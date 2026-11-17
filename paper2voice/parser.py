from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_NVIDIA_DLL_HANDLES: list[Any] = []
_NVIDIA_DLL_DIRECTORIES: set[Path] = set()


class ParsingError(RuntimeError):
    """Raised when an OCR engine cannot produce a valid paper parse."""


@dataclass(frozen=True)
class ParseArtifact:
    bundle_dir: Path
    json_path: Path
    markdown_path: Path
    assets_dir: Path


class OcrEngine(Protocol):
    name: str
    version: str

    def predict(self, pdf_path: Path) -> Iterable[Any]: ...


@dataclass(frozen=True)
class DeferredOcrEngine:
    name: str
    version: str
    factory: Callable[[], OcrEngine]

    def create(self) -> OcrEngine:
        return self.factory()


class PaddleOcrVlEngine:
    name = "paddleocr-vl"

    def __init__(self, pipeline_version: str = "v1.6", **pipeline_kwargs: Any):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        configure_nvidia_dll_directories()
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise ParsingError(
                "PaddleOCR-VL is not installed; install requirements-ocr.txt"
            ) from exc

        self.version = pipeline_version
        self._pipeline = PaddleOCRVL(
            pipeline_version=pipeline_version,
            **pipeline_kwargs,
        )

    def predict(self, pdf_path: Path) -> Iterable[Any]:
        return self._pipeline.predict(
            str(pdf_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            use_chart_recognition=True,
            format_block_content=True,
        )


class SubprocessPaddleOcrEngine:
    name = "paddleocr-vl"

    def __init__(
        self,
        pipeline_version: str = "v1.6",
        *,
        python_executable: str | Path | None = None,
        cwd: str | Path | None = None,
        pages_per_chunk: int = 8,
        chunk_timeout: float = 1800,
    ):
        self.version = pipeline_version
        self.python_executable = (
            Path(
                python_executable
                or os.environ.get("PAPER2VOICE_OCR_PYTHON")
                or _default_ocr_python()
            )
            .expanduser()
            .resolve()
        )
        self.cwd = (
            Path(cwd).expanduser().resolve()
            if cwd
            else Path(__file__).resolve().parents[1]
        )
        self.pages_per_chunk = max(1, pages_per_chunk)
        self.chunk_timeout = max(60, chunk_timeout)

    def predict(self, pdf_path: Path) -> Iterable[dict]:
        if not self.python_executable.is_file():
            raise ParsingError(
                f"OCR Python executable not found: {self.python_executable}"
            )
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError as exc:
            raise ParsingError("pypdf is required for resumable OCR chunking") from exc
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ParsingError(f"PDF contains no pages: {pdf_path}")
        source_hash = _sha256(pdf_path.read_bytes())
        checkpoint_path = pdf_path.with_name(".ocr-checkpoint.json")
        assets_dir = pdf_path.parent / "parse_assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_pages = self._load_checkpoint(
            checkpoint_path,
            source_hash,
            assets_dir,
        )
        completed_pages = {
            int(page["page_index"])
            for page in checkpoint_pages
            if isinstance(page, dict) and "page_index" in page
        }
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(self.cwd)
        pages_by_index = {
            int(page["page_index"]): page
            for page in checkpoint_pages
            if isinstance(page, dict) and "page_index" in page
        }
        with tempfile.TemporaryDirectory(
            dir=pdf_path.parent,
            prefix=".ocr-chunk-",
        ) as temporary_dir:
            temporary = Path(temporary_dir)
            for start in range(0, total_pages, self.pages_per_chunk):
                end = min(start + self.pages_per_chunk, total_pages)
                if all(index in completed_pages for index in range(start, end)):
                    continue
                chunk_pdf = temporary / f"pages-{start + 1:04d}-{end:04d}.pdf"
                chunk_writer = PdfWriter()
                for page_index in range(start, end):
                    chunk_writer.add_page(reader.pages[page_index])
                with chunk_pdf.open("wb") as handle:
                    chunk_writer.write(handle)
                output_path = temporary / f"pages-{start + 1:04d}-{end:04d}.json"
                command = [
                    str(self.python_executable),
                    "-m",
                    "paper2voice.ocr_worker",
                    "--pdf",
                    str(chunk_pdf.resolve()),
                    "--pipeline-version",
                    self.version,
                    "--page-offset",
                    str(start),
                    "--total-pages",
                    str(total_pages),
                    "--assets-dir",
                    str(assets_dir.resolve()),
                    "--output-json",
                    str(output_path.resolve()),
                ]
                try:
                    result = subprocess.run(
                        command,
                        cwd=self.cwd,
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self.chunk_timeout,
                    )
                except subprocess.TimeoutExpired as exc:
                    completed = len(completed_pages)
                    raise ParsingError(
                        f"OCR worker timed out on pages {start + 1}-{end}; "
                        f"{completed}/{total_pages} pages are saved in "
                        f"{checkpoint_path}. Rerun to resume."
                    ) from exc
                if result.returncode != 0:
                    detail = "\n".join(
                        (result.stderr or result.stdout or "unknown OCR worker error")
                        .strip()
                        .splitlines()[-12:]
                    )
                    raise ParsingError(f"OCR worker failed: {detail}")
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ParsingError(
                        "OCR worker did not return valid page JSON"
                    ) from exc
                chunk_pages = payload.get("pages")
                if not isinstance(chunk_pages, list) or not chunk_pages:
                    raise ParsingError(
                        f"OCR worker returned no pages for {start + 1}-{end}"
                    )
                for page in chunk_pages:
                    page_index = int(page["page_index"])
                    pages_by_index[page_index] = page
                    completed_pages.add(page_index)
                _atomic_write(
                    checkpoint_path,
                    json.dumps(
                        {
                            "source_sha256": source_hash,
                            "pages": [
                                pages_by_index[index]
                                for index in sorted(pages_by_index)
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n",
                )
        pages = [pages_by_index[index] for index in sorted(pages_by_index)]
        if len(pages) != total_pages:
            raise ParsingError(
                f"OCR returned {len(pages)}/{total_pages} pages; "
                f"checkpoint kept at {checkpoint_path}"
            )
        return pages

    @staticmethod
    def _load_checkpoint(
        path: Path,
        source_hash: str,
        assets_dir: Path,
    ) -> list[dict]:
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if payload.get("source_sha256") != source_hash:
            return []
        pages = payload.get("pages")
        if not isinstance(pages, list):
            return []
        for page in pages:
            for image in page.get("images", []):
                image_path = assets_dir / Path(str(image.get("path", ""))).name
                if not image_path.is_file():
                    return []
        return pages


def _default_ocr_python() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    candidates = (
        project_root / "tmp" / "ocr-venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "Scripts" / "python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def configure_nvidia_dll_directories(
    nvidia_root: str | Path | None = None,
) -> list[Path]:
    """Expose pip-installed NVIDIA DLL directories before loading Paddle."""
    root = (
        Path(nvidia_root)
        if nvidia_root is not None
        else Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    )
    directories = sorted(path.resolve() for path in root.glob("*/bin") if path.is_dir())
    new_directories = [
        directory
        for directory in directories
        if directory not in _NVIDIA_DLL_DIRECTORIES
    ]
    if not new_directories:
        return directories

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for directory in new_directories:
            _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    path_entries = [str(directory) for directory in new_directories]
    os.environ["PATH"] = os.pathsep.join([*path_entries, os.environ.get("PATH", "")])
    _NVIDIA_DLL_DIRECTORIES.update(new_directories)
    return directories


def parse_bundle(
    bundle_dir: str | Path,
    engine: OcrEngine | DeferredOcrEngine,
    *,
    force: bool = False,
) -> ParseArtifact:
    paper_dir = Path(bundle_dir)
    manifest_path = paper_dir / "paper.json"
    pdf_path = paper_dir / "paper.pdf"
    if not manifest_path.is_file():
        raise ParsingError(f"Missing bundle manifest: {manifest_path}")
    if not pdf_path.is_file():
        raise ParsingError(f"Missing PDF for parsing: {pdf_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParsingError(f"Cannot read bundle manifest: {manifest_path}") from exc

    json_path = paper_dir / "parse.json"
    markdown_path = paper_dir / "parse.md"
    assets_dir = paper_dir / "parse_assets"
    source = {
        "paper_json": "paper.json",
        "paper_json_sha256": _sha256(manifest_path.read_bytes()),
        "pdf": "paper.pdf",
        "pdf_sha256": _sha256(pdf_path.read_bytes()),
    }
    engine_info = _engine_info(engine)

    if not force and _is_current_parse(
        json_path, markdown_path, assets_dir, source, engine_info
    ):
        return ParseArtifact(paper_dir, json_path, markdown_path, assets_dir)

    active_engine = engine.create() if isinstance(engine, DeferredOcrEngine) else engine
    assets_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = list(active_engine.predict(pdf_path))
    except ParsingError:
        raise
    except Exception as exc:
        raise ParsingError(f"OCR engine failed for {pdf_path}") from exc
    if not results:
        raise ParsingError(f"OCR engine returned no pages for {pdf_path}")

    pages = []
    for index, result in enumerate(results):
        if isinstance(result, dict) and {
            "page_index",
            "page_count",
            "markdown",
            "images",
            "raw",
        }.issubset(result):
            pages.append(result)
        else:
            pages.append(
                _normalize_page(
                    result,
                    fallback_index=index,
                    assets_dir=assets_dir,
                )
            )
    title = str(manifest.get("paper", {}).get("title") or "Parsed Paper")
    markdown_sections = [
        f"<!-- page {page['page_index'] + 1}:{page['page_count']} -->\n\n"
        f"{page['markdown']}"
        for page in pages
    ]
    markdown = f"# {title}\n\n" + "\n\n---\n\n".join(markdown_sections) + "\n"
    payload = {
        "schema_version": 1,
        "source": source,
        "engine": engine_info,
        "pages": pages,
    }

    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(markdown_path, markdown)
    return ParseArtifact(paper_dir, json_path, markdown_path, assets_dir)


def _engine_info(engine: OcrEngine | DeferredOcrEngine) -> dict[str, str]:
    return {
        "name": str(getattr(engine, "name", type(engine).__name__)),
        "version": str(getattr(engine, "version", "unknown")),
    }


def _normalize_page(
    result: Any,
    *,
    fallback_index: int,
    assets_dir: Path,
    page_index_offset: int = 0,
    page_count_override: int | None = None,
) -> dict:
    json_payload = result.json
    raw = copy.deepcopy(
        json_payload.get("res", json_payload)
        if isinstance(json_payload, dict)
        else json_payload
    )
    if not isinstance(raw, dict):
        raise ParsingError("OCR result JSON must contain an object")

    markdown_payload = result.markdown
    if not isinstance(markdown_payload, dict):
        raise ParsingError("OCR result Markdown must contain an object")
    markdown_text = str(markdown_payload.get("markdown_texts") or "")
    page_index = raw.get("page_index")
    if page_index is None:
        page_index = fallback_index
    page_index = int(page_index) + page_index_offset
    page_count = page_count_override or int(
        raw.get("page_count") or max(page_index + 1, 1)
    )

    replacements = {}
    images = []
    for source_path, image in (markdown_payload.get("markdown_images") or {}).items():
        if image is None:
            continue
        filename = _safe_asset_name(Path(str(source_path)).name or "image.jpg")
        relative_path = f"parse_assets/page_{page_index + 1:04d}_{filename}"
        destination = assets_dir / Path(relative_path).name
        try:
            image.save(destination)
        except Exception as exc:
            raise ParsingError(f"Cannot save parsed image: {destination}") from exc
        replacements[str(source_path)] = relative_path
        images.append({"source": str(source_path), "path": relative_path})

    markdown_text = _replace_strings(markdown_text, replacements)
    raw = _replace_strings(raw, replacements)

    return {
        "page_index": page_index,
        "page_count": page_count,
        "width": raw.get("width"),
        "height": raw.get("height"),
        "markdown": markdown_text,
        "images": images,
        "raw": raw,
    }


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for source, target in sorted(
            replacements.items(), key=lambda item: len(item[0]), reverse=True
        ):
            value = value.replace(source, target)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_strings(item, replacements) for key, item in value.items()
        }
    return value


def _is_current_parse(
    json_path: Path,
    markdown_path: Path,
    assets_dir: Path,
    source: dict,
    engine_info: dict,
) -> bool:
    if not json_path.is_file() or not markdown_path.is_file():
        return False
    try:
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        parsed.get("schema_version") == 1
        and parsed.get("source") == source
        and parsed.get("engine") == engine_info
        and bool(parsed.get("pages"))
    ):
        return False
    for page in parsed["pages"]:
        for image in page.get("images", []):
            if not (assets_dir / Path(str(image.get("path", ""))).name).is_file():
                return False
    return True


def _safe_asset_name(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem).strip("._")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        suffix = ".png"
    return f"{stem or 'image'}{suffix}"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    part_path = path.with_name(f"{path.name}.part")
    try:
        part_path.write_text(text, encoding="utf-8")
        os.replace(part_path, path)
    finally:
        if part_path.exists():
            part_path.unlink()
