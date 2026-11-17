import hashlib
import json
from pathlib import Path

import pytest

from paper2voice.cli import build_parser, main
from paper2voice.parser import (
    ParsingError,
    SubprocessPaddleOcrEngine,
    configure_nvidia_dll_directories,
    parse_bundle,
)

IMAGE_SOURCE_PATH = "imgs/img_in_image_box_1_2_30_40.jpg"


class FakeImage:
    def save(self, path):
        Path(path).write_bytes(b"fake-image")


class FakeResult:
    def __init__(self, page_index, page_count):
        self._json = {
            "res": {
                "input_path": "paper.pdf",
                "page_index": page_index,
                "page_count": page_count,
                "width": 1000,
                "height": 1400,
                "model_settings": {
                    "use_layout_detection": True,
                    "use_chart_recognition": True,
                },
                "parsing_res_list": [
                    {
                        "block_label": "image",
                        "block_content": IMAGE_SOURCE_PATH,
                        "block_bbox": [1, 2, 30, 40],
                        "block_id": 0,
                        "block_order": None,
                        "group_id": 0,
                    }
                ],
            }
        }
        self._markdown = {
            "page_index": page_index,
            "input_path": "paper.pdf",
            "markdown_texts": (
                f'<div><img src="{IMAGE_SOURCE_PATH}" alt="Image" width="20%" /></div>'
            ),
            "markdown_images": {IMAGE_SOURCE_PATH: FakeImage()},
        }

    @property
    def json(self):
        return self._json

    @property
    def markdown(self):
        return self._markdown


class FakeEngine:
    name = "fake-ocr"
    version = "1"

    def __init__(self):
        self.calls = []

    def predict(self, pdf_path):
        self.calls.append(Path(pdf_path))
        return [FakeResult(page_index=0, page_count=2)]


def write_bundle(root, pdf_bytes=b"%PDF-1.7\nfake\n%%EOF"):
    paper_dir = root / "2501.12345"
    paper_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "paper": {
            "arxiv_id": "2501.12345",
            "full_id": "2501.12345v1",
            "title": "Fake Paper",
        },
        "assets": {"pdf": "paper.pdf", "pdf_url": "https://arxiv.org/pdf/2501.12345v1"},
    }
    (paper_dir / "paper.json").write_text(json.dumps(manifest), encoding="utf-8")
    (paper_dir / "paper.pdf").write_bytes(pdf_bytes)
    return paper_dir


def test_parse_bundle_writes_page_json_markdown_and_rewritten_image(tmp_path):
    paper_dir = write_bundle(tmp_path)
    engine = FakeEngine()

    artifact = parse_bundle(paper_dir, engine)

    assert engine.calls == [paper_dir / "paper.pdf"]
    assert artifact.json_path.is_file()
    assert artifact.markdown_path.is_file()
    assert artifact.assets_dir.is_dir()

    parsed = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == 1
    assert parsed["engine"] == {"name": "fake-ocr", "version": "1"}
    assert (
        parsed["source"]["pdf_sha256"]
        == hashlib.sha256((paper_dir / "paper.pdf").read_bytes()).hexdigest()
    )
    assert parsed["pages"][0]["page_index"] == 0
    assert parsed["pages"][0]["page_count"] == 2

    image_path = "parse_assets/page_0001_img_in_image_box_1_2_30_40.jpg"
    assert parsed["pages"][0]["images"] == [
        {"source": IMAGE_SOURCE_PATH, "path": image_path}
    ]
    assert parsed["pages"][0]["raw"]["parsing_res_list"][0]["block_content"] == (
        image_path
    )
    assert (paper_dir / image_path).read_bytes() == b"fake-image"

    markdown = artifact.markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Fake Paper")
    assert image_path in markdown
    assert IMAGE_SOURCE_PATH not in markdown


def test_parse_bundle_reuses_result_until_source_changes(tmp_path):
    paper_dir = write_bundle(tmp_path)
    engine = FakeEngine()

    parse_bundle(paper_dir, engine)
    parse_bundle(paper_dir, engine)

    assert len(engine.calls) == 1

    (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.7\nchanged\n%%EOF")
    parse_bundle(paper_dir, engine)

    assert len(engine.calls) == 2


def test_parse_bundle_rebuilds_when_referenced_image_is_missing(tmp_path):
    paper_dir = write_bundle(tmp_path)
    engine = FakeEngine()
    parse_bundle(paper_dir, engine)
    image_path = paper_dir / "parse_assets" / "page_0001_img_in_image_box_1_2_30_40.jpg"
    image_path.unlink()

    parse_bundle(paper_dir, engine)

    assert len(engine.calls) == 2
    assert image_path.is_file()


def test_parse_bundle_cache_hit_does_not_construct_ocr_engine(tmp_path):
    from paper2voice.parser import DeferredOcrEngine

    paper_dir = write_bundle(tmp_path)
    parse_bundle(paper_dir, FakeEngine())
    constructions = []

    def create_engine():
        constructions.append(True)
        return FakeEngine()

    parse_bundle(
        paper_dir,
        DeferredOcrEngine("fake-ocr", "1", create_engine),
    )

    assert constructions == []


def test_parse_bundle_rejects_missing_pdf_before_loading_engine(tmp_path):
    paper_dir = write_bundle(tmp_path)
    (paper_dir / "paper.pdf").unlink()
    engine = FakeEngine()

    with pytest.raises(ParsingError, match="paper.pdf"):
        parse_bundle(paper_dir, engine)

    assert engine.calls == []


def test_cli_parses_a_bundle_with_injected_engine(tmp_path, capsys):
    paper_dir = write_bundle(tmp_path)

    exit_code = main(
        ["parse", str(paper_dir)],
        parser_engine=FakeEngine(),
    )

    assert exit_code == 0
    assert (paper_dir / "parse.json").is_file()
    assert f"Parsed {paper_dir}" in capsys.readouterr().out


def test_parse_cli_defaults_to_paddleocr_vl_1_6():
    args = build_parser().parse_args(["parse", "bundle"])

    assert args.pipeline_version == "v1.6"


def test_configure_nvidia_dll_directories_adds_every_vendor_bin(tmp_path, monkeypatch):
    nvidia_root = tmp_path / "site-packages" / "nvidia"
    expected = [
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cudnn" / "bin",
    ]
    for directory in expected:
        directory.mkdir(parents=True)
    added = []
    monkeypatch.setattr(
        "paper2voice.parser.os.add_dll_directory",
        lambda path: added.append(Path(path)) or object(),
    )
    monkeypatch.setattr("paper2voice.parser._NVIDIA_DLL_HANDLES", [])
    monkeypatch.setattr("paper2voice.parser._NVIDIA_DLL_DIRECTORIES", set())

    configured = configure_nvidia_dll_directories(nvidia_root)

    assert configured == expected
    assert added == expected


def test_subprocess_paddle_engine_returns_normalized_pages(tmp_path, monkeypatch):
    from pypdf import PdfWriter

    pdf_path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    python_path = tmp_path / "ocr-python.exe"
    python_path.write_bytes(b"fake")
    calls = []

    def fake_run(command, **kwargs):
        assert command[:3] == [str(python_path), "-m", "paper2voice.ocr_worker"]
        assert kwargs["cwd"] == tmp_path
        assert command[command.index("--assets-dir") + 1] == str(
            (tmp_path / "parse_assets").resolve()
        )
        offset = int(command[command.index("--page-offset") + 1])
        total_pages = int(command[command.index("--total-pages") + 1])
        calls.append(offset)
        pages = [
            {
                "page_index": page_index,
                "page_count": total_pages,
                "width": 1000,
                "height": 1400,
                "markdown": f"# OCR page {page_index + 1}",
                "images": [],
                "raw": {"page_index": page_index, "page_count": total_pages},
            }
            for page_index in range(offset, min(offset + 2, total_pages))
        ]
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.write_text(
            json.dumps({"pages": pages}),
            encoding="utf-8",
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    engine = SubprocessPaddleOcrEngine(
        "v1.6",
        python_executable=python_path,
        cwd=tmp_path,
        pages_per_chunk=2,
    )
    monkeypatch.setattr("paper2voice.parser.subprocess.run", fake_run)

    pages = list(engine.predict(pdf_path))

    assert calls == [0, 2, 4]
    assert [page["page_index"] for page in pages] == [0, 1, 2, 3, 4]
    assert all(page["page_count"] == 5 for page in pages)


def test_subprocess_paddle_engine_resumes_from_checkpoint_after_timeout(
    tmp_path, monkeypatch
):
    import subprocess

    from pypdf import PdfWriter

    pdf_path = tmp_path / "paper.pdf"
    writer = PdfWriter()
    for _ in range(6):
        writer.add_blank_page(width=612, height=792)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    python_path = tmp_path / "ocr-python.exe"
    python_path.write_bytes(b"fake")
    attempts = []

    def write_pages(command, offset, total_pages):
        output_path = Path(command[command.index("--output-json") + 1])
        pages = [
            {
                "page_index": page_index,
                "page_count": total_pages,
                "width": 100,
                "height": 200,
                "markdown": f"page {page_index}",
                "images": [],
                "raw": {},
            }
            for page_index in range(offset, min(offset + 2, total_pages))
        ]
        output_path.write_text(
            json.dumps({"pages": pages}),
            encoding="utf-8",
        )

    def first_run(command, **kwargs):
        offset = int(command[command.index("--page-offset") + 1])
        attempts.append(offset)
        if offset == 2:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        write_pages(command, offset, 6)
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    engine = SubprocessPaddleOcrEngine(
        "v1.6",
        python_executable=python_path,
        cwd=tmp_path,
        pages_per_chunk=2,
        chunk_timeout=60,
    )
    monkeypatch.setattr("paper2voice.parser.subprocess.run", first_run)

    with pytest.raises(ParsingError, match="2/6 pages are saved"):
        list(engine.predict(pdf_path))

    checkpoint = json.loads(
        (tmp_path / ".ocr-checkpoint.json").read_text(encoding="utf-8")
    )
    assert [page["page_index"] for page in checkpoint["pages"]] == [0, 1]

    attempts.clear()

    def resumed_run(command, **kwargs):
        offset = int(command[command.index("--page-offset") + 1])
        attempts.append(offset)
        write_pages(command, offset, 6)
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("paper2voice.parser.subprocess.run", resumed_run)
    resumed = list(engine.predict(pdf_path))

    assert attempts == [2, 4]
    assert [page["page_index"] for page in resumed] == [0, 1, 2, 3, 4, 5]
