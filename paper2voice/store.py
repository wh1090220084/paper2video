from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from paper2voice.arxiv import PaperMetadata


class InvalidPdfError(ValueError):
    """Raised when downloaded bytes are not a PDF."""


@dataclass(frozen=True)
class PaperArtifact:
    directory: Path
    manifest_path: Path
    pdf_path: Path


class PaperRepository:
    def __init__(
        self,
        root: str | Path,
        *,
        timeout: float = 60,
        fetch_bytes: Callable[[str], bytes] | None = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._fetch_bytes = fetch_bytes or self._urlopen_bytes

    def ingest(
        self,
        paper: PaperMetadata,
        *,
        download_pdf: bool = False,
        force: bool = False,
    ) -> PaperArtifact:
        paper_dir = self.root / _safe_directory_name(paper.arxiv_id)
        paper_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = paper_dir / "paper.json"
        pdf_path = paper_dir / "paper.pdf"

        if download_pdf:
            self._download_pdf(paper.pdf_url, pdf_path, force=force)
        pdf_available = pdf_path.is_file() and pdf_path.read_bytes().startswith(
            b"%PDF-"
        )

        manifest = {
            "schema_version": 1,
            "paper": paper.to_dict(),
            "assets": {
                "pdf": "paper.pdf" if pdf_available else None,
                "pdf_url": paper.pdf_url,
            },
        }
        _atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

        return PaperArtifact(
            directory=paper_dir,
            manifest_path=manifest_path,
            pdf_path=pdf_path,
        )

    def _download_pdf(self, url: str, path: Path, *, force: bool) -> None:
        if path.is_file() and not force and path.read_bytes().startswith(b"%PDF-"):
            return

        part_path = path.with_name(f"{path.name}.part")
        try:
            payload = self._fetch_bytes(url)
            if not payload.startswith(b"%PDF-"):
                raise InvalidPdfError(
                    f"Downloaded content does not start with the PDF signature: {url}"
                )
            part_path.write_bytes(payload)
            os.replace(part_path, path)
        finally:
            if part_path.exists():
                part_path.unlink()

    def _urlopen_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "paper2voice/0.1 (https://arxiv.org/help/api/)",
                "Accept": "application/pdf",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if getattr(response, "status", 200) >= 400:
                raise OSError(f"PDF download returned HTTP {response.status}")
            return response.read()


def _safe_directory_name(arxiv_id: str) -> str:
    if "\\" in arxiv_id or arxiv_id in {".", ".."} or ".." in arxiv_id:
        raise ValueError(f"Unsafe arXiv identifier: {arxiv_id!r}")
    return arxiv_id.replace("/", "_")


def _atomic_write(path: Path, text: str) -> None:
    part_path = path.with_name(f"{path.name}.part")
    try:
        part_path.write_text(text, encoding="utf-8")
        os.replace(part_path, path)
    finally:
        if part_path.exists():
            part_path.unlink()
