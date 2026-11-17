from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paper2voice.arxiv import PaperMetadata, parse_arxiv_id


def load_pulse_json(path: str | Path) -> list[PaperMetadata]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = _records_from_payload(payload)

    papers = []
    seen = set()
    for record in records:
        paper = paper_from_pulse_record(record)
        if paper.full_id in seen:
            continue
        seen.add(paper.full_id)
        papers.append(paper)
    return papers


def paper_from_pulse_record(record: dict[str, Any]) -> PaperMetadata:
    raw_id = record.get("full_id") or record.get("arxiv_id") or record.get("id")
    if not raw_id:
        raise ValueError("arxiv-pulse record is missing an arXiv identifier")
    parsed_id = parse_arxiv_id(str(raw_id))
    authors = record.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    categories = record.get("categories") or []
    if isinstance(categories, str):
        categories = [categories]

    known_fields = {
        "arxiv_id",
        "full_id",
        "id",
        "title",
        "summary",
        "abstract",
        "authors",
        "published",
        "updated",
        "categories",
        "pdf_url",
        "abstract_url",
    }
    extra = {key: value for key, value in record.items() if key not in known_fields}

    return PaperMetadata(
        arxiv_id=parsed_id.base_id,
        version=parsed_id.version,
        title=str(record.get("title") or "").strip(),
        summary=str(record.get("summary") or record.get("abstract") or "").strip(),
        authors=[str(author) for author in authors],
        published=str(record.get("published") or ""),
        updated=str(record.get("updated") or record.get("published") or ""),
        categories=[str(category) for category in categories],
        pdf_url=_https_url(
            str(record.get("pdf_url") or f"https://arxiv.org/pdf/{parsed_id.full_id}")
        ),
        abstract_url=_https_url(
            str(
                record.get("abstract_url")
                or f"https://arxiv.org/abs/{parsed_id.full_id}"
            )
        ),
        metadata=extra,
    )


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("papers") or payload.get("records") or []
    else:
        raise TypeError("arxiv-pulse JSON must be a list or an object with papers")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("arxiv-pulse records must be JSON objects")
    return records


def _https_url(value: str) -> str:
    if value.startswith("http://"):
        return f"https://{value[len('http://') :]}"
    return value
