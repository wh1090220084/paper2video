import json

import pytest

from paper2voice.arxiv import (
    ArxivClient,
    ArxivNotFoundError,
    parse_arxiv_id,
)
from paper2voice.cli import main
from paper2voice.pulse import load_pulse_json
from paper2voice.store import InvalidPdfError, PaperRepository

ATOM_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <updated>2025-01-20T10:00:00Z</updated>
    <published>2025-01-16T09:00:00Z</published>
    <title>A
      Useful Paper</title>
    <summary>First line.
      Second line.</summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2501.12345v2" rel="alternate"/>
    <link href="http://arxiv.org/pdf/2501.12345v2" rel="related" title="pdf"/>
    <category term="cs.CV"/>
    <category term="cs.AI"/>
  </entry>
</feed>
"""


EMPTY_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
</feed>
"""


def test_parse_arxiv_id_accepts_urls_and_preserves_versions():
    parsed = parse_arxiv_id("https://arxiv.org/abs/hep-th/9901001v2")

    assert parsed.base_id == "hep-th/9901001"
    assert parsed.version == 2
    assert parsed.full_id == "hep-th/9901001v2"


def test_client_parses_atom_metadata_without_network():
    requested_urls = []

    def fake_fetch(url):
        requested_urls.append(url)
        return ATOM_FEED.encode("utf-8")

    paper = ArxivClient(fetch_bytes=fake_fetch).fetch("2501.12345v2")

    assert requested_urls == ["https://export.arxiv.org/api/query?id_list=2501.12345v2"]
    assert paper.arxiv_id == "2501.12345"
    assert paper.version == 2
    assert paper.full_id == "2501.12345v2"
    assert paper.title == "A Useful Paper"
    assert paper.summary == "First line. Second line."
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.categories == ["cs.CV", "cs.AI"]
    assert paper.pdf_url == "https://arxiv.org/pdf/2501.12345v2"
    assert paper.abstract_url == "https://arxiv.org/abs/2501.12345v2"
    assert paper.published == "2025-01-16T09:00:00Z"
    assert paper.updated == "2025-01-20T10:00:00Z"


def test_client_raises_not_found_for_empty_feed():
    client = ArxivClient(fetch_bytes=lambda _: EMPTY_FEED.encode("utf-8"))

    with pytest.raises(ArxivNotFoundError, match="2501.99999"):
        client.fetch("2501.99999")


def test_load_pulse_json_normalizes_records_and_deduplicates(tmp_path):
    path = tmp_path / "top5.json"
    record = {
        "arxiv_id": "2501.12345v2",
        "title": "Pulse Paper",
        "summary": "Abstract text.",
        "authors": ["Ada Lovelace"],
        "published": "2025-01-16T09:00:00Z",
        "updated": "2025-01-20T10:00:00Z",
        "categories": ["cs.CV"],
        "pdf_url": "http://arxiv.org/pdf/2501.12345v2",
        "abstract_url": "http://arxiv.org/abs/2501.12345v2",
        "sub_direction": "image classification",
    }
    path.write_text(json.dumps([record, record]), encoding="utf-8")

    papers = load_pulse_json(path)

    assert len(papers) == 1
    assert papers[0].full_id == "2501.12345v2"
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2501.12345v2"
    assert papers[0].abstract_url == "https://arxiv.org/abs/2501.12345v2"
    assert papers[0].metadata["sub_direction"] == "image classification"


def test_repository_writes_manifest_and_pdf(tmp_path):
    downloaded_urls = []

    def fake_fetch(url):
        downloaded_urls.append(url)
        return b"%PDF-1.7\nfake pdf\n%%EOF"

    paper = ArxivClient(fetch_bytes=lambda _: ATOM_FEED.encode("utf-8")).fetch(
        "2501.12345v2"
    )
    repository = PaperRepository(tmp_path, fetch_bytes=fake_fetch)

    artifact = repository.ingest(paper, download_pdf=True)

    assert downloaded_urls == ["https://arxiv.org/pdf/2501.12345v2"]
    assert artifact.directory == tmp_path / "2501.12345"
    assert artifact.manifest_path.is_file()
    assert artifact.pdf_path.read_bytes().startswith(b"%PDF-")
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["paper"]["full_id"] == "2501.12345v2"
    assert manifest["assets"]["pdf"] == "paper.pdf"


def test_repository_does_not_publish_invalid_pdf(tmp_path):
    paper = ArxivClient(fetch_bytes=lambda _: ATOM_FEED.encode("utf-8")).fetch(
        "2501.12345v2"
    )
    repository = PaperRepository(
        tmp_path, fetch_bytes=lambda _: b"<html>not a pdf</html>"
    )

    with pytest.raises(InvalidPdfError, match="PDF signature"):
        repository.ingest(paper, download_pdf=True)

    paper_dir = tmp_path / "2501.12345"
    assert not (paper_dir / "paper.json").exists()
    assert not (paper_dir / "paper.pdf").exists()
    assert not list(paper_dir.glob("*.part"))


def test_cli_fetches_one_paper_by_id(tmp_path, capsys):
    paper = ArxivClient(fetch_bytes=lambda _: ATOM_FEED.encode("utf-8")).fetch(
        "2501.12345v2"
    )
    client = type("FakeClient", (), {"fetch": lambda self, _: paper})()

    exit_code = main(
        ["fetch", "2501.12345v2", "--output-dir", str(tmp_path), "--no-download"],
        arxiv_client=client,
    )

    assert exit_code == 0
    manifest_path = tmp_path / "2501.12345" / "paper.json"
    assert manifest_path.is_file()
    assert "Fetched 2501.12345v2" in capsys.readouterr().out


def test_cli_imports_pulse_json_into_same_manifest_format(tmp_path):
    pulse_path = tmp_path / "pulse.json"
    pulse_path.write_text(
        json.dumps(
            [
                {
                    "arxiv_id": "2501.12345v1",
                    "title": "Pulse Paper",
                    "summary": "Abstract text.",
                    "authors": ["Ada Lovelace"],
                    "published": "2025-01-16T09:00:00Z",
                    "categories": ["cs.CV"],
                    "pdf_url": "https://arxiv.org/pdf/2501.12345v1",
                    "abstract_url": "https://arxiv.org/abs/2501.12345v1",
                }
            ]
        ),
        encoding="utf-8",
    )
    papers_dir = tmp_path / "papers"

    exit_code = main(["import-pulse", str(pulse_path), "--output-dir", str(papers_dir)])

    assert exit_code == 0
    manifest = json.loads(
        (papers_dir / "2501.12345" / "paper.json").read_text(encoding="utf-8")
    )
    assert manifest["paper"]["title"] == "Pulse Paper"
    assert manifest["assets"]["pdf"] is None
