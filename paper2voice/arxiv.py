from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
_NEW_ID = re.compile(
    r"(?<![A-Za-z0-9])(?P<base>\d{4}\.\d{4,5})(?P<version>v\d+)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_OLD_ID = re.compile(
    r"(?<![A-Za-z0-9])(?P<base>[a-z][a-z-]+(?:\.[a-z]{2})?/\d{7})"
    r"(?P<version>v\d+)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


class ArxivError(RuntimeError):
    """Base error for arXiv ingestion failures."""


class InvalidArxivIdError(ArxivError, ValueError):
    """Raised when an arXiv identifier cannot be normalized."""


class ArxivRequestError(ArxivError):
    """Raised when arXiv cannot be reached or returns an invalid response."""


class ArxivNotFoundError(ArxivError):
    """Raised when an arXiv query returns no paper."""


@dataclass(frozen=True)
class ArxivId:
    base_id: str
    version: int | None = None

    @property
    def full_id(self) -> str:
        suffix = f"v{self.version}" if self.version is not None else ""
        return f"{self.base_id}{suffix}"


def parse_arxiv_id(value: str) -> ArxivId:
    text = urllib.parse.unquote(value.strip())
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE)

    if "://" in text:
        path = urllib.parse.urlsplit(text).path.rstrip("/")
        for marker in ("/abs/", "/pdf/"):
            if marker in path:
                text = path.split(marker, 1)[1]
                break

    if text.lower().endswith(".pdf"):
        text = text[:-4]

    for pattern in (_NEW_ID, _OLD_ID):
        match = pattern.search(text)
        if not match:
            continue
        version_text = match.group("version")
        version = int(version_text[1:]) if version_text else None
        return ArxivId(base_id=match.group("base"), version=version)

    raise InvalidArxivIdError(f"Invalid arXiv identifier: {value!r}")


@dataclass
class PaperMetadata:
    arxiv_id: str
    version: int | None
    title: str
    summary: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    pdf_url: str
    abstract_url: str
    metadata: dict = field(default_factory=dict)

    @property
    def full_id(self) -> str:
        suffix = f"v{self.version}" if self.version is not None else ""
        return f"{self.arxiv_id}{suffix}"

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "full_id": self.full_id,
            "version": self.version,
            "title": self.title,
            "summary": self.summary,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "abstract_url": self.abstract_url,
            "metadata": self.metadata,
        }


class ArxivClient:
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        timeout: float = 30,
        fetch_bytes: Callable[[str], bytes] | None = None,
    ):
        self.timeout = timeout
        self._fetch_bytes = fetch_bytes or self._urlopen_bytes

    def fetch(self, arxiv_id: str) -> PaperMetadata:
        parsed_id = parse_arxiv_id(arxiv_id)
        query = urllib.parse.urlencode({"id_list": parsed_id.full_id})
        url = f"{self.BASE_URL}?{query}"
        try:
            payload = self._fetch_bytes(url)
        except ArxivError:
            raise
        except Exception as exc:
            raise ArxivRequestError(
                f"Failed to query arXiv for {parsed_id.full_id}"
            ) from exc
        return self._parse_atom(payload, parsed_id.full_id)

    def _urlopen_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "paper2voice/0.1 (https://arxiv.org/help/api/)",
                "Accept": "application/atom+xml",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if getattr(response, "status", 200) >= 400:
                raise ArxivRequestError(f"arXiv returned HTTP {response.status}")
            return response.read()

    def _parse_atom(self, payload: bytes, requested_id: str) -> PaperMetadata:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise ArxivRequestError("arXiv returned malformed Atom XML") from exc

        entry = root.find("atom:entry", ATOM_NAMESPACE)
        if entry is None:
            raise ArxivNotFoundError(f"arXiv paper not found: {requested_id}")

        raw_id = _element_text(entry, "atom:id")
        parsed_id = parse_arxiv_id(raw_id)
        pdf_url = ""
        abstract_url = ""
        for link in entry.findall("atom:link", ATOM_NAMESPACE):
            href = link.attrib.get("href", "")
            if link.attrib.get("rel") == "alternate":
                abstract_url = href
            if link.attrib.get("title") == "pdf":
                pdf_url = href

        return PaperMetadata(
            arxiv_id=parsed_id.base_id,
            version=parsed_id.version,
            title=_element_text(entry, "atom:title"),
            summary=_element_text(entry, "atom:summary"),
            authors=[
                _normalize_space(
                    author.findtext("atom:name", default="", namespaces=ATOM_NAMESPACE)
                )
                for author in entry.findall("atom:author", ATOM_NAMESPACE)
            ],
            published=_element_text(entry, "atom:published"),
            updated=_element_text(entry, "atom:updated"),
            categories=[
                category.attrib["term"]
                for category in entry.findall("atom:category", ATOM_NAMESPACE)
                if category.attrib.get("term")
            ],
            pdf_url=_https_url(pdf_url or f"https://arxiv.org/pdf/{parsed_id.full_id}"),
            abstract_url=_https_url(
                abstract_url or f"https://arxiv.org/abs/{parsed_id.full_id}"
            ),
        )


def _element_text(entry: ET.Element, path: str) -> str:
    element = entry.find(path, ATOM_NAMESPACE)
    if element is None:
        return ""
    return _normalize_space("".join(element.itertext()))


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _https_url(value: str) -> str:
    if value.startswith("http://"):
        return f"https://{value[len('http://') :]}"
    return value
