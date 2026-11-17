from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from paper2voice.explainer import ExplainerError, validate_explainer_payload

EXPLAINER_PROMPT_VERSION = 1


class TranslationError(RuntimeError):
    """Raised when a paper cannot be translated or persisted."""


@dataclass(frozen=True)
class FigureAsset:
    page_index: int
    path: str
    caption: str


@dataclass(frozen=True)
class TableAsset:
    page_index: int
    html: str


@dataclass
class DocumentAssets:
    figures: list[FigureAsset]
    tables: list[TableAsset]
    text: str


@dataclass(frozen=True)
class TranslationArtifact:
    bundle_dir: Path
    json_path: Path
    markdown_path: Path
    digest_path: Path
    explainer_path: Path


class Translator(Protocol):
    name: str
    model: str

    def translate(self, text: str, *, kind: str = "text") -> str: ...


class OpenAICompatibleTranslator:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        timeout: float = 120,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def translate(self, text: str, *, kind: str = "text") -> str:
        if not self.api_key:
            raise TranslationError("LLM API key is not configured")
        endpoint = (
            self.api_url
            if self.api_url.endswith("/chat/completions")
            else f"{self.api_url}/chat/completions"
        )
        if kind == "explainer":
            return self._generate_explainer(endpoint, text)
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的学术论文翻译器。将输入翻译为简体中文。"
                        "保留 Markdown、HTML、LaTeX、引用、表格结构、数字和专有名词。"
                        "不要添加解释、总结或原文不存在的信息，只输出翻译结果。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"内容类型：{kind}\n\n{text}",
                },
            ],
        }
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            raise TranslationError(f"LLM translation failed: {kind}") from exc

    def _generate_explainer(self, endpoint: str, text: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是严谨的 AI 论文技术编辑。根据输入生成结构化讲解 JSON。"
                        "只输出 JSON 对象，不要输出 Markdown 代码块或额外说明。"
                        "所有数字必须来自输入，并保留证据页码；缺失信息写“未提供”。"
                    ),
                },
                {"role": "user", "content": text},
            ],
        }
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
            raise TranslationError(
                "LLM structured explainer generation failed"
            ) from exc


def extract_document_assets(parse_payload: dict) -> DocumentAssets:
    figures = []
    tables = []
    text_blocks = []
    for page in parse_payload.get("pages") or []:
        page_index = int(page.get("page_index") or 0)
        raw = page.get("raw") or {}
        blocks = raw.get("parsing_res_list") or []
        captions = [
            str(block.get("block_content") or "").strip()
            for block in blocks
            if block.get("block_label")
            in {"figure_title", "chart_title", "vision_footnote"}
            and str(block.get("block_content") or "").strip()
        ]
        for image_index, image in enumerate(page.get("images") or []):
            path = str(image.get("path") or "").replace("\\", "/")
            if not path:
                continue
            caption = captions[image_index] if image_index < len(captions) else ""
            figures.append(
                FigureAsset(
                    page_index=page_index,
                    path=path,
                    caption=caption,
                )
            )
        for block in blocks:
            label = str(block.get("block_label") or "")
            content = str(block.get("block_content") or "").strip()
            if not content:
                continue
            if label in {"table", "chart"}:
                tables.append(TableAsset(page_index=page_index, html=content))
            elif label not in {
                "image",
                "chart",
                "table",
                "header",
                "header_image",
                "footer",
                "footer_image",
                "number",
                "aside_text",
            }:
                text_blocks.append(content)
    return DocumentAssets(
        figures=figures,
        tables=tables,
        text="\n\n".join(_unique(text_blocks)),
    )


def translate_bundle(
    bundle_dir: str | Path,
    translator: Translator,
    *,
    max_document_chars: int = 0,
) -> TranslationArtifact:
    paper_dir = Path(bundle_dir)
    paper_path = paper_dir / "paper.json"
    parse_path = paper_dir / "parse.json"
    if not paper_path.is_file():
        raise TranslationError(f"Missing paper manifest: {paper_path}")
    if not parse_path.is_file():
        raise TranslationError(f"Missing OCR parse: {parse_path}")
    try:
        paper_payload = json.loads(paper_path.read_text(encoding="utf-8"))
        parse_payload = json.loads(parse_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationError("Cannot read paper bundle JSON") from exc

    paper = paper_payload.get("paper") or {}
    title = str(paper.get("title") or "").strip()
    summary = str(paper.get("summary") or "").strip()
    if not title or not summary:
        raise TranslationError("Paper title and summary are required")
    json_path = paper_dir / "translation.json"
    markdown_path = paper_dir / "paper.zh.md"
    digest_path = paper_dir / "digest.zh.md"
    explainer_path = paper_dir / "explainer.zh.json"
    source = {
        "paper_sha256": _sha256(paper_path.read_bytes()),
        "parse_sha256": _sha256(parse_path.read_bytes()),
    }
    options = {"max_document_chars": max_document_chars}
    provider = {"name": translator.name, "model": translator.model}
    if _translation_is_current(
        json_path,
        markdown_path,
        digest_path,
        source=source,
        provider=provider,
        options=options,
    ):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if _explainer_is_current(payload, explainer_path):
            return TranslationArtifact(
                bundle_dir=paper_dir,
                json_path=json_path,
                markdown_path=markdown_path,
                digest_path=digest_path,
                explainer_path=explainer_path,
            )
        explainer = _generate_explainer(translator, paper, payload)
        payload["explainer"] = explainer
        payload["explainer_version"] = EXPLAINER_PROMPT_VERSION
        _write_explainer_outputs(payload, explainer_path, json_path)
        return TranslationArtifact(
            bundle_dir=paper_dir,
            json_path=json_path,
            markdown_path=markdown_path,
            digest_path=digest_path,
            explainer_path=explainer_path,
        )
    assets = extract_document_assets(parse_payload)
    parse_markdown_path = paper_dir / "parse.md"
    document_text = (
        parse_markdown_path.read_text(encoding="utf-8")
        if parse_markdown_path.is_file()
        else assets.text
    )
    if max_document_chars > 0:
        document_text = document_text[:max_document_chars]

    try:
        title_zh = _clean_plain_text(translator.translate(title, kind="title"))
        summary_zh = _clean_plain_text(translator.translate(summary, kind="abstract"))
        markdown_zh = "\n\n".join(
            _clean_translated_text(translator.translate(chunk, kind="document"))
            for chunk in _chunk_text(document_text, max_chars=12_000)
        )
        translated_figures = [
            {
                "page_index": figure.page_index,
                "path": figure.path,
                "caption_en": _plain_text(figure.caption),
                "caption_zh": (
                    _clean_plain_text(
                        translator.translate(
                            _plain_text(figure.caption),
                            kind="figure_caption",
                        )
                    )
                    if figure.caption
                    else ""
                ),
            }
            for figure in assets.figures
        ]
        translated_tables = [
            {
                "page_index": table.page_index,
                "html_en": table.html,
                "html_zh": translator.translate(table.html, kind="table"),
            }
            for table in assets.tables
        ]
    except TranslationError:
        raise
    except Exception as exc:
        raise TranslationError("LLM translation failed") from exc

    payload = {
        "schema_version": 1,
        "source": source,
        "provider": provider,
        "options": options,
        "title_zh": title_zh,
        "summary_zh": summary_zh,
        "markdown_zh": markdown_zh,
        "figure_count": len(translated_figures),
        "table_count": len(translated_tables),
        "figures": translated_figures,
        "tables": translated_tables,
        "explainer_version": EXPLAINER_PROMPT_VERSION,
    }
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        markdown_path,
        _chinese_markdown(title_zh, summary_zh, markdown_zh, payload),
    )
    _atomic_write(
        digest_path,
        _digest_markdown(title_zh, summary_zh, payload),
    )
    explainer = _generate_explainer(translator, paper, payload)
    payload["explainer"] = explainer
    _write_explainer_outputs(payload, explainer_path, json_path)
    return TranslationArtifact(
        bundle_dir=paper_dir,
        json_path=json_path,
        markdown_path=markdown_path,
        digest_path=digest_path,
        explainer_path=explainer_path,
    )


def _chinese_markdown(
    title_zh: str,
    summary_zh: str,
    markdown_zh: str,
    payload: dict,
) -> str:
    lines = [
        f"# {title_zh}",
        "",
        "## 摘要",
        "",
        summary_zh,
        "",
        "## 全文翻译",
        "",
        markdown_zh,
    ]
    if payload["tables"]:
        lines.extend(["", "## 表格", ""])
        for index, table in enumerate(payload["tables"], 1):
            lines.extend([f"### 表格 {index}", "", table["html_zh"], ""])
    if payload["figures"]:
        lines.extend(["", "## 图表说明", ""])
        for figure in payload["figures"]:
            lines.extend(
                [
                    f"![{figure['caption_zh'] or '论文图表'}]({figure['path']})",
                    "",
                    figure["caption_zh"],
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _digest_markdown(title_zh: str, summary_zh: str, payload: dict) -> str:
    return (
        "# 深读稿\n\n"
        f"## 论文\n\n{title_zh}\n\n"
        f"## 核心内容\n\n{summary_zh}\n\n"
        f"## 多媒体信息\n\n"
        f"- 图表：{payload['figure_count']}\n"
        f"- 表格：{payload['table_count']}\n"
    )


def _generate_explainer(
    translator: Translator,
    paper: dict,
    payload: dict,
) -> dict:
    explainer_request = _explainer_request(
        paper=paper,
        title_zh=str(payload["title_zh"]),
        summary_zh=str(payload["summary_zh"]),
        markdown_zh=str(payload["markdown_zh"]),
        figures=payload.get("figures") or [],
        tables=payload.get("tables") or [],
    )
    try:
        return validate_explainer_payload(
            _json_object(
                translator.translate(
                    explainer_request,
                    kind="explainer",
                )
            )
        )
    except (ExplainerError, json.JSONDecodeError) as exc:
        raise TranslationError(
            f"LLM returned an invalid structured explainer: {exc}"
        ) from exc


def _explainer_is_current(payload: dict, explainer_path: Path) -> bool:
    if (
        payload.get("explainer_version") != EXPLAINER_PROMPT_VERSION
        or not isinstance(payload.get("explainer"), dict)
        or not explainer_path.is_file()
    ):
        return False
    try:
        explainer = validate_explainer_payload(
            json.loads(explainer_path.read_text(encoding="utf-8"))
        )
    except (ExplainerError, OSError, json.JSONDecodeError):
        return False
    return explainer == payload["explainer"]


def _write_explainer_outputs(
    payload: dict,
    explainer_path: Path,
    json_path: Path,
) -> None:
    _atomic_write(
        explainer_path,
        json.dumps(payload["explainer"], ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(
        json_path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _explainer_request(
    *,
    paper: dict,
    title_zh: str,
    summary_zh: str,
    markdown_zh: str,
    figures: list[dict],
    tables: list[dict],
) -> str:
    metadata = paper.get("metadata") or {}
    paper_info = {
        "title": title_zh,
        "problem": summary_zh,
        "authors": paper.get("authors") or [],
        "institutions": metadata.get("institutions") or [],
        "venue": metadata.get("venue") or metadata.get("journal_ref") or "未提供",
        "published": paper.get("published") or "未提供",
        "links": {
            "arxiv": paper.get("abstract_url") or "未提供",
            "doi": metadata.get("doi") or "未提供",
            "code": metadata.get("code_url") or "未提供",
            "project": metadata.get("project_url") or "未提供",
        },
    }
    schema = {
        "schema_version": 1,
        "audience": "ai_graduate_developer",
        "paper_info": paper_info,
        "innovations": [
            {
                "title": "string",
                "summary": "string",
                "evidence_pages": [1],
            }
        ],
        "method": {
            "overview": "string",
            "steps": [
                {
                    "title": "string",
                    "detail": "string",
                    "evidence_pages": [1],
                }
            ],
            "formulas": [
                {
                    "latex": "string",
                    "explanation": "string",
                    "evidence_pages": [1],
                }
            ],
        },
        "experiments": {
            "setup": "string",
            "main_results": [
                {
                    "claim": "string",
                    "value": "string",
                    "evidence_pages": [1],
                }
            ],
            "ablations": [],
            "visual_figures": [],
        },
        "summary": {
            "value": "string",
            "applications": ["string"],
            "limitations": ["string"],
            "conclusion": "string",
        },
    }
    source = _select_explainer_source(markdown_zh)
    return (
        "按以下 JSON 结构返回五章技术讲解内容：\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "约束：\n"
        "- 全部章节合计目标 2500-2800 个中文字符。\n"
        "- innovations 1-3 个，默认 3 个，必须彼此不同。\n"
        "- 每个创新点 summary 写 100-160 字，讲清问题、改变和差异。\n"
        "- method.overview 写 180-240 字；每个 step.detail 写 140-220 字。\n"
        "- method.formulas 最多 3 个，只保留核心公式，不推导。\n"
        "- 每条 main_results.claim 写 80-140 字，并给出来源页码。\n"
        "- experiments.setup 写 120-180 字，说明数据集、基线和指标。\n"
        "- summary 必须包含价值、至少两个应用、至少两个局限和结论。\n"
        "- 实验数字必须能在原文找到，并填写 evidence_pages。\n"
        "- 区分绝对结果和相对提升。\n"
        "- visual_figures 使用给定图表路径。\n\n"
        f"可用图表：{json.dumps(figures, ensure_ascii=False)}\n"
        f"可用表格：{json.dumps(tables, ensure_ascii=False)}\n\n"
        f"论文翻译正文：\n{source}"
    )


def _select_explainer_source(markdown: str, *, max_chars: int = 120_000) -> str:
    if len(markdown) <= max_chars:
        return markdown
    keyword = re.compile(
        r"(摘要|引言|背景|方法|架构|实现|训练|实验|结果|消融|结论|局限|"
        r"abstract|introduction|method|approach|experiment|result|"
        r"ablation|conclusion|limitation)",
        re.IGNORECASE,
    )
    sections = re.split(r"(?m)(?=^#{1,4}\s)", markdown)
    selected = [section for section in sections if keyword.search(section[:240])]
    source = "\n\n".join(selected) if selected else markdown
    if len(source) <= max_chars:
        return source
    head = source[: max_chars // 2]
    tail = source[-max_chars // 2 :]
    return f"{head}\n\n[...中间内容已截断...]\n\n{tail}"


def _json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("JSON object not found", text, 0)
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("JSON object required", text, start)
    return payload


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    if not text.strip():
        return [""]
    chunks = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _clean_translated_text(value: str) -> str:
    text = value.strip()
    return re.sub(
        r"^(?:内容类型\s*[:：]\s*)?"
        r"(?:标题|摘要|正文|文档|图表说明|表格)"
        r"\s*[:：]?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def _plain_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return " ".join(text.split())


def _clean_plain_text(value: str) -> str:
    return _plain_text(_clean_translated_text(value))


def _translation_is_current(
    json_path: Path,
    markdown_path: Path,
    digest_path: Path,
    *,
    source: dict,
    provider: dict,
    options: dict,
) -> bool:
    if (
        not json_path.is_file()
        or not markdown_path.is_file()
        or not digest_path.is_file()
    ):
        return False
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == 1
        and payload.get("source") == source
        and payload.get("provider") == provider
        and payload.get("options") == options
    )


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
