import json

import pytest
from PIL import Image

from paper2voice.translation import (
    OpenAICompatibleTranslator,
    TranslationError,
    extract_document_assets,
    translate_bundle,
)


def explainer_payload():
    return {
        "schema_version": 1,
        "audience": "ai_graduate_developer",
        "paper_info": {
            "title": "Efficient Model",
            "problem": "Existing models are slow.",
            "authors": ["Ada"],
            "institutions": ["Test Lab"],
            "venue": "TestConf",
            "published": "2026",
            "links": {
                "arxiv": "https://arxiv.org/abs/2501.12345",
                "doi": "未提供",
                "code": "未提供",
                "project": "未提供",
            },
        },
        "innovations": [
            {
                "title": "Compact model",
                "summary": "A smaller architecture is used.",
                "evidence_pages": [1],
            }
        ],
        "method": {
            "overview": "The model uses a compact architecture.",
            "steps": [],
            "formulas": [],
        },
        "experiments": {
            "setup": "One benchmark.",
            "main_results": [
                {
                    "claim": "The model reaches 28.4 BLEU.",
                    "value": "28.4 BLEU",
                    "evidence_pages": [1],
                }
            ],
            "ablations": [],
            "visual_figures": ["parse_assets/figure.png"],
        },
        "summary": {
            "value": "The model is efficient.",
            "applications": ["Inference"],
            "limitations": ["Small benchmark."],
            "conclusion": "A practical model.",
        },
    }


class FakeTranslator:
    name = "fake-llm"
    model = "fake-model"

    def __init__(self, *, explainer_response=None):
        self.calls = []
        self.explainer_response = explainer_response

    def translate(self, text, *, kind="text"):
        self.calls.append((kind, text))
        if kind == "explainer":
            if self.explainer_response is not None:
                return self.explainer_response
            return json.dumps(explainer_payload(), ensure_ascii=False)
        if kind == "table":
            return "<table><tr><td>方法</td><td>分数</td></tr></table>"
        if kind == "title":
            return f"标题：中文[title]：{text}"
        if kind == "abstract":
            return f"内容类型：摘要\n\n中文[abstract]：{text}"
        return f"中文[{kind}]：{text}"


def write_parsed_bundle(root):
    bundle = root / "2501.12345"
    assets = bundle / "parse_assets"
    assets.mkdir(parents=True)
    Image.new("RGB", (200, 100), (30, 120, 180)).save(assets / "figure.png")
    paper = {
        "schema_version": 1,
        "paper": {
            "title": "Efficient Model",
            "summary": "We propose a compact model. It reaches 28.4 BLEU.",
            "authors": ["Ada"],
            "categories": ["cs.CL"],
        },
        "assets": {},
    }
    parsed = {
        "schema_version": 1,
        "source": {},
        "engine": {"name": "paddleocr-vl", "version": "v1.6"},
        "pages": [
            {
                "page_index": 0,
                "page_count": 1,
                "images": [
                    {
                        "source": "figure.png",
                        "path": "parse_assets/figure.png",
                    }
                ],
                "raw": {
                    "parsing_res_list": [
                        {
                            "block_label": "figure_title",
                            "block_content": (
                                '<div style="text-align: center;">'
                                "Figure 1: Model architecture.</div>"
                            ),
                        },
                        {
                            "block_label": "table",
                            "block_content": (
                                "<table><tr><td>Method</td><td>Score</td></tr></table>"
                            ),
                        },
                        {
                            "block_label": "text",
                            "block_content": "We propose a compact model.",
                        },
                    ]
                },
            }
        ],
    }
    (bundle / "paper.json").write_text(
        json.dumps(paper, ensure_ascii=False), encoding="utf-8"
    )
    (bundle / "parse.json").write_text(
        json.dumps(parsed, ensure_ascii=False), encoding="utf-8"
    )
    (bundle / "parse.md").write_text(
        "# Efficient Model\n\n"
        "![Figure 1](parse_assets/figure.png)\n\n"
        "<table><tr><td>Method</td><td>Score</td></tr></table>\n",
        encoding="utf-8",
    )
    return bundle


def test_extract_document_assets_collects_figures_tables_and_text(tmp_path):
    bundle = write_parsed_bundle(tmp_path)
    parsed = json.loads((bundle / "parse.json").read_text(encoding="utf-8"))

    assets = extract_document_assets(parsed)

    assert len(assets.figures) == 1
    assert assets.figures[0].path == "parse_assets/figure.png"
    assert assets.figures[0].caption == (
        '<div style="text-align: center;">Figure 1: Model architecture.</div>'
    )
    assert len(assets.tables) == 1
    assert "Method" in assets.tables[0].html
    assert "We propose a compact model." in assets.text


def test_translate_bundle_writes_chinese_documents_and_media_metadata(tmp_path):
    bundle = write_parsed_bundle(tmp_path)
    translator = FakeTranslator()

    artifact = translate_bundle(bundle, translator)

    payload = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    assert payload["title_zh"].startswith("中文[title]")
    assert payload["summary_zh"].startswith("中文[abstract]")
    assert payload["figure_count"] == 1
    assert payload["table_count"] == 1
    assert payload["figures"][0]["caption_zh"].startswith("中文[figure_caption]")
    assert "<div" not in payload["figures"][0]["caption_zh"]
    assert payload["tables"][0]["html_zh"].startswith("<table>")
    assert payload["explainer"]["paper_info"]["title"] == "Efficient Model"
    assert artifact.explainer_path.is_file()
    explainer = json.loads(artifact.explainer_path.read_text(encoding="utf-8"))
    assert explainer["experiments"]["main_results"][0]["value"] == "28.4 BLEU"
    chinese_markdown = artifact.markdown_path.read_text(encoding="utf-8")
    assert "# 中文[title]" in chinese_markdown
    assert "中文[figure_caption]" in chinese_markdown
    digest = artifact.digest_path.read_text(encoding="utf-8")
    assert "# 深读稿" in digest
    assert "28.4 BLEU" in digest


def test_translate_bundle_reuses_matching_translation(tmp_path):
    bundle = write_parsed_bundle(tmp_path)
    translator = FakeTranslator()
    translate_bundle(bundle, translator)
    call_count = len(translator.calls)

    translate_bundle(bundle, translator)

    assert len(translator.calls) == call_count


def test_translate_bundle_rejects_malformed_explainer_json(tmp_path):
    bundle = write_parsed_bundle(tmp_path)

    class MalformedExplainerTranslator(FakeTranslator):
        def translate(self, text, *, kind="text"):
            if kind == "explainer":
                self.calls.append((kind, text))
                return "not-json"
            return super().translate(text, kind=kind)

    translator = MalformedExplainerTranslator()

    with pytest.raises(TranslationError, match="structured explainer"):
        translate_bundle(bundle, translator)

    assert not (bundle / "explainer.zh.json").exists()
    assert (bundle / "translation.json").is_file()
    assert (bundle / "paper.zh.md").is_file()


def test_translate_bundle_reuses_full_translation_when_only_explainer_missing(
    tmp_path,
):
    bundle = write_parsed_bundle(tmp_path)
    failed = FakeTranslator(explainer_response="not-json")
    with pytest.raises(TranslationError, match="structured explainer"):
        translate_bundle(bundle, failed)

    retry = FakeTranslator()
    artifact = translate_bundle(bundle, retry)

    assert [kind for kind, _ in retry.calls] == ["explainer"]
    assert artifact.explainer_path.is_file()
    payload = json.loads(artifact.json_path.read_text(encoding="utf-8"))
    assert payload["explainer"]["paper_info"]["title"] == "Efficient Model"


def test_openai_translator_posts_chat_completion_and_returns_content(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "中文翻译"}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr("paper2voice.translation.requests.post", fake_post)
    translator = OpenAICompatibleTranslator(
        api_url="https://example.test/v1",
        api_key="secret",
        model="test-model",
    )

    result = translator.translate("Hello", kind="abstract")

    assert result == "中文翻译"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["kwargs"]["json"]["model"] == "test-model"
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer secret"


def test_openai_translator_rejects_missing_api_key():
    translator = OpenAICompatibleTranslator(
        api_url="https://example.test/v1",
        api_key="",
        model="test-model",
    )

    with pytest.raises(TranslationError, match="API key"):
        translator.translate("Hello")
