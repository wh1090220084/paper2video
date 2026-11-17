import copy
import json

import pytest

from paper2voice.explainer import (
    ExplainerError,
    load_explainer,
    validate_explainer_payload,
)


def valid_payload():
    return {
        "schema_version": 1,
        "audience": "ai_graduate_developer",
        "paper_info": {
            "title": "A Better Model",
            "problem": "Existing systems are slow.",
            "authors": ["Ada Lovelace"],
            "institutions": ["Analytical Engine Lab"],
            "venue": "TestConf 2026",
            "published": "2026-01-01",
            "links": {
                "arxiv": "https://arxiv.org/abs/2601.00001",
                "doi": "10.0000/example",
                "code": "https://example.test/code",
                "project": "https://example.test/project",
            },
        },
        "innovations": [
            {
                "title": "Compact routing",
                "summary": "Routes tokens with a smaller learned controller.",
                "evidence_pages": [2],
            }
        ],
        "method": {
            "overview": "The model maps inputs to compact representations.",
            "steps": [
                {
                    "title": "Encode",
                    "detail": "The encoder produces latent states.",
                    "evidence_pages": [3],
                }
            ],
            "formulas": [
                {
                    "latex": r"y = f(x)",
                    "explanation": "The model maps input x to output y.",
                    "evidence_pages": [3],
                }
            ],
        },
        "experiments": {
            "setup": "Three benchmark datasets and four baselines.",
            "main_results": [
                {
                    "claim": "The model reaches 28.4 BLEU.",
                    "value": "28.4 BLEU",
                    "evidence_pages": [7],
                }
            ],
            "ablations": [],
            "visual_figures": ["parse_assets/result.png"],
        },
        "summary": {
            "value": "The method makes deployment cheaper.",
            "applications": ["Edge inference"],
            "limitations": ["Only evaluated on text tasks."],
            "conclusion": "A practical efficiency improvement.",
        },
    }


def test_validate_explainer_payload_accepts_five_chapters():
    validated = validate_explainer_payload(valid_payload())

    assert validated["paper_info"]["title"] == "A Better Model"
    assert validated["paper_info"]["links"]["code"] == "https://example.test/code"
    assert len(validated["innovations"]) == 1
    assert validated["method"]["formulas"][0]["latex"] == "y = f(x)"
    assert validated["experiments"]["main_results"][0]["value"] == "28.4 BLEU"


def test_validate_explainer_payload_accepts_figure_objects():
    payload = valid_payload()
    payload["experiments"]["visual_figures"] = [
        {
            "path": "parse_assets/figure.png",
            "caption": "Architecture and results",
        },
        "{'path': 'parse_assets/result.png', 'caption': 'Results'}",
    ]

    validated = validate_explainer_payload(payload)

    assert validated["experiments"]["visual_figures"] == [
        "parse_assets/figure.png",
        "parse_assets/result.png",
    ]


def test_validate_explainer_payload_normalizes_missing_metadata():
    payload = valid_payload()
    payload["paper_info"]["authors"] = []
    payload["paper_info"]["institutions"] = []
    payload["paper_info"]["links"] = {}
    payload["paper_info"].pop("venue")

    validated = validate_explainer_payload(payload)

    assert validated["paper_info"]["authors"] == ["未提供"]
    assert validated["paper_info"]["institutions"] == ["未提供"]
    assert validated["paper_info"]["venue"] == "未提供"
    assert validated["paper_info"]["links"] == {
        "arxiv": "未提供",
        "doi": "未提供",
        "code": "未提供",
        "project": "未提供",
    }


def test_validate_explainer_payload_rejects_more_than_three_formulas():
    payload = valid_payload()
    formula = payload["method"]["formulas"][0]
    payload["method"]["formulas"] = [
        copy.deepcopy(formula),
        copy.deepcopy(formula),
        copy.deepcopy(formula),
        copy.deepcopy(formula),
    ]

    with pytest.raises(ExplainerError, match="at most three formulas"):
        validate_explainer_payload(payload)


def test_validate_explainer_payload_rejects_numeric_claim_without_evidence():
    payload = valid_payload()
    payload["experiments"]["main_results"][0]["evidence_pages"] = []

    with pytest.raises(ExplainerError, match="evidence_pages"):
        validate_explainer_payload(payload)


def test_load_explainer_reads_and_validates_bundle_file(tmp_path):
    bundle_dir = tmp_path / "paper"
    bundle_dir.mkdir()
    (bundle_dir / "explainer.zh.json").write_text(
        json.dumps(valid_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    loaded = load_explainer(bundle_dir)

    assert loaded["paper_info"]["title"] == "A Better Model"
