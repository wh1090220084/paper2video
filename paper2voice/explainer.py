from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path
from typing import Any


class ExplainerError(RuntimeError):
    """Raised when structured explainer content is missing or invalid."""


def validate_explainer_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ExplainerError("Explainer payload must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ExplainerError("Unsupported explainer schema version")
    if payload.get("audience") != "ai_graduate_developer":
        raise ExplainerError("Explainer audience must be ai_graduate_developer")

    normalized = copy.deepcopy(payload)
    paper_info = _mapping(normalized.get("paper_info"), "paper_info")
    links = _mapping(paper_info.get("links"), "paper_info.links", required=False)
    paper_info["title"] = _text(paper_info.get("title"))
    paper_info["problem"] = _text(paper_info.get("problem"))
    paper_info["authors"] = _text_list(paper_info.get("authors"))
    paper_info["institutions"] = _text_list(paper_info.get("institutions"))
    paper_info["venue"] = _text(paper_info.get("venue"))
    paper_info["published"] = _text(paper_info.get("published"))
    paper_info["links"] = {
        key: _text(links.get(key)) for key in ("arxiv", "doi", "code", "project")
    }
    normalized["paper_info"] = paper_info

    raw_innovations = normalized.get("innovations")
    if not isinstance(raw_innovations, list) or not raw_innovations:
        raise ExplainerError("At least one innovation is required")
    if len(raw_innovations) > 3:
        raise ExplainerError("At most three innovations are supported")
    normalized["innovations"] = [
        {
            "title": _text(item.get("title")),
            "summary": _text(item.get("summary")),
            "evidence_pages": _evidence_pages(
                item.get("evidence_pages"),
                f"innovations[{index}].evidence_pages",
            ),
        }
        for index, item in enumerate(
            _mapping(item, f"innovations[{index}]")
            for index, item in enumerate(raw_innovations)
        )
    ]

    method = _mapping(normalized.get("method"), "method")
    method["overview"] = _text(method.get("overview"))
    method["steps"] = [
        {
            "title": _text(item.get("title")),
            "detail": _text(item.get("detail")),
            "evidence_pages": _evidence_pages(
                item.get("evidence_pages"),
                f"method.steps[{index}].evidence_pages",
            ),
        }
        for index, item in enumerate(
            _mapping(item, f"method.steps[{index}]")
            for index, item in enumerate(method.get("steps") or [])
        )
    ]
    formulas = method.get("formulas") or []
    if not isinstance(formulas, list):
        raise ExplainerError("method.formulas must be a list")
    if len(formulas) > 3:
        raise ExplainerError("at most three formulas are supported")
    method["formulas"] = [
        {
            "latex": _text(item.get("latex")),
            "explanation": _text(item.get("explanation")),
            "evidence_pages": _evidence_pages(
                item.get("evidence_pages"),
                f"method.formulas[{index}].evidence_pages",
            ),
        }
        for index, item in enumerate(
            _mapping(item, f"method.formulas[{index}]")
            for index, item in enumerate(formulas)
        )
    ]
    normalized["method"] = method

    experiments = _mapping(normalized.get("experiments"), "experiments")
    experiments["setup"] = _text(experiments.get("setup"))
    experiments["main_results"] = _result_list(
        experiments.get("main_results"),
        "experiments.main_results",
    )
    experiments["ablations"] = _result_list(
        experiments.get("ablations"),
        "experiments.ablations",
    )
    experiments["visual_figures"] = _figure_paths(experiments.get("visual_figures"))
    normalized["experiments"] = experiments

    summary = _mapping(normalized.get("summary"), "summary")
    summary["value"] = _text(summary.get("value"))
    summary["applications"] = _text_list(summary.get("applications"))
    summary["limitations"] = _text_list(summary.get("limitations"))
    summary["conclusion"] = _text(summary.get("conclusion"))
    normalized["summary"] = summary
    return normalized


def load_explainer(bundle_dir: str | Path) -> dict:
    path = Path(bundle_dir) / "explainer.zh.json"
    if not path.is_file():
        raise ExplainerError(
            f"Missing structured explainer: {path}; enable translation first"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExplainerError(f"Cannot read structured explainer: {path}") from exc
    return validate_explainer_payload(payload)


def _result_list(value: Any, field: str) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ExplainerError(f"{field} must be a list")
    results = []
    for index, item in enumerate(value):
        item = _mapping(item, f"{field}[{index}]")
        evidence_pages = _evidence_pages(
            item.get("evidence_pages"),
            f"{field}[{index}].evidence_pages",
            required=bool(re.search(r"\d", _text(item.get("value"), default="")))
            or bool(re.search(r"\d", _text(item.get("claim"), default=""))),
        )
        results.append(
            {
                "claim": _text(item.get("claim")),
                "value": _text(item.get("value")),
                "evidence_pages": evidence_pages,
            }
        )
    return results


def _mapping(value: Any, field: str, *, required: bool = True) -> dict:
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ExplainerError(f"{field} must be an object")
    return value


def _text(value: Any, *, default: str = "未提供") -> str:
    text = str(value or "").strip()
    return text or default


def _text_list(value: Any, *, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or ["未提供"])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ExplainerError("Expected a list of strings")
    items = [str(item).strip() for item in value if str(item).strip()]
    return items or list(default or ["未提供"])


def _figure_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    paths = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("path")
        elif isinstance(item, str) and item.strip().startswith("{"):
            try:
                parsed = ast.literal_eval(item)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                item = parsed.get("path")
        path = str(item or "").strip()
        if path:
            paths.append(path)
    return paths


def _evidence_pages(
    value: Any,
    field: str,
    *,
    required: bool = False,
) -> list[int]:
    if value is None:
        pages = []
    elif not isinstance(value, list):
        raise ExplainerError(f"{field} must be a list of page numbers")
    else:
        pages = []
        for page in value:
            if isinstance(page, bool):
                raise ExplainerError(f"{field} must contain positive integers")
            try:
                number = int(page)
            except (TypeError, ValueError) as exc:
                raise ExplainerError(f"{field} must contain positive integers") from exc
            if number < 1:
                raise ExplainerError(f"{field} must contain positive integers")
            pages.append(number)
    if required and not pages:
        raise ExplainerError(f"{field} is required for numeric claims")
    return pages
