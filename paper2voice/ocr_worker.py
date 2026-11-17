from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2voice.parser import PaddleOcrVlEngine, _normalize_page


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--pipeline-version", default="v1.6")
    parser.add_argument("--page-offset", type=int, default=0)
    parser.add_argument("--total-pages", type=int)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    assets_dir = args.assets_dir
    assets_dir.mkdir(parents=True, exist_ok=True)
    engine = PaddleOcrVlEngine(args.pipeline_version)
    results = list(engine.predict(args.pdf))
    pages = [
        _normalize_page(
            result,
            fallback_index=index,
            assets_dir=assets_dir,
            page_index_offset=args.page_offset,
            page_count_override=args.total_pages,
        )
        for index, result in enumerate(results)
    ]
    args.output_json.write_text(
        json.dumps({"pages": pages}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
