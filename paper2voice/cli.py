from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from paper2voice.arxiv import ArxivClient, ArxivError
from paper2voice.cosyvoice_tts import CosyVoiceError, CosyVoiceTtsEngine
from paper2voice.dashboard import serve_dashboard
from paper2voice.parser import (
    DeferredOcrEngine,
    OcrEngine,
    ParsingError,
    SubprocessPaddleOcrEngine,
    parse_bundle,
)
from paper2voice.pulse import load_pulse_json
from paper2voice.store import PaperRepository
from paper2voice.translation import (
    OpenAICompatibleTranslator,
    TranslationError,
    translate_bundle,
)
from paper2voice.video import (
    EdgeTtsEngine,
    SapiTtsEngine,
    TtsEngine,
    VideoError,
    create_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper2voice")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Fetch one paper from the arXiv API")
    fetch.add_argument("arxiv_id", help="arXiv ID or abstract/PDF URL")
    fetch.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers"),
        help="Root directory for normalized paper bundles",
    )
    fetch.add_argument(
        "--no-download",
        action="store_true",
        help="Write metadata without downloading paper.pdf",
    )
    fetch.add_argument("--force", action="store_true", help="Replace an existing PDF")

    pulse = subparsers.add_parser(
        "import-pulse", help="Import arxiv-pulse all/top JSON records"
    )
    pulse.add_argument("json_path", type=Path, help="arxiv-pulse JSON file")
    pulse.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers"),
        help="Root directory for normalized paper bundles",
    )
    pulse.add_argument(
        "--download",
        action="store_true",
        help="Download each paper.pdf after importing metadata",
    )
    pulse.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Import at most this many records",
    )
    pulse.add_argument("--force", action="store_true", help="Replace existing PDFs")

    parse = subparsers.add_parser(
        "parse", help="Parse paper.pdf into page JSON, Markdown, and image assets"
    )
    parse.add_argument(
        "bundle_dir", type=Path, help="Normalized paper bundle directory"
    )
    parse.add_argument(
        "--pipeline-version",
        default="v1.6",
        help="PaddleOCR-VL pipeline version",
    )
    parse.add_argument(
        "--force",
        action="store_true",
        help="Reparse even when source hashes and engine version match",
    )

    translate = subparsers.add_parser(
        "translate", help="Translate a parsed paper with an OpenAI-compatible LLM"
    )
    translate.add_argument("bundle_dir", type=Path)
    translate.add_argument(
        "--api-url",
        default=os.environ.get(
            "PAPER2VOICE_LLM_API_URL",
            "https://api.deepseek.com/v1",
        ),
    )
    translate.add_argument(
        "--api-key",
        default=os.environ.get(
            "PAPER2VOICE_LLM_API_KEY",
            os.environ.get("DEEPSEEK_API_KEY", ""),
        ),
    )
    translate.add_argument(
        "--model",
        default=os.environ.get("PAPER2VOICE_LLM_MODEL", "deepseek-chat"),
    )
    translate.add_argument(
        "--max-document-chars",
        type=int,
        default=0,
        help="Maximum document characters; 0 translates the full document",
    )

    video = subparsers.add_parser(
        "video", help="Generate narration, slides, subtitles, and final MP4"
    )
    video.add_argument("bundle_dir", type=Path, help="Parsed paper bundle directory")
    video.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Video output directory (default: <bundle>/video)",
    )
    video.add_argument(
        "--tts",
        choices=("sapi", "edge", "cosyvoice"),
        default="sapi" if sys.platform == "win32" else "edge",
        help="Text-to-speech provider",
    )
    video.add_argument("--voice", default=None, help="TTS voice name")
    video.add_argument(
        "--reference-video",
        type=Path,
        default=None,
        help="Voice reference video for CosyVoice3 (default: mode/voice-reference.wav)",
    )
    video.add_argument("--intro-video", type=Path, default=None)
    video.add_argument("--outro-video", type=Path, default=None)
    video.add_argument("--max-scenes", type=int, default=18)
    video.add_argument(
        "--video-mode",
        choices=("long", "promo"),
        default="long",
        help="Long explainer or 60-second social promo",
    )
    video.add_argument("--width", type=int, default=1920)
    video.add_argument("--height", type=int, default=1080)
    video.add_argument("--fps", type=int, default=30)
    video.add_argument(
        "--theme",
        choices=("cinematic", "light_green", "minimal", "tech"),
        default="cinematic",
        help="Video color theme",
    )
    video.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Write the SRT without burning subtitles into the video",
    )

    dashboard = subparsers.add_parser(
        "dashboard", help="Open the local video scheduling dashboard"
    )
    dashboard.add_argument(
        "--settings",
        type=Path,
        default=Path("data/video-settings.json"),
        help="Settings JSON path",
    )
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--no-browser", action="store_true")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    arxiv_client: ArxivClient | None = None,
    parser_engine: OcrEngine | None = None,
    tts_engine: TtsEngine | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "dashboard":
            serve_dashboard(
                settings_path=args.settings,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
            return 0

        if args.command == "parse":
            engine = parser_engine or DeferredOcrEngine(
                SubprocessPaddleOcrEngine.name,
                args.pipeline_version,
                lambda: SubprocessPaddleOcrEngine(args.pipeline_version),
            )
            artifact = parse_bundle(args.bundle_dir, engine, force=args.force)
            print(f"Parsed {artifact.bundle_dir}: {artifact.json_path}")
            return 0

        if args.command == "translate":
            if not args.api_key:
                raise TranslationError(
                    "Set --api-key or PAPER2VOICE_LLM_API_KEY / DEEPSEEK_API_KEY"
                )
            artifact = translate_bundle(
                args.bundle_dir,
                OpenAICompatibleTranslator(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                ),
                max_document_chars=args.max_document_chars,
            )
            print(f"Translated {artifact.bundle_dir}: {artifact.markdown_path}")
            return 0

        if args.command == "video":
            engine = tts_engine or _build_tts_engine(
                args.tts,
                args.voice,
                args.reference_video,
            )
            artifact = create_video(
                args.bundle_dir,
                output_dir=args.output_dir,
                tts_engine=engine,
                max_scenes=args.max_scenes,
                width=args.width,
                height=args.height,
                fps=args.fps,
                burn_subtitles=not args.no_subtitles,
                theme=args.theme,
                video_mode=args.video_mode,
                intro_video=args.intro_video,
                outro_video=args.outro_video,
            )
            print(f"Created video {artifact.final_path}")
            return 0

        repository = PaperRepository(args.output_dir)
        if args.command == "fetch":
            client = arxiv_client or ArxivClient()
            paper = client.fetch(args.arxiv_id)
            artifact = repository.ingest(
                paper,
                download_pdf=not args.no_download,
                force=args.force,
            )
            print(f"Fetched {paper.full_id}: {artifact.manifest_path}")
            return 0

        papers = load_pulse_json(args.json_path)
        if args.limit is not None:
            papers = papers[: args.limit]
        for paper in papers:
            artifact = repository.ingest(
                paper,
                download_pdf=args.download,
                force=args.force,
            )
            print(f"Imported {paper.full_id}: {artifact.manifest_path}")
        print(f"Imported {len(papers)} paper(s) into {repository.root}")
        return 0
    except (
        ArxivError,
        CosyVoiceError,
        ParsingError,
        TranslationError,
        VideoError,
        OSError,
        ValueError,
    ) as exc:
        print(f"paper2voice: {exc}")
        return 1


def _build_tts_engine(
    provider: str,
    voice: str | None,
    reference_video: str | Path | None = None,
) -> TtsEngine:
    if provider == "cosyvoice":
        reference = reference_video or Path("mode/voice-reference.wav")
        return CosyVoiceTtsEngine(reference)
    if provider == "edge":
        return EdgeTtsEngine(voice or "zh-CN-XiaoxiaoNeural")
    return SapiTtsEngine(voice or "Microsoft Huihui Desktop")
