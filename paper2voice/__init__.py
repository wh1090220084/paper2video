"""Paper ingestion primitives for paper2voice."""

from paper2voice.arxiv import ArxivClient, PaperMetadata, parse_arxiv_id
from paper2voice.cosyvoice_tts import CosyVoiceTtsEngine
from paper2voice.parser import PaddleOcrVlEngine, parse_bundle
from paper2voice.store import PaperRepository
from paper2voice.video import build_video_plan, create_video

__all__ = [
    "ArxivClient",
    "CosyVoiceTtsEngine",
    "PaddleOcrVlEngine",
    "PaperMetadata",
    "PaperRepository",
    "build_video_plan",
    "create_video",
    "parse_arxiv_id",
    "parse_bundle",
]
