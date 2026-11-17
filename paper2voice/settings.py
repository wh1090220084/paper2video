from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

THEME_IDS = ("cinematic", "light_green", "minimal", "tech")
MODULE_IDS = ("paper_info", "innovation", "method", "experiment", "summary")
TTS_IDS = ("sapi", "edge", "cosyvoice")
WEEKDAY_IDS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
TIMEZONE = ZoneInfo("Asia/Shanghai")
LEGACY_MODULE_MAP = {
    "problem": "innovation",
    "method": "method",
    "result": "experiment",
    "figure": "experiment",
    "takeaway": "summary",
}


@dataclass
class AppSettings:
    theme: str = "cinematic"
    video_mode: str = "long"
    modules: list[str] = field(
        default_factory=lambda: [
            "paper_info",
            "innovation",
            "method",
            "experiment",
            "summary",
        ]
    )
    bundle_path: str = ""
    output_dir: str = ""
    schedule_enabled: bool = True
    schedule_time: str = "08:30"
    weekdays: list[str] = field(
        default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"]
    )
    tts: str = "sapi"
    voice: str = ""
    max_scenes: int = 18
    width: int = 1280
    height: int = 720
    fps: int = 24
    subtitles: bool = True
    ocr_pipeline_version: str = "v1.6"
    translation_enabled: bool = False
    llm_api_url: str = field(
        default_factory=lambda: os.environ.get(
            "PAPER2VOICE_LLM_API_URL",
            "https://api.deepseek.com/v1",
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "PAPER2VOICE_LLM_MODEL",
            "deepseek-chat",
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "PAPER2VOICE_LLM_API_KEY",
            os.environ.get("DEEPSEEK_API_KEY", ""),
        )
    )
    translation_max_chars: int = 0
    voice_clone_enabled: bool = True
    xtts_license_accepted: bool = False
    intro_video: str = "mode/开头.mp4"
    outro_video: str = "mode/结尾.mp4"
    voice_reference_video: str = ""

    def validate(self) -> None:
        if self.theme not in THEME_IDS:
            raise ValueError(f"Unknown theme: {self.theme}")
        if self.video_mode not in {"long", "promo"}:
            raise ValueError(f"Unknown video mode: {self.video_mode}")
        if not self.modules or any(module not in MODULE_IDS for module in self.modules):
            raise ValueError("At least one valid content module is required")
        if self.tts not in TTS_IDS:
            raise ValueError(f"Unknown TTS provider: {self.tts}")
        if not _valid_time(self.schedule_time):
            raise ValueError("Schedule time must use HH:MM")
        if not self.weekdays or any(day not in WEEKDAY_IDS for day in self.weekdays):
            raise ValueError("At least one valid weekday is required")
        if not 1 <= self.max_scenes <= 18:
            raise ValueError("max_scenes must be between 1 and 18")
        if self.width < 320 or self.height < 180:
            raise ValueError("Video dimensions are too small")
        if not 12 <= self.fps <= 60:
            raise ValueError("fps must be between 12 and 60")
        if self.translation_enabled and not self.llm_api_url.strip():
            raise ValueError("LLM API URL is required when translation is enabled")
        if self.translation_enabled and not self.llm_model.strip():
            raise ValueError("LLM model is required when translation is enabled")
        if self.translation_max_chars < 0:
            raise ValueError("translation_max_chars cannot be negative")
        if self.voice_clone_enabled and not (
            self.voice_reference_video.strip() or self.intro_video.strip()
        ):
            raise ValueError("A voice reference or intro video is required")

    def to_dict(self) -> dict:
        return asdict(self)

    def public_dict(self) -> dict:
        payload = self.to_dict()
        payload.pop("llm_api_key", None)
        payload["llm_api_key_set"] = bool(self.llm_api_key)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> AppSettings:
        allowed = cls.__dataclass_fields__
        incoming = dict(payload)
        if isinstance(incoming.get("modules"), list):
            modules = []
            for module in incoming["modules"]:
                migrated = LEGACY_MODULE_MAP.get(module, module)
                if migrated in MODULE_IDS and migrated not in modules:
                    modules.append(migrated)
            incoming["modules"] = modules
        settings = cls(
            **{key: value for key, value in incoming.items() if key in allowed}
        )
        settings.validate()
        return settings

    def next_run(self, now: datetime | None = None) -> datetime | None:
        if not self.schedule_enabled:
            return None
        current = (now or datetime.now(tz=TIMEZONE)).astimezone(TIMEZONE)
        hour, minute = (int(part) for part in self.schedule_time.split(":"))
        for offset in range(8):
            day = current.date() + timedelta(days=offset)
            if WEEKDAY_IDS[day.weekday()] not in self.weekdays:
                continue
            candidate = datetime(
                day.year,
                day.month,
                day.day,
                hour,
                minute,
                tzinfo=TIMEZONE,
            )
            if candidate > current:
                return candidate
        return None


def load_settings(path: str | Path) -> AppSettings:
    settings_path = Path(path)
    if not settings_path.is_file():
        return AppSettings()
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read settings: {settings_path}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Settings must be a JSON object")
    return AppSettings.from_dict(payload)


def save_settings(path: str | Path, settings: AppSettings) -> None:
    settings.validate()
    settings_path = Path(path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = settings_path.with_name(f"{settings_path.name}.part")
    try:
        temp_path.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, settings_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _valid_time(value: str) -> bool:
    if len(value) != 5 or value[2] != ":":
        return False
    try:
        hour, minute = (int(part) for part in value.split(":"))
    except ValueError:
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59
