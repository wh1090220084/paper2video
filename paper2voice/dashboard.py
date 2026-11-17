from __future__ import annotations

import json
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from paper2voice.arxiv import ArxivClient
from paper2voice.cosyvoice_tts import CosyVoiceError, CosyVoiceTtsEngine
from paper2voice.parser import (
    DeferredOcrEngine,
    ParsingError,
    SubprocessPaddleOcrEngine,
    parse_bundle,
)
from paper2voice.settings import (
    TIMEZONE,
    WEEKDAY_IDS,
    AppSettings,
    load_settings,
    save_settings,
)
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


class DashboardState:
    def __init__(self, settings_path: str | Path):
        self.settings_path = Path(settings_path)
        self.settings = load_settings(self.settings_path)
        if not self.settings_path.is_file():
            save_settings(self.settings_path, self.settings)
        self._lock = threading.Lock()
        self._job: dict[str, Any] = {
            "state": "idle",
            "message": "等待运行",
            "started_at": None,
            "finished_at": None,
            "final_path": None,
        }
        self._last_scheduled_date = None

    def get_settings(self) -> AppSettings:
        with self._lock:
            return AppSettings.from_dict(self.settings.to_dict())

    def update_settings(self, payload: dict) -> AppSettings:
        incoming = dict(payload)
        current = self.get_settings()
        if not incoming.get("llm_api_key"):
            incoming["llm_api_key"] = current.llm_api_key
        settings = AppSettings.from_dict(incoming)
        with self._lock:
            save_settings(self.settings_path, settings)
            self.settings = settings
        return settings

    def snapshot(self) -> dict:
        settings = self.get_settings()
        with self._lock:
            job = dict(self._job)
        next_run = settings.next_run()
        return {
            **settings.public_dict(),
            "job": job,
            "next_run": next_run.isoformat() if next_run else None,
        }

    def request_run(self) -> bool:
        settings = self.get_settings()
        if not settings.bundle_path.strip():
            raise ValueError("Set a paper bundle directory before running")
        with self._lock:
            if self._job["state"] == "running":
                return False
            self._job = {
                "state": "running",
                "message": "正在生成视频",
                "started_at": datetime.now(tz=TIMEZONE).isoformat(),
                "finished_at": None,
                "final_path": None,
            }
        threading.Thread(
            target=self._run_job,
            args=(settings,),
            daemon=True,
            name="paper2voice-video-job",
        ).start()
        return True

    def test_llm(self, payload: dict) -> dict:
        current = self.get_settings()
        api_url = str(payload.get("llm_api_url") or current.llm_api_url).strip()
        model = str(payload.get("llm_model") or current.llm_model).strip()
        api_key = str(payload.get("llm_api_key") or current.llm_api_key)
        if not api_url or not model or not api_key:
            raise ValueError("API URL, model, and API key are required")
        started = time.monotonic()
        response = OpenAICompatibleTranslator(
            api_url=api_url,
            api_key=api_key,
            model=model,
        ).translate(
            "这是一次连接测试。只回复 OK。",
            kind="configuration_test",
        )
        latency_ms = round((time.monotonic() - started) * 1000)
        return {
            "ok": True,
            "model": model,
            "latency_ms": latency_ms,
            "response": response[:200],
        }

    def scheduler_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(15):
            settings = self.get_settings()
            if not settings.schedule_enabled or not settings.bundle_path.strip():
                continue
            now = datetime.now(tz=TIMEZONE)
            if (
                now.strftime("%H:%M") != settings.schedule_time
                or self._last_scheduled_date == now.date()
            ):
                continue
            if WEEKDAY_IDS[now.weekday()] not in settings.weekdays:
                continue
            try:
                if self.request_run():
                    self._last_scheduled_date = now.date()
            except ValueError:
                continue

    def _run_job(self, settings: AppSettings) -> None:
        try:
            bundle_dir = resolve_bundle_source(settings.bundle_path)
            self._update_job_message("正在使用 PaddleOCR-VL 解析论文")
            parse_bundle(
                bundle_dir,
                DeferredOcrEngine(
                    SubprocessPaddleOcrEngine.name,
                    settings.ocr_pipeline_version,
                    lambda: SubprocessPaddleOcrEngine(settings.ocr_pipeline_version),
                ),
            )
            if settings.translation_enabled:
                self._update_job_message("正在使用大模型翻译论文")
                translate_bundle(
                    bundle_dir,
                    OpenAICompatibleTranslator(
                        api_url=settings.llm_api_url,
                        api_key=settings.llm_api_key,
                        model=settings.llm_model,
                    ),
                    max_document_chars=settings.translation_max_chars,
                )
                self._update_job_message("论文翻译完成")
            self._update_job_message("正在生成视频")
            artifact = create_video(
                bundle_dir,
                output_dir=settings.output_dir or None,
                tts_engine=_tts_engine(settings),
                max_scenes=settings.max_scenes,
                width=settings.width,
                height=settings.height,
                fps=settings.fps,
                burn_subtitles=settings.subtitles,
                theme=settings.theme,
                modules=settings.modules,
                video_mode=settings.video_mode,
                intro_video=settings.intro_video or None,
                outro_video=settings.outro_video or None,
            )
            self._set_job(
                state="success",
                message="视频已生成",
                final_path=str(artifact.final_path),
            )
        except (
            CosyVoiceError,
            ParsingError,
            TranslationError,
            VideoError,
            OSError,
            ValueError,
        ) as exc:
            self._set_job(state="error", message=str(exc))

    def _set_job(
        self,
        *,
        state: str,
        message: str,
        final_path: str | None = None,
    ) -> None:
        with self._lock:
            self._job.update(
                {
                    "state": state,
                    "message": message,
                    "finished_at": datetime.now(tz=TIMEZONE).isoformat(),
                    "final_path": final_path,
                }
            )

    def _update_job_message(self, message: str) -> None:
        with self._lock:
            if self._job["state"] == "running":
                self._job["message"] = message


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html()
            return
        if self.path == "/api/settings":
            self._send_json(self.server.dashboard_state.snapshot())
            return
        if self.path == "/api/status":
            self._send_json(self.server.dashboard_state.snapshot())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/api/settings":
                settings = self.server.dashboard_state.update_settings(payload)
                self._send_json(
                    {
                        **settings.public_dict(),
                        "next_run": settings.next_run().isoformat()
                        if settings.next_run()
                        else None,
                    }
                )
                return
            if self.path == "/api/run":
                started = self.server.dashboard_state.request_run()
                self._send_json(
                    {"started": started},
                    status=HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT,
                )
                return
            if self.path == "/api/test-llm":
                try:
                    result = self.server.dashboard_state.test_llm(payload)
                except TranslationError as exc:
                    self._send_json(
                        {"ok": False, "error": str(exc)},
                        status=HTTPStatus.BAD_GATEWAY,
                    )
                    return
                self._send_json(result)
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_000_000:
            raise ValueError("Invalid request body")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("JSON object required")
        return payload

    def _send_html(self) -> None:
        path = Path(__file__).with_name("web") / "index.html"
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        payload: dict,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], state: DashboardState):
        super().__init__(address, DashboardHandler)
        self.dashboard_state = state
        self.stop_event = threading.Event()
        self.scheduler_thread = threading.Thread(
            target=state.scheduler_loop,
            args=(self.stop_event,),
            daemon=True,
            name="paper2voice-scheduler",
        )

    def server_close(self) -> None:
        self.stop_event.set()
        super().server_close()


def create_server(
    *,
    settings_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    start_scheduler: bool = True,
) -> DashboardServer:
    state = DashboardState(settings_path)
    server = DashboardServer((host, port), state)
    if start_scheduler:
        server.scheduler_thread.start()
    return server


def serve_dashboard(
    *,
    settings_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    server = create_server(
        settings_path=settings_path,
        host=host,
        port=port,
        start_scheduler=True,
    )
    url = f"http://{host}:{server.server_port}"
    print(f"paper2voice dashboard: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _tts_engine(settings: AppSettings) -> TtsEngine:
    if settings.voice_clone_enabled or settings.tts == "cosyvoice":
        reference = settings.voice_reference_video or settings.intro_video
        return CosyVoiceTtsEngine(reference)
    if settings.tts == "edge":
        return EdgeTtsEngine(settings.voice or "zh-CN-XiaoxiaoNeural")
    return SapiTtsEngine(settings.voice or "Microsoft Huihui Desktop")


def resolve_bundle_source(
    source: str,
    *,
    papers_root: str | Path = Path("data/papers"),
    arxiv_client: ArxivClient | None = None,
    repository_fetch_bytes=None,
) -> Path:
    value = source.strip()
    if not value.lower().startswith(("http://", "https://")):
        return Path(value)
    client = arxiv_client or ArxivClient()
    paper = client.fetch(value)
    repository = PaperRepository(
        papers_root,
        fetch_bytes=repository_fetch_bytes,
    )
    return repository.ingest(paper, download_pdf=True).directory
