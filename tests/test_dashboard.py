import json
import threading
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from PIL import Image

from paper2voice.arxiv import PaperMetadata
from paper2voice.dashboard import create_server, resolve_bundle_source
from paper2voice.settings import AppSettings, load_settings, save_settings
from paper2voice.translation import OpenAICompatibleTranslator
from paper2voice.video import build_video_plan, get_video_theme, render_scene


def test_settings_round_trip_preserves_theme_modules_and_schedule(tmp_path):
    path = tmp_path / "settings.json"
    settings = AppSettings(
        theme="tech",
        modules=["method", "experiment"],
        bundle_path=r"E:\papers\2501.12345",
        schedule_enabled=True,
        schedule_time="21:15",
        weekdays=["mon", "wed", "fri"],
        tts="edge",
        max_scenes=7,
        subtitles=False,
    )

    save_settings(path, settings)
    loaded = load_settings(path)

    assert loaded == settings


def test_next_run_uses_selected_weekday_and_timezone():
    settings = AppSettings(
        schedule_enabled=True,
        schedule_time="09:30",
        weekdays=["mon", "tue", "wed", "thu", "fri"],
    )
    now = datetime(2026, 11, 7, 8, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    next_run = settings.next_run(now)

    assert next_run == datetime(2026, 11, 9, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_settings_reject_invalid_time_and_unknown_theme():
    with pytest.raises(ValueError, match="HH:MM"):
        AppSettings(schedule_time="25:90").validate()
    with pytest.raises(ValueError, match="theme"):
        AppSettings(theme="purple").validate()


def test_tech_theme_changes_rendered_background(tmp_path):
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "paper": {
            "title": "Theme Test",
            "summary": "A paper about testing.",
            "authors": ["Ada"],
            "categories": ["cs.AI"],
        },
        "assets": {},
    }
    (paper_dir / "paper.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    scene = build_video_plan(paper_dir)[0]

    light_output = tmp_path / "light.png"
    tech_output = tmp_path / "tech.png"
    render_scene(
        scene,
        paper_dir=paper_dir,
        output_path=light_output,
        width=640,
        height=360,
        theme=get_video_theme("light_green"),
    )
    render_scene(
        scene,
        paper_dir=paper_dir,
        output_path=tech_output,
        width=640,
        height=360,
        theme=get_video_theme("tech"),
    )

    with Image.open(light_output) as light, Image.open(tech_output) as tech:
        assert light.convert("RGB").getpixel((0, 0)) == (242, 249, 243)
        assert tech.convert("RGB").getpixel((0, 0)) == (7, 19, 26)


def test_dashboard_api_persists_settings(tmp_path):
    settings_path = tmp_path / "settings.json"
    server = create_server(
        settings_path=settings_path,
        host="127.0.0.1",
        port=0,
        start_scheduler=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(f"{base_url}/api/settings") as response:
            initial = json.loads(response.read())
        assert initial["theme"] == "cinematic"

        payload = json.dumps(
            {
                **initial,
                "theme": "tech",
                "schedule_time": "06:45",
                "modules": ["paper_info", "innovation", "method"],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/api/settings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            saved = json.loads(response.read())
        assert saved["theme"] == "tech"

        with urllib.request.urlopen(f"{base_url}/api/settings") as response:
            persisted = json.loads(response.read())
        assert persisted["schedule_time"] == "06:45"
        assert load_settings(settings_path).theme == "tech"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_arxiv_url_is_materialized_before_video_generation(tmp_path):
    paper = PaperMetadata(
        arxiv_id="2610.30340",
        version=1,
        title="URL Paper",
        summary="A paper supplied through an arXiv URL.",
        authors=["Ada"],
        published="2026-10-30T00:00:00Z",
        updated="2026-10-30T00:00:00Z",
        categories=["cs.AI"],
        pdf_url="https://arxiv.org/pdf/2610.30340v1",
        abstract_url="https://arxiv.org/abs/2610.30340v1",
    )
    client = type("FakeClient", (), {"fetch": lambda self, _: paper})()

    bundle_dir = resolve_bundle_source(
        "https://arxiv.org/abs/2610.30340",
        papers_root=tmp_path / "papers",
        arxiv_client=client,
        repository_fetch_bytes=lambda _: b"%PDF-1.7\nfake\n%%EOF",
    )

    assert bundle_dir == tmp_path / "papers" / "2610.30340"
    assert (bundle_dir / "paper.json").is_file()
    assert (bundle_dir / "paper.pdf").is_file()


def test_dashboard_redacts_llm_key_and_preserves_it_when_blank(tmp_path):
    path = tmp_path / "settings.json"
    server = create_server(
        settings_path=path,
        host="127.0.0.1",
        port=0,
        start_scheduler=False,
    )
    state = server.dashboard_state
    settings = AppSettings(
        translation_enabled=True,
        llm_api_url="https://api.deepseek.com/v1",
        llm_model="deepseek-chat",
        llm_api_key="secret-key",
    )
    state.update_settings(settings.to_dict())

    snapshot = state.snapshot()
    assert "llm_api_key" not in snapshot
    assert snapshot["llm_api_key_set"] is True

    payload = settings.to_dict()
    payload["llm_api_key"] = ""
    state.update_settings(payload)
    assert state.get_settings().llm_api_key == "secret-key"
    server.server_close()


def test_default_translation_limit_means_full_document():
    settings = AppSettings()
    assert settings.translation_max_chars == 0
    assert settings.theme == "cinematic"
    assert settings.voice_clone_enabled is True
    assert settings.intro_video == "mode/开头.mp4"
    assert settings.outro_video == "mode/结尾.mp4"
    assert settings.video_mode == "long"


def test_settings_reject_unknown_video_mode():
    with pytest.raises(ValueError, match="video mode"):
        AppSettings(video_mode="unknown").validate()


def test_settings_migrate_legacy_video_modules_to_five_chapters():
    settings = AppSettings.from_dict(
        {
            **AppSettings().to_dict(),
            "modules": ["problem", "method", "result", "figure", "takeaway"],
        }
    )

    assert settings.modules == [
        "innovation",
        "method",
        "experiment",
        "summary",
    ]


def test_dashboard_page_exposes_five_deep_explainer_chapters(tmp_path):
    server = create_server(
        settings_path=tmp_path / "settings.json",
        host="127.0.0.1",
        port=0,
        start_scheduler=False,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/"
        ) as response:
            html = response.read().decode("utf-8")
        for module_id in (
            "paper_info",
            "innovation",
            "method",
            "experiment",
            "summary",
        ):
            assert f'name="modules" value="{module_id}"' in html
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_dashboard_tests_llm_configuration_with_stored_key(tmp_path, monkeypatch):
    server = create_server(
        settings_path=tmp_path / "settings.json",
        host="127.0.0.1",
        port=0,
        start_scheduler=False,
    )
    state = server.dashboard_state
    state.update_settings(
        AppSettings(
            translation_enabled=True,
            llm_api_url="https://example.test/v1",
            llm_model="test-model",
            llm_api_key="stored-secret",
        ).to_dict()
    )
    captured = {}

    def fake_translate(self, text, *, kind="text"):
        captured["key"] = self.api_key
        captured["kind"] = kind
        return "OK"

    monkeypatch.setattr(OpenAICompatibleTranslator, "translate", fake_translate)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps(
            {
                "llm_api_url": "https://example.test/v1",
                "llm_model": "test-model",
                "llm_api_key": "",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/api/test-llm",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())

        assert result["ok"] is True
        assert result["model"] == "test-model"
        assert result["response"] == "OK"
        assert captured == {"key": "stored-secret", "kind": "configuration_test"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
