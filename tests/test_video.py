import json
import shutil
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from paper2voice.cli import main
from paper2voice.explainer import ExplainerError
from paper2voice.video import (
    Metric,
    SapiTtsEngine,
    VideoError,
    XttsTtsEngine,
    build_deep_video_plan,
    build_promo_plan,
    build_video_plan,
    create_video,
    extract_metrics,
    extract_voice_reference,
    probe_duration,
    render_formula_image,
    render_promo_scene,
    render_scene,
    split_tts_text,
    write_srt,
)


class SilentTtsEngine:
    name = "silent-test"
    extension = ".wav"

    def __init__(self, duration=0.3, sample_rate=8000):
        self.texts = []
        self.duration = duration
        self.sample_rate = sample_rate

    def synthesize(self, text, output_path):
        self.texts.append(text)
        output_path = Path(output_path)
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(b"\x00\x00" * int(self.sample_rate * self.duration))
        return output_path


def write_video_bundle(root, *, with_explainer=True):
    paper_dir = root / "2501.12345"
    assets_dir = paper_dir / "parse_assets"
    assets_dir.mkdir(parents=True)
    figure_path = assets_dir / "figure.png"
    Image.new("RGB", (800, 450), (30, 120, 180)).save(figure_path)
    manifest = {
        "schema_version": 1,
        "paper": {
            "arxiv_id": "2501.12345",
            "full_id": "2501.12345v1",
            "title": "A Better Model",
            "summary": (
                "Existing systems are slow and difficult to deploy. "
                "We propose a compact architecture for efficient inference. "
                "Experiments show 28.4 BLEU and outperform strong baselines. "
                "The method should make deployment practical."
            ),
            "authors": ["Ada Lovelace", "Alan Turing"],
            "categories": ["cs.CL"],
        },
        "assets": {"pdf": "paper.pdf", "pdf_url": "https://arxiv.org/pdf/2501.12345v1"},
    }
    parse = {
        "schema_version": 1,
        "source": {},
        "engine": {"name": "test", "version": "1"},
        "pages": [
            {
                "page_index": 0,
                "page_count": 1,
                "images": [{"source": "figure.png", "path": "parse_assets/figure.png"}],
                "raw": {"parsing_res_list": []},
            }
        ],
    }
    (paper_dir / "paper.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.7\nfake\n%%EOF")
    (paper_dir / "parse.json").write_text(
        json.dumps(parse, ensure_ascii=False), encoding="utf-8"
    )
    if with_explainer:
        install_deep_explainer(paper_dir)
    return paper_dir


def install_deep_explainer(paper_dir):
    fixture = Path(__file__).parent / "fixtures" / "deep-explainer.json"
    (paper_dir / "explainer.zh.json").write_text(
        fixture.read_text(encoding="utf-8"),
        encoding="utf-8",
    )


def test_build_video_plan_quotes_problem_method_and_result(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_video_plan(paper_dir, max_scenes=6)

    assert [scene.kind for scene in scenes] == [
        "cover",
        "problem",
        "method",
        "result",
        "figure",
        "takeaway",
    ]
    assert scenes[0].title == "A Better Model"
    assert "A Better Model" in scenes[0].narration
    assert "Existing systems are slow" in scenes[1].body
    assert "We propose a compact architecture" in scenes[2].body
    assert "28.4 BLEU" in scenes[3].body
    assert scenes[4].figure == "parse_assets/figure.png"


def test_build_video_plan_combines_text_scenes_with_paper_figures(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_video_plan(paper_dir, max_scenes=5)

    concept_scenes = [
        scene for scene in scenes if scene.kind in {"problem", "method", "result"}
    ]
    assert concept_scenes
    assert all(scene.figure == "parse_assets/figure.png" for scene in concept_scenes)


def test_build_deep_video_plan_uses_five_chapters_and_fixed_closing(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    install_deep_explainer(paper_dir)

    scenes = build_deep_video_plan(paper_dir)

    assert 16 <= len(scenes) <= 18
    assert [scene.section.split(" / ")[0] for scene in scenes].count("01") == 1
    assert any(scene.section.startswith("02 /") for scene in scenes)
    assert any(scene.section.startswith("03 /") for scene in scenes)
    assert any(scene.section.startswith("04 /") for scene in scenes)
    assert any(scene.section.startswith("05") for scene in scenes)
    assert sum(scene.kind == "formula" for scene in scenes) <= 3
    assert scenes[-1].narration.endswith("今天的论文讲解到此结束。")


def test_build_deep_video_plan_requires_structured_explainer(tmp_path):
    paper_dir = write_video_bundle(tmp_path, with_explainer=False)

    with pytest.raises(ExplainerError, match="explainer.zh.json"):
        build_deep_video_plan(paper_dir)


def test_build_deep_video_plan_limits_long_author_lists_in_narration(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    explainer = json.loads(
        (paper_dir / "explainer.zh.json").read_text(encoding="utf-8")
    )
    explainer["paper_info"]["authors"] = [f"Author {index}" for index in range(1, 13)]
    (paper_dir / "explainer.zh.json").write_text(
        json.dumps(explainer, ensure_ascii=False),
        encoding="utf-8",
    )

    scenes = build_deep_video_plan(paper_dir)

    assert "Author 12" not in scenes[0].narration
    assert "等" in scenes[1].narration


def test_build_deep_video_plan_matches_architecture_and_result_figures(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    architecture = paper_dir / "parse_assets" / "architecture.png"
    results = paper_dir / "parse_assets" / "results.png"
    Image.new("RGB", (800, 450), (40, 90, 180)).save(architecture)
    Image.new("RGB", (800, 450), (190, 80, 50)).save(results)
    parse = json.loads((paper_dir / "parse.json").read_text(encoding="utf-8"))
    parse["pages"][0]["images"] = [
        {"source": "architecture.png", "path": "parse_assets/architecture.png"},
        {"source": "results.png", "path": "parse_assets/results.png"},
    ]
    parse["pages"][0]["raw"]["parsing_res_list"] = [
        {
            "block_label": "figure_title",
            "block_content": "Figure 2: Model architecture and pipeline.",
        },
        {
            "block_label": "figure_title",
            "block_content": "Figure 3: Benchmark results and ablation.",
        },
    ]
    (paper_dir / "parse.json").write_text(
        json.dumps(parse, ensure_ascii=False),
        encoding="utf-8",
    )
    translation = {
        "schema_version": 1,
        "figures": [
            {
                "path": "parse_assets/architecture.png",
                "caption_zh": "图2 模型架构与训练流程。",
            },
            {
                "path": "parse_assets/results.png",
                "caption_zh": "图3 实验结果与消融对比。",
            },
        ],
    }
    (paper_dir / "translation.json").write_text(
        json.dumps(translation, ensure_ascii=False),
        encoding="utf-8",
    )

    scenes = build_deep_video_plan(paper_dir)
    by_id = {scene.scene_id: scene for scene in scenes}

    assert by_id["03-method-overview"].figure == "parse_assets/architecture.png"
    assert by_id["04-result-1"].figure == "parse_assets/results.png"


def test_deep_plan_scene_limit_preserves_method_results_and_summary(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_deep_video_plan(
        paper_dir,
        max_scenes=10,
        modules=["innovation", "method", "experiment", "summary"],
    )

    kinds = {scene.kind for scene in scenes}
    assert len(scenes) == 10
    assert {"innovation", "method", "formula", "experiment", "summary"} <= kinds
    assert any(scene.scene_id == "04-result-1" for scene in scenes)
    assert any(scene.scene_id == "05-value" for scene in scenes)


def test_create_video_long_mode_uses_five_chapter_plan(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "deep-video"

    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(duration=0.05),
        max_scenes=10,
        width=320,
        height=180,
        fps=12,
    )

    plan = json.loads(artifact.plan_path.read_text(encoding="utf-8"))
    kinds = [scene["kind"] for scene in plan["scenes"]]
    assert kinds[0] == "cover"
    assert "innovation" in kinds
    assert "method" in kinds
    assert plan["scenes"][-1]["narration"].endswith("今天的论文讲解到此结束。")


def test_create_video_records_disabled_subtitle_burn(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "clean-subtitles"

    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(duration=0.05),
        max_scenes=1,
        width=320,
        height=180,
        fps=12,
        burn_subtitles=False,
    )

    manifest = json.loads((output_dir / "video.json").read_text(encoding="utf-8"))
    assert manifest["subtitles_burned"] is False
    assert artifact.subtitle_path.is_file()


def test_create_video_long_mode_reports_missing_explainer(tmp_path):
    paper_dir = write_video_bundle(tmp_path, with_explainer=False)

    with pytest.raises(ExplainerError, match="explainer.zh.json"):
        create_video(
            paper_dir,
            output_dir=tmp_path / "missing-explainer",
            tts_engine=SilentTtsEngine(),
            max_scenes=1,
            width=320,
            height=180,
            fps=12,
        )


def test_build_video_plan_respects_selected_content_modules(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_video_plan(paper_dir, modules=["problem", "result"])

    assert [scene.kind for scene in scenes] == ["cover", "problem", "result"]


def test_build_video_plan_prefers_translated_paper_content(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    translation = {
        "schema_version": 1,
        "title_zh": "更高效的模型",
        "summary_zh": (
            "现有系统运行缓慢。我们提出一种紧凑架构。"
            "实验达到 28.4 BLEU。该方法适合实际部署。"
        ),
    }
    (paper_dir / "translation.json").write_text(
        json.dumps(translation, ensure_ascii=False), encoding="utf-8"
    )

    scenes = build_video_plan(paper_dir, max_scenes=5)

    assert scenes[0].title == "更高效的模型"
    assert "现有系统运行缓慢" in scenes[1].body
    assert "紧凑架构" in scenes[2].body


def test_build_video_plan_creates_multiple_figure_scenes(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    second_figure = paper_dir / "parse_assets" / "figure-2.png"
    Image.new("RGB", (400, 300), (190, 80, 60)).save(second_figure)
    parsed = json.loads((paper_dir / "parse.json").read_text(encoding="utf-8"))
    parsed["pages"][0]["images"].append(
        {"source": "figure-2.png", "path": "parse_assets/figure-2.png"}
    )
    parsed["pages"][0]["raw"]["parsing_res_list"] = [
        {"block_label": "figure_title", "block_content": "Figure 1: Architecture."},
        {"block_label": "figure_title", "block_content": "Figure 2: Results."},
    ]
    (paper_dir / "parse.json").write_text(
        json.dumps(parsed, ensure_ascii=False), encoding="utf-8"
    )

    scenes = build_video_plan(paper_dir, max_scenes=7)

    figures = [scene for scene in scenes if scene.kind == "figure"]
    assert len(figures) >= 2
    assert {scene.figure for scene in figures} == {
        "parse_assets/figure.png",
        "parse_assets/figure-2.png",
    }


def test_build_promo_plan_creates_short_social_story(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_promo_plan(paper_dir, max_scenes=6)

    assert [scene.kind for scene in scenes[:3]] == ["hook", "problem", "method"]
    assert scenes[0].title
    assert len(scenes[0].body) <= 80
    assert all(len(scene.narration) <= 120 for scene in scenes)
    assert scenes[-1].kind == "cta"
    assert scenes[-1].narration.endswith("今天的论文讲解到此结束。")


def test_build_promo_plan_uses_paper_figures_as_scene_visuals(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_promo_plan(paper_dir, max_scenes=6)

    assert scenes[0].figure == "parse_assets/figure.png"
    assert scenes[1].figure == "parse_assets/figure.png"
    assert scenes[2].figure == "parse_assets/figure.png"


def test_build_video_plan_adds_closing_voice_to_final_included_scene(tmp_path):
    paper_dir = write_video_bundle(tmp_path)

    scenes = build_video_plan(paper_dir, max_scenes=2)

    assert scenes[-1].narration.endswith("今天的论文讲解到此结束。")


def test_build_promo_plan_uses_distinct_chinese_sentences(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    translation = {
        "schema_version": 1,
        "title_zh": "更高效的模型",
        "summary_zh": (
            "现有系统运行缓慢。我们提出一种紧凑架构。"
            "实验达到 28.4 BLEU。该方法适合实际部署。"
        ),
    }
    (paper_dir / "translation.json").write_text(
        json.dumps(translation, ensure_ascii=False), encoding="utf-8"
    )

    scenes = build_promo_plan(paper_dir, max_scenes=6)
    bodies = {scene.kind: scene.body for scene in scenes}

    assert "运行缓慢" in bodies["problem"]
    assert "紧凑架构" in bodies["method"]
    assert "28.4 BLEU" in bodies["result"]
    assert len({bodies["problem"], bodies["method"], bodies["result"]}) == 3


def test_render_promo_scene_creates_portrait_frame(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    scene = build_promo_plan(paper_dir)[0]
    output = tmp_path / "promo-frame.png"

    render_promo_scene(
        scene,
        paper_dir=paper_dir,
        output_path=output,
        width=540,
        height=720,
        theme="tech",
    )

    image = Image.open(output).convert("RGB")
    assert image.size == (540, 720)
    assert len(image.getcolors(maxcolors=1_000_000)) > 20


def test_render_scene_creates_nonblank_requested_frame(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    scene = build_video_plan(paper_dir)[0]
    output = tmp_path / "frame.png"

    result = render_scene(
        scene,
        paper_dir=paper_dir,
        output_path=output,
        width=640,
        height=360,
    )

    assert result == output
    image = Image.open(output).convert("RGB")
    assert image.size == (640, 360)
    assert len(image.getcolors(maxcolors=1_000_000)) > 20


def test_cinematic_scene_uses_dark_background_and_paper_visual(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    scene = build_video_plan(paper_dir, max_scenes=2)[-1]
    output = tmp_path / "cinematic-frame.png"

    render_scene(
        scene,
        paper_dir=paper_dir,
        output_path=output,
        width=960,
        height=540,
        theme="cinematic",
    )

    image = Image.open(output).convert("RGB")
    corner = image.getpixel((0, 0))
    assert sum(corner) < 120
    assert len(image.getcolors(maxcolors=1_000_000)) > 200


def test_render_formula_image_creates_nonblank_png(tmp_path):
    output = tmp_path / "formula.png"

    render_formula_image(
        r"L = L_{task} + \lambda L_{geo}",
        output,
        width=900,
        height=260,
    )

    image = Image.open(output).convert("RGBA")
    assert image.size == (900, 260)
    assert image.getbbox() is not None


def test_render_formula_image_wraps_long_formula_inside_bounds(tmp_path):
    output = tmp_path / "long-formula.png"
    formula = (
        r"P(Y|I,Q)=\prod_{i=1}^{|Y|}p_\theta(y_i|V,Q,y_{<i}),"
        r"\quad V=C_\phi(I)"
    )

    render_formula_image(formula, output, width=900, height=300)

    image = Image.open(output).convert("RGBA")
    bbox = image.getbbox()
    assert bbox is not None
    assert bbox[0] >= 10
    assert bbox[2] <= 890


def test_tech_formula_scene_overlays_formula_on_figure(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    scenes = build_deep_video_plan(paper_dir)
    formula_scene = next(scene for scene in scenes if scene.kind == "formula")
    plain_output = tmp_path / "plain-formula.png"
    formula_output = tmp_path / "with-formula.png"

    render_scene(
        replace(formula_scene, formula=None),
        paper_dir=paper_dir,
        output_path=plain_output,
        width=1280,
        height=720,
        theme="tech",
    )
    render_scene(
        formula_scene,
        paper_dir=paper_dir,
        output_path=formula_output,
        width=1280,
        height=720,
        theme="tech",
    )

    with (
        Image.open(plain_output).convert("RGB") as plain,
        Image.open(formula_output).convert("RGB") as with_formula,
    ):
        difference = ImageChops.difference(plain, with_formula).convert("L")
        assert sum(difference.histogram()[20:]) > 5_000


def test_extract_metrics_keeps_value_and_units_from_result_sentence():
    metrics = extract_metrics(
        "The model reaches 28.4 BLEU, improving by over 2 BLEU, "
        "and 91.3% accuracy on the benchmark."
    )

    assert metrics == [
        Metric(value="28.4", unit="BLEU"),
        Metric(value="91.3", unit="%"),
    ]


def test_write_srt_uses_cumulative_audio_durations(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    scenes = build_video_plan(paper_dir, max_scenes=2)
    output = tmp_path / "video.srt"

    write_srt(scenes, [2.5, 61.25], output)

    srt = output.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,500" in srt
    assert "00:00:02,500 --> 00:01:03,750" in srt


def test_write_srt_wraps_english_at_word_boundaries(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    scenes = build_video_plan(paper_dir, max_scenes=1)
    scenes[0] = type(scenes[0])(
        scene_id=scenes[0].scene_id,
        kind=scenes[0].kind,
        section=scenes[0].section,
        title=scenes[0].title,
        body=scenes[0].body,
        narration=(
            "Intro abcdefghijklmnopqrstuvwxyz "
            + "supercalifragilisticexpialidocious"
            + " ending."
        ),
    )
    output = tmp_path / "video.srt"

    write_srt(scenes, [3.0], output)

    subtitle_text = "\n".join(output.read_text(encoding="utf-8").splitlines()[2:])
    assert "supercalifragilisticexpialidocious" in subtitle_text
    assert all(len(line) <= 42 for line in subtitle_text.splitlines())


@pytest.mark.skipif(sys.platform != "win32", reason="Windows SAPI only")
def test_sapi_tts_writes_playable_wave(tmp_path):
    output = tmp_path / "speech.wav"

    try:
        result = SapiTtsEngine().synthesize("这是论文转视频的测试旁白。", output)
    except VideoError as exc:
        pytest.skip(f"Windows SAPI is unavailable in this process: {exc}")

    assert result == output
    assert output.read_bytes()[:4] == b"RIFF"
    assert probe_duration(output) > 0.2


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is required",
)
def test_create_video_builds_playable_mp4_from_silent_tts(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "video"
    tts = SilentTtsEngine()

    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=tts,
        max_scenes=2,
        width=640,
        height=360,
        fps=12,
    )

    assert artifact.final_path.is_file()
    assert artifact.subtitle_path.is_file()
    assert artifact.plan_path.is_file()
    assert len(tts.texts) == 2
    assert 0.5 <= probe_duration(artifact.final_path) <= 0.8


def test_cli_video_creates_pipeline_artifacts(tmp_path, capsys):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "cli-video"

    exit_code = main(
        [
            "video",
            str(paper_dir),
            "--output-dir",
            str(output_dir),
            "--max-scenes",
            "2",
            "--width",
            "640",
            "--height",
            "360",
            "--fps",
            "12",
        ],
        tts_engine=SilentTtsEngine(),
    )

    assert exit_code == 0
    assert (output_dir / "final.mp4").is_file()
    assert "Created video" in capsys.readouterr().out


def test_create_video_does_not_pad_long_scene_beyond_audio(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "long-video"

    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(duration=13.7, sample_rate=22050),
        max_scenes=1,
        width=640,
        height=360,
        fps=24,
    )

    assert 13.5 <= probe_duration(artifact.final_path) <= 13.9


def test_create_video_removes_stale_audio_and_merge_temp_files(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "clean-video"
    output_dir.mkdir()
    stale_audio = output_dir / "scene_01.mp3"
    stale_audio.write_bytes(b"")

    create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(),
        max_scenes=1,
        width=640,
        height=360,
        fps=12,
    )

    assert not stale_audio.exists()
    assert not (output_dir / "_merged.mp4").exists()


def test_create_video_animates_static_scene(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "motion-video"
    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(duration=2.4),
        max_scenes=1,
        width=320,
        height=180,
        fps=12,
    )

    early = output_dir / "early.png"
    late = output_dir / "late.png"
    _extract_frame(artifact.final_path, 0.7, early)
    _extract_frame(artifact.final_path, 1.8, late)
    with (
        Image.open(early).convert("RGB") as first,
        Image.open(late).convert("RGB") as second,
    ):
        histogram = ImageChops.difference(first, second).convert("L").histogram()
        difference = sum(intensity * count for intensity, count in enumerate(histogram))

    assert difference > 300


def test_create_video_wraps_body_with_intro_and_outro(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    intro = tmp_path / "intro.mp4"
    outro = tmp_path / "outro.mp4"
    _make_color_audio_clip(intro, color="0x2f8f63")
    _make_color_audio_clip(outro, color="0xc24330")
    output_dir = tmp_path / "wrapped-video"

    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(duration=0.4),
        max_scenes=1,
        width=320,
        height=180,
        fps=12,
        intro_video=intro,
        outro_video=outro,
    )

    assert 1.2 <= probe_duration(artifact.final_path) <= 1.8
    assert artifact.final_path.is_file()


def test_create_video_promo_mode_uses_short_plan_without_wrappers(tmp_path):
    paper_dir = write_video_bundle(tmp_path)
    output_dir = tmp_path / "promo-video"

    artifact = create_video(
        paper_dir,
        output_dir=output_dir,
        tts_engine=SilentTtsEngine(duration=0.3),
        max_scenes=2,
        width=360,
        height=480,
        fps=12,
        video_mode="promo",
    )

    plan = json.loads(artifact.plan_path.read_text(encoding="utf-8"))
    assert plan["video_mode"] == "promo"
    assert [scene["kind"] for scene in plan["scenes"]] == ["hook", "problem"]
    assert 0.4 <= probe_duration(artifact.final_path) <= 0.8


def test_extract_voice_reference_creates_normalized_wav(tmp_path):
    intro = tmp_path / "intro.mp4"
    output = tmp_path / "voice-reference.wav"
    _make_color_audio_clip(intro, color="0x2f8f63", duration=1.0)

    result = extract_voice_reference(intro, output)

    assert result == output
    assert output.read_bytes()[:4] == b"RIFF"
    assert 0.8 <= probe_duration(output) <= 1.2


def test_xtts_engine_requires_explicit_cpml_acceptance(tmp_path):
    intro = tmp_path / "intro.mp4"
    _make_color_audio_clip(intro, color="0x2f8f63", duration=1.0)
    engine = XttsTtsEngine(intro, license_accepted=False)

    with pytest.raises(VideoError, match="CPML"):
        engine.synthesize("测试旁白", tmp_path / "speech.wav")


def test_split_tts_text_keeps_chunks_within_xtts_limit():
    text = (
        "这是一个很长的中文句子，用来验证语音合成时分段不会丢失内容。"
        "第二句话包含数字 28.4 BLEU，并且还有一些补充说明。"
        "最后一句用于确认所有内容都被保留。"
    )

    chunks = split_tts_text(text, max_chars=40)

    assert all(len(chunk) <= 40 for chunk in chunks)
    assert "".join(chunks) == text


def _extract_frame(media_path, seconds, output_path):
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(seconds),
            "-i",
            str(media_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        check=True,
    )


def _make_color_audio_clip(path, *, color, duration=0.5):
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x180:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
