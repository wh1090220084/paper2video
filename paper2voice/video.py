from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from paper2voice.explainer import load_explainer

CLOSING_NARRATION = "今天的论文讲解到此结束。"


class VideoError(RuntimeError):
    """Raised when a paper cannot be converted into a video."""


@dataclass(frozen=True)
class VideoScene:
    scene_id: str
    kind: str
    section: str
    title: str
    body: str
    narration: str
    figure: str | None = None
    formula: str | None = None


@dataclass(frozen=True)
class VideoArtifact:
    output_dir: Path
    plan_path: Path
    script_path: Path
    subtitle_path: Path
    final_path: Path


@dataclass(frozen=True)
class VideoTheme:
    name: str
    background: tuple[int, int, int]
    ink: tuple[int, int, int]
    muted: tuple[int, int, int]
    accent: tuple[int, int, int]
    rule: tuple[int, int, int]


THEMES = {
    "cinematic": VideoTheme(
        name="cinematic",
        background=(7, 12, 13),
        ink=(246, 231, 199),
        muted=(190, 176, 146),
        accent=(230, 178, 78),
        rule=(67, 55, 35),
    ),
    "light_green": VideoTheme(
        name="light_green",
        background=(242, 249, 243),
        ink=(23, 52, 42),
        muted=(91, 113, 104),
        accent=(47, 143, 99),
        rule=(211, 229, 214),
    ),
    "minimal": VideoTheme(
        name="minimal",
        background=(246, 245, 241),
        ink=(20, 34, 40),
        muted=(79, 94, 99),
        accent=(194, 67, 48),
        rule=(208, 207, 200),
    ),
    "tech": VideoTheme(
        name="tech",
        background=(7, 19, 26),
        ink=(229, 248, 240),
        muted=(139, 177, 168),
        accent=(66, 214, 178),
        rule=(29, 58, 67),
    ),
}


def get_video_theme(theme: str | VideoTheme) -> VideoTheme:
    if isinstance(theme, VideoTheme):
        return theme
    try:
        return THEMES[theme]
    except KeyError as exc:
        raise VideoError(f"Unknown video theme: {theme}") from exc


@dataclass(frozen=True)
class Metric:
    value: str
    unit: str


class TtsEngine(Protocol):
    name: str
    extension: str

    def synthesize(self, text: str, output_path: Path) -> Path: ...


class SapiTtsEngine:
    name = "windows-sapi"
    extension = ".wav"

    def __init__(self, voice: str = "Microsoft Huihui Desktop", rate: int = 170):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        try:
            self._synthesize_with_pyttsx3(text, output_path)
            return output_path
        except Exception:  # noqa: BLE001
            self._synthesize_with_powershell(text, output_path)
        if not _valid_wave(output_path):
            raise VideoError("Windows SAPI did not produce playable audio")
        return output_path

    def _synthesize_with_pyttsx3(self, text: str, output_path: Path) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise VideoError("pyttsx3 is not installed") from exc
        engine = pyttsx3.init()
        try:
            voices = engine.getProperty("voices") or []
            selected = None
            for candidate in voices:
                label = f"{candidate.name} {candidate.id}".lower()
                if self.voice.lower() in label:
                    selected = candidate.id
                    break
                if selected is None and (
                    "chinese" in label or "zh-cn" in label or "huihui" in label
                ):
                    selected = candidate.id
            if selected:
                engine.setProperty("voice", selected)
            engine.setProperty("rate", self.rate)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:  # noqa: BLE001, S110
                pass
        if not _valid_wave(output_path):
            raise VideoError("pyttsx3 did not produce playable audio")

    def _synthesize_with_powershell(self, text: str, output_path: Path) -> None:
        text_path = output_path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")
        script_path = Path(__file__).with_name("sapi_tts.ps1")
        try:
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    "-TextPath",
                    str(text_path),
                    "-OutputPath",
                    str(output_path),
                    "-Voice",
                    self.voice,
                    "-Rate",
                    "-1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise VideoError("Windows SAPI speech synthesis failed") from exc
        finally:
            if text_path.exists():
                text_path.unlink()
        if not _valid_wave(output_path):
            raise VideoError("Windows SAPI did not produce an audio file")


class EdgeTtsEngine:
    name = "edge-tts"
    extension = ".mp3"

    def __init__(
        self,
        voice: str = "zh-CN-XiaoxiaoNeural",
        rate: str = "+0%",
    ):
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import edge_tts
        except ImportError as exc:
            raise VideoError(
                "edge-tts is not installed; install requirements-video.txt"
            ) from exc
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            asyncio.run(communicate.save(str(output_path)))
        except Exception as exc:
            raise VideoError("edge-tts synthesis failed") from exc
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoError("edge-tts did not produce an audio file")
        return output_path


class XttsTtsEngine:
    name = "xtts-v2"
    extension = ".wav"

    def __init__(
        self,
        reference_video: str | Path,
        *,
        license_accepted: bool = False,
        language: str = "zh-cn",
        device: str = "cuda",
    ):
        self.reference_video = Path(reference_video)
        self.reference_wav = self.reference_video.with_name("voice-reference.wav")
        self.license_accepted = license_accepted
        self.language = language
        self.device = device
        self._tts = None

    def synthesize(self, text: str, output_path: Path) -> Path:
        if not self.license_accepted:
            raise VideoError(
                "XTTS v2 requires explicit acceptance of the non-commercial CPML "
                "or a purchased commercial license"
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.reference_wav.is_file():
            extract_voice_reference(self.reference_video, self.reference_wav)
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise VideoError(
                "coqui-tts is not installed; install requirements-video.txt"
            ) from exc
        if self._tts is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            os.environ["COQUI_TOS_AGREED"] = "1"
            self._tts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            ).to(self.device)
        chunks = split_tts_text(text, max_chars=82)
        chunk_paths = []
        try:
            for index, chunk in enumerate(chunks):
                chunk_path = output_path.with_name(
                    f"{output_path.stem}.part{index:02d}{output_path.suffix}"
                )
                self._tts.tts_to_file(
                    text=chunk,
                    speaker_wav=str(self.reference_wav),
                    language=self.language,
                    file_path=str(chunk_path),
                    split_sentences=False,
                )
                chunk_paths.append(chunk_path)
            _concatenate_wavs(chunk_paths, output_path)
        except Exception as exc:
            raise VideoError("XTTS voice-cloned synthesis failed") from exc
        finally:
            for chunk_path in chunk_paths:
                if chunk_path.exists():
                    chunk_path.unlink()
        if not _valid_wave(output_path):
            raise VideoError("XTTS did not produce a valid WAV file")
        return output_path


def split_tts_text(text: str, *, max_chars: int = 82) -> list[str]:
    units = re.findall(r"[^。！？；!?;]+[。！？；!?;]?|[。！？；!?;]", text)
    chunks = []
    for unit in units:
        remaining = unit
        while len(remaining) > max_chars:
            split_at = max_chars
            for index in range(max_chars, 0, -1):
                if remaining[index - 1] in "，,、：:":
                    split_at = index
                    break
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]
        if remaining:
            chunks.append(remaining)
    return chunks or [text]


def _concatenate_wavs(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise VideoError("No WAV chunks to concatenate")
    with wave.open(str(input_paths[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(output_path), "wb") as output:
        output.setparams(params)
        for input_path in input_paths:
            with wave.open(str(input_path), "rb") as source:
                if (
                    source.getnchannels() != params.nchannels
                    or source.getsampwidth() != params.sampwidth
                    or source.getframerate() != params.framerate
                ):
                    raise VideoError("XTTS returned incompatible WAV chunk formats")
                output.writeframes(source.readframes(source.getnframes()))


def build_video_plan(
    bundle_dir: str | Path,
    *,
    max_scenes: int = 6,
    modules: list[str] | None = None,
    include_cover: bool = True,
) -> list[VideoScene]:
    paper_dir = Path(bundle_dir)
    manifest_path = paper_dir / "paper.json"
    if not manifest_path.is_file():
        raise VideoError(f"Missing bundle manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    paper = manifest.get("paper") or {}
    translated = _translation_content(paper_dir)
    title = str(
        translated.get("title_zh") or paper.get("title") or "Untitled Paper"
    ).strip()
    summary = str(translated.get("summary_zh") or paper.get("summary") or "").strip()
    authors = [str(author) for author in paper.get("authors") or []]
    categories = [str(category) for category in paper.get("categories") or []]
    sentences = _sentences(summary)
    if not sentences:
        raise VideoError("Paper summary is empty; cannot build a video plan")

    problem = sentences[0]
    method = _select_sentence(
        sentences,
        r"\b(propose|present|introduce|develop|framework|method|architecture)\b",
        fallback=sentences[min(1, len(sentences) - 1)],
        exclude={problem},
    )
    result = _select_sentence(
        sentences,
        r"(\d|%|outperform|improv|achiev|state-of-the-art|accuracy|bleu|map|f1)",
        fallback=sentences[-1],
        exclude={problem, method},
    )
    takeaway = sentences[-1]
    if takeaway in {problem, method, result} and len(sentences) > 1:
        takeaway = sentences[-2]

    author_text = ", ".join(authors[:4])
    category_text = " / ".join(categories[:3])
    selected = (
        set(modules)
        if modules is not None
        else {"problem", "method", "result", "figure", "takeaway"}
    )
    scenes = [
        VideoScene(
            scene_id="01-cover",
            kind="cover",
            section="PAPER BRIEF",
            title=title,
            body="\n".join(part for part in (author_text, category_text) if part),
            narration=(
                f"今天介绍论文《{title}》。"
                "接下来用几分钟说明它解决的问题、核心方法和关键结果。"
            ),
        ),
    ]

    if "problem" in selected:
        scenes.append(
            VideoScene(
                scene_id=f"{len(scenes) + 1:02d}-problem",
                kind="problem",
                section="01 / 问题",
                title="论文从哪里出发",
                body=problem,
                narration=f"论文关注的起点是：{_trim(problem)}",
            )
        )
    if "method" in selected:
        scenes.append(
            VideoScene(
                scene_id=f"{len(scenes) + 1:02d}-method",
                kind="method",
                section="02 / 方法",
                title="核心思路",
                body=method,
                narration=f"作者提出的核心思路是：{_trim(method)}",
            )
        )
    if "result" in selected:
        scenes.append(
            VideoScene(
                scene_id=f"{len(scenes) + 1:02d}-result",
                kind="result",
                section="03 / 结果",
                title="最关键的结果",
                body=result,
                narration=f"从论文摘要给出的结果看：{_trim(result)}",
            )
        )

    figure = _first_figure(paper_dir)
    if figure and "figure" in selected:
        scenes.append(
            VideoScene(
                scene_id="05-figure",
                kind="figure",
                section="04 / 图示",
                title="论文图示",
                body="图表由论文解析结果直接提取，用于辅助理解方法或实验设置。",
                narration="这张图来自论文解析结果，可以帮助理解作者的方法或实验设置。",
                figure=figure,
            )
        )

    if "figure" in selected:
        for figure_index, (figure_path, caption) in enumerate(
            _figure_items(paper_dir, limit=3)[1:],
            start=2,
        ):
            scenes.append(
                VideoScene(
                    scene_id=f"{len(scenes) + 1:02d}-figure-{figure_index}",
                    kind="figure",
                    section=f"04.{figure_index} / 图示",
                    title=f"论文图示 {figure_index}",
                    body=(
                        caption
                        or "图表由论文解析结果直接提取，用于辅助理解方法或实验设置。"
                    ),
                    narration=(
                        f"我们来看论文中的图示 {figure_index}。{_trim(caption)}"
                        if caption
                        else f"我们来看论文中的图示 {figure_index}，它用于辅助理解作者的方法或实验设置。"
                    ),
                    figure=figure_path,
                )
            )

    if "takeaway" in selected:
        scenes.append(
            VideoScene(
                scene_id=f"{len(scenes) + 1:02d}-takeaway",
                kind="takeaway",
                section="05 / 结论",
                title="这意味着什么",
                body=takeaway,
                narration=f"最后总结：{_trim(takeaway)}",
            )
        )
    scenes = [_compact_scene(scene) for scene in scenes]
    scenes = _attach_scene_visuals(paper_dir, scenes)
    if not include_cover and scenes and scenes[0].kind == "cover":
        scenes = scenes[1:]
    scenes = scenes[: max(1, max_scenes)]
    if scenes:
        scenes[-1] = _append_closing_narration(scenes[-1])
    return scenes


def build_promo_plan(
    bundle_dir: str | Path,
    *,
    max_scenes: int = 6,
) -> list[VideoScene]:
    paper_dir = Path(bundle_dir)
    manifest_path = paper_dir / "paper.json"
    if not manifest_path.is_file():
        raise VideoError(f"Missing bundle manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    paper = manifest.get("paper") or {}
    translated = _translation_content(paper_dir)
    title = str(
        translated.get("title_zh") or paper.get("title") or "Untitled Paper"
    ).strip()
    summary = str(translated.get("summary_zh") or paper.get("summary") or "").strip()
    sentences = _sentences(summary)
    if not sentences:
        raise VideoError("Paper summary is empty; cannot build a promo plan")

    problem = sentences[0]
    method = _select_sentence(
        sentences,
        r"\b(propose|present|introduce|develop|framework|method|architecture)\b",
        fallback=sentences[min(1, len(sentences) - 1)],
        exclude={problem},
    )
    result = _select_sentence(
        sentences,
        r"(\d|%|outperform|improv|achiev|state-of-the-art|accuracy|bleu|map|f1)",
        fallback=sentences[-1],
        exclude={problem, method},
    )
    scenes = [
        VideoScene(
            scene_id="01-hook",
            kind="hook",
            section="PAPER SHORT",
            title="它到底解决了什么？",
            body=_trim(title, 64),
            narration=f"这项研究到底解决了什么？{_trim(title, 46)}",
        ),
        VideoScene(
            scene_id="02-problem",
            kind="problem",
            section="01 / 为什么",
            title="问题出在哪里",
            body=_trim(problem, 96),
            narration=_trim(f"过去的问题是：{problem}", 118),
        ),
        VideoScene(
            scene_id="03-method",
            kind="method",
            section="02 / 怎么做",
            title="核心方法",
            body=_trim(method, 96),
            narration=_trim(f"作者的方法是：{method}", 118),
        ),
        VideoScene(
            scene_id="04-result",
            kind="result",
            section="03 / 结果",
            title="跑出了什么结果",
            body=_trim(result, 96),
            narration=_trim(f"实验结果是：{result}", 118),
        ),
    ]
    figure_limit = min(2, max(1, max_scenes - 5))
    for figure_index, (figure_path, caption) in enumerate(
        _figure_items(paper_dir, limit=figure_limit),
        1,
    ):
        scenes.append(
            VideoScene(
                scene_id=f"{len(scenes) + 1:02d}-figure-{figure_index}",
                kind="figure",
                section=f"04.{figure_index} / 看图",
                title=f"看图 {figure_index}",
                body=_trim(caption or "论文关键图示", 76),
                narration=_trim(
                    f"看图 {figure_index}。{caption or '这张图展示了论文的关键设计。'}",
                    112,
                ),
                figure=figure_path,
            )
        )
    scenes.append(
        VideoScene(
            scene_id=f"{len(scenes) + 1:02d}-cta",
            kind="cta",
            section="05 / 了解更多",
            title="继续看原文",
            body="论文原文 · 代码 · 实验细节",
            narration="论文原文、代码和更多实验细节，都在原始论文中。",
        )
    )
    scenes = [_compact_promo_scene(scene) for scene in scenes]
    scenes = _attach_scene_visuals(paper_dir, scenes)
    scenes = scenes[: max(1, max_scenes)]
    if scenes:
        scenes[-1] = _append_closing_narration(scenes[-1])
    return scenes


def build_deep_video_plan(
    bundle_dir: str | Path,
    *,
    max_scenes: int = 18,
    modules: list[str] | None = None,
    include_cover: bool = True,
) -> list[VideoScene]:
    paper_dir = Path(bundle_dir)
    explainer = load_explainer(paper_dir)
    visuals = _figure_items(paper_dir, limit=30)

    def visual_at(index: int) -> str | None:
        return visuals[index % len(visuals)][0] if visuals else None

    info = explainer["paper_info"]
    authors = "、".join(info["authors"])
    author_label = _author_label(info["authors"])
    institutions = "、".join(info["institutions"])
    links = info["links"]
    scenes = [
        VideoScene(
            scene_id="00-cover",
            kind="cover",
            section="00 / 论文概览",
            title=info["title"],
            body=f"{authors}\n{institutions}",
            narration=(
                f"今天讲解的论文题为《{info['title']}》。"
                f"论文关注的问题是：{_trim(info['problem'], 170)}"
            ),
            figure=visual_at(0),
        ),
        VideoScene(
            scene_id="01-paper-info",
            kind="paper_info",
            section="01 / 论文信息",
            title="论文信息",
            body=(
                f"{authors}\n{institutions}\n"
                f"{info['venue']} · {info['published']}\n"
                f"代码：{links['code']}\n项目：{links['project']}"
            ),
            narration=(
                f"论文作者是{author_label}，主要机构为{institutions}。"
                f"发表信息为{info['venue']}，发布时间为{info['published']}。"
                f"arXiv 链接是 {links['arxiv']}，DOI 是 {links['doi']}。"
                f"代码链接为 {links['code']}，项目主页为 {links['project']}。"
            ),
            figure=visual_at(1),
        ),
    ]

    innovations = explainer["innovations"]
    innovation_titles = [item["title"] for item in innovations]
    scenes.append(
        VideoScene(
            scene_id="02-innovation-overview",
            kind="innovation",
            section="02 / 创新点",
            title="三个核心创新",
            body="\n".join(
                f"{index}. {title}" for index, title in enumerate(innovation_titles, 1)
            ),
            narration=(
                "这篇论文的贡献可以归纳为"
                f"{len(innovations)}个核心创新点："
                f"{'；'.join(innovation_titles)}。"
            ),
            figure=visual_at(2),
        )
    )
    for index, innovation in enumerate(innovations, 1):
        scenes.append(
            VideoScene(
                scene_id=f"02-innovation-{index}",
                kind="innovation",
                section=f"02.{index} / 创新点",
                title=innovation["title"],
                body=_trim(innovation["summary"], 150),
                narration=(
                    f"第{index}个创新点是{innovation['title']}。{innovation['summary']}"
                ),
                figure=visual_at(2 + index),
            )
        )

    method = explainer["method"]
    scenes.append(
        VideoScene(
            scene_id="03-method-overview",
            kind="method",
            section="03 / 方法",
            title="方法总览",
            body=_trim(method["overview"], 180),
            narration=f"下面进入方法部分。{method['overview']}",
            figure=visual_at(3),
        )
    )
    for index, step in enumerate(method["steps"][:4], 1):
        scenes.append(
            VideoScene(
                scene_id=f"03-method-{index}",
                kind="method",
                section=f"03.{index} / 方法",
                title=step["title"],
                body=_trim(
                    f"{step['detail']}\n来源：第 {', '.join(map(str, step['evidence_pages']))} 页",
                    180,
                ),
                narration=f"{step['title']}。{step['detail']}",
                figure=visual_at(3 + index),
            )
        )
    for index, formula in enumerate(method["formulas"][:3], 1):
        scenes.append(
            VideoScene(
                scene_id=f"03-formula-{index}",
                kind="formula",
                section=f"03 / 核心公式 {index}",
                title=f"核心公式 {index}",
                body=_trim(formula["explanation"], 150),
                narration=(
                    f"理解方法需要保留的核心公式是 {formula['latex']}。"
                    f"{formula['explanation']}"
                ),
                figure=visual_at(3 + index),
                formula=formula["latex"],
            )
        )

    experiments = explainer["experiments"]
    scenes.append(
        VideoScene(
            scene_id="04-experiment-setup",
            kind="experiment",
            section="04 / 实验结果",
            title="实验设置",
            body=_trim(experiments["setup"], 160),
            narration=f"实验部分首先看设置。{experiments['setup']}",
            figure=visual_at(4),
        )
    )
    main_results = experiments["main_results"]
    for index, result in enumerate(main_results[:4], 1):
        scenes.append(
            VideoScene(
                scene_id=f"04-result-{index}",
                kind="experiment",
                section=f"04.{index} / 主结果",
                title=f"主结果 {index}",
                body=_trim(
                    f"{result['claim']}\n"
                    f"来源：第 {', '.join(map(str, result['evidence_pages']))} 页",
                    180,
                ),
                narration=(
                    f"主实验结果 {index}：{result['claim']}。"
                    f"证据位于第 {', '.join(map(str, result['evidence_pages']))} 页。"
                ),
                figure=visual_at(5),
            )
        )
    ablations = experiments["ablations"]
    if ablations:
        ablation_text = "；".join(
            f"{item['claim']}（第 {', '.join(map(str, item['evidence_pages']))} 页）"
            for item in ablations[:4]
        )
        scenes.append(
            VideoScene(
                scene_id="04-ablations",
                kind="experiment",
                section="04.2 / 消融实验",
                title="消融实验",
                body=_trim(ablation_text, 180),
                narration=f"消融实验进一步说明：{ablation_text}。",
                figure=visual_at(5),
            )
        )
    experiment_figures = [
        path for path in experiments["visual_figures"] if (paper_dir / path).is_file()
    ]
    if experiment_figures:
        scenes.append(
            VideoScene(
                scene_id="04-visual",
                kind="experiment",
                section="04.3 / 可视化",
                title="可视化结果",
                body="论文中的可视化结果用于验证方法在实际场景中的表现。",
                narration="最后看论文中的可视化结果，它用于验证方法在实际场景中的表现。",
                figure=experiment_figures[0],
            )
        )

    summary = explainer["summary"]
    scenes.append(
        VideoScene(
            scene_id="05-value",
            kind="summary",
            section="05.1 / 总结",
            title="价值与应用",
            body=_trim(
                f"{summary['value']}\n应用：{'；'.join(summary['applications'])}",
                160,
            ),
            narration=(
                f"总结来看，{summary['value']}"
                f"它适用于{'；'.join(summary['applications'])}。"
            ),
            figure=visual_at(0),
        )
    )
    scenes.append(
        VideoScene(
            scene_id="05-limitations",
            kind="summary",
            section="05.2 / 总结",
            title="局限与结论",
            body=_trim(
                f"局限：{'；'.join(summary['limitations'])}\n{summary['conclusion']}",
                160,
            ),
            narration=(
                f"同时需要注意局限：{'；'.join(summary['limitations'])}。"
                f"{summary['conclusion']}"
            ),
            figure=visual_at(1),
        )
    )
    if modules is not None:
        selected = set(modules)
        scenes = [
            scene
            for scene in scenes
            if _deep_scene_module(scene) in selected
            and (include_cover or scene.kind != "cover")
        ]
    elif not include_cover:
        scenes = [scene for scene in scenes if scene.kind != "cover"]
    scenes = _apply_deep_scene_visuals(
        scenes,
        visuals,
        preferred_experiment_figures=experiments["visual_figures"],
    )
    scenes = _limit_deep_scenes(scenes, max_scenes=max_scenes)
    if not scenes:
        raise VideoError("No deep explainer scenes selected")
    scenes[-1] = _append_closing_narration(scenes[-1])
    return scenes


def render_scene(
    scene: VideoScene,
    *,
    paper_dir: str | Path,
    output_path: str | Path,
    width: int = 1920,
    height: int = 1080,
    theme: str | VideoTheme = "cinematic",
) -> Path:
    if width < 320 or height < 180:
        raise VideoError("Video frame must be at least 320x180")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = min(width / 1920, height / 1080)
    palette = get_video_theme(theme)
    if palette.name == "cinematic":
        return _render_cinematic_scene(
            scene,
            paper_dir=Path(paper_dir),
            output_path=output_path,
            width=width,
            height=height,
            palette=palette,
        )
    canvas = Image.new("RGB", (width, height), palette.background)
    draw = ImageDraw.Draw(canvas)

    ink = palette.ink
    muted = palette.muted
    accent = palette.accent
    rule = palette.rule
    margin = int(92 * scale)
    top = int(68 * scale)

    draw.rectangle(
        (margin, top, margin + int(72 * scale), top + int(7 * scale)),
        fill=accent,
    )
    section_font = _font(max(12, int(24 * scale)), bold=True)
    draw.text(
        (margin + int(94 * scale), top - int(3 * scale)),
        scene.section,
        font=section_font,
        fill=accent,
    )
    page_font = _font(max(11, int(20 * scale)))
    draw.text(
        (width - margin, top),
        scene.scene_id,
        font=page_font,
        fill=muted,
        anchor="ra",
    )

    title_top = int(155 * scale)
    body_top = int(360 * scale)
    figure_path = None
    if scene.figure:
        candidate = Path(paper_dir) / scene.figure
        if candidate.is_file():
            figure_path = candidate

    if figure_path:
        title_width = int(670 * scale)
        body_width = int(650 * scale)
    else:
        title_width = width - 2 * margin
        body_width = min(int(1500 * scale), width - 2 * margin)

    title_font = _fit_font(
        draw,
        scene.title,
        max_width=title_width,
        max_lines=3,
        sizes=[int(size * scale) for size in (74, 66, 58, 50, 44)],
        bold=True,
    )
    title_bottom = _draw_wrapped(
        draw,
        scene.title,
        (margin, title_top),
        title_font,
        ink,
        title_width,
        max_lines=3,
        line_spacing=1.08,
    )

    body_y = max(body_top, title_bottom + int(55 * scale))
    if scene.kind == "result":
        metrics = extract_metrics(scene.body)
        if metrics:
            metric_bottom = _draw_metrics(
                draw,
                metrics,
                x=margin,
                y=max(title_bottom + int(45 * scale), int(315 * scale)),
                width=body_width,
                scale=scale,
                accent=accent,
                ink=ink,
                muted=muted,
            )
            body_y = max(body_y, metric_bottom + int(48 * scale))
    body_font = _font(max(18, int(40 * scale)))
    body_bottom_limit = height - int(230 * scale)
    body_max_lines = max(2, int((body_bottom_limit - body_y) / (45 * scale)))
    _draw_wrapped(
        draw,
        scene.body,
        (margin, body_y),
        body_font,
        muted,
        body_width,
        max_lines=body_max_lines,
        line_spacing=1.28,
    )

    visual_box = (
        int(845 * scale),
        int(195 * scale),
        width - margin,
        height - int(150 * scale),
    )
    if figure_path:
        box = (
            int(845 * scale),
            int(195 * scale),
            width - margin,
            height - int(150 * scale),
        )
        visual_box = box
        _paste_contained(canvas, figure_path, box)
        draw.rectangle(box, outline=rule, width=max(1, int(2 * scale)))
    if scene.formula:
        formula_height = max(80, int((visual_box[3] - visual_box[1]) * 0.32))
        formula_width = max(240, visual_box[2] - visual_box[0] - int(32 * scale))
        formula_path = output_path.with_name(f"{output_path.stem}-formula.png")
        render_formula_image(
            scene.formula,
            formula_path,
            width=formula_width,
            height=formula_height,
        )
        formula_y = visual_box[3] - formula_height - int(16 * scale)
        draw.rectangle(
            (
                visual_box[0],
                formula_y - int(10 * scale),
                visual_box[2],
                visual_box[3],
            ),
            fill=palette.background,
        )
        with Image.open(formula_path) as formula_image:
            rgba_formula = formula_image.convert("RGBA")
            canvas.paste(
                rgba_formula,
                (visual_box[0] + int(16 * scale), formula_y),
                rgba_formula,
            )
        formula_path.unlink()

    draw.line(
        (margin, height - int(88 * scale), width - margin, height - int(88 * scale)),
        fill=rule,
        width=max(1, int(2 * scale)),
    )
    draw.text(
        (margin, height - int(63 * scale)),
        "paper2voice",
        font=page_font,
        fill=muted,
    )
    draw.text(
        (width - margin, height - int(63 * scale)),
        scene.kind.upper(),
        font=page_font,
        fill=muted,
        anchor="ra",
    )
    canvas.save(output_path, format="PNG")
    return output_path


def render_promo_scene(
    scene: VideoScene,
    *,
    paper_dir: str | Path,
    output_path: str | Path,
    width: int = 1080,
    height: int = 1440,
    theme: str | VideoTheme = "cinematic",
) -> Path:
    if width < 320 or height < 180:
        raise VideoError("Promo frame must be at least 320x180")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale = min(width / 1080, height / 1440)
    palette = get_video_theme(theme)
    if palette.name == "cinematic":
        return _render_cinematic_promo_scene(
            scene,
            paper_dir=Path(paper_dir),
            output_path=output_path,
            width=width,
            height=height,
            palette=palette,
        )
    figure_path = Path(paper_dir) / scene.figure if scene.figure else None
    if figure_path and figure_path.is_file():
        with Image.open(figure_path) as source:
            canvas = ImageOps.fit(
                source.convert("RGB"),
                (width, height),
                Image.Resampling.LANCZOS,
            )
        overlay = Image.new("RGBA", (width, height), (4, 12, 17, 174))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    else:
        canvas = Image.new("RGB", (width, height), palette.background)

    draw = ImageDraw.Draw(canvas)
    ink = (248, 252, 249)
    muted = (184, 210, 202)
    accent = (117, 225, 83) if palette.name == "tech" else palette.accent
    margin = int(64 * scale)
    brand_font = _font(max(12, int(24 * scale)), bold=True)
    section_font = _font(max(12, int(24 * scale)), bold=True)
    draw.text((margin, int(50 * scale)), "PAPER2VOICE", font=brand_font, fill=ink)
    draw.text(
        (width - margin, int(50 * scale)),
        scene.section,
        font=section_font,
        fill=accent,
        anchor="ra",
    )
    draw.rectangle(
        (margin, int(98 * scale), width - margin, int(103 * scale)),
        fill=accent,
    )

    title_top = int(210 * scale)
    title_width = width - 2 * margin
    title_font = _fit_font(
        draw,
        scene.title,
        max_width=title_width,
        max_lines=4,
        sizes=[int(size * scale) for size in (104, 94, 84, 74, 64)],
        bold=True,
    )
    title_bottom = _draw_wrapped(
        draw,
        scene.title,
        (margin, title_top),
        title_font,
        ink,
        title_width,
        max_lines=4,
        line_spacing=1.04,
    )

    body_font = _font(max(20, int(42 * scale)), bold=scene.kind in {"hook", "cta"})
    body_y = max(int(720 * scale), title_bottom + int(58 * scale))
    body_limit = height - int(210 * scale)
    available_lines = max(2, int((body_limit - body_y) / (52 * scale)))
    _draw_wrapped(
        draw,
        scene.body,
        (margin, body_y),
        body_font,
        ink if scene.kind in {"hook", "cta"} else muted,
        title_width,
        max_lines=min(7, available_lines),
        line_spacing=1.2,
    )

    bracket = int(34 * scale)
    bracket_width = max(1, int(4 * scale))
    corners = (
        (
            margin,
            height - int(270 * scale),
            margin + bracket,
            height - int(270 * scale),
        ),
        (
            margin,
            height - int(270 * scale),
            margin,
            height - int(270 * scale) + bracket,
        ),
        (
            width - margin - bracket,
            height - int(270 * scale),
            width - margin,
            height - int(270 * scale),
        ),
        (
            width - margin,
            height - int(270 * scale),
            width - margin,
            height - int(270 * scale) + bracket,
        ),
    )
    for line in corners:
        draw.line(line, fill=accent, width=bracket_width)

    footer_font = _font(max(11, int(20 * scale)))
    draw.text(
        (margin, height - int(54 * scale)),
        "PAPER2VOICE / SHORT",
        font=footer_font,
        fill=muted,
    )
    canvas.save(output_path, format="PNG")
    return output_path


def _render_cinematic_scene(
    scene: VideoScene,
    *,
    paper_dir: Path,
    output_path: Path,
    width: int,
    height: int,
    palette: VideoTheme,
) -> Path:
    scale = min(width / 1920, height / 1080)
    visual = _scene_visual_path(scene, paper_dir)
    canvas = _cinematic_background(width, height, visual, palette, brightness=0.42)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        (0, 0, int(width * 0.59), height),
        fill=(5, 10, 11, 224),
    )
    overlay_draw.polygon(
        (
            (int(width * 0.50), 0),
            (int(width * 0.63), 0),
            (int(width * 0.56), height),
            (int(width * 0.43), height),
        ),
        fill=(9, 18, 19, 96),
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    gold = palette.accent
    cyan = (73, 214, 205)
    margin = int(88 * scale)
    top = int(62 * scale)
    section_font = _font(max(12, int(23 * scale)), bold=True)
    tiny_font = _font(max(10, int(17 * scale)))
    draw.rectangle(
        (margin, top, margin + int(68 * scale), top + int(6 * scale)),
        fill=gold,
    )
    draw.text(
        (margin + int(86 * scale), top - int(3 * scale)),
        scene.section,
        font=section_font,
        fill=gold,
    )
    draw.text(
        (width - margin, top),
        f"{scene.scene_id} / PAPER VISION",
        font=tiny_font,
        fill=cyan,
        anchor="ra",
    )

    title_width = int(width * 0.46)
    title_top = int(160 * scale)
    title_font = _fit_font(
        draw,
        scene.title,
        max_width=title_width,
        max_lines=3,
        sizes=[int(size * scale) for size in (82, 74, 66, 58, 50)],
        bold=True,
    )
    title_bottom = _draw_wrapped(
        draw,
        scene.title,
        (margin, title_top),
        title_font,
        palette.ink,
        title_width,
        max_lines=3,
        line_spacing=1.08,
    )

    body_y = max(int(345 * scale), title_bottom + int(52 * scale))
    if scene.kind == "result":
        metrics = extract_metrics(scene.body)
        if metrics:
            metric_bottom = _draw_metrics(
                draw,
                metrics,
                x=margin,
                y=max(int(310 * scale), title_bottom + int(38 * scale)),
                width=int(width * 0.45),
                scale=scale,
                accent=gold,
                ink=palette.ink,
                muted=palette.muted,
            )
            body_y = metric_bottom + int(42 * scale)

    body_font = _font(max(17, int(33 * scale)))
    body_limit = height - int(126 * scale)
    body_lines = max(2, int((body_limit - body_y) / (41 * scale)))
    _draw_wrapped(
        draw,
        scene.body,
        (margin, body_y),
        body_font,
        palette.muted,
        title_width,
        max_lines=min(7, body_lines),
        line_spacing=1.28,
    )

    visual_box = (
        int(width * 0.59),
        int(height * 0.19),
        int(width * 0.94),
        int(height * 0.79),
    )
    if visual:
        _paste_contained(canvas, visual, visual_box)
        _draw_visual_frame(draw, visual_box, gold, cyan, scale)
        draw.text(
            (visual_box[0], visual_box[1] - int(31 * scale)),
            "SOURCE / PAPER FIGURE",
            font=tiny_font,
            fill=cyan,
        )
    else:
        _draw_signal_panel(draw, visual_box, gold, cyan, scale)
    if scene.formula:
        formula_height = max(90, int((visual_box[3] - visual_box[1]) * 0.34))
        formula_width = max(260, visual_box[2] - visual_box[0] - int(40 * scale))
        formula_path = output_path.with_name(f"{output_path.stem}-formula.png")
        render_formula_image(
            scene.formula,
            formula_path,
            width=formula_width,
            height=formula_height,
        )
        formula_y = visual_box[3] - formula_height - int(18 * scale)
        draw.rectangle(
            (
                visual_box[0],
                formula_y - int(12 * scale),
                visual_box[2],
                visual_box[3],
            ),
            fill=(5, 10, 11),
        )
        with Image.open(formula_path) as formula_image:
            canvas.paste(
                formula_image.convert("RGBA"),
                (visual_box[0] + int(20 * scale), formula_y),
                formula_image.convert("RGBA"),
            )
        formula_path.unlink()

    draw.line(
        (margin, height - int(72 * scale), width - margin, height - int(72 * scale)),
        fill=palette.rule,
        width=max(1, int(2 * scale)),
    )
    draw.text(
        (margin, height - int(51 * scale)),
        "PAPER2VOICE / CINEMATIC BRIEF",
        font=tiny_font,
        fill=palette.muted,
    )
    draw.text(
        (width - margin, height - int(51 * scale)),
        scene.kind.upper(),
        font=tiny_font,
        fill=gold,
        anchor="ra",
    )
    canvas.save(output_path, format="PNG")
    return output_path


def _render_cinematic_promo_scene(
    scene: VideoScene,
    *,
    paper_dir: Path,
    output_path: Path,
    width: int,
    height: int,
    palette: VideoTheme,
) -> Path:
    scale = min(width / 1080, height / 1440)
    visual = _scene_visual_path(scene, paper_dir)
    canvas = _cinematic_background(width, height, visual, palette, brightness=0.66)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    text_top = int(height * 0.43)
    overlay_draw.rectangle(
        (0, text_top, width, height),
        fill=(5, 9, 10, 226),
    )
    overlay_draw.rectangle(
        (0, 0, width, int(height * 0.11)),
        fill=(4, 8, 9, 155),
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    gold = palette.accent
    cyan = (73, 214, 205)
    margin = int(64 * scale)
    brand_font = _font(max(12, int(22 * scale)), bold=True)
    draw.text(
        (margin, int(42 * scale)),
        "PAPER2VOICE / CINEMATIC",
        font=brand_font,
        fill=palette.ink,
    )
    draw.text(
        (width - margin, int(42 * scale)),
        scene.section,
        font=brand_font,
        fill=cyan,
        anchor="ra",
    )
    if visual:
        frame = (
            int(width * 0.08),
            int(height * 0.14),
            int(width * 0.92),
            int(height * 0.41),
        )
        _draw_visual_frame(draw, frame, gold, cyan, scale)
        draw.text(
            (frame[0], frame[1] - int(28 * scale)),
            "PAPER FIGURE / VISUAL CONTEXT",
            font=_font(max(10, int(17 * scale)), bold=True),
            fill=cyan,
        )

    title_top = int(height * 0.47)
    title_width = width - 2 * margin
    title_font = _fit_font(
        draw,
        scene.title,
        max_width=title_width,
        max_lines=3,
        sizes=[int(size * scale) for size in (86, 78, 70, 62, 54)],
        bold=True,
    )
    title_bottom = _draw_wrapped(
        draw,
        scene.title,
        (margin, title_top),
        title_font,
        palette.ink,
        title_width,
        max_lines=3,
        line_spacing=1.05,
    )
    draw.rectangle(
        (
            margin,
            title_bottom + int(25 * scale),
            margin + int(110 * scale),
            title_bottom + int(31 * scale),
        ),
        fill=gold,
    )

    body_font = _font(max(19, int(35 * scale)))
    body_y = title_bottom + int(62 * scale)
    body_limit = height - int(128 * scale)
    body_lines = max(2, int((body_limit - body_y) / (43 * scale)))
    _draw_wrapped(
        draw,
        scene.body,
        (margin, body_y),
        body_font,
        palette.muted,
        title_width,
        max_lines=min(7, body_lines),
        line_spacing=1.24,
    )

    footer_font = _font(max(10, int(17 * scale)))
    draw.line(
        (
            margin,
            height - int(72 * scale),
            width - margin,
            height - int(72 * scale),
        ),
        fill=palette.rule,
        width=max(1, int(2 * scale)),
    )
    draw.text(
        (margin, height - int(49 * scale)),
        "PAPER2VOICE / SHORT",
        font=footer_font,
        fill=palette.muted,
    )
    draw.text(
        (width - margin, height - int(49 * scale)),
        scene.kind.upper(),
        font=footer_font,
        fill=gold,
        anchor="ra",
    )
    canvas.save(output_path, format="PNG")
    return output_path


def _cinematic_background(
    width: int,
    height: int,
    visual: Path | None,
    palette: VideoTheme,
    *,
    brightness: float,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), palette.background)
    if not visual:
        return canvas
    with Image.open(visual) as source:
        image = ImageOps.fit(
            source.convert("RGB"),
            (width, height),
            Image.Resampling.LANCZOS,
        )
    image = ImageEnhance.Brightness(image).enhance(brightness)
    tint = Image.new("RGB", (width, height), (10, 28, 27))
    return Image.blend(image, tint, 0.18)


def _scene_visual_path(scene: VideoScene, paper_dir: Path) -> Path | None:
    if not scene.figure:
        return None
    path = paper_dir / scene.figure
    return path if path.is_file() else None


def _draw_visual_frame(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    gold: tuple[int, int, int],
    cyan: tuple[int, int, int],
    scale: float,
) -> None:
    line_width = max(1, int(2 * scale))
    draw.rectangle(box, outline=gold, width=line_width)
    length = int(30 * scale)
    x0, y0, x1, y1 = box
    for line in (
        (x0, y0, x0 + length, y0),
        (x0, y0, x0, y0 + length),
        (x1 - length, y0, x1, y0),
        (x1, y0, x1, y0 + length),
        (x0, y1 - length, x0, y1),
        (x0, y1, x0 + length, y1),
        (x1 - length, y1, x1, y1),
        (x1, y1 - length, x1, y1),
    ):
        draw.line(line, fill=cyan, width=max(1, int(3 * scale)))


def _draw_signal_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    gold: tuple[int, int, int],
    cyan: tuple[int, int, int],
    scale: float,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(8, 18, 19), outline=gold, width=max(1, int(2 * scale)))
    for index in range(1, 8):
        y = y0 + int((y1 - y0) * index / 8)
        draw.line(
            (x0 + int(28 * scale), y, x1 - int(28 * scale), y),
            fill=(24, 52, 51),
            width=max(1, int(scale)),
        )
    for index in range(4):
        left = x0 + int((28 + index * 42) * scale)
        draw.line(
            (left, y0 + int(28 * scale), left, y1 - int(28 * scale)),
            fill=(20, 44, 44),
            width=max(1, int(scale)),
        )
    draw.rectangle(
        (
            x0 + int(42 * scale),
            y0 + int(52 * scale),
            x1 - int(42 * scale),
            y0 + int(58 * scale),
        ),
        fill=cyan,
    )
    _draw_visual_frame(draw, box, gold, cyan, scale)


def render_formula_image(
    formula: str,
    output_path: str | Path,
    *,
    width: int = 1200,
    height: int = 360,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        formula_lines = _formula_lines(formula)
        figure = Figure(
            figsize=(width / 100, height / 100),
            dpi=100,
            facecolor=(0, 0, 0, 0),
        )
        canvas = FigureCanvasAgg(figure)
        longest_line = max(18, max(len(line) for line in formula_lines))
        font_size = max(10, min(38, int(width * 0.8 / longest_line)))
        if len(formula_lines) == 1:
            positions = [0.5]
        else:
            positions = [
                0.7 - index * 0.4 / (len(formula_lines) - 1)
                for index in range(len(formula_lines))
            ]
        for index, line in enumerate(formula_lines):
            figure.text(
                0.5,
                positions[index],
                f"${line}$",
                ha="center",
                va="center",
                color="#f0d79b",
                fontsize=font_size,
            )
        canvas.draw()
        image = Image.frombytes(
            "RGBA",
            canvas.get_width_height(),
            canvas.buffer_rgba(),
            "raw",
            "RGBA",
        )
    except (ImportError, RuntimeError, ValueError):
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = _fit_font(
            draw,
            formula,
            max_width=width - 80,
            max_lines=3,
            sizes=[42, 36, 30, 24, 20],
            bold=True,
        )
        _draw_wrapped(
            draw,
            formula,
            (40, max(20, height // 3)),
            font,
            (240, 215, 155, 255),
            width - 80,
            max_lines=3,
            line_spacing=1.15,
        )
    image.save(output_path, format="PNG")
    return output_path


def _formula_lines(formula: str, *, max_chars: int = 52) -> list[str]:
    normalized = formula.strip().replace(r"\qquad", r"\quad")
    if len(normalized) <= max_chars:
        return [normalized]
    segments = []
    current = ""
    depth = 0
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth = max(0, depth - 1)
        if depth == 0 and normalized.startswith(r"\quad", index):
            if current.strip():
                segments.append(current.strip())
            current = ""
            index += len(r"\quad")
            continue
        if depth == 0 and character == ",":
            if current.strip():
                segments.append(current.strip())
            current = ""
            index += 1
            continue
        current += character
        index += 1
    if current.strip():
        segments.append(current.strip())
    return segments or [normalized]


def extract_metrics(text: str, *, limit: int = 3) -> list[Metric]:
    pattern = re.compile(
        r"(?<![\w.])(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>%|BLEU|mAP|AP|F1|accuracy|precision|recall)",
        re.IGNORECASE,
    )
    metrics = []
    seen = set()
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 28) : match.start()]
        if re.search(
            r"\b(?:by\s+)?(?:over|more than|at least)\s*$"
            r"|\b(?:by|increase[sd]?|gain(?:s|ed)?|improve[sd]?)\s*$",
            prefix,
            re.IGNORECASE,
        ):
            continue
        value = match.group("value")
        unit = match.group("unit")
        key = (value, unit.lower())
        if key in seen:
            continue
        seen.add(key)
        metrics.append(Metric(value=value, unit=unit))
        if len(metrics) >= limit:
            break
    return metrics


def write_srt(
    scenes: list[VideoScene],
    durations: list[float],
    output_path: str | Path,
) -> Path:
    if len(scenes) != len(durations):
        raise VideoError("Scenes and durations must have equal lengths")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current = 0.0
    blocks = []
    for index, (scene, duration) in enumerate(zip(scenes, durations, strict=True), 1):
        end = current + max(0.05, float(duration))
        text = "\n".join(_wrap_subtitle(scene.narration, width=42))
        blocks.append(f"{index}\n{_srt_time(current)} --> {_srt_time(end)}\n{text}\n")
        current = end
    output_path.write_text("\n".join(blocks), encoding="utf-8")
    return output_path


def probe_duration(media_path: str | Path) -> float:
    command = [
        _require_executable("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=duration",
        "-of",
        "json",
        str(media_path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        payload = json.loads(result.stdout)
        candidates = [payload.get("format", {}).get("duration")]
        candidates.extend(
            stream.get("duration") for stream in payload.get("streams", [])
        )
        for candidate in candidates:
            if candidate not in (None, "N/A"):
                return float(candidate)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise VideoError(f"Cannot read media duration: {media_path}") from exc
    try:
        with wave.open(str(media_path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (OSError, wave.Error):
        pass
    raise VideoError(f"Cannot read media duration: {media_path}")


def create_video(
    bundle_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    tts_engine: TtsEngine | None = None,
    max_scenes: int = 6,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    burn_subtitles: bool = True,
    theme: str | VideoTheme = "cinematic",
    modules: list[str] | None = None,
    video_mode: str = "long",
    intro_video: str | Path | None = None,
    outro_video: str | Path | None = None,
) -> VideoArtifact:
    paper_dir = Path(bundle_dir)
    output_dir = Path(output_dir) if output_dir else paper_dir / "video"
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = get_video_theme(theme)
    if video_mode not in {"long", "promo"}:
        raise VideoError(f"Unknown video mode: {video_mode}")
    if video_mode == "promo":
        scenes = build_promo_plan(paper_dir, max_scenes=max_scenes)
        intro_video = None
        outro_video = None
        burn_subtitles = False
    else:
        scenes = build_deep_video_plan(
            paper_dir,
            max_scenes=max_scenes,
            modules=modules,
            include_cover=intro_video is None,
        )
    engine = tts_engine or _default_tts_engine()
    for pattern, suffix in (("scene_*.mp3", ".mp3"), ("scene_*.wav", ".wav")):
        if suffix == engine.extension:
            continue
        for stale_path in output_dir.glob(pattern):
            stale_path.unlink()

    plan_path = output_dir / "plan.json"
    script_path = output_dir / "script.md"
    subtitle_path = output_dir / "video.srt"
    plan_payload = {
        "schema_version": 1,
        "source_bundle": paper_dir.name,
        "tts": {"name": engine.name, "voice": getattr(engine, "voice", None)},
        "video": {"width": width, "height": height, "fps": fps},
        "video_mode": video_mode,
        "theme": palette.name,
        "scenes": [asdict(scene) for scene in scenes],
    }
    plan_path.write_text(
        json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    script_path.write_text(_script_markdown(plan_payload), encoding="utf-8")

    durations = []
    scene_paths = []
    for index, scene in enumerate(scenes, 1):
        frame_path = output_dir / f"scene_{index:02d}.png"
        audio_path = output_dir / f"scene_{index:02d}{engine.extension}"
        clip_path = output_dir / f"scene_{index:02d}.mp4"
        renderer = render_promo_scene if video_mode == "promo" else render_scene
        renderer(
            scene,
            paper_dir=paper_dir,
            output_path=frame_path,
            width=width,
            height=height,
            theme=palette,
        )
        engine.synthesize(scene.narration, audio_path)
        duration = probe_duration(audio_path)
        durations.append(duration)
        _compose_scene(frame_path, audio_path, clip_path, fps=fps, duration=duration)
        scene_paths.append(clip_path)

    write_srt(scenes, durations, subtitle_path)
    concat_path = output_dir / "concat.txt"
    concat_path.write_text(
        "".join(f"file '{path.name}'\n" for path in scene_paths),
        encoding="utf-8",
    )
    merged_path = output_dir / "_merged.mp4"
    _concat_scenes(concat_path, merged_path)

    final_path = output_dir / "final.mp4"
    body_path = output_dir / "_body.mp4" if intro_video or outro_video else final_path
    if burn_subtitles:
        subtitle_scale = min(width / 1920, height / 1080)
        subtitle_size = (
            max(11, int(18 * subtitle_scale))
            if video_mode == "promo"
            else max(12, int(24 * subtitle_scale))
        )
        style = (
            "FontName=Microsoft YaHei,"
            f"FontSize={subtitle_size},"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H80000000,"
            "BorderStyle=1,Outline=1,Shadow=0,MarginV=20"
        )
        _run_ffmpeg(
            [
                "-i",
                merged_path.name,
                "-vf",
                f"subtitles={subtitle_path.name}:force_style='{style}'",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "21",
                "-c:a",
                "copy",
                body_path.name,
            ],
            cwd=output_dir,
        )
    else:
        os.replace(merged_path, body_path)
    if merged_path.exists():
        merged_path.unlink()

    if intro_video or outro_video:
        wrapped_paths = []
        if intro_video:
            wrapped_paths.append(
                _normalize_media_clip(
                    Path(intro_video),
                    output_dir / "_intro.mp4",
                    width=width,
                    height=height,
                    fps=fps,
                )
            )
        wrapped_paths.append(
            _normalize_media_clip(
                body_path,
                output_dir / "_body-normalized.mp4",
                width=width,
                height=height,
                fps=fps,
            )
        )
        if outro_video:
            wrapped_paths.append(
                _normalize_media_clip(
                    Path(outro_video),
                    output_dir / "_outro.mp4",
                    width=width,
                    height=height,
                    fps=fps,
                )
            )
        wrapped_concat = output_dir / "_wrapped.txt"
        wrapped_concat.write_text(
            "".join(f"file '{path.name}'\n" for path in wrapped_paths),
            encoding="utf-8",
        )
        _concat_scenes(wrapped_concat, final_path)
        for transient_path in (*wrapped_paths, body_path, wrapped_concat):
            if transient_path.exists() and transient_path != final_path:
                transient_path.unlink()

    manifest = {
        "schema_version": 1,
        "duration_sec": round(probe_duration(final_path), 3),
        "tts": plan_payload["tts"],
        "video": plan_payload["video"],
        "video_mode": video_mode,
        "subtitles_burned": burn_subtitles,
        "wrappers": {
            "intro": str(intro_video) if intro_video else None,
            "outro": str(outro_video) if outro_video else None,
        },
        "files": {
            "final": final_path.name,
            "subtitles": subtitle_path.name,
            "script": script_path.name,
            "plan": plan_path.name,
        },
    }
    (output_dir / "video.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return VideoArtifact(
        output_dir=output_dir,
        plan_path=plan_path,
        script_path=script_path,
        subtitle_path=subtitle_path,
        final_path=final_path,
    )


def _compose_scene(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    fps: int,
    duration: float,
) -> None:
    _run_ffmpeg(
        [
            "-loop",
            "1",
            "-framerate",
            str(fps),
            "-i",
            image_path.name,
            "-i",
            audio_path.name,
            "-vf",
            _motion_filter(
                duration=duration,
                width=image_path_width(image_path),
                height=image_path_height(image_path),
                fps=fps,
            ),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            "-t",
            f"{duration:.3f}",
            "-movflags",
            "+faststart",
            output_path.name,
        ],
        cwd=output_path.parent,
    )


def _motion_filter(*, duration: float, width: int, height: int, fps: int) -> str:
    fade_duration = min(0.35, max(0.12, duration / 8))
    fade_out_start = max(0.0, duration - fade_duration)
    return (
        f"scale={width * 2}:{height * 2}:flags=lanczos,"
        "zoompan="
        "z='min(zoom+0.0008,1.08)':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={width}x{height}:fps={fps},"
        f"fade=t=in:st=0:d={fade_duration:.3f}:color=0xF6F5F1,"
        f"fade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f}:color=0xF6F5F1"
    )


def image_path_width(image_path: Path) -> int:
    with Image.open(image_path) as image:
        return image.width


def image_path_height(image_path: Path) -> int:
    with Image.open(image_path) as image:
        return image.height


def _concat_scenes(concat_path: Path, output_path: Path) -> None:
    _run_ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path.name,
            "-c",
            "copy",
            output_path.name,
        ],
        cwd=output_path.parent,
    )


def _normalize_media_clip(
    source_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
) -> Path:
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise VideoError(f"Wrapper video not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i",
            str(source_path),
            "-vf",
            (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"fps={fps},setsar=1"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            output_path.name,
        ],
        cwd=output_path.parent,
    )
    return output_path


def extract_voice_reference(
    video_path: str | Path,
    output_path: str | Path,
    *,
    duration: float = 12,
) -> Path:
    source_path = Path(video_path).resolve()
    output_path = Path(output_path)
    if not source_path.is_file():
        raise VideoError(f"Voice reference video not found: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            "-i",
            str(source_path),
            "-vn",
            "-t",
            str(duration),
            "-af",
            "highpass=f=80,afftdn=nf=-25,loudnorm=I=-18:LRA=7:TP=-2",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_path.name,
        ],
        cwd=output_path.parent,
    )
    return output_path


def _run_ffmpeg(arguments: list[str], *, cwd: Path) -> None:
    command = [
        _require_executable("ffmpeg"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    command.extend(arguments)
    try:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        message = detail[-1] if detail else "unknown FFmpeg error"
        raise VideoError(f"FFmpeg failed: {message}") from exc


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VideoError(f"Required executable not found: {name}")
    return path


def _default_tts_engine() -> TtsEngine:
    return SapiTtsEngine() if sys.platform == "win32" else EdgeTtsEngine()


def _script_markdown(plan: dict[str, Any]) -> str:
    lines = ["# Video Script", ""]
    for index, scene in enumerate(plan["scenes"], 1):
        lines.extend(
            [
                f"## {index}. {scene['title']}",
                "",
                f"`{scene['section']}` · `{scene['kind']}`",
                "",
                scene["narration"],
                "",
            ]
        )
    return "\n".join(lines)


def _compact_scene(scene: VideoScene) -> VideoScene:
    if scene.kind in {"problem", "method", "result", "takeaway"}:
        return replace(
            scene,
            body=_trim(scene.body, 180),
            narration=_trim(scene.narration, 240),
        )
    if scene.kind == "figure":
        return replace(
            scene,
            body=_trim(scene.body, 100),
            narration=_trim(scene.narration, 220),
        )
    return scene


def _compact_promo_scene(scene: VideoScene) -> VideoScene:
    if scene.kind == "hook":
        return replace(
            scene,
            body=_trim(scene.body, 64),
            narration=_trim(scene.narration, 72),
        )
    if scene.kind in {"problem", "method", "result"}:
        return replace(
            scene,
            body=_trim(scene.body, 54),
            narration=_trim(scene.narration, 68),
        )
    if scene.kind == "figure":
        return replace(
            scene,
            body=_trim(scene.body, 60),
            narration=_trim(scene.narration, 70),
        )
    return replace(scene, narration=_trim(scene.narration, 48))


def _append_closing_narration(scene: VideoScene) -> VideoScene:
    narration = scene.narration.rstrip()
    if narration.endswith(CLOSING_NARRATION):
        return scene
    separator = " " if narration[-1:].isascii() and narration[-1:].isalnum() else ""
    return replace(
        scene,
        narration=f"{narration}{separator}{CLOSING_NARRATION}",
    )


def _attach_scene_visuals(
    paper_dir: Path,
    scenes: list[VideoScene],
) -> list[VideoScene]:
    visuals = _figure_items(paper_dir, limit=4)
    if not visuals:
        return scenes
    visual_index = 0
    visual_kinds = {"cover", "hook", "problem", "method", "result", "takeaway", "cta"}
    attached = []
    for scene in scenes:
        if scene.kind in visual_kinds and not scene.figure:
            scene = replace(scene, figure=visuals[visual_index % len(visuals)][0])
            visual_index += 1
        attached.append(scene)
    return attached


def _deep_scene_module(scene: VideoScene) -> str:
    if scene.kind in {"cover", "paper_info"}:
        return "paper_info"
    if scene.kind == "innovation":
        return "innovation"
    if scene.kind in {"method", "formula"}:
        return "method"
    if scene.kind == "experiment":
        return "experiment"
    return "summary"


def _apply_deep_scene_visuals(
    scenes: list[VideoScene],
    figures: list[tuple[str, str]],
    *,
    preferred_experiment_figures: list[str],
) -> list[VideoScene]:
    if not figures:
        return scenes
    preferred_experiment = set(preferred_experiment_figures)
    used = set()
    attached = []
    for scene in scenes:
        preferred = preferred_experiment if scene.kind == "experiment" else set()
        ranked = sorted(
            figures,
            key=lambda candidate: _figure_match_score(
                scene,
                candidate,
                preferred_paths=preferred,
                used_paths=used,
            ),
            reverse=True,
        )
        figure_path = ranked[0][0]
        used.add(figure_path)
        attached.append(replace(scene, figure=figure_path))
    return attached


def _figure_match_score(
    scene: VideoScene,
    candidate: tuple[str, str],
    *,
    preferred_paths: set[str],
    used_paths: set[str],
) -> float:
    path, caption = candidate
    scene_text = f"{scene.scene_id} {scene.title} {scene.body} {scene.section}".lower()
    caption_text = caption.lower()
    score = 6.0 if path in preferred_paths else 0.0
    score -= 3.0 if path in used_paths else 0.0

    method_terms = (
        "架构",
        "框架",
        "模型",
        "结构",
        "流程",
        "模块",
        "训练",
        "方法",
        "architecture",
        "framework",
        "pipeline",
        "model",
        "overview",
    )
    result_terms = (
        "结果",
        "实验",
        "对比",
        "基准",
        "性能",
        "表现",
        "评测",
        "准确率",
        "误差",
        "消融",
        "曲线",
        "可视化",
        "result",
        "experiment",
        "comparison",
        "benchmark",
        "ablation",
        "performance",
        "evaluation",
    )
    if scene.kind in {"method", "formula"}:
        score += sum(4 for term in method_terms if term in caption_text)
        score -= sum(2 for term in result_terms if term in caption_text)
    elif scene.kind == "experiment":
        score += sum(4 for term in result_terms if term in caption_text)
        score -= sum(2 for term in method_terms if term in caption_text)
    elif scene.kind in {"innovation", "cover", "paper_info", "summary"}:
        score += sum(1 for term in method_terms + result_terms if term in caption_text)

    english_scene = set(re.findall(r"[a-z][a-z0-9-]{2,}", scene_text))
    english_caption = set(re.findall(r"[a-z][a-z0-9-]{2,}", caption_text))
    score += 2 * len(english_scene & english_caption)
    chinese_scene = _chinese_bigrams(scene_text)
    chinese_caption = _chinese_bigrams(caption_text)
    score += min(5.0, 0.8 * len(chinese_scene & chinese_caption))
    is_overview = (
        "总览" in scene_text or "overview" in scene_text or "核心创新" in scene_text
    )
    if is_overview:
        if any(term in caption_text for term in ("架构", "框架", "architecture")):
            score += 15
        if scene.kind == "innovation" and any(
            term in caption_text for term in ("概览", "overview")
        ):
            score += 15
    return score


def _chinese_bigrams(text: str) -> set[str]:
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
    return {chinese[index : index + 2] for index in range(len(chinese) - 1)}


def _author_label(authors: list[str], *, limit: int = 5) -> str:
    if len(authors) <= limit:
        return "、".join(authors)
    return f"{'、'.join(authors[:limit])}等"


def _limit_deep_scenes(
    scenes: list[VideoScene],
    *,
    max_scenes: int,
) -> list[VideoScene]:
    limit = max(1, max_scenes)
    if len(scenes) <= limit:
        return scenes

    def first(*predicates):
        for predicate in predicates:
            for scene in scenes:
                if predicate(scene):
                    return scene
        return None

    priority = [
        first(lambda scene: scene.kind == "cover"),
        first(lambda scene: scene.kind == "paper_info"),
        first(lambda scene: scene.scene_id == "02-innovation-overview"),
        first(lambda scene: scene.scene_id == "03-method-overview"),
        first(
            lambda scene: scene.kind == "formula",
            lambda scene: scene.kind == "method",
        ),
        first(lambda scene: scene.scene_id == "04-experiment-setup"),
        first(lambda scene: scene.scene_id.startswith("04-result-")),
        first(lambda scene: scene.scene_id == "05-value"),
        first(lambda scene: scene.scene_id == "05-limitations"),
    ]
    selected = []
    for scene in priority:
        if scene is not None and scene not in selected:
            selected.append(scene)
    for scene in scenes:
        if scene not in selected:
            selected.append(scene)
        if len(selected) >= limit:
            break
    order = {scene.scene_id: index for index, scene in enumerate(scenes)}
    return sorted(selected[:limit], key=lambda scene: order[scene.scene_id])


def _figure_items(paper_dir: Path, *, limit: int) -> list[tuple[str, str]]:
    translated = _translation_content(paper_dir)
    translated_captions = {
        str(figure.get("path")): str(figure.get("caption_zh") or "")
        for figure in translated.get("figures") or []
    }
    parse_path = paper_dir / "parse.json"
    if not parse_path.is_file():
        return []
    try:
        parsed = _read_json(parse_path)
    except VideoError:
        return []
    figures = []
    for page in parsed.get("pages") or []:
        blocks = (page.get("raw") or {}).get("parsing_res_list") or []
        captions = [
            str(block.get("block_content") or "").strip()
            for block in blocks
            if block.get("block_label")
            in {"figure_title", "chart_title", "vision_footnote"}
            and str(block.get("block_content") or "").strip()
        ]
        for image_index, image in enumerate(page.get("images") or []):
            relative = str(image.get("path") or "").replace("\\", "/")
            if not relative or not (paper_dir / relative).is_file():
                continue
            lowered = relative.lower()
            if "header_image" in lowered or "footer_image" in lowered:
                continue
            try:
                with Image.open(paper_dir / relative) as image_file:
                    if image_file.width < 250 or image_file.height < 160:
                        continue
            except OSError:
                continue
            caption = translated_captions.get(relative)
            if not caption and image_index < len(captions):
                caption = captions[image_index]
            figures.append((relative, caption or ""))
            if len(figures) >= limit:
                return figures
    return figures


def _first_figure(paper_dir: Path) -> str | None:
    figures = _figure_items(paper_dir, limit=1)
    return figures[0][0] if figures else None


def _translation_content(paper_dir: Path) -> dict:
    path = paper_dir / "translation.json"
    if not path.is_file():
        return {}
    try:
        payload = _read_json(path)
    except VideoError:
        return {}
    return payload


def _sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [
        part.strip()
        for part in re.split(
            r"(?<=[!?])\s+|(?<=[。！？；])\s*|(?<=\.)\s+",
            normalized,
        )
        if part.strip()
    ]


def _select_sentence(
    sentences: list[str],
    pattern: str,
    *,
    fallback: str,
    exclude: set[str],
) -> str:
    regex = re.compile(pattern, re.IGNORECASE)
    for sentence in sentences:
        if sentence not in exclude and regex.search(sentence):
            return sentence
    return fallback


def _trim(text: str, limit: int = 360) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoError(f"Cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise VideoError(f"JSON object required: {path}")
    return payload


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc")
        if bold
        else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arialbd.ttf")
        if bold
        else Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_lines: int,
    sizes: list[int],
    bold: bool,
) -> ImageFont.FreeTypeFont:
    for size in sizes:
        font = _font(max(10, size), bold=bold)
        lines = _wrap_lines(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return font
    return _font(max(10, sizes[-1]), bold=bold)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    origin: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
    max_width: int,
    *,
    max_lines: int,
    line_spacing: float,
) -> int:
    lines = _wrap_lines(draw, text, font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = _ellipsize(draw, lines[-1] + "…", font, max_width)
    x, y = origin
    sample = draw.textbbox((0, 0), "Ag", font=font)
    line_height = int((sample[3] - sample[1]) * line_spacing)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.,;:!?%+\-]*\s*|.", paragraph)
        current = ""
        for token in tokens:
            candidate = current + token
            if current and _text_width(draw, candidate, font) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
            while current and _text_width(draw, current, font) > max_width:
                split_at = max(1, len(current) - 1)
                while (
                    split_at > 1
                    and _text_width(draw, current[:split_at], font) > max_width
                ):
                    split_at -= 1
                lines.append(current[:split_at].rstrip())
                current = current[split_at:].lstrip()
        if current:
            lines.append(current.rstrip())
        elif not tokens:
            lines.append("")
    return lines or [""]


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _ellipsize(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> str:
    while text and _text_width(draw, text, font) > max_width:
        text = text[:-1]
    return text


def _paste_contained(
    canvas: Image.Image,
    image_path: Path,
    box: tuple[int, int, int, int],
) -> None:
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    with Image.open(image_path) as source:
        image = ImageOps.contain(
            source.convert("RGB"),
            (max(1, box_width), max(1, box_height)),
            Image.Resampling.LANCZOS,
        )
    x = box[0] + (box_width - image.width) // 2
    y = box[1] + (box_height - image.height) // 2
    canvas.paste(image, (x, y))


def _draw_metrics(
    draw: ImageDraw.ImageDraw,
    metrics: list[Metric],
    *,
    x: int,
    y: int,
    width: int,
    scale: float,
    accent: tuple[int, int, int],
    ink: tuple[int, int, int],
    muted: tuple[int, int, int],
) -> int:
    gap = int(74 * scale)
    slot_width = max(
        int(150 * scale), (width - gap * (len(metrics) - 1)) // len(metrics)
    )
    value_font = _font(max(30, int(92 * scale)), bold=True)
    unit_font = _font(max(12, int(24 * scale)), bold=True)
    bottom = y
    for index, metric in enumerate(metrics):
        left = x + index * (slot_width + gap)
        draw.text((left, y), metric.value, font=value_font, fill=ink)
        unit_width = _text_width(draw, metric.value, value_font)
        draw.text(
            (left + unit_width + int(12 * scale), y + int(46 * scale)),
            metric.unit,
            font=unit_font,
            fill=accent,
        )
        draw.line(
            (
                left,
                y + int(109 * scale),
                left + min(slot_width, int(185 * scale)),
                y + int(109 * scale),
            ),
            fill=accent,
            width=max(1, int(3 * scale)),
        )
        draw.text(
            (left, y + int(122 * scale)),
            f"metric {index + 1:02d}",
            font=_font(max(10, int(18 * scale))),
            fill=muted,
        )
        bottom = max(bottom, y + int(150 * scale))
    return bottom


def _wrap_subtitle(text: str, *, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.,;:!?%+\-]*\s*|.", text)
    for token in tokens:
        if len((current + token).rstrip()) <= width:
            current += token
            continue
        if current.strip():
            lines.append(current.strip())
        current = token.lstrip()
        while len(current) > width:
            lines.append(current[:width])
            current = current[width:]
    if current:
        lines.append(current.rstrip())
    if len(lines) > 2:
        lines = lines[:2]
        lines[-1] = lines[-1][: width - 1].rstrip() + "…"
    return lines


def _srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _valid_wave(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() > 0
    except (OSError, wave.Error):
        return False
