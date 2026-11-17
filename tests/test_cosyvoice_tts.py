import os

import pytest

from paper2voice.cli import _build_tts_engine, build_parser
from paper2voice.cosyvoice_tts import (
    CosyVoiceError,
    CosyVoiceTtsEngine,
    configure_runtime_cache,
    configure_transformers_runtime,
    load_wav_soundfile,
    prepare_runtime_precision,
)
from paper2voice.dashboard import _tts_engine
from paper2voice.settings import AppSettings


def test_configure_runtime_cache_redirects_numba_to_writable_directory(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    cache_dir = tmp_path / "nested" / "numba-cache"

    configured = configure_runtime_cache(cache_dir)

    assert configured == cache_dir.resolve()
    assert configured.is_dir()
    assert os.environ["NUMBA_CACHE_DIR"] == str(configured)


def test_configure_transformers_runtime_disables_unusable_flash_attention():
    from transformers.utils import import_utils

    configure_transformers_runtime()

    assert import_utils._is_package_available("flash_attn") is False


def test_load_wav_soundfile_resamples_stereo_to_mono(tmp_path):
    soundfile = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    path = tmp_path / "reference.wav"
    soundfile.write(path, [[0.25, 0.25]] * 4410, 44100, subtype="PCM_16")

    speech = load_wav_soundfile(path, 24000)

    assert speech.shape == (1, 2400)
    assert torch.allclose(
        speech[:, 100:-100],
        torch.full((1, 2200), 0.25),
        atol=2e-4,
    )


def test_cosyvoice_engine_combines_chunks_into_playable_wave(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    reference = tmp_path / "reference.wav"
    soundfile.write(reference, [0.1] * 2400, 24000, subtype="PCM_16")
    captured = {}

    class FakeModel:
        sample_rate = 24000

        def inference_instruct2(self, text, instruction, prompt_wav, *, stream=False):
            captured["text"] = text
            captured["instruction"] = instruction
            captured["prompt_wav"] = prompt_wav
            captured["stream"] = stream
            yield {"tts_speech": torch.full((1, 240), 0.2)}
            yield {"tts_speech": torch.full((1, 120), 0.2)}

    engine = CosyVoiceTtsEngine(reference, model_dir=tmp_path / "model")
    monkeypatch.setattr(engine, "_load_model", lambda: FakeModel())
    output = tmp_path / "speech.wav"

    engine.synthesize("论文讲解测试。", output)

    info = soundfile.info(output)
    assert info.samplerate == 24000
    assert info.frames == 360
    assert captured == {
        "text": "论文讲解测试。",
        "instruction": engine.instruction,
        "prompt_wav": str(reference.resolve()),
        "stream": False,
    }


def test_cosyvoice_engine_rejects_empty_model_output(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")
    reference = tmp_path / "reference.wav"
    soundfile.write(reference, [0.1] * 2400, 24000, subtype="PCM_16")

    class EmptyModel:
        sample_rate = 24000

        def inference_instruct2(self, text, instruction, prompt_wav, *, stream=False):
            return iter(())

    engine = CosyVoiceTtsEngine(reference, model_dir=tmp_path / "model")
    monkeypatch.setattr(engine, "_load_model", lambda: EmptyModel())

    with pytest.raises(CosyVoiceError, match="did not generate audio"):
        engine.synthesize("测试", tmp_path / "speech.wav")


def test_cosyvoice_engine_splits_long_narration_before_inference(tmp_path, monkeypatch):
    soundfile = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    reference = tmp_path / "reference.wav"
    soundfile.write(reference, [0.1] * 2400, 24000, subtype="PCM_16")

    class FakeModel:
        sample_rate = 24000

        def __init__(self):
            self.texts = []

        def inference_instruct2(self, text, instruction, prompt_wav, *, stream=False):
            self.texts.append(text)
            yield {"tts_speech": torch.full((1, 100), 0.2)}

    model = FakeModel()
    engine = CosyVoiceTtsEngine(reference, model_dir=tmp_path / "model")
    monkeypatch.setattr(engine, "_load_model", lambda: model)
    output = tmp_path / "speech.wav"

    engine.synthesize(
        "第一句介绍论文问题，并说明研究背景和任务边界。"
        "第二句解释核心方法，包括模型结构和关键设计。"
        "第三句总结实验结果，并说明方法适用的范围。"
        "第四句讨论局限，并给出读者可以继续关注的方向。",
        output,
    )

    assert len(model.texts) > 1
    assert all(len(text) <= 72 for text in model.texts)
    assert soundfile.info(output).frames == 100 * len(model.texts)


def test_prepare_runtime_precision_aligns_llm_with_float32_inputs():
    torch = pytest.importorskip("torch")

    class Runtime:
        model = type(
            "Model",
            (),
            {"llm": torch.nn.Linear(2, 2).to(torch.bfloat16)},
        )()

    runtime = Runtime()
    prepare_runtime_precision(runtime)

    assert runtime.model.llm.weight.dtype == torch.float32


def test_dashboard_uses_cosyvoice3_for_default_intro_voice(tmp_path):
    reference = tmp_path / "intro.mp4"
    settings = AppSettings(
        voice_clone_enabled=True,
        intro_video=str(reference),
    )

    engine = _tts_engine(settings)

    assert isinstance(engine, CosyVoiceTtsEngine)
    assert engine.reference_source == reference.resolve()


def test_cli_builds_cosyvoice3_engine_from_voice_reference(tmp_path):
    reference = tmp_path / "intro.mp4"
    args = build_parser().parse_args(
        [
            "video",
            str(tmp_path / "bundle"),
            "--tts",
            "cosyvoice",
            "--reference-video",
            str(reference),
        ]
    )

    engine = _build_tts_engine(args.tts, args.voice, args.reference_video)

    assert isinstance(engine, CosyVoiceTtsEngine)
    assert engine.reference_source == reference.resolve()


def test_settings_accept_cosyvoice3_provider():
    settings = AppSettings(
        tts="cosyvoice",
        voice_clone_enabled=False,
    )

    settings.validate()
    assert isinstance(_tts_engine(settings), CosyVoiceTtsEngine)
