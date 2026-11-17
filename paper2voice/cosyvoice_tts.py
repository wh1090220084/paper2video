from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
DEFAULT_COSYVOICE_DIR = PROJECT_ROOT / "third_party" / "CosyVoice3"
DEFAULT_NUMBA_CACHE_DIR = PROJECT_ROOT / "tmp" / "numba-cache"
DEFAULT_INSTRUCTION = (
    "You are a helpful assistant. 请用清晰、自然、专业的普通话讲解论文。<|endofprompt|>"
)


class CosyVoiceError(RuntimeError):
    """Raised when the CosyVoice3 runtime cannot synthesize speech."""


def configure_runtime_cache(
    cache_dir: str | Path = DEFAULT_NUMBA_CACHE_DIR,
) -> Path:
    """Point Numba at a writable cache before librosa is imported."""
    configured = Path(cache_dir).expanduser().resolve()
    configured.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(configured)
    return configured


def configure_transformers_runtime() -> None:
    """Hide unusable FlashAttention installs from Transformers 4.51."""
    from transformers.utils import import_utils

    available = import_utils._is_package_available
    if getattr(available, "_paper2voice_flash_attn_disabled", False):
        return

    def filtered_availability(name, *args, **kwargs):
        if name == "flash_attn":
            return False
        return available(name, *args, **kwargs)

    filtered_availability._paper2voice_flash_attn_disabled = True
    import_utils._is_package_available = filtered_availability


def load_wav_soundfile(
    path: str | Path,
    target_sr: int,
    min_sr: int = 16000,
):
    """Load a reference WAV without TorchCodec/FFmpeg codec coupling."""
    import soundfile as sf
    import torch

    samples, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    speech = torch.from_numpy(samples.T.copy()).mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        if sample_rate < min_sr:
            raise CosyVoiceError(
                f"WAV sample rate {sample_rate} must be at least {min_sr}"
            )
        import torchaudio.functional as audio_functional

        speech = audio_functional.resample(
            speech,
            orig_freq=sample_rate,
            new_freq=target_sr,
        )
    return speech


def prepare_runtime_precision(runtime):
    """Keep CosyVoice's language model aligned with its float32 inputs."""
    llm = getattr(getattr(runtime, "model", None), "llm", None)
    if llm is None or not hasattr(llm, "float"):
        raise CosyVoiceError("CosyVoice3 runtime does not expose a language model")
    llm.float()
    return runtime


class CosyVoiceTtsEngine:
    name = "cosyvoice3"
    extension = ".wav"

    def __init__(
        self,
        reference_video: str | Path,
        *,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        instruction: str = DEFAULT_INSTRUCTION,
        numba_cache_dir: str | Path = DEFAULT_NUMBA_CACHE_DIR,
    ):
        self.reference_source = Path(reference_video).expanduser().resolve()
        if self.reference_source.suffix.lower() == ".wav":
            self.reference_wav = self.reference_source
        else:
            self.reference_wav = self.reference_source.with_name("voice-reference.wav")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.instruction = instruction
        self.numba_cache_dir = Path(numba_cache_dir).expanduser().resolve()
        self.voice = self.model_dir.name
        self._model = None

    def synthesize(self, text: str, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        if not self.reference_wav.is_file():
            self._extract_reference_audio()
        try:
            model = self._load_model()
            audios = []
            for segment in _split_narration(text):
                chunks = list(
                    model.inference_instruct2(
                        segment,
                        self.instruction,
                        str(self.reference_wav),
                        stream=False,
                    )
                )
                if not chunks:
                    raise CosyVoiceError("CosyVoice3 did not generate audio")
                for chunk in chunks:
                    audio = chunk["tts_speech"]
                    if audio.ndim == 1:
                        audio = audio.unsqueeze(0)
                    audios.append(audio.detach().float().cpu())
            if not audios:
                raise CosyVoiceError("CosyVoice3 did not generate audio")
            import soundfile as sf
            import torch

            speech = torch.cat(audios, dim=1) if len(audios) > 1 else audios[0]
            sf.write(
                str(output_path),
                speech.squeeze(0).numpy(),
                int(getattr(model, "sample_rate", 24000)),
                format="WAV",
                subtype="PCM_16",
            )
        except CosyVoiceError:
            raise
        except Exception as exc:
            raise CosyVoiceError(f"CosyVoice3 synthesis failed: {exc}") from exc
        if not _valid_wave(output_path):
            raise CosyVoiceError("CosyVoice3 did not produce a playable WAV file")
        return output_path

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_dir.is_dir():
            raise CosyVoiceError(
                f"CosyVoice3 model directory not found: {self.model_dir}"
            )
        configure_runtime_cache(self.numba_cache_dir)
        configure_transformers_runtime()
        for repo_path in (
            DEFAULT_COSYVOICE_DIR,
            DEFAULT_COSYVOICE_DIR / "third_party" / "Matcha-TTS",
        ):
            value = str(repo_path)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            import cosyvoice.cli.frontend as frontend_module
            from cosyvoice.cli.cosyvoice import AutoModel
            from cosyvoice.utils import file_utils

            file_utils.load_wav = load_wav_soundfile
            frontend_module.load_wav = load_wav_soundfile
            self._model = AutoModel(model_dir=str(self.model_dir))
            prepare_runtime_precision(self._model)
        except Exception as exc:
            raise CosyVoiceError(f"Cannot load CosyVoice3 model: {exc}") from exc
        return self._model

    def _extract_reference_audio(self) -> None:
        if not self.reference_source.is_file():
            raise CosyVoiceError(
                f"Voice reference video not found: {self.reference_source}"
            )
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise CosyVoiceError("FFmpeg is required to extract the voice reference")
        self.reference_wav.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(self.reference_source),
                    "-vn",
                    "-t",
                    "12",
                    "-af",
                    "highpass=f=80,afftdn=nf=-25,loudnorm=I=-18:LRA=7:TP=-2",
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(self.reference_wav),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CosyVoiceError(
                f"Cannot extract voice reference audio: {exc}"
            ) from exc


def _split_narration(text: str, *, max_chars: int = 72) -> list[str]:
    units = [
        unit.strip()
        for unit in re.findall(r"[^。！？!?；;\n]+[。！？!?；;]?", text)
        if unit.strip()
    ]
    chunks = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) > max_chars:
            chunks.append(current)
            current = ""
        while len(unit) > max_chars:
            split_at = max_chars
            for index in range(max_chars, 0, -1):
                if unit[index - 1] in "，,、：:":
                    split_at = index
                    break
            chunks.append(unit[:split_at].strip())
            unit = unit[split_at:].strip()
        current = f"{current}{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks or [text]


def _valid_wave(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() > 0
    except (OSError, wave.Error):
        return False
