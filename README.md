# paper2voice

Paper acquisition and parsing pipeline for turning an arXiv paper into a
normalized bundle that downstream OCR, translation, scripting, and video stages
can consume.

The acquisition layer uses only the Python standard library. OCR is an optional
dependency loaded only by the `parse` command.

## Quick Start: Launch the Web Dashboard

The recommended entry point is the local scheduling dashboard. From the
repository root:

```powershell
python -m paper2voice dashboard
```

The command starts the web service and opens the default browser. If the browser
does not open automatically, visit:

```text
http://127.0.0.1:8765
```

Use another port when `8765` is already occupied:

```powershell
python -m paper2voice dashboard --port 8766
```

The dashboard stores settings in `data/video-settings.json`. It supports four
visual themes (`cinematic`, `light_green`, `minimal`, `tech`), selectable content
modules, voice and resolution options, weekday schedules, and immediate runs.

The default `cinematic` theme follows the gold-and-cyan visual language of
`mode/开头.mp4` and combines paper figures with the explanatory text on the same
frame. Long-form videos use the fixed five-chapter structure
`论文信息 / 创新点 / 方法 / 实验结果 / 总结`, target 16-18 scenes, and keep at
most three core formulas. Scheduled jobs run while the dashboard process is
active.

The paper source field accepts either a local bundle directory or an arXiv
abstract URL; URLs are fetched and normalized into `data/papers/<arxiv_id>`
before video generation.

The page also provides a `promo` mode for 3:4 social explainers. It produces a
roughly one-minute portrait video with a hook, problem, method, result, paper
figure, and CTA, using the cloned intro voice. Promo mode keeps narration
concise and writes subtitles to a separate SRT without burning them into the
video.

The dashboard pipeline runs `PaddleOCR-VL 1.6` before video generation and
extracts figures, figure captions, tables, and text blocks from `parse.json`.
Enable `大模型翻译` in the page to translate the paper through an
OpenAI-compatible `/chat/completions` endpoint. API credentials can also be
provided through:

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:PAPER2VOICE_LLM_API_URL = "https://api.deepseek.com/v1"
$env:PAPER2VOICE_LLM_MODEL = "deepseek-flash"
```

API keys are stored only in the local settings JSON and are redacted from the
browser API. Translation produces `translation.json`, `paper.zh.md`, and
`digest.zh.md`. Video generation automatically uses the translated title and
abstract when available. `翻译正文上限` defaults to `0`, which translates the
entire parsed document; set a positive value to control API usage. The
`测试连接` button sends a minimal request through the current form values and
reports latency without persisting the key.

For the full video pipeline, install the OCR and video dependencies described
below first.

## Pipeline Scope

- fetch one paper from the official arXiv API by ID or URL;
- import `arxiv-pulse` JSON without importing its Python internals;
- normalize IDs, metadata, and URLs into one contract;
- download and validate `paper.pdf` with atomic writes;
- parse `paper.pdf` into page JSON, Markdown, and local image assets;
- generate an explainer script, narration, slides, subtitles, and MP4.

## Installation

### OCR / PDF Parsing

Install PaddleOCR-VL for PDF parsing:

```powershell
python -m venv tmp\ocr-venv
tmp\ocr-venv\Scripts\python.exe -m pip install -r requirements-ocr.txt
```

The dashboard automatically prefers `tmp/ocr-venv`, then `.venv`, and finally
the Python running the dashboard. Keep OCR in its own environment when CosyVoice
uses Torch in the main environment; mixing Torch and Paddle GPU CUDA DLLs on
Windows can cause `WinError 127` during Paddle imports.

For an NVIDIA GPU with CUDA 12.6, replace the CPU Paddle package in the isolated
environment:

```powershell
tmp\ocr-venv\Scripts\python.exe -m pip uninstall -y paddlepaddle
tmp\ocr-venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.1 `
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ `
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

PaddleOCR-VL `v1.6` requires `paddleocr>=3.7` with its `doc-parser` extras.

### Video

Install video dependencies:

```powershell
pip install -r requirements-video.txt
```

Video generation requires `ffmpeg` and `ffprobe` on `PATH`.

## CLI Usage

### Fetch a Paper by ID

Fetch the latest version by ID:

```powershell
python -m paper2voice fetch 1706.03762 --output-dir data/papers
```

`fetch` downloads the PDF by default. Use `--no-download` for metadata only.

### Import from arxiv-pulse

Import selected candidates from `arxiv-pulse`:

```powershell
python -m paper2voice import-pulse `
  ..\arxiv-pulse\arxiv_papers\classification_segmentation\top5_20261007.json `
  --output-dir data/papers `
  --limit 3 `
  --download
```

`import-pulse` is metadata-only unless `--download` is passed.

### Parse a Bundle

Parse a normalized bundle:

```powershell
python -m paper2voice parse data\papers\1706.03762
```

The first run may download PaddleOCR-VL models. Later runs reuse the parse when
both source file hashes and the pipeline version are unchanged. The CLI uses
PaddleOCR-VL `v1.6` by default.

### Translate a Parsed Bundle

To translate an already parsed bundle from the CLI:

```powershell
python -m paper2voice translate data\papers\2610.30340
```

### Generate Video

Generate a video from a normalized paper bundle:

```powershell
python -m paper2voice video data\papers\1706.03762
```

Windows uses offline `pyttsx3`/SAPI by default. Use `--tts edge` for the higher
quality Edge voice when the Microsoft speech service is reachable.

Use the existing `paper2v` environment for CosyVoice3 voice cloning:

```powershell
python -m pip install -r requirements-cosyvoice.txt
python -m paper2voice video data\papers\2610.30340 `
  --tts cosyvoice `
  --reference-video mode\voice-reference.wav `
  --intro-video mode\开头.mp4 `
  --outro-video mode\结尾.mp4
```

The CosyVoice3 adapter pins Numba's cache to `tmp/numba-cache`, avoiding the
Windows read-only-package-cache hang, and reads reference WAV files with
`soundfile` so it does not depend on TorchCodec. The default model is
`pretrained_models/Fun-CosyVoice3-0.5B`; no separate Python 3.10 environment
is required. CosyVoice's language-model dependencies must stay on the
Transformers 4.51 line; newer Transformers versions produce speech-like audio
whose content is unintelligible.

When the intro voice option is enabled, the pipeline extracts a cleaned speaker
reference from `mode/开头.mp4` and uses Fun-CosyVoice3-0.5B for body narration.
The intro and outro videos are normalized and wrapped around the generated
body.

## Bundle Contract

Each paper is stored under a stable directory based on its arXiv base ID:

```text
data/papers/
  1706.03762/
    paper.json
    paper.pdf
    parse.json
    parse.md
    parse_assets/
      page_0001_img_in_image_box_1_2_30_40.jpg
    video/
      plan.json
      script.md
      video.srt
      final.mp4
```

`paper.json` uses schema version 1:

```json
{
  "schema_version": 1,
  "paper": {
    "arxiv_id": "1706.03762",
    "full_id": "1706.03762v7",
    "version": 7,
    "title": "Attention Is All You Need",
    "summary": "...",
    "authors": ["..."],
    "published": "2017-06-12T17:57:34Z",
    "updated": "2023-08-02T00:41:18Z",
    "categories": ["cs.CL", "cs.LG"],
    "pdf_url": "https://arxiv.org/pdf/1706.03762v7",
    "abstract_url": "https://arxiv.org/abs/1706.03762v7",
    "metadata": {}
  },
  "assets": {
    "pdf": "paper.pdf",
    "pdf_url": "https://arxiv.org/pdf/1706.03762v7"
  }
}
```

`parse.json` contains source hashes, engine metadata, per-page Markdown, parsed
blocks, and local image paths. Downstream translation and video stages should
use this file rather than depending on PaddleOCR result objects.

## Design Notes

The first video implementation uses extractive narration: it quotes the abstract
to introduce the problem, method, key result, and takeaway, and only uses figures
that already exist under `parse_assets/`. This keeps the first pass grounded even
before an LLM translation/deep-reading stage is connected.

Result scenes promote absolute metrics such as `28.4 BLEU` or `91.3%` while
excluding relative-delta phrases such as "improved by 2 points". Rendered
scenes use a subtle zoom and off-white fade so the video does not remain a
sequence of completely static slides.

This keeps `arxiv-pulse` as an independent upstream and prevents scheduler or
model changes in one repository from breaking the other.

## Development

```powershell
python -m pytest -q
python -m ruff check paper2voice tests
python -m ruff format --check paper2voice tests
```