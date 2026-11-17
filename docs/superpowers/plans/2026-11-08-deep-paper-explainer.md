# Deep Paper Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 16-18 scene, 8-10 minute technical paper explainer using five fixed chapters and the existing cinematic video template.

**Architecture:** Add a validated structured explainer contract between translation and video planning. The LLM produces `explainer.zh.json`; a deterministic planner turns it into five chapters, maps paper figures to scenes, and renders core formulas without free-form hallucination.

**Tech Stack:** Python 3.12, existing OpenAI-compatible translator, JSON Schema-like validation, Pillow renderer, optional Matplotlib mathtext, pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-deep-paper-explainer-design.md`

## Global Constraints

- Target audience: AI graduate students and developers.
- Narration target: 2,500-2,800 Chinese characters.
- Scene target: 16-18 scenes.
- Chapters: paper information, innovations, method, experimental results, summary.
- Formulas: at most three core formulas; no derivations.
- Missing metadata displays as `未提供`.
- All numeric claims must retain an evidence reference.
- Final narration ends with `今天的论文讲解到此结束。`.
- Preserve existing Promo behavior and non-cinematic themes.
- Long mode requires a valid `explainer.zh.json`; never silently use the legacy
  plan in its place.

## Review Focus

- LLM returns malformed or incomplete JSON: generation must fail clearly rather than render empty scenes.
- Fewer than three defensible innovations exist: use the available evidence, not invented claims.
- A result uses relative improvement only: do not present it as an absolute score gain.
- A paper has no figures: cinematic scenes must still render a nonblank technical panel.
- Formula rendering is unavailable: fall back to readable text without dropping the scene.
- `explainer.zh.json` is missing: long generation reports a clear configuration
  error while Promo remains usable.

---

### Task 1: Define and Validate the Explainer Contract

**Files:**
- Create: `paper2voice/explainer.py`
- Create: `tests/test_explainer.py`

**Interfaces:**
- Consumes: translated JSON-shaped dictionaries.
- Produces:
  - `ExplainerError(RuntimeError)`
  - `validate_explainer_payload(payload: dict) -> dict`
  - `load_explainer(bundle_dir: str | Path) -> dict`

- [ ] **Step 1: Write failing validation tests**

Test valid five-chapter payloads, more than three formulas, missing paper fields,
and result claims without evidence.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_explainer.py -q`

Expected: import failure because `paper2voice.explainer` does not exist.

- [ ] **Step 3: Implement validation**

Normalize missing metadata to `未提供`, enforce at most three formulas and at
least one innovation, and require evidence references on numeric result claims.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_explainer.py -q`

Expected: all tests pass.

### Task 2: Generate Structured Explainer Content

**Files:**
- Modify: `paper2voice/translation.py`
- Modify: `tests/test_translation.py`

**Interfaces:**
- Consumes: translated paper markdown, figures, tables, paper metadata.
- Produces:
  - `translation.json` adds `explainer`
  - `explainer.zh.json` stored beside `translation.json`
  - `TranslationArtifact` adds `explainer_path: Path`

- [ ] **Step 1: Write failing translation tests**

Use a fake translator returning a complete JSON object for
`kind="explainer"`. Assert `explainer.zh.json` is written and cached with the
same source hashes as translation.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_translation.py -q`

Expected: no `explainer_path` and no generated file.

- [ ] **Step 3: Implement structured generation**

Extract the first JSON object from the LLM response, validate it through
`validate_explainer_payload`, and atomically write `explainer.zh.json`.

- [ ] **Step 4: Verify malformed JSON failure**

Add a test proving malformed output raises `TranslationError` and does not write
a partial explainer file.

- [ ] **Step 5: Run translation tests**

Run: `python -m pytest tests/test_translation.py -q`

Expected: all tests pass.

### Task 3: Build the Five-Chapter Deep Video Plan

**Files:**
- Modify: `paper2voice/video.py`
- Modify: `tests/test_video.py`
- Create fixture: `tests/fixtures/explainer.zh.json`

**Interfaces:**
- Consumes: `validate_explainer_payload()` output and `parse.json` figure assets.
- Produces:
  - `build_deep_video_plan(bundle_dir: str | Path) -> list[VideoScene]`
  - scene kinds: `paper_info`, `innovation`, `method`, `experiment`, `summary`

- [ ] **Step 1: Write failing plan tests**

Assert chapter order, 16-18 scenes, exactly one paper-info chapter, at most three
formula scenes, evidence-backed experiment scenes, and the fixed closing voice.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_video.py -q`

Expected: `build_deep_video_plan` is missing.

- [ ] **Step 3: Implement deterministic scene planning**

Create chapter overview scenes plus innovation, method, experiment, and summary
detail scenes. Attach figures by method/result keywords first, then round-robin
across remaining concepts.

- [ ] **Step 4: Verify plan tests**

Run: `python -m pytest tests/test_video.py -q`

Expected: all tests pass.

### Task 4: Render Core Formula Cards

**Files:**
- Modify: `paper2voice/video.py`
- Modify: `requirements-video.txt`
- Modify: `tests/test_video.py`

**Interfaces:**
- Produces:
  - `render_formula_image(formula: str, output_path: Path, width: int, height: int) -> Path`
  - method scenes with `kind="formula"` include a rendered formula image path.

- [ ] **Step 1: Write failing renderer test**

Render a short LaTeX formula and assert a nonblank PNG is produced at the
requested dimensions.

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_video.py -k formula -q`

Expected: function is missing.

- [ ] **Step 3: Implement Matplotlib mathtext rendering**

Use transparent mathtext on the cinematic dark panel. On render failure, draw a
readable plain-text fallback and keep the scene.

- [ ] **Step 4: Verify formula tests**

Run: `python -m pytest tests/test_video.py -k formula -q`

Expected: pass.

### Task 5: Integrate Long-Form Generation

**Files:**
- Modify: `paper2voice/video.py`
- Modify: `paper2voice/dashboard.py`
- Modify: `paper2voice/settings.py`
- Modify: `tests/test_dashboard.py`
- Modify: `tests/test_video.py`

**Interfaces:**
- Consumes: `build_deep_video_plan()`.
- Produces: long mode uses structured chapters; Promo still uses `build_promo_plan()`.

- [ ] **Step 1: Write failing integration tests**

Assert long mode uses the deep planner, promo remains unchanged, and module IDs
are the five chapter names. Assert long generation without an explainer fails
with a clear error while Promo still renders.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_dashboard.py tests/test_video.py -q`

Expected: long mode still uses the legacy planner.

- [ ] **Step 3: Wire settings and create_video**

Replace the legacy module IDs with the five chapter IDs and route long mode to
`build_deep_video_plan`.

- [ ] **Step 4: Run integration tests**

Run: `python -m pytest tests/test_dashboard.py tests/test_video.py -q`

Expected: all tests pass.

### Task 6: Update Dashboard and Documentation

**Files:**
- Modify: `paper2voice/web/index.html`
- Modify: `README.md`
- Modify: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: five chapter IDs.
- Produces: dashboard checkboxes show 论文信息、创新点、方法、实验结果、总结.

- [ ] **Step 1: Write failing HTML/API test**

Assert the page exposes the five chapter labels and the API preserves them.

- [ ] **Step 2: Implement dashboard copy and controls**

Replace legacy content-module labels and default selections with the five
chapters.

- [ ] **Step 3: Run browser-independent checks**

Run:

```powershell
python -m pytest -q
python -m ruff check paper2voice tests
python -m ruff format --check paper2voice tests
```

Expected: all checks pass.

### Task 7: End-to-End Verification

**Files:**
- Verify only.

- [ ] **Step 1: Generate a real deep explainer**

Run the pipeline for one paper with translation and CosyVoice enabled.

- [ ] **Step 2: Check scene plan**

Confirm five chapters, 16-18 scenes, at most three formulas, metadata links, and
the fixed closing voice.

- [ ] **Step 3: Check output media**

Run `ffprobe`, inspect key frames, and ASR the final narration.

- [ ] **Step 4: Report actual duration and any residual risk**

Record duration, chapter timings, evidence omissions, and visual issues.
