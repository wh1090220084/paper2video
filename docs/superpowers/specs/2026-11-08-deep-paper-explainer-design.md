# Deep Paper Explainer Design

## Goal

Generate a technical 8-10 minute Chinese paper explainer for AI graduate
students and developers using a fixed five-chapter structure.

## Audience

- Readers understand deep learning, Transformers, diffusion, LLMs, standard
  benchmarks, and common metrics.
- Do not explain foundational concepts unless the paper changes their meaning.
- Prefer concrete claims, architecture flow, evidence, efficiency, and limits.

## Fixed Structure

1. Paper information
2. Innovations
3. Method
4. Experimental results
5. Summary

Target 2,500-2,800 Chinese narration characters and approximately 9 minutes
30 seconds of speech.

## Chapter Contract

### Paper Information

- Title, authors, institutions, publication status, arXiv/DOI, code and project
  links.
- Explain the research problem in one sentence.
- Missing metadata is displayed as `未提供`.

### Innovations

- Default to three distinct, evidence-backed claims.
- Each claim answers: what was missing, what changed, and why the difference
  matters.
- Do not repeat implementation details that belong in the method chapter.

### Method

- Explain inputs, outputs, architecture, module flow, training objective, and
  inference flow.
- Keep at most three formulas total: one overall objective and no more than two
  critical module formulas.
- Explain symbols and dimensions on first use.
- Do not derive formulas.

### Experimental Results

- Cover datasets, baselines, metrics, main results, ablations, and one
  visualization.
- State absolute values and deltas when available.
- Include efficiency costs when reported: parameters, memory, latency, and
  throughput.
- Do not read full tables aloud.

### Summary

- State value, applications, engineering cost, limitations, and likely impact.
- End narration with `今天的论文讲解到此结束。`

## Visual Direction

- Continue the `cinematic` theme derived from `mode/开头.mp4`.
- Use gold for conclusions and important values.
- Use cyan for data flow, source markers, and technical labels.
- Paper figures remain visible in every concept scene when possible.
- Architecture figures are used for method scenes.
- Result figures and charts are used for experiment scenes.
- Formula scenes combine the formula, symbol explanation, and a source figure.

## Evidence Rules

- Every numeric claim must reference translated source text or a translated
  table.
- Never infer an absolute gain from a relative gain.
- Distinguish paper-reported numbers from calculated deltas.
- Preserve benchmark names, dataset splits, and metric names.

## Output Contract

Translation produces `explainer.zh.json` with:

```json
{
  "schema_version": 1,
  "audience": "ai_graduate_developer",
  "paper_info": {},
  "innovations": [],
  "method": {"steps": [], "formulas": []},
  "experiments": {"main_results": [], "ablations": [], "visual_figures": []},
  "summary": {"applications": [], "limitations": []}
}
```

The video planner creates 16-18 scenes from this contract. It must not invent
missing facts; it omits unavailable details or marks them as `未提供`.
Deep mode requires `explainer.zh.json`; a missing or invalid file raises a clear
configuration error instead of silently falling back to the legacy plan.
