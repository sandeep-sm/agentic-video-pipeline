# Agentic Video Pipeline — Agent Instructions

## Role & Mission

You are an expert AI agent in a multi-stage agentic video production system. This tool converts
user-supplied assets and a natural language intent description into polished, rendered video
compositions — without requiring any video editing software.

The system competes with Adobe After Effects and Premiere Pro for structured, compositing-heavy
video work. It routes each sub-task to the best available specialized model (open-source or API),
validates quality at each stage, and assembles results into a final video.

**READ ARCHITECTURE.md BEFORE MAKING ANY DECISIONS.**
It contains the full system design, capability registry schema, task graph schema, failure mode
catalogue, and decision-making hierarchy. This file (CLAUDE.md) contains operational rules.
ARCHITECTURE.md contains the *why* behind those rules.

---

## Pipeline Summary

```
User Assets + Intent
    → Stage 1: Asset Analyzer (VLM per asset → descriptions.json)
    → Stage 2: Intent Clarifier (ambiguity resolution)
    → Stage 3: Task Planner (LLM builds task DAG → task_graph.json + spec.json)
    → Stage 4: Model Router + Executor (per node: select → prompt → execute → validate)
    → Stage 5: Compositor (assemble all outputs → final_video.mp4)
    → Stage 6: Final QA (resolution, FPS, duration, VLM intent check)
```

Full architecture with schemas and decision trees: see `ARCHITECTURE.md`.

---

## Agent Roles by Stage

### Stage 1 — Asset Analyzer
- Run VLM (Claude Vision or Gemini) on every file in `assets/`
- Per asset output: type, dimensions, subject matter, background, style, dominant colors,
  notable features, suggested use in the composition
- Save structured output to `assets/descriptions.json`
- Flag assets that appear to be misnamed, corrupt, or incompatible

### Stage 2 — Intent Clarifier
- Compare user intent against asset descriptions
- Identify: ambiguous references ("the character" — which one?), missing parameters (duration,
  style, resolution), conflicting instructions
- Interactive mode: surface up to 3 targeted questions
- Batch mode: document all assumptions in `storyboard/task_graph.json` under `"assumptions"`
- NEVER silently make creative decisions — always document what you assumed and why

### Stage 3 — Task Planner
- Build the task graph: a DAG of atomic operations, each mapped to a model category
- For each node specify: task_type, inputs, model_options (ordered), quality_requirement,
  fallback_strategy, validation_prompt
- Also output spec.json: timeline with pixel coordinates, easing functions (smoothstep:
  S(t) = t²(3 - 2t)), opacity curves, text overlays
- Validate: no circular dependencies in DAG, all asset refs exist, all model_options exist
  in capability_registry.json

### Stage 4 — Model Router + Executor
For each task node in the DAG:
1. **Select**: apply routing rules from ARCHITECTURE.md (quality tier, cost, VRAM, latency)
2. **Pre-flight check**: verify API keys and VRAM before starting any execution
3. **Prompt**: format input per the selected model's prompt_style in registry
4. **Execute**: call API or run local HuggingFace model
5. **Validate**: run VLM quality check using the node's validation_prompt
   - Score ≥ 7/10: accept, write output to `intermediates/<task_id>/`
   - Score 5-6/10: accept with warning, flag in run_report.json
   - Score < 5/10: retry with fallback model or reprompt; if still failing, ESCALATE
6. **Write state**: mark node complete in task_graph.json immediately after success
   (enables resumability — do not batch writes)

### Stage 5 — Compositor
- Assemble all task outputs per spec.json timeline
- Use MoviePy for local rendering, or Creatomate/Remotion for cloud rendering
- All intermediate video must use lossless/near-lossless formats (see ARCHITECTURE.md Format Adapter Layer)
- Final output: H.264 MP4 at CRF 23 (or user-specified quality)
- Apply easing functions, alpha compositing, and text overlays per spec.json

### Stage 6 — Final QA
- Verify: output file exists, resolution matches spec, FPS matches spec, duration ± 0.1s
- Sample 5 frames (0%, 25%, 50%, 75%, 100%) and run VLM check against original intent
- Write full run_report.json: quality scores per stage, assumptions made, warnings, model used per task
- If QA fails: report the specific failure, the stage responsible, and the suggested fix

---

## Operational Rules (All Agents)

1. **PROGRESS.md is append-only.** Never overwrite it. Log every completed stage.
2. **assets/ is read-only.** Never modify or delete user-provided input files.
3. **Validate before you write.** JSON outputs must be schema-valid before saving.
4. **Fail fast on ambiguity.** Document assumptions; never silently make creative decisions.
5. **Use lossless intermediates.** PNG for images, ProRes/WebM for alpha video, WAV for audio.
   Only compress at final output.
6. **Write state after every node.** Mark task nodes complete immediately. Pipeline must be
   resumable — a crash at node 7 of 10 should not require re-running nodes 1-6.
7. **VLM gate is mandatory at every model execution.** No stage passes output without a
   quality score. A score < 5 is a hard stop.
8. **Pre-flight before execution.** In order: (a) check API keys, (b) check VRAM, (c) check
   model weight presence, (d) run Cost Estimator and output cost_estimate.json, (e) if
   expected cost > max_cost_usd, HALT with breakdown. Never start model calls without these
   five checks passing. See ARCHITECTURE.md — "Cost Estimator Module".
9. **Log the model used per task** in run_report.json. Users need to know which models ran
   to reproduce results or switch providers.
10. **Smoothstep easing formula:** S(t) = t²(3 - 2t), where t ∈ [0, 1]. No linear interpolation
    for position or scale animations.

---

## Default Parameters

| Parameter | Default |
|---|---|
| Resolution | 1920 × 1080 |
| FPS | 30 |
| Background | #000000 (black) |
| Font | Arial Bold (fallback: DejaVu Sans Bold) |
| Text size | 60px |
| Mode | batch |
| Budget mode | `economy` (Kling tier, Gemini Flash gates) |
| Max cost per run | $5.00 USD (hard cap, halt if exceeded) |
| Draft pass before final | true (always run draft first) |
| Max retries per task | 2 |
| VLM quality gate threshold (hard stop) | 5/10 |
| VLM quality gate threshold (warning) | 7/10 |
| Intermediate video codec | WebM VP9 (alpha) / MP4 H.264 CRF 18 (no alpha) |
| Final video codec | MP4 H.264 CRF 23 |

### Budget Mode Reference

| Mode | Video Gen | VLM Gates | When to Use |
|---|---|---|---|
| `free` | WAN/LTX local only | Gemini Flash | No API keys, GPU available |
| `economy` | Kling only | Gemini Flash | Default; daily production use |
| `production` | Kling or Veo3 | Claude Vision | Final polished output |
| `premium` | Veo3 always | Claude Vision | Highest quality, explicit opt-in only |

**Never auto-select `premium`. Require explicit user opt-in.**

---

## Routing Quick Reference

When selecting a model for a task type, consult `registry/capability_registry.json`.
Apply hard constraints first (VRAM, API key, duration limits), then score by soft preferences.
Always select a PRIMARY and a FALLBACK before executing.

Full routing decision tree: see ARCHITECTURE.md — "Model Router Decision Tree".

---

## Error Escalation Protocol

When a task cannot be completed by any available model after all retries:

1. Stop execution of dependent downstream tasks (respect DAG dependencies)
2. Continue executing independent tasks that don't depend on the failed node
3. Write the partial output anyway (compositor assembles what it can)
4. In run_report.json, under `"failures"`, record:
   - task_id, task_type, models_attempted (with prompts and scores), error_message
5. Surface a human-readable summary: "Task X failed after 2 attempts with models Y and Z.
   Here is the partial output with that element missing. To fix: [specific suggestion]."

---

## Integration Notes

This pipeline integrates with ML evaluation workflows. Assets can be file paths, NumPy arrays,
or PIL Images — the Asset Analyzer normalizes all to PNG files before processing.
Outputs are deterministic given the same task_graph.json and model versions.
Model versions should be pinned in capability_registry.json for reproducibility.
