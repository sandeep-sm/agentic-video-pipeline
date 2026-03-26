# Architectural Decisions Log

> Every significant design decision is recorded here with its rationale.
> A new agent picking up this project MUST read this before proposing changes.
> Do not re-litigate settled decisions without a strong new reason.
> Last updated: 2026-03-25

---

## D-001 | Code generation over diffusion rendering

**Decision:** The system generates executable Python/MoviePy code and runs it to produce video.
It does NOT use diffusion models as the primary renderer.

**Rationale:**
- Diffusion output is non-deterministic: same prompt, different video every time. Unacceptable
  for production pipelines where reproducibility matters.
- Diffusion cannot composite user-supplied assets with pixel-precise coordinate control.
- Code is inspectable, debuggable, and version-controllable. A researcher can reproduce a
  teaser video months later with the exact same output.
- Code is deterministic: same spec.json → same video, always.

**What this does NOT mean:**
- Diffusion models ARE used as specialist workers within the pipeline (e.g., WAN/Kling for
  character animation, FLUX for background generation). They are inputs to the compositor,
  not the renderer.

**Status:** Settled. Foundational.

---

## D-002 | Task graph (DAG) over linear pipeline

**Decision:** The pipeline is represented as a directed acyclic graph (DAG) of atomic tasks,
not a fixed linear sequence of stages.

**Rationale:**
- A linear pipeline cannot handle parallelism: if character animation and background generation
  are independent, they should run concurrently.
- A DAG enables partial failure: if one branch fails, independent branches still complete.
  A partial output is better than no output.
- A DAG is resumable: each completed node is written to disk immediately. A crash at node 7
  does not require re-running nodes 1–6.
- Different videos need different task sets. A simple layout composition has no video generation
  nodes. A DAG accommodates this naturally; a fixed linear pipeline would have empty stages.

**Status:** Settled. Foundational.

---

## D-003 | Capability registry as runtime configuration

**Decision:** All available models are described in `registry/capability_registry.json`.
No model names, costs, or capabilities are hardcoded in pipeline logic.

**Rationale:**
- The model landscape changes fast. Kling 2.0 today, Kling 3.0 in 6 months. New models appear
  on HuggingFace weekly. Hardcoding means code changes every time a new model is available.
- The registry enables the router to make principled decisions at runtime based on constraints
  (VRAM, API keys, budget mode) without code changes.
- Users can add their own proprietary or fine-tuned models by adding a registry entry.
- Reproducibility: pin model versions in the registry to reproduce a specific output later.

**Status:** Settled. Starter registry created at `registry/capability_registry.json`.

---

## D-004 | VLM quality gates are mandatory and non-skippable

**Decision:** Every model execution in Stage 4 is followed by a VLM quality check.
A score < 5/10 on any dimension is a hard stop. Gates cannot be disabled.

**Rationale:**
- In a multi-model pipeline, errors compound silently. A bad video generation leads to a bad
  segmentation mask, which leads to a bad composite. The user only sees the broken final output
  and has no idea which stage failed.
- The cost of a quality gate (1 VLM call ~$0.01) is trivial compared to the cost of running
  downstream stages on a bad input.
- "Optional" quality gates will be skipped under time pressure. Making them mandatory is the
  only way to guarantee pipeline reliability.

**Score thresholds (settled):**
- ≥ 7/10: auto-accept, continue
- 5–6/10: continue with warning logged in run_report.json
- < 5/10: retry with fallback model or reprompt; if still failing, ESCALATE

**Status:** Settled. Non-negotiable.

---

## D-005 | Lossless intermediate formats

**Decision:** All data passed between pipeline stages uses lossless or near-lossless formats:
PNG for images, WebM VP9 / ProRes 4444 for alpha-channel video, MP4 H.264 CRF 18 for
non-alpha video, WAV 48kHz for audio. Only the final output uses lossy compression (CRF 23).

**Rationale:**
- Each re-encode in a lossy format introduces artifacts. Across 6 stages, this compounds
  into visible quality degradation.
- Alpha channel: MP4/H.264 does not support alpha. Using it for intermediate alpha-channel
  video silently drops the transparency data. WebM VP9 or ProRes 4444 are required.
- Color space: intermediate PNG/WAV preserves full color depth; JPEG introduces chroma
  subsampling that can cause banding after color grading steps.

**Status:** Settled.

---

## D-006 | Write task state to disk immediately after each node completes

**Decision:** The moment a task node completes successfully, its output path and status are
written to `storyboard/task_graph.json`. Re-runs skip completed nodes.

**Rationale:**
- A pipeline that takes 20 minutes and crashes at minute 19 should not require starting over.
- API calls are expensive and non-refundable. Kling charges per-second whether or not the
  downstream pipeline succeeds.
- This also enables human-in-the-loop review: a user can interrupt the pipeline after the
  draft video generation step, review the output, and decide whether to proceed to final.

**Status:** Settled.

---

## D-007 | Pre-flight checks before any model execution

**Decision:** Before Stage 4 runs any model call, five checks must pass in order:
1. API keys present for all models in the task graph
2. VRAM sufficient for all local models
3. Model weights downloaded for all local models
4. Cost estimate computed and within budget cap
5. In interactive mode: user has confirmed the cost estimate

**Rationale:**
- The worst user experience: pipeline runs for 15 minutes, then fails because an API key
  was missing or a model was over budget. All prior work was wasted.
- Cost transparency is non-negotiable. Users must know what they're spending before
  they spend it. Surprises on API bills destroy trust.
- VRAM pre-check prevents cryptic CUDA OOM errors mid-run. Fail with a clear message
  ("WAN 2.1 requires 16GB VRAM, only 8GB available — switching to LTX-Video") not a
  stack trace.

**Status:** Settled.

---

## D-008 | Draft-before-final as default workflow

**Decision:** By default, the pipeline runs two passes: a draft pass (free/local models,
50% resolution) followed by a final pass (production-quality models, full resolution).
The final pass only runs after the user approves the draft (interactive) or after draft
quality gates pass (batch).

**Rationale:**
- Users don't know if their prompt produces what they imagine until they see it.
- Running Veo3 on a composition the user then wants to change costs $5–15 wasted.
- Draft mode (LTX-Video local) costs nearly nothing and gives the user a complete preview
  of composition, timing, and text placement.
- This is how all professional creative tools work: proxy edit, then online conform.

**Default budget_mode for draft pass:** `free`
**Default budget_mode for final pass:** `economy` (user can upgrade to `production` or `premium`)

**Status:** Settled.

---

## D-009 | Smoothstep easing as the standard motion curve

**Decision:** All position and scale animations use smoothstep easing: S(t) = t²(3 - 2t).
Linear interpolation is never used for animations.

**Rationale:**
- Linear motion looks mechanical and cheap. Smoothstep gives natural acceleration and
  deceleration that reads as polished.
- Smoothstep is simple to implement, deterministic, and universally understood.
- The formula is a cubic Hermite spline with zero derivative at endpoints — objects
  ease in and ease out smoothly without overshoot.
- Proven in the sigma-gen proof-of-concept to produce professional-looking results.

**Extension:** For bouncy/springy effects, cubic bezier with P1=(0.34, 1.56), P2=(0.64, 1.0)
(similar to CSS `cubic-bezier(0.34, 1.56, 0.64, 1.0)` "back-ease") can be offered as an
optional variant. Smoothstep remains the default.

**Status:** Settled.

---

## D-010 | MoviePy as primary compositor (local), with cloud renderer option

**Decision:** The primary compositor is MoviePy (Python, local). Creatomate and Remotion
are supported as optional cloud rendering backends.

**Rationale:**
- MoviePy is pure Python, ML-pipeline friendly, and accepts NumPy arrays and PIL Images
  directly — critical for integration with ML evaluation workflows.
- Local execution means no cloud dependency, no upload latency, no data privacy concerns.
- Creatomate/Remotion are available as alternatives when cloud rendering is preferred
  (e.g., no GPU available, need managed rendering at scale).
- Remotion has official Claude/LLM integration (llms.txt) — worth supporting long-term.

**Status:** Settled. MoviePy first, cloud renderer optional.

---

## D-011 | Budget mode defaults and escalation rules

**Decision:**
- Default budget_mode: `economy` (Kling tier for video gen, Gemini Flash for VLM gates)
- Default max_cost_usd: $5.00 per run
- `premium` mode requires explicit user opt-in; never auto-selected by the router
- If cost estimate > max_cost_usd: HALT in batch mode, ASK in interactive mode

**Rationale:**
- $5 covers most simple-to-medium productions at `economy` tier.
- Auto-selecting `premium` (Veo3) would produce unexpected bills. A 30s video with 6 Veo3
  clips costs ~$10.50 in video gen alone — above the default cap.
- Users who want premium quality will consciously opt in, meaning they understand the cost.
- The ROI is overwhelmingly positive at every budget tier (see ARCHITECTURE.md — Economics).
  `economy` tier already produces results worth 100–500× the cost.

**Status:** Settled.

---

## D-012 | Asset-intent mismatch is a hard stop, not a warning

**Decision:** If the VLM description of an asset conflicts with what the intent prompt
references (e.g., intent says "animate the dog" but the asset is a landscape), the pipeline
HALTS before any model calls and reports the mismatch to the user.

**Rationale:**
- Running $5 of video generation on the wrong asset is wasteful and produces a confusing
  output. The user will blame the tool, not their asset paths.
- Asset-intent mismatches are almost always user errors (wrong file path, wrong filename).
  Proceeding silently would mask the error.
- Unlike quality failures (which are retryable), asset-intent mismatches require human
  correction. No amount of model retries will fix a wrong file.

**Status:** Settled.

---

## D-013 | code_gen as the universal visual effect spawner

**Decision:** Any non-trivial visual effect (typewriter text, number counters, glitch,
particle systems, waveforms, syntax-highlighted code, blur reveals, etc.) is handled as a
`code_gen` task node — not pre-implemented in the compositor.

The executor:
1. Sends the task description + parameters to a code-gen LLM (Claude Sonnet primary,
   Gemini Pro fallback) with a strict system prompt
2. The LLM writes a self-contained Python script (stdlib + Pillow + MoviePy + NumPy only)
   that writes its output to `$OUTPUT_PATH`
3. The script executes in a subprocess (isolated, 300s timeout)
4. VLM quality gate validates the resulting clip
5. Output enters the compositor as an ordinary video/image layer

**What the compositor handles:** ONLY assembly primitives — layer ordering, position, scale,
opacity curves, smoothstep motion, audio sync. Nothing visual or creative.

**Rationale:**
- Pre-implementing every possible effect is impossible and creates permanent maintenance debt.
- An LLM can write correct MoviePy/Pillow code for any describable effect in seconds.
- Subprocess sandboxing means bad generated code cannot corrupt the pipeline process.
- The VLM gate after each code-gen execution catches effects that render incorrectly.
- This collapses the entire "visual effects" problem into a single, general-purpose executor.

**Fallback:** If code generation fails (no API key, timeout, runtime error), a labeled
placeholder clip is produced so the compositor can still assemble a partial output.

**Registry models added (2026-03-26):**
- `claude-sonnet` (tier 2, code_gen category) — primary
- `gemini-pro-code` (tier 2, code_gen category) — fallback

**Available in budget modes:** `economy`, `production`, `premium` (both are API-only, tier 2).
In `free` mode: falls back to placeholder clip.

**Status:** Settled. Implemented in `scripts/stage4_executor.py`.

---

## D-014 | Cost transparency as a first-class UX requirement

**Decision:** The pipeline surfaces cost information at three points during every run:

1. **Pre-run banner** — before any model calls execute:
   - Min / Expected / Max cost with per-task breakdown
   - Budget cap with within/over status
   - Warnings (e.g. fallback cost impact)
   - In interactive mode: pauses for explicit user confirmation

2. **Periodic heartbeat** — every 2 minutes during long-running model calls:
   - Prints task ID, model, elapsed time, running spend so far
   - Implemented as a daemon thread (`_PeriodicReporter`) so it never blocks execution

3. **Final cost summary** — at end of run:
   - Estimated vs actual with delta and sign
   - Remaining budget, wall-clock time
   - Per-task estimated vs actual breakdown

**Rationale:**
- API cost surprises (especially video generation) destroy user trust.
- Long-running calls (Kling: ~45s, WAN: ~180s) with no feedback feel broken.
- Estimated vs actual comparison teaches users what each type of video actually costs.
- The three-point structure mirrors how professional tools show budget (pre-production
  estimate → production tracking → post-production reconciliation).

**Status:** Settled. Implemented in `scripts/pipeline.py`.

---

## Open Questions (Not Yet Decided)

These are unresolved design questions. A new agent should NOT implement a solution without
first flagging the question and getting a decision from the project owner.

**OQ-1: Streaming output vs. batch render**
Should the pipeline support streaming intermediate outputs to a UI in real-time (as each
task node completes), or always batch-render and deliver the final video at the end?
Streaming is better UX but requires a server/websocket layer. Batch is simpler to implement.
Current lean: batch for v1, streaming for v2.

**OQ-2: How to handle character consistency across multiple generated clips**
If a video needs 3 separate image-to-video clips of the same character, each model call
is independent and may produce slightly different character appearances.
Possible solutions: (a) use the last frame of clip N as the input image for clip N+1,
(b) use IP-Adapter or similar conditioning, (c) use Kling's "character consistency" feature.
Not yet decided.

**OQ-3: Audio-reactive animation**
Beat detection (librosa) is solved. But which elements should pulse/scale to beats?
Should the planner LLM decide, or should the user specify? Not yet decided.

**OQ-4: Primary programming language** — **RESOLVED**: Python for v1 (see D-010).

**OQ-5: How to distribute / package the tool** — **RESOLVED**: CLI tool for v1 (simplest,
most ML-pipeline-friendly). See `scripts/pipeline.py` argparse interface.

---

## What Has Been Built (as of 2026-03-26)

| File | Status | Description |
|---|---|---|
| CLAUDE.md | ✅ Done | Agent operational instructions (6-stage pipeline) |
| ARCHITECTURE.md | ✅ Done | Full system design, schemas, failure catalogue, cost model |
| DECISIONS.md | ✅ Done | This file — 14 architectural decisions |
| FUTURE.md | ✅ Done | Roadmap, competitive landscape, v2/v3 ideas |
| PROGRESS.md | ✅ Done | Append-only session log |
| registry/capability_registry.json | ✅ Done | 25 models catalogued across 12 categories |
| scripts/pipeline.py | ✅ Done | Main CLI orchestrator — 6 stages, pre-flight, cost display, reports |
| scripts/utils.py | ✅ Done | smoothstep, load_registry, append_progress, save_json, ensure_dirs |
| scripts/stage1_asset_analyzer.py | ✅ Done | VLM asset scanner → assets/descriptions.json |
| scripts/stage2_intent_clarifier.py | ✅ Done | Ambiguity detection, asset-intent conflict check |
| scripts/stage3_task_planner.py | ✅ Done | LLM task DAG + spec.json with DAG validation |
| scripts/stage4_cost_estimator.py | ✅ Done | Pre-flight cost estimate → cost_estimate.json |
| scripts/stage4_router.py | ✅ Done | Registry-based model selection (hard constraints + scoring) |
| scripts/stage4_executor.py | ✅ Done | All task types including code_gen universal spawner |
| scripts/stage5_compositor.py | ✅ Done | MoviePy compositor with smoothstep easing + opacity |
| scripts/stage6_qa.py | ✅ Done | Resolution/FPS/duration/VLM frame sampling QA |

**Pending (end-to-end validation):**
- Drop real assets into assets/, set API keys, run full pipeline with a real intent
- Validate that task planner correctly emits code_gen nodes for effect-type requests
- Test resume behaviour (--resume flag) after a mid-run failure

**Proof of concept (separate repo):**
A working proof-of-concept exists at `/Users/osaha/Desktop/sigma-gen-release/`.
This validated Stage 3 (code generation) and Stage 5 (MoviePy compositor) for a specific
use case: 5 reference images + 1 generated output, with smoothstep animation.
The POC used Gemini to write the storyboard prompt and Claude Opus to generate the
MoviePy script. It produced a working video. This entire project is the formalization
of that POC into a general-purpose system.
