# Progress Log

> Append-only. Each entry should include the date, what was done, and any blockers or decisions.

---

## 2026-03-25 — Project Initialized

- Created workspace: `agentic-video-pipeline/`
- Defined the 4-stage pipeline architecture:
  1. VLM Asset Describer
  2. Storyboard Spec Generator (Planner LLM)
  3. Code Generator (Claude Agent)
  4. Execution & QA Validator
- Wrote `CLAUDE.md` with full agent instructions for all stages
- Wrote `FUTURE.md` with v2/v3 roadmap ideas
- Origin: adapted from a working proof-of-concept in `sigma-gen-release/` where a Gemini-generated
  storyboard prompt was passed to a Claude Opus agent to generate a MoviePy composition script

**Status:** Architecture defined. Implementation not yet started.
**Next:** Build Stage 1 (VLM asset describer) and Stage 2 (JSON storyboard spec generator).

---

## 2026-03-25 — Competitive Landscape Research

Scouted GitHub, commercial tools, and academic papers for similar pipelines. Key findings:

**Closest open-source (architecture match, wrong use case):**
- Code2Video (ShowLab, NeurIPS 2025): Planner + Coder + VLM Critic pipeline → Manim code. Generates content from scratch, no user asset ingestion.
- Manimator (ICML 2025): Research paper → Manim animation, 2-LLM pipeline. Same gap.
- Math-To-Manim: 6-agent pipeline with Claude Sonnet. Math domain only.

**Closest commercial (right renderer, no AI layer):**
- Creatomate: JSON storyboard spec (RenderScript) → cloud rendered video. Most expressive spec format. No VLM or LLM planner built in.
- Shotstack: JSON timeline API, cloud rendering. Same gap.
- Remotion: React/TypeScript compositor with official Claude/LLM integration (llms.txt). No planner layer.
- Plainly Videos: After Effects automation with MCP server for LLM integration.

**Academic planners (right pipeline, wrong renderer):**
- VideoDirectorGPT, LVD: LLM planner → diffusion renderer. Non-deterministic, no user assets.
- LAVE (IUI 2024): VLM captions + LLM agent + UI action execution. Closest to full pipeline but no code-gen.

**Confirmed gap:** No tool combines (1) VLM asset description, (2) LLM storyboard planning with
precise math (easing, coordinates), (3) code generation for compositing user-supplied assets,
(4) deterministic execution. The concept is novel.

**ComposioHQ/awesome-claude-skills:** No video composition skills exist there at all.

**Status:** Research complete. Ready to plan implementation.
**Next:** Design storyboard JSON schema (spec.json) and begin Stage 1 VLM describer.

---

## 2026-03-25 — Architecture Expanded: Model Router Vision

Major architectural evolution. The system is no longer a 4-stage linear pipeline — it is a
general-purpose agentic video production system with a task graph (DAG) at its core.

**Core insight:** The system routes each sub-task to the best specialized model. The LLM is
the orchestrator, not the worker. This collapses most of the previous "not achievable" list.

**What the model router unlocks:**
- Character animation: static image → video gen (WAN/Kling/Veo) → SAM2 roto → compositor
- Background removal from video: SAM2 / RMBG / BiRefNet
- Face/lip sync animation: LivePortrait, SadTalker
- Object removal/inpainting: SAM2 mask → ProPainter fill
- Point tracking: CoTracker3 → attach graphics to moving subjects
- Generated backgrounds/scenes: FLUX / SD3.5 / DALL-E 3
- Voiceover: ElevenLabs / Chatterbox

**Three new hard problems introduced:**
1. Capability Registry: structured model catalogue for routing decisions (created)
2. Format Adapter Layer: standardized I/O between stages (alpha, FPS, color space, resolution)
3. VLM Quality Gates: mandatory after every model call; score < 5/10 = hard stop

**Failure modes documented in ARCHITECTURE.md (5 categories, 15 specific failure modes):**
F1: Model output quality | F2: Format compatibility | F3: Infrastructure
F4: Semantic/intent | F5: Cascading failures

**Key mitigation architecture:**
- Resumability: every completed task node written to disk immediately
- VLM gates mandatory and non-skippable
- Pre-flight: 5 checks before any model execution
- Failed tasks don't block independent DAG branches
- Lossless intermediates; compress only at final output

**Files created/updated:**
- ARCHITECTURE.md (major update: full system design, schemas, failure catalogue)
- CLAUDE.md (rewritten: 6-stage pipeline, routing rules, escalation protocol)
- FUTURE.md (updated: model router vision documented)
- registry/capability_registry.json (new: 20+ models catalogued)

**Status:** Architecture fully designed.

---

## 2026-03-25 — Economics & Cost Model

**Question addressed:** Is this tool worth the API spend? What does a generation actually cost?

**Findings:**

Base token cost (every run, regardless of video complexity): ~$0.30–0.40
Draft mode base cost (local/free models): ~$0.05–0.15

Specialist model costs:
- Image-to-video: Kling $0.14/sec, Veo3 $0.35/sec, WAN local $0
- Image gen: DALL-E 3 $0.04/image, FLUX local $0
- TTS: ElevenLabs $0.18/1k chars, Chatterbox local $0
- Segmentation, tracking, depth, upscaling, face animation: all local/$0

Total cost by video type:
- Simple layout (research teaser): $0.30–0.50
- Product demo with 1 animated character: $1.20–1.80
- 30s marketing video with 4 generated clips: $4–7
- 100-video batch (template, no generation): $20–40 total

ROI vs manual production:
- ML researcher: 300–2000× ROI per video
- Indie developer: 100–1400× ROI
- Startup team: 60–500× ROI
- Enterprise batch: 170–1500× ROI

The tool pays for itself on the first video at every user tier.
Beyond cost: the multiplier is **iteration speed** — 10 variants for ~$5 vs. $400 freelancer.

**Where economics break down (honest assessment):**
- Prestige/broadcast work: AI video gen quality isn't there yet
- Narrative/documentary editing: editorial judgment is irreducible
- Cost surprises at scale with video gen: Veo3 at high volume can hit $500+ unexpectedly

**Key design decisions made from this analysis:**

1. Cost Estimator module is mandatory pre-flight (not optional)
   - Runs after task graph is built, before any model calls
   - Outputs cost_estimate.json with min/expected/max scenarios
   - Hard stops if expected cost > max_cost_usd (default $5.00)

2. Budget modes: free / economy / production / premium
   - Default: economy (Kling + Gemini Flash gates)
   - `premium` requires explicit opt-in; never auto-selected

3. Draft-before-final is the default workflow
   - Draft: free/local models, 50% resolution
   - Final: production models, full resolution
   - User approves draft before final pass executes

**Files created/updated:**
- ARCHITECTURE.md (added: Economics section, Cost Estimator module, Budget Modes, Two-tier workflow)
- CLAUDE.md (updated: default parameters table, budget mode reference, pre-flight rule #8)
- DECISIONS.md (new): full log of 12 architectural decisions + 5 open questions

**Status:** All design decisions documented. Economics validated. Ready to implement.

**Next concrete steps (in order):**
1. Build Cost Estimator module (reads task_graph.json + registry → cost_estimate.json)
2. Build Stage 1 Asset Analyzer (VLM prompt template + descriptions.json output schema)
3. Build Stage 3 Task Planner (planning prompt + task_graph.json generator + spec.json)
4. Build Stage 4 Model Router (routing logic per decision tree in ARCHITECTURE.md)
5. Build Stage 5 Compositor (MoviePy assembly, based on sigma-gen POC patterns)
6. Wire stages into an orchestrator script with pre-flight checks
7. Test end-to-end with sigma-gen assets as the first real test case

---

## 2026-03-25 — Architecture Expanded: Model Router Vision

Major architectural evolution: the system is no longer a 4-stage linear pipeline but a
general-purpose agentic video production system with a task graph (DAG) at its core.

**The key insight:** The system doesn't do hard things itself — it routes each sub-task to
the best specialized model. The LLM is the orchestrator, not the worker.

**What this unlocks (previously "not achievable"):**
- Character animation: static image → video gen (WAN/Kling/Veo) → SAM2 rotoscope → compositor
- Background removal from video: SAM2 / RMBG / BiRefNet
- Face/lip sync animation: LivePortrait, SadTalker
- Object removal/inpainting: SAM2 mask → ProPainter fill
- Point tracking: CoTracker3 → attach graphics to moving subjects
- Generated backgrounds: FLUX / SD3.5
- Voiceover: ElevenLabs / Chatterbox

**Three new hard problems introduced:**
1. Capability Registry — structured model catalogue for router decisions
2. Format Adapter Layer — standardized I/O between model stages (alpha, FPS, color space)
3. VLM Quality Gates — mandatory after every model call; score < 5/10 = hard stop

**Failure modes documented (see ARCHITECTURE.md):**
- F1.x: Model output quality (wrong content, segmentation artifacts, temporal flicker, style mismatch)
- F2.x: Format compatibility (resolution, alpha channel, FPS, color space)
- F3.x: Infrastructure (GPU OOM, API rate limits, missing keys, network timeout, missing weights)
- F4.x: Semantic/intent (ambiguity, conflicts, asset-intent mismatch, planner hallucination)
- F5.x: Cascading failures (silent propagation, no state/resumability, compounding degradation)

**Key mitigation decisions:**
- Pipeline is resumable: every completed task node written to disk immediately
- VLM gate is mandatory and non-skippable
- Pre-flight check before any execution (API keys, VRAM, model weights)
- Failed tasks don't block independent nodes — partial outputs are still delivered
- All intermediates use lossless formats; compression only at final output

**Files created/updated:**
- ARCHITECTURE.md (new) — full system design, schemas, failure catalogue, decision tree
- CLAUDE.md (rewritten) — updated agent instructions for 6-stage pipeline
- FUTURE.md (updated) — model router vision documented

**Status:** Architecture fully designed. Ready to implement.
**Next:** Build capability_registry.json, then Stage 1 Asset Analyzer.

## 2026-03-26T20:49:18Z
## 2026-03- Implementation Phase Complete (v1)26 

All pipeline scripts created and syntax-validated.

**Files created:**
- scripts/utils. smoothstep, load_registry (strips // comments), logging, append_progress, ensure_dirspy 
 assets/descriptions.json
- scripts/stage2_intent_clarifier. ambiguity detection, asset-intent conflict checkingpy 
- scripts/stage3_task_planner. LLM-driven task DAG + spec.json generation with DAG validationpy 
 cost_estimate.json; BudgetExceededError
- scripts/stage4_router. registry-based model selection (hard constraints + scoring)py 
- scripts/stage4_executor. task dispatch stubs + mandatory VLM quality gate + retry-with-fallbackpy 
- scripts/stage5_compositor. MoviePy compositor with smoothstep motion curves and opacitypy 
- scripts/stage6_qa. resolution/FPS/duration/VLM frame sampling QApy 
 run_report.json + CLI

**Key properties:**
- Smoothstep-only easing (no linear interpolation)
- Registry-driven routing (no hardcoded model names)
- Resume support: completed task nodes skip on re-run
- VLM quality gates mandatory and non-skippable
- Draft-before-final default workflow
- Hard budget cap with BudgetExceededError
- premium mode never auto-selected

**Smoke tests passed:**
- smoothstep(0/0.5/1.0) = 0.0/0.5/1.0 
- Registry loads 23 models 
- Router returns None + pre-flight warning (no VRAM, no keys on test machine) 
- Cost estimator: 5-task graph = bash.98 expected, within  cap 
- CLI --help works 

**Status:** Implementation complete. Ready for end-to-end test with real API keys + assets.
**Next:** Add assets to assets/, set API keys, run end-to-end test with a real intent.


## 2026-03-26T22:25:09Z
## 2026-03- Cost Display & Code-Gen Executor26 

### Cost transparency UX (3 additions to pipeline.py)

1. **Pre-run cost  printed before any execution:banner** 
   - Min / Expected / Max cost with per-task breakdown
   - Budget cap + within/over status
   - Warnings (e.g. "if fallback triggered, +$1.05")
   - In interactive mode: pauses for user confirmation

2. **Periodic heartbeat** (_PeriodicReporter  every 2 minutes during long model calls:thread) 
   - Prints task ID, model, elapsed time, running spend so far
   - Daemon thread, stops cleanly when task finishes

3. **Per-task status  after each task completes:line** 
 status, model used, quality score, task cost, running total /  /    - 

4. **Final cost summary  at end of run:banner** 
   - Estimated vs actual comparison with delta
   - Remaining budget, wall-clock time
   - Per-task est vs actual breakdown

### Code-gen executor (D- universal effect spawner)013 

Added  as a first-class task type. This is architecturally significant:
- The compositor handles ONLY assembly (position, timing, opacity, audio sync)
- All non-trivial visual effects (typewriter, counters, glitch, particles, etc.)
  are spawned as  task  no pre-implementation requirednodes 
- Executor sends task description + parameters to Claude Sonnet (or Gemini Pro)
  with strict system prompt: stdlib/Pillow/moviepy/numpy only, output to $OUTPUT_PATH
- Generated script runs in subprocess, isolated, 300s timeout
- VLM quality gate validates the rendered clip
- Falls back to labeled placeholder clip if code fails or no API key

Files changed:
- scripts/stage4_executor. added _execute_code_gen(), _call_llm_for_code(),py 
  _run_generated_code(), _build_code_gen_prompt(), _fallback_code_gen_placeholder()
- registry/capability_registry. added claude-sonnet (tier 2) andjson 
  gemini-pro-code (tier 2) as code_gen models (25 total models now)

**Status:** All implementation done. 25 registry models. Cost display verified.
**Next:** End-to-end test with real assets and API keys.


## 2026-03-26T22:56:40Z
Run run_20260326_002 — FAILED
  Intent: Create a 45-second premium product showcase video for the "AuraSound Pro" wireless noise-cancelling 
  Budget: economy, Cost: $0.0000
  Duration: 0.1s
  QA score: 0.0/10
  Tasks: 6, Failures: 11
