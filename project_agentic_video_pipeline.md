---
name: agentic-video-pipeline project
description: Agentic video production system — natural language + assets → rendered video. Full architecture designed, implementation not yet started.
type: project
---

Workspace: `/Users/osaha/Desktop/agentic-video-pipeline/`

## What This Is

A general-purpose agentic video production system. The user supplies assets (images, video,
audio) + a natural language intent description. The system routes each sub-task to the best
available specialized model, validates quality at each stage, and assembles a final video.

Competes with After Effects and Premiere Pro for structured, compositing-heavy video work.
Requires only API keys — no video editing software needed.

**Why:** Proof-of-concept validated in sigma-gen-release (Gemini wrote storyboard prompts →
Claude Opus generated MoviePy scripts → working videos). This is the formalization of that
POC into a general-purpose system.

## Current State (as of 2026-03-25)

**Design phase complete. Implementation not yet started.**

All architecture, decisions, and economics are documented. No code has been written yet
beyond the POC in sigma-gen-release.

## Key Files to Read First

1. `DECISIONS.md` — 12 settled architectural decisions + 5 open questions. Read before
   proposing any changes. Do not re-litigate settled decisions.
2. `ARCHITECTURE.md` — full system design: 6-stage pipeline, task graph schema, capability
   registry schema, cost estimator, failure modes (15 documented), decision tree.
3. `CLAUDE.md` — agent operational instructions for each stage.
4. `PROGRESS.md` — full session log; read the 2026-03-25 entries for current state.
5. `FUTURE.md` — roadmap, competitive landscape, v2/v3 ideas.
6. `registry/capability_registry.json` — all available models catalogued.

## Architecture Summary

6-stage pipeline built around a task graph (DAG):
1. Asset Analyzer (VLM per asset → descriptions.json)
2. Intent Clarifier (ambiguity resolution)
3. Task Planner (LLM builds DAG → task_graph.json + spec.json)
4. Model Router + Executor (per node: select → pre-flight → execute → VLM gate)
5. Compositor (MoviePy assembles all outputs → final_video.mp4)
6. Final QA (resolution, FPS, duration, VLM intent check)

Core philosophy: the system doesn't do hard things — it routes to the right model.

## What the Model Router Unlocks

Previously "impossible" tasks now handled by routing:
- Character animation: image → Kling/WAN/Veo3 (img2vid) → SAM2 (roto) → compositor
- Background removal: RMBG / BiRefNet / SAM2
- Face/lip sync animation: LivePortrait, SadTalker
- Object removal: SAM2 mask → ProPainter inpaint
- Point tracking: CoTracker3 → attach graphics to moving subjects
- Generated backgrounds: FLUX / DALL-E 3
- Voiceover: ElevenLabs / Chatterbox

## Economics (Validated)

Base token cost per run: ~$0.30–0.40 (draft: ~$0.05–0.15)
Typical total costs: $0.30–0.50 (simple), $1–3 (medium), $4–8 (complex 30s video)
ROI: 100–2000× depending on user type. Pays for itself on first video.

Budget modes: free / economy (default) / production / premium
Default max_cost_usd: $5.00 per run (hard cap)
Default workflow: draft pass first (free models), then final pass (production models)
`premium` mode (Veo3) requires explicit opt-in — never auto-selected.

## Next Implementation Steps (in order)

1. Cost Estimator module (task_graph.json + registry → cost_estimate.json)
2. Stage 1: Asset Analyzer (VLM prompt template + descriptions.json schema)
3. Stage 3: Task Planner (planning prompt + task_graph.json + spec.json generator)
4. Stage 4: Model Router (routing logic per ARCHITECTURE.md decision tree)
5. Stage 5: Compositor (MoviePy, based on sigma-gen POC patterns)
6. Orchestrator script wiring all stages with pre-flight checks
7. End-to-end test using sigma-gen assets as first real test case

## Key Constraints for Any Agent

- PROGRESS.md is append-only. Never overwrite.
- assets/ is read-only. Never modify user inputs.
- VLM quality gates are mandatory after every model call. Non-skippable.
- Pre-flight must run 5 checks before any model execution (keys, VRAM, weights, cost, confirm).
- Lossless intermediate formats only. Compress only at final output.
- Smoothstep easing: S(t) = t²(3 - 2t). No linear interpolation for animations.
- Never auto-select `premium` budget mode.
- Write task state to disk immediately after each node completes (resumability).

## How to Apply

When continuing this project: read DECISIONS.md first to understand what's settled,
check PROGRESS.md for the latest state, then refer to ARCHITECTURE.md for implementation
details. The sigma-gen-release POC is the best reference for Stage 5 (compositor) patterns.
