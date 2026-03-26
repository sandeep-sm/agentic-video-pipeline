# System Architecture — Agentic Video Pipeline

> This document is the canonical technical reference for the system design.
> All agents should read this before making architectural decisions.
> Last updated: 2026-03-25

---

## Core Philosophy

The system does not *do* the hard things — it *routes* to the right specialized model for each
hard thing. The LLM is the orchestrator, not the worker. Every task in video production can be
decomposed into atomic operations, each of which has a best-in-class model. The system's job is
to know which model to call, in what order, with what inputs, and whether the output is good enough
to pass to the next stage.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│   assets/ (images, video, audio) + intent prompt (text)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: ASSET ANALYZER                                        │
│  - VLM (Claude Vision / Gemini) runs on each asset              │
│  - Outputs: assets/descriptions.json                            │
│  - Per asset: type, dimensions, subject, background, style,     │
│    dominant colors, notable features, suggested use in video    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: INTENT CLARIFIER (optional, interactive mode only)    │
│  - LLM reviews intent prompt + asset descriptions               │
│  - Identifies ambiguities: missing info, conflicting requests   │
│  - In batch mode: documents assumptions in spec.json warnings   │
│  - In interactive mode: asks user clarifying questions          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: TASK PLANNER                                          │
│  - LLM (Claude Opus / Gemini Pro) builds a task graph (DAG)     │
│  - Each node: task_type, input_refs, model_options, quality_req │
│  - Outputs: storyboard/task_graph.json                          │
│  - Also outputs: storyboard/spec.json (timeline + coordinates)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: MODEL ROUTER + EXECUTOR (per task node in DAG)        │
│                                                                 │
│  For each node:                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 4a. Model Selector                                        │  │
│  │     Consults capability_registry.json                    │  │
│  │     Applies routing rules (quality, cost, GPU, latency)  │  │
│  │     Selects primary + fallback models                    │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ 4b. Prompt Engineer                                      │  │
│  │     Formats input for the selected model's API           │  │
│  │     Adapts intermediate formats (frames ↔ video, etc.)  │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ 4c. Model Executor                                       │  │
│  │     Calls API or runs local HuggingFace model            │  │
│  │     Handles retries, rate limits, OOM errors             │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ 4d. Quality Validator (VLM Gate)                         │  │
│  │     VLM scores output: content, quality, consistency     │  │
│  │     Routes to: accept / retry / fallback / escalate      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: COMPOSITOR                                            │
│  - All model outputs assembled into final video                 │
│  - Backend: MoviePy (local) or Remotion/Creatomate (cloud)      │
│  - Applies: timeline spec, easing functions, alpha compositing  │
│  - Outputs: outputs/final_video.mp4                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 6: FINAL QA                                              │
│  - Validates: resolution, FPS, duration, no black frames        │
│  - VLM watches key frames and scores against original intent    │
│  - Pass: deliver. Fail: report error + stage that caused it.    │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Capability Registry

Every available model is registered with a structured schema. The Model Selector reads this
to make routing decisions. File: `registry/capability_registry.json`

### Schema per model entry:

```json
{
  "model_id": "wan2.1-i2v",
  "display_name": "WAN 2.1 Image-to-Video",
  "category": "image_to_video",
  "access": "local",
  "source": "huggingface",
  "hf_repo": "Wan-AI/Wan2.1-I2V-14B-480P",
  "cost_per_call": 0.0,
  "avg_latency_seconds": 120,
  "vram_required_gb": 16,
  "input_format": {
    "type": "image",
    "accepted_formats": ["jpg", "png", "webp"],
    "max_resolution": "1280x720"
  },
  "output_format": {
    "type": "video",
    "format": "mp4",
    "typical_fps": 24,
    "max_duration_seconds": 5
  },
  "quality_tier": 2,
  "strengths": ["no_watermark", "open_source", "good_motion_fidelity"],
  "weaknesses": ["slow", "needs_gpu", "short_max_duration"],
  "prompt_style": "descriptive_natural_language",
  "negative_prompt_supported": true,
  "watermark": false,
  "notes": "Best for draft quality or when GPU available. No API cost."
}
```

### Quality Tiers (1 = best, 4 = fallback):
- **Tier 1**: Veo 3, Runway Gen-4 — highest quality, paid API
- **Tier 2**: Kling 2.x, SeedDance — good quality, paid API
- **Tier 3**: WAN 2.1, HunyuanVideo — open source, local GPU
- **Tier 4**: LTX-Video, CogVideoX — fast/free, lower quality

---

## Model Router Decision Tree

```
Given a task node, select model as follows:

1. Filter registry by task category (e.g., "image_to_video")

2. Apply hard constraints (eliminate if any fail):
   - vram_required_gb > available_vram → eliminate local models
   - access == "api" AND no API key configured → eliminate
   - max_duration_seconds < required_duration → eliminate

3. Apply soft preferences (score each remaining model):
   + quality_tier == 1         → +30 points
   + quality_tier == 2         → +20 points
   + cost_per_call == 0        → +10 points  (if budget_mode == "free")
   + avg_latency_seconds < 30  → +10 points  (if speed_priority == true)
   + "no_watermark" in strengths → +15 points (if final render, not draft)

4. Select highest scoring model as PRIMARY
   Select second highest as FALLBACK

5. If no models pass hard constraints:
   → Check if task can be SKIPPED (degraded output acceptable)
   → If not skippable: ESCALATE to user with clear explanation
```

---

## The Task Graph Schema

File: `storyboard/task_graph.json`

```json
{
  "video_params": {
    "resolution": [1920, 1080],
    "fps": 30,
    "total_duration_seconds": 9.0,
    "background_color": "#000000"
  },
  "tasks": [
    {
      "task_id": "gen_anim_001",
      "task_type": "image_to_video",
      "description": "Animate character_01.png walking right",
      "inputs": {
        "image": "assets/character_01.png",
        "prompt": "A person walking steadily to the right, neutral background, smooth motion",
        "duration_seconds": 3.0
      },
      "model_options": ["wan2.1-i2v", "kling-2.0-i2v", "ltx-video"],
      "quality_requirement": "final",
      "output_ref": "gen_anim_001_output",
      "fallback_strategy": "degrade_quality",
      "validation_prompt": "Does the video show a person walking to the right with smooth motion?"
    },
    {
      "task_id": "seg_001",
      "task_type": "video_segmentation",
      "description": "Rotoscope character from generated animation, remove background",
      "inputs": {
        "video": "gen_anim_001_output",
        "subject_description": "the walking person",
        "prompt_frame": 0
      },
      "model_options": ["sam2-video", "rmbg-video"],
      "quality_requirement": "final",
      "output_ref": "seg_001_output",
      "fallback_strategy": "skip_and_use_original",
      "validation_prompt": "Is the person cleanly isolated with transparent background and no halo artifacts?"
    },
    {
      "task_id": "composite_001",
      "task_type": "composite",
      "description": "Place rotoscoped character over background_scene.png, moving left to right",
      "inputs": {
        "foreground": "seg_001_output",
        "background": "assets/background_scene.png",
        "motion_spec": {
          "start_pos": [100, 400],
          "end_pos": [1820, 400],
          "easing": "smoothstep",
          "start_time": 0.0,
          "end_time": 3.0
        }
      },
      "model_options": ["moviepy-compositor"],
      "quality_requirement": "final",
      "output_ref": "composite_001_output",
      "fallback_strategy": "error"
    }
  ],
  "warnings": [],
  "assumptions": [
    "Character walking speed set to cross full frame in 3 seconds — adjust if too fast/slow"
  ]
}
```

### Supported task_type values (registered in capability_registry.json)

| task_type | Category | Models | Notes |
|---|---|---|---|
| `image_to_video` | Video gen | Kling, Veo3, WAN, LTX | Animates a static image |
| `text_to_video` | Video gen | Kling, Veo3, WAN, LTX | Generates video from text |
| `image_segmentation` | Segmentation | RMBG, BiRefNet | Background removal from image |
| `video_segmentation` | Segmentation | SAM2 | Object tracking + mask through video |
| `text_to_image` | Image gen | FLUX, DALL-E 3 | Generates background/scene images |
| `text_to_speech` | Audio | ElevenLabs, Chatterbox | Voiceover generation |
| `face_animation` | Animation | LivePortrait, SadTalker | Portrait/lip-sync animation |
| `video_inpainting` | Inpainting | ProPainter | Fill masked regions in video |
| `upscaling` | Enhancement | Real-ESRGAN | Upscale low-res outputs |
| `point_tracking` | Tracking | CoTracker3 | Track points/objects through video |
| `depth_estimation` | Depth | Depth Anything V2 | Monocular depth map generation |
| `speech_to_text` | Audio | Whisper | Transcription with timestamps |
| `code_gen` | **Universal effect spawner** | Claude Sonnet, Gemini Pro | See D-013 |
| `composite` | Assembly | MoviePy | Final compositor — no model call |

### The code_gen task type

`code_gen` is the universal effect spawner. Use it for any visual effect that isn't
handled by a specialist model above: typewriter text, number counters, glitch effects,
particle systems, waveform visualisers, syntax-highlighted code blocks, blur reveals, etc.

**How it works:**
1. Executor sends the task description + all inputs to a code-gen LLM
2. LLM writes a self-contained Python script (stdlib + Pillow + MoviePy + NumPy only)
   that writes its output to `$OUTPUT_PATH`
3. Script runs in a subprocess (isolated, 5-minute timeout, OUTPUT_PATH injected via env)
4. VLM gate validates the rendered clip
5. Output enters compositor as an ordinary video/image layer

**Example node:**
```json
{
  "task_id": "typewriter_001",
  "task_type": "code_gen",
  "description": "Typewriter animation: text appears character by character at 12 chars/sec with blinking cursor at 1Hz",
  "inputs": {
    "content": "How do I reverse a linked list?",
    "chars_per_second": 12,
    "cursor": true,
    "cursor_blink_hz": 1.0,
    "font_size": 48,
    "color": "white",
    "background": "black",
    "duration_seconds": 4.0,
    "output_format": "mp4"
  },
  "model_options": ["claude-sonnet", "gemini-pro-code"],
  "quality_requirement": "final",
  "validation_prompt": "Does the video show text appearing character by character with a blinking cursor?"
}
```

**Compositor principle:** The compositor handles ONLY assembly primitives (layer ordering,
position, scale, opacity curves, smoothstep motion, audio sync). All creative visual
effects come from code_gen nodes. Nothing visual is pre-implemented in the compositor.

---

## Failure Modes & Mitigation

This is the most critical section. Failures compound across stages — a bad output at Stage 4a
will silently corrupt everything downstream unless caught.

### Category 1: Model Output Quality Failures

**F1.1 — Wrong content generated**
- Example: Video gen animates the background instead of the character; SAM2 segments the wrong object
- Detection: VLM validation gate scores content accuracy < 5/10
- Mitigation: Retry with more specific prompt. If second attempt fails, try fallback model.
  If fallback also fails, ESCALATE: report which task failed and why, show the bad output.

**F1.2 — Segmentation artifacts (halos, missed edges)**
- Example: RMBG leaves a green fringe around hair; SAM2 loses track of hand
- Detection: VLM checks for "halo artifacts", "color bleeding", "missing body parts"
- Mitigation: Try BiRefNet as fallback (better at fine detail). If still failing, offer
  user the option to use the original un-rotoscoped clip instead.

**F1.3 — Temporal inconsistency in generated video**
- Example: Character flickers, deforms mid-animation, changes appearance between frames
- Detection: Sample frames at t=0, t=50%, t=100%; VLM checks for consistency
- Mitigation: Regenerate with lower temperature / different seed. If persistent, switch model tier.

**F1.4 — Style mismatch between generated and user-supplied assets**
- Example: User provides photorealistic character; generated background is painterly
- Detection: VLM compares style descriptors from Stage 1 descriptions to generated outputs
- Mitigation: Add style transfer step (Neural Style Transfer or img2img conditioning on user asset)
  OR warn user that style mismatch exists and ask if they want to proceed.

---

### Category 2: Format & Compatibility Failures

**F2.1 — Resolution mismatch**
- Example: Video gen outputs 480p; compositor expects 1080p
- Detection: Check output dimensions before passing to next stage
- Mitigation: Always upscale via Real-ESRGAN before compositing. Never silently stretch.

**F2.2 — Alpha channel loss**
- Example: SAM2 outputs RGBA PNG frames; FFmpeg re-encodes to MP4 and drops alpha
- Detection: Check that output video has alpha channel (RGBA codec) when required
- Mitigation: Use ProRes 4444 or WebM with VP9 for alpha-channel video in intermediate steps.
  Only flatten to RGB at final composite stage.

**F2.3 — FPS mismatch**
- Example: Video gen outputs 24fps clip; compositor timeline is 30fps
- Detection: Check fps metadata of every video output
- Mitigation: Always resample to target FPS using frame interpolation (RIFE) or simple repeat.
  Log the resampling so the user knows.

**F2.4 — Color space mismatch**
- Example: Model outputs linear light; compositor expects sRGB
- Detection: Check ICC profile or color space metadata where available
- Mitigation: Apply gamma correction in format adapter layer. Default: assume sRGB unless
  metadata says otherwise.

---

### Category 3: Resource & Infrastructure Failures

**F3.1 — GPU Out of Memory (OOM)**
- Example: WAN 2.1 14B model fails to load on 8GB VRAM
- Detection: Catch CUDA OOM exception; check vram_required_gb in capability registry before loading
- Mitigation: Pre-check available VRAM before model selection. If insufficient, skip local models
  and route to API tier. Always check registry constraints before attempting local inference.

**F3.2 — API rate limit / quota exceeded**
- Example: Kling API returns 429 Too Many Requests
- Detection: HTTP 429 response
- Mitigation: Exponential backoff (2s, 4s, 8s, max 3 retries). If exhausted, try next model in
  fallback list. Log the rate limit event — repeated occurrences suggest need for higher API tier.

**F3.3 — API key missing or invalid**
- Example: Veo 3 selected but GOOGLE_API_KEY not set
- Detection: Pre-flight check at pipeline start: scan task graph for required models,
  verify all needed API keys exist before starting any execution
- Mitigation: Fail fast at start with clear error: "Task gen_anim_001 requires Veo 3 but
  GOOGLE_API_KEY is not configured. Set it or switch to a local model."

**F3.4 — Network timeout during API call**
- Detection: Timeout exception after configurable threshold (default: 300s)
- Mitigation: Retry once. If second attempt times out, switch to fallback model.
  Log timeout — long timeouts on paid APIs should be flagged to user.

**F3.5 — Model weights not downloaded (local models)**
- Detection: Check that HuggingFace model cache contains required weights before invocation
- Mitigation: Pre-download step at pipeline start for all local models in the task graph.
  Estimate download size and ask user to confirm before downloading >10GB models.

---

### Category 4: Semantic & Intent Failures

**F4.1 — Ambiguous user intent**
- Example: "make the character move" — which direction? how fast? what animation style?
- Detection: Task Planner LLM flags underspecified parameters with a confidence score < 0.7
- Mitigation:
  - Interactive mode: ask user targeted clarifying questions (max 3)
  - Batch mode: make the most reasonable assumption, document it in task_graph.json warnings[],
    and include in the final output report so user can adjust

**F4.2 — Conflicting instructions**
- Example: "fade out all images at 5s" but also "keep subject_1 visible throughout"
- Detection: LLM planner checks for timeline conflicts during task graph construction
- Mitigation: Halt and report the conflict with both instructions highlighted. Do not guess.

**F4.3 — Asset-intent mismatch**
- Example: User says "animate the dog running" but the asset is a landscape photo
- Detection: VLM description from Stage 1 doesn't match the intent's subject references
- Mitigation: Report the mismatch before executing. "Asset character_01.png was described as
  a landscape photo, but the intent references a dog. Please check your asset paths."

**F4.4 — Planner LLM hallucinates model capabilities**
- Example: Planner selects "sam2-video" for a task that requires audio transcription
- Detection: Model Selector validates that selected model's category matches task_type
- Mitigation: Hard validation against capability registry. If category mismatch: error, do not
  execute. Planner must be re-prompted with corrected constraints.

---

### Category 5: Cascading Failures

**F5.1 — Silent error propagation**
- Example: Video gen produces wrong animation → SAM2 produces bad mask → compositor silently
  produces broken output → user only sees the problem at the very end
- Mitigation: The VLM quality gate after EVERY stage is non-negotiable. No stage should pass
  output to the next without a quality check. Gate thresholds:
  - Score ≥ 7/10 on all dimensions → auto-accept, continue
  - Score 5-6/10 on any dimension → flag in output report, continue with warning
  - Score < 5/10 on any dimension → STOP, do not proceed, report which stage failed

**F5.2 — Partial completion with no state**
- Example: Pipeline crashes at Stage 4c after 3 successful model calls; all progress lost
- Mitigation: Write each task output to disk immediately upon completion. Task graph tracks
  which nodes are "complete" (with output path) vs "pending". Re-runs skip completed nodes.
  This makes the pipeline resumable.

**F5.3 — Compounding quality degradation**
- Example: Each stage introduces 5% quality loss → 6 stages → ~26% total degradation
- Mitigation: Use lossless intermediate formats (PNG sequences, ProRes) between stages.
  Only compress to final codec (H.264/H.265) at the very last step.

---

## Decision-Making Hierarchy

When the system faces a decision without explicit user guidance, apply this priority order:

```
1. CORRECTNESS over QUALITY
   → A lower-quality output that matches intent beats a beautiful output that doesn't

2. EXPLICIT over INFERRED
   → If the user specified something directly, never override it

3. SAFE FALLBACK over SILENT FAILURE
   → Always produce some output rather than crashing, but always document degradations

4. ESCALATE AMBIGUITY, DON'T GUESS SILENTLY
   → If confidence < 0.7 on any intent interpretation, document the assumption
   → Never silently make creative decisions (e.g., choosing animation style)

5. FAST FEEDBACK over PERFECT FIRST DRAFT
   → In interactive mode, produce a low-res draft first, confirm intent, then render final
   → In batch mode, skip this rule and go straight to full quality
```

---

## Mode: Interactive vs. Batch

### Interactive Mode
1. Stage 1 (asset analysis) runs immediately on upload
2. Stage 2 (intent clarifier) asks up to 3 clarifying questions
3. Task Planner produces a human-readable summary: "Here's what I'm going to do..."
4. User approves or edits before execution begins
5. After Stage 4, a low-res draft is shown before final render

### Batch Mode
1. All stages run without human input
2. Ambiguities are resolved via documented assumptions
3. Quality gate failures are logged but non-blocking (degraded output is still delivered)
4. Output includes a full report: assumptions made, quality scores, warnings

---

## Format Adapter Layer

Every inter-stage data exchange uses standardized intermediate formats:

| Data Type | Intermediate Format | Rationale |
|---|---|---|
| Image (with alpha) | PNG, RGBA | Lossless, universal alpha support |
| Image (no alpha) | PNG or JPEG quality 95 | Lossless or near-lossless |
| Video (with alpha) | WebM (VP9) or ProRes 4444 | Alpha-channel preservation |
| Video (no alpha) | MP4 (H.264, CRF 18) | Near-lossless for intermediates |
| Audio | WAV 48kHz 16-bit | Lossless, universal |
| Video frames (sequence) | PNG sequence in /tmp/frames_<task_id>/ | Lossless, model-compatible |
| Masks | PNG, single-channel uint8 | Standard segmentation format |

Final output only: MP4 H.264/H.265 at user-specified quality (default CRF 23).

---

## Economics & Cost Model

### Token Cost (Base, Every Run)

These costs are incurred on every run regardless of video complexity.

| Stage | Model | Typical Tokens | Cost (USD) |
|---|---|---|---|
| Asset analysis (6 assets) | Claude Vision | 5k in / 1.5k out | ~$0.04 |
| Task planner | Claude Opus | 3k in / 2k out | ~$0.20 |
| Code generation | Claude Sonnet | 2k in / 3k out | ~$0.05 |
| VLM quality gates (8 checks) | Claude Vision | 6k in / 2k out | ~$0.05 |
| **Base subtotal** | | | **~$0.30–0.40** |

For draft mode: substitute Gemini Flash for VLM gates (~$0.01 total) and Claude Sonnet for
planning (~$0.03). Draft base cost: **~$0.05–0.15**.

### Specialist Model Costs

| Model | Category | Cost | Notes |
|---|---|---|---|
| Veo 3 | img2vid / t2v | $0.35/sec | Best quality, use for finals only |
| Kling 2.0 | img2vid / t2v | $0.14/sec | Good quality, main production model |
| WAN 2.1 (local) | img2vid / t2v | $0 + GPU time | Needs 16GB VRAM, ~120s/clip |
| LTX-Video (local) | img2vid / t2v | $0 + GPU time | Needs 8GB VRAM, ~30s/clip, draft quality |
| DALL-E 3 | t2i | $0.04/image | Background generation |
| FLUX.1 (local) | t2i | $0 + GPU time | Needs 12GB VRAM |
| ElevenLabs | TTS | $0.18/1k chars | Best quality voice |
| Chatterbox (local) | TTS | $0 + GPU time | Free fallback |
| SAM2, RMBG, BiRefNet | segmentation | $0 | All local |
| CoTracker3 | tracking | $0 | Local |
| Depth Anything V2 | depth | $0 | Local |
| Real-ESRGAN | upscaling | $0 | Local |
| Whisper (local) | STT | $0 | Local |
| ProPainter | inpainting | $0 | Local |
| LivePortrait, SadTalker | face anim | $0 | Local |

GPU rental reference: Lambda Labs A10G ~$0.75/hr, A100 ~$1.29/hr.
A typical run with 3 local model calls takes ~30–45 min = $0.38–$0.97 GPU cost.

### Total Cost by Video Complexity

| Video Type | What It Involves | API Tier (Kling) | Local/Free Tier |
|---|---|---|---|
| Simple layout (research teaser) | Compositor only, no generation | $0.30–0.50 | $0.05–0.15 |
| Product demo (1 animated character) | 1× img2vid 5s + roto + composite | $1.20–1.80 | $0.10–0.25 + GPU |
| 30s marketing video (4 gen clips) | 4× img2vid 5s + TTS + bg gen | $4–7 | $0.30–0.60 + GPU |
| 100-video batch (template, no gen) | Pure compositor, planning amortized | $20–40 total | $5–15 total |

### ROI vs Manual Production

| User Type | Hours Saved | Their Time Value | API Cost | ROI Multiple |
|---|---|---|---|---|
| ML researcher | 4–8 hrs | $30–80/hr | $0.30–0.50 | 300–2000× |
| Indie developer | 6–10 hrs or $400–800 freelancer | $100–200/hr | $1–7 | 100–1400× |
| Startup team (weekly cadence) | 8+ hrs or $300–800/video | $300–800/video | $1–8 | 60–500× |
| Enterprise batch (1000 videos) | Weeks, $50–150/video = $50k–150k | $50–150/video | $0.10–0.30/video | 170–1500× |

The tool pays for itself on the first video for every user type.
The real multiplier beyond cost is **iteration speed**: re-run 10 variants for ~$5 vs.
briefing a freelancer once for $400 and waiting days.

### Where the Economics Break Down (Honest Assessment)

1. **High-end brand / broadcast work**: Current AI video gen quality doesn't meet prestige
   production bars. Wrong tool regardless of cost.
2. **Narrative / documentary editing**: Editorial judgment (which take, emotional arc, pacing)
   is irreplaceable. This is a Premiere use case, not a compositor use case.
3. **Single video by experienced AE user**: ROI shrinks to 80–120× (still positive, but the
   user is not the primary target).
4. **Cost surprise at scale with video generation**: 50 videos × 6 Veo3 clips × 5s = $525.
   This is why the cost estimator and budget caps are non-negotiable.

---

## Cost Estimator Module

**This is a mandatory pre-flight step. No model calls execute without a cost estimate.**

### How It Works

After the Task Planner builds `task_graph.json`, and before Stage 4 executes any model call,
the Cost Estimator:

1. Reads every task node in the graph
2. For each node, looks up the PRIMARY and FALLBACK model costs in `capability_registry.json`
3. Computes estimated cost for three scenarios: min (all free/local), expected (selected models),
   max (if all fall back to highest tier)
4. Compares expected cost against `budget_mode` limits
5. Outputs `storyboard/cost_estimate.json`
6. In interactive mode: presents estimate to user and waits for confirmation
7. In batch mode: proceeds if within budget, halts with error if over cap

### cost_estimate.json Schema

```json
{
  "run_id": "run_20260325_001",
  "budget_mode": "production",
  "budget_cap_usd": 10.00,
  "estimates": {
    "minimum_usd": 0.08,
    "expected_usd": 3.40,
    "maximum_usd": 8.20
  },
  "breakdown": [
    {
      "task_id": "gen_anim_001",
      "task_type": "image_to_video",
      "primary_model": "kling2-i2v",
      "primary_cost_usd": 0.70,
      "fallback_model": "wan2.1-i2v",
      "fallback_cost_usd": 0.00,
      "note": "5 second clip at $0.14/sec"
    },
    {
      "task_id": "base_tokens",
      "task_type": "llm_orchestration",
      "primary_model": "claude-opus + claude-vision",
      "primary_cost_usd": 0.34,
      "note": "Planning + asset analysis + quality gates"
    }
  ],
  "within_budget": true,
  "warnings": [
    "If gen_anim_001 fails and falls back to veo3-i2v, cost increases by $1.05"
  ]
}
```

### Budget Modes

Set via `budget_mode` parameter. Controls which model tiers the router is allowed to select.

| Mode | Allowed Tiers | API Video Gen | VLM Gates | Typical Cost |
|---|---|---|---|---|
| `free` | 3, 4 only | WAN / LTX (local only) | Gemini Flash | $0.05–0.20 + GPU |
| `economy` | 2, 3, 4 | Kling only | Gemini Flash | $0.50–3 |
| `production` | 1, 2, 3, 4 | Kling or Veo3 | Claude Vision | $1–8 |
| `premium` | 1 only | Veo3 always | Claude Vision | $3–15 |

Default: `economy` for first runs, `production` after user explicitly upgrades.
**Never default to `premium` — always require explicit opt-in.**

### Two-Tier Draft → Final Workflow (Recommended Default)

```
DRAFT PASS  (budget_mode: "free", ~$0.05–0.20)
  ├── LTX-Video for video generation (fast, lower quality)
  ├── Gemini Flash for VLM gates
  ├── Chatterbox for TTS
  └── Output: low-res draft at 50% resolution

        ↓ User reviews draft, approves or adjusts prompt ↓

FINAL PASS  (budget_mode: "production", ~$1–8)
  ├── Kling / Veo3 for video generation
  ├── ElevenLabs for TTS
  ├── Claude Vision for quality gates
  └── Output: full-res final video
```

This mirrors how professional tools work (low-res proxy → online conform) and prevents
users from spending on final-quality generation before they're happy with the composition.

### Hard Budget Cap

Every run must accept a `max_cost_usd` parameter.
Default: `5.00` if not specified.
If cost_estimate.expected_usd > max_cost_usd:
- In interactive mode: show breakdown, ask user to raise cap or switch budget_mode
- In batch mode: HALT with error listing which tasks exceed budget, suggest alternatives

If actual cost (post-run) exceeds estimate by >20%:
- Log the discrepancy in run_report.json
- Investigate: was a fallback model invoked? Were more retries needed?

### Cost Display (UX — implemented in pipeline.py)

Three mandatory cost touchpoints during every run:

**1. Pre-run banner** (before any model calls)
- Prints min / expected / max cost with per-task breakdown
- Shows budget cap and within/over status
- Shows warnings (e.g. "if fallback triggered, cost +$1.05")
- In interactive mode: pauses for explicit user confirmation before proceeding

**2. Periodic heartbeat** (every 2 minutes during long model calls)
- Background daemon thread (`_PeriodicReporter`) — never blocks execution
- Prints: task ID, model, elapsed time, running spend so far
- Addresses the UX problem of 60–180s of silence during video generation

**3. Final cost summary** (at end of run)
- Estimated vs actual with delta and sign
- Remaining budget, wall-clock time
- Per-task estimated vs actual breakdown

---

## File & Folder Conventions

```
agentic-video-pipeline/
├── CLAUDE.md                   ← Agent instructions (all stages)
├── ARCHITECTURE.md             ← This file
├── PROGRESS.md                 ← Append-only daily log
├── FUTURE.md                   ← Roadmap and future ideas
│
├── registry/
│   └── capability_registry.json   ← All available models, their capabilities
│
├── assets/                     ← User-provided inputs (never delete)
│   └── descriptions.json       ← VLM output from Stage 1
│
├── storyboard/
│   ├── task_graph.json         ← DAG of operations (Stage 3 output)
│   └── spec.json               ← Timeline + coordinate spec (Stage 3 output)
│
├── intermediates/              ← Per-task model outputs (resumable)
│   └── <task_id>/
│       ├── output.*            ← Primary output file
│       └── quality_report.json ← VLM validation scores
│
├── scripts/
│   ├── pipeline.py             ← Main CLI orchestrator (entry point)
│   ├── utils.py                ← Shared utilities (smoothstep, registry, logging)
│   ├── stage1_asset_analyzer.py
│   ├── stage2_intent_clarifier.py
│   ├── stage3_task_planner.py
│   ├── stage4_cost_estimator.py
│   ├── stage4_router.py
│   ├── stage4_executor.py      ← All task handlers including code_gen spawner
│   ├── stage5_compositor.py
│   └── stage6_qa.py
│
└── outputs/
    ├── final_video.mp4
    └── run_report.json         ← Full run: assumptions, scores, warnings
```
