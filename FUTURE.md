# Future Ideas & Roadmap

> Ideas that are out of scope for the current version but worth tracking.

---

## Competitive Landscape (Research Summary — 2026-03-25)

Researched existing tools across GitHub, commercial APIs, and academic papers.
Full findings: see PROGRESS.md entry for 2026-03-25.

**Key finding:** No existing tool combines all four stages of this pipeline.
The concept occupies a novel intersection:
- Code2Video / Manimator / Math-To-Manim: have the LLM pipeline but generate content from scratch (no user asset ingestion)
- Creatomate / Shotstack / Remotion / Revideo: have the JSON-spec renderer but no AI planning/VLM layer
- VideoDirectorGPT / LVD: have the LLM planner but use diffusion rendering (non-deterministic, can't composite user images)
- VideoAgent: has VLM + LLM orchestration but no storyboard spec or code-gen step

**Strategic insight:** Creatomate or Shotstack could serve as the rendering backend for v1,
replacing custom MoviePy code with a battle-tested cloud renderer. The concept's unique value is
the VLM + Planner LLM layer on top.

**Potential integration paths:**
- Use Remotion as the renderer (has official Claude/LLM integration guide, llms.txt)
- Use Creatomate as the renderer (most expressive JSON spec — LLM could generate RenderScript)
- Use Plainly Videos MCP as the execution layer (LLM-native trigger via MCP)

---

---

## Version 2 — Visual Storyboard Interface

### Concept
A 2D canvas UI where users drag assets into a timeline and spatial layout, then define animation
parameters visually. The LLM translates the visual layout into the JSON storyboard spec, bypassing
the need for the Planner LLM stage entirely.

### Key Features
- Drag-and-drop asset placement onto a canvas (spatial)
- Timeline scrubber for defining keyframes (temporal)
- Per-object property panels: position, scale, opacity, rotation, easing curve
- LLM-assisted: "make this feel snappy" → auto-adjusts easing and timing parameters
- Preview mode: low-res real-time preview in the browser

### Technical Approach
- Frontend: React + Konva.js (2D canvas) or Three.js (if 3D layout needed)
- The canvas state serializes directly to the storyboard JSON spec used by Stage 3
- No separate Planner LLM needed if user defines layout manually

---

## Version 2.5 — Equation-Driven Animation

### Concept
Users (or the LLM) define object motion as mathematical equations rather than keyframes. For
example: "orbit around center at radius 200px with period 2s" → auto-generates parametric position
functions for `.set_pos()`.

### Key Features
- LLM interprets natural language motion descriptions into parametric equations
- Equations are stored in the storyboard spec under `"motion_fn"` fields
- Supports: circular orbits, sinusoidal oscillation, spring physics, bezier paths
- Deterministic and reproducible: same equation = same animation every time

### Example Storyboard Snippet
```json
{
  "asset": "subject_1.png",
  "motion_fn": {
    "x": "960 + 200 * cos(2 * pi * t / 2.0)",
    "y": "540 + 200 * sin(2 * pi * t / 2.0)"
  },
  "duration": 4.0
}
```

---

## Version 3 — 3D Scene Compositor

### Concept
Extend the pipeline to support 3D spatial layouts rendered via Blender (Python API) or a WebGL
renderer. Users define camera paths, depth layering, and lighting in natural language.

### Key Features
- LLM generates Blender Python scripts instead of MoviePy scripts
- Assets are composited as textured planes in 3D space
- Camera fly-throughs, depth-of-field, parallax effects
- Output: rendered MP4 via Blender's Cycles or EEVEE engine

---

## Version 3+ — Multi-Modal Input

### Concept
Accept audio (voiceover, music), video clips, and 3D models as assets — not just images.
The VLM/ALM layer would describe each asset type appropriately, and the storyboard spec
would handle mixed-media timelines.

### Ideas
- Auto-sync animation keyframes to audio beats (beat detection → keyframe placement)
- Video clip trimming and transitions defined in natural language
- Background music with auto-fade at video end

---

## The Model Router Architecture (Core v1 Vision — 2026-03-25)

This is the architectural leap that makes the tool genuinely powerful. The system does not
*do* the hard things — it *routes* to the right specialized model for each hard thing.

The key insight: every task in video production can be decomposed into atomic operations,
each of which has a best-in-class specialized model. The LLM is the orchestrator, not the worker.

### What this unlocks (previously "not achievable"):
- Character animation from image: img → WAN/Kling (image-to-video) → SAM2 (roto) → compositor
- Background removal from video: video → SAM2/RMBG → alpha-channel clip → compositor
- Face animation from audio: portrait + speech → LivePortrait/SadTalker → compositor
- Object removal: SAM2 mask → ProPainter inpaint → compositor
- Point tracking: CoTracker3 → track coordinates → LLM attaches graphics to moving points
- Generated backgrounds: FLUX/SD → background image → compositor
- Voiceover: script → ElevenLabs → audio track → compositor

### The three hard new problems this introduces:
1. **Capability Registry**: structured description of every available model (I/O formats, VRAM,
   cost, quality tier, strengths/weaknesses). The router reasons over this registry.
2. **Format Adapter Layer**: standardized converters between model outputs and next model's
   expected inputs. Alpha channel handling, FPS normalization, color space, resolution.
3. **VLM Quality Gates**: after every model execution, a VLM scores the output. Score < 5/10
   triggers retry or fallback. Without gates, errors compound silently across stages.

Full design: see ARCHITECTURE.md.

---

## Long-Term Vision

The tool evolves from a "video compositor" into a **general-purpose agentic video production
system** — n8n for video models, with an LLM as the workflow planner.

The user's only job: supply assets + describe intent. The system decides which models to call,
in what order, validates quality at each step, and assembles the result.

Target users:
- ML/CV researchers (paper teaser videos, demo reels)
- Product teams (feature announcement videos, social repurposing at scale)
- Indie developers (app store previews)
- Educators (animated explainers)
- Enterprises (batch-generate 1000 localized videos from templates)

Potential distribution: CLI tool → VS Code extension → web app with asset upload UI.
