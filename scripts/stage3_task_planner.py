"""
Stage 3 — Task Planner
Calls an LLM to build the task DAG (task_graph.json) and timeline spec (spec.json).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from scripts.utils import get_project_root, load_registry, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM call stub (same pattern as the VLM stub in stage1)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, model_id: str = "claude-opus") -> dict:
    """
    Call an LLM to generate structured JSON for task planning.
    Falls back to a mock response when no API key is available.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    # ── Claude (Opus / Sonnet) ────────────────────────────────────────────────
    if model_id in ("claude-opus", "claude-sonnet") and anthropic_key:
        api_model = (
            "claude-opus-4-5"
            if model_id == "claude-opus"
            else "claude-3-5-sonnet-20241022"
        )
        try:
            import anthropic  # noqa: PLC0415

            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model=api_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return {"result": message.content[0].text, "_mock": False}
        except Exception as exc:
            logger.warning("Claude LLM call failed: %s — falling back to mock", exc)

    # ── Gemini Pro ────────────────────────────────────────────────────────────
    if model_id == "gemini-pro" and google_key:
        try:
            from google import genai  # noqa: PLC0415

            client = genai.Client(api_key=google_key)
            response = client.models.generate_content(model="gemini-3.0-flash", contents=prompt)
            return {"result": response.text, "_mock": False}
        except Exception as exc:
            logger.warning("Gemini LLM call failed: %s — falling back to mock", exc)

    # ── Mock ──────────────────────────────────────────────────────────────────
    return {"result": None, "_mock": True}


# ---------------------------------------------------------------------------
# DAG validation helpers
# ---------------------------------------------------------------------------

def _topological_sort(tasks: list[dict]) -> list[str]:
    """
    Return task_ids in topological order or raise ValueError on cycles.
    Dependencies are inferred from output_ref references in inputs.
    """
    # Build output_ref → task_id index
    output_to_task: dict[str, str] = {}
    for t in tasks:
        ref = t.get("output_ref", "")
        if ref:
            output_to_task[ref] = t["task_id"]

    # Build adjacency list (task_id → set of task_ids it depends on)
    deps: dict[str, set] = {t["task_id"]: set() for t in tasks}
    for t in tasks:
        for v in t.get("inputs", {}).values():
            if isinstance(v, str) and v in output_to_task:
                deps[t["task_id"]].add(output_to_task[v])

    # Kahn's algorithm
    in_degree: dict[str, int] = {tid: 0 for tid in deps}
    for tid, dep_set in deps.items():
        for dep in dep_set:
            in_degree[tid] = in_degree.get(tid, 0) + 1

    # Recalculate using reverse edges
    in_degree = {tid: 0 for tid in deps}
    adj: dict[str, list] = {tid: [] for tid in deps}
    for tid, dep_set in deps.items():
        for dep in dep_set:
            adj[dep].append(tid)

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(tasks):
        raise ValueError("Circular dependency detected in task graph.")
    return order


def _validate_task_graph(task_graph: dict, descriptions: dict, registry: dict) -> list[str]:
    """
    Validate the task graph. Returns a list of warning strings.
    Raises ValueError on hard errors.
    """
    warnings: list[str] = []
    tasks = task_graph.get("tasks", [])

    # Check for circular deps
    _topological_sort(tasks)  # raises on cycle

    # Build known refs: asset names + output_refs of all tasks
    asset_names = set(descriptions.get("assets", {}).keys())
    output_refs: set[str] = set()
    for t in tasks:
        ref = t.get("output_ref", "")
        if ref:
            output_refs.add(ref)

    known_refs = asset_names | output_refs

    # Build registry model_id index
    registry_ids = {m["model_id"] for m in registry.get("models", [])}

    for t in tasks:
        task_id = t.get("task_id", "?")

        # Validate model_options
        for opt in t.get("model_options", []):
            if opt not in registry_ids:
                warnings.append(
                    f"Task '{task_id}': model_option '{opt}' not in capability registry"
                )

        # Validate input refs
        for key, val in t.get("inputs", {}).items():
            if isinstance(val, str) and "/" in val:
                # Looks like a file path — check asset name
                fname = val.split("/")[-1]
                if fname and fname not in asset_names and val not in known_refs:
                    warnings.append(
                        f"Task '{task_id}': input '{key}' references '{val}' "
                        "which is not a known asset or prior output"
                    )

    return warnings


# ---------------------------------------------------------------------------
# Mock plan generator (used when no LLM key)
# ---------------------------------------------------------------------------

def _generate_mock_plan(
    intent: str,
    descriptions: dict,
    clarification: dict,
    budget_mode: str,
    fps: int,
    resolution: list,
    registry: dict,
) -> tuple[dict, dict]:
    """Return a minimal but schema-valid mock task graph and spec."""
    duration = 9.0
    assets = descriptions.get("assets", {})
    asset_names = list(assets.keys())

    tasks = []
    timeline = []

    if asset_names:
        # One image-to-video task per image asset (up to 3)
        image_assets = [
            n for n, a in assets.items() if a.get("file_type") == "image"
        ][:3]

        for i, asset_name in enumerate(image_assets):
            task_id = f"gen_anim_{i + 1:03d}"
            clip_duration = min(5.0, duration / max(1, len(image_assets)))
            start_t = i * clip_duration

            # Pick model options by budget_mode
            if budget_mode == "free":
                model_options = ["ltx-video-t2v", "wan2.1-i2v"]
            elif budget_mode == "premium":
                model_options = ["veo3-i2v", "kling2-i2v"]
            elif budget_mode == "production":
                model_options = ["kling2-i2v", "veo3-i2v", "wan2.1-i2v"]
            else:  # economy
                model_options = ["kling2-i2v", "wan2.1-i2v", "ltx-video-t2v"]

            tasks.append({
                "task_id": task_id,
                "task_type": "image_to_video",
                "description": f"Animate {asset_name} based on intent",
                "inputs": {
                    "image": f"assets/{asset_name}",
                    "prompt": intent[:200],
                    "duration_seconds": clip_duration,
                },
                "model_options": model_options,
                "quality_requirement": "draft" if budget_mode == "free" else "final",
                "output_ref": f"{task_id}_output",
                "fallback_strategy": "degrade_quality",
                "validation_prompt": (
                    "Does this video clip show natural motion consistent with the source image? "
                    "Answer with a score from 1-10 and brief feedback."
                ),
                "status": "pending",
            })

            timeline.append({
                "layer_id": f"layer_{i + 1:03d}",
                "task_ref": task_id,
                "asset_type": "video",
                "start_time": start_t,
                "end_time": start_t + clip_duration,
                "position": {"x": resolution[0] // 2, "y": resolution[1] // 2},
                "scale": 1.0,
                "opacity_curve": [
                    {"t": 0.0, "v": 0.0},
                    {"t": 0.5, "v": 1.0},
                    {"t": 1.0, "v": 1.0},
                ],
                "motion": None,
                "text": None,
            })
    else:
        # No assets: single text-to-video task
        model_options = (
            ["ltx-video-t2v", "wan2.1-t2v"]
            if budget_mode == "free"
            else ["kling2-t2v", "wan2.1-t2v"]
        )
        tasks.append({
            "task_id": "gen_video_001",
            "task_type": "text_to_video",
            "description": "Generate video from intent",
            "inputs": {"prompt": intent[:200], "duration_seconds": duration},
            "model_options": model_options,
            "quality_requirement": "final",
            "output_ref": "gen_video_001_output",
            "fallback_strategy": "degrade_quality",
            "validation_prompt": (
                "Does this video match the described intent? "
                "Score 1-10 with brief feedback."
            ),
            "status": "pending",
        })
        timeline.append({
            "layer_id": "layer_001",
            "task_ref": "gen_video_001",
            "asset_type": "video",
            "start_time": 0.0,
            "end_time": duration,
            "position": {"x": resolution[0] // 2, "y": resolution[1] // 2},
            "scale": 1.0,
            "opacity_curve": [{"t": 0.0, "v": 1.0}],
            "motion": None,
            "text": None,
        })

    task_graph: dict = {
        "video_params": {
            "resolution": resolution,
            "fps": fps,
            "total_duration_seconds": duration,
            "background_color": "#000000",
        },
        "tasks": tasks,
        "warnings": ["_mock: true — generated without LLM. Review before production use."],
        "assumptions": clarification.get("assumptions", []),
    }

    spec: dict = {
        "version": "1.0",
        "video_params": {
            "resolution": resolution,
            "fps": fps,
            "total_duration_seconds": duration,
        },
        "timeline": timeline,
    }

    return task_graph, spec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_tasks(
    intent: str,
    descriptions: dict,
    clarification: dict,
    budget_mode: str = "economy",
    fps: int = 30,
    resolution: list | None = None,
) -> tuple[dict, dict]:
    """
    Build a task DAG (task_graph.json) and timeline spec (spec.json) from intent +
    asset descriptions + clarification result.

    Returns (task_graph, spec).
    """
    if resolution is None:
        resolution = [1920, 1080]

    registry = load_registry()

    # Collect available categories
    categories = sorted({m["category"] for m in registry.get("models", [])})

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    assets_summary = json.dumps(descriptions.get("assets", {}), indent=2)
    assumptions_text = "\n".join(f"- {a}" for a in clarification.get("assumptions", []))
    resolved_intent = clarification.get("resolved_intent", intent)

    task_graph_schema = json.dumps(
        {
            "video_params": {
                "resolution": resolution,
                "fps": fps,
                "total_duration_seconds": 9.0,
                "background_color": "#000000",
            },
            "tasks": [
                {
                    "task_id": "string",
                    "task_type": "string (one of: " + ", ".join(categories) + ")",
                    "description": "string",
                    "inputs": {},
                    "model_options": ["model_id_1", "model_id_2"],
                    "quality_requirement": "draft|final",
                    "output_ref": "string",
                    "fallback_strategy": "degrade_quality|skip|error",
                    "validation_prompt": "string (yes/no or scored question for VLM)",
                    "status": "pending",
                }
            ],
            "warnings": [],
            "assumptions": [],
        },
        indent=2,
    )

    spec_schema = json.dumps(
        {
            "version": "1.0",
            "video_params": {
                "resolution": resolution,
                "fps": fps,
                "total_duration_seconds": 9.0,
            },
            "timeline": [
                {
                    "layer_id": "string",
                    "task_ref": "task_id or asset filename",
                    "asset_type": "image|video|text|audio",
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "position": {"x": 960, "y": 540},
                    "scale": 1.0,
                    "opacity_curve": [{"t": 0.0, "v": 0.0}, {"t": 0.5, "v": 1.0}],
                    "motion": {
                        "start_pos": [100, 540],
                        "end_pos": [1820, 540],
                        "easing": "smoothstep",
                        "start_time": 0.0,
                        "end_time": 3.0,
                    },
                    "text": None,
                }
            ],
        },
        indent=2,
    )

    model_ids = [m["model_id"] for m in registry.get("models", [])]

    prompt = f"""You are a video production task planner for an agentic video pipeline.

USER INTENT:
{resolved_intent}

ASSUMPTIONS ALREADY MADE:
{assumptions_text or '(none)'}

ASSET DESCRIPTIONS:
{assets_summary}

AVAILABLE MODEL IDs (from capability registry):
{json.dumps(model_ids, indent=2)}

AVAILABLE TASK CATEGORIES:
{json.dumps(categories, indent=2)}

BUDGET MODE: {budget_mode}
RESOLUTION: {resolution[0]}x{resolution[1]}
FPS: {fps}

Your job is to return TWO JSON objects separated by the delimiter "---SPEC---":
1. task_graph.json following this schema:
{task_graph_schema}

---SPEC---

2. spec.json following this schema:
{spec_schema}

RULES:
- Use only model IDs from the provided list.
- Use only task_type values from the available categories list.
- Ensure tasks form a valid DAG (no circular dependencies).
- Reference prior task outputs in inputs using the output_ref value.
- Use smoothstep easing in all motion definitions (never linear).
- Budget mode '{budget_mode}': prefer tier-3/4 local models for 'free', tier-2 for 'economy', any for 'production'.
- Return ONLY valid JSON. No markdown code fences, no explanation text.
"""

    # ── LLM call ──────────────────────────────────────────────────────────────
    llm_model = "claude-opus" if budget_mode in ("production", "premium") else "claude-sonnet"
    llm_response = _call_llm(prompt, model_id=llm_model)

    task_graph: dict | None = None
    spec: dict | None = None

    if not llm_response.get("_mock") and llm_response.get("result"):
        raw_text: str = llm_response["result"]
        try:
            # Try splitting on ---SPEC---
            if "---SPEC---" in raw_text:
                parts = raw_text.split("---SPEC---", 1)
                task_graph = json.loads(_clean_json_text(parts[0]))
                spec = json.loads(_clean_json_text(parts[1]))
            else:
                # Try parsing a combined object with both keys
                combined = json.loads(_clean_json_text(raw_text))
                task_graph = combined.get("task_graph", combined)
                spec = combined.get("spec")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("LLM response parse failed: %s — using mock plan", exc)

    if task_graph is None or spec is None:
        logger.warning("Using mock task plan (no LLM key or parse failure).")
        task_graph, spec = _generate_mock_plan(
            intent, descriptions, clarification, budget_mode, fps, resolution, registry
        )
        task_graph["_mock"] = True
        spec["_mock"] = True

    # ── Validate ──────────────────────────────────────────────────────────────
    try:
        validation_warnings = _validate_task_graph(task_graph, descriptions, registry)
        task_graph.setdefault("warnings", []).extend(validation_warnings)
        for w in validation_warnings:
            logger.warning("Task graph validation: %s", w)
    except ValueError as exc:
        logger.error("Task graph validation error: %s", exc)
        raise

    # ── Persist ───────────────────────────────────────────────────────────────
    root = get_project_root()
    save_json(task_graph, root / "storyboard" / "task_graph.json")
    save_json(spec, root / "storyboard" / "spec.json")
    logger.info("Saved task_graph.json and spec.json to storyboard/")

    return task_graph, spec


def _clean_json_text(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace from LLM output."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
