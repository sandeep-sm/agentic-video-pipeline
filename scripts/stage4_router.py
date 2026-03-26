"""
Stage 4 — Model Router
Selects the best PRIMARY and FALLBACK model for each task node from the capability registry.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Budget mode → allowed quality tiers
_BUDGET_TIERS: dict[str, set[int]] = {
    "free": {3, 4},
    "economy": {2, 3, 4},
    "production": {1, 2, 3, 4},
    "premium": {1},
}


def get_available_vram() -> float:
    """
    Return the available GPU VRAM in GB.
    Calls nvidia-smi if available; returns 0.0 if no GPU found.
    """
    if not shutil.which("nvidia-smi"):
        logger.debug("nvidia-smi not found — assuming no GPU.")
        return 0.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if lines:
            # Sum free VRAM across all GPUs, return in GB
            total_mb = sum(int(v) for v in lines if v.isdigit())
            return round(total_mb / 1024.0, 2)
    except Exception as exc:
        logger.debug("nvidia-smi failed: %s", exc)
    return 0.0


def check_api_keys(task_graph: dict, registry: dict) -> list[str]:
    """
    Return a list of missing environment variable names required by the models
    referenced in the task graph.
    """
    import os

    required_keys: set[str] = set()
    model_index = {m["model_id"]: m for m in registry.get("models", [])}

    for task in task_graph.get("tasks", []):
        for model_id in task.get("model_options", []):
            m = model_index.get(model_id)
            if m and m.get("access") == "api":
                key_env = m.get("api_key_env")
                if key_env:
                    required_keys.add(key_env)

    missing = [k for k in required_keys if not os.environ.get(k)]
    return missing


def _score_model(
    model: dict,
    budget_mode: str,
    speed_priority: bool = False,
    is_final_render: bool = True,
) -> int:
    score = 0
    tier = model.get("quality_tier", 4)

    if tier == 1:
        score += 30
    elif tier == 2:
        score += 20
    elif tier == 3:
        score += 10
    # tier 4 → +0

    # Cost bonus for budget-sensitive modes
    if budget_mode in ("free", "economy"):
        if model.get("cost_per_call", -1) == 0 or model.get("access") == "local":
            score += 10

    # Latency bonus
    if speed_priority and model.get("avg_latency_seconds", 9999) < 30:
        score += 10

    # No-watermark bonus for final renders
    if is_final_render and not model.get("watermark", True):
        score += 5

    return score


def select_model(
    task_node: dict,
    registry: dict,
    budget_mode: str = "economy",
    available_vram_gb: float = 0.0,
    env_keys: dict[str, str] | None = None,
) -> dict:
    """
    Select PRIMARY and FALLBACK models for a given task node.

    Parameters
    ----------
    task_node : dict
        A single task node from task_graph["tasks"].
    registry : dict
        The full capability registry.
    budget_mode : str
        One of: free, economy, production, premium.
    available_vram_gb : float
        Free GPU VRAM in GB (0.0 if no GPU).
    env_keys : dict, optional
        Mapping of env var name → value. Defaults to os.environ.

    Returns
    -------
    dict with keys "primary" and "fallback" (each is a model entry dict or None).
    """
    import os

    if env_keys is None:
        env_keys = dict(os.environ)

    task_type = task_node.get("task_type", "")
    model_options = task_node.get("model_options", [])
    inputs = task_node.get("inputs", {})
    required_duration = float(inputs.get("duration_seconds", 0.0))
    quality_req = task_node.get("quality_requirement", "draft")
    is_final_render = quality_req == "final"

    allowed_tiers = _BUDGET_TIERS.get(budget_mode, {1, 2, 3, 4})

    model_index = {m["model_id"]: m for m in registry.get("models", [])}

    # Candidates are the model_options list from the task node, in order
    candidates: list[dict] = []
    for model_id in model_options:
        m = model_index.get(model_id)
        if m is None:
            logger.debug("Model '%s' not found in registry — skipping.", model_id)
            continue
        candidates.append(m)

    # Fall back to scanning registry by category if no valid candidates found from list
    if not candidates:
        for m in registry.get("models", []):
            if m.get("category") == task_type:
                candidates.append(m)

    # ── Hard constraints ───────────────────────────────────────────────────────
    eligible: list[dict] = []
    for m in candidates:
        model_id = m.get("model_id", "?")

        # Tier filter
        if m.get("quality_tier", 4) not in allowed_tiers:
            logger.debug("Eliminated '%s': tier %d not in %s", model_id, m.get("quality_tier", 4), allowed_tiers)
            continue

        # VRAM constraint (local models only)
        if m.get("access") == "local":
            vram_needed = m.get("vram_required_gb", 0)
            if vram_needed > available_vram_gb:
                logger.debug(
                    "Eliminated '%s': needs %.1fGB VRAM, only %.1fGB available",
                    model_id, vram_needed, available_vram_gb,
                )
                continue

        # API key constraint
        if m.get("access") == "api":
            key_env = m.get("api_key_env")
            if key_env and key_env not in env_keys:
                logger.debug("Eliminated '%s': missing API key env '%s'", model_id, key_env)
                continue

        # Duration constraint
        max_dur = m.get("output_format", {}).get("max_duration_seconds")
        if max_dur is not None and required_duration > max_dur:
            logger.debug(
                "Eliminated '%s': max_duration %ss < required %ss",
                model_id, max_dur, required_duration,
            )
            continue

        eligible.append(m)

    if not eligible:
        logger.warning(
            "No eligible models for task '%s' (type=%s, budget=%s, vram=%.1f). "
            "Check model_options and constraints.",
            task_node.get("task_id", "?"), task_type, budget_mode, available_vram_gb,
        )
        return {"primary": None, "fallback": None}

    # ── Score and sort ─────────────────────────────────────────────────────────
    scored = sorted(
        eligible,
        key=lambda m: _score_model(m, budget_mode, is_final_render=is_final_render),
        reverse=True,
    )

    primary = scored[0] if scored else None
    fallback = scored[1] if len(scored) > 1 else None

    logger.info(
        "Task '%s': primary=%s, fallback=%s",
        task_node.get("task_id", "?"),
        primary["model_id"] if primary else "none",
        fallback["model_id"] if fallback else "none",
    )
    return {"primary": primary, "fallback": fallback}
