"""
Stage 4 — Cost Estimator
Pre-flight cost analysis. Reads task_graph + registry → storyboard/cost_estimate.json.
"""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.utils import get_project_root, save_json

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when expected cost exceeds the configured maximum."""


# VLM gate costs (per check) by model
_VLM_GATE_COST: dict[str, float] = {
    "gemini-vision": 0.01,
    "claude-vision": 0.03,
}

# Budget mode → allowed quality tiers
_BUDGET_TIERS: dict[str, set] = {
    "free": {3, 4},
    "economy": {2, 3, 4},
    "production": {1, 2, 3, 4},
    "premium": {1},
}

# Base overhead costs incurred on every run
_BASE_COSTS: dict[str, dict] = {
    "task_planning": {
        "model": "claude-opus",
        "cost_usd": 0.20,
        "note": "LLM task planning (Claude Opus, ~3k in / 2k out)",
    },
    "asset_analysis": {
        "model": "claude-vision / gemini-vision",
        "cost_usd": 0.04,
        "note": "VLM asset analysis (per 6 assets, ~5k tokens)",
    },
}


def _find_model(model_id: str, registry: dict) -> dict | None:
    for m in registry.get("models", []):
        if m.get("model_id") == model_id:
            return m
    return None


def _model_cost(model: dict, task_node: dict) -> float:
    """Compute estimated cost for running model on task_node."""
    if model is None:
        return 0.0

    # Local / free models
    if model.get("access") == "local":
        return 0.0

    inputs = task_node.get("inputs", {})

    if "cost_per_second" in model:
        duration = float(inputs.get("duration_seconds", 5.0))
        return model["cost_per_second"] * duration

    if "cost_per_image" in model:
        count = int(inputs.get("count", 1))
        return model["cost_per_image"] * max(1, count)

    if "cost_per_call" in model:
        return float(model["cost_per_call"])

    if "cost_per_1k_tokens" in model:
        # Estimate 2k tokens per planning call
        return model["cost_per_1k_tokens"] * 2.0

    if "cost_per_1000_chars" in model:
        # Estimate 500 chars per call
        return model["cost_per_1000_chars"] * 0.5

    return 0.0


def _select_primary_fallback(
    task_node: dict, registry: dict, budget_mode: str
) -> tuple[dict | None, dict | None]:
    """Return (primary_model, fallback_model) dicts from model_options."""
    allowed_tiers = _BUDGET_TIERS.get(budget_mode, {1, 2, 3, 4})
    primary: dict | None = None
    fallback: dict | None = None

    for model_id in task_node.get("model_options", []):
        m = _find_model(model_id, registry)
        if m is None:
            continue
        tier = m.get("quality_tier", 4)
        if tier not in allowed_tiers:
            continue
        if primary is None:
            primary = m
        elif fallback is None:
            fallback = m
            break

    return primary, fallback


def _vlm_gate_cost_per_check(budget_mode: str) -> tuple[str, float]:
    """Return (vlm_model_name, cost_per_gate) based on budget_mode."""
    if budget_mode in ("production", "premium"):
        return "claude-vision", _VLM_GATE_COST["claude-vision"]
    return "gemini-vision", _VLM_GATE_COST["gemini-vision"]


def estimate_cost(
    task_graph: dict,
    registry: dict,
    budget_mode: str = "economy",
    max_cost_usd: float = 5.0,
    run_id: str = "run_unknown",
) -> dict:
    """
    Estimate the cost of executing task_graph.

    Writes storyboard/cost_estimate.json and returns the estimate dict.
    """
    tasks = task_graph.get("tasks", [])
    breakdown: list[dict] = []
    warnings: list[str] = []

    min_total = 0.0
    expected_total = 0.0
    max_total = 0.0

    vlm_model_name, vlm_cost_per_check = _vlm_gate_cost_per_check(budget_mode)
    n_tasks = len(tasks)

    for task_node in tasks:
        task_id = task_node.get("task_id", "unknown")
        task_type = task_node.get("task_type", "unknown")

        primary, fallback = _select_primary_fallback(task_node, registry, budget_mode)

        primary_cost = _model_cost(primary, task_node) if primary else 0.0
        fallback_cost = _model_cost(fallback, task_node) if fallback else 0.0

        # Find tier-1 cost for worst-case (max)
        worst_cost = primary_cost
        for model_id in task_node.get("model_options", []):
            m = _find_model(model_id, registry)
            if m and m.get("quality_tier") == 1:
                tier1_cost = _model_cost(m, task_node)
                worst_cost = max(worst_cost, tier1_cost)

        # Min is always $0 (local models exist for most categories)
        local_exists = any(
            _find_model(mid, registry) is not None
            and _find_model(mid, registry).get("access") == "local"
            for mid in task_node.get("model_options", [])
        )
        task_min = 0.0 if local_exists else primary_cost

        entry = {
            "task_id": task_id,
            "task_type": task_type,
            "primary_model": primary["model_id"] if primary else "none",
            "primary_cost_usd": round(primary_cost, 4),
            "fallback_model": fallback["model_id"] if fallback else "none",
            "fallback_cost_usd": round(fallback_cost, 4),
            "note": _cost_note(primary, task_node),
        }

        # Warn if fallback would be significantly more expensive
        if fallback and fallback_cost > primary_cost * 2:
            warnings.append(
                f"If '{task_id}' falls back to {fallback['model_id']}, "
                f"cost increases by ${fallback_cost - primary_cost:.2f}"
            )

        breakdown.append(entry)
        min_total += task_min
        expected_total += primary_cost
        max_total += worst_cost

    # ── VLM gate overhead (one per task + asset analysis gates) ───────────────
    vlm_gate_count = n_tasks + len(task_graph.get("tasks", []))
    vlm_total = vlm_gate_count * vlm_cost_per_check
    breakdown.append({
        "task_id": "vlm_gates",
        "task_type": "vlm_quality_gates",
        "primary_model": vlm_model_name,
        "primary_cost_usd": round(vlm_total, 4),
        "fallback_model": "none",
        "fallback_cost_usd": 0.0,
        "note": f"{vlm_gate_count} VLM quality checks at ${vlm_cost_per_check}/check",
    })
    expected_total += vlm_total
    min_total += 0.0  # local fallback not available for VLM gates in most setups
    max_total += vlm_gate_count * _VLM_GATE_COST["claude-vision"]

    # ── Base overhead ─────────────────────────────────────────────────────────
    for key, base in _BASE_COSTS.items():
        breakdown.append({
            "task_id": key,
            "task_type": "llm_orchestration",
            "primary_model": base["model"],
            "primary_cost_usd": base["cost_usd"],
            "fallback_model": "none",
            "fallback_cost_usd": 0.0,
            "note": base["note"],
        })
        expected_total += base["cost_usd"]
        max_total += base["cost_usd"]

    # ── Budget check ──────────────────────────────────────────────────────────
    within_budget = expected_total <= max_cost_usd
    if not within_budget:
        warnings.append(
            f"Expected cost ${expected_total:.2f} exceeds budget cap ${max_cost_usd:.2f}. "
            "Consider switching to a lower budget_mode or reducing task count."
        )

    # Check for models outside allowed tiers
    allowed_tiers = _BUDGET_TIERS.get(budget_mode, {1, 2, 3, 4})
    for task_node in tasks:
        primary, _ = _select_primary_fallback(task_node, registry, budget_mode)
        if primary is None:
            warnings.append(
                f"Task '{task_node.get('task_id')}': no model available for "
                f"budget_mode='{budget_mode}'. All model_options may be outside allowed tiers."
            )

    cost_estimate = {
        "run_id": run_id,
        "budget_mode": budget_mode,
        "budget_cap_usd": max_cost_usd,
        "estimates": {
            "minimum_usd": round(min_total, 4),
            "expected_usd": round(expected_total, 4),
            "maximum_usd": round(max_total, 4),
        },
        "breakdown": breakdown,
        "within_budget": within_budget,
        "warnings": warnings,
    }

    root = get_project_root()
    save_json(cost_estimate, root / "storyboard" / "cost_estimate.json")
    logger.info(
        "Cost estimate: min=$%.2f expected=$%.2f max=$%.2f within_budget=%s",
        min_total,
        expected_total,
        max_total,
        within_budget,
    )
    return cost_estimate


def _cost_note(model: dict | None, task_node: dict) -> str:
    if model is None:
        return "No model selected"
    if model.get("access") == "local":
        return f"Local model — GPU time only (no API cost)"
    inputs = task_node.get("inputs", {})
    if "cost_per_second" in model:
        dur = inputs.get("duration_seconds", 5.0)
        return f"{dur}s clip at ${model['cost_per_second']}/sec"
    if "cost_per_image" in model:
        return f"${model['cost_per_image']}/image"
    if "cost_per_call" in model:
        return f"${model['cost_per_call']}/call"
    if "cost_per_1k_tokens" in model:
        return f"${model['cost_per_1k_tokens']}/1k tokens (~2k estimated)"
    if "cost_per_1000_chars" in model:
        return f"${model['cost_per_1000_chars']}/1000 chars (~500 estimated)"
    return "unknown pricing"


def check_budget_and_halt(
    cost_estimate: dict,
    mode: str = "batch",
) -> bool:
    """
    Check whether it is safe to proceed with execution.

    Returns True if OK.
    In batch mode with within_budget=False: raises BudgetExceededError.
    In interactive mode: prints breakdown, asks user y/n.
    """
    if cost_estimate.get("within_budget", True):
        return True

    estimates = cost_estimate.get("estimates", {})
    cap = cost_estimate.get("budget_cap_usd", 5.0)
    expected = estimates.get("expected_usd", 0.0)
    budget_mode = cost_estimate.get("budget_mode", "economy")

    breakdown_lines = [
        f"  {b['task_id']}: ${b['primary_cost_usd']:.4f} ({b.get('note', '')})"
        for b in cost_estimate.get("breakdown", [])
    ]
    breakdown_str = "\n".join(breakdown_lines)
    message = (
        f"\n[BUDGET EXCEEDED]\n"
        f"  Expected cost: ${expected:.2f}\n"
        f"  Budget cap:    ${cap:.2f}\n"
        f"  Budget mode:   {budget_mode}\n"
        f"\nCost breakdown:\n{breakdown_str}\n"
        f"\nSuggestions:\n"
        f"  • Switch --budget to 'economy' or 'free'\n"
        f"  • Increase --max-cost\n"
        f"  • Reduce number of video generation tasks\n"
    )

    if mode == "interactive":
        print(message)
        try:
            answer = input("Proceed anyway? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer == "y":
            logger.warning("User chose to proceed despite budget exceeded.")
            return True

    raise BudgetExceededError(message)
