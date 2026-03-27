#!/usr/bin/env python3
"""
Agentic Video Pipeline — Main Orchestrator

Usage:
  python scripts/pipeline.py --intent "Animate my character walking across the screen" \\
    [--budget economy] [--max-cost 5.0] [--mode batch|interactive] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.utils import (
    append_progress,
    ensure_dirs,
    generate_run_id,
    get_project_root,
    load_json,
    load_registry,
    save_json,
    setup_logging,
)
from scripts.stage1_asset_analyzer import analyze_assets
from scripts.stage2_intent_clarifier import clarify_intent
from scripts.stage3_task_planner import plan_tasks, _topological_sort
from scripts.stage4_cost_estimator import BudgetExceededError, check_budget_and_halt, estimate_cost
from scripts.stage4_router import check_api_keys, get_available_vram, select_model
from scripts.stage4_executor import ExecutionResult, execute_task, retry_with_fallback
from scripts.stage5_compositor import compose_video
from scripts.stage6_qa import run_qa

logger = logging.getLogger(__name__)

_W = 62  # banner width


def _print_cost_banner(cost_estimate: dict, max_cost_usd: float) -> None:
    """Print a formatted cost estimate banner to stdout before execution starts."""
    est = cost_estimate.get("estimates", {})
    mn  = est.get("minimum_usd", 0.0)
    ex  = est.get("expected_usd", 0.0)
    mx  = est.get("maximum_usd", 0.0)
    run_id = cost_estimate.get("run_id", "?")
    mode   = cost_estimate.get("budget_mode", "?")
    within = cost_estimate.get("within_budget", True)

    sep = "─" * _W

    def _row(content: str) -> str:
        """Format content into a padded banner row of width _W."""
        # inner width = _W - 4 (for "│  " prefix and "│" suffix)
        return f"│  {content:<{_W - 3}}│"

    print(f"\n┌{sep}┐")
    print(_row(f"COST ESTIMATE  ({run_id})"))
    print(f"├{sep}┤")
    print(_row(f"Budget mode : {mode}"))
    print(_row(f"Minimum     : ${mn:.2f}   (all local / free)"))
    print(_row(f"Expected    : ${ex:.2f}   ← planned spend"))
    print(_row(f"Maximum     : ${mx:.2f}   (worst-case, all fallbacks)"))
    cap_flag = "✓ within budget" if within else "✗ OVER BUDGET"
    print(_row(f"Cap         : ${max_cost_usd:.2f}   {cap_flag}"))
    breakdown = cost_estimate.get("breakdown", [])
    if breakdown:
        print(f"├{sep}┤")
        print(_row("Task breakdown:"))
        for item in breakdown:
            tid   = item.get("task_id", "?")[:22]
            model = item.get("primary_model", "?")[:18]
            cost  = item.get("primary_cost_usd", 0.0)
            note  = item.get("note", "")[:16]
            print(_row(f"  {tid:<23} {model:<19} ${cost:.2f}   {note}"))
    warnings = cost_estimate.get("warnings", [])
    if warnings:
        print(f"├{sep}┤")
        for w in warnings[:3]:
            print(_row(f"⚠  {w[: _W - 7]}"))
    print(f"└{sep}┘")


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as m:ss or h:mm:ss."""
    s = int(seconds)
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"


class _PeriodicReporter:
    """
    Background thread that prints a one-line status update every `interval` seconds
    while a long-running model call is in progress.

    Usage:
        reporter = _PeriodicReporter(task_id, task_type, model_name, running_cost_ref, interval=120)
        reporter.start()
        ... execute model call ...
        reporter.stop()
    """

    def __init__(
        self,
        task_id: str,
        task_type: str,
        model_name: str,
        running_cost_getter,   # callable → float, returns cost spent so far
        interval: float = 120.0,
    ) -> None:
        self._task_id = task_id
        self._task_type = task_type
        self._model_name = model_name
        self._running_cost_getter = running_cost_getter
        self._interval = interval
        self._start_time = time.monotonic()
        self._timer: threading.Timer | None = None
        self._stopped = False

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._schedule()

    def stop(self) -> None:
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()

    def _schedule(self) -> None:
        if not self._stopped:
            self._timer = threading.Timer(self._interval, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self) -> None:
        if self._stopped:
            return
        elapsed = time.monotonic() - self._start_time
        spent = self._running_cost_getter()
        print(
            f"  ⏳  [{self._task_id}] still running {self._task_type} "
            f"via {self._model_name} — {_fmt_elapsed(elapsed)} elapsed "
            f"| spent so far: ${spent:.4f}",
            flush=True,
        )
        self._schedule()  # reschedule


def _print_task_status(
    task_id: str,
    model_name: str,
    score: float,
    task_cost: float,
    running_total: float,
    task_num: int,
    total_tasks: int,
    status: str,  # "✓", "⚠", "✗"
) -> None:
    """Print one line of per-task status after it completes."""
    print(
        f"  {status}  [{task_num}/{total_tasks}] {task_id:<28} "
        f"{model_name:<20} score {score:.1f}/10  "
        f"${task_cost:.2f}  |  total ${running_total:.2f}",
        flush=True,
    )


def _print_final_cost_summary(
    run_id: str,
    cost_estimate: dict,
    task_results: list,
    start_time: float,
    max_cost_usd: float,
) -> None:
    """Print a complete cost summary at the end of the run."""
    actual = sum(r.cost_actual for r in task_results)
    est_ex = cost_estimate.get("estimates", {}).get("expected_usd", 0.0)
    elapsed = time.monotonic() - start_time
    diff = actual - est_ex
    diff_sign = "+" if diff > 0 else ""
    cap_remain = max_cost_usd - actual

    sep = "─" * _W

    def _row(content: str) -> str:
        return f"│  {content:<{_W - 3}}│"

    print(f"\n┌{sep}┐")
    print(_row(f"FINAL COST SUMMARY  ({run_id})"))
    print(f"├{sep}┤")
    print(_row(f"Estimated (pre-run) : ${est_ex:.4f}"))
    print(_row(f"Actual (post-run)   : ${actual:.4f}   ({diff_sign}{diff:.4f} vs estimate)"))
    print(_row(f"Cap                 : ${max_cost_usd:.2f}"))
    cap_label = "✓ within cap" if cap_remain >= 0 else "✗ OVER CAP"
    print(_row(f"Remaining budget    : ${cap_remain:.4f}   {cap_label}"))
    print(_row(f"Wall-clock time     : {_fmt_elapsed(elapsed)}"))
    breakdown = cost_estimate.get("breakdown", [])
    if breakdown:
        print(f"├{sep}┤")
        print(_row("Per-task breakdown:"))
        for item in breakdown:
            tid   = item.get("task_id", "?")[:22]
            model = item.get("primary_model", "?")[:16]
            est_c = item.get("primary_cost_usd", 0.0)
            match = next((r for r in task_results if r.task_id == item.get("task_id")), None)
            act_c = match.cost_actual if match else est_c
            print(_row(f"  {tid:<23} {model:<17} est ${est_c:.2f}  actual ${act_c:.2f}"))
    print(f"└{sep}┘\n")


def _select_vlm(budget_mode: str) -> str:
    """Choose the VLM to use for quality gates based on budget_mode."""
    import os
    if budget_mode in ("production", "premium") and os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-vision"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini-vision"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-vision"
    return "claude-vision"  # will fall back to mock internally


def _build_run_report(
    run_id: str,
    intent: str,
    budget_mode: str,
    start_time: float,
    stage_results: dict,
    task_results: list[ExecutionResult],
    qa_result: dict,
    assumptions: list[str],
    warnings: list[str],
    failures: list[dict],
) -> dict:
    total_cost = sum(r.cost_actual for r in task_results)
    duration_s = time.monotonic() - start_time

    task_report = [
        {
            "task_id": r.task_id,
            "model_used": r.model_used,
            "quality_score": r.quality_score,
            "cost_usd": round(r.cost_actual, 4),
            "success": r.success,
            "output_path": str(r.output_path) if r.output_path else None,
        }
        for r in task_results
    ]

    return {
        "run_id": run_id,
        "intent": intent,
        "budget_mode": budget_mode,
        "total_cost_actual_usd": round(total_cost, 4),
        "duration_seconds": round(duration_s, 1),
        "stages": {
            **stage_results,
            "stage4": {"tasks": task_report},
            "stage6": {
                "status": "complete" if qa_result.get("passed") else "failed",
                "qa_passed": qa_result.get("passed", False),
                "overall_score": qa_result.get("overall_score", 0.0),
            },
        },
        "assumptions": assumptions,
        "warnings": warnings,
        "failures": failures,
    }


def run_pipeline(
    intent: str,
    budget_mode: str = "economy",
    max_cost_usd: float = 5.0,
    mode: str = "batch",
    resume: bool = False,
) -> dict:
    """
    Run the full video production pipeline.

    Returns the run_report dict.
    """
    # Safety: never auto-select premium
    if budget_mode == "premium":
        logger.warning(
            "budget_mode 'premium' was passed. This is the highest-cost tier. "
            "Continuing as requested."
        )

    start_time = time.monotonic()
    run_id = generate_run_id()
    setup_logging(verbose=False)
    logger.info("=== Agentic Video Pipeline — %s ===", run_id)

    root = get_project_root()
    ensure_dirs()

    registry = load_registry()
    env_keys = __import__("os").environ.copy()

    stage_results: dict = {}
    all_warnings: list[str] = []
    all_failures: list[dict] = []
    task_results: list[ExecutionResult] = []

    # ──────────────────────────────────────────────────────────────────────────
    # PRE-FLIGHT: API keys (before any model calls)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Pre-flight] Checking API keys...")
    # We do a lightweight check before we have a task_graph — use dummy node scan
    # Full check happens after Stage 3 when we have the actual task graph.

    available_vram = get_available_vram()
    logger.info("[Pre-flight] Available VRAM: %.1f GB", available_vram)

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1 — Asset Analyzer
    # ──────────────────────────────────────────────────────────────────────────
    descriptions_path = root / "assets" / "descriptions.json"
    if resume and descriptions_path.exists():
        logger.info("[Stage 1] Resuming — loading existing descriptions.json")
        descriptions = load_json(descriptions_path)
    else:
        logger.info("[Stage 1] Analyzing assets...")
        vlm_model = _select_vlm(budget_mode)
        descriptions = analyze_assets(vlm_model=vlm_model, budget_mode=budget_mode)

    stage_results["stage1"] = {
        "status": "complete",
        "model": descriptions.get("vlm_model", "unknown"),
        "output": "assets/descriptions.json",
        "asset_count": len(descriptions.get("assets", {})),
        "flags": descriptions.get("flags", []),
    }
    logger.info("[Stage 1] Complete — %d asset(s) analyzed.", len(descriptions.get("assets", {})))

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2 — Intent Clarifier
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 2] Clarifying intent...")
    try:
        clarification = clarify_intent(intent, descriptions, mode=mode)
    except ValueError as exc:
        logger.error("[Stage 2] Critical conflict: %s", exc)
        raise

    stage_results["stage2"] = {
        "status": "complete",
        "ambiguities": len(clarification.get("ambiguities", [])),
        "assumptions": clarification.get("assumptions", []),
    }
    all_warnings.extend(
        f"Assumption: {a}" for a in clarification.get("assumptions", [])
    )
    logger.info("[Stage 2] Complete — %d assumption(s).", len(clarification.get("assumptions", [])))

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3 — Task Planner
    # ──────────────────────────────────────────────────────────────────────────
    task_graph_path = root / "storyboard" / "task_graph.json"
    spec_path = root / "storyboard" / "spec.json"

    if resume and task_graph_path.exists() and spec_path.exists():
        logger.info("[Stage 3] Resuming — loading existing task_graph.json and spec.json")
        task_graph = load_json(task_graph_path)
        spec = load_json(spec_path)
    else:
        logger.info("[Stage 3] Planning tasks...")
        resolution = [1920, 1080]
        fps = 30
        task_graph, spec = plan_tasks(
            intent=intent,
            descriptions=descriptions,
            clarification=clarification,
            budget_mode=budget_mode,
            fps=fps,
            resolution=resolution,
        )

    task_graph.setdefault("_run_id", run_id)

    stage_results["stage3"] = {
        "status": "complete",
        "output": "storyboard/task_graph.json",
        "task_count": len(task_graph.get("tasks", [])),
        "warnings": task_graph.get("warnings", []),
    }
    all_warnings.extend(task_graph.get("warnings", []))
    logger.info("[Stage 3] Complete — %d task(s) in graph.", len(task_graph.get("tasks", [])))

    # ──────────────────────────────────────────────────────────────────────────
    # PRE-FLIGHT (continued): API keys for actual task models
    # ──────────────────────────────────────────────────────────────────────────
    missing_keys = check_api_keys(task_graph, registry)
    if missing_keys:
        logger.warning(
            "[Pre-flight] Missing API keys: %s. Affected models will fall back to local/mock.",
            ", ".join(missing_keys),
        )
        all_warnings.append(f"Missing API keys: {', '.join(missing_keys)}")

    # ──────────────────────────────────────────────────────────────────────────
    # COST ESTIMATOR (mandatory pre-flight)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Pre-flight] Estimating costs...")
    cost_estimate = estimate_cost(
        task_graph=task_graph,
        registry=registry,
        budget_mode=budget_mode,
        max_cost_usd=max_cost_usd,
        run_id=run_id,
    )
    all_warnings.extend(cost_estimate.get("warnings", []))

    try:
        check_budget_and_halt(cost_estimate, mode=mode)
    except BudgetExceededError as exc:
        logger.error("[Pre-flight] Budget exceeded. Halting.")
        raise

    # ── Print prominent cost banner before any execution ──────────────────────
    _print_cost_banner(cost_estimate, max_cost_usd)
    if mode == "interactive":
        try:
            input("  Press ENTER to proceed or Ctrl+C to cancel... ")
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            raise SystemExit(0)

    # ──────────────────────────────────────────────────────────────────────────
    # DRAFT PASS (skip in premium mode)
    # ──────────────────────────────────────────────────────────────────────────
    draft_output_path = root / "outputs" / "draft_video.mp4"
    if budget_mode != "premium":
        logger.info("[Draft Pass] Running draft pass with local/free models...")
        draft_task_outputs: dict[str, Path] = {}
        draft_tasks = task_graph.get("tasks", [])

        try:
            topo_order = _topological_sort(draft_tasks)
        except ValueError:
            topo_order = [t["task_id"] for t in draft_tasks]

        task_index = {t["task_id"]: t for t in draft_tasks}

        for task_id in topo_order:
            task_node = task_index.get(task_id)
            if task_node is None:
                continue
            # Force free-tier model for draft
            draft_node = dict(task_node)
            draft_node["quality_requirement"] = "draft"
            selected = select_model(
                draft_node, registry,
                budget_mode="free",
                available_vram_gb=available_vram,
                env_keys=env_keys,
            )
            if selected["primary"] is None:
                # No free model — skip for draft
                continue
            intermediates_dir = root / "intermediates" / "draft"
            result = execute_task(draft_node, selected, registry, intermediates_dir, spec=spec)
            if result.output_path:
                draft_task_outputs[task_id] = result.output_path
                output_ref = str(draft_node.get("output_ref", "")).strip()
                if output_ref:
                    draft_task_outputs[output_ref] = result.output_path

        # Composite draft
        try:
            compose_video(spec, draft_task_outputs, draft_output_path, draft_mode=True)
            logger.info("[Draft Pass] Draft rendered: %s", draft_output_path)
        except Exception as exc:
            logger.warning("[Draft Pass] Draft compositor failed: %s — continuing to final.", exc)

        if mode == "interactive":
            print(f"\n[Draft Pass] Draft video available at: {draft_output_path}")
            try:
                answer = input("Proceed to final render? [Y/n] ").strip().lower()
                if answer == "n":
                    logger.info("User cancelled after draft review.")
                    return _build_run_report(
                        run_id, intent, budget_mode, start_time,
                        stage_results, task_results, {"passed": False, "overall_score": 0.0},
                        clarification.get("assumptions", []), all_warnings, all_failures,
                    )
            except (EOFError, KeyboardInterrupt):
                pass
        else:
            # Batch: auto-check draft quality gates
            logger.info("[Draft Pass] Batch mode — auto-proceeding to final render.")

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 4 — Final Pass
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 4] Executing tasks (final pass)...")
    vlm_model = _select_vlm(budget_mode)
    tasks = task_graph.get("tasks", [])

    try:
        topo_order = _topological_sort(tasks)
    except ValueError as exc:
        logger.error("[Stage 4] Circular dependency in task graph: %s", exc)
        raise

    task_index = {t["task_id"]: t for t in tasks}
    task_outputs: dict[str, Path] = {}
    MAX_RETRIES = 2
    running_cost: float = 0.0  # tracks actual spend as tasks complete
    task_num = 0

    for task_id in topo_order:
        task_node = task_index.get(task_id)
        if task_node is None:
            continue

        # Resume support: skip already-complete tasks
        if resume and task_node.get("status") == "complete" and task_node.get("output_path"):
            existing_path = Path(task_node["output_path"])
            if existing_path.exists():
                task_outputs[task_id] = existing_path
                output_ref = str(task_node.get("output_ref", "")).strip()
                if output_ref:
                    task_outputs[output_ref] = existing_path
                logger.info("[Stage 4] Skipping completed task '%s' (resume).", task_id)
                continue

        task_num += 1
        total_tasks = len(topo_order)

        # Select model
        selected = select_model(
            task_node, registry,
            budget_mode=budget_mode,
            available_vram_gb=available_vram,
            env_keys=env_keys,
        )
        model_name = (selected.get("primary") or {}).get("model_id", "unknown") if selected else "unknown"

        # Start periodic heartbeat for this task
        reporter = _PeriodicReporter(
            task_id=task_id,
            task_type=task_node.get("task_type", "?"),
            model_name=model_name,
            running_cost_getter=lambda: running_cost,
            interval=120.0,
        )
        reporter.start()

        try:
            intermediates_dir = root / "intermediates"
            result = execute_task(task_node, selected, registry, intermediates_dir, spec=spec)
        finally:
            reporter.stop()

        task_results.append(result)
        running_cost += result.cost_actual

        # Quality gate + per-task status line
        score = result.quality_score
        if score >= 7.0:
            _print_task_status(task_id, model_name, score, result.cost_actual, running_cost,
                               task_num, total_tasks, "✓")
            logger.debug("[Stage 4] Task '%s': ACCEPTED (score=%.1f)", task_id, score)
        elif score >= 5.0:
            _print_task_status(task_id, model_name, score, result.cost_actual, running_cost,
                               task_num, total_tasks, "⚠")
            all_warnings.append(f"Task '{task_id}': quality score {score:.1f}/10 (below 7.0 threshold)")
        else:
            # Score < 5 — retry with fallback
            logger.warning("[Stage 4] Task '%s': FAILED (score=%.1f) — retrying with fallback.", task_id, score)
            fallback_model = selected.get("fallback")

            fallback_model_name = (fallback_model or {}).get("model_id", "none")
            fb_reporter = _PeriodicReporter(
                task_id=f"{task_id}[fallback]",
                task_type=task_node.get("task_type", "?"),
                model_name=fallback_model_name,
                running_cost_getter=lambda: running_cost,
                interval=120.0,
            )
            fb_reporter.start()
            try:
                fallback_result = retry_with_fallback(
                    task_node, result, fallback_model, registry, intermediates_dir, spec=spec
                )
            finally:
                fb_reporter.stop()

            task_results.append(fallback_result)
            running_cost += fallback_result.cost_actual

            if fallback_result.quality_score >= 5.0:
                _print_task_status(task_id, fallback_model_name, fallback_result.quality_score,
                                   fallback_result.cost_actual, running_cost, task_num, total_tasks, "⚠")
                logger.info(
                    "[Stage 4] Task '%s': fallback ACCEPTED (score=%.1f)",
                    task_id, fallback_result.quality_score,
                )
                result = fallback_result
            else:
                _print_task_status(task_id, fallback_model_name, fallback_result.quality_score,
                                   fallback_result.cost_actual, running_cost, task_num, total_tasks, "✗")
                logger.error(
                    "[Stage 4] Task '%s': ESCALATED after %d attempt(s). Score=%.1f/10.",
                    task_id, MAX_RETRIES, fallback_result.quality_score,
                )
                all_failures.append({
                    "task_id": task_id,
                    "task_type": task_node.get("task_type", "?"),
                    "models_attempted": [
                        result.model_used,
                        fallback_result.model_used if fallback_result else "none",
                    ],
                    "scores": [result.quality_score, fallback_result.quality_score],
                    "error_message": fallback_result.error_message,
                    "suggestion": (
                        f"Task '{task_id}' failed VLM gate after 2 attempts. "
                        "Check model availability, VRAM, or adjust task parameters."
                    ),
                })
                result = fallback_result  # still track the output even if low quality

        # Record output
        if result.output_path and result.output_path.exists():
            task_outputs[task_id] = result.output_path
            output_ref = str(task_node.get("output_ref", "")).strip()
            if output_ref:
                task_outputs[output_ref] = result.output_path

        # Write task status to disk immediately (resumability)
        task_node["status"] = "complete" if result.success else "failed"
        task_node["output_path"] = str(result.output_path) if result.output_path else None
        task_node["quality_score"] = result.quality_score
        save_json(task_graph, root / "storyboard" / "task_graph.json")

    logger.info("[Stage 4] Complete — %d task(s) executed.", len(task_results))

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 5 — Compositor (Final)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 5] Compositing final video...")
    final_output_path = root / "outputs" / "final_video.mp4"
    try:
        compose_video(spec, task_outputs, final_output_path, draft_mode=False)
        stage_results["stage5"] = {"status": "complete", "output": str(final_output_path)}
        logger.info("[Stage 5] Final video: %s", final_output_path)
    except Exception as exc:
        logger.error("[Stage 5] Compositor failed: %s", exc)
        stage_results["stage5"] = {"status": "failed", "error": str(exc)}
        all_failures.append({
            "task_id": "stage5_compositor",
            "error_message": str(exc),
            "suggestion": "Check moviepy installation and ffmpeg availability.",
        })
        final_output_path = None  # Mark as non-existent for QA

    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 6 — Final QA
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("[Stage 6] Running final QA...")
    if final_output_path is not None and final_output_path.exists():
        qa_result = run_qa(
            output_path=final_output_path,
            spec=spec,
            intent=intent,
            vlm_model=vlm_model,
        )
    else:
        qa_result = {
            "passed": False,
            "overall_score": 0.0,
            "failures": ["Final video file does not exist."],
            "checks": {},
            "frame_scores": [],
            "suggestions": ["Fix Stage 5 compositor errors."],
        }

    all_failures.extend([
        {"task_id": "stage6_qa", "error_message": f, "suggestion": ""}
        for f in qa_result.get("failures", [])
    ])
    logger.info(
        "[Stage 6] QA %s: score=%.1f/10",
        "PASSED" if qa_result.get("passed") else "FAILED",
        qa_result.get("overall_score", 0.0),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # REPORT
    # ──────────────────────────────────────────────────────────────────────────
    run_report = _build_run_report(
        run_id=run_id,
        intent=intent,
        budget_mode=budget_mode,
        start_time=start_time,
        stage_results=stage_results,
        task_results=task_results,
        qa_result=qa_result,
        assumptions=clarification.get("assumptions", []),
        warnings=all_warnings,
        failures=all_failures,
    )

    report_path = root / "outputs" / "run_report.json"
    save_json(run_report, report_path)
    logger.info("Run report saved: %s", report_path)

    # Print full final cost summary
    _print_final_cost_summary(run_id, cost_estimate, task_results, start_time, max_cost_usd)

    # Append to PROGRESS.md
    qa_status = "PASSED" if qa_result.get("passed") else "FAILED"
    progress_entry = (
        f"Run {run_id} — {qa_status}\n"
        f"  Intent: {intent[:100]}\n"
        f"  Budget: {budget_mode}, Cost: ${run_report['total_cost_actual_usd']:.4f}\n"
        f"  Duration: {run_report['duration_seconds']:.1f}s\n"
        f"  QA score: {qa_result.get('overall_score', 0.0):.1f}/10\n"
        f"  Tasks: {len(task_results)}, Failures: {len(all_failures)}"
    )
    append_progress(progress_entry)

    # Print final status block
    print("=" * _W)
    print(f"  Run ID : {run_id}")
    print(f"  Status : {qa_status}")
    print(f"  Output : {final_output_path}")
    print(f"  QA     : {qa_result.get('overall_score', 0.0):.1f}/10")
    print(f"  Cost   : ${run_report['total_cost_actual_usd']:.4f}")
    print(f"  Time   : {_fmt_elapsed(time.monotonic() - start_time)}")
    if all_failures:
        print(f"  Failures ({len(all_failures)}):")
        for f in all_failures[:5]:
            print(f"    - {f.get('task_id','?')}: {f.get('error_message','')[:80]}")
    print("=" * _W)

    return run_report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic Video Pipeline — orchestrate multi-stage AI video production.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--intent",
        required=True,
        help='Natural language description of the desired video. E.g. "Animate my character walking."',
    )
    parser.add_argument(
        "--budget",
        dest="budget_mode",
        choices=["free", "economy", "production", "premium"],
        default="economy",
        help="Budget mode controlling which model tiers are used (default: economy).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        dest="max_cost_usd",
        help="Hard cost cap in USD. Pipeline halts if expected cost exceeds this (default: 5.0).",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "interactive"],
        default="batch",
        help="Execution mode: batch (no user input) or interactive (asks questions).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages that have already produced output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    try:
        run_pipeline(
            intent=args.intent,
            budget_mode=args.budget_mode,
            max_cost_usd=args.max_cost_usd,
            mode=args.mode,
            resume=args.resume,
        )
    except BudgetExceededError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
