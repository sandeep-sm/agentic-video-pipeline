"""
Stage 2 — Intent Clarifier
Compares user intent against asset descriptions, flags ambiguities, and
resolves or documents them before task planning.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Parameters we consider "underspecified" if absent from the intent string
UNDERSPECIFIED_PARAMS = {
    "duration": {
        "patterns": [r"\d+\s*[- ]?\s*(?:s|sec|secs|second|seconds|min|mins|minute|minutes)\b", r"duration", r"long"],
        "default": "9 seconds",
        "default_key": "duration_seconds",
        "default_value": 9.0,
    },
    "resolution": {
        "patterns": [r"\d{3,4}[xX×]\d{3,4}", r"resolution", r"1080", r"720", r"4k"],
        "default": "1920×1080 (HD)",
        "default_key": "resolution",
        "default_value": [1920, 1080],
    },
    "fps": {
        "patterns": [r"\bfps\b", r"frame.?rate", r"\b24\b|\b30\b|\b60\b"],
        "default": "30 fps",
        "default_key": "fps",
        "default_value": 30,
    },
    "style": {
        "patterns": [r"style", r"cinematic", r"cartoon", r"realistic", r"minimal"],
        "default": "match existing asset style",
        "default_key": "style",
        "default_value": "match_asset",
    },
}


def _intent_mentions(intent: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches in the intent (case-insensitive)."""
    for pattern in patterns:
        if re.search(pattern, intent, re.IGNORECASE):
            return True
    return False


def _find_asset_references(intent: str, asset_names: list[str]) -> list[str]:
    """Return assets explicitly mentioned in the intent (by filename stem or full name)."""
    mentioned = []
    intent_lower = intent.lower()
    for name in asset_names:
        stem = name.rsplit(".", 1)[0].lower().replace("_", " ").replace("-", " ")
        if name.lower() in intent_lower or stem in intent_lower:
            mentioned.append(name)
    return mentioned


def _check_subject_references(intent: str, descriptions: dict) -> list[dict]:
    """
    Detect if the intent references a subject that doesn't match any asset.
    Returns a list of conflict dicts.
    """
    conflicts = []
    assets = descriptions.get("assets", {})
    # Simple noun extraction: words that follow "the" or "my"
    subject_matches = re.findall(r"\b(?:the|my|a|an)\s+([a-z]+(?:\s+[a-z]+)?)\b", intent.lower())
    for subject in subject_matches:
        # Check if any asset's subject_matter contains this noun
        found = any(
            subject in str(info.get("subject_matter", "")).lower()
            for info in assets.values()
        )
        if not found and len(subject.split()) <= 3:
            conflicts.append({
                "type": "subject_not_in_assets",
                "subject": subject,
                "description": (
                    f"Intent references '{subject}' but no asset describes this subject. "
                    "Verify asset names or add the missing asset."
                ),
            })
    return conflicts


def clarify_intent(
    intent: str,
    descriptions: dict,
    mode: str = "batch",
) -> dict:
    """
    Analyse user intent against asset descriptions.

    Parameters
    ----------
    intent : str
        The raw user intent text.
    descriptions : dict
        Output of stage1_asset_analyzer (the full descriptions.json dict).
    mode : str
        'batch' — document assumptions, never ask.
        'interactive' — print up to 3 clarifying questions and read stdin.

    Returns
    -------
    dict following the clarification result schema.
    """
    assets: dict[str, Any] = descriptions.get("assets", {})
    asset_names = list(assets.keys())

    ambiguities: list[dict] = []
    assumptions: list[str] = []
    asset_intent_conflicts: list[dict] = []
    missing_assets: list[str] = []
    clarifying_questions: list[str] = []

    # ── 1. Detect underspecified parameters ───────────────────────────────────
    for param, spec in UNDERSPECIFIED_PARAMS.items():
        if not _intent_mentions(intent, spec["patterns"]):
            ambiguities.append({
                "type": "missing_parameter",
                "parameter": param,
                "description": f"'{param}' not specified in intent",
                "default_used": spec["default"],
            })
            assumptions.append(f"{param.capitalize()} set to {spec['default']} (default)")

    # ── 2. Check for asset references in intent ────────────────────────────────
    explicitly_mentioned = _find_asset_references(intent, asset_names)
    if not explicitly_mentioned and asset_names:
        ambiguities.append({
            "type": "ambiguous_asset_reference",
            "description": (
                "Intent does not explicitly reference any asset by name. "
                f"Available assets: {', '.join(asset_names)}"
            ),
            "default_used": f"All {len(asset_names)} asset(s) assumed relevant",
        })
        assumptions.append(f"All available assets assumed relevant: {', '.join(asset_names)}")

    # ── 3. Detect assets referenced in intent but not in descriptions ──────────
    for name in explicitly_mentioned:
        if name not in assets:
            missing_assets.append(name)
            asset_intent_conflicts.append({
                "type": "missing_asset",
                "asset": name,
                "description": f"Intent references '{name}' but it was not found in assets/",
                "severity": "critical",
            })

    # ── 4. Flag assets with corruption or low confidence ──────────────────────
    for flag in descriptions.get("flags", []):
        asset_intent_conflicts.append({
            "type": "flagged_asset",
            "asset": flag.get("file", "unknown"),
            "description": flag.get("reason", ""),
            "severity": "warning",
        })

    # ── 5. Subject-level mismatch ─────────────────────────────────────────────
    # (skipped for empty asset sets to avoid noise)
    if assets:
        subject_conflicts = _check_subject_references(intent, descriptions)
        # Only add if not already covered by explicit name references
        non_duplicate = [c for c in subject_conflicts if c["subject"] not in intent.lower()[:20]]
        asset_intent_conflicts.extend(non_duplicate)

    # ── 6. Interactive mode: ask clarifying questions ─────────────────────────
    if mode == "interactive" and ambiguities:
        questions = []
        for amb in ambiguities[:3]:  # max 3
            if amb["type"] == "missing_parameter":
                q = (
                    f"What {amb['parameter']} would you like? "
                    f"(Press Enter for default: {amb['default_used']})"
                )
                questions.append(q)
            elif amb["type"] == "ambiguous_asset_reference":
                q = (
                    f"Which asset(s) should be used? "
                    f"Available: {', '.join(asset_names)}"
                )
                questions.append(q)

        clarifying_questions = questions
        answers: dict[str, str] = {}
        for q in questions:
            print(f"\n[Intent Clarifier] {q}")
            try:
                ans = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            answers[q] = ans
            if ans:
                assumptions.append(f"User answered: '{ans}' for question: '{q}'")

    # ── 7. Critical conflict check ────────────────────────────────────────────
    critical_conflicts = [c for c in asset_intent_conflicts if c.get("severity") == "critical"]
    if critical_conflicts:
        descriptions_str = "; ".join(c["description"] for c in critical_conflicts)
        raise ValueError(
            f"Critical asset–intent conflict(s) detected. Pipeline cannot proceed.\n"
            f"Details: {descriptions_str}"
        )

    # ── 8. Build resolved intent ──────────────────────────────────────────────
    resolved_parts = [intent.strip()]
    for assumption in assumptions:
        resolved_parts.append(f"[ASSUMED] {assumption}")
    resolved_intent = " | ".join(resolved_parts)

    result = {
        "ambiguities": ambiguities,
        "assumptions": assumptions,
        "asset_intent_conflicts": asset_intent_conflicts,
        "missing_assets": missing_assets,
        "clarifying_questions": clarifying_questions,
        "resolved_intent": resolved_intent,
    }

    # Log summary
    logger.info(
        "Intent clarification complete: %d ambiguities, %d assumptions, %d conflicts",
        len(ambiguities),
        len(assumptions),
        len(asset_intent_conflicts),
    )
    if mode == "batch" and assumptions:
        for a in assumptions:
            logger.info("  Assumption: %s", a)

    return result
