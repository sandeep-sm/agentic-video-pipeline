"""
Shared utilities for all pipeline stages.
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def smoothstep(t: float) -> float:
    """S(t) = t²(3 - 2t), clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _strip_js_comments(text: str) -> str:
    """Strip JavaScript-style // line comments from JSON-like text."""
    # Remove // comments that are not inside strings
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i - 1] != "\\"):
            in_string = not in_string
            result.append(ch)
        elif not in_string and ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            # Skip until end of line
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def load_registry() -> dict:
    """Load registry/capability_registry.json, stripping // comments."""
    registry_path = get_project_root() / "registry" / "capability_registry.json"
    raw = registry_path.read_text(encoding="utf-8")
    cleaned = _strip_js_comments(raw)
    return json.loads(cleaned)


def load_json(path) -> dict:
    """Load any JSON file and return as a dict."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path) -> None:
    """Save dict to a JSON file with indent=2. Creates parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def append_progress(entry: str) -> None:
    """Append a dated entry to PROGRESS.md. Never overwrites existing content."""
    progress_path = get_project_root() / "PROGRESS.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"\n## {timestamp}\n{entry}\n"
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(line)


def ensure_dirs() -> None:
    """Create standard pipeline directories if they don't already exist."""
    root = get_project_root()
    for d in ("assets", "storyboard", "intermediates", "outputs"):
        (root / d).mkdir(parents=True, exist_ok=True)


def generate_run_id() -> str:
    """Return a unique run ID like 'run_20260326_001'."""
    now = datetime.now(timezone.utc)
    base = now.strftime("run_%Y%m%d")
    # Find a non-colliding suffix by checking outputs/
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        candidate = f"{base}_{n:03d}"
        marker = outputs_dir / f".{candidate}"
        if not marker.exists():
            marker.touch()
            return candidate
    return f"{base}_999"
