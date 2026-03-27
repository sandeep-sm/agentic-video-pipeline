"""
Stage 1 — Asset Analyzer
Runs a VLM over every file in assets/ and produces assets/descriptions.json.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

from scripts.utils import get_project_root, load_registry, save_json

logger = logging.getLogger(__name__)

# Extensions considered "media assets" (images, video, audio)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}


def _call_vlm(model_id: str, prompt: str, image_path: Path | None = None) -> dict:
    """
    Call a VLM to analyze an asset or validate quality.
    Falls back to a mock response when no API key is available.

    Returns a dict with at least {"result": ..., "_mock": bool}.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    # ── Claude Vision ─────────────────────────────────────────────────────────
    if model_id == "claude-vision" and anthropic_key:
        try:
            import anthropic  # noqa: PLC0415

            client = anthropic.Anthropic(api_key=anthropic_key)
            content: list = []
            if image_path and image_path.exists():
                mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
                img_b64 = base64.standard_b64encode(image_path.read_bytes()).decode()
                content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_b64}})
            content.append({"type": "text", "text": prompt})
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": content}],
            )
            return {"result": message.content[0].text, "_mock": False}
        except Exception as exc:
            logger.warning("Claude Vision call failed: %s — falling back to mock", exc)

    # ── Gemini Vision ─────────────────────────────────────────────────────────
    if model_id == "gemini-vision" and google_key:
        try:
            from google import genai  # noqa: PLC0415

            client = genai.Client(api_key=google_key)
            parts: list = [prompt]
            if image_path and image_path.exists():
                from PIL import Image as PILImage  # noqa: PLC0415

                parts.insert(0, PILImage.open(image_path))
            response = client.models.generate_content(model="gemini-3.0-flash", contents=parts)
            return {"result": response.text, "_mock": False}
        except Exception as exc:
            logger.warning("Gemini Vision call failed: %s — falling back to mock", exc)

    # ── Mock ──────────────────────────────────────────────────────────────────
    mock_result = (
        '{"file_type": "image", "dimensions": [512, 512], '
        '"subject_matter": "Mock subject — no VLM key available", '
        '"background": "unknown", "style": "unknown", '
        '"dominant_colors": ["#000000"], '
        '"notable_features": "Mock analysis", '
        '"suggested_use": "unknown", '
        '"confidence": 0.5}'
    )
    return {"result": mock_result, "_mock": True}


def _parse_vlm_asset_response(raw: str) -> dict:
    """Parse the VLM text response into a structured asset description dict."""
    import json
    import re

    # Try direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: return a minimal structure
    return {
        "file_type": "unknown",
        "dimensions": [],
        "subject_matter": raw[:200],
        "background": "unknown",
        "style": "unknown",
        "dominant_colors": [],
        "notable_features": raw[:200],
        "suggested_use": "unknown",
        "confidence": 0.4,
    }


def _build_asset_prompt(asset_path: Path) -> str:
    ext = asset_path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        media_hint = "image"
    elif ext in VIDEO_EXTENSIONS:
        media_hint = "video"
    else:
        media_hint = "audio"

    return (
        f"You are analyzing a {media_hint} asset for a video production pipeline.\n"
        f"File name: {asset_path.name}\n\n"
        "Return ONLY a valid JSON object with these keys:\n"
        "  file_type       (string: 'image'|'video'|'audio')\n"
        "  dimensions      (array [width, height] for images; [duration_seconds] for video/audio)\n"
        "  subject_matter  (string: what or who is depicted)\n"
        "  background      (string: background description or 'transparent')\n"
        "  style           (string: e.g. 'photorealistic', 'cartoon', 'illustrated')\n"
        "  dominant_colors (array of hex color strings)\n"
        "  notable_features (string: key visual features)\n"
        "  suggested_use   (string: how this asset could be used in the video)\n"
        "  confidence      (float 0–1: your confidence in this analysis)\n\n"
        "Return ONLY the JSON object. No markdown, no explanation."
    )


def _get_image_dimensions(path: Path) -> list[int]:
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as img:
            return list(img.size)
    except Exception:
        return []


def _is_corrupt(path: Path) -> bool:
    """Return True if the file is 0 bytes or cannot be opened as its declared type."""
    if path.stat().st_size == 0:
        return True
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        try:
            from PIL import Image  # noqa: PLC0415

            with Image.open(path):
                pass
        except Exception:
            return True
    return False


def analyze_assets(vlm_model: str = "claude-vision", budget_mode: str = "economy") -> dict:
    """
    Scan assets/, analyze each file with a VLM, and save assets/descriptions.json.

    Returns the descriptions dict.
    """
    root = get_project_root()
    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    all_extensions = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
    asset_files = [
        p for p in sorted(assets_dir.iterdir())
        if p.is_file() and p.suffix.lower() in all_extensions
    ]

    logger.info("Found %d asset(s) to analyze in %s", len(asset_files), assets_dir)

    # In economy mode prefer gemini for cost savings on VLM gates
    effective_model = vlm_model
    if budget_mode in ("economy", "free") and vlm_model == "claude-vision":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            effective_model = "gemini-vision"

    assets_result: dict = {}
    flags: list = []

    for asset_path in asset_files:
        logger.info("Analyzing: %s", asset_path.name)

        # Corruption check
        corrupt = _is_corrupt(asset_path)
        if corrupt:
            flags.append({
                "file": asset_path.name,
                "reason": "File is 0 bytes or cannot be opened — possible corruption",
            })
            logger.warning("Flagged corrupt: %s", asset_path.name)

        # Image dimensions (fast local check — use as ground truth over VLM)
        local_dims: list[int] = []
        if asset_path.suffix.lower() in IMAGE_EXTENSIONS:
            local_dims = _get_image_dimensions(asset_path)

        # Extension vs content mismatch heuristic
        declared_ext = asset_path.suffix.lower()
        if declared_ext in IMAGE_EXTENSIONS:
            expected_type = "image"
        elif declared_ext in VIDEO_EXTENSIONS:
            expected_type = "video"
        else:
            expected_type = "audio"

        # VLM call
        prompt = _build_asset_prompt(asset_path)
        image_arg = asset_path if asset_path.suffix.lower() in IMAGE_EXTENSIONS else None
        vlm_response = _call_vlm(effective_model, prompt, image_arg)
        parsed = _parse_vlm_asset_response(vlm_response["result"])

        # Override dimensions from local check when available
        if local_dims:
            parsed["dimensions"] = local_dims

        # Extension mismatch flag
        if parsed.get("file_type") and parsed["file_type"] != expected_type:
            flags.append({
                "file": asset_path.name,
                "reason": (
                    f"Extension suggests '{expected_type}' but VLM reports "
                    f"'{parsed['file_type']}' — possible misname"
                ),
            })

        # Low confidence flag
        confidence = float(parsed.get("confidence", 1.0))
        if confidence < 0.5:
            flags.append({
                "file": asset_path.name,
                "reason": f"Low VLM confidence ({confidence:.2f}) — review manually",
            })

        if vlm_response.get("_mock"):
            parsed["_mock"] = True

        assets_result[asset_path.name] = parsed

    descriptions = {
        "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vlm_model": effective_model,
        "assets": assets_result,
        "flags": flags,
    }

    out_path = assets_dir / "descriptions.json"
    save_json(descriptions, out_path)
    logger.info("Saved descriptions to %s", out_path)
    return descriptions
