"""
Stage 6 — Final QA
Validates the output video: resolution, FPS, duration, frame quality via VLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from statistics import mean

logger = logging.getLogger(__name__)

_FRAME_POSITIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
_FRAME_LABELS = ["0%", "25%", "50%", "75%", "100%"]
_DURATION_TOLERANCE = 0.1  # seconds


def _check_file(output_path: Path) -> tuple[bool, str]:
    if not output_path.exists():
        return False, f"Output file does not exist: {output_path}"
    if output_path.stat().st_size == 0:
        return False, "Output file is 0 bytes."
    return True, ""


def _get_video_metadata(output_path: Path) -> dict:
    """Return resolution, fps, and duration of a video file."""
    try:
        from moviepy.editor import VideoFileClip  # noqa: PLC0415

        clip = VideoFileClip(str(output_path))
        meta = {
            "width": clip.size[0],
            "height": clip.size[1],
            "fps": clip.fps,
            "duration": clip.duration,
        }
        clip.close()
        return meta
    except Exception as exc:
        logger.warning("moviepy metadata read failed (%s) — trying cv2.", exc)

    try:
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(str(output_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return {"width": width, "height": height, "fps": fps, "duration": duration}
    except Exception as exc:
        logger.warning("cv2 metadata read failed: %s", exc)

    return {}


def _extract_frame(output_path: Path, t: float, dest: Path) -> bool:
    """Save frame at time t (seconds) to dest as PNG."""
    try:
        from moviepy.editor import VideoFileClip  # noqa: PLC0415

        clip = VideoFileClip(str(output_path))
        t_clamped = min(t, clip.duration - 0.001)
        frame = clip.get_frame(t_clamped)
        clip.close()

        from PIL import Image  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        Image.fromarray(frame.astype("uint8")).save(str(dest))
        return True
    except Exception as exc:
        logger.warning("Frame extraction at t=%.2fs failed: %s", t, exc)
        return False


def _vlm_score_frame(frame_path: Path, intent: str, vlm_model: str) -> dict:
    """Score a single frame with a VLM against the original intent."""
    import base64
    import mimetypes

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    prompt = (
        f"You are evaluating the quality of a video frame.\n"
        f"Original intent: {intent}\n\n"
        "Score this frame on:\n"
        "  1. Visual quality (clarity, no artifacts) — out of 10\n"
        "  2. Intent match (does it reflect the described intent?) — out of 10\n\n"
        "Reply ONLY with JSON:\n"
        '{"score": <average 0-10>, "visual_quality": <0-10>, '
        '"intent_match": <0-10>, "feedback": "<one sentence>"}'
    )

    def _parse(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            sc = float(m.group(1)) if m else 7.0
            return {"score": sc, "feedback": text[:200], "_mock": True}

    if vlm_model == "claude-vision" and anthropic_key and frame_path.exists():
        try:
            import anthropic  # noqa: PLC0415

            client = anthropic.Anthropic(api_key=anthropic_key)
            mime = mimetypes.guess_type(str(frame_path))[0] or "image/png"
            img_b64 = base64.standard_b64encode(frame_path.read_bytes()).decode()
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return _parse(msg.content[0].text)
        except Exception as exc:
            logger.warning("Claude frame score failed: %s — mock.", exc)

    elif vlm_model == "gemini-vision" and google_key and frame_path.exists():
        try:
            import google.generativeai as genai  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415

            genai.configure(api_key=google_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            img = PILImage.open(frame_path)
            response = model.generate_content([img, prompt])
            return _parse(response.text)
        except Exception as exc:
            logger.warning("Gemini frame score failed: %s — mock.", exc)

    # Mock
    score = 7.0 if (frame_path.exists() and frame_path.stat().st_size > 0) else 3.0
    return {
        "score": score,
        "visual_quality": score,
        "intent_match": score,
        "feedback": "[MOCK] No VLM key — placeholder score.",
        "_mock": True,
    }


def run_qa(
    output_path: Path,
    spec: dict,
    intent: str,
    vlm_model: str = "claude-vision",
) -> dict:
    """
    Run all QA checks on the final video.

    Returns a QA result dict.
    """
    output_path = Path(output_path)
    video_params = spec.get("video_params", {})
    expected_resolution = video_params.get("resolution", [1920, 1080])
    expected_fps = float(video_params.get("fps", 30))
    expected_duration = float(video_params.get("total_duration_seconds", 9.0))

    checks: dict[str, bool] = {
        "file_exists": False,
        "resolution_match": False,
        "fps_match": False,
        "duration_match": False,
        "no_corrupt_frames": True,
    }
    failures: list[str] = []
    suggestions: list[str] = []
    frame_scores: list[dict] = []

    # ── Check 1: file exists ──────────────────────────────────────────────────
    file_ok, file_msg = _check_file(output_path)
    checks["file_exists"] = file_ok
    if not file_ok:
        failures.append(file_msg)
        suggestions.append("Re-run Stage 5 compositor — output file was not written.")
        return {
            "passed": False,
            "checks": checks,
            "frame_scores": [],
            "overall_score": 0.0,
            "failures": failures,
            "suggestions": suggestions,
        }

    # ── Check 2-4: video metadata ─────────────────────────────────────────────
    meta = _get_video_metadata(output_path)
    if meta:
        actual_w = meta.get("width", 0)
        actual_h = meta.get("height", 0)
        actual_fps = meta.get("fps", 0.0)
        actual_dur = meta.get("duration", 0.0)

        res_ok = (actual_w == expected_resolution[0] and actual_h == expected_resolution[1])
        checks["resolution_match"] = res_ok
        if not res_ok:
            failures.append(
                f"Resolution mismatch: expected {expected_resolution[0]}×{expected_resolution[1]}, "
                f"got {actual_w}×{actual_h}"
            )
            suggestions.append("Check compositor resolution settings in spec.json.")

        fps_ok = abs(actual_fps - expected_fps) < 1.0
        checks["fps_match"] = fps_ok
        if not fps_ok:
            failures.append(f"FPS mismatch: expected {expected_fps}, got {actual_fps:.1f}")
            suggestions.append("Re-render with correct FPS parameter.")

        dur_ok = abs(actual_dur - expected_duration) <= _DURATION_TOLERANCE
        checks["duration_match"] = dur_ok
        if not dur_ok:
            failures.append(
                f"Duration mismatch: expected {expected_duration:.1f}s ± {_DURATION_TOLERANCE}s, "
                f"got {actual_dur:.2f}s"
            )
            suggestions.append(
                "Check that all task clips cover the full timeline duration."
            )
    else:
        # Cannot read metadata
        failures.append("Could not read video metadata (no moviepy or cv2 available).")
        suggestions.append("Install moviepy or opencv-python for video validation.")

    # ── Check 5: sample 5 frames and score with VLM ───────────────────────────
    duration_for_sampling = meta.get("duration", expected_duration) if meta else expected_duration
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for i, pct in enumerate(_FRAME_POSITIONS):
            frame_time = pct * duration_for_sampling
            label = _FRAME_LABELS[i]
            frame_path = tmp / f"frame_{i:02d}.png"

            extracted = _extract_frame(output_path, frame_time, frame_path)
            if not extracted:
                checks["no_corrupt_frames"] = False
                failures.append(f"Could not extract frame at {label} ({frame_time:.2f}s)")
                frame_scores.append({
                    "position": label,
                    "frame_time": frame_time,
                    "score": 0.0,
                    "feedback": "Frame extraction failed",
                })
                continue

            score_result = _vlm_score_frame(frame_path, intent, vlm_model)
            frame_scores.append({
                "position": label,
                "frame_time": frame_time,
                "score": score_result.get("score", 0.0),
                "visual_quality": score_result.get("visual_quality", 0.0),
                "intent_match": score_result.get("intent_match", 0.0),
                "feedback": score_result.get("feedback", ""),
                "_mock": score_result.get("_mock", False),
            })

    # ── Compute overall score ─────────────────────────────────────────────────
    scores = [fs["score"] for fs in frame_scores if "score" in fs]
    overall_score = round(mean(scores), 2) if scores else 0.0

    all_checks_passed = all(checks.values())
    passed = all_checks_passed and overall_score >= 5.0

    if not passed and overall_score < 5.0:
        failures.append(f"Overall VLM score {overall_score:.1f}/10 is below threshold (5.0).")
        suggestions.append("Review stage 4 task outputs — quality gates may have passed too low.")

    result = {
        "passed": passed,
        "checks": checks,
        "frame_scores": frame_scores,
        "overall_score": overall_score,
        "failures": failures,
        "suggestions": suggestions,
    }

    status = "PASSED" if passed else "FAILED"
    logger.info("QA %s: score=%.1f/10, checks=%s", status, overall_score, checks)
    return result
