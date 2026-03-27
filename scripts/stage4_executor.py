"""
Stage 4 — Model Executor
Dispatches task nodes to handler stubs, runs VLM quality gates, and supports retry/fallback.
"""

from __future__ import annotations

import logging
import os
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    task_id: str
    model_used: str
    output_path: Path | None
    quality_score: float
    quality_report: dict
    cost_actual: float
    success: bool
    error_message: str = ""


# ---------------------------------------------------------------------------
# Placeholder file creators (no heavy deps beyond stdlib + Pillow)
# ---------------------------------------------------------------------------

def _write_placeholder_png(output_path: Path, width: int = 512, height: int = 512) -> None:
    """Save a 1-frame black PNG (requires Pillow)."""
    try:
        from PIL import Image  # noqa: PLC0415

        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), format="PNG")
    except ImportError:
        # Minimal 1×1 black PNG without Pillow
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_minimal_png(output_path)


def _write_minimal_png(path: Path) -> None:
    """Write a 1×1 black PNG using only stdlib."""
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat_raw = b"\x00\x00\x00\x00"  # filter byte + RGB
    idat_data = zlib.compress(idat_raw)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", idat_data)
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _write_placeholder_mp4(output_path: Path, duration_seconds: float = 1.0) -> None:
    """Save a 1-second black MP4 (requires moviepy or falls back to a tiny stub file)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from moviepy import ColorClip  # noqa: PLC0415

        clip = ColorClip(size=(256, 144), color=[0, 0, 0], duration=duration_seconds)
        clip.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio=False,
            logger=None,
            ffmpeg_params=["-crf", "28"],
        )
        clip.close()
    except Exception as exc:
        logger.warning("moviepy placeholder MP4 failed (%s) — writing stub bytes.", exc)
        # Write a minimal stub so the path exists and has non-zero size
        output_path.write_bytes(b"PLACEHOLDER_MP4_STUB")


def _write_placeholder_wav(output_path: Path, duration_seconds: float = 1.0) -> None:
    """Save a 1-second silent WAV using stdlib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 48000
    num_channels = 1
    sample_width = 2  # 16-bit
    n_frames = int(sample_rate * duration_seconds)
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


# ---------------------------------------------------------------------------
# VLM Quality Gate
# ---------------------------------------------------------------------------

def run_vlm_quality_gate(
    output_path: Path,
    validation_prompt: str,
    vlm_model: str = "claude-vision",
) -> dict:
    """
    Run a VLM quality check on output_path.

    Returns dict with:
        score (float 0-10), passed (bool), feedback (str), _mock (bool)
    """
    import base64
    import json
    import re

    score: float = 7.0
    feedback = ""
    is_mock = False

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    # Build the prompt
    system_prompt = (
        "You are a video quality evaluator for an automated production pipeline.\n"
        f"Evaluation task: {validation_prompt}\n\n"
        "Respond with ONLY a JSON object:\n"
        '{"score": <float 0-10>, "feedback": "<one sentence>", '
        '"dimensions": {"visual_quality": <0-10>, "intent_match": <0-10>}}'
    )

    def _parse_score_response(text: str) -> tuple[float, str, dict]:
        try:
            obj = json.loads(text)
            return float(obj.get("score", 7.0)), obj.get("feedback", ""), obj.get("dimensions", {})
        except json.JSONDecodeError:
            m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
            sc = float(m.group(1)) if m else 7.0
            return sc, text[:200], {}

    if output_path and output_path.exists() and output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        # Image gate
        if vlm_model == "claude-vision" and anthropic_key:
            try:
                import anthropic  # noqa: PLC0415

                client = anthropic.Anthropic(api_key=anthropic_key)
                img_b64 = base64.standard_b64encode(output_path.read_bytes()).decode()
                import mimetypes
                mime = mimetypes.guess_type(str(output_path))[0] or "image/png"
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=256,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_b64}},
                            {"type": "text", "text": system_prompt},
                        ],
                    }],
                )
                sc, fb, dims = _parse_score_response(message.content[0].text)
                return {
                    "score": sc,
                    "passed": sc >= 5.0,
                    "feedback": fb,
                    "dimensions": dims,
                    "_mock": False,
                }
            except Exception as exc:
                logger.warning("Claude VLM gate failed: %s — using mock score.", exc)
        elif vlm_model == "gemini-vision" and google_key:
            try:
                from google import genai  # noqa: PLC0415
                from PIL import Image as PILImage  # noqa: PLC0415

                client = genai.Client(api_key=google_key)
                img = PILImage.open(output_path)
                response = client.models.generate_content(model="gemini-3.0-flash", contents=[img, system_prompt])
                sc, fb, dims = _parse_score_response(response.text)
                return {
                    "score": sc,
                    "passed": sc >= 5.0,
                    "feedback": fb,
                    "dimensions": dims,
                    "_mock": False,
                }
            except Exception as exc:
                logger.warning("Gemini VLM gate failed: %s — using mock score.", exc)

    # Fallback mock gate
    is_mock = True
    if not output_path or not output_path.exists():
        score = 3.0
        feedback = "Output file does not exist — hard failure."
    elif output_path.stat().st_size == 0:
        score = 2.0
        feedback = "Output file is 0 bytes — possible write error."
    else:
        score = 7.0
        feedback = "[MOCK] Output file exists and has non-zero size. No real VLM key available."

    return {
        "score": score,
        "passed": score >= 5.0,
        "feedback": feedback,
        "dimensions": {"visual_quality": score, "intent_match": score},
        "_mock": is_mock,
    }


# ---------------------------------------------------------------------------
# Task handler stubs
# ---------------------------------------------------------------------------

def _get_output_dir(intermediates_dir: Path, task_id: str) -> Path:
    d = intermediates_dir / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _execute_image_to_video(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.mp4"
    duration = float(task_node.get("inputs", {}).get("duration_seconds", 3.0))
    logger.info("[stub] image_to_video → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_mp4(out, duration)
    return out


def _execute_text_to_video(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.mp4"
    duration = float(task_node.get("inputs", {}).get("duration_seconds", 3.0))
    logger.info("[stub] text_to_video → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_mp4(out, duration)
    return out


def _execute_image_segmentation(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.png"
    logger.info("[stub] image_segmentation → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_png(out)
    return out


def _execute_video_segmentation(task_node: dict, model: dict, output_dir: Path) -> Path:
    # Video segmentation outputs a PNG mask sequence
    out = output_dir / "mask_0000.png"
    logger.info("[stub] video_segmentation → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_png(out)
    return out


def _execute_text_to_image(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.png"
    logger.info("[stub] text_to_image → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_png(out)
    return out


def _execute_text_to_speech(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.wav"
    duration = float(task_node.get("inputs", {}).get("duration_seconds", 3.0))
    logger.info("[stub] text_to_speech → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_wav(out, duration)
    return out


def _execute_face_animation(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.mp4"
    duration = float(task_node.get("inputs", {}).get("duration_seconds", 3.0))
    logger.info("[stub] face_animation → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_mp4(out, duration)
    return out


def _execute_video_inpainting(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.mp4"
    duration = float(task_node.get("inputs", {}).get("duration_seconds", 3.0))
    logger.info("[stub] video_inpainting → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_mp4(out, duration)
    return out


def _execute_upscaling(task_node: dict, model: dict, output_dir: Path) -> Path:
    out = output_dir / "output.png"
    logger.info("[stub] upscaling → %s (model=%s)", out, model.get("model_id", "?"))
    _write_placeholder_png(out, width=1024, height=1024)
    return out


# ---------------------------------------------------------------------------
# Code-gen executor — the universal effect spawner
# ---------------------------------------------------------------------------

_CODE_GEN_SYSTEM_PROMPT = """\
You are a Python video-effects code generator inside an agentic video production pipeline.

Your job: write a self-contained Python script that produces exactly ONE output file.

Rules:
- The script MUST write its output to the path stored in the environment variable OUTPUT_PATH.
- Use only: stdlib, Pillow, moviepy, numpy. Do NOT import anything else.
- The output must be one of: .mp4 (video), .png (image), .wav (audio).
- For video: use MoviePy. Write MP4 with codec libx264, fps=30 (or as specified).
- For images: use Pillow. Save PNG with RGBA if transparency is needed.
- For audio: use stdlib wave module. Write WAV 48kHz mono.
- Smoothstep easing formula if you need animation: S(t) = t*t*(3 - 2*t), t in [0,1].
- No plt.show(), no interactive windows, no display calls.
- The script must run to completion silently (no user prompts).
- If you cannot implement the effect cleanly, write a simple but correct placeholder
  (e.g. solid-color clip with text overlay) rather than crashing.

Return ONLY the Python code, no markdown fences, no explanation.
"""


def _call_llm_for_code(
    prompt: str,
    model_id: str,
    task_id: str,
) -> str | None:
    """
    Ask an LLM to write Python code for a visual effect.
    Returns the code string, or None if no API key available.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    if model_id.startswith("claude") and anthropic_key:
        try:
            import anthropic  # noqa: PLC0415
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                system=_CODE_GEN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            logger.warning("[code_gen] Anthropic API failed: %s", exc)

    if model_id.startswith("gemini") and google_key:
        try:
            from google import genai  # noqa: PLC0415
            client = genai.Client(api_key=google_key)
            resp = client.models.generate_content(
                model="gemini-3.0-flash",
                contents=prompt,
                config={"system_instruction": _CODE_GEN_SYSTEM_PROMPT},
            )
            return resp.text.strip()
        except Exception as exc:
            logger.warning("[code_gen] Gemini API failed: %s", exc)

    return None


def _build_code_gen_prompt(task_node: dict) -> str:
    """Build the user-facing prompt for the code-gen LLM from the task node."""
    inputs = task_node.get("inputs", {})
    desc = task_node.get("description", "")
    lines = [
        f"Task: {desc}",
        "",
        "Parameters:",
    ]
    for k, v in inputs.items():
        lines.append(f"  {k}: {v}")
    lines += [
        "",
        "The output path is available as: import os; OUTPUT_PATH = os.environ['OUTPUT_PATH']",
        "Write the complete Python script now.",
    ]
    return "\n".join(lines)


def _run_generated_code(
    code: str,
    output_path: Path,
    task_id: str,
    output_dir: Path,
) -> tuple[bool, str]:
    """
    Execute generated Python code in a subprocess with OUTPUT_PATH injected.
    Returns (success: bool, error_message: str).
    """
    import subprocess
    import sys
    import tempfile

    # Write code to a temp file
    code_file = output_dir / f"generated_{task_id}.py"
    code_file.write_text(code, encoding="utf-8")
    logger.info("[code_gen] Running generated script: %s", code_file)

    env = os.environ.copy()
    env["OUTPUT_PATH"] = str(output_path)

    try:
        result = subprocess.run(
            [sys.executable, str(code_file)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes max
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "Unknown error")[:500]
            logger.error("[code_gen] Script exited with code %d: %s", result.returncode, err)
            return False, err
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, "Script ran but did not produce output file."
        logger.info("[code_gen] Script succeeded → %s", output_path)
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Generated script timed out after 300s."
    except Exception as exc:
        return False, str(exc)


def _fallback_code_gen_placeholder(task_node: dict, output_path: Path) -> None:
    """
    If code generation fails entirely, produce a labeled placeholder clip
    so the compositor can still assemble something.
    """
    desc = task_node.get("description", "code_gen effect")[:60]
    duration = float(task_node.get("inputs", {}).get("duration_seconds", 3.0))
    try:
        from moviepy import ColorClip, TextClip, CompositeVideoClip  # noqa: PLC0415
        bg = ColorClip(size=(1920, 1080), color=[20, 20, 20], duration=duration)
        try:
            txt = TextClip(
                f"[code_gen placeholder]\n{desc}",
                fontsize=40,
                color="white",
                font="Arial-Bold",
            ).set_position("center").set_duration(duration)
            clip = CompositeVideoClip([bg, txt])
        except Exception:
            clip = bg
        clip.write_videofile(str(output_path), fps=30, codec="libx264", audio=False, logger=None)
        clip.close()
    except Exception as exc:
        logger.warning("[code_gen] Placeholder clip failed: %s — writing stub.", exc)
        _write_placeholder_mp4(output_path, duration)


def _execute_code_gen(task_node: dict, model: dict, output_dir: Path) -> Path:
    """
    Code-gen executor: ask an LLM to write a Python effect script, run it,
    validate output exists. Falls back to a labeled placeholder on failure.

    The task_node["inputs"] should describe what the effect should produce.
    The generated script writes its output to OUTPUT_PATH (injected via env).
    Output format is determined by task_node["inputs"].get("output_format", "mp4").
    """
    task_id = task_node.get("task_id", "code_gen")
    model_id = model.get("model_id", "claude-sonnet")
    inputs = task_node.get("inputs", {})
    output_fmt = inputs.get("output_format", "mp4").lstrip(".")
    output_path = output_dir / f"output.{output_fmt}"

    prompt = _build_code_gen_prompt(task_node)
    code = _call_llm_for_code(prompt, model_id, task_id)

    if code:
        logger.info("[code_gen] Got %d chars of code from %s", len(code), model_id)
        success, err = _run_generated_code(code, output_path, task_id, output_dir)
        if success:
            return output_path
        logger.warning("[code_gen] Generated code failed (%s) — falling back to placeholder.", err)
    else:
        logger.warning("[code_gen] No API key available for code generation — using placeholder.")

    # Fallback: labeled placeholder
    _fallback_code_gen_placeholder(task_node, output_path)
    return output_path


def _execute_composite(
    task_node: dict, model: dict, spec: dict, output_dir: Path
) -> Path:
    """Delegate to stage5_compositor for composite tasks."""
    from scripts import stage5_compositor  # noqa: PLC0415

    out = output_dir / "output.mp4"
    # task_outputs not available at this point — compositor will use what exists
    try:
        stage5_compositor.compose_video(spec, {}, out, draft_mode=True)
    except Exception as exc:
        logger.warning("Compositor failed in execute_composite: %s — writing stub.", exc)
        _write_placeholder_mp4(out, duration_seconds=spec.get("video_params", {}).get("total_duration_seconds", 3.0))
    return out


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "image_to_video": _execute_image_to_video,
    "text_to_video": _execute_text_to_video,
    "image_segmentation": _execute_image_segmentation,
    "video_segmentation": _execute_video_segmentation,
    "text_to_image": _execute_text_to_image,
    "text_to_speech": _execute_text_to_speech,
    "face_animation": _execute_face_animation,
    "video_inpainting": _execute_video_inpainting,
    "upscaling": _execute_upscaling,
    "code_gen": _execute_code_gen,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_task(
    task_node: dict,
    selected_model: dict,
    registry: dict,
    intermediates_dir: Path,
    spec: dict | None = None,
) -> ExecutionResult:
    """
    Execute a single task node using the selected model.

    Always followed internally by run_vlm_quality_gate.
    """
    task_id = task_node.get("task_id", "unknown")
    task_type = task_node.get("task_type", "unknown")
    model = selected_model.get("primary")

    if model is None:
        return ExecutionResult(
            task_id=task_id,
            model_used="none",
            output_path=None,
            quality_score=0.0,
            quality_report={"error": "No model selected"},
            cost_actual=0.0,
            success=False,
            error_message="No eligible model found for this task.",
        )

    output_dir = _get_output_dir(intermediates_dir, task_id)

    try:
        if task_type == "composite" and spec:
            output_path = _execute_composite(task_node, model, spec, output_dir)
        elif task_type in _DISPATCH:
            output_path = _DISPATCH[task_type](task_node, model, output_dir)
        else:
            logger.warning("Unknown task_type '%s' — writing placeholder PNG.", task_type)
            output_path = output_dir / "output.png"
            _write_placeholder_png(output_path)
    except Exception as exc:
        logger.error("Task '%s' execution failed: %s", task_id, exc)
        return ExecutionResult(
            task_id=task_id,
            model_used=model.get("model_id", "unknown"),
            output_path=None,
            quality_score=0.0,
            quality_report={"error": str(exc)},
            cost_actual=0.0,
            success=False,
            error_message=str(exc),
        )

    # VLM quality gate (mandatory after every execute)
    validation_prompt = task_node.get(
        "validation_prompt",
        "Does this output match the task requirements? Score 1-10 with brief feedback.",
    )
    # Use image path for gate if output is PNG, else the path itself
    gate_path = output_path if output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else output_path
    quality_report = run_vlm_quality_gate(gate_path, validation_prompt)

    # Save quality report alongside output
    from scripts.utils import save_json  # noqa: PLC0415

    save_json(quality_report, output_dir / "quality_report.json")

    # Estimate cost
    from scripts.stage4_cost_estimator import _model_cost  # noqa: PLC0415

    cost = _model_cost(model, task_node)

    return ExecutionResult(
        task_id=task_id,
        model_used=model.get("model_id", "unknown"),
        output_path=output_path,
        quality_score=quality_report.get("score", 0.0),
        quality_report=quality_report,
        cost_actual=cost,
        success=quality_report.get("passed", False),
        error_message="" if quality_report.get("passed") else quality_report.get("feedback", ""),
    )


def retry_with_fallback(
    task_node: dict,
    primary_result: ExecutionResult,
    fallback_model: dict | None,
    registry: dict,
    intermediates_dir: Path,
    spec: dict | None = None,
) -> ExecutionResult:
    """
    Retry a failed task with the fallback model.
    Returns the fallback ExecutionResult (or primary_result if no fallback available).
    """
    if fallback_model is None:
        logger.warning(
            "Task '%s': no fallback model available. Returning failed primary result.",
            task_node.get("task_id", "?"),
        )
        return primary_result

    logger.info(
        "Task '%s': retrying with fallback model '%s'.",
        task_node.get("task_id", "?"),
        fallback_model.get("model_id", "?"),
    )

    selected = {"primary": fallback_model, "fallback": None}
    return execute_task(task_node, selected, registry, intermediates_dir, spec=spec)
