"""
Stage 5 — Compositor
MoviePy-based compositor. Assembles task outputs per spec.json into a final video.
"""

from __future__ import annotations

import logging
from pathlib import Path

from scripts.utils import smoothstep

logger = logging.getLogger(__name__)


def _require_moviepy():
    """Import moviepy or raise a helpful ImportError."""
    try:
        import moviepy as mpy  # noqa: PLC0415

        return mpy
    except ImportError as exc:
        raise ImportError(
            "moviepy is required for compositing. Install it with:\n"
            "  pip install moviepy>=2.0.0\n"
            "Also ensure ffmpeg is available in your PATH."
        ) from exc


def _interpolate_opacity(opacity_curve: list[dict], t: float) -> float:
    """
    Interpolate opacity at time t from an opacity_curve keyframe list.
    Keyframes: [{"t": float (0-1 normalized), "v": float (0-1)}]
    Uses smoothstep between keyframes.
    """
    if not opacity_curve:
        return 1.0
    if len(opacity_curve) == 1:
        return float(opacity_curve[0]["v"])

    # Find surrounding keyframes
    for i in range(len(opacity_curve) - 1):
        kf_a = opacity_curve[i]
        kf_b = opacity_curve[i + 1]
        ta = float(kf_a["t"])
        tb = float(kf_b["t"])
        if ta <= t <= tb:
            if tb == ta:
                return float(kf_b["v"])
            raw = (t - ta) / (tb - ta)
            s = smoothstep(raw)
            return float(kf_a["v"]) + s * (float(kf_b["v"]) - float(kf_a["v"]))

    # Clamp to last value
    if t <= float(opacity_curve[0]["t"]):
        return float(opacity_curve[0]["v"])
    return float(opacity_curve[-1]["v"])


def _make_position_func(start_pos: list, end_pos: list, start_t: float, end_t: float):
    """
    Return a function pos_at(t) that uses smoothstep easing for motion.
    NEVER uses linear interpolation.
    """
    def pos_at(t: float) -> tuple[float, float]:
        if t <= start_t:
            return (float(start_pos[0]), float(start_pos[1]))
        if t >= end_t:
            return (float(end_pos[0]), float(end_pos[1]))
        raw = (t - start_t) / (end_t - start_t)
        s = smoothstep(raw)
        x = start_pos[0] + s * (end_pos[0] - start_pos[0])
        y = start_pos[1] + s * (end_pos[1] - start_pos[1])
        return (x, y)
    return pos_at


def _resolve_asset_path(task_ref: str, task_outputs: dict, project_root: Path) -> Path | None:
    """
    Resolve layer task_ref to a file path.
    task_outputs maps task_id → Path.
    Also handles direct asset paths.
    """
    if task_ref in task_outputs:
        p = task_outputs[task_ref]
        return Path(p) if p else None

    # Try as a relative path from project root
    candidate = project_root / task_ref
    if candidate.exists():
        return candidate

    # Try just the filename in assets/
    candidate2 = project_root / "assets" / task_ref
    if candidate2.exists():
        return candidate2

    return None


def compose_video(
    spec: dict,
    task_outputs: dict,
    output_path: Path,
    draft_mode: bool = False,
) -> Path:
    """
    Assemble task outputs into the final video per spec.json.

    Parameters
    ----------
    spec : dict
        The spec.json timeline dict.
    task_outputs : dict
        Maps task_id → file Path of the model output.
    output_path : Path
        Destination for the rendered video.
    draft_mode : bool
        If True, render at 50% resolution with lower bitrate.

    Returns
    -------
    Path to the rendered output file.
    """
    mpy = _require_moviepy()

    from scripts.utils import get_project_root  # noqa: PLC0415

    project_root = get_project_root()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_params = spec.get("video_params", {})
    resolution = video_params.get("resolution", [1920, 1080])
    fps = int(video_params.get("fps", 30))
    total_duration = float(video_params.get("total_duration_seconds", 9.0))

    if draft_mode:
        render_w = resolution[0] // 2
        render_h = resolution[1] // 2
    else:
        render_w = resolution[0]
        render_h = resolution[1]

    scale_factor = render_w / resolution[0]  # 0.5 in draft, 1.0 otherwise

    # ── Background clip ───────────────────────────────────────────────────────
    bg_color = [0, 0, 0]  # default black; could parse from spec
    background = mpy.ColorClip(
        size=(render_w, render_h),
        color=bg_color,
        duration=total_duration,
    )
    clips = [background]

    # ── Process each layer ────────────────────────────────────────────────────
    timeline = sorted(spec.get("timeline", []), key=lambda l: l.get("layer_id", ""))
    audio_clips = []

    for layer in timeline:
        layer_id = layer.get("layer_id", "?")
        task_ref = layer.get("task_ref", "")
        asset_type = layer.get("asset_type", "image")
        start_time = float(layer.get("start_time", 0.0))
        end_time = float(layer.get("end_time", total_duration))
        clip_duration = end_time - start_time
        position = layer.get("position", {"x": render_w // 2, "y": render_h // 2})
        scale = float(layer.get("scale", 1.0)) * scale_factor
        opacity_curve = layer.get("opacity_curve", [{"t": 0.0, "v": 1.0}])
        motion_spec = layer.get("motion")
        text_spec = layer.get("text")

        clip = None

        # ── Text layers ───────────────────────────────────────────────────────
        if asset_type == "text" or text_spec:
            text_content = (text_spec or {}).get("content", layer.get("task_ref", ""))
            font_size = int((text_spec or {}).get("font_size", 60) * scale_factor)
            font = (text_spec or {}).get("font", "Arial-Bold")
            color = (text_spec or {}).get("color", "white")
            try:
                clip = mpy.TextClip(
                    text_content,
                    fontsize=font_size,
                    font=font,
                    color=color,
                ).set_duration(clip_duration)
            except Exception as exc:
                logger.warning("TextClip failed with font '%s': %s — trying DejaVu Sans Bold.", font, exc)
                try:
                    clip = mpy.TextClip(
                        text_content,
                        fontsize=font_size,
                        font="DejaVu-Sans-Bold",
                        color=color,
                    ).set_duration(clip_duration)
                except Exception as exc2:
                    logger.error("TextClip failed entirely: %s — skipping layer '%s'.", exc2, layer_id)
                    continue

        else:
            # ── File-based layers ─────────────────────────────────────────────
            asset_path = _resolve_asset_path(task_ref, task_outputs, project_root)
            if asset_path is None or not asset_path.exists():
                logger.warning("Layer '%s': asset not found for ref '%s' — skipping.", layer_id, task_ref)
                continue

            try:
                if asset_type == "audio":
                    audio_clip = mpy.AudioFileClip(str(asset_path))
                    audio_clip = audio_clip.set_start(start_time)
                    audio_clips.append(audio_clip)
                    continue
                elif asset_type == "video":
                    clip = mpy.VideoFileClip(str(asset_path), audio=False)
                    if draft_mode:
                        clip = clip.resize(scale)
                elif asset_type == "image":
                    clip = mpy.ImageClip(str(asset_path)).set_duration(clip_duration)
                    if draft_mode:
                        clip = clip.resize(scale)
                else:
                    logger.warning("Unknown asset_type '%s' for layer '%s' — skipping.", asset_type, layer_id)
                    continue
            except Exception as exc:
                logger.error("Failed to load '%s' for layer '%s': %s — skipping.", asset_path, layer_id, exc)
                continue

        if clip is None:
            continue

        # ── Trim to layer duration ─────────────────────────────────────────────
        if clip.duration and clip.duration > clip_duration:
            clip = clip.subclip(0, clip_duration)
        elif clip.duration and clip.duration < clip_duration:
            clip = clip.loop(duration=clip_duration)

        # ── Opacity / fade (via fl_image + opacity_curve) ─────────────────────
        def _make_opacity_filter(curve, dur):
            def filter_fn(img, t):
                import numpy as np  # noqa: PLC0415

                normalized_t = t / dur if dur > 0 else 0.0
                alpha = _interpolate_opacity(curve, normalized_t)
                result = img.astype(float)
                result[..., :3] = result[..., :3] * alpha
                if result.shape[-1] == 4:
                    result[..., 3] = result[..., 3] * alpha
                return result.astype("uint8")
            return filter_fn

        try:
            clip = clip.fl_image(
                lambda img, t=0, curve=opacity_curve, dur=clip_duration: _make_opacity_filter(curve, dur)(img, t),
            )
        except Exception:
            pass  # opacity is non-critical; skip if it fails

        # ── Position (static or animated) ─────────────────────────────────────
        if motion_spec and motion_spec.get("easing") == "smoothstep":
            start_pos = motion_spec.get("start_pos", [0, 0])
            end_pos = motion_spec.get("end_pos", [0, 0])
            motion_start = float(motion_spec.get("start_time", start_time))
            motion_end = float(motion_spec.get("end_time", end_time))
            # Adjust to clip-local time
            pos_fn = _make_position_func(
                [p * scale_factor for p in start_pos],
                [p * scale_factor for p in end_pos],
                motion_start - start_time,
                motion_end - start_time,
            )
            clip = clip.set_position(pos_fn)
        else:
            px = int(position.get("x", render_w // 2) * scale_factor)
            py = int(position.get("y", render_h // 2) * scale_factor)
            # Center anchor
            try:
                w, h = clip.size
                clip = clip.set_position((px - w // 2, py - h // 2))
            except Exception:
                clip = clip.set_position((px, py))

        # ── Timing ────────────────────────────────────────────────────────────
        clip = clip.set_start(start_time)
        clips.append(clip)

    # ── Composite ─────────────────────────────────────────────────────────────
    final = mpy.CompositeVideoClip(clips, size=(render_w, render_h))

    # Attach audio if any
    if audio_clips:
        try:
            combined_audio = mpy.CompositeAudioClip(audio_clips)
            final = final.set_audio(combined_audio)
        except Exception as exc:
            logger.warning("Audio compositing failed: %s — output will be silent.", exc)

    # ── Write ──────────────────────────────────────────────────────────────────
    write_kwargs: dict = {
        "fps": fps,
        "codec": "libx264",
        "audio_codec": "aac",
        "logger": None,
    }
    if draft_mode:
        write_kwargs["ffmpeg_params"] = ["-crf", "32"]
    else:
        write_kwargs["ffmpeg_params"] = ["-crf", "23"]

    try:
        final.write_videofile(str(output_path), **write_kwargs)
    except Exception as exc:
        logger.error("write_videofile failed: %s", exc)
        raise

    logger.info("Compositor wrote: %s (draft=%s)", output_path, draft_mode)
    return output_path
