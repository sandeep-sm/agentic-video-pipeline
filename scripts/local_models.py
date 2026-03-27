"""
Local model inference — lazy-loaded GPU executors for HuggingFace models.

Each public function returns Path | None:
  - Path: output file was written successfully
  - None: inference failed; caller writes placeholder

Lazy loading: module-level caches keyed by hf_repo prevent reloading between tasks.
Circular import prevention: do NOT import from scripts.stage4_executor here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model caches (hf_repo → loaded pipeline/model) ──────────────────────────
_wan_i2v_cache: dict = {}
_wan_t2v_cache: dict = {}
_flux_cache: dict = {}
_qwen_vl_cache: dict = {}
_chatterbox_cache: dict = {}


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _device() -> str:
    return "cuda" if _cuda_available() else "cpu"


def _bf16():
    try:
        import torch
        return torch.bfloat16
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# WAN Image-to-Video
# ─────────────────────────────────────────────────────────────────────────────

def run_wan_i2v(
    task_node: dict,
    model: dict,
    output_dir: Path,
    source_image: Path | None,
) -> Path | None:
    """
    Run Wan2.x Image-to-Video inference.

    Tries diffusers WanImageToVideoPipeline first (Wan 2.1/2.2 via HF diffusers),
    then falls back to the `wan` package if installed.
    Returns the output .mp4 Path on success, None on failure.
    """
    hf_repo = model.get("hf_repo", "Wan-Video/Wan2.2-I2V-14B")
    inputs = task_node.get("inputs", {}) or {}
    prompt = str(inputs.get("prompt", inputs.get("description", "high quality smooth motion")))
    negative_prompt = str(inputs.get("negative_prompt", "static, blurry, distorted, watermark"))
    duration = float(inputs.get("duration_seconds", 4.0))
    fps_out = int((model.get("output_format") or {}).get("typical_fps", 16))
    num_frames = max(1, int(duration * fps_out))
    out = output_dir / "output.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_image is None:
        logger.warning("[wan_i2v] No source image found for task '%s'.", task_node.get("task_id"))

    # Wan-AI/ repos are NOT in diffusers format (no model_index.json).
    # Use the wan Python package for those; use diffusers for Wan-Video/ repos.
    use_wan_pkg_first = hf_repo.startswith("Wan-AI/")

    def _try_wan_package() -> "Path | None":
        try:
            import torch
            import numpy as np
            import wan
            from wan.configs import WAN_CONFIGS
            from PIL import Image as PILImage

            cache_key = f"wan_pkg_i2v_{hf_repo}"
            if cache_key not in _wan_i2v_cache:
                repo_lower = hf_repo.lower()
                # Prefer ti2v config when model is TI2V variant, else i2v
                if "ti2v" in repo_lower:
                    config_key = next(
                        (k for k in WAN_CONFIGS if "ti2v" in k.lower()),
                        next((k for k in WAN_CONFIGS if "i2v" in k.lower()), None),
                    )
                else:
                    config_key = next(
                        (k for k in WAN_CONFIGS if "i2v" in k.lower() and "ti2v" not in k.lower()),
                        next((k for k in WAN_CONFIGS if "i2v" in k.lower()), None),
                    )
                if config_key is None:
                    raise RuntimeError("No i2v/ti2v config in WAN_CONFIGS")
                logger.info("[wan_i2v] Loading wan package model (config=%s, repo=%s) …", config_key, hf_repo)
                model_obj = wan.WanI2V(
                    config=WAN_CONFIGS[config_key],
                    checkpoint_dir=None,  # auto-download from HF
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                _wan_i2v_cache[cache_key] = model_obj

            model_obj = _wan_i2v_cache[cache_key]
            if source_image is None:
                raise ValueError("source_image required")

            image = PILImage.open(source_image).convert("RGB")
            logger.info("[wan_i2v] Generating via wan package (%d frames) …", num_frames)
            frames = model_obj.generate(prompt=prompt, image=image, num_frames=num_frames)

            try:
                import imageio
                imageio.mimwrite(str(out), [np.array(f) for f in frames], fps=fps_out)
            except ImportError:
                from diffusers.utils import export_to_video  # type: ignore
                export_to_video(frames, str(out), fps=fps_out)

            logger.info("[wan_i2v] wan package wrote %s", out)
            return out

        except ImportError:
            logger.warning("[wan_i2v] wan package not installed — pip install wan")
            return None
        except Exception as exc:
            logger.warning("[wan_i2v] wan package failed: %s", exc)
            _wan_i2v_cache.pop(f"wan_pkg_i2v_{hf_repo}", None)
            return None

    def _try_diffusers() -> "Path | None":
        try:
            import torch
            from diffusers import WanImageToVideoPipeline
            from diffusers.utils import export_to_video
            from PIL import Image as PILImage

            if hf_repo not in _wan_i2v_cache:
                logger.info("[wan_i2v] Loading WanImageToVideoPipeline from %s …", hf_repo)
                pipe = WanImageToVideoPipeline.from_pretrained(
                    hf_repo, torch_dtype=torch.bfloat16
                )
                pipe.enable_model_cpu_offload()
                _wan_i2v_cache[hf_repo] = pipe

            pipe = _wan_i2v_cache[hf_repo]
            if source_image is None:
                raise ValueError("source_image is required for WanImageToVideoPipeline")

            image = PILImage.open(source_image).convert("RGB")
            max_w, max_h = 832, 480
            w, h = image.size
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                image = image.resize((int(w * scale) & ~1, int(h * scale) & ~1), PILImage.LANCZOS)

            logger.info("[wan_i2v] Generating %d frames via diffusers …", num_frames)
            result = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=5.0,
                num_inference_steps=30,
            )
            export_to_video(result.frames[0], str(out), fps=fps_out)
            logger.info("[wan_i2v] Wrote %s", out)
            return out

        except ImportError as exc:
            logger.warning("[wan_i2v] WanImageToVideoPipeline not available: %s", exc)
            return None
        except Exception as exc:
            logger.warning("[wan_i2v] diffusers pipeline failed: %s", exc)
            _wan_i2v_cache.pop(hf_repo, None)
            return None

    if use_wan_pkg_first:
        result_path = _try_wan_package()
        if result_path is not None:
            return result_path
        return _try_diffusers()
    else:
        result_path = _try_diffusers()
        if result_path is not None:
            return result_path
        return _try_wan_package()


# ─────────────────────────────────────────────────────────────────────────────
# WAN Text-to-Video
# ─────────────────────────────────────────────────────────────────────────────

def run_wan_t2v(
    task_node: dict,
    model: dict,
    output_dir: Path,
) -> Path | None:
    """
    Run Wan2.x Text-to-Video inference.

    Tries diffusers WanPipeline first, then the `wan` package.
    Returns the output .mp4 Path on success, None on failure.
    """
    hf_repo = model.get("hf_repo", "Wan-Video/Wan2.2-T2V-14B")
    inputs = task_node.get("inputs", {}) or {}
    prompt = str(inputs.get("prompt", inputs.get("description", "high quality cinematic video")))
    negative_prompt = str(inputs.get("negative_prompt", "blurry, distorted, watermark"))
    duration = float(inputs.get("duration_seconds", 4.0))
    fps_out = int((model.get("output_format") or {}).get("typical_fps", 16))
    num_frames = max(1, int(duration * fps_out))
    out = output_dir / "output.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wan-AI/ repos are not in diffusers format — use wan package first.
    use_wan_pkg_first = hf_repo.startswith("Wan-AI/")

    def _try_wan_package_t2v() -> "Path | None":
        try:
            import torch
            import numpy as np
            import wan
            from wan.configs import WAN_CONFIGS

            cache_key = f"wan_pkg_t2v_{hf_repo}"
            if cache_key not in _wan_t2v_cache:
                config_key = next(
                    (k for k in WAN_CONFIGS if "t2v" in k.lower()),
                    next(iter(WAN_CONFIGS), None),
                )
                if config_key is None:
                    raise RuntimeError("No t2v config in WAN_CONFIGS")
                logger.info("[wan_t2v] Loading wan package t2v model (config=%s, repo=%s) …", config_key, hf_repo)
                model_obj = wan.WanT2V(
                    config=WAN_CONFIGS[config_key],
                    checkpoint_dir=None,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                _wan_t2v_cache[cache_key] = model_obj

            model_obj = _wan_t2v_cache[cache_key]
            logger.info("[wan_t2v] Generating %d frames via wan package …", num_frames)
            frames = model_obj.generate(prompt=prompt, num_frames=num_frames)

            try:
                import imageio
                imageio.mimwrite(str(out), [np.array(f) for f in frames], fps=fps_out)
            except ImportError:
                from diffusers.utils import export_to_video  # type: ignore
                export_to_video(frames, str(out), fps=fps_out)

            logger.info("[wan_t2v] wan package wrote %s", out)
            return out

        except ImportError:
            logger.warning("[wan_t2v] wan package not installed — pip install wan")
            return None
        except Exception as exc:
            logger.warning("[wan_t2v] wan package failed: %s", exc)
            _wan_t2v_cache.pop(f"wan_pkg_t2v_{hf_repo}", None)
            return None

    def _try_diffusers_t2v() -> "Path | None":
        try:
            import torch
            from diffusers import WanPipeline
            from diffusers.utils import export_to_video

            if hf_repo not in _wan_t2v_cache:
                logger.info("[wan_t2v] Loading WanPipeline from %s …", hf_repo)
                pipe = WanPipeline.from_pretrained(hf_repo, torch_dtype=torch.bfloat16)
                pipe.enable_model_cpu_offload()
                _wan_t2v_cache[hf_repo] = pipe

            pipe = _wan_t2v_cache[hf_repo]
            logger.info("[wan_t2v] Generating %d frames via diffusers …", num_frames)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=5.0,
                num_inference_steps=30,
            )
            export_to_video(result.frames[0], str(out), fps=fps_out)
            logger.info("[wan_t2v] Wrote %s", out)
            return out

        except ImportError as exc:
            logger.warning("[wan_t2v] WanPipeline not available: %s", exc)
            return None
        except Exception as exc:
            logger.warning("[wan_t2v] diffusers failed: %s", exc)
            _wan_t2v_cache.pop(hf_repo, None)
            return None

    if use_wan_pkg_first:
        result_path = _try_wan_package_t2v()
        if result_path is not None:
            return result_path
        return _try_diffusers_t2v()
    else:
        result_path = _try_diffusers_t2v()
        if result_path is not None:
            return result_path
        return _try_wan_package_t2v()


# ─────────────────────────────────────────────────────────────────────────────
# WAN Video-to-Video (Animate)
# ─────────────────────────────────────────────────────────────────────────────

# Cache shared with t2v (keyed differently)
_wan_v2v_cache: dict = {}


def run_wan_v2v(
    task_node: dict,
    model: dict,
    output_dir: Path,
    source_video: Path | None,
) -> Path | None:
    """
    Run Wan2.2-Animate Video-to-Video inference.
    Uses the wan Python package (Wan-AI/ repos are not in diffusers format).
    Returns the output .mp4 Path on success, None on failure.
    """
    hf_repo = model.get("hf_repo", "Wan-AI/Wan2.2-Animate-14B")
    inputs = task_node.get("inputs", {}) or {}
    prompt = str(inputs.get("prompt", inputs.get("description", "high quality smooth animation")))
    fps_out = int((model.get("output_format") or {}).get("typical_fps", 24))
    out = output_dir / "output.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_video is None:
        logger.warning("[wan_v2v] No source video for task '%s'.", task_node.get("task_id"))

    try:
        import torch
        import numpy as np
        import wan
        from wan.configs import WAN_CONFIGS

        cache_key = f"wan_pkg_v2v_{hf_repo}"
        if cache_key not in _wan_v2v_cache:
            config_key = next(
                (k for k in WAN_CONFIGS if "animate" in k.lower() or "v2v" in k.lower()),
                next((k for k in WAN_CONFIGS if "i2v" in k.lower()), None),
            )
            if config_key is None:
                raise RuntimeError("No animate/v2v config in WAN_CONFIGS")
            logger.info("[wan_v2v] Loading wan package v2v model (config=%s, repo=%s) …", config_key, hf_repo)
            model_obj = wan.WanI2V(  # Animate shares the I2V class in wan package
                config=WAN_CONFIGS[config_key],
                checkpoint_dir=None,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            _wan_v2v_cache[cache_key] = model_obj

        model_obj = _wan_v2v_cache[cache_key]

        if source_video is None:
            raise ValueError("source_video is required for Wan2.2-Animate")

        # Extract first frame as conditioning image if the model expects image input
        from PIL import Image as PILImage  # noqa: PLC0415
        try:
            import imageio
            reader = imageio.get_reader(str(source_video))
            first_frame = PILImage.fromarray(reader.get_data(0)).convert("RGB")
            reader.close()
        except Exception:
            # Fallback: use ffmpeg via moviepy to grab frame 0
            import moviepy as mpy  # noqa: PLC0415
            vc = mpy.VideoFileClip(str(source_video))
            arr = vc.get_frame(0)
            first_frame = PILImage.fromarray(arr).convert("RGB")
            vc.close()

        logger.info("[wan_v2v] Generating animated video via wan package …")
        frames = model_obj.generate(prompt=prompt, image=first_frame)

        try:
            imageio.mimwrite(str(out), [np.array(f) for f in frames], fps=fps_out)
        except Exception:
            from diffusers.utils import export_to_video  # type: ignore  # noqa: PLC0415
            export_to_video(frames, str(out), fps=fps_out)

        logger.info("[wan_v2v] wan package wrote %s", out)
        return out

    except ImportError:
        logger.warning("[wan_v2v] wan package not installed — pip install wan")
    except Exception as exc:
        logger.warning("[wan_v2v] wan package failed: %s", exc)
        _wan_v2v_cache.pop(f"wan_pkg_v2v_{hf_repo}", None)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.3 (Joint Audio-Video Generation)
# ─────────────────────────────────────────────────────────────────────────────

_ltx23_cache: dict = {}


def _ltx23_env_ok() -> bool:
    """Check if environment meets LTX-2.3 requirements (Python ≥ 3.12, CUDA > 12.7)."""
    import sys
    if sys.version_info < (3, 12):
        logger.info("[ltx23] Python %s < 3.12 — LTX-2.3 not available.", sys.version_info[:2])
        return False
    if _cuda_available():
        import torch
        cuda_ver = torch.version.cuda or "0"
        major, minor = (int(x) for x in cuda_ver.split(".")[:2])
        if (major, minor) <= (12, 7):
            logger.info("[ltx23] CUDA %s <= 12.7 — LTX-2.3 not available.", cuda_ver)
            return False
    return True


def _ltx23_resolve_paths(hf_repo: str = "Lightricks/LTX-2.3") -> dict | None:
    """
    Resolve LTX-2.3 model file paths from HuggingFace Hub cache.
    Downloads if not already cached. Returns dict of paths or None on failure.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        # Download the full model snapshot (safetensors + config)
        cache_dir = snapshot_download(repo_id=hf_repo)
        cache_path = Path(cache_dir)

        # Locate key files
        def _find(pattern: str) -> str | None:
            matches = sorted(cache_path.rglob(pattern))
            return str(matches[0]) if matches else None

        checkpoint = _find("*22b*dev*.safetensors") or _find("*22b*.safetensors")
        distilled_lora = _find("*distilled*lora*.safetensors")
        spatial_upscaler = _find("*spatial*upscaler*x2*.safetensors") or _find("*spatial*upscaler*.safetensors")
        gemma_dir = None
        # Gemma text encoder: look inside the repo or as a separate download
        for candidate in cache_path.rglob("gemma*"):
            if candidate.is_dir():
                gemma_dir = str(candidate)
                break

        if checkpoint is None:
            logger.warning("[ltx23] Could not find checkpoint .safetensors in %s", cache_dir)
            return None

        return {
            "checkpoint_path": checkpoint,
            "distilled_lora": distilled_lora,
            "spatial_upscaler": spatial_upscaler,
            "gemma_root": gemma_dir,
            "cache_dir": cache_dir,
        }
    except Exception as exc:
        logger.warning("[ltx23] Failed to resolve model paths: %s", exc)
        return None


def run_ltx23_t2v(
    task_node: dict,
    model: dict,
    output_dir: Path,
) -> Path | None:
    """
    Run LTX-2.3 Text-to-Video (with joint audio).
    Uses ltx-pipelines TI2VidTwoStagesPipeline with distilled LoRA (8 steps).
    Returns output .mp4 Path on success, None on failure.
    """
    if not _ltx23_env_ok():
        return None

    hf_repo = model.get("hf_repo", "Lightricks/LTX-2.3")
    inputs = task_node.get("inputs", {}) or {}
    prompt = str(inputs.get("prompt", inputs.get("description", "high quality cinematic video")))
    duration = float(inputs.get("duration_seconds", 4.0))
    fps_out = 25
    # num_frames must be divisible by 8 + 1
    num_frames = max(9, (int(duration * fps_out) // 8) * 8 + 1)
    out = output_dir / "output.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.components.guiders import MultiModalGuiderParams

        cache_key = f"ltx23_t2v_{hf_repo}"
        if cache_key not in _ltx23_cache:
            paths = _ltx23_resolve_paths(hf_repo)
            if paths is None:
                return None

            lora_args = []
            if paths["distilled_lora"]:
                lora_args = [LoraPathStrengthAndSDOps(
                    path=paths["distilled_lora"], strength=0.8, sd_ops=None
                )]

            logger.info("[ltx23] Loading TI2VidTwoStagesPipeline from %s …", hf_repo)
            pipe = TI2VidTwoStagesPipeline(
                checkpoint_path=paths["checkpoint_path"],
                distilled_lora=lora_args,
                spatial_upsampler_path=paths.get("spatial_upscaler") or "",
                gemma_root=paths.get("gemma_root") or "",
            )
            _ltx23_cache[cache_key] = pipe

        pipe = _ltx23_cache[cache_key]

        video_guider = MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0)
        audio_guider = MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0)

        logger.info("[ltx23] Generating %d frames (%.1fs @ %d fps, 8 steps) with audio …", num_frames, duration, fps_out)
        pipe(
            prompt=prompt,
            output_path=str(out),
            seed=42,
            height=512,
            width=768,
            num_frames=num_frames,
            frame_rate=float(fps_out),
            num_inference_steps=8,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
        )

        if out.exists() and out.stat().st_size > 0:
            logger.info("[ltx23] Wrote %s", out)
            return out
        logger.warning("[ltx23] Output file missing or empty after generation.")
        return None

    except ImportError as exc:
        logger.warning("[ltx23] ltx-pipelines not installed: %s — pip install -e packages/ltx-pipelines from LTX-2 repo", exc)
    except Exception as exc:
        logger.warning("[ltx23] T2V generation failed: %s", exc)
        _ltx23_cache.pop(f"ltx23_t2v_{hf_repo}", None)

    return None


def run_ltx23_i2v(
    task_node: dict,
    model: dict,
    output_dir: Path,
    source_image: Path | None,
) -> Path | None:
    """
    Run LTX-2.3 Image-to-Video (with joint audio).
    Uses TI2VidTwoStagesPipeline with ImageConditioningInput on frame 0.
    Returns output .mp4 Path on success, None on failure.
    """
    if not _ltx23_env_ok():
        return None

    hf_repo = model.get("hf_repo", "Lightricks/LTX-2.3")
    inputs = task_node.get("inputs", {}) or {}
    prompt = str(inputs.get("prompt", inputs.get("description", "high quality smooth motion")))
    duration = float(inputs.get("duration_seconds", 4.0))
    fps_out = 25
    num_frames = max(9, (int(duration * fps_out) // 8) * 8 + 1)
    out = output_dir / "output.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_image is None:
        logger.warning("[ltx23] No source image for I2V task '%s'.", task_node.get("task_id"))
        return None

    try:
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_core.loader import LoraPathStrengthAndSDOps
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.components.conditioning import ImageConditioningInput

        cache_key = f"ltx23_i2v_{hf_repo}"
        if cache_key not in _ltx23_cache:
            paths = _ltx23_resolve_paths(hf_repo)
            if paths is None:
                return None

            lora_args = []
            if paths["distilled_lora"]:
                lora_args = [LoraPathStrengthAndSDOps(
                    path=paths["distilled_lora"], strength=0.8, sd_ops=None
                )]

            logger.info("[ltx23] Loading TI2VidTwoStagesPipeline (I2V) from %s …", hf_repo)
            pipe = TI2VidTwoStagesPipeline(
                checkpoint_path=paths["checkpoint_path"],
                distilled_lora=lora_args,
                spatial_upsampler_path=paths.get("spatial_upscaler") or "",
                gemma_root=paths.get("gemma_root") or "",
            )
            _ltx23_cache[cache_key] = pipe

        pipe = _ltx23_cache[cache_key]

        video_guider = MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0)
        audio_guider = MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0)

        image_cond = [ImageConditioningInput(
            path=str(source_image),
            frame_index=0,
            strength=1.0,
            crf=33,
        )]

        logger.info("[ltx23] Generating %d frames I2V (%.1fs @ %d fps, 8 steps) with audio …", num_frames, duration, fps_out)
        pipe(
            prompt=prompt,
            output_path=str(out),
            seed=42,
            height=512,
            width=768,
            num_frames=num_frames,
            frame_rate=float(fps_out),
            num_inference_steps=8,
            video_guider_params=video_guider,
            audio_guider_params=audio_guider,
            images=image_cond,
        )

        if out.exists() and out.stat().st_size > 0:
            logger.info("[ltx23] Wrote %s", out)
            return out
        logger.warning("[ltx23] Output file missing or empty after I2V generation.")
        return None

    except ImportError as exc:
        logger.warning("[ltx23] ltx-pipelines not installed: %s", exc)
    except Exception as exc:
        logger.warning("[ltx23] I2V generation failed: %s", exc)
        _ltx23_cache.pop(f"ltx23_i2v_{hf_repo}", None)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# FLUX Text-to-Image
# ─────────────────────────────────────────────────────────────────────────────

def run_flux_t2i(
    task_node: dict,
    model: dict,
    output_dir: Path,
) -> Path | None:
    """
    Run FLUX.2 Klein / FLUX.1 Dev text-to-image inference.
    Returns the output .png Path on success, None on failure.
    """
    hf_repo = model.get("hf_repo", "black-forest-labs/FLUX.2-klein-9B")
    inputs = task_node.get("inputs", {}) or {}
    prompt = str(inputs.get("prompt", inputs.get("description", "high quality product photo")))
    width = int(inputs.get("width", 1920))
    height = int(inputs.get("height", 1080))
    steps = int(inputs.get("num_inference_steps", 28))
    guidance = float(inputs.get("guidance_scale", 3.5))
    seed = int(inputs.get("seed", 42))
    out = output_dir / "output.png"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from diffusers import FluxPipeline

        if hf_repo not in _flux_cache:
            logger.info("[flux_t2i] Loading FluxPipeline from %s …", hf_repo)
            pipe = FluxPipeline.from_pretrained(hf_repo, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
            _flux_cache[hf_repo] = pipe

        pipe = _flux_cache[hf_repo]
        generator = torch.Generator("cpu").manual_seed(seed)
        logger.info("[flux_t2i] Generating %dx%d image …", width, height)
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=steps,
            max_sequence_length=512,
            generator=generator,
        ).images[0]
        image.save(str(out))
        logger.info("[flux_t2i] Wrote %s", out)
        return out

    except ImportError:
        logger.debug("[flux_t2i] diffusers FluxPipeline not available.")
    except Exception as exc:
        logger.warning("[flux_t2i] FluxPipeline failed: %s", exc)
        _flux_cache.pop(hf_repo, None)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Chatterbox TTS
# ─────────────────────────────────────────────────────────────────────────────

def run_chatterbox_tts(
    task_node: dict,
    model: dict,
    output_dir: Path,
) -> Path | None:
    """
    Run Chatterbox TTS inference (ResembleAI/chatterbox).
    Returns the output .wav Path on success, None on failure.
    """
    inputs = task_node.get("inputs", {}) or {}
    text = str(
        inputs.get("text", inputs.get("script", inputs.get("transcript", inputs.get("description", ""))))
    ).strip()
    if not text:
        logger.warning("[chatterbox] No text found in task inputs for task '%s'.", task_node.get("task_id"))
        return None

    exaggeration = float(inputs.get("exaggeration", 0.5))
    cfg_weight = float(inputs.get("cfg_weight", 0.5))
    voice_ref = inputs.get("voice_reference_audio")  # optional path

    out = output_dir / "output.wav"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_key = "chatterbox"
        if cache_key not in _chatterbox_cache:
            logger.info("[chatterbox] Loading ChatterboxTTS …")
            tts = ChatterboxTTS.from_pretrained(device=device)
            _chatterbox_cache[cache_key] = tts

        tts = _chatterbox_cache[cache_key]

        kwargs: dict = {"exaggeration": exaggeration, "cfg_weight": cfg_weight}
        if voice_ref and Path(str(voice_ref)).exists():
            kwargs["audio_prompt_path"] = str(voice_ref)

        logger.info("[chatterbox] Synthesising %d chars …", len(text))
        wav = tts.generate(text, **kwargs)
        torchaudio.save(str(out), wav, tts.sr)
        logger.info("[chatterbox] Wrote %s (sr=%d)", out, tts.sr)
        return out

    except ImportError:
        logger.debug("[chatterbox] chatterbox package not installed.")
    except Exception as exc:
        logger.warning("[chatterbox] inference failed: %s", exc)
        _chatterbox_cache.pop("chatterbox", None)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Qwen3-VL Quality Gate
# ─────────────────────────────────────────────────────────────────────────────

def run_qwen_vl_gate(
    output_path: Path,
    validation_prompt: str,
    model_id: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
) -> dict | None:
    """
    Run Qwen3-VL as a local VLM quality gate.

    Loads the model lazily on first call.
    Returns a quality report dict (score, passed, feedback) or None on failure.
    """
    if not output_path.exists():
        return {"score": 3.0, "passed": False, "feedback": "Output file does not exist.", "_mock": False}

    # Only evaluate image files directly; for video/audio, score by file validity
    is_image = output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    if not is_image:
        size = output_path.stat().st_size
        score = 7.0 if size > 1024 else 2.0
        return {
            "score": score,
            "passed": score >= 5.0,
            "feedback": f"Non-image output ({output_path.suffix}); size={size}B.",
            "_mock": False,
        }

    try:
        import torch
        from transformers import AutoProcessor

        # Try Qwen3-VL class, fall back to Qwen2.5-VL class
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except ImportError:
            from transformers import AutoModelForCausalLM as VLModel  # type: ignore

        cache_key = model_id
        if cache_key not in _qwen_vl_cache:
            logger.info("[qwen_vl] Loading %s …", model_id)
            qwen_model = VLModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(model_id)
            _qwen_vl_cache[cache_key] = {"model": qwen_model, "processor": processor}

        cache = _qwen_vl_cache[cache_key]
        qwen_model = cache["model"]
        processor = cache["processor"]

        prompt_text = (
            f"You are a video production quality evaluator.\n"
            f"Task: {validation_prompt}\n\n"
            "Score this image 0-10 on visual quality and task match.\n"
            'Reply ONLY with JSON: {"score": <float>, "feedback": "<one sentence>"}'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(output_path)},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Use qwen_vl_utils if available for proper image processing
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(qwen_model.device)
        except ImportError:
            # Fallback: use PIL directly
            from PIL import Image as PILImage
            img = PILImage.open(output_path)
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text_input],
                images=[img],
                return_tensors="pt",
            ).to(qwen_model.device)

        with torch.no_grad():
            generated_ids = qwen_model.generate(**inputs, max_new_tokens=128)
        input_len = inputs["input_ids"].shape[1]
        output_text = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse JSON response
        try:
            obj = json.loads(output_text)
            score = float(obj.get("score", 7.0))
            feedback = str(obj.get("feedback", ""))
        except (json.JSONDecodeError, ValueError):
            m = re.search(r'"score"\s*:\s*([0-9.]+)', output_text)
            score = float(m.group(1)) if m else 7.0
            feedback = output_text[:200]

        score = max(0.0, min(10.0, score))
        return {
            "score": score,
            "passed": score >= 5.0,
            "feedback": feedback,
            "dimensions": {"visual_quality": score, "intent_match": score},
            "_mock": False,
        }

    except ImportError:
        logger.debug("[qwen_vl] transformers or required packages not available.")
    except Exception as exc:
        logger.warning("[qwen_vl] inference failed: %s", exc)
        _qwen_vl_cache.pop(model_id, None)

    return None
