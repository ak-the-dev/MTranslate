"""Inpaint backend selection and adapters."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
from dataclasses import dataclass
from typing import Dict, List

from .env import env_float, env_int
from .types import InpaintMask
from .utils import copy_file


class InpaintBackend:
    name = "base"

    def warmup(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def inpaint(self, src: Path, masks: List[InpaintMask], dst: Path) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


def _default_inpaint_model_path() -> str:
    for name in ("MTRANSLATE_INFILL_MODEL", "MTRANSLATE_INPAINT_MODEL"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _default_inpaint_command() -> str:
    for name in ("MTRANSLATE_INFILL_COMMAND", "MTRANSLATE_INPAINT_COMMAND"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


class DiffusionInpaintBackend(InpaintBackend):
    name = "diffusion"
    _pipe_cache: Dict[tuple[str, str, str, str], object] = {}

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _default_inpaint_model_path()
        self.prompt = os.getenv(
            "MTRANSLATE_INPAINT_PROMPT",
            "clean manga paper texture, no text, preserve original artwork and line art",
        )
        self.negative_prompt = os.getenv(
            "MTRANSLATE_INPAINT_NEGATIVE_PROMPT",
            "letters, words, watermark, blurry, distorted anatomy, artifacts",
        )
        self.steps = env_int("MTRANSLATE_INPAINT_STEPS", 28, min_value=4, max_value=200)
        self.guidance = env_float("MTRANSLATE_INPAINT_GUIDANCE", 6.0, min_value=0.0, max_value=20.0)
        self.strength = env_float("MTRANSLATE_INPAINT_STRENGTH", 0.95, min_value=0.0, max_value=1.0)
        self.device_pref = os.getenv("MTRANSLATE_INPAINT_DEVICE", "auto").strip().lower()
        self._pipeline_mode = "inpaint"

    def warmup(self) -> None:
        if not self.model_path:
            raise RuntimeError(
                "Diffusion inpaint backend requested but MTRANSLATE_INPAINT_MODEL is not set"
            )
        path = Path(self.model_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"Diffusion inpaint model path does not exist: {path}")
        _ = self._pipeline()

    def _resolve_device_and_dtype(self):
        import torch  # type: ignore

        pref = self.device_pref
        if pref in {"", "auto"}:
            if torch.backends.mps.is_available():
                # SDXL fp16 can produce black frames on MPS; float32 is safer.
                return "mps", torch.float32
            if torch.cuda.is_available():
                return "cuda", torch.float16
            return "cpu", torch.float32
        if pref == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MTRANSLATE_INPAINT_DEVICE=mps requested but MPS is not available")
            return "mps", torch.float32
        if pref == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("MTRANSLATE_INPAINT_DEVICE=cuda requested but CUDA is not available")
            return "cuda", torch.float16
        if pref == "cpu":
            return "cpu", torch.float32
        raise RuntimeError(f"Unknown MTRANSLATE_INPAINT_DEVICE value: {pref}")

    def _detect_pipeline_mode(self, model_dir: Path) -> str:
        model_index = model_dir / "model_index.json"
        if model_index.exists():
            try:
                parsed = json.loads(model_index.read_text(encoding="utf-8"))
                cls = str(parsed.get("_class_name", ""))
                if "Inpaint" in cls:
                    return "inpaint"
            except Exception:
                pass

        unet_cfg = model_dir / "unet" / "config.json"
        if unet_cfg.exists():
            try:
                parsed = json.loads(unet_cfg.read_text(encoding="utf-8"))
                if int(parsed.get("in_channels", 4)) >= 9:
                    return "inpaint"
            except Exception:
                pass

        # Base image-to-image checkpoints can still be used for masked region cleanup
        # by generating over a seeded fill image and compositing only masked pixels back.
        return "img2img_masked"

    def _pipeline(self):
        try:
            import torch  # type: ignore
            from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Diffusion inpaint requires `torch` and `diffusers`. "
                "Install them before using MTRANSLATE_INPAINT_BACKEND=diffusion."
            ) from exc

        model_dir = Path(self.model_path).expanduser()
        model_path = str(model_dir)
        mode = self._detect_pipeline_mode(model_dir)
        device, dtype = self._resolve_device_and_dtype()
        cache_key = (model_path, device, str(dtype), mode)
        cached = DiffusionInpaintBackend._pipe_cache.get(cache_key)
        if cached is not None:
            self._pipeline_mode = mode
            return cached

        kwargs = {
            "torch_dtype": dtype,
            "local_files_only": True,
        }
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        if mode == "inpaint":
            pipe = AutoPipelineForInpainting.from_pretrained(model_path, **kwargs)
        else:
            pipe = AutoPipelineForImage2Image.from_pretrained(model_path, **kwargs)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if device == "mps":
            # Helps keep memory pressure manageable on Apple Silicon.
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
        pipe = pipe.to(device)
        self._pipeline_mode = mode
        DiffusionInpaintBackend._pipe_cache[cache_key] = pipe
        return pipe

    def inpaint(self, src: Path, masks: List[InpaintMask], dst: Path) -> None:
        if not masks:
            copy_file(src, dst)
            return
        try:
            from PIL import Image, ImageDraw  # type: ignore
            import torch  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Diffusion inpaint requires Pillow and torch") from exc

        pipe = self._pipeline()
        image = Image.open(src).convert("RGB")
        mask_img = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask_img)
        for m in masks:
            x, y, w, h = m.bbox
            if w <= 0 or h <= 0:
                continue
            draw.rectangle([x, y, x + w, y + h], fill=255)

        # SDXL inpaint works best near native resolutions; keep aspect ratio.
        width, height = image.size
        max_side = 1536
        scale = min(1.0, float(max_side) / float(max(width, height)))
        rw = max(64, int((width * scale) // 8 * 8))
        rh = max(64, int((height * scale) // 8 * 8))
        image_resized = image.resize((rw, rh), Image.Resampling.LANCZOS)
        mask_resized = mask_img.resize((rw, rh), Image.Resampling.BILINEAR)

        generator = None
        seed = os.getenv("MTRANSLATE_INPAINT_SEED", "").strip()
        if seed:
            try:
                generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
            except ValueError:
                generator = None

        if self._pipeline_mode == "inpaint":
            out = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=image_resized,
                mask_image=mask_resized,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                strength=self.strength,
                generator=generator,
            ).images[0]
        else:
            fill = Image.new("RGB", image_resized.size, color=(246, 246, 246))
            seed_image = Image.composite(fill, image_resized, mask_resized)
            generated = pipe(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=seed_image,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                strength=self.strength,
                generator=generator,
            ).images[0].convert("RGB")
            out = Image.composite(generated, image_resized, mask_resized)

        out = out.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst)


class CopyInpaintBackend(InpaintBackend):
    name = "copy"

    def warmup(self) -> None:
        return None

    def inpaint(self, src: Path, masks: List[InpaintMask], dst: Path) -> None:
        copy_file(src, dst)


class ExternalInpaintBackend(InpaintBackend):
    name = "external"

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or _default_inpaint_model_path()
        self.command = _default_inpaint_command()
        self.prompt = os.getenv("MTRANSLATE_INPAINT_PROMPT", "").strip()
        self.negative_prompt = os.getenv("MTRANSLATE_INPAINT_NEGATIVE_PROMPT", "").strip()
        self.steps = env_int("MTRANSLATE_INPAINT_STEPS", 28, min_value=1, max_value=200)
        self.guidance = env_float("MTRANSLATE_INPAINT_GUIDANCE", 6.0, min_value=0.0, max_value=20.0)
        self.strength = env_float("MTRANSLATE_INPAINT_STRENGTH", 0.95, min_value=0.0, max_value=1.0)

    def warmup(self) -> None:
        if not self.command:
            raise RuntimeError(
                "External inpaint backend requested but MTRANSLATE_INPAINT_COMMAND is not set"
            )

    def inpaint(self, src: Path, masks: List[InpaintMask], dst: Path) -> None:
        if not self.command:
            raise RuntimeError(
                "External inpaint backend requested but MTRANSLATE_INPAINT_COMMAND is not set"
            )
        payload = {
            "src": str(src),
            "dst": str(dst),
            "model": self.model_path,
            "masks": [asdict(mask) for mask in masks],
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "steps": self.steps,
            "guidance": self.guidance,
            "strength": self.strength,
        }
        cp = subprocess.run(
            shlex.split(self.command),
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout).strip()
            raise RuntimeError(detail or f"external inpaint command failed with exit code {cp.returncode}")

        if dst.exists():
            return

        stdout = (cp.stdout or "").strip()
        if stdout:
            try:
                parsed = json.loads(stdout)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                output_path = parsed.get("output") or parsed.get("dst") or parsed.get("result")
                if isinstance(output_path, str) and output_path.strip():
                    copy_file(Path(output_path.strip()), dst)
                    return

        raise RuntimeError("external inpaint command completed without producing the requested output file")


@dataclass
class InpaintSelection:
    backend: str
    backend_obj: InpaintBackend


def select_inpaint_backend() -> InpaintSelection:
    backend = (
        os.getenv("MTRANSLATE_INFILL_BACKEND")
        or os.getenv("MTRANSLATE_INPAINT_BACKEND")
        or "diffusion"
    ).strip().lower()

    if backend in {"diffusion", "diffusers", "sd", "stable_diffusion"}:
        diff = DiffusionInpaintBackend()
        diff.warmup()
        return InpaintSelection(backend="diffusion", backend_obj=diff)
    if backend in {"copy", "none", "noop"}:
        copy_backend = CopyInpaintBackend()
        copy_backend.warmup()
        return InpaintSelection(backend="copy", backend_obj=copy_backend)
    if backend in {"external", "command"}:
        ext = ExternalInpaintBackend()
        ext.warmup()
        return InpaintSelection(backend="external", backend_obj=ext)

    raise ValueError(
        f"Unsupported inpaint backend: {backend}. "
        "Supported: diffusion, copy, external"
    )
