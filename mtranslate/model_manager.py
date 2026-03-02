"""Model registry and provisioning stubs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List

from .paths import models_dir
from .utils import atomic_write_json, read_json


MODEL_PROFILES = {
    "hq": [
        {
            "id": "ocr_vision_macos",
            "kind": "ocr",
            "runtime": "vision",
            "size_gb": 0.0,
            "state": "ready",
            "notes": "Uses native Vision framework",
        },
        {
            "id": "vlm_refiner_mlx",
            "kind": "vlm",
            "runtime": "mlx",
            "size_gb": 16.0,
            "state": "missing",
            "notes": "Place local model artifacts in models/vlm_refiner_mlx",
        },
        {
            "id": "translator_llm_vllm",
            "kind": "llm",
            "runtime": "vllm",
            "size_gb": 10.0,
            "state": "missing",
            "notes": "Place Gemma 3 weights in models/google_gemma_3_4b_it",
        },
        {
            "id": "inpaint_sdxl",
            "kind": "inpaint",
            "runtime": "diffusers",
            "size_gb": 18.0,
            "state": "missing",
            "notes": "Place SDXL inpaint artifacts in models/sdxl_inpaint",
        },
    ]
}

DEFAULT_INPAINT_REPO = "Runware/Pony_Diffusion_V6_XL"
DEFAULT_TRANSLATE_REPO = "google/gemma-3-4b-it"
DEFAULT_TRANSLATE_VLLM_REPO = "google/gemma-3-4b-it"


@dataclass
class ModelRegistry:
    profile: str
    models: List[Dict]

    @property
    def ready(self) -> List[Dict]:
        return [m for m in self.models if m.get("state") == "ready"]

    @property
    def missing(self) -> List[Dict]:
        return [m for m in self.models if m.get("state") != "ready"]


def registry_path() -> Path:
    return models_dir() / "registry.json"


def load_registry() -> ModelRegistry | None:
    path = registry_path()
    if not path.exists():
        return None
    data = read_json(path)
    return ModelRegistry(profile=data["profile"], models=data["models"])


def pull_profile(profile: str) -> ModelRegistry:
    if profile not in MODEL_PROFILES:
        raise ValueError(f"Unknown model profile: {profile}")
    root = models_dir()
    models = []
    for model in MODEL_PROFILES[profile]:
        dst = root / model["id"]
        dst.mkdir(parents=True, exist_ok=True)
        info = dict(model)
        info["path"] = str(dst)
        models.append(info)
    payload = {"profile": profile, "models": models}
    atomic_write_json(registry_path(), payload)
    return ModelRegistry(profile=profile, models=models)


def pull_inpaint_model(
    repo_id: str = DEFAULT_INPAINT_REPO,
    dest: str | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "huggingface_hub is required to download inpaint models. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    root = models_dir()
    model_dir = Path(dest).expanduser().resolve() if dest else (root / "sdxl_inpaint")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
    )
    return model_dir


def _repo_slug(repo_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", repo_id.strip()).strip("_").lower()
    return slug or "translate_model"


def pull_translate_model(
    repo_id: str = DEFAULT_TRANSLATE_REPO,
    dest: str | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "huggingface_hub is required to download translation models. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    root = models_dir()
    model_dir = Path(dest).expanduser().resolve() if dest else (root / _repo_slug(repo_id))
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
    )
    return model_dir
