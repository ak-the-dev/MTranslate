from __future__ import annotations

import os
from typing import Any, Optional


class VLMBackend:
    def refine_page(self, page: Any) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class QwenVLBackend(VLMBackend):
    def refine_page(self, page: Any) -> None:
        raise RuntimeError(
            "Qwen-VL backend is not wired up yet. "
            "Download weights, integrate the model runtime, and update QwenVLBackend."
        )


def get_vlm_backend() -> Optional[VLMBackend]:
    backend = os.getenv("MTRANSLATE_VLM_BACKEND", "none").strip().lower()
    if backend in {"", "none"}:
        return None
    if backend == "qwen_vl":
        return QwenVLBackend()
    raise ValueError(f"Unknown VLM backend: {backend}")

