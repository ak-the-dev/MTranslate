"""Path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from .constants import APP_NAME


def app_support_dir() -> Path:
    base = os.getenv("MTRANSLATE_APP_SUPPORT")
    root = (Path(base).expanduser() if base else (Path.cwd() / ".mtranslate_data")).resolve()
    cwd = Path.cwd().resolve()
    if not root.is_relative_to(cwd):
        raise RuntimeError(f"MTRANSLATE_APP_SUPPORT must be inside repository: {cwd}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def models_dir() -> Path:
    path = app_support_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_root() -> Path:
    path = app_support_dir() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_dir(job_id: str) -> Path:
    path = jobs_root() / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_job_tree(job_id: str) -> dict:
    root = job_dir(job_id)
    paths = {
        "root": root,
        "work": root / "work",
        "pages": root / "work" / "pages",
        "ocr": root / "work" / "ocr",
        "masks": root / "work" / "masks",
        "inpaint": root / "work" / "inpaint",
        "typeset": root / "work" / "typeset",
        "review": root / "review",
        "logs": root / "logs",
        "exports": root / "exports",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
