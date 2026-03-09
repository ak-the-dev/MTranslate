"""Utility helpers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, List, Sequence


def run_cmd(cmd: Sequence[str], cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def detect_mime(path: Path) -> str:
    cp = run_cmd(["file", "--mime-type", "-b", str(path)])
    if cp.returncode != 0:
        return "application/octet-stream"
    return cp.stdout.strip()


def get_image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required for image dimension detection")
    with Image.open(path) as img:
        return img.size


def convert_to_png(src: Path, dst: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required for image conversion")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.save(dst, "PNG")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def natural_key(name: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise ValueError(f"Input path is not a directory: {path}")
    files = [p for p in path.iterdir() if p.is_file()]
    files.sort(key=lambda p: natural_key(p.name))
    images = []
    for p in files:
        mime = detect_mime(p)
        if mime.startswith("image/"):
            images.append(p)
    if not images:
        raise ValueError(f"No images found in input folder: {path}")
    return images


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
