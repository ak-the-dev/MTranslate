"""Utility helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Sequence


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
    cp = run_cmd(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)])
    if cp.returncode != 0:
        raise RuntimeError(f"Failed to read dimensions for {path}: {cp.stderr.strip()}")
    width = None
    height = None
    for line in cp.stdout.splitlines():
        line = line.strip()
        if line.startswith("pixelWidth:"):
            width = int(line.split(":", 1)[1].strip())
        elif line.startswith("pixelHeight:"):
            height = int(line.split(":", 1)[1].strip())
    if width is None or height is None:
        raise RuntimeError(f"Could not parse dimensions for {path}")
    return width, height


def convert_to_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cp = run_cmd(["sips", "-s", "format", "png", str(src), "--out", str(dst)])
    if cp.returncode != 0:
        raise RuntimeError(f"Failed to normalize {src}: {cp.stderr.strip()}")


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


def parse_page_ranges(spec: str, page_ids: Iterable[str]) -> set[str]:
    ids = sorted(page_ids)
    idx = {str(i + 1): pid for i, pid in enumerate(ids)}
    selected: set[str] = set()
    for part in [x.strip() for x in spec.split(",") if x.strip()]:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                pid = idx.get(str(i))
                if pid:
                    selected.add(pid)
        else:
            pid = idx.get(part)
            if pid:
                selected.add(pid)
    return selected


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def env_truthy(key: str) -> bool:
    return os.getenv(key, "").strip().lower() in {"1", "true", "yes", "on"}
