"""Native macOS helper integrations."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class NativeToolError(RuntimeError):
    pass


def _swift_env() -> dict:
    env = dict(os.environ)
    env.setdefault("SWIFT_MODULECACHE_PATH", "/tmp/swift-module-cache")
    env.setdefault("CLANG_MODULE_CACHE_PATH", "/tmp/clang-module-cache")
    return env


def _run_swift(script: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="mtranslate_native_") as td:
        req = Path(td) / "request.json"
        res = Path(td) / "response.json"
        req.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        cmd = ["swift", script, str(req), str(res)]
        cp = subprocess.run(cmd, text=True, capture_output=True, env=_swift_env(), check=False)
        if cp.returncode != 0:
            raise NativeToolError(cp.stderr.strip() or cp.stdout.strip() or "swift helper failed")
        return json.loads(res.read_text(encoding="utf-8"))


def native_ocr_batch(image_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    script = str(Path(__file__).with_name("native_tools.swift"))
    payload = {"command": "ocr", "images": image_paths}
    out = _run_swift(script, payload)
    return out.get("regions", {})


def native_typeset_batch(tasks: List[Dict[str, Any]], font_path: str | None) -> None:
    script = str(Path(__file__).with_name("native_tools.swift"))
    payload: Dict[str, Any] = {"command": "typeset", "tasks": tasks}
    if font_path:
        payload["font_path"] = font_path
    _run_swift(script, payload)


def native_images_to_pdf(images: List[str], output_pdf: str) -> None:
    script = str(Path(__file__).with_name("native_tools.swift"))
    payload = {"command": "images_to_pdf", "images": images, "output": output_pdf}
    _run_swift(script, payload)
