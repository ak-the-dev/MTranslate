"""Glossary loading and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .constants import DEFAULT_GLOSSARY


def load_glossary(path: str | None) -> Dict[str, Any]:
    data = dict(DEFAULT_GLOSSARY)
    if not path:
        return data
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Glossary file not found: {path}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        return data
    if not isinstance(raw, dict):
        raise ValueError("Glossary must be a mapping")
    for key in DEFAULT_GLOSSARY:
        if key in raw:
            data[key] = raw[key]
    return data
