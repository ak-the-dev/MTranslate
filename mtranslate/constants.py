"""Constants used across the pipeline."""

from __future__ import annotations

APP_NAME = "MTranslate"

STAGES = [
    "ingest",
    "ocr",
    "vlm_refine",
    "semantic_group",
    "translate",
    "mask",
    "inpaint",
    "typeset",
    "compose",
    "export",
]

TERMINAL_STATUSES = {"done", "failed", "cancelled"}
PAGE_STATUSES = {"pending", "running", "done", "failed", "skipped"}

DEFAULT_EXPORT = ("folder", "pdf")
ALLOWED_EXPORT_FORMATS = {"folder", "pdf"}

DEFAULT_GLOSSARY = {
    "characters": {},
    "honorific_policy": "keep",
    "catchphrases": {},
    "banned_literals": {},
}
