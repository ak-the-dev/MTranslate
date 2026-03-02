"""Regex replacement dictionaries for pre/post translation normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Pattern, Tuple


@dataclass(frozen=True)
class ReplacementRule:
    pattern: Pattern[str]
    replacement: str
    line_no: int


def _strip_inline_comment(line: str) -> str:
    # Keep simple and predictable: comments start with whitespace + # or whitespace + //
    line = re.sub(r"\s+#.*$", "", line)
    line = re.sub(r"\s+//.*$", "", line)
    return line.strip()


def _parse_rule(line: str) -> Tuple[str, str]:
    if "\t" in line:
        left, right = line.split("\t", 1)
        return left.strip(), right.strip()
    if "=>" in line:
        left, right = line.split("=>", 1)
        return left.strip(), right.strip()
    if "->" in line:
        left, right = line.split("->", 1)
        return left.strip(), right.strip()

    pieces = line.split(None, 1)
    if len(pieces) == 1:
        return pieces[0].strip(), ""
    return pieces[0].strip(), pieces[1].strip()


def load_replacement_rules(path: str | None) -> List[ReplacementRule]:
    if not path:
        return []

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dictionary file not found: {p}")

    rules: List[ReplacementRule] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text or text.startswith("#") or text.startswith("//"):
            continue
        text = _strip_inline_comment(text)
        if not text:
            continue

        src, dst = _parse_rule(text)
        if not src:
            continue
        try:
            compiled = re.compile(src)
        except re.error as exc:  # noqa: PERF203
            raise ValueError(f"Invalid regex at {p}:{line_no}: {exc}") from exc
        rules.append(ReplacementRule(pattern=compiled, replacement=dst, line_no=line_no))
    return rules


def apply_replacement_rules(text: str, rules: List[ReplacementRule]) -> tuple[str, list[dict[str, str | int]]]:
    out = text
    applied: list[dict[str, str | int]] = []
    for rule in rules:
        updated, count = rule.pattern.subn(rule.replacement, out)
        if count > 0:
            applied.append(
                {
                    "line_no": rule.line_no,
                    "pattern": rule.pattern.pattern,
                    "replacement": rule.replacement,
                    "count": count,
                }
            )
            out = updated
    return out, applied
