"""Cross-platform Python native helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_font(font_path: str | None, size: int):
    try:
        from PIL import ImageFont  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("typeset requires Pillow") from exc
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_text(draw, text: str, font, max_width: int) -> List[str]:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return []
    if max_width <= 8:
        return [cleaned]

    if " " in cleaned:
        tokens: Iterable[str] = cleaned.split(" ")
    else:
        tokens = list(cleaned)

    lines: List[str] = []
    current = ""
    for token in tokens:
        candidate = f"{current} {token}".strip() if " " in cleaned else (current + token)
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if current and width > max_width:
            lines.append(current)
            current = token
            continue
        current = candidate
    if current:
        lines.append(current)
    return lines or [cleaned]


def native_typeset_batch(tasks: List[Dict[str, Any]], font_path: str | None) -> None:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("typeset requires Pillow") from exc

    for task in tasks:
        src = str(task.get("input", ""))
        dst = str(task.get("output", ""))
        if not src or not dst:
            raise RuntimeError("typeset task missing input or output path")

        image = Image.open(src).convert("RGBA")
        draw = ImageDraw.Draw(image)
        for block in task.get("blocks", []) or []:
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            bbox = block.get("bbox") or [0, 0, 1, 1]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, w, h = [int(v) for v in bbox]
            w = max(1, w)
            h = max(1, h)
            font_size = max(6, int(block.get("font_size", max(10, int(min(w, h) * 0.2)))))
            font = _load_font(font_path, font_size)
            line_spacing = max(0.9, float(block.get("line_spacing", 1.15)))
            align = str(block.get("align", "center")).lower().strip()

            lines = _wrap_text(draw, text, font, max_width=max(8, w - 4))
            line_heights = []
            for ln in lines:
                tb = draw.textbbox((0, 0), ln, font=font)
                line_heights.append(max(1, tb[3] - tb[1]))
            text_height = int(sum(line_heights) + max(0, len(lines) - 1) * font_size * (line_spacing - 1.0))
            cursor_y = y + max(0, (h - text_height) // 2)

            for idx, ln in enumerate(lines):
                tb = draw.textbbox((0, 0), ln, font=font)
                line_width = max(1, tb[2] - tb[0])
                if align == "left":
                    cursor_x = x + 2
                elif align == "right":
                    cursor_x = x + max(0, w - line_width - 2)
                else:
                    cursor_x = x + max(0, (w - line_width) // 2)
                draw.text((cursor_x, cursor_y), ln, fill=(0, 0, 0, 255), font=font)
                cursor_y += int(line_heights[idx] * line_spacing)

        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst_path)


def native_images_to_pdf(images: List[str], output_pdf: str) -> None:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("PDF export requires Pillow") from exc

    if not images:
        raise RuntimeError("No images available to export PDF")
    opened = [Image.open(path).convert("RGB") for path in images]
    output = Path(output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    head, rest = opened[0], opened[1:]
    head.save(output, save_all=True, append_images=rest)
    for img in opened:
        img.close()
