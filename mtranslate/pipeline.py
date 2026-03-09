"""Core pipeline orchestration for MTranslate."""

from __future__ import annotations

import math
import os
import re
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .constants import ALLOWED_EXPORT_FORMATS, APP_NAME, DEFAULT_EXPORT, STAGES
from .dictionary_rules import apply_replacement_rules, load_replacement_rules
from .env import env_bool, env_float, env_int
from .glossary import load_glossary
from .inpaint_backends import select_inpaint_backend
from .jsonl_logger import EventLogger
from .native import native_images_to_pdf, native_typeset_batch
from .ocr_backends import extract_regions_batch
from .paths import app_support_dir, ensure_job_tree, job_dir
from .translator_backends import select_translation_backend
from .types import InpaintMask, JobManifest, PageManifest, TextRegion, TypesetBlock, job_from_dict
from .utils import (
    atomic_write_json,
    convert_to_png,
    copy_file,
    detect_mime,
    get_image_dimensions,
    list_images,
    read_json,
)
from .vlm_backends import get_vlm_backend


class PipelineError(RuntimeError):
    pass


def _validate_export_formats(export_formats: Sequence[str] | None) -> List[str]:
    if not export_formats:
        return list(DEFAULT_EXPORT)
    normalized = [x.strip().lower() for x in export_formats if x and x.strip()]
    if not normalized:
        raise PipelineError("At least one export format is required")
    unsupported = sorted({x for x in normalized if x not in ALLOWED_EXPORT_FORMATS})
    if unsupported:
        allowed = ",".join(sorted(ALLOWED_EXPORT_FORMATS))
        raise PipelineError(f"Unsupported export format(s): {', '.join(unsupported)}. Allowed: {allowed}")
    return normalized


def _validate_io_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise PipelineError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise PipelineError(f"Input path must be a directory: {input_path}")
    if output_path == input_path:
        raise PipelineError("Output path must be different from input path")
    try:
        if output_path.is_relative_to(input_path):
            raise PipelineError("Output path must not be inside input path")
    except AttributeError:
        # Python < 3.9 compatibility fallback.
        if str(output_path).startswith(str(input_path) + os.sep):
            raise PipelineError("Output path must not be inside input path")


def _series_id_from_input(input_path: str) -> str:
    path = Path(input_path).expanduser().resolve()
    base = path.name if path.is_dir() else path.parent.name
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return slug or "default_series"


def _series_glossary_path(series_id: str) -> Path:
    root = app_support_dir() / "glossaries"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{series_id}.yaml"


def _bootstrap_series_glossary(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "characters: {}\n"
        "honorific_policy: keep\n"
        "catchphrases: {}\n"
        "banned_literals: {}\n",
        encoding="utf-8",
    )


def _manifest_path(job_id: str) -> Path:
    return job_dir(job_id) / "manifest.json"


def _default_font_path(repo_root: Path) -> Optional[str]:
    font = repo_root / "assets" / "fonts" / "manga.ttf"
    return str(font) if font.exists() else None


def _looks_japanese(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (0x3040 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            return True
    return False


def _is_sfx_like(text: str) -> bool:
    stripped = "".join(ch for ch in text if not ch.isspace())
    if len(stripped) <= 5 and _looks_japanese(stripped):
        return True
    if any(ch in text for ch in ["！", "？", "ッ", "ッ", "ゴ", "ド", "バ", "ズ"]):
        return True
    return False


def _is_hiragana(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x309F


def _is_katakana(ch: str) -> bool:
    code = ord(ch)
    return 0x30A0 <= code <= 0x30FF


def _is_kanji(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF


def _region_text_noise_like(text: str, confidence: float, bbox: tuple[int, int, int, int]) -> bool:
    clean = re.sub(r"\s+", "", text or "")
    if not clean:
        return True

    hiragana = sum(1 for ch in clean if _is_hiragana(ch))
    katakana = sum(1 for ch in clean if _is_katakana(ch))
    kanji = sum(1 for ch in clean if _is_kanji(ch))
    latin_or_digit = sum(1 for ch in clean if ch.isascii() and (ch.isalpha() or ch.isdigit()))
    total = max(1, len(clean))
    _, _, w, h = bbox
    min_side = min(max(1, w), max(1, h))

    if len(clean) <= 1 and confidence < 0.98:
        return True
    if latin_or_digit == total and total <= 8:
        return True
    if kanji == 0 and len(clean) <= 2 and confidence < 0.98:
        return True
    if kanji == 0 and min_side < 20 and len(clean) <= 4 and confidence < 0.96:
        return True
    if kanji == 0 and (hiragana / total) >= 0.8 and len(clean) <= 6 and confidence < 0.9:
        return True
    return False


def _bbox_area(b: tuple[int, int, int, int]) -> int:
    return max(1, int(b[2])) * max(1, int(b[3]))


def _bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _axis_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = _axis_overlap(ax, ax + aw, bx, bx + bw)
    iy = _axis_overlap(ay, ay + ah, by, by + bh)
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1.0, float(union))


def _bbox_gap_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    if dx == 0 and dy == 0:
        return 0.0
    return math.hypot(float(dx), float(dy))


def _connected_components(node_ids: List[int], edges: List[tuple[int, int]]) -> List[List[int]]:
    if not node_ids:
        return []
    adj: Dict[int, set[int]] = {n: set() for n in node_ids}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    seen: set[int] = set()
    out: List[List[int]] = []
    for n in node_ids:
        if n in seen:
            continue
        stack = [n]
        comp: List[int] = []
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj.get(cur, set()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)
        out.append(comp)
    return out


def _split_region_component(
    component: List[int],
    regions: List[TextRegion],
    gamma: float = 0.55,
    sigma: float = 2.0,
) -> List[List[int]]:
    if len(component) <= 1:
        return [component]
    if len(component) == 2:
        u, v = component
        fs = max(1.0, min(regions[u].bbox[2], regions[u].bbox[3]), min(regions[v].bbox[2], regions[v].bbox[3]))
        dist = _bbox_gap_distance(regions[u].bbox, regions[v].bbox)
        if dist <= (1.0 + gamma) * fs:
            return [component]
        return [[u], [v]]

    edges: List[tuple[int, int, float]] = []
    for i in range(len(component)):
        for j in range(i + 1, len(component)):
            u = component[i]
            v = component[j]
            edges.append((u, v, _bbox_gap_distance(regions[u].bbox, regions[v].bbox)))
    if not edges:
        return [component]

    # Kruskal MST for robust split detection inside a coarse connected component.
    parent = {n: n for n in component}
    rank = {n: 0 for n in component}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst: List[tuple[int, int, float]] = []
    for u, v, w in sorted(edges, key=lambda x: x[2]):
        if union(u, v):
            mst.append((u, v, w))
    if len(mst) <= 1:
        return [component]

    sorted_mst = sorted(mst, key=lambda x: x[2], reverse=True)
    distances = [w for _, _, w in sorted_mst]
    mean = sum(distances) / float(len(distances))
    var = sum((d - mean) ** 2 for d in distances) / float(len(distances))
    std = math.sqrt(var)
    font_like = sum(min(regions[n].bbox[2], regions[n].bbox[3]) for n in component) / float(len(component))
    split_threshold = max(mean + std * sigma, font_like * (1.0 + gamma))

    top_u, top_v, top_w = sorted_mst[0]
    if top_w <= split_threshold:
        return [component]

    kept_edges = [(u, v) for u, v, _ in mst if not (u == top_u and v == top_v)]
    parts = _connected_components(component, kept_edges)
    if len(parts) <= 1:
        return [component]

    out: List[List[int]] = []
    for part in parts:
        out.extend(_split_region_component(part, regions, gamma=gamma, sigma=sigma))
    return out


def _should_merge_regions(a: TextRegion, b: TextRegion) -> bool:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    a_vertical = a.orientation == "vertical"
    b_vertical = b.orientation == "vertical"
    if a_vertical != b_vertical:
        return False

    if a_vertical:
        y_overlap = _axis_overlap(ay, ay + ah, by, by + bh)
        y_overlap_ratio = y_overlap / max(1.0, float(min(ah, bh)))
        x_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
        return y_overlap_ratio >= 0.45 and x_gap <= max(16, int(max(aw, bw) * 1.2))

    y_overlap = _axis_overlap(ay, ay + ah, by, by + bh)
    y_overlap_ratio = y_overlap / max(1.0, float(min(ah, bh)))
    x_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    y_center_delta = abs((ay + ah / 2.0) - (by + bh / 2.0))
    return (y_overlap_ratio >= 0.35 or y_center_delta <= max(ah, bh) * 0.7) and x_gap <= max(
        14, int(min(aw, bw) * 0.9)
    )


def _merge_regions(regions: List[TextRegion]) -> List[TextRegion]:
    if len(regions) < 2:
        return regions

    node_ids = list(range(len(regions)))
    coarse_edges: List[tuple[int, int]] = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _should_merge_regions(regions[i], regions[j]):
                coarse_edges.append((i, j))

    components = _connected_components(node_ids, coarse_edges)
    groups: List[List[TextRegion]] = []
    for comp in components:
        split_parts = _split_region_component(comp, regions)
        for part in split_parts:
            groups.append([regions[idx] for idx in part])

    merged: List[TextRegion] = []
    for idx, group in enumerate(groups):
        if len(group) == 1:
            merged.append(group[0])
            continue

        vertical = group[0].orientation == "vertical"
        # Japanese vertical lines are typically read right-to-left by columns.
        ordered = sorted(group, key=lambda r: (-r.bbox[0], r.bbox[1]) if vertical else (r.bbox[0], r.bbox[1]))
        text = " ".join(r.text for r in ordered)
        bbox = ordered[0].bbox
        for r in ordered[1:]:
            bbox = _bbox_union(bbox, r.bbox)
        area_sum = sum(_bbox_area(r.bbox) for r in ordered)
        conf = sum(r.confidence * _bbox_area(r.bbox) for r in ordered) / max(1.0, float(area_sum))
        x, y, w, h = bbox
        merged.append(
            TextRegion(
                id=f"merged_{idx}",
                text=text.strip(),
                bbox=bbox,
                polygon=[(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                orientation=ordered[0].orientation,
                role=ordered[0].role,
                confidence=conf,
                source=ordered[0].source,
            )
        )

    return merged


def _dedupe_overlapping_regions(regions: List[TextRegion], iou_threshold: float = 0.5) -> List[TextRegion]:
    kept: List[TextRegion] = []
    for r in sorted(regions, key=lambda x: (_bbox_area(x.bbox), x.confidence), reverse=True):
        if any(_bbox_iou(r.bbox, k.bbox) >= iou_threshold for k in kept):
            continue
        kept.append(r)
    return kept


def _snap_regions_to_beige_boxes(page: PageManifest, regions: List[TextRegion]) -> None:
    if not page.normalized_path or not regions:
        return
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return

    img = cv2.imread(page.normalized_path)
    if img is None:
        return
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Beige paper-like box interiors used for narration/dialogue cards.
    mask = ((hue > 10) & (hue < 40) & (sat > 8) & (sat < 120) & (val > 120)).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return
    page_area = float(w * h)

    for region in regions:
        x, y, bw, bh = region.bbox
        pad = max(20, int(min(bw, bh) * 0.35))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        sub = labels[y1:y2, x1:x2]
        uniq, counts = np.unique(sub, return_counts=True)
        best_bbox: tuple[int, int, int, int] | None = None
        best_hits = -1
        for lab, hits in sorted(zip(uniq.tolist(), counts.tolist()), key=lambda kv: kv[1], reverse=True):
            if lab == 0:
                continue
            sx, sy, sw, sh, area = [int(v) for v in stats[lab]]
            if area < int(bw * bh * 1.5):
                continue
            if area > int(page_area * 0.15):
                continue
            aspect = max(sw / float(max(1, sh)), sh / float(max(1, sw)))
            if aspect > 4.0:
                continue

            cx = x + bw // 2
            cy = y + bh // 2
            inside = sx <= cx <= (sx + sw) and sy <= cy <= (sy + sh)
            if not inside:
                ox = max(0, min(x + bw, sx + sw) - max(x, sx))
                oy = max(0, min(y + bh, sy + sh) - max(y, sy))
                if (ox * oy) < int(bw * bh * 0.25):
                    continue

            if hits > best_hits:
                best_hits = hits
                best_bbox = (sx, sy, sw, sh)

        if not best_bbox:
            continue
        sx, sy, sw, sh = best_bbox
        inset = max(2, int(min(sw, sh) * 0.06))
        nx = max(0, sx + inset)
        ny = max(0, sy + inset)
        nw = max(1, sw - inset * 2)
        nh = max(1, sh - inset * 2)
        region.bbox = (nx, ny, nw, nh)
        region.polygon = [(nx, ny), (nx + nw, ny), (nx + nw, ny + nh), (nx, ny + nh)]


class PipelineRunner:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        export_formats: Sequence[str] | None = None,
        glossary_path: str | None = None,
        pre_dict_path: str | None = None,
        post_dict_path: str | None = None,
        max_workers: int | None = None,
        font_path: str | None = None,
        job_id: str | None = None,
        repo_root: str | None = None,
    ) -> None:
        input_dir = Path(input_path).expanduser().resolve()
        output_dir = Path(output_path).expanduser().resolve()
        _validate_io_paths(input_dir, output_dir)

        self.input_path = str(input_dir)
        self.output_path = str(output_dir)
        self.export_formats = _validate_export_formats(export_formats)
        self.series_id = _series_id_from_input(self.input_path)
        if glossary_path:
            self.glossary_path = str(Path(glossary_path).expanduser().resolve())
        else:
            series_glossary = _series_glossary_path(self.series_id)
            _bootstrap_series_glossary(series_glossary)
            self.glossary_path = str(series_glossary)
        self.glossary = load_glossary(self.glossary_path)
        self.pre_dict_path = (
            str(Path(pre_dict_path).expanduser().resolve())
            if pre_dict_path
            else (os.getenv("MTRANSLATE_PRE_DICT", "").strip() or None)
        )
        self.post_dict_path = (
            str(Path(post_dict_path).expanduser().resolve())
            if post_dict_path
            else (os.getenv("MTRANSLATE_POST_DICT", "").strip() or None)
        )
        self.pre_dict_rules = load_replacement_rules(self.pre_dict_path)
        self.post_dict_rules = load_replacement_rules(self.post_dict_path)
        self.context_pages = env_int("MTRANSLATE_CONTEXT_PAGES", 2, min_value=0, max_value=8)
        self.context_history_lines = env_int("MTRANSLATE_CONTEXT_HISTORY_LINES", 24, min_value=0, max_value=200)
        default_workers = max(2, min(8, (os.cpu_count() or 4)))
        self.max_workers = max(1, min(32, max_workers or default_workers))
        self.repo_root = Path(repo_root or os.getcwd())
        self.font_path = font_path or _default_font_path(self.repo_root)
        self._manifest_lock = threading.Lock()
        self.vlm_backend = get_vlm_backend()
        self.inpaint_selection = select_inpaint_backend()
        self.translation_selection = None

        self.job_id = job_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.paths = ensure_job_tree(self.job_id)
        self.manifest_file = _manifest_path(self.job_id)
        self.logger = EventLogger(self.paths["logs"] / "events.jsonl")

        self.manifest = self._load_or_init_manifest()
        self._translated_page_cache: Dict[str, List[str]] = {}
        if self.pre_dict_path:
            self.manifest.notes["pre_dict_path"] = self.pre_dict_path
            self.manifest.notes["pre_dict_rules"] = len(self.pre_dict_rules)
        if self.post_dict_path:
            self.manifest.notes["post_dict_path"] = self.post_dict_path
            self.manifest.notes["post_dict_rules"] = len(self.post_dict_rules)
        self.manifest.notes["context_pages"] = self.context_pages

    @classmethod
    def from_job(cls, job_id: str, repo_root: str | None = None) -> "PipelineRunner":
        manifest = load_manifest(job_id)
        runner = cls(
            input_path=manifest.input_path,
            output_path=manifest.output_path,
            export_formats=manifest.export,
            glossary_path=manifest.glossary_path,
            pre_dict_path=manifest.notes.get("pre_dict_path"),
            post_dict_path=manifest.notes.get("post_dict_path"),
            job_id=job_id,
            repo_root=repo_root,
        )
        for page in manifest.pages.values():
            page.ensure_stages(STAGES)
        runner.manifest = manifest
        runner.save_manifest()
        return runner

    def _load_or_init_manifest(self) -> JobManifest:
        if self.manifest_file.exists():
            return job_from_dict(read_json(self.manifest_file))

        manifest = JobManifest.new(
            job_id=self.job_id,
            input_path=self.input_path,
            output_path=self.output_path,
            export=self.export_formats,
            glossary_path=self.glossary_path,
        )

        pages = list_images(Path(self.input_path))
        for idx, path in enumerate(pages, start=1):
            page_id = f"{idx:03d}"
            page = PageManifest(
                page_id=page_id,
                index=idx,
                input_path=str(path.resolve()),
            )
            page.ensure_stages(STAGES)
            manifest.pages[page_id] = page

        manifest.notes["font_path"] = self.font_path
        manifest.notes["engine"] = {
            "name": APP_NAME,
            "runtime": "python+ai-backends",
            "max_workers": self.max_workers,
            "created_by": "mtranslate run",
        }
        manifest.notes["series_id"] = self.series_id
        manifest.notes["translate_backend"] = (
            os.getenv("MTRANSLATE_TRANSLATE_BACKEND", "vllm").strip().lower() or "vllm"
        )
        manifest.notes["inpaint_backend"] = self.inpaint_selection.backend
        if self.pre_dict_path:
            manifest.notes["pre_dict_path"] = self.pre_dict_path
            manifest.notes["pre_dict_rules"] = len(self.pre_dict_rules)
        if self.post_dict_path:
            manifest.notes["post_dict_path"] = self.post_dict_path
            manifest.notes["post_dict_rules"] = len(self.post_dict_rules)
        manifest.notes["context_pages"] = self.context_pages
        return manifest

    def _ensure_translation_selection(self):
        if self.translation_selection is None:
            self.translation_selection = select_translation_backend(glossary=self.glossary)
            self.manifest.notes["translate_backend"] = self.translation_selection.backend
        return self.translation_selection

    def save_manifest(self) -> None:
        with self._manifest_lock:
            self.manifest.touch()
            atomic_write_json(self.manifest_file, self.manifest.as_dict())

    def _set_job_status(self, status: str, **extra: Any) -> None:
        self.manifest.status = status
        self.manifest.notes.update(extra)
        self.save_manifest()

    def _stage_begin(self, page: PageManifest, stage: str) -> float:
        now = datetime.now(timezone.utc)
        st = page.stages[stage]
        st.status = "running"
        st.started_at = now.isoformat()
        st.finished_at = None
        st.error_code = None
        st.error_message = None
        self.logger.emit(
            {
                "event": "stage_started",
                "job_id": self.job_id,
                "page_id": page.page_id,
                "stage": stage,
            }
        )
        self.save_manifest()
        return now.timestamp()

    def _stage_end(
        self,
        page: PageManifest,
        stage: str,
        ok: bool,
        started_ts: float,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        end = datetime.now(timezone.utc)
        st = page.stages[stage]
        st.finished_at = end.isoformat()
        st.duration_ms = int((end.timestamp() - started_ts) * 1000)
        st.status = "done" if ok else "failed"
        st.error_code = error_code
        st.error_message = error_message

        if ok:
            page.status = "done" if stage == STAGES[-1] else "running"
        else:
            page.status = "failed"

        self.logger.emit(
            {
                "event": "stage_finished",
                "job_id": self.job_id,
                "page_id": page.page_id,
                "stage": stage,
                "status": st.status,
                "duration_ms": st.duration_ms,
                "error_code": error_code,
                "error_message": error_message,
            }
        )
        self.save_manifest()

    def _run_stage_for_page(
        self,
        page: PageManifest,
        stage: str,
        fn: Callable[[PageManifest], None],
    ) -> None:
        st = page.stages[stage]
        if st.status == "done":
            return

        started_ts = self._stage_begin(page, stage)
        try:
            fn(page)
            self._stage_end(page, stage, ok=True, started_ts=started_ts)
        except Exception as exc:  # noqa: BLE001
            self._stage_end(
                page,
                stage,
                ok=False,
                started_ts=started_ts,
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )

    def _ingest_page(self, page: PageManifest) -> None:
        src = Path(page.input_path)
        mime = detect_mime(src)
        if not mime.startswith("image/"):
            raise PipelineError(f"Unsupported input mime: {mime} ({src})")

        normalized = self.paths["pages"] / f"{page.page_id}.png"
        convert_to_png(src, normalized)
        w, h = get_image_dimensions(normalized)

        page.normalized_path = str(normalized)
        page.width = w
        page.height = h
        page.mime_type = mime
        page.output_paths["normalized"] = str(normalized)

    def _ocr_stage(self, pages: List[PageManifest]) -> None:
        target_pages = [p for p in pages if p.stages["ocr"].status != "done"]
        if not target_pages:
            return

        started_ts: Dict[str, float] = {}
        for p in target_pages:
            started_ts[p.page_id] = self._stage_begin(p, "ocr")

        ocr_input = [p.normalized_path for p in target_pages if p.normalized_path]
        if not ocr_input:
            for p in target_pages:
                self._stage_end(p, "ocr", True, started_ts[p.page_id])
            return

        try:
            ocr_result = extract_regions_batch(ocr_input)
            by_path = ocr_result.by_path
            self.manifest.notes["ocr_backend"] = ocr_result.backend
            if getattr(ocr_result, "model_root", None):
                self.manifest.notes["ocr_model_root"] = ocr_result.model_root
            for p in target_pages:
                regions: List[TextRegion] = []
                for raw in by_path.get(p.normalized_path or "", []):
                    bbox = tuple(int(v) for v in raw.get("bbox", [0, 0, 1, 1]))
                    poly = [tuple(int(c) for c in pt) for pt in raw.get("polygon", [])]
                    txt = str(raw.get("text", "")).strip()
                    if not txt:
                        continue
                    regions.append(
                        TextRegion(
                            id=str(raw.get("id", f"ocr_{len(regions)}")),
                            text=txt,
                            bbox=bbox,
                            polygon=poly,
                            orientation=str(raw.get("orientation", "horizontal")),
                            role="dialogue",
                            confidence=float(raw.get("confidence", 0.5)),
                            source=str(raw.get("source", "ocr")),
                        )
                    )
                p.text_regions = regions
                p.output_paths["ocr_backend"] = ocr_result.backend
                p.output_paths["ocr"] = str(self.paths["ocr"] / f"{p.page_id}.json")
                atomic_write_json(Path(p.output_paths["ocr"]), [asdict(r) for r in regions])
                if not regions:
                    self.logger.emit(
                        {
                            "event": "ocr_empty",
                            "job_id": self.job_id,
                            "page_id": p.page_id,
                            "message": "No OCR regions detected for this page",
                        }
                    )
                self._stage_end(p, "ocr", True, started_ts[p.page_id])
        except Exception as exc:  # noqa: BLE001
            for p in target_pages:
                p.text_regions = []
                self._stage_end(
                    p,
                    "ocr",
                    False,
                    started_ts[p.page_id],
                    error_code=exc.__class__.__name__,
                    error_message=str(exc),
                )

    def _vlm_refine_page(self, page: PageManifest) -> None:
        if self.vlm_backend is not None:
            self.vlm_backend.refine_page(page)
            return

        for region in page.text_regions:
            if _is_sfx_like(region.text):
                region.role = "sfx"
            elif len(region.text) > 24:
                region.role = "narration"
            else:
                region.role = "dialogue"
            if region.orientation not in {"horizontal", "vertical"}:
                x, y, w, h = region.bbox
                region.orientation = "vertical" if h > w else "horizontal"

    def _semantic_group_page(self, page: PageManifest) -> None:
        ocr_backend = str(self.manifest.notes.get("ocr_backend", "")).strip().lower()
        min_conf = env_float(
            "MTRANSLATE_REGION_MIN_CONFIDENCE",
            0.30 if ocr_backend == "vision" else 0.55,
            min_value=0.0,
            max_value=1.0,
        )
        min_area = env_int("MTRANSLATE_REGION_MIN_AREA", 120, min_value=1, max_value=200_000)
        max_area_ratio = env_float("MTRANSLATE_REGION_MAX_AREA_RATIO", 0.22, min_value=0.01, max_value=0.95)
        max_aspect_ratio = env_float(
            "MTRANSLATE_REGION_MAX_ASPECT_RATIO",
            16.0 if ocr_backend == "vision" else 10.0,
            min_value=1.0,
            max_value=80.0,
        )
        merge_enabled = env_bool("MTRANSLATE_REGION_MERGE", default=ocr_backend != "vision")
        top_toolbar_ratio = env_float(
            "MTRANSLATE_REGION_TOP_BANNER_RATIO",
            0.015 if ocr_backend == "vision" else 0.04,
            min_value=0.0,
            max_value=0.2,
        )
        top_width_ratio = env_float("MTRANSLATE_REGION_TOP_BANNER_WIDTH_RATIO", 0.60, min_value=0.0, max_value=1.0)
        top_filter_max_conf = env_float("MTRANSLATE_REGION_TOP_BANNER_MAX_CONF", 0.90, min_value=0.0, max_value=1.0)
        page_area = max(1, int((page.width or 1) * (page.height or 1)))

        filtered: List[TextRegion] = []
        for r in page.text_regions:
            area = _bbox_area(r.bbox)
            if r.confidence < min_conf:
                continue
            if area < min_area:
                continue
            if (area / float(page_area)) > max_area_ratio:
                continue
            x, y, w, h = r.bbox
            if w <= 0 or h <= 0:
                continue
            aspect = max(w / float(h), h / float(w))
            if aspect > max_aspect_ratio:
                continue
            if _region_text_noise_like(r.text, r.confidence, r.bbox):
                continue
            top_toolbar_px = int((page.height or 0) * top_toolbar_ratio)
            if y < top_toolbar_px and w > int((page.width or 0) * top_width_ratio) and r.confidence <= top_filter_max_conf:
                continue
            filtered.append(r)

        merged = _merge_regions(filtered) if merge_enabled else list(filtered)
        merged = _dedupe_overlapping_regions(merged, iou_threshold=0.5)
        if env_bool("MTRANSLATE_SNAP_TEXT_BOXES", default=False):
            _snap_regions_to_beige_boxes(page, merged)
        max_regions = env_int("MTRANSLATE_MAX_REGIONS_PER_PAGE", 10, min_value=1, max_value=100)
        if len(merged) > max_regions:
            merged = sorted(merged, key=lambda r: (_bbox_area(r.bbox), r.confidence), reverse=True)[:max_regions]
        merged.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        page.text_regions = merged

    def _page_source_lines(self, page: PageManifest) -> List[str]:
        if not page.text_regions:
            return []
        out: List[str] = []
        for region in page.text_regions:
            src = (region.text or "").strip()
            if not src:
                continue
            if self.pre_dict_rules:
                src, _ = apply_replacement_rules(src, self.pre_dict_rules)
            if src:
                out.append(src)
        return out

    def _page_translated_lines(self, page: PageManifest) -> List[str]:
        cached = self._translated_page_cache.get(page.page_id)
        if cached is not None:
            return [x for x in cached if x.strip()]
        return [b.text.strip() for b in (page.typeset_blocks or []) if (b.text or "").strip()]

    def _history_context_lines(self, ordered_pages: List[PageManifest], current_idx: int) -> List[str]:
        if self.context_pages <= 0:
            return []
        non_empty_pages: List[List[str]] = []
        for page in reversed(ordered_pages[:current_idx]):
            lines = self._page_translated_lines(page)
            if not lines:
                continue
            non_empty_pages.append(lines)
            if len(non_empty_pages) >= self.context_pages:
                break
        if not non_empty_pages:
            return []

        flattened: List[str] = []
        for page_lines in reversed(non_empty_pages):
            flattened.extend(page_lines)
        if self.context_history_lines > 0:
            flattened = flattened[-self.context_history_lines :]
        return [f"<|{i + 1}|>{line}" for i, line in enumerate(flattened)]

    def _apply_pre_dict(self, text: str) -> tuple[str, list[dict[str, str | int]]]:
        if not self.pre_dict_rules:
            return text, []
        return apply_replacement_rules(text, self.pre_dict_rules)

    def _apply_post_dict(self, text: str) -> tuple[str, list[dict[str, str | int]]]:
        if not self.post_dict_rules:
            return text, []
        return apply_replacement_rules(text, self.post_dict_rules)

    def _translate_pages(self, pages: List[PageManifest]) -> None:
        translate_selection = self._ensure_translation_selection()
        self.manifest.notes["translate_backend"] = translate_selection.backend

        backend = translate_selection.backend_obj
        by_order = sorted(pages, key=lambda p: p.index)
        all_pages = sorted(self.manifest.pages.values(), key=lambda p: p.index)
        page_pos = {p.page_id: i for i, p in enumerate(all_pages)}

        for page in by_order:
            st = page.stages["translate"]
            if st.status == "done":
                continue
            started_ts = self._stage_begin(page, "translate")
            try:
                page.typeset_blocks = []
                idx = page_pos.get(page.page_id, 0)
                prev_page = all_pages[idx - 1] if idx > 0 else None
                next_page = all_pages[idx + 1] if idx < len(all_pages) - 1 else None
                prev_text = self._page_source_lines(prev_page) if prev_page else []
                next_text = self._page_source_lines(next_page) if next_page else []
                history_lines = self._history_context_lines(all_pages, idx)

                batch_sources: List[str] = []
                batch_contexts: List[Dict[str, Any]] = []
                batch_regions: List[TextRegion] = []

                for region in page.text_regions:
                    source_text = (region.text or "").strip()
                    if not source_text:
                        continue
                    source_text, _ = self._apply_pre_dict(source_text)
                    if not source_text:
                        continue
                    context = {
                        "prev": prev_text,
                        "next": next_text,
                        "history": history_lines,
                        "role": region.role,
                        "orientation": region.orientation,
                        "page_id": page.page_id,
                        "region_id": region.id,
                    }
                    batch_sources.append(source_text)
                    batch_contexts.append(context)
                    batch_regions.append(region)

                if batch_sources:
                    translations = backend.translate_batch(batch_sources, batch_contexts)
                else:
                    translations = []

                if len(translations) != len(batch_regions):
                    raise PipelineError(
                        f"Translation batch size mismatch for page {page.page_id}: "
                        f"sources={len(batch_regions)}, outputs={len(translations)}"
                    )

                for region, region_translation in zip(batch_regions, translations):
                    region_translation = (region_translation or "").strip()
                    if not region_translation.strip():
                        continue
                    region_translation, _ = self._apply_post_dict(region_translation)
                    if not region_translation:
                        continue
                    region.source = "translated"
                    block_bbox = region.bbox
                    if region_translation.strip().lower().startswith("here goes"):
                        x, y, w, h = block_bbox
                        target_w = max(w, int(h * 0.75))
                        page_w = page.width or (x + target_w)
                        nx = x - (target_w - w) // 2
                        nx = max(0, min(max(0, page_w - target_w), nx))
                        block_bbox = (nx, y, target_w, h)
                    block_orientation = "horizontal"
                    block = TypesetBlock(
                        region_id=region.id,
                        text=region_translation,
                        bbox=block_bbox,
                        orientation=block_orientation,
                        font_family="Comic Neue",
                        font_size=max(8, int(min(block_bbox[2], block_bbox[3]) * 0.22)),
                    )
                    page.typeset_blocks.append(block)

                self._translated_page_cache[page.page_id] = [
                    b.text.strip() for b in page.typeset_blocks if (b.text or "").strip()
                ]
                backend_debug = backend.drain_debug_events()
                if backend_debug:
                    debug_file = self.paths["logs"] / f"translate_{page.page_id}.json"
                    atomic_write_json(
                        debug_file,
                        {
                            "page_id": page.page_id,
                            "history_context": history_lines,
                            "events": backend_debug,
                        },
                    )
                    page.output_paths["translate_debug"] = str(debug_file)
                self._stage_end(page, "translate", True, started_ts)
            except Exception as exc:  # noqa: BLE001
                self._stage_end(
                    page,
                    "translate",
                    False,
                    started_ts,
                    error_code=exc.__class__.__name__,
                    error_message=str(exc),
                )

    def _mask_page(self, page: PageManifest) -> None:
        masks: List[InpaintMask] = []
        use_blocks = os.getenv("MTRANSLATE_MASK_SOURCE", "regions").strip().lower() == "blocks"
        dilation_ratio = env_float("MTRANSLATE_MASK_DILATION_RATIO", 0.10, min_value=0.0, max_value=1.5)
        dilation_offset = env_int("MTRANSLATE_MASK_DILATION_OFFSET", 0, min_value=-200, max_value=200)
        dilation_min = env_int("MTRANSLATE_MASK_DILATION_MIN", 4, min_value=0, max_value=512)
        dilation_max = env_int("MTRANSLATE_MASK_DILATION_MAX", 120, min_value=dilation_min, max_value=2048)
        if use_blocks and page.typeset_blocks:
            sources = [(b.region_id, b.bbox, 0.95) for b in page.typeset_blocks]
        else:
            sources = [(r.id, r.bbox, r.confidence) for r in page.text_regions]

        for region_id, bbox, conf in sources:
            x, y, w, h = bbox
            dil = int(min(w, h) * dilation_ratio) + dilation_offset
            dil = max(dilation_min, min(dilation_max, dil))
            masks.append(
                InpaintMask(
                    region_id=region_id,
                    bbox=(max(0, x - dil), max(0, y - dil), w + dil * 2, h + dil * 2),
                    dilation=dil,
                    confidence=float(conf),
                    path=None,
                )
            )
        page.masks = masks
        mask_file = self.paths["masks"] / f"{page.page_id}.json"
        atomic_write_json(mask_file, [asdict(m) for m in masks])
        page.output_paths["masks"] = str(mask_file)

    def _inpaint_page(self, page: PageManifest) -> None:
        if not page.normalized_path:
            raise PipelineError("Missing normalized path")
        src = Path(page.normalized_path)
        dst = self.paths["inpaint"] / f"{page.page_id}.png"
        self.manifest.notes["inpaint_backend"] = self.inpaint_selection.backend
        self.inpaint_selection.backend_obj.inpaint(src=src, masks=page.masks, dst=dst)
        page.output_paths["inpaint"] = str(dst)

    def _typeset_pages(self, pages: List[PageManifest]) -> None:
        todo = [p for p in pages if p.stages["typeset"].status != "done"]
        if not todo:
            return

        started_ts: Dict[str, float] = {}
        for p in todo:
            started_ts[p.page_id] = self._stage_begin(p, "typeset")

        tasks = []
        for p in todo:
            inpaint = p.output_paths.get("inpaint") or p.normalized_path
            output = str(self.paths["typeset"] / f"{p.page_id}.png")
            tasks.append(
                {
                    "input": inpaint,
                    "output": output,
                    "blocks": [asdict(b) for b in p.typeset_blocks],
                }
            )

        try:
            native_typeset_batch(tasks, self.font_path)
            for p in todo:
                out = str(self.paths["typeset"] / f"{p.page_id}.png")
                if not Path(out).exists():
                    raise PipelineError(f"Typeset output missing: {out}")
                p.output_paths["typeset"] = out
                self._stage_end(p, "typeset", True, started_ts[p.page_id])
        except Exception as exc:  # noqa: BLE001
            for p in todo:
                if p.stages["typeset"].status == "running":
                    self._stage_end(
                        p,
                        "typeset",
                        False,
                        started_ts[p.page_id],
                        error_code=exc.__class__.__name__,
                        error_message=str(exc),
                    )

    def _compose_page(self, page: PageManifest) -> None:
        src = Path(page.output_paths.get("typeset") or page.output_paths.get("inpaint") or page.normalized_path or page.input_path)
        dst = self.paths["exports"] / f"{page.page_id}.png"
        copy_file(src, dst)
        page.output_paths["compose"] = str(dst)

    def _export_page(self, page: PageManifest) -> None:
        export_root = Path(self.output_path)
        export_root.mkdir(parents=True, exist_ok=True)
        src = Path(page.output_paths.get("compose") or page.output_paths.get("typeset") or page.normalized_path or page.input_path)
        if "folder" in self.manifest.export:
            dst = export_root / f"{page.page_id}.png"
            copy_file(src, dst)
            page.output_paths["final_page"] = str(dst)

    def _finalize_pdf(self, pages: List[PageManifest]) -> None:
        if "pdf" in self.manifest.export:
            export_root = Path(self.output_path)
            export_root.mkdir(parents=True, exist_ok=True)
            pdf_path = export_root / "translated.pdf"
            images = [
                p.output_paths.get("final_page") or p.output_paths.get("compose")
                for p in sorted(pages, key=lambda p: p.index)
            ]
            images = [p for p in images if p]
            try:
                native_images_to_pdf(images, str(pdf_path))
                self.manifest.notes["pdf_output"] = str(pdf_path)
                self.manifest.notes.pop("pdf_error", None)
            except Exception as exc:  # noqa: BLE001
                self.manifest.notes["pdf_error"] = str(exc)

    def run(self) -> JobManifest:
        self._set_job_status("running")
        pages = [self.manifest.pages[k] for k in sorted(self.manifest.pages)]

        if not pages:
            raise PipelineError("No pages selected for execution")

        stage_funcs = {
            "ingest": lambda p: self._run_stage_for_page(p, "ingest", self._ingest_page),
            "ocr": lambda p: self._ocr_stage([p]),
            "vlm_refine": lambda p: self._run_stage_for_page(p, "vlm_refine", self._vlm_refine_page),
            "semantic_group": lambda p: self._run_stage_for_page(p, "semantic_group", self._semantic_group_page),
            "translate": lambda p: self._translate_pages([p]),
            "mask": lambda p: self._run_stage_for_page(p, "mask", self._mask_page),
            "inpaint": lambda p: self._run_stage_for_page(p, "inpaint", self._inpaint_page),
            "typeset": lambda p: self._typeset_pages([p]),
            "compose": lambda p: self._run_stage_for_page(p, "compose", self._compose_page),
            "export": lambda p: self._run_stage_for_page(p, "export", self._export_page),
        }

        for page in sorted(pages, key=lambda p: p.index):
            for stage in STAGES:
                stage_funcs[stage](page)
                if page.stages[stage].status == "failed":
                    break

        self._finalize_pdf(list(self.manifest.pages.values()))

        failed = [
            p
            for p in self.manifest.pages.values()
            if any(st.status == "failed" for st in p.stages.values())
        ]

        if failed:
            self._set_job_status("failed", failed_pages=[p.page_id for p in failed])
        else:
            self._set_job_status("done")

        return self.manifest

def load_manifest(job_id: str) -> JobManifest:
    path = _manifest_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown job id: {job_id}")
    return job_from_dict(read_json(path))

def audit_job(job_id: str) -> Dict[str, Any]:
    manifest = load_manifest(job_id)
    expected_outputs = {
        "ingest": "normalized",
        "ocr": "ocr",
        "mask": "masks",
        "inpaint": "inpaint",
        "typeset": "typeset",
        "compose": "compose",
    }
    findings: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    pages = sorted(manifest.pages.values(), key=lambda p: p.index)

    for page in pages:
        page_findings: List[str] = []
        page_warnings: List[str] = []
        failed_seen = False
        for stage in STAGES:
            state = page.stages.get(stage)
            if state is None:
                page_findings.append(f"missing stage state: {stage}")
                continue
            if state.status not in {"pending", "running", "done", "failed", "skipped"}:
                page_findings.append(f"invalid stage status '{state.status}' for {stage}")
            if failed_seen and state.status == "done":
                page_findings.append(f"stage {stage} completed after an earlier stage failed")
            output_key = expected_outputs.get(stage)
            if state.status == "done" and output_key:
                out = page.output_paths.get(output_key)
                if not out:
                    page_findings.append(f"missing output path '{output_key}' after stage {stage}")
                elif not Path(out).exists():
                    page_findings.append(f"output file missing on disk for '{output_key}': {out}")
            if state.status == "failed" and not state.error_message:
                page_warnings.append(f"missing error message for failed stage {stage}")
            if state.status == "failed":
                failed_seen = True

        if "folder" in manifest.export and page.stages.get("export") and page.stages["export"].status == "done":
            final_page = page.output_paths.get("final_page")
            if not final_page:
                page_findings.append("export stage done but final_page output is missing")
            elif not Path(final_page).exists():
                page_findings.append(f"exported page missing on disk: {final_page}")

        if page_findings:
            findings.append({"page_id": page.page_id, "issues": page_findings})
        if page_warnings:
            warnings.append({"page_id": page.page_id, "issues": page_warnings})

    if "pdf" in manifest.export:
        pdf_out = str(manifest.notes.get("pdf_output", "")).strip()
        if manifest.status == "done":
            if not pdf_out:
                findings.append({"job": "pdf", "issues": ["missing pdf_output in manifest notes"]})
            elif not Path(pdf_out).exists():
                findings.append({"job": "pdf", "issues": [f"missing PDF file on disk: {pdf_out}"]})

    return {
        "job_id": manifest.job_id,
        "status": manifest.status,
        "checks": {
            "pages_checked": len(pages),
            "findings": len(findings),
            "warnings": len(warnings),
            "ok": len(findings) == 0,
        },
        "findings": findings,
        "warnings": warnings,
    }


def summarize_manifest(manifest: JobManifest) -> Dict[str, Any]:
    pages = list(manifest.pages.values())
    total = len(pages)
    done = 0
    failed = 0
    running = 0
    pending = 0

    for page in pages:
        has_failed = any(st.status == "failed" for st in page.stages.values())
        has_running = any(st.status == "running" for st in page.stages.values())
        export_done = page.stages["export"].status == "done"
        if has_failed:
            failed += 1
        elif export_done:
            done += 1
        elif has_running:
            running += 1
        else:
            pending += 1

    return {
        "job_id": manifest.job_id,
        "status": manifest.status,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "input_path": manifest.input_path,
        "output_path": manifest.output_path,
        "export": manifest.export,
        "pages": {
            "total": total,
            "done": done,
            "failed": failed,
            "running": running,
            "pending": pending,
        },
        "notes": manifest.notes,
    }
