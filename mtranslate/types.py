"""Core types and manifest models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageState:
    name: str
    status: str = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retries: int = 0


@dataclass
class TextRegion:
    id: str
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h (top-left origin)
    polygon: List[Tuple[int, int]]
    orientation: str  # horizontal | vertical
    role: str  # dialogue | narration | sfx
    confidence: float
    source: str = "ocr"


@dataclass
class TypesetBlock:
    region_id: str
    text: str
    bbox: Tuple[int, int, int, int]
    orientation: str
    font_family: str
    font_size: int
    line_spacing: float = 1.15
    align: str = "center"


@dataclass
class InpaintMask:
    region_id: str
    bbox: Tuple[int, int, int, int]
    dilation: int
    confidence: float
    path: Optional[str] = None


@dataclass
class PageManifest:
    page_id: str
    index: int
    input_path: str
    normalized_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: Optional[str] = None
    status: str = "pending"
    stages: Dict[str, StageState] = field(default_factory=dict)
    output_paths: Dict[str, str] = field(default_factory=dict)
    text_regions: List[TextRegion] = field(default_factory=list)
    typeset_blocks: List[TypesetBlock] = field(default_factory=list)
    masks: List[InpaintMask] = field(default_factory=list)

    def ensure_stages(self, names: List[str]) -> None:
        for name in names:
            if name not in self.stages:
                self.stages[name] = StageState(name=name)


@dataclass
class JobManifest:
    job_id: str
    created_at: str
    updated_at: str
    status: str
    input_path: str
    output_path: str
    export: List[str]
    glossary_path: Optional[str]
    notes: Dict[str, Any] = field(default_factory=dict)
    pages: Dict[str, PageManifest] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        job_id: str,
        input_path: str,
        output_path: str,
        export: List[str],
        glossary_path: Optional[str],
    ) -> "JobManifest":
        now = utc_now_iso()
        return cls(
            job_id=job_id,
            created_at=now,
            updated_at=now,
            status="queued",
            input_path=input_path,
            output_path=output_path,
            export=export,
            glossary_path=glossary_path,
            notes={},
            pages={},
        )

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_stage_state(data: Dict[str, Any]) -> StageState:
    return StageState(**data)


def _to_text_region(data: Dict[str, Any]) -> TextRegion:
    data = dict(data)
    data["bbox"] = tuple(data["bbox"])
    data["polygon"] = [tuple(p) for p in data.get("polygon", [])]
    return TextRegion(**data)


def _to_typeset_block(data: Dict[str, Any]) -> TypesetBlock:
    data = dict(data)
    data["bbox"] = tuple(data["bbox"])
    return TypesetBlock(**data)


def _to_inpaint_mask(data: Dict[str, Any]) -> InpaintMask:
    data = dict(data)
    data["bbox"] = tuple(data["bbox"])
    return InpaintMask(**data)


def page_from_dict(data: Dict[str, Any]) -> PageManifest:
    stages = {k: _to_stage_state(v) for k, v in data.get("stages", {}).items()}
    text_regions = [_to_text_region(v) for v in data.get("text_regions", [])]
    typeset_blocks = [_to_typeset_block(v) for v in data.get("typeset_blocks", [])]
    masks = [_to_inpaint_mask(v) for v in data.get("masks", [])]
    return PageManifest(
        page_id=data["page_id"],
        index=data["index"],
        input_path=data["input_path"],
        normalized_path=data.get("normalized_path"),
        width=data.get("width"),
        height=data.get("height"),
        mime_type=data.get("mime_type"),
        status=data.get("status", "pending"),
        stages=stages,
        output_paths=data.get("output_paths", {}),
        text_regions=text_regions,
        typeset_blocks=typeset_blocks,
        masks=masks,
    )


def job_from_dict(data: Dict[str, Any]) -> JobManifest:
    pages = {k: page_from_dict(v) for k, v in data.get("pages", {}).items()}
    return JobManifest(
        job_id=data["job_id"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        status=data["status"],
        input_path=data["input_path"],
        output_path=data["output_path"],
        export=list(data.get("export", [])),
        glossary_path=data.get("glossary_path"),
        notes=dict(data.get("notes", {})),
        pages=pages,
    )
