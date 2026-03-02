"""OCR backend selection and PaddleOCR/FastDeploy integration."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .native import native_ocr_batch


@dataclass
class OCRResult:
    by_path: Dict[str, List[Dict[str, Any]]]
    backend: str
    model_root: Optional[str] = None


_PADDLE_OCR_INSTANCE = None
_PADDLE_OCR_LOCK = threading.Lock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_paddle_ocr():
    global _PADDLE_OCR_INSTANCE  # noqa: PLW0603
    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE
    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_INSTANCE is not None:
            return _PADDLE_OCR_INSTANCE
        from paddleocr import PaddleOCR  # type: ignore

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        base_kwargs = {
            "use_textline_orientation": _env_bool("MTRANSLATE_OCR_AUTO_ROTATE", True),
            "lang": "japan",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
        }
        optional_kwargs: Dict[str, Any] = {}
        detection_size = os.getenv("MTRANSLATE_OCR_DETECTION_SIZE", "").strip()
        if detection_size:
            optional_kwargs["text_det_limit_side_len"] = int(detection_size)
        unclip_ratio = os.getenv("MTRANSLATE_OCR_UNCLIP_RATIO", "").strip()
        if unclip_ratio:
            optional_kwargs["det_db_unclip_ratio"] = float(unclip_ratio)

        try:
            _PADDLE_OCR_INSTANCE = PaddleOCR(
                **base_kwargs,
                **optional_kwargs,
            )
        except TypeError:
            # Keep runtime compatible across PaddleOCR minor API changes.
            _PADDLE_OCR_INSTANCE = PaddleOCR(
                **base_kwargs,
            )
        return _PADDLE_OCR_INSTANCE


def _select_model_files(model_dir: Path) -> tuple[str, str, str]:
    """Return (model_path, params_path, format) for a FastDeploy model directory.

    Supports both legacy *.pdmodel + *.pdiparams and newer
    inference.json + inference.pdiparams layouts.
    """
    pdmodel = next(model_dir.glob("*.pdmodel"), None)
    pdip = next(model_dir.glob("*.pdiparams"), None)
    if pdmodel and pdip:
        return str(pdmodel), str(pdip), "pdmodel"

    json_model = model_dir / "inference.json"
    json_params = model_dir / "inference.pdiparams"
    if json_model.exists() and json_params.exists():
        return str(json_model), str(json_params), "json"

    available = sorted(p.name for p in model_dir.glob("*"))
    raise RuntimeError(
        f"model files not found in {model_dir} "
        f"(expected *.pdmodel+*.pdiparams or inference.json+inference.pdiparams; "
        f"found: {', '.join(available) or 'none'})"
    )


def _region_from_quad(idx: int, text: str, score: float, points: List[List[float]]) -> Dict[str, Any]:
    xs = [int(round(p[0])) for p in points]
    ys = [int(round(p[1])) for p in points]
    x = max(0, min(xs))
    y = max(0, min(ys))
    w = max(1, max(xs) - x)
    h = max(1, max(ys) - y)
    orientation = "vertical" if h > w else "horizontal"
    poly = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return {
        "id": f"ocr_{idx}",
        "text": text,
        "bbox": [x, y, w, h],
        "polygon": poly,
        "orientation": orientation,
        "confidence": float(score),
        "source": "paddleocr",
    }


def _ocr_thresholds() -> tuple[float, int]:
    min_score = float(os.getenv("MTRANSLATE_OCR_MIN_SCORE", "0.6"))
    min_side = int(os.getenv("MTRANSLATE_OCR_MIN_SIDE", "12"))
    return min_score, min_side


def _run_paddleocr(paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    try:
        _ = _get_paddle_ocr()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"paddleocr import failed: {exc}") from exc

    ocr = _get_paddle_ocr()
    out: Dict[str, List[Dict[str, Any]]] = {}

    min_score, min_side = _ocr_thresholds()

    for p in paths:
        page_regions: List[Dict[str, Any]] = []
        results = ocr.predict(p)
        records = results if isinstance(results, list) else [results]
        for rec_out in records:
            polys = rec_out.get("rec_polys", []) if hasattr(rec_out, "get") else []
            texts = rec_out.get("rec_texts", []) if hasattr(rec_out, "get") else []
            scores = rec_out.get("rec_scores", []) if hasattr(rec_out, "get") else []
            for idx, quad in enumerate(polys):
                text = str(texts[idx]).strip() if idx < len(texts) else ""
                score = float(scores[idx]) if idx < len(scores) else 0.0
                if not text:
                    continue
                points = [[float(pt[0]), float(pt[1])] for pt in quad]
                region = _region_from_quad(idx, text, score, points)
                w = int(region["bbox"][2])
                h = int(region["bbox"][3])
                if score < min_score or min(w, h) < min_side:
                    continue
                page_regions.append(region)
        out[p] = page_regions

    return out


def _run_fastdeploy_paddle(paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Run PP-OCR via FastDeploy Python runtime.

    This expects model directories under:
      $MTRANSLATE_FASTDEPLOY_MODELS/{det,cls,rec}
    """

    try:
        import fastdeploy as fd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"fastdeploy import failed: {exc}") from exc

    root_raw = os.getenv("MTRANSLATE_FASTDEPLOY_MODELS", "").strip()
    if not root_raw:
        raise RuntimeError("MTRANSLATE_FASTDEPLOY_MODELS is not set")
    root = Path(root_raw).expanduser()

    det_dir = root / "det"
    cls_dir = root / "cls"
    rec_dir = root / "rec"
    for d in [det_dir, cls_dir, rec_dir]:
        if not d.exists():
            raise RuntimeError(f"missing FastDeploy OCR model dir: {d}")

    det_model, det_params, _ = _select_model_files(det_dir)
    cls_model, cls_params, _ = _select_model_files(cls_dir)
    rec_model, rec_params, _ = _select_model_files(rec_dir)

    option = fd.RuntimeOption()
    option.use_paddle_backend()

    det = fd.vision.ocr.DBDetector(det_model, det_params, runtime_option=option)
    cls = fd.vision.ocr.Classifier(cls_model, cls_params, runtime_option=option)
    rec = fd.vision.ocr.Recognizer(rec_model, rec_params, label_path="", runtime_option=option)
    ppocr = fd.vision.ocr.PPOCRv3(det_model=det, cls_model=cls, rec_model=rec)

    out: Dict[str, List[Dict[str, Any]]] = {}
    min_score, min_side = _ocr_thresholds()
    for p in paths:
        pred = ppocr.predict(p)
        page_regions: List[Dict[str, Any]] = []
        boxes = getattr(pred, "boxes", []) or []
        texts = getattr(pred, "text", []) or []
        scores = getattr(pred, "rec_scores", []) or []
        for idx, box in enumerate(boxes):
            text = str(texts[idx]).strip() if idx < len(texts) else ""
            score = float(scores[idx]) if idx < len(scores) else 0.0
            if not text:
                continue
            points = [[float(pt[0]), float(pt[1])] for pt in box]
            region = _region_from_quad(idx, text, score, points)
            w = int(region["bbox"][2])
            h = int(region["bbox"][3])
            if score < min_score or min(w, h) < min_side:
                continue
            page_regions.append(region)
        out[p] = page_regions

    return out


def extract_regions_batch(image_paths: List[str]) -> OCRResult:
    backend = os.getenv("MTRANSLATE_OCR_BACKEND", "paddle").strip().lower()

    if backend in {"vision", "native"}:
        return OCRResult(by_path=native_ocr_batch(image_paths), backend="vision")

    if backend in {"paddlefast", "fastdeploy", "fastdeploy_paddle"}:
        root = Path(os.getenv("MTRANSLATE_FASTDEPLOY_MODELS", "")).expanduser()
        model_root = str(root) if str(root) else None
        return OCRResult(
            by_path=_run_fastdeploy_paddle(image_paths),
            backend="fastdeploy+paddle",
            model_root=model_root,
        )

    if backend == "paddle":
        return OCRResult(by_path=_run_paddleocr(image_paths), backend="paddleocr")

    raise ValueError(f"Unknown OCR backend: {backend}. Supported: paddle, fastdeploy, vision")
