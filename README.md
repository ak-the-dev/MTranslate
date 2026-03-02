# MTranslate

Deterministic offline manga translation pipeline.

## Runtime model

- Processes **one page at a time** from ingest to export.
- Runs inside repository-local runtime data only:
  - default: `./.mtranslate_data`
  - optional override: `MTRANSLATE_APP_SUPPORT` (must still point inside repo)
- No automatic backend fallbacks.

## Supported backends

- OCR: `paddle` (default) or `fastdeploy`
- Translation: `vllm` (Gemma 3 only)
- Inpaint: `diffusion` (SDXL inpaint only)

Configure with:

```bash
export MTRANSLATE_OCR_BACKEND=paddle
export MTRANSLATE_TRANSLATE_BACKEND=vllm
export MTRANSLATE_VLLM_MODEL=google/gemma-3-4b-it
export MTRANSLATE_INPAINT_BACKEND=diffusion
export MTRANSLATE_INPAINT_MODEL=.mtranslate_data/models/sdxl_inpaint
```

### vLLM translation model (required)

Recommended default for JP->EN manga translation:

- `google/gemma-3-4b-it`

Gemma weights are gated on Hugging Face. Accept the model license and authenticate
before pulling:

```bash
huggingface-cli login
```

If `vllm` is installed in a specific interpreter, point MTranslate to it:

```bash
export MTRANSLATE_PYTHON=/absolute/path/to/python
```

Verify interpreter has vLLM:

```bash
$MTRANSLATE_PYTHON -c "import vllm; print(vllm.__version__)"
```

Verify vLLM wiring:

```bash
./scripts/mtranslate.sh models verify-vllm
```

Optional end-to-end generation smoke test:

```bash
./scripts/mtranslate.sh models verify-vllm --smoke --model google/gemma-3-4b-it
```

Optional tuning:

```bash
export MTRANSLATE_VLLM_TEMPERATURE=0.0
export MTRANSLATE_VLLM_MAX_TOKENS=220
export MTRANSLATE_VLLM_GPU_MEMORY_UTIL=0.85
export MTRANSLATE_VLLM_CONTEXT_LINES=2
export MTRANSLATE_VLLM_CONTEXT_CHARS=260
export MTRANSLATE_VLLM_CONTEXT_LINE_CHARS=88
export MTRANSLATE_VLLM_MIN_SOURCE_CHARS_FOR_CONTEXT=6
export MTRANSLATE_CONTEXT_PAGES=2
export MTRANSLATE_CONTEXT_HISTORY_LINES=24
export MTRANSLATE_VLLM_BATCH_ENABLED=1
export MTRANSLATE_VLLM_BATCH_SIZE=6
export MTRANSLATE_VLLM_BATCH_RETRIES=2
export MTRANSLATE_VLLM_BATCH_SPLIT_DEPTH=3
export MTRANSLATE_VLLM_GLOSSARY_MAX_TERMS=12
export MTRANSLATE_VLLM_ENABLE_REASONING=1
export MTRANSLATE_VLLM_REASONING_MODEL_HINT=gemma
export MTRANSLATE_DEBUG_TRANSLATION=1
```

### SDXL inpaint model (required)

Latest compatible Pony Diffusion XL checkpoint (as of February 26, 2026):
- `Runware/Pony_Diffusion_V6_XL` (Hugging Face last modified: 2026-02-12)

Pull SDXL inpaint weights:

```bash
./scripts/mtranslate.sh models pull-inpaint --repo Runware/Pony_Diffusion_V6_XL
```

Set the local model path:

```bash
export MTRANSLATE_INPAINT_MODEL=.mtranslate_data/models/sdxl_inpaint
export MTRANSLATE_MASK_DILATION_RATIO=0.10
export MTRANSLATE_MASK_DILATION_OFFSET=8
```

The Swift editor initializes this runtime on launch and validates both:
- a Python interpreter that can import `vllm`
- local SDXL model files at `.mtranslate_data/models/sdxl_inpaint`

Note:
- Pony SDXL checkpoints are typically base SDXL checkpoints.
- For text-region infill, MTranslate uses masked img2img compositing within the diffusion backend when native inpaint UNet channels are not present.

## Series Glossaries

- Each input series gets its own glossary file automatically.
- Series id is derived from the input folder name.
- Default path: `.mtranslate_data/glossaries/<series_id>.yaml`
- Use `--glossary` to override for a specific run/create command.
- Translation prompts include only glossary terms relevant to each region.

## Pre/Post Dictionaries

- Optional regex replacement dictionaries can be applied:
  - `MTRANSLATE_PRE_DICT` before translation (OCR cleanup / normalization)
  - `MTRANSLATE_POST_DICT` after translation (style fixes / banned literal cleanup)
- CLI equivalents:
  - `mtranslate run --pre-dict /path/pre.txt --post-dict /path/post.txt`
  - `mtranslate create --pre-dict /path/pre.txt --post-dict /path/post.txt`
- Accepted line formats:
  - `regex<TAB>replacement`
  - `regex => replacement`
  - `regex -> replacement`
  - `regex` (delete match)
- `#` and `//` comments are supported.

## Quickstart

```bash
export MTRANSLATE_TRANSLATE_BACKEND=vllm
export MTRANSLATE_VLLM_MODEL=google/gemma-3-4b-it
export MTRANSLATE_INPAINT_BACKEND=diffusion
export MTRANSLATE_INPAINT_MODEL=.mtranslate_data/models/sdxl_inpaint

./scripts/mtranslate.sh run \
  --input samples/kaigeki_no_kinato_ch1 \
  --output outputs/ch1 \
  --export folder,pdf
```

## Semi-Automatic Review Flow (Manual Infill + Layout)

1. Open the native editor UI:

```bash
cd ui/MTranslateEditor
swift run
```

or from repo root:

```bash
./scripts/run_editor.sh
```

2. In the app:
- Drag and drop a PDF, folder, or image files into the drop zone.
- A new job is created automatically and loaded. No CLI pre-run is required.
- Optionally enter repo root + existing job id to reopen an earlier job.
- Use `Stage Approval` to run exactly one stage at a time (`Run Stage`).
- Inspect output, click `Approve`, then `Approve + Next` to move through:
  `ingest -> ocr -> vlm_refine -> semantic_group -> translate -> mask -> inpaint -> typeset -> compose -> export`.
- `Infill Regions` mode: disable regions you do not want erased/inpainted.
- `Text Layout` mode: edit text, drag/resize text boxes, tweak font size.
- Click `Apply Edits + Re-render Page`.

3. Backend edits are sent through `mtranslate.cli review --apply` and rerender the page from:
- `mask` stage if regions changed
- `typeset` stage if only text layout changed

Approvals are saved per-page at:
- `.mtranslate_data/jobs/<job_id>/review/approvals.json`

Review JSON format accepted by CLI:

```json
{
  "pages": {
    "001": {
      "regions": [
        {"region_id": "ocr_0", "enabled": false},
        {"region_id": "ocr_1", "bbox": [120, 340, 140, 220], "enabled": true}
      ],
      "blocks": [
        {"region_id": "ocr_1", "text": "My line", "bbox": [120, 340, 140, 220], "font_size": 24, "enabled": true}
      ]
    }
  }
}
```

## Model provisioning

```bash
python -m mtranslate.cli models pull --profile hq
python -m mtranslate.cli models pull-translate --repo google/gemma-3-4b-it
python -m mtranslate.cli models verify-translate --path .mtranslate_data/models/google_gemma_3_4b_it
python -m mtranslate.cli models verify-vllm
python -m mtranslate.cli models pull-inpaint --repo Runware/Pony_Diffusion_V6_XL
python -m mtranslate.cli models verify-inpaint --path .mtranslate_data/models/sdxl_inpaint
```

## Repository notes

- Tests were removed from this branch.
- Runtime/cache artifacts should stay out of git.
- Translation debug artifacts (if enabled) are written per page under:
  - `.mtranslate_data/jobs/<job_id>/logs/translate_<page_id>.json`
- OCR tuning knobs:
  - `MTRANSLATE_OCR_DETECTION_SIZE`
  - `MTRANSLATE_OCR_UNCLIP_RATIO`
  - `MTRANSLATE_OCR_AUTO_ROTATE`
