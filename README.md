# MTranslate

Offline manga translation pipeline for local runs.

## Current Scope

- End-to-end flow: `ingest -> ocr -> vlm_refine -> semantic_group -> translate -> mask -> inpaint -> typeset -> compose -> export`
- Automated pipeline only in this branch
- Per-job manifests and artifacts under `.mtranslate_data/jobs/<job_id>`
- Per-series glossary at `.mtranslate_data/glossaries/<series_id>.yaml`
- Regex dictionaries supported through `--pre-dict`, `--post-dict`, `MTRANSLATE_PRE_DICT`, and `MTRANSLATE_POST_DICT`

## Supported Backends

- OCR: `paddle` (default) or `fastdeploy`
- Translation: `vllm` with `VLLM_PLUGINS=vllm_mlx`
- Model: Gemma 3 MLX weights (`mlx-community/gemma-3-4b-it-qat-4bit`)
- Inpaint: `diffusion` only, using Pony Diffusion XL / SDXL-compatible weights

`vision` OCR, Docker workflows, and the old Swift review UI are not part of the current branch state.

## Python

Recommended version: `3.12.9`

```bash
python -m venv .venv_local
source .venv_local/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Models

Pull the default translation and inpaint models:

```bash
python -m mtranslate.cli models pull --profile hq
python -m mtranslate.cli models pull-translate --repo mlx-community/gemma-3-4b-it-qat-4bit
python -m mtranslate.cli models pull-inpaint --repo Runware/Pony_Diffusion_V6_XL
```

Verify the translation stack:

```bash
./scripts/mtranslate.sh models verify-vllm --plugin vllm_mlx
./scripts/mtranslate.sh models verify-vllm \
  --plugin vllm_mlx \
  --smoke \
  --model .mtranslate_data/models/mlx_community_gemma_3_4b_it_qat_4bit
```

Verify the inpaint model:

```bash
./scripts/mtranslate.sh models verify-inpaint --path .mtranslate_data/models/sdxl_inpaint
```

## Run

Minimal local smoke run:

```bash
export MTRANSLATE_TRANSLATE_BACKEND=vllm
export MTRANSLATE_VLLM_PLUGIN=vllm_mlx
export MTRANSLATE_VLLM_MODEL=.mtranslate_data/models/mlx_community_gemma_3_4b_it_qat_4bit
export MTRANSLATE_OCR_BACKEND=paddle
export MTRANSLATE_INPAINT_BACKEND=diffusion
export MTRANSLATE_INPAINT_MODEL=.mtranslate_data/models/sdxl_inpaint

./scripts/mtranslate.sh run \
  --input path/to/input_pages \
  --output outputs/run_local \
  --export folder
```

Check status or audit a job:

```bash
./scripts/mtranslate.sh status --job <job_id>
./scripts/mtranslate.sh audit --job <job_id>
```

## Notes

- Context is condensed before translation using neighboring lines and recent page history.
- Glossary prompts are filtered per region so each series keeps its own terminology without bloating prompts.
- Pony checkpoints that ship as base SDXL are used through masked img2img compositing when native inpaint weights are not present.
- There is no automated test suite in the repository right now; validation is done through smoke runs and job audits.
