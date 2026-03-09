# MTranslate

Offline manga translation pipeline for local runs.

## Scope

- End-to-end flow: `ingest -> ocr -> vlm_refine -> semantic_group -> translate -> mask -> inpaint -> typeset -> compose -> export`
- Per-job manifests and artifacts under `.mtranslate_data/jobs/<job_id>`
- Per-series glossary under `.mtranslate_data/glossaries/<series_id>.yaml`
- Regex pre/post dictionaries through `--pre-dict`, `--post-dict`, `MTRANSLATE_PRE_DICT`, and `MTRANSLATE_POST_DICT`

This branch is currently CLI/runtime focused. The old Swift review UI and Docker workflow are not part of the tracked runtime path.

## Backend Support

Translation:

- `vllm`: any vLLM-compatible model path or repo id
- `mlx_lm`: any MLX-compatible model path or repo id
- `external`: any custom LLM wrapper exposed as a command

Infill:

- `diffusion`: any diffusers-compatible inpaint or img2img checkpoint
- `copy`: pass-through mode with no model-driven infill
- `external`: any custom infill command

Gemma 3 MLX and Pony Diffusion XL remain working example defaults, not hard product requirements.

## Python

Recommended version: `3.12.9`

```bash
python -m venv .venv_local
source .venv_local/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Example Pulls

Example model pulls for the currently tested path:

```bash
python -m mtranslate.cli models pull --profile hq
python -m mtranslate.cli models pull-translate --repo mlx-community/gemma-3-4b-it-qat-4bit
python -m mtranslate.cli models pull-inpaint --repo Runware/Pony_Diffusion_V6_XL
```

You can replace those repo IDs with any other compatible model you want to use.

## Configuration

Generic LLM env vars:

```bash
export MTRANSLATE_LLM_BACKEND=vllm
export MTRANSLATE_LLM_MODEL=/path/to/your/model
export MTRANSLATE_LLM_PLUGIN=vllm_mlx
export MTRANSLATE_LLM_TEMPERATURE=0.0
export MTRANSLATE_LLM_MAX_TOKENS=220
```

Generic infill env vars:

```bash
export MTRANSLATE_INFILL_BACKEND=diffusion
export MTRANSLATE_INFILL_MODEL=/path/to/your/infill_model
export MTRANSLATE_INPAINT_PROMPT="clean page, no text, preserve artwork"
```

Backward-compatible aliases still work:

- `MTRANSLATE_TRANSLATE_BACKEND`
- `MTRANSLATE_VLLM_MODEL`
- `MTRANSLATE_VLLM_PLUGIN`
- `MTRANSLATE_INPAINT_BACKEND`
- `MTRANSLATE_INPAINT_MODEL`

## Run

Example with explicit backend overrides:

```bash
./scripts/mtranslate.sh run \
  --input path/to/input_pages \
  --output outputs/run_local \
  --export folder \
  --translate-backend vllm \
  --translate-model /path/to/llm \
  --inpaint-backend diffusion \
  --inpaint-model /path/to/infill_model
```

Direct MLX path:

```bash
./scripts/mtranslate.sh run \
  --input path/to/input_pages \
  --output outputs/run_mlx \
  --translate-backend mlx_lm \
  --translate-model /path/to/mlx_model \
  --inpaint-backend copy
```

## External Backend Contract

External translator:

- Set `MTRANSLATE_LLM_BACKEND=external`
- Set `MTRANSLATE_LLM_COMMAND="your_command"` or `MTRANSLATE_TRANSLATE_COMMAND="your_command"`
- The command receives JSON on stdin:

```json
{
  "prompt": "full translation prompt",
  "model": "/path/or/repo",
  "temperature": 0.0,
  "top_p": 0.95,
  "max_tokens": 220,
  "repetition_penalty": 1.02
}
```

- Return either plain text on stdout or JSON with one of: `text`, `translation`, `output`, `result`

External infill:

- Set `MTRANSLATE_INFILL_BACKEND=external` or `MTRANSLATE_INPAINT_BACKEND=external`
- Set `MTRANSLATE_INFILL_COMMAND="your_command"` or `MTRANSLATE_INPAINT_COMMAND="your_command"`
- The command receives JSON on stdin:

```json
{
  "src": "/abs/source.png",
  "dst": "/abs/output.png",
  "model": "/path/or/repo",
  "masks": [
    {
      "region_id": "ocr_0",
      "bbox": [0, 0, 10, 10],
      "dilation": 4,
      "confidence": 0.9,
      "path": null
    }
  ],
  "prompt": "clean page",
  "negative_prompt": "",
  "steps": 28,
  "guidance": 6.0,
  "strength": 0.95
}
```

- The command must write the requested `dst` file, or return JSON with `output`, `dst`, or `result` pointing to a generated file

## Verification

Verify the currently tested vLLM-backed path:

```bash
./scripts/mtranslate.sh models verify-vllm --plugin vllm_mlx
./scripts/mtranslate.sh models verify-vllm \
  --plugin vllm_mlx \
  --smoke \
  --model .mtranslate_data/models/mlx_community_gemma_3_4b_it_qat_4bit
```

Verify a local inpaint model directory:

```bash
./scripts/mtranslate.sh models verify-inpaint --path /path/to/infill_model
```

Check or audit a job:

```bash
./scripts/mtranslate.sh status --job <job_id>
./scripts/mtranslate.sh audit --job <job_id>
```

## Notes

- Context is condensed before translation using neighboring lines and recent page history.
- Glossary prompts are filtered per region so each series keeps its own terminology without bloating prompts.
- Diffusers checkpoints without native inpaint weights still work through masked img2img compositing.
- There is no automated test suite in the repository right now; validation is done through smoke runs and job audits.
