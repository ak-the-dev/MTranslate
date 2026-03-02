#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export MTRANSLATE_REPO_ROOT="$ROOT_DIR"
if [[ -z "${MTRANSLATE_PYTHON:-}" && -x "$HOME/.venv-vllm/bin/python" ]]; then
  export MTRANSLATE_PYTHON="$HOME/.venv-vllm/bin/python"
fi
export MTRANSLATE_TRANSLATE_BACKEND="${MTRANSLATE_TRANSLATE_BACKEND:-vllm}"
export MTRANSLATE_VLLM_MODEL="${MTRANSLATE_VLLM_MODEL:-google/gemma-3-4b-it}"
export MTRANSLATE_VLLM_ENABLE_REASONING="${MTRANSLATE_VLLM_ENABLE_REASONING:-1}"
export MTRANSLATE_VLLM_REASONING_MODEL_HINT="${MTRANSLATE_VLLM_REASONING_MODEL_HINT:-gemma}"
export MTRANSLATE_INPAINT_BACKEND="${MTRANSLATE_INPAINT_BACKEND:-diffusion}"
export MTRANSLATE_INPAINT_MODEL="${MTRANSLATE_INPAINT_MODEL:-$ROOT_DIR/.mtranslate_data/models/sdxl_inpaint}"
cd "$ROOT_DIR/ui/MTranslateEditor"
exec swift run
