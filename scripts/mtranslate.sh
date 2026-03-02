#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${MTRANSLATE_PYTHON:-}" ]]; then
  PYTHON="$MTRANSLATE_PYTHON"
else
  if [[ -x "$ROOT_DIR/.venv_local/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv_local/bin/python"
  elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
  elif [[ -x "/usr/bin/python3" ]]; then
    PYTHON="/usr/bin/python3"
  else
    PYTHON="python3"
  fi
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$ROOT_DIR"
fi

exec "$PYTHON" -m mtranslate.cli "$@"
