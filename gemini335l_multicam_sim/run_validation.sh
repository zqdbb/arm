#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT_DIR/../.venv/bin/python"

"$PYTHON_BIN" "$ROOT_DIR/simulate_8cam_reconstruction.py" \
  --output output_ideal \
  --depth-model ideal

"$PYTHON_BIN" "$ROOT_DIR/simulate_8cam_reconstruction.py" \
  --output output_spec_noise \
  --depth-model spec_noise
