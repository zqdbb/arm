#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$ROOT_DIR/../.venv/bin/python"

"$PYTHON_BIN" "$ROOT_DIR/simulate_8cam_reconstruction.py" \
  --output "$ROOT_DIR/output_4cam" \
  --depth-model ideal \
  --camera-layout four

"$PYTHON_BIN" "$ROOT_DIR/simulate_8cam_reconstruction.py" \
  --output "$ROOT_DIR/output_4cam_spec_noise" \
  --depth-model spec_noise \
  --camera-layout four
