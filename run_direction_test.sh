#!/usr/bin/env bash
set -euo pipefail

direction="${1:?direction required}"
output="${2:?output required}"

set +u
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
set -u
export PYTHONPATH=/workspace/vendor:/workspace/turntable:${PYTHONPATH:-}

python3 -u /workspace/turntable_reconstruction.py \
  --angles 0,180 \
  --pause 1 \
  --speed 5000 \
  --direction "${direction}" \
  --output "${output}"
