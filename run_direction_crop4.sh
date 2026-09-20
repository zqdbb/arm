#!/usr/bin/env bash
set -e

direction="${1:?direction required}"
output="${2:?output required}"

source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
export PYTHONPATH=/workspace/vendor:/workspace/turntable:${PYTHONPATH:-}

python3 -u /workspace/turntable_reconstruction.py \
  --angles 0,90,180,270 \
  --pause 1 \
  --speed 5000 \
  --direction "${direction}" \
  --output "${output}" \
  --crop_left 220 --crop_top 120 --crop_right 1060 --crop_bottom 680 \
  --depth_max_m 0.60
