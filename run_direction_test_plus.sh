#!/usr/bin/env bash
set -e

chmod +x /workspace/run_direction_test.sh
pkill -9 -f /workspace/rs_publisher.py 2>/dev/null || true
/workspace/run_direction_test.sh 1 /workspace/reconstruction_archive/direction_plus.ply
