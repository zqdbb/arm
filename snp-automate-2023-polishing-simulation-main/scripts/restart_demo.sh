#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CURRENT_UID="$(id -u):$(id -g)"
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
docker rm -f snp_automate_2023_sim >/dev/null 2>&1 || true
docker compose -f "$ROOT/docker/compose.sim.yml" up -d
docker compose -f "$ROOT/docker/compose.sim.yml" ps
