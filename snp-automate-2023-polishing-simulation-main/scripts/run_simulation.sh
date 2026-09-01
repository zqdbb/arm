#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CURRENT_UID="$(id -u):$(id -g)"
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
mkdir -p "$ROOT/runtime/snp_home/snp/meshes"
docker compose -f "$ROOT/docker/compose.sim.yml" pull
docker compose -f "$ROOT/docker/compose.sim.yml" up -d
docker compose -f "$ROOT/docker/compose.sim.yml" ps
docker compose -f "$ROOT/docker/compose.sim.yml" logs --tail=100 snp_automate_2023_sim || true
