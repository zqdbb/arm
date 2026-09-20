# Point Cloud Mesh Surface Analysis

Python project scaffold for industrial point cloud cleanup, surface reconstruction, and mesh quality analysis.

## What it does

- Loads common point cloud inputs such as `.ply` and `.xyz`
- Cleans and downsamples raw scans
- Detects the dominant plane and basic cluster structure
- Reconstructs a triangle mesh with PyVista surface reconstruction, Delaunay, or convex hull methods
- Reports mesh quality metrics with Trimesh
- Exports a cleaned point cloud, reconstructed mesh, and JSON report

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
make compile
make test
```

Run the CLI:

```bash
.venv/bin/python -m point_cloud_mesh_surface_analysis \
  --input data/raw/sample.ply \
  --output outputs \
  --method poisson
```

Outputs are written to:

- `outputs/meshes/` for reconstructed geometry
- `outputs/reports/` for JSON summaries
- `data/processed/` for cleaned point clouds

## Project layout

- `src/point_cloud_mesh_surface_analysis/` - package code
- `tests/` - synthetic pipeline tests
- `scripts/validate_scaffold.py` - repository health checks
- `docs/architecture.md` - pipeline design notes
- `docs/workflow.md` - operator workflow for scan processing

## Notes

- This repository assumes a local virtual environment because the host system uses an externally managed Python installation.
- The package targets Python 3.10+ and is validated against the local interpreter on this machine.
- Mesh volume and mass-like properties are only meaningful when the output mesh is watertight.