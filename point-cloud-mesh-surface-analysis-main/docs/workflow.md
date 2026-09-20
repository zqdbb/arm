# Workflow

## Typical operator flow

1. Place a raw scan in `data/raw/`.
2. Run the CLI against that scan.
3. Review the cleaned point cloud in `data/processed/`.
4. Inspect the mesh in `outputs/meshes/`.
5. Use the JSON report in `outputs/reports/` to review counts, bounding boxes, dominant-plane fit, cluster count, and mesh quality indicators.

## Recommended reconstruction methods

- `surface` for general scan-to-surface reconstruction using PyVista's dedicated surface filter
- `delaunay3d` for exploratory workflows and point sets that benefit from volumetric tetrahedralization before surface extraction
- `convex_hull` for fast bounding-envelope reconstruction when a conservative outer surface is acceptable

## Practical caution

- Dominant-plane detection is useful for fixtures, tables, and reference surfaces, but it should not be treated as a complete tolerance inspection result.
- Cluster count is a coarse segmentation signal, not a production-ready part classifier.