from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import PreprocessConfig, ReconstructionConfig
from .pipeline import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process industrial point clouds into meshes and surface-analysis reports.",
    )
    parser.add_argument("--input", required=True, help="Path to the input point cloud file")
    parser.add_argument("--output", default=".", help="Directory that receives outputs")
    parser.add_argument(
        "--method",
        choices=["surface", "delaunay3d", "convex_hull"],
        default="surface",
        help="Mesh reconstruction method",
    )
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--plane-threshold", type=float, default=0.01)
    parser.add_argument("--cluster-eps", type=float, default=0.03)
    parser.add_argument("--cluster-min-points", type=int, default=15)
    parser.add_argument("--surface-neighbor-size", type=int, default=20)
    parser.add_argument("--sample-spacing", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preprocess = PreprocessConfig(
        voxel_size=args.voxel_size,
        plane_distance_threshold=args.plane_threshold,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
    )
    reconstruction = ReconstructionConfig(
        method=args.method,
        surface_neighbor_size=args.surface_neighbor_size,
        sample_spacing=args.sample_spacing,
        alpha=args.alpha,
    )
    result = run_analysis(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        preprocess_config=preprocess,
        reconstruction_config=reconstruction,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0
