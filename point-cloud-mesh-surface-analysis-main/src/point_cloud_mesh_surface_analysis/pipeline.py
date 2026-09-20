from __future__ import annotations

import json
from pathlib import Path

from .io_utils import load_point_cloud, write_point_cloud, write_triangle_mesh
from .meshing import reconstruct_mesh
from .metrics import mesh_metrics
from .models import AnalysisResult, PreprocessConfig, ReconstructionConfig
from .processing import (
    cluster_summary,
    detect_dominant_plane,
    preprocess_point_cloud,
    summarize_point_cloud,
)


def run_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    preprocess_config: PreprocessConfig | None = None,
    reconstruction_config: ReconstructionConfig | None = None,
) -> AnalysisResult:
    preprocess_config = preprocess_config or PreprocessConfig()
    reconstruction_config = reconstruction_config or ReconstructionConfig()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    processed_dir = output_dir / "data" / "processed"
    mesh_dir = output_dir / "outputs" / "meshes"
    report_dir = output_dir / "outputs" / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    point_cloud = load_point_cloud(input_path)
    processed_cloud = preprocess_point_cloud(point_cloud, preprocess_config)
    plane = detect_dominant_plane(processed_cloud, preprocess_config)
    clusters = cluster_summary(processed_cloud, preprocess_config)
    mesh = reconstruct_mesh(processed_cloud, reconstruction_config)
    mesh_summary = mesh_metrics(mesh)

    base_name = input_path.stem
    processed_path = processed_dir / f"{base_name}_processed.ply"
    mesh_path = mesh_dir / f"{base_name}_{reconstruction_config.method}.ply"
    report_path = report_dir / f"{base_name}_{reconstruction_config.method}.json"

    write_point_cloud(processed_path, processed_cloud)
    write_triangle_mesh(mesh_path, mesh)

    result = AnalysisResult(
        point_cloud=summarize_point_cloud(processed_cloud),
        dominant_plane=plane,
        clusters=clusters,
        mesh=mesh_summary,
        files={
            "processed_point_cloud": str(processed_path),
            "mesh": str(mesh_path),
            "report": str(report_path),
        },
    )

    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result
