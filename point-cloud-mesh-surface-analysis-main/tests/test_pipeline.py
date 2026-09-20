from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from point_cloud_mesh_surface_analysis.models import PreprocessConfig, ReconstructionConfig
from point_cloud_mesh_surface_analysis.pipeline import run_analysis


def make_plane_with_box_points() -> np.ndarray:
    plane_x, plane_y = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.2, 0.2, 20))
    plane_z = np.zeros_like(plane_x)
    plane = np.column_stack((plane_x.ravel(), plane_y.ravel(), plane_z.ravel()))

    box = pv.Box(bounds=(-0.04, 0.04, -0.03, 0.03, 0.01, 0.05)).triangulate()
    box_points = box.points
    return np.vstack([plane, box_points])


class PipelineTests(unittest.TestCase):
    def test_run_analysis_exports_report_and_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "fixture.ply"
            cloud = make_plane_with_box_points()
            pv.PolyData(cloud).save(input_path)

            result = run_analysis(
                input_path=input_path,
                output_dir=tmp_path,
                preprocess_config=PreprocessConfig(
                    voxel_size=0.01,
                    plane_distance_threshold=0.008,
                    cluster_eps=0.03,
                    cluster_min_points=10,
                ),
                reconstruction_config=ReconstructionConfig(method="surface", surface_neighbor_size=12),
            )

            self.assertGreater(result.point_cloud["point_count"], 0)
            self.assertGreater(result.dominant_plane["inlier_ratio"], 0.2)
            self.assertGreater(result.mesh["vertex_count"], 0)
            self.assertTrue(Path(result.files["mesh"]).exists())
            self.assertTrue(Path(result.files["report"]).exists())

            payload = json.loads(Path(result.files["report"]).read_text(encoding="utf-8"))
            self.assertIn("mesh", payload)

    def test_alpha_reconstruction_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "fixture_alpha.ply"
            cloud = make_plane_with_box_points()
            pv.PolyData(cloud).save(input_path)

            result = run_analysis(
                input_path=input_path,
                output_dir=tmp_path,
                preprocess_config=PreprocessConfig(voxel_size=0.01),
                reconstruction_config=ReconstructionConfig(method="convex_hull"),
            )

            self.assertGreater(result.mesh["face_count"], 0)
            self.assertIn("surface_area", result.mesh)


if __name__ == "__main__":
    unittest.main()
