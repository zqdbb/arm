import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import open3d as o3d

import scan_recon_v12 as v12


class DummyModel:
    names = {0: 'chair', 1: 'dining table', 2: 'cabinet'}


class V12PipelineTests(unittest.TestCase):
    def test_segmentation_specs_match_requested_models(self):
        self.assertEqual(v12.get_segmentation_spec('chair'), (v12.GENERIC_MODEL, 'chair'))
        self.assertEqual(v12.get_segmentation_spec('table'), (v12.GENERIC_MODEL, 'dining table'))
        self.assertEqual(v12.get_segmentation_spec('cabinet'), (v12.CUSTOM_MODEL, 'cabinet'))
        self.assertEqual(v12.resolve_class_id(DummyModel(), 'dining table'), 1)

    def test_select_views_keeps_angular_separation(self):
        candidates = [
            {'index': 1, 'degree': 5.0, 'confidence': 0.8},
            {'index': 6, 'degree': 30.0, 'confidence': 0.9},
            {'index': 10, 'degree': 50.0, 'confidence': 0.7},
            {'index': 18, 'degree': 90.0, 'confidence': 0.9},
        ]
        selected = v12._select_view_indices(candidates)
        degrees = [next(c['degree'] for c in candidates if c['index'] == i)
                   for i in selected]
        self.assertGreaterEqual(len(selected), 2)
        self.assertTrue(all(abs(a - b) >= 20 for i, a in enumerate(degrees)
                            for b in degrees[i + 1:]))

    def test_validate_capture_dataset_requires_matching_rgbd(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'color').mkdir()
            (root / 'depth').mkdir()
            (root / 'pcd_frames').mkdir()
            (root / 'color/000.png').touch()
            (root / 'depth/000.png').touch()
            (root / 'pcd_frames/frame_000.ply').touch()
            (root / 'color/001.png').touch()
            (root / 'depth/001.png').touch()
            (root / 'pcd_frames/frame_001.ply').touch()
            (root / 'chair_stitched.ply').touch()
            manifest = {
                'schema_version': 1,
                'name': 'chair',
                'complete': True,
                'planned_frames': 2,
                'frames': [
                    {'index': 0, 'color': 'color/000.png', 'depth': 'depth/000.png'},
                    {'index': 1, 'color': 'color/001.png', 'depth': 'depth/001.png'},
                ],
            }
            (root / 'capture_manifest.json').write_text(json.dumps(manifest))
            loaded, stitched = v12.validate_capture_dataset(root, 'chair')
            self.assertEqual(len(loaded['frames']), 2)
            self.assertEqual(stitched, root / 'chair_stitched.ply')

    def test_robust_dimensions_ignore_outliers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'points.ply'
            rng = np.random.default_rng(7)
            points = rng.uniform([0, 0, 0], [0.05, 0.04, 0.08], size=(5000, 3))
            points = np.vstack([points, [[5, 5, 5], [-5, -5, -5]]])
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
            dims = v12.robust_dimensions_from_stitched(path)
            self.assertAlmostEqual(dims['width_m'], 0.048, delta=0.003)
            self.assertAlmostEqual(dims['depth_m'], 0.038, delta=0.003)
            self.assertAlmostEqual(dims['height_m'], 0.077, delta=0.004)

    def test_robust_dimensions_correct_rotated_xy_cloud(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rotated.ply'
            rng = np.random.default_rng(11)
            points = rng.uniform(
                [-0.025, -0.015, 0], [0.025, 0.015, 0.08], size=(10000, 3)
            )
            theta = np.radians(30)
            rotation = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ])
            points[:, :2] = points[:, :2] @ rotation.T
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
            dims = v12.robust_dimensions_from_stitched(path)
            self.assertEqual(dims['dimension_method'], 'robust_xy_min_area_rect')
            self.assertAlmostEqual(dims['width_m'], 0.048, delta=0.003)
            self.assertAlmostEqual(dims['depth_m'], 0.029, delta=0.003)
            self.assertAlmostEqual(abs(dims['xy_yaw_deg']), 30, delta=3)

    def test_robust_dimensions_correct_rotated_near_square_cloud(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rotated_square.ply'
            rng = np.random.default_rng(19)
            points = rng.uniform(
                [-0.02, -0.019, 0], [0.02, 0.019, 0.06], size=(10000, 3)
            )
            theta = np.radians(37)
            rotation = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ])
            points[:, :2] = points[:, :2] @ rotation.T
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
            dims = v12.robust_dimensions_from_stitched(path)
            self.assertAlmostEqual(dims['width_m'], 0.039, delta=0.003)
            self.assertAlmostEqual(dims['depth_m'], 0.037, delta=0.003)

    def test_surface_fit_preserves_triangle_topology(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh_path = root / 'mesh.ply'
            point_path = root / 'points.ply'
            mesh = o3d.geometry.TriangleMesh.create_box(0.05, 0.04, 0.08)
            mesh.compute_vertex_normals()
            self.assertTrue(o3d.io.write_triangle_mesh(str(mesh_path), mesh))
            points = np.asarray(mesh.vertices) + np.array([0.001, 0, 0])
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            extra = mesh.sample_points_uniformly(number_of_points=200)
            cloud += extra
            self.assertTrue(o3d.io.write_point_cloud(str(point_path), cloud))
            stats = v12.conservative_surface_fit(mesh_path, point_path)
            geometry = v12.validate_exported_mesh(mesh_path)
            self.assertEqual(geometry['triangles'], len(mesh.triangles))
            self.assertLessEqual(stats['max_displacement_m'], 0.0015 + 1e-9)

    def test_hyper3d_rejects_fal_mode_before_upload(self):
        blender = mock.Mock()
        blender.send.return_value = {
            'enabled': True,
            'message': 'Hyper3D Rodin integration is enabled. Mode: FAL_AI.',
        }
        views = [
            {'path': '/missing/a.png', 'has_mask': True},
            {'path': '/missing/b.png', 'has_mask': True},
        ]
        with self.assertRaisesRegex(RuntimeError, 'MAIN_SITE'):
            v12.do_hyper3d_generate(blender, views, {}, 'chair')
        blender.send.assert_called_once_with('get_hyper3d_status')

    def test_blender_cleanup_protects_unrelated_meshes(self):
        source = Path(v12.__file__).read_text(encoding='utf-8')
        self.assertIn('protected_mesh_names', source)
        self.assertNotIn("if o.type == 'MESH' and o.name != obj_name", source)

    @mock.patch('scan_recon_v12.subprocess.run')
    def test_final_subprocess_receives_capture_options(self, run):
        v12.run_final_pipeline('cabinet', '/dev/ttyUSB9')
        command = run.call_args.args[0]
        self.assertIn('--save-rgbd', command)
        self.assertIn('/dev/ttyUSB9', command)
        self.assertEqual(command[command.index('--name') + 1], 'cabinet')
        self.assertTrue(run.call_args.kwargs['check'])


if __name__ == '__main__':
    unittest.main()
