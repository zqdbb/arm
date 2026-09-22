#!/usr/bin/env python3
"""Register a vehicle template mesh to a multi-camera scene point cloud.

The scene cloud is expected to be in the calibrated world frame. The script
uses FPFH/RANSAC for a coarse pose and multi-scale point-to-plane ICP for the
final template-to-scene transform. ``--synthetic-pose`` applies a known pose
to the scene first, making the pipeline measurable without a real vehicle.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d

from simulate_8cam_reconstruction import load_prius_asset


def load_template(path: Path) -> o3d.geometry.TriangleMesh:
    if path.suffix.lower() == ".obj" and path.name.lower() == "hybrid.obj":
        mesh, _, _ = load_prius_asset(path)
        return mesh
    if path.suffix.lower() in {".glb", ".gltf"}:
        model = o3d.io.read_triangle_model(str(path))
        mesh = o3d.geometry.TriangleMesh()
        for item in model.meshes:
            mesh += item.mesh
    else:
        mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Template mesh has no triangles: {path}")
    mesh.compute_vertex_normals()
    return mesh


def make_transform(translation, rotation_deg):
    rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(rotation_deg))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def prepare_cloud(cloud, voxel):
    cloud = cloud.voxel_down_sample(voxel)
    radius = voxel * 2.5
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    cloud.normalize_normals()
    return cloud


def fpfh(cloud, voxel):
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )


def global_registration(source, target, voxel):
    source_down = prepare_cloud(source, voxel)
    target_down = prepare_cloud(target, voxel)
    source_feature = fpfh(source_down, voxel)
    target_feature = fpfh(target_down, voxel)
    distance = voxel * 2.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_feature,
        target_feature,
        True,
        distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result, source_down, target_down


def refine_registration(source, target, initial, voxel):
    transform = initial
    result = None
    for scale in (voxel * 2.0, voxel, voxel * 0.5):
        source_level = prepare_cloud(source, scale)
        target_level = prepare_cloud(target, scale)
        result = o3d.pipelines.registration.registration_icp(
            source_level,
            target_level,
            scale * 2.0,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=80
            ),
        )
        transform = result.transformation
    return result


def registration_quality(source, target, transform, threshold):
    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(transform)
    distances = np.asarray(aligned.compute_point_cloud_distance(target))
    inliers = distances <= threshold
    values = distances[inliers]
    return {
        "source_points": int(len(aligned.points)),
        "inlier_points": int(np.count_nonzero(inliers)),
        "inlier_fraction": float(np.mean(inliers)) if len(inliers) else 0.0,
        "mean_inlier_distance_m": float(np.mean(values)) if len(values) else None,
        "p95_inlier_distance_m": float(np.quantile(values, 0.95)) if len(values) else None,
        "max_inlier_distance_m": float(np.max(values)) if len(values) else None,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="../models/prius_hybrid/meshes/Hybrid.obj")
    parser.add_argument("--scene", default="output_ideal/merged_6cam.ply")
    parser.add_argument("--output", default="output_registration")
    parser.add_argument("--voxel", type=float, default=0.02)
    parser.add_argument("--template-points", type=int, default=120000)
    parser.add_argument("--synthetic-pose", action="store_true")
    args = parser.parse_args()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    template_path = resolve(args.template)
    scene_path = resolve(args.scene)
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)

    template_mesh = load_template(template_path)
    template = template_mesh.sample_points_uniformly(args.template_points)
    template.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel * 2.5, max_nn=40)
    )
    scene = o3d.io.read_point_cloud(str(scene_path))
    if len(scene.points) == 0:
        raise RuntimeError(f"Scene point cloud is empty: {scene_path}")
    true_transform = np.eye(4, dtype=np.float64)
    if args.synthetic_pose:
        true_transform = make_transform([0.24, -0.16, 0.08], [0.0, 0.0, 6.0])
        scene.transform(true_transform)

    coarse, _, _ = global_registration(template, scene, args.voxel)
    fine = refine_registration(template, scene, coarse.transformation, args.voxel)
    quality = registration_quality(template, scene, fine.transformation, args.voxel * 1.5)

    aligned = o3d.geometry.PointCloud(template)
    aligned.transform(fine.transformation)
    o3d.io.write_point_cloud(str(output / "scene.ply"), scene, write_ascii=False)
    o3d.io.write_point_cloud(str(output / "template_aligned.ply"), aligned, write_ascii=False)
    o3d.io.write_triangle_mesh(str(output / "template.mesh.ply"), template_mesh, write_ascii=False)

    report = {
        "template": str(template_path),
        "scene": str(scene_path),
        "voxel_m": args.voxel,
        "synthetic_pose": args.synthetic_pose,
        "global_registration": {
            "fitness": float(coarse.fitness),
            "inlier_rmse_m": float(coarse.inlier_rmse),
        },
        "icp_registration": {
            "fitness": float(fine.fitness),
            "inlier_rmse_m": float(fine.inlier_rmse),
        },
        "quality": quality,
        "template_to_scene": fine.transformation.tolist(),
        "known_scene_transform": true_transform.tolist() if args.synthetic_pose else None,
    }
    if args.synthetic_pose:
        error = np.linalg.inv(true_transform) @ fine.transformation
        report["synthetic_pose_error"] = {
            "translation_m": float(np.linalg.norm(error[:3, 3])),
            "rotation_deg": float(
                math.degrees(math.acos(np.clip((np.trace(error[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
            ),
        }
    (output / "registration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
