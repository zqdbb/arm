#!/usr/bin/env python3
"""End-to-end simulation of vehicle localization and paint-path transfer.

The system stores a reference vehicle point cloud and paint TCP poses in the
vehicle-template frame.  A synthetic live vehicle is moved inside the calibrated
four-camera workcell, its fused cloud is registered against the reference, and
the saved TCP poses are transformed into both workcell and robot-base frames.

This script validates rigid pose transfer.  It does not claim robot IK,
collision avoidance, process timing, or real paint deposition validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from register_template_to_scan import load_template, prepare_cloud


def transform_xyz(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def make_pose(translation, yaw_deg) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def rotation_error_deg(transform: np.ndarray) -> float:
    value = np.clip((np.trace(transform[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def pca_yaw(points: np.ndarray) -> float:
    centered = points[:, :2] - np.mean(points[:, :2], axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, int(np.argmax(values))]
    return float(math.atan2(axis[1], axis[0]))


def refine_icp(source, target, initial):
    transform = np.asarray(initial, dtype=np.float64)
    result = None
    for voxel, distance in ((0.10, 0.40), (0.06, 0.24), (0.035, 0.12), (0.020, 0.055)):
        source_level = prepare_cloud(source, voxel)
        target_level = prepare_cloud(target, voxel)
        result = o3d.pipelines.registration.registration_icp(
            source_level,
            target_level,
            distance,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=120
            ),
        )
        transform = result.transformation
    return result


def alignment_score(source, target, transform, threshold=0.035):
    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(transform)
    source_distance = np.asarray(aligned.compute_point_cloud_distance(target))
    target_distance = np.asarray(target.compute_point_cloud_distance(aligned))
    return {
        "source_coverage": float(np.mean(source_distance <= threshold)),
        "target_coverage": float(np.mean(target_distance <= threshold)),
        "mean_source_distance_m": float(np.mean(source_distance)),
        "p95_source_distance_m": float(np.quantile(source_distance, 0.95)),
        "score": float(
            0.5 * np.mean(source_distance <= threshold)
            + 0.5 * np.mean(target_distance <= threshold)
        ),
    }


def register_reference_to_live(reference, live):
    source_points = np.asarray(reference.points)
    target_points = np.asarray(live.points)
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    yaw_delta = pca_yaw(target_points) - pca_yaw(source_points)
    candidates = []
    for flip in (0.0, math.pi):
        initial = np.eye(4)
        initial[:3, :3] = Rotation.from_euler("z", yaw_delta + flip).as_matrix()
        initial[:3, 3] = target_center - initial[:3, :3] @ source_center
        result = refine_icp(reference, live, initial)
        quality = alignment_score(reference, live, result.transformation)
        candidates.append((quality["score"], result, quality, initial))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best, quality, initial = candidates[0]
    return best, quality, initial, [
        {"score": item[0], "fitness": float(item[1].fitness),
         "rmse_m": float(item[1].inlier_rmse)} for item in candidates
    ]


def tool_rotation(surface_normal, tangent):
    normal = np.asarray(surface_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    z_axis = -normal  # tool spray axis points from TCP toward the panel
    x_axis = np.asarray(tangent, dtype=np.float64)
    x_axis -= z_axis * np.dot(x_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(helper, z_axis)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def cast_segment(scene, origins, directions, name, standoff):
    origins = np.asarray(origins, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    rays = np.hstack((origins, directions))
    result = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    distance = result["t_hit"].numpy()
    normals = result["primitive_normals"].numpy()
    valid = np.isfinite(distance)
    hits = origins[valid] + directions[valid] * distance[valid, None]
    normals = normals[valid]
    ray_directions = directions[valid]
    if len(hits) < 3:
        return None
    for index in range(len(normals)):
        if np.dot(normals[index], ray_directions[index]) > 0:
            normals[index] *= -1
        normals[index] /= np.linalg.norm(normals[index])
    poses = []
    for index, (hit, normal) in enumerate(zip(hits, normals)):
        if index == 0:
            tangent = hits[1] - hits[0]
        elif index == len(hits) - 1:
            tangent = hits[-1] - hits[-2]
        else:
            tangent = hits[index + 1] - hits[index - 1]
        rotation = tool_rotation(normal, tangent)
        poses.append({
            "position_m": (hit + normal * standoff).tolist(),
            "surface_point_m": hit.tolist(),
            "surface_normal": normal.tolist(),
            "quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
        })
    return {"name": name, "poses": poses}


def generate_paint_path(mesh, standoff=0.28):
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    segments = []

    xs = np.linspace(-2.12, 2.12, 72)
    for side_name, y_origin, y_direction in (("left", 1.55, -1.0), ("right", -1.55, 1.0)):
        for row, z in enumerate(np.linspace(0.42, 1.34, 7)):
            ordered_x = xs if row % 2 == 0 else xs[::-1]
            origins = [[x, y_origin, z] for x in ordered_x]
            directions = [[0.0, y_direction, 0.0] for _ in ordered_x]
            segment = cast_segment(scene, origins, directions,
                                   f"{side_name}_side_row_{row:02d}", standoff)
            if segment:
                segments.append(segment)

    roof_x = np.linspace(-1.72, 1.42, 62)
    for row, y in enumerate(np.linspace(-0.68, 0.68, 6)):
        ordered_x = roof_x if row % 2 == 0 else roof_x[::-1]
        origins = [[x, y, 2.05] for x in ordered_x]
        directions = [[0.0, 0.0, -1.0] for _ in ordered_x]
        segment = cast_segment(scene, origins, directions,
                               f"roof_row_{row:02d}", standoff)
        if segment:
            segments.append(segment)

    ys = np.linspace(-0.82, 0.82, 34)
    for end_name, x_origin, x_direction in (("front", 2.75, -1.0), ("rear", -2.75, 1.0)):
        for row, z in enumerate(np.linspace(0.48, 1.24, 5)):
            ordered_y = ys if row % 2 == 0 else ys[::-1]
            origins = [[x_origin, y, z] for y in ordered_y]
            directions = [[x_direction, 0.0, 0.0] for _ in ordered_y]
            segment = cast_segment(scene, origins, directions,
                                   f"{end_name}_row_{row:02d}", standoff)
            if segment:
                segments.append(segment)

    return {
        "frame": "vehicle_template",
        "tcp_convention": "+Z points from spray TCP toward vehicle surface",
        "standoff_m": standoff,
        "segments": segments,
    }


def transform_path(path, transform, frame):
    rotation_transform = transform[:3, :3]
    transformed = {
        "frame": frame,
        "tcp_convention": path["tcp_convention"],
        "standoff_m": path["standoff_m"],
        "segments": [],
    }
    for segment in path["segments"]:
        output_segment = {"name": segment["name"], "poses": []}
        for pose in segment["poses"]:
            input_rotation = Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix()
            position = transform_xyz(transform, np.asarray([pose["position_m"]]))[0]
            surface = transform_xyz(transform, np.asarray([pose["surface_point_m"]]))[0]
            normal = rotation_transform @ np.asarray(pose["surface_normal"])
            output_rotation = rotation_transform @ input_rotation
            output_segment["poses"].append({
                "position_m": position.tolist(),
                "surface_point_m": surface.tolist(),
                "surface_normal": normal.tolist(),
                "quaternion_xyzw": Rotation.from_matrix(output_rotation).as_quat().tolist(),
            })
        transformed["segments"].append(output_segment)
    return transformed


def path_positions(path):
    return np.asarray([
        pose["position_m"]
        for segment in path["segments"]
        for pose in segment["poses"]
    ], dtype=np.float64)


def write_path_cloud(path, output, color):
    points = path_positions(path)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.paint_uniform_color(color)
    o3d.io.write_point_cloud(str(output), cloud, write_ascii=False)


def write_path_csv(path, output):
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["segment", "index", "x_m", "y_m", "z_m",
                         "qx", "qy", "qz", "qw", "spray_on"])
        for segment in path["segments"]:
            for index, pose in enumerate(segment["poses"]):
                writer.writerow([
                    segment["name"], index, *pose["position_m"],
                    *pose["quaternion_xyzw"], 1,
                ])


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="output_4cam/merged_4cam.ply")
    parser.add_argument("--live-base", default="output_charuco_calibration_4cam/merged_charuco_estimated.ply")
    parser.add_argument("--mesh", default="../models/prius_hybrid/meshes/Hybrid.obj")
    parser.add_argument("--output", default="output_paint_path_transfer")
    parser.add_argument("--live-translation", nargs=3, type=float, default=(0.55, -0.32, 0.04))
    parser.add_argument("--live-yaw-deg", type=float, default=8.0)
    parser.add_argument("--standoff", type=float, default=0.28)
    args = parser.parse_args()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print("[1/7] Loading saved and live point clouds", flush=True)
    reference = o3d.io.read_point_cloud(str(resolve(args.reference)))
    live = o3d.io.read_point_cloud(str(resolve(args.live_base)))
    if not len(reference.points) or not len(live.points):
        raise RuntimeError("Reference or live point cloud is empty")

    true_template_to_world = make_pose(args.live_translation, args.live_yaw_deg)
    live.transform(true_template_to_world)
    points = np.asarray(live.points)
    rng = np.random.default_rng(20260929)
    points += rng.normal(0.0, 0.0015, size=points.shape)
    live.points = o3d.utility.Vector3dVector(points)

    print("[2/7] Registering saved cloud to live fused cloud", flush=True)
    result, quality, initial, candidates = register_reference_to_live(reference, live)
    estimated_template_to_world = result.transformation
    aligned = o3d.geometry.PointCloud(reference)
    aligned.transform(estimated_template_to_world)
    pose_delta = np.linalg.inv(true_template_to_world) @ estimated_template_to_world
    match_threshold = 0.030
    aligned_to_live = np.asarray(aligned.compute_point_cloud_distance(live))
    live_to_aligned = np.asarray(live.compute_point_cloud_distance(aligned))
    visible_aligned = aligned.select_by_index(
        np.flatnonzero(aligned_to_live <= match_threshold).tolist()
    )

    print("[3/7] Loading vehicle mesh and generating saved paint path", flush=True)
    mesh = load_template(resolve(args.mesh))
    template_path = generate_paint_path(mesh, args.standoff)
    print(f"[4/7] Generated {sum(len(x['poses']) for x in template_path['segments'])} TCP poses", flush=True)
    estimated_world_path = transform_path(template_path, estimated_template_to_world, "workcell_world")
    true_world_path = transform_path(template_path, true_template_to_world, "workcell_world_truth")

    # Virtual robot base used only to verify the coordinate-chain calculation.
    world_to_robot_base = np.eye(4)
    world_to_robot_base[:3, 3] = np.array([0.0, 3.20, 0.0])
    robot_path = transform_path(estimated_world_path, world_to_robot_base, "robot_base")

    template_positions = path_positions(template_path)
    estimated_positions = path_positions(estimated_world_path)
    true_positions = path_positions(true_world_path)
    path_error = np.linalg.norm(estimated_positions - true_positions, axis=1)
    estimated_quaternions = [pose["quaternion_xyzw"] for segment in estimated_world_path["segments"] for pose in segment["poses"]]
    true_quaternions = [pose["quaternion_xyzw"] for segment in true_world_path["segments"] for pose in segment["poses"]]
    orientation_error = []
    for estimated_q, true_q in zip(estimated_quaternions, true_quaternions):
        delta = Rotation.from_quat(true_q).inv() * Rotation.from_quat(estimated_q)
        orientation_error.append(np.degrees(np.linalg.norm(delta.as_rotvec())))
    orientation_error = np.asarray(orientation_error)
    standoff_values = np.asarray([
        np.linalg.norm(np.asarray(pose["position_m"]) - np.asarray(pose["surface_point_m"]))
        for segment in estimated_world_path["segments"] for pose in segment["poses"]
    ])

    print("[5/7] Writing point clouds and transferred trajectories", flush=True)
    o3d.io.write_point_cloud(str(output / "reference_saved.ply"), reference, write_ascii=False)
    o3d.io.write_point_cloud(str(output / "live_fused.ply"), live, write_ascii=False)
    o3d.io.write_point_cloud(str(output / "reference_aligned.ply"), aligned, write_ascii=False)
    # Compatibility outputs used by the generic template-registration viewer.
    # These make the matching and paint-transfer pages consume the exact same
    # live pose and the exact same estimated transformation.
    o3d.io.write_point_cloud(str(output / "scene.ply"), live, write_ascii=False)
    o3d.io.write_point_cloud(str(output / "template_initial.ply"), reference, write_ascii=False)
    o3d.io.write_point_cloud(str(output / "template_aligned.ply"), aligned, write_ascii=False)
    o3d.io.write_point_cloud(str(output / "template_visible_aligned.ply"), visible_aligned, write_ascii=False)
    write_path_cloud(template_path, output / "paint_path_template.ply", [0.1, 1.0, 0.35])
    write_path_cloud(estimated_world_path, output / "paint_path_live_world.ply", [1.0, 0.15, 0.85])
    write_path_cloud(true_world_path, output / "paint_path_live_truth.ply", [0.1, 0.9, 1.0])

    for filename, payload in (
        ("paint_path_template.json", template_path),
        ("paint_path_live_world.json", estimated_world_path),
        ("paint_path_robot_base.json", robot_path),
    ):
        (output / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_path_csv(robot_path, output / "paint_path_robot_base.csv")

    print("[6/7] Computing registration and path-transfer report", flush=True)
    report = {
        "purpose": "saved vehicle point cloud + saved paint path -> live fused cloud localization -> path transfer",
        "reference_cloud": str(resolve(args.reference)),
        "live_cloud_source": str(resolve(args.live_base)),
        "true_template_to_world": true_template_to_world.tolist(),
        "estimated_template_to_world": estimated_template_to_world.tolist(),
        "coarse_initial_transform": initial.tolist(),
        "registration": {
            "method": "XY PCA two-hypothesis coarse alignment + multi-scale point-to-plane ICP",
            "icp_fitness": float(result.fitness),
            "icp_inlier_rmse_m": float(result.inlier_rmse),
            **quality,
            "candidate_summary": candidates,
        },
        "pose_error": {
            "translation_m": float(np.linalg.norm(pose_delta[:3, 3])),
            "rotation_deg": rotation_error_deg(pose_delta),
        },
        "paint_path": {
            "segments": len(template_path["segments"]),
            "poses": int(len(template_positions)),
            "standoff_m": args.standoff,
            "mean_transfer_error_m": float(np.mean(path_error)),
            "p95_transfer_error_m": float(np.quantile(path_error, 0.95)),
            "max_transfer_error_m": float(np.max(path_error)),
            "mean_orientation_error_deg": float(np.mean(orientation_error)),
            "p95_orientation_error_deg": float(np.quantile(orientation_error, 0.95)),
            "standoff_mean_m": float(np.mean(standoff_values)),
            "standoff_max_deviation_m": float(np.max(np.abs(standoff_values - args.standoff))),
        },
        "robot_frame": {
            "world_to_robot_base": world_to_robot_base.tolist(),
            "virtual_base_position_world_m": [0.0, -3.20, 0.0],
        },
        "safety_scope": {
            "rigid_path_transfer_validated": True,
            "robot_ik_validated": False,
            "collision_check_validated": False,
            "paint_process_validated": False,
            "execution_enabled": False,
            "note": "Simulation output must not be sent directly to a real robot.",
        },
        "outputs": {
            "reference": "reference_saved.ply",
            "live": "live_fused.ply",
            "aligned_reference": "reference_aligned.ply",
            "template_path": "paint_path_template.json",
            "world_path": "paint_path_live_world.json",
            "robot_path": "paint_path_robot_base.json",
            "robot_path_csv": "paint_path_robot_base.csv",
        },
    }
    (output / "paint_path_transfer_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    registration_report = {
        "template": str(resolve(args.reference)),
        "scene": str(output / "live_fused.ply"),
        "synthetic_pose": True,
        "global_registration": {
            "method": report["registration"]["method"],
            "initial_transform": initial.tolist(),
        },
        "icp_registration": {
            "fitness": float(result.fitness),
            "inlier_rmse_m": float(result.inlier_rmse),
        },
        "quality": {
            "source_points": int(len(aligned.points)),
            "inlier_points": int(np.count_nonzero(aligned_to_live <= match_threshold)),
            "inlier_fraction": float(np.mean(aligned_to_live <= match_threshold)),
            "mean_inlier_distance_m": float(np.mean(aligned_to_live[aligned_to_live <= match_threshold])),
            "p95_inlier_distance_m": float(np.quantile(aligned_to_live[aligned_to_live <= match_threshold], 0.95)),
            "scene_coverage_within_threshold": float(np.mean(live_to_aligned <= match_threshold)),
            "template_visible_fraction_within_threshold": float(np.mean(aligned_to_live <= match_threshold)),
            "match_threshold_m": match_threshold,
        },
        "template_to_scene": estimated_template_to_world.tolist(),
        "known_scene_transform": true_template_to_world.tolist(),
        "synthetic_pose_error": report["pose_error"],
        "shared_with_paint_transfer": True,
    }
    (output / "registration_report.json").write_text(
        json.dumps(registration_report, indent=2), encoding="utf-8"
    )
    print("[7/7] End-to-end paint-path transfer simulation complete", flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
