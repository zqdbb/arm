#!/usr/bin/env python3
"""Eight fixed Gemini 335L cameras: RGB-D rendering, point-cloud merge and TSDF."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import pybullet as p
from PIL import Image


BODY_COLOR = [0.10, 0.32, 0.67, 1.0]
TIRE_COLOR = [0.035, 0.04, 0.05, 1.0]
HUB_COLOR = [0.52, 0.55, 0.60, 1.0]


def as_matrix(values):
    """PyBullet exposes OpenGL matrices in column-major order."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4, order="F")


def projection_from_intrinsics(width, height, fx, fy, cx, cy, near, far):
    left = -cx * near / fx
    right = (width - cx) * near / fx
    bottom = -(height - cy) * near / fy
    top = cy * near / fy
    return as_matrix(p.computeProjectionMatrix(left, right, bottom, top, near, far))


def camera_pose(position, target):
    view = as_matrix(p.computeViewMatrix(position, target, [0.0, 0.0, 1.0]))
    # OpenGL camera: x-right/y-up/z-backward. Open3D: x-right/y-down/z-forward.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    world_to_camera = flip @ view
    return view, world_to_camera


def suv_body_mesh():
    """Create a watertight SUV-like body from longitudinal cross-sections."""
    sections = [
        (-2.40, 0.77, 0.40, 0.78),
        (-2.25, 0.90, 0.27, 1.02),
        (-1.65, 0.92, 0.25, 1.30),
        (-1.05, 0.90, 0.25, 1.66),
        (0.78, 0.90, 0.25, 1.66),
        (1.30, 0.91, 0.25, 1.27),
        (2.18, 0.88, 0.27, 1.04),
        (2.40, 0.74, 0.42, 0.76),
    ]
    vertices = []
    for x, half_width, lower, upper in sections:
        vertices.extend([
            [x, -half_width, lower],
            [x, half_width, lower],
            [x, -half_width, upper],
            [x, half_width, upper],
        ])
    triangles = []
    for i in range(len(sections) - 1):
        a = 4 * i
        b = 4 * (i + 1)
        triangles.extend([
            [a, b + 1, b], [a, a + 1, b + 1],          # underside
            [a + 2, b + 2, b + 3], [a + 2, b + 3, a + 3],  # top
            [a, b, b + 2], [a, b + 2, a + 2],          # right side (-y)
            [a + 1, b + 3, b + 1], [a + 1, a + 3, b + 3],  # left side (+y)
        ])
    last = 4 * (len(sections) - 1)
    triangles.extend([
        [0, 2, 3], [0, 3, 1],
        [last, last + 1, last + 3], [last, last + 3, last + 2],
    ])
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(BODY_COLOR[:3])
    return mesh


def cylinder_mesh(radius, width, center, color, resolution=48):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=width, resolution=resolution, split=4
    )
    rotation = mesh.get_rotation_matrix_from_xyz((math.pi / 2.0, 0.0, 0.0))
    mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
    mesh.translate(center)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color[:3])
    return mesh


def box_mesh(size, center, color):
    mesh = o3d.geometry.TriangleMesh.create_box(*size)
    mesh.translate(np.asarray(center) - np.asarray(size) / 2.0)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color[:3])
    return mesh


def make_procedural_vehicle_mesh():
    parts = [(suv_body_mesh(), BODY_COLOR)]
    for x in (-1.55, 1.55):
        for y in (-0.88, 0.88):
            parts.append((cylinder_mesh(0.36, 0.16, [x, y, 0.36], TIRE_COLOR), TIRE_COLOR))
            outer_y = y + math.copysign(0.086, y)
            parts.append((cylinder_mesh(0.21, 0.012, [x, outer_y, 0.36], HUB_COLOR), HUB_COLOR))
    # Bumpers and roof rails add geometry useful for checking alignment/detail.
    parts.extend([
        (box_mesh([0.10, 1.46, 0.12], [2.38, 0.0, 0.55], HUB_COLOR), HUB_COLOR),
        (box_mesh([0.10, 1.46, 0.12], [-2.38, 0.0, 0.55], HUB_COLOR), HUB_COLOR),
        (box_mesh([1.65, 0.045, 0.045], [-0.05, -0.71, 1.70], HUB_COLOR), HUB_COLOR),
        (box_mesh([1.65, 0.045, 0.045], [-0.05, 0.71, 1.70], HUB_COLOR), HUB_COLOR),
    ])
    combined = o3d.geometry.TriangleMesh()
    for part, _ in parts:
        combined += part
    combined.compute_vertex_normals()
    return combined, parts, {
        "type": "procedural SUV-like validation target",
        "source": "generated locally",
    }


def load_car_concept_asset(path):
    """Load, orient and dimension the downloaded Khronos Car Concept model."""
    model = o3d.io.read_triangle_model(str(path))
    if not model.meshes:
        raise RuntimeError(f"No meshes found in vehicle asset: {path}")

    # Source axes are X=width, Y=height, Z=length. Simulation uses
    # X=length, Y=width, Z=height. Normalize to a real passenger-car envelope.
    mapped_vertices = []
    for item in model.meshes:
        points = np.asarray(item.mesh.vertices)
        if len(points):
            mapped_vertices.append(points[:, [2, 0, 1]])
    all_points = np.vstack(mapped_vertices)
    source_min = all_points.min(axis=0)
    source_max = all_points.max(axis=0)
    source_extent = source_max - source_min
    target_extent = np.array([4.80, 2.05, 1.55], dtype=np.float64)
    axis_scale = target_extent / source_extent
    source_center = 0.5 * (source_min + source_max)

    parts = []
    part_names = []
    combined = o3d.geometry.TriangleMesh()
    for item in model.meshes:
        mesh = copy.deepcopy(item.mesh)
        points = np.asarray(mesh.vertices)
        mapped = points[:, [2, 0, 1]]
        mapped = (mapped - source_center) * axis_scale
        mapped[:, 2] += target_extent[2] / 2.0
        mesh.vertices = o3d.utility.Vector3dVector(mapped)
        material = model.materials[item.material_idx]
        color = np.clip(np.asarray(material.base_color, dtype=np.float64), 0.0, 1.0)
        color[3] = 1.0  # Depth simulation treats windows and paint as opaque geometry.
        if float(np.max(color[:3])) < 0.012:
            color[:3] = 0.018
        mesh.paint_uniform_color(color[:3])
        mesh.compute_vertex_normals()
        combined += mesh
        parts.append((mesh, color.tolist()))
        part_names.append(item.mesh_name)
    combined.compute_vertex_normals()
    return combined, parts, {
        "type": "downloaded high-detail concept car",
        "model_name": "Car Concept",
        "source_repository": "KhronosGroup/glTF-Sample-Assets",
        "source_url": "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CarConcept",
        "license": "Creative Commons Attribution 4.0 International",
        "attribution": "2024 Darmstadt Graphics Group GmbH; Eric Chadwick for model and textures",
        "asset_file": str(path),
        "source_bbox_after_axis_mapping": source_extent.tolist(),
        "normalized_bbox_target_m": target_extent.tolist(),
        "axis_scale": axis_scale.tolist(),
        "mesh_parts": part_names,
    }


def add_mesh_to_pybullet(mesh, color):
    vertices = np.asarray(mesh.vertices).tolist()
    indices = np.asarray(mesh.triangles, dtype=np.int32).reshape(-1).tolist()
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=vertices,
        indices=indices,
        rgbaColor=color,
        specularColor=[0.12, 0.12, 0.12],
    )
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual)


def add_vehicle_to_pybullet(parts):
    for mesh, color in parts:
        add_mesh_to_pybullet(mesh, color)


def camera_layout():
    """Four upper and four lower fixed corner cameras from the supplied scheme."""
    cameras = []
    for layer, height, target_z in (("upper", 2.45, 1.06), ("lower", 0.82, 0.72)):
        for x_sign, y_sign, corner in (
            (1.0, 1.0, "front_left"),
            (-1.0, 1.0, "rear_left"),
            (-1.0, -1.0, "rear_right"),
            (1.0, -1.0, "front_right"),
        ):
            position = np.array([3.60 * x_sign, 2.60 * y_sign, height])
            target = np.array([1.05 * x_sign, 0.0, target_z])
            cameras.append({
                "name": f"cam{len(cameras) + 1:02d}_{layer}_{corner}",
                "layer": layer,
                "position": position,
                "target": target,
            })
    return cameras


def accuracy_limit_fraction(depth_m):
    """Piecewise interpolation of the supplied <=1%@2m and <=2%@4m limits."""
    return np.clip(0.005 * depth_m, 0.0025, 0.03)


def apply_spec_noise(depth_m, rng, fill_rate):
    """Specification-derived synthetic noise, not a measured device noise model."""
    result = depth_m.copy()
    valid = result > 0.0
    if not np.any(valid):
        return result, {"valid_input_pixels": 0, "dropped_pixels": 0}

    z = result[valid]
    limit_fraction = accuracy_limit_fraction(z)
    frame_bias_fraction = float(rng.normal(0.0, np.mean(limit_fraction) / 6.0))
    # Treat the published accuracy limit as approximately three standard deviations.
    pixel_sigma_m = z * limit_fraction / 3.0
    result[valid] = z * (1.0 + frame_bias_fraction) + rng.normal(
        0.0, pixel_sigma_m, size=z.shape
    )
    dropout = valid & (rng.random(result.shape) > fill_rate)
    result[dropout] = 0.0
    return result, {
        "valid_input_pixels": int(valid.sum()),
        "dropped_pixels": int(dropout.sum()),
        "frame_bias_fraction": frame_bias_fraction,
        "mean_pixel_sigma_m": float(np.mean(pixel_sigma_m)),
    }


def rotation_from_vector(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def perturb_extrinsic(extrinsic, rng, translation_sigma_m, rotation_sigma_deg):
    camera_to_world = np.linalg.inv(extrinsic)
    translation = rng.normal(0.0, translation_sigma_m, 3)
    rotation_deg = rng.normal(0.0, rotation_sigma_deg, 3)
    estimate = camera_to_world.copy()
    estimate[:3, :3] = camera_to_world[:3, :3] @ rotation_from_vector(np.radians(rotation_deg))
    estimate[:3, 3] += translation
    return np.linalg.inv(estimate), {
        "translation_error_m": translation.tolist(),
        "rotation_error_deg_xyz": rotation_deg.tolist(),
        "translation_norm_m": float(np.linalg.norm(translation)),
        "rotation_norm_deg": float(np.linalg.norm(rotation_deg)),
    }


def point_cloud_from_rgbd(color, depth_m, intrinsic, depth_trunc):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32))),
        depth_scale=1.0,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    return rgbd, o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


def distance_stats(values):
    if len(values) == 0:
        return {"mean_m": None, "median_m": None, "p95_m": None, "max_m": None}
    return {
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p95_m": float(np.quantile(values, 0.95)),
        "max_m": float(np.max(values)),
    }


def pair_overlap_metrics(point_clouds, max_correspondence=0.08):
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    reports = []
    for a, b in pairs:
        pa = point_clouds[a].voxel_down_sample(0.025)
        pb = point_clouds[b].voxel_down_sample(0.025)
        distances = np.asarray(pa.compute_point_cloud_distance(pb))
        overlap = distances[distances <= max_correspondence]
        reports.append({
            "camera_a": a + 1,
            "camera_b": b + 1,
            "source_points": int(len(distances)),
            "overlap_points": int(len(overlap)),
            "overlap_fraction": float(len(overlap) / max(1, len(distances))),
            "overlap_distance": distance_stats(overlap),
        })
    return reports


def evaluate_mesh(reconstruction, truth, visible_truth, sample_count=180000):
    if len(reconstruction.triangles) == 0:
        return {
            "reconstruction_to_truth": distance_stats([]),
            "truth_to_reconstruction": distance_stats([]),
            "completeness": {},
            "full_mesh_completeness": {},
        }
    reconstructed_points = reconstruction.sample_points_uniformly(sample_count)
    truth_points = truth.sample_points_uniformly(sample_count)
    reconstruction_scene = o3d.t.geometry.RaycastingScene()
    reconstruction_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(reconstruction))
    truth_scene = o3d.t.geometry.RaycastingScene()
    truth_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(truth))
    r2t = truth_scene.compute_distance(o3d.core.Tensor(
        np.asarray(reconstructed_points.points), dtype=o3d.core.Dtype.Float32
    )).numpy()
    t2r = reconstruction_scene.compute_distance(o3d.core.Tensor(
        np.asarray(truth_points.points), dtype=o3d.core.Dtype.Float32
    )).numpy()
    visible_t2r = reconstruction_scene.compute_distance(o3d.core.Tensor(
        np.asarray(visible_truth.points), dtype=o3d.core.Dtype.Float32
    )).numpy()
    thresholds = (0.01, 0.02, 0.05)
    return {
        "reconstruction_to_truth": distance_stats(r2t),
        "truth_to_reconstruction": distance_stats(t2r),
        "completeness_definition": "camera-visible ground-truth exterior sampled from the eight ideal depth views",
        "completeness": {
            f"within_{int(t * 1000)}mm": float(np.mean(visible_t2r <= t)) for t in thresholds
        },
        "full_mesh_completeness_note": "Includes hidden interior, underside and occluded mechanical surfaces.",
        "full_mesh_completeness": {
            f"within_{int(t * 1000)}mm": float(np.mean(t2r <= t)) for t in thresholds
        },
        "reconstructed_surface_accuracy": {
            f"within_{int(t * 1000)}mm": float(np.mean(r2t <= t)) for t in thresholds
        },
    }


def write_camera_centers(path, cameras):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray([c["position"] for c in cameras]))
    cloud.colors = o3d.utility.Vector3dVector(np.asarray([
        [1.0, 0.45, 0.05] if c["layer"] == "upper" else [0.1, 0.95, 0.35]
        for c in cameras
    ]))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def load_official_gemini335l_mesh(path, target_triangles=12000):
    """Load Orbbec's official ROS mesh and express it in depth optical coordinates."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Cannot load official Gemini 335L mesh: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    # Official Xacro visual origin in camera_link (depth frame):
    # xyz = -0.011715 -0.0475 -0.014311.
    vertices_ros = np.asarray(mesh.vertices) + np.array([-0.011715, -0.0475, -0.014311])
    # ROS camera frame (x forward, y left, z up) -> optical frame
    # (x right, y down, z forward).
    vertices_optical = np.column_stack((
        -vertices_ros[:, 1],
        -vertices_ros[:, 2],
        vertices_ros[:, 0],
    ))
    mesh.vertices = o3d.utility.Vector3dVector(vertices_optical)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.32, 0.36, 0.42])
    return mesh


def write_official_camera_models(
    asset_path, output_path, enlarged_output_path, local_output_path, poses
):
    local_mesh = load_official_gemini335l_mesh(asset_path)
    o3d.io.write_triangle_mesh(str(local_output_path), local_mesh, write_ascii=False)
    world_mesh = o3d.geometry.TriangleMesh()
    enlarged_world_mesh = o3d.geometry.TriangleMesh()
    for pose in poses:
        instance = copy.deepcopy(local_mesh)
        instance.transform(np.linalg.inv(np.asarray(pose["world_to_camera_true"])))
        world_mesh += instance
        enlarged = copy.deepcopy(local_mesh)
        enlarged.scale(4.0, center=(0.0, 0.0, 0.0))
        enlarged.transform(np.linalg.inv(np.asarray(pose["world_to_camera_true"])))
        enlarged_world_mesh += enlarged
    world_mesh.compute_vertex_normals()
    enlarged_world_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), world_mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(
        str(enlarged_output_path), enlarged_world_mesh, write_ascii=False
    )
    return {
        "source": "Orbbec official OrbbecSDK_ROS2 repository",
        "repository": "https://github.com/orbbec/OrbbecSDK_ROS2",
        "source_mesh": str(asset_path),
        "license": "Apache License 2.0",
        "official_urdf": "assets/gemini335l_official/gemini_335_L_336_L.urdf.xacro",
        "mesh_sha256": "9de399ed805ddb004cacdf7656766d728d0e4fb9e9c83c70a3c12950c7db8303",
        "mesh_extent_m": local_mesh.get_axis_aligned_bounding_box().get_extent().tolist(),
        "display_triangle_count_per_camera": int(len(local_mesh.triangles)),
        "world_mesh_file": output_path.name,
        "enlarged_display_mesh_file": enlarged_output_path.name,
        "enlarged_display_scale": 4.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="gemini335l_spec.json")
    parser.add_argument("--output", default="output_ideal")
    parser.add_argument("--full-resolution", action="store_true")
    parser.add_argument("--depth-model", choices=["ideal", "spec_noise"], default="ideal")
    parser.add_argument("--voxel", type=float, default=0.015)
    parser.add_argument("--trunc", type=float, default=0.060)
    parser.add_argument("--noise-seed", type=int, default=335)
    parser.add_argument("--pose-noise", action="store_true")
    parser.add_argument("--pose-translation-sigma-mm", type=float, default=3.0)
    parser.add_argument("--pose-rotation-sigma-deg", type=float, default=0.10)
    parser.add_argument("--pose-seed", type=int, default=336)
    parser.add_argument(
        "--vehicle-model", choices=["car_concept", "procedural"], default="car_concept"
    )
    parser.add_argument(
        "--vehicle-asset", default="assets/car_concept/CarConcept.glb"
    )
    parser.add_argument(
        "--camera-mesh", default="assets/gemini335l_official/base_link.STL"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    spec_path = Path(args.spec)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    profile = spec["depth"] if args.full_resolution else spec["simulation_resolution"]
    width = int(profile.get("native_width", profile.get("width")))
    height = int(profile.get("native_height", profile.get("height")))
    fx = float(profile.get("fx_px"))
    fy = float(profile.get("fy_px"))
    cx = float(profile.get("cx_px"))
    cy = float(profile.get("cy_px"))
    depth_unit = float(spec["depth"]["depth_unit_m"])
    near, far = 0.17, 8.0

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    for folder in (output / "color", output / "depth", output / "pointclouds"):
        folder.mkdir(parents=True, exist_ok=True)

    if args.vehicle_model == "car_concept":
        asset_path = Path(args.vehicle_asset)
        if not asset_path.is_absolute():
            asset_path = root / asset_path
        truth, parts, vehicle_metadata = load_car_concept_asset(asset_path)
    else:
        truth, parts, vehicle_metadata = make_procedural_vehicle_mesh()
    truth_path = output / "ground_truth_vehicle.ply"
    o3d.io.write_triangle_mesh(str(truth_path), truth, write_ascii=False)
    truth_bbox = truth.get_axis_aligned_bounding_box()
    truth_center = truth_bbox.get_center()
    truth_extent = truth_bbox.get_extent()
    truth_scene = o3d.t.geometry.RaycastingScene()
    truth_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(truth))

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Cannot connect to PyBullet")
    p.resetSimulation()
    add_vehicle_to_pybullet(parts)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    projection = projection_from_intrinsics(width, height, fx, fy, cx, cy, near, far)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    cameras = camera_layout()
    depth_rng = np.random.default_rng(args.noise_seed)
    pose_rng = np.random.default_rng(args.pose_seed)
    frame_reports = []
    camera_clouds_world = []
    visible_truth_clouds = []
    poses = []

    for index, camera in enumerate(cameras):
        view, true_extrinsic = camera_pose(camera["position"], camera["target"])
        _, _, rgba, depth_buffer, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view.reshape(-1, order="F").tolist(),
            projectionMatrix=projection.reshape(-1, order="F").tolist(),
            renderer=p.ER_TINY_RENDERER,
            shadow=0,
            lightDirection=[-1.0, -1.0, 2.0],
        )
        color = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
        depth_buffer = np.asarray(depth_buffer, dtype=np.float64).reshape(height, width)
        depth_m = far * near / (far - (far - near) * depth_buffer)
        depth_m[(depth_buffer >= 0.9999) | ~np.isfinite(depth_m)] = 0.0
        ideal_depth_m = depth_m.copy()
        _, visible_cloud_camera = point_cloud_from_rgbd(color, ideal_depth_m, intrinsic, far)
        visible_cloud_world = o3d.geometry.PointCloud(visible_cloud_camera)
        visible_cloud_world.transform(np.linalg.inv(true_extrinsic))
        visible_truth_clouds.append(visible_cloud_world)
        noise_info = None
        if args.depth_model == "spec_noise":
            depth_m, noise_info = apply_spec_noise(
                depth_m, depth_rng, float(spec["depth"]["fill_rate_at_2m_fraction"])
            )
        depth_raw = np.clip(np.rint(depth_m / depth_unit), 0, 65535).astype(np.uint16)
        depth_m = depth_raw.astype(np.float64) * depth_unit

        integration_extrinsic = true_extrinsic
        pose_error = None
        if args.pose_noise:
            integration_extrinsic, pose_error = perturb_extrinsic(
                true_extrinsic,
                pose_rng,
                args.pose_translation_sigma_mm / 1000.0,
                args.pose_rotation_sigma_deg,
            )

        rgbd, cloud_camera = point_cloud_from_rgbd(color, depth_m, intrinsic, far)
        cloud_world = o3d.geometry.PointCloud(cloud_camera)
        cloud_world.transform(np.linalg.inv(integration_extrinsic))
        camera_clouds_world.append(cloud_world)
        o3d.io.write_point_cloud(
            str(output / "pointclouds" / f"cam{index + 1:02d}_world.ply"),
            cloud_world,
            write_ascii=False,
        )
        volume.integrate(rgbd, intrinsic, integration_extrinsic)
        Image.fromarray(color, mode="RGB").save(output / "color" / f"cam{index + 1:02d}.png")
        Image.fromarray(depth_raw).save(output / "depth" / f"cam{index + 1:02d}.png")

        position = camera["position"]
        surface_distance = truth_scene.compute_distance(o3d.core.Tensor(
            position.reshape(1, 3), dtype=o3d.core.Dtype.Float32
        )).numpy()[0]
        valid_count = int(np.count_nonzero(depth_m))
        frame_reports.append({
            "camera": index + 1,
            "name": camera["name"],
            "layer": camera["layer"],
            "capture_order": index + 1,
            "camera_to_vehicle_center_m": float(np.linalg.norm(position - truth_center)),
            "camera_to_nearest_surface_m": float(surface_distance),
            "valid_pixels": valid_count,
            "valid_pixel_ratio": float(valid_count / (width * height)),
            "point_count": int(len(cloud_world.points)),
            "noise": noise_info,
            "pose_error": pose_error,
        })
        poses.append({
            "camera": index + 1,
            "name": camera["name"],
            "layer": camera["layer"],
            "position_world_m": position.tolist(),
            "target_world_m": camera["target"].tolist(),
            "world_to_camera_true": true_extrinsic.tolist(),
            "world_to_camera_used": integration_extrinsic.tolist(),
        })

    p.disconnect()
    merged = o3d.geometry.PointCloud()
    for cloud in camera_clouds_world:
        merged += cloud
    visible_truth = o3d.geometry.PointCloud()
    for cloud in visible_truth_clouds:
        visible_truth += cloud
    visible_truth = visible_truth.voxel_down_sample(voxel_size=0.008)
    merged_raw_count = len(merged.points)
    merged = merged.voxel_down_sample(voxel_size=0.010)
    o3d.io.write_point_cloud(str(output / "merged_8cam.ply"), merged, write_ascii=False)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    # Remove only very small islands; keep separate wheels and roof rails.
    if len(mesh.triangles):
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        keep_labels = np.flatnonzero(counts >= max(40, int(counts.max() * 0.002)))
        remove = np.flatnonzero(~np.isin(labels, keep_labels))
        mesh.remove_triangles_by_index(remove)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output / "tsdf_mesh.ply"), mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(output / "tsdf_mesh.obj"), mesh, write_ascii=False)
    write_camera_centers(output / "camera_centers.ply", cameras)
    camera_mesh_path = Path(args.camera_mesh)
    if not camera_mesh_path.is_absolute():
        camera_mesh_path = root / camera_mesh_path
    camera_visual = write_official_camera_models(
        camera_mesh_path,
        output / "gemini335l_cameras_world.ply",
        output / "gemini335l_cameras_world_x4.ply",
        root / "assets" / "gemini335l_official" / "gemini335l_official_optical.ply",
        poses,
    )

    reconstruction_bbox = mesh.get_axis_aligned_bounding_box() if len(mesh.vertices) else None
    reconstructed_extent = reconstruction_bbox.get_extent() if reconstruction_bbox else np.zeros(3)
    overlap = pair_overlap_metrics(camera_clouds_world)
    evaluation = evaluate_mesh(mesh, truth, visible_truth)
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    report = {
        "project": "Gemini 335L eight fixed camera simulation",
        "test_definition": (
            "exact fixed extrinsics and no sensor noise" if args.depth_model == "ideal" and not args.pose_noise
            else "specification-derived synthetic depth noise and/or simulated extrinsic error"
        ),
        "important_scope": "Specification values are typical/limit values, not calibration from eight physical units.",
        "camera_model": spec["model"],
        "camera_parameter_type": spec["parameter_type"],
        "camera_parameter_source_file": spec["source_file"],
        "camera_count": 8,
        "camera_visual_model": camera_visual,
        "capture_mode": "fixed cameras, static vehicle, sequential capture cam01 to cam08",
        "infrared_interference_simulated": False,
        "resolution": [width, height],
        "full_resolution": bool(args.full_resolution),
        "intrinsics_px": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "calculated_fov_deg": {"horizontal": hfov, "vertical": vfov},
        "spec_typical_depth_fov_deg": {"horizontal": 90.0, "vertical": 65.0},
        "stereo_baseline_m": float(spec["stereo"]["baseline_m"]),
        "depth_unit_m": depth_unit,
        "depth_model": args.depth_model,
        "noise_model": None if args.depth_model == "ideal" else {
            "type": "specification-derived_not_measured_device_noise",
            "accuracy_limit_interpolation": "1 percent at 2m, 2 percent at 4m; treated as about 3 sigma",
            "fill_rate": spec["depth"]["fill_rate_at_2m_fraction"],
            "seed": args.noise_seed,
        },
        "pose_noise": None if not args.pose_noise else {
            "type": "simulated_not_measured_calibration_error",
            "translation_sigma_mm_per_axis": args.pose_translation_sigma_mm,
            "rotation_sigma_deg_per_axis": args.pose_rotation_sigma_deg,
            "seed": args.pose_seed,
        },
        "tsdf": {"backend": "Open3D ScalableTSDFVolume", "voxel_m": args.voxel, "trunc_m": args.trunc},
        "vehicle": {
            **vehicle_metadata,
            "ground_truth_bbox_m": truth_extent.tolist(),
            "reconstructed_bbox_m": reconstructed_extent.tolist(),
            "bbox_absolute_error_m": np.abs(reconstructed_extent - truth_extent).tolist(),
        },
        "per_camera": frame_reports,
        "point_cloud": {
            "raw_total_points_before_merge_downsample": int(merged_raw_count),
            "merged_points_after_10mm_voxel": int(len(merged.points)),
            "file": "merged_8cam.ply",
        },
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.triangles)),
            "ply_file": "tsdf_mesh.ply",
            "obj_file": "tsdf_mesh.obj",
        },
        "surface_evaluation": evaluation,
        "adjacent_camera_overlap": overlap,
        "files": {
            "ground_truth": truth_path.name,
            "poses": "poses.json",
            "camera_centers": "camera_centers.ply",
            "camera_models": "gemini335l_cameras_world.ply",
            "camera_models_enlarged": "gemini335l_cameras_world_x4.ply",
        },
    }
    (output / "poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
