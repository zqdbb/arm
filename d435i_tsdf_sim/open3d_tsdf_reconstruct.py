#!/usr/bin/env python3
"""PyBullet D435i RGB-D simulation fused with Open3D ScalableTSDFVolume."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
from PIL import Image


WOOD = [0.38, 0.18, 0.07, 1.0]
WOOD_DARK = [0.24, 0.09, 0.035, 1.0]
METAL = [0.16, 0.18, 0.21, 1.0]


def chair_boxes():
    boxes = [
        ([0.42, 0.42, 0.05], [0, 0, 0.43], WOOD),
        ([0.42, 0.05, 0.45], [0, 0.185, 0.655], WOOD),
        ([0.38, 0.036, 0.036], [0, -0.18, 0.31], WOOD),
        ([0.036, 0.38, 0.036], [-0.18, 0, 0.31], WOOD),
    ]
    for x in (-0.18, 0.18):
        for y in (-0.18, 0.18):
            boxes.append(([0.05, 0.05, 0.43], [x, y, 0.215], WOOD))
    return boxes


def desk_boxes():
    """Writing desk with drawers, lower shelf and a two-level hutch."""
    boxes = [
        # Main desk structure.
        ([0.76, 0.50, 0.045], [0, 0, 0.78], WOOD),
        ([0.65, 0.045, 0.14], [0, -0.205, 0.69], WOOD_DARK),
        ([0.65, 0.045, 0.14], [0, 0.205, 0.69], WOOD_DARK),
        ([0.045, 0.36, 0.14], [-0.335, 0, 0.69], WOOD_DARK),
        ([0.045, 0.36, 0.14], [0.335, 0, 0.69], WOOD_DARK),
        ([0.62, 0.36, 0.035], [0, 0, 0.25], WOOD),
        # Two deep drawers and their proud front panels.
        ([0.29, 0.40, 0.17], [-0.17, -0.005, 0.68], WOOD_DARK),
        ([0.29, 0.40, 0.17], [0.17, -0.005, 0.68], WOOD_DARK),
        ([0.30, 0.035, 0.18], [-0.17, -0.225, 0.68], WOOD),
        ([0.30, 0.035, 0.18], [0.17, -0.225, 0.68], WOOD),
        ([0.11, 0.025, 0.020], [-0.17, -0.245, 0.68], METAL),
        ([0.11, 0.025, 0.020], [0.17, -0.245, 0.68], METAL),
        # Hutch: rear panel, side posts, middle and top shelves.
        ([0.70, 0.035, 0.30], [0, 0.215, 0.952], WOOD_DARK),
        ([0.04, 0.30, 0.30], [-0.33, 0.08, 0.952], WOOD),
        ([0.04, 0.30, 0.30], [0.33, 0.08, 0.952], WOOD),
        ([0.70, 0.30, 0.030], [0, 0.08, 0.94], WOOD),
        ([0.70, 0.30, 0.035], [0, 0.08, 1.105], WOOD),
        ([0.035, 0.30, 0.165], [0, 0.08, 1.0225], WOOD_DARK),
    ]
    for x in (-0.33, 0.33):
        for y in (-0.20, 0.20):
            boxes.append(([0.055, 0.055, 0.76], [x, y, 0.38], WOOD))
    return boxes


def target_config(name):
    if name == "desk":
        return {
            "label": "procedural complex writing desk with drawers and hutch",
            "boxes": desk_boxes(),
            "size": np.array([0.76, 0.50, 1.1225]),
            "target": np.array([0.0, 0.0, 0.56]),
            "radius": 1.15,
            "heights": [0.05, 0.36, 0.68, 1.05, 1.45],
            "far": 3.0,
        }
    return {
        "label": "procedural furniture chair",
        "boxes": chair_boxes(),
        "size": np.array([0.42, 0.42, 0.88]),
        "target": np.array([0.0, 0.0, 0.44]),
        "radius": 0.78,
        "heights": [0.02, 0.22, 0.44, 0.72, 1.02],
        "far": 2.0,
    }


def add_box(size, pos, color):
    half = (np.asarray(size, dtype=np.float64) / 2.0).tolist()
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
    p.createMultiBody(0, collision, visual, basePosition=pos)


def add_furniture(boxes):
    for size, center, color in boxes:
        add_box(size, center, color)


def as_matrix(values):
    # PyBullet exposes OpenGL matrices in column-major order.
    return np.asarray(values, dtype=np.float64).reshape(4, 4, order="F")


def load_camera_parameters(path, args):
    if path is None:
        width, height = args.width, args.height
        fx = 0.5 * width / math.tan(math.radians(args.hfov / 2.0))
        fy = fx
        return {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": width / 2.0 - 0.5,
            "cy": height / 2.0 - 0.5,
            "depth_scale_m_per_unit": 0.001,
            "source": "command-line horizontal FOV approximation",
            "parameter_type": "simulated_approximation",
        }

    camera_path = Path(path).resolve()
    data = json.loads(camera_path.read_text(encoding="utf-8"))
    intrinsics = data.get("intrinsics", data)
    color = data.get("color", data)
    required = {
        "width": color.get("width"),
        "height": color.get("height"),
        "fx": intrinsics.get("fx"),
        "fy": intrinsics.get("fy"),
        "cx": intrinsics.get("cx", intrinsics.get("ppx")),
        "cy": intrinsics.get("cy", intrinsics.get("ppy")),
        "depth_scale_m_per_unit": data.get(
            "depth_scale_m_per_unit", data.get("depth_scale")
        ),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            f"camera parameter file {camera_path} is missing: {', '.join(missing)}"
        )
    required.update({
        "source": data.get("source", str(camera_path)),
        "source_file": str(camera_path),
        "parameter_type": data.get("parameter_type", "saved_camera_parameters"),
        "distortion_model": intrinsics.get("distortion_model"),
        "distortion_coeffs": intrinsics.get("coeffs"),
        "stereo_baseline_m": data.get("stereo", {}).get("baseline_m"),
    })
    return required


def projection_from_intrinsics(width, height, fx, fy, cx, cy, near, far):
    left = -cx * near / fx
    right = (width - cx) * near / fx
    bottom = -(height - cy) * near / fy
    top = cy * near / fy
    return as_matrix(p.computeProjectionMatrix(left, right, bottom, top, near, far))


def camera_pose(cam, target, projection):
    view = as_matrix(p.computeViewMatrix(cam, target, [0, 0, 1]))
    # OpenGL camera is x-right/y-up/z-backward. Open3D uses x-right/y-down/z-forward.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    extrinsic = flip @ view
    return view, projection, extrinsic


def depth_discontinuities(depth_m, threshold_m=0.015):
    valid = depth_m > 0
    edge = np.zeros_like(valid)
    boundary = np.zeros_like(valid)
    local_min = np.where(valid, depth_m, np.inf)
    local_max = np.where(valid, depth_m, -np.inf)

    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis] = slice(1, None)
        b[axis] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        pair_valid = valid[a] & valid[b]
        jump = pair_valid & (np.abs(depth_m[a] - depth_m[b]) > threshold_m)
        edge[a] |= jump
        edge[b] |= jump
        object_boundary = valid[a] ^ valid[b]
        boundary[a] |= object_boundary
        boundary[b] |= object_boundary

        min_pair = np.minimum(local_min[a], local_min[b])
        max_pair = np.maximum(local_max[a], local_max[b])
        local_min[a] = np.minimum(local_min[a], min_pair)
        local_min[b] = np.minimum(local_min[b], min_pair)
        local_max[a] = np.maximum(local_max[a], max_pair)
        local_max[b] = np.maximum(local_max[b], max_pair)

    return edge, boundary, local_min, local_max


def apply_d435_realistic_noise(depth_m, fx, baseline, subpixel_levels, rng):
    """Empirical D435-like depth degradation, not a device-specific calibration."""
    result = depth_m.copy()
    valid = result > 0
    edge, boundary, local_min, local_max = depth_discontinuities(result)

    disparity = np.zeros_like(result)
    disparity[valid] = baseline * fx / result[valid]
    # Disparity uncertainty turns into approximately quadratic axial depth noise.
    disparity_sigma_px = 0.045
    frame_disparity_bias_px = float(rng.normal(0.0, 0.012))
    disparity[valid] += frame_disparity_bias_px
    disparity[valid] += rng.normal(0.0, disparity_sigma_px, int(valid.sum()))
    disparity[valid] = np.maximum(disparity[valid], 1e-6)
    disparity[valid] = (
        np.rint(disparity[valid] * subpixel_levels) / subpixel_levels
    )
    result[valid] = baseline * fx / disparity[valid]

    # A small fraction of pixels near a depth discontinuity become mixed depths.
    finite_span = np.isfinite(local_min) & np.isfinite(local_max)
    mixed_candidates = edge & finite_span & ((local_max - local_min) > 0.015)
    flying = mixed_candidates & (rng.random(result.shape) < 0.035)
    mix = rng.uniform(0.2, 0.8, result.shape)
    result[flying] = (
        local_min[flying]
        + mix[flying] * (local_max[flying] - local_min[flying])
    )

    # Random invalid pixels plus stronger dropout at silhouettes/discontinuities.
    dropout_probability = np.full(result.shape, 0.004, dtype=np.float64)
    dropout_probability += 0.16 * edge + 0.24 * boundary
    dropout_probability += 0.003 * np.square(np.clip(result, 0.0, 3.0))
    dropout = valid & (rng.random(result.shape) < dropout_probability)
    result[dropout] = 0.0

    return result, {
        "valid_input_pixels": int(valid.sum()),
        "dropped_pixels": int(dropout.sum()),
        "flying_pixels": int(flying.sum()),
        "frame_disparity_bias_px": frame_disparity_bias_px,
    }


def rotation_from_vector(rotation_vector):
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = rotation_vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def perturb_camera_extrinsic(extrinsic, rng, translation_sigma_m, rotation_sigma_deg):
    camera_to_world = np.linalg.inv(extrinsic)
    translation_error = rng.normal(0.0, translation_sigma_m, 3)
    rotation_error_deg = rng.normal(0.0, rotation_sigma_deg, 3)
    rotation_error = rotation_from_vector(np.radians(rotation_error_deg))

    estimated_camera_to_world = camera_to_world.copy()
    estimated_camera_to_world[:3, :3] = camera_to_world[:3, :3] @ rotation_error
    estimated_camera_to_world[:3, 3] += translation_error
    return np.linalg.inv(estimated_camera_to_world), {
        "translation_error_m": translation_error.tolist(),
        "rotation_error_deg_xyz": rotation_error_deg.tolist(),
        "translation_error_norm_m": float(np.linalg.norm(translation_error)),
        "rotation_error_norm_deg": float(np.linalg.norm(rotation_error_deg)),
    }


def make_ground_truth(boxes):
    mesh = o3d.geometry.TriangleMesh()
    for size, center, _ in boxes:
        box = o3d.geometry.TriangleMesh.create_box(*size)
        origin = np.asarray(center) - np.asarray(size) / 2.0
        box.translate(origin)
        mesh += box
    mesh.compute_vertex_normals()
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output_chair_open3d")
    parser.add_argument("--views", type=int, default=40)
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument("--voxel", type=float, default=0.002)
    parser.add_argument("--trunc", type=float, default=0.008)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hfov", type=float, default=87.0)
    parser.add_argument(
        "--camera-params",
        help="JSON containing measured width/height/fx/fy/ppx/ppy/depth scale",
    )
    parser.add_argument(
        "--depth-model",
        choices=["ideal", "d435_best", "d435_realistic"],
        default="d435_best",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="stereo baseline in metres; defaults to the measured camera JSON value",
    )
    parser.add_argument("--subpixel-levels", type=int, default=32)
    parser.add_argument("--noise-seed", type=int, default=435)
    parser.add_argument("--pose-noise", action="store_true")
    parser.add_argument("--pose-translation-sigma-mm", type=float, default=2.0)
    parser.add_argument("--pose-rotation-sigma-deg", type=float, default=0.15)
    parser.add_argument("--pose-noise-seed", type=int, default=436)
    parser.add_argument("--target", choices=["chair", "desk"], default="desk")
    args = parser.parse_args()
    config = target_config(args.target)

    out = Path(args.output)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)

    camera = load_camera_parameters(args.camera_params, args)
    width, height = int(camera["width"]), int(camera["height"])
    near, far = 0.105, config["far"]
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    sensor_depth_scale = float(camera["depth_scale_m_per_unit"])
    measured_baseline = camera.get("stereo_baseline_m")
    baseline = float(
        args.baseline
        if args.baseline is not None
        else (measured_baseline if measured_baseline is not None else 0.050)
    )
    baseline_source = (
        "command-line override"
        if args.baseline is not None
        else (
            "measured camera parameter file"
            if measured_baseline is not None
            else "default simulation value"
        )
    )
    fov_y = math.degrees(2.0 * math.atan(0.5 * height / fy))
    fov_x = math.degrees(2.0 * math.atan(0.5 * width / fx))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    projection = projection_from_intrinsics(
        width, height, fx, fy, cx, cy, near, far
    )

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("cannot connect to PyBullet")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    add_furniture(config["boxes"])

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    target = config["target"]
    heights = config["heights"]
    per_ring = max(1, args.views // max(1, args.rings))
    used = 0
    poses = []
    noise_stats = []
    pose_noise_stats = []
    depth_rng = np.random.default_rng(args.noise_seed)
    pose_rng = np.random.default_rng(args.pose_noise_seed)

    for i in range(args.views):
        ring = min(args.rings - 1, i // per_ring)
        j = i % per_ring
        angle = 2.0 * math.pi * (j + 0.5 * (ring % 2)) / per_ring
        cam = np.array([config["radius"] * math.cos(angle), config["radius"] * math.sin(angle), heights[min(ring, len(heights) - 1)]])
        view, projection, extrinsic = camera_pose(cam, target, projection)
        _, _, rgba, depth_buffer, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view.reshape(-1, order="F").tolist(),
            projectionMatrix=projection.reshape(-1, order="F").tolist(),
            renderer=p.ER_TINY_RENDERER,
        )
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
        depth_buffer = np.asarray(depth_buffer, dtype=np.float64).reshape(height, width)
        depth_m = far * near / (far - (far - near) * depth_buffer)
        depth_m[(depth_buffer >= 0.9999) | ~np.isfinite(depth_m)] = 0.0
        if args.depth_model == "d435_best":
            valid = depth_m > 0
            disparity = np.zeros_like(depth_m)
            disparity[valid] = baseline * fx / depth_m[valid]
            disparity[valid] = np.rint(disparity[valid] * args.subpixel_levels) / args.subpixel_levels
            depth_m[valid] = baseline * fx / disparity[valid]
        elif args.depth_model == "d435_realistic":
            depth_m, frame_noise = apply_d435_realistic_noise(
                depth_m, fx, baseline, args.subpixel_levels, depth_rng
            )
            frame_noise["frame"] = i
            noise_stats.append(frame_noise)
        depth_raw = np.clip(
            np.rint(depth_m / sensor_depth_scale), 0, 65535
        ).astype(np.uint16)
        depth_m = depth_raw.astype(np.float64) * sensor_depth_scale
        color = np.ascontiguousarray(rgba[:, :, :3])
        Image.fromarray(color, mode="RGB").save(out / "color" / f"{i:03d}.png")
        Image.fromarray(depth_raw, mode="I;16").save(out / "depth" / f"{i:03d}.png")

        color_image = o3d.geometry.Image(color)
        depth_image = o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,
            depth_trunc=far,
            convert_rgb_to_intensity=False,
        )
        integration_extrinsic = extrinsic
        frame_pose_noise = None
        if args.pose_noise:
            integration_extrinsic, frame_pose_noise = perturb_camera_extrinsic(
                extrinsic,
                pose_rng,
                args.pose_translation_sigma_mm / 1000.0,
                args.pose_rotation_sigma_deg,
            )
            frame_pose_noise["frame"] = i
            pose_noise_stats.append(frame_pose_noise)
        volume.integrate(rgbd, intrinsic, integration_extrinsic)
        used += 1
        poses.append({
            "frame": i,
            "camera_world_m": cam.tolist(),
            "target_world_m": target.tolist(),
            "world_to_open3d_camera_true": extrinsic.tolist(),
            "world_to_open3d_camera_used": integration_extrinsic.tolist(),
            "pose_noise": frame_pose_noise,
        })

    p.disconnect()
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    # Tiny disconnected TSDF fragments are floating surface remnants, not
    # part of the reconstructed object. Keep the dominant connected surface
    # so the viewer does not present them as streaks/ghosts.
    labels, counts, _ = mesh.cluster_connected_triangles()
    if len(counts) > 1:
        largest = int(np.argmax(np.asarray(counts)))
        remove = np.flatnonzero(np.asarray(labels) != largest)
        mesh.remove_triangles_by_index(remove)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
    mesh_path = out / "tsdf_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)

    ground_truth = make_ground_truth(config["boxes"])
    o3d.io.write_triangle_mesh(str(out / f"ground_truth_{args.target}.ply"), ground_truth, write_ascii=False)
    reconstructed_points = mesh.sample_points_uniformly(number_of_points=300000)
    truth_points = ground_truth.sample_points_uniformly(number_of_points=300000)
    reconstruction_scene = o3d.t.geometry.RaycastingScene()
    reconstruction_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    truth_scene = o3d.t.geometry.RaycastingScene()
    truth_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(ground_truth))
    reconstruction_to_truth = truth_scene.compute_distance(
        o3d.core.Tensor(np.asarray(reconstructed_points.points), dtype=o3d.core.Dtype.Float32)
    ).numpy()
    truth_to_reconstruction = reconstruction_scene.compute_distance(
        o3d.core.Tensor(np.asarray(truth_points.points), dtype=o3d.core.Dtype.Float32)
    ).numpy()

    def distance_stats(values):
        return {
            "mean_m": float(np.mean(values)),
            "median_m": float(np.median(values)),
            "p95_m": float(np.quantile(values, 0.95)),
            "max_m": float(np.max(values)),
        }

    vertices = np.asarray(mesh.vertices)
    if len(vertices):
        bbox = vertices.max(axis=0) - vertices.min(axis=0)
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
    else:
        bbox = np.zeros(3)
        bbox_min = bbox_max = np.zeros(3)
    truth = config["size"]
    if args.depth_model == "d435_realistic":
        test_definition = (
            "D435 empirical sensor-noise simulation, frames rendered at exact poses, "
            "no environmental lighting/material model"
        )
        total_valid = sum(item["valid_input_pixels"] for item in noise_stats)
        noise_report = {
            "parameter_type": "empirical_simulation_not_device_measured",
            "seed": args.noise_seed,
            "disparity_sigma_px": 0.045,
            "frame_disparity_bias_sigma_px": 0.012,
            "base_dropout_probability": 0.004,
            "edge_dropout_increment": 0.16,
            "silhouette_dropout_increment": 0.24,
            "flying_pixel_probability_at_depth_edges": 0.035,
            "dropped_pixels": int(sum(item["dropped_pixels"] for item in noise_stats)),
            "flying_pixels": int(sum(item["flying_pixels"] for item in noise_stats)),
            "dropout_fraction_of_valid_input": float(
                sum(item["dropped_pixels"] for item in noise_stats) / max(1, total_valid)
            ),
            "pose_noise_simulated": bool(args.pose_noise),
            "per_frame": noise_stats,
        }
    else:
        test_definition = "D435 best-case stereo geometry, exact poses, no environmental noise"
        noise_report = None

    if args.pose_noise:
        pose_report = {
            "parameter_type": "empirical_simulation_not_measured_localization_error",
            "seed": args.pose_noise_seed,
            "translation_sigma_m_per_axis": args.pose_translation_sigma_mm / 1000.0,
            "rotation_sigma_deg_per_axis": args.pose_rotation_sigma_deg,
            "translation_error_norm_mean_m": float(np.mean([
                item["translation_error_norm_m"] for item in pose_noise_stats
            ])),
            "translation_error_norm_max_m": float(np.max([
                item["translation_error_norm_m"] for item in pose_noise_stats
            ])),
            "rotation_error_norm_mean_deg": float(np.mean([
                item["rotation_error_norm_deg"] for item in pose_noise_stats
            ])),
            "rotation_error_norm_max_deg": float(np.max([
                item["rotation_error_norm_deg"] for item in pose_noise_stats
            ])),
            "per_frame": pose_noise_stats,
        }
        test_definition += (
            f", TSDF uses noisy pose estimates with sigma="
            f"{args.pose_translation_sigma_mm:.1f} mm/axis and "
            f"{args.pose_rotation_sigma_deg:.3f} deg/axis"
        )
    else:
        pose_report = None

    report = {
        "tsdf_backend": "Open3D ScalableTSDFVolume",
        "renderer": "PyBullet TinyRenderer",
        "target": config["label"],
        "test_definition": test_definition,
        "depth_model": args.depth_model,
        "noise_model": noise_report,
        "pose_noise_model": pose_report,
        "resolution": [width, height],
        "horizontal_fov_deg": fov_x,
        "vertical_fov_deg": fov_y,
        "intrinsics_px": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "depth_scale_m_per_unit": sensor_depth_scale,
        "camera_parameter_source": camera["source"],
        "camera_parameter_source_file": camera.get("source_file"),
        "camera_parameter_type": camera["parameter_type"],
        "distortion_model": camera.get("distortion_model"),
        "distortion_coeffs": camera.get("distortion_coeffs"),
        "distortion_simulated": False,
        "near_clip_m": near,
        "far_clip_m": far,
        "stereo_baseline_m": baseline,
        "stereo_baseline_source": baseline_source,
        "stereo_subpixel_levels": args.subpixel_levels,
        "views": used,
        "rings": args.rings,
        "voxel_m": args.voxel,
        "sdf_trunc_m": args.trunc,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "ground_truth_size_m": truth.tolist(),
        "reconstructed_bbox_m": bbox.tolist(),
        "reconstructed_bbox_min_m": bbox_min.tolist(),
        "reconstructed_bbox_max_m": bbox_max.tolist(),
        "bbox_abs_error_m": np.abs(bbox - truth).tolist(),
        "surface_distance_reconstruction_to_truth": distance_stats(reconstruction_to_truth),
        "surface_distance_truth_to_reconstruction": distance_stats(truth_to_reconstruction),
        "truth_completeness_within_2mm": float(np.mean(truth_to_reconstruction <= 0.002)),
        "truth_completeness_within_5mm": float(np.mean(truth_to_reconstruction <= 0.005)),
        "reconstructed_surface_within_2mm": float(np.mean(reconstruction_to_truth <= 0.002)),
        "reconstructed_surface_within_5mm": float(np.mean(reconstruction_to_truth <= 0.005)),
        "mesh": str(mesh_path.name),
        "pose_file": "poses.json",
    }
    (out / "poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
