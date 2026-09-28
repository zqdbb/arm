#!/usr/bin/env python3
"""Simulate six-camera ChArUco extrinsic calibration and verify template alignment.

The vehicle is removed during calibration. A virtual 5x7 ChArUco board is moved
between pair-overlap stations so the camera observation graph is connected.
Only detected ChArUco corners and the known intrinsics are used to estimate the
camera rig; simulator ground truth is used only for the final error report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def camera_layout():
    cameras = []
    for sx, sy, corner in ((1, 1, "front_left"), (-1, 1, "rear_left"),
                           (-1, -1, "rear_right"), (1, -1, "front_right")):
        cameras.append({"name": f"cam{len(cameras)+1:02d}_upper_{corner}",
                        "position": np.array([3.60*sx, 2.60*sy, 2.45]),
                        "target": np.array([1.05*sx, 0.0, 1.06])})
    for sy, side in ((1, "left"), (-1, "right")):
        cameras.append({"name": f"cam{len(cameras)+1:02d}_lower_mid_{side}",
                        "position": np.array([0.0, 1.35*sy, 0.72]),
                        "target": np.array([0.0, 0.0, 0.82])})
    return cameras


def look_at(position, target):
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ position
    return transform


def transform_points(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def board_capture_pose(cameras, i, j, board_width, board_height, station, base_center):
    center = np.asarray(base_center, dtype=np.float64) + np.array(station)
    # OpenCV's generated ChArUco pattern has +X right and +Y down; with
    # camera optical coordinates this makes the board's +Z face point away
    # from the observing cameras (otherwise the rendered markers are mirrored).
    normal = center - (cameras[i]["position"] + cameras[j]["position"])*0.5
    normal /= np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(normal, up)
    if np.linalg.norm(x_axis) < 1e-5:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, normal))
    origin = center - rotation[:, 0]*board_width/2 - rotation[:, 1]*board_height/2
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = origin
    return pose


def draw_virtual_view(board_img, board_pose, world_to_camera, camera_matrix, width, height):
    board_h, board_w = board_img.shape[:2]
    physical_w, physical_h = 0.90, 1.26
    local_corners = np.array([[0, 0, 0], [physical_w, 0, 0],
                              [physical_w, physical_h, 0], [0, physical_h, 0]], dtype=np.float64)
    world_corners = transform_points(board_pose, local_corners)
    camera_corners = transform_points(world_to_camera, world_corners)
    if np.any(camera_corners[:, 2] <= 0.15):
        return None
    projected, _ = cv2.projectPoints(world_corners, cv2.Rodrigues(world_to_camera[:3, :3])[0],
                                     world_to_camera[:3, 3], camera_matrix, np.zeros(5))
    dst = projected.reshape(-1, 2).astype(np.float32)
    if (dst[:, 0].min() < 4 or dst[:, 0].max() >= width-4 or
            dst[:, 1].min() < 4 or dst[:, 1].max() >= height-4):
        return None
    src = np.array([[0, 0], [board_w-1, 0], [board_w-1, board_h-1], [0, board_h-1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    image = np.full((height, width), 220, dtype=np.uint8)
    warped = cv2.warpPerspective(board_img, homography, (width, height),
                                 flags=cv2.INTER_LINEAR, borderValue=220)
    mask_src = np.full_like(board_img, 255)
    mask = cv2.warpPerspective(mask_src, homography, (width, height),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    image[mask > 0] = warped[mask > 0]
    return image


def detect_pose(detector, board, image, camera_matrix):
    corners, ids, marker_corners, marker_ids = detector.detectBoard(image)
    if ids is None or len(ids) < 8:
        return None
    object_points = board.getChessboardCorners()[ids.reshape(-1)].astype(np.float64)
    image_points = corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, np.zeros(5),
        iterationsCount=200, reprojectionError=2.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok and inliers is not None and len(inliers) >= 8:
        fit_object, fit_image = object_points[inliers[:, 0]], image_points[inliers[:, 0]]
    else:
        # Planar, oblique views can make PnP-RANSAC reject every minimal
        # sample despite a clean ChArUco detection; use planar IPPE fallback.
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix,
                                       np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        fit_object, fit_image = object_points, image_points
    rvec, tvec = cv2.solvePnPRefineLM(fit_object, fit_image,
                                      camera_matrix, np.zeros(5), rvec, tvec)
    rotation, _ = cv2.Rodrigues(rvec)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = tvec[:, 0]
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, np.zeros(5))
    rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2)-image_points)**2, axis=1))))
    return pose, int(len(ids)), rmse, int(len(marker_ids)) if marker_ids is not None else 0


def pose_vector(transform):
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(), transform[:3, 3]]


def vector_pose(values):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    result[:3, 3] = values[3:]
    return result


def estimate_rig(edge_observations, camera_count):
    edges = []
    graph = defaultdict(set)
    for (i, j), measurements in edge_observations.items():
        for measured in measurements:
            edges.append((i, j, measured))
            graph[i].add(j)
            graph[j].add(i)
    visited = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node] - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    if len(visited) != camera_count:
        raise RuntimeError(f"ChArUco observation graph is disconnected: {sorted(visited)}")

    def unpack(x):
        poses = [np.eye(4)]
        poses.extend(vector_pose(x[k*6:(k+1)*6]) for k in range(camera_count-1))
        return poses

    def residual(x):
        poses = unpack(x)
        values = []
        for i, j, measurement in edges:
            predicted = np.linalg.inv(poses[j]) @ poses[i]
            error = np.linalg.inv(measurement) @ predicted
            values.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec()*2.0)
            values.extend(error[:3, 3])
        return np.asarray(values)

    x0 = np.zeros((camera_count-1)*6, dtype=np.float64)
    # Seed via a spanning tree, then robustly optimize all shared-board observations.
    tree = defaultdict(list)
    for i, j, t_j_i in edges:
        tree[i].append((j, t_j_i))
        tree[j].append((i, np.linalg.inv(t_j_i)))
    initial = {0: np.eye(4)}
    q = deque([0])
    while q:
        i = q.popleft()
        for j, t_j_i in tree[i]:
            if j not in initial:
                initial[j] = initial[i] @ np.linalg.inv(t_j_i)
                q.append(j)
    for k in range(1, camera_count):
        x0[(k-1)*6:k*6] = pose_vector(initial[k])
    fit = least_squares(residual, x0, loss="huber", f_scale=0.01,
                        max_nfev=300, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return unpack(fit.x), fit


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output_charuco_calibration")
    parser.add_argument("--image", default="/tmp/codex-clipboard-UXxleg.png",
                        help="User-provided board reference image; generated board uses matching 5x7 layout")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    (output / "captures").mkdir(exist_ok=True)

    cv2.setNumThreads(1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard((5, 7), 0.18, 0.135, dictionary)
    board_image = board.generateImage((900, 1260), marginSize=0)
    cv2.imwrite(str(output / "charuco_board_5x7.png"), board_image)
    if Path(args.image).is_file():
        reference = cv2.imread(args.image)
        if reference is not None:
            cv2.imwrite(str(output / "provided_board_reference.png"), reference)

    # Calibrate at the sensor's 1280x800 native specification resolution; the
    # separate fast RGB-D scene simulation remains 640x400 for runtime.
    width, height = 1280, 800
    camera_matrix = np.array([[620.0, 0, 640.0], [0, 620.0, 400.0], [0, 0, 1]], dtype=np.float64)
    cameras = camera_layout()
    truth_world_to_camera = [look_at(c["position"], c["target"]) for c in cameras]
    detector = cv2.aruco.CharucoDetector(board)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 5), (3, 5)]
    stations = [(0,0,0), (0.018,0.008,0.006), (-0.014,-0.012,-0.004),
                (0.008,-0.016,0.010), (-0.010,0.014,-0.008), (0.004,0.004,0.0)]
    measurements = defaultdict(list)
    capture_report = []
    board_poses_for_display = []
    for edge_index, (i, j) in enumerate(edges):
        # These are chosen from the cameras' actual frustum intersections.
        # The upper/lower links lie outside the vehicle footprint because the
        # vehicle itself is absent during calibration.
        if (i, j) == (0, 4): base_center = (2.0, -1.45, 1.84)
        elif (i, j) == (1, 4): base_center = (-2.50, -1.75, 2.10)
        elif (i, j) == (2, 5): base_center = (-2.0, 1.45, 1.84)
        elif (i, j) == (3, 5): base_center = (2.50, 1.75, 2.10)
        else: base_center = (0.0, 0.0, 1.05)
        detections = {}
        accepted = []
        for repeat, station in enumerate(stations[:4]):
            board_pose = board_capture_pose(cameras, i, j, 0.90, 1.26, station, base_center)
            board_poses_for_display.append({"edge": [cameras[i]["name"], cameras[j]["name"]],
                                             "pose": board_pose.tolist()})
            group = f"group_{edge_index:02d}_{repeat:02d}"
            group_path = output / "captures" / group
            group_path.mkdir(exist_ok=True)
            found = {}
            for camera_index in (i, j):
                image = draw_virtual_view(board_image, board_pose,
                                          truth_world_to_camera[camera_index],
                                          camera_matrix, width, height)
                if image is None:
                    continue
                file_path = group_path / f"cam{camera_index+1:02d}.png"
                cv2.imwrite(str(file_path), image)
                detection = detect_pose(detector, board, image, camera_matrix)
                if detection:
                    found[camera_index] = detection[0]
                    accepted.append({"camera": cameras[camera_index]["name"],
                                     "charuco_corners": detection[1],
                                     "reprojection_rmse_px": detection[2],
                                     "markers": detection[3], "image": str(file_path.relative_to(output))})
            if len(found) == 2:
                # T_cam_j_cam_i = T_cam_j_board * inverse(T_cam_i_board).
                measurements[(i, j)].append(found[j] @ np.linalg.inv(found[i]))
                detections[group] = found
        capture_report.append({"edge": [cameras[i]["name"], cameras[j]["name"]],
                               "shared_board_poses": len(measurements[(i, j)]),
                               "detections": accepted})
    if any(not measurements[e] for e in edges):
        failed = [{"edge": e, "joint_poses": len(measurements[e]),
                   "per_station": next(x["shared_board_poses"] for x in capture_report
                                       if x["edge"] == [cameras[e[0]]["name"], cameras[e[1]]["name"]])}
                  for e in edges if not measurements[e]]
        raise RuntimeError(f"No shared-board detections for camera edges: {failed}. See captures/")

    estimated_relative, optimization = estimate_rig(measurements, len(cameras))
    # Fix rig gauge to the simulator's cam01 pose. This is for validation only;
    # the real rig would instead use a measured workcell datum/board pose.
    camera01_to_world = np.linalg.inv(truth_world_to_camera[0])
    estimated_camera_to_world = [camera01_to_world @ pose for pose in estimated_relative]
    estimated_world_to_camera = [np.linalg.inv(pose) for pose in estimated_camera_to_world]
    camera_errors = []
    for index, (estimated, truth) in enumerate(zip(estimated_world_to_camera, truth_world_to_camera)):
        delta = estimated @ np.linalg.inv(truth)
        camera_errors.append({
            "camera": cameras[index]["name"],
            "translation_error_mm": float(np.linalg.norm(delta[:3, 3])*1000),
            "rotation_error_deg": float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1)))),
        })

    # Re-express each captured world cloud in camera coordinates, then merge using
    # only ChArUco-estimated transforms (aligned to world by the cam01 datum).
    import open3d as o3d
    board_points = []
    board_colors = []
    for item in board_poses_for_display:
        pose = np.asarray(item["pose"], dtype=np.float64)
        for u in np.linspace(0.0, 0.90, 25):
            for v in np.linspace(0.0, 1.26, 32):
                board_points.append((pose[:3, :3] @ np.array([u, v, 0.0])) + pose[:3, 3])
                board_colors.append((1.0, 0.78, 0.08))
    board_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(board_points)))
    board_cloud.colors = o3d.utility.Vector3dVector(np.asarray(board_colors))
    o3d.io.write_point_cloud(str(output / "charuco_board_poses.ply"), board_cloud, write_ascii=False)
    ideal_output = root / "output_ideal"
    merged = o3d.geometry.PointCloud()
    for index, estimated in enumerate(estimated_camera_to_world):
        cloud = o3d.io.read_point_cloud(str(ideal_output / "pointclouds" / f"cam{index+1:02d}_world.ply"))
        cloud.transform(truth_world_to_camera[index])
        cloud.transform(estimated)
        merged += cloud
    merged = merged.voxel_down_sample(0.010)
    o3d.io.write_point_cloud(str(output / "merged_charuco_estimated.ply"), merged, write_ascii=False)

    # Compare the calibrated merge against the simulator vehicle/template in the
    # same world frame. The estimated registration isn't seeded with camera poses.
    truth_world = o3d.io.read_point_cloud(str(ideal_output / "ground_truth_vehicle.ply"))
    truth_world.transform(camera01_to_world)
    o3d.io.write_point_cloud(str(output / "vehicle_truth_cam01_world.ply"), truth_world, write_ascii=False)
    summary = {
        "board": {"squares_x": 5, "squares_y": 7, "square_length_m": 0.18,
                  "marker_length_m": 0.135, "dictionary": "DICT_5X5_1000",
                  "provided_reference_image": str(args.image)},
        "intrinsics_px": {"width": width, "height": height, "fx": 620, "fy": 620, "cx": 640, "cy": 400},
        "world_frame": "cam01 optical frame aligned to simulator world (validation gauge only)",
        "observation_edges": capture_report,
        "pose_graph": {"edge_measurements": int(sum(len(v) for v in measurements.values())),
                       "optimizer_cost": float(optimization.cost), "optimizer_success": bool(optimization.success)},
        "camera_pose_error_vs_sim_truth": camera_errors,
        "camera_pose_error_max_translation_mm": max(x["translation_error_mm"] for x in camera_errors),
        "camera_pose_error_max_rotation_deg": max(x["rotation_error_deg"] for x in camera_errors),
        "estimated_world_to_camera": {cameras[i]["name"]: estimated_world_to_camera[i].tolist()
                                       for i in range(len(cameras))},
        "outputs": {"board": "charuco_board_5x7.png", "captures": "captures/",
                    "board_poses": "charuco_board_poses.ply",
                    "merged_scan": "merged_charuco_estimated.ply",
                    "vehicle_truth": "vehicle_truth_cam01_world.ply"},
    }
    (output / "charuco_calibration_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("pose_graph", "camera_pose_error_max_translation_mm",
                                               "camera_pose_error_max_rotation_deg", "camera_pose_error_vs_sim_truth")}, indent=2))


if __name__ == "__main__":
    main()
