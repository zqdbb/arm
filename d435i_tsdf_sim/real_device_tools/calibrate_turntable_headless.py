#!/usr/bin/env python3
"""Headless multi-view calibration of a fixed D435i and motorized turntable."""

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def rigid_transform(src, dst):
    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    trans = cd - rot @ cs
    return rot, trans


def rotation_axis(rot):
    rvec, _ = cv2.Rodrigues(rot)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-8:
        return None, 0.0
    return rvec.reshape(3) / angle, angle


def axis_point(rot, trans):
    return np.linalg.pinv(np.eye(3) - rot, rcond=1e-6) @ trans


def intersect_axis_plane(point, direction, plane):
    normal, d = plane[:3], plane[3]
    denom = float(np.dot(normal, direction))
    if abs(denom) < 1e-8:
        raise RuntimeError("estimated rotation axis is parallel to turntable plane")
    return point + (-(float(np.dot(normal, point)) + d) / denom) * direction


def normal_to_world_rotation(normal):
    normal = normal / np.linalg.norm(normal)
    target = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(np.dot(normal, target), -1.0, 1.0))
    if cosine > 0.999999:
        return np.eye(3)
    if cosine < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(normal, target)
    sine = float(np.linalg.norm(axis))
    axis /= sine
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


def capture_rgbd(pipeline, align, depth_scale, samples):
    depths, colors = [], []
    for _ in range(samples):
        frames = align.process(pipeline.wait_for_frames(5000))
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if depth and color:
            depths.append(np.asanyarray(depth.get_data()).copy())
            colors.append(np.asanyarray(color.get_data()).copy())
    if len(depths) < max(3, samples // 2):
        raise RuntimeError(f"only {len(depths)} complete RGB-D frames captured")
    depth_raw = np.median(np.stack(depths), axis=0).astype(np.uint16)
    color = np.median(np.stack(colors), axis=0).astype(np.uint8)
    return depth_raw.astype(np.float32) * depth_scale, color, depth_raw


def xyz_images(depth, intr):
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (uu - intr.ppx) / intr.fx * depth
    y = (vv - intr.ppy) / intr.fy * depth
    return x, y, depth


def fit_turntable_plane(depth, intr):
    x, y, z = xyz_images(depth, intr)
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    # Lower central image region is the visible turntable surface in the setup.
    mask = (
        (z > 0.25) & (z < 0.70)
        & (uu > int(w * 0.18)) & (uu < int(w * 0.82))
        & (vv > int(h * 0.56)) & (vv < int(h * 0.94))
    )
    points = np.column_stack([x[mask], y[mask], z[mask]])
    if len(points) < 1000:
        raise RuntimeError(f"too few turntable-plane candidates: {len(points)}")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    model, inliers = cloud.segment_plane(
        distance_threshold=0.0045, ransac_n=3, num_iterations=3000
    )
    normal = np.asarray(model[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    d = float(model[3]) / np.linalg.norm(np.asarray(model[:3], dtype=float))
    # Existing reconstruction convention: world +Z points into the turntable.
    # Positive controller motion is clockwise, hence positive about this +Z axis.
    if normal[2] < 0:
        normal = -normal
        d = -d
    return np.r_[normal, d], int(len(inliers)), int(len(points))


def feature_mask(depth, color_a, color_b, plane, intr):
    h, w = depth.shape
    x, y, z = xyz_images(depth, intr)
    signed = plane[0] * x + plane[1] * y + plane[2] * z + plane[3]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    geometry = (
        (z > 0.25) & (z < 0.62)
        & (signed > -0.28) & (signed < 0.020)
        & (uu > int(w * 0.16)) & (uu < int(w * 0.84))
        & (vv > int(h * 0.12)) & (vv < int(h * 0.91))
    )
    ga = cv2.cvtColor(color_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(color_b, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(ga, gb)
    motion = (motion > 8).astype(np.uint8) * 255
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    motion = cv2.dilate(motion, np.ones((13, 13), np.uint8), iterations=2) > 0
    mask = geometry & motion
    if mask.sum() < 2500:
        mask = geometry
    return mask


def point_at(depth, u, v, intr):
    x0, y0 = int(round(u)), int(round(v))
    h, w = depth.shape
    if x0 < 2 or x0 >= w - 2 or y0 < 2 or y0 >= h - 2:
        return None
    patch = depth[y0 - 1:y0 + 2, x0 - 1:x0 + 2]
    valid = patch[(patch > 0.20) & (patch < 0.80)]
    if valid.size < 5:
        return None
    z = float(np.median(valid))
    return np.array([(u - intr.ppx) / intr.fx * z,
                     (v - intr.ppy) / intr.fy * z, z])


def register_pair(frame0, frame1, intr, plane, commanded_deg, seed):
    depth0, color0 = frame0["depth"], frame0["color"]
    depth1, color1 = frame1["depth"], frame1["color"]
    mask0 = feature_mask(depth0, color0, color1, plane, intr)
    mask1 = feature_mask(depth1, color1, color0, plane, intr)
    orb = cv2.ORB_create(nfeatures=3500, scaleFactor=1.15, nlevels=10,
                         edgeThreshold=12, patchSize=31, fastThreshold=7)
    gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
    kp0, des0 = orb.detectAndCompute(gray0, (mask0.astype(np.uint8) * 255))
    kp1, des1 = orb.detectAndCompute(gray1, (mask1.astype(np.uint8) * 255))
    if des0 is None or des1 is None:
        raise RuntimeError("ORB found no descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    forward = matcher.knnMatch(des0, des1, k=2)
    matches = [m for m, n in forward if m.distance < 0.80 * n.distance and m.distance < 72]
    src, dst, kept_matches = [], [], []
    for match in matches:
        p0 = point_at(depth0, *kp0[match.queryIdx].pt, intr)
        p1 = point_at(depth1, *kp1[match.trainIdx].pt, intr)
        if p0 is not None and p1 is not None:
            src.append(p0)
            dst.append(p1)
            kept_matches.append(match)
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src) < 20:
        raise RuntimeError(f"too few 3D feature pairs: {len(src)}")

    rng = np.random.default_rng(seed)
    best = np.empty(0, dtype=int)
    expected = abs(commanded_deg)
    plane_normal = plane[:3]
    for _ in range(5000):
        idx = rng.choice(len(src), 3, replace=False)
        try:
            rot, trans = rigid_transform(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        axis, angle = rotation_axis(rot)
        if axis is None:
            continue
        angle_deg = math.degrees(angle)
        if abs(angle_deg - expected) > 8.0:
            continue
        if abs(float(np.dot(axis, plane_normal))) < math.cos(math.radians(18.0)):
            continue
        residual = np.linalg.norm((rot @ src.T).T + trans - dst, axis=1)
        inliers = np.flatnonzero(residual < 0.008)
        if len(inliers) > len(best):
            best = inliers
    if len(best) < 12:
        raise RuntimeError(f"RANSAC found only {len(best)} inliers from {len(src)} pairs")

    for _ in range(4):
        rot, trans = rigid_transform(src[best], dst[best])
        residual = np.linalg.norm((rot @ src.T).T + trans - dst, axis=1)
        new_best = np.flatnonzero(residual < 0.006)
        if np.array_equal(new_best, best):
            break
        best = new_best
    rot, trans = rigid_transform(src[best], dst[best])
    residual = np.linalg.norm((rot @ src[best].T).T + trans - dst[best], axis=1)
    axis, angle = rotation_axis(rot)
    angle_deg = math.degrees(angle)
    axis_plane_error = math.degrees(math.acos(np.clip(abs(np.dot(axis, plane_normal)), -1, 1)))
    point = axis_point(rot, trans)
    origin = intersect_axis_plane(point, axis, plane)

    match_image = cv2.drawMatches(
        color0, kp0, color1, kp1, [kept_matches[i] for i in best], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    mask_diag = color0.copy()
    mask_diag[~mask0] //= 4
    return {
        "rotation": rot,
        "translation": trans,
        "axis": axis,
        "angle_deg": angle_deg,
        "axis_plane_error_deg": axis_plane_error,
        "origin": origin,
        "feature_pairs": int(len(src)),
        "inliers": int(len(best)),
        "inlier_ratio": float(len(best) / len(src)),
        "rmse_m": float(np.sqrt(np.mean(residual ** 2))),
        "mask": mask0,
        "mask_diag": mask_diag,
        "match_image": match_image,
    }


def verification_image(color, intr, rotation, translation, radius=0.10):
    image = color.copy()
    center_cam = -rotation.T @ translation
    theta = np.linspace(0, 2 * np.pi, 181)
    for r, col, thickness in ((0.05, (160, 160, 160), 1),
                              (radius, (0, 255, 0), 2),
                              (0.15, (160, 160, 160), 1)):
        world = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
        camera = (rotation.T @ world.T).T + center_cam
        visible = camera[:, 2] > 0.01
        uv = np.column_stack([
            intr.fx * camera[visible, 0] / camera[visible, 2] + intr.ppx,
            intr.fy * camera[visible, 1] / camera[visible, 2] + intr.ppy,
        ]).astype(np.int32)
        if len(uv) > 5:
            cv2.polylines(image, [uv], True, col, thickness, cv2.LINE_AA)
    u = int(intr.fx * center_cam[0] / center_cam[2] + intr.ppx)
    v = int(intr.fy * center_cam[1] / center_cam[2] + intr.ppy)
    cv2.drawMarker(image, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.putText(image, f"axis center: ({u},{v})  depth={center_cam[2]:.3f}m",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return image, [u, v], center_cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=20.0)
    parser.add_argument("--speed", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--output-root", default="output")
    args = parser.parse_args()

    from turntable import TurntableController

    base = Path(__file__).resolve().parent
    output_root = (base / args.output_root).resolve()
    run_dir = output_root / ("calibration_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    device = profile.get_device()
    depth_scale = float(device.first_depth_sensor().get_depth_scale())
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    table = TurntableController(args.port)
    table.open()
    frames = {}
    movement_started = False
    try:
        for _ in range(60):
            pipeline.wait_for_frames(5000)
        table.zero()
        time.sleep(0.3)
        for angle in (0.0, args.delta, -args.delta):
            if angle != 0.0:
                movement_started = True
                print(f"MOVE absolute {angle:+.1f} deg speed={args.speed}", flush=True)
                table.move_absolute(angle, speed=args.speed)
                status = table.wait_stop(timeout=30)
                print("STATUS", json.dumps(status, ensure_ascii=False), flush=True)
                time.sleep(1.0)
            depth, color, raw = capture_rgbd(pipeline, align, depth_scale, args.samples)
            frames[angle] = {"depth": depth, "color": color, "raw": raw}
            label = f"{angle:+.0f}".replace("+", "p").replace("-", "m")
            cv2.imwrite(str(run_dir / f"color_{label}.png"), color)
            cv2.imwrite(str(run_dir / f"depth_{label}_uint16.png"), raw)
            print(f"CAPTURE {angle:+.1f} deg valid={100*np.count_nonzero(raw)/raw.size:.2f}%", flush=True)
    finally:
        if movement_started:
            try:
                print("RETURN absolute 0 deg", flush=True)
                table.move_absolute(0.0, speed=args.speed)
                print("RETURN_STATUS", json.dumps(table.wait_stop(timeout=30), ensure_ascii=False), flush=True)
            except Exception as exc:
                print("RETURN_ERROR", repr(exc), flush=True)
                try:
                    table.stop()
                except Exception:
                    pass
        table.close()
        pipeline.stop()

    plane, plane_inliers, plane_candidates = fit_turntable_plane(frames[0.0]["depth"], intr)
    print("PLANE", plane.tolist(), "inliers", plane_inliers, "/", plane_candidates, flush=True)
    results = []
    failures = []
    for index, angle in enumerate((args.delta, -args.delta)):
        try:
            result = register_pair(frames[0.0], frames[angle], intr, plane, angle, 435 + index)
            result["commanded_deg"] = angle
            results.append(result)
            tag = "plus" if angle > 0 else "minus"
            cv2.imwrite(str(run_dir / f"mask_{tag}.png"), result["mask_diag"])
            cv2.imwrite(str(run_dir / f"matches_{tag}.png"), result["match_image"])
            print("PAIR", angle, "angle", result["angle_deg"], "axis_err", result["axis_plane_error_deg"],
                  "inliers", result["inliers"], "/", result["feature_pairs"], "rmse_m", result["rmse_m"], flush=True)
        except Exception as exc:
            failures.append({"commanded_deg": angle, "error": repr(exc)})
            print("PAIR_ERROR", angle, repr(exc), flush=True)

    accepted = [r for r in results
                if abs(r["angle_deg"] - abs(r["commanded_deg"])) <= 4.0
                and r["axis_plane_error_deg"] <= 8.0
                and r["inliers"] >= 12
                and r["rmse_m"] <= 0.008]
    if not accepted:
        report = {"pass": False, "plane": plane.tolist(), "failures": failures,
                  "pairs": [{k: v for k, v in r.items() if k not in {"rotation", "translation", "axis", "origin", "mask", "mask_diag", "match_image"}} for r in results]}
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(f"calibration quality check failed; diagnostics: {run_dir}")

    motor_sign_samples = [
        np.sign(np.dot(r["axis"], plane[:3]) * np.sign(r["commanded_deg"]))
        for r in accepted
    ]
    motor_angle_sign = int(np.sign(np.median(motor_sign_samples)))
    if motor_angle_sign == 0:
        raise RuntimeError("could not determine controller angle sign")

    origins = np.stack([r["origin"] for r in accepted])
    origin = np.median(origins, axis=0)
    origin_spread = float(np.max(np.linalg.norm(origins - origin, axis=1)))
    if len(origins) > 1 and origin_spread > 0.020:
        raise RuntimeError(f"axis-center disagreement {origin_spread*1000:.1f} mm exceeds 20 mm")

    # Plane normal is less noisy than the registration axis and follows the
    # existing project convention (+Z into the table, positive motor clockwise).
    axis = plane[:3] / np.linalg.norm(plane[:3])
    rotation = normal_to_world_rotation(axis)
    translation = -rotation @ origin
    verify, center_px, center_cam = verification_image(
        frames[0.0]["color"], intr, rotation, translation
    )
    cv2.imwrite(str(run_dir / "verification.png"), verify)

    calibration = {
        "R": rotation.tolist(),
        "t": translation.tolist(),
        "plate_z": 0.0,
        "rotation_center": [0.0, 0.0],
        "radius_m": 0.10,
        "camera_serial": device.get_info(rs.camera_info.serial_number),
        "camera_usb": device.get_info(rs.camera_info.usb_type_descriptor),
        "depth_scale_m_per_unit": depth_scale,
        "motor_angle_sign_in_world": motor_angle_sign,
        "motor_positive_rotation": (
            "positive world Z" if motor_angle_sign > 0 else "negative world Z"
        ),
    }
    target = output_root / "calibrate.json"
    if target.exists():
        shutil.copy2(target, run_dir / "calibrate_previous.json")
    target.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    pair_reports = []
    for r in results:
        pair_reports.append({
            "commanded_deg": r["commanded_deg"],
            "estimated_angle_deg": r["angle_deg"],
            "axis_plane_error_deg": r["axis_plane_error_deg"],
            "feature_pairs": r["feature_pairs"],
            "inliers": r["inliers"],
            "inlier_ratio": r["inlier_ratio"],
            "rmse_m": r["rmse_m"],
            "axis_camera": r["axis"].tolist(),
            "origin_camera_m": r["origin"].tolist(),
        })
    report = {
        "pass": True,
        "created": datetime.now().isoformat(timespec="seconds"),
        "calibration_file": str(target),
        "run_dir": str(run_dir),
        "plane_camera": plane.tolist(),
        "plane_inliers": plane_inliers,
        "plane_candidates": plane_candidates,
        "axis_origin_camera_m": origin.tolist(),
        "axis_origin_spread_m": origin_spread,
        "axis_center_pixel": center_px,
        "axis_center_camera_m": center_cam.tolist(),
        "accepted_pairs": len(accepted),
        "pairs": pair_reports,
        "failures": failures,
        "calibration": calibration,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "latest_calibration.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
