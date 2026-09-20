#!/usr/bin/env python3
"""Fixed-camera, automatic-turntable RGB-D reconstruction.

The arm/camera remain stationary.  The turntable angle is converted to an
object rotation so Open3D fuses all views in the zero-angle object frame.
Run only after checking the sample is clear and the turntable is safe.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "turntable"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor"))
from turntable import TurntableController


def tf_matrix(tf):
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w), t.x],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w), t.y],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y), t.z],
        [0, 0, 0, 1]], dtype=float)


def yaw(angle_deg):
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)


def axis_rotation(angle_deg, axis):
    """Homogeneous rotation about an arbitrary unit axis through the origin."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = math.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


def turntable_transform(angle_deg, center, axis):
    """Rigid transform of the object from scan-zero to current turntable angle."""
    A = np.eye(4); A[:3, 3] = np.asarray(center, dtype=float)
    B = np.eye(4); B[:3, 3] = -np.asarray(center, dtype=float)
    return A @ axis_rotation(angle_deg, axis) @ B


class TfReader(Node):
    def __init__(self):
        super().__init__("turntable_reconstruction_tf_reader")
        self.buffer = Buffer(cache_time=Duration(seconds=10))
        self.listener = TransformListener(self.buffer, self)

    def camera_pose(self):
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.buffer.can_transform("baselink", "camera_color_optical_frame", rclpy.time.Time()):
                return tf_matrix(self.buffer.lookup_transform(
                    "baselink", "camera_color_optical_frame", rclpy.time.Time()))
        raise RuntimeError("TF baselink -> camera_color_optical_frame unavailable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--angles", default="0,30,60,90,120,150,180,210,240,270,300,330")
    ap.add_argument("--pause", type=float, default=2.0)
    ap.add_argument("--speed", type=int, default=5000)
    ap.add_argument("--output", default="/workspace/reconstruction_archive/turntable_mesh.ply")
    ap.add_argument("--center", default="-0.484,0.034,0.240",
                    help="turntable axis point in baselink (m)")
    ap.add_argument("--axis", default="-0.003,-0.022,0.9997",
                    help="turntable axis direction in baselink")
    ap.add_argument("--direction", type=float, default=1.0,
                    help="sign of physical turntable angle (+1 or -1)")
    ap.add_argument("--crop_left", type=int, default=220)
    ap.add_argument("--crop_top", type=int, default=120)
    ap.add_argument("--crop_right", type=int, default=1060)
    ap.add_argument("--crop_bottom", type=int, default=680)
    ap.add_argument("--depth_max_m", type=float, default=0.60)
    ap.add_argument("--depth_min_m", type=float, default=0.35)
    ap.add_argument("--axis_radius_m", type=float, default=0.30,
                    help="keep only points within this radius of turntable axis")
    ap.add_argument("--axis_height_min_m", type=float, default=-0.05)
    ap.add_argument("--axis_height_max_m", type=float, default=0.55)
    ap.add_argument("--icp", action="store_true", help="refine each view pose with ICP")
    ap.add_argument("--dry-run", action="store_true", help="capture no frames and send no turntable commands")
    args = ap.parse_args()
    angles = [float(x) for x in args.angles.split(",") if x.strip()]
    center = np.array([float(x) for x in args.center.split(",")], dtype=float)
    axis = np.array([float(x) for x in args.axis.split(",")], dtype=float)
    axis /= np.linalg.norm(axis)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rclpy.init()
    tf_node = TfReader()
    try:
        base_cam = tf_node.camera_pose()
        print("Base-camera TF acquired. Keep the arm stationary.", flush=True)
        if args.dry_run:
            print("Dry run: no turntable or camera command sent.")
            return

        pipe, cfg = rs.pipeline(), rs.config()
        width, height, fps = 1280, 720, 15
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal = rs.temporal_filter()
        hole = rs.hole_filling_filter(1)
        disparity_to = rs.disparity_transform(True)
        disparity_from = rs.disparity_transform(False)
        sensor = profile.get_device().first_depth_sensor()
        depth_scale = sensor.get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, intr.fx, intr.fy, intr.ppx, intr.ppy)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01, sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        target_icp = None

        with TurntableController(port=args.port) as table:
            # The controller position counter persists across power cycles.
            # Define the physically stationary start pose as this scan's 0°.
            table.zero()
            time.sleep(0.3)
            initial = table.get_status()
            if initial.get("state_name") not in ("stopped", "poweron"):
                raise RuntimeError("turntable is not stationary at scan start")
            previous_angle = 0.0
            for index, angle in enumerate(angles, 1):
                delta = angle - previous_angle
                print(f"[{index}/{len(angles)}] rotating by {delta:.1f} deg to {angle:.1f} deg", flush=True)
                if abs(delta) >= 0.01:
                    table.move_relative(delta, speed=args.speed)
                    status = table.wait_stop(timeout=60.0)
                    print(f"  stopped at {status.get('degrees', float('nan')):.1f} deg", flush=True)
                time.sleep(args.pause)
                frames = pipe.wait_for_frames(5000)
                frames = align.process(frames)
                depth = frames.get_depth_frame()
                depth = disparity_to.process(depth)
                depth = spatial.process(depth)
                depth = temporal.process(depth)
                depth = disparity_from.process(depth)
                depth = hole.process(depth)
                color = frames.get_color_frame()
                if not depth or not color:
                    print("  skipped: incomplete RGB-D frame", flush=True)
                    continue
                depth_np = np.asanyarray(depth.get_data()).copy()
                h, w = depth_np.shape[:2]
                l = max(0, min(w, args.crop_left))
                r = max(0, min(w, args.crop_right))
                t = max(0, min(h, args.crop_top))
                b = max(0, min(h, args.crop_bottom))
                crop_mask = np.zeros_like(depth_np, dtype=bool)
                crop_mask[t:b, l:r] = True
                depth_np[~crop_mask] = 0
                depth_np[depth_np < int(args.depth_min_m * 1000.0)] = 0
                depth_np[depth_np > int(args.depth_max_m * 1000.0)] = 0

                # Reject points outside the calibrated turntable working volume.
                # This removes the table/floor/background before TSDF integration.
                yy, xx = np.indices(depth_np.shape)
                zc = depth_np.astype(np.float32) * depth_scale
                valid = zc > 0
                xc = (xx.astype(np.float32) - intr.ppx) * zc / intr.fx
                yc = (yy.astype(np.float32) - intr.ppy) * zc / intr.fy
                ones = np.ones_like(zc)
                cam_pts = np.stack((xc, yc, zc, ones), axis=-1)
                base_pts = cam_pts @ base_cam.T
                q = base_pts[..., :3] - center.reshape(1, 1, 3)
                axial = np.sum(q * axis.reshape(1, 1, 3), axis=-1)
                radial = np.linalg.norm(q - axial[..., None] * axis.reshape(1, 1, 3), axis=-1)
                valid &= radial <= args.axis_radius_m
                valid &= (axial >= args.axis_height_min_m) & (axial <= args.axis_height_max_m)
                depth_np[~valid] = 0
                d = o3d.geometry.Image(depth_np)
                c = o3d.geometry.Image(np.asanyarray(color.get_data()))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    c, d, depth_scale=1.0 / depth_scale, depth_trunc=1.5, convert_rgb_to_intensity=False)
                # Convert the current object pose back to scan-zero coordinates.
                # base_cam maps camera points into baselink.  H(angle) is the
                # object's motion about the real turntable axis.  Open3D wants
                # world->camera, hence inverse(base_cam_to_object) = inv(base_cam) @ H.
                H = turntable_transform(args.direction * angle, center, axis)
                extrinsic = np.linalg.inv(base_cam) @ H
                if args.icp:
                    pcd = o3d.geometry.PointCloud.create_from_depth_image(
                        d, intrinsic, depth_scale=1.0 / depth_scale, depth_trunc=1.5,
                        stride=4)
                    pcd = pcd.voxel_down_sample(0.01)
                    if len(pcd.points) >= 50:
                        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
                        # Convert this camera view to scan-zero coordinates using
                        # the turntable model, then refine against the first view.
                        pcd.transform(np.linalg.inv(H) @ base_cam)
                        predicted_world = pcd
                        if target_icp is None:
                            target_icp = pcd
                        else:
                            result = o3d.pipelines.registration.registration_icp(
                                pcd, target_icp, 0.06, np.eye(4),
                                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                            print(f"  ICP fitness={result.fitness:.3f} rmse={result.inlier_rmse:.4f}", flush=True)
                            if result.fitness < 0.20 or result.inlier_rmse > 0.04:
                                print("  skipped: ICP registration rejected", flush=True)
                                previous_angle = angle
                                continue
                            # result maps predicted scan-zero points to target.
                            # Corrected camera->world = result @ predicted camera->world.
                            corrected_cam_to_world = result.transformation @ (np.linalg.inv(H) @ base_cam)
                            extrinsic = np.linalg.inv(corrected_cam_to_world)
                            predicted_world.transform(result.transformation)
                            target_icp = predicted_world
                volume.integrate(rgbd, intrinsic, extrinsic)
                print("  integrated", flush=True)
                previous_angle = angle
        pipe.stop()
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(args.output, mesh, write_ascii=False)
        print(f"Saved mesh: {args.output}", flush=True)
    finally:
        tf_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
