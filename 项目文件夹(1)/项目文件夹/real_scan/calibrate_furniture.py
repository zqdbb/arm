#!/usr/bin/env python3
"""转台标定 (家具旋转法).

原理:
  1. 家具放在转台上, 拍摄 RGB-D (角度 0°)
  2. 转台旋转 Δθ (默认 30°), 再拍摄 RGB-D
  3. ORB 特征匹配 + 3D 点对 → RANSAC 刚体变换 → 旋转矩阵 R_rot
  4. 从 R_rot 提取转轴方向 (Rodrigues)
  5. RANSAC 平面分割找转台面 → 转轴与转台面交点 = 世界原点
  6. 输出 calibrate.json

优点: 不依赖标记物, 直接利用家具几何, 不受光照影响.

用法: python3 calibrate_furniture.py [delta_deg]
"""

import json
import os
import sys
import time
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from turntable import TurntableController

# ── 参数 ──
DELTA_ANGLE = 30            # 两次拍摄间转台旋转角度 (度)
DEPTH_W = DEPTH_WIDTH       # 640
DEPTH_H = DEPTH_HEIGHT      # 480
N_AVG_FRAMES = 3            # 多帧平均
PLANE_THRESHOLD = 0.012     # 转台面 RANSAC 阈值 (12mm)


# ═══════════════════════════════════════════════════════════════
# 相机采集
# ═══════════════════════════════════════════════════════════════

def capture_rgbd(pipeline, align, n_avg=N_AVG_FRAMES):
    """采集 RGB-D (多帧中值滤波)."""
    depth_accum = []
    color_img = None

    for _ in range(n_avg):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        d = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001
        depth_accum.append(d)
        if color_img is None:
            color_img = np.asanyarray(color_frame.get_data())
        time.sleep(0.03)

    if len(depth_accum) < 2:
        return None, None

    depth_stack = np.stack(depth_accum, axis=0)
    depth_median = np.median(depth_stack, axis=0)

    # 空洞填充
    mask = (depth_median > 0.001).astype(np.uint8)
    depth_filled = cv2.inpaint(
        (depth_median * 1000).astype(np.uint16),
        255 - mask * 255, 3, cv2.INPAINT_NS)
    depth_filled = depth_filled.astype(np.float32) * 0.001

    return depth_filled, color_img


def depth_to_pointcloud(depth_img, intr):
    """深度图 → Open3D 点云 (相机坐标系)."""
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_W, DEPTH_H, intr.fx, intr.fy, intr.ppx, intr.ppy)
    depth_o3d = o3d.geometry.Image((depth_img * 1000).astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intr, depth_scale=1000.0)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


# ═══════════════════════════════════════════════════════════════
# 特征匹配 + 刚体变换
# ═══════════════════════════════════════════════════════════════

def get_xyz(uv, depth_img, fx, fy, ppx, ppy):
    """像素坐标 → 3D 相机坐标 (双线性插值深度)."""
    u, v = uv
    u0, v0 = int(u), int(v)
    h, w = depth_img.shape
    if u0 < 0 or u0 >= w - 1 or v0 < 0 or v0 >= h - 1:
        return None
    up, vp = u - u0, v - v0
    d00 = depth_img[v0, u0]
    d01 = depth_img[v0, u0 + 1]
    d10 = depth_img[v0 + 1, u0]
    d11 = depth_img[v0 + 1, u0 + 1]
    # 跳过无效深度
    for dd in [d00, d01, d10, d11]:
        if dd <= 0.001 or dd > 3.0:
            return None
    d = (1 - vp) * (d01 * up + d00 * (1 - up)) + vp * (d11 * up + d10 * (1 - up))
    x = (u - ppx) / fx * d
    y = (v - ppy) / fy * d
    return np.array([x, y, d])


def register_rgbd(color_0, depth_0, color_1, depth_1, fx, fy, ppx, ppy):
    """ORB 特征匹配 → 3D 对应点 → RANSAC 刚体变换.

    Returns: (R, t, inlier_ratio) or (None, None, 0) on failure.
    """
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8,
                          edgeThreshold=15, patchSize=31)

    kp0, des0 = orb.detectAndCompute(color_0, None)
    kp1, des1 = orb.detectAndCompute(color_1, None)

    if des0 is None or des1 is None or len(kp0) < 10 or len(kp1) < 10:
        print('  ORB 特征不足')
        return None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)
    matches = sorted(matches, key=lambda m: m.distance)

    # 3D 对应点
    pts_3d_0, pts_3d_1 = [], []
    for m in matches[:200]:
        pt0 = kp0[m.queryIdx].pt
        pt1 = kp1[m.trainIdx].pt
        p3d_0 = get_xyz(pt0, depth_0, fx, fy, ppx, ppy)
        p3d_1 = get_xyz(pt1, depth_1, fx, fy, ppx, ppy)
        if p3d_0 is not None and p3d_1 is not None:
            dist_0 = np.linalg.norm(p3d_0)
            dist_1 = np.linalg.norm(p3d_1)
            if 0.2 < dist_0 < 1.5 and 0.2 < dist_1 < 1.5:
                pts_3d_0.append(p3d_0)
                pts_3d_1.append(p3d_1)

    if len(pts_3d_0) < 10:
        print(f'  有效3D对应点不足: {len(pts_3d_0)}')
        return None, None, 0

    pts_0 = np.array(pts_3d_0).T  # 3×N
    pts_1 = np.array(pts_3d_1).T

    # RANSAC 刚体变换
    n_pts = pts_0.shape[1]
    best_R, best_t, best_inliers = None, None, []
    max_dist = 0.03  # 3cm inlier threshold

    for _ in range(500):
        idx = np.random.choice(n_pts, 3, replace=False)
        # 三点法求刚体变换
        src = pts_0[:, idx]
        dst = pts_1[:, idx]

        # 计算质心
        c_src = src.mean(axis=1, keepdims=True)
        c_dst = dst.mean(axis=1, keepdims=True)

        # 去中心化
        src_c = src - c_src
        dst_c = dst - c_dst

        # SVD 求 R
        H = src_c @ dst_c.T
        U, s, Vt = np.linalg.svd(H)
        R_est = Vt.T @ U.T
        if np.linalg.det(R_est) < 0:
            Vt[2, :] *= -1
            R_est = Vt.T @ U.T

        t_est = (c_dst - R_est @ c_src).flatten()

        # 计算内点
        diff = pts_1 - (R_est @ pts_0 + t_est.reshape(3, 1))
        dists = np.linalg.norm(diff, axis=0)
        inliers = np.where(dists < max_dist)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R_est, t_est

    if best_R is None or len(best_inliers) < 8:
        print(f'  RANSAC 失败, 内点: {len(best_inliers)}')
        return None, None, 0

    # 用内点 refine
    src_in = pts_0[:, best_inliers]
    dst_in = pts_1[:, best_inliers]
    c_src = src_in.mean(axis=1, keepdims=True)
    c_dst = dst_in.mean(axis=1, keepdims=True)
    H = (src_in - c_src) @ (dst_in - c_dst).T
    U, s, Vt = np.linalg.svd(H)
    R_ref = Vt.T @ U.T
    if np.linalg.det(R_ref) < 0:
        Vt[2, :] *= -1
        R_ref = Vt.T @ U.T
    t_ref = (c_dst - R_ref @ c_src).flatten()

    inlier_ratio = len(best_inliers) / n_pts
    print(f'  特征匹配: {len(matches)} 对 → 3D点 {n_pts} 对 → 内点 {len(best_inliers)} ({inlier_ratio:.1%})')

    return R_ref, t_ref, inlier_ratio


# ═══════════════════════════════════════════════════════════════
# 旋转轴提取
# ═══════════════════════════════════════════════════════════════

def extract_rotation_axis(R):
    """从旋转矩阵 R 提取旋转轴方向和角度.

    家具绕转轴旋转 Δθ, R 是相机坐标系下的旋转矩阵.
    用 Rodrigues 分解得到轴方向和角度.
    """
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-6:
        return None, 0
    axis = rvec.flatten() / angle
    return axis, angle


def point_on_axis(R, t):
    """从 (I-R) @ c = t 求轴上一点 (最小范数解)."""
    A = np.eye(3) - R
    # SVD 伪逆
    U, s, Vt = np.linalg.svd(A)
    s_inv = np.zeros(3)
    s_inv[:2] = 1.0 / s[:2]  # 前两个奇异值非零, 第三个接近0 (零空间=转轴)
    pinv = Vt.T @ np.diag(s_inv) @ U.T
    c = pinv @ t
    return c


# ═══════════════════════════════════════════════════════════════
# 转台面检测
# ═══════════════════════════════════════════════════════════════

def find_turntable_plane(pcd):
    """RANSAC 找转台顶面 (最大平面, 法向量大致朝上/朝相机)."""
    pts = np.asarray(pcd.points)
    dist = np.linalg.norm(pts, axis=1)
    near = (dist > 0.2) & (dist < 1.5)

    if near.sum() < 1000:
        return None

    pcd_near = pcd.select_by_index(np.where(near)[0])
    plane_model, inliers = pcd_near.segment_plane(
        distance_threshold=PLANE_THRESHOLD, ransac_n=3, num_iterations=1000)

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # 法向量朝相机 (相机在 Z+)
    if normal[2] > 0:
        normal = -normal
        d = -d

    print(f'  转台面: {len(inliers):,} inliers  n=[{normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}]')
    return np.array([normal[0], normal[1], normal[2], d])


def intersect_axis_plane(axis_dir, axis_point, plane_model):
    """转轴与转台面的交点."""
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    # axis_point + λ * axis_dir 在平面上:  dot(normal, axis_point + λ*axis_dir) + d = 0
    denom = np.dot(normal, axis_dir)
    if abs(denom) < 1e-6:
        return axis_point  # 平行, 退回最小范数解
    lam = -(d + np.dot(normal, axis_point)) / denom
    return axis_point + lam * axis_dir


# ═══════════════════════════════════════════════════════════════
# 坐标变换
# ═══════════════════════════════════════════════════════════════

def normal_to_R(normal):
    """Rodrigues: 法向量 → 旋转矩阵 (cam→world, Z-up)."""
    n_world = np.array([0., 0., 1.])
    cos_theta = np.dot(normal, n_world)
    if cos_theta > 0.9999:
        return np.eye(3)
    if cos_theta < -0.9999:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    k = np.cross(normal, n_world)
    k = k / np.linalg.norm(k)
    sin_theta = np.linalg.norm(np.cross(normal, n_world))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)


def make_verify_image(color_img, R, t, radius_m, fx, fy, ppx, ppy):
    """叠加拟合圆到 RGB 验证图上."""
    h, w = color_img.shape[:2]
    img = color_img.copy()

    n_pts = 72
    theta = np.linspace(0, 2 * np.pi, n_pts)
    circle_w = np.column_stack([
        radius_m * np.cos(theta),
        radius_m * np.sin(theta),
        np.zeros(n_pts),
    ])

    center_cam = -R.T @ t
    circle_c = (R.T @ circle_w.T).T + center_cam
    front = circle_c[:, 2] > 0.01

    if front.sum() >= 6:
        u = fx * circle_c[front, 0] / circle_c[front, 2] + ppx
        v = fy * circle_c[front, 1] / circle_c[front, 2] + ppy
        pts_uv = np.column_stack([u, v]).astype(np.int32)
        for i in range(len(pts_uv)):
            j = (i + 1) % len(pts_uv)
            if np.sqrt((pts_uv[i][0]-pts_uv[j][0])**2 +
                       (pts_uv[i][1]-pts_uv[j][1])**2) < 150:
                cv2.line(img, tuple(pts_uv[i]), tuple(pts_uv[j]), (0, 255, 0), 2)

    if center_cam[2] > 0.01:
        cu = int(fx * center_cam[0] / center_cam[2] + ppx)
        cv_ = int(fy * center_cam[1] / center_cam[2] + ppy)
        cv2.drawMarker(img, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

    cv2.putText(img, f'Radius: {radius_m*100:.1f} cm', (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, f'Center (cam): [{center_cam[0]:.3f} {center_cam[1]:.3f} {center_cam[2]:.3f}]',
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    delta_deg = float(sys.argv[1]) if len(sys.argv) > 1 else DELTA_ANGLE

    print('=' * 55)
    print(f'  转台标定 (家具旋转法, Δθ={delta_deg}°)')
    print('  放家具 → 拍两帧 → 配准 → 提取转轴')
    print('=' * 55)

    # ── 相机初始化 ──
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print('\n错误: 未检测到 RealSense 设备!')
        return

    dev = devices[0]
    print(f'设备: {dev.get_info(rs.camera_info.name)}')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, DEPTH_FPS)
    cfg.enable_stream(rs.stream.color, DEPTH_W, DEPTH_H, rs.format.bgr8, DEPTH_FPS)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print(f'内参: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}')

    # 预热
    for _ in range(30):
        pipeline.wait_for_frames()
        time.sleep(0.05)

    # ── 转台初始化 ──
    tt = None
    try:
        tt = TurntableController(port=TURNTABLE_PORT)
        tt.open()
        print(f'转台已连接: {TURNTABLE_PORT}')
    except Exception as e:
        print(f'转台未检测到 ({e}), 将手动旋转模式')

    try:
        # ── 第一帧: 角度 0° ──
        print('\n[1/3] 拍摄角度 0°...')
        if tt:
            tt.zero()
        depth_0, color_0 = capture_rgbd(pipeline, align)
        if depth_0 is None:
            print('采集失败'); return

        pcd_0 = depth_to_pointcloud(depth_0, intr)
        print(f'  点云: {len(pcd_0.points):,} 点')

        # ── 旋转 ──
        print(f'\n[2/3] 旋转转台 {delta_deg}°...')
        if tt:
            tt.move_and_wait(delta_deg)
            actual_deg = delta_deg
        else:
            input(f'  请手动旋转转台 {delta_deg}°, 完成后按 Enter...')
            actual_deg = delta_deg

        time.sleep(0.5)

        # ── 第二帧: 角度 Δθ ──
        print(f'\n[2/3] 拍摄角度 {delta_deg}°...')
        depth_1, color_1 = capture_rgbd(pipeline, align)
        if depth_1 is None:
            print('采集失败'); return

        pcd_1 = depth_to_pointcloud(depth_1, intr)
        print(f'  点云: {len(pcd_1.points):,} 点')

        # ── 配准 ──
        print('\n[3/3] 特征匹配配准...')
        R_rot, t_rot, inlier_ratio = register_rgbd(
            color_0, depth_0, color_1, depth_1,
            intr.fx, intr.fy, intr.ppx, intr.ppy)

        if R_rot is None:
            print('\n配准失败! 可能原因:')
            print('  1. 家具纹理不够 (试试有木纹的家具)')
            print('  2. 旋转角度太大导致重叠不足 (试试 20-30°)')
            print('  3. 深度数据在特征点处缺失')
            return

        # ── 提取转轴 ──
        axis_dir, axis_angle = extract_rotation_axis(R_rot)
        actual_deg_extracted = np.degrees(axis_angle)
        print(f'\n  旋转角度: {actual_deg_extracted:.1f}° (期望 {delta_deg}°)')
        print(f'  转轴方向 (cam): [{axis_dir[0]:.4f} {axis_dir[1]:.4f} {axis_dir[2]:.4f}]')

        # 验证角度一致性
        angle_error = abs(actual_deg_extracted - delta_deg)
        if angle_error > 15:
            print(f'  ⚠ 角度偏差较大 ({angle_error:.1f}°), 配准可能不准确')

        # ── 求轴上一点 ──
        axis_pt = point_on_axis(R_rot, t_rot)
        print(f'  轴上点 (cam, 最小范数): [{axis_pt[0]:.4f} {axis_pt[1]:.4f} {axis_pt[2]:.4f}]')

        # ── 找转台面 ──
        plane_model = find_turntable_plane(pcd_0)
        if plane_model is not None:
            origin_pt = intersect_axis_plane(axis_dir, axis_pt, plane_model)
            print(f'  世界原点 (转轴∩转台面): [{origin_pt[0]:.4f} {origin_pt[1]:.4f} {origin_pt[2]:.4f}]')
        else:
            print('  ⚠ 未检测到转台面, 使用最小范数点作为原点')
            origin_pt = axis_pt

        # ── 构建标定 ──
        # 确保 axis_dir 朝上 (Z+ in world)
        if axis_dir[2] < 0:
            axis_dir = -axis_dir

        R_calib = normal_to_R(axis_dir)
        t_calib = -R_calib @ origin_pt

        # 验证: 原点投影到世界坐标应该为 (0, 0, 0)
        world_origin = R_calib @ origin_pt + t_calib
        print(f'  原点验证 (world, 应≈0): [{world_origin[0]:.4f} {world_origin[1]:.4f} {world_origin[2]:.4f}]')

        radius = 0.10  # 转台半径 10cm

        # ── 可视化验证 ──
        verify_img = make_verify_image(
            color_0, R_calib, t_calib, radius,
            intr.fx, intr.fy, intr.ppx, intr.ppy)

        title_bar = np.zeros((40, verify_img.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_bar, "Calibration Result - [S]=save [R]=retry [Q]=quit",
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        verify_display = np.vstack([title_bar, verify_img])

        cv2.imshow('Verify', verify_display)
        print('\nS=保存  R=重来  Q=退出')

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s') or key == ord('S'):
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                calib_data = {
                    'R': R_calib.tolist(),
                    't': t_calib.tolist(),
                    'plate_z': 0.0,
                    'rotation_center': [0.0, 0.0],
                    'radius_m': float(radius),
                }
                with open(CALIB_FILE, 'w') as f:
                    json.dump(calib_data, f, indent=2, default=float)

                print(f'\n标定已保存 → {CALIB_FILE}')
                print(json.dumps(calib_data, indent=2, default=float))
                cv2.destroyAllWindows()
                return
            elif key == ord('r') or key == ord('R'):
                cv2.destroyWindow('Verify')
                print('重新标定...\n')
                break  # 回到外层循环重新来
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if tt:
            tt.close()


if __name__ == '__main__':
    main()
