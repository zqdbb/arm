#!/usr/bin/env python3
"""D435i 转台全自动标定: 深度图 → 平面分割 → 边界圆拟合 → 旋转轴.

原理:
  1. 采集多帧深度图取中值 (降噪)
  2. RANSAC 分割转台顶面
  3. 提取顶面边界点 (ConvexHull)
  4. 2D 圆拟合边界点 → 圆心+半径
  5. 映射回 3D → 旋转轴方向 + 轴上一点

用法: python3 calibrate_auto.py
输出: output/calibrate.json (与 calibrate.py 格式兼容)
"""

import json
import os
import sys
import time
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
import pyransac3d as pyrsc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# ── 参数 ──
DEPTH_W = DEPTH_WIDTH       # 640 (与现有配置一致)
DEPTH_H = DEPTH_HEIGHT      # 480
PLANE_THRESHOLD = 0.015     # 平面分割阈值 (15mm, 容忍纸板翘曲)
CIRCLE_INLIER_THRESH = 0.008  # 圆拟合内点阈值 (8mm)
MIN_PLANE_POINTS = 3000     # 顶面最少点数
N_AVG_FRAMES = 5            # 多帧平均帧数


def capture_filtered_depth(pipeline, align, n_avg=N_AVG_FRAMES):
    """采集 N 帧深度 → 时域中值滤波 → 空间滤波 → 空洞填充."""
    depth_frames = []
    for _ in range(n_avg):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001
        depth_frames.append(depth_img)
        time.sleep(0.02)

    if len(depth_frames) < 2:
        return None, None, None

    depth_stack = np.stack(depth_frames, axis=0)
    depth_median = np.median(depth_stack, axis=0)

    # 空洞填充: 用最近邻插值
    mask = (depth_median > 0.001).astype(np.uint8)
    depth_filled = cv2.inpaint(
        (depth_median * 1000).astype(np.uint16),
        255 - mask * 255, 3, cv2.INPAINT_NS)
    depth_filled = depth_filled.astype(np.float32) * 0.001

    # 也返回最后一帧的 color
    color_img = np.asanyarray(color_frame.get_data())
    return depth_filled, color_img, depth_median


def depth_to_pointcloud(depth_img, intrinsics):
    """深度图 → Open3D 点云 (相机坐标系)."""
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_W, DEPTH_H, fx, fy, ppx, ppy)
    depth_o3d = o3d.geometry.Image((depth_img * 1000).astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intr, depth_scale=1000.0)
    # 统计滤波去飞点
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def segment_plane(pcd, distance_threshold=PLANE_THRESHOLD, center_depth=None):
    """RANSAC 分割最大平面 → 返回 (平面点云, 法向量, 平面方程).

    Args:
        center_depth: 图像中心点的深度 (米), 用于空间 ROI 过滤背景桌面.
                     斜俯视时桌面和转台顶面深度不同, 用此参数裁剪 Z 范围.
    """
    pts = np.asarray(pcd.points)
    dist = np.linalg.norm(pts, axis=1)
    near_mask = (dist > 0.15) & (dist < 1.5)

    # 空间 ROI: 以中心深度为参考, 排除桌面背景
    if center_depth is not None and center_depth > 0:
        # Z 深度裁剪: 转台顶面 ±15cm (斜俯视下圆盘深度跨度 < 15cm)
        z_mask = (pts[:, 2] > center_depth - 0.15) & (pts[:, 2] < center_depth + 0.15)
        # XY 圆柱裁剪: 以光轴为中心半径 20cm (足够容纳 20cm 直径转台)
        xy_dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        xy_mask = xy_dist < 0.20
        roi_mask = z_mask & xy_mask
        near_mask = near_mask & roi_mask
        print(f'  ROI: Z∈[{center_depth-0.15:.2f},{center_depth+0.15:.2f}]  XY<0.20m'
              f'  保留 {near_mask.sum():,}/{len(pts):,} 点')

    pcd_near = pcd.select_by_index(np.where(near_mask)[0])

    if len(pcd_near.points) < MIN_PLANE_POINTS:
        raise RuntimeError(f'有效深度点不足: {len(pcd_near.points)}')

    plane_model, inliers = pcd_near.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3, num_iterations=2000)

    if len(inliers) < MIN_PLANE_POINTS:
        raise RuntimeError(
            f'平面点数不足: {len(inliers)} < {MIN_PLANE_POINTS}\n'
            '  请确保转台在视野内且未被遮挡')

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # 确保法向量朝相机 (相机在 Z+ 方向看物体, 平面法向量应大致朝 -Z)
    if normal[2] > 0:
        normal = -normal
        d = -d

    plane_pcd = pcd_near.select_by_index(inliers)
    print(f'  平面点: {len(inliers):,}  法向量: n=[{normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}]')
    return plane_pcd, normal, np.array([normal[0], normal[1], normal[2], d])


def _project_to_2d(pts, normal):
    """将 3D 点投影到平面局部 2D 坐标系. Returns (x_2d, y_2d, centroid, u, v)."""
    arb = np.array([1., 0., 0.])
    if abs(np.dot(normal, arb)) > 0.9:
        arb = np.array([0., 1., 0.])
    u = np.cross(normal, arb)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    x_2d = np.dot(centered, u)
    y_2d = np.dot(centered, v)
    return x_2d, y_2d, centroid, u, v


def fit_circle_robust(x, y, n_iter=3, outlier_sigma=2.0):
    """鲁棒圆拟合: 全部点迭代最小二乘, 每轮剔除离群点.

    代数圆拟合: min Σ [(x²+y²) - A·x - B·y - C]²
    → cx = A/2, cy = B/2, r = sqrt(cx²+cy²+C)

    Args:
        x, y: 1D arrays of 2D points
        n_iter: 迭代轮数
        outlier_sigma: 离群点阈值 (标准差倍数)

    Returns: (cx, cy, radius, inlier_ratio) or None
    """
    mask = np.ones(len(x), dtype=bool)
    for iteration in range(n_iter):
        xi, yi = x[mask], y[mask]
        if len(xi) < 10:
            return None
        # 代数圆拟合
        A_mat = np.column_stack([xi, yi, np.ones(len(xi))])
        b_vec = xi**2 + yi**2
        try:
            sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        cx = sol[0] / 2.0
        cy = sol[1] / 2.0
        r = np.sqrt(max(cx**2 + cy**2 + sol[2], 1e-12))
        if r <= 0.005 or r > 0.3:
            return None
        # 残差
        residuals = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r)
        if iteration < n_iter - 1:
            thresh = np.mean(residuals[mask]) + outlier_sigma * np.std(residuals[mask])
            mask = residuals < thresh
    inlier_ratio = mask.sum() / len(x)
    return float(cx), float(cy), float(r), float(inlier_ratio)


def extract_boundary_2d(plane_pcd, normal):
    """将平面点云投影到 2D → ConvexHull 取边界 → 返回边界 3D 点."""
    pts = np.asarray(plane_pcd.points)
    if len(pts) < 20:
        return None, None, None, None

    # 构建平面局部坐标系 (u, v 正交于 normal)
    arb = np.array([1., 0., 0.])
    if abs(np.dot(normal, arb)) > 0.9:
        arb = np.array([0., 1., 0.])
    u = np.cross(normal, arb)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # 投影到 2D
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    x_2d = np.dot(centered, u)
    y_2d = np.dot(centered, v)

    # ConvexHull 提取外边界
    from scipy.spatial import ConvexHull
    hull = ConvexHull(np.column_stack([x_2d, y_2d]))
    boundary_2d = np.column_stack([x_2d[hull.vertices], y_2d[hull.vertices]])

    # 转回 3D
    boundary_3d = centroid + boundary_2d[:, 0:1] * u + boundary_2d[:, 1:2] * v

    return boundary_2d, boundary_3d, centroid, (u, v)


def fit_circle_ransac(points_2d, inlier_thresh=CIRCLE_INLIER_THRESH,
                       max_iters=500):
    """RANSAC 圆拟合 (2D). 返回 (cx, cy, radius, inlier_mask)."""
    if len(points_2d) < 5:
        return None

    x, y = points_2d[:, 0], points_2d[:, 1]
    n = len(x)
    best_inliers = []
    best_model = None

    for _ in range(max_iters):
        # 随机选 3 点
        idx = np.random.choice(n, 3, replace=False)
        x3, y3 = x[idx], y[idx]

        # 三点确定圆: 解线性方程
        A = np.column_stack([2 * x3, 2 * y3, np.ones(3)])
        b = x3**2 + y3**2
        try:
            sol = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        cx, cy, c0 = sol
        r = np.sqrt(cx**2 + cy**2 + c0)
        if r <= 0.005 or r > 0.3:
            continue

        # 计算内点
        dist = np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r)
        inliers = np.where(dist < inlier_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = (cx, cy, r)

    if best_model is None or len(best_inliers) < 10:
        return None

    # 用所有内点 refine
    x_in, y_in = x[best_inliers], y[best_inliers]
    A = np.column_stack([2 * x_in, 2 * y_in, np.ones(len(x_in))])
    b = x_in**2 + y_in**2
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy, c0 = sol
    r = np.sqrt(max(cx**2 + cy**2 + c0, 1e-9))

    inlier_ratio = len(best_inliers) / n
    return cx, cy, r, inlier_ratio


def normal_to_R(normal):
    """Rodrigues: 法向量 → 旋转矩阵 R (cam→world, Z-up)."""
    n_world = np.array([0., 0., 1.])
    cos_theta = np.dot(normal, n_world)
    if cos_theta > 0.9999:
        return np.eye(3)
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

    # World → Camera: P_cam = R^T @ P_world + center_cam, 其中 t = -R @ center_cam
    # 所以 R^T @ (P_w - t) = R^T @ P_w + center_cam
    center_cam = -R.T @ t  # 从 t = -R @ center_cam 反推
    circle_c = (R.T @ circle_w.T).T + center_cam
    front = circle_c[:, 2] > 0.01

    if front.sum() >= 6:
        u = (fx * circle_c[front, 0] / circle_c[front, 2] + ppx)
        v = (fy * circle_c[front, 1] / circle_c[front, 2] + ppy)
        pts_uv = np.column_stack([u, v]).astype(np.int32)
        for i in range(len(pts_uv)):
            j = (i + 1) % len(pts_uv)
            if np.sqrt((pts_uv[i][0]-pts_uv[j][0])**2 +
                       (pts_uv[i][1]-pts_uv[j][1])**2) < 150:
                cv2.line(img, tuple(pts_uv[i]), tuple(pts_uv[j]), (0, 255, 0), 2)

    # 圆心标记
    if center_cam[2] > 0.01:
        cu = int(fx * center_cam[0] / center_cam[2] + ppx)
        cv_ = int(fy * center_cam[1] / center_cam[2] + ppy)
        cv2.drawMarker(img, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

    cv2.putText(img, f'Radius: {radius_m*100:.1f} cm', (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img, f'Center (cam): [{center_cam[0]:.3f} {center_cam[1]:.3f} {center_cam[2]:.3f}]',
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 255, 0), 3)
    return img


def main():
    print('=' * 55)
    print('  D435i 转台全自动标定')
    print('  深度图 → 平面分割 → ConvexHull边界 → 圆拟合')
    print('=' * 55)

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
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)

    print(f'depth_scale: {depth_scale:.6f}')

    # ── Advanced Mode preset (暂不加载, 避免固件参数残留影响其他程序) ──
    preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'HighAccuracyPreset-custom.json')
    if False and os.path.exists(preset_path):
        # 如需启用, 请在程序退出前重置 disparityShift
        pass

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    print(f'内参: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}')

    # 预热
    for _ in range(30):
        pipeline.wait_for_frames()
        time.sleep(0.05)

    try:
        while True:
            print('\n按 SPACE 开始自动标定, Q 退出...')
            colorizer = rs.colorizer()

            while True:
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                df = aligned.get_depth_frame()
                cf = aligned.get_color_frame()
                if not df or not cf:
                    continue

                color_img = np.asanyarray(cf.get_data())
                depth_image = np.asanyarray(df.get_data())
                depth_colored = np.asanyarray(colorizer.colorize(df).get_data())
                h, w = color_img.shape[:2]
                depth_disp = cv2.resize(depth_colored, (w, h))

                # ── 彩色图 HUD: 中心十字 + 距离 ──
                cx, cy = w // 2, h // 2
                info_overlay = color_img.copy()
                roi = depth_image[cy-1:cy+2, cx-1:cx+2]
                valid = roi[roi > 0]
                center_dist = np.mean(valid) * depth_scale if len(valid) > 0 else 0
                cv2.putText(info_overlay, f"Center: {center_dist:.3f}m",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_overlay, f"Depth Scale: {depth_scale:.6f}",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.drawMarker(info_overlay, (cx, cy), (0, 0, 255),
                              cv2.MARKER_CROSS, 20, 2)

                # ── 深度图标签 ──
                cv2.putText(depth_disp, "Depth", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # ── 拼接 ──
                combined = np.hstack([info_overlay, depth_disp])
                title_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
                cv2.putText(title_bar, "RealSense D435i - Color | Depth  [SPACE=calibrate | Q=quit]",
                           (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                preview = np.vstack([title_bar, combined])

                cv2.imshow('Auto Calibrate', preview)

                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    return

            # ── Step 1: 采集 + 生成点云 ──
            print('\n[1/4] 采集深度图 (多帧中值滤波)...')
            depth_clean, color_for_verify, _ = capture_filtered_depth(pipeline, align)
            if depth_clean is None:
                print('  采集失败, 重试')
                continue

            # ── 图像中心深度 (用于深度掩膜) ──
            h, w = depth_clean.shape
            cy_img, cx_img = h // 2, w // 2
            center_depth = float(depth_clean[cy_img, cx_img])
            if center_depth <= 0:
                for r in range(1, 30, 2):
                    for dy in range(-r, r+1, r):
                        for dx in range(-r, r+1, r):
                            ny, nx = cy_img+dy, cx_img+dx
                            if 0 <= ny < h and 0 <= nx < w and depth_clean[ny, nx] > 0:
                                center_depth = float(depth_clean[ny, nx])
                                break
                        if center_depth > 0:
                            break
                    if center_depth > 0:
                        break
            print(f'  中心深度: {center_depth:.3f}m')

            if center_depth > 0:
                # 全局深度阈值: |depth - center_depth| < 8cm
                # 不依赖连通性, 纸板翘曲/悬空不会阻断
                depth_range = 0.12  # 12cm, 宽松覆盖斜俯视圆盘+翘曲
                depth_mask = (depth_clean > 0) & (np.abs(depth_clean - center_depth) < depth_range)
                print(f'  深度阈值: |d - {center_depth:.3f}| < {depth_range:.2f} → {depth_mask.sum():,} px')

                # HSV 饱和度分离: 纸板和黑色转台深度相同, 但饱和度不同
                # 黑色/灰色饱和度≈0, 棕色纸板即使在阴影中饱和度也>25
                if color_for_verify is not None and depth_mask.sum() > 500:
                    hsv = cv2.cvtColor(color_for_verify, cv2.COLOR_BGR2HSV)
                    sat = hsv[:, :, 1]  # 饱和度通道, 0-255
                    sat_mask = sat > 25  # 低阈值, 阴影中也能保留
                    mask_2d = depth_mask & sat_mask
                    if mask_2d.sum() < 500:
                        mask_2d = depth_mask  # 兜底
                    print(f'  饱和度分离: {depth_mask.sum():,}px → {mask_2d.sum():,}px')
                else:
                    mask_2d = depth_mask

                # 2D 中心裁剪: 丢弃图像边缘, 只留转盘区域
                crop_sz = 200
                y1_c = max(0, h // 2 - crop_sz)
                y2_c = min(h, h // 2 + crop_sz)
                x1_c = max(0, w // 2 - crop_sz)
                x2_c = min(w, w // 2 + crop_sz)
                crop_mask = np.zeros((h, w), dtype=bool)
                crop_mask[y1_c:y2_c, x1_c:x2_c] = True
                mask_2d = mask_2d & crop_mask
                print(f'  ROI裁剪: [{x1_c}:{x2_c}, {y1_c}:{y2_c}] → 掩膜 {mask_2d.sum():,}px')

                # 诊断: 保存掩膜叠加图
                diag = color_for_verify.copy()
                diag[~mask_2d] = diag[~mask_2d] // 2  # 非掩膜区域变暗
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'mask_diag.png'), diag)
                print(f'  诊断图已保存 → {OUTPUT_DIR}/mask_diag.png')

                # 应用掩膜 → 干净深度图 → 点云
                depth_masked = depth_clean.copy()
                depth_masked[~mask_2d] = 0
                pcd = depth_to_pointcloud(depth_masked, intr)
                print(f'  过滤后点云: {len(pcd.points):,} 点')
            else:
                print('  中心深度无效, 使用全图')
                mask_2d = None
                pcd = depth_to_pointcloud(depth_clean, intr)

            # ── Step 2: 分割转台顶面 ──
            print('\n[2/4] 分割转台顶面...')
            try:
                plane_pcd, normal, plane_model = segment_plane(pcd, center_depth=center_depth)
            except RuntimeError as e:
                print(f'  分割失败: {e}')
                cv2.destroyWindow('Auto Calibrate')
                continue

            # ── Step 2.5: DBSCAN 聚类 ──
            if len(plane_pcd.points) > 100:
                labels = np.asarray(plane_pcd.cluster_dbscan(
                    eps=0.02, min_points=50, print_progress=False))
                if labels.max() >= 0:
                    best_label = int(np.argmax(np.bincount(labels[labels >= 0])))
                    cluster_mask = labels == best_label
                    plane_pcd = plane_pcd.select_by_index(np.where(cluster_mask)[0])
                    print(f'  聚类: 保留最大簇 {cluster_mask.sum():,} 点')
                else:
                    print('  聚类: 全为噪声, 保留原点集')

            # ── Step 3: 2D 轮廓最小外接圆 → 射线-平面求交 → 圆心 3D ──
            print('\n[3/4] 轮廓圆拟合...')

            # 在掩膜上找最大轮廓 (纸板外边界)
            mask_u8 = mask_2d.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print('  未找到轮廓, 重试')
                cv2.destroyWindow('Auto Calibrate')
                continue

            cnt = max(contours, key=cv2.contourArea)
            (cx_px, cy_px), r_px = cv2.minEnclosingCircle(cnt)
            print(f'  轮廓面积: {cv2.contourArea(cnt):.0f}px²'
                  f'  外接圆: 中心({cx_px:.1f},{cy_px:.1f}) 半径{r_px:.1f}px')

            # 圆心 2D → 3D: 射线-平面求交
            a, b_pl, c_pl, d_pl = plane_model
            dir_x = (cx_px - intr.ppx) / intr.fx
            dir_y = (cy_px - intr.ppy) / intr.fy
            dir_z = 1.0
            dot = a * dir_x + b_pl * dir_y + c_pl * dir_z
            t = -d_pl / dot
            center_3d = np.array([t * dir_x, t * dir_y, t * dir_z])

            # 半径: 轮廓可能被掩膜裁切, 物理半径硬编码 10cm
            radius = 0.10

            # 内点率验证
            all_pts = np.asarray(pcd.points)
            x_2d, y_2d, _, _, _ = _project_to_2d(all_pts, normal)
            # 把 center_3d 也投影到 2D
            pts_for_centroid = np.asarray(plane_pcd.points)
            _, _, centroid_2d, u, v = _project_to_2d(pts_for_centroid, normal)
            centered = center_3d - centroid_2d
            cx_2d_c = float(np.dot(centered, u))
            cy_2d_c = float(np.dot(centered, v))
            all_dist = np.abs(np.sqrt((x_2d - cx_2d_c)**2 + (y_2d - cy_2d_c)**2) - radius)
            inlier_ratio = (all_dist < CIRCLE_INLIER_THRESH).sum() / len(x_2d)

            print(f'  半径: {radius*100:.1f}cm  圆心: [{center_3d[0]:.4f} {center_3d[1]:.4f} {center_3d[2]:.4f}]')
            print(f'  内点率: {inlier_ratio:.1%}')

            # ── Step 4: 构造标定参数 ──
            print('\n[4/4] 计算标定参数...')
            R = normal_to_R(normal)
            t = -R @ center_3d
            rotation_center = np.array([0., 0.])
            plate_z_final = 0.0

            cv2.destroyWindow('Auto Calibrate')
            verify_img = make_verify_image(
                color_for_verify, R, t, radius,
                intr.fx, intr.fy, intr.ppx, intr.ppy)

            # 加标题栏
            title_bar = np.zeros((40, verify_img.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, "Calibration Result - Green=Circle  Red=Center  [S=save | R=retry]",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            verify_display = np.vstack([title_bar, verify_img])

            cv2.imshow('Verify', verify_display)
            print('S=保存  R=重来')

            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('s') or key == ord('S'):
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    calib_data = {
                        'R': R.tolist(),
                        't': t.tolist(),
                        'plate_z': float(plate_z_final),
                        'rotation_center': rotation_center.tolist(),
                        'radius_m': float(radius),
                    }
                    with open(CALIB_FILE, 'w') as f:
                        json.dump(calib_data, f, indent=2, default=float)
                    print(f'\n标定已保存 → {CALIB_FILE}')
                    print(json.dumps(calib_data, indent=2, default=float))
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    return
                elif key == ord('r') or key == ord('R'):
                    cv2.destroyWindow('Verify')
                    print('重新标定...')
                    break
                elif key == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    pipeline.stop()
                    return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
