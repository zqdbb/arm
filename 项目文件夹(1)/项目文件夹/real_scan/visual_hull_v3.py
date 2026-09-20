#!/usr/bin/env python3
"""
Visual Hull v3 — 纯几何遮罩: 用 D435i 深度找转台面 → 转台面以上 = 椅子.
完全不依赖颜色/纹理, 只依赖转台面深度 (D435i 能稳定看到).
"""
import numpy as np
import cv2
import json
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCAN_DIR = BASE_DIR / 'output/open3d_scan'
COLOR_DIR = SCAN_DIR / 'color'
DEPTH_DIR = SCAN_DIR / 'depth'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = SCAN_DIR / 'camera_intrinsic.json'
OUTPUT_DIR = BASE_DIR / 'output/visual_hull_v3'
MASK_DIR = OUTPUT_DIR / 'masks'

VOXEL_SIZE = 0.003
VOLUME_X = 0.15
VOLUME_Y = 0.15
VOLUME_Z_MIN = -0.02
VOLUME_Z_MAX = -0.70


def load_data():
    calib = json.load(open(CALIB_PATH))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])
    intrinsic = json.load(open(INTRINSIC_PATH))
    M = intrinsic['intrinsic_matrix']
    return R_calib, t_calib, (M[0], M[4], M[6], M[7]), (intrinsic['width'], intrinsic['height'])


def compute_virtual_poses(R_calib, t_calib, n_frames=36):
    R_w2c = R_calib.T
    t_w2c = -R_calib.T @ t_calib
    poses = []
    for i in range(n_frames):
        theta = np.radians(i * 10)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        poses.append((R_w2c @ R_z, t_w2c))
    return poses


def create_geometry_masks(w, h, R_calib, t_calib, fx, fy, ppx, ppy):
    """
    几何遮罩: 用深度图找转台面 → 不在转台面上的像素 = 前景.
    原理: D435i 稳定看到转台面 (灰色塑料). 椅子上方像素要么
    1) 深度 > 转台面以上 或 2) 无深度 (深色木头) → 都是前景.
    """
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    depth_files = sorted(DEPTH_DIR.glob('*.png'))
    n_frames = len(depth_files)

    u_arr, v_arr = np.meshgrid(np.arange(w), np.arange(h))

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    masks = []
    print(f'逐帧几何遮罩: {n_frames} 帧')

    for i, df in enumerate(depth_files):
        depth_mm = cv2.imread(str(df), -1).astype(np.float32)
        depth_m = depth_mm / 1000.0
        valid = (depth_m > 0.01) & (depth_m < 2.0)

        # 像素 → 相机坐标
        X_c = np.full((h, w), np.nan)
        Y_c = np.full((h, w), np.nan)
        Z_c = np.full((h, w), np.nan)
        X_c[valid] = (u_arr[valid] - ppx) / fx * depth_m[valid]
        Y_c[valid] = (v_arr[valid] - ppy) / fy * depth_m[valid]
        Z_c[valid] = depth_m[valid]

        # 相机 → world
        p_c = np.stack([X_c[valid], Y_c[valid], Z_c[valid]], axis=-1)
        p_w = (R_calib @ p_c.T).T + t_calib

        # RANSAC 找转台面 (world Z ≈ 0 的水平面)
        world_z = np.full((h, w), np.nan)
        world_z[valid] = p_w[:, 2]
        world_xy = np.full((h, w), np.nan)
        world_xy[valid] = np.sqrt(p_w[:, 0]**2 + p_w[:, 1]**2)

        # 转台面附近: Z ≈ 0 ± 1.5cm, XY < 12cm
        on_table = valid & (np.abs(world_z) < 0.015) & (world_xy < 0.12)

        # 前景 mask: 满足以下任一条件
        # a) 在转台区域内 (XY<12cm) 且 Z 显著高于转台面 (Z < -0.01 = 高于转台面)
        above_table = valid & (world_xy < 0.15) & (world_z < -0.008)

        # b) 在转台区域内但没有深度值 (深色木头造成的空洞)
        no_depth_in_roi = (~valid) & _in_roi_mask(w, h, R_calib, t_calib, fx, fy, ppx, ppy)

        # c) 深度无效但在画面中心区域 (保守兜底)
        center_margin = 80
        in_center = np.zeros((h, w), dtype=bool)
        in_center[h//2-center_margin:h//2+center_margin,
                  w//2-center_margin:w//2+center_margin] = True
        no_depth_center = (~valid) & in_center

        # 合并
        fg = above_table | no_depth_in_roi | no_depth_center

        mask = fg.astype(np.uint8) * 255

        # 形态学: 闭运算填小孔 + 膨胀保证连续性
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)

        # 只保留最大连通区
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                mask = (labels == (np.argmax(areas) + 1)).astype(np.uint8) * 255

        masks.append(mask)

        cv2.imwrite(str(MASK_DIR / f'{i:06d}_mask.png'), mask)

        if (i + 1) % 6 == 0:
            fg_pct = (mask > 0).sum() / mask.size * 100
            print(f'  [{i+1}/{n_frames}] 前景={fg_pct:.1f}%')

    # 保存样本
    color0 = cv2.imread(str(sorted(COLOR_DIR.glob('*.jpg'))[0]))
    cv2.imwrite(str(OUTPUT_DIR / 'mask_sample.png'), masks[0])
    cv2.imwrite(str(OUTPUT_DIR / 'masked_sample.jpg'),
                cv2.bitwise_and(color0, color0, mask=masks[0]))

    # 一致性诊断
    fg_sum = np.sum([m > 0 for m in masks], axis=0).astype(np.float32)
    fg_freq = fg_sum / n_frames
    cv2.imwrite(str(OUTPUT_DIR / 'fg_frequency.png'),
                (fg_freq * 255).astype(np.uint8))
    consistent = (fg_freq > 0.5).sum() / fg_freq.size * 100
    print(f'  多帧一致前景 (>50%帧): {consistent:.1f}%')
    return masks


def _in_roi_mask(h, w, R_calib, t_calib, fx, fy, ppx, ppy):
    """在转台圆柱区域内 (world XY < 15cm) 但没有有效深度的像素."""
    # 简化: 对画面中心 60% 区域做标记
    mask = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    margin_y, margin_x = int(h * 0.35), int(w * 0.35)
    mask[cy - margin_y:cy + margin_y, cx - margin_x:cx + margin_x] = True
    return mask


def build_visual_hull(masks, poses, fx, fy, ppx, ppy, w, h):
    nx = int(2 * VOLUME_X / VOXEL_SIZE) + 1
    ny = int(2 * VOLUME_Y / VOXEL_SIZE) + 1
    nz = int(abs(VOLUME_Z_MAX - VOLUME_Z_MIN) / VOXEL_SIZE) + 1

    print(f'\n体素网格: {nx}×{ny}×{nz} = {nx*ny*nz/1e6:.1f}M')
    print(f'体素: {VOXEL_SIZE*1000:.0f}mm | X±{VOLUME_X} Y±{VOLUME_Y} Z[{VOLUME_Z_MIN},{VOLUME_Z_MAX}]')

    x = np.linspace(-VOLUME_X, VOLUME_X, nx, dtype=np.float32)
    y = np.linspace(-VOLUME_Y, VOLUME_Y, ny, dtype=np.float32)
    z = np.linspace(VOLUME_Z_MIN, VOLUME_Z_MAX, nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    n_voxels = len(voxels)
    inside = np.ones(n_voxels, dtype=bool)
    n_views = len(masks)

    print(f'空间雕刻: {n_views} 视角')
    t0 = time.time()

    for i in range(n_views):
        R_v, t_v = poses[i]
        mask = masks[i]

        p_cam = (R_v @ voxels[inside].T).T + t_v
        u_proj = (fx * p_cam[:, 0] / p_cam[:, 2] + ppx).astype(int)
        v_proj = (fy * p_cam[:, 1] / p_cam[:, 2] + ppy).astype(int)
        in_img = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h) & (p_cam[:, 2] > 0.01)

        inside_indices = np.where(inside)[0]
        mask_vals = np.zeros(len(p_cam), dtype=bool)
        valid_idx = np.where(in_img)[0]
        if len(valid_idx) > 0:
            mask_vals[valid_idx] = mask[v_proj[valid_idx], u_proj[valid_idx]] > 128
        inside[inside_indices] = in_img & mask_vals

        if (i + 1) % 6 == 0:
            print(f'  [{i+1}/{n_views}] {inside.sum():,} ({inside.sum()/n_voxels*100:.2f}%)')

    dt = time.time() - t0
    pts = voxels[inside]
    print(f'  耗时: {dt:.0f}s | Visual Hull: {len(pts):,} 点')
    return pts


def supplement_depth_points(R_calib, t_calib, fx, fy, ppx, ppy, w, h):
    depth_files = sorted(DEPTH_DIR.glob('*.png'))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    all_pts = []

    for i, df in enumerate(depth_files):
        depth_m = cv2.imread(str(df), -1).astype(np.float32) * 0.001
        valid = (depth_m > 0.2) & (depth_m < 1.0)
        if valid.sum() < 1000:
            continue
        X_c = (u[valid] - ppx) / fx * depth_m[valid]
        Y_c = (v[valid] - ppy) / fy * depth_m[valid]
        Z_c = depth_m[valid]
        p_c = np.column_stack([X_c, Y_c, Z_c])
        p_w = (R_calib @ p_c.T).T + t_calib

        theta = np.radians(i * 10)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_undo = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        p_obj = (R_undo @ p_w.T).T

        in_vol = ((np.abs(p_obj[:, 0]) <= VOLUME_X) &
                  (np.abs(p_obj[:, 1]) <= VOLUME_Y) &
                  (p_obj[:, 2] >= VOLUME_Z_MIN) &
                  (p_obj[:, 2] <= VOLUME_Z_MAX))
        if in_vol.sum() > 500:
            step = max(1, in_vol.sum() // 3000)
            all_pts.append(p_obj[in_vol][::step])

    return np.vstack(all_pts) if all_pts else np.empty((0, 3))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('=' * 55)
    print('  Visual Hull v3 — 几何遮罩 + 空间雕刻')
    print('=' * 55)

    R_calib, t_calib, (fx, fy, ppx, ppy), (w, h) = load_data()
    print(f'内参: fx={fx:.1f} fy={fy:.1f} 尺寸={w}×{h}')
    cam_world = -R_calib.T @ t_calib
    print(f'相机 world: [{cam_world[0]:.3f},{cam_world[1]:.3f},{cam_world[2]:.3f}]')

    # Step 1: 几何遮罩
    print(f'\n{"─"*50}')
    print('遮罩生成 (深度几何约束)')
    masks = create_geometry_masks(w, h, R_calib, t_calib, fx, fy, ppx, ppy)

    # Step 2: 位姿
    print(f'\n{"─"*50}')
    poses = compute_virtual_poses(R_calib, t_calib, len(masks))
    for idx in [0, 9, 18, 27]:
        R_v, t_v = poses[idx]
        c = -R_v.T @ t_v
        print(f'  帧{idx:2d} ({idx*10:3d}°): cam=[{c[0]:.3f},{c[1]:.3f},{c[2]:.3f}]')

    # Step 3: Visual Hull
    print(f'\n{"─"*50}')
    hull_pts = build_visual_hull(masks, poses, fx, fy, ppx, ppy, w, h)

    # Step 4: 深度补充
    print(f'\n{"─"*50}')
    depth_pts = supplement_depth_points(R_calib, t_calib, fx, fy, ppx, ppy, w, h)
    print(f'深度点: {len(depth_pts):,}')

    if len(hull_pts) > 0 and len(depth_pts) > 0:
        merged = np.vstack([hull_pts, depth_pts])
    elif len(hull_pts) > 0:
        merged = hull_pts
    else:
        merged = depth_pts

    if len(merged) == 0:
        print('\n无点生成! 检查相机位姿或深度数据.')
        return

    # 翻转 Z
    merged_out = merged.copy()
    merged_out[:, 2] = -merged_out[:, 2]

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_out.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if len(pcd.points) > 20:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    out = OUTPUT_DIR / 'visual_hull.ply'
    o3d.io.write_point_cloud(str(out), pcd)

    pts = np.asarray(pcd.points)
    print(f'\n{"="*55}')
    print(f'  输出: {out} ({len(pts):,} 点)')
    print(f'  XYZ: [{pts[:,0].min():.3f},{pts[:,0].max():.3f}] '
          f'[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] '
          f'[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]')
    for lo, hi, label in [(0.0, 0.02, '转盘'), (0.02, 0.15, '腿'),
                           (0.15, 0.40, '座面'), (0.40, 0.75, '靠背')]:
        n = ((pts[:, 2] >= lo) & (pts[:, 2] < hi)).sum()
        print(f'    Z [{lo:.2f},{hi:.2f}) {label}: {n:6d}')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
