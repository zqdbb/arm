#!/usr/bin/env python3
"""
Visual Hull v2 — CPU-only 3D reconstruction for turntable + dark wood furniture.
改进遮罩策略: 中值背景建模 + 自适应阈值 + 多帧一致性 + GrabCut 精化.
完全不依赖深度图, 仅用 RGB + 已知位姿 → 空间雕刻 → 点云.
"""
import numpy as np
import cv2
import json
import os
import time
from pathlib import Path

# ── 路径 ──
BASE_DIR = Path(__file__).parent
SCAN_DIR = BASE_DIR / 'output/open3d_scan'
COLOR_DIR = SCAN_DIR / 'color'
DEPTH_DIR = SCAN_DIR / 'depth'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = SCAN_DIR / 'camera_intrinsic.json'
OUTPUT_DIR = BASE_DIR / 'output/visual_hull_v2'
MASK_DIR = OUTPUT_DIR / 'masks'

# ── 体素参数 ──
VOXEL_SIZE = 0.003          # 体素 3mm
VOLUME_X = 0.15             # X 半范围 (m)
VOLUME_Y = 0.15             # Y 半范围 (m)
VOLUME_Z_MIN = -0.02         # Z 下界 (转盘面略下), world Z 朝下
VOLUME_Z_MAX = -0.70        # Z 上界 (靠背顶), 越负越高

# ── 遮罩参数 ──
BG_DIFF_THRESH = 30          # 背景差分基础阈值
MORPH_CLOSE_SIZE = 9         # 闭运算核大小
MORPH_OPEN_SIZE = 5          # 开运算核大小
MIN_FG_AREA = 500            # 最小前景面积


def load_data():
    calib = json.load(open(CALIB_PATH))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])

    intrinsic = json.load(open(INTRINSIC_PATH))
    M = intrinsic['intrinsic_matrix']
    fx, fy = M[0], M[4]
    ppx, ppy = M[6], M[7]
    w, h = intrinsic['width'], intrinsic['height']
    return R_calib, t_calib, (fx, fy, ppx, ppy), (w, h)


def compute_virtual_poses(R_calib, t_calib, n_frames=36):
    """对象固定, 虚拟相机绕 Z 轴转 (相机→世界 的逆)."""
    R_w2c = R_calib.T
    t_w2c = -R_calib.T @ t_calib

    poses = []
    for i in range(n_frames):
        theta = np.radians(i * 10)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z = np.array([[cos_t, -sin_t, 0],
                        [sin_t,  cos_t, 0],
                        [0,       0,     1]])
        poses.append((R_w2c @ R_z, t_w2c))
    return poses


# ═══════════════════════════════════════════════════════════
#  遮罩生成: 改进版背景减除 + 深度辅助
# ═══════════════════════════════════════════════════════════

def create_improved_masks(w, h, R_calib, t_calib, fx, fy, ppx, ppy):
    """
    两步法遮罩:
    1. 中值背景建模 + 逐帧自适应差分 → 粗遮罩
    2. GrabCut 以粗遮罩为初始 → 精遮罩
    """
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    color_files = sorted(COLOR_DIR.glob('*.jpg'))
    depth_files = sorted(DEPTH_DIR.glob('*.png'))
    n_frames = len(color_files)
    print(f'  {n_frames} 帧')

    # ── 加载所有帧 ──
    all_frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    for i, cf in enumerate(color_files):
        all_frames[i] = cv2.imread(str(cf))

    # ── Step 1: 中值背景模型 ──
    print('  [1/3] 中值背景模型...')
    bg_model = np.median(all_frames, axis=0).astype(np.uint8)
    bg_gray = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(OUTPUT_DIR / 'background_model.jpg'), bg_model)

    # ── 逐像素方差 (用于自适应阈值) ──
    all_gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in all_frames])
    per_pixel_std = np.std(all_gray.astype(np.float32), axis=0)
    # 平滑方差图
    per_pixel_std = cv2.GaussianBlur(per_pixel_std, (5, 5), 0)
    cv2.imwrite(str(OUTPUT_DIR / 'pixel_std.png'),
                (per_pixel_std / per_pixel_std.max() * 255).astype(np.uint8))

    # ── Step 2: 逐帧差分 + 自适应阈值 ──
    print('  [2/3] 逐帧差分...')
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_SIZE, MORPH_CLOSE_SIZE))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_SIZE, MORPH_OPEN_SIZE))

    raw_masks = []
    for i in range(n_frames):
        color = all_frames[i]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # 彩色差分 (各通道最大)
        diff = cv2.absdiff(color.astype(np.int16), bg_model.astype(np.int16))
        diff_gray = diff.max(axis=2).astype(np.uint8)

        # 灰度差分
        gray_diff = cv2.absdiff(gray.astype(np.int16), bg_gray.astype(np.int16)).astype(np.uint8)

        # 自适应阈值: 方差大的区域用高阈值
        local_thresh = BG_DIFF_THRESH + per_pixel_std * 0.5
        local_thresh = np.clip(local_thresh, BG_DIFF_THRESH, BG_DIFF_THRESH * 3)

        # 综合彩色+灰度差分
        combined = cv2.addWeighted(diff_gray, 0.7, gray_diff, 0.3, 0)

        # 逐像素自适应
        fg = (combined > local_thresh).astype(np.uint8) * 255

        # 形态学清理
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)

        # 找最大连通区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest = np.argmax(areas) + 1
                fg = (labels == largest).astype(np.uint8) * 255

        raw_masks.append(fg)

        if (i + 1) % 6 == 0:
            fg_frac = (fg > 0).sum() / fg.size * 100
            print(f'    [{i+1}/{n_frames}] 前景={fg_frac:.1f}%')

    # ── Step 3: 多帧一致性过滤 ──
    print('  [3/3] 多帧一致性 + GrabCut 精化...')
    # 对于背景减除: 如果某个像素在 >70% 的帧中是前景 → 可能是背景噪声
    # 如果 <20% 的帧中是前景 → 可能是椅子某部分被遮挡太严重
    fg_sum = np.sum([m > 0 for m in raw_masks], axis=0).astype(np.float32)
    fg_freq = fg_sum / n_frames
    cv2.imwrite(str(OUTPUT_DIR / 'fg_frequency.png'),
                (fg_freq * 255).astype(np.uint8))

    masks = []
    for i in range(n_frames):
        color = all_frames[i]
        raw = raw_masks[i]

        # 用深度图+标定筛选确信前/背景 (只有转台区域内的深度点才是前景)
        depth = cv2.imread(str(depth_files[i]), -1).astype(np.float32) * 0.001
        depth_valid = (depth > 0.2) & (depth < 1.2)

        sure_fg = np.zeros((h, w), np.uint8)
        sure_bg = np.zeros((h, w), np.uint8)

        if depth_valid.sum() > 100:
            u_arr, v_arr = np.meshgrid(np.arange(w), np.arange(h))
            dv = depth[depth_valid]
            X_c = (u_arr[depth_valid] - ppx) / fx * dv
            Y_c = (v_arr[depth_valid] - ppy) / fy * dv
            Z_c = dv
            p_c = np.column_stack([X_c, Y_c, Z_c])
            p_w = (R_calib @ p_c.T).T + t_calib
            dist_xy = np.sqrt(p_w[:, 0]**2 + p_w[:, 1]**2)

            fg_idx = np.where(depth_valid)
            sure_fg[fg_idx[0][dist_xy < 0.12], fg_idx[1][dist_xy < 0.12]] = cv2.GC_FGD
            sure_bg[fg_idx[0][dist_xy > 0.20], fg_idx[1][dist_xy > 0.20]] = cv2.GC_BGD

        # 图像边缘 → 确信背景
        border = 15
        sure_bg[:border, :] = cv2.GC_BGD
        sure_bg[-border:, :] = cv2.GC_BGD
        sure_bg[:, :border] = cv2.GC_BGD
        sure_bg[:, -border:] = cv2.GC_BGD

        # 粗遮罩作为可能前景
        probable_fg = np.zeros((h, w), np.uint8)
        probable_fg[raw > 0] = cv2.GC_PR_FGD

        # GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        mask_gc = np.where(sure_fg > 0, cv2.GC_FGD,
                           np.where(sure_bg > 0, cv2.GC_BGD,
                                    np.where(probable_fg > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD)))

        try:
            mask_gc, _, _ = cv2.grabCut(color, mask_gc, None, bgd_model, fgd_model,
                                        3, cv2.GC_INIT_WITH_MASK)
            final = ((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        except cv2.error:
            final = raw

        # 最终形态学
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel_close)

        masks.append(final)

        cv2.imwrite(str(MASK_DIR / f'{i:06d}_mask.png'), final)
        cv2.imwrite(str(MASK_DIR / f'{i:06d}_masked.jpg'),
                    cv2.bitwise_and(color, color, mask=final))

        if (i + 1) % 12 == 0:
            fg_frac = (final > 0).sum() / final.size * 100
            print(f'    [{i+1}/{n_frames}] 精化后前景={fg_frac:.1f}%')

    # 保存样本
    cv2.imwrite(str(OUTPUT_DIR / 'mask_sample.png'), masks[0])
    cv2.imwrite(str(OUTPUT_DIR / 'masked_sample.jpg'),
                cv2.bitwise_and(all_frames[0], all_frames[0], mask=masks[0]))
    return masks


# ═══════════════════════════════════════════════════════════
#  Visual Hull 空间雕刻
# ═══════════════════════════════════════════════════════════

def build_visual_hull(masks, poses, fx, fy, ppx, ppy, w, h):
    nx = int(2 * VOLUME_X / VOXEL_SIZE) + 1
    ny = int(2 * VOLUME_Y / VOXEL_SIZE) + 1
    nz = int(abs(VOLUME_Z_MAX - VOLUME_Z_MIN) / VOXEL_SIZE) + 1

    print(f'\n体素网格: {nx}×{ny}×{nz} = {nx*ny*nz/1e6:.1f}M 体素')
    print(f'体素尺寸: {VOXEL_SIZE*1000:.0f}mm')
    print(f'范围: X±{VOLUME_X}m  Y±{VOLUME_Y}m  Z[{VOLUME_Z_MIN},{VOLUME_Z_MAX}]m')

    x = np.linspace(-VOLUME_X, VOLUME_X, nx, dtype=np.float32)
    y = np.linspace(-VOLUME_Y, VOLUME_Y, ny, dtype=np.float32)
    z = np.linspace(VOLUME_Z_MIN, VOLUME_Z_MAX, nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    n_voxels = len(voxels)

    inside = np.ones(n_voxels, dtype=bool)
    n_views = len(masks)
    print(f'\n空间雕刻: {n_views} 个视角')
    t0 = time.time()

    for i in range(n_views):
        R_v, t_v = poses[i]
        mask = masks[i]

        if mask.sum() < MIN_FG_AREA:
            print(f'  [{i+1}/{n_views}] 跳过 (遮罩面积不足)')
            continue

        p_cam = (R_v @ voxels[inside].T).T + t_v

        u_proj = (fx * p_cam[:, 0] / p_cam[:, 2] + ppx).astype(int)
        v_proj = (fy * p_cam[:, 1] / p_cam[:, 2] + ppy).astype(int)

        in_img = (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h) & (p_cam[:, 2] > 0.01)

        inside_indices = np.where(inside)[0]
        mask_vals = np.zeros(len(p_cam), dtype=bool)
        valid_indices = np.where(in_img)[0]
        if len(valid_indices) > 0:
            mask_vals[valid_indices] = mask[v_proj[valid_indices], u_proj[valid_indices]] > 128

        inside[inside_indices] = in_img & mask_vals

        if (i + 1) % 6 == 0:
            surviving = inside.sum()
            pct = surviving / n_voxels * 100
            print(f'  [{i+1}/{n_views}] 剩余 {surviving:,} 体素 ({pct:.2f}%)')

    dt = time.time() - t0
    hull_points = voxels[inside]
    print(f'  耗时: {dt:.0f}s  |  Visual Hull: {len(hull_points):,} 点')
    return hull_points


# ═══════════════════════════════════════════════════════════
#  深度点补充 (D435i 能看到的部分)
# ═══════════════════════════════════════════════════════════

def supplement_depth_points(R_calib, t_calib, fx, fy, ppx, ppy, w, h):
    """提取 D435i 有效深度点, 变换到对象坐标系."""
    depth_files = sorted(DEPTH_DIR.glob('*.png'))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    all_pts = []

    for i, df in enumerate(depth_files):
        depth_mm = cv2.imread(str(df), -1).astype(np.float32)
        depth_m = depth_mm / 1000.0
        valid = (depth_m > 0.15) & (depth_m < 0.8)
        if valid.sum() < 1000:
            continue

        X_c = (u[valid] - ppx) / fx * depth_m[valid]
        Y_c = (v[valid] - ppy) / fy * depth_m[valid]
        Z_c = depth_m[valid]
        p_c = np.column_stack([X_c, Y_c, Z_c])
        p_w = (R_calib @ p_c.T).T + t_calib

        # undo turntable rotation
        theta = np.radians(i * 10)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_undo = np.array([[cos_t, -sin_t, 0],
                           [sin_t,  cos_t, 0],
                           [0,       0,     1]])
        p_obj = (R_undo @ p_w.T).T

        in_vol = ((np.abs(p_obj[:, 0]) <= VOLUME_X) &
                  (np.abs(p_obj[:, 1]) <= VOLUME_Y) &
                  (p_obj[:, 2] >= VOLUME_Z_MIN) &
                  (p_obj[:, 2] <= VOLUME_Z_MAX))

        if in_vol.sum() > 500:
            sample_step = max(1, in_vol.sum() // 3000)
            all_pts.append(p_obj[in_vol][::sample_step])

    if all_pts:
        return np.vstack(all_pts)
    return np.empty((0, 3))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('  Visual Hull v2 — CPU 空间雕刻重建')
    print('=' * 55)

    R_calib, t_calib, (fx, fy, ppx, ppy), (w, h) = load_data()
    print(f'内参: fx={fx:.1f} fy={fy:.1f} 尺寸={w}×{h}')
    cam_world = -R_calib.T @ t_calib
    print(f'相机 world 位置: [{cam_world[0]:.3f}, {cam_world[1]:.3f}, {cam_world[2]:.3f}]')

    # Step 1: 改进遮罩
    print(f'\n{"─" * 50}')
    print('遮罩生成 (背景减除 + GrabCut)')
    masks = create_improved_masks(w, h, R_calib, t_calib, fx, fy, ppx, ppy)

    # Step 2: 虚拟相机位姿
    print(f'\n{"─" * 50}')
    poses = compute_virtual_poses(R_calib, t_calib, len(masks))
    for idx in [0, len(poses)//4, len(poses)//2]:
        R_v, t_v = poses[idx]
        center = -R_v.T @ t_v
        print(f'  帧{idx:2d} ({idx*10:3d}°): cam_center=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}]')

    # Step 3: Visual Hull
    print(f'\n{"─" * 50}')
    hull_pts = build_visual_hull(masks, poses, fx, fy, ppx, ppy, w, h)

    # Step 4: 深度补充
    print(f'\n{"─" * 50}')
    print('深度点补充...')
    depth_pts = supplement_depth_points(R_calib, t_calib, fx, fy, ppx, ppy, w, h)
    print(f'  深度点: {len(depth_pts):,}')

    if len(hull_pts) > 0 and len(depth_pts) > 0:
        merged = np.vstack([hull_pts, depth_pts])
    elif len(hull_pts) > 0:
        merged = hull_pts
    else:
        merged = depth_pts

    # Step 5: 翻转 Z (world Z 朝下 → 标准 Z 朝上)
    merged_out = merged.copy()
    merged_out[:, 2] = -merged_out[:, 2]

    # Step 6: 保存
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_out.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    out_path = OUTPUT_DIR / 'visual_hull.ply'
    o3d.io.write_point_cloud(str(out_path), pcd)

    # 统计
    pts = np.asarray(pcd.points)
    print(f'\n{"=" * 55}')
    print(f'  输出: {out_path} ({len(pcd.points):,} 点)')
    print(f'  XYZ 范围: X[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] '
          f'Y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] '
          f'Z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]')
    for lo, hi, label in [(0.0, 0.02, '转盘面'), (0.02, 0.15, '椅腿'),
                           (0.15, 0.40, '座面'), (0.40, 0.75, '靠背')]:
        n = ((pts[:, 2] >= lo) & (pts[:, 2] < hi)).sum()
        print(f'    Z [{lo:.2f},{hi:.2f}) {label}: {n:6d} 点')
    print(f'{"=" * 55}')
    return out_path


if __name__ == '__main__':
    main()
