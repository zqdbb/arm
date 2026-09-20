#!/usr/bin/env python3
"""
Space Carving — 纯 RGB + 标定位姿, 不碰深度图.
体素色彩一致性雕刻: 各视角颜色方差大的体素 → 雕掉(空气),
方差小的 → 保留(物体表面/内部).
"""
import numpy as np
import cv2, json, os, glob, time
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCAN_DIR = BASE_DIR / 'output/open3d_scan'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = SCAN_DIR / 'camera_intrinsic.json'
OUTPUT_DIR = BASE_DIR / 'output/space_carving'

# 体素参数
VOXEL_SIZE = 0.001       # 1mm 体素
VOL_X = 0.06             # X ±6cm
VOL_Y = 0.06             # Y ±6cm
VOL_Z_MIN = -0.105       # 转台面以上 10.5cm
VOL_Z_MAX = 0.005        # 转台面以下一点点

# 色彩一致性阈值 (RGB 标准差, 0-255)
COLOR_THRESHOLD = 35     # 标准差 < 35 → 认为是物体


def load_data():
    calib = json.load(open(CALIB_PATH))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])

    intrinsic = json.load(open(INTRINSIC_PATH))
    M = intrinsic['intrinsic_matrix']
    fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]
    w, h = intrinsic['width'], intrinsic['height']
    return R_calib, t_calib, (fx, fy, ppx, ppy), (w, h)


def load_images():
    """预加载所有 RGB 图像."""
    color_files = sorted(glob.glob(str(SCAN_DIR / 'color' / '*.jpg')))
    images = [cv2.imread(f) for f in color_files]
    print(f'加载 {len(images)} 张 RGB 图像 ({images[0].shape[1]}×{images[0].shape[0]})')
    return images, len(images)


def compute_camera_poses(R_calib, t_calib, n_frames):
    """计算每帧的相机外参 (aligned frame 中).

    管线公式: p_aligned = R_z(+θ) @ (R_calib @ p_cam + t_calib)
    反推:     p_cam = R_calib^T @ (R_z(-θ) @ p_aligned - t_calib)
    """
    poses = []
    for i in range(n_frames):
        theta = np.radians(i * 10)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z_inv = np.array([[cos_t, sin_t, 0],
                           [-sin_t, cos_t, 0],
                           [0, 0, 1]])  # R_z(-θ)

        R_inv = R_calib.T @ R_z_inv
        t_inv = -R_calib.T @ t_calib
        poses.append((R_inv, t_inv))

    return poses


def build_volume():
    """创建体素网格."""
    nx = int(2 * VOL_X / VOXEL_SIZE) + 1
    ny = int(2 * VOL_Y / VOXEL_SIZE) + 1
    nz = int((VOL_Z_MAX - VOL_Z_MIN) / VOXEL_SIZE) + 1

    x = np.linspace(-VOL_X, VOL_X, nx, dtype=np.float32)
    y = np.linspace(-VOL_Y, VOL_Y, ny, dtype=np.float32)
    z = np.linspace(VOL_Z_MIN, VOL_Z_MAX, nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    print(f'体素网格: {nx}×{ny}×{nz} = {len(voxels)/1e6:.1f}M 体素 '
          f'(分辨率 {VOXEL_SIZE*1000:.0f}mm)')
    return voxels, (nx, ny, nz)


def space_carve(voxels, shape, poses, images, fx, fy, ppx, ppy, w, h):
    """
    色彩一致性雕刻.
    voxels: (N, 3) 体素世界坐标
    shape: (nx, ny, nz) 用于重构 3D 网格
    """
    n_voxels = len(voxels)
    n_views = len(images)
    BATCH = 200000  # 每批处理 200K 体素

    # 每个体素的颜色统计量
    color_sum = np.zeros((n_voxels, 3), dtype=np.float64)
    color_sq = np.zeros((n_voxels, 3), dtype=np.float64)
    vis_count = np.zeros(n_voxels, dtype=np.int32)

    print(f'\n逐视角雕刻: {n_views} 视角 × {n_voxels/1e6:.1f}M 体素')
    t0_total = time.time()

    for i_view, (R_inv, t_inv) in enumerate(poses):
        img = images[i_view].astype(np.float32)
        t0 = time.time()

        for start in range(0, n_voxels, BATCH):
            end = min(start + BATCH, n_voxels)
            batch = voxels[start:end]

            # 变换到相机坐标: p_cam = R_inv @ p_world + t_inv
            p_cam = (R_inv @ batch.T).T + t_inv

            # 可见性检查
            z_cam = p_cam[:, 2]
            in_front = z_cam > 0.05
            if not in_front.any():
                continue

            idx = np.where(in_front)[0]
            u = (fx * p_cam[idx, 0] / z_cam[idx] + ppx).astype(int)
            v = (fy * p_cam[idx, 1] / z_cam[idx] + ppy).astype(int)
            in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)

            if not in_img.any():
                continue

            idx = idx[in_img]
            u, v = u[in_img], v[in_img]

            # 采样颜色
            colors = img[v, u]  # BGR
            global_idx = start + idx

            color_sum[global_idx] += colors
            color_sq[global_idx] += colors ** 2
            vis_count[global_idx] += 1

        dt = time.time() - t0
        if (i_view + 1) % 6 == 0 or i_view == 0:
            print(f'  [{i_view+1}/{n_views}] {dt:.1f}s')

    dt_total = time.time() - t0_total
    print(f'  总耗时: {dt_total:.0f}s')

    # 计算色彩方差
    visible = vis_count >= 6  # 至少 6 个视角看到
    print(f'  多视角可见体素: {visible.sum():,}/{n_voxels:,}')

    if visible.sum() == 0:
        print('  无足够可见体素!')
        return None

    mean = color_sum[visible] / vis_count[visible, None]
    sq_mean = color_sq[visible] / vis_count[visible, None]
    var = sq_mean - mean ** 2
    std = np.sqrt(np.maximum(var, 0))
    max_std = std.max(axis=1)  # RGB 三通道最大标准差

    # 色彩一致 = 物体; 不一致 = 空气
    occupied = max_std < COLOR_THRESHOLD
    n_occupied = occupied.sum()
    print(f'  色彩一致体素: {n_occupied:,} ({n_occupied/visible.sum()*100:.1f}%)')
    print(f'  std 范围: [{max_std.min():.1f}, {max_std.max():.1f}]')
    print(f'  std 分位数: 10%={np.percentile(max_std,10):.1f} '
          f'25%={np.percentile(max_std,25):.1f} '
          f'50%={np.percentile(max_std,50):.1f} '
          f'75%={np.percentile(max_std,75):.1f} '
          f'90%={np.percentile(max_std,90):.1f}')

    if n_occupied < 100:
        print('  物体体素太少, 尝试降低阈值')
        return None

    return voxels[visible][occupied], max_std[occupied], vis_count[visible][occupied]


def export_results(pts, shape, std_vals):
    """导出体素模型."""
    import open3d as o3d

    # 翻转 Z (world Z↓ → 视觉 Z↑)
    pts_out = pts.copy()
    pts_out[:, 2] = -pts_out[:, 2]

    # 导出体素点云（占用体素的中心点）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))

    # 用色彩方差着色 (低方差=绿, 高方差=红)
    std_norm = np.clip(std_vals / COLOR_THRESHOLD, 0, 1)
    colors = np.zeros((len(pts_out), 3))
    colors[:, 0] = std_norm  # R: 方差越大越红
    colors[:, 1] = 1 - std_norm  # G: 方差越小越绿
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_ply = str(OUTPUT_DIR / 'carved_volume.ply')
    o3d.io.write_point_cloud(out_ply, pcd)

    # 也导出为体素网格 (每个体素一个 cube)
    # 降采样到 2mm 避免太大
    pcd_down = pcd.voxel_down_sample(voxel_size=0.002)
    out_down = str(OUTPUT_DIR / 'carved_volume_2mm.ply')
    o3d.io.write_point_cloud(out_down, pcd_down)

    print(f'\n{"="*55}')
    print(f'  Space Carving 完成')
    print(f'  占用体素: {len(pts):,} (1mm 分辨率)')
    print(f'  降采样: {len(pcd_down.points):,} (2mm 分辨率)')
    print(f'  XYZ: [{pts_out[:,0].min():.3f},{pts_out[:,0].max():.3f}] '
          f'[{pts_out[:,1].min():.3f},{pts_out[:,1].max():.3f}] '
          f'[{pts_out[:,2].min():.3f},{pts_out[:,2].max():.3f}]')
    print(f'  输出: {out_ply}')
    print(f'  输出: {out_down}')
    print(f'{"="*55}')

    return out_ply


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('  Space Carving — RGB 色彩一致性体素雕刻')
    print('=' * 55)

    R_calib, t_calib, (fx, fy, ppx, ppy), (w, h) = load_data()
    print(f'内参: fx={fx:.1f} fy={fy:.1f} 尺寸={w}×{h}')

    images, n_frames = load_images()

    print('\n计算相机位姿...')
    poses = compute_camera_poses(R_calib, t_calib, n_frames)
    for idx in [0, 9, 18, 27]:
        R_inv, t_inv = poses[idx]
        # 相机位置: p_cam = 0 → 0 = R_inv @ cam + t_inv → cam = -R_inv.T @ t_inv
        c = -R_inv.T @ t_inv
        print(f'  帧{idx:2d} ({idx*10:3d}°): cam=[{c[0]:.3f},{c[1]:.3f},{c[2]:.3f}]')

    print('\n构建体素网格...')
    voxels, shape = build_volume()

    print(f'\n色彩一致性阈值: std < {COLOR_THRESHOLD} (0-255)')
    result = space_carve(voxels, shape, poses, images, fx, fy, ppx, ppy, w, h)

    if result is not None:
        pts, std_vals, _ = result
        export_results(pts, shape, std_vals)
    else:
        print('\n雕刻失败')

    # 诊断: 输出不同阈值的结果对比
    print(f'\n{"─"*50}')
    print('阈值诊断 (不同 std 阈值下的占用率):')
    visible_voxels = voxels  # saved from space_carve
    for thresh in [15, 25, 35, 45, 60, 80]:
        if result is not None:
            occupied = (result[1] < thresh).sum()
            pct = occupied / len(result[1]) * 100
            print(f'  std < {thresh:2d}: {occupied:7,} / {len(result[1]):,} ({pct:.1f}%)')


if __name__ == '__main__':
    main()
