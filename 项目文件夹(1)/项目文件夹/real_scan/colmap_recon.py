#!/usr/bin/env python3
"""
COLMAP RGB 重建管线 — 从彩色图重建家具3D点云
策略：深度图做前景遮罩 → COLMAP SfM+MVS → 稠密点云
"""
import numpy as np
import cv2
import json
import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path

# ─── 路径 ───
BASE_DIR = Path(__file__).parent
SCAN_DIR = BASE_DIR / 'output/open3d_scan'
COLOR_DIR = SCAN_DIR / 'color'
DEPTH_DIR = SCAN_DIR / 'depth'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = SCAN_DIR / 'camera_intrinsic.json'
COLMAP_DIR = BASE_DIR / 'output/colmap_scan'
MASKED_DIR = COLMAP_DIR / 'masked'
SPARSE_DIR = COLMAP_DIR / 'sparse'
DENSE_DIR = COLMAP_DIR / 'dense'
DATABASE_PATH = COLMAP_DIR / 'database.db'

# ─── 参数 ───
MASK_DILATE_ITER = 15       # 膨胀迭代次数 (填充椅子空洞)
MASK_DEPTH_NEAR = -0.05     # World Z 下界 (转盘以下5cm)
MASK_DEPTH_FAR = 0.50       # World Z 上界
MASK_RADIUS = 0.20          # XY 半径限制
MAX_DEPTH_MM = 1500         # 有效深度上限


def load_calibration():
    calib = json.load(open(CALIB_PATH))
    R = np.array(calib['R'])
    t = np.array(calib['t'])
    K_data = json.load(open(INTRINSIC_PATH))
    M = K_data['intrinsic_matrix']
    fx, fy = M[0], M[4]
    ppx, ppy = M[6], M[7]
    w, h = K_data['width'], K_data['height']
    return R, t, (fx, fy, ppx, ppy), (w, h)


def create_foreground_mask(depth_mm, R, t, fx, fy, ppx, ppy):
    """
    深度图 → 前景遮罩 (0=背景, 255=前景)
    策略: 标记转台区域(有深度+Z≈0+XY小) → 膨胀填充椅子空洞
    """
    h, w = depth_mm.shape
    depth_m = depth_mm.astype(float) / 1000.0

    # 像素→相机坐标
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X_c = (u - ppx) / fx * depth_m
    Y_c = (v - ppy) / fy * depth_m
    Z_c = depth_m

    # 有效深度
    valid = (depth_m > 0.01) & (depth_m < MAX_DEPTH_MM / 1000.0)

    # 变换到 world
    world_z = np.full((h, w), np.nan)
    world_xy = np.full((h, w), np.nan)

    p_c = np.stack([X_c[valid], Y_c[valid], Z_c[valid]], axis=-1)
    p_w = (R @ p_c.T).T + t
    world_z[valid] = p_w[:, 2]
    world_xy[valid] = np.sqrt(p_w[:, 0]**2 + p_w[:, 1]**2)

    # 前景条件: 在转台面附近 (Z ≈ 0) 且 在转台半径内
    on_table = valid & (world_z > MASK_DEPTH_NEAR) & (world_z < MASK_DEPTH_FAR) & (world_xy < MASK_RADIUS)
    mask = on_table.astype(np.uint8) * 255

    # 膨胀填充椅子造成的空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=MASK_DILATE_ITER)

    # 再腐蚀回来 (保持边界
    mask = cv2.erode(mask, kernel, iterations=max(1, MASK_DILATE_ITER - 3))

    return mask


def prepare_masked_images():
    """为所有36帧生成前景遮罩 + 遮罩后图片"""
    R, t, (fx, fy, ppx, ppy), _ = load_calibration()

    MASKED_DIR.mkdir(parents=True, exist_ok=True)

    color_files = sorted(COLOR_DIR.glob('*.jpg'))
    depth_files = sorted(DEPTH_DIR.glob('*.png'))

    print(f'生成遮罩图片: {len(color_files)} 帧')

    for i, (cf, df) in enumerate(zip(color_files, depth_files)):
        color = cv2.imread(str(cf))
        depth_mm = cv2.imread(str(df), -1)

        mask = create_foreground_mask(depth_mm, R, t, fx, fy, ppx, ppy)

        # 遮罩应用到彩图
        masked = color.copy()
        masked[mask == 0] = 0

        # 可选: crop 到非零区域
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            y1, y2 = max(0, ys.min() - 10), min(color.shape[0], ys.max() + 10)
            x1, x2 = max(0, xs.min() - 10), min(color.shape[1], xs.max() + 10)
            # 保持原尺寸用于COLMAP，只是设黑色背景
            pass

        out_path = MASKED_DIR / cf.name
        cv2.imwrite(str(out_path), masked)

        if (i + 1) % 12 == 0:
            print(f'  [{i+1}/{len(color_files)}]')

        # 保存第一帧mask用于调试
        if i == 0:
            cv2.imwrite(str(COLMAP_DIR / 'mask_debug.png'), mask)
            cv2.imwrite(str(COLMAP_DIR / 'masked_debug.jpg'), masked)

    # 也复制内参文件
    shutil.copy(INTRINSIC_PATH, COLMAP_DIR / 'camera_intrinsic.json')
    print(f'遮罩图片完成 → {MASKED_DIR}/')
    print(f'调试: {COLMAP_DIR}/mask_debug.png, masked_debug.jpg')


def run_colmap(use_masked=True):
    """运行 COLMAP SfM + MVS 管线

    use_masked=True:  用遮罩图 (需要足够特征)
    use_masked=False: 用原始图 (背景纹理 → SfM更稳), 后处理裁剪
    """
    K_data = json.load(open(INTRINSIC_PATH))
    M = K_data['intrinsic_matrix']
    fx, fy = M[0], M[4]
    ppx, ppy = M[6], M[7]
    w, h = K_data['width'], K_data['height']

    image_dir = MASKED_DIR if use_masked else COLOR_DIR
    tag = 'masked' if use_masked else 'original'

    SPARSE_DIR.mkdir(parents=True, exist_ok=True)
    DENSE_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. 特征提取 ──
    print(f'\n[1/6] 特征提取 ({tag})...')
    cmd = [
        'colmap', 'feature_extractor',
        '--database_path', str(DATABASE_PATH),
        '--image_path', str(image_dir),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'PINHOLE',
        f'--ImageReader.camera_params', f'{fx},{fy},{ppx},{ppy}',
        '--SiftExtraction.max_num_features', '8192',
        '--SiftExtraction.estimate_affine_shape', '0',
        '--SiftExtraction.domain_size_pooling', '1',
    ]
    subprocess.run(cmd, check=True)

    # ── 2. 特征匹配 ──
    print(f'\n[2/6] 特征匹配 (sequential, overlap=10)...')
    # overlap=10: 匹配前后10帧 (100°范围) 以应对低纹理
    cmd = [
        'colmap', 'sequential_matcher',
        '--database_path', str(DATABASE_PATH),
        '--SequentialMatching.overlap', '10',
        '--SiftMatching.max_num_matches', '32768',
        '--SiftMatching.max_ratio', '0.8',  # 宽松一点
    ]
    subprocess.run(cmd, check=True)

    # ── 3. 稀疏重建 (SfM) ──
    print('\n[3/5] 稀疏重建 (SfM)...')
    cmd = [
        'colmap', 'mapper',
        '--database_path', str(DATABASE_PATH),
        '--image_path', str(MASKED_DIR),
        '--output_path', str(SPARSE_DIR),
        '--Mapper.ba_refine_extra_params', '0',
        '--Mapper.ba_global_max_num_iterations', '50',
    ]
    subprocess.run(cmd, check=True)

    # 找最大的 bin (可能是0/)
    bins = sorted(SPARSE_DIR.glob('*/'))
    if not bins:
        print('ERROR: SfM 失败, 没有重建结果')
        return None
    # 找 images.bin 最大的
    best_bin = max(bins, key=lambda b: os.path.getsize(b / 'images.bin') if (b / 'images.bin').exists() else 0)
    print(f'SfM 完成 → {best_bin}')

    # ── 4. 畸变校正 + 稠密重建准备 ──
    print('\n[4/5] 图像去畸变...')
    undistort_dir = DENSE_DIR / 'images'
    cmd = [
        'colmap', 'image_undistorter',
        '--image_path', str(MASKED_DIR),
        '--input_path', str(best_bin),
        '--output_path', str(DENSE_DIR),
        '--output_type', 'images',
    ]
    subprocess.run(cmd, check=True)

    # ── 5. 稠密重建 (PatchMatch MVS) ──
    print('\n[5/5] 稠密重建 (MVS)...')
    cmd = [
        'colmap', 'patch_match_stereo',
        '--workspace_path', str(DENSE_DIR),
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.window_radius', '5',
        '--PatchMatchStereo.window_step', '2',
        '--PatchMatchStereo.num_iterations', '5',
        '--PatchMatchStereo.geom_consistency', '1',
        '--PatchMatchStereo.filter_min_num_consistent', '2',
    ]
    subprocess.run(cmd, check=True)

    # ── 6. 融合 ──
    print('\n[5/5续] 深度融合...')
    cmd = [
        'colmap', 'stereo_fusion',
        '--workspace_path', str(DENSE_DIR),
        '--workspace_format', 'COLMAP',
        '--input_type', 'photometric',
        '--output_path', str(DENSE_DIR / 'fused.ply'),
    ]
    subprocess.run(cmd, check=True)

    # 找 fused.ply
    fused_ply = DENSE_DIR / 'fused.ply'
    if fused_ply.exists():
        size_mb = os.path.getsize(fused_ply) / 1024 / 1024
        print(f'\n✓ 稠密点云: {fused_ply} ({size_mb:.1f} MB)')
        return fused_ply
    else:
        print('\n寻找融合结果...')
        for p in DENSE_DIR.glob('*.ply'):
            print(f'  {p}')
        return None


def main():
    os.makedirs(COLMAP_DIR, exist_ok=True)

    # 清理旧数据库
    for f in [DATABASE_PATH]:
        if os.path.exists(f):
            os.remove(f)

    print('=' * 55)
    print('  COLMAP RGB 重建管线 (转台场景)')
    print('=' * 55)

    # Step 1: 遮罩
    prepare_masked_images()

    # Step 2: COLMAP
    result = run_colmap()

    if result:
        print(f'\n完成! 点云: {result}')
    else:
        print('\nCOLMAP 重建失败, 可能是特征不足')
        print('尝试备选方案: DUSt3R 或增加转台纹理')


if __name__ == '__main__':
    main()
