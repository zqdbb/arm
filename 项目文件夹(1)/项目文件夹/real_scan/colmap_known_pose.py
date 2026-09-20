#!/usr/bin/env python3
"""
COLMAP MVS with known poses — 跳过 SfM, 直接稠密重建
用标定位姿 + 原始彩色图 → patch_match_stereo → 点云
"""
import numpy as np
import cv2
import json
import os
import subprocess
import shutil
import time
from pathlib import Path

# ── 路径 ──
BASE_DIR = Path(__file__).parent
SCAN_DIR = BASE_DIR / 'output/open3d_scan'
COLOR_DIR = SCAN_DIR / 'color'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = SCAN_DIR / 'camera_intrinsic.json'
COLMAP_DIR = BASE_DIR / 'output/colmap_known_pose'
SPARSE_DIR = COLMAP_DIR / 'sparse' / '0'
DENSE_DIR = COLMAP_DIR / 'dense'
DATABASE = COLMAP_DIR / 'database.db'

# ── 参数 ──
N_FRAMES = 36
ANGLE_STEP = 10  # 度


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


def rot_to_quat(R):
    """3x3 rotation matrix → quaternion [qw, qx, qy, qz]"""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz])


def compute_poses(R_calib, t_calib, flip_rotation=False):
    """
    计算 COLMAP 相机位姿 (world→camera).
    对象中心坐标系: 对象固定, 虚拟相机绕 Z 轴转.
    """
    R_w2c = R_calib.T
    t_w2c = -R_calib.T @ t_calib

    poses = []
    for i in range(N_FRAMES):
        theta = np.radians(i * ANGLE_STEP)
        if flip_rotation:
            theta = -theta

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z = np.array([[cos_t, -sin_t, 0],
                        [sin_t,  cos_t, 0],
                        [0,       0,     1]])

        R_v = R_w2c @ R_z
        t_v = t_w2c
        poses.append((R_v, t_v))
    return poses


def prepare_images():
    """复制原始彩色图到 COLMAP 图片目录"""
    img_dir = COLMAP_DIR / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    color_files = sorted(COLOR_DIR.glob('*.jpg'))
    for cf in color_files:
        shutil.copy(cf, img_dir / cf.name)
    print(f'复制 {len(color_files)} 张图片 → {img_dir}')
    return img_dir


def create_sparse_model(poses, fx, fy, ppx, ppy, w, h, flip_rotation=False):
    """创建 COLMAP 稀疏模型 (cameras.txt + images.txt + 空 points3D.txt)"""
    SPARSE_DIR.mkdir(parents=True, exist_ok=True)

    # cameras.txt — COLMAP 用空格分隔参数
    with open(SPARSE_DIR / 'cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {ppx:.6f} {ppy:.6f}\n')

    # images.txt
    with open(SPARSE_DIR / 'images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('# POINTS2D[] as (X, Y, POINT3D_ID)\n')

        for i, (R_v, t_v) in enumerate(poses):
            q = rot_to_quat(R_v)
            # COLMAP: qw, qx, qy, qz, tx, ty, tz
            f.write(f'{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} ')
            f.write(f'{t_v[0]:.10f} {t_v[1]:.10f} {t_v[2]:.10f} 1 {i:06d}.jpg\n')
            f.write('\n')  # 空点

    # 空 points3D.txt
    with open(SPARSE_DIR / 'points3D.txt', 'w') as f:
        f.write('# 3D point list\n')

    tag = 'flip' if flip_rotation else 'normal'
    print(f'创建稀疏模型 ({tag}): {SPARSE_DIR}')
    print(f'  相机: PINHOLE {w}x{h} {fx:.1f},{fy:.1f},{ppx:.1f},{ppy:.1f}')
    print(f'  图像: {N_FRAMES} 帧')
    for i in [0, N_FRAMES//4, N_FRAMES//2]:
        R_v, t_v = poses[i]
        center = -R_v.T @ t_v
        print(f'    帧{i:2d} ({i*ANGLE_STEP:3d}°): center=[{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}]')


def run_pipeline():
    """运行 COLMAP: 特征提取 + 匹配 + 三角化 + MVS"""
    img_dir = COLMAP_DIR / 'images'

    # ── 1. 特征提取 ──
    print('\n[1/5] 特征提取 (原始图)...')
    t0 = time.time()
    subprocess.run([
        'colmap', 'feature_extractor',
        '--database_path', str(DATABASE),
        '--image_path', str(img_dir),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'PINHOLE',
    ], check=True, capture_output=True)
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # ── 2. 特征匹配 ──
    print('\n[2/5] 特征匹配 (sequential, overlap=5)...')
    t0 = time.time()
    subprocess.run([
        'colmap', 'sequential_matcher',
        '--database_path', str(DATABASE),
        '--SequentialMatching.overlap', '5',
    ], check=True, capture_output=True)
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # ── 3. 三角化 (用已知位姿) ──
    print('\n[3/5] 三角化 (已知位姿)...')
    t0 = time.time()
    ret = subprocess.run([
        'colmap', 'point_triangulator',
        '--database_path', str(DATABASE),
        '--image_path', str(img_dir),
        '--input_path', str(SPARSE_DIR),
        '--output_path', str(SPARSE_DIR),
        '--Mapper.tri_min_angle', '1.0',
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f'  三角化失败:\n{ret.stderr[-500:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # 检查三角化结果
    points3d = SPARSE_DIR / 'points3D.bin'
    if points3d.exists():
        print(f'  points3D.bin: {os.path.getsize(points3d)/1024:.0f} KB')
    else:
        print('  警告: 没有三角化出 3D 点, MVS 可能效果不佳')

    # ── 4. 去畸变 ──
    print('\n[4/5] 图像去畸变...')
    if DENSE_DIR.exists():
        shutil.rmtree(DENSE_DIR)
    DENSE_DIR.mkdir(parents=True)

    t0 = time.time()
    subprocess.run([
        'colmap', 'image_undistorter',
        '--image_path', str(img_dir),
        '--input_path', str(SPARSE_DIR),
        '--output_path', str(DENSE_DIR),
        '--output_type', 'COLMAP',
    ], check=True, capture_output=True)
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # ── 5. 稠密 MVS ──
    print('\n[5/5] 稠密 MVS (PatchMatch)...')
    t0 = time.time()
    ret = subprocess.run([
        'colmap', 'patch_match_stereo',
        '--workspace_path', str(DENSE_DIR),
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.max_image_size', '1200',
        '--PatchMatchStereo.geom_consistency', '1',
        '--PatchMatchStereo.window_radius', '7',
        '--PatchMatchStereo.num_iterations', '5',
        '--PatchMatchStereo.num_samples', '15',
        '--PatchMatchStereo.filter_min_num_consistent', '2',
    ], capture_output=True, text=True)

    # patch_match_stereo can return non-zero for warnings
    if 'ERROR' in (ret.stderr or '') and ret.returncode != 0:
        print(f'  MVS 可能有问题:\n{ret.stderr[-500:]}')

    elapsed = time.time() - t0
    print(f'  完成 ({elapsed:.0f}s)')

    # ── 6. 融合 ──
    print('\n[融合] 深度融合...')
    fusion_ply = DENSE_DIR / 'fused.ply'
    subprocess.run([
        'colmap', 'stereo_fusion',
        '--workspace_path', str(DENSE_DIR),
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', str(fusion_ply),
        '--StereoFusion.min_num_pixels', '3',
    ], check=True, capture_output=True)

    if fusion_ply.exists():
        size_mb = os.path.getsize(fusion_ply) / (1024 * 1024)
        print(f'  点云: {fusion_ply} ({size_mb:.1f} MB)')
        return fusion_ply
    else:
        print('  点云文件未生成')
        return None


def main():
    for f in [DATABASE]:
        if f.exists():
            f.unlink()

    COLMAP_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('  COLMAP + 已知位姿 MVS 重建')
    print('=' * 55)

    R_calib, t_calib, (fx, fy, ppx, ppy), (w, h) = load_data()
    print(f'内参: fx={fx:.1f} fy={fy:.1f} 尺寸={w}×{h}')
    print(f'标定: t_calib={t_calib.round(4)}')

    # Step 1: 复制图片
    prepare_images()

    # Step 2: 计算位姿
    poses = compute_poses(R_calib, t_calib)

    # Step 3: 创建稀疏模型
    create_sparse_model(poses, fx, fy, ppx, ppy, w, h)

    # Step 4: 运行 COLMAP
    result = run_pipeline()

    if result:
        print(f'\n{"=" * 55}')
        print(f'  完成! {result}')
        print(f'{"=" * 55}')
    else:
        print('\n重建失败')


if __name__ == '__main__':
    main()
