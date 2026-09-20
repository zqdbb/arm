#!/usr/bin/env python3
"""
COLMAP MVS 稠密重建 — 5° 步长 72 帧 + 已知标定位姿.
"""
import numpy as np, cv2, json, os, subprocess, shutil, time
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / 'output/dense_scan'
CALIB = BASE / 'output/calibrate.json'
INTR = DATA / 'camera_intrinsic.json'

OUT = BASE / 'output/colmap_dense'
SPARSE = OUT / 'sparse' / '0'
DENSE = OUT / 'dense'
DB = OUT / 'database.db'
IMG = OUT / 'images'

STEP = 5
N_FRAMES = 72


def rot_to_quat(R):
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


def main():
    # 加载数据
    calib = json.load(open(CALIB))
    R_calib = np.array(calib['R']); t_calib = np.array(calib['t'])
    intr = json.load(open(INTR))
    M = intr['intrinsic_matrix']
    fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]
    w, h = intr['width'], intr['height']

    print('=' * 55)
    print('  COLMAP MVS: 5° × 72 帧 + 已知位姿')
    print(f'  内参: {fx:.1f} {fy:.1f}  {w}×{h}')
    print('=' * 55)

    # 清理旧数据
    for f in [DB]:
        if f.exists(): f.unlink()
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # 复制图片
    IMG.mkdir(parents=True)
    color_files = sorted((DATA / 'color').glob('*.jpg'))
    for i, cf in enumerate(color_files):
        shutil.copy(cf, IMG / f'{i:06d}.jpg')
    print(f'复制 {len(color_files)} 张图片')

    # 位姿: aligned frame → camera
    # p_cam = R_calib^T @ R_z(-θ) @ p_aligned - R_calib^T @ t_calib
    SPARSE.mkdir(parents=True)

    with open(SPARSE / 'cameras.txt', 'w') as f:
        f.write('# Camera list\n')
        f.write(f'1 PINHOLE {w} {h} {fx:.6f} {fy:.6f} {ppx:.6f} {ppy:.6f}\n')

    with open(SPARSE / 'images.txt', 'w') as f:
        f.write('# Image list\n')
        for i in range(N_FRAMES):
            theta = np.radians(i * STEP)
            c, s = np.cos(theta), np.sin(theta)
            R_z_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            R_w2c = R_calib.T @ R_z_inv
            q = rot_to_quat(R_w2c)
            t_w2c = -R_calib.T @ t_calib
            f.write(f'{i+1} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} ')
            f.write(f'{t_w2c[0]:.10f} {t_w2c[1]:.10f} {t_w2c[2]:.10f} 1 {i:06d}.jpg\n')
            f.write('\n')

    with open(SPARSE / 'points3D.txt', 'w') as f:
        f.write('# Empty\n')

    print(f'稀疏模型: {N_FRAMES} 帧, 位姿已写入')

    # ── COLMAP Pipeline ──

    # 1. 特征提取
    print('\n[1/4] 特征提取...')
    t0 = time.time()
    subprocess.run(['colmap', 'feature_extractor', '--database_path', str(DB),
                    '--image_path', str(IMG), '--ImageReader.single_camera', '1',
                    '--ImageReader.camera_model', 'PINHOLE'],
                   check=True, capture_output=True)
    print(f'  {time.time()-t0:.0f}s')

    # 2. 特征匹配
    print('\n[2/4] 特征匹配...')
    t0 = time.time()
    subprocess.run(['colmap', 'sequential_matcher', '--database_path', str(DB),
                    '--SequentialMatching.overlap', '6'],
                   check=True, capture_output=True)
    print(f'  {time.time()-t0:.0f}s')

    # 3. 三角化 (已知位姿)
    print('\n[3/4] 三角化...')
    t0 = time.time()
    ret = subprocess.run(['colmap', 'point_triangulator',
                          '--database_path', str(DB), '--image_path', str(IMG),
                          '--input_path', str(SPARSE), '--output_path', str(SPARSE),
                          '--Mapper.tri_min_angle', '1.0'],
                         capture_output=True, text=True)
    if ret.returncode != 0:
        print(f'  失败: {ret.stderr[-300:]}')
        return
    print(f'  {time.time()-t0:.0f}s')

    # 4. MVS (去畸变 + 稠密)
    print('\n[4/4] 稠密 MVS...')
    DENSE.mkdir(parents=True)

    subprocess.run(['colmap', 'image_undistorter', '--image_path', str(IMG),
                    '--input_path', str(SPARSE), '--output_path', str(DENSE),
                    '--output_type', 'COLMAP'],
                   check=True, capture_output=True)

    t0 = time.time()
    ret = subprocess.run(['colmap', 'patch_match_stereo',
                          '--workspace_path', str(DENSE),
                          '--workspace_format', 'COLMAP',
                          '--PatchMatchStereo.max_image_size', '1200',
                          '--PatchMatchStereo.geom_consistency', '1',
                          '--PatchMatchStereo.window_radius', '5',
                          '--PatchMatchStereo.num_iterations', '5',
                          '--PatchMatchStereo.num_samples', '15',
                          '--PatchMatchStereo.filter_min_num_consistent', '2'],
                         capture_output=True, text=True)
    print(f'  PatchMatch: {time.time()-t0:.0f}s')

    # 融合
    print('\n[融合] 深度融合...')
    fusion = DENSE / 'fused.ply'
    subprocess.run(['colmap', 'stereo_fusion', '--workspace_path', str(DENSE),
                    '--workspace_format', 'COLMAP', '--input_type', 'geometric',
                    '--output_path', str(fusion),
                    '--StereoFusion.min_num_pixels', '3'],
                   check=True, capture_output=True)

    if fusion.exists():
        size_mb = os.path.getsize(fusion) / 1e6
        # 统计点数
        pts = np.fromfile(fusion, dtype=np.float32, offset=0)
        print(f'  输出: {fusion} ({size_mb:.1f} MB)')
    else:
        print('  融合失败!')

    print(f'{"="*55}')


if __name__ == '__main__':
    main()
