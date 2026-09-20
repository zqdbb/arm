#!/usr/bin/env python3
"""
V5 COLMAP多视图重建: 背景减除 + COLMAP SfM/MVS → 网格
72帧RGB → 中值背景 → 前景剪影 → COLMAP密集匹配 → 网格

使用: python3 scan_recon_v5.py
需要: sudo apt install colmap
"""

import cv2, json, time, os, subprocess, shutil
import numpy as np
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ.pop('DISPLAY', None)

BASE = Path(__file__).parent
OUT = BASE / 'output/v5'
COLOR_SRC = BASE / 'output/v4/color'
W, H = 640, 480

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'images').mkdir(exist_ok=True)
(OUT / 'masks').mkdir(exist_ok=True)


def check_colmap():
    """检查 COLMAP 是否安装"""
    try:
        r = subprocess.run(['colmap', '--help'], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            print('COLMAP 已安装')
            return True
    except FileNotFoundError:
        pass
    print('错误: COLMAP 未安装')
    print('  sudo apt install colmap')
    return False


def check_cuda():
    """检查 COLMAP 是否有 CUDA 支持"""
    try:
        r = subprocess.run(['colmap', 'feature_extractor', '--help'],
                           capture_output=True, text=True, timeout=10)
        if 'cuda' in r.stdout.lower() or 'gpu' in r.stdout.lower():
            return True
    except:
        pass
    return False


def generate_masked_images():
    """
    用中值背景减除生成椅子剪影和遮罩图片。
    在所有72帧上取中值 → 自然得到不含椅子的背景。
    """
    color_files = sorted(COLOR_SRC.glob('*.jpg'))
    if len(color_files) < 36:
        print(f'错误: 图片太少 ({len(color_files)} 张), 需要 v4 采集的72帧')
        return False
    print(f'输入图片: {len(color_files)} 张 ({COLOR_SRC})')

    # 计算中值背景 (椅子每帧位置不同，中值=背景)
    print('计算中值背景...')
    t0 = time.time()
    samples = []
    for i in range(0, len(color_files), 2):  # 隔帧采样加快速度
        img = cv2.imread(str(color_files[i]))
        if img is not None and img.shape == (H, W, 3):
            samples.append(img.astype(np.float32))
    bg = np.median(np.stack(samples), axis=0).astype(np.uint8)
    cv2.imwrite(str(OUT / 'background.png'), bg)
    print(f'  背景计算完成 ({time.time()-t0:.1f}s) → {OUT / "background.png"}')

    # 逐帧减背景生成遮罩
    print(f'\n生成遮罩 ({len(color_files)} 帧)...')
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    t0 = time.time()
    n_valid = 0

    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None:
            continue

        # 颜色差
        diff = cv2.absdiff(img, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = (binary > 0).sum() / binary.size
        if fg_ratio > 0.5:
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # 形态学清理
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # 最大连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest = np.argmax(areas) + 1
                binary = (labels == largest).astype(np.uint8) * 255

        if binary.sum() < 200:
            continue

        # 保存遮罩
        mask_name = f'{cf.stem}.png'
        cv2.imwrite(str(OUT / 'masks' / mask_name), binary)

        # 遮罩后图片 (背景填中灰色，减少 SIFT 边界效应)
        masked = img.copy()
        bg_gray = np.full_like(masked, 128)
        masked[binary == 0] = bg_gray[binary == 0]
        cv2.imwrite(str(OUT / 'images' / f'{cf.stem}.jpg'), masked)

        n_valid += 1

        if (i + 1) % 18 == 0:
            print(f'  [{i+1}/{len(color_files)}] 有效={n_valid}')

    print(f'遮罩完成 ({time.time()-t0:.1f}s): {n_valid} 帧有效')
    return n_valid >= 30


def run_colmap():
    """运行 COLMAP 完整管线"""
    images_dir = str(OUT / 'images')
    masks_dir = str(OUT / 'masks')
    db_path = str(OUT / 'colmap.db')
    sparse_dir = str(OUT / 'sparse')
    dense_dir = str(OUT / 'dense')

    # 清理旧数据库
    for p in [db_path, sparse_dir, dense_dir]:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.unlink(p)
    os.makedirs(sparse_dir, exist_ok=True)

    has_cuda = check_cuda()
    gpu_str = 'CUDA' if has_cuda else 'CPU'
    print(f'\nGPU: {gpu_str}')

    # Step 1: 特征提取
    print('\n--- COLMAP 特征提取 ---')
    t0 = time.time()
    cmd = [
        'colmap', 'feature_extractor',
        '--database_path', db_path,
        '--image_path', images_dir,
        '--ImageReader.mask_path', masks_dir,
        '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.max_num_features', '8192',
        '--SiftExtraction.estimate_affine_shape', '0',
        '--SiftExtraction.use_gpu', '0',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
    if r.returncode != 0:
        print(f'STDERR: {r.stderr[-500:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # Step 2: 序列匹配 (转台相邻帧高重叠)
    print('\n--- COLMAP 序列匹配 ---')
    t0 = time.time()
    cmd = [
        'colmap', 'sequential_matcher',
        '--database_path', db_path,
        '--SequentialMatching.overlap', '5',
        '--SequentialMatching.loop_detection', '0',
        '--SiftMatching.use_gpu', '0',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
    if r.returncode != 0:
        print(f'STDERR: {r.stderr[-500:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # Step 3: 稀疏重建
    print('\n--- COLMAP 稀疏重建 ---')
    t0 = time.time()
    r = subprocess.run([
        'colmap', 'mapper',
        '--database_path', db_path,
        '--image_path', images_dir,
        '--output_path', sparse_dir,
    ], capture_output=True, text=True, timeout=1200)
    print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)
    if r.returncode != 0:
        print(f'STDERR: {r.stderr[-500:]}')

    # 找最大的稀疏模型
    sparse_models = sorted(Path(sparse_dir).glob('*/cameras.bin'))
    if not sparse_models:
        # 尝试 images.bin 或 points3D.bin
        sparse_models = sorted(Path(sparse_dir).glob('*/*.bin'))
        sparse_dirs = set(p.parent for p in sparse_models)
        if not sparse_dirs:
            print('稀疏重建失败: 没有生成模型')
            return False
        sparse_model_dir = str(sorted(sparse_dirs)[-1])
    else:
        sparse_model_dir = str(sparse_models[0].parent)
    print(f'  稀疏模型: {sparse_model_dir}')
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # Step 4: 密集重建
    print('\n--- COLMAP 密集重建 ---')

    # 4a: 图像去畸变
    print('  undistortion...')
    t0 = time.time()
    r = subprocess.run([
        'colmap', 'image_undistorter',
        '--image_path', images_dir,
        '--input_path', sparse_model_dir,
        '--output_path', str(dense_dir),
        '--output_type', 'COLMAP',
    ], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f'  去畸变失败: {r.stderr[-300:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # 4b: 立体匹配
    print('  patch_match_stereo...')
    t0 = time.time()
    cmd = [
        'colmap', 'patch_match_stereo',
        '--workspace_path', str(dense_dir),
        '--PatchMatchStereo.window_radius', '4',
        '--PatchMatchStereo.num_samples', '7',
        '--PatchMatchStereo.num_iterations', '3',
        '--PatchMatchStereo.geom_consistency', '1',
    ]
    if has_cuda:
        cmd += ['--PatchMatchStereo.gpu_index', '0']
    else:
        cmd += ['--PatchMatchStereo.max_image_size', '1000']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)
    if r.returncode != 0:
        print(f'  立体匹配失败: {r.stderr[-300:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # 4c: 深度融合
    print('  stereo_fusion...')
    t0 = time.time()
    r = subprocess.run([
        'colmap', 'stereo_fusion',
        '--workspace_path', str(dense_dir),
        '--output_path', str(OUT / 'dense_fused.ply'),
    ], capture_output=True, text=True, timeout=1800)
    print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)
    if r.returncode != 0:
        print(f'  深度融合失败: {r.stderr[-300:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # 4d: Poisson 网格
    print('  poisson_mesher...')
    t0 = time.time()
    r = subprocess.run([
        'colmap', 'poisson_mesher',
        '--input_path', str(OUT / 'dense_fused.ply'),
        '--output_path', str(OUT / 'dense_mesh.ply'),
    ], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f'  网格生成失败: {r.stderr[-300:]}')
        return False
    print(f'  完成 ({time.time()-t0:.0f}s)')

    return True


def post_process():
    """方向修正 + 输出"""
    import open3d as o3d

    mesh_path = OUT / 'dense_mesh.ply'
    if not mesh_path.exists():
        print(f'错误: 网格不存在 {mesh_path}')
        return

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        print('错误: 空网格')
        return

    print(f'\n原始网格: {len(verts):,} 顶点, {len(np.asarray(mesh.triangles)):,} 面')

    # COLMAP 坐标系任意，需要检测方向。
    # 假设椅子占最大尺寸的轴是高度(Y=up)
    # 检测主力方向
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, -1]  # 最大方差方向

    # 找到主方向后，将 verts 旋转使主方向对齐 Y 轴
    target = np.array([0., 1., 0.])
    axis = np.cross(main_axis, target)
    if np.linalg.norm(axis) > 1e-9:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(main_axis, target), -1, 1))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    else:
        R_align = np.eye(3) if np.dot(main_axis, target) > 0 else np.diag([1, -1, -1])

    verts_aligned = (R_align @ verts.T).T
    verts_aligned[:, 1] -= verts_aligned[:, 1].min()  # 底部 Y=0

    mesh_aligned = o3d.geometry.TriangleMesh()
    mesh_aligned.vertices = o3d.utility.Vector3dVector(verts_aligned)
    mesh_aligned.triangles = mesh.triangles

    # 只保留最大连通分量
    labels, counts, _ = mesh_aligned.cluster_connected_triangles()
    if len(counts) > 1:
        largest = np.argmax(counts)
        mesh_aligned.remove_triangles_by_index(np.where(labels != largest)[0])
    mesh_aligned.remove_unreferenced_vertices()
    mesh_aligned.compute_vertex_normals()

    # 保存
    verts_out = np.asarray(mesh_aligned.vertices)
    o3d.io.write_triangle_mesh(str(OUT / 'chair.obj'), mesh_aligned)
    o3d.io.write_triangle_mesh(str(OUT / 'chair.ply'), mesh_aligned)

    print(f'\n{"=" * 50}')
    print(f'  输出: {OUT}/chair.obj')
    print(f'  方向: Y=↑')
    for i, a in enumerate('XYZ'):
        print(f'  {a}: [{verts_out[:,i].min():.3f}, {verts_out[:,i].max():.3f}] '
              f'span={np.ptp(verts_out[:,i])*100:.1f}cm')
    print(f'  预期: X≈4.6cm Y≈8.8cm(高) Z≈4.0cm')
    print(f'  顶点: {len(verts_out):,}  面: {len(np.asarray(mesh_aligned.triangles)):,}')
    print(f'{"=" * 50}')


# ═══════════════════ Main ═══════════════════
if __name__ == '__main__':
    if not check_colmap():
        exit(1)

    print('=' * 50)
    print('  V5 COLMAP 多视图重建')
    print('=' * 50)

    # Step 1: 生成遮罩图片
    if not generate_masked_images():
        print('遮罩生成失败')
        exit(1)

    # Step 2: COLMAP 管线
    if not run_colmap():
        print('\nCOLMAP 重建失败')
        print('可能原因:')
        print('  1. 椅子纹理太弱 → 尝试用更小 --SiftExtraction.max_num_features')
        print('  2. GPU 内存不足 → CPU模式虽然慢但应该能跑')
        print('  3. 遮罩质量差 → 检查 output/v5/masks/')
        exit(1)

    # Step 3: 后处理
    post_process()
