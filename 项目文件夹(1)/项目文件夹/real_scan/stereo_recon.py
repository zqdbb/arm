#!/usr/bin/env python3
"""
多视角 SGBM 立体匹配 + 深度融合 → 替代 D435i 深度.

输入: 密集采集的 RGB 图 (5° 步长) + calibrate.json 标定位姿
输出: 三角网格 mesh (PLY + OBJ)

流程:
  1. 每帧配对其最佳相邻帧 (skip=2 → 10° 基线)
  2. 极线校正 → SGBM → 深度图
  3. 深度图 → 3D 点云 (用已知位姿投影)
  4. 累积点云 → TSDF 融合 → Marching Cubes mesh
"""
import numpy as np
import cv2, json, os, glob, time
from pathlib import Path
import open3d as o3d

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'output/dense_scan'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = DATA_DIR / 'camera_intrinsic.json'

# SGBM 参数
SGBM_BLOCK = 5
SGBM_NUM_DISP = 256      # 视差搜索范围 (5°基线 = 65像素@44cm, 近处更大)
SGBM_MIN_DISP = 0
SGBM_UNIQUENESS = 8      # 唯一性比率 (降低, 接受更多匹配)
SGBM_SPECKLE = 100       # 散斑过滤窗口

PAIR_SKIP = 1             # 隔几帧配对 (1=相邻5°, 2=隔帧10°)

# TSDF 参数
VOXEL_SIZE = 0.003        # 3mm 体素
SDF_TRUNC = 0.012          # 截断距离

OUTPUT_DIR = BASE_DIR / 'output/stereo_recon'
SCENE_DIR = OUTPUT_DIR / 'scene'


def load_data():
    calib = json.load(open(CALIB_PATH))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])

    intr_path = INTRINSIC_PATH if INTRINSIC_PATH.exists() else \
                BASE_DIR / 'output/open3d_scan/camera_intrinsic.json'
    if not intr_path.exists():
        raise FileNotFoundError(f'内参文件不存在: {intr_path}')
    intr = json.load(open(intr_path))
    M = intr['intrinsic_matrix']
    fx, fy = M[0], M[4]
    ppx, ppy = M[6], M[7]
    w, h = intr['width'], intr['height']
    return R_calib, t_calib, (fx, fy, ppx, ppy), (w, h)


def compute_poses(R_calib, t_calib, n_frames, step_angle=5):
    """计算每帧相机外参 (aligned frame → camera)."""
    poses = []
    for i in range(n_frames):
        theta = np.radians(i * step_angle)
        c, s = np.cos(theta), np.sin(theta)
        R_z_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_w2c = R_calib.T @ R_z_inv
        t_w2c = -R_calib.T @ t_calib
        cam = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ t_calib  # R_z(+θ)
        poses.append((R_w2c, t_w2c, cam))
    return poses


def stereo_match(img0, img1, R0, cam0, R1, cam1, K, w, h):
    """对一对图像做极线校正 + SGBM 立体匹配, 返回深度图.
    R0, cam0: 参考帧的相机旋转矩阵(W2C)和相机中心(world坐标)
    R1, cam1: 配对帧的相机旋转矩阵(W2C)和相机中心(world坐标)
    """
    # 相对位姿: camera0 → camera1
    R_rel = R1 @ R0.T
    t_rel = R1 @ (cam0 - cam1)

    # 极线校正
    R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, np.zeros(5), K, np.zeros(5), (w, h), R_rel, t_rel, alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(
        K, np.zeros(5), R1_rect, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K, np.zeros(5), R2_rect, P2, (w, h), cv2.CV_32FC1)

    rect0 = cv2.remap(img0, map1x, map1y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, map2x, map2y, cv2.INTER_LINEAR)

    gray0 = cv2.cvtColor(rect0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=SGBM_MIN_DISP,
        numDisparities=SGBM_NUM_DISP,
        blockSize=SGBM_BLOCK,
        P1=8 * 3 * SGBM_BLOCK**2,
        P2=32 * 3 * SGBM_BLOCK**2,
        disp12MaxDiff=1,
        uniquenessRatio=SGBM_UNIQUENESS,
        speckleWindowSize=SGBM_SPECKLE,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(gray0, gray1).astype(np.float32) / 16.0

    # 视差 → 深度 (校正后焦距)
    fx_rect = P1[0, 0]
    baseline = np.linalg.norm(t_rel)
    depth = np.full_like(disparity, np.nan)
    valid = disparity > 0.5
    depth[valid] = fx_rect * baseline / disparity[valid]

    return depth, R1_rect, P1


def depth_to_points(depth, R_w2c_ref, t_w2c_ref, R_rect, P, fx, fy, ppx, ppy, w, h):
    """深度图 → 3D 点云 (world aligned frame 坐标)."""
    # 校正后相机内参
    fx_r = P[0, 0]
    fy_r = P[1, 1]
    ppx_r = P[0, 2]
    ppy_r = P[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = ~np.isnan(depth)
    if valid.sum() < 1000:
        return None

    Z = depth[valid]
    X = (u[valid] - ppx_r) / fx_r * Z
    Y = (v[valid] - ppy_r) / fy_r * Z

    pts_rect = np.column_stack([X, Y, Z])

    # 校正坐标系 → 原始相机坐标系
    # p_cam = R_rect^T @ p_rect
    pts_cam = (R_rect.T @ pts_rect.T).T

    # 相机坐标系 → world aligned frame
    # p_world = R_w2c_ref^T @ (p_cam - t_w2c_ref)
    R_c2w = R_w2c_ref.T
    pts_world = (R_c2w @ pts_cam.T).T + R_c2w @ (-t_w2c_ref)

    return pts_world, valid


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SCENE_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('  多视角 SGBM 立体匹配 + 深度融合')
    print('=' * 55)

    R_calib, t_calib, (fx, fy, ppx, ppy), (w, h) = load_data()
    K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
    print(f'内参: fx={fx:.1f} fy={fy:.1f} 尺寸={w}×{h}')

    # 加载图像
    color_dir = DATA_DIR / 'color'
    if not color_dir.exists():
        color_dir = BASE_DIR / 'output/open3d_scan/color'
    color_files = sorted(glob.glob(str(color_dir / '*.jpg')))
    n_frames = len(color_files)

    # 自动检测步长
    try:
        intr = json.load(open(str(INTRINSIC_PATH) if INTRINSIC_PATH.exists()
                           else str(BASE_DIR / 'output/open3d_scan/camera_intrinsic.json')))
        step_angle = intr.get('step_angle', 10)
    except:
        step_angle = 10

    print(f'图像: {n_frames} 帧, 步长 {step_angle}°')

    if n_frames < 3:
        print('帧数不足! 先运行 capture_dense.py 采集')
        return

    # 预加载图像
    print('预加载图像...')
    images = [cv2.imread(f) for f in color_files]

    # 计算位姿
    print('计算相机位姿...')
    poses = compute_poses(R_calib, t_calib, n_frames, step_angle)

    # ── 逐帧 SGBM ──
    print(f'\n{"─"*50}')
    print(f'SGBM 立体匹配: {n_frames} 帧, 配对间隔 {PAIR_SKIP} 帧 ({PAIR_SKIP*step_angle}° 基线)')
    all_pts = []
    all_colors = []

    t0 = time.time()
    n_good = 0

    for i in range(n_frames):
        j = (i + PAIR_SKIP) % n_frames

        img0 = images[i]
        img1 = images[j]
        R_ref, t_ref, cam_ref = poses[i]
        R_pair, t_pair, cam_pair = poses[j]

        baseline = np.linalg.norm(cam_pair - cam_ref)

        # 相对位姿
        R_rel = R_pair @ R_ref.T
        t_rel = R_pair @ (cam_ref - cam_pair)

        # 跳过基线太小的配对
        if baseline < 0.02:
            continue

        # SGBM
        depth, R_rect, P = stereo_match(img0, img1,
                                         R_ref, cam_ref, R_pair, cam_pair,
                                         K, w, h)

        # 深度图 → 3D 点
        result = depth_to_points(depth, R_ref, t_ref, R_rect, P,
                                 fx, fy, ppx, ppy, w, h)
        if result is None:
            continue

        pts_world, valid_mask = result
        n_good += 1

        # 颜色
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        rgb = img0[v[valid_mask], u[valid_mask]][:, ::-1] / 255.0

        # 圆柱裁切 (只保留转台中心区域)
        dist_xy = np.sqrt(pts_world[:, 0]**2 + pts_world[:, 1]**2)
        keep = dist_xy <= 0.12
        pts_world = pts_world[keep]
        rgb = rgb[keep]

        if len(pts_world) < 500:
            continue

        # 随机降采样 (控制总点数)
        if len(pts_world) > 15000:
            idx = np.random.choice(len(pts_world), 15000, replace=False)
            pts_world = pts_world[idx]
            rgb = rgb[idx]

        all_pts.append(pts_world.astype(np.float32))
        all_colors.append(rgb.astype(np.float32))

        if (i + 1) % 12 == 0 or i == 0:
            elapsed = time.time() - t0
            n_pts = sum(len(p) for p in all_pts)
            print(f'  [{i+1}/{n_frames}] {i*step_angle:3d}° '
                  f'基线={baseline*100:.1f}cm depth={np.nanmedian(depth):.3f}m '
                  f'已用时{elapsed:.0f}s  累积{n_pts:,}点')

    dt = time.time() - t0
    print(f'  有效帧: {n_good}/{n_frames}  耗时: {dt:.0f}s')

    if len(all_pts) < 3:
        print('有效深度帧数不足!')
        return

    pts_all = np.vstack(all_pts)
    colors_all = np.vstack(all_colors)
    print(f'  累积点数: {len(pts_all):,}')

    # ── 保存累积点云 ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)
    o3d.io.write_point_cloud(str(SCENE_DIR / 'accumulated_sgbm.ply'), pcd)

    # ── TSDF 融合 ──
    print(f'\n{"─"*50}')
    print('TSDF 融合...')
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    # 从点云构建 TSDF (Open3D 不直接支持, 但可以用 ScalableTSDFVolume)
    # 这里用另一种方式: 先做 Alpha Shapes 再简化为 mesh

    print(f'  降采样后: {len(pcd.points):,} 点')
    pts = np.asarray(pcd.points)

    # Z 翻转 (world Z↓ → 视觉 Z↑)
    pts[:, 2] = -pts[:, 2]
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 裁切
    dist_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    keep = (dist_xy <= 0.08) & (pts[:, 2] >= 0.002)
    pts = pts[keep]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    print(f'  裁切后: {len(pts):,} 点')
    print(f'  XYZ: X[{pts[:,0].min():.3f},{pts[:,0].max():.3f}] '
          f'Y[{pts[:,1].min():.3f},{pts[:,1].max():.3f}] '
          f'Z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]')

    # 尝试 Alpha Shapes (适合带棱角的物体)
    print(f'\nAlpha Shapes 表面重建...')
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=0.008)
        print(f'  Alpha=0.008: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面')
    except Exception as e:
        print(f'  Alpha Shapes 失败: {e}')
        # 退回 Poisson
        print('  退回 Poisson 表面重建...')
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0., 0., 1.]))
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, n_threads=4)

    out_ply = str(SCENE_DIR / 'stereo_mesh.ply')
    out_obj = str(SCENE_DIR / 'stereo_mesh.obj')
    o3d.io.write_triangle_mesh(out_ply, mesh)
    o3d.io.write_triangle_mesh(out_obj, mesh)

    verts = np.asarray(mesh.vertices)
    print(f'\n{"="*55}')
    print(f'  输出: {out_ply}')
    print(f'  顶点: {len(verts):,}  面: {len(mesh.triangles):,}')
    if len(verts) > 0:
        print(f'  XYZ: X[{verts[:,0].min():.3f},{verts[:,0].max():.3f}] '
              f'Y[{verts[:,1].min():.3f},{verts[:,1].max():.3f}] '
              f'Z[{verts[:,2].min():.3f},{verts[:,2].max():.3f}]')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
