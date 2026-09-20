#!/usr/bin/env python3
"""
深度融合: 已知角度反旋 + 几何过滤 + Poisson 网格重建.
不需要 rembg, 靠转台平面分离前景背景.
"""
import numpy as np
import cv2, json, time
from pathlib import Path
import open3d as o3d

BASE = Path(__file__).parent
DATA = BASE / 'output/dense_scan'
CALIB = BASE / 'output/calibrate.json'
INTR = DATA / 'camera_intrinsic.json'
OUT = BASE / 'output/depth_fusion'
OUT.mkdir(parents=True, exist_ok=True)

VOXEL_SIZE = 0.0015  # 1.5mm downsampling
MAX_CAM_DEPTH = 0.5   # D405 可靠深度上限 50cm
Z_CHAIR_MIN = -0.10   # 椅子底部 (10cm above turntable, 含余量)
Z_CHAIR_MAX = -0.001  # 椅子顶部 (1mm above turntable)
R_CHAIR_MAX = 0.15    # 转台半径 15cm 内的点


def load_data():
    calib = json.load(open(CALIB))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])
    intr = json.load(open(INTR))
    M = intr['intrinsic_matrix']
    fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]
    w, h = intr['width'], intr['height']
    step = intr.get('step_angle', 5)
    n_frames = intr.get('n_frames', 72)
    return R_calib, t_calib, (fx, fy, ppx, ppy), (w, h), step, n_frames


def main():
    R_calib, t_calib, K, (w, h), step, n_frames = load_data()
    fx, fy, ppx, ppy = K
    cam_pos = -R_calib.T @ t_calib
    print(f'相机 world 位置: [{cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f}]')

    color_files = sorted((DATA / 'color').glob('*.jpg'))
    depth_files = sorted((DATA / 'depth').glob('*.png'))
    n_frames = min(n_frames, len(color_files), len(depth_files))

    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))

    all_pts = []
    all_colors = []

    print(f'\n深度融合: {n_frames} 帧 (无反旋)')
    t0 = time.time()

    for i in range(n_frames):
        img = cv2.imread(str(color_files[i]))
        depth = cv2.imread(str(depth_files[i]), -1).astype(np.float32) / 1000.0

        valid_depth = (depth > 0.01) & (depth < MAX_CAM_DEPTH)
        if valid_depth.sum() < 100:
            continue

        z_c = depth[valid_depth]
        x_c = (u_grid[valid_depth] - ppx) / fx * z_c
        y_c = (v_grid[valid_depth] - ppy) / fy * z_c
        pts_cam = np.column_stack([x_c, y_c, z_c])

        # 相机 → world
        pts_world = (R_calib @ pts_cam.T).T + t_calib

        # 反旋: world → object
        theta = np.radians(i * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
        pts_obj = (R_z_inv @ pts_world.T).T

        all_pts.append(pts_obj)
        colors_bgr = img[valid_depth].astype(np.float32) / 255.0
        all_colors.append(colors_bgr[:, ::-1])

        if (i + 1) % 18 == 0:
            pts_total = sum(len(p) for p in all_pts)
            print(f'  [{i+1}/{n_frames}] {pts_total/1e6:.2f}M pts  {time.time()-t0:.0f}s')

    pts_all = np.concatenate(all_pts, axis=0)
    colors_all = np.concatenate(all_colors, axis=0)
    print(f'\n原始点云: {len(pts_all)/1e6:.2f}M 点')

    # 几何过滤: 只保留转台面以上的点 (椅子)
    r = np.hypot(pts_all[:, 0], pts_all[:, 1])
    print(f'  Z: [{pts_all[:,2].min():.3f}, {pts_all[:,2].max():.3f}]')
    print(f'  r: [{r.min():.3f}, {r.max():.3f}]')

    in_z = (pts_all[:, 2] >= Z_CHAIR_MIN) & (pts_all[:, 2] <= Z_CHAIR_MAX)
    in_r = r < R_CHAIR_MAX
    keep = in_z & in_r
    pts_filtered = pts_all[keep]
    colors_filtered = colors_all[keep]
    print(f'过滤后 (Z∈[{Z_CHAIR_MIN},{Z_CHAIR_MAX}], r<{R_CHAIR_MAX}m): {len(pts_filtered)/1e6:.2f}M 点')

    if len(pts_filtered) < 500:
        print('点数太少! 退出')
        return

    # 统计
    pts = pts_filtered
    print(f'  X: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]')
    print(f'  Y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]')
    print(f'  Z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]')

    # 体素降采样
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_filtered.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    print(f'降采样后: {len(pcd.points):,} 点 @ {VOXEL_SIZE*1000:.1f}mm')

    # 统计离群点去除
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Z 翻转 (world Z↓ → visual Z↑)
    pts_pcd = np.asarray(pcd.points)
    pts_pcd[:, 2] = -pts_pcd[:, 2]
    pcd.points = o3d.utility.Vector3dVector(pts_pcd)

    # 保存点云
    o3d.io.write_point_cloud(str(OUT / 'fused.ply'), pcd)
    print(f'点云已保存: {OUT}/fused.ply')

    # Poisson 重建
    print('\nPoisson 表面重建...')
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE*5, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., -2.]))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8, width=0, scale=1.1, linear_fit=False)

    # 去低密度面
    density_thresh = np.quantile(densities, 0.1)
    to_remove = densities < density_thresh
    mesh.remove_vertices_by_mask(to_remove)
    mesh.remove_unreferenced_vertices()

    # Z 翻转后, chair 在 Z>0; 去掉 Z<0 的碎面
    verts = np.asarray(mesh.vertices)
    keep = verts[:, 2] >= 0.001
    if keep.sum() > 0:
        indices = np.where(keep)[0]
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(verts[indices])
        old_to_new = np.full(len(verts), -1, dtype=int)
        old_to_new[indices] = np.arange(len(indices))
        faces = np.asarray(mesh.triangles)
        face_mask = (old_to_new[faces[:, 0]] >= 0) & \
                    (old_to_new[faces[:, 1]] >= 0) & \
                    (old_to_new[faces[:, 2]] >= 0)
        new_faces = np.column_stack([
            old_to_new[faces[face_mask, 0]],
            old_to_new[faces[face_mask, 1]],
            old_to_new[faces[face_mask, 2]],
        ])
        new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
        new_mesh.compute_vertex_normals()
        mesh = new_mesh

    o3d.io.write_triangle_mesh(str(OUT / 'poisson.ply'), mesh)
    o3d.io.write_triangle_mesh(str(OUT / 'poisson.obj'), mesh)

    verts = np.asarray(mesh.vertices)
    print(f'\n{"="*55}')
    print(f'  顶点: {len(verts):,}  面: {len(mesh.triangles):,}')
    if len(verts) > 0:
        dx = verts[:, 0].max() - verts[:, 0].min()
        dy = verts[:, 1].max() - verts[:, 1].min()
        dz = verts[:, 2].max() - verts[:, 2].min()
        print(f'  尺寸 X: {dx*100:.1f}cm  Y: {dy*100:.1f}cm  Z: {dz*100:.1f}cm')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
