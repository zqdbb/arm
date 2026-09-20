#!/usr/bin/env python3
"""V20 Step 2: 累积点云 + Poisson 重建 (简单位姿模型)."""
import cv2, numpy as np, json, time, gc
from pathlib import Path

BASE = Path(__file__).parent
DEPTH_DIR = BASE / 'output/v20_depth_sgbm'
CAPTURE_DIR = BASE / 'output/v20_capture'
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

with open(DEPTH_DIR / 'meta.json') as f:
    meta = json.load(f)

W_s, H_s = meta['width'], meta['height']
fx_s, fy_s = meta['fx'], meta['fy']
cx_s, cy_s = meta['cx'], meta['cy']
n_frames = meta['n_frames']
step_deg = meta['step_deg']

print(f'V20 Point Cloud: {W_s}x{H_s}  {n_frames} frames')

# 像素射线 (预计算)
u = np.arange(W_s)
v = np.arange(H_s)
uu, vv = np.meshgrid(u, v)
ray_x = (uu - cx_s) / fx_s
ray_y = (vv - cy_s) / fy_s

# 估计深度均值
sample_depths = []
for i in [0, 8, 16, 24, 32, 40, 48, 56, 64]:
    d = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    cv = d[H_s//2-50:H_s//2+50, W_s//2-50:W_s//2+50]
    cv = cv[cv > 0]
    if len(cv) > 10:
        sample_depths.append(np.median(cv))
cam_dist = np.median(sample_depths) if sample_depths else 0.42
print(f'中心深度: {cam_dist*1000:.0f}mm')

# ── 逐帧累积 (简化位姿: 只 undo 旋转) ──
all_pts = []
subsample = 4  # 每 4 像素取 1
t0 = time.time()

for i in range(n_frames):
    depth = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    valid_s = depth[::subsample, ::subsample] > 0
    if valid_s.sum() < 30:
        continue

    z = depth[::subsample, ::subsample][valid_s]
    x = ray_x[::subsample, ::subsample][valid_s] * z
    y = ray_y[::subsample, ::subsample][valid_s] * z
    pts_cam = np.stack([x, y, z], axis=1)  # (N,3) in camera frame

    # Undo turntable rotation:
    # 转台旋转轴 ≈ 相机视线方向, 距相机的深度 = cam_dist
    # 把相机坐标绕 Z=0 的 Y 轴旋转 -angle (等效于 undo 转台旋转)
    th = np.radians(i * step_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)

    # 旋转轴在相机坐标系的位置 (在画面中心, 深度 cam_dist)
    axis_z = cam_dist

    # 平移使旋转轴经过原点 → Y轴旋转 → 平移回来
    pz = pts_cam[:, 2] - axis_z
    x_rot = cos_t * pts_cam[:, 0] + sin_t * pz
    z_rot = -sin_t * pts_cam[:, 0] + cos_t * pz + axis_z
    y_rot = pts_cam[:, 1]

    pts_w = np.stack([x_rot, y_rot, z_rot], axis=1)
    all_pts.append(pts_w.astype(np.float32))

    del depth, valid_s, z, x, y, pts_cam, pts_w
    gc.collect()

    if (i + 1) % 18 == 0:
        elapsed = time.time() - t0
        n_pts = sum(len(p) for p in all_pts)
        print(f'  [{i+1}/{n_frames}] {n_pts} pts  {elapsed:.0f}s')

elapsed = time.time() - t0
total_pts = sum(len(p) for p in all_pts)
print(f'累积: {total_pts} 点 / {elapsed:.0f}s')

# ── 合并 + 检查位置 ──
pts_all = np.vstack(all_pts)
print(f'\n原始点云: {len(pts_all)} 点')
print(f'  X: [{pts_all[:,0].min()*1000:.0f}, {pts_all[:,0].max()*1000:.0f}] mm')
print(f'  Y: [{pts_all[:,1].min()*1000:.0f}, {pts_all[:,1].max()*1000:.0f}] mm')
print(f'  Z: [{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}] mm')

# 椅子应该在 Z≈0 (转台中心) 附近, 而不是在 Z≈cam_dist
# 把 Z 平移使物体在原点
pts_all[:, 2] -= cam_dist
print(f'Centered: Z=[{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}] mm')

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_all)

# 体素下采样
pcd = pcd.voxel_down_sample(voxel_size=0.0005)
print(f'下采样: {len(pcd.points)} 点')

# 去噪
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f'去噪: {len(pcd.points)} 点')

# 椅子过滤: 原点附近 8cm×8cm×10cm
pts_arr = np.asarray(pcd.points)
print(f'  过滤前范围:')
print(f'    X: [{pts_arr[:,0].min()*1000:.0f}, {pts_arr[:,0].max()*1000:.0f}] mm')
print(f'    Y: [{pts_arr[:,1].min()*1000:.0f}, {pts_arr[:,1].max()*1000:.0f}] mm')
print(f'    Z: [{pts_arr[:,2].min()*1000:.0f}, {pts_arr[:,2].max()*1000:.0f}] mm')

dists_xz = np.linalg.norm(pts_arr[:, [0, 2]], axis=1)
chair_mask = (dists_xz < 0.08) & (pts_arr[:, 1] > -0.005) & (pts_arr[:, 1] < 0.10)
print(f'  在椅子区域内: {chair_mask.sum()} / {len(pts_arr)} ({chair_mask.sum()/max(len(pts_arr),1)*100:.1f}%)')

pcd_chair = pcd.select_by_index(np.where(chair_mask)[0])
if len(pcd_chair.points) < 100:
    print('ERROR: chair points too few, saving unfiltered')
    pcd_chair = pcd

o3d.io.write_point_cloud(str(OUT / 'v20_sgbm_pcd.ply'), pcd_chair)
print(f'点云: {OUT}/v20_sgbm_pcd.ply  ({len(pcd_chair.points)} 点)')

# ── Poisson 重建 ──
if len(pcd_chair.points) < 100:
    print('ERROR: Not enough points for Poisson')
    sys.exit(1)

print('\nPoisson 重建 ...')
pcd_chair.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_chair, depth=8, width=0, scale=1.1, linear_fit=False)

dens_arr = np.asarray(densities)
thresh = np.percentile(dens_arr, 10)
mesh.remove_vertices_by_mask(dens_arr < thresh)
mesh.compute_vertex_normals()

verts = np.asarray(mesh.vertices)
d_xz = np.linalg.norm(verts[:, [0, 2]], axis=1)
keep = (d_xz < 0.07) & (verts[:, 1] > 0.0) & (verts[:, 1] < 0.095)
keep_idx = np.where(keep)[0]

if len(keep_idx) > 100:
    tris = np.asarray(mesh.triangles)
    tk = np.all(np.isin(tris, keep_idx), axis=1)
    old = -np.ones(len(verts), dtype=int)
    old[keep_idx] = np.arange(len(keep_idx))
    mesh_f = o3d.geometry.TriangleMesh()
    mesh_f.vertices = o3d.utility.Vector3dVector(verts[keep_idx])
    mesh_f.triangles = o3d.utility.Vector3iVector(old[tris[np.where(tk)[0]]])
    mesh_f.compute_vertex_normals()
else:
    mesh_f = mesh
    print('WARNING: filter too aggressive, using unfiltered')

print(f'Mesh: {len(mesh_f.vertices)} verts, {len(mesh_f.triangles)} tris')
o3d.io.write_triangle_mesh(str(OUT / 'v20_sgbm_mesh.ply'), mesh_f)
print(f'Saved: {OUT}/v20_sgbm_mesh.ply')
print('\nV20 Complete')
