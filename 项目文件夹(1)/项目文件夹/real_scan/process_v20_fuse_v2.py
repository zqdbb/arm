#!/usr/bin/env python3
"""V20 融合 v2: 中心裁剪 + 简化竖直转轴模型 + Poisson 重建."""
import numpy as np, json, time, sys
from pathlib import Path

BASE = Path(__file__).parent
DEPTH_DIR = BASE / 'output/v20_depth_sgbm'
CAPTURE_DIR = BASE / 'output/v20_capture'
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

with open(DEPTH_DIR / 'meta.json') as f:
    meta = json.load(f)
W_s, H_s = meta['width'], meta['height']
fx, fy = meta['fx'], meta['fy']
cx, cy = meta['cx'], meta['cy']
n_frames = meta['n_frames']
step_deg = meta['step_deg']

print(f'SGBM: {W_s}x{H_s} fx={fx:.1f}  {n_frames} frames @ {step_deg}°')

# ── 从数据估算转轴位置 ──
# 椅子在画面中心，用中心区域深度中值作为相机-椅子距离
sample_depths = []
for i in range(0, n_frames, 8):
    d = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    center = d[H_s//2-30:H_s//2+30, W_s//2-40:W_s//2+40]  # 60x80 center
    cv = center[center > 0]
    if len(cv) > 20:
        sample_depths.append(np.median(cv))
cam_dist = np.median(sample_depths)
print(f'Estimated cam_dist: {cam_dist*1000:.0f}mm (from center depth)')

# 裁剪到椅子区域: 椅子 ~70x30px, 用 160x120 裁剪留余量
CROP_W, CROP_H = 160, 120
x1 = W_s // 2 - CROP_W // 2
x2 = x1 + CROP_W
y1 = H_s // 2 - CROP_H // 2
y2 = y1 + CROP_H
print(f'Crop: [{x1}:{x2}, {y1}:{y2}] = {CROP_W}x{CROP_H}')

# 裁剪后内参
cx_c = W_s // 2 - x1
cy_c = H_s // 2 - y1

# 预计算射线 (使用原始内参)
u = np.arange(W_s)
v = np.arange(H_s)
uu, vv = np.meshgrid(u, v)
ray_x = ((uu - cx) / fx)[y1:y2, x1:x2]
ray_y = ((vv - cy) / fy)[y1:y2, x1:x2]

# ── 逐帧累积 ──
all_pts = []
t0 = time.time()

for i in range(n_frames):
    depth_full = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    depth = depth_full[y1:y2, x1:x2]
    valid = depth > 0

    if valid.sum() < 50:
        continue

    z = depth[valid]
    x = ray_x[valid] * z
    y = ray_y[valid] * z
    pts_cam = np.stack([x, y, z], axis=1)

    # 简化模型: 转轴竖直 (Y轴), 位于 Z=cam_dist
    # 反旋转角度 = -i*step_deg
    th = np.radians(i * step_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)

    # 绕 Y=cam_dist 处的竖直轴反旋转
    dz = pts_cam[:, 2] - cam_dist
    x_rot = cos_t * pts_cam[:, 0] + sin_t * dz
    z_rot = -sin_t * pts_cam[:, 0] + cos_t * dz + cam_dist

    # 平移到原点 (椅子中心)
    pts_world = np.stack([x_rot, pts_cam[:, 1], z_rot - cam_dist], axis=1)
    all_pts.append(pts_world.astype(np.float32))

    if (i + 1) % 36 == 0:
        n = sum(len(p) for p in all_pts)
        print(f'  [{i+1}/{n_frames}] {n} pts  {time.time()-t0:.0f}s')

# ── 合并 ──
pts_all = np.vstack(all_pts)
print(f'\nAccumulated: {len(pts_all)} pts')
print(f'  X: [{pts_all[:,0].min()*1000:.0f}, {pts_all[:,0].max()*1000:.0f}] mm')
print(f'  Y: [{pts_all[:,1].min()*1000:.0f}, {pts_all[:,1].max()*1000:.0f}] mm')
print(f'  Z: [{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}] mm')

# ── Poisson 重建 ──
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_all)
pcd = pcd.voxel_down_sample(voxel_size=0.0005)
print(f'Downsampled: {len(pcd.points)} pts')

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f'De-noised: {len(pcd.points)} pts')

# 椅子过滤器: 预期尺寸 43x88x37mm (XxZxY)
pts_arr = np.asarray(pcd.points)
dists_xz = np.linalg.norm(pts_arr[:, [0, 2]], axis=1)
chair_mask = (dists_xz < 0.06) & (pts_arr[:, 1] > -0.01) & (pts_arr[:, 1] < 0.06)
print(f'Chair region: {chair_mask.sum()}/{len(pts_arr)} ({chair_mask.sum()/max(len(pts_arr),1)*100:.1f}%)')

if chair_mask.sum() < 200:
    print('WARNING: too few chair points, using all points')
    pcd_f = pcd
else:
    pcd_f = pcd.select_by_index(np.where(chair_mask)[0])

o3d.io.write_point_cloud(str(OUT / 'v20_v2_pcd.ply'), pcd_f)
print(f'Point cloud saved: {OUT}/v20_v2_pcd.ply')

# Poisson
print('Running Poisson reconstruction...')
pcd_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_f, depth=8, width=0, scale=1.1, linear_fit=False)

d_arr = np.asarray(densities)
mesh.remove_vertices_by_mask(d_arr < np.percentile(d_arr, 10))
mesh.compute_vertex_normals()

# 清理外围
verts = np.asarray(mesh.vertices)
d_xz = np.linalg.norm(verts[:, [0, 2]], axis=1)
keep = (d_xz < 0.05) & (verts[:, 1] > 0.0) & (verts[:, 1] < 0.055)
k_idx = np.where(keep)[0]

if len(k_idx) > 100:
    tris = np.asarray(mesh.triangles)
    tk = np.all(np.isin(tris, k_idx), axis=1)
    old = -np.ones(len(verts), dtype=int)
    old[k_idx] = np.arange(len(k_idx))
    mesh_f = o3d.geometry.TriangleMesh()
    mesh_f.vertices = o3d.utility.Vector3dVector(verts[k_idx])
    mesh_f.triangles = o3d.utility.Vector3iVector(old[tris[np.where(tk)[0]]])
    mesh_f.compute_vertex_normals()
else:
    mesh_f = mesh

vf = np.asarray(mesh_f.vertices)
print(f'\nMesh: {len(vf)} verts, {len(mesh_f.triangles)} tris')
print(f'Size: X={np.ptp(vf[:,0])*1000:.0f} Y={np.ptp(vf[:,1])*1000:.0f} Z={np.ptp(vf[:,2])*1000:.0f} mm')
print(f'Expected: 43x88x37mm (XxZxY)')
o3d.io.write_triangle_mesh(str(OUT / 'v20_v2_mesh.ply'), mesh_f)
print(f'Saved: {OUT}/v20_v2_mesh.ply')
print('Done.')
