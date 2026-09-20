#!/usr/bin/env python3
"""V20 高分辨率融合: 用全分辨率 SGBM 深度 + 标定转轴."""
import numpy as np, json, time, sys
from pathlib import Path

BASE = Path(__file__).parent
DEPTH_DIR = BASE / 'output/v20_depth_sgbm_hires'
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

with open(BASE / 'output/calibrate.json') as f:
    cal = json.load(f)
with open(DEPTH_DIR / 'meta.json') as f:
    meta = json.load(f)

R_cal = np.array(cal['R'])
W_s, H_s = meta['width'], meta['height']
fx, fy = meta['fx'], meta['fy']
cx, cy = meta['cx'], meta['cy']
n_frames = meta['n_frames']; step_deg = meta['step_deg']

print(f'Hi-res depth: {W_s}x{H_s} fx={fx:.1f}')
print(f'Coverage: {meta["coverage_mean"]:.1f}%')

# ── 转轴参数 (使用搜索最佳) ──
axis_dir = R_cal[:, 2].copy()
if axis_dir[1] > 0:
    axis_dir = -axis_dir
axis_dir = axis_dir / np.linalg.norm(axis_dir)

# 搜索最佳: X=-20mm, Y=0mm, Z=408mm
axis_origin = np.array([-0.020, 0.0, 0.408])
print(f'Axis origin: {axis_origin*1000} mm')
print(f'Axis dir: {axis_dir}')

# ── 预计算射线 ──
u = np.arange(W_s); v = np.arange(H_s)
uu, vv = np.meshgrid(u, v)
ray_x = (uu - cx) / fx
ray_y = (vv - cy) / fy

def rodrigues(axis, angle):
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# ── 逐帧融合 ──
all_pts = []
t0 = time.time()

for i in range(n_frames):
    depth = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    valid = ~np.isnan(depth) & (depth > 0)

    if valid.sum() < 30:
        continue

    z = depth[valid]
    x = ray_x[valid] * z
    y = ray_y[valid] * z
    pts_cam = np.stack([x, y, z], axis=1)

    th = np.radians(i * step_deg)
    R_rot = rodrigues(axis_dir, -th)
    pts_centered = pts_cam - axis_origin
    pts_rotated = (R_rot @ pts_centered.T).T + axis_origin
    pts_world = pts_rotated - axis_origin

    all_pts.append(pts_world.astype(np.float32))

    if (i + 1) % 36 == 0:
        n = sum(len(p) for p in all_pts)
        print(f'  [{i+1}/{n_frames}] {n} pts  {time.time()-t0:.0f}s')

pts_all = np.vstack(all_pts)
print(f'\nAccumulated: {len(pts_all)} pts')
print(f'  X: [{pts_all[:,0].min()*1000:.0f}, {pts_all[:,0].max()*1000:.0f}] mm')
print(f'  Y: [{pts_all[:,1].min()*1000:.0f}, {pts_all[:,1].max()*1000:.0f}] mm')
print(f'  Z: [{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}] mm')

# ── Mesh ──
import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_all)
pcd = pcd.voxel_down_sample(voxel_size=0.0005)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f'Filtered: {len(pcd.points)} pts')

pts_arr = np.asarray(pcd.points)
dists_xz = np.linalg.norm(pts_arr[:, [0, 2]], axis=1)
chair = (dists_xz < 0.06) & (pts_arr[:, 1] > -0.01) & (pts_arr[:, 1] < 0.06)
print(f'Chair region: {chair.sum()}/{len(pts_arr)} ({chair.sum()/max(len(pts_arr),1)*100:.1f}%)')

pcd_f = pcd if chair.sum() < 200 else pcd.select_by_index(np.where(chair)[0])
o3d.io.write_point_cloud(str(OUT / 'v20_hires_pcd.ply'), pcd_f)

# Poisson
pcd_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.003, max_nn=30))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_f, depth=8, width=0, scale=1.1, linear_fit=False)

d_arr = np.asarray(densities)
mesh.remove_vertices_by_mask(d_arr < np.percentile(d_arr, 10))
mesh.compute_vertex_normals()

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
o3d.io.write_triangle_mesh(str(OUT / 'v20_hires_mesh.ply'), mesh_f)
print(f'Saved: {OUT}/v20_hires_mesh.ply')
print('Done.')
