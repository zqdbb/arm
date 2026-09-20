#!/usr/bin/env python3
"""V20 融合 v3: 用标定转轴（正确变换方向）+ 中心裁剪."""
import numpy as np, json, time, sys
from pathlib import Path

BASE = Path(__file__).parent
DEPTH_DIR = BASE / 'output/v20_depth_sgbm'
CAPTURE_DIR = BASE / 'output/v20_capture'
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

# ── 加载 ──
with open(BASE / 'output/calibrate.json') as f:
    cal = json.load(f)
with open(DEPTH_DIR / 'meta.json') as f:
    meta = json.load(f)

R_cal = np.array(cal['R'])
t_cal = np.array(cal['t'])
cx_w, cy_w = cal['cx'], cal['cy']

W_s, H_s = meta['width'], meta['height']
fx, fy = meta['fx'], meta['fy']
cx, cy = meta['cx'], meta['cy']
n_frames = meta['n_frames']
step_deg = meta['step_deg']

print(f'Calib: color cam 640x360 crop, fx={cal["fx"]:.1f}')
print(f'SGBM:  IR cam {W_s}x{H_s} downsample, fx={fx:.1f}')
print(f'R_cal:\n{R_cal}')
print(f't_cal: {t_cal*1000} mm')
print(f'cx_w={cx_w*1000:.1f}mm, cy_w={cy_w*1000:.1f}mm')

# ── 正确变换: p_cam = R_cal @ p_plane + t_cal ──
# 旋转中心 (plane 坐标)
center_plane = np.array([cx_w, cy_w, 0.0])
# 旋转中心 (color camera 坐标)
center_color = R_cal @ center_plane + t_cal
# 旋转轴方向 (plane 法线在 camera frame)
axis_dir = R_cal[:, 2].copy()  # p_cam = R @ p_plane + t, plane Z轴 = R 第三列
if axis_dir[1] > 0:  # 让 Y 分量朝下（物理朝上）
    axis_dir = -axis_dir
axis_dir = axis_dir / np.linalg.norm(axis_dir)

print(f'\nRotation axis (color cam frame):')
print(f'  center: {center_color*1000} mm')
print(f'  direction: {axis_dir}')

# IR 左相机 ≈ 在彩色相机左边 25mm
# 世界点在 IR 相机坐标 = 世界点在 color 坐标 + [0.025, 0, 0]
ir_offset = np.array([0.025, 0.0, 0.0])
axis_origin = center_color + ir_offset
print(f'  IR-corrected center: {axis_origin*1000} mm')

# ── 裁剪到椅子区域 ──
CROP_W, CROP_H = 200, 150
x1 = W_s // 2 - CROP_W // 2
x2 = x1 + CROP_W
y1 = H_s // 2 - CROP_H // 2
y2 = y1 + CROP_H
print(f'\nCrop: [{x1}:{x2}, {y1}:{y2}] = {CROP_W}x{CROP_H}')

# 预计算射线
u = np.arange(W_s); v = np.arange(H_s)
uu, vv = np.meshgrid(u, v)
ray_x = ((uu - cx) / fx)[y1:y2, x1:x2]
ray_y = ((vv - cy) / fy)[y1:y2, x1:x2]

# ── 逐帧融合 ──
all_pts = []
t0 = time.time()

# Rodrigues: 绕 axis_dir 旋转 -th
def rodrigues(axis, angle):
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

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

    # 反旋转: 绕 axis_dir, 角度 -i*step_deg
    th = np.radians(i * step_deg)
    R_rot = rodrigues(axis_dir, -th)

    pts_centered = pts_cam - axis_origin
    pts_rotated = (R_rot @ pts_centered.T).T + axis_origin
    pts_world = pts_rotated - axis_origin  # 椅子中心移到原点

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

# ── 点云 + Mesh ──
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

if chair.sum() < 200:
    print('WARNING: too few chair points, using all')
    pcd_f = pcd
else:
    pcd_f = pcd.select_by_index(np.where(chair)[0])

o3d.io.write_point_cloud(str(OUT / 'v20_v3_pcd.ply'), pcd_f)

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
o3d.io.write_triangle_mesh(str(OUT / 'v20_v3_mesh.ply'), mesh_f)
print(f'Saved: {OUT}/v20_v3_mesh.ply')
print('Done.')
