#!/usr/bin/env python3
"""V20 Step 2: TSDF 融合 + Mesh 提取 (分段处理, 限制内存)."""
import cv2, numpy as np, json, time, sys, gc
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

print(f'TSDF Fusion: {W_s}x{H_s}  {n_frames} frames')

# ── 估计相机距离 ──
sample_depths = []
for i in [0, 8, 16, 24, 32, 40, 48, 56, 64]:
    d = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    cv = d[H_s//2-50:H_s//2+50, W_s//2-50:W_s//2+50]
    cv = cv[cv > 0]
    if len(cv) > 10:
        sample_depths.append(np.median(cv))
cam_dist = np.median(sample_depths) if sample_depths else 0.42
print(f'  相机距离: {cam_dist*1000:.0f}mm')

# ── TSDF 融合 ──
import open3d as o3d

# 限定体素范围: 物体在原点附近, 15×15×15 cm 足够
voxel_size = 0.002  # 2mm
vol_bounds = np.array([[-0.1, -0.1, -0.02], [0.1, 0.1, 0.12]])  # 20×20×14cm

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size, sdf_trunc=0.008,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    volume_unit_resolution=16, depth_sampling_stride=4)

intrinsic = o3d.camera.PinholeCameraIntrinsic(W_s, H_s, fx_s, fy_s, cx_s, cy_s)

fused = 0
t0 = time.time()

for i in range(n_frames):
    depth_f16 = np.load(str(DEPTH_DIR / f'{i:03d}.npy'))
    depth = depth_f16.astype(np.float32)
    del depth_f16
    gc.collect()

    if (depth > 0).sum() < 200:
        continue

    depth_mm = (depth * 1000).astype(np.uint16)

    # 相机绕 Y 轴环绕
    th = np.radians(i * step_deg)
    cam_pos = np.array([cam_dist * np.sin(th), 0.0, cam_dist * np.cos(th)])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.vstack([right, up, forward])
    t = -R @ cam_pos
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t

    gray = np.full((H_s, W_s, 3), 128, dtype=np.uint8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(gray), o3d.geometry.Image(depth_mm),
        depth_scale=1000.0, depth_trunc=0.7, convert_rgb_to_intensity=False)

    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    fused += 1

    del depth, depth_mm
    gc.collect()

    if (i + 1) % 18 == 0:
        elapsed = time.time() - t0
        print(f'  [{i+1}/{n_frames}] {elapsed:.0f}s')

elapsed = time.time() - t0
print(f'TSDF 完成: {fused}/{n_frames} 帧融合, {elapsed:.0f}s')

# ── 提取 Mesh ──
print('提取 mesh ...')
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
verts = np.asarray(mesh.vertices)
print(f'Mesh: {len(verts)} verts, {len(mesh.triangles)} tris')

# 过滤
dists_xz = np.linalg.norm(verts[:, [0, 2]], axis=1)
keep = (dists_xz < 0.08) & (verts[:, 1] > -0.005) & (verts[:, 1] < 0.10)
keep_idx = np.where(keep)[0]

if len(keep_idx) > 100:
    tris = np.asarray(mesh.triangles)
    tri_keep = np.all(np.isin(tris, keep_idx), axis=1)
    old_to_new = -np.ones(len(verts), dtype=int)
    old_to_new[keep_idx] = np.arange(len(keep_idx))
    mesh_f = o3d.geometry.TriangleMesh()
    mesh_f.vertices = o3d.utility.Vector3dVector(verts[keep_idx])
    mesh_f.triangles = o3d.utility.Vector3iVector(old_to_new[tris[np.where(tri_keep)[0]]])
    mesh_f.compute_vertex_normals()
else:
    mesh_f = mesh

print(f'Filtered: {len(mesh_f.vertices)} verts, {len(mesh_f.triangles)} tris')
o3d.io.write_triangle_mesh(str(OUT / 'v20_sgbm_mesh.ply'), mesh_f)
print(f'Saved: {OUT}/v20_sgbm_mesh.ply')

# ── 同样方式处理内置深度 (对比) ──
print(f'\n── 内置深度 TSDF (对比) ──')
with open(CAPTURE_DIR / 'calib.json') as f:
    calib = json.load(f)
dscale = calib['depth_scale']

volume2 = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size, sdf_trunc=0.008,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    volume_unit_resolution=16, depth_sampling_stride=4)

fused2 = 0
covs2 = []
for i in range(n_frames):
    dr = cv2.imread(str(CAPTURE_DIR / f'depth/{i:03d}.png'), -1)
    if dr is None:
        continue
    dr_s = cv2.resize(dr, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
    dm = dr_s.astype(np.float32) * dscale
    valid_mask = (dm > 0.2) & (dm < 0.7)
    dm[~valid_mask] = 0
    if valid_mask.sum() < 200:
        del dr_s, dm, valid_mask
        continue

    covs2.append(valid_mask.sum() / valid_mask.size * 100)

    dmm = (dm * 1000).astype(np.uint16)
    th = np.radians(i * step_deg)
    cp = np.array([cam_dist * np.sin(th), 0.0, cam_dist * np.cos(th)])
    fwd = -cp / np.linalg.norm(cp)
    up = np.array([0.0, 1.0, 0.0])
    rt = np.cross(up, fwd); rt /= np.linalg.norm(rt)
    up = np.cross(fwd, rt)
    R2 = np.vstack([rt, up, fwd])
    P2 = np.eye(4); P2[:3, :3] = R2; P2[:3, 3] = -R2 @ cp

    gray = np.full((H_s, W_s, 3), 128, dtype=np.uint8)
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(gray), o3d.geometry.Image(dmm),
        depth_scale=1000.0, depth_trunc=0.7, convert_rgb_to_intensity=False)
    volume2.integrate(rgbd2, intrinsic, np.linalg.inv(P2))
    fused2 += 1
    del dr_s, dm, dmm, valid_mask
    gc.collect()

    if (i + 1) % 18 == 0:
        print(f'  [{i+1}/{n_frames}]')

if covs2:
    print(f'  内置覆盖率: {np.mean(covs2):.1f}%  融合 {fused2} 帧')
else:
    print('  ⚠️ 无有效内置深度帧')

mesh2 = volume2.extract_triangle_mesh()
mesh2.compute_vertex_normals()
verts2 = np.asarray(mesh2.vertices)
d2 = np.linalg.norm(verts2[:, [0, 2]], axis=1)
k2 = (d2 < 0.08) & (verts2[:, 1] > -0.005) & (verts2[:, 1] < 0.10)
kidx2 = np.where(k2)[0]
if len(kidx2) > 100:
    t2 = np.asarray(mesh2.triangles)
    tk2 = np.all(np.isin(t2, kidx2), axis=1)
    o2 = -np.ones(len(verts2), dtype=int)
    o2[kidx2] = np.arange(len(kidx2))
    mesh2f = o3d.geometry.TriangleMesh()
    mesh2f.vertices = o3d.utility.Vector3dVector(verts2[kidx2])
    mesh2f.triangles = o3d.utility.Vector3iVector(o2[t2[np.where(tk2)[0]]])
    mesh2f.compute_vertex_normals()
else:
    mesh2f = mesh2

o3d.io.write_triangle_mesh(str(OUT / 'v20_builtin_mesh.ply'), mesh2f)
print(f'Saved: {OUT}/v20_builtin_mesh.ply')

# ── 简单统计 ──
sgbm_covs = meta['coverage_per_frame']
print(f'\n{"="*55}')
print(f'V20 Complete ({W_s}x{H_s})')
print(f'  SGBM 覆盖率: {np.mean(sgbm_covs):.1f}%')
if covs2:
    print(f'  内置覆盖率: {np.mean(covs2):.1f}%')
print(f'  SGBM mesh:  {OUT}/v20_sgbm_mesh.ply')
print(f'  内置 mesh:  {OUT}/v20_builtin_mesh.ply')
print(f'{"="*55}')
