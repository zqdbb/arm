#!/usr/bin/env python3
"""
V20: IR SGBM 处理 + TSDF 融合 + Mesh 提取
用法: python3 process_v20.py
输入: output/v20_capture/
输出: output/v20_ir_stereo/
"""

import cv2, numpy as np, json, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/v20_capture'
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

# ── 加载内参 + 降采样 ──
with open(CAPTURE_DIR / 'calib.json') as f:
    calib = json.load(f)
ir = calib['ir_intrinsics']
baseline = calib['baseline_m']
step_deg = calib['step_deg']
n_frames = calib['n_frames']

# 降采样到 1/4 内存 (640×360)
SCALE = 0.5
W_s = int(ir['width'] * SCALE)
H_s = int(ir['height'] * SCALE)
fx_s = ir['fx'] * SCALE
fy_s = ir['fy'] * SCALE
cx_s = ir['ppx'] * SCALE
cy_s = ir['ppy'] * SCALE
print(f'IR: {W_s}x{H_s} (降采样 {SCALE:.1f}x) fx={fx_s:.1f} fy={fy_s:.1f}')
print(f'步进: {step_deg}°  帧数: {n_frames}  基线: {baseline*1000:.0f}mm')

# ── SGBM (小分辨率) ──
# 视差范围: 0.4m距离 → disp = 318*0.05/0.4 ≈ 40, 0.2m → ~80
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=5,
    P1=8 * 3 * 5**2, P2=32 * 3 * 5**2,
    disp12MaxDiff=3, uniquenessRatio=5,
    speckleWindowSize=100, speckleRange=2,
    preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)


def process_frame(i):
    """加载 + 降采样 + SGBM → 深度图 (米)."""
    ir_l = cv2.imread(str(CAPTURE_DIR / f'ir_left/{i:03d}.png'), cv2.IMREAD_GRAYSCALE)
    ir_r = cv2.imread(str(CAPTURE_DIR / f'ir_right/{i:03d}.png'), cv2.IMREAD_GRAYSCALE)
    if ir_l is None or ir_r is None:
        return None

    ir_l_s = cv2.resize(ir_l, (W_s, H_s))
    ir_r_s = cv2.resize(ir_r, (W_s, H_s))

    disp = sgbm.compute(ir_l_s, ir_r_s).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > 0.5
    depth[valid] = (fx_s * baseline) / disp[valid]

    # 只保留 0.2–0.7m
    depth[(depth < 0.2) | (depth > 0.7)] = 0
    return depth


def compute_pose(angle_deg, dist_m):
    """相机绕 Y 轴看原点. angle_deg = 转台角度."""
    th = np.radians(angle_deg)
    cam_pos = np.array([dist_m * np.sin(th), 0.0, dist_m * np.cos(th)])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R = np.vstack([right, up, forward])
    t = -R @ cam_pos
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ── 第一步: 估计相机距离 ──
print('\n── 估计相机距离 ──')
sample_frames = [0, 8, 16, 24, 32, 40, 48, 56, 64]
sample_depths = []
for i in sample_frames:
    d = process_frame(i)
    if d is not None and (d > 0).sum() > 100:
        center = d[H_s//2-50:H_s//2+50, W_s//2-50:W_s//2+50]
        cv = center[center > 0]
        if len(cv) > 20:
            sample_depths.append(np.median(cv))

cam_dist = np.median(sample_depths) if sample_depths else 0.42
print(f'  估计相机距离: {cam_dist*1000:.0f}mm ({len(sample_depths)} 采样帧)')

# ── 第二步: 处理所有帧 ──
print(f'\n── SGBM 处理 {n_frames} 帧 ──')
t0 = time.time()
all_depths = []
coverage_stats = []

for i in range(n_frames):
    depth = process_frame(i)
    all_depths.append(depth)
    if depth is not None:
        cov = (depth > 0).sum() / depth.size * 100
        coverage_stats.append(cov)
    if (i + 1) % 18 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_frames - i - 1)
        print(f'  [{i+1}/{n_frames}] 已用{elapsed:.0f}s  剩余~{eta:.0f}s')

elapsed = time.time() - t0
print(f'SGBM 完成: {elapsed:.0f}s ({elapsed/n_frames:.1f}s/帧)')
if coverage_stats:
    print(f'覆盖率: 均值 {np.mean(coverage_stats):.1f}%  最差 {np.min(coverage_stats):.1f}%  '
          f'最好 {np.max(coverage_stats):.1f}%')

# ── 第三步: TSDF 融合 ──
print(f'\n── TSDF 融合 ──')
import open3d as o3d

voxel_size = 0.0015  # 1.5mm 体素 (降采样补偿)
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size, sdf_trunc=0.006,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

intrinsic = o3d.camera.PinholeCameraIntrinsic(W_s, H_s, fx_s, fy_s, cx_s, cy_s)
fused_count = 0

for i in range(n_frames):
    depth = all_depths[i]
    if depth is None or (depth > 0).sum() < 200:
        continue

    depth_mm = (depth * 1000).astype(np.uint16)
    angle_deg = i * step_deg
    pose = compute_pose(angle_deg, cam_dist)

    # 灰色 dummy color
    gray = np.full((H_s, W_s, 3), 128, dtype=np.uint8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(gray), o3d.geometry.Image(depth_mm),
        depth_scale=1000.0, depth_trunc=0.7, convert_rgb_to_intensity=False)

    volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    fused_count += 1

print(f'  融合: {fused_count}/{n_frames} 帧')

# ── 第四步: 提取 + 过滤 Mesh ──
print(f'\n── 提取 Mesh ──')
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
verts = np.asarray(mesh.vertices)
print(f'  原始 mesh: {len(verts)} 顶点, {len(mesh.triangles)} 面')

dists_xz = np.linalg.norm(verts[:, [0, 2]], axis=1)
keep = (dists_xz < 0.08) & (verts[:, 1] > -0.01) & (verts[:, 1] < 0.12)
keep_idx = np.where(keep)[0]

if len(keep_idx) > 100:
    tris = np.asarray(mesh.triangles)
    tri_keep = np.all(np.isin(tris, keep_idx), axis=1)
    old_to_new = -np.ones(len(verts), dtype=int)
    old_to_new[keep_idx] = np.arange(len(keep_idx))
    chair_mesh = o3d.geometry.TriangleMesh()
    chair_mesh.vertices = o3d.utility.Vector3dVector(verts[keep_idx])
    chair_mesh.triangles = o3d.utility.Vector3iVector(old_to_new[tris[np.where(tri_keep)[0]]])
    chair_mesh.compute_vertex_normals()
else:
    chair_mesh = mesh

print(f'  最终 mesh: {len(chair_mesh.vertices)} 顶点, {len(chair_mesh.triangles)} 面')
o3d.io.write_triangle_mesh(str(OUT / 'v20_sgbm_mesh.ply'), chair_mesh)

# ── 第五步: 内置深度融合 (对比) ──
print(f'\n── 内置深度融合 (对比) ──')
volume2 = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_size, sdf_trunc=0.006,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)

depth_scale = calib['depth_scale']
builtin_covs = []
builtin_fused = 0

for i in range(n_frames):
    depth_raw = cv2.imread(str(CAPTURE_DIR / f'depth/{i:03d}.png'), -1)
    if depth_raw is None:
        continue
    depth_raw_s = cv2.resize(depth_raw, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
    depth_m = depth_raw_s.astype(np.float32) * depth_scale
    valid = (depth_m > 0.2) & (depth_m < 0.7)
    depth_m[~valid] = 0
    if valid.sum() < 200:
        continue

    cov = valid.sum() / valid.size * 100
    builtin_covs.append(cov)

    depth_mm = (depth_m * 1000).astype(np.uint16)
    pose = compute_pose(i * step_deg, cam_dist)
    gray = np.full((H_s, W_s, 3), 128, dtype=np.uint8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(gray), o3d.geometry.Image(depth_mm),
        depth_scale=1000.0, depth_trunc=0.7, convert_rgb_to_intensity=False)
    volume2.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    builtin_fused += 1

if builtin_covs:
    print(f'  内置深度覆盖率: 均值 {np.mean(builtin_covs):.1f}%  融合 {builtin_fused} 帧')
else:
    print(f'  ⚠️ 无有效内置深度帧')

mesh2 = volume2.extract_triangle_mesh()
mesh2.compute_vertex_normals()
verts2 = np.asarray(mesh2.vertices)
dists2 = np.linalg.norm(verts2[:, [0, 2]], axis=1)
keep2 = (dists2 < 0.08) & (verts2[:, 1] > -0.01) & (verts2[:, 1] < 0.12)
keep_idx2 = np.where(keep2)[0]

if len(keep_idx2) > 100:
    tris2 = np.asarray(mesh2.triangles)
    tri_keep2 = np.all(np.isin(tris2, keep_idx2), axis=1)
    old2 = -np.ones(len(verts2), dtype=int)
    old2[keep_idx2] = np.arange(len(keep_idx2))
    builtin_mesh = o3d.geometry.TriangleMesh()
    builtin_mesh.vertices = o3d.utility.Vector3dVector(verts2[keep_idx2])
    builtin_mesh.triangles = o3d.utility.Vector3iVector(old2[tris2[np.where(tri_keep2)[0]]])
    builtin_mesh.compute_vertex_normals()
else:
    builtin_mesh = mesh2

o3d.io.write_triangle_mesh(str(OUT / 'v20_builtin_mesh.ply'), builtin_mesh)

# ── 第六步: 覆盖率对比图 ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(len(coverage_stats)), coverage_stats, 'o-', markersize=3,
        color='steelblue', label=f'SGBM (mean {np.mean(coverage_stats):.1f}%)')
if builtin_covs:
    ax.plot(range(len(builtin_covs)), builtin_covs, 's-', markersize=3,
            color='darkorange', label=f'Built-in (mean {np.mean(builtin_covs):.1f}%)')
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Frame')
ax.set_ylabel('Depth Coverage (%)')
ax.set_title(f'V20 Coverage ({n_frames} frames, {W_s}x{H_s})')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(OUT / 'v20_coverage_comparison.png', dpi=150)

# ── 统计 ──
print(f'\n{"="*55}')
print(f'V20 完成 ({W_s}x{H_s})')
print(f'  SGBM 覆盖率: {np.mean(coverage_stats):.1f}%')
if builtin_covs:
    print(f'  内置覆盖率: {np.mean(builtin_covs):.1f}%')
print(f'  SGBM mesh:  {OUT}/v20_sgbm_mesh.ply')
print(f'  内置 mesh:  {OUT}/v20_builtin_mesh.ply')
print(f'{"="*55}')
