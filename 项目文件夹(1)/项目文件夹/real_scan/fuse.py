#!/usr/bin/env python3
"""两步融合：
  Step 1: 每帧 YOLO mask → 反投影 → PLY（相机坐标系）
  Step 2: 标定转轴旋转对齐 → 体素一致性过滤 → 融合
"""
import cv2, numpy as np, json, sys, time
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/capture'
OUT_DIR = BASE / 'output/fuse'
STEP1_DIR = OUT_DIR / 'step1_frames'
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEP1_DIR.mkdir(exist_ok=True)

# ── 加载 meta ──
with open(CAPTURE_DIR / 'meta.json') as f:
    meta = json.load(f)

fx, fy = meta['fx'], meta['fy']
ppx, ppy = meta['ppx'], meta['ppy']
W, H = meta['wxH_zoomed']
n_frames = meta['n_frames']
step_deg = meta['step_deg']
calib = meta.get('calib', {})
print(f'相机: {meta["camera"]}  zoom: {meta["zoom"]}x')
print(f'{W}x{H}  {n_frames} 帧  step={step_deg}°')

# ── YOLO ──
from ultralytics import YOLO
yolo = YOLO(str(BASE / 'yolov8n-seg.pt'))

# ── 射线 ──
u = np.arange(W); v = np.arange(H)
uu, vv = np.meshgrid(u, v)
ray_x = (uu - ppx) / fx
ray_y = (vv - ppy) / fy

color_files = sorted((CAPTURE_DIR / 'color').glob('*.jpg'))
depth_files = sorted((CAPTURE_DIR / 'depth').glob('*.png'))

# ═══════════════════════════════════════════════════════════════
# Step 1: 每帧 → YOLO mask → 反投影 → 存 PLY
# ═══════════════════════════════════════════════════════════════
step1_done = all((STEP1_DIR / f'{i:03d}.ply').exists() for i in range(min(n_frames, len(depth_files))))

if not step1_done:
    print(f'\n{"=" * 50}')
    print('Step 1: 逐帧反投影 (YOLO mask) → PLY')
    print('=' * 50)

    for i in range(min(n_frames, len(depth_files))):
        out_path = STEP1_DIR / f'{i:03d}.ply'
        if out_path.exists():
            continue

        color = cv2.imread(str(color_files[i]))
        depth_mm = cv2.imread(str(depth_files[i]), cv2.IMREAD_UNCHANGED)
        if color is None or depth_mm is None:
            continue

        if color.shape[:2] != (H, W):
            color = cv2.resize(color, (W, H))

        depth_m = depth_mm.astype(np.float32) * 0.001

        # YOLO mask
        results = yolo(color, verbose=False, classes=[56])
        mask = np.zeros((H, W), dtype=bool)
        if results[0].masks is not None:
            best, best_c = None, 0
            for j in range(len(results[0].boxes)):
                if results[0].names[int(results[0].boxes.cls[j])] == 'chair':
                    c = float(results[0].boxes.conf[j])
                    if c > best_c:
                        best_c = c
                        best = j
            if best is not None:
                m = results[0].masks.data[best].cpu().numpy()
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = m > 0.5

        valid = (depth_m > 0.01) & mask
        if valid.sum() < 50:
            print(f'  [{i:3d}] YOLO 未检测到椅子或有效点不足')
            continue

        # 深度范围过滤
        d_median = np.median(depth_m[valid])
        DEPTH_RANGE = 0.07  # ±70mm
        valid = valid & (depth_m > d_median - DEPTH_RANGE) & (depth_m < d_median + DEPTH_RANGE)

        z = depth_m[valid]
        x = ray_x[valid] * z
        y = -ray_y[valid] * z  # Y flip: world up
        pts = np.stack([x, y, z], axis=1)

        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(out_path), pcd)

        if (i + 1) % 18 == 0:
            print(f'  [{i+1:3d}/{n_frames}]')

    print(f'Step 1 完成 → {STEP1_DIR}/')
else:
    print('Step 1 已完成，跳过')

# ═══════════════════════════════════════════════════════════════
# Step 2: 加载各帧点云 → 旋转对齐 → 体素一致性过滤 → 融合
# ═══════════════════════════════════════════════════════════════
print(f'\n{"=" * 50}')
print('Step 2: 旋转对齐 + 体素一致性融合')
print('=' * 50)

# 转轴方向：优先用标定 R[2,:]（RANSAC 平面法向，比质心拟合稳定）
if calib:
    R_cal = np.array(calib['R'])
    axis_dir = np.array([R_cal[2, 0], -R_cal[2, 1], R_cal[2, 2]])  # Y flip
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    print('使用标定转轴方向 (R_cal[2,:], Y-flipped)')
else:
    fit_path = BASE / 'output/axis_fit.json'
    if fit_path.exists():
        with open(fit_path) as f:
            fit = json.load(f)
        axis_dir = np.array(fit['axis_dir'])
        print('使用拟合转轴 (axis_fit.json fallback)')
    else:
        print('错误: 无标定数据且无 axis_fit.json')
        sys.exit(1)

print(f'转轴方向: [{axis_dir[0]:.3f}, {axis_dir[1]:.3f}, {axis_dir[2]:.3f}]')


def rodrigues(axis, angle):
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


import open3d as o3d

all_pts = []
centroids = []
t0 = time.time()

for i in range(n_frames):
    ply_path = STEP1_DIR / f'{i:03d}.ply'
    if not ply_path.exists():
        continue

    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_cam = np.asarray(pcd.points)
    if len(pts_cam) < 50:
        continue

    centroids.append(pts_cam.mean(axis=0))

# 转轴点：所有帧质心均值
axis_point = np.array(centroids).mean(axis=0)
print(f'转轴点 (质心均值): [{axis_point[0]*1000:.0f}, {axis_point[1]*1000:.0f}, {axis_point[2]*1000:.0f}] mm')

# 旋转对齐
for i in range(n_frames):
    ply_path = STEP1_DIR / f'{i:03d}.ply'
    if not ply_path.exists():
        continue

    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_cam = np.asarray(pcd.points)
    if len(pts_cam) < 50:
        continue

    theta = np.radians(i * step_deg)
    R_undo = rodrigues(axis_dir, -theta)
    pts_w = (R_undo @ (pts_cam - axis_point).T).T
    all_pts.append(pts_w.astype(np.float32))

if not all_pts:
    print('错误: 无有效帧')
    sys.exit(1)

pts_all = np.vstack(all_pts)
elapsed = time.time() - t0
print(f'\n融合: {len(pts_all):,} pts / {elapsed:.0f}s')
print(f'  全范围: X=[{pts_all[:,0].min()*1000:.0f}, {pts_all[:,0].max()*1000:.0f}]mm')
print(f'          Y=[{pts_all[:,1].min()*1000:.0f}, {pts_all[:,1].max()*1000:.0f}]mm')
print(f'          Z=[{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}]mm')

# ── 体素一致性过滤 ──
# 真正椅子表面的点应该在多帧都被看到
VOXEL_SIZE = 0.002  # 2mm
MIN_FRAME_CONSISTENCY = 3

pts_shifted = pts_all - pts_all.min(axis=0)
voxel_indices = (pts_shifted / VOXEL_SIZE).astype(np.int32)
voxel_counts = Counter()
for vi in voxel_indices:
    voxel_counts[tuple(vi)] += 1

keep_voxels = {k for k, c in voxel_counts.items() if c >= MIN_FRAME_CONSISTENCY}
keep_mask = np.array([tuple(vi) in keep_voxels for vi in voxel_indices])
pts_filtered = pts_all[keep_mask]
print(f'体素一致性 (≥{MIN_FRAME_CONSISTENCY}帧, {VOXEL_SIZE*1000:.0f}mm): {len(pts_filtered):,} pts '
      f'(保留 {100*len(pts_filtered)/len(pts_all):.0f}%)')

# 保存原始融合
pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(pts_filtered)
pcd_ds = pcd_all.voxel_down_sample(voxel_size=0.001)

# 统计离群点去除 (更激进)
pcd_ds, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
o3d.io.write_point_cloud(str(OUT_DIR / 'fused_raw.ply'), pcd_ds)
print(f'原始融合 (1mm下采样 + stat outlier) → {OUT_DIR}/fused_raw.ply')

# RANSAC 去转台平面
plane_model, inliers = pcd_ds.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=500)
pcd_noplane = pcd_ds.select_by_index(inliers, invert=True)

pts_np = np.asarray(pcd_noplane.points)
print(f'去平面: {len(pts_np):,} pts')
print(f'  包围盒: X={np.ptp(pts_np[:,0])*1000:.0f}mm  '
      f'Y={np.ptp(pts_np[:,1])*1000:.0f}mm  '
      f'Z={np.ptp(pts_np[:,2])*1000:.0f}mm')

o3d.io.write_point_cloud(str(OUT_DIR / 'fused_noplane.ply'), pcd_noplane)
print(f'去平面后 → {OUT_DIR}/fused_noplane.ply')

# DBSCAN → 取最大簇
if len(pts_np) > 100:
    labels = np.array(pcd_noplane.cluster_dbscan(eps=0.005, min_points=10, print_progress=True))
    n_clusters = labels.max() + 1
    print(f'DBSCAN: {n_clusters} 簇')
    if n_clusters > 0:
        best_label = max(range(n_clusters), key=lambda lb: (labels == lb).sum())
        chair_idx = np.where(labels == best_label)[0]
        pcd_chair = pcd_noplane.select_by_index(chair_idx)
        vc = np.asarray(pcd_chair.points)
        print(f'椅子簇: {len(vc):,} pts')
        print(f'  尺寸: X={np.ptp(vc[:,0])*1000:.0f}mm  '
              f'Y={np.ptp(vc[:,1])*1000:.0f}mm  '
              f'Z={np.ptp(vc[:,2])*1000:.0f}mm')
        o3d.io.write_point_cloud(str(OUT_DIR / 'fused_chair.ply'), pcd_chair)
        print(f'椅子点云 → {OUT_DIR}/fused_chair.ply')
    else:
        pcd_chair = pcd_noplane
else:
    pcd_chair = pcd_noplane

# Poisson
print('\nPoisson 重建...')
pcd_chair.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
pcd_chair.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_chair, depth=9, width=0, scale=1.05, linear_fit=False)

d_arr = np.asarray(densities)
mesh.remove_vertices_by_mask(d_arr < np.percentile(d_arr, 5))
mesh.compute_vertex_normals()

# 去除小连通分量
labels, counts, _ = mesh.cluster_connected_triangles()
if len(counts) > 1:
    mesh.remove_triangles_by_index(np.where(labels != np.argmax(counts))[0])
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

o3d.io.write_triangle_mesh(str(OUT_DIR / 'chair.ply'), mesh)
o3d.io.write_triangle_mesh(str(OUT_DIR / 'chair.obj'), mesh)

vf = np.asarray(mesh.vertices)
print(f'\n最终 mesh: {len(vf):,} verts  '
      f'尺寸: X={np.ptp(vf[:,0])*1000:.0f}mm  '
      f'Y={np.ptp(vf[:,1])*1000:.0f}mm  '
      f'Z={np.ptp(vf[:,2])*1000:.0f}mm')
print(f'\n输出:')
print(f'  {OUT_DIR}/fused_raw.ply     — 原始融合（验证对齐）')
print(f'  {OUT_DIR}/fused_noplane.ply  — 去转台平面后')
print(f'  {OUT_DIR}/fused_chair.ply    — DBSCAN 椅子簇')
print(f'  {OUT_DIR}/chair.ply / .obj   — 最终 mesh')
