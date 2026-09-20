#!/usr/bin/env python3
"""V20: 标定位姿融合 (修正版 + IR/Color 外参 + 内置深度对照)."""
import cv2, numpy as np, json, time, gc
from pathlib import Path

BASE = Path(__file__).parent
DEPTH_DIR = BASE / 'output/v20_depth_sgbm'
CAPTURE_DIR = BASE / 'output/v20_capture'
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load calibration ──
with open(BASE / 'output/calibrate.json') as f:
    cal = json.load(f)
with open(DEPTH_DIR / 'meta.json') as f:
    meta = json.load(f)
with open(CAPTURE_DIR / 'calib.json') as f:
    v20_cal = json.load(f)

R_cal = np.array(cal['R'])
t_cal = np.array(cal['t'])
cx_w, cy_w = cal['cx'], cal['cy']

fx_s, fy_s = meta['fx'], meta['fy']
cx_s, cy_s = meta['cx'], meta['cy']
W_s, H_s = meta['width'], meta['height']
step_deg = meta['step_deg']
n_frames = meta['n_frames']

print(f'Calib: {cal["width"]}x{cal["height"]} crop, fx={cal["fx"]:.1f}')
print(f'SGBM:  {W_s}x{H_s} downsample, fx={fx_s:.1f}')
# FOV correction: calib 640x360 is center CROP (fx unchanged), SGBM 640x360 is DOWNSAMPLE (fx/2)
# The calib sees center half of FOV. SGBM sees full FOV at half res.
# For the same pixel (u,v) in both: different ray directions.
# → Crop SGBM depth to center half to match calibration FOV
CROP = 0.5  # use center 50% of SGBM = center of full FOV = calibration FOV
x1 = int(W_s * (1 - CROP) / 2)
x2 = int(W_s * (1 + CROP) / 2)
y1 = int(H_s * (1 - CROP) / 2)
y2 = int(H_s * (1 + CROP) / 2)
print(f'Crop to center: [{x1}:{x2}, {y1}:{y2}] ({x2-x1}x{y2-y1})')

# Cropped intrinsics: fx unchanged (same as full-res, crop preserves fx)
# ppx cropped to center of crop region
cx_c = (W_s / 2 - x1)  # center of cropped region
cy_c = (H_s / 2 - y1)
# But fx is from meta (fx_s = fx_full/2). For crop, fx should be fx_full = fx_s * 2
fx_c = fx_s * 2  # undo downsample
fy_c = fy_s * 2
print(f'Cropped intrinsic: fx={fx_c:.1f} fy={fy_c:.1f} cx={cx_c:.1f} cy={cy_c:.1f}')
# This should be close to calib["fx"] but for IR camera (~637 vs ~908 for color)

# ── Rotation axis ──
# p_plane = R_cal @ p_cam + t_cal
# p_cam = R_cal.T @ (p_plane - t_cal)
center_plane = np.array([cx_w, cy_w, 0.0])
center_cam = R_cal.T @ (center_plane - t_cal)
axis_y = R_cal.T[:, 2].copy()  # plane normal
if axis_y[1] < 0:
    axis_y = -axis_y
print(f'\nRotation axis (color cam frame):')
print(f'  center: [{center_cam[0]*1000:.1f}, {center_cam[1]*1000:.1f}, {center_cam[2]*1000:.1f}] mm')
print(f'  direction: [{axis_y[0]:.3f}, {axis_y[1]:.3f}, {axis_y[2]:.3f}]')

# Apply IR-Color extrinsic (simplified: just use the values as-is, D435i offset ~25mm)
# IR cam is ~25mm to the left of color cam
ir_offset = np.array([-0.025, 0.0, 0.0])  # approximate
axis_origin = center_cam + ir_offset
print(f'  IR-corrected: [{axis_origin[0]*1000:.1f}, {axis_origin[1]*1000:.1f}, {axis_origin[2]*1000:.1f}] mm')


# ── Fusion function ──
def fuse_depth_maps(depth_dir, depth_scale_factor=1.0, label=""):
    """Fuse depth maps from disk. Returns point cloud."""
    all_pts = []
    t0 = time.time()
    subsample = 2

    u = np.arange(W_s)
    v = np.arange(H_s)
    uu, vv = np.meshgrid(u, v)
    ray_x_full = (uu - cx_s) / fx_s
    ray_y_full = (vv - cy_s) / fy_s

    # Crop to calibration FOV
    ray_x_c = ray_x_full[y1:y2, x1:x2]
    ray_y_c = ray_y_full[y1:y2, x1:x2]

    for i in range(n_frames):
        fpath = depth_dir / f'{i:03d}.npy'
        if fpath.exists():
            depth_full = np.load(str(fpath)).astype(np.float32) * depth_scale_factor
        else:
            # built-in depth from PNG
            dr = cv2.imread(str(CAPTURE_DIR / f'depth/{i:03d}.png'), -1)
            if dr is None:
                continue
            dr = cv2.resize(dr, (W_s, H_s), interpolation=cv2.INTER_NEAREST)
            depth_full = dr.astype(np.float32) * depth_scale_factor

        depth_c = depth_full[y1:y2, x1:x2]
        vs = depth_c[::subsample, ::subsample] > 0
        if vs.sum() < 30:
            continue

        z = depth_c[::subsample, ::subsample][vs]
        x = ray_x_c[::subsample, ::subsample][vs] * z
        y = ray_y_c[::subsample, ::subsample][vs] * z
        pts_cam = np.stack([x, y, z], axis=1)

        # Undo turntable rotation
        th = np.radians(i * step_deg)
        K = np.array([
            [0, -axis_y[2], axis_y[1]],
            [axis_y[2], 0, -axis_y[0]],
            [-axis_y[1], axis_y[0], 0]
        ])
        R_rot = np.eye(3) + np.sin(-th) * K + (1 - np.cos(-th)) * (K @ K)

        pts_centered = pts_cam - axis_origin
        pts_rotated = (R_rot @ pts_centered.T).T + axis_origin
        pts_world = pts_rotated - axis_origin

        all_pts.append(pts_world.astype(np.float32))

        if (i + 1) % 18 == 0:
            elapsed = time.time() - t0
            print(f'  [{label}] [{i+1}/{n_frames}] {sum(len(p) for p in all_pts)} pts  {elapsed:.0f}s')

    if not all_pts:
        return None, None

    pts_all = np.vstack(all_pts)
    print(f'  [{label}] Accumulated: {len(pts_all)} pts / {time.time()-t0:.0f}s')
    print(f'    X: [{pts_all[:,0].min()*1000:.0f}, {pts_all[:,0].max()*1000:.0f}] mm')
    print(f'    Y: [{pts_all[:,1].min()*1000:.0f}, {pts_all[:,1].max()*1000:.0f}] mm')
    print(f'    Z: [{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}] mm')

    return pts_all, time.time() - t0


# ── Fuse SGBM depth ──
print('\n--- SGBM Fusion ---')
pts_sgbm, _ = fuse_depth_maps(DEPTH_DIR, depth_scale_factor=1.0, label="SGBM")

# ── Fuse built-in depth ──
print('\n--- Built-in Fusion ---')
with open(CAPTURE_DIR / 'calib.json') as f:
    cap_cal = json.load(f)
dscale = cap_cal['depth_scale']
pts_builtin, _ = fuse_depth_maps(CAPTURE_DIR / 'depth', depth_scale_factor=dscale, label="Builtin")

if pts_sgbm is None and pts_builtin is None:
    print('ERROR: No points!')
    sys.exit(1)

# ── Make meshes ──
import open3d as o3d

for pts, name in [(pts_sgbm, 'sgbm'), (pts_builtin, 'builtin')]:
    if pts is None or len(pts) < 100:
        print(f'\nSkipping {name}: insufficient points')
        continue

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size=0.0005)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pts_arr = np.asarray(pcd.points)
    dists_xz = np.linalg.norm(pts_arr[:, [0, 2]], axis=1)
    chair = (dists_xz < 0.07) & (pts_arr[:, 1] > -0.01) & (pts_arr[:, 1] < 0.10)
    print(f'\n[{name}] Chair pts: {chair.sum()}/{len(pts_arr)} ({chair.sum()/max(len(pts_arr),1)*100:.1f}%)')

    if chair.sum() < 500:
        print(f'[{name}] WARNING: too few chair points, using all')
        pcd_f = pcd
    else:
        pcd_f = pcd.select_by_index(np.where(chair)[0])

    o3d.io.write_point_cloud(str(OUT / f'v20_{name}_pcd.ply'), pcd_f)

    # Poisson
    pcd_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_f, depth=8, width=0, scale=1.1, linear_fit=False)

    if len(densities) > 0:
        d_arr = np.asarray(densities)
        mesh.remove_vertices_by_mask(d_arr < np.percentile(d_arr, 10))

    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    d_xz = np.linalg.norm(verts[:, [0, 2]], axis=1)
    keep = (d_xz < 0.06) & (verts[:, 1] > 0.0) & (verts[:, 1] < 0.09)
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
    print(f'[{name}] Mesh: {len(vf)}v, {len(mesh_f.triangles)}t')
    print(f'  Size: X={np.ptp(vf[:,0])*1000:.0f} Y={np.ptp(vf[:,1])*1000:.0f} Z={np.ptp(vf[:,2])*1000:.0f} mm')
    o3d.io.write_triangle_mesh(str(OUT / f'v20_{name}_mesh.ply'), mesh_f)
    print(f'  Saved: {OUT}/v20_{name}_mesh.ply')

print('\nDone. Expected chair: 43x88x37mm')
