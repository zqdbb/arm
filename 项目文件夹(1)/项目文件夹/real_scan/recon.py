#!/usr/bin/env python3
"""Camera-adaptive 重建脚本 — 已知转台位姿 + 多帧深度融合 + Poisson mesh."""
import cv2, numpy as np, json, time, sys
from pathlib import Path
import open3d as o3d

BASE = Path(__file__).parent

# ── 配置 ──
CAPTURE_DIR = BASE / 'output/capture'
OUT_DIR = BASE / 'output/recon'
VOXEL_SIZE = 0.001       # 1mm 体素下采样
CHAIR_RADIUS = 0.06      # 椅子裁剪半径 (m)
CHAIR_Z_MIN = -0.01       # 椅子底部 (m)
CHAIR_Z_MAX = 0.10        # 椅子顶部 (m)
USE_YOLO = True           # YOLO mask 预过滤（去掉转台背景）


def rodrigues(axis, angle):
    """Rodrigues rotation formula: axis must be unit vector."""
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def main():
    # ── 加载元数据 ──
    meta_path = CAPTURE_DIR / 'meta.json'
    if not meta_path.exists():
        print(f'错误: 未找到 {meta_path}，请先运行 capture.py')
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    fx = meta['fx']; fy = meta['fy']
    ppx = meta['ppx']; ppy = meta['ppy']
    W, H = meta['wxH_zoomed']
    depth_scale = meta['depth_scale']
    n_frames = meta['n_frames']
    step_deg = meta['step_deg']
    calib = meta.get('calib', {})

    print(f'相机: {meta["camera"]}')
    print(f'分辨率: {W}x{H}  fx={fx:.1f} fy={fy:.1f} pp=({ppx:.1f},{ppy:.1f})')
    print(f'深度 scale: {depth_scale}')
    print(f'帧数: {n_frames}  步进: {step_deg}°')

    if not calib:
        print('错误: meta.json 中无标定数据')
        sys.exit(1)

    # ── 转轴参数 ──
    # 优先从72帧质心拟合结果加载，更准确
    fit_path = BASE / 'output/axis_fit.json'
    if fit_path.exists():
        with open(fit_path) as f:
            fit = json.load(f)
        axis_dir = np.array(fit['axis_dir'])
        axis_point = np.array(fit['axis_point'])
        print(f'使用拟合转轴 (axis_fit.json, {fit["n_frames"]}帧)')
    else:
        R_cal = np.array(calib['R'])
        t_cal = np.array(calib['t'])
        axis_dir = R_cal[2, :].copy()  # 第三行 = 平面法向
        axis_dir = axis_dir / np.linalg.norm(axis_dir)
        axis_point = t_cal
        print(f'使用标定转轴 (fallback, 建议先跑 fit_axis.py)')

    print(f'转轴方向: [{axis_dir[0]:.3f}, {axis_dir[1]:.3f}, {axis_dir[2]:.3f}]')
    print(f'转轴点: [{axis_point[0]*1000:.0f}, {axis_point[1]*1000:.0f}, {axis_point[2]*1000:.0f}] mm')

    # ── YOLO ──
    if USE_YOLO:
        from ultralytics import YOLO
        yolo = YOLO(str(BASE / 'yolov8n-seg.pt'))
        print('YOLO loaded')

    # ── 预计算射线 ──
    u = np.arange(W); v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    ray_x = (uu - ppx) / fx
    ray_y = (vv - ppy) / fy

    # ── 逐帧融合 ──
    all_pts = []
    color_files = sorted((CAPTURE_DIR / 'color').glob('*.jpg'))
    depth_files = sorted((CAPTURE_DIR / 'depth').glob('*.png'))

    if len(depth_files) < n_frames:
        print(f'警告: 深度文件不足 ({len(depth_files)}/{n_frames})')

    t0 = time.time()
    print(f'\n{"=" * 50}')
    print('逐帧反投影 + 旋转对齐...')
    print('=' * 50)

    for i, df in enumerate(depth_files[:n_frames]):
        # 加载深度 (毫米存储)
        depth_mm = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth_m = depth_mm.astype(np.float32) * 0.001  # mm → m
        valid = depth_m > 0.01  # 排除零点

        # YOLO mask
        if USE_YOLO and i < len(color_files):
            img = cv2.imread(str(color_files[i]))
            if img is not None:
                results = yolo(img, verbose=False, classes=[56])
                if results[0].masks is not None:
                    m = results[0].masks.data[0].cpu().numpy()
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    valid = valid & (m > 0.5)

        if valid.sum() < 100:
            continue

        # 深度范围过滤：YOLO 2D mask 内包含背景深度，只保留椅子深度附近
        d_median = np.median(depth_m[valid])
        DEPTH_RANGE = 0.07  # ±70mm，覆盖椅子深度+旋转偏移
        valid = valid & (depth_m > d_median - DEPTH_RANGE) & (depth_m < d_median + DEPTH_RANGE)

        # 反投影
        z = depth_m[valid]
        x = ray_x[valid] * z
        y = -ray_y[valid] * z  # Y flip: world up
        pts_cam = np.stack([x, y, z], axis=1)

        # 去旋转
        theta = np.radians(i * step_deg)
        R_undo = rodrigues(axis_dir, -theta)
        pts_centered = pts_cam - axis_point
        pts_rotated = (R_undo @ pts_centered.T).T + axis_point
        pts_world = pts_rotated - axis_point  # 原点移到转轴中心

        all_pts.append(pts_world.astype(np.float32))

        if (i + 1) % 18 == 0:
            n = sum(len(p) for p in all_pts)
            print(f'  [{i+1:3d}/{n_frames}]  {n:,} pts  {time.time()-t0:.0f}s')

    if not all_pts:
        print('错误: 无有效点')
        sys.exit(1)

    pts_all = np.vstack(all_pts)
    elapsed = time.time() - t0
    print(f'\n融合完成: {len(pts_all):,} pts / {elapsed:.0f}s')
    print(f'  X: [{pts_all[:,0].min()*1000:.0f}, {pts_all[:,0].max()*1000:.0f}] mm')
    print(f'  Y: [{pts_all[:,1].min()*1000:.0f}, {pts_all[:,1].max()*1000:.0f}] mm')
    print(f'  Z: [{pts_all[:,2].min()*1000:.0f}, {pts_all[:,2].max()*1000:.0f}] mm')

    # ── 点云处理 ──
    print(f'\n{"=" * 50}')
    print('点云滤波 + Poisson 重建...')
    print('=' * 50)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)

    # 体素下采样
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f'体素下采样 ({VOXEL_SIZE*1000:.1f}mm): {len(pcd.points):,} pts')

    # 统计离群点去除
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    print(f'去离群点: {len(pcd.points):,} pts')

    # 保存原始融合点云（调试用）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(OUT_DIR / 'fused_raw.ply'), pcd)
    print(f'原始融合点云 → {OUT_DIR}/fused_raw.ply')

    # RANSAC 找转台平面 → 删除
    pts_arr = np.asarray(pcd.points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.003,
                                              ransac_n=3, num_iterations=500)
    n_plane = len(inliers)
    print(f'转台平面 inliers: {n_plane:,} / {len(pts_arr):,} '
          f'({100*n_plane/len(pts_arr):.0f}%)')
    pcd = pcd.select_by_index(inliers, invert=True)
    print(f'去平面后: {len(pcd.points):,} pts')

    if len(pcd.points) < 100:
        print('错误: 去平面后点数不足，检查标定或数据')
        sys.exit(1)

    # DBSCAN → 取最大簇（椅子）
    labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=10, print_progress=True))
    n_clusters = labels.max() + 1
    print(f'DBSCAN: {n_clusters} 簇')
    if n_clusters > 0:
        best_label = max(range(n_clusters), key=lambda lb: (labels == lb).sum())
        chair_idx = np.where(labels == best_label)[0]
        pcd_f = pcd.select_by_index(chair_idx)
        print(f'  椅子簇: {len(chair_idx):,} pts')
    else:
        pcd_f = pcd

    # 保存过滤后点云
    o3d.io.write_point_cloud(str(OUT_DIR / 'fused_filtered.ply'), pcd_f)
    v = np.asarray(pcd_f.points)
    print(f'过滤后: {len(v):,} pts')
    print(f'  包围盒: X={np.ptp(v[:,0])*1000:.0f}mm  '
          f'Y={np.ptp(v[:,1])*1000:.0f}mm  '
          f'Z={np.ptp(v[:,2])*1000:.0f}mm')

    # 法向量估计
    pcd_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    pcd_f.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

    # Poisson 重建
    print('Poisson 重建...')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_f, depth=9, width=0, scale=1.05, linear_fit=False)

    # 密度过滤 (去掉低密度 outlier 面片)
    d_arr = np.asarray(densities)
    mesh.remove_vertices_by_mask(d_arr < np.percentile(d_arr, 5))
    mesh.compute_vertex_normals()
    print(f'Poisson mesh: {len(np.asarray(mesh.vertices)):,} verts, {len(np.asarray(mesh.triangles)):,} tris')

    # ── 后处理 ──
    # 移除小连通分量
    labels, counts, _ = mesh.cluster_connected_triangles()
    if len(counts) > 1:
        mesh.remove_triangles_by_index(np.where(labels != np.argmax(counts))[0])
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    d_xz_f = np.linalg.norm(verts[:, [0, 2]], axis=1)
    keep = (d_xz_f < CHAIR_RADIUS) & (verts[:, 1] > CHAIR_Z_MIN) & (verts[:, 1] < CHAIR_Z_MAX)
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
    print(f'\n最终 mesh: {len(vf):,} verts, {len(np.asarray(mesh_f.triangles)):,} tris')
    print(f'尺寸: X={np.ptp(vf[:,0])*1000:.0f}mm  Y={np.ptp(vf[:,1])*1000:.0f}mm  Z={np.ptp(vf[:,2])*1000:.0f}mm')

    # ── 保存 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(OUT_DIR / 'chair.ply'), mesh_f)
    o3d.io.write_triangle_mesh(str(OUT_DIR / 'chair.obj'), mesh_f)
    o3d.io.write_point_cloud(str(OUT_DIR / 'fused.ply'), pcd_f)
    print(f'\n已保存: {OUT_DIR}/chair.ply, chair.obj, fused.ply')


if __name__ == '__main__':
    main()
