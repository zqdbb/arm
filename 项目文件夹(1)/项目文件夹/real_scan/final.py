#!/usr/bin/env python3
"""
3D 物体点云采集 + 分区域旋转拼接重建
支持 --name chair / table / cabinet 区分不同物体

用法:
  python gemini_reconstruct.py --name chair    # 采集+重建椅子
  python gemini_reconstruct.py --name table    # 采集+重建桌子
  python gemini_reconstruct.py --name cabinet  # 采集+重建柜子
  python gemini_reconstruct.py --name chair --skip-capture  # 跳过采集, 直接重建
"""
import time
import sys
import argparse
import json
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict

try:
    from turntable import TurntableController
except ImportError:
    TurntableController = None

# ─── 物体配置 (不同物体的裁剪框、输出目录、重建参数) ────────────────
OBJECT_CONFIG = {
    "chair": {
        # 空间裁剪框 (相机坐标系: X左右, Y上下, Z前后/深度)
        "x_min": -0.12, "x_max": 0.12,   # 左右 ±12cm
        "y_min": -0.12, "y_max": 0.12,   # 上下 ±12cm
        "z_min": 0.18,  "z_max": 0.40,   # 深度 18~40cm
        # 采集时的过滤参数
        "y_percentile": 50,   # Y方向: 保留上方50%(靠背+座位), 不删腿
        "x_low": 10, "x_high": 90,  # X方向: 保留中间80%
        "plane_ratio": 0.3,   # 平面占比>30%才删(转台)
        "plane_dist": 0.005,
        # 重建: 分区域 (靠背少帧防重叠, 腿多帧保密度)
        "split_region": True,
        "backrest_frames": 30,
        "leg_frames": 50,
        "backrest_z_thresh": 0.035,
        "backrest_min_consensus": 3,
        "leg_min_consensus": 4,
    },
    "table": {
        "x_min": -0.25, "x_max": 0.25,
        "y_min": -0.25, "y_max": 0.25,
        "z_min": 0.15,  "z_max": 0.50,
        "y_percentile": 45,
        "x_low": 15, "x_high": 90,
        "plane_ratio": 0.3,
        "plane_dist": 0.005,
        "split_region": False,
        "total_frames": 50,
        "min_consensus": 4,
    },
    "cabinet": {
        "x_min": -0.25, "x_max": 0.25,
        "y_min": -0.30, "y_max": 0.30,
        "z_min": 0.15,  "z_max": 0.55,
        "y_percentile": 45,
        "x_low": 10, "x_high": 90,
        "plane_ratio": 0.3,
        "plane_dist": 0.005,
        "split_region": False,
        "total_frames": 72,
        "min_consensus": 3,
        "mirror_fill": True,   # 柜子侧面用Y方向镜像对称补齐
    },
}

# ─── 全局参数 ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# 相机与采集参数
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 3
LASER_POWER = 300

# 重建通用参数
ROTATION_AXIS = np.array([-0.0405, 0.8023, 0.5956])
ROTATION_AXIS = ROTATION_AXIS / np.linalg.norm(ROTATION_AXIS)
ROTATION_CENTER = np.array([0.02921, -0.20124, 0.12441])
VOXEL_SIZE = 0.0015
FINAL_VOXEL = 0.001

# 运行时配置 (由--name填充)
CFG = None
OUT_DIR = None
FINAL_PLY_PATH = None


# ═══════════════════════════════════════════════════════════════════
#  采集部分
# ═══════════════════════════════════════════════════════════════════

def setup_filters():
    filters = []
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    filters.append(spatial)
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    filters.append(temporal)
    hole_filling = rs.hole_filling_filter(1)
    filters.append(hole_filling)
    return filters


def safe_crop_pcd(pcd):
    """根据当前物体配置的裁剪框裁掉背景"""
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(CFG["x_min"], CFG["y_min"], CFG["z_min"]),
        max_bound=(CFG["x_max"], CFG["y_max"], CFG["z_max"])
    )
    return pcd.crop(bbox)


def preview_camera_view(pipe, align, intrinsics):
    print("\n" + "=" * 60)
    print(f"【视角确认 - {CFG['name']}】开启实时画面...")
    print(f"  裁剪框: X[{CFG['x_min']*100:.0f},{CFG['x_max']*100:.0f}] "
          f"Y[{CFG['y_min']*100:.0f},{CFG['y_max']*100:.0f}] "
          f"Z[{CFG['z_min']*100:.0f},{CFG['z_max']*100:.0f}] cm")
    print(" 按【空格键】开始采集，按【Q】退出。")
    print("=" * 60 + "\n")
    colorizer = rs.colorizer()
    while True:
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        cf = aligned.get_color_frame()
        if not df or not cf:
            continue
        color_img = np.asanyarray(cf.get_data())
        depth_frame_colorized = np.asanyarray(colorizer.colorize(df).get_data())
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * 0.001
        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_m)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0,
            depth_trunc=CFG["z_max"], convert_rgb_to_intensity=False
        )
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height,
            intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
        obj_pcd = safe_crop_pcd(pcd)

        # 黄色高亮: 将裁剪后的3D点投影回2D图像
        overlay = color_img.copy()
        obj_pts = np.asarray(obj_pcd.points)
        if len(obj_pts) > 0:
            fx, fy = intrinsics.fx, intrinsics.fy
            ppx, ppy = intrinsics.ppx, intrinsics.ppy
            # 投影到像素坐标
            u = (fx * obj_pts[:, 0] / (obj_pts[:, 2] + 1e-8) + ppx).astype(int)
            v = (fy * obj_pts[:, 1] / (obj_pts[:, 2] + 1e-8) + ppy).astype(int)
            # 只保留图像范围内的点
            valid = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
            u, v = u[valid], v[valid]
            # 黄色叠加 (alpha=0.6)
            overlay[v, u] = (overlay[v, u] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)

        cv2.putText(overlay, f"{CFG['name']} | Points: {len(obj_pcd.points)} | Press [SPACE]",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        combined = np.hstack((overlay, depth_frame_colorized))
        cv2.imshow("Preview", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)


def _write_capture_manifest(path, manifest):
    temp_path = path.with_suffix('.json.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def capture_clean_pcd(save_rgbd=False, turntable_port='/dev/ttyUSB0'):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_root = OUT_DIR.parent
    color_dir = output_root / 'color'
    depth_dir = output_root / 'depth'
    manifest_path = output_root / 'capture_manifest.json'
    if save_rgbd:
        color_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    tt = None
    pipe_started = False
    try:
        profile = pipe.start(cfg)
        pipe_started = True
        ds = profile.get_device().first_depth_sensor()
        if ds.supports(rs.option.laser_power):
            ds.set_option(rs.option.laser_power, LASER_POWER)
        align = rs.align(rs.stream.color)
        filters = setup_filters()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
        )
        preview_camera_view(pipe, align, intr)
        if TurntableController is None:
            raise RuntimeError("未找到 turntable.py")

        tt = TurntableController(port=turntable_port)
        tt.open()
        tt.move_absolute(0, speed=10000)
        tt.wait_stop(timeout=30)

        manifest = None
        if save_rgbd:
            manifest = {
                'schema_version': 1,
                'capture_id': time.strftime('%Y%m%d_%H%M%S'),
                'name': CFG['name'],
                'complete': False,
                'step_deg': STEP_DEG,
                'planned_frames': N_FRAMES,
                'depth_average_count': N_AVG,
                'depth_aligned_to': 'color',
                'depth_scale_m_per_unit': ds.get_depth_scale(),
                'color': {'width': intr.width, 'height': intr.height, 'format': 'png'},
                'depth': {'width': intr.width, 'height': intr.height, 'format': 'uint16_png'},
                'intrinsics': {
                    'fx': intr.fx,
                    'fy': intr.fy,
                    'ppx': intr.ppx,
                    'ppy': intr.ppy,
                    'distortion_model': str(intr.model),
                    'coeffs': list(intr.coeffs),
                },
                'stitched_coordinates': 'X=width,Y=depth,Z=up,meters',
                'frames': [],
            }
            _write_capture_manifest(manifest_path, manifest)

        t0 = time.time()
        print(f"\n开始采集 {CFG['name']}: {N_FRAMES} 帧点云...")
        for i in range(N_FRAMES):
            deg = i * STEP_DEG
            tt.move_absolute(deg, speed=5000)
            tt.wait_stop(timeout=30)
            depth_frames_list = []
            color_img = None
            for _ in range(N_AVG):
                frames = pipe.wait_for_frames(timeout_ms=5000)
                aligned = align.process(frames)
                df = aligned.get_depth_frame()
                cf = aligned.get_color_frame()
                if not df or not cf:
                    continue
                for f in filters:
                    df = f.process(df)
                depth_frames_list.append(np.asanyarray(df.get_data()).astype(np.float32))
                if color_img is None:
                    color_img = np.asanyarray(cf.get_data())
            if not depth_frames_list or color_img is None:
                raise RuntimeError(f"帧 {i:03d} 未获得完整 RGB-D 数据")

            depth_stack = np.stack(depth_frames_list, axis=0)
            valid_mask = depth_stack > 0
            valid_counts = np.sum(valid_mask, axis=0)
            depth_sum = np.sum(depth_stack, axis=0)
            mean_depth = np.zeros_like(depth_sum)
            mean_depth[valid_counts > 0] = depth_sum[valid_counts > 0] / valid_counts[valid_counts > 0]
            depth_m = (mean_depth * ds.get_depth_scale()).astype(np.float32)
            o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            o3d_depth = o3d.geometry.Image(depth_m)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, depth_scale=1.0,
                depth_trunc=CFG["z_max"], convert_rgb_to_intensity=False
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
            pcd = safe_crop_pcd(pcd)

            if len(pcd.points) > 20:
                pcd = pcd.voxel_down_sample(voxel_size=0.002)
                labels = np.array(pcd.cluster_dbscan(eps=0.015, min_points=10))
                if len(labels) > 0:
                    valid = labels[labels != -1]
                    if len(valid) > 0:
                        unique, counts = np.unique(valid, return_counts=True)
                        largest = unique[np.argmax(counts)]
                        pcd = pcd.select_by_index(np.where(labels == largest)[0])

                if len(pcd.points) > 30:
                    pts = np.asarray(pcd.points)
                    _, inliers = pcd.segment_plane(
                        distance_threshold=CFG["plane_dist"], ransac_n=3, num_iterations=100)
                    if len(inliers) / len(pts) > CFG["plane_ratio"]:
                        pcd = pcd.select_by_index(inliers, invert=True)

                if len(pcd.points) > 20 and CFG["y_percentile"] < 100:
                    pts = np.asarray(pcd.points)
                    y_thresh = np.percentile(pts[:, 1], CFG["y_percentile"])
                    keep = pts[:, 1] < y_thresh
                    pcd = pcd.select_by_index(np.where(keep)[0])

                if len(pcd.points) > 20:
                    pts = np.asarray(pcd.points)
                    x_low = np.percentile(pts[:, 0], CFG["x_low"])
                    x_high = np.percentile(pts[:, 0], CFG["x_high"])
                    keep = (pts[:, 0] > x_low) & (pts[:, 0] < x_high)
                    pcd = pcd.select_by_index(np.where(keep)[0])

            save_path = OUT_DIR / f"frame_{i:03d}.ply"
            if not o3d.io.write_point_cloud(str(save_path), pcd):
                raise IOError(f"点云写入失败: {save_path}")

            if save_rgbd:
                color_path = color_dir / f'{i:03d}.png'
                depth_path = depth_dir / f'{i:03d}.png'
                depth_u16 = np.clip(
                    np.rint(mean_depth), 0, np.iinfo(np.uint16).max
                ).astype(np.uint16)
                if not cv2.imwrite(str(color_path), color_img):
                    raise IOError(f"彩色图写入失败: {color_path}")
                if not cv2.imwrite(str(depth_path), depth_u16):
                    raise IOError(f"深度图写入失败: {depth_path}")
                manifest['frames'].append({
                    'index': i,
                    'angle_deg': deg,
                    'color': str(color_path.relative_to(output_root)),
                    'depth': str(depth_path.relative_to(output_root)),
                    'point_cloud': str(save_path.relative_to(output_root)),
                })
                _write_capture_manifest(manifest_path, manifest)

            print(f"  [{i+1:02d}/{N_FRAMES}] {deg:3d}° | {len(pcd.points):5d}点 -> {save_path.name}")

        if manifest is not None:
            manifest['complete'] = True
            manifest['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            _write_capture_manifest(manifest_path, manifest)
        print(f"\n采集完毕，耗时 {time.time()-t0:.1f} 秒！")
    finally:
        if tt is not None:
            try:
                tt.close()
            finally:
                if pipe_started:
                    pipe.stop()
        elif pipe_started:
            pipe.stop()


# ═══════════════════════════════════════════════════════════════════
#  重建部分
# ═══════════════════════════════════════════════════════════════════

def voxel_down(pts, vs):
    keys = np.floor(pts / vs).astype(np.int64)
    vmap = defaultdict(list)
    for i, k in enumerate(keys):
        vmap[tuple(k)].append(i)
    op = np.zeros((len(vmap), 3))
    for i, (k, idx) in enumerate(vmap.items()):
        op[i] = pts[idx].mean(0)
    return op


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_align_matrix(axis):
    axis = axis / np.linalg.norm(axis)
    z_axis = np.array([0, 0, 1])
    v = np.cross(axis, z_axis)
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - axis[2]) / (s * s)


def auto_detect_rotation_center(frames_data, R_align, z_center=0.12):
    """
    自动估计旋转中心: 用帧0和帧36(相差180°)在XY平面搜索最优中心
    返回 np.array([cx, cy, z_center])
    """
    if 0 not in frames_data or 36 not in frames_data:
        # 没有帧36就用帧18(90°), 精度稍差
        ref_idx, rot_idx = (0, 18) if 18 in frames_data else (0, 0)
        if ref_idx == rot_idx:
            print("  警告: 无法自动估计旋转中心(缺少对比帧), 使用默认值")
            return ROTATION_CENTER.copy()
    else:
        ref_idx, rot_idx = 0, 36

    p_ref = frames_data[ref_idx] @ R_align.T
    p_rot = frames_data[rot_idx] @ R_align.T
    angle_diff = (rot_idx - ref_idx) * STEP_DEG  # 180°或90°

    # 粗搜索: 步长5mm
    best_score = float('inf')
    best_cx, best_cy = 0, -0.2
    for cx in np.arange(-0.06, 0.06, 0.005):
        for cy in np.arange(-0.32, -0.10, 0.005):
            center = np.array([cx, cy, z_center])
            R = rot_z(np.radians(-angle_diff))
            p_rot_aligned = (p_rot - center) @ R.T + center
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(p_ref[:, :2])
            dist, _ = nbrs.kneighbors(p_rot_aligned[:, :2])
            score = np.median(dist)
            if score < best_score:
                best_score = score
                best_cx, best_cy = cx, cy

    # 精搜索: 步长1mm
    for cx in np.arange(best_cx - 0.005, best_cx + 0.006, 0.001):
        for cy in np.arange(best_cy - 0.005, best_cy + 0.006, 0.001):
            center = np.array([cx, cy, z_center])
            R = rot_z(np.radians(-angle_diff))
            p_rot_aligned = (p_rot - center) @ R.T + center
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(p_ref[:, :2])
            dist, _ = nbrs.kneighbors(p_rot_aligned[:, :2])
            score = np.median(dist)
            if score < best_score:
                best_score = score
                best_cx, best_cy = cx, cy

    center = np.array([best_cx, best_cy, z_center])
    print(f"  自动估计旋转中心: [{best_cx*100:.1f}, {best_cy*100:.1f}, {z_center*100:.1f}]cm, "
          f"帧{ref_idx}vs帧{rot_idx}中位距离={best_score*1000:.1f}mm")
    return center


def read_frame_ply(filepath):
    pcd = o3d.io.read_point_cloud(str(filepath))
    return np.asarray(pcd.points)


def dbscan_clean(pts, eps=0.01, min_samples=10):
    if len(pts) < min_samples:
        return pts
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique) == 0:
        return pts
    return pts[labels == unique[np.argmax(counts)]]


def consensus_filter(frame_indices, frames_data, R_align, center, min_consensus):
    voxel_counts = Counter()
    for i in frame_indices:
        if i not in frames_data:
            continue
        p_aligned = frames_data[i] @ R_align.T
        R = rot_z(np.radians(i * STEP_DEG))
        tp = (p_aligned - center) @ R.T + center
        keys = np.floor(tp / VOXEL_SIZE).astype(np.int64)
        for k in keys:
            voxel_counts[tuple(k)] += 1
    selected = [k for k, v in voxel_counts.items() if v >= min_consensus]
    if not selected:
        return np.empty((0, 3), dtype=float)
    return np.array([[
        (k[0] + 0.5) * VOXEL_SIZE,
        (k[1] + 0.5) * VOXEL_SIZE,
        (k[2] + 0.5) * VOXEL_SIZE
    ] for k in selected])


def stitch_point_clouds(show_visualization=True):
    ply_files = sorted(list(OUT_DIR.glob("frame_*.ply")))
    if not ply_files:
        print(f"\n错误: 未找到 PLY 数据！目录: {OUT_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"【重建 - {CFG['name']}】读取到 {len(ply_files)} 帧")
    if CFG["split_region"]:
        print(f"  分区域: 靠背前{CFG['backrest_frames']}帧>={CFG['backrest_min_consensus']}, "
              f"腿前{CFG['leg_frames']}帧>={CFG['leg_min_consensus']}")
    else:
        print(f"  统一拼接: 前{CFG['total_frames']}帧, >={CFG['min_consensus']}帧一致")
    print(f"{'='*60}")

    # 读取所有帧 (先下采样减少点数, 再DBSCAN去噪, 避免16万点卡死)
    frames_data = {}
    for fp in ply_files:
        idx = int(fp.stem.split('_')[1])
        p = read_frame_ply(fp)
        if len(p) < 50:
            continue
        # 采集时已做过背景剔除, 这里只做体素下采样, 不再DBSCAN(避免拆碎椅子)
        p = voxel_down(p, VOXEL_SIZE)
        frames_data[idx] = p
        if idx % 10 == 0:
            print(f"  已处理帧 {idx}/{len(ply_files)}, 当前帧 {len(p)} 点")
    print(f"有效帧: {len(frames_data)}")
    if not frames_data:
        raise RuntimeError("没有可用于拼接的有效点云帧")

    R_align = build_align_matrix(ROTATION_AXIS)
    # 自动估计旋转中心 (用帧0和帧36在XY平面搜索)
    center = auto_detect_rotation_center(frames_data, R_align, z_center=ROTATION_CENTER[2])

    # 异常帧检测
    abnormal = []
    for i in sorted(frames_data.keys()):
        p_aligned = frames_data[i] @ R_align.T
        R = rot_z(np.radians(i * STEP_DEG))
        tp = (p_aligned - center) @ R.T + center
        if tp[:, 2].max() - tp[:, 2].min() < 0.02:
            abnormal.append(i)
    if abnormal:
        print(f"异常帧(已排除): {abnormal}")

    if CFG["split_region"]:
        # ── 椅子: 分区域 ──
        back_idx = [i for i in range(CFG["backrest_frames"])
                    if i in frames_data and i not in abnormal]
        leg_idx = [i for i in range(CFG["leg_frames"])
                   if i in frames_data and i not in abnormal]

        back_pts = consensus_filter(back_idx, frames_data, R_align, center,
                                    CFG["backrest_min_consensus"])
        back_pts = dbscan_clean(back_pts, eps=0.006, min_samples=5)

        leg_pts = consensus_filter(leg_idx, frames_data, R_align, center,
                                   CFG["leg_min_consensus"])
        leg_pts = dbscan_clean(leg_pts, eps=0.006, min_samples=5)
        if len(back_pts) == 0 or len(leg_pts) == 0:
            raise RuntimeError(
                f"点云共识不足: 靠背 {len(back_pts)} 点, 腿 {len(leg_pts)} 点"
            )

        # 中心化
        back_c = back_pts - back_pts.mean(0)
        back_c[:, 2] -= back_c[:, 2].min()
        leg_c = leg_pts - leg_pts.mean(0)
        leg_c[:, 2] -= leg_c[:, 2].min()

        # 分割
        z_th = CFG["backrest_z_thresh"]
        back_part = back_c[back_c[:, 2] < z_th]
        leg_part = leg_c[leg_c[:, 2] >= z_th]

        # XY对齐
        ob = back_c[(back_c[:, 2] >= 0.03) & (back_c[:, 2] <= 0.04)]
        ol = leg_c[(leg_c[:, 2] >= 0.03) & (leg_c[:, 2] <= 0.04)]
        if len(ob) > 0 and len(ol) > 0:
            xy_off = ob[:, :2].mean(0) - ol[:, :2].mean(0)
            leg_part = leg_part.copy()
            leg_part[:, :2] += xy_off

        combined = np.vstack([back_part, leg_part])
        if len(combined) == 0:
            raise RuntimeError("椅子分区后无有效点，请重新采集")
        print(f"靠背: {len(back_part)}点, 腿: {len(leg_part)}点")
    else:
        # ── 桌子/柜子: 统一全帧 ──
        all_idx = [i for i in range(CFG["total_frames"])
                   if i in frames_data and i not in abnormal]
        combined = consensus_filter(all_idx, frames_data, R_align, center,
                                    CFG["min_consensus"])
        combined = dbscan_clean(combined, eps=0.006, min_samples=5)
        if len(combined) == 0:
            raise RuntimeError("点云共识过滤后无有效点，请重新采集")

        # 镜像对称补齐侧面 (柜子用, 长方体左右对称)
        if CFG.get("mirror_fill", False):
            mirror = combined.copy()
            mirror[:, 1] = -mirror[:, 1]  # Y方向镜像
            combined = np.vstack([combined, mirror])
            combined = voxel_down(combined, VOXEL_SIZE)  # 去重
            combined = dbscan_clean(combined, eps=0.008, min_samples=5)
            print(f"  镜像补齐侧面后: {len(combined)}点")

        combined = combined - combined.mean(0)
        combined[:, 2] -= combined[:, 2].min()

    # 最终下采样 + 中心化
    combined = voxel_down(combined, FINAL_VOXEL)
    final = combined - combined.mean(0)
    final[:, 2] -= final[:, 2].min()

    dims = (final.max(0) - final.min(0)) * 100
    print(f"\n最终: {len(final)}点, 尺寸 {dims.round(1)}cm")

    # 保存 (带颜色)
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(final)
    colors = np.zeros((len(final), 3))
    z_norm = (final[:, 2] - final[:, 2].min()) / (final[:, 2].max() - final[:, 2].min() + 1e-8)
    colors[:, 0] = z_norm
    colors[:, 2] = 1 - z_norm
    colors[:, 1] = 0.3
    pcd_final.colors = o3d.utility.Vector3dVector(colors)

    FINAL_PLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(FINAL_PLY_PATH), pcd_final)
    print(f"\n保存成功: {FINAL_PLY_PATH}")

    if show_visualization:
        try:
            o3d.visualization.draw_geometries(
                [pcd_final], window_name=f"3D {CFG['name']} Point Cloud",
                width=1024, height=768
            )
        except Exception as e:
            print(f"可视化跳过: {e}")


# ═══════════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════════

def main():
    global CFG, OUT_DIR, FINAL_PLY_PATH

    parser = argparse.ArgumentParser(description="3D物体点云采集与重建")
    parser.add_argument("--name", required=True, choices=["chair", "table", "cabinet"],
                        help="物体名称: chair / table / cabinet")
    parser.add_argument("--skip-capture", action="store_true",
                        help="跳过采集, 直接用已有数据重建")
    parser.add_argument("--save-rgbd", action="store_true",
                        help="同步保存对齐的彩色图、深度图和采集清单")
    parser.add_argument("--turntable-port", default="/dev/ttyUSB0",
                        help="转台串口 (默认: /dev/ttyUSB0)")
    parser.add_argument("--no-visualize", action="store_true",
                        help="保存结果后不打开 Open3D 窗口")
    args = parser.parse_args()

    # 加载物体配置
    CFG = OBJECT_CONFIG[args.name].copy()
    CFG["name"] = args.name
    OUT_DIR = BASE_DIR / f"output/{args.name}/pcd_frames"
    FINAL_PLY_PATH = BASE_DIR / f"output/{args.name}/{args.name}_stitched.ply"

    print(f"\n物体: {args.name}")
    print(f"输出目录: {OUT_DIR}")
    print(f"裁剪框: X[{CFG['x_min']*100:.0f},{CFG['x_max']*100:.0f}] "
          f"Y[{CFG['y_min']*100:.0f},{CFG['y_max']*100:.0f}] "
          f"Z[{CFG['z_min']*100:.0f},{CFG['z_max']*100:.0f}] cm")

    if not args.skip_capture:
        capture_clean_pcd(save_rgbd=args.save_rgbd,
                          turntable_port=args.turntable_port)
    stitch_point_clouds(show_visualization=not args.no_visualize)


if __name__ == "__main__":
    main()
