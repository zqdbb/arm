#!/usr/bin/env python3
"""Open3D RGBD 重建管线: D435i 硬件深度 + RGBD Odometry + TSDF 融合.

基于 Open3D Reconstruction System, 适配电动转台.

输出: triangle mesh (.ply + .obj), 非点云. 后续可自行采样转点云.

用法:
  python3 open3d_scan.py                    # 完整流程: 放置确认 → 零点标定 → 旋转轴标定 → 采集 + 重建
  python3 open3d_scan.py --skip-capture     # 跳过采集, 仅重建
  python3 open3d_scan.py --skip-zero        # 跳过零点标定
  python3 open3d_scan.py --skip-calib       # 跳过旋转轴标定 (使用已有 calibrate.json)

旋转轴标定: 放家具 → 拍两帧 (间隔30°) → ORB特征匹配 → 提取转轴方向 + 转台面交点 → 保存 calibrate.json

输出: output/open3d_scan/scene/integrated_calib_mesh.ply (mesh)
      output/open3d_scan/scene/integrated_calib_mesh.obj (mesh)

对比:
  real_scan.py   — DA-V2 单目深度 (尺度不一致)
  colmap_scan.py — COLMAP MVS (CPU 慢, 需去背景)
  open3d_scan.py — D435i 硬件深度 + TSDF (尺度正确, 速度快)
"""

import os, sys, time, json, argparse, shutil
from math import tan, radians
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from turntable import TurntableController

# Open3D 重建模块路径
_RECON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'open3d_recon')
sys.path.insert(0, _RECON_DIR)

# ── 输出目录 ──
SCAN_DIR = os.path.join(OUTPUT_DIR, 'open3d_scan')
COLOR_DIR = os.path.join(SCAN_DIR, 'color')
DEPTH_DIR = os.path.join(SCAN_DIR, 'depth')
SCENE_DIR = os.path.join(SCAN_DIR, 'scene')
INTRINSIC_PATH = os.path.join(SCAN_DIR, 'camera_intrinsic.json')
FINAL_PLY = os.path.join(SCENE_DIR, 'integrated.ply')

# ── 采集参数 ──
STEP_ANGLE = 10            # 步进角度
DEFAULT_DISTANCE_CM = 45.0 # 默认物体中心距离 (cm)
DEFAULT_RADIUS_CM = 10.0   # 默认物体半径 (cm)
DEFAULT_ZOOM = 1.0         # 默认数码变焦倍数 (1.0=不裁切, 先看全貌)

# ── 重建参数 ──
VOXEL_SIZE = 0.005         # 体素大小 5mm (越小越精细, 越大越快)
MAX_DEPTH = 2.0            # 深度截断 2m
MAX_DEPTH_DIFF = 0.03      # RGBD odometry 深度差阈值
TSDF_CUBIC_SIZE = 2.0      # TSDF 体素网格尺寸
ICCP_METHOD = 'color'      # ICP 方法: color / point_to_plane / point_to_point
N_FRAMES_PER_FRAGMENT = 100  # 每 fragment 帧数


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def draw_center_hud(preview, depth_data, color_w, color_h):
    """在预览图上叠加: 中心十字 + 深度距离读数 (用于判断是否在盲区)."""
    cx, cy = color_w // 2, color_h // 2
    cv2.line(preview, (cx, cy - 20), (cx, cy + 20), (0, 0, 255), 1)
    cv2.line(preview, (cx - 20, cy), (cx + 20, cy), (0, 0, 255), 1)
    # depth_data 是 uint16 数组 (单位 mm), 取中心3x3中值
    if depth_data is not None and depth_data.size > 0:
        roi = depth_data[max(cy-1,0):cy+2, max(cx-1,0):cx+2]
        valid = roi[roi > 0]
        if len(valid) > 0:
            dist_m = np.median(valid) / 1000.0
            cv2.putText(preview, f'Center: {dist_m:.3f}m', (cx + 25, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def save_intrinsic(filename, fx, fy, ppx, ppy, width, height):
    """保存相机内参 JSON (支持变焦修正)."""
    with open(filename, 'w') as f:
        json.dump({
            'width': width,
            'height': height,
            'intrinsic_matrix': [fx, 0, 0, 0, fy, 0, ppx, ppy, 1],
        }, f, indent=2)


def compute_disparity_shift(distance_cm, radius_cm, width=640, h_fov=87, baseline_mm=50):
    """动态计算 D435i disparity shift, 优化工作距离处的深度精度.

    公式: shift = (focal_length * baseline) / max_range / 10
    """
    focal_length = 0.5 * (width / tan(radians(h_fov / 2)))
    max_range_mm = (distance_cm + radius_cm) * 10  # cm → mm
    shift = int((focal_length * baseline_mm) / max_range_mm)
    return max(shift, 1)


def apply_zoom(color_img, depth_img, zoom, intrinsics):
    """数码变焦: 裁切中心 → 缩放回原分辨率 → 修正内参.

    Returns: (color_zoomed, depth_zoomed, fx, fy, ppx, ppy)
    """
    if zoom <= 1.0:
        return color_img, depth_img, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

    h, w = color_img.shape[:2]
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2

    color_crop = color_img[y0:y0 + crop_h, x0:x0 + crop_w]
    depth_crop = depth_img[y0:y0 + crop_h, x0:x0 + crop_w]

    color_zoom = cv2.resize(color_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    depth_zoom = cv2.resize(depth_crop, (w, h), interpolation=cv2.INTER_NEAREST)

    fx = intrinsics.fx * zoom
    fy = intrinsics.fy * zoom
    ppx = (intrinsics.ppx - x0) * zoom
    ppy = (intrinsics.ppy - y0) * zoom

    return color_zoom, depth_zoom, fx, fy, ppx, ppy


def setup_workspace():
    """初始化输出目录 (不删除已有数据, 仅确保目录存在)."""
    ensure_dir(COLOR_DIR)
    ensure_dir(DEPTH_DIR)
    ensure_dir(SCENE_DIR)


# ══════════════════════════════════════════════════════════════════
# Phase 1: RGBD 采集
# ══════════════════════════════════════════════════════════════════

N_ACCUM_FRAMES = 3           # 每个角度累积帧数 (时域中值滤波填空洞)

def capture_rgbd_frames(pipeline, align, tt, distance_cm, radius_cm, zoom=1.0,
                        diagnose=False, skip_hole_fill=False):
    """步进拍照: 转台每步停 → 多帧深度累积 → 变焦 → 保存."""
    import pyrealsense2 as rs

    angles = list(range(0, 360, STEP_ANGLE))
    print(f'\n[采集] {len(angles)} 帧, 每{STEP_ANGLE}°, '
          f'距离={distance_cm}cm, 半径={radius_cm}cm, '
          f'深度累积={N_ACCUM_FRAMES}帧/位置')
    if skip_hole_fill:
        print(f'  ⚠ 填孔已禁用 (保留原始空洞用于诊断)')

    cv2.namedWindow('Open3D Scan Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Open3D Scan Capture', 960, 360)

    # 背景裁切距离: 物体中心距离 + 半径 + 50cm 余量 (避免误伤)
    # D435i 盲区: 最小有效距离 ~280mm, 小于此值的深度不可靠
    clip_mm = int((distance_cm + radius_cm + 50) * 10)
    min_valid_mm = 280
    print(f'  深度过滤: <{min_valid_mm}mm 或 >{clip_mm}mm 清零')

    # RealSense 深度后处理滤镜链
    spat_filter = rs.spatial_filter()
    spat_filter.set_option(rs.option.filter_magnitude, 2)
    spat_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat_filter.set_option(rs.option.filter_smooth_delta, 20)
    temp_filter = rs.temporal_filter()
    temp_filter.set_option(rs.option.filter_smooth_alpha, 0.4)   # 时域平滑权重
    temp_filter.set_option(rs.option.filter_smooth_delta, 20)
    hole_filter = rs.hole_filling_filter()

    saved = 0
    total_start = time.time()

    for i, target_deg in enumerate(angles):
        print(f'  [{i+1}/{len(angles)}] → {target_deg}°', end='')
        tt.move_absolute(target_deg, speed=TURNTABLE_DEFAULT_SPEED)
        tt.wait_stop(timeout=30)
        time.sleep(0.5)  # 防抖

        # ── 多帧深度累积: 每位置采集 N 帧 → 空间+时域滤波 → 中值累积 ──
        depth_accum = []
        color_final = None

        for j in range(N_ACCUM_FRAMES):
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 滤镜链: 空间平滑 → 时域去抖 → 填孔
            depth_frame = spat_filter.process(depth_frame)
            if j > 0 and len(depth_accum) > 0:
                depth_frame = temp_filter.process(depth_frame)
            if not skip_hole_fill:
                depth_frame = hole_filter.process(depth_frame)

            depth_img = np.asanyarray(depth_frame.get_data()).astype(np.float32)

            # 诊断: 保存第一帧原始深度 (填孔前)
            if diagnose and i == 0 and j == 0:
                raw_depth = depth_img.copy()
                print(f'  [诊断] 原始深度空洞: {(raw_depth==0).sum()} 像素')

            depth_accum.append(depth_img)

            if j == 0:
                color_final = np.asanyarray(color_frame.get_data())

            if j < N_ACCUM_FRAMES - 1:
                time.sleep(0.02)

        if len(depth_accum) == 0:
            print('  ✗ 采集失败')
            continue

        # 时域中值: 消除随机空洞 (某一帧的0值被另一帧的有效值替代)
        depth_stack = np.stack(depth_accum, axis=0)
        depth_median = np.median(depth_stack, axis=0)
        # 用中值后的有效值替换原帧的空洞
        n_zeros_before = (depth_accum[0] == 0).sum()
        n_zeros_after = (depth_median == 0).sum()
        n_filled = n_zeros_before - n_zeros_after
        if n_filled > 500:
            print(f'  [+{n_filled}空洞填充]', end='')
        print()

        depth_img = depth_median.astype(np.uint16)

        # 深度过滤: 盲区 (<280mm) + 远背景 (>clip_mm) 清零
        # RGB 只涂黑远背景, 盲区保留彩色画面 (方便观察家具位置)
        far_mask = (depth_img > clip_mm)
        blind_mask = (depth_img < min_valid_mm) & (depth_img > 0)
        depth_img[far_mask] = 0
        depth_img[blind_mask] = 0
        color_final[far_mask] = [0, 0, 0]

        # 诊断: 保存第一帧对比 (原始 vs 处理后)
        if diagnose and i == 0:
            # 保存处理后深度
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'diagnose_depth_filtered.png'), depth_img)
            # 保存原始深度 (处理后但含远背景, 方便对比空洞位置)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'diagnose_depth_raw.png'),
                        raw_depth.astype(np.uint16))
            # 保存空洞掩膜: 原始有深度 → 处理后空缺 = 被错误填孔的区域 (高亮)
            hole_mask = np.where((raw_depth > 0) & (raw_depth <= clip_mm), 255, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'diagnose_valid_mask.png'), hole_mask)
            # 保存颜色图
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'diagnose_color.png'), color_final)
            # 计算椅子候选区域: 原始深度图中值为0但颜色图中非背景的像素
            color_gray = cv2.cvtColor(color_final, cv2.COLOR_BGR2GRAY)
            fg_mask = (color_gray > 20)  # 非黑区域
            chair_candidate = ((raw_depth == 0) & fg_mask).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'diagnose_chair_holes.png'), chair_candidate)
            n_chair_holes = (chair_candidate > 0).sum()
            print(f'  [诊断] 椅子候选空洞 (有色但无深度): {n_chair_holes} 像素')
            if n_chair_holes > 1000:
                print(f'  ⚠ 深色表面导致大量深度缺失! 建议: 提高激光功率 / 喷涂显影剂')

        # 数码变焦
        color_zoom, depth_zoom, zfx, zfy, zppx, zppy = apply_zoom(
            color_final, depth_img, zoom,
            color_frame.profile.as_video_stream_profile().intrinsics)

        cv2.imwrite(os.path.join(COLOR_DIR, f'{saved:06d}.jpg'), color_zoom)
        cv2.imwrite(os.path.join(DEPTH_DIR, f'{saved:06d}.png'), depth_zoom)

        # 保存第一帧内参 (变焦修正后)
        if saved == 0:
            h, w = color_zoom.shape[:2]
            save_intrinsic(INTRINSIC_PATH, zfx, zfy, zppx, zppy, w, h)

        saved += 1

        # 预览
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_zoom, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = cv2.resize(depth_colormap, (color_zoom.shape[1], color_zoom.shape[0]))
        preview = np.hstack([color_zoom, depth_colormap])
        draw_center_hud(preview, depth_zoom, color_zoom.shape[1], color_zoom.shape[0])
        cv2.putText(preview, f'{target_deg} deg [{saved}/{len(angles)}]', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Open3D Scan Capture', preview)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    elapsed = time.time() - total_start
    print(f'  采集完成: {saved} 帧, {elapsed:.0f}s ({(elapsed/len(angles)):.1f}s/帧)')
    return saved


# ══════════════════════════════════════════════════════════════════
# Phase 2: Open3D 重建管线
# ══════════════════════════════════════════════════════════════════

def run_reconstruction():
    """运行 Open3D RGBD 重建管线: make_fragments → register → refine → integrate."""
    import open3d as o3d
    from initialize_config import initialize_config
    import make_fragments
    import register_fragments
    import refine_registration
    import integrate_scene

    n_color = len([f for f in os.listdir(COLOR_DIR) if f.endswith('.jpg')])
    n_depth = len([f for f in os.listdir(DEPTH_DIR) if f.endswith('.png')])
    if n_color < 5 or n_depth < 5:
        sys.exit(f'图片不足: color={n_color}, depth={n_depth}')

    print(f'\n{"="*55}')
    print(f'  Open3D RGBD 重建: {n_color} 帧')
    print(f'  TSDF voxel={VOXEL_SIZE*1000:.0f}mm  ICP={ICCP_METHOD}')
    print(f'{"="*55}')

    # 深度缩放因子
    depth_scale = 1000.0  # D435i z16 → 1mm/unit → 1000 units/m

    config = {
        'path_dataset': SCAN_DIR,
        'path_intrinsic': INTRINSIC_PATH,
        'depth_scale': depth_scale,
        'max_depth': MAX_DEPTH,
        'voxel_size': VOXEL_SIZE,
        'max_depth_diff': MAX_DEPTH_DIFF,
        'tsdf_cubic_size': TSDF_CUBIC_SIZE,
        'icp_method': ICCP_METHOD,
        'global_registration': 'ransac',
        'python_multi_threading': True,
        'n_frames_per_fragment': min(N_FRAMES_PER_FRAGMENT, n_color),
        'n_keyframes_per_n_frame': 5,
        'min_depth': 0.15,
        'debug_mode': False,
    }
    initialize_config(config)

    times = []

    # 1. Make fragments: RGBD odometry + 局部 TSDF
    print('\n[1/4] 制作碎片 (RGBD odometry + 局部TSDF)...')
    t0 = time.time()
    make_fragments.run(config)
    dt = time.time() - t0
    times.append(dt)
    print(f'  耗时: {dt:.0f}s')

    # 2. Register fragments: FPFH + ICP 配准
    print('\n[2/4] 碎片配准 (FPFH + ICP)...')
    t0 = time.time()
    register_fragments.run(config)
    dt = time.time() - t0
    times.append(dt)
    print(f'  耗时: {dt:.0f}s')

    # 3. Refine registration: 多尺度 ICP 精配准
    print('\n[3/4] 精配准 (Multi-scale ICP)...')
    t0 = time.time()
    refine_registration.run(config)
    dt = time.time() - t0
    times.append(dt)
    print(f'  耗时: {dt:.0f}s')

    # 4. Integrate scene: 全局 TSDF 融合
    print('\n[4/4] 场景融合 (全局 TSDF)...')
    t0 = time.time()
    integrate_scene.run(config)
    dt = time.time() - t0
    times.append(dt)
    print(f'  耗时: {dt:.0f}s')

    total = sum(times)
    print(f'\n  重建总耗时: {total:.0f}s '
          f'(fragments={times[0]:.0f}s, register={times[1]:.0f}s, '
          f'refine={times[2]:.0f}s, integrate={times[3]:.0f}s)')

    # mesh 后处理 + 保存 OBJ
    mesh = o3d.io.read_triangle_mesh(FINAL_PLY)
    if len(mesh.vertices) > 0:
        mesh = _clean_mesh(mesh, radius=0.12)
        o3d.io.write_triangle_mesh(FINAL_PLY, mesh)
        print(f'  PLY mesh: {FINAL_PLY}')

        out_obj = FINAL_PLY.replace('.ply', '.obj')
        o3d.io.write_triangle_mesh(out_obj, mesh)
        print(f'  OBJ mesh: {out_obj}')

    return FINAL_PLY


def _clean_mesh(mesh, radius=0.12):
    """mesh 后处理: 裁切圆柱区域 + 去除孤立碎片."""
    import open3d as o3d

    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh

    # 1. 圆柱裁切: 只保留转台中心 radius 内的顶点
    dist_xy = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    keep_vert = dist_xy <= radius

    if keep_vert.sum() < 3:
        return mesh

    # 重建索引: 删除圆柱外的顶点 → 更新面索引
    old_to_new = np.full(len(verts), -1, dtype=int)
    old_to_new[keep_vert] = np.arange(keep_vert.sum())
    new_verts = verts[keep_vert]

    tris = np.asarray(mesh.triangles)
    keep_tri = keep_vert[tris].all(axis=1)
    new_tris = old_to_new[tris[keep_tri]]

    mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    mesh.vertices = o3d.utility.Vector3dVector(new_verts)

    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[keep_vert])

    # 2. 去孤立碎片: 只保留最大连通分量
    triangles = np.asarray(mesh.triangles)
    if len(triangles) > 0:
        mesh.remove_unreferenced_vertices()
        result = mesh.cluster_connected_triangles()
        # Open3D 0.19 returns (triangle_labels, cluster_counts, cluster_areas)
        if len(result) >= 2 and len(result[0]) > 0:
            labels = np.asarray(result[0])
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) > 1:
                largest_label = unique[np.argmax(counts)]
                keep_tri = labels == largest_label
                verts2 = np.asarray(mesh.vertices)
                tris2 = np.asarray(mesh.triangles)[keep_tri]
                # 重建索引
                used = np.unique(tris2.ravel())
                old2new2 = np.full(len(verts2), -1, dtype=int)
                old2new2[used] = np.arange(len(used))
                new_verts2 = verts2[used]
                new_tris2 = old2new2[tris2]
                mesh.triangles = o3d.utility.Vector3iVector(new_tris2)
                mesh.vertices = o3d.utility.Vector3dVector(new_verts2)
                if mesh.has_vertex_colors():
                    c2 = np.asarray(mesh.vertex_colors)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(c2[used])

    return mesh


def _clean_mesh_above_plate(mesh, radius=0.10, z_min=0.01):
    """mesh 后处理 (累积+泊松路径): 裁圆柱 + 切转盘面 + 去碎片."""
    import open3d as o3d

    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh

    # 1. 圆柱裁切 + Z 下限 (删除转盘面)
    dist_xy = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    keep_vert = (dist_xy <= radius) & (verts[:, 2] >= z_min)

    if keep_vert.sum() < 3:
        return mesh

    old_to_new = np.full(len(verts), -1, dtype=int)
    old_to_new[keep_vert] = np.arange(keep_vert.sum())
    new_verts = verts[keep_vert]

    tris = np.asarray(mesh.triangles)
    keep_tri = keep_vert[tris].all(axis=1)
    new_tris = old_to_new[tris[keep_tri]]

    mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    mesh.vertices = o3d.utility.Vector3dVector(new_verts)
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors[keep_vert])

    mesh.remove_unreferenced_vertices()

    # 2. 只保留最大连通分量
    tris2 = np.asarray(mesh.triangles)
    if len(tris2) > 0:
        result = mesh.cluster_connected_triangles()
        if len(result) >= 2 and len(result[0]) > 0:
            labels = np.asarray(result[0])
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) > 1:
                largest_label = unique[np.argmax(counts)]
                keep_t = labels == largest_label
                verts3 = np.asarray(mesh.vertices)
                tris3 = tris2[keep_t]
                used = np.unique(tris3.ravel())
                old2new2 = np.full(len(verts3), -1, dtype=int)
                old2new2[used] = np.arange(len(used))
                mesh.triangles = o3d.utility.Vector3iVector(old2new2[tris3])
                mesh.vertices = o3d.utility.Vector3dVector(verts3[used])
                if mesh.has_vertex_colors():
                    c3 = np.asarray(mesh.vertex_colors)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(c3[used])

    return mesh


def _crop_mesh_cylinder(mesh, radius=0.10, z_min=0.01):
    """简单裁切: 圆柱 + Z下限 (不跑连通分量, 保留薄结构)."""
    import open3d as o3d

    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        return mesh

    dist_xy = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    keep_vert = (dist_xy <= radius) & (verts[:, 2] >= z_min)

    if keep_vert.sum() < 3:
        return mesh

    old_to_new = np.full(len(verts), -1, dtype=int)
    old_to_new[keep_vert] = np.arange(keep_vert.sum())
    new_verts = verts[keep_vert]

    tris = np.asarray(mesh.triangles)
    keep_tri = keep_vert[tris].all(axis=1)
    new_tris = old_to_new[tris[keep_tri]]

    mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    mesh.vertices = o3d.utility.Vector3dVector(new_verts)
    if mesh.has_vertex_colors():
        c = np.asarray(mesh.vertex_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(c[keep_vert])
    mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct_with_known_poses():
    """基于标定的直接 TSDF 融合 (跳过 RGBD odometry).

    用 calibrate.json 的转轴参数 + 已知转台角度, 直接算每帧相机外参,
    一步融合到 TSDF, 比 odometry 更精确.
    """
    import glob
    import open3d as o3d

    if not os.path.exists(CALIB_FILE):
        print('  标定文件不存在, 退回 odometry 管线')
        return run_reconstruction()

    with open(CALIB_FILE, 'r') as f:
        calib = json.load(f)
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])

    # 加载内参
    if not os.path.exists(INTRINSIC_PATH):
        print('  内参文件不存在, 退回 odometry 管线')
        return run_reconstruction()
    with open(INTRINSIC_PATH, 'r') as f:
        intr_data = json.load(f)
    im = intr_data['intrinsic_matrix']
    fx, fy, ppx, ppy = im[0], im[4], im[6], im[7]
    w_img = intr_data['width']
    h_img = intr_data['height']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w_img, h_img, fx, fy, ppx, ppy)

    # 帧列表
    color_files = sorted(glob.glob(os.path.join(COLOR_DIR, '*.jpg')))
    depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, '*.png')))
    n_frames = len(color_files)

    if n_frames < 3:
        print('  帧数不足, 退回 odometry 管线')
        return run_reconstruction()

    # 帧对应的转台角度 (按采集顺序: 0°, STEP_ANGLE°, 2*STEP_ANGLE°, ...)
    angles = np.arange(n_frames) * STEP_ANGLE

    print(f'\n{"="*55}')
    print(f'  直接 TSDF 融合 (标定位姿): {n_frames} 帧')
    print(f'  voxel={VOXEL_SIZE*1000:.0f}mm  angles={angles[0]:.0f}°~{angles[-1]:.0f}°')
    print(f'{"="*55}')

    # 创建 TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    t0 = time.time()
    n_good = 0
    for i, (cpath, dpath, ang) in enumerate(zip(color_files, depth_files, angles)):
        color = o3d.io.read_image(cpath)
        depth = o3d.io.read_image(dpath)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False)

        # 相机外参: 物体随转台转 +θ → 反旋对齐
        theta = np.radians(ang)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_z_rot = np.array([[cos_t, -sin_t, 0],
                            [sin_t, cos_t, 0],
                            [0, 0, 1]])  # R_z(+θ)

        R_ext = R_z_rot @ R_calib
        t_ext = R_z_rot @ t_calib

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_ext
        extrinsic[:3, 3] = t_ext

        volume.integrate(rgbd, intrinsic, np.linalg.inv(extrinsic))
        n_good += 1

        if (i + 1) % 12 == 0 or i == n_frames - 1:
            print(f'  [{i+1}/{n_frames}] {ang:.0f}° integrated')

    dt = time.time() - t0
    print(f'  融合耗时: {dt:.0f}s ({n_good} 帧)')

    # 提取 mesh (triangle mesh, 不是点云)
    mesh = volume.extract_triangle_mesh()
    print(f'  mesh 原始: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面')

    # mesh 后处理: 裁切圆柱区域 + 去孤立碎片
    mesh = _clean_mesh(mesh, radius=0.12)
    print(f'  mesh 清理后: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面')

    # 保存 PLY (带顶点颜色)
    out_ply = os.path.join(SCENE_DIR, 'integrated_calib_mesh.ply')
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f'  PLY: {out_ply}')

    # 保存 OBJ (通用格式, 方便你后续转点云)
    out_obj = os.path.join(SCENE_DIR, 'integrated_calib_mesh.obj')
    o3d.io.write_triangle_mesh(out_obj, mesh)
    print(f'  OBJ: {out_obj}')

    return out_ply


# ══════════════════════════════════════════════════════════════════
# Phase 2b: 简单点云累积 (替代 TSDF, 方便调试)
# ══════════════════════════════════════════════════════════════════

def reconstruct_simple_accumulation(cylinder_radius=0.12, flip_rotation=False):
    """简单点云累积: 每帧先切除转盘面 → 旋转对齐 → 累加.

    核心改进: 在每帧相机空间 RANSAC 找转盘面→切除, 然后再变换累积.
    这样转盘面不会污染累积结果.
    """
    import glob, open3d as o3d

    if not os.path.exists(CALIB_FILE):
        print('  标定文件不存在')
        return None

    with open(CALIB_FILE, 'r') as f:
        calib = json.load(f)
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])
    calib_radius = calib.get('radius_m', 0.1)

    if not os.path.exists(INTRINSIC_PATH):
        print('  内参文件不存在')
        return None
    with open(INTRINSIC_PATH, 'r') as f:
        intr_data = json.load(f)
    im = intr_data['intrinsic_matrix']
    fx, fy, ppx, ppy = im[0], im[4], im[6], im[7]

    color_files = sorted(glob.glob(os.path.join(COLOR_DIR, '*.jpg')))
    depth_files = sorted(glob.glob(os.path.join(DEPTH_DIR, '*.png')))
    n_frames = len(color_files)
    angles = np.arange(n_frames) * STEP_ANGLE

    sign = -1 if flip_rotation else 1
    effective_radius = min(cylinder_radius, calib_radius * 1.5) if cylinder_radius > 0 else calib_radius * 1.5

    print(f'\n{"="*55}')
    print(f'  逐帧去转盘 + 累积: {n_frames} 帧')
    print(f'  旋转: {"R_z(-θ)" if sign < 0 else "R_z(+θ)"}  |  圆柱mask: {effective_radius:.3f}m')
    print(f'{"="*55}')

    all_pts, all_colors = [], []
    t0 = time.time()

    for i, (cpath, dpath, ang) in enumerate(zip(color_files, depth_files, angles)):
        color = cv2.imread(cpath)
        depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth > 0.001) & (depth < MAX_DEPTH)
        if valid.sum() < 1000:
            continue

        # ── 像素 → 相机坐标 → world ──
        all_idx = np.where(valid)
        X_c = (u[valid] - ppx) / fx * depth[valid]
        Y_c = (v[valid] - ppy) / fy * depth[valid]
        Z_c = depth[valid]
        pts_cam = np.column_stack([X_c, Y_c, Z_c])
        pts_world = (R_calib @ pts_cam.T).T + t_calib

        # ── 旋转对齐 ──
        theta = np.radians(ang)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        if sign < 0:
            R_align = np.array([[cos_t, sin_t, 0], [-sin_t, cos_t, 0], [0, 0, 1]])
        else:
            R_align = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        pts_world = (R_align @ pts_world.T).T

        # ── world Z 阈值去转盘面 (标定后转盘面在 world Z≈0) ──
        # 只保留转盘面以上的点 (Z < -0.003 = 高于转盘面 3mm)
        # 不用 RANSAC 相机空间找平面, 避免误删靠背等垂直面
        n_before_plate = len(pts_world)
        above_plate = pts_world[:, 2] < -0.003
        pts_world = pts_world[above_plate]
        all_idx = (all_idx[0][above_plate], all_idx[1][above_plate])
        n_removed = n_before_plate - len(pts_world)

        if len(pts_world) < 500:
            continue

        # ── 圆柱 mask (去除远离转轴的地面/墙壁) ──
        if effective_radius > 0:
            dist_xy = np.sqrt(pts_world[:, 0]**2 + pts_world[:, 1]**2)
            keep = dist_xy <= effective_radius
            pts_world = pts_world[keep]
            all_idx = (all_idx[0][keep], all_idx[1][keep])

        # ── 颜色 ──
        row_idx = all_idx[0]
        col_idx = all_idx[1]
        rgb = color[row_idx, col_idx][:, ::-1] / 255.0

        all_pts.append(pts_world.astype(np.float32))
        all_colors.append(rgb.astype(np.float32))

        if (i + 1) % 6 == 0 or i == 0 or i == n_frames - 1:
            n_so_far = sum(len(p) for p in all_pts)
            pct_removed = n_removed / max(n_before_plate, 1) * 100
            print(f'  [{i+1}/{n_frames}] {ang:3.0f}°  '
                  f'去转盘{pct_removed:.0f}% → 家具{len(pts_world):,}点 | 累积{n_so_far:,}点')

        # 诊断帧
        if i == 0:
            debug0 = pts_world.copy()
        if i == 1:
            debug1 = pts_world.copy()

    if not all_pts:
        print('  无有效点')
        return None

    # 保存对齐诊断: frame 0 和 frame 1 (应对齐到同一位置)
    if 'debug0' in dir() and 'debug1' in dir():
        pcd_d0 = o3d.geometry.PointCloud()
        pcd_d0.points = o3d.utility.Vector3dVector(debug0)
        pcd_d0.paint_uniform_color([1, 0, 0])  # 红色 = frame 0
        pcd_d1 = o3d.geometry.PointCloud()
        pcd_d1.points = o3d.utility.Vector3dVector(debug1)
        pcd_d1.paint_uniform_color([0, 1, 0])  # 绿色 = frame 1
        debug_path = os.path.join(SCENE_DIR, 'debug_alignment.ply')
        pcd_debug = pcd_d0 + pcd_d1
        o3d.io.write_point_cloud(debug_path, pcd_debug)
        print(f'  对齐诊断: {debug_path} (红色=0°  绿色=10°  应重叠)')
        print(f'    → 如果红绿交错无重叠 → 旋转方向错误, 加 --flip 重试')

    dt = time.time() - t0
    pts_all = np.vstack(all_pts)
    colors_all = np.vstack(all_colors)
    print(f'  累计总点数: {len(pts_all):,}  耗时: {dt:.0f}s')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)
    pcd.colors = o3d.utility.Vector3dVector(colors_all)

    # 降采样 (为 Poisson 准备, 1.5mm 网格保留更多细节)
    pcd = pcd.voxel_down_sample(voxel_size=0.0015)
    # 保存降采样后的点云供检查
    pts_before = np.asarray(pcd.points)
    print(f'  累积点云 (降采样后): {len(pts_before):,} 点')
    print(f'    XY range: X[{pts_before[:,0].min():.4f},{pts_before[:,0].max():.4f}] '
          f'Y[{pts_before[:,1].min():.4f},{pts_before[:,1].max():.4f}]')
    r_xy = np.sqrt(pts_before[:,0]**2 + pts_before[:,1]**2)
    print(f'    r_xy: max={r_xy.max():.4f}  median={np.median(r_xy):.4f}  '
          f'95%={np.percentile(r_xy, 95):.4f}')
    # 保存点云供可视化
    pcd_debug = o3d.geometry.PointCloud()
    pcd_debug.points = o3d.utility.Vector3dVector(pts_before)
    o3d.io.write_point_cloud(os.path.join(SCENE_DIR, 'accumulated_points.ply'), pcd_debug)

    # Poisson 表面重建 → triangle mesh
    print('  Poisson 表面重建 (depth=10)...')
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.006, max_nn=30))
    # 相机在 Z 正上方 (world 坐标翻转后)
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 1.]))
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, n_threads=4)

    # Z 翻转 (world Z↓ → 视觉 Z↑)
    verts = np.asarray(mesh.vertices)
    verts[:, 2] = -verts[:, 2]
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # 裁切圆柱 + 切除转盘面 (不跑连通分量, 保留薄结构)
    mesh = _crop_mesh_cylinder(mesh, radius=0.08, z_min=0.002)

    out_ply = os.path.join(SCENE_DIR, 'accumulated_mesh.ply')
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f'  PLY mesh: {out_ply}')
    print(f'  顶点: {len(mesh.vertices):,}  面: {len(mesh.triangles):,}')

    out_obj = os.path.join(SCENE_DIR, 'accumulated_mesh.obj')
    o3d.io.write_triangle_mesh(out_obj, mesh)
    print(f'  OBJ mesh: {out_obj}')

    return out_ply


# ══════════════════════════════════════════════════════════════════
# Phase 3: 后处理
# ══════════════════════════════════════════════════════════════════

def remove_turntable_plane(pcd, distance_threshold=0.005):
    """RANSAC 找水平面 → 删除转盘面点 (保留家具)."""
    import open3d as o3d

    pts = np.asarray(pcd.points)
    if len(pts) < 500:
        return pcd

    # 只在下半部分找转盘面 (家具在上半部)
    z_all = pts[:, 2]
    z_median = np.median(z_all)
    lower = pts[z_all < z_median]

    if len(lower) < 500:
        return pcd

    pcd_lower = o3d.geometry.PointCloud()
    pcd_lower.points = o3d.utility.Vector3dVector(lower)

    plane_model, inliers = pcd_lower.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)

    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n_norm = n / np.linalg.norm(n)

    # 只处理水平面 (法向量接近 Z 轴)
    z_alignment = abs(n_norm[2])
    if z_alignment < 0.85:
        print(f'  转盘切除: 未找到水平面 (Z_align={z_alignment:.2f}), 跳过')
        return pcd

    # 删除平面上及其附近点
    all_dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)
    above = pts[all_dists >= distance_threshold]
    n_removed = len(pts) - len(above)

    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(above)

    print(f'  转盘切除: {n_removed:,} 点 ({n_removed/len(pts)*100:.0f}%) '
          f'| 法向量 Z={z_alignment:.2f}')
    return result


def postprocess(ply_path):
    """mesh 后处理: 转盘切除 + 去碎片 (仅当输入是 mesh 时)."""
    import open3d as o3d

    if not os.path.exists(ply_path):
        print(f'输出不存在: {ply_path}')
        return

    # 尝试作为 mesh 读取
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if len(mesh.vertices) == 0:
        print('读取 mesh 失败, 跳过')
        return

    n_raw_v = len(mesh.vertices)
    n_raw_t = len(mesh.triangles)
    print(f'\n[后处理] mesh 原始: {n_raw_v:,} 顶点, {n_raw_t:,} 面')

    # 裁切 + 去碎片
    mesh = _clean_mesh(mesh, radius=0.12)

    n_final_v = len(mesh.vertices)
    n_final_t = len(mesh.triangles)
    print(f'  清理后: {n_final_v:,} 顶点, {n_final_t:,} 面')

    # 简化 (可选, 保持质量)
    if n_final_t > 50000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
        print(f'  简化后: {len(mesh.triangles):,} 面')

    out_ply = ply_path.replace('.ply', '_clean.ply')
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f'  输出: {out_ply}')

    out_obj = ply_path.replace('.ply', '_clean.obj')
    o3d.io.write_triangle_mesh(out_obj, mesh)
    print(f'  输出: {out_obj}')


# ══════════════════════════════════════════════════════════════════
# 零点标定 (自动 + 手动)
# ══════════════════════════════════════════════════════════════════

def auto_zero_calibration(pipeline, align, colorizer, tt,
                          distance_cm=45.0, radius_cm=10.0,
                          coarse_step=20, fine_step=5):
    """自动零点标定: 基于深度数据搜索家具正面.

    原理: 旋转转台360°, 每个角度计算"正面得分".
    得分 = 中心区域有效深度密度 × 距离匹配度.
    家具正对/侧对相机时, 画面中央有效深度点最多、距离最接近预期.

    输出: 最佳角度 (0~360)
    """
    min_valid = 280                       # D435i 盲区
    clip_max = int((distance_cm + radius_cm + 50) * 10)

    def _capture_score():
        """采集一帧并计算正面得分."""
        for _ in range(3):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        cf = aligned.get_color_frame()
        if not df or not cf:
            return 0.0, None
        depth_img = np.asanyarray(df.get_data())
        color_img = np.asanyarray(cf.get_data())
        h, w = depth_img.shape
        # 中心 40% ROI
        rhh, rwh = int(h * 0.2), int(w * 0.2)
        cy, cx = h // 2, w // 2
        roi = depth_img[cy - rhh:cy + rhh, cx - rwh:cx + rwh]
        valid = roi[(roi > min_valid) & (roi < clip_max)]
        if len(valid) < 100:
            return 0.0, color_img
        density = len(valid) / roi.size
        avg_d = np.median(valid)
        expected_d = distance_cm * 10
        dist_score = max(0, 1.0 - abs(avg_d - expected_d) / expected_d)
        return density * dist_score, color_img

    print('\n' + '=' * 55)
    print('  自动零点标定 (深度搜索)')
    print(f'  粗搜步长={coarse_step}°  细搜步长={fine_step}°')
    print('=' * 55)

    # ── 第一步: 粗搜 ──
    print('\n[1/3] 粗搜 360° ...')
    coarse = {}
    for ang in range(0, 360, coarse_step):
        tt.move_absolute(ang, speed=TURNTABLE_DEFAULT_SPEED)
        tt.wait_stop(timeout=30)
        time.sleep(0.8)
        score, _ = _capture_score()
        coarse[ang] = score
        bar = '#' * int(score * 20) if score > 0 else ''
        print(f'  {ang:3d}°  score={score:.4f}  {bar}')
    best_c = max(coarse, key=coarse.get)
    print(f'粗搜最佳: {best_c}° (score={coarse[best_c]:.4f})')

    # ── 第二步: 细搜 ──
    print('\n[2/3] 细搜 ...')
    fine = {}
    for ang in range(max(0, best_c - coarse_step),
                     min(360, best_c + coarse_step + 1), fine_step):
        tt.move_absolute(ang, speed=TURNTABLE_DEFAULT_SPEED)
        tt.wait_stop(timeout=30)
        time.sleep(0.5)
        score, _ = _capture_score()
        fine[ang] = score
        marker = ' <--' if score == max(fine.values()) else ''
        print(f'  {ang:3d}°  score={score:.4f}{marker}')
    best_f = max(fine, key=fine.get)

    # ── 第三步: 抛物线插值 ──
    angles = np.array(list(fine.keys()))
    scores = np.array(list(fine.values()))
    mask = np.abs(angles - best_f) <= fine_step * 2
    local_a, local_s = angles[mask], scores[mask]
    if len(local_a) >= 3:
        coeffs = np.polyfit(local_a, local_s, 2)
        if coeffs[0] < 0:
            peak = (-coeffs[1] / (2 * coeffs[0])) % 360
        else:
            peak = float(best_f)
    else:
        peak = float(best_f)

    print(f'\n[3/3] 插值结果: {peak:.1f}°')

    tt.move_absolute(int(peak), speed=TURNTABLE_DEFAULT_SPEED)
    tt.wait_stop(timeout=30)
    return peak


def manual_zero_calibration(pipeline, align, pc, colorizer, tt):
    """手动零点标定: 用户旋转转台直到家具侧面正对相机, 按回车确认.

    键盘:
      A/D 或 ← →  — 微调角度 (1°/5°)
      SPACE/ENTER  — 确认当前位置为喷涂零点
      Q/W          — ±5° 快调
    """
    print('\n' + '=' * 55)
    print('  零点标定')
    print('  A/D 或 ← → 旋转转台 → 侧面正对相机后按 ENTER')
    print('=' * 55)

    cv2.namedWindow('Zero Calibration - AD=rotate | ENTER=confirm', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Zero Calibration - AD=rotate | ENTER=confirm', 960, 360)

    total_rotated = 0.0

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        h, w = color_img.shape[:2]
        depth_disp = cv2.resize(depth_colored, (w, h))
        preview = np.hstack([color_img, depth_disp])

        cx_disp = w + w // 2
        cy_disp = h // 2
        cv2.line(preview, (cx_disp, cy_disp - 30), (cx_disp, cy_disp + 30), (0, 0, 255), 1)
        cv2.line(preview, (cx_disp - 30, cy_disp), (cx_disp + 30, cy_disp), (0, 0, 255), 1)

        cv2.putText(preview, f'Rotated: {total_rotated:+.0f} deg', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, 'A/← : -1 deg | D/→ : +1 deg', (10, h + h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(preview, 'Q/W : -5/+5 deg | ENTER : confirm', (10, h + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(preview, 'Align furniture face to camera center, then ENTER',
                    (w // 4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow('Zero Calibration - AD=rotate | ENTER=confirm', preview)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('a') or key == 81:  # ←
            tt.move_and_wait(-1, speed=TURNTABLE_DEFAULT_SPEED)
            total_rotated -= 1
        elif key == ord('d') or key == 83:  # →
            tt.move_and_wait(1, speed=TURNTABLE_DEFAULT_SPEED)
            total_rotated += 1
        elif key == ord('q'):
            tt.move_and_wait(-5, speed=TURNTABLE_DEFAULT_SPEED)
            total_rotated -= 5
        elif key == ord('w'):
            tt.move_and_wait(5, speed=TURNTABLE_DEFAULT_SPEED)
            total_rotated += 5
        elif key == ord(' ') or key == 13:  # SPACE or ENTER
            break
        elif key == 27:  # ESC
            total_rotated = 0.0
            break

    cv2.destroyAllWindows()
    return total_rotated


# ══════════════════════════════════════════════════════════════════
# 旋转轴标定 (家具旋转法)
# ══════════════════════════════════════════════════════════════════

def _capture_single_rgbd(pipeline, align, n_avg=3):
    """采集单帧 RGB-D (多帧中值滤波)."""
    depth_accum = []
    color = None
    for _ in range(n_avg):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        cf = aligned.get_color_frame()
        if not df or not cf:
            continue
        d = np.asanyarray(df.get_data()).astype(np.float32) * 0.001
        depth_accum.append(d)
        if color is None:
            color = np.asanyarray(cf.get_data())
        time.sleep(0.03)
    if len(depth_accum) < 2:
        return None, None
    depth_stack = np.stack(depth_accum, axis=0)
    depth_median = np.median(depth_stack, axis=0)
    mask = (depth_median > 0.001).astype(np.uint8)
    depth_filled = cv2.inpaint((depth_median * 1000).astype(np.uint16),
                               255 - mask * 255, 3, cv2.INPAINT_NS)
    return depth_filled.astype(np.float32) * 0.001, color


def _get_xyz(uv, depth_img, fx, fy, ppx, ppy):
    """像素坐标 → 3D 相机坐标."""
    u, v = uv
    u0, v0 = int(u), int(v)
    h, w = depth_img.shape
    if u0 < 0 or u0 >= w - 1 or v0 < 0 or v0 >= h - 1:
        return None
    up, vp = u - u0, v - v0
    d00, d01 = depth_img[v0, u0], depth_img[v0, u0 + 1]
    d10, d11 = depth_img[v0 + 1, u0], depth_img[v0 + 1, u0 + 1]
    for dd in [d00, d01, d10, d11]:
        if dd <= 0.001 or dd > 3.0:
            return None
    d = (1 - vp) * (d01 * up + d00 * (1 - up)) + vp * (d11 * up + d10 * (1 - up))
    x = (u - ppx) / fx * d
    y = (v - ppy) / fy * d
    return np.array([x, y, d])


def _register_rgbd_ransac(pts_0, pts_1):
    """RANSAC 刚体变换 (3×N → 3×N). Returns (R, t, n_inliers)."""
    n_pts = pts_0.shape[1]
    best_R, best_t, best_inliers = None, None, []
    max_dist = 0.03
    for _ in range(500):
        idx = np.random.choice(n_pts, 3, replace=False)
        src = pts_0[:, idx]; dst = pts_1[:, idx]
        c_src = src.mean(axis=1, keepdims=True)
        c_dst = dst.mean(axis=1, keepdims=True)
        H = (src - c_src) @ (dst - c_dst).T
        U, s, Vt = np.linalg.svd(H)
        R_est = Vt.T @ U.T
        if np.linalg.det(R_est) < 0:
            Vt[2, :] *= -1; R_est = Vt.T @ U.T
        t_est = (c_dst - R_est @ c_src).flatten()
        diff = pts_1 - (R_est @ pts_0 + t_est.reshape(3, 1))
        inliers = np.where(np.linalg.norm(diff, axis=0) < max_dist)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers; best_R, best_t = R_est, t_est
    if best_R is None or len(best_inliers) < 8:
        return None, None, 0
    # refine with all inliers
    src_in = pts_0[:, best_inliers]; dst_in = pts_1[:, best_inliers]
    c_src = src_in.mean(axis=1, keepdims=True)
    c_dst = dst_in.mean(axis=1, keepdims=True)
    H = (src_in - c_src) @ (dst_in - c_dst).T
    U, s, Vt = np.linalg.svd(H)
    R_ref = Vt.T @ U.T
    if np.linalg.det(R_ref) < 0:
        Vt[2, :] *= -1; R_ref = Vt.T @ U.T
    t_ref = (c_dst - R_ref @ c_src).flatten()
    return R_ref, t_ref, len(best_inliers)


def _axis_from_R(R):
    """Rodrigues: 旋转矩阵 → 轴方向 + 角度."""
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle < 1e-6:
        return None, 0
    return rvec.flatten() / angle, angle


def _axis_point(R, t):
    """(I-R)@c = t → 轴上一点 (SVD 伪逆)."""
    U, s, Vt = np.linalg.svd(np.eye(3) - R)
    s_inv = np.zeros(3); s_inv[:2] = 1.0 / s[:2]
    return (Vt.T @ np.diag(s_inv) @ U.T @ t)


def _normal_to_R(normal):
    """法向量 → 旋转矩阵 (cam→world, Z-up)."""
    n_w = np.array([0., 0., 1.])
    cos_t = np.dot(normal, n_w)
    if cos_t > 0.9999:
        return np.eye(3)
    if cos_t < -0.9999:
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    k = np.cross(normal, n_w); k = k / np.linalg.norm(k)
    sin_t = np.linalg.norm(np.cross(normal, n_w))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)


def calibrate_rotation_axis(pipeline, align, intr, tt, delta_deg=30):
    """家具旋转法标定转轴 → 保存 calibrate.json.

    放家具在转台上 → 拍两帧 (间隔 delta_deg°) → 特征匹配配准
    → 提取旋转轴方向 → 找转台面 → 轴面交点 = 世界原点.
    """
    import open3d as o3d

    fx, fy = intr.fx, intr.fy
    ppx, ppy = intr.ppx, intr.ppy

    print(f'\n{"="*55}')
    print(f'  旋转轴标定 (家具旋转法, Δθ={delta_deg}°)')
    print(f'{"="*55}')

    # ── Frame 0 ──
    print('\n[轴标定] 拍摄角度 0°...')
    depth_0, color_0 = _capture_single_rgbd(pipeline, align)
    if depth_0 is None:
        print('  采集失败, 跳过旋转轴标定')
        return False

    # ── Rotate ──
    print(f'[轴标定] 旋转转台 {delta_deg}°...')
    tt.move_and_wait(delta_deg)

    # ── Frame 1 ──
    print(f'[轴标定] 拍摄角度 {delta_deg}°...')
    depth_1, color_1 = _capture_single_rgbd(pipeline, align)
    if depth_1 is None:
        print('  采集失败, 跳过旋转轴标定')
        return False

    # ── ICP 点云配准 (替代特征匹配, 更鲁棒) ──
    print('[轴标定] ICP 点云配准...')

    h, w = depth_0.shape

    # 运动掩膜: 帧差 → 只保留转动的家具区域
    gray_0 = cv2.cvtColor(color_0, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.cvtColor(color_1, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray_0, gray_1)
    _, motion_mask = cv2.threshold(frame_diff, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    motion_mask = (motion_mask > 0)

    # 诊断图
    diag = color_0.copy()
    diag[~motion_mask] = diag[~motion_mask] // 2
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'calib_motion_mask.png'), diag)

    # 用运动掩膜过滤深度图 → 创建两个点云
    d0_masked = depth_0.copy(); d0_masked[~motion_mask] = 0
    d1_masked = depth_1.copy(); d1_masked[~motion_mask] = 0

    pcd_0 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image((d0_masked * 1000).astype(np.uint16)),
        o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, ppx, ppy),
        depth_scale=1000.0)
    pcd_1 = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image((d1_masked * 1000).astype(np.uint16)),
        o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, ppx, ppy),
        depth_scale=1000.0)

    # 去飞点 + 降采样
    pcd_0, _ = pcd_0.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_1, _ = pcd_1.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_0 = pcd_0.voxel_down_sample(0.003)  # 3mm
    pcd_1 = pcd_1.voxel_down_sample(0.003)

    print(f'  点云: 0°={len(pcd_0.points):,}点  30°={len(pcd_1.points):,}点')

    if len(pcd_0.points) < 500 or len(pcd_1.points) < 500:
        print('  点云点数不足, 跳过旋转轴标定')
        return False

    # 估算法向量 (point-to-plane ICP 需要)
    pcd_0.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd_1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

    # ICP: pcd_1 → pcd_0, 初始为单位矩阵
    init_transform = np.eye(4)
    result = o3d.pipelines.registration.registration_icp(
        pcd_1, pcd_0,
        max_correspondence_distance=0.05,  # 5cm, 适应30°旋转的位移
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200))

    T_icp = result.transformation  # 4×4
    R_rot = T_icp[:3, :3]
    t_rot = T_icp[:3, 3]

    print(f'  ICP: fitness={result.fitness:.3f}  rmse={result.inlier_rmse*1000:.1f}mm')

    if result.fitness < 0.3:
        print('  ⚠ ICP 匹配率过低, 配准可能不准')

    # ── Extract axis ──
    axis_dir, axis_angle_rad = _axis_from_R(R_rot)
    print(f'  旋转角度: {np.degrees(axis_angle_rad):.1f}° (期望 {delta_deg}°)')
    print(f'  转轴方向 (cam): [{axis_dir[0]:.4f} {axis_dir[1]:.4f} {axis_dir[2]:.4f}]')

    if abs(np.degrees(axis_angle_rad) - delta_deg) > 15:
        print('  ⚠ 角度偏差较大, 使用转台面法向量作为转轴方向')
        # 兜底: 用转台平面法向量作为转轴方向
        pcd_full = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image((depth_0 * 1000).astype(np.uint16)),
            o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, ppx, ppy),
            depth_scale=1000.0)
        pts_full = np.asarray(pcd_full.points)
        near_full = np.where((np.linalg.norm(pts_full, axis=1) > 0.2) &
                              (np.linalg.norm(pts_full, axis=1) < 1.5))[0]
        if len(near_full) > 1000:
            pcd_near_full = pcd_full.select_by_index(near_full)
            fallback_plane, _ = pcd_near_full.segment_plane(
                distance_threshold=0.012, ransac_n=3, num_iterations=1000)
            a, b, c, d = fallback_plane
            fallback_normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            if fallback_normal[2] > 0:
                fallback_normal = -fallback_normal
            axis_dir = fallback_normal
            print(f'  使用平面法向量: [{axis_dir[0]:.4f} {axis_dir[1]:.4f} {axis_dir[2]:.4f}]')

    # ── Axis point + turntable plane ──
    axis_pt = _axis_point(R_rot, t_rot)

    # Find turntable plane from depth_0
    pcd_o3d = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image((depth_0 * 1000).astype(np.uint16)),
        o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, ppx, ppy),
        depth_scale=1000.0)
    pts = np.asarray(pcd_o3d.points)
    near = np.where((np.linalg.norm(pts, axis=1) > 0.2) & (np.linalg.norm(pts, axis=1) < 1.5))[0]
    plane_model = None
    if len(near) > 1000:
        pcd_near = pcd_o3d.select_by_index(near)
        plane_model, inliers = pcd_near.segment_plane(
            distance_threshold=0.012, ransac_n=3, num_iterations=1000)
        a, b, c, d = plane_model
        normal = np.array([a, b, c]) / np.linalg.norm([a, b, c])
        if normal[2] > 0:
            normal = -normal; d = -d
        print(f'  转台面: {len(inliers):,} inliers  n=[{normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f}]')
        plane_model = np.array([normal[0], normal[1], normal[2], d])

    if plane_model is not None:
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        denom = np.dot(normal, axis_dir)
        if abs(denom) > 1e-6:
            lam = -(d + np.dot(normal, axis_pt)) / denom
            origin_pt = axis_pt + lam * axis_dir
        else:
            origin_pt = axis_pt
    else:
        print('  ⚠ 未检测到转台面, 使用最小范数解')
        origin_pt = axis_pt

    print(f'  世界原点 (cam): [{origin_pt[0]:.4f} {origin_pt[1]:.4f} {origin_pt[2]:.4f}]')

    # ── Build calibration ──
    if axis_dir[2] < 0:
        axis_dir = -axis_dir
    R_calib = _normal_to_R(axis_dir)
    t_calib = -R_calib @ origin_pt

    # ── 验证图 ──
    verify_img = color_0.copy()
    center_cam = -R_calib.T @ t_calib
    radius = 0.10  # 转台半径 10cm
    n_pts = 120

    # 半透明绿色填充转台圆盘
    overlay = verify_img.copy()
    theta_arr = np.linspace(0, 2 * np.pi, n_pts)
    circle_w = np.column_stack([radius * np.cos(theta_arr),
                                radius * np.sin(theta_arr),
                                np.zeros(n_pts)])
    circle_c = (R_calib.T @ circle_w.T).T + center_cam
    front = circle_c[:, 2] > 0.01
    if front.sum() >= 6:
        u = fx * circle_c[front, 0] / circle_c[front, 2] + ppx
        v = fy * circle_c[front, 1] / circle_c[front, 2] + ppy
        pts_uv = np.column_stack([u, v]).astype(np.int32)
        cv2.fillPoly(overlay, [pts_uv], (0, 200, 0))
    verify_img = cv2.addWeighted(verify_img, 0.65, overlay, 0.35, 0)

    # 画 3 个同心圆 (5cm, 10cm, 15cm) 用于判断圆心和半径
    for r_m, color in [(0.05, (100, 100, 100)), (0.10, (0, 255, 0)), (0.15, (100, 100, 100))]:
        cw = np.column_stack([r_m * np.cos(theta_arr),
                              r_m * np.sin(theta_arr),
                              np.zeros(n_pts)])
        cc = (R_calib.T @ cw.T).T + center_cam
        f = cc[:, 2] > 0.01
        if f.sum() >= 6:
            u = fx * cc[f, 0] / cc[f, 2] + ppx
            v = fy * cc[f, 1] / cc[f, 2] + ppy
            pu = np.column_stack([u, v]).astype(np.int32)
            for i in range(len(pu)):
                j = (i + 1) % len(pu)
                if np.sqrt((pu[i][0]-pu[j][0])**2 + (pu[i][1]-pu[j][1])**2) < 150:
                    cv2.line(verify_img, tuple(pu[i]), tuple(pu[j]), color, 2 if r_m == 0.10 else 1)

    # 中心十字线 (延伸整个画面)
    if center_cam[2] > 0.01:
        cu = int(fx * center_cam[0] / center_cam[2] + ppx)
        cv_ = int(fy * center_cam[1] / center_cam[2] + ppy)
        cv2.line(verify_img, (cu - 60, cv_), (cu + 60, cv_), (0, 0, 255), 1)
        cv2.line(verify_img, (cu, cv_ - 60), (cu, cv_ + 60), (0, 0, 255), 1)
        cv2.drawMarker(verify_img, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(verify_img, (cu, cv_), 5, (0, 0, 255), -1)

    # 信息文字
    y0 = 28
    cv2.putText(verify_img, f'Turntable r=10cm (green) | Axis: [{axis_dir[0]:.3f} {axis_dir[1]:.3f} {axis_dir[2]:.3f}]',
                (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2); y0 += 22
    cv2.putText(verify_img, f'Angle: {np.degrees(axis_angle_rad):.1f} deg  Center(cam): [{center_cam[0]:.3f} {center_cam[1]:.3f} {center_cam[2]:.3f}]',
                (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2); y0 += 22
    cv2.putText(verify_img, f'Gray circles: 5cm / 15cm reference',
                (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1); y0 += 22
    cv2.putText(verify_img, 'S=Save  R=Retry  Q=Skip',
                (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

    cv2.imshow('Axis Calibration - Verify', verify_img)
    print('\n  S=保存  R=重来  Q=跳过继续扫描')

    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s') or key == ord('S'):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            calib_data = {
                'R': R_calib.tolist(),
                't': t_calib.tolist(),
                'plate_z': 0.0,
                'rotation_center': [0.0, 0.0],
                'radius_m': float(radius),
            }
            with open(CALIB_FILE, 'w') as f:
                json.dump(calib_data, f, indent=2, default=float)
            print(f'\n  标定已保存 → {CALIB_FILE}')
            cv2.destroyWindow('Axis Calibration - Verify')
            tt.move_and_wait(-delta_deg)
            return True
        elif key == ord('r') or key == ord('R'):
            cv2.destroyWindow('Axis Calibration - Verify')
            print('  重新标定...')
            tt.move_and_wait(-delta_deg)
            return calibrate_rotation_axis(pipeline, align, intr, tt, delta_deg)
        elif key == ord('q') or key == 27:
            cv2.destroyWindow('Axis Calibration - Verify')
            print('  跳过轴标定, 继续扫描...')
            tt.move_and_wait(-delta_deg)
            return False

    # unreachable, fallback
    tt.move_and_wait(-delta_deg)
    return True


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import pyrealsense2 as rs

    parser = argparse.ArgumentParser(description='Open3D RGBD 重建管线')
    parser.add_argument('--skip-capture', action='store_true',
                        help='跳过采集, 直接重建 (颜色/深度图需已存在)')
    parser.add_argument('--distance', type=float, default=DEFAULT_DISTANCE_CM,
                        help=f'物体中心到相机距离 cm (默认: {DEFAULT_DISTANCE_CM})')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS_CM,
                        help=f'物体半径 cm (默认: {DEFAULT_RADIUS_CM})')
    parser.add_argument('--step-angle', type=float, default=STEP_ANGLE,
                        help=f'步进角度 (默认: {STEP_ANGLE}°)')
    parser.add_argument('--voxel-size', type=float, default=VOXEL_SIZE,
                        help=f'TSDF 体素大小 m (默认: {VOXEL_SIZE})')
    parser.add_argument('--zoom', type=float, default=DEFAULT_ZOOM,
                        help=f'数码变焦倍数 (默认: {DEFAULT_ZOOM}x)')
    parser.add_argument('--no-post', action='store_true', help='跳过后处理')
    parser.add_argument('--skip-zero', action='store_true',
                        help='跳过零点标定, 当前位置作为起点')
    parser.add_argument('--skip-calib', action='store_true',
                        help='跳过旋转轴标定 (使用已有 calibrate.json)')
    parser.add_argument('--auto-zero', action='store_true',
                        help='自动零点标定 (深度搜索, 无需手动)')
    parser.add_argument('--manual-zero', action='store_true',
                        help='手动零点标定 (默认: 手动)')
    parser.add_argument('--diagnose', action='store_true',
                        help='保存第一帧诊断图 (原始深度 vs 处理后)')
    parser.add_argument('--no-hole-fill', action='store_true',
                        help='禁用深度填孔 (保留真实空洞, 诊断暗表面用)')
    parser.add_argument('--laser-power', type=float, default=0,
                        help='激光功率 0-360 (默认: 自动, 暗表面建议150-300)')
    parser.add_argument('--simple-accum', action='store_true',
                        help='使用简单点云累积代替 TSDF 融合 (方便调试)')
    parser.add_argument('--flip', action='store_true',
                        help='翻转旋转对齐方向 (方向反了时切换)')
    args = parser.parse_args()

    print('=' * 55)
    print('  Open3D RGBD 重建管线')
    print(f'  深度: D435i 硬件 (尺度正确)')
    print(f'  重建: RGBD Odometry + TSDF 体素融合')
    if args.zoom > 1.0:
        print(f'  变焦: {args.zoom:.1f}x')
    print('=' * 55)

    if not args.skip_capture:
        setup_workspace()

        # ── 连接相机 ──
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            sys.exit('未检测到 RealSense 相机')

        dev = devices[0]
        print(f'相机: {dev.get_info(rs.camera_info.name)}')

        # ── 加载 High Accuracy preset ──
        preset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'HighAccuracyPreset-custom.json')

        # ── 配置相机流 ──
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT,
                          rs.format.z16, DEPTH_FPS)
        cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT,
                          rs.format.bgr8, DEPTH_FPS)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        colorizer = rs.colorizer()
        pc = rs.pointcloud()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        # ── D435i 传感器优化 (最大化深度覆盖率) ──
        depth_sensor = profile.get_device()
        depth_sensors = depth_sensor.query_sensors()
        for s in depth_sensors:
            if s.is_depth_sensor():
                depth_sensor = s
                break
        try:
            # 激光功率: 暗表面 (深色木材) 需要更高功率来反射足够 IR
            if args.laser_power > 0:
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, float(args.laser_power))
                    print(f'激光功率: {args.laser_power:.0f} (手动)')
            else:
                # 默认: 提高到 150 (出厂默认通常 30-100, 深色物体需要 >100)
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, 150.0)
                    print(f'激光功率: 150 (自动提高, 适配深色表面)')
            # 发射器常亮 (避免曝光变化引起的深度波动)
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
                print(f'发射器: 常亮')
        except Exception as e:
            print(f'传感器优化失败: {e}')

        # ── Advanced Mode preset / disparity shift (暂时禁用, 保持出厂默认) ──
        # depth_sensor_dev = profile.get_device()
        # try:
        #     advnc_mode = rs.rs400_advanced_mode(depth_sensor_dev)
        #     if os.path.exists(preset_path):
        #         preset_json = json.load(open(preset_path))
        #         advnc_mode.load_json(str(preset_json).replace("'", '"'))
        #         print(f'深度 preset 已加载: HighAccuracy')
        # except Exception as e:
        #     print(f'Advanced mode 不支持: {e}')

        # 预热
        for _ in range(30):
            pipeline.wait_for_frames()
            time.sleep(0.05)

        # ── 家具放置确认 ──
        print('\n' + '=' * 55)
        print('  请把家具放在转台上, 然后按 SPACE')
        print('  Q/ESC = 退出')
        print('=' * 55)

        cv2.namedWindow('Place furniture - SPACE to capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Place furniture - SPACE to capture', 960, 360)

        while True:
            frames = pipeline.wait_for_frames()
            aligned_f = align.process(frames)
            depth_f = aligned_f.get_depth_frame()
            color_f = aligned_f.get_color_frame()
            if not depth_f or not color_f:
                continue

            color_img = np.asanyarray(color_f.get_data())
            depth_data = np.asanyarray(depth_f.get_data())
            depth_colored = np.asanyarray(colorizer.colorize(depth_f).get_data())
            h, w = color_img.shape[:2]
            depth_disp = cv2.resize(depth_colored, (w, h))
            preview = np.hstack([color_img, depth_disp])

            # 中心十字 + 深度距离 (RGB侧)
            draw_center_hud(preview, depth_data, w, h)

            cv2.putText(preview, 'Place furniture & press SPACE', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, 'RGB', (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, 'Depth', (w + 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Place furniture - SPACE to capture', preview)

            key = cv2.waitKey(10) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                return

        cv2.destroyAllWindows()
        print('家具已放置, 开始标定流程...')

        # ── 连接转台 ──
        print(f'连接转台 {TURNTABLE_PORT} ...')
        tt = TurntableController(port=TURNTABLE_PORT)
        tt.open()
        tt.zero()

        # ── 零点标定 ──
        if args.skip_zero:
            print('\n[零点] 跳过, 当前位置设为零点')
            tt.zero()
        elif args.auto_zero:
            print('\n[零点] 自动标定模式')
            best_angle = auto_zero_calibration(
                pipeline, align, colorizer, tt,
                distance_cm=args.distance, radius_cm=args.radius)
            print(f'[零点] 自动找到正面 ≈ {best_angle:.1f}°  → 设为零点')
            tt.zero()
        else:
            # 默认: 手动零点标定
            print('\n[零点] 手动标定模式')
            total_rotated = manual_zero_calibration(pipeline, align, pc, colorizer, tt)
            print(f'[零点] 手动旋转 {total_rotated:+.0f}  → 设为零点')
            tt.zero()

        # ── 旋转轴标定 (家具旋转法) ──
        if not args.skip_calib:
            calibrate_rotation_axis(pipeline, align, intr, tt, delta_deg=30)
        else:
            print('\n[轴标定] 跳过 (--skip-calib)')

        # ── 扫描采集 ──
        try:
            n_frames = capture_rgbd_frames(
                pipeline, align, tt, args.distance, args.radius, args.zoom,
                diagnose=args.diagnose, skip_hole_fill=args.no_hole_fill)
            if n_frames < 5:
                sys.exit('采集帧数不足')
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            tt.stop()
            tt.close()
            print('相机 + 转台已释放.')
    else:
        if not os.path.isdir(COLOR_DIR) or not os.path.isdir(DEPTH_DIR):
            sys.exit(f'图片目录不存在: {SCAN_DIR}')

    # ── 重建 ──
    if args.simple_accum:
        print('\n使用累积+泊松重建模式')
        ply_path = reconstruct_simple_accumulation(cylinder_radius=0.12,
                                                    flip_rotation=args.flip)
    elif os.path.exists(CALIB_FILE):
        print('\n检测到标定文件, 使用标定位姿直接融合')
        ply_path = reconstruct_with_known_poses()
    else:
        print('\n未检测到标定文件, 使用 RGBD odometry 管线')
        ply_path = run_reconstruction()

    print(f'\n{"="*55}')
    print(f'  完成!')
    if os.path.exists(ply_path):
        fsize = os.path.getsize(ply_path)
        print(f'  输出: {ply_path} ({fsize/1024/1024:.1f} MB)')
    else:
        print(f'  警告: 输出文件未找到, 检查 scene/ 目录')
    print(f'{"="*55}')

    if not args.no_post and os.path.exists(ply_path):
        postprocess(ply_path)


if __name__ == '__main__':
    main()
