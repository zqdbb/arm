#!/usr/bin/env python3
"""
V11 椅子重建: YOLO mask 抠图 + COLMAP 密集匹配
核心: 全分辨率(640×480)采集 → YOLO 精准 mask → 只保留椅子像素 → COLMAP 密集重建
      与 V5 的区别: YOLO mask 替代背景减除, 全分辨率替代数字变焦
"""

import cv2, json, time, os, subprocess, shutil
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d

os.environ['QT_QPA_PLATFORM'] = 'xcb'

BASE = Path(__file__).parent
OUT = BASE / 'output/v11'
W, H = 1280, 720
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150
DISPARITY_SHIFT = 0
DIGITAL_ZOOM = 2.0  # 标定/采集用裁剪模式，同 V6
# COLMAP 阶段使用全分辨率 640×480（从裁剪标定换算内参）

OUT.mkdir(parents=True, exist_ok=True)


def set_disparity_shift(value=DISPARITY_SHIFT):
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return False
        advnc = rs.rs400_advanced_mode(devices[0])
        if not advnc.is_enabled():
            advnc.toggle_advanced_mode(True)
            time.sleep(1)
        dt = advnc.get_depth_table()
        dt.disparityShift = value
        advnc.set_depth_table(dt)
        print(f'disparityShift → {value}')
        return True
    except Exception as e:
        print(f'警告: disparityShift 设置失败: {e}')
        return False


def do_capture(calib):
    """采集: 全分辨率 72 帧 (不裁剪, 给 COLMAP 用)"""
    print('\n' + '=' * 50)
    print(f'  采集: {STEP_DEG}deg × {N_FRAMES} 帧 (全分辨率 {W}×{H})')
    print('=' * 50)

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    # 启动相机（多次尝试）
    pipe = rs.pipeline()
    profile = None
    for attempt in range(3):
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
            profile = pipe.start(cfg)
            time.sleep(0.5)
            pipe.wait_for_frames(timeout_ms=5000)
            break
        except RuntimeError as e:
            print(f'  启动尝试 {attempt+1}/3 失败: {e}')
            try:
                pipe.stop()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.0)
                pipe = rs.pipeline()
    else:
        # 硬件复位后重试
        print('  尝试硬件复位...')
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            try:
                devices[0].hardware_reset()
                time.sleep(2.0)
            except Exception as e:
                print(f'  硬件复位失败: {e}')
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        profile = pipe.start(cfg)
        time.sleep(0.5)
        pipe.wait_for_frames(timeout_ms=10000)

    align = rs.align(rs.stream.color)

    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)
    dscale = ds.get_depth_scale()

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    calib['fx_full'] = intr.fx
    calib['fy_full'] = intr.fy
    calib['ppx_full'] = intr.ppx
    calib['ppy_full'] = intr.ppy
    calib['width_full'] = intr.width
    calib['height_full'] = intr.height
    calib['depth_scale'] = dscale

    calib_path = OUT / 'calibrate.json'
    with open(calib_path, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'V11 标定已保存: {calib_path}')
    print(f'  fx={intr.fx:.1f} fy={intr.fy:.1f} ppx={intr.ppx:.1f} ppy={intr.ppy:.1f} {intr.width}x{intr.height}')
    print(f'相机就绪 (全分辨率 {intr.width}x{intr.height})')

    (OUT / 'color').mkdir(exist_ok=True)
    (OUT / 'depth').mkdir(exist_ok=True)

    t0 = time.time()
    for i in range(N_FRAMES):
        deg = i * STEP_DEG
        tt.move_absolute(deg, speed=5000)
        tt.wait_stop(timeout=30)

        d_sum = np.zeros((H, W), np.float64)
        color_full = None
        n_good = 0
        for _ in range(N_AVG):
            frames = pipe.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            df = aligned.get_depth_frame()
            cf = aligned.get_color_frame()
            if not df or not cf: continue
            d_sum += np.asanyarray(df.get_data()).astype(np.float64)
            if color_full is None:
                color_full = np.asanyarray(cf.get_data())
            n_good += 1

        if n_good == 0:
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  无数据')
            continue

        depth_mm = (d_sum / n_good) * dscale * 1000.0
        depth_u16 = depth_mm.astype(np.uint16)

        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.jpg'), color_full,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'), depth_u16)

        dvals = depth_mm[depth_mm > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  '
              f'valid:{len(dvals)//1000}k  med={np.median(dvals):.0f}mm  {el:.0f}s')

    pipe.stop()
    print(f'采集完成 ({time.time()-t0:.0f}s)')
    if tt: tt.close()


def rotation_to_quaternion(R):
    """3x3 rotation matrix → [w, x, y, z] quaternion (COLMAP convention)."""
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def measure_chair_from_masks(masks_dict, calib):
    """从 YOLO mask + 深度图推算椅子真实尺寸（像素-距离法）"""
    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    dscale = calib.get('depth_scale', 0.001)

    depth_dir = OUT / 'depth'

    heights_m = []
    widths_m = []

    for i, mask in masks_dict.items():
        if mask is None or mask.max() == 0:
            continue
        ys, xs = np.where(mask > 128)
        if len(ys) < 100:
            continue

        dep = cv2.imread(str(depth_dir / f'{i:03d}.png'), cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue

        # mask 内有效深度中位数 = 相机到椅子距离
        dvals = dep[ys, xs].astype(float)
        valid = dvals > 0
        if valid.sum() < 100:
            continue
        dist_m = float(np.median(dvals[valid])) * dscale  # mm -> m
        if dist_m < 0.1:
            continue

        h_px = ys.max() - ys.min()
        w_px = xs.max() - xs.min()

        # 相似三角形: real = pixel x distance / focal
        h_m = h_px * dist_m / fy
        w_m = w_px * dist_m / fx
        heights_m.append(h_m)
        widths_m.append(w_m)

    if not heights_m:
        return None

    heights_m = np.array(heights_m)
    widths_m = np.array(widths_m)

    # 高度取最大值（部分帧 YOLO mask 可能未覆盖顶部）
    chair_h = float(np.max(heights_m))

    # 宽度随转台旋转交替: 正面=宽度, 侧面=深度
    w_sorted = np.sort(widths_m)
    n = len(w_sorted)
    chair_d = float(np.median(w_sorted[:max(1, n // 2)]))   # 窄 = 深度
    chair_w = float(np.median(w_sorted[n // 2:]))           # 宽 = 宽度

    result = {
        'height_m': round(chair_h, 3),
        'width_m': round(chair_w, 3),
        'depth_m': round(chair_d, 3),
        'n_measured': len(heights_m),
    }

    print(f'\n  自动测量 (像素-距离法, {result["n_measured"]} 帧):')
    print(f'    椅子高度: {result["height_m"]*100:.0f}cm')
    print(f'    椅子宽度: {result["width_m"]*100:.0f}cm')
    print(f'    椅子深度: {result["depth_m"]*100:.0f}cm')

    calib['chair_height_m'] = result['height_m']
    calib['chair_width_m'] = result['width_m']
    calib['chair_depth_m'] = result['depth_m']

    return result


def do_fusion(calib):
    """V11 融合: 转台标定位姿 + YOLO mask + COLMAP MVS (跳过 SfM)"""
    from ultralytics import YOLO

    print('\n' + '=' * 50)
    print('  V11: 已知位姿 + YOLO mask + COLMAP MVS')
    print('=' * 50)

    cW = calib.get('width_full', W)
    cH = calib.get('height_full', H)
    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    ppx = calib.get('ppx_full', 320.0)
    ppy = calib.get('ppy_full', 240.0)
    step = calib.get('step_deg', STEP_DEG)
    n_frames = calib.get('n_frames', N_FRAMES)

    color_files = sorted((OUT / 'color').glob('*.jpg'))
    if len(color_files) < 36:
        print(f'图片不足: {len(color_files)} 张')
        return

    # ── Step 1: YOLO mask ──
    print('\n[1/5] 生成 YOLO mask...')
    yolo_obj = YOLO('yolov8n-seg.pt')

    colmap_img_dir = OUT / 'colmap/images'
    colmap_mask_dir = OUT / 'colmap/masks'
    colmap_img_dir.mkdir(parents=True, exist_ok=True)
    colmap_mask_dir.mkdir(parents=True, exist_ok=True)

    detected = 0
    masks_dict = {}
    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None: continue

        results = yolo_obj(str(cf), classes=[56], verbose=False)
        r = results[0]

        mask_binary = np.zeros((cH, cW), dtype=np.uint8)
        if r.boxes is not None and len(r.boxes) > 0:
            conf = r.boxes.conf[0].item()
            if conf >= 0.5:
                detected += 1
                mask_raw = r.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_raw, (cW, cH))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

        cv2.imwrite(str(colmap_mask_dir / f'{i:03d}.png'), mask_binary)
        masks_dict[i] = mask_binary

        cv2.imwrite(str(colmap_img_dir / f'{i:03d}.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        if i < 3 or (i + 1) % 18 == 0:
            print(f'  [{i+1}/{len(color_files)}] {i*step:3d}deg  '
                  f'mask={(mask_binary>0).sum()//1000}k px  {"✓" if mask_binary.any() else "✗"}')

    print(f'\n  YOLO 检出: {detected}/{len(color_files)}')

    # ── 自动测量椅子尺寸 ──
    chair_dims = measure_chair_from_masks(masks_dict, calib)

    # ── Step 2: 用转台标定计算每帧相机位姿 ──
    print('\n[2/5] 计算相机位姿 (转台标定)...')

    R_calib = np.array(calib['R'])  # camera → turntable plane
    t_calib = np.array(calib['t'])  # plane center in camera coords
    R_plane = R_calib.T             # world(turntable) → camera
    t_world = t_calib

    # 写 COLMAP sparse model (直接用 TXT 格式，跳过 model_converter)
    sparse_model_dir = OUT / 'colmap/sparse'
    if sparse_model_dir.is_dir(): shutil.rmtree(str(sparse_model_dir))
    sparse_model_dir.mkdir(parents=True, exist_ok=True)

    f_mean = (fx + fy) / 2.0
    with open(sparse_model_dir / 'cameras.txt', 'w') as f:
        f.write('# Camera list\n')
        f.write(f'1 SIMPLE_RADIAL {cW} {cH} {f_mean:.6f} {ppx:.6f} {ppy:.6f} 0.0\n')

    with open(sparse_model_dir / 'images.txt', 'w') as f:
        f.write('# Image list\n')
        for i in range(n_frames):
            deg = i * step
            theta = np.radians(deg)
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta),  np.cos(theta), 0],
                           [0,              0,             1]])
            R_i = R_plane @ Rz
            t_i = t_world.copy()

            q = rotation_to_quaternion(R_i)  # [w, x, y, z]
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            tx, ty, tz = t_i[0], t_i[1], t_i[2]
            f.write(f'{i+1} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} '
                    f'{tx:.10f} {ty:.10f} {tz:.10f} 1 {i:03d}.jpg\n\n')

    with open(sparse_model_dir / 'points3D.txt', 'w') as f:
        f.write('# Empty\n')

    print(f'  已生成 {n_frames} 个相机位姿')
    print(f'  内参: f={f_mean:.1f} {cW}x{cH}')
    print(f'  稀疏模型 (TXT): {sparse_model_dir}')

    # ── Step 3: YOLO mask 抠图 ──
    print('\n[3/5] 应用 YOLO mask → 抠图...')
    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None: continue
        if i in masks_dict and masks_dict[i].max() > 0:
            mask_3ch = masks_dict[i][:, :, None] / 255.0
            masked = (img.astype(np.float32) * mask_3ch).astype(np.uint8)
        else:
            masked = np.zeros_like(img)
        cv2.imwrite(str(colmap_img_dir / f'{i:03d}.jpg'), masked,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f'  已覆盖 {len(color_files)} 张图像')

    # ── Step 4: COLMAP MVS ──
    print('\n[4/5] COLMAP 密集重建 (抠图)...')
    dense_dir = str(OUT / 'colmap/dense')
    if os.path.isdir(dense_dir): shutil.rmtree(dense_dir)

    print('  undistortion...')
    t0 = time.time()
    r = subprocess.run(['colmap', 'image_undistorter',
                        '--image_path', str(colmap_img_dir),
                        '--input_path', str(sparse_model_dir),
                        '--output_path', dense_dir,
                        '--output_type', 'COLMAP'],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f'  去畸变失败: {r.stderr[-300:]}')
        return
    print(f'  完成 ({time.time()-t0:.0f}s)')

    print('  patch_match_stereo (CPU, 预计 20-40 分钟)...')
    t0 = time.time()
    cmd = ['colmap', 'patch_match_stereo',
           '--workspace_path', dense_dir,
           '--PatchMatchStereo.window_radius', '5',
           '--PatchMatchStereo.num_samples', '11',
           '--PatchMatchStereo.num_iterations', '5',
           '--PatchMatchStereo.geom_consistency', '1',
           '--PatchMatchStereo.max_image_size', '1200']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)
    if r.returncode != 0:
        print(f'  立体匹配失败: {r.stderr[-300:]}')
        return
    print(f'  完成 ({time.time()-t0:.0f}s)')

    print('  stereo_fusion...')
    t0 = time.time()
    fused_ply = str(OUT / 'dense_fused.ply')
    r = subprocess.run(['colmap', 'stereo_fusion',
                        '--workspace_path', dense_dir,
                        '--output_path', fused_ply],
                       capture_output=True, text=True, timeout=1800)
    print(r.stdout[-300:] if len(r.stdout) > 300 else r.stdout)
    if r.returncode != 0:
        print(f'  深度融合失败: {r.stderr[-300:]}')
        return
    print(f'  完成 ({time.time()-t0:.0f}s)')

    # ── Step 5: Poisson + 后处理 ──
    print('\n[5/5] Poisson 网格 + 后处理...')
    pcd = o3d.io.read_point_cloud(fused_ply)
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print('错误: 空点云')
        return
    print(f'MVS 点云: {len(points):,} 点')

    # 几何过滤兜底
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=500)
    n = np.array(plane_model[:3])
    d_val = plane_model[3]
    if n[2] < 0:
        n = -n; d_val = -d_val

    dists = points @ n + d_val
    above = dists > 0.003
    chair_h = calib.get('chair_height_m', 0.15)
    within_height = dists < chair_h
    z_axis = n / np.linalg.norm(n)
    x_axis = np.cross(np.array([0., 0., 1.]), z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([1., 0., 0.]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R_plane_local = np.column_stack([x_axis, y_axis, z_axis])
    pts_centered = points - (-d_val * n)
    pts_local = pts_centered @ R_plane_local
    r2 = pts_local[:, 0]**2 + pts_local[:, 1]**2
    chair_r = max(calib.get('chair_width_m', 0.15), calib.get('chair_depth_m', 0.15)) / 2
    within_radius = r2 < chair_r**2

    keep = above & within_height & within_radius
    filtered_points = points[keep]
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f'几何过滤后: {len(filtered_points):,} 点')

    if len(filtered_points) < 100:
        print('点数不足')
        return

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[keep])

    print('  poisson_mesher...')
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(filtered_pcd, depth=8)

    # PCA 方向对齐
    verts = np.asarray(mesh.vertices)
    cov = np.cov(verts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, -1]
    target = np.array([0., 1., 0.])
    rot_axis = np.cross(main_axis, target)
    if np.linalg.norm(rot_axis) > 1e-9:
        rot_axis /= np.linalg.norm(rot_axis)
        angle = np.arccos(np.clip(np.dot(main_axis, target), -1, 1))
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                      [rot_axis[2], 0, -rot_axis[0]],
                      [-rot_axis[1], rot_axis[0], 0]])
        R_align = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    else:
        R_align = np.eye(3) if np.dot(main_axis, target) > 0 else np.diag([1, -1, -1])

    verts_aligned = (R_align @ verts.T).T
    verts_aligned[:, 1] -= verts_aligned[:, 1].min()
    mesh.vertices = o3d.utility.Vector3dVector(verts_aligned)
    mesh.compute_vertex_normals()

    labels, counts, _ = mesh.cluster_connected_triangles()
    if len(counts) > 1:
        mesh.remove_triangles_by_index(np.where(labels != np.argmax(counts))[0])
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(OUT / 'chair.obj'), mesh)
    o3d.io.write_triangle_mesh(str(OUT / 'chair.ply'), mesh)

    verts_out = np.asarray(mesh.vertices)
    print(f'\n{"=" * 50}')
    print(f'  输出: {OUT}/chair.obj')
    for i, a in enumerate('XYZ'):
        span = np.ptp(verts_out[:, i]) * 100
        print(f'  {a}: span={span:.1f}cm  [{verts_out[:,i].min():.3f}, {verts_out[:,i].max():.3f}]')
    cw, cd, ch = calib.get('chair_width_m', 0), calib.get('chair_depth_m', 0), calib.get('chair_height_m', 0)
    print(f'  实测: 宽={cw*100:.1f}cm  高={ch*100:.1f}cm  深={cd*100:.1f}cm (YOLO自动测量)')
    print(f'  顶点: {len(verts_out):,}  面: {len(np.asarray(mesh.triangles)):,}')
    print(f'{"=" * 50}')

    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'method': 'V11_KNOWN_POSE_YOLO_MVS',
           'n_frames': len(color_files), 'yolo_detected': detected,
           'raw_points': len(points), 'filtered_points': len(filtered_points),
           'vertices': len(verts_out),
           'faces': len(np.asarray(mesh.triangles))}
    with open(OUT / 'log_fusion.json', 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    import sys

    calib_path = OUT / 'calibrate.json'
    ts_calib_path = BASE / 'output/calibrate.json'

    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        if 'fx_full' in calib:
            print(f'已有 V11 标定: {calib_path}')
            resp = input('使用现有标定? [Y/n]: ').strip().lower()
            if resp == 'y':
                calib = calib
            else:
                calib = None

    if calib is None:
        if not ts_calib_path.exists():
            print(f'错误: 未找到 turntable_set_v2 标定文件 {ts_calib_path}')
            print('请先运行 turntable_set_v2.py 完成标定（按 S 保存）')
            sys.exit(1)

        with open(ts_calib_path) as f:
            ts_calib = json.load(f)
        print(f'读取 turntable_set_v2 标定: {ts_calib_path}')
        print(f'  R={ts_calib["R"]}')
        print(f'  t={ts_calib["t"]}')
        print(f'  radius={ts_calib["radius_m"]*100:.1f}cm')

        calib = {
            'R': ts_calib['R'], 't': ts_calib['t'],
            'radius_m': ts_calib['radius_m'],
            'cx': ts_calib.get('cx', 0.0), 'cy': ts_calib.get('cy', 0.0),
            'step_deg': STEP_DEG, 'n_frames': N_FRAMES,
        }

    resp = input('\n开始采集? [Y/n]: ').strip().lower()
    if resp != 'n':
        do_capture(calib)
    else:
        print('跳过采集')

    do_fusion(calib)
