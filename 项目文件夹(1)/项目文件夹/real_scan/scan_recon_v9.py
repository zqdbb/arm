#!/usr/bin/env python3
"""
V9 椅子重建: YOLOv8-seg Visual Hull 空间雕刻
核心: 72个角度的YOLO 2D mask → 反投影圆锥 → 3D交集 = 椅子体积
      完全不需要 D435i 深度，只用 RGB + 相机位姿
"""

import cv2, json, time, os
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d

os.environ['QT_QPA_PLATFORM'] = 'xcb'

BASE = Path(__file__).parent
OUT = BASE / 'output/v9'
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150
DISPARITY_SHIFT = 150
DIGITAL_ZOOM = 2.0

# V9 Visual Hull 参数
VOXEL_SIZE = 0.0015        # 1.5mm 体素
VOL_X = 0.06               # X ±6cm
VOL_Y = 0.06               # Y ±6cm
VOL_Z_MIN = -0.12           # 盘面下方 12cm（椅子在负Z方向，高~9cm）
VOL_Z_MAX = 0.03            # 盘面上方 3cm
YOLO_CONF_THRESHOLD = 0.5

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'color').mkdir(exist_ok=True)
(OUT / 'depth').mkdir(exist_ok=True)
(OUT / 'depth_color').mkdir(exist_ok=True)


def get_cropped_intrinsics(fx, fy, ppx, ppy):
    crop_w = int(W / DIGITAL_ZOOM)
    crop_h = int(H / DIGITAL_ZOOM)
    crop_x = (W - crop_w) // 2
    crop_y = (H - crop_h) // 2
    return {
        'fx': fx, 'fy': fy,
        'ppx': ppx - crop_x, 'ppy': ppy - crop_y,
        'width': crop_w, 'height': crop_h,
        'crop_x': crop_x, 'crop_y': crop_y
    }


def crop_frame(color, depth=None):
    crop_w = int(W / DIGITAL_ZOOM)
    crop_h = int(H / DIGITAL_ZOOM)
    crop_x = (W - crop_w) // 2
    crop_y = (H - crop_h) // 2
    color_crop = color[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    if depth is None:
        return color_crop
    depth_crop = depth[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    return color_crop, depth_crop


def set_disparity_shift(value=None):
    if value is None:
        value = DISPARITY_SHIFT
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print('警告: 未找到 RealSense 设备')
            return False
        device = devices[0]
        advnc = rs.rs400_advanced_mode(device)
        if not advnc.is_enabled():
            print('启用 advanced mode...')
            advnc.toggle_advanced_mode(True)
            time.sleep(1)
        dt = advnc.get_depth_table()
        dt.disparityShift = value
        advnc.set_depth_table(dt)
        print(f'disparityShift 设为 {value}')
        return True
    except Exception as e:
        print(f'警告: 无法设置 disparityShift: {e}')
        return False


def do_calibrate():
    """标定: 同 V6/V7/V8"""
    print('\n' + '=' * 50)
    print('  V9 标定: D435i 转台标定')
    print('=' * 50)

    set_disparity_shift(0)
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    dscale = ds.get_depth_scale()
    print(f'fx={fx:.1f} fy={fy:.1f} scale={dscale:.6f}')

    ci = get_cropped_intrinsics(fx, fy, ppx, ppy)
    fx_c, fy_c = ci['fx'], ci['fy']
    ppx_c, ppy_c = ci['ppx'], ci['ppy']
    cW, cH = ci['width'], ci['height']
    print(f'数字变焦 {DIGITAL_ZOOM}x: {W}x{H} -> {cW}x{cH}')

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    print('\nSPACE=标定  S=保存  Q=退出')
    cv2.namedWindow('calib', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('calib', cW * 2, cH * 2)

    calib = None
    while True:
        frames = pipe.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)
        color = np.asanyarray(aligned.get_color_frame().get_data())
        depth_frame = aligned.get_depth_frame()
        if not depth_frame:
            continue
        depth = np.asanyarray(depth_frame.get_data())

        color_crop, depth_crop = crop_frame(color, depth)
        depth_m = depth_crop.astype(np.float64) * dscale

        valid = depth_m > 0.01
        if valid.sum() < 500:
            cv2.imshow('calib', color_crop)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'): break
            continue

        vv, uu = np.mgrid[0:cH, 0:cW]
        z_cam = depth_m[valid]
        x_cam = (uu[valid].astype(np.float64) - ppx_c) / fx_c * z_cam
        y_cam = (vv[valid].astype(np.float64) - ppy_c) / fy_c * z_cam

        pcd_raw = o3d.geometry.PointCloud()
        pcd_raw.points = o3d.utility.Vector3dVector(np.column_stack([x_cam, y_cam, z_cam]))

        plane_model, inliers = pcd_raw.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=500)
        n = plane_model[:3]
        d_val = plane_model[3]
        if n[2] < 0:
            n = -n
            d_val = -d_val

        R_plane = np.eye(3)
        z_axis = n / np.linalg.norm(n)
        x_axis = np.cross(np.array([0., 1., 0.]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1., 0., 0.]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_plane = np.column_stack([x_axis, y_axis, z_axis])

        R_calib = R_plane.T
        t_calib = -d_val * n

        pts3d = np.column_stack([x_cam[inliers], y_cam[inliers], z_cam[inliers]])

        def detect_center_blob(gray_img):
            th = np.percentile(gray_img, 20)
            dark = (gray_img < max(th, 20)).astype(np.uint8) * 255
            n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
            if n_labels <= 1:
                return None, 0
            best_label = None
            best_score = -1
            cy_img, cx_img = cH / 2, cW / 2
            for lb in range(1, n_labels):
                area = stats[lb, cv2.CC_STAT_AREA]
                if area < 20 or area > 5000:
                    continue
                cx_b, cy_b = centroids[lb]
                dist = np.hypot(cx_b - cx_img, cy_b - cy_img)
                if dist > min(cW, cH) * 0.4:
                    continue
                mask_lb = (labels == lb)
                mean_brightness = gray_img[mask_lb].mean()
                score = (255 - mean_brightness) * 0.6 + (1.0 - dist / (min(cW, cH) * 0.4)) * 0.4
                if score > best_score:
                    best_score = score
                    best_label = lb
            if best_label is not None:
                return centroids[best_label], stats[best_label, cv2.CC_STAT_AREA]
            return None, 0

        def detect_disc_edge(gray_img, cx_px, cy_px):
            edge_pts = []
            disc_brightness = np.median(gray_img)
            for ang in range(0, 360, 2):
                rad = np.radians(ang)
                for r in range(5, min(cW, cH) // 2):
                    px = int(cx_px + r * np.cos(rad))
                    py = int(cy_px + r * np.sin(rad))
                    if px < 0 or px >= cW or py < 0 or py >= cH:
                        break
                    if gray_img[py, px] < disc_brightness * 0.55:
                        edge_pts.append([px, py])
                        break
            return edge_pts

        def ray_plane_intersect(px, py, n, d_val, fx, fy, ppx, ppy):
            ray = np.array([(px - ppx) / fx, (py - ppy) / fy, 1.0])
            ray /= np.linalg.norm(ray)
            denom = np.dot(n, ray)
            if abs(denom) < 1e-8:
                return None
            t = -d_val / denom
            if t <= 0:
                return None
            return ray * t

        gray = cv2.cvtColor(color_crop, cv2.COLOR_BGR2GRAY)
        center_result = detect_center_blob(gray)
        center_px = None
        cx, cy = 0.0, 0.0

        if center_result[0] is not None:
            center_px = center_result[0]
            cp3d = ray_plane_intersect(center_px[0], center_px[1], n, d_val, fx_c, fy_c, ppx_c, ppy_c)
            if cp3d is not None:
                cpl = R_plane @ cp3d
                cx, cy = cpl[0], cpl[1]

        edge_pts_raw = detect_disc_edge(gray, center_px[0] if center_px is not None else cW/2,
                                        center_px[1] if center_px is not None else cH/2)

        radius = 0.05
        if len(edge_pts_raw) >= 15 and center_px is not None:
            ed = np.array(edge_pts_raw)
            px_dists = np.hypot(ed[:, 0] - center_px[0], ed[:, 1] - center_px[1])
            med_px = np.median(px_dists)
            good = np.abs(px_dists - med_px) < med_px * 0.4
            edge_pts = ed[good].tolist()
            edge_3d_dists = []
            for pe in edge_pts:
                pe3d = ray_plane_intersect(pe[0], pe[1], n, d_val, fx_c, fy_c, ppx_c, ppy_c)
                if pe3d is not None:
                    pel = R_plane @ pe3d
                    edge_3d_dists.append(np.hypot(pel[0] - cx, pel[1] - cy))
            if len(edge_3d_dists) >= 8:
                radius = float(np.median(edge_3d_dists))
                radius = max(radius, 0.03)
                radius = min(radius, 0.15)

        vimg = color_crop.copy()
        vimg = cv2.cvtColor(vimg, cv2.COLOR_RGB2BGR)

        for idx in inliers:
            u = int(uu.ravel()[valid.ravel()][idx])
            v_i = int(vv.ravel()[valid.ravel()][idx])
            if 0 <= u < cW and 0 <= v_i < cH:
                vimg[v_i, u] = [0, 255, 0]

        for pe in edge_pts_raw:
            if 0 <= pe[0] < cW and 0 <= pe[1] < cH:
                vimg[pe[1], pe[0]] = [0, 0, 255]

        if center_px is not None:
            th = np.linspace(0, 2 * np.pi, 72)
            p_w_circle = np.column_stack([cx + radius * np.cos(th),
                                          cy + radius * np.sin(th),
                                          np.zeros(72)])
            p_c_circle = (R_calib @ p_w_circle.T).T + t_calib
            fvis = p_c_circle[:, 2] > 0.01
            if fvis.sum() > 6:
                u_c = (fx_c * p_c_circle[fvis, 0] / p_c_circle[fvis, 2] + ppx_c).astype(int)
                v_c = (fy_c * p_c_circle[fvis, 1] / p_c_circle[fvis, 2] + ppy_c).astype(int)
                for i in range(len(u_c)):
                    j = (i + 1) % len(u_c)
                    if np.hypot(u_c[i] - u_c[j], v_c[i] - v_c[j]) < 300:
                        cv2.line(vimg, (u_c[i], v_c[i]), (u_c[j], v_c[j]), (255, 0, 0), 2)
            cv2.circle(vimg, (int(center_px[0]), int(center_px[1])), 4, (0, 255, 255), -1)

        print(f'\r  3D点: {valid.sum():,}  '
              f'黑点: ({center_px[0]:.1f},{center_px[1]:.1f}) -> 局({cx*100:.1f},{cy*100:.1f})cm  '
              f'边缘: {len(edge_pts_raw)} R={radius*100:.1f}cm  '
              f'S=保存  SPACE=重标', end='')

        cv2.imshow('calib', vimg)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s') and len(edge_pts_raw) >= 20:
            calib = {'R': R_calib.tolist(), 't': t_calib.tolist(), 'radius_m': radius,
                     'cx': float(cx), 'cy': float(cy),
                     'fx': fx_c, 'fy': fy_c, 'ppx': ppx_c, 'ppy': ppy_c,
                     'width': cW, 'height': cH,
                     'step_deg': STEP_DEG, 'n_frames': N_FRAMES,
                     'digital_zoom': DIGITAL_ZOOM, 'disparity_shift': DISPARITY_SHIFT,
                     'depth_scale': dscale}
            with open(OUT / 'calibrate.json', 'w') as f:
                json.dump(calib, f, indent=2)
            print(f'\n标定保存: {OUT}/calibrate.json')
            break
        elif key == ord(' ') or key == 13:
            print('\n重标...')
        elif key == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()
    return calib


def do_capture(calib):
    """采集: 同 V6/V7/V8"""
    print('\n' + '=' * 50)
    print(f'  采集: {STEP_DEG}deg x {N_FRAMES} 帧')
    print('=' * 50)

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    set_disparity_shift(0)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)
    dscale = ds.get_depth_scale()
    pipe.wait_for_frames(timeout_ms=10000)
    print(f'相机就绪')

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
            if not df or not cf:
                continue
            d_sum += np.asanyarray(df.get_data()).astype(np.float64)
            if color_full is None:
                color_full = np.asanyarray(cf.get_data())
            n_good += 1

        if n_good == 0:
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  无数据')
            continue

        depth_mm_full = (d_sum / n_good) * dscale * 1000.0
        depth_u16 = depth_mm_full.astype(np.uint16)

        color, depth_mm = crop_frame(color_full, depth_mm_full)
        depth_u16_crop = depth_mm.astype(np.uint16)

        dv = np.clip(depth_mm, 50, 500)
        dv8 = (dv / 500 * 255).astype(np.uint8)
        dv_color = cv2.applyColorMap(dv8, cv2.COLORMAP_JET)

        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'), depth_u16_crop)
        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.png'), color)
        cv2.imwrite(str(OUT / 'depth_color' / f'{i:03d}.jpg'), dv_color)

        dvals = depth_mm[depth_mm > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  '
              f'valid:{len(dvals)//1000}k  '
              f'depth∈[{dvals.min():.0f},{dvals.max():.0f}]mm  '
              f'med={np.median(dvals):.0f}mm  {el:.0f}s')

    pipe.stop()
    print(f'采集完成 ({time.time()-t0:.0f}s)')
    if tt:
        tt.close()


def do_fusion(calib):
    """V9 融合: YOLO Visual Hull 空间雕刻"""
    from ultralytics import YOLO

    print('\n' + '=' * 50)
    print('  V9 YOLO Visual Hull 空间雕刻')
    print('=' * 50)

    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])
    fx, fy = calib['fx'], calib['fy']
    ppx, ppy = calib['ppx'], calib['ppy']
    cW = calib['width']; cH = calib['height']
    step = calib.get('step_deg', STEP_DEG)
    cx_disc = calib.get('cx', 0.0)
    cy_disc = calib.get('cy', 0.0)

    color_files = sorted([f for f in (OUT / 'color').glob('*.png') if len(f.stem) == 3])
    n_frames = min(N_FRAMES, len(color_files))
    if n_frames == 0:
        print('无数据!')
        return

    # ── 加载 YOLO ──
    print('加载 YOLOv8n-seg...')
    yolo = YOLO('yolov8n-seg.pt')

    # ── 获取所有帧的 YOLO mask ──
    print(f'\n逐帧 YOLO 推理 ({n_frames} 帧)...')
    masks = []
    angles = []
    detected = 0
    for i in range(n_frames):
        results = yolo(str(color_files[i]), classes=[56], verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        conf = r.boxes.conf[0].item()
        if conf < YOLO_CONF_THRESHOLD:
            continue
        mask_raw = r.masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (cW, cH))
        mask_binary = (mask_resized > 0.7).astype(np.uint8)
        mask_binary = cv2.erode(mask_binary, np.ones((3, 3), np.uint8), iterations=1)
        if mask_binary.sum() < 100:
            continue
        masks.append(mask_binary)
        angles.append(np.radians(i * step))
        detected += 1
        if i < 3 or (i + 1) % 18 == 0:
            print(f'  [{i+1}/{n_frames}] {i*step:3d}deg  '
                  f'mask={mask_binary.sum():,}px  conf={conf:.2f}')

    print(f'\nYOLO 检出: {detected}/{n_frames} 帧')

    if detected < 10:
        print('检出帧不足')
        return

    # ── 创建体素网格 ──
    print(f'\n创建体素网格 ({VOXEL_SIZE*1000:.0f}mm)...')
    nx = int(2 * VOL_X / VOXEL_SIZE)
    ny = int(2 * VOL_Y / VOXEL_SIZE)
    nz = int((VOL_Z_MAX - VOL_Z_MIN) / VOXEL_SIZE)
    print(f'  网格: {nx} x {ny} x {nz} = {nx*ny*nz:,} 体素')
    print(f'  范围: X=[{-VOL_X*100:.0f},{VOL_X*100:.0f}] Y=[{-VOL_Y*100:.0f},{VOL_Y*100:.0f}] '
          f'Z=[{VOL_Z_MIN*100:.0f},{VOL_Z_MAX*100:.0f}]cm')

    x_vals = np.linspace(-VOL_X, VOL_X, nx)
    y_vals = np.linspace(-VOL_Y, VOL_Y, ny)
    z_vals = np.linspace(VOL_Z_MIN, VOL_Z_MAX, nz)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    voxels_obj = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (N, 3)
    n_voxels = len(voxels_obj)
    print(f'  体素总数: {n_voxels:,}')

    # ── 空间雕刻（投票制）──
    print(f'\n空间雕刻 ({detected} 视角, 投票阈值>={int(detected*0.75)}/{detected})...')
    # 投票: 每个体素统计在多少视角的 mask 内
    vote_count = np.zeros(n_voxels, dtype=np.int32)

    for idx, (mask, theta) in enumerate(zip(masks, angles)):
        # 物体坐标 → 相机坐标
        c, s = np.cos(theta), np.sin(theta)
        R_z_pos = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        p_centered = (R_z_pos @ voxels_obj.T).T  # (N, 3)

        p_world = p_centered.copy()
        p_world[:, 0] += cx_disc
        p_world[:, 1] += cy_disc

        p_cam = (R_calib @ p_world.T).T + t_calib  # (N, 3)

        # 投影到图像
        z_cam = p_cam[:, 2]
        in_front = z_cam > 0.01
        u = np.zeros(n_voxels, dtype=np.float64)
        v = np.zeros(n_voxels, dtype=np.float64)
        u[in_front] = fx * p_cam[in_front, 0] / z_cam[in_front] + ppx
        v[in_front] = fy * p_cam[in_front, 1] / z_cam[in_front] + ppy

        # 检查是否在 mask 内
        in_image = in_front & (u >= 0) & (u < cW) & (v >= 0) & (v < cH)
        ui = np.clip(u[in_image].astype(int), 0, cW - 1)
        vi = np.clip(v[in_image].astype(int), 0, cH - 1)
        mask_hit = np.zeros(n_voxels, dtype=bool)
        mask_hit[in_image] = mask[vi, ui] > 0

        vote_count += mask_hit.astype(np.int32)

        if (idx + 1) % 18 == 0 or idx < 3:
            n_keep = (vote_count >= max(int(detected * 0.75), 3)).sum()
            print(f'  [{idx+1}/{detected}] {np.degrees(theta):.0f}deg  '
                  f'投票>75%体素: {n_keep:,} ({n_keep/n_voxels*100:.1f}%)')

    vote_threshold = max(int(detected * 0.75), 3)
    inside = vote_count >= vote_threshold

    print(f'\n雕刻完成: {inside.sum():,} 体素 ({inside.sum()/n_voxels*100:.1f}%)')

    if inside.sum() < 10:
        print('错误: 无剩余体素')
        return

    # ── 提取表面点 → Poisson ──
    hull_pts = voxels_obj[inside]

    # 提取表面体素（至少有一个邻居在外部）
    print('提取表面体素...')
    shape = (nx, ny, nz)
    inside_grid = inside.reshape(shape)

    # 形态学边缘: 膨胀 → 与原体素做差
    from scipy import ndimage
    eroded = ndimage.binary_erosion(inside_grid, iterations=1)
    surface_grid = inside_grid & ~eroded
    surface_mask = surface_grid.ravel()
    surface_pts = voxels_obj[surface_mask]
    print(f'  表面体素: {surface_mask.sum():,}')

    # 体素中心作为点云
    pcd_hull = o3d.geometry.PointCloud()
    pcd_hull.points = o3d.utility.Vector3dVector(surface_pts)

    # 方向修正: Z↑ → Y↑
    pts = np.asarray(pcd_hull.points)
    pts_out = np.zeros_like(pts)
    pts_out[:, 0] = pts[:, 0]
    pts_out[:, 1] = pts[:, 2]
    pts_out[:, 2] = pts[:, 1]
    pts_out[:, 2] -= pts_out[:, 2].min()

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_out)
    o3d.io.write_point_cloud(str(OUT / 'hull_points.ply'), pcd_out)

    # ── 保存完整体素（调试） ──
    pcd_full = o3d.geometry.PointCloud()
    pts_full = hull_pts.copy()
    pts_full_out = np.zeros_like(pts_full)
    pts_full_out[:, 0] = pts_full[:, 0]
    pts_full_out[:, 1] = pts_full[:, 2]
    pts_full_out[:, 2] = pts_full[:, 1]
    pts_full_out[:, 2] -= pts_full_out[:, 2].min()
    pcd_full.points = o3d.utility.Vector3dVector(pts_full_out)
    o3d.io.write_point_cloud(str(OUT / 'hull_full.ply'), pcd_full)

    # ── Poisson 表面重建 ──
    print(f'\nPoisson 表面重建 (表面 {len(pcd_out.points):,} 点)...')
    pcd_out.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 3, max_nn=30))
    pcd_out.orient_normals_towards_camera_location(np.array([0., 2., 0.]))

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_out, depth=8, width=0, scale=1.1, linear_fit=False)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
    mesh.remove_unreferenced_vertices()

    labels_m, counts_m, _ = mesh.cluster_connected_triangles()
    if len(counts_m) > 1:
        mesh.remove_triangles_by_index(np.where(labels_m != np.argmax(counts_m))[0])
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(OUT / 'chair.ply'), mesh)
    o3d.io.write_triangle_mesh(str(OUT / 'chair.obj'), mesh)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    print(f'\n{"=" * 50}')
    print(f'  输出: {OUT}/chair.obj')
    print(f'  方向: Y=↑')
    for i, a in enumerate('XYZ'):
        print(f'  {a}: [{verts[:,i].min():.3f}, {verts[:,i].max():.3f}] '
              f'span={np.ptp(verts[:,i])*100:.1f}cm')
    y_span = np.ptp(verts[:, 1]) * 100
    print(f'  预期: Y≈8.8cm(高)  X≈4.6cm  Z≈4.0cm')
    print(f'  实际: Y={y_span:.1f}cm(高)')
    print(f'  顶点: {len(verts):,}  面: {len(faces):,}')
    print(f'{"=" * 50}')

    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'method': 'V9_YOLO_VisualHull',
           'n_frames': n_frames, 'detected_frames': detected,
           'voxel_size_mm': VOXEL_SIZE * 1000,
           'volume_x_cm': VOL_X * 200, 'volume_y_cm': VOL_Y * 200,
           'volume_z_range_cm': [VOL_Z_MIN * 100, VOL_Z_MAX * 100],
           'hull_voxels': int(inside.sum()),
           'surface_pts': len(surface_pts),
           'vertices': len(verts),
           'faces': len(faces) if len(faces) > 0 else 0}
    with open(OUT / 'log_fusion.json', 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    import sys

    calib_path = OUT / 'calibrate.json'
    v6_calib_path = BASE / 'output/v6/calibrate.json'
    has_depth = len(list((OUT / 'depth').glob('*.png'))) >= 36

    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        print(f'加载 V9 标定: {calib_path}')
    elif v6_calib_path.exists():
        import shutil
        shutil.copy(v6_calib_path, calib_path)
        with open(calib_path) as f:
            calib = json.load(f)
        print(f'从 V6 迁移标定: {v6_calib_path} -> {calib_path}')
    else:
        calib = do_calibrate()
        if calib is None:
            print('标定取消')
            sys.exit(0)

    # 复用 V6 数据
    v6_color = list((BASE / 'output/v6/color').glob('*.png'))
    v9_color = list((OUT / 'color').glob('*.png'))
    if len(v9_color) < 36 and len(v6_color) >= 36:
        print(f'从 V6 复用彩色数据 ({len(v6_color)} 帧)')
        for sub in ['color']:
            v6_sub = BASE / 'output/v6' / sub
            v9_sub = OUT / sub
            if v6_sub.exists() and not any(v9_sub.iterdir()):
                for f in v6_sub.iterdir():
                    (v9_sub / f.name).symlink_to(f.resolve())

    resp = input('跳过采集? [Y/n]: ').strip().lower()
    if resp == 'n':
        do_capture(calib)
    else:
        print('跳过采集\n')

    do_fusion(calib)
