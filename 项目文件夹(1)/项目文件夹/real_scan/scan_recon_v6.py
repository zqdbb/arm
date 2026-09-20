#!/usr/bin/env python3
"""
V6 D435i Disparity Shift + 深度差分 + 点云融合
核心: 调 disparityShift 让 D435i 能拍到 20cm 处的椅子 → TSDF融合

使用: python3 scan_recon_v6.py
"""

import cv2, json, time, os
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d

os.environ['QT_QPA_PLATFORM'] = 'xcb'

BASE = Path(__file__).parent
OUT = BASE / 'output/v6'
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150
DISPARITY_SHIFT = 150  # 关键：降低最近深度到 ~10cm
DIGITAL_ZOOM = 2.0      # 中心裁剪倍数

# TSDF/融合参数
VOXEL_SIZE = 0.001        # 1mm 体素
COLOR_BRIGHT_THRESHOLD = 0.50  # 灰度>0.50 = 椅子（局部搜索窗内，椅最亮）
DEPTH_RANGE_MARGIN = 0.15  # 深度中值±15cm
CHAIR_SEARCH_RADIUS = 40   # 盘心周围搜索半径（px），椅子~4cm≈50px

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'color').mkdir(exist_ok=True)
(OUT / 'depth').mkdir(exist_ok=True)
(OUT / 'depth_color').mkdir(exist_ok=True)


def get_cropped_intrinsics(fx, fy, ppx, ppy):
    """返回中心裁剪后的内参"""
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
    """中心裁剪 color 和 depth 帧"""
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
    """在启动 pipeline 前设置 disparity shift。value=None 使用默认 DISPARITY_SHIFT"""
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
        print(f'disparityShift 设为 {value} (最近深度 ~10cm)' if value > 0
              else f'disparityShift 设为 {value} (默认)')
        return True
    except Exception as e:
        print(f'警告: 无法设置 disparityShift: {e}')
        return False


def do_calibrate():
    print('\n' + '=' * 50)
    print('  V6 标定: D435i 转台标定')
    print('=' * 50)

    # 标定前关闭 disparity shift，保证深度质量
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

    # 裁剪内参
    ci = get_cropped_intrinsics(fx, fy, ppx, ppy)
    fx_c, fy_c = ci['fx'], ci['fy']
    ppx_c, ppy_c = ci['ppx'], ci['ppy']
    cW, cH = ci['width'], ci['height']
    print(f'数字变焦 {DIGITAL_ZOOM}x: {W}×{H} → {cW}×{cH}')

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

        # RANSAC 找平面
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
        d = plane_model[3]
        if n[2] < 0:
            n = -n
            d = -d

        # 相机外参
        R_plane = np.eye(3)
        z_axis = n / np.linalg.norm(n)
        x_axis = np.cross(np.array([0., 1., 0.]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1., 0., 0.]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_plane = np.column_stack([x_axis, y_axis, z_axis])

        R_calib = R_plane.T
        t_calib = -d * n
        n_cam = R_calib[:, 2]

        pts3d = np.column_stack([x_cam[inliers], y_cam[inliers], z_cam[inliers]])
        p_w = (R_calib @ pts3d.T).T + t_calib

        # 黑点检测（盘面圆心）
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
                if area < 10 or area > 20000:
                    continue
                cx_b, cy_b = centroids[lb]
                dist = np.hypot(cx_b - cx_img, cy_b - cy_img)
                if dist > min(cW, cH) * 0.49:
                    continue

                mask_lb = (labels == lb)
                mean_brightness = gray_img[mask_lb].mean()
                score = (255 - mean_brightness) * 0.6 + (1.0 - dist / (min(cW, cH) * 0.49)) * 0.4
                if score > best_score:
                    best_score = score
                    best_label = lb

            if best_label is not None:
                return centroids[best_label], stats[best_label, cv2.CC_STAT_AREA]
            return None, 0

        # 边缘检测（亮度突变）
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

        # 射线-平面求交
        def ray_plane_intersect(px, py, n, d, fx, fy, ppx, ppy):
            ray = np.array([(px - ppx) / fx, (py - ppy) / fy, 1.0])
            ray /= np.linalg.norm(ray)
            denom = np.dot(n, ray)
            if abs(denom) < 1e-8:
                return None
            t = -d / denom
            if t <= 0:
                return None
            return ray * t

        gray = cv2.cvtColor(color_crop, cv2.COLOR_BGR2GRAY)

        center_result = detect_center_blob(gray)
        center_px = None
        cx, cy = 0.0, 0.0

        if center_result[0] is not None:
            center_px = center_result[0]
            cp3d = ray_plane_intersect(center_px[0], center_px[1], n, d, fx_c, fy_c, ppx_c, ppy_c)
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
                pe3d = ray_plane_intersect(pe[0], pe[1], n, d, fx_c, fy_c, ppx_c, ppy_c)
                if pe3d is not None:
                    pel = R_plane @ pe3d
                    edge_3d_dists.append(np.hypot(pel[0] - cx, pel[1] - cy))
            if len(edge_3d_dists) >= 8:
                radius = float(np.median(edge_3d_dists))
                radius = max(radius, 0.03)
                radius = min(radius, 0.15)

        # 可视化
        vimg = color_crop.copy()
        vimg = cv2.cvtColor(vimg, cv2.COLOR_RGB2BGR)

        # 显示平面内点（绿）
        for idx in inliers:
            u = int(uu.ravel()[valid.ravel()][idx])
            v_i = int(vv.ravel()[valid.ravel()][idx])
            if 0 <= u < cW and 0 <= v_i < cH:
                vimg[v_i, u] = [0, 255, 0]

        # 显示边缘点（红）
        for pe in edge_pts_raw:
            if 0 <= pe[0] < cW and 0 <= pe[1] < cH:
                vimg[pe[1], pe[0]] = [0, 0, 255]

        # 蓝色 3D 圆投影
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

        n_inliers = len(inliers)
        cp_str = f'({center_px[0]:.1f},{center_px[1]:.1f})' if center_px is not None else '(?,?)'
        print(f'\r  3D点: {valid.sum():,}  盘面点: {n_inliers:,} / {valid.sum():,}  '
              f'黑点: {cp_str} → 局({cx*100:.1f},{cy*100:.1f})cm  '
              f'R={radius*100:.1f}cm 圆心局=({cx*100:.1f},{cy*100:.1f})cm  '
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
                     'depth_scale': dscale,
                     'n_cam': n_cam.tolist(), 'plane_d': d, 'plane_n': n.tolist()}
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
    print('\n' + '=' * 50)
    print(f'  采集: {STEP_DEG}° × {N_FRAMES} 帧')
    print('=' * 50)

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    # 采集时用 shift=0（深度准确），彩色图分割定位椅子
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
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}°  无数据')
            continue

        depth_mm_full = (d_sum / n_good) * dscale * 1000.0
        depth_u16 = depth_mm_full.astype(np.uint16)

        color, depth_mm = crop_frame(color_full, depth_mm_full)
        depth_u16_crop = depth_mm.astype(np.uint16)

        # 可视化深度
        dv = np.clip(depth_mm, 50, 500)
        dv8 = (dv / 500 * 255).astype(np.uint8)
        dv_color = cv2.applyColorMap(dv8, cv2.COLORMAP_JET)

        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'), depth_u16_crop)
        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.png'), color)
        cv2.imwrite(str(OUT / 'depth_color' / f'{i:03d}.jpg'), dv_color)

        dvals = depth_mm[depth_mm > 0]
        dvals_full = depth_mm_full[depth_mm_full > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}°  '
              f'valid:{len(dvals)//1000}k  '
              f'depth∈[{dvals.min():.0f},{dvals.max():.0f}]mm  '
              f'med={np.median(dvals):.0f}mm  '
              f'full∈[{dvals_full.min():.0f},{dvals_full.max():.0f}]mm  '
              f'dscale={dscale:.6f}  {el:.0f}s')

    pipe.stop()
    print(f'采集完成 ({time.time()-t0:.0f}s)')
    if tt:
        tt.close()


def do_fusion(calib):
    print('\n' + '=' * 50)
    print('  V6 彩色分割 + 深度提取 + TSDF 融合')
    print('=' * 50)

    R = np.array(calib['R']); t = np.array(calib['t'])
    fx, fy, ppx, ppy = calib['fx'], calib['fy'], calib['ppx'], calib['ppy']
    cW = calib['width']; cH = calib['height']
    step = calib.get('step_deg', STEP_DEG)
    cx_chair = calib.get('cx', 0.0)
    cy_chair = calib.get('cy', 0.0)
    print(f'  椅子中心: cx={cx_chair*100:.1f}cm cy={cy_chair*100:.1f}cm')

    depth_files = sorted(list((OUT / 'depth').glob('*.png')))
    color_files = sorted(list((OUT / 'color').glob('*.png')) + list((OUT / 'color').glob('*.jpg')))
    n_frames = min(N_FRAMES, len(depth_files), len(color_files))
    if n_frames == 0:
        print('无数据!')
        return

    # 预计算射线方向
    vv, uu = np.mgrid[0:cH, 0:cW]
    dx = (uu.astype(np.float64) - ppx) / fx
    dy = (vv.astype(np.float64) - ppy) / fy

    # 计算盘面中心在裁剪图中的投影像素位置（用于限定搜索区域）
    p_w_center = np.array([cx_chair, cy_chair, 0.0])
    p_c_center = R @ p_w_center + t
    if p_c_center[2] > 0.01:
        u_center = int(fx * p_c_center[0] / p_c_center[2] + ppx)
        v_center = int(fy * p_c_center[1] / p_c_center[2] + ppy)
    else:
        u_center, v_center = cW // 2, cH // 2
    disc_radius_px = int(calib.get('radius_m', 0.05) / p_c_center[2] * fx * 1.15) if p_c_center[2] > 0.01 else 80
    print(f'  盘面投影: ({u_center},{v_center}) r={disc_radius_px}px')

    print(f'\n逐帧颜色分割 + 深度验证 ({n_frames} 帧)...')
    print(f'  亮度阈值={COLOR_BRIGHT_THRESHOLD*255:.0f}/255  搜索半径={CHAIR_SEARCH_RADIUS}px')
    t0 = time.time()

    all_chair_pts = []
    frame_ok_count = 0

    for i in range(n_frames):
        depth_u16 = cv2.imread(str(depth_files[i]), cv2.IMREAD_UNCHANGED)
        color_bgr = cv2.imread(str(color_files[i]))
        if depth_u16 is None or color_bgr is None:
            continue

        depth_m = depth_u16.astype(np.float64) * 0.001
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # 小搜索窗: 只搜索盘面投影中心周围 (椅子 ~4cm≈50px)
        yy, xx = np.mgrid[0:cH, 0:cW]
        near_center = (xx - u_center)**2 + (yy - v_center)**2 < CHAIR_SEARCH_RADIUS**2

        # 局部自适应亮度: 窗内最亮的像素是椅子
        window_gray = gray[near_center]
        if len(window_gray) < 100:
            continue
        local_median = float(np.median(window_gray))
        bright = gray > max(local_median + 0.06, COLOR_BRIGHT_THRESHOLD)

        chair_mask = near_center & bright

        if chair_mask.sum() < 15:
            if i < 3 or (i + 1) % 18 == 0:
                print(f'  [{i+1}/{n_frames}] 候选<15 (local_med={local_median*255:.0f} bright={bright.sum()}), skip')
            continue

        # 形态学: 只做 CLOSE 填间隙，不做 OPEN（保留细椅腿/栅格）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        chair_mask_u8 = chair_mask.astype(np.uint8) * 255
        chair_mask_u8 = cv2.morphologyEx(chair_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 取离盘心最近的连通域
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(chair_mask_u8, connectivity=8)
        if n_labels <= 1:
            continue
        best_label = None
        best_dist = float('inf')
        for lb in range(1, n_labels):
            if stats[lb, cv2.CC_STAT_AREA] < 30:
                continue
            cx_b, cy_b = centroids[lb]
            dist = np.hypot(cx_b - u_center, cy_b - v_center)
            if dist < best_dist:
                best_dist = dist
                best_label = lb
        if best_label is None:
            continue
        chair_mask_clean = labels == best_label

        if chair_mask_clean.sum() < 15:
            continue

        # 提取深度 → 3D 坐标
        z_cam = depth_m[chair_mask_clean]
        pts_cam = np.column_stack([
            dx[chair_mask_clean] * z_cam,
            dy[chair_mask_clean] * z_cam,
            z_cam
        ])
        pts_world = (R.T @ (pts_cam - t).T).T

        # Z 验证: 椅子应在盘面上方 0~12cm
        z_ok = (pts_world[:, 2] > -0.12) & (pts_world[:, 2] < 0.02)
        if z_ok.sum() < 10:
            continue
        pts_world = pts_world[z_ok]
        frame_ok_count += 1

        # 去旋转
        theta = np.radians(i * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        pts_obj = (R_z.T @ pts_world.T).T

        all_chair_pts.append(pts_obj)

        if i < 3 or (i + 1) % 18 == 0:
            print(f'  [{i+1}/{n_frames}] mask={chair_mask_clean.sum()} '
                  f'pts={len(pts_obj)} Z∈[{pts_obj[:,2].min()*100:.1f},{pts_obj[:,2].max()*100:.1f}]cm '
                  f'local_med={local_median*255:.0f}')

    if frame_ok_count == 0:
        print('\n错误: 所有帧均未提取到椅子点')
        return

    print(f'\n有效帧: {frame_ok_count}/{n_frames}')

    if len(all_chair_pts) == 0:
        print('\n错误: 未提取到椅子点')
        return

    merged = np.concatenate(all_chair_pts, axis=0)
    print(f'\n椅子点合计: {len(merged):,}')

    # 先体素降采样减少点数，再聚类
    print('降采样 (2mm)...')
    pcd_merged = o3d.geometry.PointCloud()
    pcd_merged.points = o3d.utility.Vector3dVector(merged)
    pcd_merged = pcd_merged.voxel_down_sample(0.002)
    merged = np.asarray(pcd_merged.points)
    print(f'降采样后: {len(merged):,} 点')

    # DBSCAN 聚类
    print('DBSCAN 聚类 (eps=5mm, min_samples=50)...')
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(merged)
    labels = np.array(pcd_tmp.cluster_dbscan(eps=0.005, min_points=50, print_progress=False))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'  簇数: {n_clusters}')
    for lid in sorted(set(labels)):
        count = (labels == lid).sum()
        if count > 100:
            cp = merged[labels == lid]
            print(f'  簇 {lid}: {count:,} 点  '
                  f'中心=({cp[:,0].mean()*100:.1f},{cp[:,1].mean()*100:.1f},{cp[:,2].mean()*100:.1f})cm')

    valid_labels = [l for l in set(labels) if l != -1]
    if not valid_labels:
        print('DBSCAN 未找到簇')
        return
    largest_label = max(valid_labels, key=lambda l: (labels == l).sum())
    chair_pts = merged[labels == largest_label]
    print(f'\n椅子簇 ({largest_label}): {len(chair_pts):,} 点')

    # 保存调试点云
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(merged)
    colors = np.zeros((len(merged), 3))
    if largest_label >= 0:
        colors[labels == largest_label] = [0, 1, 0]
    colors[labels == -1] = [0.5, 0.5, 0.5]
    for lid in valid_labels:
        if lid != largest_label:
            colors[labels == lid] = [1, 0.5, 0]
    pcd_raw.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(OUT / 'debug_clusters.ply'), pcd_raw)

    # 体素降采样 + 统计滤波
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(chair_pts)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f'滤波后: {len(pcd.points):,} 点')

    if len(pcd.points) < 100:
        print('滤波后点数太少')
        return

    # 方向修正: Z↑ → Y↑
    pts = np.asarray(pcd.points)
    pts_out = np.zeros_like(pts)
    pts_out[:, 0] = pts[:, 0]
    pts_out[:, 1] = pts[:, 2]
    pts_out[:, 2] = pts[:, 1]
    pts_out[:, 2] -= pts_out[:, 2].min()

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_out)
    o3d.io.write_point_cloud(str(OUT / 'chair_points.ply'), pcd_out)

    # Poisson 表面重建
    print('Poisson 表面重建...')
    pcd_out.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=30))
    pcd_out.orient_normals_towards_camera_location(np.array([0., 2., 0.]))

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_out, depth=9, width=0, scale=1.1, linear_fit=False)
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
           'method': 'ColorSegmentation+Depth+DBSCAN',
           'disparity_shift': DISPARITY_SHIFT,
           'n_frames': n_frames, 'total_chair_pts': len(chair_pts),
           'voxel_size_mm': VOXEL_SIZE * 1000,
           'color_threshold': COLOR_BRIGHT_THRESHOLD,
           'vertices': len(verts),
           'faces': len(faces) if len(faces) > 0 else 0}
    with open(OUT / 'log_fusion.json', 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    import sys

    calib_path = OUT / 'calibrate.json'
    has_depth = len(list((OUT / 'depth').glob('*.png'))) >= 36

    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        print(f'加载已有标定: {calib_path}')
    else:
        calib = do_calibrate()
        if calib is None:
            print('标定取消')
            sys.exit(0)

    if has_depth:
        resp = input('深度图已存在，跳过采集? [Y/n]: ').strip().lower()
        if resp == 'n':
            do_capture(calib)
        else:
            print('跳过采集\n')
    else:
        do_capture(calib)

    do_fusion(calib)
