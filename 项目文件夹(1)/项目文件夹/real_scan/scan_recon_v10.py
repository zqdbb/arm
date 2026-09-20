#!/usr/bin/env python3
"""
V10 椅子重建: YOLO mask + Depth Anything V2 稠密深度投影
核心: YOLO找椅子 → Depth Anything V2预测稠密深度 → RANSAC度量缩放 → 3D投影 → Poisson
      用 AI 单目深度替代 D435i 立体深度（后者在光滑木面上彻底失效）
"""

import cv2, json, time, os
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d
import torch

os.environ['QT_QPA_PLATFORM'] = 'xcb'

BASE = Path(__file__).parent
OUT = BASE / 'output/v10'
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150
DISPARITY_SHIFT = 150
DIGITAL_ZOOM = 2.0

VOXEL_SIZE = 0.001          # 1mm 降采样
YOLO_CONF_THRESHOLD = 0.5
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 0.80
Z_CHAIR_MIN = -0.12
Z_CHAIR_MAX = 0.03

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'color').mkdir(exist_ok=True)
(OUT / 'depth').mkdir(exist_ok=True)
(OUT / 'depth_color').mkdir(exist_ok=True)


def get_cropped_intrinsics(fx, fy, ppx, ppy):
    crop_w = int(W / DIGITAL_ZOOM)
    crop_h = int(H / DIGITAL_ZOOM)
    crop_x = (W - crop_w) // 2
    crop_y = (H - crop_h) // 2
    return {'fx': fx, 'fy': fy, 'ppx': ppx - crop_x, 'ppy': ppy - crop_y,
            'width': crop_w, 'height': crop_h, 'crop_x': crop_x, 'crop_y': crop_y}


def crop_frame(color, depth=None):
    crop_w = int(W / DIGITAL_ZOOM)
    crop_h = int(H / DIGITAL_ZOOM)
    crop_x = (W - crop_w) // 2
    crop_y = (H - crop_h) // 2
    color_crop = color[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    if depth is None:
        return color_crop
    return color_crop, depth[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]


def set_disparity_shift(value=None):
    if value is None:
        value = DISPARITY_SHIFT
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print('警告: 未找到 RealSense 设备')
            return False
        advnc = rs.rs400_advanced_mode(devices[0])
        if not advnc.is_enabled():
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
    print('\n' + '=' * 50)
    print('  V10 标定: D435i 转台标定')
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

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)

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
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
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
            n = -n; d_val = -d_val

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
            best_label, best_score = None, -1
            cy_img, cx_img = cH / 2, cW / 2
            for lb in range(1, n_labels):
                area = stats[lb, cv2.CC_STAT_AREA]
                if area < 20 or area > 5000:
                    continue
                cx_b, cy_b = centroids[lb]
                if np.hypot(cx_b - cx_img, cy_b - cy_img) > min(cW, cH) * 0.4:
                    continue
                mean_brightness = gray_img[labels == lb].mean()
                score = (255 - mean_brightness) * 0.6 + (1.0 - np.hypot(cx_b - cx_img, cy_b - cy_img) / (min(cW, cH) * 0.4)) * 0.4
                if score > best_score:
                    best_score = score; best_label = lb
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
            return ray * t if t > 0 else None

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

        edge_pts_raw = detect_disc_edge(gray, center_px[0] if center_px is not None else cW / 2,
                                        center_px[1] if center_px is not None else cH / 2)

        radius = 0.05
        if len(edge_pts_raw) >= 15 and center_px is not None:
            ed = np.array(edge_pts_raw)
            px_dists = np.hypot(ed[:, 0] - center_px[0], ed[:, 1] - center_px[1])
            med_px = np.median(px_dists)
            good = np.abs(px_dists - med_px) < med_px * 0.4
            edge_3d_dists = []
            for pe in ed[good]:
                pe3d = ray_plane_intersect(pe[0], pe[1], n, d_val, fx_c, fy_c, ppx_c, ppy_c)
                if pe3d is not None:
                    pel = R_plane @ pe3d
                    edge_3d_dists.append(np.hypot(pel[0] - cx, pel[1] - cy))
            if len(edge_3d_dists) >= 8:
                radius = float(np.median(edge_3d_dists))
                radius = max(radius, 0.03)
                radius = min(radius, 0.15)

        vimg = cv2.cvtColor(color_crop.copy(), cv2.COLOR_RGB2BGR)
        for idx in inliers:
            u = int(uu.ravel()[valid.ravel()][idx])
            vi = int(vv.ravel()[valid.ravel()][idx])
            if 0 <= u < cW and 0 <= vi < cH:
                vimg[vi, u] = [0, 255, 0]

        for pe in edge_pts_raw:
            if 0 <= pe[0] < cW and 0 <= pe[1] < cH:
                vimg[pe[1], pe[0]] = [0, 0, 255]

        if center_px is not None:
            th = np.linspace(0, 2 * np.pi, 72)
            p_w_circle = np.column_stack([cx + radius * np.cos(th), cy + radius * np.sin(th), np.zeros(72)])
            p_c_circle = (R_calib @ p_w_circle.T).T + t_calib
            fvis = p_c_circle[:, 2] > 0.01
            if fvis.sum() > 6:
                u_c = (fx_c * p_c_circle[fvis, 0] / p_c_circle[fvis, 2] + ppx_c).astype(int)
                v_c = (fy_c * p_c_circle[fvis, 1] / p_c_circle[fvis, 2] + ppy_c).astype(int)
                for k in range(len(u_c)):
                    nk = (k + 1) % len(u_c)
                    if np.hypot(u_c[k] - u_c[nk], v_c[k] - v_c[nk]) < 300:
                        cv2.line(vimg, (u_c[k], v_c[k]), (u_c[nk], v_c[nk]), (255, 0, 0), 2)
            cv2.circle(vimg, (int(center_px[0]), int(center_px[1])), 4, (0, 255, 255), -1)

        print(f'\r  3D点: {valid.sum():,}  黑点: ({center_px[0]:.1f},{center_px[1]:.1f}) '
              f'局({cx*100:.1f},{cy*100:.1f})cm  边缘: {len(edge_pts_raw)} R={radius*100:.1f}cm  '
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
        elif key in (ord(' '), 13):
            print('\n重标...')
        elif key == ord('q'):
            break

    pipe.stop()
    cv2.destroyAllWindows()
    return calib


def do_capture(calib):
    print('\n' + '=' * 50)
    print(f'  采集: {STEP_DEG}deg x {N_FRAMES} 帧')
    print('=' * 50)

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)

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
    print('相机就绪')

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
        dv_color = cv2.applyColorMap((dv / 500 * 255).astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'), depth_u16_crop)
        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.png'), color)
        cv2.imwrite(str(OUT / 'depth_color' / f'{i:03d}.jpg'), dv_color)

        dvals = depth_mm[depth_mm > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  valid:{len(dvals)//1000}k  '
              f'depth∈[{dvals.min():.0f},{dvals.max():.0f}]mm  med={np.median(dvals):.0f}mm  {el:.0f}s')

    pipe.stop()
    print(f'采集完成 ({time.time()-t0:.0f}s)')
    if tt:
        tt.close()


def do_fusion(calib):
    """V10 融合: YOLO mask + Depth Anything V2 稠密深度 → 3D 投影 → Poisson"""
    from ultralytics import YOLO
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    print('\n' + '=' * 50)
    print('  V10 YOLO + Depth Anything V2 稠密深度投影')
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
    depth_files = sorted([f for f in (OUT / 'depth').glob('*.png') if len(f.stem) == 3])
    n_frames = min(N_FRAMES, len(color_files), len(depth_files))
    if n_frames == 0:
        print('无数据!')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'设备: {device}')

    print('加载 YOLOv8n-seg...')
    yolo = YOLO('yolov8n-seg.pt')

    print('加载 Depth Anything V2 Small...')
    depth_processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    depth_model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    depth_model = depth_model.to(device)
    depth_model.eval()

    # 预计算每个像素在纸板平面上的期望深度（float32 省内存）
    vv, uu = np.mgrid[0:cH, 0:cW]
    ray_x = ((uu.astype(np.float32) - ppx) / fx).astype(np.float32)
    ray_y = ((vv.astype(np.float32) - ppy) / fy).astype(np.float32)
    cam_center_world = (-R_calib.T @ t_calib).astype(np.float32)
    ray_world_z = (R_calib[0, 2] * ray_x + R_calib[1, 2] * ray_y + R_calib[2, 2]).astype(np.float32)
    d_plane = (-cam_center_world[2] / ray_world_z).astype(np.float32)

    all_pts_obj = []
    detected = 0
    scale_fail = 0
    total_chair_px = 0

    print(f'\n逐帧推理 ({n_frames} 帧)...')
    t0 = time.time()

    for i in range(n_frames):
        color_bgr = cv2.imread(str(color_files[i]))
        depth_u16 = cv2.imread(str(depth_files[i]), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_u16 is None:
            continue

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        d435i_depth = depth_u16.astype(np.float32) / 1000.0

        # ── YOLO mask ──
        results = yolo(str(color_files[i]), classes=[56], verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        conf = r.boxes.conf[0].item()
        if conf < YOLO_CONF_THRESHOLD:
            continue
        mask_raw = r.masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_raw, (cW, cH))
        mask_binary = mask_resized > 0.5
        n_chair_px = mask_binary.sum()
        if n_chair_px < 100:
            continue

        # ── Depth Anything V2 推理 ──
        inputs = depth_processor(images=color_rgb, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = depth_model(**inputs)
        pred_depth = outputs.predicted_depth[0].cpu().numpy()

        if pred_depth.shape != (cH, cW):
            pred_depth = cv2.resize(pred_depth, (cW, cH))

        # ── 深度缩放：只用纸板平面上的像素拟合（D435i 筛选，纸板深度可靠）──
        on_plane = (~mask_binary) & (d435i_depth > 0.01) & (np.abs(d435i_depth - d_plane) < 0.04)
        if on_plane.sum() < 100:
            scale_fail += 1
            continue

        d_rel_card = pred_depth[on_plane]
        d_plane_card = d_plane[on_plane]

        # 离群值过滤 + 鲁棒线性拟合
        d_rel_lo, d_rel_hi = np.percentile(d_rel_card, [3, 97])
        d_plane_lo, d_plane_hi = np.percentile(d_plane_card, [3, 97])
        inlier = ((d_rel_card >= d_rel_lo) & (d_rel_card <= d_rel_hi) &
                  (d_plane_card >= d_plane_lo) & (d_plane_card <= d_plane_hi))

        if inlier.sum() < 50:
            scale_fail += 1
            continue

        a, b = np.polyfit(d_rel_card[inlier], d_plane_card[inlier], 1)

        # 应用缩放 + 逐像素物理约束（椅子在纸板上方，深度 < 纸板深度）
        d_metric = a * pred_depth + b
        d_metric = np.clip(d_metric, d_plane - 0.15, d_plane - 0.001)

        # ── 提取椅子 3D 点 ──
        chair = mask_binary & (d_metric > 0.01)
        if chair.sum() < 50:
            continue

        v_px, u_px = np.where(chair)
        z_cam = d_metric[chair].astype(np.float32)
        x_cam = (u_px.astype(np.float32) - ppx) / fx * z_cam
        y_cam = (v_px.astype(np.float32) - ppy) / fy * z_cam
        pts_cam = np.column_stack([x_cam, y_cam, z_cam])

        # 相机 → 世界
        pts_world = ((R_calib.T @ (pts_cam - t_calib).T).T).astype(np.float32)

        # 平移到转轴中心
        pts_centered = pts_world.copy()
        pts_centered[:, 0] -= cx_disc
        pts_centered[:, 1] -= cy_disc

        # 去旋转
        theta = np.radians(i * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z_neg = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts_obj = (R_z_neg @ pts_centered.T).T

        all_pts_obj.append(pts_obj)
        detected += 1
        total_chair_px += n_chair_px

        if i < 3 or (i + 1) % 18 == 0:
            el = time.time() - t0
            # 检查缩放质量
            d_plane_range = np.ptp(d_plane_card[inlier]) * 100
            print(f'  [{i+1}/{n_frames}] {i*step:3d}deg  '
                  f'mask={n_chair_px}px  a={a:.4f} b={b:.3f}  '
                  f'planeΔ={d_plane_range:.1f}cm  pts={len(pts_obj)}  conf={conf:.2f}  {el:.0f}s')

    print(f'\n检出: {detected}/{n_frames}  缩放失败: {scale_fail}')

    if len(all_pts_obj) == 0:
        print('无有效帧')
        return

    # ── 合并 ──
    merged = np.concatenate(all_pts_obj, axis=0)
    print(f'累积点: {len(merged):,}  平均 {total_chair_px//max(detected,1):,}px/帧')

    # ── Z 过滤 ──
    z_ok = (merged[:, 2] > Z_CHAIR_MIN) & (merged[:, 2] < Z_CHAIR_MAX)
    merged = merged[z_ok]
    print(f'Z 过滤 [{Z_CHAIR_MIN*100:.0f},{Z_CHAIR_MAX*100:.0f}]cm: {len(merged):,}')

    if len(merged) < 200:
        print('点数太少')
        return

    # ── 体素降采样 ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pts_ds = np.asarray(pcd.points)
    print(f'降采样 ({VOXEL_SIZE*1000:.0f}mm): {len(pts_ds):,}')

    # ── DBSCAN ──
    pcd_ds = o3d.geometry.PointCloud()
    pcd_ds.points = o3d.utility.Vector3dVector(pts_ds)
    labels = np.array(pcd_ds.cluster_dbscan(eps=0.005, min_points=30, print_progress=False))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'DBSCAN: {n_clusters} 簇')

    # 各簇信息
    for lid in sorted(set(labels)):
        count = (labels == lid).sum()
        if count > 50:
            cp = pts_ds[labels == lid]
            print(f'  簇 {lid}: {count:,}pt  '
                  f'中心=({cp[:,0].mean()*100:.1f},{cp[:,1].mean()*100:.1f},{cp[:,2].mean()*100:.1f})cm')

    valid_labels = [l for l in set(labels) if l != -1]
    if not valid_labels:
        print('无有效簇，使用所有点')
        chair_pts = pts_ds
    else:
        best_label = max(valid_labels, key=lambda l: (labels == l).sum())
        chair_pts = pts_ds[labels == best_label]
        print(f'选中最大簇 {best_label}: {len(chair_pts):,}')

    # ── 保存调试 ──
    pcd_dbg = o3d.geometry.PointCloud()
    pcd_dbg.points = o3d.utility.Vector3dVector(pts_ds)
    if n_clusters > 0:
        cmap = np.random.rand(max(n_clusters, 1), 3)
        dbg_colors = np.zeros((len(pts_ds), 3))
        for lid in set(labels):
            if lid == -1:
                dbg_colors[labels == lid] = [0.5, 0.5, 0.5]
            else:
                dbg_colors[labels == lid] = cmap[lid % len(cmap)]
        pcd_dbg.colors = o3d.utility.Vector3dVector(dbg_colors)
    o3d.io.write_point_cloud(str(OUT / 'debug_clusters.ply'), pcd_dbg)

    # ── 统计滤波 ──
    pcd_chair = o3d.geometry.PointCloud()
    pcd_chair.points = o3d.utility.Vector3dVector(chair_pts)
    pcd_chair, _ = pcd_chair.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    print(f'统计滤波后: {len(pcd_chair.points):,}')

    # ── 方向修正 Z→Y ──
    pts_final = np.asarray(pcd_chair.points)
    pts_out = np.zeros_like(pts_final)
    pts_out[:, 0] = pts_final[:, 0]
    pts_out[:, 1] = pts_final[:, 2]
    pts_out[:, 2] = pts_final[:, 1]
    pts_out[:, 2] -= pts_out[:, 2].min()

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_out)
    o3d.io.write_point_cloud(str(OUT / 'chair_points.ply'), pcd_out)

    # ── Poisson ──
    print(f'\nPoisson 表面重建 ({len(pcd_out.points):,} 点)...')
    pcd_out.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=30))
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
    print(f'  V10 输出: {OUT}/chair.obj')
    print(f'  方向: Y=↑')
    for i, a in enumerate('XYZ'):
        print(f'  {a}: [{verts[:,i].min():.3f}, {verts[:,i].max():.3f}] '
              f'span={np.ptp(verts[:,i])*100:.1f}cm')
    print(f'  预期: Y≈8.8cm(高)  X≈4.6cm  Z≈4.0cm')
    print(f'  顶点: {len(verts):,}  面: {len(faces):,}')
    print(f'{"=" * 50}')

    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'method': 'V10_YOLO_DepthAnythingV2_Projection',
           'n_frames': n_frames, 'detected_frames': detected,
           'scale_failures': scale_fail,
           'total_merged_pts': len(merged),
           'chair_pts_final': len(pts_final),
           'voxel_size_mm': VOXEL_SIZE * 1000,
           'yolo_conf_threshold': YOLO_CONF_THRESHOLD,
           'vertices': len(verts),
           'faces': int(len(faces))}
    with open(OUT / 'log_fusion.json', 'w') as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    import sys

    calib_path = OUT / 'calibrate.json'
    v6_calib_path = BASE / 'output/v6/calibrate.json'

    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        print(f'加载 V10 标定: {calib_path}')
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
    v10_color = list((OUT / 'color').glob('*.png'))
    if len(v10_color) < 36 and len(v6_color) >= 36:
        print(f'从 V6 复用数据 ({len(v6_color)} 帧)')
        for sub in ['color', 'depth', 'depth_color']:
            v6_sub = BASE / 'output/v6' / sub
            v10_sub = OUT / sub
            if v6_sub.exists() and not any(v10_sub.iterdir()):
                for f in v6_sub.iterdir():
                    (v10_sub / f.name).symlink_to(f.resolve())

    has_depth = len([f for f in (OUT / 'depth').glob('*.png') if len(f.stem) == 3]) >= 36

    if has_depth:
        resp = input('数据已存在，跳过采集? [Y/n]: ').strip().lower()
        if resp == 'n':
            do_capture(calib)
        else:
            print('跳过采集\n')
    else:
        do_capture(calib)

    do_fusion(calib)
