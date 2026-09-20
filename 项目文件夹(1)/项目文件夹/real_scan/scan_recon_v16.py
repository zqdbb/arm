#!/usr/bin/env python3
"""
V16 家具重建 — 标定+扫描同次完成 + High Accuracy 深度预设
══════════════════════════════════════════════════════════════
  交互标定转台(按S保存) → 放椅子 → 按回车 → 72帧采集
  → YOLO+SAM抠图 → 深度反投影 → 标定轴旋转对齐 → Poisson重建

V15→V16 核心改动:
  1. High Accuracy visual preset (提升深度质量)
  2. 标定后不退出，直接进入采集 (消除相机位移误差)
  3. 旋转轴用标定结果 (非相机Y轴猜测)
  4. 旋转中心用标定圆心 (非数据驱动估计)
  5. 相机尽量靠近 ~30cm (用户控制，标定时看深度)

使用方式:
  python3 scan_recon_v16.py --name chair
  python3 scan_recon_v16.py --skip-capture  # 用已有数据重建
  python3 scan_recon_v16.py --skip-calib     # 用已有标定
"""

import cv2
import json
import numpy as np
import open3d as o3d
import os
import pyrealsense2 as rs
import sys
import time
import argparse
from pathlib import Path

# ─── 常量 ────────────────────────────────────────────
BASE = Path(__file__).parent
OUT = BASE / 'output/v16/chair'
W, H = 640, 480
W_OUT, H_OUT = 500, 500
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG  # 72
N_AVG = 5
LASER_POWER = 150

CUSTOM_MODEL = 'yolov8n-seg.pt'
SAM_MODEL = BASE / 'sam_vit_b_01ec64.pth'

# 重建参数 (同 V15)
POISSON_DEPTH = 8
VOXEL_SIZE = 0.001
OUTLIER_NB = 30
OUTLIER_STD = 1.5
DEPTH_BILATERAL_D = 5
DEPTH_BILATERAL_SIGMA = 50
MASK_ERODE_ITER = 2
MAD_MIN_TOLERANCE = 10


# ╔══════════════════════════════════════════════════════════╗
# ║  TurntableCalibrator — 精简版转台标定                     ║
# ╚══════════════════════════════════════════════════════════╝

class TurntableCalibrator:
    """检测转台中心暗点 + RANSAC 平面拟合 + 计算旋转轴."""

    def __init__(self):
        self.center_px = None       # (cx, cy) 图像坐标
        self.locked = False
        self.plane_model = None     # [a, b, c, d] 平面方程
        self.center_3d = None       # 圆心 3D 相机坐标
        self.n_axis = None          # 旋转轴方向 (相机坐标)
        self.fx = self.fy = self.ppx = self.ppy = 0
        self.depth_scale = 0.001
        self.calib_saved = False

    def detect_center(self, gray):
        """连通域分析：找近图像中心的最暗区域."""
        h, w = gray.shape
        th = np.percentile(gray, 20)
        dark = (gray < max(th, 20)).astype(np.uint8) * 255

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dark, connectivity=8)
        if n_labels <= 1:
            return None

        cx_img, cy_img = w / 2, h / 2
        best_label, best_score = None, -1

        for lb in range(1, n_labels):
            area = stats[lb, cv2.CC_STAT_AREA]
            if area < 10 or area > 20000:
                continue
            cx_b, cy_b = centroids[lb]
            dist = np.hypot(cx_b - cx_img, cy_b - cy_img)
            if dist > min(w, h) * 0.49:
                continue
            mask_lb = (labels == lb)
            mean_brightness = gray[mask_lb].mean()
            score = ((255 - mean_brightness) * 0.6 +
                     (1.0 - dist / (min(w, h) * 0.49)) * 0.4)
            if score > best_score:
                best_score = score
                best_label = lb

        if best_label is not None:
            cx, cy = centroids[best_label]
            r = np.sqrt(stats[best_label, cv2.CC_STAT_AREA] / np.pi)
            return (float(cx), float(cy), float(r))
        return None

    def fit_plane_and_center(self, depth_image):
        """RANSAC 拟合转台平面 + 计算圆心 3D 坐标."""
        h, w = depth_image.shape
        stride = 2
        d_sub = depth_image[::stride, ::stride].astype(np.float64) * self.depth_scale
        valid = (d_sub > 0.01) & (d_sub < 0.8)

        if valid.sum() < 300:
            return False

        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z_cam = d_sub[valid]
        x_cam = (uu[valid].astype(np.float64) - self.ppx) / self.fx * z_cam
        y_cam = (vv[valid].astype(np.float64) - self.ppy) / self.fy * z_cam

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.column_stack([x_cam, y_cam, z_cam]))
        plane_model, _ = pcd.segment_plane(
            distance_threshold=0.005, ransac_n=3, num_iterations=300)

        n = np.array(plane_model[:3])
        d = plane_model[3]
        if n[2] < 0:
            n = -n
            d = -d

        self.plane_model = plane_model
        self.n_axis = n / np.linalg.norm(n)

        # 计算圆心 3D 坐标 (射线-平面求交)
        if self.center_px is not None:
            cx_px, cy_px = self.center_px[0], self.center_px[1]
            ray = np.array([
                (cx_px - self.ppx) / self.fx,
                (cy_px - self.ppy) / self.fy,
                1.0
            ])
            ray /= np.linalg.norm(ray)
            denom = np.dot(self.n_axis, ray)
            if abs(denom) > 1e-8:
                t_int = -d / denom
                if t_int > 0:
                    self.center_3d = ray * t_int
                    return True
        return False

    def draw(self, image, fps=0.0):
        """绘制标定覆盖层."""
        result = image.copy()
        h, w = image.shape[:2]

        # FPS
        cv2.putText(result, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if not self.locked:
            if self.center_px:
                cx, cy, r = self.center_px
                cv2.circle(result, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
                cv2.drawMarker(result, (int(cx), int(cy)), (0, 0, 255),
                              cv2.MARKER_CROSS, 16, 2)
            cv2.putText(result, "SPACE=锁定圆心  Q=退出",
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
        else:
            # 圆心十字
            if self.center_px:
                cv2.drawMarker(result,
                              (int(self.center_px[0]), int(self.center_px[1])),
                              (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

            # 旋转轴信息
            if self.n_axis is not None:
                cv2.putText(result,
                    f"Axis: [{self.n_axis[0]:.3f} {self.n_axis[1]:.3f} {self.n_axis[2]:.3f}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.center_3d is not None:
                cz = self.center_3d[2]
                cv2.putText(result, f"Center Z: {cz*100:.1f}cm",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if cz > 0.40:
                    hint = f"!! 太远 ({cz*100:.0f}cm > 40cm), 请靠近 !!"
                    cv2.putText(result, hint, (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif cz < 0.25:
                    cv2.putText(result, f"!! 太近 ({cz*100:.0f}cm < 25cm)", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(result, f"距离 OK ({cz*100:.0f}cm)", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(result, "S=保存并继续  R=重锁  Q=退出",
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (200, 200, 200), 1)

        return result


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 1: 标定 + 采集 (同一次运行)                        ║
# ╚══════════════════════════════════════════════════════════╝

def do_calibrate_and_capture(calib_input=None):
    """交互标定转台 → 放椅子 → 72帧采集.

    Args:
        calib_input: 如果提供, 跳过交互标定, 直接使用已有标定数据采集.
    """
    from turntable import TurntableController

    # ── 启动相机 ──
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    for attempt in range(3):
        try:
            profile = pipe.start(cfg)
            time.sleep(0.5)
            pipe.wait_for_frames(timeout_ms=5000)
            break
        except RuntimeError as e:
            print(f'  启动尝试 {attempt+1}/3: {e}')
            try:
                pipe.stop()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.0)
                pipe = rs.pipeline()
    else:
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

    align = rs.align(rs.stream.color)
    ds = profile.get_device().first_depth_sensor()

    # High Accuracy 预设
    try:
        ds.set_option(rs.option.visual_preset, 3.0)
        print('  Depth preset: High Accuracy')
    except Exception as e:
        print(f'  [WARN] 无法设置 High Accuracy 预设: {e}')

    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)

    dscale = ds.get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # 计算裁剪后的内参 (640×480 → crop 480×480 → resize 500×500)
    crop_size = min(W, H)  # 480
    x0 = (W - crop_size) // 2  # 80
    y0 = (H - crop_size) // 2  # 0
    scale = W_OUT / crop_size  # 500/480

    calib = {
        'fx_full': intr.fx * scale,
        'fy_full': intr.fy * scale,
        'ppx_full': (intr.ppx - x0) * scale,
        'ppy_full': (intr.ppy - y0) * scale,
        'width_full': W_OUT,
        'height_full': H_OUT,
        'depth_scale': dscale,
        'step_deg': STEP_DEG,
        'n_frames': N_FRAMES,
        'R': None, 't': None, 'center_3d': None, 'n_axis': None,
    }

    if calib_input is not None:
        # 使用已有标定
        calib.update(calib_input)
        print('\n使用已有标定, 跳过交互标定')
    else:
        # ── 交互标定 ──
        calib_obj = TurntableCalibrator()
        calib_obj.fx = intr.fx
        calib_obj.fy = intr.fy
        calib_obj.ppx = intr.ppx
        calib_obj.ppy = intr.ppy
        calib_obj.depth_scale = dscale

        print('\n' + '=' * 60)
        print('  转台标定 (无椅子)')
        print('  SPACE=锁定圆心  S=保存并继续  R=重锁  Q=退出')
        print('  请将相机对准空转台中心, 距离 25-40cm')
        print('=' * 60)

        cv2.namedWindow('V16 Calibration', cv2.WINDOW_NORMAL)
        frame_counter = 0
        fps = 0.0
        last_fps_time = time.time()

        try:
            while True:
                frames = pipe.wait_for_frames(timeout_ms=5000)
                aligned = align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # 检测圆心 (未锁定时)
                if not calib_obj.locked:
                    center_det = calib_obj.detect_center(gray)
                    if center_det:
                        calib_obj.center_px = center_det  # (cx, cy, r)

                # 已锁定: 持续拟合平面 (每30帧)
                if calib_obj.locked and frame_counter % 30 == 0:
                    calib_obj.fit_plane_and_center(depth_image)

                # FPS
                frame_counter += 1
                now = time.time()
                if now - last_fps_time >= 1.0:
                    fps = frame_counter / (now - last_fps_time)
                    frame_counter = 0
                    last_fps_time = now

                display = calib_obj.draw(color_image, fps=fps)
                cv2.imshow('V16 Calibration', display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('用户退出')
                    pipe.stop()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord(' '):
                    if not calib_obj.locked and calib_obj.center_px:
                        calib_obj.locked = True
                        calib_obj.fit_plane_and_center(depth_image)
                        if calib_obj.center_3d is not None:
                            print(f'  圆心锁定: ({calib_obj.center_px[0]:.1f}, '
                                  f'{calib_obj.center_px[1]:.1f})')
                            print(f'  圆心 3D: Z={calib_obj.center_3d[2]*100:.1f}cm')
                            print(f'  旋转轴: {calib_obj.n_axis}')
                    elif calib_obj.locked:
                        # 已锁定, 重锁
                        calib_obj.locked = False
                        calib_obj.center_px = None
                        calib_obj.plane_model = None
                        calib_obj.center_3d = None
                        calib_obj.n_axis = None
                        print('  已重置, 请重新锁定')
                elif key == ord('r'):
                    calib_obj.locked = False
                    calib_obj.center_px = None
                    calib_obj.plane_model = None
                    calib_obj.center_3d = None
                    calib_obj.n_axis = None
                    print('  已重置')
                elif key == ord('s'):
                    if not calib_obj.locked or calib_obj.center_3d is None:
                        print('  请先锁定圆心 (按SPACE)')
                    else:
                        # 最后一次精确拟合
                        calib_obj.fit_plane_and_center(depth_image)

                        # 构建 R_calib (同 turntable_set_v2 逻辑)
                        n = calib_obj.n_axis
                        d_val = calib_obj.plane_model[3]

                        z_axis = n / np.linalg.norm(n)
                        x_axis = np.cross(np.array([0., 1., 0.]), z_axis)
                        if np.linalg.norm(x_axis) < 1e-6:
                            x_axis = np.cross(np.array([1., 0., 0.]), z_axis)
                        x_axis /= np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)
                        R_plane = np.column_stack([x_axis, y_axis, z_axis])
                        R_calib = R_plane.T.astype(np.float64)
                        t_calib = (-d_val * n).astype(np.float64)

                        calib['R'] = R_calib.tolist()
                        calib['t'] = t_calib.tolist()
                        calib['center_3d'] = calib_obj.center_3d.tolist()
                        calib['n_axis'] = calib_obj.n_axis.tolist()

                        print(f'\n  标定已保存!')
                        print(f'  旋转轴: {calib["n_axis"]}')
                        print(f'  圆心 3D: {calib["center_3d"]}')
                        print(f'  Z距离: {calib_obj.center_3d[2]*100:.1f}cm')
                        break
        finally:
            cv2.destroyAllWindows()

    # ── 保存标定 ──
    OUT.mkdir(parents=True, exist_ok=True)
    calib_path = OUT / 'calibrate.json'
    # 转换 numpy array 为 list
    calib_serializable = {}
    for k, v in calib.items():
        if isinstance(v, np.ndarray):
            calib_serializable[k] = v.tolist()
        else:
            calib_serializable[k] = v
    with open(calib_path, 'w') as f:
        json.dump(calib_serializable, f, indent=2)
    print(f'标定文件: {calib_path}')

    # ── 连接转台 ──
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    # ── 等待放椅子 ──
    print('\n' + '=' * 60)
    print('  请将椅子放在转台中心')
    input('  按回车开始采集...')

    # ── 72 帧采集 ──
    print('\n' + '=' * 60)
    print(f'  Step 1/4  采集: {STEP_DEG}deg × {N_FRAMES} 帧 ({W}×{H} → {W_OUT}×{H_OUT})')
    print('=' * 60)

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
            if not df or not cf:
                continue
            d_sum += np.asanyarray(df.get_data()).astype(np.float64)
            if color_full is None:
                color_full = np.asanyarray(cf.get_data())
            n_good += 1

        if n_good == 0:
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  无数据')
            continue

        crop_size = min(W, H)
        x0 = (W - crop_size) // 2
        y0 = (H - crop_size) // 2
        color_crop = color_full[y0:y0+crop_size, x0:x0+crop_size]
        color_out = cv2.resize(color_crop, (W_OUT, H_OUT),
                               interpolation=cv2.INTER_LINEAR)
        depth_mm = (d_sum / n_good) * dscale * 1000.0
        depth_crop = depth_mm[y0:y0+crop_size, x0:x0+crop_size]
        depth_out = cv2.resize(depth_crop, (W_OUT, H_OUT),
                               interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.png'), color_out)
        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'),
                    depth_out.astype(np.uint16))

        dvals = depth_out[depth_out > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  '
              f'valid:{len(dvals)//1000}k  med={np.median(dvals):.0f}mm  {el:.0f}s')

    pipe.stop()
    tt.close()
    print(f'采集完成 ({time.time()-t0:.0f}s)\n')
    return calib


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 2: YOLO + SAM 抠图 + 测量 (同 V15)                 ║
# ╚══════════════════════════════════════════════════════════╝

def measure_chair_from_masks(masks_dict, calib):
    """4 个 SAM 精修帧做像素-距离法测量."""
    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    dscale = calib.get('depth_scale', 0.001)
    depth_dir = OUT / 'depth'
    view_indices = [0, 18, 36, 54]

    heights, widths = [], []

    for idx in view_indices:
        mask = masks_dict.get(idx)
        if mask is None or mask.max() == 0:
            continue
        ys, xs = np.where(mask > 128)
        if len(ys) < 200:
            continue

        dep = cv2.imread(str(depth_dir / f'{idx:03d}.png'),
                         cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue

        dvals = dep[ys, xs].astype(float)
        valid = dvals > 0
        if valid.sum() < 200:
            continue

        ys_v, xs_v = ys[valid], xs[valid]
        d_valid = dvals[valid]
        hist, edges = np.histogram(
            d_valid, bins=min(30, max(5, len(d_valid)//50)))
        peak_bin = np.argmax(hist)
        d_center = (edges[peak_bin] +
                    edges[min(peak_bin+1, len(edges)-1)]) / 2

        in_layer = np.abs(d_valid - d_center) < 15
        if in_layer.sum() < 100:
            continue

        ys_f = ys_v[in_layer]
        xs_f = xs_v[in_layer]
        d_layer = d_valid[in_layer]
        dist_m = float(np.median(d_layer)) * dscale

        h_px = ys_f.max() - ys_f.min()
        w_px = np.percentile(xs_f, 98) - np.percentile(xs_f, 2)
        h_m = h_px * dist_m / fy
        w_m = w_px * dist_m / fx
        heights.append(h_m)
        widths.append(w_m)

    if not heights:
        return None

    heights = np.array(heights)
    widths = np.array(widths)
    chair_h = float(np.median(heights))
    sorted_w = np.sort(widths)
    chair_w = float(np.mean(sorted_w[-2:]))
    chair_d = float(np.mean(sorted_w[:2]))

    result = {
        'height_m': round(chair_h, 3),
        'width_m': round(chair_w, 3),
        'depth_m': round(chair_d, 3),
        'n_measured': len(heights),
    }

    print(f'\n  ┌─ SAM 视角测量 ({result["n_measured"]} 帧, 深度峰值层)')
    print(f'  ├─ 椅子高度: {result["height_m"]*100:.1f}cm')
    print(f'  ├─ 椅子宽度: {result["width_m"]*100:.1f}cm')
    print(f'  └─ 椅子深度: {result["depth_m"]*100:.1f}cm')
    return result


def do_yolo_and_measure(calib):
    """YOLOv8n-seg 初筛 + SAM 精修 + 自动测量 (同 V15)."""
    from ultralytics import YOLO

    cW = calib.get('width_full', W_OUT)
    cH = calib.get('height_full', H_OUT)
    step = calib.get('step_deg', STEP_DEG)

    color_files = sorted((OUT / 'color').glob('*.png'))
    if len(color_files) < 36:
        print(f'图片不足: {len(color_files)} 张')
        return None, None

    print('=' * 60)
    print('  Step 2/4  YOLO 初筛 + SAM 精修 + 自动测量')
    print('=' * 60)

    print(f'  加载 YOLO: {CUSTOM_MODEL}')
    yolo_obj = YOLO(str(CUSTOM_MODEL))

    sam_ok = Path(str(SAM_MODEL)).exists()
    sam_predictor = None
    if sam_ok:
        print(f'  加载 SAM: {SAM_MODEL}')
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_MODEL))
        sam_predictor = SamPredictor(sam)

    # Pass 1: YOLO 全部帧
    masks_dict = {}
    bboxes = {}
    detected = 0

    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None:
            continue

        results = yolo_obj(img, verbose=False)
        r = results[0]
        mask_binary = np.zeros((cH, cW), dtype=np.uint8)
        bboxes[i] = None

        if r.boxes is not None and len(r.boxes) > 0:
            best_idx = int(r.boxes.conf.argmax().item())
            conf = r.boxes.conf[best_idx].item()
            if conf >= 0.3:
                detected += 1
                xyxy = r.boxes.xyxy[best_idx].cpu().numpy()
                bboxes[i] = xyxy
                mask_raw = r.masks.data[best_idx].cpu().numpy()
                mask_resized = cv2.resize(mask_raw, (cW, cH))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        masks_dict[i] = mask_binary

    print(f'  YOLO 检出: {detected}/{len(color_files)}')

    # Pass 2: SAM 精修 4 视角帧
    view_indices = [0, 18, 36, 54]
    if sam_predictor:
        print(f'\n  SAM 精修 {len(view_indices)} 个视角帧...')
        for idx in view_indices:
            if bboxes.get(idx) is None:
                continue
            cf = color_files[idx]
            img = cv2.imread(str(cf))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)

            if idx == 0:
                print(f'    视角   0° (frame 000): 跳过 SAM, 使用 YOLO mask')
                continue
            else:
                masks_sam, scores, _ = sam_predictor.predict(
                    box=bboxes[idx][None, :], multimask_output=False)

            best_m = masks_sam[scores.argmax()]
            masks_dict[idx] = (best_m > 0).astype(np.uint8) * 255
            n_px = int(best_m.sum())
            print(f'    视角 {idx*step:3d}° (frame {idx:03d}): '
                  f'SAM mask={n_px//1000}k px')

    # Pass 3: 深度峰值过滤 + 形态学清理
    for i in range(len(color_files)):
        mask = masks_dict.get(i)
        if mask is None or mask.sum() < 500:
            continue

        dep = cv2.imread(str(OUT / 'depth' / f'{i:03d}.png'),
                         cv2.IMREAD_UNCHANGED)
        if dep is not None:
            ys, xs = np.where(mask > 128)
            d_vals = dep[ys, xs].astype(float)
            valid = d_vals > 0
            if valid.sum() > 50:
                d_valid = d_vals[valid]
                median_d = np.median(d_valid)
                mad = np.median(np.abs(d_valid - median_d))
                tolerance = max(3.0 * mad * 1.4826, MAD_MIN_TOLERANCE)
                keep = np.abs(d_valid - median_d) < tolerance
                refined = np.zeros_like(mask)
                refined[ys[valid][keep], xs[valid][keep]] = 255
                mask = refined

        if mask.sum() > 500:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            if n_labels > 1:
                largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = (labels == largest).astype(np.uint8) * 255

        masks_dict[i] = mask

    # 统计
    for i in range(len(color_files)):
        if i < 3 or (i+1) % 18 == 0:
            m = masks_dict.get(i)
            n_px = (m > 0).sum() if m is not None else 0
            tag = ('SAM' if i in view_indices and sam_ok else
                   ('YOLO' if n_px > 500 else '✗'))
            print(f'  [{i+1}/{len(color_files)}] {i*step:3d}deg  '
                  f'mask={n_px//1000}k px  [{tag}]')

    # 保存 4 视角 RGBA 抠图
    views_dir = OUT / 'views'
    views_dir.mkdir(exist_ok=True)
    for idx in view_indices:
        if masks_dict.get(idx) is None or masks_dict[idx].sum() < 500:
            continue
        img = cv2.imread(str(color_files[idx]))
        if img is None:
            continue
        mask = masks_dict[idx]
        rgba = np.dstack([img, mask])
        cv2.imwrite(str(views_dir / f'view_{idx:03d}.png'), rgba)

    # 测量
    chair_dims = measure_chair_from_masks(masks_dict, calib)
    if chair_dims:
        print(f'\n  ★ 测量结果: 高{chair_dims["height_m"]*100:.1f}cm '
              f'宽{chair_dims["width_m"]*100:.1f}cm '
              f'深{chair_dims["depth_m"]*100:.1f}cm')

    return masks_dict, chair_dims


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 3: 深度点云融合 (使用标定旋转轴)                    ║
# ╚══════════════════════════════════════════════════════════╝

def do_depth_reconstruction(masks_dict, calib):
    """72帧深度反投影 → 标定旋转轴对齐 → Poisson 重建."""

    print('=' * 60)
    print('  Step 3/4  深度反投影 + 标定轴旋转 + Poisson 重建')
    print('=' * 60)

    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    ppx = calib.get('ppx_full', 250.0)
    ppy = calib.get('ppy_full', 250.0)
    dscale = calib.get('depth_scale', 0.001)

    depth_dir = OUT / 'depth'
    depth_files = sorted(depth_dir.glob('*.png'))

    if len(depth_files) < 4:
        print(f'深度图不足: {len(depth_files)} 张')
        return None

    # ── 旋转参数 (来自标定) ──
    n_axis = np.array(calib.get('n_axis', [0.0, 1.0, 0.0]))
    center_cam = np.array(calib.get('center_3d', [0.0, 0.0, 0.3]))
    print(f'  旋转轴 (标定): [{n_axis[0]:.4f} {n_axis[1]:.4f} {n_axis[2]:.4f}]')
    print(f'  旋转中心(标定): [{center_cam[0]:.3f} {center_cam[1]:.3f} {center_cam[2]:.3f}]m')

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ── Pass A: 反投影 72 帧 ──
    print('  Pass A: 反投影 72 帧...')
    frame_data = []  # (pts_cam, idx)

    for i, dp in enumerate(depth_files):
        idx = int(dp.stem)
        mask = masks_dict.get(idx)
        if mask is None or mask.sum() < 500:
            continue

        dep = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue

        dep_f = cv2.bilateralFilter(
            dep.astype(np.float32), DEPTH_BILATERAL_D,
            DEPTH_BILATERAL_SIGMA, DEPTH_BILATERAL_SIGMA)
        mask_eroded = cv2.erode(mask, erode_kernel, iterations=MASK_ERODE_ITER)

        valid_mask = (mask_eroded > 128) & (dep_f > 0)
        ys, xs = np.where(valid_mask)
        if len(ys) < 200:
            continue

        d_m = dep_f[ys, xs].astype(float) * dscale
        pts_cam = np.stack([
            (xs - ppx) * d_m / fx,
            (ys - ppy) * d_m / fy,
            d_m
        ], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE * 2)

        if len(pcd.points) < 50:
            continue

        frame_data.append((np.asarray(pcd.points), idx))

        if (i+1) % 12 == 0 or i == 0:
            print(f'  [{i+1}/{len(depth_files)}] frame {idx:03d} → '
                  f'{len(pcd.points)} 点')

    print(f'  有效帧: {len(frame_data)}/{len(depth_files)}')
    if len(frame_data) < 4:
        print('  有效帧太少')
        return None

    # ── Pass B: 标定轴旋转对齐 ──
    print('  Pass B: 标定轴旋转对齐...')

    global_pcd = o3d.geometry.PointCloud()
    for pts_cam, idx in frame_data:
        angle = idx * STEP_DEG
        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        K = np.array([[0, -n_axis[2], n_axis[1]],
                       [n_axis[2], 0, -n_axis[0]],
                       [-n_axis[1], n_axis[0], 0]])
        R_rot = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        pts_aligned = (pts_cam - center_cam) @ R_rot.T + center_cam

        pcd_aligned = o3d.geometry.PointCloud()
        pcd_aligned.points = o3d.utility.Vector3dVector(pts_aligned)
        global_pcd += pcd_aligned

    print(f'  融合点云: {len(global_pcd.points)} 点')

    # ── Pass C: 后处理 + Poisson ──
    print('  Pass C: 后处理 + Poisson 重建...')
    (OUT / 'models').mkdir(exist_ok=True)

    o3d.io.write_point_cloud(
        str(OUT / 'models' / 'pointcloud_raw.ply'), global_pcd)
    print(f'  已导出原始点云: {OUT}/models/pointcloud_raw.ply')

    pcd = global_pcd.voxel_down_sample(VOXEL_SIZE)
    print(f'  体素下采样 (voxel={VOXEL_SIZE*1000:.0f}mm): {len(pcd.points)} 点')

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB, std_ratio=OUTLIER_STD)
    print(f'  去噪后: {len(pcd.points)} 点')

    o3d.io.write_point_cloud(
        str(OUT / 'models' / 'pointcloud_clean.ply'), pcd)
    print(f'  已导出干净点云: {OUT}/models/pointcloud_clean.ply')

    print('  估计法线...')
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 5, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    print(f'  Poisson 重建 (depth={POISSON_DEPTH})...')
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    print(f'  Poisson 完成 ({time.time()-t0:.0f}s), '
          f'{len(mesh.vertices)} 顶点')

    if len(densities) > 0:
        d_arr = np.asarray(densities)
        threshold = np.percentile(d_arr, 5)
        vertices_remove = d_arr < threshold
        mesh.remove_vertices_by_mask(vertices_remove)
        print(f'  去低密度后: {len(mesh.vertices)} 顶点, '
              f'{len(mesh.triangles)} 面')

    return mesh


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 4: 缩放 + 导出                                    ║
# ╚══════════════════════════════════════════════════════════╝

def do_scale_export(mesh, chair_dims, name):
    """trimesh 分轴缩放 + 导出 OBJ/PLY."""
    import trimesh

    print('=' * 60)
    print('  Step 4/4  缩放 + 导出')
    print('=' * 60)

    TARGET_HEIGHT = 0.088  # 8.8cm
    ch = chair_dims.get('height_m', TARGET_HEIGHT)
    cw = chair_dims.get('width_m', 0.06)
    cd = chair_dims.get('depth_m', 0.05)

    verts = np.asarray(mesh.vertices)
    cur_w = verts[:, 0].max() - verts[:, 0].min()
    cur_d = verts[:, 1].max() - verts[:, 1].min()
    cur_h = verts[:, 2].max() - verts[:, 2].min()
    print(f'  重建尺寸: {cur_w*100:.1f}×{cur_d*100:.1f}×{cur_h*100:.1f}cm')

    if chair_dims.get('n_measured', 0) >= 2:
        print(f'  目标尺寸: {cw*100:.1f}×{cd*100:.1f}×{ch*100:.1f}cm (测量)')
        sx = cw / cur_w if cur_w > 0.0001 else 1.0
        sy = cd / cur_d if cur_d > 0.0001 else 1.0
        sz = ch / cur_h if cur_h > 0.0001 else 1.0
    else:
        print(f'  目标高度: {TARGET_HEIGHT*100:.1f}cm (固定)')
        sz = TARGET_HEIGHT / cur_h if cur_h > 0.0001 else 1.0
        sx = sz
        sy = sz

    print(f'  缩放: ×{sx:.3f} ×{sy:.3f} ×{sz:.3f}')

    verts[:, 0] *= sx
    verts[:, 1] *= sy
    verts[:, 2] *= sz
    verts[:, 2] -= verts[:, 2].min()

    faces = np.asarray(mesh.triangles)
    scaled = trimesh.Trimesh(vertices=verts, faces=faces)

    (OUT / 'models').mkdir(exist_ok=True)
    obj_path = OUT / 'models' / f'{name}_v16.obj'
    ply_path = OUT / 'models' / f'{name}_v16.ply'
    scaled.export(str(obj_path))
    scaled.export(str(ply_path))

    vs = scaled.vertices
    print(f'  最终: {vs[:,0].max()-vs[:,0].min():.3f}×'
          f'{vs[:,1].max()-vs[:,1].min():.3f}×'
          f'{vs[:,2].max()-vs[:,2].min():.3f}m, '
          f'{len(vs)} 顶点')
    print(f'  已导出: {obj_path}')
    print(f'  已导出: {ply_path}')
    return scaled


# ╔══════════════════════════════════════════════════════════╗
# ║  Main                                                    ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    global OUT

    parser = argparse.ArgumentParser(
        description='V16 家具重建 — 标定+扫描同次 + High Accuracy')
    parser.add_argument('--name', default='chair', help='家具名称')
    parser.add_argument('--skip-capture', action='store_true',
                        help='跳过采集, 用已有数据重建')
    parser.add_argument('--skip-calib', action='store_true',
                        help='跳过交互标定, 使用已有 calibrate.json')
    parser.add_argument('--poisson-depth', type=int, default=POISSON_DEPTH)
    parser.add_argument('--voxel', type=float, default=VOXEL_SIZE)
    args = parser.parse_args()

    OUT = BASE / f'output/v16/{args.name}'
    OUT.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'  V16 家具重建 [{args.name}]')
    print(f'  标定+扫描同次 + High Accuracy 预设')
    print(f'  输出: {OUT}')
    print('=' * 60)

    # ── 标定 + 采集 ──
    calib = None
    calib_path = OUT / 'calibrate.json'

    if args.skip_capture:
        print('跳过采集 (--skip-capture)')
        if not calib_path.exists():
            print(f'错误: 标定文件不存在 {calib_path}')
            sys.exit(1)
        with open(calib_path) as f:
            calib = json.load(f)
    else:
        if args.skip_calib and calib_path.exists():
            with open(calib_path) as f:
                calib = json.load(f)
            print(f'使用已有标定: {calib_path}')
        elif args.skip_calib:
            print('错误: --skip-calib 但标定文件不存在')
            sys.exit(1)

        if calib is not None:
            calib = do_calibrate_and_capture(calib_input=calib)
        else:
            calib = do_calibrate_and_capture(calib_input=None)

    if calib is None:
        print('标定/采集失败')
        sys.exit(1)

    # ── 抠图 + 测量 ──
    masks_dict, chair_dims = do_yolo_and_measure(calib)
    if masks_dict is None:
        print('抠图失败')
        sys.exit(1)

    if chair_dims is None:
        print('警告: 自动测量失败, 使用默认高度 8.8cm')
        chair_dims = {
            'height_m': 0.088, 'width_m': 0.06,
            'depth_m': 0.05, 'n_measured': 0
        }

    # ── 深度重建 ──
    mesh = do_depth_reconstruction(masks_dict, calib)
    if mesh is None:
        print('重建失败')
        sys.exit(1)

    # ── 缩放导出 ──
    final = do_scale_export(mesh, chair_dims, args.name)

    # ── 日志 ──
    log = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'V16_CALIB_SAME_SESSION',
        'n_frames': N_FRAMES,
        'poisson_depth': args.poisson_depth,
        'voxel_size': args.voxel,
        'chair_dims': chair_dims,
        'changes_from_v15': [
            'high_accuracy_preset',
            'calib_and_capture_same_session',
            'calibration_rotation_axis',
            'calibration_rotation_center',
            'closer_camera_distance',
        ],
        'output_dir': str(OUT),
    }
    with open(OUT / 'log_v16.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\n{"=" * 60}')
    print(f'  V16 完成!')
    print(f'  椅子尺寸: {chair_dims["height_m"]*100:.1f}×'
          f'{chair_dims["width_m"]*100:.1f}×{chair_dims["depth_m"]*100:.1f}cm')
    print(f'  输出目录: {OUT}')
    print(f'  {"=" * 60}')


if __name__ == '__main__':
    main()
