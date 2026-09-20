#!/usr/bin/env python3
"""真机扫描全流程: 拍照识别 → 喷涂零点 → 多角度扫描 → 拼接.

用法: python3 real_scan.py
前提: 先运行 calibrate.py 完成标定.

输出 (在 real_scan/output/):
  classify_rgb.png    — 抠图后的家具RGB (给你的分类模型)
  scan.ply            — 完整拼接点云 (Y-up, 给你的分割模型)
"""

import json
import os
import sys
import time
import math
import numpy as np
import open3d as o3d
import cv2
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/xie/项目文件夹/real_scan')
sys.path.insert(0, '/home/xie/furniture_spray_deploy/ultralytics')
sys.path.insert(0, '/home/xie/项目文件夹/Depth-Anything-V2')
from config import *
from utils import world_rotate_z_around, save_ply, post_process
from turntable import TurntableController
from depth_anything_v2.dpt import DepthAnythingV2


# ── Depth-Anything-V2 深度补全 ──
_DA_CKPT = '/home/xie/项目文件夹/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
_da_model = None


def _get_da():
    global _da_model
    if _da_model is None:
        print(f'  加载 Depth-Anything-V2 Small...')
        _da_model = DepthAnythingV2(
            encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        _da_model.load_state_dict(torch.load(_DA_CKPT, map_location='cpu'))
        _da_model.eval()
    return _da_model


def da_predict_depth(img_bgr):
    """RGB(BGR) → Depth-Anything-V2 相对深度 [0,1] near→far."""
    model = _get_da()
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scale = 518 / max(h_orig, w_orig)
    new_h = round(h_orig * scale / 14) * 14
    new_w = round(w_orig * scale / 14) * 14
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        depth = model(tensor)
    depth = F.interpolate(depth.unsqueeze(1), size=(h_orig, w_orig),
                          mode='bicubic', align_corners=False).squeeze().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def da_align_to_metric(pred_depth, d435i_depth):
    """相对深度 → metric (mm). 用 D435i 中心区域最小二乘."""
    h, w = d435i_depth.shape
    cy, cx = h // 2, w // 2
    center = np.zeros((h, w), dtype=bool)
    center[cy - h // 8:cy + h // 8, cx - w // 8:cx + w // 8] = True
    valid = (d435i_depth > 300) & (d435i_depth < 2000) & center
    if valid.sum() < 100:
        valid = (d435i_depth > 300) & (d435i_depth < 2000)
    indices = np.where(valid)
    n = min(5000, len(indices[0]))
    if n < 50:
        return d435i_depth.copy(), 1.0, 0.0
    sel = np.random.choice(len(indices[0]), n, replace=False)
    rows, cols = indices[0][sel], indices[1][sel]
    disp = 1.0 / (pred_depth[rows, cols] + 0.01)
    metric = d435i_depth[rows, cols]
    A = np.column_stack([disp, np.ones(n)])
    coeff, _, _, _ = np.linalg.lstsq(A, metric, rcond=None)
    a, b = coeff
    result = a / (pred_depth + 0.01) + b
    result = np.clip(result, 0, 10000)
    return result.astype(np.float32), a, b


def da_depth_to_points(depth_metric, fx, fy, ppx, ppy):
    """Metric深度 (mm) → 相机坐标点云 (m)."""
    h, w = depth_metric.shape
    u, v = np.arange(w), np.arange(h)
    uu, vv = np.meshgrid(u, v)
    z = depth_metric / 1000.0
    valid = z > 0.05
    x = (uu - ppx) * z / fx
    y = (vv - ppy) * z / fy
    return np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)


def da_frame_to_world(color_img, depth_frame, R, t_calib, plate_z,
                       fx, fy, ppx, ppy,
                       clip_radius_m=0.14, plate_band=0.004):
    """单帧: RGB → DA-V2稠密深度 → D435i对齐 → 过滤转台面 → world坐标."""
    d435i_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    return da_frame_to_world_arr(color_img, d435i_depth, R, t_calib, plate_z,
                                  fx, fy, ppx, ppy, clip_radius_m, plate_band)


def da_frame_to_world_arr(color_img, d435i_depth, R, t_calib, plate_z,
                           fx, fy, ppx, ppy,
                           clip_radius_m=0.14, plate_band=0.004):
    """da_frame_to_world 的 numpy 数组版本 (支持变焦后的 depth array).

    返回 (pts_world, depth_metric).
    """
    # 1. DA-V2 预测 + D435i 对齐
    pred_depth = da_predict_depth(color_img)
    depth_metric, a, b = da_align_to_metric(pred_depth, d435i_depth)

    # 2. 用 world-Z 过滤转台面
    # 转台面在 world 中 Z=plate_z, 对应深度 ≈ d_plate
    # 简化: 过滤深度 ≈ 转台面深度的像素
    # 转台面在图像底部的深度值 → 作为盘面参考深度
    h, w = depth_metric.shape
    bottom = depth_metric[int(h * 0.85):, :]
    bottom_valid = bottom[(bottom > 100) & (bottom < 5000)]
    if len(bottom_valid) > 200:
        disc_ref = np.median(bottom_valid)
        # 距盘面深度 ±15mm 的像素 → 删除
        near_disc = np.abs(d435i_depth - disc_ref) < 15
    else:
        near_disc = np.zeros_like(d435i_depth, dtype=bool)

    # 3. 深度 → 相机点云 (跳过盘面像素)
    h_d, w_d = depth_metric.shape
    u, v = np.arange(w_d), np.arange(h_d)
    uu, vv = np.meshgrid(u, v)
    z = depth_metric / 1000.0
    valid = (z > 0.05) & (~near_disc)
    x = (uu - ppx) * z / fx
    y = (vv - ppy) * z / fy
    pts_cam = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)

    # 4. 世界坐标变换
    pts_world = (R @ pts_cam.T).T + t_calib

    # 5. 用 plate_z 二次过滤 (world坐标系直接砍转台面高度)
    is_plate = (np.abs(pts_world[:, 2] - plate_z) < plate_band)
    r_plate = np.sqrt(pts_world[:, 0]**2 + pts_world[:, 1]**2)
    is_disc = is_plate & (r_plate < clip_radius_m + 0.02)
    pts_world = pts_world[~is_disc]

    # 6. 径向裁剪
    r = np.sqrt((pts_world[:, 0])**2 + (pts_world[:, 1])**2)
    pts_world = pts_world[r < clip_radius_m]

    return pts_world.astype(np.float32), depth_metric


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        sys.exit(f'标定文件不存在: {CALIB_FILE}\n请先运行 python3 calibrate.py')
    with open(CALIB_FILE, 'r') as f:
        data = json.load(f)
    R = np.array(data['R'])
    t = np.array(data['t'])
    plate_z = data['plate_z']
    disc_radius = data.get('radius_m', 0.1)
    # world坐标原点已在旋转中心 (t 做了平移), 所以 rotation_center = [0, 0]
    rotation_center = np.array([0.0, 0.0])
    print(f'标定: plate_z={plate_z:.4f}, disc_radius={disc_radius*100:.1f}cm, '
          f'rotation_center=[0, 0] (world原点)')
    return R, t, plate_z, disc_radius, rotation_center


def cam_to_world(pts, R, t_calib):
    """相机坐标 → 世界坐标 (Z-up)."""
    return (R @ pts.T).T + t_calib


_YOLO_MODEL_PATH = os.path.join(os.path.expanduser('~'),
    'furniture_spray_deploy/models/best.pt')
_yolo_model = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print(f'  加载 YOLO: {_YOLO_MODEL_PATH}')
        _yolo_model = YOLO(_YOLO_MODEL_PATH)
    return _yolo_model


def yolo_find_front(pipeline, align, tt, colorizer=None):
    """YOLO 置信度正面查找: 30粗搜 5精调 设零点.

    流程:
      1. 转台旋转 0~330 每30一停, 拍照跑 YOLO, 记录最高置信度
      2. 找出置信度最高的 30 区间
      3. 区间内每 5 精扫, 确认峰值
      4. 转到峰值角度, 设为零点
    """
    yolo = _get_yolo()

    print('\n' + '=' * 55)
    print('  YOLO 正面查找')
    print('  粗搜: 12个角度 (30步长)  精调: 15 (5步长)')
    print('=' * 55)

    # ── Phase 1: 30 粗搜 ──
    coarse_angles = list(range(0, 360, 30))
    coarse_results = []

    cv2.namedWindow('YOLO Front Search - Coarse', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Front Search - Coarse', 640, 360)

    for i, target_deg in enumerate(coarse_angles):
        print(f'\n  [{i+1}/{len(coarse_angles)}] 转到 {target_deg} ...')
        tt.move_absolute(target_deg, speed=TURNTABLE_DEFAULT_SPEED)
        tt.wait_stop(timeout=30)
        time.sleep(0.3)

        frames = pipeline.wait_for_frames()
        aligned_f = align.process(frames)
        color_frame = aligned_f.get_color_frame()
        if not color_frame:
            coarse_results.append((target_deg, 0.0, 'none'))
            continue
        color_img = np.asanyarray(color_frame.get_data())

        results = yolo(color_img, verbose=False)[0]
        top_conf = 0.0
        top_cls = 'none'
        if results.boxes is not None and len(results.boxes) > 0:
            top_conf = float(results.boxes.conf[0])
            top_cls = yolo.names.get(int(results.boxes.cls[0]), '?')

        coarse_results.append((target_deg, top_conf, top_cls))
        print(f'    检测: {top_cls} conf={top_conf:.3f}')

        cv2.putText(color_img, f'{top_cls} {top_conf:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(color_img, f'{target_deg} deg', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('YOLO Front Search - Coarse', color_img)
        cv2.waitKey(1)

    cv2.destroyWindow('YOLO Front Search - Coarse')

    # 打印粗搜结果
    print('\n  ── 粗搜结果 ──')
    for deg, conf, cls_name in coarse_results:
        bar = '#' * int(conf * 40)
        print(f'    {deg:3d}: {cls_name:8s} {conf:.4f} {bar}')

    best = max(coarse_results, key=lambda x: x[1])
    best_deg, best_conf = best[0], best[1]
    print(f'\n  粗搜峰值: {best_deg} conf={best_conf:.4f}')

    if best_conf < 0.1:
        print('  警告: 置信度过低, 默认不旋转')
        cv2.destroyAllWindows()
        tt.zero()
        return 0.0, 0.0

    # ── Phase 2: 5 精调 ──
    fine_start = (best_deg - 15) % 360
    fine_angles = [(fine_start + 5 * i) % 360 for i in range(7)]
    fine_results = []

    cv2.namedWindow('YOLO Front Search - Fine', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Front Search - Fine', 640, 360)

    for target_deg in fine_angles:
        print(f'  精调: 转到 {target_deg} ...')
        tt.move_absolute(target_deg, speed=TURNTABLE_DEFAULT_SPEED)
        tt.wait_stop(timeout=30)
        time.sleep(0.3)

        frames = pipeline.wait_for_frames()
        aligned_f = align.process(frames)
        color_frame = aligned_f.get_color_frame()
        if not color_frame:
            fine_results.append((target_deg, 0.0))
            continue
        color_img = np.asanyarray(color_frame.get_data())

        results = yolo(color_img, verbose=False)[0]
        top_conf = 0.0
        if results.boxes is not None and len(results.boxes) > 0:
            top_conf = float(results.boxes.conf[0])

        fine_results.append((target_deg, top_conf))

        cv2.putText(color_img, f'Fine: {target_deg} deg conf={top_conf:.4f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('YOLO Front Search - Fine', color_img)
        cv2.waitKey(1)

    cv2.destroyWindow('YOLO Front Search - Fine')

    fine_best = max(fine_results, key=lambda x: x[1])
    fine_deg, fine_conf = fine_best

    print('\n  ── 精调结果 ──')
    for deg, conf in fine_results:
        marker = ' <-- BEST' if deg == fine_deg else ''
        print(f'    {deg:3d}: {conf:.4f}{marker}')
    print(f'\n  YOLO 正面: {fine_deg} conf={fine_conf:.4f}')

    # 转到最佳角度, 设为零点
    tt.move_absolute(fine_deg, speed=TURNTABLE_DEFAULT_SPEED)
    tt.wait_stop(timeout=30)
    time.sleep(0.3)
    tt.zero()

    return fine_deg, fine_conf


def find_spraying_zero(pts_world, rotation_center):
    """PCA 找家具主方向, 返回需要旋转的角度使侧面正对相机.

    相机大致从 +Y 方向看向原点. 旋转使家具主方向对齐 X 轴 (侧面正对相机).
    返回旋转角度(度).
    """
    # 过滤: 转台中心附近, 在转台面上方
    r = np.sqrt((pts_world[:, 0] - rotation_center[0])**2 +
                (pts_world[:, 1] - rotation_center[1])**2)
    mask = (r < 0.12) & (pts_world[:, 2] > -0.05)
    pts = pts_world[mask]
    if len(pts) < 100:
        print('  PCA: 点数不足, 默认不旋转')
        return 0.0

    # 对水平面(XY)做PCA
    centered = pts[:, :2] - pts[:, :2].mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 主方向 (最大特征值)
    main_dir = eigenvectors[:, -1]
    ratio = eigenvalues[-1] / max(eigenvalues[0], 1e-10)
    print(f'  PCA 主方向: [{main_dir[0]:.3f}, {main_dir[1]:.3f}] '
          f'(特征值比 {ratio:.1f}:1)')

    # 如果接近圆形对称 (ratio < 1.5), 不旋转
    if ratio < 1.5:
        print(f'  PCA: 接近圆形对称, 不旋转')
        return 0.0

    # 主方向当前角度, 旋转使它对齐 X 轴
    angle = math.degrees(math.atan2(main_dir[1], main_dir[0]))
    # 对齐最近的 X 轴方向 (0° 或 90° 或 180° 或 270°)
    # 选 4 个候选方向中最接近的
    targets = [0, 90, 180, 270]
    best = min(targets, key=lambda t: abs((t - angle + 180) % 360 - 180))
    rot = best - angle
    # 归一化到 [-90, 90]
    if rot > 90:
        rot -= 180
    elif rot < -90:
        rot += 180

    print(f'  主方向当前={angle:.1f}°, 对齐{best}° → 转台旋转 {rot:.1f}°')
    return rot


def manual_zero_calibration(pipeline, align, pc, colorizer, tt):
    """手动零点标定: 用户旋转转台直到家具侧面正对相机, 按回车确认.

    键盘:
      A/D 或 ← →  — 微调角度 (1°/5°)
      SPACE/ENTER  — 确认当前位置为喷涂零点
      Q            — 跳过(不旋转)
    """
    print('\n' + '=' * 55)
    print('  喷涂零点标定')
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

        # 画中心十字参考线
        cx, cy = w // 2, h // 2
        cv2.line(preview, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 1)
        cv2.line(preview, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 1)

        cv2.putText(preview, f'Rotated: {total_rotated:+.0f} deg', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, 'A/← : -1 deg | D/→ : +1 deg', (10, h + h//2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(preview, 'Q/W : -5/+5 deg | ENTER : confirm', (10, h + h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(preview, 'Align furniture face to camera center, then ENTER',
                    (w // 4, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow('Zero Calibration - AD=rotate | ENTER=confirm', preview)

        key = cv2.waitKey(1) & 0xFF

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


def capture_frame(pipeline, align, pc):
    """采集一帧: RGB图 + 深度帧 + 点云(相机坐标). 返回 (color_img, depth_frame, pts_cam)."""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None

    color_img = np.asanyarray(color_frame.get_data())
    pc.map_to(aligned)
    vertices = np.asanyarray(pc.calculate(depth_frame).get_vertices()).view(np.float32)
    pts = vertices.reshape(-1, 3)
    valid = np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0.01)
    pts = pts[valid]

    return color_img, depth_frame, pts


def capture_stable_frame(pipeline, align, pc, n_warmup=5):
    """预热后采集稳定帧."""
    for _ in range(n_warmup):
        pipeline.wait_for_frames()
        time.sleep(0.05)
    return capture_frame(pipeline, align, pc)


def zoom_frame(color_img, depth_frame, zoom=1.0):
    """数码变焦: 裁切中心 → 缩放回原分辨率. 返回 (color_zoomed, depth_zoomed, fx, fy, ppx, ppy)."""
    if zoom <= 1.0:
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_arr = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        return color_img, depth_arr, intr.fx, intr.fy, intr.ppx, intr.ppy

    h, w = color_img.shape[:2]
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2

    # 裁切 RGB 和深度
    color_crop = color_img[y0:y0 + crop_h, x0:x0 + crop_w]
    depth_arr = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_crop = depth_arr[y0:y0 + crop_h, x0:x0 + crop_w]

    # 缩放回原分辨率
    color_zoom = cv2.resize(color_crop, (w, h), interpolation=cv2.INTER_LANCZOS4)
    depth_zoom = cv2.resize(depth_crop, (w, h), interpolation=cv2.INTER_NEAREST)

    # 调整内参
    intr = depth_frame.profile.as_video_stream_profile().intrinsics
    fx = intr.fx * zoom
    fy = intr.fy * zoom
    ppx = (intr.ppx - x0) * zoom
    ppy = (intr.ppy - y0) * zoom

    return color_zoom, depth_zoom, fx, fy, ppx, ppy


def crop_center(img, size=160, y_offset=-90):
    """裁剪图像中心区域, y_offset负值=向上偏移."""
    h, w = img.shape[:2]
    x0 = max(0, (w - size) // 2)
    y0 = max(0, (h - size) // 2 + y_offset)
    y0 = max(0, min(y0, h - size))
    return img[y0:y0+size, x0:x0+size]


def remove_background_rgb(color_img, depth_frame, bg_depth_frame=None):
    """背景差分: 用深度图差分提取前景家具.

    如果有 bg_depth_frame (空转台的深度), 则用差分mask.
    否则用深度截断: 只保留比中心最近深度+30cm内的像素.
    """
    h, w = color_img.shape[:2]
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

    if bg_depth_frame is not None:
        bg_depth = np.asanyarray(bg_depth_frame.get_data()).astype(np.float32)
        # 深度差分: 有家具的地方比空转台更近(深度更小)
        diff = bg_depth - depth
        fg_mask = (depth > 0) & (bg_depth > 0) & (diff > 80)  # 差距>8cm=家具
    else:
        # 取中心点深度作参考, 只保留中心深度±0.2m范围
        cx, cy = w // 2, h // 2
        center_d = depth[cy-10:cy+10, cx-10:cx+10]
        center_valid = center_d[(center_d > 100) & (center_d < 5000)]
        if len(center_valid) < 10:
            return crop_center(color_img)
        ref = np.median(center_valid)
        fg_mask = (depth > ref - 100) & (depth < ref + 200)  # 参考深度-10cm~+20cm

    if fg_mask.sum() < 500:
        return crop_center(color_img)

    # 找前景包围盒
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    if len(y_idx) < 5 or len(x_idx) < 5:
        return crop_center(color_img)

    y0, y1 = y_idx[0], y_idx[-1]
    x0, x1 = x_idx[0], x_idx[-1]
    # 加10% padding
    pad_y = max(10, (y1 - y0) // 10)
    pad_x = max(10, (x1 - x0) // 10)
    y0 = max(0, y0 - pad_y)
    y1 = min(h, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    return color_img[y0:y1, x0:x1]


def remove_disc_ransac(pts, disc_radius, distance_threshold=0.004, expected_z=None):
    """RANSAC找转台水平面 → 只删平面内点, 保留同高度但不在平面上的椅腿.

    expected_z: 期望的转台面Z坐标. 设置后, 找到的平面必须在expected_z±2cm内才删除.
                用于合并后防止误删椅子座面.
    """
    if len(pts) < 100:
        return pts

    # 只在转台半径内找平面
    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    in_disc = r < disc_radius + 0.01
    disc_candidate = pts[in_disc]

    if len(disc_candidate) < 500:
        return pts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(disc_candidate)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)

    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n_norm = n / np.linalg.norm(n)

    # 检查是否是水平面 (法向量接近Z轴)
    z_alignment = abs(n_norm[2])
    if z_alignment < 0.85:
        print(f'  RANSAC切盘: 平面不水平 (Z_align={z_alignment:.2f}), 跳过')
        return pts

    # 检查平面高度是否匹配预期转台面 (防止误删椅子座面)
    if expected_z is not None:
        plane_z = -d / c if abs(c) > 0.01 else 0
        z_error = abs(plane_z - expected_z)
        if z_error > 0.02:
            print(f'  RANSAC切盘: 平面Z={plane_z:.4f} 距预期转台面{expected_z:.4f}太远'
                  f'({z_error*100:.1f}cm), 跳过 (防止误删座面)')
            return pts

    # 计算整个点云到该平面的距离
    all_dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)
    is_plane = all_dists < distance_threshold

    # 只删在转台半径内且在平面上的点
    is_disc = is_plane & in_disc
    result = pts[~is_disc]

    n_removed = len(pts) - len(result)
    n_str = f'[{n_norm[0]:.2f},{n_norm[1]:.2f},{n_norm[2]:.2f}]'
    print(f'  RANSAC切盘: n={n_str} d={d:.4f}, thr={distance_threshold*1000:.0f}mm, '
          f'移除 {n_removed:,} 点 ({n_removed/len(pts)*100:.0f}%)')
    return result


def remove_disc_zhist(pts, disc_radius, z_band=0.004, expected_z=None):
    """合并后Z直方图切盘: 24帧盘面点堆积在同一Z高度形成峰值, 精确铲除.

    expected_z: 限定搜索范围在expected_z±2cm内, 避免误删座面高度.
    """
    if len(pts) < 1000:
        return pts

    r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    in_disc = pts[r < disc_radius + 0.01]
    if len(in_disc) < 500:
        return pts

    z_all = in_disc[:, 2]

    # 如果指定了expected_z, 限定搜索范围
    if expected_z is not None:
        z_mask = np.abs(z_all - expected_z) < 0.02
        z_search = z_all[z_mask]
        if len(z_search) < 100:
            print(f'  Z-hist切盘: expected_z={expected_z:.4f} 附近点数不足, 跳过')
            return pts
    else:
        z_search = z_all

    # Z histogram
    hist, edges = np.histogram(z_search, bins=100)
    peak_idx = np.argmax(hist)
    z_peak = (edges[peak_idx] + edges[peak_idx + 1]) / 2
    peak_count = hist[peak_idx]

    if peak_count < max(len(z_search) * 0.02, 30):
        print(f'  Z-hist切盘: Z_peak={z_peak:.4f} 峰值不够显著({peak_count/len(z_search)*100:.0f}%), 跳过')
        return pts

    # 删掉 Z_peak±z_band 内且在转台半径内的点
    is_disc = (np.abs(pts[:, 2] - z_peak) < z_band) & (r < disc_radius + 0.01)
    result = pts[~is_disc]

    n_removed = len(pts) - len(result)
    print(f'  Z-hist切盘: Z_peak={z_peak:.4f}, band={z_band*1000:.0f}mm, '
          f'峰值={peak_count}点({peak_count/len(z_search)*100:.0f}%), '
          f'移除 {n_removed:,}/{len(pts):,} ({n_removed/len(pts)*100:.0f}%)')
    return result


def clip_radius(pts_world, rotation_center, radius=0.06):
    """径向过滤: 只保留旋转中心附近, 不限制Z (留到后面RANSAC切面)."""
    r = np.sqrt((pts_world[:, 0] - rotation_center[0])**2 +
                (pts_world[:, 1] - rotation_center[1])**2)
    return pts_world[r < radius]


def remove_plane_ransac(pts, distance_threshold=0.006, z_direction_only=True):
    """RANSAC拟合并移除转台面.

    z_direction_only=True: 只移除平面法向量接近Z轴的点 (只切转台面, 不误切家具面).
    返回 (above_plane_points, plane_model).
    """
    if len(pts) < 100:
        return pts, None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=2000)

    a, b, c, d = plane_model
    n = np.array([a, b, c])
    n_norm = n / np.linalg.norm(n)

    if z_direction_only:
        # 检查法向量是否接近Z轴 (转台面应该是水平的)
        z_alignment = abs(n_norm[2])
        if z_alignment < 0.7:  # 与Z轴夹角>45°, 不是转台面
            print(f'  平面法向量偏离Z轴 ({z_alignment:.2f}), 跳过切面')
            return pts, None

    inlier_set = set(inliers)
    above = pts[[i for i in range(len(pts)) if i not in inlier_set]]

    n_str = f'[{n_norm[0]:.2f},{n_norm[1]:.2f},{n_norm[2]:.2f}]'
    print(f'  RANSAC切面: 法向量={n_str} d={d:.4f}')
    print(f'  移除 {len(inliers):,} 点 (转台面), 保留 {len(above):,} 点 (家具)')
    return above, plane_model


# ── SAM 背景去除 (Segment Anything Model) ──

SAM_CHECKPOINT = os.path.join(os.path.expanduser('~'), '.cache/sam/sam_vit_b_01ec64.pth')
_sam_predictor = None
_sam_available = None


def _check_sam():
    global _sam_available
    if _sam_available is None:
        if not os.path.exists(SAM_CHECKPOINT):
            print(f'  SAM模型未找到: {SAM_CHECKPOINT}')
            _sam_available = False
        else:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                _sam_available = True
            except ImportError:
                print('  segment-anything 未安装: pip install segment-anything')
                _sam_available = False
    return _sam_available


def _get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None and _check_sam():
        from segment_anything import sam_model_registry, SamPredictor
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'  加载SAM模型 ({device})...')
        sam = sam_model_registry['vit_b'](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def remove_background_sam(img_bgr, depth_frame=None):
    """用SAM分割家具前景: 中心点+四角负点 → 选出最优mask."""
    if not _check_sam():
        return remove_background_rgb(img_bgr, depth_frame)

    # 1. SAM预测 — 中心点正提示 + 四角负提示
    predictor = _get_sam_predictor()
    if predictor is None:
        return remove_background_rgb(img_bgr, depth_frame)

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    # 中心区域3x3正提示 (椅子在画面中央30%区域)
    cx, cy = w // 2, h // 2
    span_x, span_y = int(w * 0.12), int(h * 0.12)
    xs = [cx - span_x, cx, cx + span_x]
    ys = [cy - span_y, cy, cy + span_y]
    grid_pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    # 四角+四边中点负提示
    corners = np.array([
        [5, 5], [w//2, 5], [w-5, 5],
        [5, h-5], [w//2, h-5], [w-5, h-5],
        [5, h//2], [w-5, h//2],
    ], dtype=np.float32)

    point_coords = np.vstack([grid_pts, corners]).astype(np.float32)
    point_labels = np.array([1]*len(grid_pts) + [0]*len(corners))

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False,
    )

    mask = masks[0].astype(np.float32)
    print(f'  SAM mask score={scores[0]:.3f}, 前景={mask.sum()/mask.size*100:.0f}%')

    # 2. 用深度辅助去掉底盘: 底部20%区域的深度估计底盘距离
    if depth_frame is not None:
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        h_d, w_d = depth.shape
        # 取底部20%行的有效深度
        bottom_strip = depth[int(h_d*0.8):, :]
        bottom_valid = bottom_strip[(bottom_strip > 100) & (bottom_strip < 5000)]
        if len(bottom_valid) > 200:
            disc_depth = np.median(bottom_valid)
            # 底盘 = 深度≈disc_depth ± 30mm
            disc_region = (depth > 100) & (depth < 5000) & (np.abs(depth - disc_depth) < 15)
            mask[disc_region] = 0
            print(f'  SAM去底盘: disc_depth={disc_depth:.0f}mm, '
                  f'移除 {disc_region.sum()/disc_region.size*100:.0f}% 像素')

    # 3. 蒙版合成: 前景保留, 背景白色
    mask_3ch = np.stack([mask] * 3, axis=-1)
    fg = img_bgr.astype(np.float32) * mask_3ch
    bg = np.full_like(img_bgr, 255, dtype=np.float32) * (1 - mask_3ch)
    result = (fg + bg).astype(np.uint8)

    # 4. 裁剪到mask区域
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    if len(y_idx) > 10 and len(x_idx) > 10:
        pad = 15
        y0 = max(0, y_idx[0] - pad)
        y1 = min(h, y_idx[-1] + pad)
        x0 = max(0, x_idx[0] - pad)
        x1 = min(w, x_idx[-1] + pad)
        result = result[y0:y1, x0:x1]

    return result


def main():
    import pyrealsense2 as rs

    skip_zero = '--skip-zero' in sys.argv
    manual_zero = '--manual-zero' in sys.argv
    # 数码变焦倍数 (默认1.5)
    zoom_val = 1.0
    for arg in sys.argv:
        if arg.startswith('--zoom='):
            try:
                zoom_val = float(arg.split('=')[1])
            except ValueError:
                pass

    if skip_zero:
        print('>>> 跳过零点标定, 当前位置作为零点 <<<')
    elif manual_zero:
        print('>>> 手动零点标定模式 <<<')
    else:
        print('>>> YOLO 自动正面查找模式 (默认) <<<')
    if zoom_val > 1.0:
        print(f'>>> 数码变焦: {zoom_val:.1f}x <<<')

    R_calib, t_calib, plate_z, disc_radius, rotation_center = load_calibration()

    # 预加载 Depth-Anything-V2 模型
    print('\n[初始化 AI 深度补全]')
    _get_da()
    print(f'  Depth-Anything-V2 Small 就绪')

    # ── 连接相机 ──
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit('未检测到 RealSense 相机')
    dev = devices[0]
    print(f'相机: {dev.get_info(rs.camera_info.name)}')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, DEPTH_FPS)
    profile = pipeline.start(cfg)

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    pc = rs.pointcloud()

    for _ in range(30):
        pipeline.wait_for_frames()
        time.sleep(0.05)

    # 获取相机内参
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    _intr = color_profile.get_intrinsics()
    _fx, _fy, _ppx, _ppy = _intr.fx, _intr.fy, _intr.ppx, _intr.ppy
    print(f'相机内参: fx={_fx:.1f} fy={_fy:.1f} cx={_ppx:.1f} cy={_ppy:.1f}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 清理旧文件
    for f in os.listdir(OUTPUT_DIR):
        if f in ('classify_rgb.png', 'scan.ply'):
            os.remove(os.path.join(OUTPUT_DIR, f))
            print(f'  清理旧文件: {f}')

    # ── 连接转台 ──
    print(f'连接转台 {TURNTABLE_PORT} ...')
    tt = TurntableController(port=TURNTABLE_PORT)
    tt.open()

    try:
        total_rotation = 0.0  # 跟踪总旋转量, 用于回零

        # ============================================================
        # Step 1: 放家具 + 拍识别照
        # ============================================================
        print('\n' + '=' * 55)
        print('  请把家具放在转台上, 然后按 SPACE')
        print('=' * 55)

        cv2.namedWindow('Place furniture - SPACE to capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Place furniture - SPACE to capture', 960, 360)

        classify_rgb = None
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

            cv2.putText(preview, 'Place furniture & press SPACE', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, 'RGB', (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, 'Depth', (w + 10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Place furniture - SPACE to capture', preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                classify_rgb, classify_depth, pts_first = capture_stable_frame(pipeline, align, pc)
                if classify_rgb is None or len(pts_first) < 1000:
                    print(f'采集失败 ({len(pts_first) if pts_first is not None else 0}点), 重试')
                    continue
                print(f'采集: {len(pts_first):,} 有效点')
                break
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                tt.close()
                return

        cv2.destroyAllWindows()

        # 保存分类用RGB: 原图直接保存
        classify_path = os.path.join(OUTPUT_DIR, 'classify_rgb.png')
        cv2.imwrite(classify_path, classify_rgb)
        print(f'>>> 分类原图: {classify_path} ({classify_rgb.shape[1]}x{classify_rgb.shape[0]})')

        # ============================================================
        # Step 2: 喷涂零点标定 (YOLO 置信度正面查找)
        # ============================================================
        if skip_zero:
            print('\n[喷涂零点] 跳过, 当前位置设为零点')
            tt.zero()
        elif manual_zero:
            total_rotated = manual_zero_calibration(pipeline, align, pc, colorizer, tt)
            print(f'\n[喷涂零点] 手动旋转 {total_rotated:+.0f}  → 设为零点')
            tt.zero()
        else:
            # YOLO 自动正面查找 (30粗搜 + 5精调)
            yolo_best_deg, yolo_conf = yolo_find_front(pipeline, align, tt, colorizer)
            print(f'[喷涂零点] YOLO 正面={yolo_best_deg} conf={yolo_conf:.4f}')
        total_rotation = 0.0

        # ============================================================
        # Step 3: 多角度扫描 (每15°一帧, 共24帧, 匹配仿真质量)
        # ============================================================
        SCAN_ANGLES = list(range(0, 360, 15))  # [0, 15, 30, ..., 345]
        print(f'\n[扫描] {len(SCAN_ANGLES)} 帧, 每15° (匹配仿真24帧密度)')

        frames_ply = {}
        total_rotation = 0.0  # 跟踪总旋转量用于回零

        cv2.namedWindow('Scanning...', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Scanning...', 960, 360)

        for i, target_deg in enumerate(SCAN_ANGLES):
            # 绝对运动到目标角度 (避免累积偏差)
            print(f'\n[{i+1}/{len(SCAN_ANGLES)}] 旋转到 {target_deg}°')
            tt.move_absolute(target_deg, speed=TURNTABLE_DEFAULT_SPEED)
            tt.wait_stop(timeout=30)
            total_rotation = target_deg
            time.sleep(0.3)

            # DA-V2 深度补全: 变焦 → RGB → 稠密深度 → D435i对齐 → 过滤转台 → world坐标
            t0 = time.time()
            color_img, depth_frame, _ = capture_stable_frame(pipeline, align, pc)
            if color_img is None:
                print(f'  采集失败, 重试...')
                color_img, depth_frame, _ = capture_stable_frame(pipeline, align, pc)
                if color_img is None:
                    print(f'  跳过 {target_deg}°')
                    continue

            # 数码变焦 (默认1.5x)
            if zoom_val > 1.0:
                color_zoom, depth_zoom, zfx, zfy, zppx, zppy = \
                    zoom_frame(color_img, depth_frame, zoom_val)
            else:
                depth_zoom = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                color_zoom = color_img
                zfx, zfy, zppx, zppy = _fx, _fy, _ppx, _ppy

            pts_world, depth_metric = da_frame_to_world_arr(
                color_zoom, depth_zoom, R_calib, t_calib, plate_z,
                zfx, zfy, zppx, zppy,
                clip_radius_m=0.14, plate_band=0.004)
            t1 = time.time()

            # 保存点云
            frames_ply[target_deg] = pts_world

            print(f'  [{i+1}/{len(SCAN_ANGLES)}] {target_deg:3d}°: '
                  f'{len(pts_world):,}点 ({t1-t0:.1f}s)')

            # 预览: RGB(变焦) | D435i深度(变焦) | DA-V2补全深度
            h, w = color_zoom.shape[:2]
            da_viz = np.clip(depth_metric / 4000 * 255, 0, 255).astype(np.uint8)
            da_viz_color = cv2.applyColorMap(da_viz, cv2.COLORMAP_INFERNO)
            d435i_viz = np.clip(depth_zoom / 4000 * 255, 0, 255).astype(np.uint8)
            d435i_viz_color = cv2.applyColorMap(d435i_viz, cv2.COLORMAP_INFERNO)
            preview = np.hstack([color_zoom, d435i_viz_color, da_viz_color])
            cv2.putText(preview, f'{target_deg} deg | DA-V2 {len(pts_world):,}pts', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(preview, 'RGB', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(preview, 'D435i', (w+10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.putText(preview, 'DA-V2', (w*2+10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.imshow('Scanning...', preview)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        if len(frames_ply) < 3:
            print(f'有效帧不足 ({len(frames_ply)}), 扫描失败')
            pipeline.stop()
            tt.close()
            return

        # ============================================================
        # Step 4: 拼接 — 反旋对齐 → 合并 → 后处理 (DA-V2已过滤转台)
        # ============================================================
        print(f'\n[拼接] {len(frames_ply)} 帧 (DA-V2已过滤转台)...')

        angles = sorted(frames_ply.keys())
        ref_angle = angles[0]

        # 反旋对齐 + 合并
        aligned = []
        for deg in angles:
            rot_angle = (deg - ref_angle)
            pts_aligned = world_rotate_z_around(frames_ply[deg], rot_angle, rotation_center)
            aligned.append(pts_aligned)
            print(f'  {deg:3d}° → 反旋 {rot_angle:+.0f}°: {len(pts_aligned):,}点')

        merged = np.vstack(aligned)
        print(f'合并: {len(merged):,} 点')

        # 统计离群点去除 + 半径滤波: 底盘残余稀疏点 → 当噪点删
        n_before = len(merged)
        pcd_tmp = o3d.geometry.PointCloud()
        pcd_tmp.points = o3d.utility.Vector3dVector(merged)
        pcd_tmp, _ = pcd_tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_tmp, _ = pcd_tmp.remove_radius_outlier(nb_points=10, radius=0.015)
        merged = np.asarray(pcd_tmp.points)
        print(f'  统计去噪+半径滤波: {n_before:,} → {len(merged):,}')

        # MLS 平滑: 消除多帧拼接的"双墙"效应 (来自仿真 v17)
        if len(merged) > 500:
            print('[MLS] 局部平面投影平滑...')
            pcd_mls = o3d.geometry.PointCloud()
            pcd_mls.points = o3d.utility.Vector3dVector(merged)
            pcd_mls = pcd_mls.voxel_down_sample(voxel_size=0.0015)
            kdtree = o3d.geometry.KDTreeFlann(pcd_mls)
            pts_arr = np.asarray(pcd_mls.points)
            smoothed = np.zeros_like(pts_arr)
            mls_radius = 0.006
            mls_min_nb = 8
            for i in range(len(pts_arr)):
                [_, idx, _] = kdtree.search_radius_vector_3d(pts_arr[i], mls_radius)
                if len(idx) < mls_min_nb:
                    smoothed[i] = pts_arr[i]
                    continue
                nb = pts_arr[idx]
                center = nb.mean(axis=0)
                cov = (nb - center).T @ (nb - center)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]
                v = pts_arr[i] - center
                smoothed[i] = pts_arr[i] - np.dot(v, normal) * normal
            merged = smoothed
            print(f'  MLS平滑: {len(merged):,} 点')

        # 后处理 (降采样 → 去噪 → DBSCAN → Y-up)
        print('[后处理] 降采样 → 去噪 → DBSCAN → Y-up ...')
        pts_final = post_process(merged, voxel_size=VOXEL_SIZE,
                                 outlier_nb=OUTLIER_NB, outlier_radius=OUTLIER_RADIUS,
                                 dbscan_eps=DBSCAN_EPS, dbscan_min=DBSCAN_MIN)
        if len(pts_final) == 0:
            print('后处理结果为空!')
            pipeline.stop()
            tt.close()
            return

        # 上下翻转 (viewer Y-up 方向修正)
        pts_final[:, 1] *= -1

        scan_path = SCAN_OUTPUT
        save_ply(pts_final, scan_path)
        fsize = os.path.getsize(scan_path)
        print(f'\n[完成] Y-up → {scan_path}')
        print(f'  点数: {len(pts_final):,}  |  文件: {fsize:,} bytes')

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        # 回到喷涂零点 (绝对运动, total_rotation 已跟踪)
        print(f'转台返回零点... (从 {total_rotation:.0f}°)')
        tt.move_absolute(0, speed=max(TURNTABLE_DEFAULT_SPEED, 20000))
        tt.wait_stop(timeout=60)
        tt.close()
        print('转台已归零.')


if __name__ == '__main__':
    main()
