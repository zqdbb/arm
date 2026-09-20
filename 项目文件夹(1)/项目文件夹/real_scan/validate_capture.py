#!/usr/bin/env python3
"""采集数据验证工具 — 实时彩色图/深度图/YOLO mask/IR左右/单帧点云.

用法:
  python3 validate_capture.py                # 默认椅子
  python3 validate_capture.py --name table   # 桌子
  python3 validate_capture.py --name cabinet # 柜子
"""
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2, numpy as np, pyrealsense2 as rs, time, sys, argparse
from pathlib import Path
import open3d as o3d

BASE = Path(__file__).parent

# ── 解析参数 ──
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='chair', choices=['chair', 'table', 'cabinet'],
                    help='物体类型')
args = parser.parse_args()
ITEM = args.name

# 模型路径
CUSTOM_MODEL = BASE / 'custom_model/weights/best.pt'  # train.zip 自定义模型

# 类别映射  (chair 用 YOLOv8n-seg, table/cabinet 用自定义模型)
CLASS_MAP = {
    'chair':   {'classes': [56], 'name': 'chair', 'model': 'yolo'},
    'table':   {'classes': None, 'name': 'table', 'model': 'custom'},
    'cabinet': {'classes': None, 'name': 'cabinet', 'model': 'custom'},
}
W_FULL, H_FULL = 1280, 720
FPS = 15
ZOOM = 3  # 3× 数字变焦

# ── 全局变量 ──
pipe = None; cfg = None; profile = None; align = None
depth_sensor = None; depth_scale = 0.001
fx = fy = 0.0; ppx = ppy = 0.0
W, H = W_FULL // ZOOM, H_FULL // ZOOM  # 变焦后有效分辨率
camera_name = 'Unknown'; max_laser = 360
x0, y0 = (W_FULL - W*ZOOM)//2, (H_FULL - H*ZOOM)//2  # 裁剪偏移 (ZOOM=2时 x0=0 y0=0 因为刚好一半)
ir_enabled = False


def crop_zoom(img):
    """中心裁剪 ZOOM 倍"""
    if ZOOM == 1:
        return img
    h, w = img.shape[:2]
    x0_ = (w - w // ZOOM) // 2
    y0_ = (h - h // ZOOM) // 2
    return img[y0_:y0_ + h // ZOOM, x0_:x0_ + w // ZOOM]


def start_pipe(with_ir=False):
    global pipe, cfg, profile, depth_sensor, depth_scale, fx, fy, ppx, ppy, align, W, H, camera_name, max_laser

    try:
        if pipe is not None:
            pipe.stop()
    except Exception:
        pass

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W_FULL, H_FULL, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W_FULL, H_FULL, rs.format.bgr8, FPS)
    if with_ir:
        cfg.enable_stream(rs.stream.infrared, 1, W_FULL, H_FULL, rs.format.y8, FPS)
        cfg.enable_stream(rs.stream.infrared, 2, W_FULL, H_FULL, rs.format.y8, FPS)
    profile = pipe.start(cfg)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    # 检测相机型号
    camera_name = device.get_info(rs.camera_info.name) if hasattr(rs, 'camera_info') else 'Unknown'
    print(f'检测到相机: {camera_name}')

    # 从实际相机读取内参
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx_full = intr.fx
    fy_full = intr.fy
    ppx_full = intr.ppx
    ppy_full = intr.ppy

    # 变焦后内参: fx/fy 不变，光心偏移
    if ZOOM > 1:
        x0_ = (W_FULL - W_FULL // ZOOM) // 2
        y0_ = (H_FULL - H_FULL // ZOOM) // 2
        fx = fx_full
        fy = fy_full
        ppx = ppx_full - x0_
        ppy = ppy_full - y0_
        W = W_FULL // ZOOM
        H = H_FULL // ZOOM
    else:
        fx, fy = fx_full, fy_full
        ppx, ppy = ppx_full, ppy_full
        W, H = W_FULL, H_FULL

    # D405 近距离模式
    if 'D405' in camera_name:
        try:
            depth_sensor.set_option(rs.option.visual_preset, 2)  # Hand
            print('  visual_preset → Hand')
        except Exception as e:
            print(f'  visual_preset 失败: {e}')

    # D405 激光最大 100mW 左右，D435i 最大 360mW
    try:
        max_laser = depth_sensor.get_option_range(rs.option.laser_power).max
    except Exception:
        max_laser = 360
    try:
        depth_sensor.set_option(rs.option.laser_power, max_laser)
        print(f'  laser_power → {max_laser:.0f}')
    except Exception:
        pass

    print(f'  {W_FULL}x{H_FULL} → {W}x{H} (zoom={ZOOM}x)  '
          f'fx={fx:.1f} fy={fy:.1f} pp=({ppx:.1f},{ppy:.1f}) '
          f'scale={depth_scale:.4f}')


# ── 初次启动 ──
start_pipe(with_ir=False)

# ── YOLO ──
from ultralytics import YOLO

target_class = CLASS_MAP[ITEM]['name']
target_classes = CLASS_MAP[ITEM]['classes']
use_custom_model = CLASS_MAP[ITEM]['model'] == 'custom'

if use_custom_model:
    model_path = str(CUSTOM_MODEL)
    print(f'加载自定义模型: {model_path}')
else:
    model_path = str(BASE / 'yolov8n-seg.pt')
    print(f'加载 YOLOv8n-seg: {model_path}')

yolo = YOLO(model_path)
print(f'目标类别: {target_class}')

# ── 状态 ──
depth_min, depth_max = 0.20, 0.60
ir_enabled = False
paused = False

print()
print('=' * 60)
print('  键盘:')
print(f'  SPACE : 拍{ITEM}点云 ({"YOLO" if ITEM != "cabinet" else "自定义"} mask 内) → Open3D')
print('  F     : 拍全帧点云 → Open3D')
print('  I     : 开启/关闭 IR 左右图')
print('  Z     : 切换数字变焦 1×/2×')
print('  P     : 暂停/继续')
print('  ↑↓   : 调整深度显示范围')
print('  1-5   : 切换 visual preset')
print('  Q/ESC : 退出')
print('=' * 60)

vis = None
frame_idx = 0
yolo_mask = np.zeros((H, W), dtype=bool)
yolo_conf = 0.0
window_ok = False

# 尝试创建窗口
try:
    cv2.namedWindow('Capture Validator', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('Capture Validator', 1200, 500)

    # 显示一帧测试
    test_img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.putText(test_img, 'Loading...', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('Capture Validator', test_img)
    key = cv2.waitKey(100)
    if key >= 0 or cv2.getWindowProperty('Capture Validator', cv2.WND_PROP_VISIBLE) >= 0:
        window_ok = True
        print('GUI 窗口已创建')
    else:
        print('GUI 窗口不可用，使用终端模式 (按 SPACE 拍点云, Q 退出)')
except cv2.error:
    print('GUI 窗口不可用，使用终端模式 (按 SPACE 拍点云, Q 退出)')


def get_yolo_mask(color_bgr):
    kwargs = {'verbose': False}
    if target_classes is not None:
        kwargs['classes'] = target_classes
    results = yolo(color_bgr, **kwargs)
    mask = np.zeros((H, W), dtype=bool)
    conf = 0.0
    if results[0].masks is not None:
        best, best_c = None, 0
        for i in range(len(results[0].boxes)):
            if results[0].names[int(results[0].boxes.cls[i])] == target_class:
                c = float(results[0].boxes.conf[i])
                if c > best_c:
                    best_c = c
                    best = i
        if best is not None:
            m = results[0].masks.data[best].cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = m > 0.5
            conf = best_c
    return mask, conf


def make_pointcloud(color, depth_m, mask=None):
    if mask is not None and mask.sum() > 0:
        valid = (depth_m > 0) & (depth_m < 10) & mask
    else:
        valid = (depth_m > 0) & (depth_m < 10)
    if valid.sum() < 100:
        return None
    v, u = np.where(valid)
    z = depth_m[v, u]
    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    clr = color[v, u, :].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(clr)
    return pcd


def show_pointcloud(pcd, title):
    global vis
    if len(pcd.points) < 50:
        print('  点数不足，跳过')
        return
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=900, height=700)
        vis.add_geometry(coord)
        vis.add_geometry(pcd)
    else:
        vis.clear_geometries()
        vis.add_geometry(coord, reset_bounding_box=False)
        vis.add_geometry(pcd, reset_bounding_box=False)
    vc = vis.get_view_control()
    vc.set_front([0, 0, -1])
    vc.set_up([0, -1, 0])
    vc.set_lookat([0, 0, 0.4])
    vc.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()


while True:
    t0 = time.time()

    if not paused:
        try:
            frames = pipe.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            print('  丢帧，跳过...')
            continue

        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data())
        depth_m = depth.astype(np.float32) * depth_scale

        # 数字变焦裁剪
        color = crop_zoom(color)
        depth = crop_zoom(depth)
        depth_m = crop_zoom(depth_m)

        # YOLO (每 3 帧)
        if frame_idx % 3 == 0:
            yolo_mask, yolo_conf = get_yolo_mask(color)

        # IR
        if ir_enabled:
            ir_left = crop_zoom(np.asanyarray(frames.get_infrared_frame(1).get_data()))
            ir_right = crop_zoom(np.asanyarray(frames.get_infrared_frame(2).get_data()))
        else:
            ir_left = np.zeros((H, W), dtype=np.uint8)
            ir_right = np.zeros((H, W), dtype=np.uint8)

    # ── 深度热力图 ──
    d_clipped = np.clip(depth_m, depth_min, depth_max)
    d_norm = ((d_clipped - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    d_heat = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    d_heat[depth == 0] = (60, 60, 60)

    # ── YOLO mask 叠加 ──
    color_disp = color.copy()
    if yolo_mask.sum() > 0:
        overlay = color_disp.copy()
        overlay[yolo_mask] = (0, 255, 0)
        color_disp = cv2.addWeighted(color_disp, 0.7, overlay, 0.3, 0)
        cnts, _ = cv2.findContours(yolo_mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_disp, cnts, -1, (0, 255, 0), 2)

    # ── 统计 ──
    total = W * H
    valid_all = (depth > 0).sum()

    if yolo_mask.sum() > 100:
        d_chair = depth_m[yolo_mask]
        chair_valid = (d_chair > 0).sum()
        chair_total = yolo_mask.sum()
        chair_in_range = ((d_chair > depth_min) & (d_chair < depth_max)).sum()
        chair_pct = 100 * chair_valid / chair_total
        chair_med = np.median(d_chair[d_chair > 0]) if chair_valid > 0 else 0

        if chair_pct > 50:
            chair_color = (0, 255, 0)
        elif chair_pct > 20:
            chair_color = (0, 220, 255)
        elif chair_pct > 5:
            chair_color = (0, 140, 255)
        else:
            chair_color = (0, 0, 255)
    else:
        chair_valid = chair_total = chair_in_range = 0
        chair_pct = chair_med = 0.0
        chair_color = (150, 150, 150)

    # ── HUD ──
    hud_h = 140
    hud = np.zeros((hud_h, W, 3), dtype=np.uint8)

    # ── 状态栏（醒目）──
    if yolo_mask.sum() > 200 and chair_pct > 30:
        status_text = "READY - Chair in frame"
        status_bg = (0, 100, 0)
        status_fg = (0, 255, 0)
    elif yolo_mask.sum() > 100 and chair_pct > 10:
        status_text = "CHECK - Chair weak"
        status_bg = (0, 80, 80)
        status_fg = (0, 255, 255)
    else:
        status_text = "NO CHAIR - Adjust camera"
        status_bg = (0, 0, 100)
        status_fg = (0, 0, 255)

    cv2.rectangle(hud, (0, 0), (W, 28), status_bg, -1)
    cv2.putText(hud, status_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_fg, 2)
    ry = 48

    fps = 1.0 / max(time.time() - t0, 0.001)
    cv2.putText(hud, f'{camera_name}  Laser: {max_laser:.0f}  Zoom: {ZOOM}x  FPS: {fps:.0f}  IR: {"ON" if ir_enabled else "OFF"}',
                (10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    ry += 22
    cv2.putText(hud,
        f'Full: valid={valid_all:,} ({100*valid_all/total:.1f}%)  Range=[{depth_min:.2f},{depth_max:.2f}]m',
        (10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    ry += 20
    cv2.putText(hud,
        f'{ITEM.upper()} MASK: {int(chair_total):,}px  '
        f'DepthValid={chair_valid:,}/{chair_total:,} = {chair_pct:.1f}%  '
        f'InRange={chair_in_range:,}  Med={chair_med:.3f}m  YOLO_conf={yolo_conf:.2f}',
        (10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.5, chair_color, 1)
    ry += 22
    cv2.putText(hud,
        f'SPACE={ITEM.title()}PCD  F=FullPCD  I=IR  Z=Zoom  P=Pause  Q=Quit',
        (10, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)

    # ── 画面合成 ──
    frame_idx += 1

    if window_ok:
        def resize_half(img):
            return cv2.resize(img, (W // 2, H // 2))

        color_half = resize_half(color_disp)
        depth_half = resize_half(d_heat)

        if ir_enabled:
            ir_l_disp = cv2.cvtColor(ir_left, cv2.COLOR_GRAY2BGR)
            ir_r_disp = cv2.cvtColor(ir_right, cv2.COLOR_GRAY2BGR)
            if yolo_mask.sum() > 0:
                cnts2, _ = cv2.findContours(yolo_mask.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ir_l_disp, cnts2, -1, (0, 255, 0), 2)
                cv2.drawContours(ir_r_disp, cnts2, -1, (0, 255, 0), 2)
            ir_l_half = resize_half(ir_l_disp)
            ir_r_half = resize_half(ir_r_disp)
            cv2.putText(ir_l_half, 'IR LEFT', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(ir_r_half, 'IR RIGHT', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            top_row = np.hstack([color_half, depth_half])
            bot_row = np.hstack([ir_l_half, ir_r_half])
            mid = np.vstack([top_row, bot_row])
        else:
            mid = np.hstack([color_half, depth_half])

        display = np.vstack([hud, mid])
        dw = mid.shape[1]
        dh = hud_h + mid.shape[0]
        scale = min(1200 / dw, 900 / dh, 1.0)
        display_small = cv2.resize(display, (int(dw * scale), int(dh * scale)))
        cv2.imshow('Capture Validator', display_small)
        key = cv2.waitKey(1)

    else:
        # 终端模式：每 10 帧打印一次核心指标
        if frame_idx % 10 == 0:
            print(f'  [{frame_idx:04d}]  '
                  f'全图有效={100*valid_all/total:.1f}%  |  '
                  f'{ITEM.upper()}: {int(chair_total):,}px  '
                  f'深度有效={chair_pct:.1f}%  '
                  f'中值={chair_med:.3f}m  '
                  f'YOLO={yolo_conf:.2f}')
        # 第 30 帧自动拍点云
        if frame_idx == 30 and yolo_mask.sum() > 100:
            pcd = make_pointcloud(color, depth_m, yolo_mask)
            if pcd is not None:
                pts_arr = np.asarray(pcd.points)
                dists = np.linalg.norm(pts_arr, axis=1)
                keep = (dists > 0.20) & (dists < 0.80)
                pcd = pcd.select_by_index(np.where(keep)[0])
                out_dir = BASE / 'output/validate'
                out_dir.mkdir(parents=True, exist_ok=True)
                path = str(out_dir / f'{ITEM}_auto.ply')
                o3d.io.write_point_cloud(path, pcd)
                cv2.imwrite(str(out_dir / 'color_auto.jpg'), color,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f'\n  已自动保存: {path}')
                print(f'  {ITEM}点云点数: {len(pcd.points):,}')
            print('\n按 Enter 退出...')
            input()
            break
        if frame_idx > 100:
            print('超时，退出')
            break
        key = -1
        time.sleep(0.05)

    if key in (27, ord('q')):
        break
    elif key == 32:  # SPACE
        print(f'\n[{ITEM}点云] mask={yolo_mask.sum():,}px')
        pcd = make_pointcloud(color, depth_m, yolo_mask)
        if pcd is None:
            print(f'  {ITEM}区域有效深度不足！')
            continue
        print(f'  点数: {len(pcd.points):,}')
        pts_arr = np.asarray(pcd.points)
        dists = np.linalg.norm(pts_arr, axis=1)
        keep = (dists > 0.20) & (dists < 0.80)
        pcd = pcd.select_by_index(np.where(keep)[0])
        print(f'  裁剪 0.2-0.8m: {len(pcd.points):,} 点')
        show_pointcloud(pcd, f'{ITEM.title()} Point Cloud')
        out_dir = BASE / 'output/validate'
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f'{ITEM}_{frame_idx:04d}.ply')
        o3d.io.write_point_cloud(path, pcd)
        cv2.imwrite(str(out_dir / f'color_{frame_idx:04d}.jpg'), color,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f'  已保存: {path}')

    elif key == ord('f') or key == ord('F'):
        print(f'\n[全帧点云]')
        pcd = make_pointcloud(color, depth_m)
        if pcd is None:
            print('  有效深度不足！')
            continue
        print(f'  点数: {len(pcd.points):,}')
        pts_arr = np.asarray(pcd.points)
        dists = np.linalg.norm(pts_arr, axis=1)
        keep = dists < 0.80
        pcd = pcd.select_by_index(np.where(keep)[0])
        print(f'  裁剪后: {len(pcd.points):,}')
        show_pointcloud(pcd, 'Full Frame Point Cloud')
        out_dir = BASE / 'output/validate'
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f'full_{frame_idx:04d}.ply')
        o3d.io.write_point_cloud(path, pcd)
        print(f'  已保存: {path}')

    elif key in (ord('i'), ord('I')):
        ir_enabled = not ir_enabled
        print(f'  IR → {"ON" if ir_enabled else "OFF"} (重启流...)')
        start_pipe(with_ir=ir_enabled)

    elif key in (ord('z'), ord('Z')):
        ZOOM = 1 if ZOOM == 2 else 2
        print(f'  Zoom → {ZOOM}x (重启流...)')
        start_pipe(with_ir=ir_enabled)

    elif key in (ord('p'), ord('P')):
        paused = not paused
        print(f'  {"暂停" if paused else "继续"}')
    elif key == 82 or key == 0:
        depth_max = min(2.0, depth_max + 0.05)
    elif key == 84 or key == 1:
        depth_max = max(depth_min + 0.05, depth_max - 0.05)
    elif ord('1') <= key <= ord('5'):
        presets = {1: 'Default', 2: 'Hand', 3: 'HighAccuracy', 4: 'HighDensity', 5: 'MediumDensity'}
        cur = key - ord('0')
        try:
            depth_sensor.set_option(rs.option.visual_preset, float(cur))
            print(f'  Preset → {presets[cur]}')
        except Exception as e:
            print(f'  Preset 切换失败 (需停流): {e}')

pipe.stop()
if vis is not None:
    vis.destroy_window()
cv2.destroyAllWindows()
print('Done.')
