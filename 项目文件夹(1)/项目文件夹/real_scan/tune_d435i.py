#!/usr/bin/env python3
"""D435i 参数调优工具: 实时滑动条调节传感器参数 + 后处理滤镜.

用法: python3 tune_d435i.py

键盘:
  q     — 退出
  s     — 保存当前参数到 tune_preset.json
  l     — 加载上次保存的参数
  r     — 恢复 D435i 出厂默认
  f     — 开关后处理滤镜
  h     — 开关填补区域高亮
  1-4   — 切换预设 (1=默认 2=高覆盖 3=高质量 4=家具扫描)

Trackbar 说明:
  传感器参数 (每次修改实时生效):
    Laser Power    — IR 激光功率 (0~360, 默认150, 越高穿透越强)
    Confidence Thr — 深度置信度阈值 (0~15, 默认3, 越低点越多噪声越大)
    Exposure       — 曝光时间 (1~165000, 默认~16600=30fps)
    Gain           — 传感器增益 (16~248, 默认16, 暗场景调高)
    Disp Shift     — 视差偏移 (0~255, 越小近距离精度越高)

  后处理滤镜:
    Spat Magnitude — 空域滤波强度 (0~5, 默认2)
    Spat Alpha     — 空域平滑权重 (0.0~1.0, 步长0.05, 默认0.5)
    Spat Delta     — 空域边缘保持 (1~100, 默认20, 越大边缘越糊)
    Spat HolesFill — 空域填孔等级 (0~5, 默认0)

    Temp Alpha     — 时域平滑权重 (0.0~1.0, 步长0.05, 默认0.4)
    Temp Delta     — 时域边缘保持 (1~100, 默认20)

    Hole Fill      — 最终填孔模式 (0=FAR 1=NEAREST 2=NEAR)
"""

import json
import os
import numpy as np
import cv2
import pyrealsense2 as rs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESET_FILE = os.path.join(SCRIPT_DIR, 'tune_preset.json')

# ── Trackbar 窗口名 ──
WIN_CONTROL = 'D435i Tuning Controls'
WIN_PREVIEW = 'D435i Preview'

# ── 传感器参数默认值 ──
DEFAULTS = {
    'laser_power': 150,       # 0-360
    'confidence': 3,          # 0-15
    'exposure': 16600,        # 1-165000
    'gain': 16,               # 16-248
    'disp_shift': 0,          # 0-255

    'spat_magnitude': 2,      # 0-5
    'spat_alpha': 50,         # 0-100 → /100
    'spat_delta': 20,         # 1-100
    'spat_holes': 0,          # 0-5

    'temp_alpha': 40,         # 0-100 → /100
    'temp_delta': 20,         # 1-100

    'hole_fill': 1,           # 0-2
}

# ── 预设 ──
PRESETS = {
    '1_default': {
        'laser_power': 150, 'confidence': 3, 'exposure': 16600, 'gain': 16,
        'disp_shift': 0, 'spat_magnitude': 2, 'spat_alpha': 50, 'spat_delta': 20,
        'spat_holes': 0, 'temp_alpha': 40, 'temp_delta': 20, 'hole_fill': 1,
    },
    '2_short_exp': {
        # 缩短曝光 + 降增益 → 减轻镜面眩光
        'laser_power': 150, 'confidence': 5, 'exposure': 2000, 'gain': 16,
        'disp_shift': 0, 'spat_magnitude': 2, 'spat_alpha': 50, 'spat_delta': 20,
        'spat_holes': 2, 'temp_alpha': 40, 'temp_delta': 20, 'hole_fill': 1,
    },
    '3_passive': {
        # 关 IR 发射器, 纯靠环境光纹理做双目匹配 (需要环境光足够亮)
        'laser_power': 0, 'confidence': 3, 'exposure': 16600, 'gain': 16,
        'disp_shift': 0, 'spat_magnitude': 2, 'spat_alpha': 50, 'spat_delta': 20,
        'spat_holes': 2, 'temp_alpha': 40, 'temp_delta': 20, 'hole_fill': 1,
    },
    '4_darkroom': {
        # 关室内灯 + IR 发射器打满 + 强力填孔 (暗环境专用)
        'laser_power': 360, 'confidence': 1, 'exposure': 33000, 'gain': 32,
        'disp_shift': 0, 'spat_magnitude': 3, 'spat_alpha': 40, 'spat_delta': 10,
        'spat_holes': 5, 'temp_alpha': 30, 'temp_delta': 15, 'hole_fill': 2,
    },
}

state = dict(DEFAULTS)  # 当前参数
use_filter = False
highlight_fill = False
depth_scale = 0.001


def nothing(x):
    pass


def create_control_window():
    """创建 Trackbar 控制面板."""
    cv2.namedWindow(WIN_CONTROL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CONTROL, 520, 520)

    cv2.createTrackbar('Laser Power', WIN_CONTROL,
                       DEFAULTS['laser_power'], 360, nothing)
    cv2.createTrackbar('Confidence Thr', WIN_CONTROL,
                       DEFAULTS['confidence'], 15, nothing)
    cv2.createTrackbar('Exposure', WIN_CONTROL,
                       DEFAULTS['exposure'], 165000, nothing)
    cv2.createTrackbar('Gain', WIN_CONTROL,
                       DEFAULTS['gain'], 248, nothing)
    cv2.createTrackbar('Disp Shift', WIN_CONTROL,
                       DEFAULTS['disp_shift'], 255, nothing)

    cv2.createTrackbar('─ Post Filters ─', WIN_CONTROL, 0, 1, nothing)  # separator

    cv2.createTrackbar('Spat Magnitude', WIN_CONTROL,
                       DEFAULTS['spat_magnitude'], 5, nothing)
    cv2.createTrackbar('Spat Alpha', WIN_CONTROL,
                       DEFAULTS['spat_alpha'], 100, nothing)
    cv2.createTrackbar('Spat Delta', WIN_CONTROL,
                       DEFAULTS['spat_delta'], 100, nothing)
    cv2.createTrackbar('Spat HolesFill', WIN_CONTROL,
                       DEFAULTS['spat_holes'], 5, nothing)

    cv2.createTrackbar('Temp Alpha', WIN_CONTROL,
                       DEFAULTS['temp_alpha'], 100, nothing)
    cv2.createTrackbar('Temp Delta', WIN_CONTROL,
                       DEFAULTS['temp_delta'], 100, nothing)

    cv2.createTrackbar('Hole Fill', WIN_CONTROL,
                       DEFAULTS['hole_fill'], 2, nothing)


def read_trackbars():
    """读取所有 Trackbar 值 → state 字典."""
    state['laser_power'] = cv2.getTrackbarPos('Laser Power', WIN_CONTROL)
    state['confidence'] = cv2.getTrackbarPos('Confidence Thr', WIN_CONTROL)
    state['exposure'] = cv2.getTrackbarPos('Exposure', WIN_CONTROL)
    state['gain'] = cv2.getTrackbarPos('Gain', WIN_CONTROL)
    state['disp_shift'] = cv2.getTrackbarPos('Disp Shift', WIN_CONTROL)

    state['spat_magnitude'] = cv2.getTrackbarPos('Spat Magnitude', WIN_CONTROL)
    state['spat_alpha'] = cv2.getTrackbarPos('Spat Alpha', WIN_CONTROL)
    state['spat_delta'] = cv2.getTrackbarPos('Spat Delta', WIN_CONTROL)
    state['spat_holes'] = cv2.getTrackbarPos('Spat HolesFill', WIN_CONTROL)

    state['temp_alpha'] = cv2.getTrackbarPos('Temp Alpha', WIN_CONTROL)
    state['temp_delta'] = cv2.getTrackbarPos('Temp Delta', WIN_CONTROL)

    state['hole_fill'] = cv2.getTrackbarPos('Hole Fill', WIN_CONTROL)


def set_trackbars(params):
    """从 state 同步到 Trackbar."""
    cv2.setTrackbarPos('Laser Power', WIN_CONTROL, int(params['laser_power']))
    cv2.setTrackbarPos('Confidence Thr', WIN_CONTROL, int(params['confidence']))
    cv2.setTrackbarPos('Exposure', WIN_CONTROL, int(params['exposure']))
    cv2.setTrackbarPos('Gain', WIN_CONTROL, int(params['gain']))
    cv2.setTrackbarPos('Disp Shift', WIN_CONTROL, int(params['disp_shift']))
    cv2.setTrackbarPos('Spat Magnitude', WIN_CONTROL, int(params['spat_magnitude']))
    cv2.setTrackbarPos('Spat Alpha', WIN_CONTROL, int(params['spat_alpha']))
    cv2.setTrackbarPos('Spat Delta', WIN_CONTROL, int(params['spat_delta']))
    cv2.setTrackbarPos('Spat HolesFill', WIN_CONTROL, int(params['spat_holes']))
    cv2.setTrackbarPos('Temp Alpha', WIN_CONTROL, int(params['temp_alpha']))
    cv2.setTrackbarPos('Temp Delta', WIN_CONTROL, int(params['temp_delta']))
    cv2.setTrackbarPos('Hole Fill', WIN_CONTROL, int(params['hole_fill']))


def apply_sensor_settings(depth_sensor, params):
    """将当前参数写入 D435i 传感器."""
    opts = {
        rs.option.laser_power: 'laser_power',
        rs.option.confidence_threshold: 'confidence',
        rs.option.exposure: 'exposure',
        rs.option.gain: 'gain',
    }
    for opt, key in opts.items():
        if depth_sensor.supports(opt):
            try:
                val = float(params[key])
                rng = depth_sensor.get_option_range(opt)
                val = max(rng.min, min(rng.max, val))
                depth_sensor.set_option(opt, val)
            except Exception:
                pass

    # Emitter: laser_power > 0 就开启
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled,
                                1 if params['laser_power'] > 0 else 0)

    # 曝光优先 (关闭自动曝光才能手动调)
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
    # Gain 也手动
    if depth_sensor.supports(rs.option.enable_auto_white_balance):
        try:
            depth_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        except Exception:
            pass


def apply_disparity_shift(dev, shift):
    """通过 Advanced Mode 设置 disparity shift."""
    try:
        adv = rs.rs400_advanced_mode(dev)
        dt = adv.get_depth_table()
        dt.disparityShift = int(shift)
        adv.set_depth_table(dt)
    except Exception:
        pass


def update_filters(params):
    """根据当前参数重建后处理滤镜."""
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, int(params['spat_magnitude']))
    spatial.set_option(rs.option.filter_smooth_alpha, params['spat_alpha'] / 100.0)
    spatial.set_option(rs.option.filter_smooth_delta, int(params['spat_delta']))
    spatial.set_option(rs.option.holes_fill, int(params['spat_holes']))

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, params['temp_alpha'] / 100.0)
    temporal.set_option(rs.option.filter_smooth_delta, int(params['temp_delta']))

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, int(params['hole_fill']))

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    return spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth


def apply_filters(depth_frame, filters):
    """视差域滤波管道: depth→disparity→spatial→temporal→depth→hole_fill."""
    spatial, temporal, hole_filling, d2d, d2d_inv = filters
    filtered = d2d.process(depth_frame)
    filtered = spatial.process(filtered)
    filtered = temporal.process(filtered)
    filtered = d2d_inv.process(filtered)
    filtered = hole_filling.process(filtered)
    return filtered


def save_preset(params, filepath=PRESET_FILE):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)
    print(f'参数已保存 → {filepath}')


def load_preset(filepath=PRESET_FILE):
    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


def colorize_depth(depth_frame, colorizer):
    """将深度帧转为 BGR 色彩图."""
    return np.asanyarray(colorizer.colorize(depth_frame).get_data())


def draw_hud(color_img, depth_data, params):
    """叠加 HUD: 中心十字 + 距离 (与 1.py 一致)."""
    h, w = color_img.shape[:2]
    cx, cy = w // 2, h // 2
    overlay = color_img.copy()

    # 中心 3x3 中值距离
    roi = depth_data[max(cy-1,0):cy+2, max(cx-1,0):cx+2]
    valid = roi[roi > 0]
    dist_m = np.median(valid) * depth_scale if len(valid) > 0 else 0

    cv2.putText(overlay, f'Center Distance: {dist_m:.3f}m',
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(overlay, f'Depth Scale: {depth_scale:.6f}',
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.drawMarker(overlay, (cx, cy), (0, 0, 255),
                  cv2.MARKER_CROSS, 20, 2)
    return overlay


def highlight_filled_areas(depth_colormap, raw_mask):
    """在填补区域覆盖白色网格, 区分真实深度 vs 填补深度."""
    fill_mask = (raw_mask == 0)
    if fill_mask.sum() == 0:
        return depth_colormap

    overlay = depth_colormap.copy()
    overlay[fill_mask] = (overlay[fill_mask] * 0.3 + np.array([60, 60, 60]) * 0.7).astype(np.uint8)

    grid = np.zeros_like(fill_mask)
    grid[::20, :] = True
    grid[:, ::20] = True
    grid_mask = fill_mask & grid
    overlay[grid_mask] = [255, 255, 255]
    return overlay


def main():
    global depth_scale

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print('未检测到 RealSense 设备')
        return

    dev = devices[0]
    print(f'相机: {dev.get_info(rs.camera_info.name)}')
    print(f'序列号: {dev.get_info(rs.camera_info.serial_number)}')
    print(f'固件: {dev.get_info(rs.camera_info.firmware_version)}')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    depth_sensor = profile.get_device()
    depth_sensors = depth_sensor.query_sensors()
    for s in depth_sensors:
        if s.is_depth_sensor():
            depth_sensor = s
            break

    print(f'Depth Scale: {depth_scale:.6f}')

    # 控制窗口
    create_control_window()

    # 初始滤镜
    filters = update_filters(DEFAULTS)

    # 预热
    for _ in range(30):
        pipeline.wait_for_frames()

    global use_filter, highlight_fill

    print('\n' + '=' * 55)
    print('  D435i 参数调优工具')
    print('  q=退出  s=保存  l=加载  r=恢复出厂')
    print('  f=开关滤镜  h=开关填补高亮')
    print('  1=默认  2=短曝光(抗眩光)  3=被动模式(关IR)  4=暗室模式')
    print('=' * 55)

    # 上次保存的参数
    saved = load_preset()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # ── 读取 Trackbar ──
            read_trackbars()

            # ── 应用传感器参数 ──
            apply_sensor_settings(depth_sensor, state)

            # ── 更新滤镜 ──
            filters = update_filters(state)

            # ── 原始深度 (用于判断哪些是真实测到的点) ──
            raw_depth = np.asanyarray(depth_frame.get_data())
            valid_mask = (raw_depth > 0).astype(np.uint8)

            # ── 滤镜处理 ──
            if use_filter:
                filtered = apply_filters(depth_frame, filters)
                depth_for_vis = filtered
                filter_status = 'FILTERED'
            else:
                depth_for_vis = depth_frame
                filter_status = 'RAW'

            # ── 深度色彩图 ──
            depth_colormap = colorize_depth(depth_for_vis, colorizer)
            depth_colormap = cv2.resize(depth_colormap, (640, 480))

            # ── 填补区域高亮 ──
            if use_filter and highlight_fill:
                depth_colormap = highlight_filled_areas(depth_colormap, valid_mask)

            # ── 彩色图 HUD ──
            color_img = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_for_vis.get_data())
            info_overlay = draw_hud(color_img, depth_data, state)

            # ── 拼接显示 (与 1.py 一致) ──
            combined = np.hstack([info_overlay, depth_colormap])
            title_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            suffix = f' - [{filter_status}]' if use_filter else ''
            cv2.putText(title_bar, f"RealSense D435i - Color (Left) | Depth (Right){suffix}",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            display = np.vstack([title_bar, combined])

            cv2.imshow(WIN_PREVIEW, display)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                save_preset(state)

            elif key == ord('l'):
                loaded = load_preset()
                if loaded:
                    state.update(loaded)
                    set_trackbars(state)
                    print('参数已加载')

            elif key == ord('r'):
                state.update(DEFAULTS)
                set_trackbars(DEFAULTS)
                # 硬件重置高级模式参数
                try:
                    adv = rs.rs400_advanced_mode(dev)
                    dt = adv.get_depth_table()
                    dt.disparityShift = 0
                    adv.set_depth_table(dt)
                except Exception:
                    pass
                print('已恢复出厂默认')

            elif key == ord('f'):
                use_filter = not use_filter
                print(f'滤镜: {"开" if use_filter else "关"}')

            elif key == ord('h'):
                highlight_fill = not highlight_fill
                print(f'填补高亮: {"开" if highlight_fill else "关"}')

            elif key >= ord('1') and key <= ord('4'):
                preset_name = f'{chr(key)}_'
                for k, v in PRESETS.items():
                    if k.startswith(preset_name):
                        state.update(v)
                        set_trackbars(v)
                        # 应用 disparity shift
                        apply_disparity_shift(dev, v['disp_shift'])
                        apply_sensor_settings(depth_sensor, v)
                        print(f'预设已加载: {k}')
                        break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('相机已关闭')


if __name__ == '__main__':
    main()
