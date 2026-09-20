#!/usr/bin/env python3
"""
D435i 增强采集: 每角度多帧平均降噪 + 时域滤波.
解决深色木头深度噪声问题.
"""
import numpy as np
import cv2
import json
import sys
import time
from pathlib import Path
import pyrealsense2 as rs

BASE_DIR = Path(__file__).parent

STEP_ANGLE = 5
W, H = 640, 480
DEPTH_FPS = 30
N_AVG = 7                    # 每角度采样帧数 (时域平均)
RGB_EXPOSURE = 156
RGB_GAIN = 64
LASER_POWER = 150

OUTPUT_DIR = BASE_DIR / 'output' / 'dense_scan_v2'
COLOR_DIR = OUTPUT_DIR / 'color'
DEPTH_DIR = OUTPUT_DIR / 'depth'


def setup_d435i(profile):
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()

    # RGB 手动曝光
    sensors = dev.query_sensors()
    for s in sensors:
        if s.is_color_sensor() or 'RGB' in str(s.get_info(rs.camera_info.name)):
            s.set_option(rs.option.enable_auto_exposure, 0)
            s.set_option(rs.option.exposure, RGB_EXPOSURE)
            s.set_option(rs.option.gain, RGB_GAIN)
            s.set_option(rs.option.enable_auto_white_balance, 0)
            print(f'RGB: 曝光={RGB_EXPOSURE} 增益={RGB_GAIN} 白平衡=锁定')
            break

    depth_sensor.set_option(rs.option.laser_power, LASER_POWER)
    depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
    print(f'激光功率: {LASER_POWER}  深度预设: High Accuracy')

    # 启用深度后处理滤波器
    return depth_sensor


def main():
    if len(sys.argv) > 1:
        global STEP_ANGLE
        STEP_ANGLE = int(sys.argv[1])

    n_frames = 360 // STEP_ANGLE
    print('=' * 55)
    print(f'  D435i 增强采集: {STEP_ANGLE}° × {n_frames} 帧')
    print(f'  每角度 {N_AVG} 帧平均  |  {W}×{H} @ {DEPTH_FPS}fps')
    print('=' * 55)

    # 转台
    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台: 已归零')

    # 相机
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, DEPTH_FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, DEPTH_FPS)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # 稳定 + 手动曝光
    print('等待稳定 (2s)...')
    for _ in range(DEPTH_FPS * 2):
        pipe.wait_for_frames()
    setup_d435i(profile)

    # 深度后处理滤波器
    temporal = rs.temporal_filter(smooth_alpha=0.4, smooth_delta=20, persistence_control=3)
    hole_filling = rs.hole_filling_filter(mode=1)  # 从邻近像素填充

    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    # 保存内参
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    intrinsic_data = {
        'width': color_intr.width, 'height': color_intr.height,
        'intrinsic_matrix': [color_intr.fx, 0, 0, 0, color_intr.fy, 0,
                             color_intr.ppx, color_intr.ppy, 1],
        'depth_intrinsic_matrix': [depth_intr.fx, 0, 0, 0, depth_intr.fy, 0,
                                    depth_intr.ppx, depth_intr.ppy, 1],
        'step_angle': STEP_ANGLE, 'n_frames': n_frames, 'camera': 'D435i',
        'n_avg_frames': N_AVG,
    }
    with open(OUTPUT_DIR / 'camera_intrinsic.json', 'w') as f:
        json.dump(intrinsic_data, f, indent=2)
    print(f'内参: fx={color_intr.fx:.1f} fy={color_intr.fy:.1f}')

    # 采集
    angles = list(range(0, 360, STEP_ANGLE))
    print(f'\n开始采集 {n_frames} 帧...')
    t0 = time.time()

    for i, target_deg in enumerate(angles):
        tt.move_absolute(target_deg, speed=10000)
        tt.wait_stop(timeout=30)
        time.sleep(0.4)

        color_frames = []
        depth_frames = []

        for _ in range(N_AVG):
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            # 时域滤波
            depth_filtered = temporal.process(depth_frame)
            depth_filtered = hole_filling.process(depth_filtered)

            color_frames.append(np.asanyarray(color_frame.get_data()).astype(np.float32))
            depth_frames.append(np.asanyarray(depth_filtered.get_data()).astype(np.float32))

        # 中值/平均
        color_avg = np.median(color_frames, axis=0).astype(np.uint8)
        depth_avg = np.median(depth_frames, axis=0).astype(np.uint16)

        cv2.imwrite(str(COLOR_DIR / f'{i:06d}.jpg'), color_avg)
        cv2.imwrite(str(DEPTH_DIR / f'{i:06d}.png'), depth_avg)

        if (i + 1) % 12 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_frames - i - 1)
            valid_pct = (depth_avg > 0).sum() / depth_avg.size * 100
            print(f'  [{i+1}/{n_frames}] {target_deg:3d}°  '
                  f'深度有效{valid_pct:.0f}%  已用{elapsed:.0f}s  剩余~{eta:.0f}s')

    pipe.stop()
    elapsed = time.time() - t0
    print(f'\n采集完成! {n_frames} 帧 / {elapsed:.0f}s')
    print(f'输出: {OUTPUT_DIR}/')
    print(f'下一步: calibrate_furniture.py 标定, 然后 depth_fusion.py 重建')


if __name__ == '__main__':
    main()
