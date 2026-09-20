#!/usr/bin/env python3
"""
密集采集脚本: 5° 步长 72 帧, 固定 RGB 曝光/白平衡.
用于后续 SGBM 多视角立体匹配 + 深度融合 → 替代 D435i 深度.
"""
import numpy as np
import cv2
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── 采集参数 ──
STEP_ANGLE = 5              # 步进角度 (5° = 72 帧)
RGB_EXPOSURE = 156          # RGB 手动曝光 (默认 156, 根据环境调)
RGB_GAIN = 64               # RGB 手动增益 (默认 64)
RGB_BRIGHTNESS = 0
RGB_CONTRAST = 50
RGB_GAMMA = 300
LASER_POWER = 150           # 红外激光功率 (深度用作参考, 调低减少发热)
DEPTH_FPS = 30

OUTPUT_DIR = BASE_DIR / 'output' / 'dense_scan'
COLOR_DIR = OUTPUT_DIR / 'color'
DEPTH_DIR = OUTPUT_DIR / 'depth'


def setup_camera(pipe, cfg, profile):
    """设置手动曝光和白平衡."""
    import pyrealsense2 as rs
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()

    # RGB 传感器手动控制
    sensors = dev.query_sensors()
    rgb_sensor = None
    for s in sensors:
        if s.is_color_sensor() or str(s.get_info(rs.camera_info.name)).startswith('RGB'):
            rgb_sensor = s
            break

    if rgb_sensor is None:
        print('⚠ 未找到 RGB 传感器, 跳过手动曝光设置')
    else:
        rgb_sensor.set_option(rs.option.enable_auto_exposure, 0)
        rgb_sensor.set_option(rs.option.exposure, RGB_EXPOSURE)
        rgb_sensor.set_option(rs.option.gain, RGB_GAIN)
        rgb_sensor.set_option(rs.option.brightness, RGB_BRIGHTNESS)
        rgb_sensor.set_option(rs.option.contrast, RGB_CONTRAST)
        rgb_sensor.set_option(rs.option.gamma, RGB_GAMMA)
        rgb_sensor.set_option(rs.option.enable_auto_white_balance, 0)
        print(f'RGB 手动: 曝光={RGB_EXPOSURE} 增益={RGB_GAIN} 白平衡=锁定')

    # 红外激光功率
    depth_sensor.set_option(rs.option.laser_power, LASER_POWER)
    print(f'激光功率: {LASER_POWER}')

    return rgb_sensor


def main():
    import pyrealsense2 as rs

    if len(sys.argv) > 1:
        try:
            step = int(sys.argv[1])
            global STEP_ANGLE
            STEP_ANGLE = step
        except ValueError:
            pass

    n_frames = 360 // STEP_ANGLE
    print('=' * 55)
    print(f'  密集采集: {STEP_ANGLE}° 步长 × {n_frames} 帧')
    print(f'  手动曝光: {RGB_EXPOSURE}  增益: {RGB_GAIN}')
    print('=' * 55)

    # ── 转台 ──
    try:
        from turntable import TurntableController
        tt = TurntableController(port='/dev/ttyUSB0')
        tt.open()
        tt.move_absolute(0, speed=10000)
        tt.wait_stop(timeout=30)
        print('转台: 已归零')
    except Exception as e:
        print(f'转台连接失败: {e}')
        print('请确认: 1) USB已插  2) /dev/ttyUSB0 存在  3) 有权限')
        return

    # ── 相机 ──
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, DEPTH_FPS)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, DEPTH_FPS)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # 等待自动曝光稳定 (1秒后锁定)
    print('等待曝光稳定 (1s)...')
    for _ in range(30):
        pipe.wait_for_frames()
    rgb_sensor = setup_camera(pipe, cfg, profile)

    # 创建目录
    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)

    # ── 保存相机内参 ──
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsic_data = {
        'width': intr.width,
        'height': intr.height,
        'intrinsic_matrix': [intr.fx, 0, 0, 0, intr.fy, 0, intr.ppx, intr.ppy, 1],
        'model': str(intr.model),
        'step_angle': STEP_ANGLE,
        'n_frames': n_frames,
    }
    with open(OUTPUT_DIR / 'camera_intrinsic.json', 'w') as f:
        json.dump(intrinsic_data, f, indent=2)
    print(f'内参已保存: fx={intr.fx:.1f} fy={intr.fy:.1f}')

    # ── 逐帧采集 ──
    angles = list(range(0, 360, STEP_ANGLE))
    print(f'\n开始采集 {n_frames} 帧...')
    t0 = time.time()

    for i, target_deg in enumerate(angles):
        tt.move_absolute(target_deg, speed=10000)
        tt.wait_stop(timeout=30)
        time.sleep(0.3)

        # 采集 (取一帧即可, 固定曝光下RGB不跳)
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())

        cv2.imwrite(str(COLOR_DIR / f'{i:06d}.jpg'), color_img)
        cv2.imwrite(str(DEPTH_DIR / f'{i:06d}.png'), depth_img)

        if (i + 1) % 12 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_frames - i - 1)
            print(f'  [{i+1}/{n_frames}] {target_deg:3d}°  '
                  f'已用时{elapsed:.0f}s  预计剩余{eta:.0f}s')

    pipe.stop()
    elapsed = time.time() - t0
    print(f'\n采集完成! {n_frames} 帧 / {elapsed:.0f}s')
    print(f'RGB: {COLOR_DIR}')
    print(f'深度: {DEPTH_DIR}')
    print(f'内参: {OUTPUT_DIR / "camera_intrinsic.json"}')


if __name__ == '__main__':
    main()
