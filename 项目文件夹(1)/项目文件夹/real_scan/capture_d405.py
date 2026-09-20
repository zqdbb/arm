#!/usr/bin/env python3
"""
D405 密集采集: 5° × 72 帧. D405 专为近距离优化 (7-50cm).
与 D435i 不同: 单 Stereo Module, 无激光, 基线 18mm.
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
RGB_EXPOSURE = 30000      # D405 曝光 (1-165000)
RGB_GAIN = 64              # D405 增益 (16-248)
DEPTH_PRESET = 2          # 2=Hand (近距离优化, 7-50cm)
W, H = 1280, 720          # D405 原生分辨率
DEPTH_FPS = 15            # 15fps 足够

OUTPUT_DIR = BASE_DIR / 'output' / 'dense_scan'
COLOR_DIR = OUTPUT_DIR / 'color'
DEPTH_DIR = OUTPUT_DIR / 'depth'
DEPTH_COLOR_DIR = OUTPUT_DIR / 'depth_color'


def setup_d405(profile):
    """D405 配置: 对齐4.py的设置."""
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()

    # 视觉预设: Close Range (与4.py一致)
    depth_sensor.set_option(rs.option.visual_preset, DEPTH_PRESET)
    preset_names = {0:"Custom", 1:"Default", 2:"Hand", 3:"High Accuracy",
                    4:"High Density", 5:"Medium Density"}
    print(f'深度预设: {DEPTH_PRESET} ({preset_names.get(DEPTH_PRESET, "?")})')

    # 自动曝光 (与4.py一致, 让相机自己调)
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    print(f'曝光: 自动')

    return depth_sensor


def main():
    if len(sys.argv) > 1:
        try:
            global STEP_ANGLE
            STEP_ANGLE = int(sys.argv[1])
        except ValueError:
            pass

    n_frames = 360 // STEP_ANGLE
    print('=' * 55)
    print(f'  D405 密集采集: {STEP_ANGLE}° × {n_frames} 帧')
    print(f'  {W}×{H} @ {DEPTH_FPS}fps  曝光={RGB_EXPOSURE}  增益={RGB_GAIN}')
    print('=' * 55)

    # 转台
    try:
        from turntable import TurntableController
        tt = TurntableController(port='/dev/ttyUSB0')
        tt.open()
        tt.move_absolute(0, speed=10000)
        tt.wait_stop(timeout=30)
        print('转台: 已归零')
    except Exception as e:
        print(f'转台连接失败: {e}')
        return

    # 相机
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, DEPTH_FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, DEPTH_FPS)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    # 等待稳定
    print('等待曝光稳定 (2s)...')
    for _ in range(DEPTH_FPS * 2):
        pipe.wait_for_frames()

    depth_sensor = setup_d405(profile)
    depth_scale = depth_sensor.get_depth_scale()  # D405: ~0.0001 (0.1mm/unit)

    # 创建目录
    COLOR_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    DEPTH_COLOR_DIR.mkdir(parents=True, exist_ok=True)

    # 保存内参
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_profile.get_intrinsics()

    intrinsic_data = {
        'width': intr.width,
        'height': intr.height,
        'intrinsic_matrix': [intr.fx, 0, 0, 0, intr.fy, 0, intr.ppx, intr.ppy, 1],
        'depth_intrinsic_matrix': [depth_intr.fx, 0, 0, 0, depth_intr.fy, 0,
                                    depth_intr.ppx, depth_intr.ppy, 1],
        'model': str(intr.model),
        'step_angle': STEP_ANGLE,
        'n_frames': n_frames,
        'camera': 'D405',
        'depth_scale': float(depth_scale),
    }
    with open(OUTPUT_DIR / 'camera_intrinsic.json', 'w') as f:
        json.dump(intrinsic_data, f, indent=2)
    print(f'内参: fx={intr.fx:.1f} fy={intr.fy:.1f}  '
          f'depth: fx={depth_intr.fx:.1f} fy={depth_intr.fy:.1f}')

    # 逐帧采集
    angles = list(range(0, 360, STEP_ANGLE))
    print(f'\n开始采集 {n_frames} 帧...')
    t0 = time.time()

    for i, target_deg in enumerate(angles):
        tt.move_absolute(target_deg, speed=10000)
        tt.wait_stop(timeout=30)
        time.sleep(0.3)

        # 取多帧平均降噪 (D405 帧率高, 可以这样用)
        color_imgs = []
        depth_imgs = []
        for _ in range(3):  # 3 帧平均
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            color_imgs.append(np.asanyarray(color_frame.get_data()).astype(np.float32))
            depth_imgs.append(np.asanyarray(depth_frame.get_data()).astype(np.float32))

        color_img = np.mean(color_imgs, axis=0).astype(np.uint8)
        # 用实际 depth_scale 转换为毫米再保存
        depth_mm = np.mean(depth_imgs, axis=0) * depth_scale * 1000.0
        depth_img = depth_mm.astype(np.uint16)

        cv2.imwrite(str(COLOR_DIR / f'{i:06d}.jpg'), color_img)
        cv2.imwrite(str(DEPTH_DIR / f'{i:06d}.png'), depth_img)

        # 伪彩色深度预览 (JET, 60-500mm → 椅子可见)
        depth_viz = depth_mm.copy()
        depth_viz_clip = np.clip(depth_viz, 60, 500)
        depth_viz_norm = ((depth_viz_clip - 60) / (500 - 60) * 255).astype(np.uint8)
        depth_viz_color = cv2.applyColorMap(depth_viz_norm, cv2.COLORMAP_JET)
        depth_viz_color[depth_img == 0] = 0
        cv2.imwrite(str(DEPTH_COLOR_DIR / f'{i:06d}.jpg'), depth_viz_color)

        if (i + 1) % 12 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_frames - i - 1)
            print(f'  [{i+1}/{n_frames}] {target_deg:3d}°  '
                  f'已用{elapsed:.0f}s  剩余~{eta:.0f}s')

    pipe.stop()
    elapsed = time.time() - t0
    print(f'\n采集完成! {n_frames} 帧 / {elapsed:.0f}s')
    print(f'RGB: {COLOR_DIR}/')
    print(f'深度: {DEPTH_DIR}/')
    print(f'内参: {OUTPUT_DIR / "camera_intrinsic.json"}')
    print(f'\n下一步: calibrate_furniture.py 用 {OUTPUT_DIR.name} 标定')


if __name__ == '__main__':
    main()
