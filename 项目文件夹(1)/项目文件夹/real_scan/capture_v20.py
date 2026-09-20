#!/usr/bin/env python3
"""
V20: IR 立体采集 — 转台多角度 IR 左右图采集
用法: python3 capture_v20.py [步进度数, 默认5]
输出: output/v20_capture/
  ir_left/    — IR 左图 (infra1)
  ir_right/   — IR 右图 (infra2)
  color/      — RGB 参考图
  depth/      — 内置深度 (对比用)
  calib.json  — 相机内参
"""

import numpy as np, cv2, json, sys, time
from pathlib import Path
import pyrealsense2 as rs

BASE = Path(__file__).parent
STEP_DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 5
W, H = 1280, 720
LASER_POWER = 360

OUT = BASE / 'output/v20_capture'
for d in [OUT / 'ir_left', OUT / 'ir_right', OUT / 'color', OUT / 'depth']:
    d.mkdir(parents=True, exist_ok=True)


def main():
    n_frames = 360 // STEP_DEG
    print(f'V20 IR 立体采集: {STEP_DEG}° × {n_frames} 帧  {W}×{H}')

    # ── 转台 ──
    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台: 已连接')

    # ── 相机 ──
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    pipe = rs.pipeline()
    profile = pipe.start(cfg)

    # IR 投影仪全开
    dev = profile.get_device()
    depth_sensor = dev.first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1)
    depth_sensor.set_option(rs.option.laser_power, LASER_POWER)
    print(f'激光功率: {LASER_POWER}')

    # 获取 IR 内参
    ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir_intr = ir_profile.get_intrinsics()
    calib = {
        'ir_intrinsics': {
            'width': ir_intr.width, 'height': ir_intr.height,
            'fx': ir_intr.fx, 'fy': ir_intr.fy,
            'ppx': ir_intr.ppx, 'ppy': ir_intr.ppy,
            'model': 'Brown Conrady',
            'coeffs': list(ir_intr.coeffs),
        },
        'baseline_m': 0.05,
        'depth_scale': depth_sensor.get_depth_scale(),
        'laser_power': LASER_POWER,
        'step_deg': STEP_DEG,
        'n_frames': n_frames,
    }
    with open(OUT / 'calib.json', 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'IR 内参: fx={ir_intr.fx:.1f} fy={ir_intr.fy:.1f} '
          f'cx={ir_intr.ppx:.1f} cy={ir_intr.ppy:.1f}')

    # 稳定
    print('等待稳定 (2s)...')
    for _ in range(60):
        pipe.wait_for_frames()

    # ── 采集 ──
    angles = list(range(0, 360, STEP_DEG))
    print(f'\n开始采集 {n_frames} 帧 ...')
    t0 = time.time()

    for i, target_deg in enumerate(angles):
        tt.move_absolute(target_deg, speed=10000)
        tt.wait_stop(timeout=30)
        time.sleep(0.3)

        frames = pipe.wait_for_frames()

        ir_l = np.asanyarray(frames.get_infrared_frame(1).get_data())
        ir_r = np.asanyarray(frames.get_infrared_frame(2).get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        color = np.asanyarray(frames.get_color_frame().get_data())

        cv2.imwrite(str(OUT / f'ir_left/{i:03d}.png'), ir_l)
        cv2.imwrite(str(OUT / f'ir_right/{i:03d}.png'), ir_r)
        cv2.imwrite(str(OUT / f'depth/{i:03d}.png'), depth)
        cv2.imwrite(str(OUT / f'color/{i:03d}.jpg'), color)

        if (i + 1) % 12 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_frames - i - 1)
            print(f'  [{i+1}/{n_frames}] {target_deg:3d}°  已用{elapsed:.0f}s  剩余~{eta:.0f}s')

    pipe.stop()
    tt.close()
    elapsed = time.time() - t0
    print(f'\n采集完成! {n_frames} 帧 / {elapsed:.0f}s ({elapsed/n_frames:.1f}s/帧)')
    print(f'输出: {OUT}/')


if __name__ == '__main__':
    main()
