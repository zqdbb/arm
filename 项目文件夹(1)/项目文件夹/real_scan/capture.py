#!/usr/bin/env python3
"""Camera-adaptive 采集脚本 — 支持 D405 / D435i + 转台 72 帧."""
import cv2, numpy as np, pyrealsense2 as rs, time, json
from pathlib import Path

BASE = Path(__file__).parent

# ── 可调参数 ──
W_FULL, H_FULL = 1280, 720
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG  # 72
N_AVG = 3                     # 每帧平均次数
ZOOM_DEFAULT = 1              # 默认无变焦，全分辨率
TURNTABLE_PORT = '/dev/ttyUSB0'

OUT_DIR = BASE / 'output/capture'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 读取标定（先读 zoom，再算裁剪参数）──
calib_path = BASE / 'output/calibrate.json'
if calib_path.exists():
    with open(calib_path) as f:
        calib = json.load(f)
    ZOOM = calib.get('digital_zoom', ZOOM_DEFAULT)
    print(f'已加载标定: zoom={ZOOM}x  radius={calib["radius_m"]*100:.1f}cm')
else:
    print('警告: 未找到标定文件，请先运行 turntable_set_v2.py')
    calib = {}
    ZOOM = ZOOM_DEFAULT

# ── 连接转台 ──
from turntable import TurntableController
tt = TurntableController(port=TURNTABLE_PORT)
tt.open()
tt.move_absolute(0, speed=10000)
tt.wait_stop(timeout=30)
print('转台已归零')

# ── 启动相机 ──
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W_FULL, H_FULL, rs.format.z16, FPS)
cfg.enable_stream(rs.stream.color, W_FULL, H_FULL, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 相机型号检测
camera_name = device.get_info(rs.camera_info.name)
print(f'相机: {camera_name}')

# 读取内参
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx_full = intr.fx
fy_full = intr.fy
ppx_full = intr.ppx
ppy_full = intr.ppy
print(f'全分辨率内参: fx={fx_full:.1f} fy={fy_full:.1f} pp=({ppx_full:.1f},{ppy_full:.1f})')

# 变焦后内参
x0_ = (W_FULL - int(W_FULL / ZOOM)) // 2
y0_ = (H_FULL - int(H_FULL / ZOOM)) // 2
W, H = int(W_FULL / ZOOM), int(H_FULL / ZOOM)
fx, fy = fx_full, fy_full
ppx, ppy = ppx_full - x0_, ppy_full - y0_
print(f'变焦后: {W}x{H} pp=({ppx:.1f},{ppy:.1f})')

# ── 相机参数配置 ──
if 'D405' in camera_name:
    visual_preset = 2     # Hand — 近距离优化
    laser_power = None
    preset_name = 'Hand'
    print('D405 模式: VisualPreset=Hand, 无激光')
else:
    visual_preset = 3     # High Accuracy
    laser_power = 360
    preset_name = 'HighAccuracy'
    print(f'D435i 模式: VisualPreset=HighAccuracy, Laser=360')

# 设 visual preset（需停流）
pipe.stop()
try:
    depth_sensor.set_option(rs.option.visual_preset, visual_preset)
    print(f'  visual_preset → {preset_name}')
except Exception as e:
    print(f'  visual_preset 失败: {e}')

if laser_power is not None:
    try:
        depth_sensor.set_option(rs.option.laser_power, float(laser_power))
        print(f'  laser_power → {laser_power}')
    except Exception as e:
        print(f'  laser_power 失败: {e}')

# 重启（新建 config，旧的已被消费）
cfg2 = rs.config()
cfg2.enable_stream(rs.stream.depth, W_FULL, H_FULL, rs.format.z16, FPS)
cfg2.enable_stream(rs.stream.color, W_FULL, H_FULL, rs.format.bgr8, FPS)
profile = pipe.start(cfg2)
align = rs.align(rs.stream.color)
print(f'深度 scale: {depth_scale}')

# ── 创建输出目录 ──
(OUT_DIR / 'color').mkdir(exist_ok=True)
(OUT_DIR / 'depth').mkdir(exist_ok=True)

# ── 保存元数据 ──
meta = {
    'camera': camera_name,
    'fx_full': fx_full, 'fy_full': fy_full,
    'ppx_full': ppx_full, 'ppy_full': ppy_full,
    'width_full': W_FULL, 'height_full': H_FULL,
    'zoom': ZOOM, 'wxH_zoomed': [W, H],
    'fx': fx, 'fy': fy, 'ppx': ppx, 'ppy': ppy,
    'depth_scale': depth_scale,
    'visual_preset': preset_name,
    'laser_power': laser_power,
    'step_deg': STEP_DEG, 'n_frames': N_FRAMES,
    'n_avg': N_AVG, 'fps': FPS,
    'calib': calib,
}
with open(OUT_DIR / 'meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

# ── 采集循环 ──
print(f'\n{"=" * 55}')
print(f'  采集: {STEP_DEG}° × {N_FRAMES} 帧  ({W}x{H} @ {ZOOM}x zoom)')
print(f'{"=" * 55}')

# 让相机稳定几帧
for _ in range(10):
    pipe.wait_for_frames()

t0 = time.time()
for i in range(N_FRAMES):
    deg = i * STEP_DEG
    tt.move_absolute(deg, speed=5000)
    tt.wait_stop(timeout=30)

    d_sum = np.zeros((H_FULL, W_FULL), np.float64)
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
        print(f'  [{i+1:3d}/{N_FRAMES}] {deg:3d}°  无数据')
        continue

    # 平均深度
    depth_mm_full = (d_sum / n_good) * depth_scale * 1000.0

    # 变焦裁剪 → 保存
    color_save = color_full[y0_:y0_+H, x0_:x0_+W]
    depth_mm = depth_mm_full[y0_:y0_+H, x0_:x0_+W]
    depth_u16 = depth_mm.astype(np.uint16)

    cv2.imwrite(str(OUT_DIR / 'color' / f'{i:03d}.jpg'), color_save,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(OUT_DIR / 'depth' / f'{i:03d}.png'), depth_u16)

    # 统计（裁剪后有效区域内）
    dvals = depth_mm[depth_mm > 0]
    elapsed = time.time() - t0
    print(f'  [{i+1:3d}/{N_FRAMES}] {deg:3d}°  '
          f'valid={len(dvals)//1000}k  '
          f'med={np.median(dvals):.0f}mm  '
          f'{elapsed:.0f}s')

pipe.stop()
tt.close()

elapsed = time.time() - t0
print(f'\n采集完成: {N_FRAMES} 帧 / {elapsed:.0f}s ({elapsed/N_FRAMES:.1f}s/帧)')
print(f'输出: {OUT_DIR}/')
