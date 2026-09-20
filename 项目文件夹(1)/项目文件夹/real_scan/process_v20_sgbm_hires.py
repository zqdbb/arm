#!/usr/bin/env python3
"""V20 高分辨率 SGBM: 1280x720 IR 中心裁剪 → SGBM (不降采样)."""
import cv2, numpy as np, json, time, gc
from pathlib import Path

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/v20_capture'
OUT_DIR = BASE / 'output/v20_depth_sgbm_hires'
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CAPTURE_DIR / 'calib.json') as f:
    cal = json.load(f)

# 全分辨率 IR 内参
fx = cal['ir_intrinsics']['fx']
fy = cal['ir_intrinsics']['fy']
ppx = cal['ir_intrinsics']['ppx']
ppy = cal['ir_intrinsics']['ppy']
W_full, H_full = cal['ir_intrinsics']['width'], cal['ir_intrinsics']['height']
baseline = cal['baseline_m']
step_deg = cal['step_deg']
n_frames = cal['n_frames']

print(f'IR full res: {W_full}x{H_full} fx={fx:.2f} baseline={baseline}m')

# 裁剪中心区域: 椅子 ~67x58px, 裁剪 400x300 留余量
CROP_W, CROP_H = 400, 300
x1 = W_full // 2 - CROP_W // 2
x2 = x1 + CROP_W
y1 = H_full // 2 - CROP_H // 2
y2 = y1 + CROP_H
print(f'Crop: [{x1}:{x2}, {y1}:{y2}] = {CROP_W}x{CROP_H}')

# 裁剪后内参
cx_c = CROP_W / 2.0
cy_c = CROP_H / 2.0
# fx, fy 不变 (裁剪不改变焦距)
print(f'Cropped intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx_c:.2f} cy={cy_c:.2f}')

# StereoSGBM — 全分辨率参数
# depth = fx * baseline / disparity
# 椅子 350-480mm → disparity = 636.75*0.05/depth = 31.84/depth
#   350mm → 91px, 480mm → 66px
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,
    disp12MaxDiff=1,
    uniquenessRatio=5,
    speckleWindowSize=100,
    speckleRange=1,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

coverage_list = []
t0 = time.time()

for i in range(n_frames):
    # 读取全分辨率 IR
    imgL = cv2.imread(str(CAPTURE_DIR / f'ir_left/{i:03d}.png'), 0)
    imgR = cv2.imread(str(CAPTURE_DIR / f'ir_right/{i:03d}.png'), 0)

    # 裁剪中心
    cropL = imgL[y1:y2, x1:x2]
    cropR = imgR[y1:y2, x1:x2]

    # SGBM
    disp = stereo.compute(cropL, cropR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    # 深度
    depth = np.full_like(disp, np.nan)
    valid = disp > 0
    depth[valid] = fx * baseline / disp[valid]

    # 只保留椅子距离范围
    depth[(depth < 0.30) | (depth > 0.55)] = np.nan

    # 保存 float16
    depth_f16 = depth.astype(np.float16)
    np.save(str(OUT_DIR / f'{i:03d}.npy'), depth_f16)

    cov = np.sum(valid) / (CROP_W * CROP_H) * 100
    coverage_list.append(cov)

    del imgL, imgR, cropL, cropR, disp, depth, depth_f16
    gc.collect()

    if (i + 1) % 18 == 0:
        elapsed = time.time() - t0
        print(f'  [{i+1}/{n_frames}] cov={np.mean(coverage_list[-18:]):.1f}%  {elapsed:.0f}s')

elapsed = time.time() - t0
print(f'\nDone: {n_frames} frames / {elapsed:.0f}s')
print(f'Coverage: mean={np.mean(coverage_list):.1f}% min={np.min(coverage_list):.1f}% max={np.max(coverage_list):.1f}%')

# 保存 meta
meta = {
    'width': CROP_W, 'height': CROP_H,
    'fx': fx, 'fy': fy, 'cx': cx_c, 'cy': cy_c,
    'baseline': baseline,
    'n_frames': n_frames, 'step_deg': step_deg,
    'coverage_mean': float(np.mean(coverage_list)),
    'coverage_per_frame': [float(c) for c in coverage_list],
}
with open(OUT_DIR / 'meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'Saved to {OUT_DIR}/')
