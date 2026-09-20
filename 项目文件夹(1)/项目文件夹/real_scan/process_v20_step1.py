#!/usr/bin/env python3
"""V20 Step 1: SGBM 处理所有帧, 深度图存盘."""
import cv2, numpy as np, json, time, sys, gc
from pathlib import Path

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/v20_capture'
DEPTH_DIR = BASE / 'output/v20_depth_sgbm'
DEPTH_DIR.mkdir(parents=True, exist_ok=True)

with open(CAPTURE_DIR / 'calib.json') as f:
    calib = json.load(f)
ir = calib['ir_intrinsics']
baseline = calib['baseline_m']

SCALE = 0.5
W_s = int(ir['width'] * SCALE)
H_s = int(ir['height'] * SCALE)
fx_s = ir['fx'] * SCALE
fy_s = ir['fy'] * SCALE
n_frames = calib['n_frames']

print(f'SGBM: {W_s}x{H_s}  {n_frames} frames')

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=96, blockSize=5,
    P1=8 * 3 * 5**2, P2=32 * 3 * 5**2,
    disp12MaxDiff=3, uniquenessRatio=5,
    speckleWindowSize=100, speckleRange=2,
    preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

t0 = time.time()
covs = []

for i in range(n_frames):
    ir_l = cv2.imread(str(CAPTURE_DIR / f'ir_left/{i:03d}.png'), cv2.IMREAD_GRAYSCALE)
    ir_r = cv2.imread(str(CAPTURE_DIR / f'ir_right/{i:03d}.png'), cv2.IMREAD_GRAYSCALE)

    ir_l_s = cv2.resize(ir_l, (W_s, H_s))
    ir_r_s = cv2.resize(ir_r, (W_s, H_s))
    del ir_l, ir_r

    disp = sgbm.compute(ir_l_s, ir_r_s).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > 0.5
    depth[valid] = (fx_s * baseline) / disp[valid]
    depth[(depth < 0.2) | (depth > 0.7)] = 0

    # 存半精度 (float16) 省内存
    np.save(str(DEPTH_DIR / f'{i:03d}.npy'), depth.astype(np.float16))

    cov = (depth > 0).sum() / depth.size * 100
    covs.append(cov)

    del ir_l_s, ir_r_s, disp, depth, valid
    gc.collect()

    if (i + 1) % 18 == 0:
        elapsed = time.time() - t0
        print(f'  [{i+1}/{n_frames}] cov={np.mean(covs[-18:]):.1f}%  '
              f'{elapsed:.0f}s  ~{elapsed/(i+1)*(n_frames-i-1):.0f}s left')

elapsed = time.time() - t0
print(f'\nSGBM 完成: {elapsed:.0f}s ({elapsed/n_frames:.1f}s/frame)')
print(f'覆盖率: mean={np.mean(covs):.1f}% min={np.min(covs):.1f}% max={np.max(covs):.1f}%')

# 保存元数据
meta = {
    'width': W_s, 'height': H_s, 'fx': fx_s, 'fy': fy_s,
    'cx': ir['ppx'] * SCALE, 'cy': ir['ppy'] * SCALE,
    'baseline': baseline, 'n_frames': n_frames,
    'step_deg': calib['step_deg'],
    'coverage_mean': float(np.mean(covs)),
    'coverage_min': float(np.min(covs)),
    'coverage_max': float(np.max(covs)),
    'coverage_per_frame': [float(c) for c in covs],
}
with open(DEPTH_DIR / 'meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(f'输出: {DEPTH_DIR}/ ({n_frames} .npy files)')
