#!/usr/bin/env python3
"""SGBM 立体匹配测试: 用一对相邻帧的 RGB 算出深度图, 替代 D435i 深度."""
import numpy as np
import cv2, json, os
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCAN_DIR = BASE_DIR / 'output/open3d_scan'
CALIB_PATH = BASE_DIR / 'output/calibrate.json'
INTRINSIC_PATH = SCAN_DIR / 'camera_intrinsic.json'

R_calib = np.array(json.load(open(CALIB_PATH))['R'])
t_calib = np.array(json.load(open(CALIB_PATH))['t'])
intr = json.load(open(INTRINSIC_PATH))
M = intr['intrinsic_matrix']
fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]
w, h = intr['width'], intr['height']

# 读取帧0和帧1 (0° 和 10°)
img0 = cv2.imread(str(SCAN_DIR / 'color' / '000000.jpg'))
img1 = cv2.imread(str(SCAN_DIR / 'color' / '000001.jpg'))
depth_gt = cv2.imread(str(SCAN_DIR / 'depth' / '000000.png'), -1).astype(np.float32) / 1000.0

# 帧 i 的相机外参 (aligned frame → camera):
#   R_w2c_i = R_calib^T @ R_z(-θ_i)
#   t_w2c_i = -R_calib^T @ t_calib (常数!)
#   cam_i   = R_z(+θ_i) @ t_calib

def get_extrinsics(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    R_z_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    R_w2c = R_calib.T @ R_z_inv
    t_w2c = -R_calib.T @ t_calib
    cam = R_z @ t_calib
    return R_w2c, t_w2c, cam

R0, t0, cam0 = get_extrinsics(0)
R1, t1, cam1 = get_extrinsics(10)

# 相对位姿: frame0 → frame1
R_rel = R1 @ R0.T
t_rel = R1 @ (cam0 - cam1)

print(f'帧0 相机位置: {cam0.round(3)}')
print(f'帧1 相机位置: {cam1.round(3)}')
print(f'基线长度: {np.linalg.norm(cam1 - cam0):.4f}m = {np.linalg.norm(cam1-cam0)*100:.1f}cm')
print(f'R_rel (应接近 I):\n{np.round(R_rel, 4)}')
print(f't_rel (应主要是 X 分量): {np.round(t_rel, 4)}')

# 极线校正
K = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]], dtype=np.float64)
R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
    K, np.zeros(5), K, np.zeros(5),
    (w, h), R_rel, t_rel,
    alpha=0)

# 校正映射
map1x, map1y = cv2.initUndistortRectifyMap(K, np.zeros(5), R1_rect, P1, (w, h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K, np.zeros(5), R2_rect, P2, (w, h), cv2.CV_32FC1)
rect0 = cv2.remap(img0, map1x, map1y, cv2.INTER_LINEAR)
rect1 = cv2.remap(img1, map2x, map2y, cv2.INTER_LINEAR)

# 保存对照图: 原始 vs 校正后 + 水平线
for row in range(50, h, 60):
    cv2.line(rect0, (0, row), (w, row), (0, 255, 0), 1)
    cv2.line(rect1, (0, row), (w, row), (0, 255, 0), 1)
side = np.hstack([rect0, rect1])
cv2.imwrite(str(BASE_DIR / 'output' / 'rectified_pair.png'), side)

# SGBM 立体匹配
gray0 = cv2.cvtColor(rect0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,    # 搜索范围 128 像素
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
disparity = stereo.compute(gray0, gray1).astype(np.float32) / 16.0

# 视差 → 深度
disparity[disparity < 1] = np.nan
depth_sgbm = np.full_like(disparity, np.nan)
fx_rect = P1[0, 0]  # 校正后的焦距
baseline = np.linalg.norm(t_rel)
valid = disparity > 0
depth_sgbm[valid] = fx_rect * baseline / disparity[valid]

# 可视化
disparity_vis = np.nan_to_num(disparity, 0)
disparity_norm = cv2.normalize(disparity_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)

depth_vis = np.nan_to_num(depth_sgbm, 0)
depth_norm = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

# D435i 深度作为参考
depth_gt_norm = cv2.normalize(np.nan_to_num(depth_gt, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
depth_gt_color = cv2.applyColorMap(depth_gt_norm, cv2.COLORMAP_JET)

# 拼图保存
row1 = np.hstack([img0, depth_gt_color])
row2 = np.hstack([disparity_color, depth_color])
result = np.vstack([row1, row2])
cv2.imwrite(str(BASE_DIR / 'output' / 'sgbm_result.png'), result)
cv2.imwrite(str(BASE_DIR / 'output' / 'rectified_pair.png'), side)

# 统计
sgbm_valid = (~np.isnan(depth_sgbm)).sum()
sgbm_median = np.nanmedian(depth_sgbm)
d435i_valid = ((depth_gt > 0.01) & (depth_gt < 2.0)).sum()
print(f'\nSGBM 有效深度: {sgbm_valid:,} / {w*h:,} ({sgbm_valid/(w*h)*100:.1f}%)')
print(f'  中位数: {sgbm_median:.3f}m')
print(f'D435i 有效深度: {d435i_valid:,} ({d435i_valid/(w*h)*100:.1f}%)')
print(f'\n输出: output/sgbm_result.png')
print(f'输出: output/rectified_pair.png')
