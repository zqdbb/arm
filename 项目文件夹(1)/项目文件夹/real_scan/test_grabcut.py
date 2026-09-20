#!/usr/bin/env python3
"""GrabCut 抠椅子轮廓 — 用世界坐标定位椅子, 精准 ROI."""
import numpy as np, cv2, json
from pathlib import Path

DATA = Path(__file__).parent / 'output/dense_scan'
CALIB = Path(__file__).parent / 'output/calibrate.json'
INTR = DATA / 'camera_intrinsic.json'
OUT = Path(__file__).parent / 'output/visual_hull/grabcut_test'
OUT.mkdir(parents=True, exist_ok=True)

calib = json.load(open(CALIB))
R_calib = np.array(calib['R']); t_calib = np.array(calib['t'])
intr = json.load(open(INTR))
M = intr['intrinsic_matrix']
fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]
w, h = intr['width'], intr['height']; step = 5

color_files = sorted((DATA / 'color').glob('*.jpg'))
depth_files = sorted((DATA / 'depth').glob('*.png'))

u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32),
                              np.arange(h, dtype=np.float32))

indices = [0, 6, 12, 18, 24, 36, 48, 60]

for idx in indices:
    img = cv2.imread(str(color_files[idx]))
    depth = cv2.imread(str(depth_files[idx]), -1).astype(float) / 1000

    # ── 世界坐标过滤: 找到椅子像素位置 ──
    valid = depth > 0.01
    z_c = depth[valid]
    x_c = (u_grid[valid] - ppx) / fx * z_c
    y_c = (v_grid[valid] - ppy) / fy * z_c
    pts_cam = np.column_stack([x_c, y_c, z_c])
    pts_world = (R_calib @ pts_cam.T).T + t_calib

    # undo rotation → canonical object frame
    theta = np.radians(idx * step)
    c, s = np.cos(theta), np.sin(theta)
    R_z_inv = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
    pts_obj = (R_z_inv @ pts_world.T).T

    # 椅子: 转台面以上 + XY 半径 < 5cm
    above = pts_obj[:, 2] < -0.005
    in_r = np.hypot(pts_obj[:, 0], pts_obj[:, 1]) < 0.05
    chair_mask = above & in_r

    if chair_mask.sum() < 50:
        print(f'帧{idx:2d}: 椅子点太少, 跳过')
        continue

    # ROI: 椅子在图像中的包围盒 + margin
    chair_px_v = v_grid[valid][chair_mask].astype(int)
    chair_px_u = u_grid[valid][chair_mask].astype(int)
    margin = 15
    x0 = max(0, chair_px_u.min() - margin)
    y0 = max(0, chair_px_v.min() - margin)
    x1 = min(w, chair_px_u.max() + margin)
    y1 = min(h, chair_px_v.max() + margin)

    rect = (x0, y0, x1 - x0, y1 - y0)
    print(f'帧{idx:2d} ({idx*step:3d}°): ROI=({x0},{y0},{x1},{y1}) size={x1-x0}×{y1-y0}')

    # ── GrabCut ──
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5,
                cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 只保留 ROI 内最大的连通域
    roi = fg_mask[y0:y1, x0:x1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    if num_labels > 1:
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        roi = (labels == largest).astype(np.uint8) * 255
    fg_mask[y0:y1, x0:x1] = roi

    fg_pct = (fg_mask > 0).sum() / fg_mask.size * 100
    print(f'  前景占比: {fg_pct:.1f}%')

    # 可视化
    overlay = img.copy()
    overlay[fg_mask == 0] = overlay[fg_mask == 0] // 2

    mask_rgb = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    extracted = cv2.bitwise_and(img, img, mask=fg_mask)
    row1 = np.hstack([img, mask_rgb])
    row2 = np.hstack([extracted, overlay])
    result = np.vstack([row1, row2])
    cv2.imwrite(str(OUT / f'grabcut_{idx:03d}.jpg'), result)
    cv2.imwrite(str(OUT / f'mask_{idx:03d}.png'), fg_mask)

    # ROI debug
    debug = img.copy()
    cv2.rectangle(debug, (rect[0], rect[1]),
                  (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    cv2.imwrite(str(OUT / f'roi_{idx:03d}.jpg'), debug)

print(f'\n输出: {OUT}/')

