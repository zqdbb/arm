#!/usr/bin/env python3
"""rembg 抠图测试 — 和 GrabCut 对比."""
import numpy as np, cv2, json, time
from pathlib import Path
from rembg import remove, new_session

DATA = Path(__file__).parent / 'output/dense_scan'
CALIB = Path(__file__).parent / 'output/calibrate.json'
INTR = DATA / 'camera_intrinsic.json'
OUT = Path(__file__).parent / 'output/visual_hull/rembg_test'
OUT.mkdir(parents=True, exist_ok=True)

calib = json.load(open(CALIB))
R_calib = np.array(calib['R']); t_calib = np.array(calib['t'])
intr = json.load(open(INTR))
M = intr['intrinsic_matrix']
fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]; w, h = intr['width'], intr['height']

color_files = sorted((DATA / 'color').glob('*.jpg'))
depth_files = sorted((DATA / 'depth').glob('*.png'))
u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

# 创建 session (isnet-general-use 精度最好)
print('加载 rembg 模型 (isnet-general-use)...')
session = new_session('isnet-general-use')

indices = [0, 12, 24, 36, 48, 60]

for idx in indices:
    img = cv2.imread(str(color_files[idx]))
    t0 = time.time()

    # rembg 抠图
    result = remove(img, session=session, only_mask=False)
    # result 是 RGBA, alpha 通道就是 mask
    alpha = result[:, :, 3]
    rembg_mask = (alpha > 128).astype(np.uint8) * 255

    dt = time.time() - t0

    # 对比深度 chair 点
    depth = cv2.imread(str(depth_files[idx]), -1).astype(float)/1000
    valid = depth > 0.01
    z_c = depth[valid]; x_c = (u_grid[valid]-ppx)/fx*z_c; y_c = (v_grid[valid]-ppy)/fy*z_c
    pts_cam = np.column_stack([x_c, y_c, z_c])
    pts_world = (R_calib @ pts_cam.T).T + t_calib
    theta = np.radians(idx*5); c,s = np.cos(theta), np.sin(theta)
    R_z_inv = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    pts_obj = (R_z_inv @ pts_world.T).T
    above = pts_obj[:,2] < -0.003
    in_r = np.hypot(pts_obj[:,0], pts_obj[:,1]) < 0.05
    chair_idx = np.where(above & in_r)[0]

    depth_mask = np.zeros((h,w), dtype=np.uint8)
    if len(chair_idx) > 100:
        pts_obj_chair = pts_obj[chair_idx]
        R_z_fwd = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        pts_w = (R_z_fwd @ pts_obj_chair.T).T
        pts_c = (R_calib.T @ (pts_w - t_calib).T).T
        u_px = (fx*pts_c[:,0]/pts_c[:,2]+ppx).astype(int)
        v_px = (fy*pts_c[:,1]/pts_c[:,2]+ppy).astype(int)
        in_img = (u_px>=0)&(u_px<w)&(v_px>=0)&(v_px<h)&(pts_c[:,2]>0.01)
        depth_mask[v_px[in_img], u_px[in_img]] = 255

    depth_px = (depth_mask > 0).sum()
    overlap = (rembg_mask > 0) & (depth_mask > 0)
    union = (rembg_mask > 0) | (depth_mask > 0)
    iou = overlap.sum() / union.sum() * 100 if union.sum() > 0 else 0
    recall = overlap.sum() / depth_px * 100 if depth_px > 0 else 0
    precision = overlap.sum() / (rembg_mask > 0).sum() * 100 if (rembg_mask > 0).sum() > 0 else 0

    fg_pct = (rembg_mask > 0).sum() / rembg_mask.size * 100

    print(f'帧{idx:2d} ({idx*5:3d}°): 前景={fg_pct:.1f}%  IoU={iou:.0f}%  '
          f'recall={recall:.0f}%  precision={precision:.0f}%  {dt:.1f}s')

    # 保存可视化
    overlay = img.copy()
    overlay[rembg_mask == 0] = overlay[rembg_mask == 0] // 2
    mask_rgb = cv2.cvtColor(rembg_mask, cv2.COLOR_GRAY2BGR)
    extracted = cv2.bitwise_and(img, img, mask=rembg_mask)
    row1 = np.hstack([img, mask_rgb])
    row2 = np.hstack([extracted, overlay])
    result_img = np.vstack([row1, row2])
    cv2.imwrite(str(OUT / f'rembg_{idx:03d}.jpg'), result_img)
    cv2.imwrite(str(OUT / f'mask_{idx:03d}.png'), rembg_mask)

print(f'\n输出: {OUT}/')
