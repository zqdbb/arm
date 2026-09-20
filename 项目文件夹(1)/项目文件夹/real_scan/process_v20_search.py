#!/usr/bin/env python3
"""V20 转轴参数搜索: 试不同原点+方向, 找最紧致的点云."""
import numpy as np, json, time
from pathlib import Path

BASE = Path(__file__).parent
DEPTH_DIR = BASE / 'output/v20_depth_sgbm'

with open(BASE / 'output/calibrate.json') as f:
    cal = json.load(f)
with open(DEPTH_DIR / 'meta.json') as f:
    meta = json.load(f)

R_cal = np.array(cal['R']); t_cal = np.array(cal['t'])
W_s, H_s = meta['width'], meta['height']
fx, fy = meta['fx'], meta['fy']
cx, cy = meta['cx'], meta['cy']
n_frames = meta['n_frames']; step_deg = meta['step_deg']

# 轴方向: 标定的 + 纯竖直
axis_dirs = {
    'calib': R_cal[:, 2] / np.linalg.norm(R_cal[:, 2]),
    'vertical': np.array([0.0, 1.0, 0.0]),
}
# 确保 Y 分量朝下 (物理朝上)
for k in axis_dirs:
    if axis_dirs[k][1] > 0:
        axis_dirs[k] = -axis_dirs[k]

# 预加载深度数据
print('Loading depth...')
frames = []
for i in range(0, n_frames, 4):  # 每 4 帧 (20度步进)
    d = np.load(str(DEPTH_DIR / f'{i:03d}.npy')).astype(np.float32)
    # 紧裁剪
    x1, x2 = W_s//2-40, W_s//2+40
    y1, y2 = H_s//2-30, H_s//2+30
    dc = d[y1:y2, x1:x2]
    valid = (dc > 0.30) & (dc < 0.55)
    if valid.sum() < 30:
        continue
    z = dc[valid]
    u = np.arange(W_s); v = np.arange(H_s)
    uu, vv = np.meshgrid(u, v)
    rx = ((uu - cx) / fx)[y1:y2, x1:x2][valid]
    ry = ((vv - cy) / fy)[y1:y2, x1:x2][valid]
    x = rx * z; y = ry * z
    frames.append((i, np.stack([x, y, z], axis=1).astype(np.float32)))

print(f'Using {len(frames)} frames')

def rodrigues(axis, angle):
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def fuse_and_score(axis_dir, axis_origin, rot_sign):
    """融合所有帧并返回紧致度分数 (越小越好 = 点云越紧致)."""
    all_pts = []
    for i, pts_cam in frames:
        th = np.radians(i * step_deg) * rot_sign
        R_rot = rodrigues(axis_dir, -th)
        pts_centered = pts_cam - axis_origin
        pts_rotated = (R_rot @ pts_centered.T).T + axis_origin
        pts_world = pts_rotated - axis_origin
        all_pts.append(pts_world)

    pts_all = np.vstack(all_pts)
    # 只取原点附近 8cm 内的点
    dists = np.linalg.norm(pts_all[:, [0, 2]], axis=1)
    near = pts_all[(dists < 0.08) & (np.abs(pts_all[:, 1]) < 0.06)]
    if len(near) < 100:
        return 999, 0
    # 分数 = mean distance to centroid (越小越紧致)
    centroid = near.mean(axis=0)
    score = np.mean(np.linalg.norm(near - centroid, axis=1))
    return score, len(near)

# ── 搜索 ──
print('\nSearching...')
results = []
Z = 0.408  # 从数据估算的深度

for axis_name, axis_dir in axis_dirs.items():
    for ox in [-0.02, 0.0, 0.02, 0.06]:
        for oy in [-0.02, 0.0, 0.02]:
            for rot_sign in [1.0, -1.0]:
                origin = np.array([ox, oy, Z])
                score, n = fuse_and_score(axis_dir, origin, rot_sign)
                results.append((score, n, axis_name, ox, oy, rot_sign))
                if n > 200:
                    print(f'  dir={axis_name:10s} origin=({ox*1000:+.0f},{oy*1000:+.0f},{Z*1000:.0f})mm sign={rot_sign:+.0f} score={score*1000:.1f}mm n={n}')

# 排序
results.sort(key=lambda x: x[0])
print(f'\n=== TOP 10 ===')
for score, n, aname, ox, oy, rs in results[:10]:
    print(f'  score={score*1000:.1f}mm n={n:5d}  dir={aname:10s} origin=({ox*1000:+.0f},{oy*1000:+.0f},{Z*1000:.0f})mm sign={rs:+.0f}')

best = results[0]
print(f'\nBest: score={best[0]*1000:.1f}mm inlier_radius, dir={best[2]}, X={best[3]*1000:.0f}mm Y={best[4]*1000:.0f}mm sign={best[5]:+.0f}')
