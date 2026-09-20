#!/usr/bin/env python3
"""从72帧椅子质心轨迹拟合转台旋转轴（不依赖校准）."""
import cv2, numpy as np, json
from pathlib import Path

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/capture'

with open(CAPTURE_DIR / 'meta.json') as f:
    meta = json.load(f)

fx, fy = meta['fx'], meta['fy']
ppx, ppy = meta['ppx'], meta['ppy']
W, H = meta['wxH_zoomed']
step_deg = meta['step_deg']
n_frames = meta['n_frames']

from ultralytics import YOLO
yolo = YOLO(str(BASE / 'yolov8n-seg.pt'))

u = np.arange(W); v = np.arange(H)
uu, vv = np.meshgrid(u, v)
ray_x = (uu - ppx) / fx
ray_y = (vv - ppy) / fy

color_files = sorted((CAPTURE_DIR / 'color').glob('*.jpg'))
depth_files = sorted((CAPTURE_DIR / 'depth').glob('*.png'))

centroids = []  # 每帧的3D质心

for i in range(min(n_frames, len(depth_files))):
    color = cv2.imread(str(color_files[i]))
    depth_mm = cv2.imread(str(depth_files[i]), cv2.IMREAD_UNCHANGED)
    if color is None or depth_mm is None:
        continue
    if color.shape[:2] != (H, W):
        color = cv2.resize(color, (W, H))
    depth_m = depth_mm.astype(np.float32) * 0.001

    # YOLO
    results = yolo(color, verbose=False, classes=[56])
    mask = np.zeros((H, W), dtype=bool)
    if results[0].masks is not None:
        best, best_c = None, 0
        for j in range(len(results[0].boxes)):
            if results[0].names[int(results[0].boxes.cls[j])] == 'chair':
                c = float(results[0].boxes.conf[j])
                if c > best_c: best_c = c; best = j
        if best is not None:
            m = results[0].masks.data[best].cpu().numpy()
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = m > 0.5

    valid = (depth_m > 0.01) & mask
    if valid.sum() < 100:
        continue

    z = depth_m[valid]
    x = ray_x[valid] * z
    y = -ray_y[valid] * z  # 翻转Y：相机Y朝下→世界Y朝上
    pts = np.stack([x, y, z], axis=1)
    centroids.append(pts.mean(axis=0))

centroids = np.array(centroids)
print(f'有效帧: {len(centroids)}/{n_frames}')

# ── 拟合 3D 圆 ──
# 1. 拟合平面 (圆心在平面上)
c_mean = centroids.mean(axis=0)
C = centroids - c_mean
U, S, Vt = np.linalg.svd(C)
normal = Vt[2, :]  # 法向 = 转轴方向
normal = normal / np.linalg.norm(normal)
print(f'拟合转轴方向: {normal}')

# 2. 投影到平面 → 拟合 2D 圆
# 平面局部坐标: 两个基向量
if abs(normal[0]) < 0.9:
    e1 = np.cross(normal, [1, 0, 0])
else:
    e1 = np.cross(normal, [0, 1, 0])
e1 = e1 / np.linalg.norm(e1)
e2 = np.cross(normal, e1)

u_proj = C @ e1
v_proj = C @ e2

# 最小二乘圆拟合: (u-a)^2 + (v-b)^2 = r^2
# 展开: u^2 + v^2 = 2a*u + 2b*v + (r^2 - a^2 - b^2)
A = np.column_stack([2*u_proj, 2*v_proj, np.ones(len(u_proj))])
b = u_proj**2 + v_proj**2
x = np.linalg.lstsq(A, b, rcond=None)[0]
a, b_2d, c = x
r = np.sqrt(c + a**2 + b_2d**2)
center_2d = np.array([a, b_2d])

# 转回 3D
axis_point_est = c_mean + e1 * a + e2 * b_2d
print(f'拟合圆心 (轴点): {axis_point_est*1000} mm')
print(f'拟合半径: {r*1000:.1f} mm')

# 验证：各帧质心到轴的距离
dists = []
for ci in centroids:
    v = ci - axis_point_est
    dist = np.linalg.norm(np.cross(v, normal))  # 到轴的距离
    dists.append(dist)
dists = np.array(dists) * 1000
print(f'各帧到轴距离: mean={dists.mean():.0f}mm std={dists.std():.0f}mm')

# 与原校准对比
calib = meta.get('calib', {})
if calib:
    R_cal = np.array(calib['R'])
    t_cal = np.array(calib['t'])
    print(f'\n原校准轴点: {t_cal*1000} mm')
    print(f'原校准 R[2,:] (法向): {R_cal[2,:]}')
    print(f'原校准 R[1,:] (y轴):  {R_cal[1,:]}')

# 保存
result = {
    'axis_dir': normal.tolist(),
    'axis_point': axis_point_est.tolist(),
    'radius_m': float(r),
    'n_frames': len(centroids),
    'note': '由72帧椅子质心轨迹拟合，不依赖标定'
}
out_path = BASE / 'output/axis_fit.json'
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\n已保存 → {out_path}')
