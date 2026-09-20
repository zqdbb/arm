#!/usr/bin/env python3
"""离线验证采集数据：逐帧查看 color/depth/点云 + YOLO mask 过滤."""
import cv2, numpy as np, json, sys
from pathlib import Path

CAPTURE_DIR = Path(__file__).parent / 'output/capture'

meta_path = CAPTURE_DIR / 'meta.json'
if not meta_path.exists():
    print('未找到 meta.json，请先运行 capture.py')
    sys.exit(1)

with open(meta_path) as f:
    meta = json.load(f)

fx, fy = meta['fx'], meta['fy']
ppx, ppy = meta['ppx'], meta['ppy']
W, H = meta['wxH_zoomed']
depth_scale = meta['depth_scale']
print(f'相机: {meta["camera"]}  zoom: {meta["zoom"]}x')
print(f'{W}x{H}  fx={fx:.1f} fy={fy:.1f} ppx={ppx:.1f} ppy={ppy:.1f}')
print()

color_files = sorted((CAPTURE_DIR / 'color').glob('*.jpg'))
depth_files = sorted((CAPTURE_DIR / 'depth').glob('*.png'))

if not depth_files:
    print('无数据文件')
    sys.exit(1)

# ── YOLO ──
from ultralytics import YOLO
yolo = YOLO(str(Path(__file__).parent / 'yolov8n-seg.pt'))
use_mask = True  # 默认开 mask
mask_bool = np.zeros((H, W), dtype=bool)
print('YOLO 已加载  按 M 切换 mask  按 P 存点云\n')

# ── 射线 ──
u = np.arange(W); v = np.arange(H)
uu, vv = np.meshgrid(u, v)
ray_x = (uu - ppx) / fx
ray_y = (vv - ppy) / fy

idx = 0
need_redo_mask = True

while True:
    cf = color_files[idx]
    df = depth_files[idx]

    color = cv2.imread(str(cf))
    depth_mm = cv2.imread(str(df), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        print(f'{df.name}: 读取失败')
        idx = (idx + 1) % len(depth_files)
        continue

    if color.shape[:2] != (H, W):
        color = cv2.resize(color, (W, H))

    depth_m = depth_mm.astype(np.float32) * 0.001

    # YOLO mask（换帧时重新算）
    if use_mask and need_redo_mask:
        results = yolo(color, verbose=False, classes=[56])
        mask_bool.fill(False)
        if results[0].masks is not None:
            best, best_c = None, 0
            for j in range(len(results[0].boxes)):
                if results[0].names[int(results[0].boxes.cls[j])] == 'chair':
                    c = float(results[0].boxes.conf[j])
                    if c > best_c:
                        best_c = c
                        best = j
            if best is not None:
                m = results[0].masks.data[best].cpu().numpy()
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                mask_bool = m > 0.5
        need_redo_mask = False

    valid = depth_m > 0.01
    valid_masked = valid & mask_bool if use_mask else valid

    # 深度热力图
    d_clip = np.clip(depth_m, 0.2, 0.6)
    d_norm = ((d_clip - 0.2) / 0.4 * 255).astype(np.uint8)
    d_heat = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

    if use_mask:
        d_heat[~mask_bool] = (d_heat[~mask_bool] * 0.15).astype(np.uint8)
        d_heat[~valid] = 50
        # mask 轮廓
        cnts, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(d_heat, cnts, -1, (255, 255, 255), 2)

    # 拼接
    bar = np.full((H, 4, 3), 100, dtype=np.uint8)
    display = np.hstack([color, bar, d_heat])

    # HUD
    total = W * H
    n_full = valid.sum()
    n_masked = valid_masked.sum()
    pct_full = 100 * n_full / total
    pct_masked = 100 * n_masked / (mask_bool.sum() + 1) if use_mask else 0
    mode_str = 'MASK ON' if use_mask else 'MASK OFF'
    color_stat = (0, 255, 0) if pct_masked > 20 else (0, 200, 255) if pct_masked > 5 else (0, 0, 255)

    cv2.putText(display, f'Frame {idx}/{len(depth_files)-1}  {cf.name}  [{mode_str}]',
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(display, f'Full valid: {pct_full:.1f}%  |  Mask valid: {pct_masked:.1f}%',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_stat, 2)
    cv2.putText(display, f'Mask px: {mask_bool.sum():,}  Valid in mask: {n_masked:,}',
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(display, 'M=mask  P=save PLY  L/R=browse  Q=quit',
                (10, display.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    scale = min(1100 / display.shape[1], 700 / display.shape[0], 1.0)
    ds = cv2.resize(display, None, fx=scale, fy=scale)
    cv2.imshow('Data Checker', ds)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('m'), ord('M')):
        use_mask = not use_mask
        need_redo_mask = True
        print(f'  Mask: {"ON" if use_mask else "OFF"}')
    elif key == ord('p'):
        # 反投影 → 点云（只取 mask 内或全图、看当前模式）
        sample = 4
        vs = valid_masked[::sample, ::sample]
        zs = depth_m[::sample, ::sample][vs]
        xs = (ray_x[::sample, ::sample][vs] * zs).flatten()
        ys = (ray_y[::sample, ::sample][vs] * zs).flatten()
        zs = zs.flatten()
        pts = np.stack([xs, ys, zs], axis=1)

        d_ok = (zs > 0.15) & (zs < 0.7)
        pts = pts[d_ok]

        center_z_vals = depth_m[H//2-10:H//2+10, W//2-10:W//2+10]
        center_z_vals = center_z_vals[center_z_vals > 0.01]
        suf = '_masked' if use_mask else ''
        out = CAPTURE_DIR.parent / f'check_frame_{idx:03d}{suf}.ply'
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(str(out), pcd)

        print(f'\n  --- Frame {idx} ({cf.name}) [{mode_str}] ---')
        if len(center_z_vals) > 0:
            print(f'  中心深度: {np.median(center_z_vals)*1000:.0f}mm')
        print(f'  3D点数: {len(pts):,}')
        if len(pts) > 50:
            print(f'  包围盒: X={np.ptp(pts[:,0])*1000:.0f}mm  '
                  f'Y={np.ptp(pts[:,1])*1000:.0f}mm  '
                  f'Z={np.ptp(pts[:,2])*1000:.0f}mm')
        print(f'  → {out}\n')

    elif key == 81:   # LEFT
        idx = (idx - 1) % len(depth_files)
        need_redo_mask = True
    elif key == 83:   # RIGHT
        idx = (idx + 1) % len(depth_files)
        need_redo_mask = True

cv2.destroyAllWindows()
