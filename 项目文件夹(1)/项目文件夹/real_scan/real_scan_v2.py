#!/usr/bin/env python3
"""真机扫描 v2: Depth-Anything-V2 深度补全替代 D435i 原始深度.

核心改动: 每帧RGB → Depth-Anything → 稠密深度 → D435i可靠点对齐 → 点云
D435i 只用于提供 metric scale 参考, 不再直接生成点云.

用法: python3 real_scan_v2.py [--skip-zero]
"""

import json
import os
import sys
import time
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/xie/项目文件夹/real_scan')
sys.path.insert(0, '/home/xie/项目文件夹/Depth-Anything-V2')
from config import *
from utils import world_rotate_z_around, save_ply, post_process
from turntable import TurntableController
from depth_anything_v2.dpt import DepthAnythingV2

# ── SAM 模型配置 ──
SAM_CKPT = os.path.join(os.path.expanduser('~'), '.cache/sam/sam_vit_b_01ec64.pth')
_sam_predictor = None
_sam_ok = None

# ── Depth-Anything 模型配置 ──
DA_CKPT = '/home/xie/项目文件夹/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
_da_model = None


def _check_sam():
    global _sam_ok
    if _sam_ok is None:
        _sam_ok = os.path.exists(SAM_CKPT)
    return _sam_ok


def _get_sam_predictor():
    global _sam_predictor
    if _sam_predictor is None and _check_sam():
        from segment_anything import sam_model_registry, SamPredictor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'  加载 SAM ({device})...')
        sam = sam_model_registry['vit_b'](checkpoint=SAM_CKPT)
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
    return _sam_predictor


def _get_da_model():
    global _da_model
    if _da_model is None:
        print(f'  加载 Depth-Anything-V2 Small...')
        _da_model = DepthAnythingV2(
            encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        ckpt = torch.load(DA_CKPT, map_location='cpu')
        _da_model.load_state_dict(ckpt)
        _da_model.eval()
        print(f'  Depth-Anything 就绪 ({sum(p.numel() for p in _da_model.parameters()):,} 参数)')
    return _da_model


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        sys.exit(f'标定文件不存在: {CALIB_FILE}')
    with open(CALIB_FILE, 'r') as f:
        data = json.load(f)
    R = np.array(data['R'])
    t = np.array(data['t'])
    plate_z = data['plate_z']
    disc_radius = data.get('radius_m', 0.1)
    rotation_center = np.array([0.0, 0.0])
    return R, t, plate_z, disc_radius, rotation_center


def predict_dense_depth(img_bgr):
    """Depth-Anything-V2: RGB(BGR) → relative depth [0,1] (near→far)."""
    model = _get_da_model()
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    scale = 518 / max(h_orig, w_orig)
    new_h = round(h_orig * scale / 14) * 14
    new_w = round(w_orig * scale / 14) * 14
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        depth = model(tensor)
    depth = F.interpolate(depth.unsqueeze(1), size=(h_orig, w_orig),
                          mode='bicubic', align_corners=False).squeeze().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def align_to_metric(pred_depth, d435i_depth):
    """相对深度 → metric 深度 (mm), 用 D435i 中心区域做最小二乘拟合."""
    h, w = d435i_depth.shape
    cy, cx = h // 2, w // 2
    # Center 25% region + valid depth range
    center = np.zeros((h, w), dtype=bool)
    center[cy - h // 8:cy + h // 8, cx - w // 8:cx + w // 8] = True
    valid = (d435i_depth > 300) & (d435i_depth < 2000) & center
    if valid.sum() < 100:
        valid = (d435i_depth > 300) & (d435i_depth < 2000)

    indices = np.where(valid)
    n = min(5000, len(indices[0]))
    if n < 50:
        # Fallback: use all non-zero depth
        indices = np.where(d435i_depth > 100)
        n = min(5000, len(indices[0]))
    if n < 50:
        return d435i_depth.copy(), 1.0, 0.0

    sel = np.random.choice(len(indices[0]), n, replace=False)
    rows, cols = indices[0][sel], indices[1][sel]
    disp = 1.0 / (pred_depth[rows, cols] + 0.01)
    metric = d435i_depth[rows, cols]
    A = np.column_stack([disp, np.ones(n)])
    coeff, _, _, _ = np.linalg.lstsq(A, metric, rcond=None)
    a, b = coeff

    result = a / (pred_depth + 0.01) + b
    result = np.clip(result, 0, 10000)
    return result.astype(np.float32), a, b


def remove_turntable_via_depth(depth_metric, plate_z_cam, z_tolerance=0.004):
    """在深度图层面过滤转台面: 比板面高出不足 tolerance 的深度设0."""
    h, w = depth_metric.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    # D435i intrinsics (hardcoded for now, from config)
    fx, fy, ppx, ppy = 617.0, 617.0, 320.0, 240.0
    z_cam = depth_metric / 1000.0  # mm → m
    # Project to world Z (approximate, using camera intrinsics)
    # For points near the center, world_Z ≈ -d_plane + cam_Z * cos(theta)
    # Simpler: use depth-based heuristic — anything within z_tolerance of plate_z_cam height
    # plate_z is in world coords. In cam coords, it's ~R.T @ [0,0,plate_z] - t
    # But we know plate is the lowest surface. Just filter depths within a narrow band.

    # Identify the "lowest" valid depth in each column (closest to plate)
    # The turntable is the closest "flat" surface → lowest Z values after alignment
    # Simpler: find the mode of valid depths in the center, remove pixels at that depth
    return depth_metric  # For now, just pass through


def depth_to_pointcloud(depth_metric, fx, fy, ppx, ppy, mask=None):
    """深度图 (mm) → 相机坐标系点云 (m). 可选 mask 过滤."""
    h, w = depth_metric.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    z = depth_metric / 1000.0
    valid = z > 0.05
    if mask is not None:
        valid = valid & mask
    x = (uu - ppx) * z / fx
    y = (vv - ppy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1).astype(np.float32)
    return pts


def get_sam_chair_mask(img_bgr):
    """SAM分割: 返回椅子mask (H,W) bool."""
    predictor = _get_sam_predictor()
    if predictor is None:
        return None
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)
    # 3x3 center positive + edge negative points
    cx, cy = w // 2, h // 2
    span_x, span_y = int(w * 0.12), int(h * 0.12)
    xs = [cx - span_x, cx, cx + span_x]
    ys = [cy - span_y, cy, cy + span_y]
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)
    edges = np.array([
        [5, 5], [w//2, 5], [w-5, 5],
        [5, h-5], [w//2, h-5], [w-5, h-5],
        [5, h//2], [w-5, h//2],
    ], dtype=np.float32)
    point_coords = np.vstack([grid, edges])
    point_labels = np.array([1]*len(grid) + [0]*len(edges))
    masks, scores, _ = predictor.predict(
        point_coords=point_coords, point_labels=point_labels,
        multimask_output=False)
    return masks[0].astype(bool)


def remove_plate_by_z(pts_world, plate_z, z_band=0.003):
    """在 world 坐标中: Z ≈ plate_z ± z_band → 转台面 → 删除."""
    is_plate = np.abs(pts_world[:, 2] - plate_z) < z_band
    r = np.sqrt(pts_world[:, 0]**2 + pts_world[:, 1]**2)
    is_disc = is_plate & (r < 0.15)  # within 15cm radius
    return pts_world[~is_disc]


def main():
    import pyrealsense2 as rs

    skip_zero = '--skip-zero' in sys.argv
    print('=' * 55)
    print('  Real Scan v2: Depth-Anything 深度补全')
    print('  RGB → DA-V2稠密深度 → D435i对齐 → 点云')
    print('=' * 55)

    # Pre-load models
    print('\n[初始化模型]')
    _get_da_model()
    sam_ok = _check_sam()
    if sam_ok:
        _get_sam_predictor()
    else:
        print('  SAM 未安装, 跳过 mask 辅助')

    R_calib, t_calib, plate_z, disc_radius, rotation_center = load_calibration()
    print(f'\n标定: plate_z={plate_z:.4f}, radius={disc_radius*100:.0f}cm')

    # Camera intrinsics
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        sys.exit('No RealSense camera')
    dev = devices[0]

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, DEPTH_FPS)
    profile = pipeline.start(cfg)
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(f'相机内参: fx={fx:.1f} fy={fy:.1f} cx={ppx:.1f} cy={ppy:.1f}')

    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()
    colorizer = rs.colorizer()

    for _ in range(30):
        pipeline.wait_for_frames()
        time.sleep(0.05)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        if f in ('classify_rgb.png', 'scan.ply'):
            os.remove(os.path.join(OUTPUT_DIR, f))

    tt = TurntableController(port=TURNTABLE_PORT)
    tt.open()

    try:
        total_rotation = 0.0

        # ── Step 1: 初始化拍照 ──
        print('\n放好家具后按 SPACE...')
        cv2.namedWindow('Place furniture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Place furniture', 960, 360)

        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_img = np.asanyarray(color_frame.get_data())
            depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            h, w = color_img.shape[:2]
            preview = np.hstack([color_img, cv2.resize(depth_colored, (w, h))])
            cv2.imshow('Place furniture', preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                tt.close()
                return
        cv2.destroyAllWindows()

        # ── Step 2: 零点标定 ──
        if skip_zero:
            print('\n跳过零点标定')
            tt.zero()
        else:
            print('\nPCA 自动零点...')
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            pc.map_to(aligned)
            verts = np.asanyarray(pc.calculate(depth).get_vertices()).view(np.float32)
            pts = verts.reshape(-1, 3)
            valid = np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0.01)
            pts = pts[valid]
            pts_w = (R_calib @ pts.T).T + t_calib
            # Simple PCA
            r = np.sqrt((pts_w[:, 0] - rotation_center[0])**2 +
                        (pts_w[:, 1] - rotation_center[1])**2)
            mask = (r < 0.12) & (pts_w[:, 2] > -0.05)
            pca_pts = pts_w[mask][:, :2]
            if len(pca_pts) > 100:
                centered = pca_pts - pca_pts.mean(axis=0)
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                main_dir = eigvecs[:, -1]
                ratio = eigvals[-1] / max(eigvals[0], 1e-10)
                if ratio > 1.5:
                    angle = math.degrees(math.atan2(main_dir[1], main_dir[0]))
                    targets = [0, 90, 180, 270]
                    best = min(targets, key=lambda t: abs((t - angle + 180) % 360 - 180))
                    rot = best - angle
                    if abs(rot) > 0.5:
                        print(f'  转台旋转 {rot:.1f}°')
                        tt.move_and_wait(rot, speed=TURNTABLE_DEFAULT_SPEED)
                        time.sleep(0.3)
            tt.zero()

        # ── Step 3: 多角度扫描 ──
        SCAN_ANGLES = list(range(0, 360, 15))
        print(f'\n扫描 {len(SCAN_ANGLES)} 帧 (每15°), Depth-Anything 补全深度...')

        frames_rgb = {}
        frames_da_depth = {}  # Depth-Anything completed metric depth

        cv2.namedWindow('Scanning', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Scanning', 960, 360)

        for i, target_deg in enumerate(SCAN_ANGLES):
            print(f'\n[{i+1}/{len(SCAN_ANGLES)}] → {target_deg}°')
            tt.move_absolute(target_deg, speed=TURNTABLE_DEFAULT_SPEED)
            tt.wait_stop(timeout=30)
            total_rotation = target_deg
            time.sleep(0.3)

            # Capture
            frames_list = pipeline.wait_for_frames()
            aligned = align.process(frames_list)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_img = np.asanyarray(color_frame.get_data())
            d435i_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

            # Depth-Anything 预测 + 对齐
            t0 = time.time()
            pred_depth = predict_dense_depth(color_img)
            depth_metric, a, b = align_to_metric(pred_depth, d435i_depth)
            t1 = time.time()

            frames_rgb[target_deg] = color_img
            frames_da_depth[target_deg] = depth_metric

            # Generate point cloud from completed depth
            # Use SAM mask to isolate chair (optional, skip for speed)
            chair_mask = None
            if sam_ok:
                chair_mask = get_sam_chair_mask(color_img)

            pts_cam = depth_to_pointcloud(depth_metric, fx, fy, ppx, ppy, mask=chair_mask)
            n_pts = len(pts_cam)
            print(f'  DA-V2 {t1-t0:.1f}s a={a:.0f} → {n_pts:,} 点')

            # Preview
            da_viz = np.clip(depth_metric / 4000 * 255, 0, 255).astype(np.uint8)
            da_viz_color = cv2.applyColorMap(da_viz, cv2.COLORMAP_INFERNO)
            d435i_viz = np.clip(d435i_depth / 4000 * 255, 0, 255).astype(np.uint8)
            d435i_viz_color = cv2.applyColorMap(d435i_viz, cv2.COLORMAP_INFERNO)
            preview = np.hstack([color_img, d435i_viz_color, da_viz_color])
            cv2.putText(preview, f'{target_deg} deg | DA-V2 {n_pts:,} pts',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(preview, 'RGB', (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(preview, 'D435i', (w + 10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(preview, 'DA-V2', (w*2 + 10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.imshow('Scanning', preview)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        # ── Step 4: 拼接 ──
        print(f'\n拼接 {len(frames_da_depth)} 帧...')
        angles = sorted(frames_da_depth.keys())
        ref_angle = angles[0]

        aligned_frames = []
        for deg in angles:
            depth_metric = frames_da_depth[deg]
            chair_mask = None
            if sam_ok and deg in frames_rgb:
                chair_mask = get_sam_chair_mask(frames_rgb[deg])
            pts_cam = depth_to_pointcloud(depth_metric, fx, fy, ppx, ppy, mask=chair_mask)

            # World transform
            pts_world = (R_calib @ pts_cam.T).T + t_calib

            # Remove turntable plate by Z
            pts_world = remove_plate_by_z(pts_world, plate_z, z_band=0.003)

            # Clip radius
            r = np.sqrt((pts_world[:, 0] - rotation_center[0])**2 +
                        (pts_world[:, 1] - rotation_center[1])**2)
            pts_world = pts_world[r < 0.14]

            # Rotate to reference angle
            rot_angle = ref_angle - deg
            pts_world = world_rotate_z_around(pts_world, rot_angle, rotation_center)

            aligned_frames.append(pts_world)
            print(f'  {deg:3d}°: {len(pts_world):,} 点')

        merged = np.vstack(aligned_frames)
        print(f'\n合并: {len(merged):,} 点')

        # ── Step 5: 后处理 ──
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged)
        pcd = pcd.voxel_down_sample(voxel_size=0.0015)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd, _ = pcd.remove_radius_outlier(nb_points=10, radius=0.015)
        merged_clean = np.asarray(pcd.points)
        print(f'去噪: {len(merged_clean):,} 点')

        pts_final = post_process(merged_clean, voxel_size=VOXEL_SIZE,
                                 outlier_nb=OUTLIER_NB, outlier_radius=OUTLIER_RADIUS,
                                 dbscan_eps=DBSCAN_EPS, dbscan_min=DBSCAN_MIN)
        if len(pts_final) == 0:
            print('后处理结果为空!')
            return

        # Y-up flip
        pts_final[:, 1] *= -1

        scan_path = SCAN_OUTPUT
        save_ply(pts_final, scan_path)
        fsize = os.path.getsize(scan_path)
        print(f'\n完成 → {scan_path}')
        print(f'  点数: {len(pts_final):,}  |  文件: {fsize:,} bytes')

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f'\n转台归零...')
        tt.move_absolute(0, speed=max(TURNTABLE_DEFAULT_SPEED, 20000))
        tt.wait_stop(timeout=60)
        tt.close()


if __name__ == '__main__':
    main()
