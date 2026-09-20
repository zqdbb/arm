#!/usr/bin/env python3
"""Depth-Anything-V2 深度补全: RGB → 稠密深度图 → 替换/融合 D435i 原始深度.

用法:
  python3 depth_completion.py <rgb_path>                    # 纯预测, 保存深度图
  python3 depth_completion.py <rgb_path> <depth_frame>      # D435i深度对齐, 生成点云
"""

import sys
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/xie/项目文件夹/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

MODEL_CKPT = '/home/xie/项目文件夹/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth'
MODEL_ENCODER = 'vits'
MODEL_MAX_SIZE = 518  # ViT-S input size

_model = None


def get_model():
    global _model
    if _model is None:
        _model = DepthAnythingV2(
            encoder=MODEL_ENCODER,
            features=64,
            out_channels=[48, 96, 192, 384]
        )
        ckpt = torch.load(MODEL_CKPT, map_location='cpu')
        _model.load_state_dict(ckpt)
        _model.eval()
    return _model


def predict_depth(img_bgr):
    """输入 BGR 图像 (H,W,3), 返回归一化相对深度图 (H,W) [0,1] near→far."""
    model = get_model()
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize so longest side = MODEL_MAX_SIZE, pad to multiple of 14
    scale = MODEL_MAX_SIZE / max(h_orig, w_orig)
    new_h = round(h_orig * scale / 14) * 14
    new_w = round(w_orig * scale / 14) * 14
    img_resized = cv2.resize(img_rgb, (new_w, new_h))
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)  # 1,3,H,W

    with torch.no_grad():
        depth = model(tensor)
    # depth is (1, H_out, W_out) → upsample to original
    depth = F.interpolate(depth.unsqueeze(1), size=(h_orig, w_orig),
                          mode='bicubic', align_corners=False).squeeze()
    depth = depth.numpy()
    # Normalize to [0, 1]: 0 = far, 1 = near
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)


def align_depth_to_metric(pred_depth, d435i_depth, mask=None):
    """把相对深度对齐到 D435i 的 metric 深度 (mm).

    使用稳健的线性回归: metric = a / (pred_depth + eps) + b
    拟合可靠点 (D435i 深度在 300-2000mm 范围内, 排除 0).
    """
    if mask is None:
        mask = (d435i_depth > 300) & (d435i_depth < 2000)

    # 取子集做拟合: 中心区域 + 随机采样
    h, w = d435i_depth.shape
    cy, cx = h // 2, w // 2
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[cy - h//8:cy + h//8, cx - w//8:cx + w//8] = True

    valid = mask & center_mask
    if valid.sum() < 100:
        valid = mask

    n_sample = min(5000, valid.sum())
    indices = np.where(valid)
    if len(indices[0]) > n_sample:
        sel = np.random.choice(len(indices[0]), n_sample, replace=False)
        y_sample = indices[0][sel]
        x_sample = indices[1][sel]
    else:
        y_sample, x_sample = indices

    # metric = a * disparity + b  where disparity = 1/(pred_depth + eps)
    eps = 0.01
    disp = 1.0 / (pred_depth[y_sample, x_sample] + eps)
    metric = d435i_depth[y_sample, x_sample]

    # Robust linear regression (RANSAC-like)
    A = np.column_stack([disp, np.ones_like(disp)])
    try:
        coeff, _, _, _ = np.linalg.lstsq(A, metric, rcond=None)
        a, b = coeff
    except np.linalg.LinAlgError:
        a, b = 1.0, 0.0

    # Apply
    depth_metric = a / (pred_depth + eps) + b
    depth_metric = np.clip(depth_metric, 0, 10000)
    return depth_metric.astype(np.float32), a, b


def depth_to_pointcloud(depth_metric, fx, fy, ppx, ppy):
    """深度图 → 点云 (相机坐标系)."""
    h, w = depth_metric.shape
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)
    z = depth_metric / 1000.0  # mm → m
    valid = z > 0.01
    x = (uu - ppx) * z / fx
    y = (vv - ppy) * z / fy
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return pts.astype(np.float32)


if __name__ == '__main__':
    import pyrealsense2 as rs

    rgb_path = sys.argv[1] if len(sys.argv) > 1 else None
    use_d435i = '--d435i' in sys.argv

    if use_d435i:
        # Connect to D435i and capture one frame
        print('Connecting to D435i...')
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            sys.exit('No camera')
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(devices[0].get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(cfg)
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipeline.wait_for_frames()
            time.sleep(0.05)
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        img_bgr = np.asanyarray(color_frame.get_data())
        d435i_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
        pipeline.stop()
    elif rgb_path:
        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            sys.exit(f'Cannot read {rgb_path}')
        d435i_depth = None
        # Default D435i intrinsics
        fx, fy, ppx, ppy = 617.0, 617.0, 320.0, 240.0
    else:
        sys.exit('Usage: python3 depth_completion.py <rgb.png> [--d435i]')

    print(f'Image: {img_bgr.shape[1]}x{img_bgr.shape[0]}')

    # 1. Predict dense depth
    t0 = time.time()
    pred_depth = predict_depth(img_bgr)
    t1 = time.time()
    print(f'Depth prediction: {t1 - t0:.1f}s, '
          f'range=[{pred_depth.min():.3f}, {pred_depth.max():.3f}]')

    # 2. Save predicted depth visualization
    depth_viz = (pred_depth * 255).astype(np.uint8)
    depth_viz_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_INFERNO)

    out_dir = '/home/xie/项目文件夹/real_scan/output'
    cv2.imwrite(os.path.join(out_dir, 'depth_predicted.png'), depth_viz_color)
    print(f'Saved: {out_dir}/depth_predicted.png')

    # 3. If D435i available, align and generate point cloud
    if d435i_depth is not None:
        depth_metric, a, b = align_depth_to_metric(pred_depth, d435i_depth)
        print(f'Metric alignment: a={a:.1f}, b={b:.1f}')

        # Save comparison
        d435i_viz = np.clip(d435i_depth / 4000 * 255, 0, 255).astype(np.uint8)
        d435i_viz_color = cv2.applyColorMap(d435i_viz, cv2.COLORMAP_INFERNO)
        completed_viz = np.clip(depth_metric / 4000 * 255, 0, 255).astype(np.uint8)
        completed_viz_color = cv2.applyColorMap(completed_viz, cv2.COLORMAP_INFERNO)

        side_by_side = np.hstack([img_bgr, d435i_viz_color, completed_viz_color])
        cv2.imwrite(os.path.join(out_dir, 'depth_comparison.png'), side_by_side)
        print(f'Saved: {out_dir}/depth_comparison.png (RGB | D435i | DA-V2)')

        # Generate point clouds
        pts_d435i = depth_to_pointcloud(d435i_depth, fx, fy, ppx, ppy)
        pts_completed = depth_to_pointcloud(depth_metric, fx, fy, ppx, ppy)

        import open3d as o3d

        # D435i raw
        pcd_d435i = o3d.geometry.PointCloud()
        pcd_d435i.points = o3d.utility.Vector3dVector(pts_d435i)
        o3d.io.write_point_cloud(os.path.join(out_dir, 'depth_d435i.ply'), pcd_d435i)

        # DA-V2 completed
        pcd_completed = o3d.geometry.PointCloud()
        pcd_completed.points = o3d.utility.Vector3dVector(pts_completed)
        o3d.io.write_point_cloud(os.path.join(out_dir, 'depth_completed.ply'), pcd_completed)

        print(f'D435i point cloud: {len(pts_d435i):,} pts')
        print(f'DA-V2 completed:  {len(pts_completed):,} pts')
        print(f'Saved: {out_dir}/depth_d435i.ply, {out_dir}/depth_completed.ply')
