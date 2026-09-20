#!/usr/bin/env python3
"""
V14 家具重建 — 纯深度点云融合 (无 AI 生成)
══════════════════════════════════════════════════════════════
  采集 → YOLO+SAM 抠 mask → 72帧深度反投影 → 点云融合 → Poisson 重建

和 V12/V13 的区别:
  - 不调 Hyper3D、不用 TripoSR
  - 模型 100% 来自 D435i 实测深度数据
  - 精度取决于深度传感器，不靠 AI "脑补"
  - 不需要 Blender (trimesh 直接导出)

使用方式:
  python3 scan_recon_v14.py --name chair
  python3 scan_recon_v14.py --skip-capture
"""

import cv2, json, time, os, sys, argparse
import numpy as np
from pathlib import Path

# ─── 常量 ────────────────────────────────────────────
BASE = Path(__file__).parent
OUT = BASE / 'output/v14/chair'
W, H = 640, 480
W_OUT, H_OUT = 500, 500
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG  # 72
N_AVG = 5
LASER_POWER = 150

CUSTOM_MODEL = 'yolov8n-seg.pt'
SAM_MODEL = BASE / 'sam_vit_b_01ec64.pth'

# Poisson 重建参数
POISSON_DEPTH = 8       # 重建深度 (越大越精细，也越慢)
VOXEL_SIZE = 0.002      # 点云下采样体素 (米), 2mm
OUTLIER_NB = 30         # 统计离群点过滤邻居数
OUTLIER_STD = 1.5       # 离群点标准差阈值
FRAME_SAMPLE_MAX = 3000 # 每帧最多采样点数


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 1: 采集 (同 V12)                                  ║
# ╚══════════════════════════════════════════════════════════╝

def do_capture(calib):
    import pyrealsense2 as rs
    from turntable import TurntableController

    print('\n' + '=' * 60)
    print(f'  Step 1/4  采集: {STEP_DEG}deg × {N_FRAMES} 帧 ({W}×{H} → {W_OUT}×{H_OUT})')
    print('=' * 60)

    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    pipe = rs.pipeline()
    for attempt in range(3):
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
            profile = pipe.start(cfg)
            time.sleep(0.5)
            pipe.wait_for_frames(timeout_ms=5000)
            break
        except RuntimeError as e:
            print(f'  启动尝试 {attempt+1}/3: {e}')
            try:
                pipe.stop()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.0)
                pipe = rs.pipeline()
    else:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            try:
                devices[0].hardware_reset()
                time.sleep(2.0)
            except Exception as e:
                print(f'  硬件复位失败: {e}')
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        profile = pipe.start(cfg)
        time.sleep(0.5)

    align = rs.align(rs.stream.color)
    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)
    dscale = ds.get_depth_scale()

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    crop_size = min(W, H)
    x0 = (W - crop_size) // 2
    scale = W_OUT / crop_size
    calib['fx_full'] = intr.fx * scale
    calib['fy_full'] = intr.fy * scale
    calib['ppx_full'] = (intr.ppx - x0) * scale
    calib['ppy_full'] = (intr.ppy - (H - crop_size) // 2) * scale
    calib['width_full'] = W_OUT
    calib['height_full'] = H_OUT
    calib['depth_scale'] = dscale

    calib_path = OUT / 'calibrate.json'
    with open(calib_path, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'标定已保存: {calib_path}')

    (OUT / 'color').mkdir(exist_ok=True)
    (OUT / 'depth').mkdir(exist_ok=True)

    t0 = time.time()
    for i in range(N_FRAMES):
        deg = i * STEP_DEG
        tt.move_absolute(deg, speed=5000)
        tt.wait_stop(timeout=30)

        d_sum = np.zeros((H, W), np.float64)
        color_full = None
        n_good = 0
        for _ in range(N_AVG):
            frames = pipe.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            df = aligned.get_depth_frame()
            cf = aligned.get_color_frame()
            if not df or not cf:
                continue
            d_sum += np.asanyarray(df.get_data()).astype(np.float64)
            if color_full is None:
                color_full = np.asanyarray(cf.get_data())
            n_good += 1

        if n_good == 0:
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  无数据')
            continue

        crop_size = min(W, H)
        x0 = (W - crop_size) // 2
        y0 = (H - crop_size) // 2
        color_crop = color_full[y0:y0+crop_size, x0:x0+crop_size]
        color_out = cv2.resize(color_crop, (W_OUT, H_OUT), interpolation=cv2.INTER_LINEAR)
        depth_mm = (d_sum / n_good) * dscale * 1000.0
        depth_crop = depth_mm[y0:y0+crop_size, x0:x0+crop_size]
        depth_out = cv2.resize(depth_crop, (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.png'), color_out)
        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'), depth_out.astype(np.uint16))

        dvals = depth_out[depth_out > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  '
              f'valid:{len(dvals)//1000}k  med={np.median(dvals):.0f}mm  {el:.0f}s')

    pipe.stop()
    tt.close()
    print(f'采集完成 ({time.time()-t0:.0f}s)\n')


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 2: YOLO + SAM 抠图 + 测量 (同 V12)                ║
# ╚══════════════════════════════════════════════════════════╝

def measure_chair_from_masks(masks_dict, calib):
    """4 个 SAM 精修帧做像素-距离法测量（深度直方图峰值层）."""
    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    dscale = calib.get('depth_scale', 0.001)
    depth_dir = OUT / 'depth'
    view_indices = [0, 18, 36, 54]

    heights, widths = [], []

    for idx in view_indices:
        mask = masks_dict.get(idx)
        if mask is None or mask.max() == 0:
            continue
        ys, xs = np.where(mask > 128)
        if len(ys) < 200:
            continue

        dep = cv2.imread(str(depth_dir / f'{idx:03d}.png'), cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue

        dvals = dep[ys, xs].astype(float)
        valid = dvals > 0
        if valid.sum() < 200:
            continue

        ys_v, xs_v = ys[valid], xs[valid]
        d_valid = dvals[valid]
        hist, edges = np.histogram(d_valid, bins=min(30, max(5, len(d_valid)//50)))
        peak_bin = np.argmax(hist)
        d_center = (edges[peak_bin] + edges[min(peak_bin+1, len(edges)-1)]) / 2

        in_layer = np.abs(d_valid - d_center) < 15
        if in_layer.sum() < 100:
            continue

        ys_f = ys_v[in_layer]
        xs_f = xs_v[in_layer]
        d_layer = d_valid[in_layer]
        dist_m = float(np.median(d_layer)) * dscale

        h_px = ys_f.max() - ys_f.min()
        w_px = np.percentile(xs_f, 98) - np.percentile(xs_f, 2)
        h_m = h_px * dist_m / fy
        w_m = w_px * dist_m / fx
        heights.append(h_m)
        widths.append(w_m)

    if not heights:
        return None

    heights = np.array(heights)
    widths = np.array(widths)
    chair_h = float(np.median(heights))
    sorted_w = np.sort(widths)
    chair_w = float(np.mean(sorted_w[-2:]))
    chair_d = float(np.mean(sorted_w[:2]))

    result = {
        'height_m': round(chair_h, 3),
        'width_m': round(chair_w, 3),
        'depth_m': round(chair_d, 3),
        'n_measured': len(heights),
    }

    print(f'\n  ┌─ SAM 视角测量 ({result["n_measured"]} 帧, 深度峰值层)')
    print(f'  ├─ 椅子高度: {result["height_m"]*100:.1f}cm')
    print(f'  ├─ 椅子宽度: {result["width_m"]*100:.1f}cm')
    print(f'  └─ 椅子深度: {result["depth_m"]*100:.1f}cm')
    return result


def do_yolo_and_measure(calib):
    """YOLOv8n-seg 初筛 + SAM 精修 + 自动测量."""
    from ultralytics import YOLO

    cW = calib.get('width_full', W_OUT)
    cH = calib.get('height_full', H_OUT)
    step = calib.get('step_deg', STEP_DEG)

    color_files = sorted((OUT / 'color').glob('*.png'))
    if len(color_files) < 36:
        print(f'图片不足: {len(color_files)} 张')
        return None, None

    print('=' * 60)
    print('  Step 2/4  YOLO 初筛 + SAM 精修 + 自动测量')
    print('=' * 60)

    print(f'  加载 YOLO: {CUSTOM_MODEL}')
    yolo_obj = YOLO(str(CUSTOM_MODEL))

    sam_ok = Path(str(SAM_MODEL)).exists()
    sam_predictor = None
    if sam_ok:
        print(f'  加载 SAM: {SAM_MODEL}')
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_MODEL))
        sam_predictor = SamPredictor(sam)

    # Pass 1: YOLO 全部帧
    masks_dict = {}
    bboxes = {}
    detected = 0

    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None:
            continue

        results = yolo_obj(img, verbose=False)
        r = results[0]
        mask_binary = np.zeros((cH, cW), dtype=np.uint8)
        bboxes[i] = None

        if r.boxes is not None and len(r.boxes) > 0:
            best_idx = int(r.boxes.conf.argmax().item())
            conf = r.boxes.conf[best_idx].item()
            if conf >= 0.3:
                detected += 1
                xyxy = r.boxes.xyxy[best_idx].cpu().numpy()
                bboxes[i] = xyxy
                mask_raw = r.masks.data[best_idx].cpu().numpy()
                mask_resized = cv2.resize(mask_raw, (cW, cH))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
        masks_dict[i] = mask_binary

    print(f'  YOLO 检出: {detected}/{len(color_files)}')

    # Pass 2: SAM 精修 4 视角帧
    view_indices = [0, 18, 36, 54]
    if sam_predictor:
        print(f'\n  SAM 精修 {len(view_indices)} 个视角帧...')
        for idx in view_indices:
            if bboxes.get(idx) is None:
                continue
            cf = color_files[idx]
            img = cv2.imread(str(cf))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)
            masks_sam, scores, _ = sam_predictor.predict(
                box=bboxes[idx][None, :], multimask_output=False
            )
            best_m = masks_sam[scores.argmax()]
            masks_dict[idx] = (best_m > 0).astype(np.uint8) * 255
            n_px = int(best_m.sum())
            print(f'    视角 {idx*step:3d}° (frame {idx:03d}): SAM mask={n_px//1000}k px')

    # Pass 3: 所有帧深度峰值过滤 + 形态学清理
    for i in range(len(color_files)):
        mask = masks_dict.get(i)
        if mask is None or mask.sum() < 500:
            continue

        dep = cv2.imread(str(OUT / 'depth' / f'{i:03d}.png'), cv2.IMREAD_UNCHANGED)
        if dep is not None:
            ys, xs = np.where(mask > 128)
            d_vals = dep[ys, xs].astype(float)
            valid = d_vals > 0
            if valid.sum() > 50:
                ys_v, xs_v = ys[valid], xs[valid]
                d_vals = d_vals[valid]
                hist, edges = np.histogram(d_vals, bins=min(30, max(5, len(d_vals)//50)))
                peak_bin = np.argmax(hist)
                d_center = (edges[peak_bin] + edges[min(peak_bin+1, len(edges)-1)]) / 2
                keep = np.abs(d_vals - d_center) < 30
                refined = np.zeros_like(mask)
                refined[ys_v[keep], xs_v[keep]] = 255
                mask = refined

        if mask.sum() > 500:
            k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n_labels > 1:
                largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = (labels == largest).astype(np.uint8) * 255

        masks_dict[i] = mask

    # 统计
    for i in range(len(color_files)):
        if i < 3 or (i+1) % 18 == 0:
            m = masks_dict.get(i)
            n_px = (m > 0).sum() if m is not None else 0
            tag = 'SAM' if i in view_indices and sam_ok else ('YOLO' if n_px > 500 else '✗')
            print(f'  [{i+1}/{len(color_files)}] {i*step:3d}deg  mask={n_px//1000}k px  [{tag}]')

    # ── 保存 4 视角 RGBA 抠图供验证 ──
    views_dir = OUT / 'views'
    views_dir.mkdir(exist_ok=True)
    for idx in view_indices:
        if masks_dict.get(idx) is None or masks_dict[idx].sum() < 500:
            continue
        img = cv2.imread(str(color_files[idx]))
        if img is None:
            continue
        mask = masks_dict[idx]
        rgba = np.dstack([img, mask])
        cv2.imwrite(str(views_dir / f'view_{idx:03d}.png'), rgba)

    # 测量
    chair_dims = measure_chair_from_masks(masks_dict, calib)
    if chair_dims:
        print(f'\n  ★ 测量结果: 高{chair_dims["height_m"]*100:.1f}cm '
              f'宽{chair_dims["width_m"]*100:.1f}cm 深{chair_dims["depth_m"]*100:.1f}cm')

    return masks_dict, chair_dims


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 3: 深度点云融合 + Poisson 重建                    ║
# ╚══════════════════════════════════════════════════════════╝

def do_depth_reconstruction(masks_dict, calib):
    """72帧深度图 × mask → 反投影 → 旋转对齐 → 点云融合 → Poisson → mesh."""
    import open3d as o3d

    print('=' * 60)
    print('  Step 3/4  深度点云融合 + Poisson 重建')
    print('=' * 60)

    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    ppx = calib.get('ppx_full', 250.0)
    ppy = calib.get('ppy_full', 250.0)
    dscale = calib.get('depth_scale', 0.001)

    depth_dir = OUT / 'depth'
    depth_files = sorted(depth_dir.glob('*.png'))

    if len(depth_files) < 4:
        print(f'深度图不足: {len(depth_files)} 张')
        return None

    all_pts = []
    total_valid = 0

    # ── 逐帧反投影 ──
    for i, dp in enumerate(depth_files):
        idx = int(dp.stem)
        mask = masks_dict.get(idx)
        if mask is None or mask.sum() < 500:
            continue

        dep = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue

        # mask 过滤深度
        valid_mask = (mask > 128) & (dep > 0)
        ys, xs = np.where(valid_mask)
        if len(ys) < 200:
            continue

        d_m = dep[ys, xs].astype(float) * dscale  # 深度 (米)

        # 反投影: 像素 → 相机 3D
        X_cam = (xs - ppx) * d_m / fx
        Y_cam = (ys - ppy) * d_m / fy
        Z_cam = d_m
        pts_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)

        # 绕转台轴旋转 -angle → 对齐到 0° 帧
        #   相机坐标下绕轴 n (转台法线) 旋转, 中心=转台中心在相机3D坐标
        angle = idx * STEP_DEG
        rad = np.radians(-angle)

        R_calib = np.array(calib['R'])
        t_calib = np.array(calib['t'])
        n = R_calib[2]                           # 转台法线=旋转轴 (已归一化)
        cx_w = calib.get('cx', 0.0)
        cy_w = calib.get('cy', 0.0)
        # 平面坐标(cx_w, cy_w, ||t||) → 相机坐标: center = R_calib @ [cx_w,cy_w,||t||]
        center_cam = R_calib @ np.array([cx_w, cy_w, np.linalg.norm(t_calib)])

        # Rodrigues: 绕 n 旋转 rad
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R_rot = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        pts_world = (pts_cam - center_cam) @ R_rot.T + center_cam

        # 3D 空间裁切: 只保留转台中心附近的点 (椅子区域 ~10cm)
        dist = np.linalg.norm(pts_world - center_cam, axis=1)
        keep_3d = dist < 0.12  # 12cm 内
        pts_world = pts_world[keep_3d]

        if len(pts_world) < 50:
            continue

        # 下采样
        if len(pts_world) > FRAME_SAMPLE_MAX:
            idx_sample = np.random.choice(len(pts_world), FRAME_SAMPLE_MAX, replace=False)
            pts_world = pts_world[idx_sample]

        all_pts.append(pts_world)
        total_valid += 1

        if (i+1) % 12 == 0 or i == 0:
            print(f'  [{i+1}/{len(depth_files)}] frame {idx:03d} '
                  f'{angle:3d}deg → {len(pts_world)} 点')

    print(f'\n  有效帧: {total_valid}/{len(depth_files)}')

    if len(all_pts) < 4:
        print('  有效点云太少，无法重建')
        return None

    # ── 合并 + 下采样 ──
    merged = np.vstack(all_pts)
    print(f'  合并点云: {len(merged)} 点')

    # 导出去噪前的原始合并点云
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(merged)
    (OUT / 'models').mkdir(exist_ok=True)
    o3d.io.write_point_cloud(str(OUT / 'models' / 'pointcloud_raw.ply'), raw_pcd)
    print(f'  已导出原始点云: {OUT}/models/pointcloud_raw.ply')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)

    # 体素下采样
    print(f'  体素下采样 (voxel={VOXEL_SIZE*1000:.0f}mm)...')
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    print(f'  下采样后: {len(pcd.points)} 点')

    # 统计离群点过滤
    print(f'  去噪 ({OUTLIER_NB}近邻, {OUTLIER_STD}σ)...')
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=OUTLIER_NB, std_ratio=OUTLIER_STD)
    print(f'  去噪后: {len(pcd.points)} 点')

    o3d.io.write_point_cloud(str(OUT / 'models' / 'pointcloud_clean.ply'), pcd)
    print(f'  已导出干净点云: {OUT}/models/pointcloud_clean.ply')

    # 估计法线
    print('  估计法线...')
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)

    # ── Poisson 表面重建 ──
    print(f'  Poisson 重建 (depth={POISSON_DEPTH})...')
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    print(f'  Poisson 完成 ({time.time()-t0:.0f}s), '
          f'{len(mesh.vertices)} 顶点')

    # 去除低密度区域（噪声）
    if len(densities) > 0:
        d_arr = np.asarray(densities)
        threshold = np.percentile(d_arr, 5)
        vertices_remove = d_arr < threshold
        mesh.remove_vertices_by_mask(vertices_remove)
        print(f'  去低密度后: {len(mesh.vertices)} 顶点, '
              f'{len(mesh.triangles)} 面')

    return mesh


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 4: 缩放 + 导出                                    ║
# ╚══════════════════════════════════════════════════════════╝

def do_scale_export(mesh, chair_dims, name):
    """trimesh 分轴缩放 + 导出 OBJ/PLY."""
    import trimesh

    print('=' * 60)
    print('  Step 4/4  缩放 + 导出')
    print('=' * 60)

    ch = chair_dims.get('height_m', 0.09)
    cw = chair_dims.get('width_m', 0.06)
    cd = chair_dims.get('depth_m', 0.05)

    verts = np.asarray(mesh.vertices)
    cur_w = verts[:, 0].max() - verts[:, 0].min()
    cur_d = verts[:, 1].max() - verts[:, 1].min()
    cur_h = verts[:, 2].max() - verts[:, 2].min()
    print(f'  重建尺寸: {cur_w*100:.1f}×{cur_d*100:.1f}×{cur_h*100:.1f}cm')
    print(f'  测量尺寸: {cw*100:.1f}×{cd*100:.1f}×{ch*100:.1f}cm')

    sx = cw / cur_w if cur_w > 0.0001 else 1.0
    sy = cd / cur_d if cur_d > 0.0001 else 1.0
    sz = ch / cur_h if cur_h > 0.0001 else 1.0
    print(f'  缩放: ×{sx:.3f} ×{sy:.3f} ×{sz:.3f}')

    verts[:, 0] *= sx
    verts[:, 1] *= sy
    verts[:, 2] *= sz
    verts[:, 2] -= verts[:, 2].min()

    faces = np.asarray(mesh.triangles)
    scaled = trimesh.Trimesh(vertices=verts, faces=faces)

    (OUT / 'models').mkdir(exist_ok=True)
    obj_path = OUT / 'models' / f'{name}_v14.obj'
    ply_path = OUT / 'models' / f'{name}_v14.ply'
    scaled.export(str(obj_path))
    scaled.export(str(ply_path))

    vs = scaled.vertices
    print(f'  最终: {vs[:,0].max()-vs[:,0].min():.3f}×'
          f'{vs[:,1].max()-vs[:,1].min():.3f}×'
          f'{vs[:,2].max()-vs[:,2].min():.3f}m, '
          f'{len(vs)} 顶点')
    print(f'  已导出: {obj_path}')
    print(f'  已导出: {ply_path}')
    return scaled


# ╔══════════════════════════════════════════════════════════╗
# ║  Main                                                    ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    global OUT, POISSON_DEPTH, VOXEL_SIZE

    parser = argparse.ArgumentParser(
        description='V14 家具重建 — 纯深度点云融合 (无 AI 生成)')
    parser.add_argument('--name', default='chair', help='家具名称 (默认: chair)')
    parser.add_argument('--skip-capture', action='store_true', help='跳过采集')
    parser.add_argument('--step-deg', type=int, default=STEP_DEG)
    parser.add_argument('--turntable-port', default='/dev/ttyUSB0')
    parser.add_argument('--poisson-depth', type=int, default=POISSON_DEPTH,
                        help=f'Poisson 重建深度 (默认: {POISSON_DEPTH})')
    parser.add_argument('--voxel', type=float, default=VOXEL_SIZE,
                        help=f'体素下采样大小 米 (默认: {VOXEL_SIZE})')
    args = parser.parse_args()

    OUT = BASE / f'output/v14/{args.name}'
    OUT.mkdir(parents=True, exist_ok=True)
    POISSON_DEPTH = args.poisson_depth
    VOXEL_SIZE = args.voxel

    step_deg = args.step_deg
    n_frames = 360 // step_deg

    print('=' * 60)
    print(f'  V14 家具重建 [{args.name}]')
    print(f'  深度点云融合 + Poisson 重建 | 输出: {OUT}')
    print(f'  {step_deg}deg × {n_frames} 帧')
    print('=' * 60)

    # ── 标定 ──
    calib_path = OUT / 'calibrate.json'
    ts_calib_path = BASE / 'output/calibrate.json'

    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        if 'fx_full' in calib:
            print(f'已有 V14 标定: {calib_path}')
            if not args.skip_capture:
                resp = input('使用现有标定? [Y/n]: ').strip().lower()
                if resp == 'n':
                    calib_path.unlink()
                    calib = None

    if calib is None:
        if ts_calib_path.exists():
            resp = input(f'已有转台标定 {ts_calib_path}，是否重新标定? [y/N]: ').strip().lower()
            if resp == 'y':
                ts_calib_path.unlink()
            if not ts_calib_path.exists():
                import subprocess
                subprocess.run([sys.executable, str(BASE / 'turntable_set_v2.py')])
                if not ts_calib_path.exists():
                    print('标定未保存，退出')
                    sys.exit(1)
        with open(ts_calib_path) as f:
            ts_calib = json.load(f)
        calib = {
            'R': ts_calib['R'], 't': ts_calib['t'],
            'radius_m': ts_calib['radius_m'],
            'cx': ts_calib.get('cx', 0.0), 'cy': ts_calib.get('cy', 0.0),
            'step_deg': STEP_DEG, 'n_frames': N_FRAMES,
        }

    # ── 采集 ──
    if not args.skip_capture:
        resp = input('\n开始采集? [Y/n]: ').strip().lower()
        if resp != 'n':
            do_capture(calib)
        else:
            print('跳过采集')
    else:
        print('跳过采集 (--skip-capture)')

    # ── 抠图 + 测量 ──
    masks_dict, chair_dims = do_yolo_and_measure(calib)
    if masks_dict is None:
        print('抠图失败，退出')
        sys.exit(1)

    if chair_dims is None:
        print('警告: 自动测量失败，使用默认值 9×6×5cm')
        chair_dims = {'height_m': 0.09, 'width_m': 0.06, 'depth_m': 0.05}

    # ── 深度点云融合重建 ──
    mesh = do_depth_reconstruction(masks_dict, calib)
    if mesh is None:
        print('重建失败，退出')
        sys.exit(1)

    # ── 缩放 + 导出 ──
    final = do_scale_export(mesh, chair_dims, args.name)

    # ── 日志 ──
    log = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'V14_DEPTH_FUSION',
        'n_frames': N_FRAMES,
        'poisson_depth': POISSON_DEPTH,
        'voxel_size': VOXEL_SIZE,
        'chair_dims': chair_dims,
        'output_dir': str(OUT),
    }
    with open(OUT / 'log_v14.json', 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\n{"=" * 60}')
    print(f'  V14 完成!')
    print(f'  方法: 纯深度点云融合 + Poisson 重建 (无 AI 生成)')
    print(f'  椅子尺寸: {chair_dims["height_m"]*100:.1f}×'
          f'{chair_dims["width_m"]*100:.1f}×{chair_dims["depth_m"]*100:.1f}cm (高×宽×深)')
    print(f'  输出目录: {OUT}')
    print(f'  {"=" * 60}')


if __name__ == '__main__':
    main()
