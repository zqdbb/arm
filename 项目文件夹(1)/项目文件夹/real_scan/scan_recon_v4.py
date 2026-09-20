#!/usr/bin/env python3
"""
V4 椅子重建: 空背景减除 + Visual Hull 空间雕刻
核心: 先拍空转台背景 → 放椅子 → 每帧减背景得剪影 → 72视角雕刻

使用: python3 scan_recon_v4.py
"""

import cv2, json, time, os
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d

os.environ['QT_QPA_PLATFORM'] = 'xcb'

# ═══════════════════ 参数 ═══════════════════
BASE = Path(__file__).parent
OUT = BASE / 'output/v4'
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150

# Visual Hull 参数
VOXEL_SIZE = 0.0015       # 1.5mm 体素
VOL_X = 0.035             # X ±3.5cm (椅子 2.3cm + 余量)
VOL_Y = 0.035             # Y ±3.5cm (转台面方向)
VOL_Z_MIN = 0.002         # 转台面上 2mm (跳过盘面)
VOL_Z_MAX = 0.095         # 椅子顶部 (8.8cm + 余量)
VIS_CONSENSUS = 50        # 至少 50/72 视角一致才保留

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'color').mkdir(exist_ok=True)
(OUT / 'depth').mkdir(exist_ok=True)
(OUT / 'depth_color').mkdir(exist_ok=True)
(OUT / 'masks').mkdir(exist_ok=True)


# ═══════════════════ 标定 ═══════════════════
def do_calibrate():
    print('=' * 50)
    print('  标定: D435i 转台标定')
    print('=' * 50)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)

    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)
    dscale = ds.get_depth_scale()

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    print(f'fx={fx:.1f} fy={fy:.1f}  scale={dscale:.6f}')

    for i in range(60):
        try:
            pipe.wait_for_frames(timeout_ms=5000)
            break
        except:
            if i < 5:
                time.sleep(0.5)
    else:
        print('相机超时! 重插 USB 后重试')
        pipe.stop()
        exit(1)
    print('相机就绪')

    win = 'Calib'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    st = {'pt': None, 'ok': False}

    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            st['pt'] = (x, y); st['ok'] = False
        elif e == cv2.EVENT_RBUTTONDOWN and st['pt']:
            st['ok'] = True
    cv2.setMouseCallback(win, click)

    def pick_point(img, prompt, marker_color):
        st['pt'] = None; st['ok'] = False
        while not st['ok']:
            d = img.copy()
            cv2.putText(d, prompt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if st['pt']:
                cv2.drawMarker(d, st['pt'], marker_color, cv2.MARKER_CROSS, 30, 3)
            cv2.imshow(win, d)
            if cv2.waitKey(30) & 0xFF == 27:
                return None
        return st['pt']

    def plane_intersect(px, py, R, pd, fx, fy, ppx, ppy):
        n = R[2]; dx = (px - ppx) / fx; dy = (py - ppy) / fy
        denom = n[0] * dx + n[1] * dy + n[2]
        if abs(denom) < 1e-9: return None
        z = -pd / denom
        return np.array([dx * z, dy * z, z]) if z > 0 else None

    R_calib = t_calib = radius = None

    print('\nSPACE=标定  S=保存  Q=退出')
    while True:
        aligned = align.process(pipe.wait_for_frames())
        color = np.asanyarray(aligned.get_color_frame().get_data())
        dframe = aligned.get_depth_frame()

        cv2.putText(color, 'SPACE=calib  S=save  Q=quit', (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.imshow(win, color)
        k = cv2.waitKey(10) & 0xFF

        if k == ord(' '):
            pc = rs.pointcloud(); pc.map_to(aligned)
            verts = np.asanyarray(pc.calculate(dframe).get_vertices()).view(np.float32).reshape(-1, 3)
            ok_mask = np.isfinite(verts).all(axis=1) & (verts[:, 2] > 0.05) & (verts[:, 2] < 2.0)
            print(f'  3D点: {ok_mask.sum():,}')

            z_lo = float(np.percentile(verts[ok_mask, 2], 3))
            plate = verts[ok_mask][(verts[ok_mask, 2] < z_lo + 0.03)]
            if len(plate) < 100:
                print('  平面点太少!')
                continue
            tpcd = o3d.geometry.PointCloud()
            tpcd.points = o3d.utility.Vector3dVector(plate)
            pl, _ = tpcd.segment_plane(distance_threshold=0.006, ransac_n=3, num_iterations=2000)
            a, b, c, d = pl
            n = np.array([a, b, c]); n = n / np.linalg.norm(n)
            if n[2] < 0: n = -n; d = -d
            nw = np.array([0., 0., 1.])
            ct = np.dot(n, nw)
            if ct > 0.9999:
                R_plane = np.eye(3)
            else:
                kk = np.cross(n, nw); kk = kk / np.linalg.norm(kk)
                st_val = np.linalg.norm(np.cross(n, nw))
                K = np.array([[0, -kk[2], kk[1]], [kk[2], 0, -kk[0]], [-kk[1], kk[0], 0]])
                R_plane = np.eye(3) + st_val * K + (1 - ct) * (K @ K)

            cp = pick_point(color, '点转台圆心 (左键选 右键确认)', (0, 0, 255))
            if not cp: continue
            cc = plane_intersect(cp[0], cp[1], R_plane, d, fx, fy, ppx, ppy)
            if cc is None: print('圆心3D失败!'); continue

            ep = pick_point(color, '点转台边缘 (左键选 右键确认)', (255, 0, 0))
            if not ep: continue
            ec = plane_intersect(ep[0], ep[1], R_plane, d, fx, fy, ppx, ppy)
            if ec is None: print('边缘3D失败!'); continue

            radius = float(np.linalg.norm(ec - cc))
            t_calib = -R_plane @ cc
            R_calib = R_plane
            print(f'  半径: {radius*100:.1f}cm')

            vimg = color.copy()
            cam_w = -R_calib.T @ t_calib
            th = np.linspace(0, 2 * np.pi, 72)
            cw = np.column_stack([radius * np.cos(th), radius * np.sin(th), np.zeros(72)])
            c_cam = (R_plane.T @ cw.T).T + cam_w
            fvis = c_cam[:, 2] > 0.01
            if fvis.sum() >= 6:
                u = (fx * c_cam[fvis, 0] / c_cam[fvis, 2] + ppx).astype(int)
                v = (fy * c_cam[fvis, 1] / c_cam[fvis, 2] + ppy).astype(int)
                for i in range(len(u)):
                    j = (i + 1) % len(u)
                    if np.hypot(u[i] - u[j], v[i] - v[j]) < 150:
                        cv2.line(vimg, (u[i], v[i]), (u[j], v[j]), (0, 255, 0), 2)
            cv2.drawMarker(vimg, cp, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.drawMarker(vimg, ep, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.line(vimg, cp, ep, (255, 255, 0), 2)
            cv2.putText(vimg, f'R={radius*100:.1f}cm', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow('Verify', vimg)
            print('  S=保存  SPACE=重标')

        elif k == ord('s') and R_calib is not None:
            break
        elif k == ord('q') or k == 27:
            pipe.stop(); cv2.destroyAllWindows(); exit()

    pipe.stop()
    cv2.destroyAllWindows()

    calib = {'R': R_calib.tolist(), 't': t_calib.tolist(), 'radius_m': radius,
             'fx': fx, 'fy': fy, 'ppx': ppx, 'ppy': ppy,
             'width': W, 'height': H, 'depth_scale': float(dscale),
             'step_deg': STEP_DEG, 'n_frames': N_FRAMES, 'camera': 'D435i'}
    with open(OUT / 'calibrate.json', 'w') as f:
        json.dump(calib, f, indent=2)
    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'radius_m': radius}
    with open(OUT / 'log_calib.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f'\n标定保存: {OUT / "calibrate.json"}')
    return calib


# ═══════════════════ 采集 ═══════════════════
def do_capture(calib):
    print('\n' + '=' * 50)
    print(f'  采集: {STEP_DEG}° × {N_FRAMES} 帧 + 空背景')
    print('=' * 50)

    from turntable import TurntableController
    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)
    dscale = ds.get_depth_scale()
    pipe.wait_for_frames(timeout_ms=10000)

    for _ in range(30):
        try:
            pipe.wait_for_frames(timeout_ms=5000)
        except:
            pass

    # Step 1: 采集空转台背景
    print('\n--- 采集空转台背景 ---')
    print('请确保转台上没有椅子! 回车继续...')
    input()

    bg_sum = np.zeros((H, W, 3), dtype=np.float64)
    for i in range(N_AVG):
        aligned = align.process(pipe.wait_for_frames())
        bg_sum += np.asanyarray(aligned.get_color_frame().get_data()).astype(np.float64)
        print(f'  背景 {i+1}/{N_AVG}')
    background = (bg_sum / N_AVG).astype(np.uint8)
    cv2.imwrite(str(OUT / 'background.png'), background)
    print(f'背景保存: {OUT / "background.png"}')

    # Step 2: 放椅子
    print('\n--- 请放置椅子 ---')
    print('把椅子放在转台中心，回车开始采集...')
    input()

    frame_log = []
    t0 = time.time()
    angles = list(range(0, 360, STEP_DEG))

    for i, deg in enumerate(angles):
        tt.move_absolute(deg, speed=10000)
        tt.wait_stop(timeout=30)

        c_sum = np.zeros((H, W, 3), dtype=np.float64)
        d_sum = np.zeros((H, W), dtype=np.float64)
        for _ in range(N_AVG):
            aligned = align.process(pipe.wait_for_frames())
            c_sum += np.asanyarray(aligned.get_color_frame().get_data()).astype(np.float64)
            d_sum += np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float64)

        color = (c_sum / N_AVG).astype(np.uint8)
        depth_mm = (d_sum / N_AVG) * dscale * 1000.0
        depth_u16 = depth_mm.astype(np.uint16)

        cv2.imwrite(str(OUT / f'color/{i:06d}.jpg'), color)
        cv2.imwrite(str(OUT / f'depth/{i:06d}.png'), depth_u16)

        dv = np.clip(depth_mm, 70, 600)
        dv_n = ((dv - 70) / (600 - 70) * 255).astype(np.uint8)
        dv_c = cv2.applyColorMap(dv_n, cv2.COLORMAP_JET)
        dv_c[depth_u16 == 0] = 0
        cv2.imwrite(str(OUT / f'depth_color/{i:06d}.jpg'), dv_c)

        dvals = depth_mm[depth_mm > 0]
        nv = len(dvals)
        frame_log.append({'frame': i, 'deg': deg, 'valid': nv,
                          'median_mm': float(np.median(dvals)) if nv else 0})

        if (i + 1) % 12 == 0 or i == 0:
            el = time.time() - t0
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}°  valid:{nv//1000}k  {el:.0f}s')

    pipe.stop()
    el = time.time() - t0
    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'elapsed_s': round(el, 1), 'frames': frame_log}
    with open(OUT / 'log_capture.json', 'w') as f:
        json.dump(log, f, indent=2)
    print(f'采集完成 ({el:.0f}s)')


# ═══════════════════ Visual Hull 融合 ═══════════════════
def do_fusion(calib):
    print('\n' + '=' * 50)
    print('  V4 背景减除 + Visual Hull')
    print('=' * 50)

    R = np.array(calib['R']); t = np.array(calib['t'])
    fx, fy, ppx, ppy = calib['fx'], calib['fy'], calib['ppx'], calib['ppy']
    step = calib.get('step_deg', STEP_DEG)

    # 加载背景
    bg_path = OUT / 'background.png'
    if not bg_path.exists():
        print(f'错误: 背景图不存在 {bg_path}')
        return
    background = cv2.imread(str(bg_path))
    print(f'背景: {bg_path}')

    color_files = sorted((OUT / 'color').glob('*.jpg'))
    n_frames = min(N_FRAMES, len(color_files))

    # 形态学核
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Step 1: 逐帧背景减除提取椅子剪影
    print(f'\n--- 背景减除提取剪影 ({n_frames} 帧) ---')
    masks = []

    for i in range(n_frames):
        img = cv2.imread(str(color_files[i]))

        # 颜色差
        diff = cv2.absdiff(img, background)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # 高斯模糊降噪
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu 自动二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 检查 Otsu 阈值是否太低 (全图都被判为前景)
        fg_ratio = (binary > 0).sum() / binary.size
        if fg_ratio > 0.5:
            # Otsu 找的阈值太低，手动提高
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # 形态学清理
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # 只保留最大连通区
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest = np.argmax(areas) + 1
                binary = (labels == largest).astype(np.uint8) * 255

        if binary.sum() < 50:
            masks.append(np.zeros((H, W), dtype=np.uint8))
            continue

        masks.append(binary)

        # 保存样本
        if i < 5 or i % 18 == 0:
            cv2.imwrite(str(OUT / f'masks/{i:06d}_mask.png'), binary)
            overlay = img.copy()
            overlay[binary > 0] = overlay[binary > 0] // 2 + np.array([0, 0, 128], dtype=np.uint8) // 2
            cv2.imwrite(str(OUT / f'masks/{i:06d}_overlay.jpg'), overlay)

        if (i + 1) % 18 == 0:
            avg = np.mean([m.sum() for m in masks[-18:]]) / 255
            print(f'  [{i+1}/{n_frames}] 平均前景={avg:.0f}px')

    valid_masks = [m for m in masks if m.sum() > 500]
    print(f'有效剪影: {len(valid_masks)}/{len(masks)}  '
          f'平均前景={np.mean([m.sum() for m in valid_masks])/255:.0f}px')

    if len(valid_masks) < 20:
        print('有效剪影太少! 检查 masks/ 目录和背景图')
        return

    # Step 2: 构建体素网格
    nx = int(2 * VOL_X / VOXEL_SIZE) + 1
    ny = int(2 * VOL_Y / VOXEL_SIZE) + 1
    nz = int((VOL_Z_MAX - VOL_Z_MIN) / VOXEL_SIZE) + 1

    x = np.linspace(-VOL_X, VOL_X, nx, dtype=np.float32)
    y = np.linspace(-VOL_Y, VOL_Y, ny, dtype=np.float32)
    z = np.linspace(VOL_Z_MIN, VOL_Z_MAX, nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float64)
    n_voxels = len(voxels)
    print(f'\n体素: {nx}×{ny}×{nz} = {n_voxels/1e3:.1f}K  分辨率={VOXEL_SIZE*1000:.1f}mm')
    print(f'  X∈[-{VOL_X},{VOL_X}] Y∈[-{VOL_Y},{VOL_Y}] Z∈[{VOL_Z_MIN},{VOL_Z_MAX}]')

    # Step 3: Visual Hull 空间雕刻
    print(f'\n空间雕刻: {n_frames} 视角  一致性阈值≥{VIS_CONSENSUS}')
    t0 = time.time()

    # 存活体素计数
    vis_counts = np.zeros(n_voxels, dtype=np.int16)
    BATCH = 50000

    for i_view in range(n_frames):
        mask = masks[i_view]
        if mask.sum() < 100:
            continue

        theta = np.radians(i_view * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_view = R.T @ R_z
        t_view = -R.T @ t

        for start in range(0, n_voxels, BATCH):
            end = min(start + BATCH, n_voxels)
            batch = voxels[start:end]

            p_cam = (R_view @ batch.T).T + t_view
            z_cam = p_cam[:, 2]

            in_front = z_cam > 0.01
            if in_front.sum() < 10:
                continue

            idx = np.where(in_front)[0]
            u = (fx * p_cam[idx, 0] / z_cam[idx] + ppx).astype(int)
            v = (fy * p_cam[idx, 1] / z_cam[idx] + ppy).astype(int)
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)

            if in_img.sum() < 10:
                continue

            idx = idx[in_img]
            u, v = u[in_img], v[in_img]
            fg = mask[v, u] > 128
            if fg.sum() > 0:
                vis_counts[start + idx[fg]] += 1

        if (i_view + 1) % 18 == 0:
            n_survive = (vis_counts >= VIS_CONSENSUS).sum()
            print(f'  [{i_view+1}/{n_frames}] 存活={n_survive:,} '
                  f'({n_survive/n_voxels*100:.1f}%)  {time.time()-t0:.0f}s')

    dt = time.time() - t0
    survived = vis_counts >= VIS_CONSENSUS
    hull_pts = voxels[survived]
    print(f'雕刻完成: {dt:.0f}s  存活={survived.sum():,} '
          f'({survived.sum()/n_voxels*100:.1f}%)')

    if survived.sum() < 100:
        print('存活体素太少! 降低 VIS_CONSENSUS')
        return

    # Step 4: 方向修正 + 点云输出
    # 当前: X=左右 Y=前后 Z=↑(椅高)
    # 目标: X=左右 Y=↑(椅高) Z=前后  (匹配 reference chair_ai.obj)
    # 变换: (x,y,z) → (x, z, y)  即 R_x(-90°)
    pts_obj = hull_pts.copy()
    pts_out = np.zeros_like(pts_obj)
    pts_out[:, 0] = pts_obj[:, 0]      # X → X (左右不变)
    pts_out[:, 1] = pts_obj[:, 2]      # Z → Y (椅高变↑)
    pts_out[:, 2] = pts_obj[:, 1]      # Y → Z (前后变深度)

    # 平移使底部在 Z=0 (Blender 地面)
    pts_out[:, 2] -= pts_out[:, 2].min()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_out.astype(np.float64))
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    # 保存点云
    o3d.io.write_point_cloud(str(OUT / 'visual_hull.ply'), pcd)
    print(f'\n点云: {len(pcd.points):,} 点 → {OUT / "visual_hull.ply"}')

    # Poisson 表面重建
    if len(pcd.points) > 500:
        print('Poisson 表面重建...')
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 5, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.array([0., 2., 0.]))

        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.05))
        mesh.remove_unreferenced_vertices()

        # 只保留最大连通分量
        labels, counts, _ = mesh.cluster_connected_triangles()
        if len(counts) > 1:
            largest = np.argmax(counts)
            mesh.remove_triangles_by_index(np.where(labels != largest)[0])
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        o3d.io.write_triangle_mesh(str(OUT / 'visual_hull.ply'), mesh)
        o3d.io.write_triangle_mesh(str(OUT / 'visual_hull.obj'), mesh)

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        print(f'  顶点: {len(verts):,}  面: {len(faces):,}')
    else:
        verts = pts_out
        faces = np.array([])
        o3d.io.write_point_cloud(str(OUT / 'visual_hull.ply'), pcd)

    # 诊断
    print(f'\n{"=" * 50}')
    print(f'  输出: {OUT}/visual_hull.obj')
    print(f'  方向: Y=↑ (匹配 reference chair_ai.obj)')
    if len(verts) > 0:
        for i, a in enumerate('XYZ'):
            print(f'  {a}: [{verts[:,i].min():.3f}, {verts[:,i].max():.3f}] '
                  f'span={np.ptp(verts[:,i])*100:.1f}cm')
        y_span = np.ptp(verts[:, 1]) * 100
        xz = np.ptp(verts[:, [0, 2]]) * 100
        print(f'  预期: X≈4.6cm Y≈8.8cm(高) Z≈4.0cm')
        print(f'  实际: Y={y_span:.1f}cm(高) XZ={xz:.1f}cm')
    print(f'{"=" * 50}')

    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'method': '背景减除+Visual Hull',
           'n_frames': n_frames, 'valid_masks': len(valid_masks),
           'voxel_size_mm': VOXEL_SIZE * 1000,
           'vis_consensus': VIS_CONSENSUS,
           'hull_voxels': int(survived.sum()),
           'vertices': len(verts),
           'faces': len(faces) if len(faces) > 0 else 0,
           'size_cm': {a: round(np.ptp(verts[:, i]) * 100, 1) for i, a in enumerate('XYZ')}
           if len(verts) > 0 else {}}
    with open(OUT / 'log_fusion.json', 'w') as f:
        json.dump(log, f, indent=2)


# ═══════════════════ Main ═══════════════════
if __name__ == '__main__':
    calib = do_calibrate()
    do_capture(calib)
    do_fusion(calib)
