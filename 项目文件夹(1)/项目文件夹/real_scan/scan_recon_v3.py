#!/usr/bin/env python3
"""
V3 椅子重建: 逐像素深度差分 + TSDF 体素融合
核心改进: 每像素计算转台面期望深度 → 深度差分离椅子 → TSDF融合 → Marching Cubes

使用: python3 scan_recon_v3.py
"""

import cv2, json, time, os
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d

os.environ['QT_QPA_PLATFORM'] = 'xcb'

# ═══════════════════ 参数 ═══════════════════
BASE = Path(__file__).parent
OUT = BASE / 'output/v3'
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150
MAX_DEPTH = 0.70

# 深度差分参数
DEPTH_DIFF_THRESH = 0.005   # 比转台面近 5mm → 椅子

# TSDF 参数
VOXEL_SIZE = 0.0015          # 1.5mm 体素
SDF_TRUNC = 0.005            # 5mm 截断距离
VOL_X = 0.05                 # X ±5cm (椅子 2cm + 余量)
VOL_Y = 0.05                 # Y ±5cm
VOL_Z_MIN = -0.01            # 转台面下 1cm
VOL_Z_MAX = 0.105            # 椅子上 1.5cm (椅子 8.8cm)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'color').mkdir(exist_ok=True)
(OUT / 'depth').mkdir(exist_ok=True)
(OUT / 'depth_color').mkdir(exist_ok=True)
(OUT / 'debug').mkdir(exist_ok=True)


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
    print(f'  采集: {STEP_DEG}° × {N_FRAMES} 帧')
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


# ═══════════════════ TSDF 融合 ═══════════════════
def do_fusion(calib):
    print('\n' + '=' * 50)
    print('  V3 TSDF 深度融合')
    print('=' * 50)

    R = np.array(calib['R']); t = np.array(calib['t'])
    fx, fy, ppx, ppy = calib['fx'], calib['fy'], calib['ppx'], calib['ppy']
    step = calib.get('step_deg', STEP_DEG)

    color_files = sorted((OUT / 'color').glob('*.jpg'))
    depth_files = sorted((OUT / 'depth').glob('*.png'))
    n_frames = min(N_FRAMES, len(color_files), len(depth_files))

    # 预计算像素射线方向
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                   np.arange(H, dtype=np.float32))
    dx = (u_grid - ppx) / fx
    dy = (v_grid - ppy) / fy
    denom_plane = R[2, 0] * dx + R[2, 1] * dy + R[2, 2]

    # 诊断: 采样几个像素的深度差
    print(f'\n--- 诊断: 深度差分采样 (第一帧) ---')
    depth0 = cv2.imread(str(depth_files[0]), -1).astype(np.float32) / 1000.0
    expected_0 = -t[2] / np.maximum(denom_plane, 0.001)
    diff_0 = expected_0 - depth0
    valid0 = (depth0 > 0.01) & (depth0 < MAX_DEPTH)
    chair0 = (diff_0 > DEPTH_DIFF_THRESH) & valid0
    print(f'  期望深度: [{expected_0[valid0].min():.3f}, {expected_0[valid0].max():.3f}]')
    print(f'  实际深度: [{depth0[valid0].min():.3f}, {depth0[valid0].max():.3f}]')
    print(f'  深度差: [{diff_0[valid0].min():.3f}, {diff_0[valid0].max():.3f}]')
    print(f'  椅子像素: {chair0.sum()} / {valid0.sum()} ({chair0.sum()/max(valid0.sum(),1)*100:.1f}%)')
    if chair0.sum() > 0:
        chair_depths = depth0[chair0]
        print(f'  椅子深度: [{chair_depths.min():.3f}, {chair_depths.max():.3f}]')
        # 保存椅子mask样本
        mask_viz = (chair0.astype(np.uint8) * 255)
        cv2.imwrite(str(OUT / 'debug/chair_mask_sample.png'), mask_viz)
        cv2.imwrite(str(OUT / 'debug/chair_diff_sample.jpg'),
                    cv2.applyColorMap(np.clip(diff_0 * 1000, 0, 50).astype(np.uint8) * 5, cv2.COLORMAP_JET))

    # TSDF volume
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, ppx, ppy)

    print(f'\nTSDF 融合: {n_frames} 帧  体素={VOXEL_SIZE*1000:.1f}mm  截断={SDF_TRUNC*1000:.0f}mm')
    t0 = time.time()
    integrated = 0

    for i in range(n_frames):
        img = cv2.imread(str(color_files[i]))
        depth_raw = cv2.imread(str(depth_files[i]), -1).astype(np.float32) / 1000.0

        # 逐像素深度差分: 转台面期望深度 - 实际深度
        diff = expected_0 - depth_raw  # 注意: 所有帧用同一个 denom (相机不动)
        valid = (depth_raw > 0.01) & (depth_raw < MAX_DEPTH)
        chair_mask = (diff > DEPTH_DIFF_THRESH) & valid

        if chair_mask.sum() < 200:
            continue

        # 中值滤波去飞点
        depth = cv2.medianBlur(depth_raw, 5)

        # 创建 mask 后的深度图 (非椅子区域=0, 不参与融合)
        depth_masked = depth.copy()
        depth_masked[~chair_mask] = 0.0

        # 椅子区域 RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth_o3d = o3d.geometry.Image(depth_masked)
        rgb_o3d = o3d.geometry.Image(rgb)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=MAX_DEPTH,
            convert_rgb_to_intensity=False)

        # 相机外参
        # p_obj → world: p_w = R_z(+θ) @ p_obj
        # p_w → camera: p_c = R_calib^T @ p_w - R_calib^T @ t
        # 合并: p_c = R_calib^T @ R_z(+θ) @ p_obj - R_calib^T @ t
        theta = np.radians(i * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_obj2cam = R.T @ R_z
        t_obj2cam = -R.T @ t

        extr = np.eye(4)
        extr[:3, :3] = R_obj2cam
        extr[:3, 3] = t_obj2cam

        vol.integrate(rgbd, intrinsic, extr)
        integrated += 1

        if (i + 1) % 18 == 0:
            print(f'  [{i+1}/{n_frames}] {time.time()-t0:.0f}s')

    print(f'融合完成: {integrated}/{n_frames} 帧  {time.time()-t0:.0f}s')

    if integrated < 6:
        print('融合帧数太少! 检查深度差阈值或数据')
        return

    # 提取 mesh (在 object 坐标系中: XY=转台面, Z=↑朝相机)
    print('\n提取 mesh (Marching Cubes)...')
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(f'  提取: {len(verts):,} 顶点  {len(faces):,} 面')
    if len(verts) > 0:
        for i, a in enumerate('XYZ'):
            print(f'  {a}: [{verts[:,i].min():.4f}, {verts[:,i].max():.4f}] '
                  f'span={np.ptp(verts[:,i])*100:.1f}cm')

    # 移除孤立面片
    if len(faces) > 0:
        labels, counts, _ = mesh.cluster_connected_triangles()
        if len(counts) > 1:
            largest = np.argmax(counts)
            to_remove = np.where(labels != largest)[0]
            mesh.remove_triangles_by_index(to_remove)
        mesh.remove_unreferenced_vertices()

    # 圆柱裁切: 半径 4.5cm (椅子 2cm + 余量)
    verts = np.asarray(mesh.vertices)
    dist_xy = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    # Z > -0.01: 去掉TSDF截断区以下的噪声
    keep = (dist_xy <= 0.045) & (verts[:, 2] >= -0.005)
    if keep.sum() > 0:
        indices = np.where(keep)[0]
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(verts[indices])
        old_to_new = np.full(len(verts), -1, dtype=int)
        old_to_new[indices] = np.arange(len(indices))
        faces_arr = np.asarray(mesh.triangles)
        face_mask = (old_to_new[faces_arr[:, 0]] >= 0) & \
                    (old_to_new[faces_arr[:, 1]] >= 0) & \
                    (old_to_new[faces_arr[:, 2]] >= 0)
        new_faces = np.column_stack([
            old_to_new[faces_arr[face_mask, 0]],
            old_to_new[faces_arr[face_mask, 1]],
            old_to_new[faces_arr[face_mask, 2]],
        ])
        new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
        new_mesh.compute_vertex_normals()
        mesh = new_mesh

    # ═══════════════════ 方向修正 ═══════════════════
    # TSDF 输出: XY=转台面, Z=↑朝相机 (已经正确)
    # Blender: Z=up. 椅子底在 Z=0 (转台面), 椅子顶在 Z>0.
    # 需要确保输出时 chair 站的起来:
    #   - 不需要翻转 Z (它已经是正的)
    #   - 把转台面 (chair 底) 移到 Z=0 以上
    #   - 如果 chair 高度 < 0.005, 整体抬高
    verts = np.asarray(mesh.vertices)
    z_min_v = verts[:, 2].min()
    if z_min_v < 0:
        verts[:, 2] -= z_min_v  # 整体抬高使最低点=0
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # 保存
    o3d.io.write_triangle_mesh(str(OUT / 'tsdf.ply'), mesh)
    o3d.io.write_triangle_mesh(str(OUT / 'tsdf.obj'), mesh)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(f'\n{"=" * 50}')
    print(f'  输出: {OUT}/tsdf.obj')
    print(f'  顶点: {len(verts):,}  面: {len(faces):,}')
    if len(verts) > 0:
        for i, a in enumerate('XYZ'):
            print(f'  {a}: [{verts[:,i].min():.4f}, {verts[:,i].max():.4f}] '
                  f'span={np.ptp(verts[:,i])*100:.2f}cm')

    # 高度分层统计
    if len(verts) > 0:
        print(f'  分层:')
        for lo, hi, label in [(0, 0.015, '转盘余量'), (0.015, 0.025, '椅腿底'),
                               (0.025, 0.045, '椅腿+座'), (0.045, 0.065, '座面'),
                               (0.065, 0.09, '靠背'), (0.09, 0.5, '椅顶')]:
            n = ((verts[:, 2] >= lo) & (verts[:, 2] < hi)).sum()
            print(f'    Z[{lo:.3f},{hi:.3f}) {label}: {n:6d}')
    print(f'  方向: Blender Z=up, 椅子直立在 XY 平面上')
    print(f'{"=" * 50}')

    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'method': 'TSDF+深度差分',
           'integrated_frames': integrated,
           'voxel_size_mm': VOXEL_SIZE * 1000,
           'sdf_trunc_mm': SDF_TRUNC * 1000,
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
