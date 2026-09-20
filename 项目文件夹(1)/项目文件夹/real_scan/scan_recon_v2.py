#!/usr/bin/env python3
"""
V2 椅子重建: 标定 → 采集 → 融合 → Poisson 网格
改进: Z过滤方向修正 + 多帧一致性 + 紧半径 + 诊断输出

使用: python3 scan_recon_v2.py
"""

import cv2, json, time, os
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
import open3d as o3d
from collections import defaultdict

os.environ['QT_QPA_PLATFORM'] = 'xcb'

# ═══════════════════ 参数 ═══════════════════
BASE = Path(__file__).parent
OUT = BASE / 'output/v2'
W, H = 640, 480
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG
N_AVG = 5
LASER_POWER = 150

# 椅子物理参数 (用户确认)
CHAIR_RADIUS = 0.035     # 椅子宽4cm → 半径略大一点
Z_MIN = 0.003            # 转台面上方3mm (跳过盘面噪声)
Z_MAX = 0.090            # 椅子顶部 (8.8cm)
MAX_DEPTH = 0.70         # 相机最远

# 融合参数
VOXEL_SIZE = 0.002       # 2mm
CONSISTENCY_MIN = 15     # 至少15/72帧出现
POISSON_DEPTH = 9
DENSITY_QUANTILE = 0.05

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

    # 预热
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
            ok = np.isfinite(verts).all(axis=1) & (verts[:, 2] > 0.05) & (verts[:, 2] < 2.0)
            print(f'  3D点: {ok.sum():,}')

            z_lo = float(np.percentile(verts[ok, 2], 3))
            plate = verts[ok][(verts[ok, 2] < z_lo + 0.03)]
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


# ═══════════════════ 融合 ═══════════════════
def do_fusion(calib):
    print('\n' + '=' * 50)
    print('  融合重建 (V2)')
    print('=' * 50)

    R = np.array(calib['R']); t = np.array(calib['t'])
    fx, fy, ppx, ppy = calib['fx'], calib['fy'], calib['ppx'], calib['ppy']
    step = calib.get('step_deg', STEP_DEG)

    color_files = sorted((OUT / 'color').glob('*.jpg'))
    depth_files = sorted((OUT / 'depth').glob('*.png'))
    n_frames = min(N_FRAMES, len(color_files), len(depth_files))

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32),
                                   np.arange(H, dtype=np.float32))

    # 诊断: Z直方图 (采样前3帧)
    print(f'\n--- 诊断: World Z 分布 (前3帧采样) ---')
    for idx in [0, n_frames//4, n_frames//2]:
        depth = cv2.imread(str(depth_files[idx]), -1).astype(np.float32) / 1000.0
        valid = (depth > 0.01) & (depth < MAX_DEPTH)
        if valid.sum() < 500:
            continue
        z_c = depth[valid]
        x_c = (u_grid[valid] - ppx) / fx * z_c
        y_c = (v_grid[valid] - ppy) / fy * z_c
        pts_cam = np.column_stack([x_c, y_c, z_c])
        pts_w = (R @ pts_cam.T).T + t
        zw = pts_w[:, 2]
        # 椅子区域
        chair_z = (zw >= Z_MIN) & (zw <= Z_MAX)
        print(f'  帧{idx} ({idx*step}°): Z [{zw.min():.4f}, {zw.max():.4f}]  '
              f'椅区Z∈[{Z_MIN},{Z_MAX}] = {chair_z.sum():,}点')

    # 逐帧提取 + 体素投票
    print(f'\n{n_frames} 帧 → world 3D + 反旋 + 体素投票')
    t0 = time.time()
    voxel_votes = defaultdict(int)
    voxel_colors = defaultdict(list)
    total_raw = 0

    for i in range(n_frames):
        depth = cv2.imread(str(depth_files[i]), -1).astype(np.float32) / 1000.0
        img = cv2.imread(str(color_files[i]))
        valid = (depth > 0.01) & (depth < MAX_DEPTH)
        if valid.sum() < 500:
            continue

        z_c = depth[valid]
        x_c = (u_grid[valid] - ppx) / fx * z_c
        y_c = (v_grid[valid] - ppy) / fy * z_c
        pts_cam = np.column_stack([x_c, y_c, z_c])
        pts_w = (R @ pts_cam.T).T + t

        # 反旋
        theta = np.radians(i * step)
        c, s = np.cos(theta), np.sin(theta)
        R_undo = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)
        pts_obj = (R_undo @ pts_w.T).T

        # 过滤: Z范围 + XY半径
        zw = pts_obj[:, 2]
        r = np.hypot(pts_obj[:, 0], pts_obj[:, 1])
        keep = (zw >= Z_MIN) & (zw <= Z_MAX) & (r < CHAIR_RADIUS)

        if keep.sum() < 10:
            continue

        pts_k = pts_obj[keep]
        clr_k = img[valid][keep].astype(np.float32) / 255.0

        # 量化到体素 (2mm)
        voxel_indices = np.floor(pts_k / VOXEL_SIZE).astype(np.int64)
        for vi, clr in zip(voxel_indices, clr_k):
            key = tuple(vi)
            voxel_votes[key] += 1
            if len(voxel_colors[key]) < 3:  # 只存前3个颜色样本
                voxel_colors[key].append(clr)

        total_raw += keep.sum()
        if (i + 1) % 18 == 0:
            print(f'  [{i+1}/{n_frames}] {total_raw/1e6:.1f}M raw | {len(voxel_votes)/1e3:.1f}K voxels')

    dt1 = time.time() - t0
    print(f'提取耗时: {dt1:.0f}s | 总原始点: {total_raw/1e6:.2f}M | 体素: {len(voxel_votes):,}')

    # 体素投票: 保留 >= CONSISTENCY_MIN 帧的体素
    consistent = {k: v for k, v in voxel_votes.items() if v >= CONSISTENCY_MIN}
    print(f'一致性过滤 (≥{CONSISTENCY_MIN}帧): {len(voxel_votes):,} → {len(consistent):,} 体素')

    if len(consistent) < 500:
        print('一致体素太少! 降低 CONSISTENCY_MIN 或检查数据')
        return

    # 体素 → 点云
    pts_list = []
    clr_list = []
    for key, _ in consistent.items():
        pts_list.append(np.array(key, dtype=np.float64) * VOXEL_SIZE + VOXEL_SIZE / 2)
        if key in voxel_colors and voxel_colors[key]:
            clr_list.append(np.mean(voxel_colors[key], axis=0))
        else:
            clr_list.append([0.5, 0.5, 0.5])

    pts_cons = np.array(pts_list)
    clr_cons = np.array(clr_list)

    # 诊断: Z分布
    print(f'\n--- 诊断: 一致性点云 Z 分布 ---')
    for lo, hi, label in [(Z_MIN, 0.02, '腿根'), (0.02, 0.04, '腿中-座'),
                           (0.04, 0.06, '座面-背'), (0.06, Z_MAX, '靠背')]:
        n = ((pts_cons[:, 2] >= lo) & (pts_cons[:, 2] < hi)).sum()
        print(f'  Z [{lo:.2f},{hi:.2f}) {label:8s}: {n:6d}')

    # Open3D 处理
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_cons)
    pcd.colors = o3d.utility.Vector3dVector(clr_cons)

    # 统计滤波
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)

    # 体素降采样 (万一投票后有密集区)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)

    # Z翻转 (使椅子朝上)
    pts_p = np.asarray(pcd.points)
    pts_p[:, 2] = -pts_p[:, 2]
    pcd.points = o3d.utility.Vector3dVector(pts_p)

    # 保存中间点云
    o3d.io.write_point_cloud(str(OUT / 'debug/fused_clean.ply'), pcd)
    print(f'\n融合点云: {len(pcd.points):,} 点 → {OUT / "debug/fused_clean.ply"}')

    # Poisson 重建
    print('Poisson 重建 (depth=9)...')
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_SIZE * 6, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., -2.]))

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH, width=0, scale=1.1, linear_fit=False)

    mesh.remove_vertices_by_mask(dens < np.quantile(dens, DENSITY_QUANTILE))
    mesh.remove_unreferenced_vertices()

    # 去 Z<0 碎面 (椅子在 Z>0)
    verts = np.asarray(mesh.vertices)
    kv = verts[:, 2] >= 0.001
    if kv.sum() > 0:
        idx = np.where(kv)[0]
        nm = o3d.geometry.TriangleMesh()
        nm.vertices = o3d.utility.Vector3dVector(verts[idx])
        o2n = np.full(len(verts), -1, dtype=int); o2n[idx] = np.arange(len(idx))
        faces = np.asarray(mesh.triangles)
        fm = (o2n[faces[:, 0]] >= 0) & (o2n[faces[:, 1]] >= 0) & (o2n[faces[:, 2]] >= 0)
        nf = np.column_stack([o2n[faces[fm, 0]], o2n[faces[fm, 1]], o2n[faces[fm, 2]]])
        nm.triangles = o3d.utility.Vector3iVector(nf)
        nm.compute_vertex_normals()
        mesh = nm

    o3d.io.write_triangle_mesh(str(OUT / 'poisson.ply'), mesh)
    o3d.io.write_triangle_mesh(str(OUT / 'poisson.obj'), mesh)

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(f'\n{"=" * 50}')
    print(f'  顶点: {len(verts):,}  面: {len(faces):,}')
    for i, a in enumerate('XYZ'):
        print(f'  {a}: [{verts[:,i].min():.3f}, {verts[:,i].max():.3f}] '
              f'span={np.ptp(verts[:,i])*100:.1f}cm')
    print(f'  输出: {OUT}/poisson.obj')
    print(f'{"=" * 50}')

    log = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
           'raw_pts': total_raw, 'voxels': len(voxel_votes),
           'consistent_voxels': len(consistent),
           'fused_pts': len(pcd.points),
           'vertices': len(verts), 'faces': len(faces),
           'size_cm': {a: round(np.ptp(verts[:, i]) * 100, 1) for i, a in enumerate('XYZ')}}
    with open(OUT / 'log_fusion.json', 'w') as f:
        json.dump(log, f, indent=2)


# ═══════════════════ Main ═══════════════════
if __name__ == '__main__':
    calib = do_calibrate()
    do_capture(calib)
    do_fusion(calib)
