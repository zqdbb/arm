#!/usr/bin/env python3
"""转台标定: 采集 → 平面拟合 → 点圆心 → 点边缘 → calibrate.json."""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import numpy as np
import cv2

CLICK_WIN = 'Click'
click_state = {'pt': None, 'confirmed': False, 'step': 'center'}  # center | edge

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_state['pt'] = (x, y)
        click_state['confirmed'] = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        if click_state['pt'] is not None:
            click_state['confirmed'] = True

cv2.namedWindow(CLICK_WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(CLICK_WIN, cv2.WND_PROP_TOPMOST, 1)
cv2.setMouseCallback(CLICK_WIN, on_click)
print('[init] 交互窗口已就绪')

import json
import sys
import time
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


def fit_plane_and_rotation(pts):
    """RANSAC 平面拟合 → 旋转矩阵 R, plane_d."""
    if len(pts) < 50:
        return None
    z = pts[:, 2]
    z_min = float(np.percentile(z, 3))
    z_plate_max = z_min + 0.030
    plate_pts = pts[z < z_plate_max]
    if len(plate_pts) < 50:
        return None
    print(f'  转台面点: {len(plate_pts):,} (Z=[{z_min:.3f}, {z_plate_max:.3f}])')

    tpcd = o3d.geometry.PointCloud()
    tpcd.points = o3d.utility.Vector3dVector(plate_pts)
    plane_model, inliers = tpcd.segment_plane(distance_threshold=0.006, ransac_n=3, num_iterations=2000)
    a, b, c, d_plate = plane_model
    n = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
    if n[2] < 0:
        n = -n
        d_plate = -d_plate
    print(f'  转台面法线: n=[{n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f}] d={d_plate:.4f}')

    n_world = np.array([0., 0., 1.])
    cos_theta = np.dot(n, n_world)
    if cos_theta > 0.9999:
        R = np.eye(3)
    else:
        k = np.cross(n, n_world)
        k = k / np.linalg.norm(k)
        sin_theta = np.linalg.norm(np.cross(n, n_world))
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
    return R, d_plate


def pixel_to_plane(px, py, R, plane_d, intr):
    """像素 → 射线-平面交点 (3D 相机坐标)."""
    n = R[2, :]
    Kx = (px - intr.ppx) / intr.fx
    Ky = (py - intr.ppy) / intr.fy
    denom = n[0] * Kx + n[1] * Ky + n[2]
    if abs(denom) < 1e-9:
        return None
    z_cam = -plane_d / denom
    if z_cam <= 0:
        return None
    return np.array([Kx * z_cam, Ky * z_cam, z_cam])


def make_verify_image(color_img, R, t, radius_m, center_px, edge_px, fx, fy, ppx, ppy):
    h, w = color_img.shape[:2]
    img = color_img.copy()

    center_cam = -R.T @ t
    n_pts = 72
    theta = np.linspace(0, 2 * np.pi, n_pts)
    circle_w = np.column_stack([radius_m * np.cos(theta), radius_m * np.sin(theta), np.zeros(n_pts)])
    circle_c = (R.T @ circle_w.T).T + center_cam
    front = circle_c[:, 2] > 0.01
    if front.sum() >= 6:
        u = (fx * circle_c[front, 0] / circle_c[front, 2] + ppx)
        v = (fy * circle_c[front, 1] / circle_c[front, 2] + ppy)
        pts_uv = np.column_stack([u, v]).astype(np.int32)
        for i in range(len(pts_uv)):
            j = (i + 1) % len(pts_uv)
            if np.sqrt((pts_uv[i][0]-pts_uv[j][0])**2 + (pts_uv[i][1]-pts_uv[j][1])**2) < 150:
                cv2.line(img, tuple(pts_uv[i]), tuple(pts_uv[j]), (0, 255, 0), 2)

    # 画圆心和边缘点
    cv2.drawMarker(img, center_px, (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
    cv2.drawMarker(img, edge_px, (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.line(img, center_px, edge_px, (255, 255, 0), 2)

    if center_cam[2] > 0.01:
        cu = int(fx * center_cam[0] / center_cam[2] + ppx)
        cv_ = int(fy * center_cam[1] / center_cam[2] + ppy)
        cv2.drawMarker(img, (cu, cv_), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)

    for i, line in enumerate([
        f'Radius: {radius_m*100:.1f} cm',
        f'Center cam: [{center_cam[0]:.3f} {center_cam[1]:.3f} {center_cam[2]:.3f}]',
        f't: [{t[0]:.4f} {t[1]:.4f} {t[2]:.4f}]',
    ]):
        cv2.putText(img, line, (12, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return img


def click_step(best_color, prompt_text):
    """等待用户点击并确认, 返回 (px, py)."""
    print(f'  {prompt_text}')
    click_state['pt'] = None
    click_state['confirmed'] = False
    while not click_state['confirmed']:
        disp = best_color.copy()
        cv2.putText(disp, prompt_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(disp, 'LEFT=select  RIGHT=confirm  ESC=cancel', (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if click_state['pt']:
            cv2.drawMarker(disp, click_state['pt'], (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.imshow(CLICK_WIN, disp)
        if cv2.waitKey(30) & 0xFF == 27:
            return None
    return click_state['pt']


def main():
    import pyrealsense2 as rs

    print('=' * 55)
    print('  转台标定: SPACE=采集 → 点圆心 → 点边缘 → S=保存')
    print('=' * 55)

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print('未检测到设备!')
        return
    dev = devices[0]
    print(f'设备: {dev.get_info(rs.camera_info.name)}')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, DEPTH_FPS)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()
    colorizer = rs.colorizer()

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_stream.get_intrinsics()
    print(f'内参: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}')

    for _ in range(30):
        pipeline.wait_for_frames()
        time.sleep(0.05)

    best_color, R, t, radius_m = None, None, None, None
    center_px, edge_px = None, None
    print('\n按 SPACE 开始标定...')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_img = np.asanyarray(aligned.get_color_frame().get_data())
            depth_colored = np.asanyarray(colorizer.colorize(aligned.get_depth_frame()).get_data())
            h, w = color_img.shape[:2]
            preview = np.hstack([color_img, cv2.resize(depth_colored, (w, h))])
            cv2.putText(preview, 'RGB', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, 'Depth', (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            status = 'S=保存 R=重标' if R is not None else 'SPACE=标定 Q=退出'
            cv2.putText(preview, status, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if R is not None else (255, 255, 255), 2)
            cv2.imshow('Preview', preview)
            key = cv2.waitKey(10) & 0xFF

            if key == ord(' '):
                # ---- 采集 ----
                print('\n[标定] 采集最佳帧...')
                best_vertices, best_color, max_valid = None, None, 0
                for _ in range(CALIB_FRAMES):
                    f = pipeline.wait_for_frames()
                    a = align.process(f)
                    d, c = a.get_depth_frame(), a.get_color_frame()
                    if not d or not c:
                        continue
                    pc.map_to(a)
                    v = np.asanyarray(pc.calculate(d).get_vertices()).view(np.float32).reshape(-1, 3)
                    nv = np.all(np.isfinite(v), axis=1).sum()
                    if nv > max_valid:
                        max_valid = nv
                        best_vertices = v.copy()
                        best_color = np.asanyarray(c.get_data())
                    time.sleep(0.03)
                print(f'  有效点: {max_valid:,}')

                # ---- 平面拟合 ----
                valid = np.all(np.isfinite(best_vertices), axis=1) & (best_vertices[:, 2] > 0.05) & (best_vertices[:, 2] < 2.0)
                result = fit_plane_and_rotation(best_vertices[valid])
                if result is None:
                    print('  平面拟合失败!')
                    continue
                R, plane_d = result

                # ---- 第1步: 点圆心 ----
                pt = click_step(best_color, 'STEP 1/2: Click turntable CENTER')
                if pt is None:
                    continue
                center_px = pt
                center_cam_pt = pixel_to_plane(center_px[0], center_px[1], R, plane_d, intr)
                if center_cam_pt is None:
                    print('  圆心计算失败!')
                    continue
                center_cam = center_cam_pt
                print(f'  圆心像素: {center_px} → 3D: [{center_cam[0]:.3f} {center_cam[1]:.3f} {center_cam[2]:.3f}]')

                # ---- 第2步: 点边缘 ----
                pt = click_step(best_color, 'STEP 2/2: Click turntable EDGE')
                if pt is None:
                    continue
                edge_px = pt
                edge_cam = pixel_to_plane(edge_px[0], edge_px[1], R, plane_d, intr)
                if edge_cam is None:
                    print('  边缘点计算失败!')
                    continue
                print(f'  边缘像素: {edge_px} → 3D: [{edge_cam[0]:.3f} {edge_cam[1]:.3f} {edge_cam[2]:.3f}]')

                # ---- 半径 = 圆心到边缘的3D距离 ----
                radius_m = float(np.linalg.norm(edge_cam - center_cam))
                print(f'  计算半径: {radius_m*100:.1f} cm')

                t = -R @ center_cam

                # ---- 验证图 ----
                verify_img = make_verify_image(best_color, R, t, radius_m, center_px, edge_px,
                                               intr.fx, intr.fy, intr.ppx, intr.ppy)
                cv2.imshow('Verify', verify_img)
                print('  S=保存  R=重标')

            elif key == ord('s') and R is not None:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                calib_data = {
                    'R': R.tolist(), 't': t.tolist(),
                    'plate_z': 0.0, 'rotation_center': [0.0, 0.0], 'radius_m': float(radius_m),
                }
                with open(CALIB_FILE, 'w') as f:
                    json.dump(calib_data, f, indent=2, default=float)
                print(f'\n标定已保存 → {CALIB_FILE}')
                print(json.dumps(calib_data, indent=2, default=float))
                break

            elif key == ord('r'):
                R, t, radius_m = None, None, None
                center_px, edge_px = None, None
                print('重置, 按 SPACE 重新标定')

            elif key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
