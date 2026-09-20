#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D405/D435i 双阶段圆盘标定 —— 自适应分辨率版（v7）
自动适配不同相机分辨率，无需手动调参。
"""

import cv2
import json
import numpy as np
import open3d as o3d
import os
import pyrealsense2 as rs
import time

class TwoPhaseLocator:
    STATE_SEARCH = 0
    STATE_TRACK = 1

    def __init__(self):
        self.state = self.STATE_SEARCH
        self.center = None
        self.ellipse = None
        self.HIST_SIZE = 10
        self.ell_history = []
        self.last_radius_3d = 0.10
        self.plane_model = None
        self._plane_refit_counter = 0
        self.alpha = 0.25

    def detect_center(self, gray):
        """V6 同款连通域分析：暗色像素 + 连通域 + 近中心 + 暗度评分"""
        h, w = gray.shape

        # 取最暗 20% 像素作为候选
        th = np.percentile(gray, 20)
        dark = (gray < max(th, 20)).astype(np.uint8) * 255

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if n_labels <= 1:
            return None

        cx_img, cy_img = w / 2, h / 2
        best_label = None
        best_score = -1

        for lb in range(1, n_labels):
            area = stats[lb, cv2.CC_STAT_AREA]
            # 面积过滤：太小忽略噪声，太大忽略背景
            if area < 10 or area > 20000:
                continue
            cx_b, cy_b = centroids[lb]
            dist = np.hypot(cx_b - cx_img, cy_b - cy_img)
            # 距离过滤：不超过图像半对角线的 49%
            if dist > min(w, h) * 0.49:
                continue

            mask_lb = (labels == lb)
            mean_brightness = gray[mask_lb].mean()
            # 综合评分：越暗越好 + 越近中心越好
            score = (255 - mean_brightness) * 0.6 + (1.0 - dist / (min(w, h) * 0.49)) * 0.4
            if score > best_score:
                best_score = score
                best_label = lb

        if best_label is not None:
            cx, cy = centroids[best_label]
            r = np.sqrt(stats[best_label, cv2.CC_STAT_AREA] / np.pi)
            return (float(cx), float(cy), float(r))
        return None

    def detect_ring_precise(self, color_bgr, depth_image, depth_scale, fx, fy, ppx, ppy):
        """RANSAC 平面 → 梯度搜索最优半径 → 3D 圆投影 → 椭圆"""
        h, w = depth_image.shape
        cx, cy = self.center

        # 1. RANSAC 平面拟合（缓存，每 5 帧刷新一次减少抖动）
        if self.plane_model is None or self._plane_refit_counter >= 5:
            stride = 2
            d_sub = depth_image[::stride, ::stride].astype(np.float64) * depth_scale
            valid = (d_sub > 0.01) & (d_sub < 0.8)
            if valid.sum() < 300:
                return None
            vv, uu = np.mgrid[0:h:stride, 0:w:stride]
            z_cam = d_sub[valid]
            x_cam = (uu[valid].astype(np.float64) - ppx) / fx * z_cam
            y_cam = (vv[valid].astype(np.float64) - ppy) / fy * z_cam
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.column_stack([x_cam, y_cam, z_cam]))
            plane_model, _ = pcd.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=300)
            self.plane_model = plane_model
            self._plane_refit_counter = 0
        else:
            self._plane_refit_counter += 1

        n = np.array(self.plane_model[:3])
        d_val = self.plane_model[3]
        if n[2] < 0:
            n = -n; d_val = -d_val

        # 2. 射线-平面求交 → 3D 中心
        ray = np.array([(cx - ppx) / fx, (cy - ppy) / fy, 1.0])
        ray /= np.linalg.norm(ray)
        denom = np.dot(n, ray)
        if abs(denom) < 1e-8:
            return None
        t_int = -d_val / denom
        if t_int <= 0:
            return None
        center_3d = ray * t_int

        # 3. 平面局部坐标系
        z_axis = n / np.linalg.norm(n)
        x_axis = np.cross(np.array([0., 1., 0.]), z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross(np.array([1., 0., 0.]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        R_plane = np.column_stack([x_axis, y_axis, z_axis])

        # 4. 梯度搜索最优 3D 半径
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        angles = np.linspace(0, 2 * np.pi, 72)
        best_r = self.last_radius_3d
        best_score = -1.0

        for r_3d in np.linspace(0.04, 0.16, 25):
            cl = np.column_stack([r_3d * np.cos(angles), r_3d * np.sin(angles), np.zeros(72)])
            cc = (R_plane @ cl.T).T + center_3d
            px_s = np.clip((cc[:, 0] * fx / cc[:, 2] + ppx).astype(int), 0, w - 1)
            py_s = np.clip((cc[:, 1] * fy / cc[:, 2] + ppy).astype(int), 0, h - 1)
            score = float(grad_mag[py_s, px_s].mean())
            if score > best_score:
                best_score = score
                best_r = r_3d

        self.last_radius_3d = best_r

        # 5. 最优半径 → 3D 圆 → 2D 投影 → 椭圆
        angles_full = np.linspace(0, 2 * np.pi, 180)
        cl = np.column_stack([best_r * np.cos(angles_full), best_r * np.sin(angles_full), np.zeros(180)])
        cc = (R_plane @ cl.T).T + center_3d
        px = cc[:, 0] * fx / cc[:, 2] + ppx
        py = cc[:, 1] * fy / cc[:, 2] + ppy
        valid_p = (cc[:, 2] > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pts = np.column_stack([px[valid_p], py[valid_p]])
        if len(pts) < 15:
            return None

        ellipse = cv2.fitEllipse(pts.astype(np.float32).reshape(-1, 1, 2))
        return ellipse

    def smooth_ellipse(self, ellipse):
        """只平滑轴长和角度，中心保持与红叉对齐"""
        if ellipse is None:
            return None
        if self.ellipse is None:
            self.ellipse = ellipse
            self.ell_history.append(ellipse)
            return ellipse

        prev = self.ellipse
        (cx1, cy1), (MA1, ma1), ang1 = prev
        (cx2, cy2), (MA2, ma2), ang2 = ellipse

        alpha = self.alpha
        MA = alpha * MA2 + (1 - alpha) * MA1
        ma = alpha * ma2 + (1 - alpha) * ma1
        ang1_rad = np.deg2rad(ang1)
        ang2_rad = np.deg2rad(ang2)
        diff = ang2_rad - ang1_rad
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        ang = np.rad2deg(ang1_rad + alpha * diff) % 180

        smoothed = ((cx2, cy2), (MA, ma), ang)
        self.ellipse = smoothed
        self.ell_history.append(smoothed)
        if len(self.ell_history) > self.HIST_SIZE:
            self.ell_history.pop(0)
        return smoothed

    def get_3d(self, depth_image, center, depth_scale, fx, fy, ppx, ppy):
        if center is None:
            return None
        cx, cy = center
        cx = np.clip(cx, 0, depth_image.shape[1] - 1)
        cy = np.clip(cy, 0, depth_image.shape[0] - 1)
        rgn = 5
        y1, y2 = max(0, int(cy - rgn)), min(depth_image.shape[0], int(cy + rgn) + 1)
        x1, x2 = max(0, int(cx - rgn)), min(depth_image.shape[1], int(cx + rgn) + 1)
        patch = depth_image[y1:y2, x1:x2]
        depths = patch[patch > 0]
        if len(depths) == 0:
            z_raw = depth_image[int(cy), int(cx)]
            if z_raw == 0:
                return None
            sample_count = 1
        else:
            z_med = np.median(depths)
            valid_d = depths[(depths > z_med * 0.9) & (depths < z_med * 1.1)]
            z_raw = float(np.mean(valid_d)) if len(valid_d) > 0 else float(z_med)
            sample_count = len(depths)
        z = z_raw * depth_scale
        X = (cx - ppx) * z / fx
        Y = (cy - ppy) * z / fy
        return {
            'position': (X, Y, z),
            'position_mm': (X * 1000, Y * 1000, z * 1000),
            'depth_raw': z_raw, 'sample_count': sample_count
        }

    def draw(self, image, center_det, pos_3d=None, fps=0.0):
        result = image.copy()
        cv2.putText(result, f"FPS: {fps:.1f}", (image.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.state == self.STATE_SEARCH:
            if center_det:
                cx, cy, r = center_det
                cv2.circle(result, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
                cv2.drawMarker(result, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.putText(result, f"Center: ({cx:.1f}, {cy:.1f})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(result, ">>> Press SPACE to lock <<<",
                           (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
            else:
                cv2.putText(result, "Searching center dot...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if self.ellipse:
                cv2.ellipse(result, self.ellipse, (255, 0, 0), 2)
                ecx, ecy = self.ellipse[0]
                cv2.drawMarker(result, (int(ecx), int(ecy)), (255, 0, 0),
                              cv2.MARKER_CROSS, 24, 2)

            if self.center:
                cv2.drawMarker(result, (int(self.center[0]), int(self.center[1])),
                              (0, 0, 255), cv2.MARKER_CROSS, 16, 2)

            if self.ellipse:
                (ecx, ecy), (MA, ma), ang = self.ellipse
                cv2.putText(result, f"Ellipse: ({ecx:.1f}, {ecy:.1f})  semi-axes=({MA/2:.1f}, {ma/2:.1f})px  ang={ang:.0f}deg",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(result, f"Radius: {self.last_radius_3d*100:.1f}cm",
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if pos_3d:
                x, y, z = pos_3d['position_mm']
                cv2.putText(result, f"GRIP(mm): X={x:+.2f} Y={y:+.2f} Z={z:.2f}",
                           (10, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(result, f"DepthSamples: {pos_3d['sample_count']}",
                           (10, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(result, "SPACE=re-search  R=reset  S=save calib  Q=quit",
                       (10, image.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)

        return result


# ---------- 相机初始化（保持不变） ----------
def _try_start_pipeline(pipeline, config, width, height, fps):
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        profile = pipeline.start(cfg)
        return profile, (width, height, fps)
    except RuntimeError:
        try:
            pipeline.stop()
        except Exception:
            pass
        return None, None

def setup_realsense():
    pipeline = rs.pipeline()
    candidates = [
        (1280, 720, 30), (1280, 720, 15),
        (848, 480, 30), (848, 480, 15),
        (640, 480, 30), (640, 480, 15),
        (640, 360, 30), (480, 270, 30), (424, 240, 30),
    ]
    profile = None
    for width, height, fps in candidates:
        profile, chosen = _try_start_pipeline(pipeline, rs.config(), width, height, fps)
        if profile is not None:
            print(f"[INFO] RealSense 启动成功: {width}x{height} @ {fps}fps")
            break
    if profile is None:
        print("[WARN] 常规配置全部失败，尝试硬件复位相机...")
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            try:
                devices[0].hardware_reset()
                time.sleep(2.0)
            except Exception as e:
                print(f"[WARN] 硬件复位失败: {e}")
        pipeline = rs.pipeline()
        for width, height, fps in candidates:
            profile, chosen = _try_start_pipeline(pipeline, rs.config(), width, height, fps)
            if profile is not None:
                print(f"[INFO] 硬件复位后启动成功: {width}x{height} @ {fps}fps")
                break
    if profile is None:
        raise RuntimeError(
            "\n========================================\n"
            "无法启动 RealSense，所有分辨率/帧率组合均失败。\n\n"
            "请检查 USB 连接、驱动和固件版本。\n"
            "========================================"
        )
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"Depth scale: {depth_scale} m")
    print(f"Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
          f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale, intrinsics


# ---------- 主函数 ----------
def main():
    pipeline, align, depth_scale, intrinsics = setup_realsense()
    locator = TwoPhaseLocator()

    print("\n" + "=" * 55)
    print("双阶段圆盘标定 —— 自适应分辨率版 v7（适配 D405/D435i）")
    print("=" * 55)
    print("SPACE - 锁定中心 / 重新搜索外圈")
    print("R     - 重置")
    print("S     - 保存标定 (R/t/内参/半径)")
    print("Q     - 退出")
    print("=" * 55)

    window_name = "Precise Disc Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    zoom_factor = 2.0  # 固定 2x 数码变焦
    zoom_x0, zoom_y0 = 0, 0  # 裁剪偏移（用于修正内参）

    frame_counter = 0
    fps = 0.0
    last_fps_time = time.time()
    detect_counter = 0
    DETECT_EVERY_N_FRAMES = 2

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 数码变焦：裁剪中心，不放大
            if zoom_factor > 1.0:
                h_full, w_full = color_image.shape[:2]
                crop_w = int(w_full / zoom_factor)
                crop_h = int(h_full / zoom_factor)
                zoom_x0 = (w_full - crop_w) // 2
                zoom_y0 = (h_full - crop_h) // 2
                color_image = color_image[zoom_y0:zoom_y0+crop_h, zoom_x0:zoom_x0+crop_w]
                depth_image = depth_image[zoom_y0:zoom_y0+crop_h, zoom_x0:zoom_x0+crop_w]
            else:
                zoom_x0, zoom_y0 = 0, 0

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            h, w = color_image.shape[:2]

            # 有效内参（考虑裁剪偏移）
            fx_c = intrinsics.fx
            fy_c = intrinsics.fy
            ppx_c = intrinsics.ppx - zoom_x0
            ppy_c = intrinsics.ppy - zoom_y0

            center_det = None

            if locator.state == locator.STATE_SEARCH:
                center_det = locator.detect_center(gray)
            else:
                detect_counter += 1
                if detect_counter % DETECT_EVERY_N_FRAMES == 0:
                    raw = locator.detect_ring_precise(color_image, depth_image, depth_scale,
                                                       fx_c, fy_c, ppx_c, ppy_c)
                    if raw is not None:
                        smoothed = locator.smooth_ellipse(raw)
                        locator.ellipse = smoothed
                    else:
                        print("[WARN] 边缘检测失败")
                center_det = (locator.center[0], locator.center[1], 5)

            pos_3d = None
            if locator.center:
                pos_3d = locator.get_3d(depth_image, locator.center, depth_scale,
                                        fx_c, fy_c, ppx_c, ppy_c)

            frame_counter += 1
            now = time.time()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                last_fps_time = now

            display = locator.draw(color_image, center_det, pos_3d, fps=fps)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if locator.state == locator.STATE_SEARCH and center_det:
                    locator.center = (center_det[0], center_det[1])
                    locator.state = locator.STATE_TRACK
                    locator.plane_model = None
                    locator._plane_refit_counter = 0
                    print(f"中心锁定: ({locator.center[0]:.1f}, {locator.center[1]:.1f})")
                    raw = locator.detect_ring_precise(color_image, depth_image, depth_scale,
                                                       fx_c, fy_c, ppx_c, ppy_c)
                    if raw is not None:
                        smoothed = locator.smooth_ellipse(raw)
                        locator.ellipse = smoothed
                        print(f"外圈找到: 半径={locator.last_radius_3d*100:.1f}cm")
                else:
                    locator.ell_history.clear()
                    locator.plane_model = None
                    locator._plane_refit_counter = 0
                    raw = locator.detect_ring_precise(color_image, depth_image, depth_scale,
                                                       fx_c, fy_c, ppx_c, ppy_c)
                    if raw is not None:
                        smoothed = locator.smooth_ellipse(raw)
                        locator.ellipse = smoothed
                    print(f"外圈重新搜索: 半径={locator.last_radius_3d*100:.1f}cm")
            elif key == ord('r'):
                locator.state = locator.STATE_SEARCH
                locator.center = None
                locator.ellipse = None
                locator.ell_history.clear()
                locator.last_radius_3d = 0.10
                locator.plane_model = None
                locator._plane_refit_counter = 0
                detect_counter = 0
                print("已重置")
            elif key == ord('s') or key == ord('S'):
                if locator.ellipse is None:
                    print("请先锁定圆盘（按空格）再保存标定")
                else:
                    depth_m = depth_image.astype(np.float64) * depth_scale
                    valid = (depth_m > 0.01) & (depth_m < 0.8)

                    if valid.sum() < 500:
                        print(f"有效深度点不足 ({valid.sum()})")
                    else:
                        vv, uu = np.mgrid[0:h, 0:w]
                        z_cam = depth_m[valid]
                        x_cam = (uu[valid].astype(np.float64) - ppx_c) / fx_c * z_cam
                        y_cam = (vv[valid].astype(np.float64) - ppy_c) / fy_c * z_cam

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.column_stack([x_cam, y_cam, z_cam]))
                        plane_model, inliers = pcd.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=500)

                        n = np.array(plane_model[:3])
                        d = plane_model[3]
                        if n[2] < 0:
                            n = -n
                            d = -d

                        z_axis = n / np.linalg.norm(n)
                        x_axis = np.cross(np.array([0., 1., 0.]), z_axis)
                        if np.linalg.norm(x_axis) < 1e-6:
                            x_axis = np.cross(np.array([1., 0., 0.]), z_axis)
                        x_axis /= np.linalg.norm(x_axis)
                        y_axis = np.cross(z_axis, x_axis)
                        R_plane = np.column_stack([x_axis, y_axis, z_axis])
                        R_calib = R_plane.T.astype(np.float64)
                        t_calib = (-d * n).astype(np.float64)

                        # 中心点 3D 坐标（射线-平面求交）
                        cx_w, cy_w = 0.0, 0.0
                        if locator.center is not None:
                            cx_px, cy_px = locator.center
                            ray = np.array([(cx_px - ppx_c) / fx_c, (cy_px - ppy_c) / fy_c, 1.0])
                            ray /= np.linalg.norm(ray)
                            denom = np.dot(n, ray)
                            if abs(denom) > 1e-8:
                                t_int = -d / denom
                                if t_int > 0:
                                    cp = ray * t_int
                                    cpl = R_plane @ cp
                                    cx_w, cy_w = float(cpl[0]), float(cpl[1])

                        # 圆盘半径（来自梯度搜索）
                        radius_m = locator.last_radius_3d

                        calib = {
                            'R': R_calib.tolist(),
                            't': t_calib.tolist(),
                            'radius_m': radius_m,
                            'cx': cx_w, 'cy': cy_w,
                            'fx': fx_c, 'fy': fy_c,
                            'ppx': ppx_c, 'ppy': ppy_c,
                            'width': w, 'height': h,
                            'depth_scale': depth_scale,
                            'digital_zoom': zoom_factor,
                            'zoom_x0': zoom_x0, 'zoom_y0': zoom_y0,
                        }

                        out_dir = 'output'
                        os.makedirs(out_dir, exist_ok=True)
                        calib_path = f'{out_dir}/calibrate.json'
                        with open(calib_path, 'w') as f:
                            json.dump(calib, f, indent=2)
                        print(f'\n标定已保存 → {calib_path}')
                        print(f'  内参: fx={fx_c:.1f} fy={fy_c:.1f} ppx={ppx_c:.1f} ppy={ppy_c:.1f}')
                        print(f'  盘面中心(世界): ({cx_w*100:.1f}, {cy_w*100:.1f})cm')
                        print(f'  盘面半径: {radius_m*100:.1f}cm')
                        print(f'  R={R_calib.tolist()}')
                        print(f'  t={t_calib.tolist()}')
                        break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()