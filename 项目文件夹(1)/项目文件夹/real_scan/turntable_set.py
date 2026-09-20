#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D405 双阶段圆盘标定 —— 亚像素边缘精修版
"""

import time

import cv2
import numpy as np
import pyrealsense2 as rs


class TwoPhaseLocator:
    STATE_SEARCH = 0
    STATE_TRACK = 1

    def __init__(self):
        self.state = self.STATE_SEARCH
        self.center = None
        self.ellipse = None
        self.ring_points = []
        self.edge_mode = 'inner'  # 'inner', 'outer', 'center'
        self.HIST_SIZE = 15
        self.ell_history = []

    # ========== 第一阶段：中心黑点 ==========
    def detect_center(self, gray):
        blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=60,
            param1=80, param2=20,
            minRadius=3, maxRadius=25
        )
        if circles is None:
            return None

        h, w = gray.shape
        cx_frame, cy_frame = w / 2, h / 2
        best = None
        best_score = -1

        for c in circles[0, :]:
            cx, cy, r = c
            dist = np.sqrt((cx - cx_frame)**2 + (cy - cy_frame)**2)
            score = 1 - dist / (max(w, h) * 0.5)
            y1, y2 = max(0, int(cy - r)), min(h, int(cy + r))
            x1, x2 = max(0, int(cx - r)), min(w, int(cx + r))
            if y2 > y1 and x2 > x1:
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0 and np.mean(roi) < 120:
                    score += 0.5
            if score > best_score:
                best_score = score
                best = (cx, cy, r)

        return best

    # ========== 第二阶段：亚像素径向边缘 ==========
    def detect_ring_precise(self, gray, center):
        """
        亚像素级边缘定位。
        沿每条射线采样灰度，找最陡的下降沿（黑线内边缘）。
        """
        cx, cy = center
        h, w = gray.shape

        # 自适应阈值：中心附近是亮的，黑线是暗的
        # 采样中心区域亮度作为背景
        bg_vals = []
        for r_test in range(10, 60, 5):
            for ang in np.linspace(0, 2*np.pi, 16, endpoint=False):
                x = int(cx + r_test * np.cos(ang))
                y = int(cy + r_test * np.sin(ang))
                if 0 <= x < w and 0 <= y < h:
                    bg_vals.append(gray[y, x])
        bg_mean = np.mean(bg_vals) if bg_vals else 180
        target_dark = bg_mean * 0.35  # 黑线目标亮度

        points = []
        debug_profiles = []  # 调试用

        for i in range(360):  # 360条射线，1度一条
            theta = np.deg2rad(i)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # 沿射线密集采样（步长=1像素）
            rsamp, vals = [], []
            for r in range(60, 550, 1):
                x = cx + r * cos_t
                y = cy + r * sin_t
                ix, iy = int(x), int(y)
                if not (0 <= ix < w and 0 <= iy < h):
                    break
                rsamp.append(r)
                # 双线性插值采样，亚像素精度
                fx, fy = x - ix, y - iy
                v00 = gray[iy, ix]
                v01 = gray[iy, min(ix+1, w-1)]
                v10 = gray[min(iy+1, h-1), ix]
                v11 = gray[min(iy+1, h-1), min(ix+1, w-1)]
                val = (1-fx)*(1-fy)*v00 + fx*(1-fy)*v01 + (1-fx)*fy*v10 + fx*fy*v11
                vals.append(val)

            if len(rsamp) < 30:
                continue

            vals = np.array(vals, dtype=np.float32)

            # 策略：找"下降到黑线"的最陡梯度点
            # 计算梯度
            grad = np.gradient(vals)
            # 找负梯度最大的位置（最陡下降）
            neg_grad = -grad
            # 只考虑亮度低于背景一半的区域
            valid_mask = vals < bg_mean * 0.7

            if not np.any(valid_mask):
                continue

            # 在有效区域内找最陡下降沿
            best_idx = None
            best_g = -999

            for idx in range(1, len(vals)-1):
                if not valid_mask[idx]:
                    continue
                # 下降沿：前面亮，后面暗
                g = (vals[idx-1] - vals[idx+1]) / 2.0
                if g > best_g and vals[idx] < target_dark + 20:
                    best_g = g
                    best_idx = idx

            if best_idx is None:
                continue

            # 亚像素精修：在最佳点周围3个点做抛物线拟合
            if 1 <= best_idx < len(vals) - 2:
                y0, y1, y2 = vals[best_idx-1], vals[best_idx], vals[best_idx+1]
                # 抛物线顶点偏移
                denom = (y0 - 2*y1 + y2)
                if abs(denom) > 1e-6:
                    offset = (y0 - y2) / (2 * denom)
                    offset = np.clip(offset, -1, 1)
                else:
                    offset = 0
                r_precise = rsamp[best_idx] + offset
            else:
                r_precise = rsamp[best_idx]

            # 根据 edge_mode 微调
            if self.edge_mode == 'inner':
                # 已经是最陡下降沿 = 内边缘
                pass
            elif self.edge_mode == 'outer':
                # 找上升沿（外边缘）：在下降沿之后找最陡上升
                post_vals = vals[best_idx:]
                post_grad = np.gradient(post_vals)
                if len(post_grad) > 2:
                    up_idx = best_idx + np.argmax(post_grad[1:-1]) + 1
                    r_precise = rsamp[min(up_idx, len(rsamp)-1)]
            elif self.edge_mode == 'center':
                # 找黑线中心：下降到最暗点
                dark_idx = best_idx + np.argmin(vals[best_idx:])
                r_precise = rsamp[min(dark_idx, len(rsamp)-1)]

            points.append((cx + r_precise * cos_t, cy + r_precise * sin_t))
            debug_profiles.append((rsamp, vals, best_idx))

        self.ring_points = points
        self.debug_profiles = debug_profiles

        if len(points) < 30:
            return None

        # 离群点剔除：点到中心距离的中值滤波
        dists = [np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in points]
        med_r = np.median(dists)
        filtered = [p for p, d in zip(points, dists) if abs(d - med_r) < med_r * 0.15]
        
        if len(filtered) < 20:
            filtered = points  # fallback

        pts = np.array(filtered, dtype=np.float32).reshape(-1, 1, 2)
        try:
            ellipse = cv2.fitEllipse(pts)
            return ellipse
        except cv2.error:
            return None

    def smooth_ellipse(self, ellipse):
        if ellipse is None:
            return None
        self.ell_history.append(ellipse)
        if len(self.ell_history) > self.HIST_SIZE:
            self.ell_history.pop(0)

        centers = np.array([e[0] for e in self.ell_history])
        axes = np.array([e[1] for e in self.ell_history])
        angles = np.array([e[2] for e in self.ell_history])

        angles_rad = np.deg2rad(angles)
        sin_a = np.median(np.sin(angles_rad))
        cos_a = np.median(np.cos(angles_rad))
        angle_med = np.rad2deg(np.arctan2(sin_a, cos_a))

        return (tuple(np.median(centers, axis=0)),
                tuple(np.median(axes, axis=0)),
                angle_med)

    def get_3d(self, depth_image, center, depth_scale, intrinsics):
        if center is None:
            return None
        cx, cy = center
        cx = np.clip(cx, 0, depth_image.shape[1] - 1)
        cy = np.clip(cy, 0, depth_image.shape[0] - 1)

        w = 5
        y1, y2 = max(0, int(cy - w)), min(depth_image.shape[0], int(cy + w) + 1)
        x1, x2 = max(0, int(cx - w)), min(depth_image.shape[1], int(cx + w) + 1)
        patch = depth_image[y1:y2, x1:x2]
        depths = patch[patch > 0]

        if len(depths) == 0:
            z_raw = depth_image[int(cy), int(cx)]
            if z_raw == 0:
                return None
            sample_count = 1
        else:
            z_med = np.median(depths)
            valid = depths[(depths > z_med * 0.9) & (depths < z_med * 1.1)]
            z_raw = float(np.mean(valid)) if len(valid) > 0 else float(z_med)
            sample_count = len(depths)

        z = z_raw * depth_scale
        X = (cx - intrinsics.ppx) * z / intrinsics.fx
        Y = (cy - intrinsics.ppy) * z / intrinsics.fy

        return {
            'position': (X, Y, z),
            'position_mm': (X * 1000, Y * 1000, z * 1000),
            'depth_raw': z_raw, 'sample_count': sample_count
        }

    def draw(self, image, center_det, pos_3d=None):
        result = image.copy()

        if self.state == self.STATE_SEARCH:
            if center_det:
                cx, cy, r = center_det
                cv2.circle(result, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
                cv2.drawMarker(result, (int(cx), int(cy)), (0, 0, 255),
                              cv2.MARKER_CROSS, 16, 2)
                cv2.putText(result, f"Center: ({cx:.1f}, {cy:.1f})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(result, ">>> Press SPACE to lock <<<",
                           (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
            else:
                cv2.putText(result, "Searching center dot...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 画径向点（红色）
            for px, py in self.ring_points:
                cv2.circle(result, (int(px), int(py)), 1, (0, 0, 255), -1)

            # 画椭圆（绿色）
            if self.ellipse:
                cv2.ellipse(result, self.ellipse, (0, 255, 0), 2)
                cx, cy = self.ellipse[0]
                cv2.drawMarker(result, (int(cx), int(cy)), (0, 255, 0),
                              cv2.MARKER_CROSS, 24, 2)

            # 锁定中心（蓝色）
            if self.center:
                cv2.drawMarker(result, (int(self.center[0]), int(self.center[1])),
                              (255, 0, 0), cv2.MARKER_CROSS, 16, 2)

            if self.ellipse:
                (cx, cy), (MA, ma), ang = self.ellipse
                mode_str = {'inner': '内边缘', 'outer': '外边缘', 'center': '中心'}[self.edge_mode]
                cv2.putText(result, f"Disc: ({cx:.1f}, {cy:.1f}) px  [{mode_str}]",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result, f"Axes: ({MA/2:.1f}, {ma/2:.1f}) px  Angle: {ang:.1f} deg",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if pos_3d:
                x, y, z = pos_3d['position_mm']
                cv2.putText(result, f"GRIP(mm): X={x:+.2f} Y={y:+.2f} Z={z:.2f}",
                           (10, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(result, f"DepthSamples: {pos_3d['sample_count']}",
                           (10, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(result, "SPACE=re-search  R=reset  1=inner 2=outer 3=center",
                       (10, image.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)

        return result


def setup_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"Depth scale: {depth_scale} m")
    print(f"Intrinsics: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, "
          f"cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale, intrinsics


def main():
    pipeline, align, depth_scale, intrinsics = setup_realsense()
    locator = TwoPhaseLocator()

    print("\n" + "=" * 55)
    print("双阶段圆盘标定 —— 亚像素边缘精修版")
    print("=" * 55)
    print("SPACE - 锁定中心 / 重新搜索外圈")
    print("R     - 重置")
    print("S     - 保存位姿")
    print("1/2/3 - 切换边缘模式：内边缘/外边缘/黑线中心")
    print("=" * 55)

    window_name = "Precise Disc Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            center_det = None

            if locator.state == locator.STATE_SEARCH:
                center_det = locator.detect_center(gray)
            else:
                locator.ellipse = locator.detect_ring_precise(gray, locator.center)
                if locator.ellipse:
                    locator.ellipse = locator.smooth_ellipse(locator.ellipse)
                center_det = (locator.center[0], locator.center[1], 5)

            pos_3d = None
            if locator.center:
                pos_3d = locator.get_3d(depth_image, locator.center, depth_scale, intrinsics)

            display = locator.draw(color_image, center_det, pos_3d)
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if locator.state == locator.STATE_SEARCH and center_det:
                    locator.center = (center_det[0], center_det[1])
                    locator.state = locator.STATE_TRACK
                    print(f"中心锁定: ({locator.center[0]:.1f}, {locator.center[1]:.1f})")
                    locator.ellipse = locator.detect_ring_precise(gray, locator.center)
                    if locator.ellipse:
                        print(f"外圈找到: 轴长=({locator.ellipse[1][0]:.1f}, {locator.ellipse[1][1]:.1f})")
                else:
                    locator.ell_history.clear()
                    locator.ellipse = locator.detect_ring_precise(gray, locator.center)
                    print("外圈重新搜索")
            elif key == ord('r'):
                locator.state = locator.STATE_SEARCH
                locator.center = None
                locator.ellipse = None
                locator.ell_history.clear()
                print("已重置")
            elif key == ord('s') and pos_3d:
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"calib_{ts}.png", display)
                x, y, z = pos_3d['position_mm']
                print(f"[{ts}] 位姿: X={x:.2f} Y={y:.2f} Z={z:.2f} mm")
            elif key == ord('1'):
                locator.edge_mode = 'inner'
                locator.ell_history.clear()
                print("边缘模式: 内边缘（最精确）")
            elif key == ord('2'):
                locator.edge_mode = 'outer'
                locator.ell_history.clear()
                print("边缘模式: 外边缘")
            elif key == ord('3'):
                locator.edge_mode = 'center'
                locator.ell_history.clear()
                print("边缘模式: 黑线中心")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()