#!/usr/bin/env python3
"""D435i 转台扫描: 实时预览 + 按空格采集 + 拼接导出PLY.

用法: python3 capture_scan.py
前提: 先运行 calibrate.py 完成标定.
"""

import json
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, '/home/xie/项目文件夹/real_scan')
from config import *
from utils import world_rotate_z_around, save_ply, post_process


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        print(f'错误: 标定文件不存在 → {CALIB_FILE}')
        print('请先运行: python3 calibrate.py')
        return None
    with open(CALIB_FILE, 'r') as f:
        data = json.load(f)
    R = np.array(data['R'])
    t = np.array(data['t'])
    plate_z = data['plate_z']
    rotation_center = np.array(data.get('rotation_center', [0.0, 0.0]))
    print(f'标定加载: plate_z={plate_z:.3f}, 旋转中心={rotation_center}')
    return R, t, plate_z, rotation_center


def capture_from_depth(depth_frame, n_frames=3):
    """从深度帧生成点云 (相机坐标系). 多次采样稳定."""

    # 将当前帧保存, 直接用它计算
    pts = np.asanyarray(depth_frame.get_data())  # H×W raw depth mm

    # 生成点云用 rs.pointcloud (复用 connect 的 pipeline 不方便, 用手动计算)
    # 直接从深度图 + 内参算
    # 但我们有 align 和 pc object... 简便做法: accept depth_frame, use pre-created pc
    # 这需要传递更多上下文. 简化: 在 main loop 里直接处理.
    return pts


def main():
    import pyrealsense2 as rs

    # 1. 加载标定
    result = load_calibration()
    if result is None:
        return
    R, t, plate_z, rotation_center = result

    # 2. 连接相机 (RGB+Depth 双流, 用于预览)
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print('错误: 未检测到 RealSense 相机!')
        return
    dev = devices[0]

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, DEPTH_FPS)
    cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.bgr8, DEPTH_FPS)
    pipeline.start(cfg)

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    pc = rs.pointcloud()

    # 预热
    for _ in range(30):
        pipeline.wait_for_frames()
        time.sleep(0.05)

    # 3. 扫描循环
    frames = {}
    cap_idx = 0
    total = len(TURNTABLE_ANGLES)
    target_deg = TURNTABLE_ANGLES[0]

    cv2.namedWindow('Scan - SPACE 采集 | Q 退出', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Scan - SPACE 采集 | Q 退出', 960, 360)

    print('\n' + '=' * 50)
    print('  SPACE = 采集当前角度    Q = 退出')
    print('=' * 50)

    flash_t = 0  # 采集闪烁计时

    try:
        while cap_idx < total:
            frames_rs = pipeline.wait_for_frames()
            aligned = align.process(frames_rs)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 预览图像
            color_img = np.asanyarray(color_frame.get_data())
            depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # 缩放拼接
            h, w = color_img.shape[:2]
            depth_disp = cv2.resize(depth_colored, (w, h))
            preview = np.hstack([color_img, depth_disp])

            # 叠加信息
            target_deg = TURNTABLE_ANGLES[cap_idx]
            cv2.putText(preview, 'RGB', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, 'Depth', (w + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(preview, f'[{cap_idx}/{total}]', (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # 下一个目标角度提示
            hint = f'>> 旋转到 {target_deg} [SPACE采集]'
            cv2.putText(preview, hint, (w//2 - 150, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 已采集角度列表
            done_str = '已采集: ' + ', '.join(f'{d}' for d in sorted(frames.keys()))
            cv2.putText(preview, done_str, (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # 采集闪烁效果
            if time.time() - flash_t < 0.3:
                overlay = preview.copy()
                cv2.rectangle(overlay, (0, 0), (preview.shape[1], preview.shape[0]),
                              (0, 255, 0), -1)
                preview = cv2.addWeighted(preview, 0.6, overlay, 0.4, 0)

            cv2.imshow('Scan - SPACE 采集 | Q 退出', preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # SPACE
                # 采集点云
                pc.map_to(aligned)
                vertices = np.asanyarray(pc.calculate(depth_frame).get_vertices()).view(np.float32)
                pts = vertices.reshape(-1, 3)
                valid = np.all(np.isfinite(pts), axis=1) & (pts[:, 2] > 0.01)
                pts = pts[valid]

                if len(pts) < 50000:
                    print(f'  采集失败 ({len(pts)}点), 重试...')
                    continue

                # raw → world
                pts_world = (R @ pts.T).T + t
                frames[target_deg] = pts_world

                print(f'  [{cap_idx+1}/{total}] {target_deg:3d}°: {len(pts_world):,}点')
                cap_idx += 1
                flash_t = time.time()

            elif key == ord('q') or key == 27:  # Q or ESC
                print('\n用户退出.')
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(frames) < 3:
        print(f'错误: 有效帧不足 ({len(frames)})')
        return

    # 4. 反旋
    print(f'\n[反旋] 旋转中心={rotation_center} ...')
    angles = sorted(frames.keys())
    ref_angle = angles[0]
    aligned_frames = []
    for deg in angles:
        rot_angle = -(deg - ref_angle)
        pts_aligned = world_rotate_z_around(frames[deg], rot_angle, rotation_center)
        aligned_frames.append(pts_aligned)
        print(f'  {deg:3d} -> 反旋 {rot_angle:+.0f}: {len(pts_aligned):,}点')

    # 5. 合并 + 后处理
    merged = np.vstack(aligned_frames)
    print(f'\n[合并] {len(aligned_frames)}帧, {len(merged):,}点')

    print('[后处理] 降采样 -> 去噪 -> DBSCAN -> Y-up ...')
    pts_final = post_process(merged,
                             voxel_size=VOXEL_SIZE,
                             outlier_nb=OUTLIER_NB,
                             outlier_radius=OUTLIER_RADIUS,
                             dbscan_eps=DBSCAN_EPS,
                             dbscan_min=DBSCAN_MIN)

    if len(pts_final) == 0:
        print('错误: 后处理结果为空!')
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_ply(pts_final, SCAN_OUTPUT)
    fsize = os.path.getsize(SCAN_OUTPUT)
    print(f'\n[完成] -> {SCAN_OUTPUT}')
    print(f'  点数: {len(pts_final):,}  |  文件: {fsize:,} bytes')


if __name__ == '__main__':
    main()
