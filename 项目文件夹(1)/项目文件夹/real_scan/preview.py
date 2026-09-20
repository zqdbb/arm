#!/usr/bin/env python3
"""D435i 相机预览 — 同时显示 RGB 和深度图, 按 ESC 退出.

用法: python3 preview.py
"""

import sys
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    print('请先安装: pip3 install pyrealsense2')
    sys.exit(1)


def main():
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print('错误: 未检测到 RealSense 相机! 请检查 USB 连接.')
        return
    dev = devices[0]
    print(f'设备: {dev.get_info(rs.camera_info.name)}')
    print(f'序列号: {dev.get_info(rs.camera_info.serial_number)}')

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)

    # 对齐: 深度对齐到彩色
    align = rs.align(rs.stream.color)

    # 深度色映射
    colorizer = rs.colorizer()

    print('\n预览中... 按 ESC 退出')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转换
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 深度着色
            depth_colored = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())

            # 拼接: RGB左, 深度右
            h, w = color_image.shape[:2]
            # 深度图缩放到和RGB一样
            depth_display = cv2.resize(depth_colored, (w, h))
            combined = np.hstack([color_image, depth_display])

            # 叠加帧率
            cv2.putText(combined, 'RGB', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, 'Depth', (w + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('D435i Preview (ESC to quit)', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                # 按 S 截图
                import time
                ts = time.strftime('%Y%m%d_%H%M%S')
                fname = f'd435i_snapshot_{ts}.png'
                cv2.imwrite(fname, combined)
                print(f'  截图: {fname}')

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print('预览结束.')


if __name__ == '__main__':
    main()
