import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # IMU 需要 root 权限, 跳过
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    # ========== RealSense 后处理滤波器 ==========
    # 1. 空域滤波：去噪 + 边缘保持 + 空洞填补
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 3)  # 0~5，越大填得越猛

    # 2. 时域滤波：多帧融合，减少闪烁
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    # 3. 空洞填补滤波：专门补黑色空洞
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)  # 0=近邻, 1=下方, 2=远方

    print("按 'q' 退出 | 按 's' 保存截图 | 按 'c' 切换深度模式")
    show_colorized = True

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            accel_frame = None
            gyro_frame = None

            if not aligned_depth_frame or not color_frame:
                continue

            # ========== 应用后处理滤波器（新增） ==========
            filtered_depth = spatial.process(aligned_depth_frame)
            filtered_depth = temporal.process(filtered_depth)
            filtered_depth = hole_filling.process(filtered_depth)
            # 现在 filtered_depth 就是"修过"的深度帧，后面的代码都用它

            # 用过滤后的深度帧做可视化
            if show_colorized:
                depth_colormap = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
            else:
                depth_image = np.asanyarray(filtered_depth.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

            color_image = np.asanyarray(color_frame.get_data())

            # 中心测距也用过滤后的深度（更稳定）
            depth_image = np.asanyarray(filtered_depth.get_data())
            h, w = depth_image.shape
            cy, cx = h // 2, w // 2
            roi = depth_image[cy-1:cy+2, cx-1:cx+2]
            valid = roi[roi > 0]
            center_depth = np.mean(valid) * depth_scale if len(valid) > 0 else 0

            # HUD 绘制（和原来一样）
            info_overlay = color_image.copy()
            cv2.putText(info_overlay, f"Center Distance: {center_depth:.3f}m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_overlay, f"Depth Scale: {depth_scale:.6f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.drawMarker(info_overlay, (cx, cy), (0, 0, 255),
                          cv2.MARKER_CROSS, 20, 2)

            if accel_frame:
                a = accel_frame.as_motion_frame().get_motion_data()
                cv2.putText(info_overlay, f"Accel: x={a.x:.2f} y={a.y:.2f} z={a.z:.2f}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if gyro_frame:
                g = gyro_frame.as_motion_frame().get_motion_data()
                cv2.putText(info_overlay, f"Gyro: x={g.x:.2f} y={g.y:.2f} z={g.z:.2f}",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            depth_colormap_resized = cv2.resize(depth_colormap, (640, 480))
            combined = np.hstack((info_overlay, depth_colormap_resized))

            title_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, "RealSense D435i - Color (Left) | Depth (Right) - Filtered",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            final_display = np.vstack((title_bar, combined))

            cv2.imshow('RealSense D435i Viewer', final_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = cv2.getTickCount()
                cv2.imwrite(f"realsense_capture_{timestamp}.png", final_display)
                print(f"截图已保存: realsense_capture_{timestamp}.png")
            elif key == ord('c'):
                show_colorized = not show_colorized
                print(f"深度显示模式: {'彩色映射' if show_colorized else '伪彩色'}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")

if __name__ == "__main__":
    main()