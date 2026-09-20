import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # 创建管道
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用彩色流和深度流
    # 彩色流: 640x480, 30fps, BGR8格式
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 深度流: 640x480, 30fps, Z16格式
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # IMU 需要 root 权限, 跳过

    # 启动管道
    profile = pipeline.start(config)

    # 获取深度传感器的深度比例（用于将深度值转换为米）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例: {depth_scale}")

    # 创建对齐对象，将深度对齐到彩色帧
    align = rs.align(rs.stream.color)

    # 创建颜色映射器，用于可视化深度图
    colorizer = rs.colorizer()

    print("按 'q' 退出，按 's' 保存截图")
    print("按 'c' 切换深度显示模式（彩色/灰度）")
    print("按 'a' 显示/隐藏加速度计数据")
    print("按 'g' 显示/隐藏陀螺仪数据")

    show_colorized = True  # 深度显示模式
    show_accel = False
    show_gyro = False

    try:
        while True:
            # 等待一对匹配的帧（深度和彩色）
            frames = pipeline.wait_for_frames()

            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)

            # 获取对齐后的帧
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # 获取IMU数据
            accel_frame = aligned_frames.first_or_default(rs.stream.accel)
            gyro_frame = aligned_frames.first_or_default(rs.stream.gyro)

            if not aligned_depth_frame or not color_frame:
                continue

            # 转换为numpy数组
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 深度图可视化
            if show_colorized:
                # 使用RealSense内置颜色映射
                depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
            else:
                # 灰度显示，归一化到0-255
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )

            # 创建深度信息的覆盖层
            depth_info = depth_image.copy()
            # 在图像中心点获取深度值
            h, w = depth_image.shape
            center_x, center_y = w // 2, h // 2
            center_depth = depth_image[center_y, center_x]
            distance_meters = center_depth * depth_scale

            # 添加文字信息到彩色图像
            info_overlay = color_image.copy()
            cv2.putText(info_overlay, f"Center Distance: {distance_meters:.3f}m", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(info_overlay, f"Depth Scale: {depth_scale:.6f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.drawMarker(info_overlay, (center_x, center_y), (0, 0, 255), 
                          cv2.MARKER_CROSS, 20, 2)

            # 添加IMU数据
            if show_accel and accel_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                cv2.putText(info_overlay, f"Accel: x={accel_data.x:.2f} y={accel_data.y:.2f} z={accel_data.z:.2f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            if show_gyro and gyro_frame:
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                cv2.putText(info_overlay, f"Gyro: x={gyro_data.x:.2f} y={gyro_data.y:.2f} z={gyro_data.z:.2f}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # 创建组合显示
            # 调整深度图大小以匹配彩色图
            depth_colormap_resized = cv2.resize(depth_colormap, (640, 480))

            # 水平拼接
            combined = np.hstack((info_overlay, depth_colormap_resized))

            # 添加标题栏
            title_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, "RealSense D435i - Color (Left) | Depth (Right)", 
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            final_display = np.vstack((title_bar, combined))

            # 显示
            cv2.imshow('RealSense D435i Viewer', final_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存截图
                timestamp = cv2.getTickCount()
                cv2.imwrite(f"realsense_capture_{timestamp}.png", final_display)
                print(f"截图已保存: realsense_capture_{timestamp}.png")
            elif key == ord('c'):
                show_colorized = not show_colorized
                print(f"深度显示模式: {'彩色映射' if show_colorized else '伪彩色'}")
            elif key == ord('a'):
                show_accel = not show_accel
            elif key == ord('g'):
                show_gyro = not show_gyro

    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")

if __name__ == "__main__":
    main()