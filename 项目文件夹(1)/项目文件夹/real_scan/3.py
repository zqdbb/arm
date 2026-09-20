import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # IMU 需要 root 权限, 跳过
    # config.enable_stream(rs.stream.accel)
    # config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    # ========== 改进版后处理管道：视差域滤波 + 保守参数 ==========
    
    # 1. 深度 ↔ 视差 转换（在视差域滤波边缘更清晰）
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # 2. 空域滤波：保守参数，只去噪，不猛糊
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)   # 降低：0.5→0.25，更保守
    spatial.set_option(rs.option.filter_smooth_delta, 5)      # 降低：20→5，边缘保持更好
    spatial.set_option(rs.option.holes_fill, 2)             # 降低：3→2，轻度填洞

    # 3. 时域滤波
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.2)
    temporal.set_option(rs.option.filter_smooth_delta, 10)

    # 4. 空洞填补：用 mode 1（farest_from_around），比近邻填充更自然
    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)

    print("=" * 55)
    print("按 'q' 退出 | 按 's' 截图 | 按 'c' 切换深度模式")
    print("按 'f' 开关滤波器对比 | 按 'h' 开关填补区域高亮")
    print("=" * 55)
    
    show_colorized = True
    use_filter = True
    highlight_fill = True  # 高亮显示"被填补"的区域

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            accel_frame = aligned_frames.first_or_default(rs.stream.accel)
            gyro_frame = aligned_frames.first_or_default(rs.stream.gyro)

            if not depth_frame or not color_frame:
                continue

            # 保存原始深度掩码（>0 表示真实测到的深度）
            raw_depth = np.asanyarray(depth_frame.get_data())
            valid_mask = (raw_depth > 0).astype(np.uint8)

            # ========== 应用视差域滤波管道 ==========
            if use_filter:
                # 深度 → 视差 → 空域滤波 → 时域滤波 → 视差 → 深度
                filtered = depth_to_disparity.process(depth_frame)
                filtered = spatial.process(filtered)
                filtered = temporal.process(filtered)
                filtered = disparity_to_depth.process(filtered)
                # 最后再做一次轻度的空洞填补
                filtered = hole_filling.process(filtered)
                depth_for_vis = filtered
                title_suffix = " - Filtered"
            else:
                depth_for_vis = depth_frame
                title_suffix = " - RAW"

            # 颜色映射
            if show_colorized:
                depth_colormap = np.asanyarray(colorizer.colorize(depth_for_vis).get_data())
            else:
                d = np.asanyarray(depth_for_vis.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(d, alpha=0.03),
                    cv2.COLORMAP_JET
                )

            # ========== 关键改进：区分"真实深度"和"填补区域" ==========
            if highlight_fill and use_filter:
                # 把原始没有深度（mask=0）但被滤波器填补上的区域标出来
                # 方法：在这些区域覆盖半透明灰色，并画虚线网格
                fill_mask = (valid_mask == 0)
                
                # 创建覆盖层
                overlay = depth_colormap.copy()
                
                # 填补区域变暗 + 偏灰
                overlay[fill_mask] = (overlay[fill_mask] * 0.3 + np.array([80, 80, 80]) * 0.7).astype(np.uint8)
                
                # 在填补区域画稀疏网格线（每20px一条）
                h, w = fill_mask.shape
                grid = np.zeros_like(fill_mask)
                grid[::20, :] = True
                grid[:, ::20] = True
                grid_mask = fill_mask & grid
                overlay[grid_mask] = [255, 255, 255]  # 白色网格
                
                depth_colormap = overlay

            # 提取深度数据用于测距
            depth_data = np.asanyarray(depth_for_vis.get_data())
            h, w = depth_data.shape
            cy, cx = h // 2, w // 2
            
            # 中心测距：优先用原始有效深度，没有再用滤波后的
            roi_raw = raw_depth[cy-1:cy+2, cx-1:cx+2]
            valid_raw = roi_raw[roi_raw > 0]
            if len(valid_raw) > 0:
                center_depth = np.mean(valid_raw) * depth_scale
                depth_source = "RAW"
            else:
                roi_fil = depth_data[cy-1:cy+2, cx-1:cx+2]
                valid_fil = roi_fil[roi_fil > 0]
                center_depth = np.mean(valid_fil) * depth_scale if len(valid_fil) > 0 else 0
                depth_source = "FILL"

            # 彩色图 HUD
            color_image = np.asanyarray(color_frame.get_data())
            info_overlay = color_image.copy()
            
            dist_color = (0, 255, 0) if depth_source == "RAW" else (0, 165, 255)
            cv2.putText(info_overlay, f"Center: {center_depth:.3f}m [{depth_source}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2)
            cv2.putText(info_overlay, f"Depth Scale: {depth_scale:.6f}",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.drawMarker(info_overlay, (cx, cy), (0, 0, 255),
                          cv2.MARKER_CROSS, 20, 2)

            # IMU
            if accel_frame:
                a = accel_frame.as_motion_frame().get_motion_data()
                cv2.putText(info_overlay, f"A: x={a.x:.2f} y={a.y:.2f} z={a.z:.2f}",
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if gyro_frame:
                g = gyro_frame.as_motion_frame().get_motion_data()
                cv2.putText(info_overlay, f"G: x={g.x:.2f} y={g.y:.2f} z={g.z:.2f}",
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

            # 深度图加图例说明
            depth_display = cv2.resize(depth_colormap, (640, 480))
            cv2.putText(depth_display, "Grid=Filled Area" if highlight_fill else "Depth",
                       (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 拼接
            combined = np.hstack((info_overlay, depth_display))
            title_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, f"RealSense D435i - Color | Depth{title_suffix}",
                       (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            final_display = np.vstack((title_bar, combined))

            cv2.imshow('RealSense D435i Viewer', final_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = cv2.getTickCount()
                cv2.imwrite(f"capture_{ts}.png", final_display)
                print(f"截图已保存: capture_{ts}.png")
            elif key == ord('c'):
                show_colorized = not show_colorized
            elif key == ord('f'):
                use_filter = not use_filter
                print(f"滤波器: {'开启' if use_filter else '关闭'}")
            elif key == ord('h'):
                highlight_fill = not highlight_fill
                print(f"填补高亮: {'开启' if highlight_fill else '关闭'}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("相机已关闭")

if __name__ == "__main__":
    main()