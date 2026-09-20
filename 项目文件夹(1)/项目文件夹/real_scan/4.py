#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel RealSense D405 可视化工具（最终版）
修复：自动探测可用选项，避免报错
"""

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import pyrealsense2 as rs
import numpy as np
import cv2
import time

# ==================== 配置参数 ====================
WIDTH, HEIGHT = 1280, 720
FPS = 30
DEPTH_WIDTH, DEPTH_HEIGHT = 1280, 720

MIN_DISTANCE = 0.07   # 7cm
MAX_DISTANCE = 0.50   # 50cm

SAVE_DIR = "./d405_captures"
os.makedirs(SAVE_DIR, exist_ok=True)


def list_supported_options(sensor, label):
    """打印传感器支持的所有选项（调试用）"""
    print(f"\n--- {label} 支持的选项 ---")
    for opt in range(int(rs.option.count)):
        try:
            if sensor.supports(rs.option(opt)):
                val = sensor.get_option(rs.option(opt))
                print(f"  {rs.option(opt).name}: {val}")
        except:
            pass


def init_camera():
    """初始化 RealSense D405 相机"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    print("=" * 50)
    print("Intel RealSense D405 可视化工具")
    print("=" * 50)
    print(f"设备名称: {device.get_info(rs.camera_info.name)}")
    print(f"序列号: {device.get_info(rs.camera_info.serial_number)}")
    print(f"固件版本: {device.get_info(rs.camera_info.firmware_version)}")
    
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    print(f"\n深度单位 (depth_scale): {depth_scale} m")
    print(f"即 1 个深度单位 = {depth_scale * 1000:.2f} mm")
    
    # === 尝试设置视觉预设 ===
    if depth_sensor.supports(rs.option.visual_preset):
        try:
            preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
            print(f"\n可用预设范围: {preset_range.min} ~ {preset_range.max}")
            
            # 尝试找 "Close Range" 或 "Short Range" 预设
            preset_name_map = {
                0: "Custom", 1: "Default", 2: "Hand", 
                3: "High Accuracy", 4: "High Density", 
                5: "Medium Density", 6: "Close Range"
            }
            
            # 先尝试 6 (Close Range)，如果不行就尝试 0 (Custom)
            for preset_id in [6, 0, 1]:
                if preset_range.min <= preset_id <= preset_range.max:
                    depth_sensor.set_option(rs.option.visual_preset, preset_id)
                    name = preset_name_map.get(preset_id, f"Preset {preset_id}")
                    print(f"✓ 已设置视觉预设: {name} (ID={preset_id})")
                    break
        except Exception as e:
            print(f"设置预设失败: {e}")
    else:
        print("该设备不支持 visual_preset 选项")
    
    # === 尝试设置激光/投影功率 ===
    if depth_sensor.supports(rs.option.laser_power):
        try:
            laser_range = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, laser_range.max)
            print(f"✓ 激光功率已设置为: {laser_range.max}")
        except Exception as e:
            print(f"设置激光功率失败: {e}")
    else:
        print("该设备不支持 laser_power 选项（D405 使用结构光，无激光器）")
    
    # === 尝试设置发射器功率（D405 可能是这个选项）===
    if depth_sensor.supports(rs.option.emitter_enabled):
        try:
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
            print("✓ 红外发射器已启用")
        except:
            pass
    
    align = rs.align(rs.stream.color)
    return pipeline, align, depth_scale


def process_frames(frames, align):
    """对齐并提取帧"""
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None, None, None
    
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    
    return depth_image, color_image, intrinsics


def visualize_depth(depth_image, depth_scale, min_dist, max_dist):
    """深度图伪彩色化"""
    depth_meters = depth_image.astype(np.float32) * depth_scale
    depth_clipped = np.clip(depth_meters, min_dist, max_dist)
    depth_norm = ((depth_clipped - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_colormap[depth_image == 0] = [0, 0, 0]
    return depth_colormap


def create_display(color_image, depth_colormap, min_dist, max_dist, fps=0):
    """创建并排显示"""
    h, w = color_image.shape[:2]
    depth_colormap = cv2.resize(depth_colormap, (w, h))
    
    # 深度标尺条
    bar_w = 40
    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)
    for i in range(h):
        v = int(255 * (1 - i / h))
        bar[i] = cv2.applyColorMap(np.uint8([[v]]), cv2.COLORMAP_JET)[0, 0]
    
    # 标尺文字
    for i in range(6):
        y = int(i * h / 5)
        val = max_dist - (max_dist - min_dist) * (i / 5)
        cv2.putText(bar, f"{val*100:.0f}", (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    combined = np.hstack([color_image, depth_colormap, bar])
    
    # 信息文字
    info = f"Range:{min_dist*100:.0f}-{max_dist*100:.0f}cm | q=退出 s=截图 c=点云 1/2/3=量程 | 鼠标悬停看距离"
    cv2.putText(combined, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if fps > 0:
        cv2.putText(combined, f"FPS:{fps:.1f}", (combined.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return combined


def main():
    # 先创建窗口（在 RealSense 初始化之前，避免 Qt 状态冲突）
    window_name = "D405 Depth Visualization (press keys on this window)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    # 鼠标回调
    mouse_x, mouse_y = 0, 0
    def on_mouse(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y
    cv2.setMouseCallback(window_name, on_mouse)

    pipeline, align, depth_scale = init_camera()
    
    global MIN_DISTANCE, MAX_DISTANCE
    frame_count = 0
    start_time = time.time()
    save_count = 0
    
    print("\n" + "=" * 50)
    print("⚠️  重要：请用鼠标点击弹出的可视化窗口")
    print("    然后在那个窗口里按键盘按键！")
    print("    不要在终端/控制台里输入！")
    print("=" * 50)
    print("\n按键说明：")
    print("  q  → 退出程序")
    print("  s  → 保存截图（彩色+深度+原始数据）")
    print("  c  → 保存点云（需安装 open3d）")
    print("  1  → 近距离模式 (7-50cm)")
    print("  2  → 中距离模式 (20-100cm)")
    print("  3  → 远距离模式 (50-300cm)")
    print("  鼠标悬停 → 查看该像素距离\n")
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_image, color_image, intrinsics = process_frames(frames, align)
            
            if depth_image is None:
                continue
            
            depth_colormap = visualize_depth(depth_image, depth_scale, MIN_DISTANCE, MAX_DISTANCE)
            
            # 计算 FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            if elapsed > 1.0:
                frame_count = 0
                start_time = time.time()
            
            display = create_display(color_image, depth_colormap, MIN_DISTANCE, MAX_DISTANCE, fps)
            
            # 鼠标位置距离显示
            if mouse_x < color_image.shape[1]:  # 只在左侧彩色图区域有效
                dist = depth_image[mouse_y, mouse_x] * depth_scale if 0 <= mouse_y < depth_image.shape[0] else 0
                if dist > 0:
                    cv2.putText(display, f"Dist: {dist:.4f}m", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # 画十字准星
                    cx = mouse_x
                    cy = mouse_y
                    cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
            cv2.imshow(window_name, display)
            
            # 等待按键（1ms），焦点必须在窗口上
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n收到退出指令，正在关闭...")
                break
            
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"{SAVE_DIR}/color_{ts}.png", color_image)
                cv2.imwrite(f"{SAVE_DIR}/depth_{ts}.png", depth_colormap)
                np.save(f"{SAVE_DIR}/depth_raw_{ts}.npy", depth_image)
                save_count += 1
                print(f"[{save_count}] 截图已保存到 {SAVE_DIR}/")
            
            elif key == ord('c'):
                try:
                    import open3d as o3d
                    depth_o3d = o3d.geometry.Image(depth_image)
                    color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color_o3d, depth_o3d,
                        depth_scale=1.0 / depth_scale,
                        depth_trunc=MAX_DISTANCE,
                        convert_rgb_to_intensity=False
                    )
                    pinhole = o3d.camera.PinholeCameraIntrinsic(
                        intrinsics.width, intrinsics.height,
                        intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
                    )
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
                    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    o3d.io.write_point_cloud(f"{SAVE_DIR}/pointcloud_{ts}.ply", pcd)
                    print(f"点云已保存: pointcloud_{ts}.ply")
                    o3d.visualization.draw_geometries([pcd], window_name="D405 Point Cloud")
                except ImportError:
                    print("请先安装 open3d: pip install open3d")
            
            elif key == ord('1'):
                MIN_DISTANCE, MAX_DISTANCE = 0.07, 0.50
                print("量程切换: 近距离 7-50cm")
            elif key == ord('2'):
                MIN_DISTANCE, MAX_DISTANCE = 0.20, 1.00
                print("量程切换: 中距离 20-100cm")
            elif key == ord('3'):
                MIN_DISTANCE, MAX_DISTANCE = 0.50, 3.00
                print("量程切换: 远距离 50-300cm")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    main()