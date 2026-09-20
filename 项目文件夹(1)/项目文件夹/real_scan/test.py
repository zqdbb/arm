#!/usr/bin/env python3
"""
D435i 拍摄脚本 - 从640x480裁剪到500x500
按 's' 保存当前画面，按 'q' 退出
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

class D435iCapture:
    def __init__(self, output_dir="captured_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 原始分辨率和目标分辨率
        self.raw_width = 640
        self.raw_height = 480
        self.target_size = 500
        
        # 初始化相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置流 - 使用原始分辨率
        self.config.enable_stream(rs.stream.depth, self.raw_width, self.raw_height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.raw_width, self.raw_height, rs.format.bgr8, 30)
        
        # 对齐深度到彩色
        self.align = rs.align(rs.stream.color)
        
    def crop_to_square(self, image):
        """从图像中心裁剪出500x500的正方形"""
        h, w = image.shape[:2]
        
        # 计算裁剪区域
        crop_size = min(self.target_size, h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        
        # 裁剪
        cropped = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # 如果需要，调整到精确的500x500
        if cropped.shape[0] != self.target_size or cropped.shape[1] != self.target_size:
            cropped = cv2.resize(cropped, (self.target_size, self.target_size))
        
        return cropped
        
    def start(self):
        """启动相机"""
        print("启动D435i相机...")
        self.pipeline.start(self.config)
        
        # 等待相机稳定
        print("等待相机自动曝光稳定...")
        for i in range(30):
            frames = self.pipeline.wait_for_frames()
        
        print("相机就绪！")
        print("按 's' 保存图像，按 'q' 退出")
        print(f"输出分辨率: {self.target_size}x{self.target_size}")
        
    def save_frame(self, color_image, depth_image):
        """保存图像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 裁剪图像
        color_cropped = self.crop_to_square(color_image)
        depth_cropped = self.crop_to_square(depth_image)
        
        # 保存原始图像
        color_filename = os.path.join(self.output_dir, f"color_{timestamp}.png")
        depth_filename = os.path.join(self.output_dir, f"depth_{timestamp}.png")
        
        cv2.imwrite(color_filename, color_cropped)
        cv2.imwrite(depth_filename, depth_cropped)
        
        # 创建并保存深度彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_cropped, alpha=0.03),
            cv2.COLORMAP_JET
        )
        depth_colormap_filename = os.path.join(self.output_dir, f"depth_colormap_{timestamp}.png")
        cv2.imwrite(depth_colormap_filename, depth_colormap)
        
        print(f"\n图像已保存:")
        print(f"  彩色图: {color_filename} ({color_cropped.shape[1]}x{color_cropped.shape[0]})")
        print(f"  深度图: {depth_filename}")
        print(f"  深度彩色图: {depth_colormap_filename}")
        
    def run(self):
        """运行预览和拍摄"""
        self.start()
        
        try:
            while True:
                # 获取对齐后的帧
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # 转换为numpy数组
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 裁剪图像用于显示
                color_display = self.crop_to_square(color_image)
                depth_display = self.crop_to_square(depth_image)
                
                # 创建深度彩色图用于显示
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_display, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # 水平拼接显示
                display_image = np.hstack((color_display, depth_colormap))
                
                # 添加提示文字
                cv2.putText(display_image, "Press 's' to save, 'q' to quit", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                # 显示分辨率信息
                cv2.putText(display_image, f"Output: {self.target_size}x{self.target_size}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 0), 2)
                
                # 显示图像
                cv2.imshow(f'D435i Capture - {self.target_size}x{self.target_size}', display_image)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):
                    # 保存图像（保存裁剪后的500x500）
                    self.save_frame(color_image, depth_image)
                    
                elif key == ord('q') or key == 27:  # 'q'或ESC
                    print("退出程序")
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            print("相机已关闭")

if __name__ == "__main__":
    capture = D435iCapture()
    capture.run()