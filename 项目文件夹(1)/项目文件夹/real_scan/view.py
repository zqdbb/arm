import pyrealsense2 as rs
import numpy as np
import cv2
import rclpy
from robot import RealmanController

# 简单的实时显示 + 机械臂关节控制
def simple_viewer():
    # ── 机械臂初始化（慢速）──
    rclpy.init()
    robot = RealmanController(speed=10)
    current = robot.get_joints_deg()
    targets = list(current) if current else [0.0] * 6
    joint_idx = 0

    # ── 相机初始化 ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("实时显示已启动")
    print("按键控制：[ / ] 切换关节 | - / = 选中关节减/加 2 度 | q 退出")
    print("机械臂运动时请留意周围，紧急情况按机械臂上的急停按钮")

    try:
        while True:
            # 获取帧
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转换为 numpy 数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 深度图彩色化
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # 水平拼接显示
            images = np.hstack((color_image, depth_colormap))

            # 叠加当前选中关节和目标角度
            cv2.putText(
                images,
                f"J{joint_idx+1}: {targets[joint_idx]:.1f} deg (speed 10)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            cv2.imshow('RealSense D435i', images)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('['):
                joint_idx = (joint_idx - 1) % 6
                print(f"选中关节 J{joint_idx+1}")
            elif key == ord(']'):
                joint_idx = (joint_idx + 1) % 6
                print(f"选中关节 J{joint_idx+1}")
            elif key in (ord('-'), ord(',')):
                targets[joint_idx] -= 2.0
                robot.move_joints_deg(targets, speed=10, block=False)
            elif key in (ord('='), ord('.')):
                targets[joint_idx] += 2.0
                robot.move_joints_deg(targets, speed=10, block=False)

            # 处理机械臂 ROS 回调（非阻塞）
            rclpy.spin_once(robot, timeout_sec=0)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    simple_viewer()
