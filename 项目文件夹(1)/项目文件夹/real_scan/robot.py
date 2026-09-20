#!/usr/bin/env python3
"""瑞尔曼 ECO65-B 机械臂关节控制（ROS2 版）

依赖：rm_driver 节点已启动（ros2 launch rm_driver rm_eco65_driver.launch.py）
运行前先 source ROS 环境：
    source /opt/ros/humble/setup.bash
    source ~/ros2_ws/install/setup.bash

关节角度单位：
    - move_joints_deg / get_joints_deg  用「度」（日常习惯）
    - move_joints / get_joints          用「弧度」（ROS2 底层）
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from rm_ros_interfaces.msg import Movej, Armstate

DEG = math.pi / 180.0


class RealmanController(Node):
    def __init__(self, speed=10):
        super().__init__('realman_controller')
        # 默认速度（1~100，越小越慢）
        self.default_speed = speed

        # 关节运动：发布命令 / 订阅结果
        self.movej_pub = self.create_publisher(Movej, '/rm_driver/movej_cmd', 10)
        self.movej_sub = self.create_subscription(Bool, '/rm_driver/movej_result', self._on_movej_result, 10)

        # 状态查询：发布命令 / 订阅结果
        self.state_pub = self.create_publisher(Empty, '/rm_driver/get_current_arm_state_cmd', 10)
        self.state_sub = self.create_subscription(Armstate, '/rm_driver/get_current_arm_state_result', self._on_state, 10)

        self._move_ok = None
        self._joints_rad = None

    # ── 回调 ──
    def _on_movej_result(self, msg):
        self._move_ok = bool(msg.data)

    def _on_state(self, msg):
        self._joints_rad = list(msg.joint)

    # ── 关节运动 ──
    def move_joints_deg(self, angles_deg, speed=None, block=True):
        """按「度」移动 6 个关节。speed 默认用构造时的慢速值。"""
        joint_rad = [a * DEG for a in angles_deg]
        return self.move_joints(joint_rad, speed, block)

    def move_joints(self, joint_rad, speed=None, block=True):
        """按「弧度」移动 6 个关节。"""
        if len(joint_rad) != 6:
            self.get_logger().error(f'需要 6 个关节角度，收到 {len(joint_rad)} 个')
            return False

        speed = speed if speed is not None else self.default_speed
        speed = max(1, min(100, int(speed)))

        msg = Movej()
        msg.joint = [float(j) for j in joint_rad]
        msg.speed = speed
        msg.block = bool(block)
        msg.trajectory_connect = 0
        msg.dof = 6

        deg = [round(a / DEG, 2) for a in joint_rad]
        self.get_logger().info(f'发送关节命令(度): {deg}，速度: {speed}%')
        self._move_ok = None
        self.movej_pub.publish(msg)

        if block:
            return self._wait_move_result(timeout=60.0)
        return True

    def _wait_move_result(self, timeout=60.0):
        """等待 MoveJ 结果（block 模式下，运动完成后 driver 才返回结果）。"""
        t0 = time.time()
        while self._move_ok is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._move_ok is None:
            self.get_logger().warn('等待 MoveJ 结果超时')
            return False
        self.get_logger().info('MoveJ ' + ('成功' if self._move_ok else '失败'))
        return self._move_ok

    # ── 状态查询 ──
    def get_joints_deg(self):
        """读取当前 6 关节角度，返回「度」。"""
        rad = self.get_joints()
        if rad is None:
            return None
        return [a / DEG for a in rad]

    def get_joints(self):
        """读取当前 6 关节角度，返回「弧度」。"""
        self._joints_rad = None
        self.state_pub.publish(Empty())
        t0 = time.time()
        while self._joints_rad is None and time.time() - t0 < 5.0:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._joints_rad is None:
            self.get_logger().warn('读取关节角度超时（rm_driver 是否在运行？）')
            return None
        deg = [round(a / DEG, 2) for a in self._joints_rad]
        self.get_logger().info(f'当前关节(度): {deg}')
        return self._joints_rad


def main():
    rclpy.init()
    robot = RealmanController(speed=20)  # 慢速，可调

    # 等节点订阅/发布器就绪
    time.sleep(1)

    # 1. 读当前关节角度
    robot.get_joints_deg()

    # 2. 慢速小幅测试：仅动 J1 到 5.7 度，其余保持 0
    #    改这里的角度即可控制机械臂到目标位姿
    target_deg = [-41.94, 80.476, -146.443, 22.351, 121.555, -120.882]
    robot.move_joints_deg(target_deg, speed=10)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
