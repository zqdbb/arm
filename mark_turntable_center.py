#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import cv2
import numpy as np


class Marker(Node):
    def __init__(self):
        super().__init__('turntable_center_marker')
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.info = None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(Image, '/camera/color/image_raw', self.rgb_cb, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, 10)

    def rgb_cb(self, msg):
        try: self.rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception: pass

    def depth_cb(self, msg):
        try: self.depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        except Exception: pass

    def info_cb(self, msg): self.info = msg

    def click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.depth is None or self.info is None:
            return
        h, w = self.depth.shape[:2]
        r = 3
        patch = self.depth[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)]
        vals = patch[patch > 0]
        if vals.size == 0:
            print('点击位置没有有效深度，请点击转台台面或边缘。', flush=True); return
        z = float(np.median(vals)) / 1000.0
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        p = np.array([(x-cx)*z/fx, (y-cy)*z/fy, z, 1.0])
        try:
            tf = self.buffer.lookup_transform('baselink', 'camera_color_optical_frame', rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            R = np.array([
                [1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w), 2*(q.x*q.z+q.y*q.w)],
                [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z), 2*(q.y*q.z-q.x*q.w)],
                [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w), 1-2*(q.x*q.x+q.y*q.y)]])
            b = R @ p[:3] + np.array([t.x,t.y,t.z])
            print(f'clicked pixel=({x},{y}), depth={z:.3f} m', flush=True)
            print(f'baselink point: x={b[0]:.4f}, y={b[1]:.4f}, z={b[2]:.4f}', flush=True)
        except Exception as e: print('TF 查询失败:', e, flush=True)


def main():
    rclpy.init(); n = Marker()
    cv2.namedWindow('Turntable Center Marker', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Turntable Center Marker', n.click)
    print('请点击转台圆盘中心；按 q 退出。', flush=True)
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.03)
            if n.rgb is not None:
                frame = n.rgb.copy()
                cv2.putText(frame, 'Click turntable center | q: quit', (20,35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2)
                cv2.imshow('Turntable Center Marker', frame)
            if cv2.waitKey(1) & 0xff == ord('q'): break
    finally:
        cv2.destroyAllWindows(); n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__': main()
