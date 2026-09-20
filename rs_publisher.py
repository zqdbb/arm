#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np
import time

class RSPublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')

        # Publishers
        self.pub_depth = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.pub_color = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        # The D435 USB control endpoint can remain busy briefly after stream
        # startup, so retry each setting independently.
        time.sleep(0.5)
        options = (
            (rs.option.visual_preset, 4, 'visual_preset'),
            (rs.option.emitter_enabled, 1, 'emitter_enabled'),
            (rs.option.laser_power, 180, 'laser_power'),
        )
        for option, value, name in options:
            for attempt in range(3):
                try:
                    depth_sensor.set_option(option, value)
                    actual = depth_sensor.get_option(option)
                    self.get_logger().info(f'Depth {name}={actual:.0f}')
                    break
                except Exception as exc:
                    if attempt == 2:
                        self.get_logger().warn(
                            f'Depth {name} setup failed: {exc}')
                    time.sleep(0.5)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.6)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        # Static validation benefits from a short temporal history.  During
        # manual eye-in-hand capture, wait for the image to settle at a pose.
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.temporal.set_option(rs.option.holes_fill, 2)
        self.hole_filling = rs.hole_filling_filter(1)
        self.logged_depth_stats = False

        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()

        self.timer = self.create_timer(1.0 / 15.0, self.publish_data)
        self.get_logger().info('RealSense publisher started')

    def publish_data(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                return

            if not self.logged_depth_stats:
                raw = np.asanyarray(depth_frame.get_data())
                raw_valid = raw[(raw > 0) & (raw < 10000)]
                self.get_logger().info(
                    'Raw depth: invalid_or_out_of_range='
                    f'{100.0 * (1.0 - raw_valid.size / raw.size):.2f}%, '
                    f'median={float(np.median(raw_valid)) if raw_valid.size else 0:.0f} mm')

            depth_frame = self.spatial.process(depth_frame)
            if self.temporal is not None:
                depth_frame = self.temporal.process(depth_frame)
            depth_frame = self.hole_filling.process(depth_frame)

            if not self.logged_depth_stats:
                filtered = np.asanyarray(depth_frame.get_data())
                filtered_valid = filtered[(filtered > 0) & (filtered < 10000)]
                self.get_logger().info(
                    'Filtered depth: invalid_or_out_of_range='
                    f'{100.0 * (1.0 - filtered_valid.size / filtered.size):.2f}%, '
                    f'median={float(np.median(filtered_valid)) if filtered_valid.size else 0:.0f} mm')
                self.logged_depth_stats = True

            timestamp = self.get_clock().now().to_msg()

            # Depth image
            depth_data = np.asanyarray(depth_frame.get_data())
            # RealSense uses 0 for missing depth.  Some filter paths can emit
            # 65535 for an unresolved sample; never publish that as 65 m.
            depth_data = np.where(depth_data > 10000, 0, depth_data).astype(np.uint16)
            depth_msg = Image()
            # After align(depth -> color), pixels are expressed in the color
            # optical frame and must use the same frame as CameraInfo/RGB.
            depth_msg.header = Header(stamp=timestamp, frame_id='camera_color_optical_frame')
            depth_msg.height, depth_msg.width = depth_data.shape
            depth_msg.encoding = '16UC1'
            depth_msg.is_bigendian = 0
            depth_msg.step = depth_msg.width * 2
            depth_msg.data = depth_data.tobytes()
            self.pub_depth.publish(depth_msg)

            # Color image
            color_data = np.asanyarray(color_frame.get_data())
            color_msg = Image()
            color_msg.header = Header(stamp=timestamp, frame_id='camera_color_optical_frame')
            color_msg.height, color_msg.width = color_data.shape[:2]
            color_msg.encoding = 'bgr8'
            color_msg.is_bigendian = 0
            color_msg.step = color_msg.width * 3
            color_msg.data = color_data.tobytes()
            self.pub_color.publish(color_msg)

            # Camera info
            info_msg = CameraInfo()
            info_msg.header = Header(stamp=timestamp, frame_id='camera_color_optical_frame')
            info_msg.width = self.intrinsics.width
            info_msg.height = self.intrinsics.height
            info_msg.k = [
                float(self.intrinsics.fx), 0.0, float(self.intrinsics.ppx),
                0.0, float(self.intrinsics.fy), float(self.intrinsics.ppy),
                0.0, 0.0, 1.0
            ]
            self.pub_info.publish(info_msg)

        except Exception as e:
            self.get_logger().warn(f'Frame error: {str(e)[:50]}')

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main():
    rclpy.init()
    node = RSPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
