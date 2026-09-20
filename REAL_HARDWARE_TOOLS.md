# 根目录真实硬件工具说明

本文件说明仓库根目录中与 Intel RealSense、ROS 2、机械臂 TF 和电动转台有关的工具。它们用于真机诊断，不属于 Gemini 335L 纯仿真。

## 硬件与软件前提

- Intel RealSense D435/D435i。
- ROS 2 Humble 环境。
- 已发布 `baselink -> camera_color_optical_frame` 的 TF。
- Y200RA60/兼容串口转台及对应 `turntable` Python 模块。
- Python 包：`rclpy`、`tf2_ros`、`cv_bridge`、`pyrealsense2`、`open3d`、`opencv-python`、`numpy`。

脚本中的容器路径默认为 `/workspace`，ROS 工作空间默认为 `/workspace/ros2_ws`。在宿主机运行时需要改成实际路径或自行设置环境。

## 数据与坐标链路

```text
D435 depth + color
  → depth 对齐到 color
  → /camera/aligned_depth_to_color/image_raw
  → /camera/color/image_raw
  → /camera/color/camera_info
  → TF: baselink <- camera_color_optical_frame
  → 转台物体位姿补偿
  → Open3D ScalableTSDFVolume
  → PLY Mesh
```

深度图使用 `16UC1` 毫米单位。相机内参来自当前 RealSense 流，不使用硬编码仿真参数。

## `rs_publisher.py`

功能：

- 以 1280×720、15 FPS 打开彩色和深度流。
- 把深度对齐到彩色光学坐标。
- 尝试设置视觉预设、发射器和激光功率。
- 应用空间滤波、时间滤波和孔洞填充。
- 把无效的大深度值重新置零。
- 发布 RGB、深度和 CameraInfo。

主要话题：

```text
/camera/color/image_raw
/camera/aligned_depth_to_color/image_raw
/camera/color/camera_info
```

启动前确认没有 RealSense Viewer 或其他进程占用设备。

## `mark_turntable_center.py`

功能：显示 RGB 图像，点击转台中心后读取附近有效深度，通过相机内参反投影，再用 TF 转换到 `baselink`。

输出可作为 `turntable_reconstruction.py --center x,y,z` 的初始值。点击位置必须有有效深度，且 TF 必须可用。

## `turntable_reconstruction.py`

功能：机械臂和相机保持不动，转台按给定角度旋转；脚本把转台角度转换为物体在零度坐标系中的逆运动，并将多视角 RGB-D 融合到 Open3D TSDF。

默认关键参数：

```text
串口               /dev/ttyUSB0
角度               0,30,...,330（12 视角）
转台速度           5000
TSDF voxel         10 mm
TSDF truncation    40 mm
深度范围           0.35–0.60 m
```

重要参数：

- `--center`：转轴上一点在 `baselink` 中的位置，单位米。
- `--axis`：转轴在 `baselink` 中的单位方向。
- `--direction`：实际转动方向与数学正方向的符号。
- `--crop_*`：图像裁剪区域。
- `--axis_radius_m`：只保留转轴附近指定半径内的点。
- `--icp`：使用 ICP 对每一帧位姿做小范围优化。
- `--dry-run`：不采集帧、不发送转台命令，用于检查 TF 和参数解析。

先执行：

```bash
python3 turntable_reconstruction.py --dry-run
```

确认 TF、设备和参数无误后再执行真实扫描。`--dry-run` 仍会初始化 ROS 2 并查询 TF，但不会启动 RealSense 或控制转台。

## 方向检查脚本

`run_direction_test_plus.sh` 和 `run_direction_test_minus.sh` 分别用 `+1`、`-1` 方向做 0°/180°测试。选择重影较小且几何方向正确的一组，再运行完整扫描。

`run_direction_crop4.sh` 使用 4 个视角，`run_direction_crop12.sh` 使用 12 个视角。脚本会实际转动设备，不提供交互确认。

## 运行前安全检查

1. 确认急停可触达。
2. 确认机械臂在扫描期间保持静止。
3. 确认转台和物体不会碰撞相机、机械臂或线缆。
4. 确认串口对应目标转台，而不是其他执行器。
5. 先检查 0°/180°方向，再进行 4 或 12 视角扫描。
6. 先低速、少视角验证裁剪范围和转轴中心。
7. 不要同时运行多个 RealSense 进程。

## 常见重建问题

### 多层重影

优先检查：

- `--direction` 正负号。
- 转轴中心 `--center`。
- 转轴方向 `--axis`。
- `world_to_camera` 与 `camera_to_world` 是否混用。
- 采集时机械臂是否移动。

### 尺寸不对

确认深度比例。RealSense 原始深度单位应通过设备读取，ROS 发布的当前脚本按毫米 `16UC1` 使用。

### 空洞或发散

检查深度有效率、反光/黑色表面、工作距离、裁剪区域、TSDF 截断距离和相邻视角重叠。

### 停止服务响应慢

仓库中的 `industrial_reconstruction.py` 已移除固定 30 帧延迟队列，并在诊断模式下限制为 100 帧，减少位姿错配和队列阻塞。
