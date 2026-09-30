# Gemini 330 系列仿真复现说明

> 当前代码使用 Gemini 335L 规格作为仿真参数基线，最终实际设备型号不预先锁定为 335L。对于静止车辆，正式多机采集建议顺序触发，避免多台主动红外相机同时工作产生互扰。硬件选型依据见 [`docs/project/CAMERA_SELECTION.md`](docs/project/CAMERA_SELECTION.md)。

本文档专门说明 `gemini335l_multicam_sim`：使用四台固定的 Gemini 330 系列相机基线对车辆进行 RGB-D 仿真采集，融合点云，验证 TSDF 重建，并将系统保存的车辆模板和喷涂路径迁移到现场车辆坐标。

这是软件链路仿真，不是四台真实相机的 SDK 采集结果。相机内参来自用户提供的 Gemini 335L 规格书典型值；真实部署时必须替换为每台设备从 SDK/EEPROM 读取并经过标定的参数。

## 1. 仿真目标

当前默认方案验证以下完整流程：

```text
Prius 模板 Mesh
  -> 四台固定相机渲染 RGB-D
  -> 各相机点云转换到工位世界坐标
  -> Open3D ScalableTSDFVolume / 多视角点云融合
  -> 现场融合点云
  -> 模板点云配准，估计车辆位姿
  -> 将模板坐标系中的喷涂 TCP 路径变换到现场和机器人基座
```

当前默认相机布局是车辆四角高位 `cam01`–`cam04`。脚本仍支持 `six` 和 `eight` 布局，但它们只用于覆盖率和历史方案对照，不是当前网页和默认结果。

## 2. 仓库位置和主要文件

从仓库根目录看，相关文件如下：

```text
gemini335l_multicam_sim/
├── simulate_8cam_reconstruction.py   # RGB-D 渲染、点云融合和 TSDF
├── simulate_charuco_calibration.py   # 四相机 ChArUco 外参仿真
├── register_template_to_scan.py      # FPFH/RANSAC + 多尺度 ICP 模板配准
├── simulate_paint_path_transfer.py   # 车辆定位和喷涂路径迁移
├── run_validation.sh                 # 默认理想/噪声两组重建
├── viewer.html                       # Three.js 结果查看器
├── gemini335l_spec.json              # 相机规格和仿真参数
├── assets/gemini335l_official/       # 官方 Gemini 335L/336L 外观模型
├── output_4cam/                      # 理想深度四相机结果
├── output_4cam_spec_noise/           # 规格书推导噪声结果
└── output_paint_path_transfer/       # 端到端配准和路径迁移结果
```

默认车辆模型为仓库根目录的 `models/prius_hybrid/meshes/Hybrid.obj`。仿真统一使用米，深度图使用 16-bit PNG 保存，深度单位为 1 mm。

## 3. 环境准备

已验证环境：Ubuntu、Python 3.12、Open3D 0.19、PyBullet 3.2.7、NumPy、Pillow。建议使用仓库根目录的独立虚拟环境：

```bash
cd /home/azh/桌面/arm
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install \
  open3d==0.19.0 \
  pybullet==3.2.7 \
  numpy \
  pillow \
  scipy \
  opencv-contrib-python
```

说明：

- `open3d` 用于 RGB-D、点云、TSDF、ICP 和 Mesh 输出。
- `pybullet` 用于车辆和相机的无窗口渲染。
- `opencv-contrib-python` 用于 ChArUco 仿真。
- `scipy` 用于姿态和路径变换。
- 如果机器已经有可用环境，可以把下面命令中的 `./.venv/bin/python` 替换成对应 Python；不要混用系统包和虚拟环境包。

## 4. 一键复现默认重建

在仓库根目录执行：

```bash
./gemini335l_multicam_sim/run_validation.sh
```

脚本会依次运行：

1. 理想深度：无传感器深度噪声，输出到 `gemini335l_multicam_sim/output_4cam/`。
2. 规格书噪声：按工作距离、深度精度上限和填充率构造近似噪声，输出到 `gemini335l_multicam_sim/output_4cam_spec_noise/`。

如果需要单独重跑一组：

```bash
./.venv/bin/python gemini335l_multicam_sim/simulate_8cam_reconstruction.py \
  --output gemini335l_multicam_sim/output_4cam \
  --depth-model ideal \
  --camera-layout four

./.venv/bin/python gemini335l_multicam_sim/simulate_8cam_reconstruction.py \
  --output gemini335l_multicam_sim/output_4cam_spec_noise \
  --depth-model spec_noise \
  --noise-seed 335 \
  --camera-layout four
```

默认参数为 640×400、`fx=fy=310 px`、TSDF voxel `15 mm`、截断距离 `60 mm`。使用规格书原始分辨率时：

```bash
./.venv/bin/python gemini335l_multicam_sim/simulate_8cam_reconstruction.py \
  --output gemini335l_multicam_sim/output_4cam_full_resolution \
  --depth-model ideal \
  --camera-layout four \
  --full-resolution
```

可用 `--voxel` 和 `--trunc` 调整 TSDF：

```text
--voxel 0.015     体素边长，单位 m
--trunc 0.060     TSDF 截断距离，单位 m
--pose-noise      加入相机外参误差
--vehicle-model prius_hybrid|car_concept|procedural
--camera-layout four|six|eight
```

加入相机定位误差的示例：

```bash
./.venv/bin/python gemini335l_multicam_sim/simulate_8cam_reconstruction.py \
  --output gemini335l_multicam_sim/output_4cam_pose_noise \
  --depth-model spec_noise \
  --pose-noise \
  --pose-translation-sigma-mm 3.0 \
  --pose-rotation-sigma-deg 0.10 \
  --pose-seed 336 \
  --camera-layout four
```

## 5. 输出和验收

每个重建输出目录重点查看：

| 文件 | 含义 |
| --- | --- |
| `color/cam01.png` … `cam04.png` | 各相机 RGB 图像 |
| `depth/cam01.png` … `cam04.png` | 16-bit 深度图，单位 mm |
| `pointclouds/cam*_world.ply` | 单相机世界坐标点云 |
| `merged_4cam.ply` | 四相机合并点云 |
| `tsdf_mesh.ply` / `tsdf_mesh.obj` | TSDF 输出 Mesh |
| `ground_truth_vehicle.ply` | 米制车辆真值表面 |
| `poses.json` | 相机中心、注视点和实际外参 |
| `report.json` | 参数、尺寸、表面误差和完整度 |

已提交的 640×400 Prius 基线大致为：

| 指标 | 理想深度 | 规格书推导噪声 |
| --- | ---: | ---: |
| 重建 Mesh 顶点数 | 124,958 | 184,205 |
| 重建→真值平均距离 | 6.02 mm | 7.99 mm |
| 重建→真值 P95 距离 | 13.20 mm | 18.63 mm |
| 可见外表面 10 mm 完整度 | 99.02% | 95.84% |

这些指标只评价四个视角实际可见的外表面；车底、车内和完全遮挡区域没有被四相机观测，不应据此判断为重建失败。

## 6. ChArUco 外参标定仿真

车辆标定阶段被移除，虚拟 5×7 ChArUco 板在相机重叠区域移动。运行：

```bash
./.venv/bin/python gemini335l_multicam_sim/simulate_charuco_calibration.py \
  --output gemini335l_multicam_sim/output_charuco_calibration_4cam \
  --vehicle-output gemini335l_multicam_sim/output_4cam
```

主要输出：

```text
output_charuco_calibration_4cam/
├── charuco_board_5x7.png
├── captures/                         # 各组虚拟采集图像
├── charuco_calibration_report.json   # 角点和外参误差
├── merged_charuco_estimated.ply      # 按估计外参拼接的车辆点云
└── charuco_board_trajectory.ply      # 标定板采样轨迹
```

该步骤验证的是“ChArUco 外参 -> 多相机统一坐标系”，不是直接用标定板重建车辆。真实系统中应把脚本中的典型内参替换为每台相机的 SDK 参数，并以工位基准定义世界坐标。

## 7. 单独运行模板车辆配准

模板配准输入是现场四相机融合点云，算法为：

```text
模板 Mesh -> 均匀采样点云 -> FPFH/RANSAC 粗配准 -> 多尺度点到面 ICP
```

运行可量化的已知位姿回归：

```bash
cd gemini335l_multicam_sim
../.venv/bin/python register_template_to_scan.py \
  --template ../models/prius_hybrid/meshes/Hybrid.obj \
  --scene output_4cam/merged_4cam.ply \
  --output output_registration_4cam \
  --synthetic-pose
cd ..
```

`--synthetic-pose` 会先给场景点云施加已知的平移/旋转，再检查配准估计与真值的差异。真实扫描时去掉该参数，并将 `--scene` 改成已经完成背景剔除、外参变换和融合的现场点云。

主要输出：

- `output_registration_4cam/scene.ply`：模拟现场点云。
- `output_registration_4cam/template_aligned.ply`：完整模板配准结果。
- `output_registration_4cam/template_visible_aligned.ply`：与现场点云距离不超过阈值的可见模板表面。
- `output_registration_4cam/registration_report.json`：RANSAC、ICP、覆盖率和位姿误差。

## 8. 端到端喷涂路径迁移

这个步骤对应实际业务：系统已有标准车辆点云和模板坐标系中的喷涂 TCP 路径，现场相机得到融合点云后，先求车辆位姿，再变换路径。

运行：

```bash
cd gemini335l_multicam_sim
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../.venv/bin/python simulate_paint_path_transfer.py \
  --output output_paint_path_transfer
cd ..
```

默认现场车辆相对模板平移 `(0.55, -0.32, 0.04) m`、偏航 `8°`。现场点云默认来自 `output_charuco_calibration_4cam/merged_charuco_estimated.ply`，并加入 1.5 mm 的模拟现场点云噪声。

坐标变换关系为：

```text
T_world_tcp = T_world_template @ T_template_tcp
T_base_tcp  = T_base_world @ T_world_tcp
```

主要输出：

| 文件 | 含义 |
| --- | --- |
| `reference_saved.ply` | 系统保存的标准车辆点云 |
| `live_fused.ply` | 模拟现场融合点云 |
| `reference_aligned.ply` | 配准到现场的标准车辆 |
| `paint_path_template.json` | 模板坐标系路径 |
| `paint_path_live_world.json` | 工位世界坐标路径 |
| `paint_path_robot_base.json/.csv` | 机器人基座坐标路径 |
| `paint_path_transfer_report.json` | 配准、路径误差和安全范围 |

已提交回归的典型结果：车辆位姿误差约 `7.0 mm / 0.316°`，ICP RMSE 约 `9.5 mm`，1,566 个 TCP 位姿的平均迁移误差约 `14.2 mm`、P95 约 `18.8 mm`，喷枪离车身约 `280 mm`。

该输出不能直接发送给真实机械臂。真实执行前还需加入对应机械臂 IK、关节限位、碰撞检测、轨迹平滑、喷枪时序、通信和急停联锁。

## 9. 网页查看器

不要直接双击 `viewer.html`，因为浏览器会限制本地文件加载和 ES Module。启动 HTTP 服务：

```bash
python3 -m http.server 8877 \
  --bind 0.0.0.0 \
  --directory gemini335l_multicam_sim
```

本机浏览器打开：

```text
http://127.0.0.1:8877/viewer.html
```

同一局域网其他电脑访问时，将 `127.0.0.1` 替换为运行服务电脑的局域网 IP，并确保防火墙放行 TCP `8877`。网页依赖 jsDelivr 的 Three.js，因此浏览器需要网络访问 CDN。

网页中的主要视图：

- 理想 TSDF Mesh。
- 规格噪声 TSDF Mesh。
- 四相机合并点云。
- 四相机现场车辆模板匹配。
- ChArUco 标定场景。
- 喷涂路径迁移和虚拟喷枪播放。

颜色约定：橙色为现场融合点云，灰色为完整模板，蓝色为匹配后可见模板，绿色为模板路径，紫色为迁移后路径。

## 10. 复现顺序建议

首次验证建议按以下顺序执行：

```text
1. 安装 .venv 依赖
2. run_validation.sh
3. simulate_charuco_calibration.py
4. register_template_to_scan.py
5. simulate_paint_path_transfer.py
6. 启动 http.server 查看 viewer.html
```

如果只关心当前应用目标，至少要保留以下三个报告：

```text
output_charuco_calibration_4cam/charuco_calibration_report.json
output_registration_4cam/registration_report.json
output_paint_path_transfer/paint_path_transfer_report.json
```

## 11. 常见问题

### Python 找不到 Open3D

确认命令使用的是 `./.venv/bin/python`，并重新安装 `open3d==0.19.0`。不要只运行系统的 `python3`。

### 网页没有模型

确认通过 `http://` 访问，并检查浏览器能否访问 `https://cdn.jsdelivr.net`。修改输出后需要刷新页面，查看器会读取固定的 `output_4cam`、`output_4cam_spec_noise` 和 `output_paint_path_transfer` 目录。

### 点云或 Mesh 有重影

优先检查 `poses.json` 中的外参方向、米/毫米单位和相机坐标系。Open3D TSDF 需要 `world_to_camera`；把 `camera_to_world` 直接传入会造成严重错位。

### 车顶有空洞

四台高位相机只覆盖其可见外表面。可以调整上方相机高度和注视点，或运行六/八相机布局做覆盖率对照；真实相机还需考虑玻璃、黑色车漆、反光和无效深度。

### 配准结果看起来只有一部分模板

`template_visible_aligned.ply` 只保留现场点云附近的模板表面，车底、背面和遮挡区域会被过滤。完整模板在 `template_aligned.ply`。

## 12. 仿真边界

当前仿真没有完整模拟双目匹配、红外投射、镜头畸变、环境光、透明/反光/黑色材质、多路径、真实 SDK 时间同步或真实设备逐台标定。因此它适合验证：

- 坐标系和尺度是否一致。
- 四相机点云是否能进入 TSDF。
- 模板配准和路径刚体迁移的数学链路是否正确。
- 噪声和外参误差对结果的敏感性。

它不能单独证明真实车辆现场一定达到相同精度。

## 13. 相关文档

- [当前仿真子项目说明](gemini335l_multicam_sim/README.md)
- [四相机 ChArUco 实机标定方法](gemini335l_multicam_sim/CHARUCO_CALIBRATION.md)
- [总项目 README](README.md)
