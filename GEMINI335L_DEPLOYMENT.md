# Gemini 330 系列仿真基线新电脑部署步骤

> 本部署文档运行的是仓库现有的 Gemini 335L 参数仿真基线，不代表最终硬件采购型号。实际部署前请根据 [`docs/project/CAMERA_SELECTION.md`](docs/project/CAMERA_SELECTION.md) 在奥比中光 Gemini 330 系列中完成选型。车辆静止时，真实多机采集应优先采用顺序触发以降低主动红外互扰。

本文从一台全新的 Ubuntu Linux 电脑开始，说明如何从 GitHub 下载项目、创建 Python 环境、运行四相机 Gemini 330 系列仿真基线，并打开网页查看器。所有命令默认在终端执行。

## 1. 系统要求

建议环境：

- Ubuntu 22.04 或 24.04，64 位。
- Python 3.10–3.12，推荐 Python 3.12。
- 至少 8 GB 内存；运行完整分辨率或多相机对照时建议 16 GB 以上。
- 能访问 GitHub 和 PyPI。
- 如果要让同一局域网其他电脑访问网页，服务器电脑和客户端必须在同一网段。

仿真不需要真实 Gemini 相机、机械臂、ROS 2 或 Gazebo。PyBullet 负责无窗口渲染，Open3D 负责点云、TSDF、ICP 和 Mesh。

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip build-essential
```

确认版本：

```bash
git --version
python3 --version
```

## 3. 从 GitHub 下载项目

仓库地址：<https://github.com/zqdbb/arm>

```bash
cd ~
git clone https://github.com/zqdbb/arm.git
cd arm
git checkout main
```

检查 Gemini 仿真文件是否存在：

```bash
test -f gemini335l_multicam_sim/simulate_8cam_reconstruction.py && echo "simulation source: OK"
test -f models/prius_hybrid/meshes/Hybrid.obj && echo "vehicle model: OK"
test -f gemini335l_multicam_sim/viewer.html && echo "viewer: OK"
```

如果仓库需要更新到最新版本：

```bash
git pull --ff-only origin main
```

## 4. 创建独立 Python 环境

必须在仓库根目录创建 `.venv`，因为 `run_validation.sh` 默认从这里查找 Python：

```bash
cd ~/arm
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

安装仿真依赖：

```bash
./.venv/bin/python -m pip install \
  open3d==0.19.0 \
  pybullet==3.2.7 \
  numpy \
  pillow \
  scipy \
  opencv-contrib-python
```

检查依赖：

```bash
./.venv/bin/python - <<'PY'
import cv2
import numpy
import open3d
import pybullet
import scipy

print("Open3D:", open3d.__version__)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("OpenCV:", cv2.__version__)
print("PyBullet: import OK")
PY
```

看到 `Open3D: 0.19.0` 且没有 traceback，即表示 Python 环境可用。之后的命令都使用 `./.venv/bin/python`，避免误用系统 Python。

## 5. 运行默认四相机仿真

一键生成理想深度和规格书推导噪声两组结果：

```bash
cd ~/arm
./gemini335l_multicam_sim/run_validation.sh
```

生成的目录：

```text
gemini335l_multicam_sim/output_4cam/
gemini335l_multicam_sim/output_4cam_spec_noise/
```

理想组不加入传感器深度噪声；噪声组按照 Gemini 335L 规格书中的深度精度和填充率构造近似噪声。默认分辨率是 640×400，TSDF 体素是 15 mm，车辆是仓库中的 Prius Hybrid 模型。

如果只运行一组：

```bash
./.venv/bin/python gemini335l_multicam_sim/simulate_8cam_reconstruction.py \
  --output gemini335l_multicam_sim/output_4cam \
  --depth-model ideal \
  --camera-layout four
```

带规格书噪声：

```bash
./.venv/bin/python gemini335l_multicam_sim/simulate_8cam_reconstruction.py \
  --output gemini335l_multicam_sim/output_4cam_spec_noise \
  --depth-model spec_noise \
  --noise-seed 335 \
  --camera-layout four
```

模拟相机定位/标定误差：

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

## 6. 运行 ChArUco 标定仿真

这一步模拟四台固定相机共同观测一块移动的 5×7 ChArUco 板，并估计相机外参：

```bash
cd ~/arm
./.venv/bin/python gemini335l_multicam_sim/simulate_charuco_calibration.py \
  --output gemini335l_multicam_sim/output_charuco_calibration_4cam \
  --vehicle-output gemini335l_multicam_sim/output_4cam
```

主要结果：

```text
gemini335l_multicam_sim/output_charuco_calibration_4cam/
├── charuco_calibration_report.json
├── merged_charuco_estimated.ply
├── charuco_board_5x7.png
└── captures/
```

`charuco_calibration_report.json` 用于检查外参误差；`merged_charuco_estimated.ply` 是按估计外参拼接的点云，后续可以作为现场扫描输入。

## 7. 运行车辆模板配准

此步骤把系统保存的标准车辆模板配准到现场点云。`--synthetic-pose` 仅用于仿真回归，会先施加一个已知车辆位姿，便于量化配准误差：

```bash
cd ~/arm/gemini335l_multicam_sim
../.venv/bin/python register_template_to_scan.py \
  --template ../models/prius_hybrid/meshes/Hybrid.obj \
  --scene output_4cam/merged_4cam.ply \
  --output output_registration_4cam \
  --synthetic-pose
cd ~/arm
```

配准流程为 FPFH/RANSAC 粗配准加多尺度点到面 ICP。主要输出：

```text
gemini335l_multicam_sim/output_registration_4cam/
├── registration_report.json
├── scene.ply
├── template_aligned.ply
└── template_visible_aligned.ply
```

真实扫描输入不使用 `--synthetic-pose`，而是将 `--scene` 替换为完成背景剔除和外参融合后的真实点云。

## 8. 运行喷涂路径迁移仿真

这一步模拟实际业务：已有标准车辆点云和模板坐标系中的喷涂 TCP 路径，现场点云配准后，将路径变换到现场世界坐标和机器人基座坐标。

```bash
cd ~/arm/gemini335l_multicam_sim
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
../.venv/bin/python simulate_paint_path_transfer.py \
  --output output_paint_path_transfer
cd ~/arm
```

主要输出：

```text
gemini335l_multicam_sim/output_paint_path_transfer/
├── paint_path_transfer_report.json
├── registration_report.json
├── reference_saved.ply
├── live_fused.ply
├── reference_aligned.ply
├── paint_path_template.json
├── paint_path_live_world.json
└── paint_path_robot_base.json / .csv
```

当前回归结果通常约为：车辆位姿误差 7.0 mm / 0.316°，路径迁移平均误差 14.2 mm，P95 约 18.8 mm。该输出只验证刚体坐标变换，不能直接发送给真实机械臂。

## 9. 启动网页查看器

网页必须从仿真目录提供静态文件，不要在仓库根目录直接启动，否则 `/viewer.html` 会返回 404：

```bash
cd ~/arm
python3 -m http.server 8877 \
  --bind 0.0.0.0 \
  --directory "$PWD/gemini335l_multicam_sim"
```

保持这个终端运行，然后在服务器本机打开：

```text
http://127.0.0.1:8877/viewer.html
```

获取服务器局域网 IP：

```bash
hostname -I
```

同一局域网其他电脑打开：

```text
http://服务器局域网IP:8877/viewer.html
```

例如服务器 IP 是 `192.168.100.69`：

```text
http://192.168.100.69:8877/viewer.html
```

如果服务器同时连接有线和无线网络，使用与客户端处于同一网段的那个 IP。客户端浏览器还需要能够访问 jsDelivr，因为查看器从 CDN 加载 Three.js。

### 后台运行网页服务

需要关闭终端后仍保持服务时：

```bash
cd ~/arm
nohup python3 -m http.server 8877 \
  --bind 0.0.0.0 \
  --directory "$PWD/gemini335l_multicam_sim" \
  > /tmp/gemini335l_viewer.log 2>&1 &
```

检查监听状态：

```bash
ss -ltnp | grep ':8877'
curl -I http://127.0.0.1:8877/viewer.html
```

停止服务：

```bash
pkill -f 'python3 -m http.server 8877'
```

如果 Ubuntu 防火墙启用，允许局域网访问端口：

```bash
sudo ufw allow from 192.168.100.0/24 to any port 8877 proto tcp
```

将 `192.168.100.0/24` 换成实际局域网网段。

## 10. 推荐复现顺序

首次部署按以下顺序执行：

```text
1. 安装系统依赖
2. git clone 仓库
3. 创建 .venv 并安装 Python 依赖
4. 运行 run_validation.sh
5. 运行 ChArUco 标定仿真
6. 运行模板车辆配准
7. 运行喷涂路径迁移
8. 绑定 0.0.0.0 启动网页
```

只想确认环境是否成功时，执行到第 4 步即可。只想查看已提交结果时，跳过重新生成，直接执行第 8 步。

## 11. 常见问题

### `ModuleNotFoundError: No module named open3d`

命令使用了错误的 Python。确认使用：

```bash
./.venv/bin/python -c 'import open3d; print(open3d.__version__)'
```

### `run_validation.sh` 找不到 Python

脚本默认寻找仓库根目录的 `.venv/bin/python`。请确认当前目录是仓库根目录，并重新执行第 4 步。

### 网页返回 404

HTTP 服务必须带有：

```bash
--directory ~/arm/gemini335l_multicam_sim
```

不能只在 `~/arm` 下运行不带 `--directory` 的服务。

### 局域网电脑无法访问

依次检查：

```bash
ss -ltnp | grep ':8877'
```

输出必须包含 `0.0.0.0:8877`，不能只有 `127.0.0.1:8877`。然后确认客户端与服务器在同一网段，并检查 `ufw` 是否放行 TCP 8877。

### 运行时间或内存不足

先使用默认 640×400 和 `voxel=0.015`。不要一开始使用 `--full-resolution`；降低分辨率或增大 voxel 会减少内存和计算时间。

## 12. 安全边界

本仿真不会控制真实相机、机械臂或转台。喷涂路径输出仅用于验证坐标变换，未验证真实机械臂 IK、碰撞、关节限位、喷枪控制和急停联锁，禁止直接发送给真实设备。

相关文档：

- [仿真功能和结果说明](GEMINI335L_SIMULATION_README.md)
- [Gemini 仿真子项目 README](gemini335l_multicam_sim/README.md)
- [总项目 README](README.md)
