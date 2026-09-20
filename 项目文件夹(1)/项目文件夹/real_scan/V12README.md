# V12 家具重建管线

## 概述

V12 以 `scan_recon_v12.py` 为总入口，支持椅子、桌子和柜子。一次转台扫描同时生成 RGB-D、逐帧点云和 `final.py` 拼接点云；YOLO 抠图供 Hyper3D 生成完整网格，拼接点云提供真实尺寸约束。

```text
scan_recon_v12.py
  -> final.py 单圈采集 RGB-D + PLY
  -> final.py 拼接 <name>_stitched.ply
  -> YOLO 分割并生成白底多视图
  -> Hyper3D 生成完整三角网格
  -> Blender 清理并按 stitched 点云缩放
  -> output/<name>/<name>_v12.obj/.ply
```

Hyper3D 不直接接收 PLY。最终网格表面来自 Hyper3D，`final.py` 的点云只负责真实宽、深、高和坐标约束。

## 模型分工

| 类别 | 分割模型 | 目标类名 |
|---|---|---|
| chair | `yolov8n-seg.pt` | `chair` |
| table | `yolov8n-seg.pt` | `dining table` |
| cabinet | `custom_model/weights/best.pt` | `cabinet` |

默认直接使用 YOLO segmentation mask。只有显式传入 `--sam-refine` 才加载 `sam_vit_b_01ec64.pth`，且 SAM 结果必须通过与 YOLO mask 的一致性检查。

## 依赖与安装

### 硬件

- Intel RealSense D435i，建议直连 USB 3.x 接口
- Y200RA60 电动转台 + 1SC 控制器，RS232/CH340 串口
- Blender 4.x；可以和采集程序在同一台电脑，也可以运行在另一台电脑

### Ubuntu 系统依赖

本项目当前在 Python 3.10 下验证。Ubuntu 22.04 可以直接执行：

```bash
sudo apt update
sudo apt install -y \
  python3 python3-venv python3-pip python3-dev \
  build-essential libgl1 libglib2.0-0 libusb-1.0-0 \
  usbutils v4l-utils
```

Ubuntu 24.04 默认 Python 3.12；如果 `pyrealsense2` 或 Open3D 安装失败，建议使用 Python 3.10。可以通过 deadsnakes PPA 安装：

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 --version
```

这是第三方 PPA；受管控环境应改用管理员批准的 Python 3.10 软件源。安装后用 `python3.10 -m venv .venv` 创建环境，不要混用多个 Python 的 `pip`。

### Python 虚拟环境

在项目根目录创建独立环境：

```bash
cd /你的路径/real_scan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

安装运行必需依赖：

```bash
python -m pip install \
  pyrealsense2 \
  ultralytics \
  opencv-python \
  numpy \
  open3d \
  scikit-learn \
  pyserial
```

当前主机已经验证过的版本如下。新主机遇到兼容性问题时，可以用这些版本复现环境：

```bash
python -m pip install \
  pyrealsense2==2.58.3.10794 \
  ultralytics==8.4.120 \
  opencv-python==5.0.0.93 \
  numpy==1.26.4 \
  open3d==0.19.0 \
  scikit-learn==1.7.2 \
  pyserial==3.5
```

Ultralytics 会安装 PyTorch。通常让它选择与新主机匹配的版本即可，不建议从旧主机硬拷贝 CUDA/PyTorch。如果没有可用 NVIDIA 驱动，YOLO 会自动使用 CPU，只是推理速度较慢。

只有使用 `--sam-refine` 时才需要安装 SAM：

```bash
python -m pip install segment-anything==1.0
```

并把 `sam_vit_b_01ec64.pth` 放在项目根目录。不使用 `--sam-refine` 时不需要安装 SAM，也不需要复制这个 375 MB 权重。

### 模型文件

以下文件不会由当前程序自动下载，新主机必须复制到相同的项目相对路径：

```text
real_scan/
├── yolov8n-seg.pt
└── custom_model/
    └── weights/
        └── best.pt
```

检查模型和类别：

```bash
source .venv/bin/activate
python - <<'PY'
from ultralytics import YOLO
print('通用模型:', YOLO('yolov8n-seg.pt').names)
print('自训练模型:', YOLO('custom_model/weights/best.pt').names)
PY
```

自训练模型必须包含 `chair`、`table`、`cabinet`；当前流程实际使用其中的 `cabinet`。

### 串口权限

把当前用户加入 `dialout` 组，然后注销并重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

重新登录后检查转台串口：

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
python - <<'PY'
import serial.tools.list_ports
for port in serial.tools.list_ports.comports():
    print(port.device, port.description)
PY
```

如果实际端口不是 `/dev/ttyUSB0`，运行时传入：

```bash
python3 scan_recon_v12.py --name chair --turntable-port /dev/ttyUSB1
```

不要依赖 `/dev/ttyUSB0` 永远不变；拔插设备或增加其他 USB 串口后编号可能变化。

### RealSense 检查

先验证 Python 能识别 D435i：

```bash
source .venv/bin/activate
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
print('RealSense 数量:', len(ctx.devices))
for dev in ctx.devices:
    print(dev.get_info(rs.camera_info.name),
          dev.get_info(rs.camera_info.serial_number))
PY
```

如果数量为 0：

1. 确认使用 USB 3.x 数据线和接口，不要只接供电线。
2. 执行 `lsusb` 检查 Intel RealSense 是否出现。
3. 安装对应 Ubuntu 版本的 Intel librealsense udev rules，并重新插拔相机。
4. 确认没有其他程序正在占用相机。

### Blender 安装

可以从 Blender 官网安装 Blender 4.x，Ubuntu 也可使用：

```bash
sudo snap install blender --classic
blender --version
```

随后在 Blender 中安装项目根目录的 `blender_mcp_addon.py`：

1. `Edit -> Preferences -> Add-ons -> Install from Disk`
2. 选择 `blender_mcp_addon.py`
3. 启用 `Interface: Blender MCP`
4. 在 3D View 按 `N`，打开 BlenderMCP 面板

如果 addon 报告缺少 `requests`，给 Blender 自带的 Python 安装：

```bash
blender --background --python-expr \
  "import ensurepip; ensurepip.bootstrap(); import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','requests'])"
```

### 安装后检查

```bash
source .venv/bin/activate
python -c "import cv2,numpy,open3d,sklearn,serial,pyrealsense2,ultralytics; print('Python 依赖正常')"
python3 -m py_compile final.py scan_recon_v12.py test_scan_recon_v12.py
python3 -m unittest discover -s . -p 'test_scan_recon_v12.py' -v
```

测试应显示 10 项通过。离线测试不会连接转台、相机、Blender 或 Hyper3D。

## Blender 和 Hyper3D

1. 在 Blender 4.x 中安装并启用 `blender_mcp_addon.py`。
2. 在 BlenderMCP 面板选择 `MAIN_SITE` 并配置 Hyper3D API key。当前本地 base64 图片上传不支持 FAL_AI 模式。
3. 启动 BlenderMCP Server，默认监听 `localhost:9876`。
4. API key 保存在 Blender addon，不写入本项目脚本或日志。

## 运行

```bash
cd /你的路径/real_scan
source .venv/bin/activate

# 椅子完整流程
python3 scan_recon_v12.py --name chair

# 桌子/柜子
python3 scan_recon_v12.py --name table
python3 scan_recon_v12.py --name cabinet

# 串口不是默认值
python3 scan_recon_v12.py --name chair --turntable-port /dev/ttyUSB1

# 复用 output/<name> 已有采集，不访问相机或转台
python3 scan_recon_v12.py --name chair --skip-capture

# 只校验数据、运行抠图并计算点云尺寸，不连接 Blender/Hyper3D
python3 scan_recon_v12.py --name chair --skip-capture --no-blender

# 读取旧的 V12 数据，最终结果仍写到 output/chair
python3 scan_recon_v12.py --name chair --skip-capture \
  --input-dir output/v12/chair --no-blender

# 已有 Blender 对象 chair_v12 时跳过收费的 Hyper3D 生成
python3 scan_recon_v12.py --name chair --skip-capture --skip-hyper3d

# 远程 Blender
python3 scan_recon_v12.py --name chair --blender-host 192.168.1.100
```

如果 `output/<name>` 已有采集数据，完整采集会默认拒绝覆盖。需要重采时使用：

```bash
python3 scan_recon_v12.py --name chair --overwrite-capture
```

旧的 `color/`、`depth/`、`pcd_frames/`、manifest 和 stitched PLY 会先移动到 `output/<name>/history/<timestamp>/`，不会直接删除。

## 更换主机或迁移部署

项目中的业务路径都基于 `Path(__file__).parent`，所以项目可以放到新主机的任意目录，不需要把代码中的 `/home/xie/...` 改成新用户名。进入新的项目目录运行即可：

```bash
cd /新主机上的路径/real_scan
source .venv/bin/activate
python3 scan_recon_v12.py --name chair
```

### 情况一：相机、转台和 Blender 全部接在新主机

需要完成：

1. 复制项目代码以及两个必要模型权重。
2. 按“依赖与安装”创建新的虚拟环境；不要复制旧主机的 `.venv`。
3. 配置新主机的串口权限并确认实际串口名。
4. 确认 D435i 能被 `pyrealsense2` 识别。
5. 在新主机 Blender 中重新安装 BlenderMCP addon。
6. 在 BlenderMCP 中重新选择 `MAIN_SITE`、填写 Hyper3D API key 并启动 Server。
7. 如果相机、支架或转台的相对位置改变，重新标定旋转轴。

完整运行示例：

```bash
python3 scan_recon_v12.py \
  --name chair \
  --turntable-port /dev/ttyUSB1 \
  --blender-host localhost \
  --blender-port 9876
```

### 情况二：新主机负责采集，Blender 在另一台电脑

采集机安装 Python、RealSense 和串口依赖；Blender 电脑安装 Blender 4.x、BlenderMCP addon，并配置 Hyper3D API key。

当前 addon 默认只监听 Blender 电脑自己的 `localhost`。推荐使用 SSH 端口转发，不需要把 BlenderMCP 端口直接暴露到局域网：

```bash
# 在采集机执行；把 blender-user 和 IP 改成 Blender 电脑的信息
ssh -L 9876:localhost:9876 -N blender-user@192.168.1.100
```

保持该终端运行，再开一个终端执行：

```bash
cd /新主机上的路径/real_scan
source .venv/bin/activate
python3 scan_recon_v12.py --name chair \
  --turntable-port /dev/ttyUSB0 \
  --blender-host localhost --blender-port 9876
```

也可以通过环境变量设置默认 Blender 地址：

```bash
export BLENDER_HOST=localhost
export BLENDER_PORT=9876
python3 scan_recon_v12.py --name chair
```

命令行 `--blender-host/--blender-port` 优先用于本次运行；环境变量适合写进新主机自己的 shell 启动配置。不要把 API key 写进项目代码或提交到版本控制。

### 情况三：只把已有扫描数据移到新主机离线重跑

至少复制对应类别的以下内容：

```text
output/<name>/
├── color/
├── depth/
├── pcd_frames/
├── capture_manifest.json
└── <name>_stitched.ply
```

然后执行：

```bash
python3 scan_recon_v12.py --name chair --skip-capture --no-blender
```

需要重新调用 Hyper3D 时去掉 `--no-blender`。只做离线处理不需要连接 D435i 和转台，但仍需要 YOLO 权重；调用 Hyper3D 还需要 BlenderMCP 和 API key。

如果迁移的是旧目录 `output/v12/chair`，使用：

```bash
python3 scan_recon_v12.py --name chair --skip-capture \
  --input-dir /迁移后的路径/output/v12/chair --no-blender
```

### 新主机需要检查或修改的参数

| 项目 | 修改方式 | 什么时候需要改 |
|---|---|---|
| 项目路径 | 不改代码，进入新目录运行 | 项目放到任意新路径时 |
| 转台串口 | `--turntable-port /dev/...` | 新主机串口编号变化时 |
| Blender 地址 | `--blender-host` / `BLENDER_HOST` | Blender 不在采集机时 |
| Blender 端口 | `--blender-port` / `BLENDER_PORT` | addon 不使用 9876 时 |
| Hyper3D key | BlenderMCP 面板重新填写 | 每台新 Blender 主机都要配置 |
| YOLO 权重 | 复制到项目相对路径 | 每台运行分割的主机都需要 |
| 转台每度脉冲 | `config.py` 的 `TURNTABLE_PULSES_PER_DEG` | 实际旋转角度不准时 |
| 旋转轴方向 | `final.py` 的 `ROTATION_AXIS` | 相机/支架/转台相对姿态改变时 |
| 旋转轴中心 | `final.py` 的 `ROTATION_CENTER` | 相机/支架/转台相对位置改变时 |

`config.py` 当前 `TURNTABLE_PULSES_PER_DEG = 400`。如果发出 360° 指令却没有准确转一圈，需要重新测量：

```text
新值 = 旧值 × 指令角度 / 实际转过角度
```

例如旧值 400，指令 360°，实际只转 350°，说明脉冲数不足，新值约为 `400 × 360 / 350 = 411`。如果实际转了 370°，新值约为 `400 × 360 / 370 = 389`。修改后先用低速、小角度验证，再测试整圈，避免撞线或过度旋转。



### 新主机验收清单

```bash
# 1. Python 依赖
python -c "import cv2,numpy,open3d,sklearn,serial,pyrealsense2,ultralytics; print('OK')"

# 2. 模型文件
ls -lh yolov8n-seg.pt custom_model/weights/best.pt

# 3. 相机
lsusb | grep -i realsense

# 4. 串口
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

# 5. 离线测试
python3 -m unittest discover -s . -p 'test_scan_recon_v12.py' -v

# 6. Blender 端口（启动 BlenderMCP Server 后）
python - <<'PY'
import socket
s = socket.create_connection(('localhost', 9876), timeout=3)
s.close()
print('BlenderMCP 端口正常')
PY
```

## 输出

```text
output/chair/
├── color/                    # 72 帧对齐彩色 PNG
├── depth/                    # 72 帧 uint16 原始深度 PNG
├── pcd_frames/               # final.py 逐帧点云
├── views/                    # 提交给 Hyper3D 的白底抠图
│   └── views_manifest.json
├── capture_manifest.json     # 内参、深度比例、帧号和角度
├── chair_stitched.ply        # final.py 拼接点云，无三角面
├── chair_dims.json           # stitched 点云稳健尺寸
├── chair_v12.obj             # Hyper3D + 点云尺寸约束网格
├── chair_v12.ply             # 最终三角网格
└── log_v12.json
```

桌子和柜子使用同样结构及对应文件名前缀。

## 质量和失败条件

- RGB、深度和 PLY 帧必须编号一致；manifest 未完成时停止。
- YOLO 至少需要两个合格且角度分离的 mask，少于两个时不会请求 Hyper3D，避免浪费额度。
- 尺寸使用 stitched 点云 2%/98% 稳健范围；XY 使用稳健内点的最小面积旋转矩形，避免斜放造成包围盒膨胀。
- 最终 PLY 必须包含顶点和三角面；点云不会通过 Poisson 强行变成最终表面。
- `--surface-fit` 是默认关闭的实验选项，只对点云附近顶点做低权重、小位移贴合。

## 常见问题

`模型未找到`：确认两个 `.pt` 文件位于上面的固定路径。

`合格抠图视角不足`：检查物体是否完整进入画面、类别是否正确、光照是否足够；脚本不会回退提交未抠图原图。

`Hyper3D 未启用`：在 BlenderMCP 面板检查 addon、API key 和 Server 状态。

`Connection refused`：确认 BlenderMCP Server 已启动，且 `--blender-host/--blender-port` 与实际监听地址一致。

`已有采集数据`：使用 `--skip-capture` 复用，或使用 `--overwrite-capture` 归档后重采。


