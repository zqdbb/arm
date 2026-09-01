# 抛光仿真项目完整流程与运行说明

> 本文是 `snp-automate-2023-polishing-simulation` 仓库的项目交接文档，记录项目目标、软件组成、目录结构、启动方法、扫描重建、工具路径规划、运动规划、两种执行模式、结果查看和资源清理流程。

## 1. 项目概述

本项目基于 ROS Industrial 的 SNP Automate 2023，用于演示机器人抛光工艺的完整软件流程：

```text
仿真工作站启动
    ↓
机器人扫描运动规划
    ↓
仿真扫描执行
    ↓
工业重建生成工件网格
    ↓
在 RViz 中选择待加工区域
    ↓
TPP（Tool Path Planning）生成工具路径
    ↓
机器人运动规划
    ↓
仿真执行抛光轨迹
```

本仓库是原始项目的副本，主要用于学习、演示和复现实验流程。

### 重要边界

- 本项目默认使用 ROS 2 Jazzy 和 Docker。
- 本项目使用虚拟机器人和仿真视觉，不连接真实机器人。
- 默认执行器是软件仿真，不会产生真实抛光加工。
- 即使使用 ROS 2 Control 的 `mock_components/GenericSystem`，也只会驱动 RViz 中的虚拟机器人模型。
- 真实机器人执行需要另外配置机器人驱动、控制器、通信网络、坐标标定和安全策略，本仓库不包含这些内容。

## 2. 软件架构和主要组件

| 组件 | 作用 |
|---|---|
| ROS 2 Jazzy | 提供节点、话题、服务、Action 和启动系统 |
| Docker 镜像 | 提供 SNP Automate 2023 及其依赖，避免在主机上重复安装大量依赖 |
| RViz2 | 显示机器人、工件网格、工具路径、规划轨迹和执行状态 |
| SNPApplication | 在 RViz 中提供扫描、重建、TPP 和运动执行流程界面 |
| `reconstruction_sim_node` | 模拟扫描重建过程并输出重建网格 |
| `tool_path_planning_server` | 根据选区和起点生成工具路径 |
| `snp_planning_server` | 将工具路径转换为机器人运动轨迹 |
| `motoros2_simulator` | 模拟 MotoROS2 相关服务和机器人状态 |
| `execution_simulator` | 默认的简化执行器，只接受轨迹并返回成功，不播放完整关节运动 |
| `ros2_control_node` | 可选的 ROS 2 Control 模拟控制器 |
| `mock_components/GenericSystem` | 可选的模拟硬件，让 RViz 中的机器人响应关节轨迹 |

### 数据流

```text
scan_traj.yaml
    → 扫描运动规划和执行
    → reconstruction_sim_node
    → runtime/snp_home/snp/meshes/results_mesh.ply
    → RViz Polygon 选区 + Start Point
    → noether_ros_tool_path_planning_server
    → 工具路径
    → snp_motion_planning_node
    → FollowJointTrajectory
    → execution_simulator 或 ros2_control 模拟控制器
```

## 3. 仓库目录说明

```text
snp-automate-2023-polishing-simulation/
├── config/                         SNP、RViz、控制器和 TPP 参数
├── docker/                         Docker Compose、Dockerfile 和容器配置
├── docs/                           运行指南、故障排查、结果和交接文档
├── launch/                         ROS 2 启动文件
├── meshes/                         工作台、工具、机器人和工件网格资源
├── motoros2/                       MotoROS2 仿真配置
├── runtime/                        挂载到容器的运行时目录和输出结果
│   ├── results_mesh_view.blend     Blender 查看文件
│   └── snp_home/snp/meshes/
│       └── results_mesh.ply        仿真扫描重建结果
├── scripts/
│   ├── run_simulation.sh           首次启动脚本，会拉取镜像
│   └── restart_demo.sh             重启已有仿真容器
├── urdf/                           机器人和 ros2_control 描述
├── README.md                       项目入口说明
└── package.xml                     ROS 2 包元数据
```

## 4. 运行环境

推荐环境：

- Windows 11 + WSL 2；
- Ubuntu 24.04 WSL 发行版；
- ROS 2 Jazzy；
- Docker Desktop，并开启 WSL 2 集成；
- 支持 WSLg/X11 图形转发；
- 足够的磁盘空间保存 Docker 镜像和运行缓存。

本次交接使用的 Docker 镜像为：

```text
ghcr.io/ros-industrial-consortium/snp_automate_2023:jazzy-master
```

不需要额外安装 ROS 2 Humble。Ubuntu 24.04 对应 ROS 2 Jazzy，本仓库也按 Jazzy 镜像验证。

## 5. 第一次启动仿真

### 5.1 启动 Docker Desktop

先启动 Docker Desktop，确认 Docker 引擎已经运行。然后打开 Ubuntu-24.04 WSL：

```powershell
wsl.exe -d Ubuntu-24.04
```

### 5.2 进入仓库

如果仓库位于 Windows 的 `C:` 盘：

```bash
cd /mnt/c/Users/19256/snp-automate-2023-polishing-simulation
```

如果仓库位于 `E:` 盘，则把路径替换为：

```bash
cd /mnt/e/coding/shixi/snp-automate-2023-polishing-simulation
```

### 5.3 配置图形显示并启动

```bash
export DISPLAY=${DISPLAY:-:0}
export QT_QPA_PLATFORM=xcb
export QT_X11_NO_MITSHM=1
chmod +x scripts/*.sh
./scripts/run_simulation.sh
```

`run_simulation.sh` 会完成以下操作：

1. 创建运行结果目录；
2. 拉取 Docker 镜像；
3. 创建并启动 `snp_automate_2023_sim` 容器；
4. 启动 RViz2、扫描重建、TPP、运动规划和仿真控制节点；
5. 输出容器状态和最近日志。

正常情况下会出现 RViz2 窗口。SNPApplication 的流程面板集成在 RViz 界面中，不一定表现为独立的 Windows 应用窗口。

### 5.4 检查容器和节点

在 WSL 中检查容器：

```bash
docker ps --filter name=snp_automate_2023_sim
```

检查 ROS 2 节点：

```bash
docker exec snp_automate_2023_sim bash -lc \
  'source /opt/ros/jazzy/setup.bash && ros2 node list'
```

至少应能看到以下类型的节点：

```text
/rviz
/reconstruction_sim_node
/tool_path_planning_server
/snp_planning_server
/motoros2_simulator
```

默认模式下还会看到：

```text
/motion_execution_simulator
```

ROS 2 Control 模拟硬件模式下会看到：

```text
/controller_manager
/joint_trajectory_position_controller
/joint_state_broadcaster
```

## 6. 两种机器人执行模式

### 6.1 默认模式：简化执行模式

仓库 `docker/compose.sim.yml` 默认配置为：

```yaml
SNP_SIM_ROBOT: "true"
SNP_SIM_VISION: "true"
SNP_BYPASS_EXECUTION: "true"
```

此时启动的是 `execution_simulator`。它的作用是接收运动执行请求并返回成功，适合快速验证：

- 扫描流程；
- 工业重建；
- TPP 工具路径；
- 运动规划服务；
- 整体行为树流程。

在这个模式下，日志出现 `SUCCESS` 不代表真实机器人已经逐点播放轨迹，RViz 中可能主要显示虚影、轨迹预览或状态变化。

### 6.2 ROS 2 Control 模拟硬件模式

如果希望 RViz 中的实体机器人模型按照关节轨迹运动，使用仓库提供的专用启动脚本：

```bash
./scripts/run_ros2_control_simulation.sh
```

该脚本通过 `docker/compose.ros2_control.override.yml` 将：

```yaml
SNP_BYPASS_EXECUTION: "true"
```

覆盖为：

```yaml
SNP_BYPASS_EXECUTION: "false"
```

也可以手动使用 Compose 启动：

```bash
docker rm -f snp_automate_2023_sim 2>/dev/null || true
docker compose -f docker/compose.sim.yml -f docker/compose.ros2_control.override.yml up -d
docker compose -f docker/compose.sim.yml -f docker/compose.ros2_control.override.yml ps
```

该模式会启动：

```text
ros2_control_node
joint_trajectory_position_controller
joint_state_broadcaster
mock_components/GenericSystem
```

可用的关节轨迹 Action 为：

```text
/joint_trajectory_position_controller/follow_joint_trajectory
```

切换模式后建议从 TPP 规划开始重新执行，不要直接复用上一次容器中的旧执行状态。执行时观察 RViz 中不透明的机器人模型，而不是 `preview` 命名空间中的半透明规划预览。

执行完成后如需恢复仓库默认简化模式，停止当前容器后运行 `./scripts/run_simulation.sh`，即可使用基础 Compose 配置重新启动。

> 该模式仍然是虚拟机器人运动，不连接真实 HC10，也不会产生真实抛光力、接触和材料去除。

## 7. SNPApplication 完整操作流程

### 阶段 A：初始化

1. 确认 RViz 已打开，机器人、工作台和工具模型可见；
2. 在 SNPApplication 中点击初始化/开始流程；
3. 等待节点和服务就绪；
4. 如果界面没有在前台，使用 `Alt + Tab` 切换到 RViz。

### 阶段 B：生成扫描运动并执行扫描

依次批准扫描相关步骤：

```text
Initialize Flags
Approve Scan Motion Plan Creation
Load Trajectory From File
Approve Scan Execution
```

扫描流程中常见的成功日志包括：

```text
Start Reconstruction -> SUCCESS
Execute Scan Approach Motion -> SUCCESS
Execute Scan Motion -> SUCCESS
Stop Reconstruction -> SUCCESS
Remove Scan Link -> SUCCESS
Add Scan Link -> SUCCESS
```

如果第一次出现：

```text
Get Current Joint State -> FAILED
Failed to find a joint state
```

先等待关节状态发布，再重新批准当前扫描执行步骤。不要连续快速点击多个批准按钮。

扫描和重建完成后检查结果：

```bash
ls -lh runtime/snp_home/snp/meshes/results_mesh.ply
```

### 阶段 C：确认重建模型

在 RViz 中查看扫描得到的工件网格。结果文件为：

```text
runtime/snp_home/snp/meshes/results_mesh.ply
```

也可以在 Blender 中打开：

```text
runtime/results_mesh_view.blend
```

或在 Blender 里通过导入 PLY 的方式打开 `results_mesh.ply`。

### 阶段 D：绘制 TPP 加工区域

1. 在 RViz 中选择 TPP/Polygon 多边形工具；
2. 在重建网格表面圈选一个较小、连续的待加工区域；
3. 尽量避免选区过大、过窄、跨越空洞或包含自交线；
4. 如果轨迹画乱，清除当前 Polygon/选区后重新绘制；
5. 在 TPP 工具中明确设置 **Start Point**；
6. 确认起点位于网格表面，再点击 Plan。

只画多边形而没有设置起点，会出现：

```text
Start point has not been set
```

### 阶段 E：生成工具路径

点击工具路径规划相关按钮，等待：

```text
Plan Tool Paths -> SUCCESS
```

如果显示：

```text
Tool paths are empty! Check tool path planner parameters.
```

或：

```text
Error invoking tool path planner on mesh at index 0 in TPP pipeline.
Start point has not been set
```

请先清除选区，重新在网格表面选择更小的连续区域，并重新设置 Start Point。

### 阶段 F：生成机器人运动规划

工具路径成功后，依次批准运动规划生成：

```text
Approve Motion Plan Generation
GenerateMotionPlanService
Combine Trajectories
```

期望日志为：

```text
Plan Tool Paths -> SUCCESS
Approve Motion Plan Generation -> SUCCESS
GenerateMotionPlanService -> SUCCESS
Combine Trajectories -> SUCCESS
```

如果 `GenerateMotionPlanService` 失败，优先检查：

- 机器人当前关节状态是否已发布；
- TPP 工具路径是否为空；
- 选区是否太大或产生不可达姿态；
- 工具路径是否超出机器人工作空间；
- 规划服务节点是否仍在运行。

### 阶段 G：批准并执行抛光轨迹

当界面出现：

```text
Approve Process Motion Execution -> RUNNING
```

点击对应的批准/确认按钮。

在默认简化执行模式中，该步骤主要验证执行请求是否被仿真执行器接受；在 ROS 2 Control 模拟硬件模式中，应观察 RViz 中机器人模型沿规划轨迹运动。

成功判断包括：

- 执行步骤不再反复回到失败状态；
- 运动执行服务返回 `SUCCESS`；
- ROS 2 Control 模式下机器人关节状态和 RViz 模型发生变化；
- 轨迹显示和工具头运动方向符合规划结果。

## 8. 结果查看

### 8.1 RViz 中查看

RViz 可查看：

- 机器人当前姿态；
- `preview` 规划预览；
- 扫描/重建网格；
- TPP 选区；
- 工具路径；
- 机器人运动轨迹；
- 工具头和工作台的相对位置。

### 8.2 Blender 中查看

推荐打开：

```text
runtime/results_mesh_view.blend
```

也可以导入：

```text
runtime/snp_home/snp/meshes/results_mesh.ply
```

PLY 文件是扫描重建结果，不是 Blender 工程文件。若直接双击打不开，应先启动 Blender，再使用 `File → Import → Stanford (.ply)` 导入。

### 8.3 GitHub 页面中查看

仓库 README 中的 GIF 用于快速展示流程。完整运行说明、故障排查和资源清单位于：

- `docs/PROJECT_WORKFLOW_CN.md`：本文，完整项目流程；
- `docs/RUN_GUIDE_CN.md`：简明运行步骤；
- `docs/TROUBLESHOOTING_CN.md`：常见故障处理；
- `docs/DEMO_RESULT.md`：已验证的 Demo 结果；
- `docs/SNP_DEMO_RESOURCE_LEDGER.md`：镜像、容器、缓存和路径清单。

## 9. 常见故障处理

### `remove_scan_link` 或 `add_scan_link` service unreachable

这是服务节点尚未就绪、流程切换过快或节点重启造成的。先等待几秒，再重新批准当前步骤；持续失败时重启 Demo：

```bash
./scripts/restart_demo.sh
```

然后从扫描流程重新开始。

### `Get Current Joint State -> FAILED`

等待 `/robot_joint_states` 或关节状态发布后重试。不要在关节状态还未发布时连续点击执行。

### `Tool paths are empty`

重新绘制较小的连续区域，并确认区域确实落在重建网格上；同时设置 Start Point。

### `Start point has not been set`

说明没有为 TPP 工具路径设置起点。画完 Polygon 后必须单独设置起点。

### `generate_motion_plan service unreachable`

检查容器是否还在运行：

```bash
docker ps --filter name=snp_automate_2023_sim
docker logs --tail=100 snp_automate_2023_sim
```

必要时重启容器并重新规划。

### RViz 卡顿、窗口关闭或打开多个实例

只保留一个仿真容器和一个 RViz。重启前先执行：

```bash
docker rm -f snp_automate_2023_sim 2>/dev/null || true
docker compose -f docker/compose.sim.yml up -d
```

Windows 下可使用 `Alt + Tab` 切换窗口，不要重复启动多个脚本。

### `Approve Process Motion Execution -> RUNNING` 长时间不结束

先确认当前使用哪种执行模式：

- `SNP_BYPASS_EXECUTION=true`：可能只是简化执行器返回状态，不会播放完整机器人关节动画；
- `SNP_BYPASS_EXECUTION=false`：检查 `ros2_control_node` 和 `joint_trajectory_position_controller` 是否运行。

可查看节点：

```bash
docker exec snp_automate_2023_sim bash -lc \
  'source /opt/ros/jazzy/setup.bash && ros2 node list'
```

不要反复点击同一个批准按钮，以免提交重复执行请求。

## 10. 重启和停止

### 重启整个 Demo

```bash
cd /path/to/snp-automate-2023-polishing-simulation
./scripts/restart_demo.sh
```

Windows + WSL 示例：

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc \
  'cd /mnt/c/Users/19256/snp-automate-2023-polishing-simulation && ./scripts/restart_demo.sh'
```

### 停止容器但保留镜像

```bash
docker compose -f docker/compose.sim.yml down
```

### 删除容器和镜像

确认不再需要本地仿真环境后：

```bash
docker rm -f snp_automate_2023_sim 2>/dev/null || true
docker image rm ghcr.io/ros-industrial-consortium/snp_automate_2023:jazzy-master
```

不要手动删除 Docker Desktop 的 WSL 虚拟磁盘，也不要直接执行 `docker system prune -a`，以免误删其他项目资源。

## 11. 项目输出与交接内容

主要输出：

```text
runtime/snp_home/snp/meshes/results_mesh.ply
runtime/results_mesh_view.blend
```

主要配置：

```text
config/app.rviz
config/snp_automate.xml
config/snp_automate.btproj
config/tpp.yaml
config/controllers.yaml
config/scan_traj.yaml
config/calibration.yaml
```

主要启动入口：

```text
scripts/run_simulation.sh
scripts/restart_demo.sh
docker/compose.sim.yml
launch/start.launch.xml
```

仓库不包含以下内容：

- Docker 镜像本体；
- Docker Desktop 的 WSL 虚拟磁盘；
- 完整 ROS 日志缓存；
- 原始录屏 MP4；
- 真实机器人驱动和硬件配置。

删除本机大文件前，应先确认 GitHub 仓库、README、本文档、GIF、配置文件和最终结果文件已经可以正常访问。

## 12. 最简复现清单

接手人员只需要按以下顺序操作：

```text
1. 安装/启动 Docker Desktop
2. 打开 Ubuntu-24.04 WSL
3. 进入仓库目录
4. 运行 scripts/run_simulation.sh
5. 在 RViz/SNPApplication 中初始化
6. 批准扫描规划并执行扫描
7. 等待 Start Reconstruction -> SUCCESS
8. 在 RViz 中绘制小范围 Polygon
9. 设置 TPP Start Point
10. Plan Tool Paths -> SUCCESS
11. Approve Motion Plan Generation
12. GenerateMotionPlanService -> SUCCESS
13. Combine Trajectories -> SUCCESS
14. Approve Process Motion Execution
15. 在 RViz 中查看规划/执行结果
16. 在 Blender 中查看 results_mesh.ply 或 results_mesh_view.blend
```

如果需要观察 RViz 中机器人模型沿轨迹运动，先把 `docker/compose.sim.yml` 中的 `SNP_BYPASS_EXECUTION` 改为 `"false"`，再重启并重新执行第 8 步之后的规划流程。
