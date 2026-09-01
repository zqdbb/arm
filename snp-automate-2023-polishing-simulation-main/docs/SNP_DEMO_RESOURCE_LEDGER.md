# SNP Automate 2023 – resource ledger

> 本文件记录原始仿真机器和本交接仓库的资源位置。执行清理命令前，必须逐项确认路径；不要直接复制执行删除命令。

## Result
- Demo mode: simulated robot + simulated vision + bypassed physical execution.
- Status on August 19, 2026: running successfully in Docker container; RViz2 and the SNP planning/reconstruction nodes started.
- Verified nodes include `snp_planning_server`, `tool_path_planning_server`, `reconstruction_sim_node`, `motion_execution_simulator`, `motoros_simulator`, and `rviz`.
- Verified services include `/start_reconstruction`, `/stop_reconstruction`, `/plan_tool_path`, and `/generate_motion_plan`.

## Local project and runtime paths
- Project clone: `E:\coding\shixi\snp_automate_2023`
- Runtime/configuration/output directory: `E:\coding\shixi\snp_automate_2023_runtime`
- Simulated output mount: `E:\coding\shixi\snp_automate_2023_runtime\snp_home\snp\meshes`
- Compose file: `E:\coding\shixi\snp_automate_2023_runtime\compose.sim.yml`
- Run helper: `E:\coding\shixi\snp_automate_2023_runtime\run_simulation.sh`

## Docker resources
- Application image: `ghcr.io/ros-industrial-consortium/snp_automate_2023:jazzy-master`
- Image digest: `sha256:0e25537a9d6ec8944121462a2140cc92f9b5106bb29dfecb360f8e5227dbb569`
- Image ID: `0e25537a9d6e`
- Docker image disk usage: about `13.6GB` (compressed content about `3.49GB`)
- Container: `snp_automate_2023_sim` (currently running)
- Container mode: host network, privileged, WSLg display forwarding

## Docker storage location
Docker Desktop is configured to store internal WSL data below `E:\WSL\DockerDesktopData\DockerDesktopWSL`. Do **not** manually delete files there. Remove the named container/image through Docker so Docker can reclaim layers safely.

## ROS decision
Ubuntu 24.04 already has ROS 2 Jazzy at `/opt/ros/jazzy`. The repository CI publishes/tests both Humble and Jazzy variants, so this run uses the Jazzy prebuilt image. ROS 2 Humble was not installed and is not needed for this demo run.

## Stop and cleanup

### 本交接仓库位置
- Handover repository: `C:\Users\19256\snp-automate-2023-polishing-simulation`
- GitHub repository (待创建/上传后): `wjia051123-tech/snp-automate-2023-polishing-simulation`

以下旧路径是原始仿真机器上的实际位置，不是本仓库中的相对路径：
仅在确认不再需要本机仿真环境后，在 Ubuntu-24.04 WSL 中执行：
```bash
docker rm -f snp_automate_2023_sim
docker image rm ghcr.io/ros-industrial-consortium/snp_automate_2023:jazzy-master
rm -rf /mnt/e/coding/shixi/snp_automate_2023 /mnt/e/coding/shixi/snp_automate_2023_runtime
```

等价的 Windows PowerShell 命令（执行前再次确认每一个绝对路径）：
```powershell
wsl.exe -d Ubuntu-24.04 -- docker rm -f snp_automate_2023_sim
wsl.exe -d Ubuntu-24.04 -- docker image rm ghcr.io/ros-industrial-consortium/snp_automate_2023:jazzy-master
Remove-Item -LiteralPath 'E:\coding\shixi\snp_automate_2023','E:\coding\shixi\snp_automate_2023_runtime' -Recurse -Force
```

Do not run broad commands such as `docker system prune -a`, because that can remove unrelated Docker resources.


> GitHub 仓库只包含源码、配置、结果模型、文档和 GIF，不包含 Docker 镜像本身、Docker Desktop 虚拟磁盘、ROS 日志缓存或原始 MP4。