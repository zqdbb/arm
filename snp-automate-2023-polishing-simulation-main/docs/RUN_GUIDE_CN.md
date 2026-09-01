# 运行指南（中文）

## 1. 启动

确保 Docker Desktop 已启动，并在 Ubuntu-24.04 WSL 中执行：

```bash
cd /mnt/c/Users/19256/snp-automate-2023-polishing-simulation
export DISPLAY=${DISPLAY:-:0}
chmod +x scripts/*.sh
./scripts/run_simulation.sh
```

容器使用以下仿真参数：

```text
SNP_SIM_ROBOT=true
SNP_SIM_VISION=true
SNP_BYPASS_EXECUTION=true
```

启动后通常会出现 RViz2 和 SNPApplication 窗口。若窗口被关闭，执行 `./scripts/restart_demo.sh`。

## 2. 扫描和重建

在 SNPApplication 中按顺序批准扫描运动规划、扫描执行和重建流程。日志出现以下结果时，扫描和重建已经完成：

```text
Start Reconstruction -> SUCCESS
Execute Scan Motion -> SUCCESS
Stop Reconstruction -> SUCCESS
```

随后检查：

```bash
ls -lh runtime/snp_home/snp/meshes/results_mesh.ply
```

该 PLY 是仿真扫描重建得到的工件网格。

## 3. TPP 工具路径

在 RViz 中切换到 TPP 操作：

1. 选择 Polygon/多边形区域工具；
2. 在工件表面圈选一个较小、连续的区域；
3. 清除错误选区后重新绘制，不要保留重叠或自交多边形；
4. 设置 TPP 的 **Start Point**（起点）；
5. 点击 Plan。

正常日志为：

```text
Plan Tool Paths -> SUCCESS
```

若提示 `Start point has not been set`，说明只画了区域但没有指定起点。

## 4. 运动规划与执行

继续批准：

```text
Approve Motion Plan Generation -> SUCCESS
GenerateMotionPlanService -> SUCCESS
Combine Trajectories -> SUCCESS
Approve Process Motion Execution
```

最后一个批准步骤进入 `RUNNING` 后，仿真机器人会执行轨迹；执行结束后应返回 `SUCCESS` 或流程进入下一个完成状态。

## 5. 查看结果

- RViz：查看机器人、扫描网格、工具路径和运动轨迹。
- Blender：打开 `runtime/results_mesh_view.blend`，或在 Blender 中导入 `runtime/snp_home/snp/meshes/results_mesh.ply`。
- GitHub 页面：README 中的两个 GIF 用于快速展示运行过程。


> 如果仓库放在其他目录，只需把上面的 `cd` 路径替换为实际路径。