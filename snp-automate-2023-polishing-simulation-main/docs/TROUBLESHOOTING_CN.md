# 常见问题与处理记录

## `Get Current Joint State -> FAILED`

如果随后出现关节起始状态误差超出容差，先等待仿真关节状态发布，再重新批准扫描执行。`launch/test.launch.xml` 已为主要关节补充零位参数，减少该问题。

## `remove_scan_link` 或 `add_scan_link` service unreachable

这是扫描链接服务在节点重启或流程切换时尚未就绪造成的。不要连续快速点击；等待节点稳定后重新批准当前步骤。若持续不可达，执行：

```bash
./scripts/restart_demo.sh
```

然后从扫描流程重新开始。

## `Tool paths are empty`

通常是待加工区域太大、太小、选区不在重建网格上，或起点没有设置。清除当前 Polygon 后，在网格表面重新画一个较小的连续区域，设置 Start Point，再 Plan。

## `Start point has not been set`

必须在 TPP 面板/交互工具中明确点击一个起点；仅绘制多边形不会自动生成起点。

## `generate_motion_plan service unreachable`

说明规划服务节点没有启动或已退出。检查：

```bash
docker ps --filter name=snp_automate_2023_sim
docker logs --tail=100 snp_automate_2023_sim
```

必要时重启容器，再重新执行工具路径规划。

## RViz 卡顿或窗口关闭

执行 `./scripts/restart_demo.sh`。不要同时打开多个 RViz 或 SNPApplication 实例。

## `Approve Process Motion Execution -> RUNNING` 很久

这是仿真执行阶段，先观察机器人是否仍在运动及容器日志；如果没有任何状态变化，检查容器和规划服务，而不是反复点击。
