# 验证结果

执行命令：

```bash
python simulate_tsdf.py --output output
```

输入是解析生成的无噪声深度（16 位 PNG，单位 mm），相机环绕物体 12 个等间隔位姿，使用 D435i 彩色流近似内参 320×240、fx=fy=280 px。没有加入传感器噪声、深度空洞、遮挡误差、颜色分割误差或位姿漂移。

`output/report.json`：

* 真值尺寸：0.240 × 0.180 × 0.200 m
* 重建包围盒：0.24745 × 0.18800 × 0.20229 m
* 绝对误差：7.45、8.00、2.29 mm
* 判定：`pass: true`（最大误差 ≤ 3×4 mm）

误差主要来自体素离散化和 marching-cubes 插值；这是无现实噪声条件下该融合流程的理想上限。若要直接测试项目的 Open3D TSDF，可在 Python ≤3.12 环境安装 `open3d`，将 `output/depth/*.png` 和对应相机位姿接入 `real_scan/tsdf_fusion.py` 的 `vol.integrate` 循环。
