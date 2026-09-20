# 3D 物体点云采集与重建系统

基于瑞尔曼 ECO65-B 机械臂 + Intel D435i 深度相机 + Y200RA60 转台的微型家具 3D 扫描系统。

## 硬件配置

| 设备 | 型号 | 说明 |
|------|------|------|
| 机械臂 | 瑞尔曼 ECO65-B | 眼在手上，采集时全程固定不动 |
| 深度相机 | Intel RealSense D435i | 固定在机械臂末端，斜俯视转台 |
| 转台 | Y200RA60 | 水平转台，直径 20cm，步进电机控制 |
| 物体 | 微型家具 | 椅子 / 桌子 / 柜子 |

## 环境依赖

```bash
pip install numpy opencv-python pyrealsense2 open3d scikit-learn
```

转台控制需要 `turntable.py`（与本脚本同目录）。

## 采集前准备

### 1. 启动机械臂驱动

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch rm_driver rm_eco65_driver.launch.py
```

### 2. 调节机械臂位姿

新开一个终端，运行位姿调节脚本：

```bash
cd ~/项目文件夹/real_scan
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
python3 robot.py
```

将机械臂调节到合适的斜俯视位姿（相机对准转台中心），调节完成后保持机械臂不动。

> 参考关节角（度）：`[-41.94, 80.476, -146.443, 22.351, 121.555, -120.882]`

## 快速开始

### 1. 采集 + 重建（一步完成）

```bash
# 采集椅子
python3 final.py --name chair

# 采集桌子
python3 final.py --name table

# 采集柜子
python3 final.py --name cabinet
```

运行后：
1. 弹出实时预览窗口，**黄色高亮**显示当前采集区域
2. 确认物体在黄色区域内，按**空格键**开始采集
3. 转台自动旋转，每 5° 采集一帧，共 72 帧（360°）
4. 采集完成后自动进行点云拼接重建
5. 弹出 3D 可视化窗口显示重建结果

### 2. 跳过采集，直接重建已有数据

```bash
python3 final.py --name chair --skip-capture
```

## 输出文件

采集和重建结果保存在 `output/<物体名>/` 目录下：

```
output/
├── chair/
│   ├── pcd_frames/
│   │   ├── frame_000.ply    # 第0帧(0°)
│   │   ├── frame_001.ply    # 第1帧(5°)
│   │   └── ...
│   └── chair_stitched.ply   # 最终重建结果
├── table/
│   └── ...
└── cabinet/
    └── ...
```

用 MeshLab 或 CloudCompare 打开 `.ply` 文件即可查看 3D 模型。

## 核心原理

### 旋转中心自动估计

每次采集的数据旋转中心可能不同（机械臂位置、背景剔除后坐标偏移等），代码会**自动估计**旋转中心：

1. 取帧 0 和帧 36（相差 180°）
2. 在 XY 平面搜索最优中心，使两帧对齐距离最小
3. 粗搜索 5mm 步长 + 精搜索 1mm 步长
4. 输出自动估计的中心坐标和对齐精度

### 分区域重建（椅子专用）

椅子结构特殊，靠背和腿需要不同处理：
- **靠背**（下方，Z < 3.5cm）：前 30 帧，≥3 帧一致 → 防止重叠重影
- **椅腿**（上方，Z ≥ 3.5cm）：前 50 帧，≥4 帧一致 → 保证密度

> 注意：Z 大的一端是椅腿（上方），Z 小的一端是靠背（下方）。

### 统一重建（桌子 / 柜子）

- 桌子：前 50 帧，≥4 帧一致
- 柜子：全部 72 帧，≥3 帧一致 + Y 方向镜像对称补齐侧面

### 背景剔除流程

每帧点云采集后自动进行背景剔除：

1. **体素下采样**到 2mm（减少计算量）
2. **DBSCAN 聚类**取最大簇（去掉零散噪点）
3. **RANSAC 去平面**（去掉转台平面）
4. **Y 方向百分位过滤**（去掉下方残留）
5. **X 方向左右过滤**（去掉两侧背景）

## 参数调整

在 `final.py` 的 `OBJECT_CONFIG` 中调整各物体参数：

### 裁剪框（空间范围）

```python
"x_min": -0.12, "x_max": 0.12,  # X方向左右范围(米)
"y_min": -0.12, "y_max": 0.12,  # Y方向上下范围(米)
"z_min": 0.18,  "z_max": 0.40,  # Z方向前后范围(米)
```

调整裁剪框可以控制采集区域，预览窗口的黄色高亮会实时显示。

### 背景剔除参数

```python
"y_percentile": 30,   # Y方向保留小于此分位数的点(上方)
"x_low": 10,          # X方向左边界分位数
"x_high": 90,         # X方向右边界分位数
"plane_ratio": 0.3,   # RANSAC平面占比阈值(超过则删除)
"plane_dist": 0.005,  # RANSAC平面距离阈值(米)
```

### 重建参数

```python
# 椅子分区域
"backrest_frames": 30,        # 靠背用前多少帧
"leg_frames": 50,             # 腿用前多少帧
"backrest_min_consensus": 3,  # 靠背最少几帧一致
"leg_min_consensus": 4,       # 腿最少几帧一致

# 桌子/柜子统一
"total_frames": 50,           # 用前多少帧
"min_consensus": 4,           # 最少几帧一致
"mirror_fill": True,          # 是否镜像补齐侧面(柜子用)
```

## 常见问题

### Q: 采集的点云包含转台/背景怎么办？

A: 调整裁剪框参数，缩小 `x_min/x_max/y_min/y_max/z_min/z_max`，预览窗口黄色高亮会显示采集区域，确保只有物体在黄色区域内。

### Q: 重建结果不像物体，是弧形/圆环？

A: 旋转中心估计错误。检查：
1. 帧 0 和帧 36 是否都有效（点数 > 50）
2. 物体是否在裁剪框内
3. 自动估计的中心是否合理（X 约 [-5, 5]cm，Y 约 [-30, -10]cm）

### Q: 椅子靠背有重叠/重影？

A: 减少靠背帧数 `backrest_frames`（如从 30 降到 20），或提高一致性阈值 `backrest_min_consensus`（如从 3 升到 4）。

### Q: 椅子腿糊成一块？

A: 减少腿的帧数 `leg_frames`（如从 50 降到 40），或提高一致性阈值 `leg_min_consensus`。

### Q: 柜子侧面没有点？

A: 柜子已启用 `mirror_fill: True`，会自动用 Y 方向镜像对称补齐侧面。如果仍不满意，可降低 `min_consensus`（如从 3 降到 2），但会引入噪声。

### Q: 采集速度慢？

A: 可降低 `N_AVG`（平均帧数，默认 3）或提高 `FPS`（默认 15），但会影响点云质量。

### Q: 相机被占用（Device or resource busy）？

A: 关闭其他使用 RealSense 相机的程序（如 preview_crop.py、Realsense Viewer 等）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `final.py` | 主程序（采集 + 重建一体化） |
| `turntable.py` | 转台控制模块（需同目录） |
| `output/` | 采集数据和重建结果输出目录 |

## 采集流程

1. 启动机械臂驱动：`ros2 launch rm_driver rm_eco65_driver.launch.py`
2. 运行 `python3 robot.py` 调节机械臂到斜俯视位姿，完成后保持不动
3. 将物体放在转台中心
4. 运行 `python3 final.py --name <物体>`
5. 预览窗口确认黄色高亮只包含物体
6. 按空格键开始采集
7. 等待 72 帧采集完成（约 5 分钟）
8. 自动重建并显示 3D 结果
9. 在 `output/<物体>/` 查看最终 PLY 文件
