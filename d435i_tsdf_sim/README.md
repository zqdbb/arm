# D435i 无噪声 RGB-D / TSDF 仿真验证

该目录是一个独立、无需 ROS/Open3D 的回归测试。`download_d435i_urdf.sh` 下载 Intel RealSense 官方 D435i URDF；`simulate_tsdf.py` 用 D435i 彩色相机内参，在 12 个环绕视角对一个已知尺寸的盒状物体做解析光线投射，生成理想深度图并融合 TSDF，最后提取 PLY 网格和误差报告。

若需要真正的仿真器成像，运行 `simulate_d435i.py`：它在 PyBullet 中加载 `urdf/d435i_sim.urdf`，通过 TinyRenderer 的 `getCameraImage` 深度缓冲生成 RGB-D，再执行同样的 TSDF 融合。

运行：

```bash
cd d435i_tsdf_sim
bash download_d435i_urdf.sh
python simulate_tsdf.py --output output
```

默认物体尺寸为 0.24 x 0.18 x 0.20 m，体素 4 mm。`output/report.json` 中 `pass` 为 true 表示重建包围盒误差小于 2 个体素且表面采样误差合格。该测试特意不加入深度噪声、孔洞、位姿误差或背景，可作为项目 TSDF 算法的理想上限基线。

## Open3D TSDF 验证

`open3d_tsdf_reconstruct.py` 将 PyBullet 中的模拟 D435i RGB-D 帧直接送入项目同款的 Open3D `ScalableTSDFVolume`：

```bash
cd /home/azh/桌面/arm/d435i_tsdf_sim
/home/azh/桌面/arm/.venv/bin/python open3d_tsdf_reconstruct.py \
  --output output_chair_open3d --views 40 --rings 5
```

输出包括 `color/`、`depth/`、`poses.json`、`report.json` 和 `tsdf_mesh.ply`。D435 上限测试使用 1280×720、水平 FOV 87°、50 mm 基线、1/32 像素视差量化、1 mm 体素和 4 mm 截断距离，并在导出前移除孤立碎片；报告中的 `tsdf_backend` 明确为 `Open3D ScalableTSDFVolume`。
