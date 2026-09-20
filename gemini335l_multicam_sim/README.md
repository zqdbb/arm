# Gemini 335L 八固定相机仿真

该目录独立于原 `d435i_tsdf_sim`。它使用用户提供的 Gemini 335L 精简规格书参数，模拟上层四台、下层四台固定相机依次采集静止车辆，然后用固定外参拼接点云并进行 Open3D TSDF 融合。默认目标是 Khronos glTF Sample Assets 的高细节 Car Concept 模型，模型采用 CC BY 4.0，来源和许可证保存在 `assets/car_concept`。

场景中的八台相机使用奥比中光官方 `OrbbecSDK_ROS2` 仓库提供的 Gemini 335L/336L `base_link.STL` 和 Xacro 坐标定义，不是按外形尺寸手工近似的模型。官方源文件及 Apache 2.0 许可证保存在 `assets/gemini335l_official`。

默认快速验证使用规格书 1280×800 内参的精确 0.5 倍分辨率：640×400，`fx=fy=310`，`cx=320`，`cy=200`。加 `--full-resolution` 可运行 1280×800。

运行：

```bash
./run_validation.sh
```

查看：

```bash
cd ..
python3 -m http.server 8877 --bind 127.0.0.1
```

浏览器打开 `http://127.0.0.1:8877/gemini335l_multicam_sim/viewer.html`。

`output_ideal` 是精确外参、无传感器噪声的几何基线；`output_spec_noise` 是依据规格书精度上限构造的合成噪声对照。后者不是八台实机的逐台标定数据或实测噪声模型。
