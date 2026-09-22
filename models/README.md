# 仿真模型资产

## `prius_hybrid`

该目录包含用户新增的 Gazebo Toyota Prius Hybrid 模型：

- `model.sdf`：Gazebo 模型定义。
- `meshes/Hybrid.obj`：车辆主体、车窗、内饰和车轮网格。
- `meshes/*.mtl`、`*.png`：材质和纹理。

`model.sdf` 中声明的网格缩放为 `0.01`。Gemini 六相机仿真会读取该 OBJ，并将源坐标转换为项目统一的车辆坐标：`+X` 朝车头、`+Y` 朝车辆左侧、`+Z` 向上。模型包原始包围盒约为 `4.618 × 2.012 × 1.537 m`。

该模型现在是 [`gemini335l_multicam_sim`](../gemini335l_multicam_sim/README.md) 的默认车辆和模板匹配资产。模型来源和再分发许可应以原始 Gazebo 模型包的授权信息为准；仓库仅记录用户提供的本地模型资产。

## `office_cabinet`

这是原有家具喷涂仿真的简化柜体模型，与车辆模板匹配流程无关。
