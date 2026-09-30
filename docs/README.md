# 文档导航

这里集中放置仓库级说明，不放运行时源码和生成结果。新成员先看根目录 [`README.md`](../README.md)，再按任务进入对应子项目。

## 文档分类

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| `project/` | 项目规划、技术难点和阶段性方案报告 | [`project/README.md`](project/README.md) |
| `history/` | Git 提交顺序、每次更新的目的和当前主线 | [`history/PROJECT_HISTORY.md`](history/PROJECT_HISTORY.md) |
| `archive/` | 不参与运行的原始压缩包和本地安装包归档 | [`archive/README.md`](archive/README.md) |

## 当前主线文档

- [Gemini 335L 四相机仿真说明](../gemini335l_multicam_sim/README.md)
- [Gemini 330 系列相机选型建议](project/CAMERA_SELECTION.md)
- [Gemini 330 系列联网选型报告](project/GEMINI330_SELECTION_RESEARCH.md)
- [Gemini 335L 新电脑部署步骤](../GEMINI335L_DEPLOYMENT.md)
- [D435i TSDF 仿真说明](../d435i_tsdf_sim/README.md)
- [真实 D435i 工具和安全说明](../REAL_HARDWARE_TOOLS.md)
- [完整更新历史](history/PROJECT_HISTORY.md)

## 文档维护规则

1. 仓库级决策、阶段结论和更新顺序写在 `docs/` 或根目录 README。
2. 子项目的运行参数、依赖和输出定义写在子项目自己的 README。
3. 每次新增实验必须同时记录输入、参数、输出位置和结论，不能只提交截图。
4. 生成的 Mesh、日志、缓存和本机环境不作为新的根目录文件提交。
