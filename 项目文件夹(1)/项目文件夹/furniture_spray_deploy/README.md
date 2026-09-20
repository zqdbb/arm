# 家具喷漆 Web 服务 — 部署包

## 系统要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **内存 (RAM)** | **4 GB** | **8 GB** |
| 硬盘空间 | 500 MB | 1 GB |
| Python | 3.8+ | 3.10 |
| 操作系统 | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |

### 内存说明

服务同时加载 3 个模型：

| 模型 | 参数量 | 权重占用 | 推理时内存 |
|------|--------|---------|-----------|
| ResNet-18 (图片分类) | 1170 万 | 45 MB | ~300 MB (含 PyTorch) |
| PointNeXt-S (柜子分割) | 89 万 | 3 MB | ~100 MB (含 openpoints) |
| PointCNN (椅子/桌子分割) | ~500 万 | 95 MB | ~1 GB (含 TensorFlow 1.x runtime) |

峰值内存场景（TF1 + PyTorch 同时加载）：约 **2-3 GB**，加上操作系统和 web 服务开销，**4 GB 勉强可用，8 GB 稳定运行**。

> 注：PointCNN 使用 TensorFlow 1.x 兼容模式，TF runtime 本身占用较大（500 MB - 1 GB）。
> 如果目标机器只有 4 GB 内存，可以只使用 PointNeXt (柜子)，禁用 PointCNN (椅子/桌子) 以节省 ~1 GB。

## 快速启动

```bash
# 1. 解压
tar xzf furniture_spray_deploy.tar.gz
cd furniture_spray_deploy

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
bash run.sh

# 4. 浏览器访问
# http://localhost:5000
```

## 目录结构

```
furniture_spray_deploy/
├── webapp/                    # Flask web 应用
│   ├── app.py                 # 主服务 (API + 前端)
│   ├── pipeline_worker.py     # 管线桥接层
│   ├── pointnext_inference.py # PointNeXt 柜子分割 (24类)
│   ├── pointcnn_inference.py  # PointCNN 椅子/桌子分割
│   ├── path_planner.py        # 喷涂路径规划
│   └── templates/
│       └── index.html         # Web 前端页面
├── pretrained_models/         # 预训练模型权重
│   └── partnet/
│       ├── tf1_compat.py
│       ├── pointcnn_seg_partnet_sem_seg_150_Chair_1/  # 椅子 (6类)
│       └── pointcnn_seg_partnet_sem_seg_150_Table_1/  # 桌子 (11类)
├── models/                    # 图片分类模型
│   ├── furniture_cls_weights.pth   # ResNet-18 权重
│   └── furniture_classes.json      # 类别映射
├── log/checkpoint/            # PointNeXt 模型权重
├── cfgs/partnet/              # PointNeXt 模型配置
├── openpoints/                # PointNeXt 依赖库 (纯 Python)
├── spray_pipeline.py          # 几何管线 (回退用)
├── auto_run.py                # 点云加载工具
├── requirements.txt           # Python 依赖
├── run.sh                     # 一键启动脚本
└── README.md
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 界面 |
| `/api/health` | GET | 健康检查 |
| `/api/classify-image` | POST | 上传照片 → 分类 (cabinet/chair/table) |
| `/api/upload` | POST | 上传点云 (.ply/.npy/.xyz/.pcd/.pts) |
| `/api/segment-ai` | POST | AI 部件分割 |
| `/api/generate-path` | POST | 生成喷涂路径 |
