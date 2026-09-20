#!/usr/bin/env python3
"""
PointNeXt PartNet StorageFurniture 推理模块.

加载训练好的 PointNeXt-S 模型（24类），对 cabinet 点云进行逐点部件分割。
与 Flask 解耦，可独立测试。CPU-only 模式（无 CUDA）。

训练配置: cfgs/partnet/pointnext-s.yaml
  - feature_keys: pos → 仅 3 通道 xyz 输入
  - num_points: 2048
  - num_classes: 24
"""

import glob
import os
import sys
import numpy as np
import torch

# ── CPU 回退: 替代 CUDA pointnet2 操作 ──
_FAKE_SETUP_DONE = False


def _setup_fake_cuda():
    global _FAKE_SETUP_DONE
    if _FAKE_SETUP_DONE:
        return
    _FAKE_SETUP_DONE = True

    class FakePointnet2Cuda:
        @staticmethod
        def furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output):
            centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
            distance = torch.ones(B, N, device=xyz.device) * 1e10
            farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
            batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)
            for i in range(npoint):
                centroids[:, i] = farthest
                centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
                dist = torch.sum((xyz - centroid) ** 2, -1)
                distance = torch.min(distance, dist)
                farthest = torch.max(distance, -1)[1]
            output.copy_(centroids)

        @staticmethod
        def gather_points_wrapper(B, C, N, npoint, features, idx, output):
            idx_flat = torch.clamp(idx.unsqueeze(1).expand(-1, C, -1), 0, N - 1).long()
            output.copy_(features.gather(2, idx_flat))

        @staticmethod
        def gather_points_grad_wrapper(*args):
            pass

        @staticmethod
        def ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx):
            for b in range(B):
                for i in range(npoint):
                    d = torch.sum((xyz[b] - new_xyz[b, i]) ** 2, -1)
                    v = torch.nonzero(d < radius * radius).squeeze(-1)[:nsample]
                    if v.numel() > 0:
                        idx[b, i, :v.numel()] = v.long()

        @staticmethod
        def group_points_wrapper(B, C, N, npoint, nsample, features, idx, output):
            idx_flat = torch.clamp(
                idx.reshape(B, -1).unsqueeze(1).expand(-1, C, -1), 0, N - 1
            ).long()
            output.copy_(features.gather(2, idx_flat).reshape(B, C, npoint, nsample))

        @staticmethod
        def group_points_grad_wrapper(*args):
            pass

        @staticmethod
        def three_nn_wrapper(B, N, m, unknown, known, dist2, idx):
            for b in range(B):
                d = torch.cdist(unknown[b], known[b])
                top_d, top_i = d.topk(3, dim=1, largest=False)
                dist2[b].copy_(top_d)
                idx[b].copy_(top_i.int())

        @staticmethod
        def three_interpolate_wrapper(B, c, m, n, features, idx, weight, output):
            for b in range(B):
                for j in range(c):
                    i_flat = idx[b].long().clamp(0, m - 1)
                    gathered = features[b, j][i_flat]
                    output[b, j] = (gathered * weight[b]).sum(dim=1)

        @staticmethod
        def three_interpolate_grad_wrapper(B, c, n, m, grad_out, idx, weight, grad_features):
            for b in range(B):
                for j in range(c):
                    i_flat = idx[b].long().clamp(0, m - 1)
                    grad_features[b, j].scatter_add_(
                        0, i_flat[:, 0], grad_out[b, j] * weight[b, :, 0]
                    )
                    grad_features[b, j].scatter_add_(
                        0, i_flat[:, 1], grad_out[b, j] * weight[b, :, 1]
                    )
                    grad_features[b, j].scatter_add_(
                        0, i_flat[:, 2], grad_out[b, j] * weight[b, :, 2]
                    )

    _sys = sys
    _sys.modules.setdefault("pointnet2_batch_cuda", FakePointnet2Cuda)
    _sys.modules.setdefault("chamber", type("FakeChamfer", (), {}))

    try:
        import openpoints.models.layers.subsample as _sub

        _sub.furthest_point_sample = lambda xyz, npoint: (
            FakePointnet2Cuda.furthest_point_sampling_wrapper(
                xyz.shape[0], xyz.shape[1], npoint, xyz, None,
                torch.zeros(xyz.shape[0], npoint, dtype=torch.long, device=xyz.device)
            )
        )
    except Exception:
        pass


_setup_fake_cuda()

# ── 标签名和语义组 ──
PARTNET_LABEL_NAMES = [
    "background",           # 0
    "countertop",           # 1
    "shelf",                # 2
    "frame_vertical_bar",   # 3
    "back_panel",           # 4
    "top_panel",            # 5
    "vertical_side_panel",  # 6
    "frame_horizontal_bar", # 7
    "vertical_front_panel", # 8
    "bottom_panel",         # 9
    "vertical_divider_panel",  # 10
    "drawer_back",          # 11
    "drawer_bottom",        # 12
    "drawer_side",          # 13
    "drawer_front",         # 14
    "drawer_handle",        # 15
    "base_bottom_panel",    # 16
    "base_side_panel",      # 17
    "foot",                 # 18
    "wheel",                # 19
    "caster_stem",          # 20
    "hinge",                # 21
    "door_handle",          # 22
    "cabinet_door_surface", # 23
]

# 语义组 → 颜色映射
SEMANTIC_GROUP_COLORS = {
    "body":    [0.3, 0.6, 1.0],   # 蓝色 — 柜体面板
    "door":    [1.0, 0.5, 0.2],   # 橙色 — 门板
    "drawer":  [0.4, 0.8, 0.3],   # 绿色 — 抽屉
    "shelf":   [1.0, 0.8, 0.1],   # 黄色 — 搁板
    "base":    [0.6, 0.4, 0.2],   # 棕色 — 底座/脚
    "handle":  [1.0, 0.3, 0.6],   # 粉色 — 拉手
    "hinge":   [0.7, 0.7, 0.7],   # 灰色 — 铰链
    "frame":   [0.5, 0.3, 0.8],   # 紫色 — 框架条
    "countertop": [0.1, 0.7, 0.9],  # 青色 — 台面
    "background": [0.2, 0.2, 0.2],  # 深灰 — 背景
}


def _get_semantic_group(label_idx):
    """返回 label 的语义组名和颜色."""
    if label_idx in {4, 5, 6, 8, 9, 10}:
        return "body", SEMANTIC_GROUP_COLORS["body"]
    if label_idx == 23:
        return "door", SEMANTIC_GROUP_COLORS["door"]
    if label_idx in {11, 12, 13, 14, 15}:
        return "drawer", SEMANTIC_GROUP_COLORS["drawer"]
    if label_idx == 2:
        return "shelf", SEMANTIC_GROUP_COLORS["shelf"]
    if label_idx in {16, 17, 18, 19, 20}:
        return "base", SEMANTIC_GROUP_COLORS["base"]
    if label_idx == 22:
        return "handle", SEMANTIC_GROUP_COLORS["handle"]
    if label_idx == 21:
        return "hinge", SEMANTIC_GROUP_COLORS["hinge"]
    if label_idx in {3, 7}:
        return "frame", SEMANTIC_GROUP_COLORS["frame"]
    if label_idx == 1:
        return "countertop", SEMANTIC_GROUP_COLORS["countertop"]
    return "background", SEMANTIC_GROUP_COLORS["background"]


# 中文显示名
PARTNET_LABEL_NAMES_CN = {
    0: "背景", 1: "台面", 2: "搁板", 3: "竖框条", 4: "背板",
    5: "顶板", 6: "侧板", 7: "横框条", 8: "前竖板", 9: "底板",
    10: "竖隔板", 11: "抽屉背", 12: "抽屉底", 13: "抽屉侧",
    14: "抽屉面", 15: "抽屉拉手", 16: "底座底板", 17: "底座侧板",
    18: "脚", 19: "轮子", 20: "轮轴", 21: "铰链",
    22: "门拉手", 23: "门板",
}

# ── 模型封装 ──


class PointNeXtPartNetInference:
    """PointNeXt PartNet 推理器（StorageFurniture, 24 类）"""

    def __init__(self, config_path, checkpoint_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        from openpoints.utils import EasyConfig
        from openpoints.models.build import build_model_from_cfg
        from openpoints.utils.ckpt_util import load_checkpoint

        cfg = EasyConfig()
        cfg.load(config_path, recursive=True)
        if cfg.model.get("in_channels", None) is None:
            cfg.model.in_channels = cfg.model.encoder_args.in_channels
        self.cfg = cfg

        self.model = build_model_from_cfg(cfg.model).to(self.device)
        load_checkpoint(self.model, checkpoint_path)
        self.model.eval()

        n_params = sum(p.nelement() for p in self.model.parameters())
        print(f"[PointNeXt] 已加载模型: {n_params / 1e6:.2f}M 参数, "
              f"设备={self.device}, 类别={cfg.num_classes}")

    @torch.no_grad()
    def predict(self, points, num_points=2048):
        """
        对 cabinet 点云预测逐点部件标签.

        输入:
          points: (N, 3) numpy float32, 原始 xyz（已归一化或未归一化均可）
          num_points: 采样点数（默认 2048，匹配训练配置）

        输出:
          pred_labels: (N,) numpy int32 — 每个点的标签 0-23
          label_names: list[str] — 24 个标签英文名
          regions: list[dict] — 按标签分组的区域列表
        """
        N = points.shape[0]
        pts_np = points.astype(np.float32).copy()

        # 预处理: center + unit sphere normalize
        pts_np = pts_np - pts_np.mean(axis=0, keepdims=True)
        max_dist = np.max(np.sqrt(np.sum(pts_np ** 2, axis=-1)))
        if max_dist > 1e-10:
            pts_np = pts_np / max_dist

        pts_t = torch.from_numpy(pts_np).float().to(self.device)

        # 均匀采样到 2048 点（匹配训练输入大小）
        if N > num_points:
            idx = torch.linspace(0, N - 1, num_points).long()
            sampled_pts = pts_t[idx]
        else:
            sampled_pts = pts_t
            pad = torch.zeros(num_points - N, 3, device=self.device)
            sampled_pts = torch.cat([sampled_pts, pad], dim=0)

        # 构建输入: p0=(B,N,3), f0=(B,3,N), cls0=(B,1)
        p0 = sampled_pts.unsqueeze(0)                    # (1, 2048, 3)
        f0 = p0.transpose(1, 2).contiguous()              # (1, 3, 2048)
        cls0 = torch.tensor([[0]], device=self.device, dtype=torch.long)  # StorageFurniture

        logits = self.model(p0, f0, cls0)                 # (1, 24, 2048)
        sampled_labels = logits.argmax(dim=1).squeeze(0)  # (2048,)

        # 将采样点的标签映射回原始点云（最近邻）
        if N > num_points:
            sampled_pts_np = sampled_pts.cpu().numpy()
            pred_full = np.zeros(N, dtype=np.int32)
            sampled_lbls_np = sampled_labels.cpu().numpy()
            for i in range(N):
                dists = np.sum((sampled_pts_np - pts_np[i]) ** 2, axis=1)
                pred_full[i] = sampled_lbls_np[np.argmin(dists)]
        elif N < num_points:
            pred_full = sampled_labels[:N].cpu().numpy().astype(np.int32)
        else:
            pred_full = sampled_labels.cpu().numpy().astype(np.int32)

        regions = self._group_into_regions(points, pred_full)
        return pred_full, PARTNET_LABEL_NAMES, regions

    def _group_into_regions(self, points, pred_labels, min_ratio=0.002):
        """按预测标签分组，每个标签生成一个 region."""
        N = len(points)
        unique_labels = np.unique(pred_labels)
        regions = []

        for lbl in unique_labels:
            if lbl == 0:  # 跳过 background
                continue
            mask = pred_labels == lbl
            region_pts = points[mask]
            coverage = len(region_pts) / N

            if coverage < min_ratio:
                continue

            group_name, color = _get_semantic_group(int(lbl))
            cn_name = PARTNET_LABEL_NAMES_CN.get(int(lbl), f"part_{lbl}")

            # PCA 表面类型估计: planar/curved/skeleton
            if len(region_pts) >= 5:
                centered = region_pts - region_pts.mean(0)
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                normal = eigvecs[:, 0]
                lam0, lam1, lam2 = eigvals[0], eigvals[1], eigvals[2]
                sum_lam = lam0 + lam1 + lam2 + 1e-10
                extent = np.sqrt(lam2)

                # 点偏离平面的中位数 / 整体尺度 → 曲率指标
                plane_center = region_pts.mean(0)
                deviations = np.abs(np.dot(region_pts - plane_center, normal))
                curvature = np.median(deviations) / (extent + 1e-10)

                linearity = lam2 / sum_lam
                if linearity > 0.85:
                    surf_type = "skeleton"
                elif curvature > 0.02:
                    surf_type = "curved"
                else:
                    surf_type = "planar"
            else:
                normal = np.array([0, 0, 1], dtype=np.float32)
                surf_type = "planar"

            # 喷涂策略
            if surf_type == "planar":
                spray = "2D栅格扫描"
            elif surf_type == "curved":
                spray = "等高线轮廓跟随"
            else:
                spray = "轴线环绕"

            regions.append({
                "points": region_pts,
                "type": surf_type,
                "normal": normal.astype(np.float32),
                "center": region_pts.mean(0).astype(np.float32),
                "coverage": float(coverage),
                "part_name": cn_name,
                "part_id": PARTNET_LABEL_NAMES[int(lbl)],
                "label": int(lbl),
                "semantic_group": group_name,
                "color": [float(c) for c in color],
                "spray": spray,
            })

        regions.sort(key=lambda r: r["coverage"], reverse=True)
        return regions


# ── 单例 ──

_MODEL = None


def get_pointnext_model(config_path=None, checkpoint_path=None, device=None):
    """获取 PointNeXt 推理器单例（首次调用加载模型）。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    base = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base, "..")

    if config_path is None:
        config_path = os.path.join(project_root, "cfgs", "partnet", "pointnext-s.yaml")
    if checkpoint_path is None:
        # Auto-find checkpoint in log/checkpoint/
        ckpt_dir = os.path.join(project_root, "log", "checkpoint")
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "*_ckpt_best.pth"))
        if ckpt_files:
            checkpoint_path = ckpt_files[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    _MODEL = PointNeXtPartNetInference(config_path, checkpoint_path, device)
    return _MODEL


# ── 测试 ──
if __name__ == "__main__":
    print("=== 测试 PointNeXt 推理模块 ===")
    model = get_pointnext_model()

    # 合成一个简单 box 点云
    rng = np.random.RandomState(42)
    pts = rng.randn(5000, 3).astype(np.float32)

    labels, names, regions = model.predict(pts)
    print(f"预测标签: {np.unique(labels)}")
    print(f"区域数: {len(regions)}")
    for r in regions[:5]:
        print(f"  [{r['part_name']}] type={r['type']} "
              f"coverage={r['coverage']:.3f} color={r['color']}")
