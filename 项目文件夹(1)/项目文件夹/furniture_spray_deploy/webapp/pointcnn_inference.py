#!/usr/bin/env python3
"""
PointCNN PartNet 推理模块 — 封装 chair/table 的分割推理.

基于 pretrained_models/partnet/ 下的预训练 PointCNN 模型.
- Chair:   6 类  (chair_head/back/arm/base/seat)
- Table:   11 类 (regular_tabletop/base, pool_tabletop/base, etc.)
- Cabinet: 7 类  (countertop/shelf/frame/drawer/base/door)

注意: 此模块使用 TensorFlow 1.x, 与 PyTorch 可在同一进程中共存 (CPU 模式).
"""

import os
import sys
import math
import h5py
import tempfile
import importlib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Patch tf.contrib BEFORE PointCNN imports
_PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'pretrained_models', 'partnet')
sys.path.insert(0, _PRETRAINED_DIR)
import tf1_compat  # noqa: E402

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

# ── 配置 ──

CATEGORY_CONFIG = {
    'chair': {
        'partnet_name': 'Chair',
        'num_class': 6,
        'part_names': ['background', 'chair_head', 'chair_back', 'chair_arm',
                       'chair_base', 'chair_seat'],
        'part_names_cn': ['背景', '椅头', '靠背', '扶手', '底座', '座面'],
    },
    'table': {
        'partnet_name': 'Table',
        'num_class': 11,
        'part_names': ['background', 'ping_pong_net', 'ping_pong_tabletop',
                       'ping_pong_table_base', 'pool_tabletop', 'pool_table_base',
                       'picnic_tabletop', 'picnic_table_base', 'picnic_bench',
                       'regular_tabletop', 'regular_table_base'],
        'part_names_cn': ['背景', '乒乓网', '乒乓桌面', '乒乓桌腿', '台球桌面',
                          '台球桌腿', '野餐桌面', '野餐桌腿', '野餐长凳',
                          '桌面', '桌腿'],
    },
}

# 颜色映射
CHAIR_COLORS = [
    [0.2, 0.2, 0.2],       # 0 background
    [1.0, 0.2, 0.2],       # 1 chair_head - 红
    [0.2, 1.0, 0.2],       # 2 chair_back - 绿
    [0.2, 0.2, 1.0],       # 3 chair_arm - 蓝
    [1.0, 1.0, 0.2],       # 4 chair_base - 黄
    [1.0, 0.2, 1.0],       # 5 chair_seat - 品红
]

TABLE_COLORS = [
    [0.2, 0.2, 0.2],       # 0 background
    [0.8, 0.8, 0.8],       # 1 ping_pong_net
    [0.3, 0.6, 1.0],       # 2 ping_pong_tabletop
    [0.5, 0.3, 0.1],       # 3 ping_pong_table_base
    [0.3, 0.6, 1.0],       # 4 pool_tabletop
    [0.5, 0.3, 0.1],       # 5 pool_table_base
    [0.3, 0.6, 1.0],       # 6 picnic_tabletop
    [0.5, 0.3, 0.1],       # 7 picnic_table_base
    [0.2, 0.8, 0.2],       # 8 picnic_bench
    [0.3, 0.6, 1.0],       # 9 regular_tabletop
    [0.5, 0.3, 0.1],       # 10 regular_table_base
]

CATEGORY_COLORS = {'chair': CHAIR_COLORS, 'table': TABLE_COLORS}

# ── 全局 TF 状态 ──
_TF_SESSIONS = {}  # category → (sess, net_ops, setting)


def _get_pointcnn_session(category, fold=1):
    """获取或创建 PointCNN TF1 session (单例)."""
    key = (category, fold)
    if key in _TF_SESSIONS:
        return _TF_SESSIONS[key]

    config = CATEGORY_CONFIG[category]
    partnet_name = config['partnet_name']
    num_class = config['num_class']

    model_dir = os.path.join(
        _PRETRAINED_DIR,
        f'pointcnn_seg_partnet_sem_seg_150_{partnet_name}_{fold}')
    pointcnn_dir = os.path.join(model_dir, 'PointCNN')
    ckpt_dir = os.path.join(model_dir, 'ckpts')

    # 查找 checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'iter-168336')
    if not os.path.exists(ckpt_path + '.index'):
        ckpt_file = os.path.join(ckpt_dir, 'checkpoint')
        if os.path.exists(ckpt_file):
            with open(ckpt_file) as f:
                for line in f:
                    if line.startswith('model_checkpoint_path'):
                        name = line.split(':')[1].strip().strip('"')
                        ckpt_path = os.path.join(ckpt_dir, name)
                        break

    if not os.path.exists(ckpt_path + '.index'):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 导入设置模块
    if category == 'chair' or category == 'table':
        setting_module = 'partnet_sem_seg_plus_100'
    else:
        setting_module = 'partnet_sem_seg_150'

    sys.path.insert(0, pointcnn_dir)
    setting_path = os.path.join(pointcnn_dir, 'pointcnn_seg')
    sys.path.append(setting_path)
    setting = importlib.import_module(setting_module)
    setting.num_class = num_class

    model_module = importlib.import_module('pointcnn_seg')
    sample_num = setting.sample_num
    batch_size = 1

    # 构建 TF 图
    tf.compat.v1.reset_default_graph()
    indices_ph = tf.compat.v1.placeholder(tf.int32, shape=(batch_size, None, 2))
    is_training_ph = tf.compat.v1.placeholder(tf.bool)
    pts_fts_ph = tf.compat.v1.placeholder(
        tf.float32, shape=(batch_size, None, setting.data_dim))

    pts_fts_sampled = tf.gather_nd(pts_fts_ph, indices=indices_ph)
    if setting.data_dim > 3:
        points_sampled, features_sampled = tf.split(
            pts_fts_sampled, [3, setting.data_dim - 3], axis=-1)
        if not setting.use_extra_features:
            features_sampled = None
    else:
        points_sampled = pts_fts_sampled
        features_sampled = None

    net = model_module.Net(points_sampled, features_sampled, is_training_ph, setting)
    seg_probs_op = tf.nn.softmax(net.logits, name='seg_probs')

    saver = tf.compat.v1.train.Saver()

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config_proto)
    saver.restore(sess, ckpt_path)

    ops = {
        'sess': sess,
        'indices_ph': indices_ph,
        'pts_fts_ph': pts_fts_ph,
        'is_training_ph': is_training_ph,
        'seg_probs_op': seg_probs_op,
        'sample_num': sample_num,
        'batch_size': batch_size,
        'data_dim': setting.data_dim,
        'num_class': num_class,
    }

    _TF_SESSIONS[key] = (ops, setting)
    print(f"[PointCNN] 已加载 {category} 模型: {num_class} 类, fold={fold}")
    return (ops, setting)


def predict(category, points, fold=1, max_points=None):
    """
    使用 PointCNN 对点云进行部件分割.

    输入:
      category: 'chair' | 'table'
      points: (N, 3) numpy float32 原始 xyz
      fold: 模型 fold (1/2/3)
      max_points: 推理点数, None=自适应 (根据输入点数选最优值, 避免稀疏点云过度填充)

    输出:
      labels: (N,) numpy int32 — 逐点标签
      config: dict — 类别名、部件名、颜色等
    """
    if category not in CATEGORY_CONFIG:
        raise ValueError(f"不支持类别: {category}, 仅支持 chair/table")

    cat_config = CATEGORY_CONFIG[category]
    colors = CATEGORY_COLORS[category]

    N = points.shape[0]

    # 自适应 max_points: 避免稀疏点云过度填充导致重复点过多
    if max_points is None:
        if N <= 2048:
            max_points = 2048
        elif N <= 4096:
            max_points = 4096
        else:
            max_points = 8192

    # 采样/填充到 max_points
    if N < max_points:
        # 升采样: 加微小抖动避免完全重复的点破坏局部邻域
        n_dup = max_points - N
        idx = np.random.choice(N, n_dup, replace=True)
        dup_pts = points[idx] + np.random.randn(n_dup, 3).astype(np.float32) * 0.0005
        pts_padded = np.vstack([points, dup_pts])
    elif N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts_padded = points[idx]
    else:
        pts_padded = points
        idx = np.arange(N)

    pts_padded = pts_padded.astype(np.float32)

    # 写临时 H5
    tmp_h5 = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    h5_path = tmp_h5.name
    tmp_h5.close()

    try:
        data = pts_padded[np.newaxis, ...]
        data_num = np.array([max_points], dtype=np.int32)
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('data', data=data, dtype='float32')
            f.create_dataset('data_num', data=data_num, dtype='int32')

        # 获取 TF session
        ops, setting = _get_pointcnn_session(category, fold=fold)
        batch_size = ops['batch_size']
        sample_num = ops['sample_num']

        tile_num = math.ceil((sample_num * batch_size) / max_points)

        # 构建 indices
        indices_batch_indices = np.tile(
            np.reshape(np.arange(batch_size), (batch_size, 1, 1)),
            (1, sample_num, 1))
        indices_shuffle = np.tile(np.arange(max_points), tile_num)[:sample_num * batch_size]
        np.random.shuffle(indices_shuffle)
        indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
        indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

        pts_batch = pts_padded[np.newaxis, ...]  # (1, max_points, 3)

        seg_probs = ops['sess'].run(
            [ops['seg_probs_op']],
            feed_dict={
                ops['pts_fts_ph']: pts_batch,
                ops['indices_ph']: indices_batch,
                ops['is_training_ph']: False,
            })
        probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

        # 投票: 每个点多次采样, 取最高置信度标签
        predictions = [(-1, 0.0)] * max_points
        for i in range(sample_num * batch_size):
            point_idx = indices_shuffle[i]
            prob = probs_2d[i, :]
            conf = np.amax(prob)
            label = int(np.argmax(prob))
            if conf > predictions[point_idx][1]:
                predictions[point_idx] = [label, conf]

        labels_padded = np.array([l for l, _ in predictions], dtype=np.int32)

        # 映射回原始点云
        if N > max_points:
            # 最近邻映射
            labels_out = np.zeros(N, dtype=np.int32)
            for i in range(N):
                dists = np.sum((pts_padded - points[i]) ** 2, axis=1)
                labels_out[i] = labels_padded[np.argmin(dists)]
        elif N < max_points:
            labels_out = labels_padded[:N]
        else:
            labels_out = labels_padded

    finally:
        os.unlink(h5_path)

    return labels_out, cat_config, colors


def _estimate_region_type(pts, eps=None):
    """PCA + DBSCAN 估计区域表面类型: planar / curved / skeleton."""
    if len(pts) < 5:
        return 'planar', np.array([0, 0, 1], dtype=np.float32)

    # 先 PCA 分析 — 平面/曲面有明确的特征值模式
    centered = pts - pts.mean(0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    lam0, lam1, lam2 = eigvals[0], eigvals[1], eigvals[2]
    sum_lam = lam0 + lam1 + lam2 + 1e-10
    linearity = lam2 / sum_lam       # ~0.33=各向同性, ~1.0=纯线
    planarity = 1 - (lam0 / (lam1 + 1e-10))  # ~0=各向同性, ~1=纯平面

    # 高度线性的 → skeleton (杆状结构)
    if linearity > 0.85:
        return 'skeleton', normal.astype(np.float32)

    # 明显是平面的 → planar (不依赖 DBSCAN, 避免噪声分簇误判)
    if planarity > 0.4:
        extent = np.sqrt(lam2)
        deviations = np.abs(np.dot(pts - pts.mean(0), normal))
        curvature = np.median(deviations) / (extent + 1e-10)
        if curvature > 0.025:
            return 'curved', normal.astype(np.float32)
        return 'planar', normal.astype(np.float32)

    # 模糊情况: DBSCAN 辅助 (5+ 簇 → skeleton)
    if eps is None:
        bbox = pts.max(0) - pts.min(0)
        eps = float(np.median(bbox) * 0.12)
    try:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=max(eps, 0.02), min_samples=10).fit(pts)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 5:
            return 'skeleton', normal.astype(np.float32)
    except Exception:
        pass

    # 默认 planar
    extent = np.sqrt(lam2)
    deviations = np.abs(np.dot(pts - pts.mean(0), normal))
    curvature = np.median(deviations) / (extent + 1e-10)
    if curvature > 0.03:
        return 'curved', normal.astype(np.float32)
    return 'planar', normal.astype(np.float32)

def group_into_regions(points, labels, cat_config, colors, min_ratio=0.015):
    """将预测标签分组为 regions (兼容 webapp 输出格式).

    min_ratio: 最小覆盖率阈值, 低于此值的部件被过滤.
               chair 默认 0.015 (1.5%), 可过滤掉无扶手椅子的虚假扶手.
    """
    N = len(points)
    part_names = cat_config['part_names']
    part_names_cn = cat_config['part_names_cn']
    unique_labels = np.unique(labels)
    regions = []

    # AI 标签 → 区域类型映射 (比 PCA 在噪声点集上更准确)
    # chair: 1=椅头 2=靠背 3=扶手 4=底座 5=座面
    # table: 9=桌面 10=桌腿
    SKELETON_LABELS = {3, 4, 8, 10}  # 扶手/底座/桌腿 → skeleton
    PLANAR_LABELS = {2, 5, 6, 9}      # 靠背/座面/桌面 → planar

    for lbl in unique_labels:
        if lbl == 0:
            continue
        mask = labels == lbl
        region_pts = points[mask]
        coverage = len(region_pts) / N
        if coverage < min_ratio:
            continue

        if lbl in SKELETON_LABELS:
            surf_type = 'skeleton'
            normal = np.array([0, 0, 1], dtype=np.float32)
        elif lbl in PLANAR_LABELS:
            surf_type = 'planar'
            centered = region_pts - region_pts.mean(0)
            _, eigvecs = np.linalg.eigh(np.cov(centered.T))
            normal = eigvecs[:, 0].astype(np.float32)
        else:
            surf_type, normal = _estimate_region_type(region_pts)
        cn_name = part_names_cn[lbl] if lbl < len(part_names_cn) else f'part_{lbl}'
        en_name = part_names[lbl] if lbl < len(part_names) else f'part_{lbl}'

        spray_map = {'planar': '2D栅格扫描', 'curved': '等高线轮廓跟随', 'skeleton': '轴线环绕'}

        regions.append({
            'points': region_pts,
            'type': surf_type,
            'normal': normal,
            'center': region_pts.mean(0).astype(np.float32),
            'coverage': float(coverage),
            'part_name': cn_name,
            'part_id': en_name,
            'label': int(lbl),
            'color': colors[lbl] if lbl < len(colors) else [0.5, 0.5, 0.5],
            'spray': spray_map.get(surf_type, '2D栅格扫描'),
        })

    regions.sort(key=lambda r: r['coverage'], reverse=True)
    return regions


# ── 测试 ──
if __name__ == '__main__':
    print("=== 测试 PointCNN 推理 ===")
    rng = np.random.RandomState(42)
    pts = rng.randn(5000, 3).astype(np.float32)

    for cat in ['chair', 'table']:
        print(f"\n--- {cat} ---")
        labels, cfg, colors = predict(cat, pts)
        print(f"  标签分布: {np.unique(labels)}")
        regions = group_into_regions(pts, labels, cfg, colors)
        print(f"  区域数: {len(regions)}")
        for r in regions[:3]:
            print(f"    [{r['part_name']}] coverage={r['coverage']:.3f}")
