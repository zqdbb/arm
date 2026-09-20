#!/usr/bin/env python3
"""
喷涂管线 v2 — 表面级分解 + 多策略组合路径
家具 → 区域分解(平面/曲面/骨架) → 每区域独立路径 → 合并输出
用法:
  python spray_pipeline.py --file <ply/npy/h5>
  python spray_pipeline.py --synth           # 合成测试
  python spray_pipeline.py --modelnet40       # ModelNet40测试
"""
import os, sys, argparse, json
import numpy as np
import open3d as o3d

# 模型推理模块 (可选)
try:
    from partseg_inference import (PartSegModel, segment_furniture_with_model,
                                    PROTOTYPE_COLORS, PART_TO_PROTOTYPE,
                                    map_part_to_prototype, map_prototype_to_part)
    HAS_PARTSEG = True
except ImportError:
    HAS_PARTSEG = False

# 路径优化模块 (可选)
try:
    from path_optimizer import optimize_path, optimize_scan_direction
    HAS_PATH_OPT = True
except ImportError:
    HAS_PATH_OPT = False

# ===================================================================
# 表面类型定义 (替代旧的5类家具分类)
# ===================================================================
SURFACE_TYPES = {
    'planar': {
        'name': '平面区域',
        'path_algo': '2D栅格扫描',
        'gun_pose': '固定法向',
        'speed': '匀速',
        'color': [0.4, 0.7, 1.0],   # 蓝色
    },
    'curved': {
        'name': '曲面区域',
        'path_algo': '等高线轮廓跟随',
        'gun_pose': '实时跟随法线',
        'speed': '变速',
        'color': [1.0, 0.4, 0.7],   # 粉红
    },
    'skeleton': {
        'name': '细长结构',
        'path_algo': '轴线环绕',
        'gun_pose': '环绕跟随',
        'speed': '中低速',
        'color': [0.4, 1.0, 0.5],   # 绿色
    },
}

# 部件颜色调色板 — 每个部件用不同颜色, 视觉可区分
PART_COLORS = [
    [0.21, 0.49, 0.94],  # 蓝
    [0.94, 0.35, 0.16],  # 橙
    [0.22, 0.78, 0.35],  # 绿
    [0.88, 0.18, 0.55],  # 粉
    [0.95, 0.78, 0.06],  # 黄
    [0.53, 0.27, 0.73],  # 紫
    [0.00, 0.68, 0.73],  # 青
    [0.80, 0.55, 0.30],  # 棕
    [0.35, 0.35, 0.35],  # 深灰
    [0.95, 0.45, 0.30],  # 橙红
    [0.45, 0.75, 0.95],  # 浅蓝
    [0.95, 0.60, 0.75],  # 浅粉
    [0.50, 0.60, 0.30],  # 橄榄
    [0.70, 0.35, 0.50],  # 玫红
    [0.25, 0.55, 0.45],  # 墨绿
    [0.80, 0.72, 0.50],  # 卡其
]

def _compute_gravity_axis(regions):
    """推断重力方向 — 找法向最"一致"的平面群, 即为水平面族.
    不硬编码世界Z, 适应任意模型朝向 (Z-up / Y-up 均可)."""
    planar = [r for r in regions if r['type'] == 'planar']
    if not planar:
        all_pts = np.vstack([r['points'] for r in regions])
        cov = np.cov(all_pts.T)
        _, eigvecs = np.linalg.eigh(cov)
        return eigvecs[:, 0]  # 最小方差方向

    # 用法向聚类找最大覆盖率的平面族 (这些是水平面)
    # 对每对平面计算法向一致性 (dot 的绝对值)
    normals = np.array([r['normal'] for r in planar])
    coverages = np.array([r['coverage'] for r in planar])

    best_group_coverage = 0
    best_group_normal = None

    if len(planar) == 1:
        best_group_normal = normals[0].copy()
    else:
        for i in range(len(planar)):
            group_cov = 0
            for j in range(len(planar)):
                if abs(np.dot(normals[i], normals[j])) > 0.7:
                    group_cov += coverages[j]
            if group_cov > best_group_coverage:
                best_group_coverage = group_cov
                best_group_normal = normals[i].copy()

    if best_group_normal is None:
        best_group_normal = planar[0]['normal']

    # 确保重力轴朝上: 最高的水平面应在重力范围的上半部分
    # 关键: 参考最高的水平面 (柜子顶板/桌面/座面), 不是最大的平面
    all_pts = np.vstack([r['points'] for r in regions])
    all_h = np.dot(all_pts, best_group_normal)
    g_min, g_max = float(all_h.min()), float(all_h.max())

    # 找最高处的水平面 (abs dot > 0.7, 不受重力符号影响)
    horizontal_planar = [r for r in planar
                         if abs(np.dot(r['normal'], best_group_normal)) > 0.7]
    if horizontal_planar:
        # 用最高的水平面 (沿候选重力方向投影最大的) 作为"上方"参考
        topmost = max(horizontal_planar,
                      key=lambda r: float(np.dot(r['center'], best_group_normal)))
        ref_h = float(np.dot(topmost['center'], best_group_normal))
    else:
        # 无水平面 → 用最大平面, 假设它在重力方向的上方
        main_planar = max(planar, key=lambda r: r['coverage'])
        ref_h = float(np.dot(main_planar['center'], best_group_normal))

    # 参考平面离底部近 → 重力方向反了, 翻转
    if (ref_h - g_min) < (g_max - ref_h) * 0.5:
        best_group_normal = -best_group_normal

    return best_group_normal


def _is_horizontal(normal, gravity_axis=None, threshold=0.7):
    """法向与重力轴夹角 < arccos(0.7)≈45° → 水平面"""
    if gravity_axis is None:
        gravity_axis = np.array([0, 0, 1.0])
    return abs(np.dot(normal, gravity_axis)) > threshold


def _is_vertical(normal, gravity_axis=None, threshold=0.35):
    """法向与重力轴夹角 > arccos(0.35)≈70° → 垂直面"""
    if gravity_axis is None:
        gravity_axis = np.array([0, 0, 1.0])
    return abs(np.dot(normal, gravity_axis)) < threshold

# ===================================================================
# G1-G4 家具级几何大类
# ===================================================================
GEOMETRY_CLASS = {
    'G1_large_plane': {
        'name': '大平面',
        'description': '门板、面板、木饰面 — 单一大平面主导',
        'typical': ['door'],
        'spray_strategy': 'Z字扫描, 1站',
        'difficulty': '低',
    },
    'G2_multi_plane': {
        'name': '多平面',
        'description': '柜体、衣柜、电视柜 — 多个正交平面',
        'typical': ['dresser', 'night_stand', 'wardrobe', 'bookshelf', 'tv_stand'],
        'spray_strategy': '逐面喷涂, 1-2站',
        'difficulty': '中低',
    },
    'G3_curved': {
        'name': '曲面',
        'description': '弧形床头、曲面靠背、异形家具 — 包含连续曲面',
        'typical': ['sofa'],
        'spray_strategy': '法向跟踪等高线, 2-3站',
        'difficulty': '中高',
    },
    'G4_mixed_skeleton': {
        'name': '杆件混合',
        'description': '椅腿、桌腿、花格 — 平面+细长杆件混合',
        'typical': ['chair', 'table', 'bed'],
        'spray_strategy': '平面栅格 + 杆件环绕, 2-3站',
        'difficulty': '高',
    },
}

# ===================================================================
# 部件语义命名 — 基于类别 + 位置 + 几何类型的启发式规则
# ===================================================================
def name_parts(category, regions):
    """
    为每个区域赋予语义部件名称.
    输入: category (如 'chair'), regions (decompose_furniture 输出)
    输出: regions (原地修改, 添加 part_name 字段)
    """
    if not regions:
        return regions

    # 计算每个 region 的相对位置
    all_centers = np.array([r['center'] for r in regions])
    global_center = np.mean([r['center'] for r in regions], axis=0)
    all_pts = np.vstack([r['points'] for r in regions])

    # 自动推断重力方向 (最大平面的法向)
    gravity = _compute_gravity_axis(regions)

    # 沿重力轴投影位置: 在重力方向上的相对高度
    grav_proj = np.dot(all_pts, gravity)
    g_min, g_max = grav_proj.min(), grav_proj.max()
    g_range = g_max - g_min + 1e-10

    for r in regions:
        c = r['center']
        rel_h = (np.dot(c, gravity) - g_min) / g_range  # 0=bottom, 1=top
        to_center = c - global_center
        r['_gravity'] = gravity
        r['_rel_h'] = rel_h
        r['_rel_xy'] = np.linalg.norm(to_center[:2])
        r['_g_range'] = g_range

    # 按类别分配部件名
    if category == 'chair':
        _name_chair_parts(regions, g_range, gravity)
    elif category == 'table':
        _name_table_parts(regions, g_range, gravity)
    elif category == 'bed':
        _name_bed_parts(regions, gravity)
    elif category == 'sofa':
        _name_sofa_parts(regions, g_range, gravity)
    elif category in ('desk', 'dresser', 'night_stand', 'wardrobe',
                       'bookshelf', 'tv_stand', 'cabinet'):
        _name_cabinet_parts(regions, category, gravity)
    elif category == 'door':
        _name_door_parts(regions)
    else:
        _name_generic_parts(regions)

    # 清理临时字段
    for r in regions:
        r.pop('_gravity', None)
        r.pop('_rel_h', None)
        r.pop('_rel_xy', None)
        r.pop('_g_range', None)

    return regions


def _is_leg_face(r, gravity=None):
    """判断 planar 区域是否实为腿面. 核心逻辑:
    腿沿重力方向延伸, 故腿面的最长PCA轴应与重力平行;
    扶手/靠背支撑则沿水平方向延伸, 主方向垂直于重力."""
    pts = r['points']
    if len(pts) < 10:
        return False
    centered = pts - pts.mean(0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    face_aspect = eigvals[2] / (eigvals[1] + 1e-10)
    # 面内长宽比 > 1.3 → 至少有点细长
    if face_aspect < 1.3:
        return False
    # 检查主方向(最长轴)是否平行于重力 — 这是腿 vs 扶手的关键区别
    if gravity is not None:
        principal_dir = eigvecs[:, 2]  # λ₃ 对应的特征向量 (最长方向)
        gravity_norm = gravity / (np.linalg.norm(gravity) + 1e-10)
        dot = abs(np.dot(principal_dir, gravity_norm))
        if dot < 0.65:  # 夹角 > 50° → 主方向不沿重力 → 不是腿
            return False
    # 小覆盖面 + 细长形状 + 沿重力方向 = 腿面
    if r['coverage'] < 0.15:
        return True
    return False


def _name_chair_parts(regions, g_range, gravity):
    """椅子部件命名. 核心规则:
    1. 最大水平面 → 座面
    2. 骨架区域: 座面以下→腿, 座面以上→靠背/扶手框架
    3. 座面以下的所有非水平区域 → 腿 (铁律, 不检查几何)
    4. 座面以上: 最大垂直面 → 靠背, 其余 → 扶手
    5. 剩余 → 辅面/辅件"""
    planar = [r for r in regions if r['type'] == 'planar']
    skeletons = [r for r in regions if r['type'] == 'skeleton']

    horizontal = [r for r in planar if _is_horizontal(r['normal'], gravity)]
    vertical = [r for r in planar if _is_vertical(r['normal'], gravity)]
    oblique = [r for r in planar
               if not _is_horizontal(r['normal'], gravity)
               and not _is_vertical(r['normal'], gravity)]

    # Step 1: 找座面 → 同时计算重力方向分割高度
    seat_h = None
    if horizontal:
        horizontal.sort(key=lambda r: -r['coverage'])
        horizontal[0]['part_name'] = '座面(seat)'
        horizontal[0]['part_id'] = 'seat'
        seat_h = float(np.dot(horizontal[0]['center'], gravity))
        for r in horizontal[1:]:
            r['part_name'] = '辅面(aux_plane)'
            r['part_id'] = 'aux_plane'
    else:
        # 兜底: 无水平面时用所有区域中心的中位数作为座面参考高度
        all_centers_h = [float(np.dot(r['center'], gravity)) for r in regions]
        if all_centers_h:
            seat_h = float(np.median(all_centers_h))

    # Step 2: skeleton → 座面以下=腿, 座面以上=框架(靠背/扶手)
    leg_idx = 0
    above_seat_skeletons = []
    for r in skeletons:
        if seat_h is not None:
            r_h = float(np.dot(r['center'], gravity))
            if r_h > seat_h:
                above_seat_skeletons.append(r)
                continue
        leg_idx += 1
        r['part_name'] = f'腿(leg_{leg_idx})'
        r['part_id'] = f'leg_{leg_idx}'

    # Step 3: vertical + oblique → 按座面高度分割
    # 座面以下 = 腿 (铁律, 不检查几何特征)
    # 座面以上 = 靠背/扶手候选
    above_seat = []
    for r in vertical + oblique:
        if 'part_name' in r:
            continue
        if seat_h is not None:
            r_h = float(np.dot(r['center'], gravity))
            if r_h < seat_h:
                # 座面以下 → 腿, 无例外
                leg_idx += 1
                r['part_name'] = f'腿(leg_{leg_idx})'
                r['part_id'] = f'leg_{leg_idx}'
            else:
                above_seat.append(r)
        else:
            # 无座面 → 兜底
            above_seat.append(r)

    # Step 4: 座面以上的 vertical/oblique → 靠背(最大) / 扶手(其余)
    above_vert = [r for r in above_seat if _is_vertical(r['normal'], gravity)]
    above_obl  = [r for r in above_seat if not _is_vertical(r['normal'], gravity)]

    if above_vert:
        above_vert.sort(key=lambda r: -r['coverage'])
        above_vert[0]['part_name'] = '靠背(backrest)'
        above_vert[0]['part_id'] = 'backrest'
        for r in above_vert[1:]:
            r['part_name'] = '扶手(armrest)'
            r['part_id'] = 'armrest'
    elif above_obl:
        # 无纯垂直面 → 斜靠背
        above_obl.sort(key=lambda r: -r['coverage'])
        above_obl[0]['part_name'] = '靠背(backrest)'
        above_obl[0]['part_id'] = 'backrest'
        for r in above_obl[1:]:
            r['part_name'] = '扶手(armrest)'
            r['part_id'] = 'armrest'

    for r in above_obl:
        if 'part_name' not in r:
            r['part_name'] = '扶手(armrest)'
            r['part_id'] = 'armrest'

    # Step 4.5: 座面以上的 skeleton → 最高=靠背框架, 其余=扶手框架
    if above_seat_skeletons:
        above_seat_skeletons.sort(key=lambda r: -float(np.dot(r['center'], gravity)))
        above_seat_skeletons[0]['part_name'] = '靠背框架(backrest_frame)'
        above_seat_skeletons[0]['part_id'] = 'backrest_frame'
        for r in above_seat_skeletons[1:]:
            r['part_name'] = '扶手框架(armrest_frame)'
            r['part_id'] = 'armrest_frame'

    # Step 5: 未命名的 planar → 辅面
    for r in planar:
        if 'part_name' not in r:
            r['part_name'] = '辅面(aux_plane)'
            r['part_id'] = 'aux_plane'

    # Step 6: 剩余非 planar → 按类型
    for r in regions:
        if 'part_name' not in r:
            if r['type'] == 'curved':
                r['part_name'] = '曲面辅件(curved_aux)'
                r['part_id'] = 'curved_aux'
            else:
                r['part_name'] = '辅件(aux)'
                r['part_id'] = 'aux'


def _name_table_parts(regions, g_range, gravity):
    """桌子: 水平面在上=桌面, 水平面在下=横撑, 骨架=腿"""
    planar = [r for r in regions if r['type'] == 'planar']
    skeletons = [r for r in regions if r['type'] == 'skeleton']

    horizontal = [r for r in planar if _is_horizontal(r['normal'], gravity)]
    vertical = [r for r in planar if _is_vertical(r['normal'], gravity)]

    if horizontal:
        horizontal.sort(key=lambda r: -np.dot(r['center'], gravity))
        horizontal[0]['part_name'] = '桌面(tabletop)'
        horizontal[0]['part_id'] = 'tabletop'
        for r in horizontal[1:]:
            if r['coverage'] > 0.03:
                r['part_name'] = '横撑(stretcher)'
                r['part_id'] = 'stretcher'
            else:
                r['part_name'] = '小平面(small_plane)'
                r['part_id'] = 'small_plane'

    for r in vertical:
        if r['coverage'] > 0.03:
            r['part_name'] = '侧板(side_panel)'
            r['part_id'] = 'side_panel'
        else:
            r['part_name'] = '小平面(small_plane)'
            r['part_id'] = 'small_plane'

    for i, r in enumerate(skeletons):
        r['part_name'] = f'腿(leg_{i+1})'
        r['part_id'] = f'leg_{i+1}'

    for r in regions:
        if 'part_name' not in r:
            r['part_name'] = '辅件(aux)'
            r['part_id'] = 'aux'


def _name_bed_parts(regions, gravity):
    """床: 水平面=床板, 垂直面=床侧板, 骨架=床架"""
    planar = [r for r in regions if r['type'] == 'planar']
    skeletons = [r for r in regions if r['type'] == 'skeleton']

    horizontal = [r for r in planar if _is_horizontal(r['normal'], gravity)]
    vertical = [r for r in planar if _is_vertical(r['normal'], gravity)]

    if horizontal:
        horizontal.sort(key=lambda r: -len(r['points']))
        horizontal[0]['part_name'] = '床板(bed_board)'
        horizontal[0]['part_id'] = 'bed_board'
        for r in horizontal[1:]:
            r['part_name'] = '床板辅面(bed_aux)'
            r['part_id'] = 'bed_aux'

    for i, r in enumerate(vertical):
        r['part_name'] = f'床侧板(bed_side_{i+1})'
        r['part_id'] = f'bed_side_{i+1}'

    for r in planar:
        if 'part_name' not in r:
            r['part_name'] = '辅面(aux_plane)'
            r['part_id'] = 'aux_plane'

    for i, r in enumerate(skeletons):
        r['part_name'] = f'床架(frame_{i+1})'
        r['part_id'] = f'frame_{i+1}'

    for r in regions:
        if 'part_name' not in r:
            r['part_name'] = '辅件(aux)'
            r['part_id'] = 'aux'


def _name_sofa_parts(regions, g_range, gravity):
    """沙发: 水平面=座面, 垂直面=靠背/扶手, 骨架=腿"""
    planar = [r for r in regions if r['type'] == 'planar']

    if not planar:
        for r in regions:
            r['part_name'] = '曲面主体(curved_body)'
            r['part_id'] = 'curved_body'
        return

    horizontal = [r for r in planar if _is_horizontal(r['normal'], gravity)]
    vertical = [r for r in planar if _is_vertical(r['normal'], gravity)]

    if horizontal:
        horizontal.sort(key=lambda r: np.dot(r['center'], gravity))  # 最低→座面
        horizontal[0]['part_name'] = '座面(seat)'
        horizontal[0]['part_id'] = 'seat'
        for r in horizontal[1:]:
            r['part_name'] = '辅面(aux_plane)'
            r['part_id'] = 'aux_plane'

    if vertical:
        vertical.sort(key=lambda r: -len(r['points']))
        vertical[0]['part_name'] = '靠背(backrest)'
        vertical[0]['part_id'] = 'backrest'
        for i, r in enumerate(vertical[1:], 1):
            r['part_name'] = f'扶手(armrest_{i})'
            r['part_id'] = f'armrest_{i}'

    for r in planar:
        if 'part_name' not in r:
            r['part_name'] = '辅面(aux_plane)'
            r['part_id'] = 'aux_plane'

    for r in regions:
        if 'part_name' not in r:
            if r['type'] == 'curved':
                r['part_name'] = '曲面扶手(curved_armrest)'
                r['part_id'] = 'curved_armrest'
            elif r['type'] == 'skeleton':
                r['part_name'] = '沙发腿(sofa_leg)'
                r['part_id'] = 'sofa_leg'
            else:
                r['part_name'] = '辅件(aux)'
                r['part_id'] = 'aux'


def _name_cabinet_parts(regions, category, gravity):
    """柜体: 水平面=顶板/底板, 垂直面=侧板/面板"""
    planar = [r for r in regions if r['type'] == 'planar']

    horizontal = [r for r in planar if _is_horizontal(r['normal'], gravity)]
    vertical = [r for r in planar if _is_vertical(r['normal'], gravity)]

    if horizontal:
        horizontal.sort(key=lambda r: -np.dot(r['center'], gravity))  # 最高→顶板
        horizontal[0]['part_name'] = '顶板(top)'
        horizontal[0]['part_id'] = 'top'
        if len(horizontal) >= 2:
            horizontal[-1]['part_name'] = '底板(bottom)'
            horizontal[-1]['part_id'] = 'bottom'
        for r in horizontal[1:-1]:
            r['part_name'] = '搁板(shelf)'
            r['part_id'] = 'shelf'

    for i, r in enumerate(vertical):
        r['part_name'] = f'侧板(side_{i+1})'
        r['part_id'] = f'side_{i+1}'

    for r in planar:
        if 'part_name' not in r:
            r['part_name'] = '面板(panel)'
            r['part_id'] = 'panel'

    for r in regions:
        if 'part_name' not in r:
            if r['type'] == 'skeleton':
                r['part_name'] = '边框(frame)'
                r['part_id'] = 'frame'
            else:
                r['part_name'] = '辅件(aux)'
                r['part_id'] = 'aux'


def _name_door_parts(regions):
    """门: 门板"""
    for r in regions:
        if r['type'] == 'planar':
            r['part_name'] = '门板(door_panel)'
            r['part_id'] = 'door_panel'
        elif r['type'] == 'skeleton':
            r['part_name'] = '门框(door_frame)'
            r['part_id'] = 'door_frame'
        else:
            r['part_name'] = '门饰(door_trim)'
            r['part_id'] = 'door_trim'


def _name_generic_parts(regions):
    """通用命名: 按几何类型 + 序号"""
    counts = {'planar': 0, 'curved': 0, 'skeleton': 0}
    for r in regions:
        t = r['type']
        counts[t] += 1
        if t == 'planar':
            r['part_name'] = f'平面_{counts[t]}'
            r['part_id'] = f'planar_{counts[t]}'
        elif t == 'curved':
            r['part_name'] = f'曲面_{counts[t]}'
            r['part_id'] = f'curved_{counts[t]}'
        else:
            r['part_name'] = f'杆件_{counts[t]}'
            r['part_id'] = f'skeleton_{counts[t]}'


def classify_geometry_class(regions):
    """
    根据区域组成判断家具级几何大类 (G1-G4).

    G1: 单一平面主导 — 最大平面覆盖>70%, 骨架/曲面占比<5%
    G2: 多个平面 (>=3 planar), 无显著骨架/曲面
    G3: 包含曲面区域 (曲面覆盖>5%)
    G4: 包含骨架区域 (骨架覆盖>5%, 平面+杆件混合)
    """
    if not regions:
        return 'G1_large_plane'

    n_planar = sum(1 for r in regions if r['type'] == 'planar')
    n_curved = sum(1 for r in regions if r['type'] == 'curved')
    n_skeleton = sum(1 for r in regions if r['type'] == 'skeleton')

    planar_cov = sum(r['coverage'] for r in regions if r['type'] == 'planar')
    curved_cov = sum(r['coverage'] for r in regions if r['type'] == 'curved')
    skel_cov = sum(r['coverage'] for r in regions if r['type'] == 'skeleton')
    max_planar_cov = max((r['coverage'] for r in regions if r['type'] == 'planar'), default=0)

    # 单一大平面主导 → G1 (即使有少量骨架/曲面)
    if max_planar_cov > 0.70 and curved_cov < 0.05 and skel_cov < 0.08:
        return 'G1_large_plane'

    if curved_cov > 0.05:
        return 'G3_curved'
    if skel_cov > 0.05:
        return 'G4_mixed_skeleton'
    if max_planar_cov > 0.70:
        return 'G1_large_plane'
    return 'G2_multi_plane'


def classify_furniture_type(points):
    """
    基于几何特征自动识别家具类型 — 仅3类: chair / table / cabinet.

    输入: (N,3) 点云
    输出: (category_name, confidence, features_dict)
      - category_name: 'chair' | 'table' | 'cabinet'
      - confidence: 0.0~1.0
      - features: 特征字典 (供调试)
    """
    regions = decompose_furniture(points)

    n_planar = sum(1 for r in regions if r['type'] == 'planar')
    n_curved = sum(1 for r in regions if r['type'] == 'curved')
    n_skeleton = sum(1 for r in regions if r['type'] == 'skeleton')
    planar_cov = sum(r['coverage'] for r in regions if r['type'] == 'planar')
    skel_cov = sum(r['coverage'] for r in regions if r['type'] == 'skeleton')
    max_planar_cov = max((r['coverage'] for r in regions if r['type'] == 'planar'), default=0)

    gravity = _compute_gravity_axis(regions)
    planar_regions = [r for r in regions if r['type'] == 'planar']
    n_horizontal = sum(1 for r in planar_regions if _is_horizontal(r['normal'], gravity))
    n_vertical = sum(1 for r in planar_regions if _is_vertical(r['normal'], gravity))

    all_pts = np.vstack([r['points'] for r in regions])
    bbox = all_pts.max(0) - all_pts.min(0)

    features = {
        'n_regions': len(regions),
        'n_planar': n_planar, 'n_curved': n_curved, 'n_skeleton': n_skeleton,
        'planar_cov': round(planar_cov, 3), 'skel_cov': round(skel_cov, 3),
        'max_planar_cov': round(max_planar_cov, 3),
        'n_horizontal': n_horizontal, 'n_vertical': n_vertical,
        'bbox': bbox.tolist(),
    }

    # 统计小平面 — 可能是被RANSAC拆散的腿面
    n_small_planar = sum(1 for r in planar_regions if r['coverage'] < 0.08)
    bbox_h = max(bbox)
    bbox_wh_ratio = bbox_h / (min(bbox) + 1e-10) if min(bbox) > 0 else 1.0

    # 大垂直面和大水平面计数 (柜体特征)
    n_large_vert = sum(1 for r in planar_regions
                       if _is_vertical(r['normal'], gravity) and r['coverage'] > 0.06)
    n_large_horiz = sum(1 for r in planar_regions
                        if _is_horizontal(r['normal'], gravity) and r['coverage'] > 0.06)
    vertical_cov = sum(r['coverage'] for r in planar_regions
                       if _is_vertical(r['normal'], gravity))
    horizontal_cov = sum(r['coverage'] for r in planar_regions
                         if _is_horizontal(r['normal'], gravity))

    # ============================================================
    # 优先柜体检测 — 柜体特征: 多大垂直面 + 多水平面 + 少骨架
    # ============================================================
    # 强柜体信号: ≥2大垂直面(侧板) + ≥2大水平面(顶/底板) + 骨架少
    if n_large_vert >= 2 and n_large_horiz >= 2 and skel_cov < 0.12:
        return 'cabinet', 0.82, features
    # 垂直面覆盖高 + 多大垂直面 + 多水平面 + 骨架少 → 柜体
    if vertical_cov > 0.25 and n_large_vert >= 2 and n_horizontal >= 2 and skel_cov < 0.10:
        return 'cabinet', 0.75, features
    # 即使有少量骨架, 垂直大面多 + 水平面多 → 仍是柜体
    if n_large_vert >= 2 and n_horizontal >= 2 and skel_cov < 0.08:
        return 'cabinet', 0.68, features
    # 4+个平面都是大面 + 无骨架 → 典型柜体
    if n_planar >= 4 and n_large_vert >= 2 and n_large_horiz >= 1 and n_skeleton == 0:
        return 'cabinet', 0.78, features

    # ============================================================
    # 空间分析: 椅子靠背在水平面上方, 桌子没有
    # 同时计算主水平面在重力方向的位置 (桌面在顶部, 座面在中间)
    # ============================================================
    horizontal_regions = [r for r in planar_regions if _is_horizontal(r['normal'], gravity)]
    has_backrest = False
    main_horiz_rel_h = 0.5  # 默认中间
    g_min = float(np.dot(all_pts, gravity).min())
    if horizontal_regions:
        main_horiz = max(horizontal_regions, key=lambda r: r['coverage'])
        h_center = float(np.dot(main_horiz['center'], gravity))
        g_span = float(np.dot(all_pts, gravity).ptp())
        main_horiz_rel_h = (h_center - g_min) / (g_span + 1e-10)  # 0=底, 1=顶
        margin = max(0.005, 0.01 * g_span)
        for r in planar_regions:
            if r is main_horiz:
                continue
            r_h = float(np.dot(r['center'], gravity))
            if r_h > h_center + margin:
                if _is_vertical(r['normal'], gravity) or not _is_horizontal(r['normal'], gravity):
                    has_backrest = True
                    break

    # ============================================================
    # 椅子 vs 桌子区分
    # 桌子: 桌面在顶部 (主水平面位置高, 上方无结构)
    # 椅子: 座面在中间 (主水平面位置中等, 上方有靠背/扶手)
    # ============================================================
    # 桌子强信号: 主水平面接近顶部 + 无靠背
    table_like = (main_horiz_rel_h > 0.65 and not has_backrest and n_horizontal >= 1)
    # 椅子强信号: 有靠背 或 主水平面在中间
    chair_like = (has_backrest or (0.25 < main_horiz_rel_h <= 0.65 and n_horizontal >= 1))

    if n_skeleton > 0:
        # 桌子信号强 → table
        if table_like and n_horizontal >= 1 and n_vertical <= 2:
            return 'table', 0.82, features
        # 椅子信号强 → chair
        if chair_like and n_horizontal >= 1:
            return 'chair', 0.82, features
        # 核心区分: 椅子有靠背(水平面上方的垂直面), 桌子只有桌面+腿
        if n_skeleton >= 3 and n_horizontal >= 1:
            if has_backrest:
                return 'chair', 0.85, features
            elif bbox_wh_ratio > 1.3 and main_horiz_rel_h < 0.65:
                return 'chair', 0.60, features
            else:
                return 'table', 0.75, features
        if n_horizontal >= 1 and n_vertical >= 1:
            if has_backrest:
                return 'chair', 0.80, features
            # 有水平面+垂直面但无靠背 → 可能是桌子侧板
            elif n_skeleton >= 2:
                return 'table', 0.70, features
            return 'chair', 0.55, features
        elif n_horizontal >= 1:
            if has_backrest:
                return 'chair', 0.65, features
            elif n_skeleton >= 2:
                return 'table', 0.70, features
            return 'table', 0.75, features
        else:
            return 'chair', 0.40, features

    # 无骨架 — 腿可能被RANSAC拆成了小平面
    # 小垂直面(<5%覆盖)通常是腿/扶手而非柜子侧板
    n_tiny_vert = sum(1 for r in planar_regions
                      if r['coverage'] < 0.05 and _is_vertical(r['normal'], gravity))
    # 无skeleton但有大量小平面+水平/垂直面 → 椅子(腿面被误判为planar)
    if n_horizontal >= 1 and n_vertical >= 1 and n_small_planar >= 3:
        return 'chair', 0.65, features
    # 有水平面+很多小平面 → table
    if n_horizontal >= 1 and n_small_planar >= 4 and n_planar >= 5:
        return 'table', 0.55, features
    # 极细长+少量水平面+有高度 → 椅子 (靠背+座面)
    # 极细长但扁平 → 平板 (桌子/木板)
    if bbox_wh_ratio > 10.0 and n_horizontal >= 2 and n_planar <= 4:
        all_grav = np.dot(all_pts, gravity)
        g_height = float(all_grav.ptp())  # 重力方向高度
        if g_height > min(bbox) * 3:  # 重力方向有明显高度 → 椅子
            return 'chair', 0.40, features
        else:
            return 'table', 0.40, features

    # 多平面无骨架 → cabinet (除非特征暗示椅子)
    if n_planar >= 2:
        if n_horizontal >= 1 and n_tiny_vert >= 2 and n_planar <= 6:
            return 'chair', 0.45, features
        if bbox_wh_ratio > 3.0 and n_horizontal >= 1 and n_vertical >= 1:
            return 'chair', 0.40, features
        # 只有水平面+少量平面+无垂直面无骨架 → 平板类, 更可能是简易桌子
        if n_horizontal >= 2 and n_vertical == 0 and n_planar <= 4:
            return 'table', 0.40, features
        return 'cabinet', 0.70, features

    # 单一大平面 → table (桌面), 非柜子
    if max_planar_cov > 0.8 and n_planar <= 2:
        return 'table', 0.50, features
    if max_planar_cov > 0.6:
        return 'cabinet', 0.50, features

    return 'cabinet', 0.35, features


def analyze_furniture(points, category=None):
    """
    完整的家具几何分析 (Point Cloud Geometry Analyzer).

    输入: (N,3) 点云, 可选类别名
    输出: {
        'category': 类别名,
        'geometry_class': 'G1'~'G4',
        'geometry_class_info': {...},
        'n_regions': 区域数,
        'parts': [{name, part_id, type, geometry_class, normal, coverage, points}, ...],
        'spray_summary': '喷涂策略摘要',
    }
    """
    regions = decompose_furniture(points)

    # 家具级几何大类
    geo_class = classify_geometry_class(regions)
    geo_info = GEOMETRY_CLASS[geo_class]

    # 部件命名
    if category:
        name_parts(category, regions)
    else:
        _name_generic_parts(regions)

    # 构建输出
    parts = []
    for r in regions:
        part = {
            'name': r.get('part_name', r['type']),
            'part_id': r.get('part_id', r['type']),
            'type': r['type'],
            'geometry': 'plane' if r['type'] == 'planar' else (
                'curve' if r['type'] == 'curved' else 'rod'),
            'spray': SURFACE_TYPES[r['type']]['path_algo'],
            'normal': r['normal'].tolist() if isinstance(r['normal'], np.ndarray) else r['normal'],
            'coverage': round(r['coverage'], 4),
            'n_points': len(r['points']),
            'points': r['points'],
        }
        parts.append(part)

    # 喷涂摘要
    strategies = []
    for t in ['planar', 'curved', 'skeleton']:
        n = sum(1 for p in parts if p['type'] == t)
        if n > 0:
            strategies.append(f"{n}×{SURFACE_TYPES[t]['name']}({SURFACE_TYPES[t]['path_algo']})")

    return {
        'category': category or 'unknown',
        'geometry_class': geo_class,
        'geometry_class_info': geo_info,
        'n_regions': len(regions),
        'parts': parts,
        'spray_summary': ' + '.join(strategies),
        'difficulty': geo_info['difficulty'],
    }

# ===================================================================
# 0.5 模型驱动部件分割 (替代几何启发式)
# ===================================================================
_PARTSEG_MODEL = None  # 全局单例

def get_partseg_model(checkpoint_path=None, config_path=None, device='cuda'):
    """获取/初始化 PartSegModel 单例"""
    global _PARTSEG_MODEL
    if _PARTSEG_MODEL is not None:
        return _PARTSEG_MODEL
    if not HAS_PARTSEG:
        raise ImportError("partseg_inference 模块不可用")

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__),
                                   'cfgs/partnet/pointnext-s.yaml')
    if checkpoint_path is None:
        import glob
        ckpts = sorted(glob.glob(os.path.join(
            os.path.dirname(__file__),
            'log/partnet/*/checkpoint/*_ckpt_best.pth')))
        if not ckpts:
            raise FileNotFoundError("未找到 PartNet checkpoint，请先训练模型")
        checkpoint_path = ckpts[-1]

    _PARTSEG_MODEL = PartSegModel(config_path, checkpoint_path, device)
    return _PARTSEG_MODEL


def _verify_part_surface(part_points, claimed_type, scale=None):
    """
    几何分析验证部件表面类型是否正确.

    返回: (verified_type, confidence)
      - 如果几何判断与声称的一致 → (claimed_type, >0.8)
      - 如果几何判断不同 → (corrected_type, <0.6)
    """
    if len(part_points) < 15:
        return claimed_type, 0.5
    normal = _estimate_normal(part_points)
    geometric_type = _classify_region_type(part_points, normal, scale)
    if geometric_type == claimed_type:
        return claimed_type, 0.90
    # 冲突: 几何覆盖模型
    return geometric_type, 0.50


def analyze_furniture_with_model(points, category=None, model=None):
    """
    融合部件分割: 模型提供语义部件名, 几何分析校正面类型.

    - chair/table: PointNeXt 预测部件 → 每个部件几何验证表面类型
    - cabinet: 纯几何启发式 + 规则命名
    - 冲突时几何结论优先 (喷漆路径选算法靠表面类型)

    输入: (N,3) 点云, 类别名, 可选模型实例
    输出: 与 analyze_furniture() 相同格式, 附加 _confidence 和 _warnings
    """
    _MODEL_CATEGORIES = {'chair', 'table'}
    scale = _compute_scale_factor(points)
    warnings = []

    if category and category.lower() in _MODEL_CATEGORIES and model is not None:
        # ── 模型路径: chair/table (需要 model 实例) ──
        try:
            regions = segment_furniture_with_model(model, points, category=category)
        except Exception as e:
            warnings.append({
                'part': '全局', 'issue': 'model_inference_failed',
                'detail': f'模型推理失败: {e}, 回退几何分析'
            })
            print(f"  [回退] 模型推理失败 → 纯几何分析")
            regions = None

        if regions is None:
            # 回退到纯几何
            geo_result = analyze_furniture(points, category=category)
            geo_result['_model_based'] = False
            geo_result['_warnings'] = warnings
            return geo_result

        parts = []
        for r in regions:
            rtype = r['type']  # 模型推断的表面类型
            proto = r.get('prototype', 'unknown')

            # 几何验证表面类型
            verified_type, surf_conf = _verify_part_surface(r['points'], rtype, scale)
            if verified_type != rtype:
                warnings.append({
                    'part': r['part_name'],
                    'issue': 'surface_type_mismatch',
                    'model_said': rtype,
                    'geometry_says': verified_type,
                    'detail': f"模型标记为{SURFACE_TYPES[rtype]['name']}, 几何分析判定为{SURFACE_TYPES[verified_type]['name']}"
                })
                rtype = verified_type

            # 模型预测置信度 (从逐点 softmax 均值估算)
            model_conf = float(np.mean(r.get('_point_conf', [0.7]))) if '_point_conf' in r else 0.70
            confidence = round(min(model_conf, surf_conf), 2)

            part = {
                'name': r['part_name'],
                'part_id': r['part_id'],
                'type': rtype,
                'geometry': 'plane' if rtype == 'planar' else (
                    'curve' if rtype == 'curved' else 'rod'),
                'spray': SURFACE_TYPES[rtype]['path_algo'],
                'normal': r['normal'].tolist() if isinstance(r['normal'], np.ndarray) else r['normal'],
                'coverage': round(r['coverage'], 4),
                'n_points': len(r['points']),
                'points': r['points'],
                'prototype': proto,
                'label': r.get('label', -1),
                'confidence': confidence,
            }
            parts.append(part)

        geo_class = classify_geometry_class(regions)
        model_based = True

    else:
        # ── 几何路径: cabinet 及未知类 ──
        geo_result = analyze_furniture(points, category=category)
        parts = geo_result['parts']
        geo_class = geo_result['geometry_class']
        # 每个部件默认置信度
        for p in parts:
            p['confidence'] = 0.70
            p['prototype'] = 'unknown'
            p['label'] = -1
        model_based = False

    geo_info = GEOMETRY_CLASS[geo_class]

    strategies = []
    for t in ['planar', 'curved', 'skeleton']:
        n = sum(1 for p in parts if p['type'] == t)
        if n > 0:
            strategies.append(f"{n}×{SURFACE_TYPES[t]['name']}({SURFACE_TYPES[t]['path_algo']})")

    return {
        'category': category or 'unknown',
        'geometry_class': geo_class,
        'geometry_class_info': geo_info,
        'n_regions': len(parts),
        'parts': parts,
        'spray_summary': ' + '.join(strategies) if strategies else 'N/A',
        'difficulty': geo_info['difficulty'],
        '_model_based': model_based,
        '_warnings': warnings,
    }


# ===================================================================
# 自检 & 纠正
# ===================================================================
def validate_segmentation(analysis):
    """
    分割结果自检 — 返回检查报告.

    检查项:
      1. 法向一致性: "座面"的法向必须接近重力方向
      2. 尺寸合理性: "腿"的宽度不超过物体宽度的 20%
      3. 空间关系: 靠背在座面后方/上方
      4. 覆盖率: 单个部件 < 2% → 可能是噪声
      5. 模型置信度: < 0.5 → 标记低置信

    返回: list of {part, check, level, detail}
      level: 'PASS' | 'WARN' | 'FAIL'
    """
    reports = []
    parts = analysis.get('parts', [])
    if not parts:
        return reports

    all_pts = np.vstack([p['points'] for p in parts])
    bbox = all_pts.max(0) - all_pts.min(0)
    object_width = max(bbox[0], bbox[1])
    gravity = np.array([0, 0, 1.0])

    # 找座面 (找最高的水平大面 → 桌面, 中等高度的水平面 → 座面)
    seat_candidates = [p for p in parts
                       if p.get('prototype') == 'horizontal_main_plane'
                       or '座面' in p.get('name', '')
                       or '桌面' in p.get('name', '')
                       or '顶板' in p.get('name', '')]

    leg_parts = [p for p in parts
                 if p.get('prototype') == 'vertical_leg'
                 or '腿' in p.get('name', '')]

    back_parts = [p for p in parts
                  if p.get('prototype') == 'end_vertical_plane'
                  or '靠背' in p.get('name', '')]

    for p in parts:
        name = p.get('name', '?')
        conf = p.get('confidence', 0.7)

        # 1. 法向检查
        normal = np.array(p['normal']) if isinstance(p['normal'], list) else p['normal']
        if '座面' in name or '桌面' in name or '顶板' in name or '床板' in name:
            dot = abs(np.dot(normal, gravity))
            if dot < 0.6:
                reports.append({
                    'part': name, 'check': '法向',
                    'level': 'WARN',
                    'detail': f'水平面法向偏斜 (dot={dot:.2f}, 期望>0.7)'
                })
            else:
                reports.append({'part': name, 'check': '法向', 'level': 'PASS', 'detail': f'dot={dot:.2f}'})

        # 2. 尺寸检查
        if '腿' in name or 'leg' in p.get('part_id', ''):
            leg_pts = p['points']
            leg_width = (leg_pts.max(0) - leg_pts.min(0))[:2].max()
            if leg_width > object_width * 0.25:
                reports.append({
                    'part': name, 'check': '尺寸',
                    'level': 'WARN',
                    'detail': f'腿宽度({leg_width:.3f})超过物体宽度20%({object_width*0.2:.3f})'
                })
            else:
                reports.append({'part': name, 'check': '尺寸', 'level': 'PASS', 'detail': f'宽度={leg_width:.3f}'})

        # 4. 覆盖率
        if p.get('coverage', 0) < 0.02:
            reports.append({
                'part': name, 'check': '覆盖率',
                'level': 'WARN',
                'detail': f'占比仅{p["coverage"]:.1%}, 可能为噪声'
            })

        # 5. 置信度
        if conf < 0.5:
            reports.append({
                'part': name, 'check': '置信度',
                'level': 'WARN',
                'detail': f'低置信度({conf:.2f}), 建议人工确认'
            })
        elif conf >= 0.7:
            reports.append({'part': name, 'check': '置信度', 'level': 'PASS', 'detail': f'{conf:.2f}'})

    # 3. 空间关系: 靠背 vs 座面
    if seat_candidates and back_parts:
        seat_center = seat_candidates[0]['points'].mean(0)
        back_center = back_parts[0]['points'].mean(0)
        if back_center[2] < seat_center[2] - 0.05:
            reports.append({
                'part': f'{back_parts[0]["name"]} vs {seat_candidates[0]["name"]}',
                'check': '位置',
                'level': 'FAIL',
                'detail': '靠背在座面下方'
            })
        else:
            reports.append({
                'part': f'{back_parts[0]["name"]} vs {seat_candidates[0]["name"]}',
                'check': '位置',
                'level': 'PASS',
                'detail': '靠背在座面上方/同高'
            })

    return reports


def apply_correction(analysis, correction_path):
    """
    应用用户纠正规则 (YAML).

    correction.yaml 格式:
      overrides:
        - match: "扶手"       # 匹配部件名 (模糊)
          type: "planar"      # 强制表面类型

    返回: 修改后的 analysis
    """
    import yaml as _yaml
    with open(correction_path) as f:
        rules = _yaml.safe_load(f)

    overrides = rules.get('overrides', [])
    for r in overrides:
        match_name = r.get('match', '').lower()
        new_type = r.get('type')
        for p in analysis['parts']:
            pname = p.get('name', '').lower()
            if match_name in pname and new_type and new_type in SURFACE_TYPES:
                old_type = p['type']
                p['type'] = new_type
                p['spray'] = SURFACE_TYPES[new_type]['path_algo']
                p['geometry'] = 'plane' if new_type == 'planar' else (
                    'curve' if new_type == 'curved' else 'rod')
                p['confidence'] = 1.0  # 人工确认
                analysis['_warnings'].append({
                    'part': p['name'], 'issue': 'user_correction',
                    'model_said': old_type, 'geometry_says': new_type,
                    'detail': f'用户纠正: {old_type} → {new_type}'
                })

    return analysis


def auto_pipeline(points, model=None, correction=None, visualize=True, save_html=False):
    """
    端到端自动管线: 分类 → 融合分割 → 自检 → 纠正 → 路径 → 可视化.

    输入: (N,3) 点云
    输出: 完整结果字典 + 自检报告
    """
    # Step 0: 自动分类
    category, cls_conf, features = classify_furniture_type(points)
    print(f"\n{'='*60}")
    print(f"[自动识别] {category} (置信度: {cls_conf:.0%})")
    print(f"  特征: {features['n_regions']}区域 "
          f"P:{features['n_planar']} C:{features['n_curved']} S:{features['n_skeleton']}")
    print(f"{'='*60}")

    # Step 1: 融合分割
    if model is None and category in ('chair', 'table'):
        try:
            model = get_partseg_model()
        except Exception as e:
            print(f"[警告] 模型加载失败: {e}, 回退几何分析")

    analysis = analyze_furniture_with_model(points, category=category, model=model)
    source = "模型+几何融合" if analysis.get('_model_based') else "几何启发式"
    print(f"[分割] {source} → {analysis['n_regions']}部件: {analysis['spray_summary']}")
    for p in analysis['parts']:
        tag = f" [conf={p.get('confidence',0):.2f}]" if 'confidence' in p else ""
        print(f"  [{p['name']}] {p['type']} → {p['spray']}{tag}")

    # Step 2: 自检
    print(f"\n[自检]")
    reports = validate_segmentation(analysis)
    n_warn = 0
    n_fail = 0
    for r in reports:
        icon = {'PASS': '  ✓', 'WARN': '  ⚠', 'FAIL': '  ✗'}[r['level']]
        print(f"{icon} [{r['check']}] {r['part']}: {r['detail']}")
        if r['level'] == 'WARN':
            n_warn += 1
        elif r['level'] == 'FAIL':
            n_fail += 1
    status = 'FAIL' if n_fail > 0 else ('WARN' if n_warn > 0 else 'PASS')
    print(f"  综合: {status} ({len(reports)}项检查, {n_warn}警告, {n_fail}失败)")

    # Step 3: 纠正
    if correction:
        print(f"\n[纠正] 应用: {correction}")
        analysis = apply_correction(analysis, correction)

    # Step 4: 从分析结果生成路径
    print("\n[路径生成]")
    regions = []
    for p in analysis['parts']:
        if isinstance(p['normal'], list):
            normal = np.array(p['normal'])
        else:
            normal = p['normal']
        regions.append({
            'points': p['points'],
            'type': p['type'],
            'normal': normal,
            'center': p['points'].mean(0),
            'coverage': p['coverage'],
            'part_name': p['name'],
            'part_id': p.get('part_id', ''),
        })

    regions_with_paths = []
    for ri, region in enumerate(regions):
        wp, wn, segs = generate_region_path(region)
        regions_with_paths.append((region, (wp, wn, segs)))
        pname = region.get('part_name', f'部件{ri+1}')
        print(f"  [{pname}] {SURFACE_TYPES[region['type']]['name']}: "
              f"{len(wp)} 路径点, {len(segs)} 段")

    combined_wp, combined_wn, combined_seg = merge_region_paths(regions_with_paths)
    print(f"  总路径: {len(combined_wp)}点, {len(combined_seg)}段")

    # Step 5: 可视化
    if visualize:
        print("\n[可视化]")
        visualize_decomposed(points, regions, combined_wp, combined_seg,
                            title=f"{category} | {source} | {analysis['spray_summary']}",
                            save_html=save_html)

    return {
        'name': category,
        'geometry_class': analysis['geometry_class'],
        'geometry_class_info': analysis['geometry_class_info'],
        'category': category,
        'n_regions': analysis['n_regions'],
        'regions': regions,
        'parts': analysis['parts'],
        'spray_summary': analysis['spray_summary'],
        'waypoints': combined_wp,
        'normals': combined_wn,
        'segments': combined_seg,
        '_auto_category': category,
        '_cls_confidence': cls_conf,
        '_validation': reports,
        '_validation_status': status,
        '_model_based': analysis.get('_model_based', False),
        '_warnings': analysis.get('_warnings', []),
    }


# ===================================================================
# 1. 区域分解 — 核心函数
# ===================================================================
def decompose_furniture(points, min_region_ratio=0.02):
    """
    将家具点云分解为不同类型的表面区域.

    策略 (v3 增强):
    1. 自适应尺度 + 逐点曲率预计算
    2. RANSAC 提取所有平面片 (自适应阈值)
    3. 曲率 + 截面圆度 + 长宽比 三特征分类 (planar/curved/skeleton)
    4. 合并相邻同向平面 → 大平面; 相邻异向平面 → 曲面
    5. 剩余点 DBSCAN → 曲率驱动归类
    6. 孤儿点距离上限分配 + 小碎片合并

    输入: (N,3) numpy 点云 (已归一化)
    输出: list of {points, type, normal, coverage}
    """
    N = len(points)
    scale = _compute_scale_factor(points)

    # ── 预计算逐点几何特征 (曲率+平面性+法向) ──
    global_planarity, global_curvature, global_normals = \
        _compute_pointwise_features(points)

    raw_regions = []
    used = np.zeros(N, dtype=bool)
    remaining_idx = np.arange(N)
    remaining_pts = points.copy()

    # 自适应 RANSAC 阈值
    ransac_thresh = 0.012 * scale   # 适中: 不过于严格, 捕获更多平面点
    dbscan_eps_plane = 0.06 * scale   # 平面内 DBSCAN 连通分离
    dbscan_eps_resid = 0.06 * scale   # 残余点 DBSCAN (降低以分离相邻腿)

    # ── 1. RANSAC 提取所有平面 → 曲率 + 截面法分类 ──
    for _ in range(40):
        if len(remaining_pts) < max(12, N * 0.01):
            break
        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(remaining_pts)
        try:
            plane_model, inliers = pcd_t.segment_plane(
                distance_threshold=ransac_thresh, ransac_n=3, num_iterations=800)
        except RuntimeError:
            break
        if len(inliers) / N < 0.005:
            break

        a, b, c, d = plane_model
        ransac_normal = np.array([a, b, c])
        ransac_normal /= np.linalg.norm(ransac_normal) + 1e-10
        plane_pts = remaining_pts[inliers]
        global_inliers = remaining_idx[inliers]  # 映射回原始索引

        # 连通分量分离
        if len(plane_pts) > 20:
            pcd_p = o3d.geometry.PointCloud()
            pcd_p.points = o3d.utility.Vector3dVector(plane_pts)
            try:
                cl = np.array(pcd_p.cluster_dbscan(
                    eps=dbscan_eps_plane, min_points=12, print_progress=False))
            except Exception:
                cl = np.zeros(len(plane_pts), dtype=int)

            any_used = False
            for lbl in set(cl):
                if lbl == -1:
                    continue
                cm = cl == lbl
                cpts = plane_pts[cm]
                if len(cpts) / N < 0.005:
                    continue
                gids = global_inliers[cm]  # 原始索引
                rtype = _classify_skeleton(
                    cpts, ransac_normal, scale,
                    global_curvature[gids], global_planarity[gids])
                raw_regions.append({
                    'points': cpts, 'type': rtype,
                    'normal': ransac_normal,
                    'center': cpts.mean(0),
                    'coverage': len(cpts) / N,
                    '_orig_ids': gids,
                })
                used[gids] = True
                any_used = True

            # 兜底: 整块
            if not any_used and len(plane_pts) / N >= 0.005:
                rtype = _classify_skeleton(
                    plane_pts, ransac_normal, scale,
                    global_curvature[global_inliers], global_planarity[global_inliers])
                raw_regions.append({
                    'points': plane_pts, 'type': rtype,
                    'normal': ransac_normal,
                    'center': plane_pts.mean(0),
                    'coverage': len(plane_pts) / N,
                    '_orig_ids': global_inliers,
                })
                used[global_inliers] = True

        elif len(plane_pts) / N >= 0.005:
            rtype = _classify_skeleton(
                plane_pts, ransac_normal, scale,
                global_curvature[global_inliers], global_planarity[global_inliers])
            raw_regions.append({
                'points': plane_pts, 'type': rtype,
                'normal': ransac_normal,
                'center': plane_pts.mean(0),
                'coverage': len(plane_pts) / N,
                '_orig_ids': global_inliers,
            })
            used[global_inliers] = True

        mask = np.ones(len(remaining_pts), dtype=bool)
        mask[inliers] = False
        remaining_pts = remaining_pts[mask]
        remaining_idx = remaining_idx[mask]

    # ── 2. 合并: 同向平面→大平面, 异向相邻平面→曲面 ──
    regions = _merge_with_curved_detection(raw_regions, N, scale)

    # ── 2.5 清理: 高平面性残余点 → 归入最近平面区域 (消除腿间桥接) ──
    if len(regions) > 0:
        rest_mask = ~used
        rest_gidx = np.where(rest_mask)[0]
        planar_region_centers = np.array([r['center'] for r in regions if r['type'] == 'planar'])
        if len(planar_region_centers) > 0:
            for gi in rest_gidx:
                pt = points[gi]
                if global_planarity[gi] > 0.5:
                    dists = np.linalg.norm(planar_region_centers - pt, axis=1)
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < 0.3 * scale:
                        # 找到对应的 region 并合并
                        pi = [j for j, r in enumerate(regions) if r['type'] == 'planar'][nearest]
                        regions[pi]['points'] = np.vstack([regions[pi]['points'], pt.reshape(1, 3)])
                        regions[pi]['center'] = regions[pi]['points'].mean(0)
                        regions[pi]['coverage'] = len(regions[pi]['points']) / N
                        used[gi] = True

    # ── 3. DBSCAN 剩余点 (曲率驱动分类) ──
    rest_mask = ~used
    rest_pts = points[rest_mask]
    rest_gidx = np.where(rest_mask)[0]
    if len(rest_pts) > 15:
        pcd_r = o3d.geometry.PointCloud()
        pcd_r.points = o3d.utility.Vector3dVector(rest_pts)
        try:
            rl = np.array(pcd_r.cluster_dbscan(
                eps=dbscan_eps_resid, min_points=6, print_progress=False))
        except Exception:
            rl = np.zeros(len(rest_pts), dtype=int)

        for lbl in set(rl):
            if lbl == -1:
                continue
            cm = rl == lbl
            cpts = rest_pts[cm]
            if len(cpts) / N < 0.003:
                continue
            gids = rest_gidx[cm]
            normal = _estimate_normal(cpts)
            rtype = _classify_skeleton(
                cpts, normal, scale,
                global_curvature[gids], global_planarity[gids])
            regions.append({
                'points': cpts, 'type': rtype,
                'normal': normal,
                'center': cpts.mean(0),
                'coverage': len(cpts) / N,
            })
            used[gids] = True

    # ── 3.5 腿拆分: 所有区域用更小 eps 二次 DBSCAN 分离单根腿 ──
    leg_eps = 0.03 * scale  # 比 dbscan_eps_resid 小很多, 分离相邻细杆
    split_regions = []
    for r in regions:
        if len(r['points']) < 20:
            split_regions.append(r)
            continue
        pcd_l = o3d.geometry.PointCloud()
        pcd_l.points = o3d.utility.Vector3dVector(r['points'])
        try:
            ll = np.array(pcd_l.cluster_dbscan(
                eps=leg_eps, min_points=6, print_progress=False))
        except Exception:
            ll = np.zeros(len(r['points']), dtype=int)

        # 按 DBSCAN 标签拆分
        n_clusters = len(set(ll) - {-1})
        if n_clusters <= 1:
            split_regions.append(r)
            continue

        for lbl in set(ll):
            if lbl == -1:
                continue
            cm = ll == lbl
            cpts = r['points'][cm]
            if len(cpts) / N < 0.003:
                continue
            normal = _estimate_normal(cpts)
            # 重新判定子区域类型
            subtype = _classify_region_type(cpts, normal, scale)
            split_regions.append({
                'points': cpts,
                'type': subtype,
                'normal': normal,
                'center': cpts.mean(0),
                'coverage': len(cpts) / N,
            })
        # 噪声点附加到最近的拆分子区域
        noise_mask = ll == -1
        if noise_mask.any():
            sub_regions = split_regions[-n_clusters:]
            if len(sub_regions) > 0:
                noise_pts = r['points'][noise_mask]
                centers = np.array([sr['center'] for sr in sub_regions])
                for pt in noise_pts:
                    dists = np.linalg.norm(centers - pt, axis=1)
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < leg_eps * 2:
                        sub_regions[nearest]['points'] = np.vstack([
                            sub_regions[nearest]['points'], pt.reshape(1, 3)])
                        sub_regions[nearest]['center'] = sub_regions[nearest]['points'].mean(0)
                        sub_regions[nearest]['coverage'] = len(sub_regions[nearest]['points']) / N

    regions = split_regions

    # ── 3.6 杆件聚合: 相邻小平面区域合成 skeleton (方腿每面被RANSAC单独检出) ──
    group_eps = 0.12 * scale  # 聚合距离: 相邻面在此距离内认为属同一杆件
    small_planar = [(i, r) for i, r in enumerate(regions)
                    if r['type'] == 'planar' and r['coverage'] < 0.08 and len(r['points']) >= 15]
    # 按距离聚类小平面
    grouped_indices = []  # list of sets
    used_for_group = set()
    for i, (si, sr) in enumerate(small_planar):
        if si in used_for_group:
            continue
        group = {si}
        # 扩张: 找所有触达的小平面
        changed = True
        while changed:
            changed = False
            for sj, s2 in small_planar:
                if sj in group or sj in used_for_group:
                    continue
                for gk in list(group):
                    d = np.linalg.norm(regions[gk]['center'] - regions[sj]['center'])
                    if d < group_eps:
                        group.add(sj)
                        changed = True
                        break
        if len(group) >= 2:
            # 检查合并后是否呈柱状
            merged_pts = np.vstack([regions[gi]['points'] for gi in group])
            normal = _estimate_normal(merged_pts)
            merged_rtype = _classify_region_type(merged_pts, normal, scale)
            if merged_rtype == 'skeleton':
                grouped_indices.append(group)
                used_for_group.update(group)
        else:
            used_for_group.add(si)

    if grouped_indices:
        new_regions = [r for i, r in enumerate(regions) if i not in used_for_group]
        for group in grouped_indices:
            merged_pts = np.vstack([regions[gi]['points'] for gi in group])
            normal = _estimate_normal(merged_pts)
            new_regions.append({
                'points': merged_pts,
                'type': 'skeleton',
                'normal': normal,
                'center': merged_pts.mean(0),
                'coverage': len(merged_pts) / N,
            })
        regions = new_regions

    # ── 4. 孤儿点 → 最近区域 (距离上限 = 0.15 * scale) ──
    orphan = points[~used]
    if len(orphan) > 0 and len(regions) > 0:
        centers = np.array([r['center'] for r in regions])
        orphan_max_dist = 0.08 * scale
        for pt in orphan:
            dists = np.linalg.norm(centers - pt, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < orphan_max_dist:
                regions[nearest]['points'] = np.vstack([
                    regions[nearest]['points'], pt.reshape(1, 3)])

    # ── 5. 清理: 合并小碎片到最近的同类型区域 ──
    for r in regions:
        r['coverage'] = len(r['points']) / N
        r['center'] = r['points'].mean(0)
        r.pop('_orig_ids', None)

    # 迭代合并小碎片 (覆盖率 < 3%)
    fragment_merge_dist = 0.8 * scale
    changed = True
    while changed:
        changed = False
        for i in range(len(regions) - 1, -1, -1):
            if regions[i]['coverage'] >= 0.03 or len(regions) <= 2:
                continue
            best_j, best_d = -1, float('inf')
            for j, rj in enumerate(regions):
                if j == i or rj['type'] != regions[i]['type']:
                    continue
                d = np.linalg.norm(regions[i]['center'] - rj['center'])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d < fragment_merge_dist:
                regions[best_j]['points'] = np.vstack([
                    regions[best_j]['points'], regions[i]['points']])
                regions[best_j]['coverage'] = len(regions[best_j]['points']) / N
                regions[best_j]['center'] = regions[best_j]['points'].mean(0)
                regions.pop(i)
                changed = True

    if len(regions) == 0:
        regions.append({
            'points': points, 'type': 'curved',
            'normal': np.array([0, 0, 1]),
            'coverage': 1.0, 'center': points.mean(0)})

    return regions


def _classify_skeleton(points, normal, scale=None, curvatures=None, planarity_vals=None):
    """截面分析 + 曲率 + 圆度 → planar | curved | skeleton.

    兼容旧调用签名, 内部委托给 _classify_region_type."""
    return _classify_region_type(points, normal, scale, curvatures, planarity_vals)


def _estimate_normal(points):
    """PCA 估计主法向"""
    centered = points - points.mean(0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, 0]


# ===================================================================
# 几何特征增强: 曲率 + 自适应阈值 + 截面圆度
# ===================================================================
def _compute_scale_factor(points):
    """点云包围盒对角线, 用于归一化所有绝对阈值. 下限 0.3 避免极小物体除零."""
    bbox = points.max(0) - points.min(0)
    diag = float(np.linalg.norm(bbox))
    return max(diag, 0.3)


def _compute_pointwise_features(points, k=30):
    """逐点局部 PCA 特征 — 曲率 + 平面性 + 法向.

    用 Open3D KDTree 搜索 k 近邻, 对邻域协方差矩阵做 PCA:
      λ₁ ≤ λ₂ ≤ λ₃  (升序)
      curvature  = λ₁ / (λ₁+λ₂+λ₃)   → 高 = 弯曲/边缘
      planarity  = (λ₂-λ₁) / λ₃      → 高 = 平坦

    Returns:
      planarity:  (N,)  float
      curvature:  (N,)  float
      normals:    (N,3) 局部法向 (最小特征向量)
    """
    N = len(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    planarity = np.zeros(N, dtype=np.float64)
    curvature = np.zeros(N, dtype=np.float64)
    normals = np.zeros((N, 3), dtype=np.float64)
    k_actual = min(k, N - 1)

    for i in range(N):
        _, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], k_actual + 1)
        idx = idx[1:]  # 跳过自身
        nb = points[idx]
        c = nb - nb.mean(0)
        if len(c) < 3:
            planarity[i] = 0.5
            curvature[i] = 0.0
            normals[i] = [0, 0, 1]
            continue
        cov = np.cov(c.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        lam1, lam2, lam3 = eigvals[0], eigvals[1], eigvals[2]
        total = lam1 + lam2 + lam3 + 1e-10
        curvature[i] = lam1 / total
        planarity[i] = (lam2 - lam1) / (lam3 + 1e-10)
        normals[i] = eigvecs[:, 0]

    return planarity.astype(np.float32), curvature.astype(np.float32), normals.astype(np.float32)


def _classify_region_type(points, normal, scale=None,
                          curvatures=None, planarity_vals=None):
    """将点簇分类为 planar | curved | skeleton.

    三关:
      1. 曲率关 — 逐点曲率均值高 → curved (真正的曲面, 非拼装)
      2. 骨架关 — PCA 长宽比 > 4 + 截面圆度检查 (区分细杆 vs 薄板)
      3. 默认 → planar

    Args:
      points:           (M,3) 点坐标
      normal:           (3,) 参考法向
      scale:            包围盒对角线, 自适应阈值; None=自动计算
      curvatures:       (M,) 预计算的逐点曲率, None=现场采样计算
      planarity_vals:   (M,) 预计算的逐点平面性, None=现场采样计算
    """
    Np = len(points)
    if Np < 15:
        return 'planar'

    if scale is None:
        scale = _compute_scale_factor(points)

    # ── 0. 全局 PCA 扁平度 (整体形状, 不受边缘效应影响) ──
    global_centered = points - points.mean(0)
    global_cov = np.cov(global_centered.T)
    global_eig = np.linalg.eigvalsh(global_cov)
    global_eig.sort()  # 升序
    global_flatness = global_eig[0] / (global_eig[1] + 1e-10)  # 越小越扁平

    # ── 1. 曲率特征 ──
    if curvatures is not None and len(curvatures) == Np:
        mean_curve = float(np.mean(curvatures))
        mean_planarity = float(np.mean(planarity_vals)) if planarity_vals is not None else 0.5
    else:
        # 采样计算 (最多 150 点)
        n_sample = min(150, Np)
        idx_sample = np.random.choice(Np, n_sample, replace=False)
        sp, sc, _ = _compute_pointwise_features(points[idx_sample], k=min(25, Np - 1))
        mean_curve = float(sc.mean())
        mean_planarity = float(sp.mean())

    # 高平面性 → 直接判为平面 (跳过后续曲率/骨架判断)
    # 降低阈值: 薄板边缘局部邻域平面性也会下降, 只要整体趋势是平的即可
    if mean_planarity > 0.35:
        return 'planar'

    # 全局扁平 → 平面 (薄板保护: 整体形状是片状, 不会被局部曲率误导)
    if global_flatness < 0.15:
        return 'planar'

    # ── 2. PCA 长宽比 + 截面分析 (必须在曲率之前: 细杆的局部邻域呈3D, 曲率高) ──
    centered = points - points.mean(0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]  # 降序
    axis1 = eigvecs[:, idx[0]]

    proj1 = np.dot(centered, axis1)
    t_min, t_max = proj1.min(), proj1.max()
    length = t_max - t_min

    if length > 1e-8:
        # 沿主轴切 3~10 段, 每段统计截面宽度和圆度
        n_seg = min(10, max(3, Np // 30))
        widths, circularities = [], []
        for i in range(n_seg):
            t0 = t_min + length * i / n_seg
            t1 = t_min + length * (i + 1) / n_seg
            mask = (proj1 >= t0) & (proj1 < t1)
            if mask.sum() < 5:
                continue
            seg = centered[mask]
            radial = seg - np.outer(np.dot(seg, axis1), axis1)
            r = np.linalg.norm(radial, axis=1)
            w = float(np.median(r)) * 2
            if w < 1e-8:
                continue
            widths.append(w)
            circ = 1.0 - min(float(np.std(r) / (np.mean(r) + 1e-10)), 1.0)
            circularities.append(circ)

        if widths:
            median_width = float(np.median(widths))
            aspect_ratio = length / (median_width + 1e-10)
            mean_circ = float(np.mean(circularities)) if circularities else 0.0
            skel_width_thresh = 0.22 * scale

            if aspect_ratio > 3.0 and median_width < skel_width_thresh:
                # 截面扁平 → 是薄板不是杆
                if mean_circ < 0.30 and aspect_ratio < 5.0:
                    return 'planar'
                return 'skeleton'

    # ── 3. 曲率关 (骨架之后: 排除细杆误判) ──
    # 曲率显著且平面性低 → 曲面
    if mean_curve > 0.03 and mean_planarity < 0.40:
        return 'curved'
    # 中等曲率 + 很低平面性 → 曲面 (球面/自由曲面)
    if mean_curve > 0.015 and mean_planarity < 0.25:
        return 'curved'

    return 'planar'


def _merge_with_curved_detection(raw_regions, N, scale=None):
    """合并相邻区域 (自适应距离阈值):
    - 同向 planar → 合并为更大 planar
    - 相邻 planar 但法向不同 → 合并为 curved (曲面由多片不同向平面拼成)
    - 相邻 skeleton → 合并
    - curved 区域直接合并 (已由曲率分类识别)
    """
    if scale is None:
        all_pts = np.vstack([r['points'] for r in raw_regions]) if raw_regions else np.zeros((1, 3))
        scale = _compute_scale_factor(all_pts)

    n = len(raw_regions)
    if n == 0:
        return []
    if n == 1:
        r = raw_regions[0]
        return [{'points': r['points'], 'type': r['type'],
                 'normal': r['normal'], 'coverage': r['coverage'],
                 'center': r['center']}]

    # 自适应合并距离
    d_skel = 0.08 * scale    # 骨架合并 (降低: 避免不同腿合并)
    d_planar = 0.6 * scale   # 平面合并
    d_curved = 0.3 * scale   # 曲面合并
    d_adjacent = 0.35 * scale  # 曲面检测邻接距离

    # 预处理: 计算每个 region 的点云法向 (供后续曲面判断)
    for r in raw_regions:
        if 'pcd_normals' not in r and len(r['points']) >= 30:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(r['points'])
            try:
                pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamKNN(min(30, len(r['points']))))
                r['pcd_normals'] = np.asarray(pcd.normals)
            except Exception:
                r['pcd_normals'] = None
        else:
            r['pcd_normals'] = None

    flags = [False] * n
    merged = []

    for i in range(n):
        if flags[i]:
            continue
        flags[i] = True
        cur = {'points': raw_regions[i]['points'].copy(),
               'type': raw_regions[i]['type'],
               'normal': raw_regions[i]['normal'].copy(),
               'center': raw_regions[i]['center'].copy(),
               'coverage': raw_regions[i]['coverage'],
               'pcd_normals': raw_regions[i]['pcd_normals']}

        # 迭代合并同类
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if flags[j]:
                    continue
                if raw_regions[j]['type'] != cur['type']:
                    continue
                dist = np.linalg.norm(cur['center'] - raw_regions[j]['center'])

                if cur['type'] == 'skeleton':
                    if dist < d_skel:
                        cur['points'] = np.vstack([cur['points'], raw_regions[j]['points']])
                        cur['center'] = cur['points'].mean(0)
                        cur['coverage'] = len(cur['points']) / N
                        cur['pcd_normals'] = None
                        flags[j] = True
                        changed = True

                elif cur['type'] == 'planar':
                    ns = abs(np.dot(cur['normal'], raw_regions[j]['normal']))
                    if dist < d_planar and ns > 0.7:
                        cur['points'] = np.vstack([cur['points'], raw_regions[j]['points']])
                        cur['center'] = cur['points'].mean(0)
                        cur['coverage'] = len(cur['points']) / N
                        cur['pcd_normals'] = None
                        flags[j] = True
                        changed = True

                else:  # curved
                    if dist < d_curved:
                        cur['points'] = np.vstack([cur['points'], raw_regions[j]['points']])
                        cur['center'] = cur['points'].mean(0)
                        cur['coverage'] = len(cur['points']) / N
                        flags[j] = True
                        changed = True

        merged.append(cur)

    # ── 相邻 planar 但法向渐变(非尖角) → 合并为 curved (兜底逻辑) ──
    planar_idx = [i for i, r in enumerate(merged) if r['type'] == 'planar']
    if len(planar_idx) >= 3:
        group_flags = [False] * len(planar_idx)
        new_curved = []
        for pi, i in enumerate(planar_idx):
            if group_flags[pi]:
                continue
            group = [i]
            group_flags[pi] = True
            changed = True
            while changed:
                changed = False
                for pj, j in enumerate(planar_idx):
                    if group_flags[pj]:
                        continue
                    for g in group:
                        d = np.linalg.norm(merged[g]['center'] - merged[j]['center'])
                        if d < d_adjacent:
                            group.append(j)
                            group_flags[pj] = True
                            changed = True
                            break
            if len(group) >= 3:
                g_normals = np.array([merged[g]['normal'] for g in group])
                g_ref = np.median(g_normals, axis=0)
                g_ref /= np.linalg.norm(g_ref) + 1e-10
                g_aligned = []
                for gn in g_normals:
                    if np.dot(gn, g_ref) < 0:
                        gn = -gn
                    g_aligned.append(gn)
                g_aligned = np.array(g_aligned)

                min_ns = 1.0
                for a in range(len(g_aligned)):
                    for b in range(a + 1, len(g_aligned)):
                        ns_ab = abs(np.dot(g_aligned[a], g_aligned[b]))
                        if ns_ab < min_ns:
                            min_ns = ns_ab
                if min_ns < 0.30 or min_ns > 0.85:
                    continue

                g_c = g_aligned - g_aligned.mean(0)
                g_cov = np.cov(g_c.T)
                g_eig = np.linalg.eigvalsh(g_cov)
                g_eig = np.sort(np.abs(g_eig))[::-1]
                g_spread = g_eig[1] / (g_eig[0] + 1e-10)
                if g_spread > 0.10:
                    merged_pts = np.vstack([merged[g]['points'] for g in group])
                    new_curved.append({
                        'points': merged_pts,
                        'type': 'curved',
                        'normal': g_ref,
                        'center': merged_pts.mean(0),
                        'coverage': len(merged_pts) / N,
                    })
                    for g in group:
                        merged[g]['_remove'] = True

        merged = [r for r in merged if not r.get('_remove')]
        merged.extend(new_curved)

    # 清理临时字段
    for r in merged:
        r.pop('pcd_normals', None)
        r.pop('_remove', None)

    return merged



# ===================================================================
# 2. 区域级路径生成
# ===================================================================
# ===================================================================
# 2. 区域级路径生成
# ===================================================================
def generate_region_path(region, spacing=0.06):
    """为单个区域生成喷涂路径"""
    points = region['points']
    rtype = region['type']

    if rtype == 'planar':
        return _path_planar_raster(points, region['normal'], spacing)
    elif rtype == 'curved':
        return _path_curved_contour(points, spacing)
    elif rtype == 'skeleton':
        return _path_skeleton_helix(points, spacing)
    else:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), dtype=int)


def _path_planar_raster(points, normal, spacing):
    """平面2D栅格: 投影到平面, 之字形扫描"""
    normal = normal / (np.linalg.norm(normal) + 1e-10)

    # 找平面上两个方向
    basis_z = np.array([0, 0, 1])
    if abs(np.dot(normal, basis_z)) > 0.95:
        basis_z = np.array([1, 0, 0])
    u = basis_z - np.dot(basis_z, normal) * normal
    u /= np.linalg.norm(u) + 1e-10
    v = np.cross(normal, u)

    # 投影到平面坐标系
    center = points.mean(0)
    proj_u = np.dot(points - center, u)
    proj_v = np.dot(points - center, v)

    u_min, u_max = proj_u.min(), proj_u.max()
    v_min, v_max = proj_v.min(), proj_v.max()

    wp, segs, norms = [], [], []
    u_vals = np.arange(u_min, u_max + spacing * 0.5, spacing)
    for i, ui in enumerate(u_vals):
        v_start = v_min if i % 2 == 0 else v_max
        v_end = v_max if i % 2 == 0 else v_min
        p1 = center + u * ui + v * v_start
        p2 = center + u * ui + v * v_end
        wp.extend([p1, p2])
        norms.extend([normal, normal])
        if i > 0:
            segs.append([len(wp) - 2, len(wp) - 1])

    return (np.array(wp) if wp else np.zeros((0, 3)),
            np.array(norms) if norms else np.zeros((0, 3)),
            np.array(segs) if segs else np.zeros((0, 2), dtype=int))


def _path_curved_contour(points, spacing):
    """曲面等高线跟随: 多层切片轮廓"""
    wp, norms, segs = [], [], []
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    n_levels = max(4, min(20, int((z_max - z_min) / spacing)))

    for zi, z in enumerate(np.linspace(z_min + 0.01, z_max - 0.01, n_levels)):
        slice_mask = np.abs(points[:, 2] - z) < spacing * 0.6
        slice_pts = points[slice_mask]
        if len(slice_pts) < 8:
            continue

        # XY平面角度排序 → 轮廓
        xy = slice_pts[:, :2]
        center = xy.mean(0)
        angles = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = slice_pts[order[:min(len(order), 30)]]

        for i, p in enumerate(ordered):
            wp.append(p)
            # 局部法向
            local_mask = np.linalg.norm(slice_pts - p, axis=1) < spacing * 1.5
            if local_mask.sum() >= 5:
                local = slice_pts[local_mask]
                cov_l = np.cov((local - p).T)
                _, eigvecs = np.linalg.eigh(cov_l)
                norms.append(eigvecs[:, 0])
            else:
                norms.append(np.array([0, 0, 1]))

        if len(ordered) > 1:
            st = len(wp) - len(ordered)
            for j in range(len(ordered) - 1):
                segs.append([st + j, st + j + 1])

    return (np.array(wp) if wp else np.zeros((0, 3)),
            np.array(norms) if norms else np.zeros((0, 3)),
            np.array(segs) if segs else np.zeros((0, 2), dtype=int))


def _path_skeleton_helix(points, spacing):
    """细长结构轴线环绕: 沿主轴螺旋"""
    wp, norms, segs = [], [], []
    if len(points) < 10:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), dtype=int)

    # PCA找主轴
    centered = points - points.mean(0)
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]  # 最大特征值方向(长轴)

    # 沿主轴投影
    t_vals = np.dot(points, axis)
    t_min, t_max = t_vals.min(), t_vals.max()
    center = points.mean(0)
    radius = np.median(np.linalg.norm(
        centered - np.outer(np.dot(centered, axis), axis), axis=1))

    n_steps = max(6, int((t_max - t_min) / spacing))
    for i, t in enumerate(np.linspace(t_min, t_max, n_steps)):
        for ang in np.linspace(0, 2 * np.pi, 8):
            p = center + axis * t
            radial_2d = np.array([np.cos(ang), np.sin(ang)])
            # 投影到垂直于轴的平面
            radial_3d = radial_2d[0] * _perpendicular(axis)[0] + radial_2d[1] * _perpendicular(axis)[1]
            p = p + radial_3d * radius * 0.7
            wp.append(p)
            # 法向朝外
            n = radial_3d / (np.linalg.norm(radial_3d) + 1e-10)
            norms.append(n)

    # 连接相邻点成段
    for i in range(len(wp) - 1):
        segs.append([i, i + 1])

    return (np.array(wp) if wp else np.zeros((0, 3)),
            np.array(norms) if norms else np.zeros((0, 3)),
            np.array(segs) if segs else np.zeros((0, 2), dtype=int))


def _perpendicular(v):
    """返回两个与v正交的单位向量"""
    if abs(v[2]) < 0.9:
        u1 = np.cross(v, [0, 0, 1])
    else:
        u1 = np.cross(v, [1, 0, 0])
    u1 /= np.linalg.norm(u1) + 1e-10
    u2 = np.cross(v, u1)
    u2 /= np.linalg.norm(u2) + 1e-10
    return u1, u2


# ===================================================================
# 3. 合并区域路径
# ===================================================================
def merge_region_paths(regions_with_paths):
    """合并各区域路径, 区域间加过渡线"""
    all_wp, all_n, all_seg = [], [], []
    offset = 0

    for region, (wp, wn, segs) in regions_with_paths:
        if len(wp) == 0:
            continue
        all_wp.extend(wp)
        all_n.extend(wn)
        for s in segs:
            all_seg.append([int(s[0]) + offset, int(s[1]) + offset])
        offset += len(wp)

    return (np.array(all_wp) if all_wp else np.zeros((0, 3)),
            np.array(all_n) if all_n else np.zeros((0, 3)),
            np.array(all_seg) if all_seg else np.zeros((0, 2), dtype=int))


# ===================================================================
# 4. 可视化
# ===================================================================
def _try_open3d():
    """检测 Open3D EGL 离屏渲染是否可用"""
    try:
        import open3d as _o3d
        render = _o3d.visualization.rendering.OffscreenRenderer(10, 10)
        del render
        return True
    except Exception:
        return False

_HAS_OPEN3D_WINDOW = None

def _can_use_open3d():
    global _HAS_OPEN3D_WINDOW
    if _HAS_OPEN3D_WINDOW is None:
        _HAS_OPEN3D_WINDOW = _try_open3d()
    return _HAS_OPEN3D_WINDOW


def render_geometries_to_file(geometries, filepath, title="", width=1200, height=800,
                               front=None, lookat=None, up=None):
    """EGL 离屏渲染几何体列表到 PNG 文件 (Wayland 兼容)"""
    import os as _os
    render = o3d.visualization.rendering.OffscreenRenderer(width, height)

    bbox = None
    for geo in geometries:
        if hasattr(geo, 'get_axis_aligned_bounding_box'):
            b = geo.get_axis_aligned_bounding_box()
            if bbox is None:
                bbox = b
            else:
                bbox.min_bound = np.minimum(bbox.min_bound, b.min_bound)
                bbox.max_bound = np.maximum(bbox.max_bound, b.max_bound)
        elif hasattr(geo, 'get_max_bound'):
            mn, mx = geo.get_min_bound(), geo.get_max_bound()
            if bbox is None:
                bbox = o3d.geometry.AxisAlignedBoundingBox(mn, mx)
            else:
                bbox.min_bound = np.minimum(bbox.min_bound, mn)
                bbox.max_bound = np.maximum(bbox.max_bound, mx)

    if bbox is None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            np.array([-1, -1, -1]), np.array([1, 1, 1]))

    for i, geo in enumerate(geometries):
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = 'defaultUnlit'

        if isinstance(geo, o3d.geometry.PointCloud):
            render.scene.add_geometry(f'geo_{i}', geo, mtl)
        elif isinstance(geo, o3d.geometry.LineSet):
            mtl_line = o3d.visualization.rendering.MaterialRecord()
            mtl_line.shader = 'unlitLine'
            mtl_line.line_width = 2.0
            render.scene.add_geometry(f'geo_{i}', geo, mtl_line)
        elif isinstance(geo, o3d.geometry.TriangleMesh):
            render.scene.add_geometry(f'geo_{i}', geo, mtl)
        elif isinstance(geo, o3d.geometry.AxisAlignedBoundingBox):
            mn = geo.min_bound
            mx = geo.max_bound
            corners = np.array([
                [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]], [mn[0], mn[1], mx[2]],
                [mx[0], mx[1], mn[2]], [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
            ])
            lines = np.array([
                [0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],
                [3,5],[3,6],[4,7],[5,7],[6,7]
            ])
            ls_box = o3d.geometry.LineSet()
            ls_box.points = o3d.utility.Vector3dVector(corners)
            ls_box.lines = o3d.utility.Vector2iVector(lines)
            color = geo.color if hasattr(geo, 'color') else [1.0, 0.0, 0.0]
            ls_box.paint_uniform_color(color)
            mtl_bbox = o3d.visualization.rendering.MaterialRecord()
            mtl_bbox.shader = 'unlitLine'
            mtl_bbox.line_width = 3.0
            render.scene.add_geometry(f'geo_{i}', ls_box, mtl_bbox)

    center = bbox.get_center().astype(np.float64)
    extent = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    if front is None:
        front = np.array([0.3, -1.0, 0.5], dtype=np.float64)
    eye = center + front / np.linalg.norm(front) * extent * 1.5
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    render.setup_camera(60.0, center.astype(np.float32),
                        eye.astype(np.float32), up.astype(np.float32))
    img = render.render_to_image()
    _os.makedirs(_os.path.dirname(filepath) or '.', exist_ok=True)
    o3d.io.write_image(filepath, img)
    del render
    return filepath


def visualize_geometries_browser(geometries, labels=None, title="3D Viewer",
                                    width=1400, height=800, output_path=None):
    """
    用 Plotly 在浏览器中显示可交互 3D 视图.

    特性:
      - orbit 自由旋转 (类似 CAD 软件)
      - 深色背景 + 图例
      - hover 显示部件名
      - 自适应点大小
      - 自动适配相机视角

    参数:
      geometries: list of Open3D geometries
      labels: 每个 geometry 的标签 (用于图例/hover), None=不显示
      title: 标题
      output_path: 输出 HTML 路径, None=自动生成
    """
    import plotly.graph_objects as go
    import os as _os, subprocess as _sp

    traces = []
    legend_shown = set()
    all_points_list = []

    for gi, geo in enumerate(geometries):
        label = labels[gi] if labels and gi < len(labels) else None
        show_legend = label and label not in legend_shown
        if show_legend:
            legend_shown.add(label)

        if isinstance(geo, o3d.geometry.PointCloud):
            pts = np.asarray(geo.points)
            colors = np.asarray(geo.colors) if geo.has_colors() else None
            all_points_list.append(pts)

            # 自适应采样: 总点数越多, 采样越稀疏
            n_pts = len(pts)
            if n_pts > 4000:
                n_sample = 4000
            elif n_pts > 2000:
                n_sample = min(n_pts, 3000)
            else:
                n_sample = n_pts

            if n_pts > n_sample:
                idx = np.random.choice(n_pts, n_sample, replace=False)
                pts = pts[idx]
                if colors is not None:
                    colors = colors[idx]

            # 自适应点大小
            marker_size = 3 if n_pts < 1000 else (2 if n_pts < 3000 else 1.5)

            if colors is not None and len(colors) > 1:
                clr_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
                           for r, g, b in colors]
                marker = dict(size=marker_size, color=clr_str, opacity=0.85)
            elif colors is not None and len(colors) == 1:
                r, g, b = colors[0]
                marker = dict(size=marker_size, color=f'rgb({int(r*255)},{int(g*255)},{int(b*255)})', opacity=0.85)
            else:
                marker = dict(size=marker_size, color='#5b9bd5', opacity=0.85)

            hover = label or '点云'
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=marker,
                name=label or '',
                text=hover,
                hoverinfo='text',
                showlegend=show_legend,
                legendgroup=label,
            ))

        elif isinstance(geo, o3d.geometry.LineSet):
            pts = np.asarray(geo.points)
            lines = np.asarray(geo.lines) if geo.has_lines() else None
            all_points_list.append(pts)
            color = np.asarray(geo.colors)[0] if geo.has_colors() else [1, 0.3, 0.1]
            clr_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            if lines is not None and len(lines) > 0:
                x, y, z = [], [], []
                for a, b in lines:
                    x.extend([pts[a, 0], pts[b, 0], None])
                    y.extend([pts[a, 1], pts[b, 1], None])
                    z.extend([pts[a, 2], pts[b, 2], None])
                hover = label or '路径'
                traces.append(go.Scatter3d(
                    x=x, y=y, z=z, mode='lines',
                    line=dict(color=clr_str, width=2.5),
                    name=label or '',
                    hoverinfo='text', text=hover,
                    showlegend=show_legend, legendgroup=label))

        elif isinstance(geo, o3d.geometry.TriangleMesh):
            verts = np.asarray(geo.vertices)
            tris = np.asarray(geo.triangles)
            all_points_list.append(verts)
            color = np.asarray(geo.vertex_colors)[0] if geo.has_vertex_colors() else [1, 1, 1]
            n_show = min(len(tris), 1500)
            step = max(1, len(tris) // n_show)
            x, y, z = [], [], []
            for ti in range(0, len(tris), step):
                a, b, c = tris[ti]
                x.extend([verts[a, 0], verts[b, 0], verts[c, 0], None])
                y.extend([verts[a, 1], verts[b, 1], verts[c, 1], None])
                z.extend([verts[a, 2], verts[b, 2], verts[c, 2], None])
            c_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            hover = label or 'mesh'
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode='lines',
                line=dict(color=c_str, width=1),
                name=label or '',
                hoverinfo='text', text=hover,
                showlegend=show_legend, legendgroup=label))

        elif isinstance(geo, o3d.geometry.AxisAlignedBoundingBox):
            mn, mx = geo.min_bound, geo.max_bound
            all_points_list.append(np.array([mn, mx]))
            corners = np.array([
                [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]], [mn[0], mn[1], mx[2]],
                [mx[0], mx[1], mn[2]], [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
            ])
            edge_pairs = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
            color = np.asarray(geo.color) if hasattr(geo, 'color') and geo.color is not None else [1.0, 0.0, 0.0]
            clr_str = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            x, y, z = [], [], []
            for a, b in edge_pairs:
                x.extend([corners[a, 0], corners[b, 0], None])
                y.extend([corners[a, 1], corners[b, 1], None])
                z.extend([corners[a, 2], corners[b, 2], None])
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode='lines',
                line=dict(color=clr_str, width=3),
                name=label or '包围盒',
                hoverinfo='text', text=label or '包围盒',
                showlegend=show_legend, legendgroup=label))

    # 自动计算相机视角
    if all_points_list:
        all_pts = np.vstack(all_points_list)
        center = all_pts.mean(0)
        extent = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0)))
        eye_dist = extent * 1.8
        eye = dict(x=center[0] + eye_dist * 0.6,
                   y=center[1] + eye_dist * 0.8,
                   z=center[2] + eye_dist * 0.5)
    else:
        eye = dict(x=1.5, y=1.5, z=1.2)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#e0e0e0')),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='', showgrid=True, gridcolor='#333',
                       backgroundcolor='#1a1a2e', showticklabels=False),
            yaxis=dict(title='', showgrid=True, gridcolor='#333',
                       backgroundcolor='#1a1a2e', showticklabels=False),
            zaxis=dict(title='', showgrid=True, gridcolor='#333',
                       backgroundcolor='#1a1a2e', showticklabels=False),
            camera=dict(eye=eye),
            dragmode='orbit',
        ),
        paper_bgcolor='#111',
        plot_bgcolor='#111',
        font=dict(color='#ccc'),
        legend=dict(
            x=0.82, y=0.98,
            bgcolor='rgba(20,20,40,0.85)',
            bordercolor='#444',
            borderwidth=1,
            font=dict(size=12, color='#ddd'),
            itemsizing='constant',
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        width=width, height=height,
        modebar_add=['resetCameraDefault3d', 'toImage'],
    )

    if output_path:
        filepath = output_path
    else:
        import time as _time
        ts = _time.strftime('%m%d_%H%M%S')
        safe_title = "".join(c if c.isalnum() or c in '._-' else '_' for c in title)[:30]
        fname = f'viewer_{safe_title}_{ts}.html'
        filepath = _os.path.join(_os.path.expanduser('~'), 'PointNeXt', fname)
    fig.write_html(filepath)
    # 尝试浏览器打开, 失败不报错
    try:
        _sp.Popen(['xdg-open', filepath], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
    except Exception:
        pass
    return filepath


def visualize_decomposed(points, regions, combined_wp, combined_seg, title="喷涂管线v2", save_html=False):
    """分色显示不同区域 + 组合路径 + 部件名称标注"""
    _print_analysis_summary(points, regions, combined_wp, combined_seg, title)

    if _can_use_open3d():
        _visualize_open3d(points, regions, combined_wp, combined_seg, title, save_html=save_html)
    else:
        print("[可视化] Open3D 窗口不可用, 使用 matplotlib")
        _visualize_matplotlib(points, regions, combined_wp, combined_seg, title)


def _print_analysis_summary(points, regions, combined_wp, combined_seg, title):
    """控制台输出分析摘要"""
    type_counts = {}
    for ri, region in enumerate(regions):
        rtype = region['type']
        type_counts[rtype] = type_counts.get(rtype, 0) + 1

    parts_list = []
    for t in ['planar', 'curved', 'skeleton']:
        if type_counts.get(t):
            parts_list.append(f"{type_counts[t]}×{SURFACE_TYPES[t]['name']}")
    print(f"[{title} — 红框] {len(regions)}部件: {' + '.join(parts_list)}")
    for ri, region in enumerate(regions):
        rtype = region['type']
        pname = region.get('part_name', f'#{ri+1}')
        print(f"  [{pname}] {region.get('geometry', rtype)} "
              f"→ {SURFACE_TYPES[rtype]['path_algo']}")
    print(f"  总路径: {len(combined_wp)}点, {len(combined_seg)}段")
    print(f"  ⬤ 蓝=平面   ⬤ 粉=曲面   ⬤ 绿=细长结构(骨架)")


def _visualize_open3d(points, regions, combined_wp, combined_seg, title, save_html=False):
    """EGL 离屏渲染 PNG (默认) + 可选 Plotly HTML"""
    import time as _time
    ts = _time.strftime('%m%d_%H%M%S')
    safe_title = "".join(c if c.isalnum() or c in '._-' else '_' for c in title)[:40]
    out_dir = os.path.join(os.path.expanduser('~'), 'PointNeXt')

    geometries = []

    # 1. 按区域用不同颜色
    for ri, region in enumerate(regions):
        rtype = region['type']
        color = SURFACE_TYPES[rtype]['color']
        pts = region['points'].astype(np.float64)
        if len(pts) > 3000:
            pts = pts[np.random.choice(len(pts), 3000, replace=False)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    # 2. 喷涂路径
    if len(combined_wp) > 0:
        wp = combined_wp.astype(np.float64)
        if len(combined_seg) > 0:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(wp)
            ls.lines = o3d.utility.Vector2iVector(combined_seg)
            ls.paint_uniform_color([1.0, 0.3, 0.1])
            geometries.append(ls)
        # 起点/终点球
        sp0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        sp0.translate(wp[0])
        sp0.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(sp0)
        sp1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
        sp1.translate(wp[-1])
        sp1.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(sp1)

    # 3. 坐标轴
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    geometries.append(frame)

    # 4. 渲染 PNG
    print(f"\n[渲染] EGL 离屏渲染 → PNG ...")
    try:
        png_path = os.path.join(out_dir, f'vis_{safe_title}_{ts}.png')
        render_geometries_to_file(geometries, png_path, title=title, width=1600, height=1000)
        print(f"[PNG] {png_path}  ({os.path.getsize(png_path)/1024:.0f} KB)")
    except Exception as e:
        print(f"[PNG] 渲染失败: {e}")

    # 5. 可选: 轻量 HTML (大幅降采样)
    if save_html:
        try:
            html_geos, html_labels = [], []
            for ri, region in enumerate(regions):
                rtype = region['type']
                color = SURFACE_TYPES[rtype]['color']
                pname = region.get('part_name', region.get('name', f'部件{ri+1}'))
                pts = region['points'].astype(np.float64)
                if len(pts) > 600:
                    pts = pts[np.random.choice(len(pts), 600, replace=False)]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.paint_uniform_color(color)
                html_geos.append(pcd)
                html_labels.append(pname)
            if len(combined_wp) > 0:
                # 用原始坐标和线段, 只取前 60 段避免过大
                wp2 = combined_wp.astype(np.float64)
                n_seg_show = min(60, len(combined_seg))
                seg2 = combined_seg[:n_seg_show] if len(combined_seg) > 0 else np.zeros((0, 2), dtype=int)
                if len(seg2) > 0:
                    ls2 = o3d.geometry.LineSet()
                    ls2.points = o3d.utility.Vector3dVector(wp2)
                    ls2.lines = o3d.utility.Vector2iVector(seg2)
                    ls2.paint_uniform_color([1.0, 0.3, 0.1])
                    html_geos.append(ls2)
                    html_labels.append('喷漆路径')
            html_path = os.path.join(out_dir, f'vis_{safe_title}_{ts}.html')
            visualize_geometries_browser(html_geos, labels=html_labels, title=title, output_path=html_path)
            if os.path.exists(html_path):
                print(f"[HTML] {html_path}  ({os.path.getsize(html_path)/1024/1024:.1f} MB)")
            else:
                print(f"[HTML] 写入失败")
        except Exception as e:
            print(f"[HTML] 生成失败: {e}")


def _visualize_matplotlib(points, regions, combined_wp, combined_seg, title):
    """matplotlib 3D 可视化 (回退)"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for ri, region in enumerate(regions):
        rtype = region['type']
        color = SURFACE_TYPES[rtype]['color']
        pts = region['points']
        pname = region.get('part_name', f'#{ri+1}')
        n_sample = min(len(pts), 800)
        idx = np.random.choice(len(pts), n_sample, replace=False) if len(pts) > n_sample else np.arange(len(pts))
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                   c=[color], s=3, alpha=0.7, label=pname)

    if len(combined_wp) > 0:
        ax.plot(combined_wp[:, 0], combined_wp[:, 1], combined_wp[:, 2],
                'r-', linewidth=1.5, alpha=0.8, label='路径')
        ax.scatter(*combined_wp[0], c='red', s=80, marker='*', label='起点')
        ax.scatter(*combined_wp[-1], c='green', s=80, marker='o', label='终点')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


# ===================================================================
# 5. 主管线
# ===================================================================
def pipeline(points, name="unknown", category=None, visualize=True,
             use_partseg=False, optimize=False, save_html=False):
    """端到端: 点云 → 几何分析(G1-G4) → 部件命名 → 多策略路径 → 可视化"""
    print(f"\n[管线v2] 处理: {name} ({len(points)} 点)")

    # Step 1: 几何分析 + 部件命名
    if use_partseg and HAS_PARTSEG:
        print("[Step1] 模型推理 + 部件分割...")
        analysis = analyze_furniture_with_model(points, category=category)
    else:
        print("[Step1] 几何分析 + 部件命名...")
        analysis = analyze_furniture(points, category=category)
    regions = [p for p in analysis['parts']]

    geo_class = analysis['geometry_class']
    geo_info = analysis['geometry_class_info']
    print(f"  几何大类: {geo_class} — {geo_info['name']} ({geo_info['description']})")
    print(f"  难度: {geo_info['difficulty']}  策略: {geo_info['spray_strategy']}")

    # 还原 region 格式以兼容后续函数
    raw_regions = []
    for p in analysis['parts']:
        raw_regions.append({
            'points': p['points'],
            'type': p['type'],
            'normal': np.array(p['normal']) if isinstance(p['normal'], list) else p['normal'],
            'center': p['points'].mean(0),
            'coverage': p['coverage'],
            'part_name': p['name'],
            'part_id': p['part_id'],
            'geometry': p['geometry'],
        })

    # Step 2: 每区域生成路径
    print("[Step2] 生成区域路径...")
    regions_with_paths = []
    for ri, region in enumerate(raw_regions):
        if optimize and HAS_PATH_OPT and region['type'] == 'planar':
            wp, wn, segs = optimize_scan_direction(region)
        else:
            wp, wn, segs = generate_region_path(region)
        regions_with_paths.append((region, (wp, wn, segs)))
        print(f"  [{region.get('part_name', ri+1)}] "
              f"{SURFACE_TYPES[region['type']]['name']}: "
              f"{len(wp)} 路径点, {len(segs)} 段")

    # Step 3: 合并/优化路径
    if optimize and HAS_PATH_OPT:
        print("[Step3] TSP优化路径排序 + 过渡...")
        opt_result = optimize_path(regions_with_paths, points, safe_height=0.15)
        combined_wp = opt_result['waypoints']
        combined_wn = opt_result['normals']
        combined_seg = opt_result['segments']
        order_info = opt_result['stats']['order']
        print(f"  访问顺序: {order_info}")
        print(f"  喷涂距离: {opt_result['stats']['spray_dist']:.2f}  "
              f"过渡距离: {opt_result['stats']['travel_dist']:.2f}  "
              f"过渡段数: {len(opt_result['transitions'])}")
    else:
        print("[Step3] 合并区域路径...")
        combined_wp, combined_wn, combined_seg = merge_region_paths(regions_with_paths)

    # Step 4: 可视化
    if visualize:
        print("[Step4] 可视化...")
        visualize_decomposed(points, raw_regions, combined_wp, combined_seg,
                            title=f"{name} [{geo_info['name']}] — {len(raw_regions)}部件",
                            save_html=save_html)

    return {
        'name': name,
        'geometry_class': geo_class,
        'geometry_class_info': geo_info,
        'category': analysis['category'],
        'n_regions': len(raw_regions),
        'regions': raw_regions,
        'parts': analysis['parts'],
        'spray_summary': analysis['spray_summary'],
        'waypoints': combined_wp,
        'normals': combined_wn,
        'segments': combined_seg,
    }


# ===================================================================
# 兼容旧接口 — 用于 build_hotel_furniture_kb
# ===================================================================
def analyze_geometry(points, category=None):
    """兼容旧接口, 返回区域分解的统计摘要"""
    analysis = analyze_furniture(points, category=category)
    if analysis is None or not analysis['parts']:
        return None

    parts = analysis['parts']
    regions = [{
        'points': p['points'], 'type': p['type'],
        'normal': np.array(p['normal']), 'coverage': p['coverage'],
        'center': p['points'].mean(0),
        'part_name': p['name'], 'part_id': p['part_id'],
    } for p in parts]

    feats = {
        'geometry_class': analysis['geometry_class'],
        'n_regions': analysis['n_regions'],
        'n_planar': sum(1 for r in regions if r['type'] == 'planar'),
        'n_curved': sum(1 for r in regions if r['type'] == 'curved'),
        'n_skeleton': sum(1 for r in regions if r['type'] == 'skeleton'),
        'regions': regions,
        'parts': parts,
        'plane_inlier_ratio': sum(r['coverage'] for r in regions if r['type'] == 'planar'),
        'total_plane_coverage': sum(r['coverage'] for r in regions if r['type'] == 'planar'),
        'max_plane_coverage': max((r['coverage'] for r in regions if r['type'] == 'planar'), default=0),
        'n_plane_groups': sum(1 for r in regions if r['type'] == 'planar'),
        'n_large_planes': sum(1 for r in regions if r['type'] == 'planar' and r['coverage'] > 0.08),
        'n_small_planes': sum(1 for r in regions if r['type'] == 'planar' and r['coverage'] <= 0.08),
        'has_skeleton': any(r['type'] == 'skeleton' for r in regions),
        'n_clusters': sum(1 for r in regions if r['type'] in ('skeleton', 'curved')),
        'mean_curvature': 0.0,
        'normal_entropy': 0.0,
        'bbox_height': np.vstack([r['points'] for r in regions])[:, 2].max() - np.vstack([r['points'] for r in regions])[:, 2].min(),
    }
    return feats


def classify_spray_type(feats, category=None):
    """
    兼容旧接口, 返回几何大类摘要.
    """
    if feats is None:
        if category:
            geo_class = 'G2_multi_plane'
            info = GEOMETRY_CLASS[geo_class]
            return 0, {'name': f'类别:{category} ({info["name"]})'}, 0.70, []
        return 0, None, 0, []

    geo_class = feats.get('geometry_class', 'G2_multi_plane')
    info = GEOMETRY_CLASS.get(geo_class, GEOMETRY_CLASS['G2_multi_plane'])

    n_p = feats.get('n_planar', 0)
    n_c = feats.get('n_curved', 0)
    n_s = feats.get('n_skeleton', 0)

    strategies = []
    if n_p > 0:
        strategies.append(f'{n_p}个平面(2D栅格)')
    if n_c > 0:
        strategies.append(f'{n_c}个曲面(等高线跟随)')
    if n_s > 0:
        strategies.append(f'{n_s}个骨架(轴线环绕)')

    summary = ' + '.join(strategies) if strategies else '未知'
    spray_info = {
        'name': f'{info["name"]}: {summary}',
        'geometry_class': geo_class,
        'path_algo': info['spray_strategy'],
        'gun_pose': '区域自适应',
        'speed': '区域变速',
        'agv_stations': '1站' if n_p + n_c + n_s <= 3 else '1-2站',
        'difficulty': info['difficulty'],
    }
    # 返回旧 type_id (1-4 对应 G1-G4)
    type_map = {'G1_large_plane': 1, 'G2_multi_plane': 2, 'G3_curved': 3, 'G4_mixed_skeleton': 4}
    old_type = type_map.get(geo_class, 2)

    return old_type, spray_info, 0.75, []


# ===================================================================
# 测试入口
# ===================================================================
def run_synthetic_test(optimize=False, visualize=True, save_html=False):
    """合成测试"""
    opt_tag = " + TSP路径优化" if optimize else ""
    print("=" * 70)
    print(f"  合成家具测试 — 区域分解 + 多策略路径 (v2){opt_tag}")
    print("=" * 70)

    # 桌子: 大平面 + 4条腿
    top = np.random.rand(600, 3) * [1.5, 0.8, 0.03] - [0.75, 0.4, 0]
    legs = []
    for dx, dy in [(-0.6, -0.3), (0.6, -0.3), (-0.6, 0.3), (0.6, 0.3)]:
        leg = np.random.rand(106, 3) * [0.06, 0.06, 0.8] + [dx, dy, -0.8]
        legs.append(leg)
    pts = np.vstack([top] + legs)
    pipeline(pts, "合成-桌子", category="table", optimize=optimize, visualize=visualize, save_html=save_html)

    # 椅子: 座面 + 靠背 + 腿
    seat = np.random.rand(300, 3) * [0.5, 0.5, 0.04] - [0.25, 0.25, 0]
    back = np.random.rand(200, 3) * [0.5, 0.04, 0.4] + [-0.25, -0.27, 0.1]
    legs2 = []
    for dx, dy in [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)]:
        l = np.random.rand(81, 3) * [0.04, 0.04, 0.45] + [dx, dy, -0.45]
        legs2.append(l)
    pts2 = np.vstack([seat, back] + legs2)
    pipeline(pts2, "合成-椅子", category="chair", optimize=optimize, visualize=visualize, save_html=save_html)

    # 曲面沙发
    phi = np.random.rand(1024) * np.pi * 0.5
    theta = np.random.rand(1024) * 2 * np.pi
    r = 0.5 + np.random.rand(1024) * 0.2
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi) * 0.5
    pts3 = np.column_stack([x, y, z])
    pipeline(pts3, "合成-曲面沙发", category="sofa", optimize=optimize, visualize=visualize, save_html=save_html)

    print("\n" + "=" * 70)
    print("  合成测试完成。PNG 已保存到 ~/PointNeXt/vis_*.png")
    print("=" * 70)


def run_modelnet40_test(use_partseg=False, optimize=False, visualize=True, save_html=False):
    """ModelNet40 真实数据测试"""
    import h5py, glob

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'modelnet40_ply_hdf5_2048')
    test_files = sorted(glob.glob(os.path.join(data_dir, 'ply_data_test*.h5')))

    if not test_files:
        print("✗ ModelNet40 数据未找到")
        return

    names_file = os.path.join(data_dir, 'shape_names.txt')
    with open(names_file) as f:
        categories = [l.strip() for l in f.readlines()]

    target_cats = ['door', 'desk', 'table', 'chair', 'bookshelf',
                   'dresser', 'night_stand', 'tv_stand', 'wardrobe', 'bed', 'sofa']

    mode = "模型推理" if use_partseg else "几何启发式"
    print(f"\n{'='*60}")
    print(f"  ModelNet40 测试 — {mode}")
    print(f"{'='*60}")

    for fpath in test_files:
        with h5py.File(fpath, 'r') as hf:
            data = hf['data'][:]
            labels = hf['label'][:]

        for cat_name in target_cats:
            if cat_name not in categories:
                continue
            cat_idx = categories.index(cat_name)
            cat_mask = (labels[:, 0] == cat_idx)
            cat_data = data[cat_mask]
            if len(cat_data) == 0:
                continue

            pts = cat_data[0].astype(np.float32)
            pts -= pts.mean(0)
            s = np.abs(pts).max()
            if s > 0:
                pts /= s

            result = pipeline(pts, f"ModelNet40/{cat_name}", category=cat_name,
                            visualize=False, use_partseg=use_partseg, optimize=optimize, save_html=save_html)
            if result:
                tag = "[模型]" if result.get('_model_based') else "[几何]"
                print(f"  {tag} {cat_name:15s} → [{result['geometry_class']}] "
                      f"{result['n_regions']}部件 "
                      f"{result['spray_summary']}")

    # 选一个做可视化
    if visualize:
        print("\n选择一个样本可视化...")
        fpath = test_files[0]
        with h5py.File(fpath, 'r') as hf:
            data = hf['data'][:]
            labels = hf['label'][:]
        for cat_name in ['table', 'chair']:
            if cat_name not in categories:
                continue
            cat_idx = categories.index(cat_name)
            mask = labels[:, 0] == cat_idx
            if mask.any():
                pts = data[mask][0].astype(np.float32)
                pts -= pts.mean(0)
                pts /= np.abs(pts).max()
                pipeline(pts, f"ModelNet40/{cat_name}", category=cat_name,
                        visualize=True, use_partseg=use_partseg, optimize=optimize, save_html=save_html)
                break
    else:
        print("\n[跳过可视化]")


# ===================================================================
# CLI
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='喷涂管线 v2 — 表面分解+多策略')
    parser.add_argument('--synth', action='store_true', help='合成测试')
    parser.add_argument('--modelnet40', action='store_true', help='ModelNet40测试')
    parser.add_argument('--file', type=str, help='输入点云文件 (.ply/.npy)')
    parser.add_argument('--no-viz', action='store_true', help='不显示可视化')
    parser.add_argument('--partseg', action='store_true',
                        help='使用 PointNeXt 模型做部件分割 (替代几何启发式)')
    parser.add_argument('--partseg-ckpt', type=str, default=None,
                        help='模型 checkpoint 路径 (默认自动查找)')
    parser.add_argument('--optimize', action='store_true',
                        help='启用 TSP 路径排序 + 扫描方向优化')
    parser.add_argument('--html', action='store_true',
                        help='同时生成交互式 HTML (默认仅 PNG)')
    args = parser.parse_args()

    # 若启用模型模式, 预加载模型
    if args.partseg:
        print("[初始化] 加载 PointNeXt 部件分割模型...")
        try:
            get_partseg_model(args.partseg_ckpt)
            print("[初始化] 模型加载成功\n")
        except Exception as e:
            print(f"[警告] 模型加载失败: {e}")
            print("[回退] 使用几何启发式方法\n")
            args.partseg = False

    if args.synth:
        run_synthetic_test(optimize=args.optimize, visualize=not args.no_viz, save_html=args.html)
    elif args.modelnet40:
        run_modelnet40_test(use_partseg=args.partseg, optimize=args.optimize,
                           visualize=not args.no_viz, save_html=args.html)
    elif args.file:
        ext = os.path.splitext(args.file)[1].lower()
        if ext == '.ply':
            pcd = o3d.io.read_point_cloud(args.file)
            pts = np.asarray(pcd.points, dtype=np.float32)
        elif ext == '.npy':
            pts = np.load(args.file).astype(np.float32)
        else:
            print(f"不支持: {ext}")
            sys.exit(1)
        pipeline(pts, os.path.basename(args.file), visualize=not args.no_viz,
                use_partseg=args.partseg, optimize=args.optimize, save_html=args.html)
    else:
        print("用法:")
        print("  python spray_pipeline.py --synth                  # 合成测试")
        print("  python spray_pipeline.py --modelnet40              # ModelNet40测试(几何)")
        print("  python spray_pipeline.py --modelnet40 --partseg    # ModelNet40测试(模型)")
        print("  python spray_pipeline.py --file x.ply --partseg    # 单个文件(模型)")
        print("  python spray_pipeline.py --synth --optimize        # 合成测试+路径优化")
        print("\n模型驱动特性: PointNeXt 部件分割 → 几何原型映射 → 11类家具泛化")
