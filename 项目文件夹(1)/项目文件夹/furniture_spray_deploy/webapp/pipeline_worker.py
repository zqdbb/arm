#!/usr/bin/env python3
"""喷漆管线桥接层 — Flask 与 spray_pipeline 之间的薄封装."""

import os, sys, base64
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spray_pipeline import (
    decompose_furniture, classify_geometry_class,
    name_parts, SURFACE_TYPES, PART_COLORS, GEOMETRY_CLASS,
    _compute_gravity_axis, _is_horizontal, _is_vertical
)

# 点云格式 — 直接读点
_POINT_CLOUD_EXTS = {'.ply', '.npy', '.xyz', '.pcd', '.pts'}
# Mesh 格式 — 读 mesh 后采样顶点
_MESH_EXTS = {'.obj', '.stl', '.off', '.glb', '.gltf'}


def _load_point_cloud(filepath):
    """加载点云格式 (.ply/.npy/.xyz/.pcd/.pts). 复用 auto_run.load_points."""
    from auto_run import load_points
    return load_points(filepath)


def _parse_obj_fallback(filepath):
    """手动解析 OBJ 文件提取顶点 + 面索引. 处理 Open3D 不支持的 OBJ 变体."""
    verts = []
    faces = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == 'f':
                # 处理 v, v/vt, v/vt/vn 格式 — 取第一个数字
                idxs = []
                for p in parts[1:]:
                    vidx = int(p.split('/')[0])
                    idxs.append(vidx - 1 if vidx > 0 else len(verts) + vidx)
                if len(idxs) >= 3:
                    faces.append(idxs[:3])  # 只取前三个 (三角化)
                    if len(idxs) >= 4:
                        faces.append([idxs[0], idxs[2], idxs[3]])  # 四边形→两个三角
    return np.array(verts, dtype=np.float32), faces


def _file_hash_seed(filepath):
    """从文件内容生成确定性种子 (同文件→同结果)."""
    import hashlib
    with open(filepath, 'rb') as f:
        h = hashlib.md5(f.read(65536)).hexdigest()  # 读前 64KB 足够区分
    return int(h[:8], 16) % (2**31)


def _sample_faces(verts, faces, n_target, seed=42):
    """从三角面片均匀采样点 (固定种子保证可复现)."""
    rng = np.random.RandomState(seed)
    if not faces:
        if len(verts) <= n_target:
            return verts
        idx = rng.choice(len(verts), n_target, replace=False)
        return verts[idx]

    tri_verts = np.array([[verts[f[0]], verts[f[1]], verts[f[2]]] for f in faces], dtype=np.float32)
    n_tris = len(tri_verts)
    a = tri_verts[:, 1] - tri_verts[:, 0]
    b = tri_verts[:, 2] - tri_verts[:, 0]
    areas = np.linalg.norm(np.cross(a, b), axis=1) * 0.5
    areas = np.maximum(areas, 1e-12)
    probs = areas / areas.sum()

    idx = rng.choice(n_tris, n_target, p=probs)
    r1 = rng.rand(n_target, 1)
    r2 = rng.rand(n_target, 1)
    mask = (r1 + r2) > 1
    r1[mask] = 1 - r1[mask]
    r2[mask] = 1 - r2[mask]

    sampled = (tri_verts[idx, 0] +
               r1 * (tri_verts[idx, 1] - tri_verts[idx, 0]) +
               r2 * (tri_verts[idx, 2] - tri_verts[idx, 0]))
    return sampled.astype(np.float32)


def _load_mesh_as_points(filepath, target_points=10000):
    """加载 mesh 格式 → 点云.
    优先用 trimesh 做面片采样 (与桌面 obj2npy.py 一致),
    回退: Open3D → 手动 OBJ 解析 + 三角面采样.
    使用文件内容哈希作为随机种子, 同文件多次上传结果一致.
    """
    seed = _file_hash_seed(filepath)

    # 尝试 trimesh (最可靠的面片采样)
    try:
        import trimesh as _trimesh
        mesh = _trimesh.load(filepath, force='mesh')
        if isinstance(mesh, _trimesh.Scene):
            mesh = _trimesh.util.concatenate(
                [g for g in mesh.geometry.values() if hasattr(g, 'vertices')])

        if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
            v = mesh.vertices.astype(np.float32)
            # 过滤离群顶点 (同 obj2npy.py)
            c = v.mean(0)
            dist = np.sqrt(((v - c) ** 2).sum(1))
            keep = dist < np.median(dist) * 5
            if keep.sum() >= 100:
                v = v[keep]

            # 面片均匀采样
            n_sample = max(target_points, 2048)
            rng = np.random.RandomState(seed)
            pts, _ = _trimesh.sample.sample_surface(mesh, n_sample, seed=seed)
            pts = pts.astype(np.float32)
            return pts
    except Exception as e:
        print(f"[trimesh 回退] {e}")

    # 回退: Open3D / 手动 OBJ 解析
    verts = np.empty((0, 3), dtype=np.float32)
    faces = []

    try:
        mesh = o3d.io.read_triangle_mesh(filepath)
        if len(mesh.vertices) > 0:
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            if len(mesh.triangles) > 0:
                faces = np.asarray(mesh.triangles).tolist()
    except Exception:
        pass

    if len(verts) == 0:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.obj':
            try:
                verts, faces = _parse_obj_fallback(filepath)
            except Exception as e:
                raise ValueError(f"OBJ 解析失败: {e}")
        elif ext == '.stl':
            try:
                verts = _parse_stl_binary(filepath)
            except Exception:
                pass

    if len(verts) == 0:
        raise ValueError(f"无法从文件中提取顶点数据, 请检查文件格式是否正确")

    n_sample = max(target_points, 2048)
    if len(verts) >= n_sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(verts), n_sample, replace=False)
        pts = verts[idx]
    else:
        sampled = _sample_faces(verts, faces, n_sample - len(verts), seed=seed)
        pts = np.vstack([verts, sampled]).astype(np.float32) if len(sampled) > 0 else verts

    return pts


def _parse_stl_binary(filepath):
    """解析二进制 STL 文件提取顶点."""
    verts = []
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        n_tris = int.from_bytes(f.read(4), 'little')
        for _ in range(n_tris):
            f.read(12)  # normal
            for _ in range(3):
                v = [float(int.from_bytes(f.read(4), 'little', signed=True)) for _ in range(3)]
                verts.append(v)
            f.read(2)  # attribute
    return np.array(verts, dtype=np.float32) if verts else np.empty((0, 3), dtype=np.float32)


def load_and_normalize(filepath):
    """解析上传文件并归一化, 返回 (points, n_original).
    支持: .ply .npy .xyz .pcd .pts (点云) + .obj .stl .off .glb .gltf (mesh→点云).
    大点云自动降采样到 ~15K 点以保证处理速度.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in _POINT_CLOUD_EXTS:
        points = _load_point_cloud(filepath)
    elif ext in _MESH_EXTS:
        points = _load_mesh_as_points(filepath)
    else:
        raise ValueError(f"不支持格式: {ext}")

    if len(points) == 0:
        raise ValueError("文件无点云数据")

    # 归一化 (同 auto_run.load_points)
    center = points.mean(0)
    points = points - center
    scale = np.abs(points).max()
    if scale > 1e-10:
        points = points / scale

    # 大点云降采样 — RANSAC+DBSCAN 对 >20K 点会很慢
    MAX_POINTS = 15000
    n_original = len(points)
    if n_original > MAX_POINTS:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        voxel_size = 0.015  # 归一化坐标下, 约 1.5% 的包围盒边长
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        points = np.asarray(pcd.points, dtype=np.float32)
        # 如果降采样后还是太多, 随机抽样
        if len(points) > MAX_POINTS:
            idx = np.random.choice(len(points), MAX_POINTS, replace=False)
            points = points[idx]
        print(f"[降采样] {n_original} → {len(points)} 点 (voxel={voxel_size})")

    return points, n_original


def run_classify_with_regions(points, retry_category=None):
    """执行 decompose + 分类, 返回 (category, confidence, features, regions, geo_class).
    这是瓶颈操作 (RANSAC+DBSCAN, 2-5s), 结果中的 regions 将被缓存供 run_segment 复用.

    retry_category: 上次分类结果 (如 'chair'), 本次重分类会排除它, 换逻辑再判.
    """
    regions = decompose_furniture(points)

    # 复制 classify_furniture_type 的分类逻辑，但复用已计算的 regions
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
    bbox_h = max(bbox)  # 最长边
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
        category, confidence = 'cabinet', 0.82
    # 垂直面覆盖高 + 多大垂直面 + 多水平面 + 骨架少 → 柜体
    elif vertical_cov > 0.25 and n_large_vert >= 2 and n_horizontal >= 2 and skel_cov < 0.10:
        category, confidence = 'cabinet', 0.75
    # 即使有少量骨架, 垂直大面多 + 水平面多 → 仍是柜体
    elif n_large_vert >= 2 and n_horizontal >= 2 and skel_cov < 0.08:
        category, confidence = 'cabinet', 0.68
    # 4+个平面都是大面 + 无骨架 → 典型柜体
    elif n_planar >= 4 and n_large_vert >= 2 and n_large_horiz >= 1 and n_skeleton == 0:
        category, confidence = 'cabinet', 0.78
    else:
        # 非柜体 → 进入椅子/桌子区分
        category, confidence = None, 0.0

    # ============================================================
    # 空间分析: 椅子靠背在水平面上方, 桌子没有
    # 同时计算主水平面在重力方向的位置 (桌面在顶部, 座面在中间)
    # ============================================================
    if category is None:
        horizontal_regions = [r for r in planar_regions if _is_horizontal(r['normal'], gravity)]
        has_backrest = False
        main_horiz_rel_h = 0.5
        g_min = float(np.dot(all_pts, gravity).min())
        if horizontal_regions:
            main_horiz = max(horizontal_regions, key=lambda r: r['coverage'])
            h_center = float(np.dot(main_horiz['center'], gravity))
            g_span = float(np.dot(all_pts, gravity).ptp())
            main_horiz_rel_h = (h_center - g_min) / (g_span + 1e-10)
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
        table_like = (main_horiz_rel_h > 0.65 and not has_backrest and n_horizontal >= 1)
        chair_like = (has_backrest or (0.25 < main_horiz_rel_h <= 0.65 and n_horizontal >= 1))

        if n_skeleton > 0:
            if table_like and n_horizontal >= 1 and n_vertical <= 2:
                category, confidence = 'table', 0.82
            elif chair_like and n_horizontal >= 1:
                category, confidence = 'chair', 0.82
            elif n_skeleton >= 3 and n_horizontal >= 1:
                if has_backrest:
                    category, confidence = 'chair', 0.85
                elif bbox_wh_ratio > 1.3 and main_horiz_rel_h < 0.65:
                    category, confidence = 'chair', 0.60
                else:
                    category, confidence = 'table', 0.75
            elif n_horizontal >= 1 and n_vertical >= 1:
                if has_backrest:
                    category, confidence = 'chair', 0.80
                elif n_skeleton >= 2:
                    category, confidence = 'table', 0.70
                else:
                    category, confidence = 'chair', 0.55
            elif n_horizontal >= 1:
                if has_backrest:
                    category, confidence = 'chair', 0.65
                elif n_skeleton >= 2:
                    category, confidence = 'table', 0.70
                else:
                    category, confidence = 'table', 0.75
            else:
                category, confidence = 'chair', 0.40
        else:
            # 无骨架腿 — 腿被RANSAC拆成了小平面
            n_tiny_vert = sum(1 for r in planar_regions
                              if r['coverage'] < 0.05 and _is_vertical(r['normal'], gravity))
            if n_horizontal >= 1 and n_vertical >= 1 and n_small_planar >= 3:
                category, confidence = 'chair', 0.65
            elif n_horizontal >= 1 and n_small_planar >= 4 and n_planar >= 5:
                category, confidence = 'table', 0.55
            elif bbox_wh_ratio > 10.0 and n_horizontal >= 2 and n_planar <= 4:
                all_grav = np.dot(all_pts, gravity)
                g_height = float(all_grav.ptp())
                if g_height > min(bbox) * 3:
                    category, confidence = 'chair', 0.40
                else:
                    category, confidence = 'table', 0.40
            elif n_planar >= 2:
                if n_horizontal >= 1 and n_tiny_vert >= 2 and n_planar <= 6:
                    category, confidence = 'chair', 0.45
                elif bbox_wh_ratio > 3.0 and n_horizontal >= 1 and n_vertical >= 1:
                    category, confidence = 'chair', 0.40
                elif n_horizontal >= 2 and n_vertical == 0 and n_planar <= 4:
                    category, confidence = 'table', 0.40
                else:
                    category, confidence = 'cabinet', 0.70
            elif max_planar_cov > 0.8 and n_planar <= 2:
                category, confidence = 'table', 0.50
            elif max_planar_cov > 0.6:
                category, confidence = 'cabinet', 0.50
            else:
                category, confidence = 'cabinet', 0.35

    # retry 模式: 排除上次分类, 重新判断
    is_retry = retry_category is not None

    # retry 模式: 如果结果与上次相同, 强制切换到次优类别
    if is_retry and category == retry_category:
        alt_categories = ['chair', 'table', 'cabinet']
        alt_categories.remove(retry_category)
        # 根据特征选最合理的替代类别
        # 柜体特征 → 优先试 cabinet
        if n_large_vert >= 2 and n_horizontal >= 2 and 'cabinet' in alt_categories:
            category = 'cabinet'
        elif vertical_cov > 0.20 and n_horizontal >= 2 and 'cabinet' in alt_categories:
            category = 'cabinet'
        elif n_skeleton >= 3:
            # 有很多骨架 → 非柜子 → 在 chair/table 间切换
            category = alt_categories[0] if alt_categories[0] != 'cabinet' else alt_categories[1]
        elif n_horizontal >= 1 and n_vertical >= 1:
            category = alt_categories[0]
        elif max_planar_cov > 0.5 and n_planar <= 3:
            category = 'table' if 'table' in alt_categories else alt_categories[0]
        else:
            category = alt_categories[0]
        confidence = round(confidence * 0.7, 2)

    geo_class = classify_geometry_class(regions)
    geo_info = GEOMETRY_CLASS.get(geo_class, GEOMETRY_CLASS['G2_multi_plane'])

    # 将 regions 转为可 JSON 序列化的格式存入缓存
    regions_cache = []
    for r in regions:
        regions_cache.append({
            'points': r['points'].astype(np.float32),
            'type': r['type'],
            'normal': r['normal'].astype(np.float32).tolist(),
            'center': r['center'].astype(np.float32).tolist(),
            'coverage': float(r['coverage']),
        })

    return {
        'category': category,
        'confidence': round(confidence, 2),
        'features': features,
        'geometry_class': geo_class,
        'geometry_class_info': {
            'name': geo_info['name'],
            'description': geo_info['description'],
            'spray_strategy': geo_info['spray_strategy'],
            'difficulty': geo_info['difficulty'],
        },
        '_regions_cache': regions_cache,
        '_retry': is_retry,
    }


def _color_for_role(name):
    """按部件角色分配固定颜色 — 同一角色在不同重力下颜色一致, 切换时视觉可对比."""
    if '座面' in name or '桌面' in name or '桌板' in name:
        return [0.21, 0.49, 0.94]     # 蓝色 — 水平主面
    if '靠背' in name:
        return [0.94, 0.35, 0.16]     # 橙色 — 靠背
    if '腿' in name:
        return [0.22, 0.78, 0.35]     # 绿色 — 腿/杆件
    if '扶手' in name:
        return [0.88, 0.18, 0.55]     # 粉色 — 扶手
    if '横撑' in name:
        return [0.53, 0.27, 0.73]     # 紫色
    if '侧板' in name:
        return [0.00, 0.68, 0.73]     # 青色
    if '顶板' in name:
        return [0.95, 0.45, 0.30]     # 橙红
    if '底板' in name:
        return [0.50, 0.60, 0.30]     # 橄榄
    if '搁板' in name or '面板' in name:
        return [0.45, 0.75, 0.95]     # 浅蓝
    if '床板' in name or '床架' in name or '床侧板' in name:
        return [0.80, 0.55, 0.30]     # 棕色
    if '门板' in name or '门框' in name or '门饰' in name:
        return [0.70, 0.35, 0.50]     # 玫红
    if '曲面' in name or '沙发' in name:
        return [1.0, 0.4, 0.7]        # 粉红-曲面
    if '辅面' in name or '辅件' in name or '小平面' in name:
        return [0.60, 0.60, 0.60]     # 灰色-辅件
    return None  # 让调用方用 fallback


def _build_parts_from_regions(regions, gravity, category):
    """用指定重力轴给 regions 命名, 返回 parts 列表."""
    import spray_pipeline as sp
    original_fn = sp._compute_gravity_axis
    sp._compute_gravity_axis = lambda r: np.array(gravity, dtype=np.float64)
    try:
        sp.name_parts(category, regions)
    finally:
        sp._compute_gravity_axis = original_fn

    parts = []
    fallback_idx = 0
    for i, r in enumerate(regions):
        pts = r['points']
        n = len(pts)
        name = r.get('part_name', f'{SURFACE_TYPES[r["type"]]["name"]}_{i+1}')
        color = _color_for_role(name)
        if color is None:
            color = PART_COLORS[fallback_idx % len(PART_COLORS)]
            fallback_idx += 1
        name = r.get('part_name', f'{SURFACE_TYPES[r["type"]]["name"]}_{i+1}')
        dot_g = float(np.dot(r['normal'], gravity))
        if r['type'] == 'skeleton':
            orientation = 'skeleton'
        elif abs(dot_g) > 0.7:
            orientation = 'horizontal'
        elif abs(dot_g) < 0.35:
            orientation = 'vertical'
        else:
            orientation = f'oblique(dot={dot_g:.2f})'
        parts.append({
            'name': name,
            'type': r['type'],
            'type_name': SURFACE_TYPES[r['type']]['name'],
            'spray': SURFACE_TYPES[r['type']]['path_algo'],
            'coverage': round(float(r['coverage']), 3),
            'n_points': n,
            'color': color,
            'normal': r['normal'].tolist(),
            'points_b64': base64.b64encode(pts.astype(np.float32).tobytes()).decode('ascii'),
            '_debug': {
                'type': r['type'],
                'normal': [round(float(x), 3) for x in r['normal']],
                'dot_gravity': round(dot_g, 3),
                'orientation': orientation,
                'coverage': round(float(r['coverage']), 3),
                'n_points': n,
            },
        })
    return parts


def _regions_to_parts(regions):
    """将 regions 列表转为 webapp parts 格式."""
    type_names = {'planar': '平面区域', 'curved': '曲面区域', 'skeleton': '细长结构'}
    spray_map = {'planar': '2D栅格扫描', 'curved': '等高线轮廓跟随', 'skeleton': '轴线环绕'}

    parts = []
    for r in regions:
        pts = r['points']
        n = len(pts)
        parts.append({
            'name': r['part_name'],
            'type': r['type'],
            'type_name': type_names.get(r['type'], r['type']),
            'spray': spray_map.get(r['type'], r['type']),
            'coverage': round(float(r['coverage']), 3),
            'n_points': n,
            'color': [float(c) for c in r['color']],
            'normal': r['normal'].tolist() if hasattr(r['normal'], 'tolist') else r['normal'],
            'points_b64': base64.b64encode(
                pts.astype(np.float32).tobytes()
            ).decode('ascii'),
            '_debug': {
                'type': r['type'],
                'normal': [round(float(x), 3) for x in (
                    r['normal'] if hasattr(r['normal'], '__iter__') else [0, 0, 0])],
                'dot_gravity': 0,
                'orientation': 'ai_labeled',
                'coverage': round(float(r['coverage']), 3),
                'n_points': n,
            },
        })
    return parts


def _wrap_parts_response(parts, total_points, model_name):
    """包装 parts 列表为前端兼容的响应格式."""
    candidate = {
        'axis': 'ai',
        'gravity': [0, 0, 0],
        'summary': f"AI分割 {len(parts)} 类部件",
        'detail': model_name,
        'n_parts': len(parts),
        'parts': parts,
    }
    return {
        'candidates': [candidate],
        'default_axis': 'ai',
        'parts': parts,
        'n_regions': len(parts),
        'total_points': int(total_points),
        'model_based': True,
    }


def _correct_drawer_vs_door(regions):
    """修正抽屉面/门板混淆: 竖长面板应标记为门板而非抽屉面.

    PartNet 定义: 抽屉面=横向短面板, 门板=竖向大面板.
    模型有时把大面板误判为抽屉面(14), 用 PCA 纵横比修正为门板(23).
    """
    DOOR_COLOR = [1.0, 0.5, 0.2]   # orange, matches "door" semantic group
    DRAWER_COLOR = [0.4, 0.8, 0.3]  # green, matches "drawer" semantic group

    for r in regions:
        if r.get('label') != 14:  # only fix drawer_front mispredictions
            continue

        pts = r['points']
        coverage = r['coverage']
        if len(pts) < 10:
            continue

        # 抽屉面通常占柜子的 2%-8%, 超过 10% 很可能是门板
        # 再用 PCA 确认: 门板是竖向大面板, 抽屉面是横向短面板
        if coverage < 0.08:
            continue

        # PCA: get principal axes extent ratio
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        extent_primary = np.sqrt(eigvals[2])
        extent_secondary = np.sqrt(eigvals[1])
        aspect = extent_primary / (extent_secondary + 1e-10)

        # 覆盖率 > 8% 且 elongated → 大面板, 应为门板而非抽屉面
        if aspect > 1.3:
            r['label'] = 23
            r['part_name'] = '门板'
            r['part_id'] = 'cabinet_door_surface'
            r['semantic_group'] = 'door'
            r['color'] = DOOR_COLOR


def generate_path_for_part(region, spacing=0.06, spray_distance=0.15):
    """为单个部件区域生成喷涂路径 (gen_table 风格干净 zigzag + 腿线).

    输入: region dict (含 points, type, normal, part_id)
          spacing — 路径间距
          spray_distance — 喷枪距离表面
    输出: (waypoints, normals, segments) 或 None
    """
    from clean_path import generate_clean_path
    return generate_clean_path(region, spacing, spray_distance)


def generate_cabinet_path(points, spacing=0.06, spray_distance=0.15):
    """为整件柜子生成喷涂路径 (顶面+前面+后面+左面+右面, 不按部件拆).

    输入: points — 归一化后的柜子整件点云 (N,3)
    输出: (waypoints, normals, segments) 或 None
    """
    from clean_path import generate_cabinet_path as _gcp
    return _gcp(points, spacing, spray_distance)


def _densify_sparse_cloud(points, target=8000):
	    """对稀疏点云做 BPA 网格重建 + 均匀采样, 提升 AI 模型分割质量.
	    返回 densified points 或 None (点数足够不需要处理).
	    """
	    MIN_DENSE = 5000
	    if len(points) >= MIN_DENSE:
	        return None

	    try:
	        pcd = o3d.geometry.PointCloud()
	        pcd.points = o3d.utility.Vector3dVector(points)
	        pcd.estimate_normals(
	            o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

	        radii = [0.001, 0.002, 0.004, 0.008, 0.016]
	        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
	            pcd, o3d.utility.DoubleVector(radii))

	        if len(mesh.vertices) > 0:
	            n_sample = max(target, len(points) * 2)
	            pcd_dense = mesh.sample_points_uniformly(
	                number_of_points=n_sample)
	            pts_dense = np.asarray(pcd_dense.points, dtype=np.float32)
	            if len(pts_dense) >= len(points) * 1.5:
	                return pts_dense
	    except Exception:
	        pass

	    # BPA 失败时的 fallback: 对原始点做微小抖动复制
	    n_target = max(target, len(points) * 2)
	    n_dup = n_target - len(points)
	    idx = np.random.choice(len(points), n_dup, replace=True)
	    rng = np.random.RandomState(42)
	    jitter = rng.randn(n_dup, 3).astype(np.float32) * 0.0003
	    return np.vstack([points, points[idx] + jitter]).astype(np.float32)


def run_segment_pointnext(points):
    """PointNeXt 分割 — cabinet (24 类精细标签)."""
    from pointnext_inference import get_pointnext_model
    model = get_pointnext_model()
    _labels, _names, regions = model.predict(points)
    _correct_drawer_vs_door(regions)
    parts = _regions_to_parts(regions)
    resp = _wrap_parts_response(parts, len(points), 'PointNeXt (24类)')
    resp['_regions'] = regions
    return resp


# ── 柜子几何分解 (对齐 gen_guizi_path, 尺寸自适应) ──

CABINET_FACE_COLORS = {
    '后面': [0.30, 0.30, 0.90],
    '左面上': [0.30, 0.90, 0.30],
    '左面下': [0.25, 0.70, 0.25],
    '右面上': [0.90, 0.50, 0.30],
    '右面下': [0.80, 0.40, 0.20],
    '顶面': [0.50, 0.90, 0.90],
    '前面上': [0.90, 0.30, 0.50],
    '前面下': [0.90, 0.60, 0.60],
}


def _face_band(pts, ax, side, ratio=0.06):
    """取某轴外侧 ratio 比例的点带 (面板厚度), 尺寸自适应.

    用分位数求面厚带, 避免离群点/归一化尺度影响, 同结构不同尺寸比例一致.
    """
    v = pts[:, ax]
    lo, hi = np.percentile(v, [1.0, 99.0])
    span = hi - lo
    if span <= 1e-9:
        span = (v.max() - v.min()) + 1e-9
    if side == 'min':
        return v <= lo + ratio * span
    return v >= hi - ratio * span


def decompose_cabinet(points, min_points=10):
    """按 gen_guizi_path 的几何规则把柜子点云拆成 8 个外表面部件.

    柜子 Y-up (竖直=Y, 宽=X, 深=Z), 与 load_and_normalize 约定一致.
    上/下分隔由门板(前面 max-Z)顶部 Y 推导 —— 尺寸自适应, 不写死 Y=0.

    产出 8 个面板部件:
      后面 / 左面上 / 左面下 / 右面上 / 右面下 / 顶面 / 前面上 / 前面下
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 20:
        return []

    back = _face_band(pts, 2, 'min')    # 后面 (Z min)
    top = _face_band(pts, 1, 'max')     # 顶面 (Y max)
    left = _face_band(pts, 0, 'min')    # 左面 (X min)
    right = _face_band(pts, 0, 'max')   # 右面 (X max)
    front = _face_band(pts, 2, 'max')   # 前面 (门板, Z max)

    # 上/下分隔: 门板顶部 Y (无门板则取中点), 加 1% 高度余量避免门板顶缘误入上半段
    front_pts = pts[front]
    if front_pts.shape[0] >= min_points:
        split_y = float(front_pts[:, 1].max())
    else:
        split_y = float(np.percentile(pts[:, 1], 50.0))
    split_y += 0.01 * float(np.ptp(pts[:, 1]))

    upper = pts[:, 1] > split_y
    lower = pts[:, 1] <= split_y

    # 前面上: 上半段前框 (上半段 max-Z 处一条)
    front_upper = np.zeros(len(pts), dtype=bool)
    if upper.sum() >= min_points:
        uz = pts[upper, 2]
        u_lo, u_hi = np.percentile(uz, [1.0, 99.0])
        front_upper = upper & (pts[:, 2] > u_hi - 0.06 * (u_hi - u_lo + 1e-9))

    defs = [
        ('cabinet_back',        '后面',   back,                  [0, 0, -1]),
        ('cabinet_left_upper',  '左面上', left & upper,          [-1, 0, 0]),
        ('cabinet_left_lower',  '左面下', left & lower,          [-1, 0, 0]),
        ('cabinet_right_upper', '右面上', right & upper,         [1, 0, 0]),
        ('cabinet_right_lower', '右面下', right & lower,         [1, 0, 0]),
        ('cabinet_top',         '顶面',   top,                   [0, 1, 0]),
        ('cabinet_front_upper', '前面上', front_upper,           [0, 0, 1]),
        ('cabinet_front_lower', '前面下', front & lower,         [0, 0, 1]),
    ]

    regions = []
    total = len(pts)
    for part_id, name, mask, normal in defs:
        if mask.sum() < min_points:
            continue
        rp = pts[mask]
        regions.append({
            'points': rp.astype(np.float32),
            'type': 'planar',
            'normal': np.array(normal, dtype=np.float32),
            'center': rp.mean(0).astype(np.float32),
            'coverage': float(mask.sum() / total),
            'part_name': name,
            'part_id': part_id,
            'semantic_group': 'cabinet_face',
            'color': CABINET_FACE_COLORS.get(name, [0.6, 0.6, 0.6]),
            'spray': '2D栅格扫描',
        })
    return regions


def run_segment_cabinet_geometric(points):
    """柜子几何分解分割 (对齐 gen_guizi_path), 输出与 AI 分割同格式."""
    regions = decompose_cabinet(points)
    parts = _regions_to_parts(regions)
    resp = _wrap_parts_response(parts, len(points), '几何分解 (gen_guizi_path)')
    resp['_regions'] = regions
    return resp


def run_segment_pointcnn(points, category):
    """PointCNN 分割 — chair (6类) / table (11类)."""
    from pointcnn_inference import predict, group_into_regions, CATEGORY_CONFIG, CATEGORY_COLORS
    labels, cat_cfg, colors = predict(category, points)
    # chair 用更高阈值 (2.5%), 过滤无扶手椅子的虚假扶手分割
    min_ratio = 0.025 if category == 'chair' else 0.015
    regions = group_into_regions(points, labels, cat_cfg, colors, min_ratio=min_ratio)
    parts = _regions_to_parts(regions)
    num_class = cat_cfg['num_class']
    resp = _wrap_parts_response(parts, len(points), f'PointCNN ({num_class}类)')
    resp['_regions'] = regions
    return resp


def run_segment_ai(points, category):
    """统一的 AI 分割入口 — 按类别分发到 PointNeXt 或 PointCNN.

    输入:
      points: (N,3) numpy 原始点云
      category: 'cabinet' | 'chair' | 'table'

    输出:
      dict 与 run_segment_from_cache() 格式兼容
    """
    if category == 'cabinet':
        return run_segment_cabinet_geometric(points)
    elif category in ('chair', 'table'):
        return run_segment_pointcnn(points, category)
    else:
        raise ValueError(
            f"不支持的家具类别: {category}，仅支持 cabinet/chair/table"
        )


def run_segment_from_cache(regions_cache, category):
    """用 X/Y/Z 三个重力轴 + 自动检测 分别命名, 返回 4 个候选供用户选择."""
    # 还原 regions
    base_regions = []
    for rc in regions_cache:
        base_regions.append({
            'points': np.array(rc['points'], dtype=np.float32),
            'type': rc['type'],
            'normal': np.array(rc['normal'], dtype=np.float32),
            'center': np.array(rc['center'], dtype=np.float32),
            'coverage': rc['coverage'],
        })

    # 4 个重力候选
    candidates = []
    axes = {
        'X': [1.0, 0.0, 0.0],
        'Y': [0.0, 1.0, 0.0],
        'Z': [0.0, 0.0, 1.0],
    }
    # 自动检测的也加入
    auto_gravity = _compute_gravity_axis(base_regions).tolist()
    axes['auto'] = [round(float(x), 4) for x in auto_gravity]

    for label, grav in axes.items():
        # 深拷贝 regions 避免 name_parts 原地修改互相干扰
        regions = []
        for br in base_regions:
            regions.append({
                'points': br['points'].copy(),
                'type': br['type'],
                'normal': br['normal'].copy(),
                'center': br['center'].copy(),
                'coverage': br['coverage'],
            })
        parts = _build_parts_from_regions(regions, grav, category)

        # 统计
        n_seat = sum(1 for p in parts if '座面' in p['name'])
        n_back = sum(1 for p in parts if '靠背' in p['name'])
        n_legs = sum(1 for p in parts if '腿' in p['name'])
        n_horiz = sum(1 for p in parts if abs(p.get('_debug', {}).get('dot_gravity', 0)) > 0.7)
        n_vert  = sum(1 for p in parts if abs(p.get('_debug', {}).get('dot_gravity', 0)) < 0.35)
        n_skel  = sum(1 for p in parts if p['type'] == 'skeleton')

        candidates.append({
            'axis': label,
            'gravity': [round(float(x), 3) for x in grav],
            'summary': f"座面×{n_seat} 靠背×{n_back} 腿×{n_legs}",
            'detail': f"水平{n_horiz} 垂直{n_vert} 骨架{n_skel}",
            'n_parts': len(parts),
            'parts': parts,
        })

    # 默认选 auto, 同时生成统一云 (用第一个候选的 points_b64)
    default = next((c for c in candidates if c['axis'] == 'auto'), candidates[-1])

    return {
        'candidates': candidates,
        'default_axis': 'auto',
        'parts': default['parts'],
        'n_regions': default['n_parts'],
        'total_points': int(sum(len(p['points']) for p in base_regions)),
    }
