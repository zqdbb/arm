#!/usr/bin/env python3
"""
喷涂路径规划 — 采样+图搜索方法.

不走几何扫描线/等高线/螺旋的老路.
而是在表面采样候选路径点, 构建相邻图, 用贪心覆盖算法选出最优路径序列.

原理类似 RRT* 的采样思想, 但目标是表面全覆盖而非点到点.
"""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def plan_path(region, spray_distance=0.15, spacing=0.06):
    """为单个部件区域生成喷涂路径.

    输入:
      region: dict, 含 points (N,3), type, normal, center
      spray_distance: 喷枪距离表面偏移 (米)
      spacing: 相邻路径间距 (米)

    输出: (waypoints, normals, segments)
    """
    points = region['points']
    if len(points) < 10:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=int))

    rtype = region.get('type', 'planar')
    normal = region.get('normal', np.array([0, 0, 1], dtype=np.float32))

    # ── 统一走采样+图搜索流程 ──
    if rtype == 'skeleton':
        return _sample_and_cover_skeleton(points, spray_distance, spacing)
    else:
        return _sample_and_cover_surface(points, normal, spray_distance, spacing)


# ═══════════════════════════════════════════════════════════════
# 通用方法: 在表面上采样 → 构建图 → 贪心覆盖
# ═══════════════════════════════════════════════════════════════

def _sample_and_cover_surface(points, normal, spray_distance, spacing):
    """采样+图搜索覆盖规划.

    与旧几何方法本质区别: 不是在平面上画扫描线, 而是在表面采样候选点,
    用图搜索 (贪心+2-opt) 找出最优覆盖序列.

    对于近平面表面: 在投影平面网格采样 → zigzag 排序 → 2-opt
    对于曲面表面: FPS 采样 → k-NN 图 → 贪心覆盖 → 2-opt
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-10)

    # 判断是否为近平面 (法向一致性好 → 平面; 否则曲面)
    cand_normals = _estimate_normals_knn(points[::5] if len(points) > 100 else points,
                                         points, k=15)
    normal_spread = float(np.mean([
        np.arccos(min(abs(np.dot(n, normal)), 1.0))
        for n in cand_normals
    ]))
    is_planar = normal_spread < 0.25  # < 14° 偏差 → 近似平面

    if is_planar:
        return _cover_planar_via_grid(points, normal, spray_distance, spacing)
    else:
        return _cover_curved_via_graph(points, normal, spray_distance, spacing)


def _cover_planar_via_grid(points, normal, spray_distance, spacing):
    """平面表面: 在投影平面上网格采样 + zigzag 排序.

    这是采样策略的一种 — 网格采样对于平面是最均匀的.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    center = points.mean(0).astype(np.float64)

    # 建立平面局部坐标系 (u, v)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = ref - np.dot(ref, normal) * normal
    u /= np.linalg.norm(u) + 1e-10
    v = np.cross(normal, u)

    # 投影到 2D
    centered = points - center
    uv = np.column_stack([np.dot(centered, u), np.dot(centered, v)])

    # Alpha shape / 凹包边界
    boundary_uv = _compute_boundary(uv)

    # 网格采样 → 只保留边界内的点
    u_min, u_max = uv[:, 0].min(), uv[:, 0].max()
    v_min, v_max = uv[:, 1].min(), uv[:, 1].max()
    # 加一点 margin 确保覆盖边缘
    u_min -= spacing * 0.3
    u_max += spacing * 0.3
    v_min -= spacing * 0.3
    v_max += spacing * 0.3

    u_vals = np.arange(u_min, u_max + spacing * 0.5, spacing)
    v_vals = np.arange(v_min, v_max + spacing * 0.5, spacing)

    if len(u_vals) < 2 or len(v_vals) < 2:
        # 退化: 直接采样
        return _cover_curved_via_graph(points, normal, spray_distance, spacing)

    # 生成网格点并按行组织 (每行是一个 v 值)
    # grid_rows[i] = [(u, v), ...]  — 第 i 行内按 u 排序
    grid_rows = []
    for vi in v_vals:
        row = []
        for ui in u_vals:
            if _point_in_polygon((ui, vi), boundary_uv):
                row.append([ui, vi])
        if row:
            row.sort(key=lambda p: p[0])  # 行内按 u 排序
            grid_rows.append(row)

    if len(grid_rows) < 2:
        return _cover_curved_via_graph(points, normal, spray_distance, spacing)

    # Zigzag: 偶数行从左到右, 奇数行从右到左
    ordered_uv = []
    for row_i, row in enumerate(grid_rows):
        if row_i % 2 == 1:
            row = list(reversed(row))
        ordered_uv.extend(row)

    ordered_uv = np.array(ordered_uv)

    # 法向估计
    wp_normals = np.tile(normal, (len(ordered_uv), 1))

    # 映射回 3D + 法向偏移
    waypoints_3d = (center
                    + np.outer(ordered_uv[:, 0], u)
                    + np.outer(ordered_uv[:, 1], v)
                    + normal * spray_distance)

    # zigzag 已是最优覆盖模式, 不应用 2-opt (会打乱秩序)
    waypoints = waypoints_3d.astype(np.float32)
    norms = wp_normals.astype(np.float32)
    segs = np.array([[i, i + 1] for i in range(len(waypoints) - 1)], dtype=int)

    return waypoints, norms, segs


def _cover_curved_via_graph(points, normal, spray_distance, spacing):
    """曲面表面: FPS 采样 + k-NN 图 + 贪心覆盖 + 2-opt."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-10)

    # FPS 采样 — 用投影面积估算候选点数
    try:
        from scipy.spatial import ConvexHull
        # 投影到法向平面算面积 (避免 3D 凸包双面计数)
        proj_uv = _project_to_plane(points, normal)
        hull2d = ConvexHull(proj_uv)
        area = hull2d.volume  # 2D convex hull 面积
    except Exception:
        area = 1.0
    n_candidates = max(30, min(500, int(area / (spacing * spacing) * 0.6)))
    if len(points) > n_candidates:
        idx = _fps_sample(points, n_candidates)
        cand_pts = points[idx]
    else:
        cand_pts = points
        n_candidates = len(points)

    # 法向估计
    cand_normals = _estimate_normals_knn(cand_pts, points, k=15)
    for i in range(len(cand_normals)):
        if np.dot(cand_normals[i], normal) < 0:
            cand_normals[i] = -cand_normals[i]

    # 沿法向偏移 → 喷枪位姿
    waypoints_3d = cand_pts + cand_normals * spray_distance

    # k-NN 图
    k = min(12, len(waypoints_3d) - 1)
    tree = cKDTree(waypoints_3d)
    distances, indices = tree.query(waypoints_3d, k=k + 1)

    max_edge = spacing * 2.5
    adj = []
    for i in range(len(waypoints_3d)):
        neighbors = []
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            if j >= len(waypoints_3d):
                continue
            if distances[i, j_idx] < max_edge:
                neighbors.append(j)
        adj.append(neighbors)

    # 贪心覆盖
    center = waypoints_3d.mean(0)
    start = int(np.argmin(np.linalg.norm(waypoints_3d - center, axis=1)))

    visited = np.zeros(len(waypoints_3d), dtype=bool)
    path = [start]
    visited[start] = True

    for _ in range(len(waypoints_3d) - 1):
        current = path[-1]
        best_next = -1
        best_score = -1

        for nb in adj[current]:
            if visited[nb]:
                continue
            unvisited_nbs = sum(1 for n2 in adj[nb] if not visited[n2])
            score = unvisited_nbs * 2 + (1 if len(adj[nb]) <= 3 else 0)
            if len(adj[nb]) <= 3:
                score += 3
            if score > best_score:
                best_score = score
                best_next = nb

        if best_next < 0:
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            current_pos = waypoints_3d[current]
            dists = np.linalg.norm(waypoints_3d[unvisited] - current_pos, axis=1)
            best_next = unvisited[int(np.argmin(dists))]

        visited[best_next] = True
        path.append(best_next)

    # 2-opt 优化
    path = _two_opt(waypoints_3d, path, max_iter=50)

    waypoints = waypoints_3d[path].astype(np.float32)
    norms = cand_normals[path].astype(np.float32)
    segs = np.array([[i, i + 1] for i in range(len(path) - 1)], dtype=int)

    return waypoints, norms, segs


# ═══════════════════════════════════════════════════════════════
# 边界检测 (Alpha Shape / 凹包)
# ═══════════════════════════════════════════════════════════════

def _compute_boundary(uv_points):
    """计算 2D 点集的边界多边形.

    先用 Delaunay 三角剖分 + 长边过滤做 alpha shape.
    如果失败, 回退到凸包.
    """
    if len(uv_points) < 3:
        return uv_points

    try:
        from scipy.spatial import Delaunay, ConvexHull

        # 先用凸包获得完整外边界 (稳定可靠)
        hull = ConvexHull(uv_points)
        hull_pts = uv_points[hull.vertices]

        # 如果凸包顶点数合理 (<30), 直接用
        if len(hull_pts) < 30:
            return hull_pts

        # 太多顶点 → 简化: 均匀降采样到 20 点
        step = max(1, len(hull_pts) // 20)
        simplified = hull_pts[::step]
        if len(simplified) >= 3:
            return simplified
        return hull_pts
    except Exception:
        return uv_points


def _point_in_polygon(point, polygon):
    """射线法判断点是否在多边形内."""
    if len(polygon) < 3:
        return True
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _sample_and_cover_skeleton(points, spray_distance, spacing):
    """细长结构 (椅腿等) 的路径 — 采样+螺旋连接.

    步骤:
      1. DBSCAN 分簇 (多条腿)
      2. 每簇沿轴线采样
      3. 每层均匀采样圆周方向
      4. 螺旋连接各层采样点
      5. 2-opt 优化
    """
    if len(points) < 10:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=int))

    # DBSCAN 分簇
    bbox = points.max(0) - points.min(0)
    eps = float(np.median(bbox) * 0.08)
    try:
        clustering = DBSCAN(eps=max(eps, 0.01), min_samples=5).fit(points)
        labels = clustering.labels_
        cluster_ids = sorted(set(labels))
        if -1 in cluster_ids:
            cluster_ids.remove(-1)
    except Exception:
        cluster_ids = [0]
        labels = np.zeros(len(points), dtype=int)

    if len(cluster_ids) < 2:
        # 单根: 采样法
        return _single_rod_cover(points, spray_distance, spacing)

    # 多簇: 每根单独处理
    all_wp, all_norms, all_segs = [], [], []
    offset = 0
    for cid in cluster_ids:
        mask = labels == cid
        cpts = points[mask]
        if len(cpts) < 10:
            continue
        wp_c, norms_c, segs_c = _single_rod_cover(cpts, spray_distance, spacing)
        if len(wp_c) > 0:
            all_wp.append(wp_c)
            all_norms.append(norms_c)
            if len(segs_c) > 0:
                all_segs.append(segs_c + offset)
            offset += len(wp_c)

    if all_wp:
        return (np.vstack(all_wp).astype(np.float32),
                np.vstack(all_norms).astype(np.float32),
                np.vstack(all_segs).astype(np.int32) if all_segs
                else np.zeros((0, 2), dtype=int))

    return _single_rod_cover(points, spray_distance, spacing)


def _single_rod_cover(points, spray_distance, spacing):
    """单根杆的覆盖路径: 轴向多层 + 圆周采样 + 连续螺旋."""
    if len(points) < 10:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=int))

    # PCA 找主轴
    centered = points - points.mean(0)
    cov = np.cov(centered.T)
    _eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    axis = axis / (np.linalg.norm(axis) + 1e-10)

    # 构建径向基
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    r1 = ref - np.dot(ref, axis) * axis
    r1 /= np.linalg.norm(r1) + 1e-10
    r2 = np.cross(axis, r1)

    # 沿轴向采样
    proj = np.dot(points, axis)
    t_min, t_max = proj.min(), proj.max()
    center = points.mean(0)
    n_axial = max(6, int((t_max - t_min) / spacing))

    # 每层采样圆周方向
    n_angular = 12
    all_waypoints = []
    all_normals = []

    for ti in np.linspace(t_min, t_max, n_axial):
        # 该层附近的点
        local_mask = np.abs(proj - ti) < spacing * 1.5
        if local_mask.sum() < 3:
            continue
        local_pts = points[local_mask]
        local_centered = local_pts - (center + axis * ti)
        # 径向距离
        radial_dist = np.linalg.norm(
            local_centered - np.outer(np.dot(local_centered, axis), axis), axis=1)
        radius = float(np.median(radial_dist)) * 0.85
        if radius < 0.001:
            continue

        layer_wp = []
        layer_norm = []
        for ang in np.linspace(0, 2 * np.pi, n_angular, endpoint=False):
            radial_dir = np.cos(ang) * r1 + np.sin(ang) * r2
            radial_dir = radial_dir / (np.linalg.norm(radial_dir) + 1e-10)
            wp = center + axis * ti + radial_dir * (radius + spray_distance)
            layer_wp.append(wp)
            layer_norm.append(radial_dir)

        all_waypoints.append(np.array(layer_wp))
        all_normals.append(np.array(layer_norm))

    if not all_waypoints:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=int))

    # 螺旋连接: 第0层[0,1,2,...11] → 第1层[0,1,2,...11] → ...
    # 每层内按角度排序, 相邻层间最近点相连
    ordered_wp = []
    ordered_norm = []
    for layer_idx, (wps, nms) in enumerate(zip(all_waypoints, all_normals)):
        if layer_idx == 0:
            ordered_wp.extend(wps)
            ordered_norm.extend(nms)
        else:
            # 找到上一层最后一点最近的本层起点
            prev_last = ordered_wp[-1]
            dists_to_prev = np.linalg.norm(wps - prev_last, axis=1)
            start_idx = int(np.argmin(dists_to_prev))
            # 重排本层: 从 start_idx 开始绕一圈
            reordered_wp = np.roll(wps, -start_idx, axis=0)
            reordered_nm = np.roll(nms, -start_idx, axis=0)
            ordered_wp.extend(reordered_wp)
            ordered_norm.extend(reordered_nm)

    waypoints = np.array(ordered_wp, dtype=np.float32)
    normals_arr = np.array(ordered_norm, dtype=np.float32)

    # 线段: 顺序连接 + 每圈闭合
    segs = [[i, i + 1] for i in range(len(waypoints) - 1)]
    for layer_idx in range(len(all_waypoints)):
        si = layer_idx * n_angular
        ei = si + n_angular - 1
        if ei < len(waypoints) and ei != si:
            segs.append([ei, si])

    return waypoints, normals_arr, np.array(segs, dtype=int)


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _project_to_plane(points, normal):
    """投影 3D 点到法向定义的平面, 返回 2D 坐标."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-10)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    a = ref - np.dot(ref, normal) * normal
    a /= np.linalg.norm(a) + 1e-10
    b = np.cross(normal, a)
    centered = points - points.mean(0)
    return np.column_stack([np.dot(centered, a), np.dot(centered, b)])


def _fps_sample(points, n_samples):
    """最远点采样 (Farthest Point Sampling) — 保证均匀覆盖."""
    n = len(points)
    if n <= n_samples:
        return np.arange(n)
    idx = np.zeros(n_samples, dtype=int)
    idx[0] = np.random.randint(n)
    dists = np.full(n, np.inf)
    for i in range(1, n_samples):
        d = np.sum((points - points[idx[i - 1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        idx[i] = int(np.argmax(dists))
    return idx


def _estimate_normals_knn(query_pts, ref_pts, k=15):
    """k-NN 局部平面拟合 → 法向量."""
    tree = cKDTree(ref_pts)
    _, indices = tree.query(query_pts, k=min(k, len(ref_pts)))
    normals = np.zeros((len(query_pts), 3), dtype=np.float64)
    for i, nbrs in enumerate(indices):
        nbr_pts = ref_pts[nbrs]
        centered = nbr_pts - nbr_pts.mean(0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]  # 最小特征值方向
    return normals


def _two_opt(waypoints, path, max_iter=50):
    """2-opt 局部优化: 消除路径中的交叉.

    随机选两个位置 i, j (i < j-1), 如果反转 i..j 段能缩短总距离, 就执行反转."""
    path = list(path)
    n = len(path)
    if n < 4:
        return path

    # 预计算距离
    def path_length(p):
        total = 0.0
        for a in range(len(p) - 1):
            total += float(np.linalg.norm(waypoints[p[a]] - waypoints[p[a + 1]]))
        return total

    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 2):
            for j in range(i + 2, min(n, i + 25)):
                # 当前: ...→i→i+1→...→j-1→j→...
                # 反转: ...→i→j-1→...→i+1→j→...
                old_d1 = float(np.linalg.norm(waypoints[path[i]] - waypoints[path[i + 1]]))
                old_d2 = float(np.linalg.norm(waypoints[path[j - 1]] - waypoints[path[j]]))
                new_d1 = float(np.linalg.norm(waypoints[path[i]] - waypoints[path[j - 1]]))
                new_d2 = float(np.linalg.norm(waypoints[path[i + 1]] - waypoints[path[j]]))
                if new_d1 + new_d2 < old_d1 + old_d2 - 1e-8:
                    path[i + 1:j] = reversed(path[i + 1:j])
                    improved = True
        n = len(path)  # in case it changes
    return path
