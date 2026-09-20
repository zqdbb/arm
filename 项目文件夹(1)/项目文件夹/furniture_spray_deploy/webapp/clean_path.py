#!/usr/bin/env python3
"""gen_table_path 风格的干净喷涂路径生成 (替代 Open3D 曲面重建的 plan_path).

把 AI 分割得到的部件 region (归一化点云) 生成结构化路径:
  桌面/座面 -> 顶面 + 4 侧面 (Z字 zigzag)
  靠背     -> 前面 + 后面 + 顶面
  腿/底座  -> 单根腿竖直扫描线
  扶手     -> 沿最长轴扫描线

坐标系: 与 webapp 一致 (load_and_normalize 后未旋转), 输出归一化坐标,
直接叠加在分割点云上显示.
"""

import numpy as np

LINE_STEP = 0.05      # zigzag 行距 (归一化坐标)
PT_DENSITY = 0.02     # 沿线点距
LEG_N_PTS = 12        # 单根腿线点数

TOP_SO = 0.06         # 顶面 standoff
FACE_SO = 0.06        # 侧面 standoff
LEG_SO = 0.15         # 腿 standoff
LEG_FEED = 0.12       # 腿进给(切线方向线距, 相对腿宽的比例)


def role_of(part_id):
    """部件 id -> 喷涂角色."""
    pid = (part_id or '').lower()
    if 'tabletop' in pid:
        return '桌面'
    if 'table_base' in pid:
        return '腿'
    if 'chair_seat' in pid:
        return '座面'
    if 'chair_back' in pid or 'chair_head' in pid:
        return '靠背'
    if 'chair_arm' in pid:
        return '扶手'
    if 'chair_base' in pid:
        return '腿'
    return None


def _role_from_name(name):
    nm = name or ''
    if '座面' in nm or '桌面' in nm or '桌板' in nm:
        return '座面' if '座' in nm else '桌面'
    if '靠背' in nm:
        return '靠背'
    if '腿' in nm or '底座' in nm:
        return '腿'
    if '扶手' in nm:
        return '扶手'
    return None


def _zigzag(scan_axis, scan_range, step_axis, step_range,
            const_axis, const_val, normal, standoff):
    s0, s1 = scan_range
    t0, t1 = step_range
    n_lines = max(2, int(round((t1 - t0) / LINE_STEP)) + 1)
    t_vals = np.linspace(t0, t1, n_lines)
    n_pts = max(3, int(round((s1 - s0) / PT_DENSITY)) + 1)
    s_fwd = np.linspace(s0, s1, n_pts)
    s_rev = s_fwd[::-1]
    normal = np.asarray(normal, dtype=float)
    wps = []
    for li, t in enumerate(t_vals):
        ss = s_fwd if li % 2 == 0 else s_rev
        for s in ss:
            p = np.zeros(3)
            p[scan_axis] = s
            p[step_axis] = t
            p[const_axis] = const_val
            p += normal * standoff
            wps.append(p)
    return np.array(wps)


def _up_axis(region):
    """桌面/座面是水平薄板, 竖直轴 = 法线方向 (取法向量的最大分量轴)."""
    n = region.get('normal')
    if n is not None and len(n) == 3:
        n = np.abs(np.asarray(n, dtype=float))
        if n.max() > 0.5:
            return int(np.argmax(n))
    # fallback: PCA 最小方差轴
    pts = np.asarray(region.get('points'), dtype=np.float32)
    c = pts - pts.mean(0)
    cov = np.cov(c.T)
    _, evecs = np.linalg.eigh(cov)
    return int(np.argmax(np.abs(evecs[:, 0])))


def _clip_horizontal_slab(pts, up_axis, rel_gap=0.25):
    """裁剪水平薄板: 只保留竖直轴最高处的密集层, 丢掉往下延伸的稀疏误分点.

    当分割把桌腿/支撑误并入桌面时, 桌面点云会在竖直轴方向往下伸出稀疏尾部.
    沿竖直轴做直方图, 找到最高处的密集块, 下方稀疏尾部即被裁掉.
    对本身就是干净薄板的部件不会误删 (密集块覆盖整个薄板).
    """
    if len(pts) < 20:
        return pts
    v = pts[:, up_axis]
    n_bins = max(20, min(48, len(pts) // 150))
    hist, edges = np.histogram(v, bins=n_bins)
    peak = float(hist.max())
    dense = hist >= peak * rel_gap

    low_edge = edges[-1]
    for i in range(n_bins - 1, -1, -1):
        if dense[i]:
            low_edge = edges[i]
        else:
            # 当前 bin 稀疏: 下方若还有密集 bin 则视为薄板内部空隙, 否则薄板结束
            if not dense[:i].any():
                break
    return pts[v >= low_edge]


def _box_faces(pts, prefix, top_so, face_so, include_sides=True):
    """把近似长方体的部件拆成 顶面/前面/后面/(左面/右面) 的 zigzag 路径."""
    x0, x1 = pts[:, 0].min(), pts[:, 0].max()
    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    z0, z1 = pts[:, 2].min(), pts[:, 2].max()
    defs = [
        (f'{prefix}顶面', 0, (x0, x1), 2, (z0, z1), 1, y1, [0, 1, 0], top_so),
        (f'{prefix}前面', 0, (x0, x1), 1, (y0, y1), 2, z1, [0, 0, 1], face_so),
        (f'{prefix}后面', 0, (x0, x1), 1, (y0, y1), 2, z0, [0, 0, -1], face_so),
    ]
    if include_sides:
        defs += [
            (f'{prefix}左面', 2, (z0, z1), 1, (y0, y1), 0, x0, [-1, 0, 0], face_so),
            (f'{prefix}右面', 2, (z0, z1), 1, (y0, y1), 0, x1, [1, 0, 0], face_so),
        ]
    out = {}
    for name, sa, sr, ta, tr, ca, cv, normal, so in defs:
        wps = _zigzag(sa, sr, ta, tr, ca, cv, normal, so)
        if len(wps):
            out[name] = wps
    return out


def _split_legs(pts):
    """把整块腿/底座点云拆成单根 (处理横撑把腿连起来的情况).

    分别假设 3 个轴为竖直轴, 取底部 ~45% 的点 (那里只有腿、没有横撑),
    在水平面上 DBSCAN, 选能拆出最多根的那个轴作为竖直轴, 再把全部点
    按水平位置就近分到各腿."""
    from sklearn.cluster import DBSCAN

    best_vertical, best_clusters = None, []
    for vertical in (0, 1, 2):
        v = pts[:, vertical]
        v_lo, v_hi = v.min(), v.max()
        thr = v_lo + 0.45 * (v_hi - v_lo)
        bottom = pts[v < thr]
        if len(bottom) < 5:
            continue
        horiz = [a for a in (0, 1, 2) if a != vertical]
        bbox2d = bottom[:, horiz].max(0) - bottom[:, horiz].min(0)
        eps = max(float(np.median(bbox2d) * 0.15), 0.02)
        labels = DBSCAN(eps=eps, min_samples=5).fit(bottom[:, horiz]).labels_
        clusters = []
        for cid in set(labels):
            if cid == -1:
                continue
            cpts = bottom[labels == cid]
            if len(cpts) >= 5:
                clusters.append(cpts)
        if len(clusters) > len(best_clusters):
            best_vertical = vertical
            best_clusters = clusters

    if not best_clusters:
        return [pts]

    horiz = [a for a in (0, 1, 2) if a != best_vertical]
    centers = np.array([c[:, horiz].mean(0) for c in best_clusters])
    dist = np.linalg.norm(pts[:, None, horiz] - centers[None, :, :], axis=2)
    assign = np.argmin(dist, axis=1)
    return [pts[assign == j] for j in range(len(centers))]


def _line_for_leg(gv, center, standoff, n_pts=LEG_N_PTS):
    """单根腿 -> 3 条竖直扫描线 (之字形, 对齐 spray_table_all_adaptive 的 _build_tcp_leg_zpattern).

    在水平面内取该腿朝外(相对整体 center)的法向 n, 沿切线方向铺 3 条线
    (feed offsets [-feed, 0, +feed]), 每条线沿竖直轴从上到下交替扫描.
    机械臂喷单条腿走的正是这 3 条线 (LEG_FEED 进给的之字形).
    """
    rng = gv.max(0) - gv.min(0)
    along = int(np.argmax(rng))          # 竖直轴
    lo, hi = gv[:, along].min(), gv[:, along].max()
    ts = np.linspace(hi, lo, n_pts)
    horiz = [a for a in (0, 1, 2) if a != along]

    # 水平面内朝外的单位法向 (该腿相对整体中心在哪一侧)
    nv = np.zeros(3)
    for a in horiz:
        nv[a] = 1.0 if gv[:, a].mean() >= center[a] else -1.0
    nv /= (np.linalg.norm(nv) + 1e-9)
    # 切线方向 = 法向在水平面内旋转 90°
    tv = np.zeros(3)
    tv[horiz[0]] = -nv[horiz[1]]
    tv[horiz[1]] = nv[horiz[0]]

    # 进给: 按腿宽比例 (对齐 LEG_FEED 相对腿宽的尺度)
    width = float(np.median([rng[a] for a in horiz]))
    feed = max(width * LEG_FEED, 1e-3)
    offsets = (-feed, 0.0, feed)

    lines = []
    for oi, off in enumerate(offsets):
        base = np.zeros(3)
        base[along] = gv[:, along].mean()
        for a in horiz:
            base[a] = gv[:, a].mean() + nv[a] * standoff + tv[a] * off
        pts = np.tile(base, (n_pts, 1))
        pts[:, along] = ts if oi % 2 == 0 else ts[::-1]
        lines.append(pts)
    return lines


def _dense_band(pts, axis, rel_gap=0.25):
    """沿 axis 找密集主带范围 [lo, hi], 忽略稀疏噪声/误分点."""
    v = pts[:, axis]
    if len(v) < 10:
        return float(v.min()), float(v.max())
    n_bins = max(20, min(48, len(pts) // 150))
    hist, edges = np.histogram(v, bins=n_bins)
    peak = float(hist.max())
    dense = hist >= peak * rel_gap
    di = np.where(dense)[0]
    if len(di) == 0:
        return float(v.min()), float(v.max())
    return float(edges[di.min()]), float(edges[di.max() + 1])


def _robust_range(pts, axis, lo_pct=2.0, hi_pct=98.0):
    """沿 axis 取分位数范围 [lo, hi], 对均匀面板更稳 (避开少量噪声点)."""
    v = pts[:, axis]
    return float(np.percentile(v, lo_pct)), float(np.percentile(v, hi_pct))


def _detect_up_axis(pts):
    """水平薄板(座面)的竖直轴 = 密集主带最薄的轴."""
    best_axis, best_thick = None, np.inf
    for a in (0, 1, 2):
        lo, hi = _dense_band(pts, a)
        if hi - lo < best_thick:
            best_axis, best_thick = a, hi - lo
    return best_axis


def _chair_side_edge(edge_axis, edge_lo, edge_hi, const_axis, const_val,
                     up_axis, up_val, normal, standoff, feed, n_pts=10):
    """椅子座面侧面薄条: 沿边界的一条水平线, 3线之字沿 -up 方向下移."""
    wps = []
    for oi, off in enumerate((0.0, -feed, -2.0 * feed)):
        ts = np.linspace(edge_lo, edge_hi, n_pts)
        if oi % 2 == 1:
            ts = ts[::-1]
        base = np.zeros(3)
        base[up_axis] = up_val + off
        base[const_axis] = const_val
        base += np.asarray(normal, dtype=float) * standoff
        line = np.tile(base, (n_pts, 1))
        line[:, edge_axis] = ts
        wps.append(line)
    return np.vstack(wps)


def _chair_seat_path(pts, top_so, face_so):
    """座面: 顶面 + 4 侧面薄条 (对齐 chair_all_adaptive 的 SEAT_SIDE_EDGES).

    座面是水平薄板, 全部路点都贴着顶面高度(密集主带上沿)生成,
    侧面是沿边界的薄条, 不再把整个座面厚度扫进去.
    """
    up = _detect_up_axis(pts)
    horiz = [a for a in (0, 1, 2) if a != up]
    h0, h1 = horiz

    lo_u, hi_u = _dense_band(pts, up)
    seat_top = hi_u

    # 顶面附近点求水平足迹 (避开靠背/腿误分点)
    u_span = hi_u - lo_u
    top_pts = pts[pts[:, up] >= seat_top - max(u_span * 0.5, 0.02)]
    if len(top_pts) < 5:
        top_pts = pts
    r0 = _dense_band(top_pts, h0)
    r1 = _dense_band(top_pts, h1)

    up_normal = np.zeros(3)
    up_normal[up] = 1.0
    out = {'座面顶面': _zigzag(h0, r0, h1, r1, up, seat_top, up_normal, top_so)}

    feed = max((r0[1] - r0[0] + r1[1] - r1[0]) * 0.5 * 0.06, 0.01)
    sides = [
        (h1, r1, h0, r0[0], -1.0, h0),   # h0 最小侧
        (h1, r1, h0, r0[1], +1.0, h0),   # h0 最大侧
        (h0, r0, h1, r1[0], -1.0, h1),   # h1 最小侧
        (h0, r0, h1, r1[1], +1.0, h1),   # h1 最大侧
    ]
    for i, (e_axis, e_rng, c_axis, c_val, sgn, n_axis) in enumerate(sides):
        normal = np.zeros(3)
        normal[n_axis] = sgn
        out[f'座面侧{i + 1}'] = _chair_side_edge(
            e_axis, e_rng[0], e_rng[1], c_axis, c_val,
            up, seat_top, normal, face_so, feed)
    return out


def _chair_back_path(pts, top_so, face_so):
    """靠背: 顶面薄条 + 前面 + 后面 (对齐 chair_all_adaptive 靠背三面).

    靠背是竖直薄板: 厚度轴 = 最小跨度; 竖直轴 = 全局 Y (Y-up 场景,
    load_and_normalize 不旋转); 宽度 = 剩下的轴.
    不能用跨度最大当竖直轴 —— 靠背的高和宽跨度接近, 会把竖直轴误判成宽,
    于是顶面薄条被错放到侧面去.
    """
    rng = pts.max(0) - pts.min(0)
    thin = int(np.argmin(rng))          # 厚度轴 (靠背面对的方向)
    up = 1                              # 全局竖直轴 (Y-up 场景)
    if up == thin:                      # 异常回退: 靠背不该薄在竖直轴
        up = int(np.argmax(rng))
    width = [a for a in (0, 1, 2) if a not in (thin, up)][0]

    t_lo, t_hi = _robust_range(pts, thin)    # 厚度 (前/后)
    h_lo, h_hi = _robust_range(pts, up)      # 底 / 顶
    w_lo, w_hi = _robust_range(pts, width)   # 宽

    up_normal = np.zeros(3)
    up_normal[up] = 1.0
    front_n = np.zeros(3)
    front_n[thin] = -1.0
    back_n = np.zeros(3)
    back_n[thin] = +1.0

    return {
        '靠背顶面': _zigzag(width, (w_lo, w_hi), thin, (t_lo, t_hi), up, h_hi, up_normal, top_so),
        '靠背前面': _zigzag(width, (w_lo, w_hi), up, (h_lo, h_hi), thin, t_lo, front_n, face_so),
        '靠背后面': _zigzag(width, (w_lo, w_hi), up, (h_lo, h_hi), thin, t_hi, back_n, face_so),
    }


def _chair_legs_path(pts, leg_so):
    """底座(4腿+4横杆): 4面 × (2腿竖线 + 1横杆线), 每条3线之字.

    对齐 chair_all_adaptive 的 build_side_items / _spray_leg_side:
    每个面喷该侧 2 条腿的竖直扫描线 + 1 根连接横杆.
    """
    along = 1                              # 竖直轴 = 全局 Y (Y-up 场景)
    horiz = [a for a in (0, 1, 2) if a != along]
    h0, h1 = horiz
    lo, hi = pts[:, along].min(), pts[:, along].max()

    # 1) 底部 15% 点找腿中心 (避开横杆/座面把腿连起来)
    thr = lo + 0.15 * (hi - lo)
    bottom = pts[pts[:, along] < thr]
    if len(bottom) < 5:
        return {}
    from sklearn.cluster import DBSCAN, KMeans
    bbox2d = bottom[:, horiz].max(0) - bottom[:, horiz].min(0)

    # DBSCAN 试多个 eps, 取簇数最多的 (腿最可能被分开)
    best_labels, best_n = None, 0
    for factor in (0.10, 0.08, 0.06, 0.05):
        eps = max(float(np.median(bbox2d) * factor), 0.015)
        labels = DBSCAN(eps=eps, min_samples=5).fit(bottom[:, horiz]).labels_
        n = len(set(labels)) - (1 if -1 in labels else 0)
        if n > best_n:
            best_n, best_labels = n, labels

    if best_n < 2:
        # 回退: 默认 4 条腿
        km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(bottom[:, horiz])
        centers = km.cluster_centers_
    else:
        centers = []
        for cid in set(best_labels):
            if cid == -1:
                continue
            cpts = bottom[best_labels == cid]
            if len(cpts) >= 5:
                centers.append(cpts[:, horiz].mean(0))
        centers = np.array(centers)

    # 2) 全部底座点就近分配, 得每条腿竖直范围
    d = np.linalg.norm(pts[:, None, horiz] - centers[None, :, :], axis=2)
    assign = np.argmin(d, axis=1)
    dmin = d.min(1)

    # 3) 横杆高度: 中低处离腿中心远的点
    low_mask = pts[:, along] < lo + 0.6 * (hi - lo)
    bar_y = lo + 0.45 * (hi - lo)
    if low_mask.sum() > 20:
        dl = dmin[low_mask]
        thr_d = np.percentile(dl, 80)
        bp = pts[low_mask][dl > thr_d]
        if len(bp) >= 10:
            bar_y = float(np.median(bp[:, along]))

    leg_width = max(float(np.median(bbox2d)) * 0.15, 0.02)
    feed = leg_width * 0.5
    n_pts = LEG_N_PTS
    # 全局统一腿高 (所有腿同高, 单腿 min/max 会被误分点/横杆拉长)
    leg_lo_g, leg_hi_g = _robust_range(pts, along)

    paths = {}
    face_idx = 0
    for axis, axis_idx in ((h0, 0), (h1, 1)):
        tangent_idx = 1 - axis_idx
        tangent_axis = h1 if axis == h0 else h0
        order = np.argsort(centers[:, axis_idx])
        for side, sgn, idxs in (('min', -1.0, order[:2]), ('max', +1.0, order[-2:])):
            face_idx += 1
            normal = np.zeros(3)
            normal[axis] = sgn
            # 2 条腿竖线
            for li, li_idx in enumerate(idxs):
                # 所有腿同高: 用全局鲁棒竖直范围, 不用单腿 min/max (会被误分点拉长)
                leg_lo_i, leg_hi_i = leg_lo_g, leg_hi_g
                lines = []
                for oi, off in enumerate((-feed, 0.0, feed)):
                    base = np.zeros(3)
                    base[along] = 0.0
                    base[h0] = centers[li_idx, 0]
                    base[h1] = centers[li_idx, 1]
                    base += normal * leg_so
                    base[tangent_axis] += off
                    ts = np.linspace(leg_hi_i, leg_lo_i, n_pts)
                    if oi % 2 == 1:
                        ts = ts[::-1]
                    line = np.tile(base, (n_pts, 1))
                    line[:, along] = ts
                    lines.append(line)
                paths[f'腿{face_idx}-{li + 1}'] = np.vstack(lines)
            # 1 根横杆 (连接两腿, 3线之字沿竖直方向)
            bar_axis_val = float(centers[idxs, axis_idx].mean())
            t0 = float(centers[idxs[0], tangent_idx])
            t1 = float(centers[idxs[1], tangent_idx])
            bar_lines = []
            for oi, off in enumerate((-feed, 0.0, feed)):
                base = np.zeros(3)
                base[along] = bar_y + off
                base[axis] = bar_axis_val
                base += normal * leg_so
                ts = np.linspace(t0, t1, n_pts)
                if oi % 2 == 1:
                    ts = ts[::-1]
                line = np.tile(base, (n_pts, 1))
                line[:, tangent_axis] = ts
                bar_lines.append(line)
            paths[f'横杆{face_idx}'] = np.vstack(bar_lines)
    return paths


def generate_clean_path(region, spacing=0.06, spray_distance=0.15):
    """region -> (waypoints, normals, segments) 或 None.

    与 generate_path_for_part 输出格式一致, 供 app.py 直接使用.
    """
    pts = np.asarray(region.get('points'), dtype=np.float32)
    if pts is None or len(pts) < 5:
        return None

    part_id = (region.get('part_id') or '')
    if part_id.startswith('cabinet_'):
        return _cabinet_face_path(region, spacing)

    role = role_of(region.get('part_id', ''))
    if role is None:
        role = _role_from_name(region.get('part_name', ''))
    if role is None:
        return None

    top_so = max(FACE_SO, float(spacing) * 0.8)   # 用 spacing 微调 standoff
    face_so = max(FACE_SO, float(spacing) * 0.8)
    leg_so = max(LEG_SO, float(spacing) * 1.5)

    # 椅子部件走 chair_all_adaptive 风格的路径, 桌子保持原逻辑
    is_chair = 'chair' in (region.get('part_id') or '').lower()

    paths = {}
    if role == '座面':
        paths = _chair_seat_path(pts, top_so, face_so)
    elif role == '靠背':
        paths = _chair_back_path(pts, top_so, face_so)
    elif role == '腿' and is_chair:
        paths = _chair_legs_path(pts, leg_so)
    elif role == '桌面':
        # 桌面是水平薄板: 先裁掉竖直方向向下延伸的稀疏误分点(桌腿等),
        # 再对真正的薄板生成顶面 + 侧面, 避免侧面把腿也扫进去.
        up_axis = _up_axis(region)
        slab = _clip_horizontal_slab(pts, up_axis)
        if len(slab) >= 5:
            pts = slab
        paths = _box_faces(pts, role, top_so, face_so, include_sides=True)
    elif role == '腿':
        legs = _split_legs(pts)
        center = pts.mean(0)             # 整个腿部件中心, 判断内外
        for i, lp in enumerate(legs):
            for j, wps in enumerate(_line_for_leg(lp, center, leg_so)):
                paths[f'{role}{i + 1}-{j + 1}'] = wps
    elif role == '扶手':
        center = pts.mean(0)
        for j, wps in enumerate(_line_for_leg(pts, center, leg_so)):
            paths[f'扶手-{j + 1}'] = wps

    return _paths_to_output(paths)


def _paths_to_output(paths):
    """把 {名称: (N,3)路点} 合并成 (waypoints, normals, segments)."""
    wp_list, seg_list = [], []
    offset = 0
    for wps in paths.values():
        if len(wps) < 2:
            continue
        wp_list.append(wps)
        n = len(wps)
        seg_list.append(np.stack([np.arange(n - 1), np.arange(1, n)], 1) + offset)
        offset += n

    if not wp_list:
        return None

    wp = np.vstack(wp_list).astype(np.float32)
    segs = np.vstack(seg_list).astype(np.int64)
    norms = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (len(wp), 1))
    return wp, norms, segs


def _cabinet_face_axes(ca):
    """柜子面板的 scan/step 轴 (对齐 gen_guizi_path 的 zigzag 约定).

    const=Y(顶面): scan X, step Z; 竖面(front/back/side): scan 水平轴, step Y.
    """
    if ca == 1:
        return 0, 2
    return (0 if ca == 2 else 2), 1


def _cabinet_face_path(region, spacing=0.06):
    """柜子单个外表面部件 -> 平面 zigzag 路径 (对齐 gen_guizi_path).

    面板法向决定 const 轴 (法向主导轴) 与朝外方向, 尺寸范围取自该面板点云,
    尺寸自适应 —— 同结构不同尺寸的柜子面板也能正确挂上路径.
    """
    pts = np.asarray(region.get('points'), dtype=np.float32)
    if pts is None or len(pts) < 5:
        return None

    normal = np.asarray(region.get('normal'), dtype=np.float64)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-9:
        return None
    n = normal / nrm

    ca = int(np.argmax(np.abs(n)))           # const 轴
    sgn = 1.0 if n[ca] > 0 else -1.0
    cv = float(pts[:, ca].max()) if sgn > 0 else float(pts[:, ca].min())

    sa, ta = _cabinet_face_axes(ca)
    so = max(FACE_SO, float(spacing) * 0.8)
    wps = _zigzag(sa, (float(pts[:, sa].min()), float(pts[:, sa].max())),
                  ta, (float(pts[:, ta].min()), float(pts[:, ta].max())),
                  ca, cv, n, so)
    name = region.get('part_name') or '柜子面板'
    return _paths_to_output({name: wps})


def generate_cabinet_path(points, spacing=0.06, spray_distance=0.15):
    """柜子整件路径: 顶面 + 前面 + 后面 + 左面 + 右面 (对齐 guizi_all_adaptive).

    柜子当作一个长方体盒子整体喷涂 (分割太抽象, 不按部件拆), 五个外表面全部
    由实际点云包围盒推导 —— 同结构不同尺寸的柜子也能正确挂上路径.
    竖直轴 = 全局 Y (Y-up 场景, 与 guizi.obj 的 OBJ 约定一致); 宽=X, 深=Z.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts is None or len(pts) < 5:
        return None

    top_so = max(FACE_SO, float(spacing) * 0.8)
    face_so = max(FACE_SO, float(spacing) * 0.8)
    paths = _box_faces(pts, '柜子', top_so, face_so, include_sides=True)
    return _paths_to_output(paths)
