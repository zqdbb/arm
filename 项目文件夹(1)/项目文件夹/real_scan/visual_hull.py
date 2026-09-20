#!/usr/bin/env python3
"""
Shape from Silhouette — Visual Hull 体素雕刻 + Marching Cubes 网格.
基于深度剪影 (不是深度值), 不依赖 IR 深度精度.

输入: 密集采集的 RGB + 深度图 (5° × 72 帧) + calibrate.json
输出: 三角网格 PLY + OBJ

原理:
  1. 从深度图提取椅子剪影 (深度阈值 + 形态学)
  2. 物体坐标系建体素网格
  3. 体素投影到各视角, 被所有剪影包含 → 保留
  4. Marching Cubes → 三角网格
"""
import numpy as np
import cv2, json, time
from pathlib import Path
from skimage import measure
from rembg import remove, new_session
import open3d as o3d

BASE = Path(__file__).parent
DATA = BASE / 'output/dense_scan'          # 5° × 72 帧
CALIB = BASE / 'output/calibrate.json'
INTR = DATA / 'camera_intrinsic.json'
OUT = BASE / 'output/visual_hull'

# ── 体素参数 ──
VOXEL_SIZE = 0.001          # 1mm 体素
VOL_RADIUS = 0.05           # X/Y 半范围 5cm (匹配椅子尺寸+偏移余量)
Z_MIN = -0.095              # 物体 Z 下界 (world Z↓, 靠背最高点)
Z_MAX = 0.005               # 物体 Z 上界 (world Z↓, 转台面)

# ── 剪影参数 ──
PLANE_MARGIN = 0.005        # 高于转台面多少算椅子 (m)
MORPH_KERNEL = 7            # 形态学闭运算核


def load_data():
    calib = json.load(open(CALIB))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])

    intr = json.load(open(INTR))
    M = intr['intrinsic_matrix']
    fx, fy = M[0], M[4]
    ppx, ppy = M[6], M[7]
    w, h = intr['width'], intr['height']
    step = intr.get('step_angle', 5)
    n_frames = intr.get('n_frames', 72)
    return R_calib, t_calib, (fx, fy, ppx, ppy), (w, h), step, n_frames


def extract_silhouettes(n_frames, w, h, R_calib, t_calib, K, step):
    """rembg (isnet-general-use) 深度学习抠图 → 椅子剪影."""
    color_dir = DATA / 'color'
    color_files = sorted(color_dir.glob('*.jpg'))
    actual = len(color_files)
    if actual < n_frames:
        n_frames = actual

    masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 膨胀核: 5px 确保靠背全覆盖
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    print(f'提取剪影: {n_frames} 帧 (rembg isnet-general-use + dilate 5px)')
    session = new_session('isnet-general-use')
    t0 = time.time()

    for i in range(n_frames):
        img = cv2.imread(str(color_files[i]))

        # rembg 抠图 (RGBA → alpha通道 = mask)
        result = remove(img, session=session, only_mask=False)
        alpha = result[:, :, 3]
        fg = (alpha > 128).astype(np.uint8) * 255

        # 膨胀 5px → 确保椅子边缘/靠背全覆盖
        fg = cv2.dilate(fg, dilate_kernel)

        # 形态学闭运算
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        # 保留最大连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            fg = (labels == largest).astype(np.uint8) * 255

        masks.append(fg)

        if (i + 1) % 18 == 0:
            elapsed = time.time() - t0
            fg_pct = (fg > 0).sum() / fg.size * 100
            print(f'  [{i+1}/{n_frames}] 前景 {fg_pct:.1f}%  {elapsed:.0f}s')

    return masks


def build_volume():
    """建体素网格 (物体坐标系: aligned frame @ θ=0, world Z↓)."""
    nx = int(2 * VOL_RADIUS / VOXEL_SIZE) + 1
    ny = int(2 * VOL_RADIUS / VOXEL_SIZE) + 1
    nz = int((Z_MAX - Z_MIN) / VOXEL_SIZE) + 1

    x = np.linspace(-VOL_RADIUS, VOL_RADIUS, nx, dtype=np.float32)
    y = np.linspace(-VOL_RADIUS, VOL_RADIUS, ny, dtype=np.float32)
    z = np.linspace(Z_MIN, Z_MAX, nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    print(f'体素网格: {nx}×{ny}×{nz} = {len(voxels)/1e6:.2f}M @ {VOXEL_SIZE*1000:.0f}mm')
    print(f'  X: [{x[0]:.3f},{x[-1]:.3f}] Y: [{y[0]:.3f},{y[-1]:.3f}] Z: [{z[0]:.3f},{z[-1]:.3f}]')
    return voxels, (nx, ny, nz)


def carve(voxels, shape, masks, R_calib, t_calib, K, w, h, step):
    """Visual Hull: 逐视角雕刻体素."""
    fx, fy, ppx, ppy = K
    n_frames = len(masks)
    n_voxels = len(voxels)

    # 每帧的外参 (虚拟相机绕固定物体)
    # R_v = R_calib^T @ R_z(+θ),  t_v = -R_calib^T @ t_calib (常数)
    t_v = (-R_calib.T @ t_calib).astype(np.float32)

    occupied = np.ones(n_voxels, dtype=bool)
    BATCH = 80000

    print(f'\n体素雕刻: {n_frames} 视角 × {n_voxels/1e6:.2f}M 体素')
    t0 = time.time()

    for i_frame, mask in enumerate(masks):
        theta = np.radians(i_frame * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        R_v = (R_calib.T @ R_z).astype(np.float32)

        surviving = 0
        for start in range(0, n_voxels, BATCH):
            end = min(start + BATCH, n_voxels)
            idx_batch = np.where(occupied[start:end])[0]
            if len(idx_batch) == 0:
                continue

            global_idx = start + idx_batch
            batch = voxels[global_idx]

            # 物体坐标 → 相机坐标
            p_cam = (R_v @ batch.T).T + t_v
            z_cam = p_cam[:, 2]
            in_front = z_cam > 0.05
            if not in_front.any():
                occupied[global_idx] = False
                continue

            # 投影到图像
            idx = np.where(in_front)[0]
            u = (fx * p_cam[idx, 0] / z_cam[idx] + ppx).astype(int)
            v = (fy * p_cam[idx, 1] / z_cam[idx] + ppy).astype(int)
            in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not in_img.any():
                occupied[global_idx] = False
                continue

            idx = idx[in_img]
            u, v = u[in_img], v[in_img]

            # 检查是否在剪影内
            in_mask = mask[v, u] > 0
            idx_inside = idx[in_mask]

            # 更新占据状态
            keep = np.zeros(len(batch), dtype=bool)
            keep[idx_inside] = True
            occupied[global_idx] = keep
            surviving += len(idx_inside)

        n_survive = occupied.sum()
        if (i_frame + 1) % 12 == 0:
            elapsed = time.time() - t0
            print(f'  [{i_frame+1:2d}/{n_frames}] {i_frame*step:3d}°  剩余 {n_survive:,} '
                  f'({n_survive/n_voxels*100:.1f}%)  {elapsed:.0f}s')

    print(f'  完成: {occupied.sum():,} 体素  ({time.time()-t0:.0f}s)')
    return occupied


def to_mesh(voxels, occupied, shape):
    """Marching Cubes → 三角网格."""
    nx, ny, nz = shape
    grid = occupied.reshape(nx, ny, nz).astype(np.float32)

    verts, faces, _, _ = measure.marching_cubes(
        grid, level=0.5, spacing=(1, 1, 1))  # 体素索引

    # 体素索引 → 物体坐标系 (world Z↓)
    verts[:, 0] = -VOL_RADIUS + verts[:, 0] * VOXEL_SIZE
    verts[:, 1] = -VOL_RADIUS + verts[:, 1] * VOXEL_SIZE
    verts[:, 2] = Z_MIN + verts[:, 2] * VOXEL_SIZE

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def postprocess(mesh):
    """清理孤立面片 + Z翻转 + 圆柱裁切."""
    # 移除孤立面片
    labels, counts, _ = mesh.cluster_connected_triangles()
    if len(counts) > 1:
        largest = np.argmax(counts)
        to_remove = np.where(labels != largest)[0]
        mesh.remove_triangles_by_index(to_remove)
    mesh.remove_unreferenced_vertices()

    # Z 翻转 (world Z↓ → 视觉 Z↑)
    verts = np.asarray(mesh.vertices)
    verts[:, 2] = -verts[:, 2]
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # 圆柱裁切 (只保留转台中心区域)
    verts = np.asarray(mesh.vertices)
    dist_xy = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    keep = (dist_xy <= 0.06) & (verts[:, 2] >= 0.001)
    if keep.sum() > 0:
        indices = np.where(keep)[0]
        # 构建新 mesh (select_by_index 不能处理重复索引)
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(verts[indices])

        # 重新映射面
        old_to_new = np.full(len(verts), -1, dtype=int)
        old_to_new[indices] = np.arange(len(indices))
        faces = np.asarray(mesh.triangles)
        face_mask = (old_to_new[faces[:, 0]] >= 0) & \
                    (old_to_new[faces[:, 1]] >= 0) & \
                    (old_to_new[faces[:, 2]] >= 0)
        new_faces = np.column_stack([
            old_to_new[faces[face_mask, 0]],
            old_to_new[faces[face_mask, 1]],
            old_to_new[faces[face_mask, 2]],
        ])
        new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)
        new_mesh.compute_vertex_normals()
        mesh = new_mesh

    return mesh


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print('=' * 55)
    print('  Visual Hull — Shape from Silhouette')
    print('=' * 55)

    R_calib, t_calib, K, (w, h), step, n_frames = load_data()
    fx, fy, ppx, ppy = K
    print(f'内参: fx={fx:.1f} fy={fy:.1f}  {w}×{h} 步长={step}° 帧数={n_frames}')
    cam_pos = -R_calib.T @ t_calib
    print(f'相机 world 位置: [{cam_pos[0]:.3f} {cam_pos[1]:.3f} {cam_pos[2]:.3f}]')

    # 1. 提取剪影
    masks = extract_silhouettes(n_frames, w, h, R_calib, t_calib, K, step)
    n_frames = len(masks)

    # 2. 建体素网格
    voxels, shape = build_volume()

    # 3. Visual Hull 雕刻
    occupied = carve(voxels, shape, masks, R_calib, t_calib, K, w, h, step)

    n_pts = occupied.sum()
    if n_pts < 50:
        print(f'\n保留体素太少 ({n_pts})! 检查:')
        print(f'  1. 椅子是否在体素范围内?')
        print(f'  2. 剪影是否正确? 查看 {OUT}/masks/')
        # 保存剪影样本供调试
        debug_dir = OUT / 'masks'
        debug_dir.mkdir(exist_ok=True)
        for i in [0, 18, 36, 54]:
            cv2.imwrite(str(debug_dir / f'mask_{i:03d}.png'), masks[i])
        print(f'  调试输出: {OUT}/masks/mask_*.png')
        return

    # 4. Marching Cubes → 网格
    print(f'\nMarching Cubes...')
    mesh = to_mesh(voxels, occupied, shape)
    print(f'  原始网格: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 面')

    # 5. 后处理
    mesh = postprocess(mesh)

    # 6. 缩放校正 (Visual Hull 通常偏大)
    # 不做缩放, 先看原始结果

    # 保存
    out_ply = str(OUT / 'visual_hull.ply')
    out_obj = str(OUT / 'visual_hull.obj')
    o3d.io.write_triangle_mesh(out_ply, mesh)
    o3d.io.write_triangle_mesh(out_obj, mesh)

    verts = np.asarray(mesh.vertices)
    print(f'\n{"="*55}')
    print(f'  输出: {out_ply}')
    print(f'  顶点: {len(verts):,}  面: {len(mesh.triangles):,}')
    if len(verts) > 0:
        dx = verts[:, 0].max() - verts[:, 0].min()
        dy = verts[:, 1].max() - verts[:, 1].min()
        dz = verts[:, 2].max() - verts[:, 2].min()
        print(f'  尺寸 X: {dx*100:.1f}cm  Y: {dy*100:.1f}cm  Z: {dz*100:.1f}cm  (预期 4×4×8.8cm)')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
