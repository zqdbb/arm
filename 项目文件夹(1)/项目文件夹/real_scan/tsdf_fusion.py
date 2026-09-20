#!/usr/bin/env python3
"""
TSDF 深度融合: 标准 RGB-D 重建管线.
直接对深度图进行体素融合, 抗噪 + 紧致表面.
"""
import numpy as np
import cv2, json, time
from pathlib import Path
from rembg import remove, new_session
import open3d as o3d

BASE = Path(__file__).parent
DATA = BASE / 'output/dense_scan'
CALIB = BASE / 'output/calibrate.json'
INTR = DATA / 'camera_intrinsic.json'
OUT = BASE / 'output/tsdf_fusion'
OUT.mkdir(parents=True, exist_ok=True)

VOXEL_SIZE = 0.001  # 1mm
VOL_RADIUS = 0.04   # X/Y 半范围 4cm (椅子 2cm + 深度噪声余量 2cm)
Z_MIN_OBJ = -0.005  # 物体坐标 Z 下界 (world Z↓, 转台面)
Z_MAX_OBJ = 0.095   # 物体坐标 Z 上界 (world Z↓, 椅子最高点)

DEPTH_TRUNC = 0.5   # TSDF 截断距离 (m), > 这个距离的深度不融合


def load_data():
    calib = json.load(open(CALIB))
    R_calib = np.array(calib['R'])
    t_calib = np.array(calib['t'])
    intr = json.load(open(INTR))
    M = intr['intrinsic_matrix']
    fx, fy, ppx, ppy = M[0], M[4], M[6], M[7]
    w, h = intr['width'], intr['height']
    step = intr.get('step_angle', 5)
    n_frames = intr.get('n_frames', 72)
    return R_calib, t_calib, (fx, fy, ppx, ppy), (w, h), step, n_frames


def main():
    R_calib, t_calib, K, (w, h), step, n_frames = load_data()
    fx, fy, ppx, ppy = K

    print('=' * 55)
    print('  TSDF 深度融合')
    print(f'  内参: {fx:.1f} {fy:.1f}  {w}×{h}')
    print(f'  体素: {VOXEL_SIZE*1000:.0f}mm  截断: {DEPTH_TRUNC}m')
    print('=' * 55)

    color_files = sorted((DATA / 'color').glob('*.jpg'))
    depth_files = sorted((DATA / 'depth').glob('*.png'))
    n_frames = min(n_frames, len(color_files), len(depth_files))

    # rembg session
    print('加载 rembg (isnet-general-use)...')
    session = new_session('isnet-general-use')

    # TSDF volume
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=VOXEL_SIZE * 5,  # 5 倍体素 = 5mm 截断距离
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # 内参
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, ppx, ppy)

    print(f'\n融合: {n_frames} 帧')
    t0 = time.time()
    integrated = 0

    for i in range(n_frames):
        img = cv2.imread(str(color_files[i]))
        depth_raw = cv2.imread(str(depth_files[i]), -1).astype(np.float32) / 1000.0

        # rembg mask → 过滤深度
        result = remove(img, session=session, only_mask=False)
        alpha = result[:, :, 3]
        fg_mask = alpha > 128

        # 深度有效 + 在椅子深度范围内 + 前景
        valid_depth = (depth_raw > 0.01) & (depth_raw < DEPTH_TRUNC)

        # 对深度做中值滤波 (减少飞点)
        depth = cv2.medianBlur(depth_raw, 5)

        keep = valid_depth & fg_mask

        if keep.sum() < 200:
            continue

        # 创建 masked depth (非椅子区域设为 0)
        depth_masked = depth.copy()
        depth_masked[~keep] = 0.0

        # RGB (Open3D 需要 RGB 格式)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_masked = rgb.copy()
        rgb_masked[~keep] = 0

        depth_o3d = o3d.geometry.Image(depth_masked)
        rgb_o3d = o3d.geometry.Image(rgb_masked)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False)

        # 相机外参: object frame @ θ=0 → camera
        # 标定: p_world = R_calib @ p_cam + t_calib  (camera → world)
        # 逆:   p_cam = R_calib^T @ p_world - R_calib^T @ t_calib
        # 物体旋转: p_world = R_z(+θ) @ p_obj
        # 所以: p_cam = R_calib^T @ R_z(+θ) @ p_obj - R_calib^T @ t_calib
        theta = np.radians(i * step)
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        R_obj2cam = R_calib.T @ R_z
        t_obj2cam = -R_calib.T @ t_calib

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_obj2cam
        extrinsic[:3, 3] = t_obj2cam

        vol.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1

        if (i + 1) % 18 == 0:
            print(f'  [{i+1}/{n_frames}] {time.time()-t0:.0f}s')

    print(f'  融合完成: {integrated} 帧  {time.time()-t0:.0f}s')

    # 提取 mesh
    print('\n提取 mesh...')
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    print(f'  原始: {len(verts):,} 顶点, {len(mesh.triangles):,} 面')
    if len(verts) > 0:
        print(f'  X: [{verts[:,0].min():.3f},{verts[:,0].max():.3f}]  '
              f'Y: [{verts[:,1].min():.3f},{verts[:,1].max():.3f}]  '
              f'Z: [{verts[:,2].min():.3f},{verts[:,2].max():.3f}]')

    # Z 翻转 (world Z↓ → visual Z↑)
    verts = np.asarray(mesh.vertices)
    verts[:, 2] = -verts[:, 2]
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # 移除孤立面片
    labels, counts, _ = mesh.cluster_connected_triangles()
    if len(counts) > 1:
        largest = np.argmax(counts)
        to_remove = np.where(labels != largest)[0]
        mesh.remove_triangles_by_index(to_remove)
    mesh.remove_unreferenced_vertices()

    # 圆柱裁切: 半径 4cm
    verts = np.asarray(mesh.vertices)
    dist_xy = np.sqrt(verts[:, 0]**2 + verts[:, 1]**2)
    keep = (dist_xy <= 0.04) & (verts[:, 2] >= 0.001)
    if keep.sum() > 0:
        indices = np.where(keep)[0]
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(verts[indices])
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

    # 保存
    o3d.io.write_triangle_mesh(str(OUT / 'tsdf.ply'), mesh)
    o3d.io.write_triangle_mesh(str(OUT / 'tsdf.obj'), mesh)

    verts = np.asarray(mesh.vertices)
    print(f'\n{"="*55}')
    print(f'  输出: {OUT}/tsdf.ply')
    print(f'  顶点: {len(verts):,}  面: {len(mesh.triangles):,}')
    if len(verts) > 0:
        dx = verts[:, 0].max() - verts[:, 0].min()
        dy = verts[:, 1].max() - verts[:, 1].min()
        dz = verts[:, 2].max() - verts[:, 2].min()
        print(f'  尺寸 X: {dx*100:.1f}cm  Y: {dy*100:.1f}cm  Z: {dz*100:.1f}cm')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
