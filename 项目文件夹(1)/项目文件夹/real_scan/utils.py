"""点云工具函数 — 从仿真复用, 不做修改."""

import math
import numpy as np
import open3d as o3d
import os


def world_rotate_z(pts_world, angle_deg):
    """绕世界Z轴旋转."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (Rz @ pts_world.T).T


def world_rotate_z_around(pts_world, angle_deg, center_xy):
    """绕指定XY中心旋转Z轴."""
    pts = pts_world.copy()
    pts[:, :2] -= center_xy
    pts = world_rotate_z(pts, angle_deg)
    pts[:, :2] += center_xy
    return pts


def save_ply(pts, filepath):
    """导出PLY文件."""
    if len(pts) == 0:
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(filepath, pcd)


def post_process(merged_pts, voxel_size=0.002, outlier_nb=20, outlier_radius=0.01,
                 dbscan_eps=0.025, dbscan_min=80):
    """后处理管线: 降采样 → 去噪 → DBSCAN聚类 → Y-up.

    Returns (N, 3) numpy array in Y-up coordinates.
    """
    if len(merged_pts) == 0:
        return np.zeros((0, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)

    # 降采样
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 半径滤波去噪
    pcd, _ = pcd.remove_radius_outlier(nb_points=outlier_nb, radius=outlier_radius)

    # DBSCAN 聚类, 保留前3大簇 (避免丢失靠背/椅腿等与主体DBSCAN分离的稀疏部件)
    labels = np.array(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min))
    if len(set(labels)) > 1 or -1 not in labels:
        lbl_counts = sorted(
            [(lbl, (labels == lbl).sum()) for lbl in set(labels) if lbl >= 0],
            key=lambda x: x[1], reverse=True)
        if lbl_counts:
            # 保留前3大簇, 舍弃小噪点簇
            top_labels = {lbl for lbl, cnt in lbl_counts[:3] if cnt >= dbscan_min}
            keep_idx = np.where([l in top_labels for l in labels])[0]
            if len(keep_idx) > 0:
                pcd = pcd.select_by_index(keep_idx)

    # Z-up → Y-up
    R_yup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    pts_final = (R_yup @ np.asarray(pcd.points).T).T
    pts_final -= pts_final.mean(axis=0)

    return pts_final
