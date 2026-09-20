#!/usr/bin/env python3
"""
裁剪框预览 (按物体设置过滤参数)
1. DBSCAN取最大簇
2. 簇内RANSAC去转台平面
3. Y方向过滤 (椅子用, 桌子/柜子不用)
4. X方向左右过滤
黄色=物体 灰色=剔除 Q=退出
"""
import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# 每个物体的裁剪框 + 过滤参数
OBJECT_CONFIG = {
    "chair": {
        "x_min": -0.12, "x_max": 0.12, "y_min": -0.12, "y_max": 0.12,
        "z_min": 0.18, "z_max": 0.40,
        "y_percentile": 30,   # 椅子: 保留上方40%, 去掉下方转台残留
        "x_low": 10, "x_high": 90,
    },
    "table": {
        "x_min": -0.25, "x_max": 0.25, "y_min": -0.25, "y_max": 0.25,
        "z_min": 0.15, "z_max": 0.50,
        "y_percentile": 45,   # 桌子: 几乎不过滤Y, 保留桌腿
        "x_low": 15, "x_high": 90,
    },
    "cabinet": {
        "x_min": -0.25, "x_max": 0.25, "y_min": -0.30, "y_max": 0.30,
        "z_min": 0.15, "z_max": 0.55,
        "y_percentile": 45,   # 柜子: 几乎不过滤Y, 保留整体
        "x_low": 10, "x_high": 90,
    },
}

W, H = 640, 480
FPS = 15
DOWNSAMPLE = 0.002
DBSCAN_EPS = 0.015
DBSCAN_MIN = 10
PLANE_DIST = 0.005
PLANE_RATIO = 0.3
INTERVAL = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=["chair", "table", "cabinet"])
    args = parser.parse_args()
    cfg = OBJECT_CONFIG[args.name]

    print(f"\n{args.name} 裁剪框 X[{cfg['x_min']*100:.0f},{cfg['x_max']*100:.0f}] "
          f"Y[{cfg['y_min']*100:.0f},{cfg['y_max']*100:.0f}] Z[{cfg['z_min']*100:.0f},{cfg['z_max']*100:.0f}]cm")
    print(f"过滤: Y<{cfg['y_percentile']}分位, X在{cfg['x_low']}~{cfg['x_high']}分位")
    print(f"黄=物体 灰=剔除 Q=退出\n")

    pipe = rs.pipeline()
    c = rs.config()
    c.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    c.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(c)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    ds = profile.get_device().first_depth_sensor().get_depth_scale()

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    frame_idx = 0
    cached_keep = np.zeros((H, W), dtype=bool)
    cached_info = (0, 0, 0, 0, 0)

    try:
        while True:
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            df = aligned.get_depth_frame()
            cf = aligned.get_color_frame()
            if not df or not cf:
                continue

            color = np.asanyarray(cf.get_data()).copy()
            depth = np.asanyarray(df.get_data()).astype(np.float32) * ds
            Z = depth
            X = (u_grid - ppx) * Z / fx
            Y = (v_grid - ppy) * Z / fy

            crop = (Z > cfg["z_min"]) & (Z < cfg["z_max"]) & \
                   (X > cfg["x_min"]) & (X < cfg["x_max"]) & \
                   (Y > cfg["y_min"]) & (Y < cfg["y_max"]) & (Z > 0)

            if frame_idx % INTERVAL == 0:
                pts = np.stack([X[crop], Y[crop], Z[crop]], axis=1)
                pix = np.stack([u_grid[crop], v_grid[crop]], axis=1)

                keep_mask = np.zeros(len(pts), dtype=bool)
                n_clusters = 0
                n_cluster = 0
                n_plane = 0
                n_after_y = 0
                n_keep = 0

                if len(pts) > 100:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd = pcd.voxel_down_sample(DOWNSAMPLE)
                    down_pts = np.asarray(pcd.points)

                    if len(down_pts) > DBSCAN_MIN:
                        # 1. DBSCAN取最大簇
                        labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN))
                        valid = labels[labels != -1]
                        if len(valid) > 0:
                            unique, counts = np.unique(valid, return_counts=True)
                            n_clusters = len(unique)
                            largest_label = unique[np.argmax(counts)]
                            cluster_pts = down_pts[labels == largest_label]
                            n_cluster = len(cluster_pts)

                            # 2. 簇内RANSAC去平面
                            cluster_pcd = o3d.geometry.PointCloud()
                            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
                            plane_model, inliers = cluster_pcd.segment_plane(
                                distance_threshold=PLANE_DIST, ransac_n=3, num_iterations=100)

                            if len(inliers) / n_cluster > PLANE_RATIO:
                                n_plane = len(inliers)
                                obj_pts = cluster_pts[~np.isin(np.arange(n_cluster), inliers)]
                            else:
                                obj_pts = cluster_pts

                            # 3. Y方向过滤 (按物体参数)
                            if len(obj_pts) > 20 and cfg["y_percentile"] < 100:
                                y_thresh = np.percentile(obj_pts[:, 1], cfg["y_percentile"])
                                obj_pts = obj_pts[obj_pts[:, 1] < y_thresh]
                                n_after_y = len(obj_pts)
                            else:
                                n_after_y = len(obj_pts)

                            # 4. X方向左右过滤
                            if len(obj_pts) > 20:
                                x_low = np.percentile(obj_pts[:, 0], cfg["x_low"])
                                x_high = np.percentile(obj_pts[:, 0], cfg["x_high"])
                                obj_pts = obj_pts[
                                    (obj_pts[:, 0] > x_low) & (obj_pts[:, 0] < x_high)]

                            n_keep = len(obj_pts)

                            if n_keep > 0:
                                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(obj_pts)
                                dist, _ = nbrs.kneighbors(pts)
                                keep_mask = (dist[:, 0] < DOWNSAMPLE * 1.5)

                keep_full = np.zeros((H, W), dtype=bool)
                if keep_mask.any():
                    kp = pix[keep_mask]
                    keep_full[kp[:, 1], kp[:, 0]] = True
                cached_keep = keep_full
                cached_info = (n_clusters, n_cluster, n_plane, n_after_y, n_keep)

            other = crop & ~cached_keep
            color[other] = (color[other] * 0.4 + np.array([128, 128, 128]) * 0.6).astype(np.uint8)
            color[cached_keep] = (color[cached_keep] * 0.4 + np.array([0, 255, 255]) * 0.6).astype(np.uint8)

            n_clust, n_clu, n_pln, n_ay, n_kp = cached_info
            info = f"{args.name} 簇{n_clust} 最大簇{n_clu} 平面{n_pln} Y后{n_ay} X后{n_kp} Q退出"
            cv2.putText(color, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(f"preview {args.name}", color)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
