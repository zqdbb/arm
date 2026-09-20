#!/usr/bin/env python3
"""
高清 3D 椅子点云拼接管线（全局 Pose Graph 闭环图优化版）
原理：
  1. 建立全局姿态图 (Pose Graph)。
  2. 计算相邻帧之间的 ICP 匹配（构建边 Constraint）。
  3. 增加第 N 帧与第 0 帧的闭环匹配（Loop Closure constraint），形成 360 度闭环。
  4. 运行 Open3D 姿态图全局优化，彻底平摊累积误差，消除螺线与重影。
"""

import time
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from pathlib import Path

try:
    from turntable import TurntableController
except ImportError:
    TurntableController = None

# ─── 全局参数配置 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output/v12/chair/pcd_frames"
FINAL_PLY_PATH = BASE_DIR / "output/v12/chair/chair_stitched.ply"

# 相机与采集参数
W, H = 640, 480       
FPS = 15
STEP_DEG = 5                    # 单步旋转角度 (72帧)
N_FRAMES = 360 // STEP_DEG      
N_AVG = 5                       
LASER_POWER = 300               

# 空间裁剪框 (相机坐标系：X 左右，Y 上下，Z 前后/深度)
X_MIN, X_MAX = -0.20, 0.20      # 左右 ±20cm
Y_MIN, Y_MAX = -0.20, 0.20      # 上下 ±20cm
Z_MIN, Z_MAX = 0.15, 0.45      # 深度 15cm ~ 45cm


def setup_filters():
    """配置 RealSense 硬件滤波链"""
    filters = []
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    filters.append(spatial)
    
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    filters.append(temporal)
    
    hole_filling = rs.hole_filling_filter(1)
    filters.append(hole_filling)
    return filters


def safe_crop_pcd(pcd):
    """纯空间框安全裁剪"""
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(X_MIN, Y_MIN, Z_MIN),
        max_bound=(X_MAX, Y_MAX, Z_MAX)
    )
    return pcd.crop(bbox)


def preview_camera_view(pipe, align, intrinsics):
    """实时视角确认"""
    print("\n" + "=" * 60)
    print("【视角确认】开启实时画面...")
    print(" 按【空格键 (Space)】开始采集，按【Q】退出。")
    print("=" * 60 + "\n")

    colorizer = rs.colorizer()

    while True:
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        cf = aligned.get_color_frame()
        if not df or not cf:
            continue

        color_img = np.asanyarray(cf.get_data())
        depth_frame_colorized = np.asanyarray(colorizer.colorize(df).get_data())

        depth_scale = 0.001
        depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale
        
        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_m)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=Z_MAX, convert_rgb_to_intensity=False
        )
        pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_intrinsics)
        chair_pcd = safe_crop_pcd(pcd)

        overlay = color_img.copy()
        cv2.putText(overlay, f"Points: {len(chair_pcd.points)} | Press [SPACE] to Start", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        combined = np.hstack((overlay, depth_frame_colorized))
        cv2.imshow("Preview (Left: Color | Right: Depth)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)


def capture_clean_pcd():
    """单帧点云自动采集"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)

    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)

    align = rs.align(rs.stream.color)
    filters = setup_filters()

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )

    preview_camera_view(pipe, align, intr)

    if TurntableController is None:
        print("错误: 未找到 turntable.py，无法驱动转台！")
        pipe.stop()
        return

    tt = TurntableController(port='/dev/ttyUSB0')
    try:
        tt.open()
        tt.move_absolute(0, speed=10000)
        tt.wait_stop(timeout=30)
    except Exception as e:
        print(f"转台连接错误: {e}")
        pipe.stop()
        return

    t0 = time.time()
    print(f"\n开始采集 {N_FRAMES} 帧点云...")

    for i in range(N_FRAMES):
        deg = i * STEP_DEG
        tt.move_absolute(deg, speed=5000)
        tt.wait_stop(timeout=30)

        depth_frames_list = []
        color_img = None

        for _ in range(N_AVG):
            frames = pipe.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            df = aligned.get_depth_frame()
            cf = aligned.get_color_frame()
            if not df or not cf:
                continue

            for f in filters:
                df = f.process(df)

            d_data = np.asanyarray(df.get_data()).astype(np.float32)
            depth_frames_list.append(d_data)
            if color_img is None:
                color_img = np.asanyarray(cf.get_data())

        if not depth_frames_list:
            continue

        depth_stack = np.stack(depth_frames_list, axis=0)
        valid_mask = depth_stack > 0
        valid_counts = np.sum(valid_mask, axis=0)

        depth_sum = np.sum(depth_stack, axis=0)
        mean_depth = np.zeros_like(depth_sum)
        mean_depth[valid_counts > 0] = depth_sum[valid_counts > 0] / valid_counts[valid_counts > 0]

        depth_scale = ds.get_depth_scale()
        depth_m = (mean_depth * depth_scale).astype(np.float32)

        o3d_color = o3d.geometry.Image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        o3d_depth = o3d.geometry.Image(depth_m)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1.0, depth_trunc=Z_MAX, convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_intrinsics)
        pcd = safe_crop_pcd(pcd)

        save_path = OUT_DIR / f"frame_{i:03d}.ply"
        o3d.io.write_point_cloud(str(save_path), pcd)

        print(f"  [{i+1:02d}/{N_FRAMES}] {deg:3d}° 保存成功 | 点数: {len(pcd.points):5d}")

    pipe.stop()
    tt.close()
    print(f"\n数据采集完毕，耗时 {time.time()-t0:.1f} 秒！")


def voxel_down_sample_compat(pcd, voxel_size):
    if hasattr(pcd, "voxel_down_sample"):
        return pcd.voxel_down_sample(voxel_size=voxel_size)
    elif hasattr(pcd, "voxel_downsample"):
        return pcd.voxel_downsample(voxel_size=voxel_size)
    return pcd


def remove_outlier_compat(pcd, nb_neighbors, std_ratio):
    if hasattr(pcd, "remove_statistical_outlier"):
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd_clean
    elif hasattr(pcd, "remove_statistical_outliers"):
        pcd_clean, _ = pcd.remove_statistical_outliers(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd_clean
    return pcd


def pair_icp(source, target, max_dist=0.015, init_trans=np.identity(4)):
    """计算两帧之间的 ICP 配准，并返回变换矩阵与信息矩阵"""
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_dist,
        init=init_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    info_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_dist, reg_p2p.transformation
    )
    return reg_p2p.transformation, info_matrix


def stitch_point_clouds():
    """采用姿态图 (Pose Graph) + 闭环优化 (Loop Closure) 的 ICP 全局拼接"""
    ply_files = sorted(list(OUT_DIR.glob("frame_*.ply")))
    if not ply_files:
        print(f"\n错误: 未找到 PLY 数据！")
        return

    print(f"\n" + "=" * 60)
    print(f"【姿态图全局优化】读取到 {len(ply_files)} 帧点云，构建 Pose Graph...")
    print("=" * 60)

    raw_pcds = []
    for file_path in ply_files:
        pcd = o3d.io.read_point_cloud(str(file_path))
        pcd = safe_crop_pcd(pcd)
        if len(pcd.points) > 50:
            pcd = voxel_down_sample_compat(pcd, voxel_size=0.002)
            pcd.estimate_normals()
            raw_pcds.append(pcd)

    n_pcds = len(raw_pcds)
    if n_pcds < 2:
        print("有效点云不足，请重新运行并输入 n 进行采集！")
        return

    # 1. 建立 5 度绝对旋转的初始猜想矩阵 Guess
    rad = np.radians(-STEP_DEG)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    T_step_init = np.array([
        [cos_a,  0, sin_a, 0],
        [0,      1,     0, 0],
        [-sin_a, 0, cos_a, 0],
        [0,      0,     0, 1]
    ])

    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    print("正在构建边约束 (Edges) 与 ICP 微调...")
    
    # 2. 依次计算相邻帧之间的 ICP 约束 (Odometry Edges)
    for i in range(n_pcds - 1):
        source = raw_pcds[i + 1]
        target = raw_pcds[i]

        transformation, info_matrix = pair_icp(source, target, max_dist=0.015, init_trans=T_step_init)

        # 更新相对积累位姿并放入节点
        odometry = odometry @ transformation
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))

        # 添加非闭环边 (uncertain=False)
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i + 1, i, transformation, info_matrix, uncertain=False
            )
        )

    # 3. 添加闭环约束 (Loop Closure Edge: 最后一帧 -> 第一帧)
    print("正在计算首尾闭环约束 (Loop Closure)...")
    last_idx = n_pcds - 1
    first_idx = 0
    
    # 估计最后一帧到第一帧的相对初始变换
    loop_init = np.linalg.inv(odometry)
    transformation_loop, info_matrix_loop = pair_icp(
        raw_pcds[last_idx], raw_pcds[first_idx], max_dist=0.03, init_trans=loop_init
    )
    
    # 添加闭环边 (uncertain=True 表示闭环边，求解器会重点优化拉平误差)
    pose_graph.edges.append(
        o3d.pipelines.registration.PoseGraphEdge(
            last_idx, first_idx, transformation_loop, info_matrix_loop, uncertain=True
        )
    )

    # 4. 执行 Levenberg-Marquardt 全局姿态图优化
    print("正在进行 Levenberg-Marquardt 全局姿态图优化，平摊累积误差...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.02,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    criterion = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        criterion,
        option
    )

    # 5. 将优化后的全局位姿映射并融合到点云中
    merged_pcd = o3d.geometry.PointCloud()
    for i in range(n_pcds):
        p_temp = o3d.geometry.PointCloud(raw_pcds[i])
        p_temp.transform(pose_graph.nodes[i].pose)
        merged_pcd += p_temp

    # 全局降采样与去噪
    print("\n正在对优化后的模型进行滤波与去噪...")
    merged_pcd = voxel_down_sample_compat(merged_pcd, voxel_size=0.0015)
    merged_pcd = remove_outlier_compat(merged_pcd, nb_neighbors=20, std_ratio=2.0)

    # 翻转 180 度矫正视角
    center = merged_pcd.get_center()
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    merged_pcd.rotate(R_flip, center=center)

    FINAL_PLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(FINAL_PLY_PATH), merged_pcd)

    print(f"\n拼接成功！高清 3D 模型已保存至:\n  -> {FINAL_PLY_PATH}")
    print(f"拼接点云总点数: {len(merged_pcd.points)}")

    o3d.visualization.draw_geometries(
        [merged_pcd],
        window_name="3D Chair Point Cloud (Pose Graph Global Optimized)",
        width=1024, height=768
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    user_choice = input("是否跳过数据采集，直接使用已有数据进行拼接？(y/n): ").strip().lower()
    print("=" * 60)

    if user_choice == 'y':
        stitch_point_clouds()
    else:
        capture_clean_pcd()
        stitch_point_clouds()