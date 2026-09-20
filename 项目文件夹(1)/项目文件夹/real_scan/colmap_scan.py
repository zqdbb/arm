#!/usr/bin/env python3
"""COLMAP 多视几何重建管线: D435i RGB → SfM + MVS → 稠密点云.

用法:
  python3 colmap_scan.py                    # 步进拍照 + COLMAP 重建 (默认)
  python3 colmap_scan.py --video            # 连续旋转录视频 + ffmpeg 抽帧 + COLMAP
  python3 colmap_scan.py --skip-capture     # 跳过采集, 直接用已有图片跑 COLMAP

输出: output/colmap_scan/scan_colmap.ply (稠密点云, COLMAP 原始坐标系)

流程:
  采集 → ffmpeg抽帧 (video模式) → SfM稀疏重建 → MVS稠密重建 → PLY导出
"""

import os
import sys
import time
import subprocess
import shutil
import argparse
import threading
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from turntable import TurntableController


# ── COLMAP 输出目录 ──
COLMAP_DIR = os.path.join(OUTPUT_DIR, 'colmap_scan')
IMAGES_DIR = os.path.join(COLMAP_DIR, 'images')
SPARSE_DIR = os.path.join(COLMAP_DIR, 'sparse')
DENSE_DIR = os.path.join(COLMAP_DIR, 'dense')
DB_PATH = os.path.join(COLMAP_DIR, 'database.db')
FINAL_PLY = os.path.join(COLMAP_DIR, 'scan_colmap.ply')

# ── 采集参数 ──
STEP_ANGLE = 10           # 步进模式: 每步角度
VIDEO_SPEED = 2000         # 连续模式: 转台速度 Hz (5°/s)
VIDEO_FPS = 30             # 连续模式: 录制帧率
EXTRACT_FPS = 2.0          # 连续模式: 抽帧率 (每秒抽N帧, 相邻~2.5°)

# ── COLMAP 参数 ──
COLMAP_BIN = 'colmap'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def setup_workspace():
    """初始化 COLMAP 工作目录."""
    if os.path.exists(COLMAP_DIR):
        shutil.rmtree(COLMAP_DIR)
    ensure_dir(IMAGES_DIR)
    ensure_dir(SPARSE_DIR)
    ensure_dir(DENSE_DIR)
    print(f'工作区: {COLMAP_DIR}')


# ══════════════════════════════════════════════════════════════════
# Phase 1: 图像采集
# ══════════════════════════════════════════════════════════════════

def mask_background(img_bgr, depth_frame, max_dist_mm=550):
    """用深度图遮罩静态背景: 超过 max_dist_mm 或深度=0 的像素涂黑.

    COLMAP 会提取物体和背景的 SIFT 特征。背景静止、物体随转台旋转，
    矛盾的运动关系会导致 SfM 轨迹破裂。涂黑背景让 COLMAP 只关注物体。
    """
    depth = np.asanyarray(depth_frame.get_data())
    bg_mask = (depth > max_dist_mm) | (depth == 0)
    img_bgr[bg_mask] = [0, 0, 0]
    return img_bgr


def capture_step_mode(pipeline, align, tt):
    """步进拍照: 转台每 N° 停 → 拍 RGB → 保存 PNG."""
    angles = list(range(0, 360, STEP_ANGLE))
    print(f'\n[采集·步进] {len(angles)} 帧, 每{STEP_ANGLE}°')

    cv2.namedWindow('COLMAP Capture - Step', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('COLMAP Capture - Step', 640, 360)

    saved = 0
    for i, target_deg in enumerate(angles):
        print(f'  [{i+1}/{len(angles)}] 旋转到 {target_deg}° ...')
        tt.move_absolute(target_deg, speed=TURNTABLE_DEFAULT_SPEED)
        tt.wait_stop(timeout=30)
        time.sleep(0.5)  # 等相机稳定

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            print(f'    采集失败, 跳过')
            continue

        img = np.asanyarray(color_frame.get_data())
        # 深度遮罩去背景: 涂黑转台以外的静态背景
        if depth_frame:
            img = mask_background(img, depth_frame)
        path = os.path.join(IMAGES_DIR, f'frame_{saved:04d}.png')
        cv2.imwrite(path, img)
        saved += 1

        cv2.putText(img, f'{target_deg} deg  [{saved}/{len(angles)}]', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('COLMAP Capture - Step', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(f'  保存 {saved} 张图片 → {IMAGES_DIR}')


def capture_video_mode(pipeline, align, tt):
    """连续旋转录视频: 转台匀速旋转 → D435i 录 MJPG AVI → ffmpeg 抽帧."""
    import pyrealsense2 as rs

    video_path = os.path.join(COLMAP_DIR, 'turntable.avi')

    # 计算旋转 360° 所需时间
    degrees_per_sec = VIDEO_SPEED / TURNTABLE_PULSES_PER_DEG  # Hz / (pulses/deg)
    duration_360 = 360.0 / degrees_per_sec
    print(f'\n[采集·视频] 转台 {degrees_per_sec:.1f}°/s, '
          f'一圈 {duration_360:.0f}s, 录制 {VIDEO_FPS}fps')

    # 初始化 VideoWriter (MJPG 无损)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (DEPTH_WIDTH, DEPTH_HEIGHT))
    if not writer.isOpened():
        print('  VideoWriter 打开失败, 改用 XVID')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (DEPTH_WIDTH, DEPTH_HEIGHT))
        if not writer.isOpened():
            sys.exit('无法创建视频文件')

    # 开始连续旋转
    print(f'  转台开始旋转...')
    tt.rotate_forward(speed=VIDEO_SPEED)

    # 录制循环
    cv2.namedWindow('COLMAP Capture - Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('COLMAP Capture - Video', 640, 360)

    n_frames = int(duration_360 * VIDEO_FPS)
    t_start = time.time()
    frames_written = 0

    for i in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        if depth_frame:
            img = mask_background(img, depth_frame)
        writer.write(img)
        frames_written += 1

        elapsed = time.time() - t_start
        if i % 10 == 0:
            cv2.putText(img, f'Recording: {elapsed:.1f}s / {duration_360:.0f}s', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('COLMAP Capture - Video', img)
            cv2.waitKey(1)

    writer.release()
    tt.stop()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f'  录制完成: {frames_written} 帧, {elapsed:.1f}s')
    print(f'  视频: {video_path}')

    # ffmpeg 抽帧
    print(f'\n[ffmpeg] fps={EXTRACT_FPS} 抽帧...')
    result = subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vf', f'fps={EXTRACT_FPS}',
        os.path.join(IMAGES_DIR, 'frame_%04d.png'),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ffmpeg 错误:\n{result.stderr}')
        sys.exit(1)

    n_extracted = len([f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
    print(f'  抽帧 {n_extracted} 张 → {IMAGES_DIR}')


# ══════════════════════════════════════════════════════════════════
# Phase 2: COLMAP SfM + MVS 重建
# ══════════════════════════════════════════════════════════════════

def run_colmap(camera_model='SIMPLE_RADIAL', single_camera=True):
    """运行完整 COLMAP 管线.

    Args:
        camera_model: PINHOLE, SIMPLE_RADIAL, SIMPLE_PINHOLE, RADIAL
        single_camera: 所有图片共享同一相机内参
    """
    n_images = len([f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
    if n_images < 5:
        sys.exit(f'图片不足: {n_images} 张 (需 >= 5)')

    print(f'\n{"="*55}')
    print(f'  COLMAP 重建: {n_images} 张图片')
    print(f'  相机模型: {camera_model}  |  共享内参: {single_camera}')
    print(f'{"="*55}')

    # ── 1. 特征提取 ──
    print('\n[1/6] 特征提取 (SIFT)...')
    t0 = time.time()
    cmd = [
        COLMAP_BIN, 'feature_extractor',
        '--database_path', DB_PATH,
        '--image_path', IMAGES_DIR,
        '--SiftExtraction.use_gpu', '0',
        '--SiftExtraction.max_image_size', '1200',
    ]
    if single_camera:
        cmd += ['--ImageReader.single_camera', '1']
        cmd += ['--ImageReader.camera_model', camera_model]
    subprocess.run(cmd, check=True)
    t1 = time.time()
    print(f'  耗时: {t1-t0:.0f}s')

    # ── 2. 顺序匹配 (视频帧 → 相邻帧匹配, 快 5-10 倍) ──
    print('\n[2/6] 顺序匹配 (sequential_matcher)...')
    cmd = [
        COLMAP_BIN, 'sequential_matcher',
        '--database_path', DB_PATH,
        '--SequentialMatching.overlap', '5',        # 匹配相邻5帧
        '--SequentialMatching.quadratic_overlap', '0',  # 关闭二次重叠
        '--SiftMatching.use_gpu', '0',
    ]
    subprocess.run(cmd, check=True)
    t2 = time.time()
    print(f'  耗时: {t2-t1:.0f}s')

    # ── 3. 稀疏重建 (SfM) ──
    print('\n[3/6] 稀疏重建 (SfM Mapper)...')
    cmd = [
        COLMAP_BIN, 'mapper',
        '--database_path', DB_PATH,
        '--image_path', IMAGES_DIR,
        '--output_path', SPARSE_DIR,
    ]
    subprocess.run(cmd, check=True)
    t3 = time.time()
    print(f'  耗时: {t3-t2:.0f}s')

    # 检查 SfM 结果
    sparse_models = [d for d in os.listdir(SPARSE_DIR) if d.isdigit()]
    if not sparse_models:
        sys.exit('SfM 失败: 无重建模型输出')
    sparse_model = os.path.join(SPARSE_DIR, sparse_models[0])
    print(f'  SfM 完成: {sparse_model}')

    # ── 4. 去畸变 + 导出 ──
    print('\n[4/6] 去畸变 (image_undistorter)...')
    cmd = [
        COLMAP_BIN, 'image_undistorter',
        '--image_path', IMAGES_DIR,
        '--input_path', sparse_model,
        '--output_path', DENSE_DIR,
        '--output_type', 'COLMAP',
    ]
    subprocess.run(cmd, check=True)
    t4 = time.time()
    print(f'  耗时: {t4-t3:.0f}s')

    # ── 5. 稠密重建 (MVS - PatchMatch Stereo) ──
    print('\n[5/6] 稠密重建 (PatchMatch Stereo, CPU)...')
    cmd = [
        COLMAP_BIN, 'patch_match_stereo',
        '--workspace_path', DENSE_DIR,
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.gpu_index', '-1',
        '--PatchMatchStereo.geom_consistency', '0',     # CPU关几何一致性
        '--PatchMatchStereo.max_image_size', '800',     # 限制分辨率提速
    ]
    subprocess.run(cmd, check=True)
    t5 = time.time()
    print(f'  耗时: {(t5-t4)/60:.0f}min')

    # ── 6. 融合 + 导出 PLY ──
    print('\n[6/6] 点云融合 (Stereo Fusion)...')
    cmd = [
        COLMAP_BIN, 'stereo_fusion',
        '--workspace_path', DENSE_DIR,
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', FINAL_PLY,
    ]
    subprocess.run(cmd, check=True)
    t6 = time.time()

    fsize = os.path.getsize(FINAL_PLY)
    print(f'\n  总耗时: {(t6-t0)/60:.0f}min')
    print(f'  输出: {FINAL_PLY} ({fsize/1024/1024:.1f} MB)')


# ══════════════════════════════════════════════════════════════════
# Phase 3: 后处理 (降采样, 去噪, 保存)
# ══════════════════════════════════════════════════════════════════

def postprocess_colmap():
    """对 COLMAP 输出的 PLY 做基本后处理."""
    import open3d as o3d

    if not os.path.exists(FINAL_PLY):
        print(f'COLMAP 输出不存在: {FINAL_PLY}')
        return

    print(f'\n[后处理] 降采样 + 去噪...')

    pcd = o3d.io.read_point_cloud(FINAL_PLY)
    n_before = len(pcd.points)
    print(f'  原始点数: {n_before:,}')

    if n_before < 1000:
        print('  点数过少, 跳过后处理')
        return

    # 降采样
    pcd = pcd.voxel_down_sample(voxel_size=0.001)  # 1mm
    n_voxel = len(pcd.points)
    print(f'  降采样 (1mm): {n_before:,} → {n_voxel:,}')

    # 统计去噪
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    n_clean = len(pcd.points)
    print(f'  统计去噪: {n_voxel:,} → {n_clean:,}')

    # 保存精简版
    lite_path = FINAL_PLY.replace('.ply', '_lite.ply')
    o3d.io.write_point_cloud(lite_path, pcd)
    print(f'  精简版: {lite_path} ({n_clean:,} 点)')


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import pyrealsense2 as rs

    parser = argparse.ArgumentParser(description='COLMAP 多视几何重建管线')
    parser.add_argument('--video', action='store_true',
                        help='连续旋转录视频模式 (默认: 步进拍照)')
    parser.add_argument('--skip-capture', action='store_true',
                        help='跳过采集, 直接用已有图片跑 COLMAP')
    parser.add_argument('--camera-model', default='RADIAL',
                        choices=['SIMPLE_PINHOLE', 'PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'OPENCV'],
                        help='COLMAP 相机模型 (默认: RADIAL)')
    parser.add_argument('--multi-camera', action='store_true',
                        help='每张图片独立内参 (默认: 共享内参)')
    parser.add_argument('--step-angle', type=float, default=STEP_ANGLE,
                        help=f'步进角度 (默认: {STEP_ANGLE}°)')
    parser.add_argument('--extract-fps', type=float, default=EXTRACT_FPS,
                        help=f'视频抽帧率 (默认: {EXTRACT_FPS})')
    parser.add_argument('--no-post', action='store_true',
                        help='跳过后处理')
    args = parser.parse_args()

    skip_capture = args.skip_capture

    print('=' * 55)
    print('  COLMAP 多视几何重建管线')
    print(f'  相机: D435i RGB (不用深度)')
    print(f'  重建: SfM + MVS (CPU)')
    print('=' * 55)

    if not skip_capture:
        setup_workspace()

        # ── 连接相机 ──
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            sys.exit('未检测到 RealSense 相机')

        dev = devices[0]
        print(f'相机: {dev.get_info(rs.camera_info.name)}')

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT,
                          rs.format.z16, DEPTH_FPS)
        cfg.enable_stream(rs.stream.color, DEPTH_WIDTH, DEPTH_HEIGHT,
                          rs.format.bgr8, DEPTH_FPS)
        profile = pipeline.start(cfg)
        align = rs.align(rs.stream.color)

        # 预热
        for _ in range(30):
            pipeline.wait_for_frames()
            time.sleep(0.05)

        # ── 连接转台 ──
        print(f'连接转台 {TURNTABLE_PORT} ...')
        tt = TurntableController(port=TURNTABLE_PORT)
        tt.open()
        tt.zero()

        try:
            if args.video:
                capture_video_mode(pipeline, align, tt)
            else:
                capture_step_mode(pipeline, align, tt)
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            tt.stop()
            tt.close()
            print('相机 + 转台已释放.')
    else:
        if not os.path.isdir(IMAGES_DIR):
            sys.exit(f'图片目录不存在: {IMAGES_DIR}\n请先采集图片')
        n_imgs = len([f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')])
        print(f'跳过采集, 使用已有图片: {n_imgs} 张')

    # ── COLMAP 重建 ──
    run_colmap(
        camera_model=args.camera_model,
        single_camera=not args.multi_camera,
    )

    # ── 后处理 ──
    if not args.no_post:
        postprocess_colmap()

    print(f'\n{"="*55}')
    print(f'  完成!')
    print(f'  稠密点云: {FINAL_PLY}')
    print(f'  坐标系: COLMAP 原始坐标系 (需手动对齐到世界坐标)')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
