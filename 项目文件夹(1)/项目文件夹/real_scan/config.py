"""D435i 扫描参数配置."""

import os

# ── 相机参数 ──────────────────────
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
DEPTH_FPS = 30

# ── 电动转台参数 (Y200RA60 + 1SC控制器) ──
TURNTABLE_PORT = '/dev/ttyUSB0'
TURNTABLE_BAUD = 115200
TURNTABLE_PULSES_PER_DEG = 400    # ★ 每度脉冲数, 角度不准就调这个
                                #   调大 → 同样指令转得少
                                #   调小 → 同样指令转得多
TURNTABLE_MAX_SPEED = 50000       # 最大速度 Hz (50°/s)
TURNTABLE_DEFAULT_SPEED = 10000   # 扫描速度 Hz (10°/s)

# 扫描角度
TURNTABLE_ANGLES = list(range(0, 360, 30))  # 12 帧, 每30°

# 标定参数
CALIB_FRAMES = 20  # 标定时采集帧数 (取中值防抖)

# 后处理参数
VOXEL_SIZE = 0.002       # 降采样 2mm
OUTLIER_NB = 20          # 半径滤波邻居数
OUTLIER_RADIUS = 0.01    # 半径滤波半径 1cm
DBSCAN_EPS = 0.03        # 聚类距离 3cm (放宽连接靠背和座面)
DBSCAN_MIN = 25          # 聚类最小点数 (放宽保留稀疏靠背)

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
CALIB_FILE = os.path.join(OUTPUT_DIR, 'calibrate.json')
SCAN_OUTPUT = os.path.join(OUTPUT_DIR, 'scan.ply')
CLASSIFY_OUTPUT = os.path.join(OUTPUT_DIR, 'classify_rgb.png')
FRAME_RGB_PATTERN = os.path.join(OUTPUT_DIR, 'frame_{i:03d}_rgb.png')
FRAME_PLY_PATTERN = os.path.join(OUTPUT_DIR, 'frame_{i:03d}.ply')
