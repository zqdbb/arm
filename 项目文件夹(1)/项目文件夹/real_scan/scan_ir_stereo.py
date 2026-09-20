#!/usr/bin/env python3
"""
V20: D435i 原始红外立体匹配
绕过内部 ASIC 芯片，直接读 IR 左右图像 + OpenCV StereoSGBM 匹配。
用法: python3 scan_ir_stereo.py
"""

import cv2, numpy as np, json, time
from pathlib import Path
import pyrealsense2 as rs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
OUT = BASE / 'output/v20_ir_stereo'
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. 启动 D435i 管线 ──
print('启动 D435i ...')
pipe = rs.pipeline()
cfg = rs.config()

# IR 立体对 (左=infra1, 右=infra2) + 深度
cfg.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipe.start(cfg)

# 强制打开 IR 投影仪
dev = profile.get_device()
depth_sensor = dev.first_depth_sensor()
depth_sensor.set_option(rs.option.emitter_enabled, 1)  # always on
depth_sensor.set_option(rs.option.laser_power, 360)    # max power
print(f'  激光功率: {depth_sensor.get_option(rs.option.laser_power)}')

# 踢掉前 30 帧让曝光稳定
for _ in range(30):
    pipe.wait_for_frames()

# ── 2. 提取内参 ──
ir_profile = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
ir_intr = ir_profile.get_intrinsics()
fx_ir, fy_ir = ir_intr.fx, ir_intr.fy
cx_ir, cy_ir = ir_intr.ppx, ir_intr.ppy
w_ir, h_ir = ir_intr.width, ir_intr.height
baseline = 0.05  # D435i IR 双目基线 50mm

print(f'  IR 内参: {w_ir}x{h_ir}, fx={fx_ir:.1f}, fy={fy_ir:.1f}, cx={cx_ir:.1f}, cy={cy_ir:.1f}')
print(f'  基线: {baseline*1000:.0f}mm')

# 获取 depth scale
depth_scale = depth_sensor.get_depth_scale()
print(f'  depth_scale: {depth_scale}')

# ── 3. 采集 ──
print('\n采集 IR 立体对 ...')
frames = pipe.wait_for_frames()
ir_l_frame = frames.get_infrared_frame(1)
ir_r_frame = frames.get_infrared_frame(2)
depth_frame = frames.get_depth_frame()
color_frame = frames.get_color_frame()

ir_left = np.asanyarray(ir_l_frame.get_data())   # uint8
ir_right = np.asanyarray(ir_r_frame.get_data())  # uint8
depth_builtin = np.asanyarray(depth_frame.get_data()).astype(np.float32)  # uint16 → float
color = np.asanyarray(color_frame.get_data())

pipe.stop()
print(f'  IR 左:  min={ir_left.min()}, max={ir_left.max()}, mean={ir_left.mean():.1f}')
print(f'  IR 右:  min={ir_right.min()}, max={ir_right.max()}, mean={ir_right.mean():.1f}')

# ── 4. 可视化 IR 斑点 ──
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
axes[0].imshow(ir_left, cmap='gray')
axes[0].set_title('IR 左 (infra1)', fontsize=12)
axes[0].axis('off')

axes[1].imshow(ir_right, cmap='gray')
axes[1].set_title('IR 右 (infra2)', fontsize=12)
axes[1].axis('off')

# 放大中心区域看斑点纹理
h, w = ir_left.shape
cy_h, cx_h = h // 2, w // 2
crop_size = 200
crop_l = ir_left[cy_h-crop_size:cy_h+crop_size, cx_h-crop_size:cx_h+crop_size]
crop_r = ir_right[cy_h-crop_size:cy_h+crop_size, cx_h-crop_size:cx_h+crop_size]

axes[2].imshow(crop_l, cmap='gray')
axes[2].set_title(f'IR 左中心 {crop_size*2}x{crop_size*2} (找斑点)', fontsize=12)
axes[2].axis('off')

# 增强对比度看斑点
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
crop_l_enhanced = clahe.apply(crop_l)
axes[3].imshow(crop_l_enhanced, cmap='gray')
axes[3].set_title('CLAHE 增强后 (斑点可见性)', fontsize=12)
axes[3].axis('off')

plt.tight_layout()
plt.savefig(OUT / 'ir_raw_and_spots.png', dpi=150, bbox_inches='tight')
print(f'已保存: {OUT}/ir_raw_and_spots.png')

# ── 5. StereoSGBM 匹配 ──
print('\nStereoSGBM 匹配 ...')

# 参数解释:
# - numDisparities: 搜索范围。对 42cm 距离，视差 ~50px，设 128 足够
# - blockSize: 匹配窗口。纹理弱时用小窗口（3-5），避免平滑掉细节
# - uniquenessRatio: 降低→允许更多模糊匹配。木头表面用低值
# - speckleWindowSize: 斑点过滤窗口，0=关
# - disp12MaxDiff: 左右一致性检查，降低→宽松
# - preFilterCap: 预滤波截断值，降低→保留弱信号

# 方案 A: 保守（针对弱纹理）
sgbm_weak = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=3,
    uniquenessRatio=5,
    speckleWindowSize=0,
    speckleRange=0,
    preFilterCap=31,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disp_weak = sgbm_weak.compute(ir_left, ir_right).astype(np.float32) / 16.0

# 方案 B: 激进（块更小、约束更松）
sgbm_aggro = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=3,
    P1=4 * 3 * 3**2,
    P2=16 * 3 * 3**2,
    disp12MaxDiff=5,
    uniquenessRatio=3,
    speckleWindowSize=0,
    speckleRange=0,
    preFilterCap=15,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disp_aggro = sgbm_aggro.compute(ir_left, ir_right).astype(np.float32) / 16.0

# 方案 C: 用 BM (更快, 可能对弱纹理更鲁棒)
bm = cv2.StereoBM_create(numDisparities=128, blockSize=9)
disp_bm = bm.compute(ir_left, ir_right).astype(np.float32) / 16.0

# ── 6. 视差 → 深度 ──
def disparity_to_depth(disp, fx, baseline):
    """视差图 → 深度图 (米)"""
    depth = np.zeros_like(disp)
    valid = disp > 0.5
    depth[valid] = (fx * baseline) / disp[valid]
    return depth

depth_sgbm_weak = disparity_to_depth(disp_weak, fx_ir, baseline)
depth_sgbm_aggro = disparity_to_depth(disp_aggro, fx_ir, baseline)
depth_bm_custom = disparity_to_depth(disp_bm, fx_ir, baseline)

# 内置深度 (mm → m)
depth_builtin_m = depth_builtin * depth_scale

# YOLO mask 区域（简单用中心矩形近似椅子区域）
h, w = ir_left.shape
mask_center = np.zeros((h, w), dtype=bool)
cy, cx = h // 2, w // 2
# 椅子在画面中心约 100×100px 区域（在 720p 上按比例约 144×144）
half = 100
mask_center[cy-half:cy+half, cx-half:cx+half] = True

def compute_coverage(depth, mask):
    """有效深度覆盖率"""
    total = mask.sum()
    if total == 0:
        return 0
    valid = (depth > 0) & mask
    return valid.sum() / total * 100

cov_builtin = compute_coverage(depth_builtin_m, mask_center)
cov_sgbm_weak = compute_coverage(depth_sgbm_weak, mask_center)
cov_sgbm_aggro = compute_coverage(depth_sgbm_aggro, mask_center)
cov_bm = compute_coverage(depth_bm_custom, mask_center)

print(f'\n  中心区域深度覆盖率 ({half*2}x{half*2}px):')
print(f'    内置深度 (ASIC):   {cov_builtin:.1f}%')
print(f'    SGBM 保守:          {cov_sgbm_weak:.1f}%')
print(f'    SGBM 激进:          {cov_sgbm_aggro:.1f}%')
print(f'    BM:                 {cov_bm:.1f}%')

# ── 7. 对比可视化 ──
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

def show_depth(ax, depth_m, title, vmin=0.3, vmax=0.55):
    disp = depth_m.copy()
    valid = disp > 0
    colored = np.zeros((*disp.shape, 3), dtype=np.uint8)
    if valid.any():
        d_valid = disp[valid]
        norm = (np.clip(d_valid, vmin, vmax) - vmin) / (vmax - vmin)
        colored[valid] = (plt.cm.inferno(norm)[:, :3] * 255).astype(np.uint8)
    colored[~valid] = [40, 40, 40]
    ax.imshow(colored)
    cov = valid.sum() / valid.size * 100
    ax.set_title(f'{title} (覆盖率 {cov:.1f}%)', fontsize=12)
    ax.axis('off')

def show_depth_center(ax, depth_m, title, vmin=0.3, vmax=0.55):
    disp = depth_m[cy-half:cy+half, cx-half:cx+half].copy()
    valid = disp > 0
    colored = np.zeros((*disp.shape, 3), dtype=np.uint8)
    if valid.any():
        d_valid = disp[valid]
        norm = (np.clip(d_valid, vmin, vmax) - vmin) / (vmax - vmin)
        colored[valid] = (plt.cm.inferno(norm)[:, :3] * 255).astype(np.uint8)
    colored[~valid] = [40, 40, 40]
    ax.imshow(colored)
    cov = compute_coverage(depth_m, mask_center)
    ax.set_title(f'{title}\n中心覆盖率 {cov:.1f}%', fontsize=12)
    ax.axis('off')

# Row 1: 全图深度对比
show_depth(axes[0, 0], depth_builtin_m, '内置深度 (ASIC)')
show_depth(axes[0, 1], depth_sgbm_weak, 'SGBM 保守')
show_depth(axes[0, 2], depth_sgbm_aggro, 'SGBM 激进')

# Row 2: 中心区域放大
show_depth_center(axes[1, 0], depth_builtin_m, '内置深度 (ASIC)')
show_depth_center(axes[1, 1], depth_sgbm_weak, 'SGBM 保守')
show_depth_center(axes[1, 2], depth_sgbm_aggro, 'SGBM 激进')

plt.tight_layout()
plt.savefig(OUT / 'depth_comparison.png', dpi=150, bbox_inches='tight')
print(f'已保存: {OUT}/depth_comparison.png')

# ── 8. 保存数据和统计 ──
stats = {
    'resolution': f'{w_ir}x{h_ir}',
    'fx_ir': fx_ir, 'fy_ir': fy_ir,
    'cx_ir': cx_ir, 'cy_ir': cy_ir,
    'baseline_m': baseline,
    'depth_scale': depth_scale,
    'center_coverage_builtin': cov_builtin,
    'center_coverage_sgbm_weak': cov_sgbm_weak,
    'center_coverage_sgbm_aggro': cov_sgbm_aggro,
    'center_coverage_bm': cov_bm,
}

cv2.imwrite(str(OUT / 'ir_left.png'), ir_left)
cv2.imwrite(str(OUT / 'ir_right.png'), ir_right)
np.save(str(OUT / 'depth_builtin.npy'), depth_builtin_m)
np.save(str(OUT / 'depth_sgbm_weak.npy'), depth_sgbm_weak)
np.save(str(OUT / 'depth_sgbm_aggro.npy'), depth_sgbm_aggro)

# 打印斑点质量分析
print(f'\n── IR 斑点质量分析 ──')
# 局部方差 = 纹理丰富度
local_std = cv2.GaussianBlur(ir_left.astype(np.float32), (5,5), 0)
# 用小窗口标准差衡量
kernel = np.ones((5,5)) / 25
local_mean = cv2.filter2D(ir_left.astype(np.float32), -1, kernel)
local_sq_mean = cv2.filter2D(ir_left.astype(np.float32)**2, -1, kernel)
local_var = local_sq_mean - local_mean**2
local_std = np.sqrt(np.maximum(local_var, 0))

print(f'  全图纹理标准差 均值: {local_std.mean():.2f}')
print(f'  中心区域纹理标准差 均值: {local_std[cy-half:cy+half, cx-half:cx+half].mean():.2f}')
print(f'  椅子区域 (更小, 50x50): {local_std[cy-50:cy+50, cx-50:cx+50].mean():.2f}')

print(f'\n✅ V20 IR 立体测试完成, 输出: {OUT}')
