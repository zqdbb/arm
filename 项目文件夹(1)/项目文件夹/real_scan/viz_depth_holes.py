#!/usr/bin/env python3
"""
可视化 D435i 深度空洞: 彩色图 | 深度热力图 | 空洞高亮
用法: python3 viz_depth_holes.py
"""

import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import cv2, numpy as np, json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

DATA = Path(__file__).parent / 'output/v12/chair'
OUT = Path(__file__).parent / 'output/depth_holes'
OUT.mkdir(parents=True, exist_ok=True)

with open(DATA / 'calibrate.json') as f:
    calib = json.load(f)
depth_scale = calib['depth_scale']  # 0.001 → raw * 0.001 = meters

FRAMES = ['000', '018', '036', '054']
LABELS = ['正面 0°', '侧面 90°', '背面 180°', '侧面 270°']

fig, axes = plt.subplots(len(FRAMES), 3, figsize=(15, 5 * len(FRAMES)))
plt.subplots_adjust(wspace=0.02, hspace=0.15)

stats = {}

for row, (fid, label) in enumerate(zip(FRAMES, LABELS)):
    color = cv2.cvtColor(cv2.imread(str(DATA / f'color/{fid}.png')), cv2.COLOR_BGR2RGB)
    depth_raw = cv2.imread(str(DATA / f'depth/{fid}.png'), -1)  # uint16
    depth_m = depth_raw.astype(np.float32) * depth_scale  # meters
    depth_mm = depth_m * 1000  # millimeters

    # 椅子 mask (alpha = 255 → 椅子区域)
    view = cv2.imread(str(DATA / f'views/view_{fid}.png'), -1)
    chair_mask = (view[:, :, 3] > 128)

    # 空洞 = depth == 0
    hole_mask = (depth_raw == 0)
    overall_hole = hole_mask.sum() / hole_mask.size * 100
    chair_hole = (hole_mask & chair_mask).sum() / max(chair_mask.sum(), 1) * 100
    chair_density = 100 - chair_hole
    stats[fid] = (overall_hole, chair_hole, chair_density)

    # Col 1: 原图 + mask 轮廓
    ax = axes[row, 0]
    ax.imshow(color)
    contours, _ = cv2.findContours(chair_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        ax.plot(c[:, 0, 0], c[:, 0, 1], 'lime', linewidth=1.5)
    ax.set_title(f'{label} (frame {fid})', fontsize=11)
    ax.axis('off')

    # Col 2: 深度热力图 (有效深度彩色，空洞灰色)
    ax = axes[row, 1]
    depth_disp = depth_mm.copy()
    valid = depth_disp > 0
    vmin, vmax = depth_mm[valid].min(), depth_mm[valid].max()
    depth_colored = np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    depth_colored[valid] = (plt.cm.inferno(
        (depth_mm[valid] - vmin) / (vmax - vmin))[:, :3] * 255).astype(np.uint8)
    depth_colored[~valid] = [60, 60, 60]
    ax.imshow(depth_colored)
    ax.set_title(f'深度 ({vmin:.0f}–{vmax:.0f} mm)', fontsize=11)
    ax.axis('off')

    # Col 3: 空洞高亮 (红色=空洞，绿色=有效深度)
    ax = axes[row, 2]
    overlay = color.copy()
    overlay[hole_mask] = [255, 40, 40]  # 红色 = 空洞
    # 椅子轮廓
    for c in contours:
        ax.plot(c[:, 0, 0], c[:, 0, 1], 'cyan', linewidth=1.5)
    ax.imshow(overlay)
    ax.set_title(
        f'空洞: 整体 {overall_hole:.1f}% | 椅子 {chair_hole:.1f}%\n'
        f'椅子深度覆盖率 {chair_density:.1f}%',
        fontsize=11)
    ax.axis('off')

fig.suptitle('D435i 深度空洞可视化 — V12 椅子 (距离 42cm)', fontsize=14, y=1.01)
plt.savefig(OUT / 'depth_holes_overview.png', dpi=150, bbox_inches='tight')
print(f'已保存: {OUT}/depth_holes_overview.png')

# ── 统计汇总 ──
print(f'\n{"Frame":>8}  {"角度":>10}  {"整体空洞%":>10}  {"椅子空洞%":>11}  {"椅子覆盖率%":>11}')
print('-' * 55)
for fid, label in zip(FRAMES, LABELS):
    oh, ch, cd = stats[fid]
    print(f'{fid:>8}  {label:>10}  {oh:>10.1f}  {ch:>11.1f}  {cd:>11.1f}')

# ── 所有 72 帧统计 ──
print('\n\n── 全部 72 帧深度覆盖率 ──')
all_color = sorted((DATA / 'color').glob('*.png'))
covs = []
for cp in all_color:
    fid = cp.stem
    dp = DATA / f'depth/{fid}.png'
    vp = DATA / f'views/view_{fid}.png'
    if not dp.exists() or not vp.exists():
        continue
    dr = cv2.imread(str(dp), -1)
    view = cv2.imread(str(vp), -1)
    cm = view[:, :, 3] > 128
    hole = (dr == 0)
    ch = (hole & cm).sum() / max(cm.sum(), 1) * 100
    covs.append(100 - ch)

plt.figure(figsize=(14, 4))

ax1 = plt.subplot(1, 2, 1)
plt.plot(range(len(covs)), covs, 'o-', markersize=3, color='steelblue')
plt.axhline(y=np.mean(covs), color='red', linestyle='--', label=f'均值 {np.mean(covs):.1f}%')
plt.axhline(y=50, color='gray', linestyle=':', label='50%')
plt.xlabel('Frame')
plt.ylabel('椅子深度覆盖率 (%)')
plt.title('72 帧椅子区域深度覆盖率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 105)

ax2 = plt.subplot(1, 2, 2)
plt.hist(covs, bins=20, edgecolor='black', color='steelblue', alpha=0.8)
plt.axvline(x=np.mean(covs), color='red', linestyle='--', label=f'均值 {np.mean(covs):.1f}%')
plt.xlabel('椅子深度覆盖率 (%)')
plt.ylabel('帧数')
plt.title('覆盖率分布')
plt.legend()

plt.tight_layout()
plt.savefig(OUT / 'depth_coverage_72frames.png', dpi=150, bbox_inches='tight')
print(f'已保存: {OUT}/depth_coverage_72frames.png')
print(f'\n全部 72 帧椅子深度覆盖率:')
print(f'  均值: {np.mean(covs):.1f}%')
print(f'  中位数: {np.median(covs):.1f}%')
print(f'  最差: {np.min(covs):.1f}% (frame {all_color[np.argmin(covs)].stem})')
print(f'  最好: {np.max(covs):.1f}% (frame {all_color[np.argmax(covs)].stem})')
