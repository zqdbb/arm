#!/usr/bin/env python3
"""
V16 数据验证工具 — 逐帧检查 mask 质量 + 深度数据质量
用法: python3 validate_v16.py [帧号]  # 不传帧号则检查关键帧
输出: output/v16/chair/validate/overlay_XXX.png (mask叠加图)
"""
import cv2, numpy as np, sys, os
from pathlib import Path

OUT = Path('output/v16/chair')
VALIDATE_DIR = OUT / 'validate'
VALIDATE_DIR.mkdir(exist_ok=True)

KEY_FRAMES = [0, 9, 18, 27, 36, 45, 54, 63]  # 每45度一个

def validate_frame(idx):
    """检查一帧的数据质量, 保存叠加图."""
    color = cv2.imread(str(OUT / f'color/{idx:03d}.png'))
    depth = cv2.imread(str(OUT / f'depth/{idx:03d}.png'), cv2.IMREAD_UNCHANGED)

    # 尝试读取mask (从views目录或重新算)
    view_path = OUT / f'views/view_{idx:03d}.png'
    mask = None
    mask_source = 'none'
    if view_path.exists():
        rgba = cv2.imread(str(view_path), cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.shape[-1] == 4:
            mask = (rgba[:,:,3] > 128).astype(np.uint8) * 255
            mask_source = 'SAM/YOLO'

    if color is None:
        print(f'  frame {idx:03d}: 彩色图不存在')
        return

    h, w = color.shape[:2]

    # === 深度统计 ===
    depth_stats = {}
    if depth is not None:
        d_all = depth[depth > 0]
        depth_stats['全图有效点'] = f'{len(d_all)} ({100*len(d_all)/(w*h):.1f}%)'
        if len(d_all) > 0:
            depth_stats['全图深度范围'] = f'{d_all.min()}~{d_all.max()}mm'

        if mask is not None:
            d_masked = depth[mask > 0]
            d_valid = d_masked[d_masked > 0]
            depth_stats['mask内有效点'] = f'{len(d_valid)}'
            if len(d_valid) > 0:
                depth_stats['mask内深度'] = f'median={np.median(d_valid):.0f} min={d_valid.min():.0f} max={d_valid.max():.0f}mm'
                depth_stats['mask内深度跨度'] = f'{d_valid.max()-d_valid.min():.0f}mm = {(d_valid.max()-d_valid.min())/10:.1f}cm'

            d_outside = depth[(mask == 0) & (depth > 0)]
            if len(d_outside) > 0 and len(d_valid) > 0:
                diff = np.median(d_outside) - np.median(d_valid)
                depth_stats['转台-椅子深度差'] = f'{diff:.0f}mm = {diff/10:.1f}cm'

    # === 生成叠加图 ===
    overlay = color.copy()

    # 深度可视化 (放大20倍亮度)
    if depth is not None:
        depth_viz = np.clip(depth.astype(float) * 20, 0, 65535).astype(np.uint16)
        depth_viz = cv2.applyColorMap(
            (depth_viz.astype(float) / 256).astype(np.uint8), cv2.COLORMAP_JET)
        # 无效深度区域显示为灰色
        depth_viz[depth == 0] = [128, 128, 128]
    else:
        depth_viz = np.zeros_like(color)

    # mask轮廓 (绿色=有mask, 红色=无mask区域)
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        # 半透明mask
        alpha = 0.3
        mask_color = np.zeros_like(color)
        mask_color[:,:,1] = 255  # 绿色
        overlay = cv2.addWeighted(overlay, 1, mask_color, alpha, 0)

    # 文字信息
    y = 25
    cv2.putText(overlay, f'Frame {idx:03d} ({idx*5}deg)', (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    y += 25
    for key, val in depth_stats.items():
        cv2.putText(overlay, f'{key}: {val}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        y += 18

    # 保存
    out_path = VALIDATE_DIR / f'overlay_{idx:03d}.png'
    # 并排: 原图 | 深度热力图 | 叠加图
    combined = np.hstack([color, depth_viz, overlay])
    cv2.imwrite(str(out_path), combined)

    # === 终端报告 ===
    print(f'\n{"="*50}')
    print(f'Frame {idx:03d} ({idx*5}deg)  mask来源: {mask_source}')
    print(f'{"="*50}')
    for key, val in depth_stats.items():
        print(f'  {key}: {val}')

    # 关键判断
    if mask is not None and mask.sum() > 0:
        h_proj = mask.sum(axis=1)
        w_proj = mask.sum(axis=0)
        print(f'  mask尺寸: H={h_proj.max()}px × W={w_proj.max()}px, 总面积={mask.sum()}px')

        # 预期椅子在画面中约150px高
        if h_proj.max() < 30:
            print(f'  ⚠️  mask高度<30px, 可能没抠到椅子!')
        if w_proj.max() < 20:
            print(f'  ⚠️  mask宽度<20px, 可能没抠到椅子!')
    else:
        print(f'  ⚠️  无mask!')

    if depth_stats.get('mask内深度跨度', '').endswith('cm'):
        span_cm = float(depth_stats['mask内深度跨度'].split('=')[-1].replace('cm',''))
        if span_cm > 8:
            print(f'  ⚠️  mask内深度跨度>{span_cm:.0f}cm, 可能混入了非椅子像素!')

    print(f'\n  叠加图已保存: {out_path}')
    return out_path


if __name__ == '__main__':
    if len(sys.argv) > 1:
        frames = [int(sys.argv[1])]
    else:
        frames = KEY_FRAMES

    print(f'检查 {len(frames)} 帧...')
    for f in frames:
        validate_frame(f)

    print(f'\n所有叠加图保存在: {VALIDATE_DIR}/')
    print('请打开 overlay_*.png 检查:')
    print('  左=原图 | 中=深度热力图(彩色=近, 灰色=无效) | 右=mask叠加(绿色轮廓)')
