#!/usr/bin/env python3
"""
YOLO 椅子分割可视化：逐帧 YOLOv8n-seg 推理，在原图上叠加 mask 和检测框，保存可视化结果。
"""

import cv2
import numpy as np
from pathlib import Path
import time

BASE = Path(__file__).parent
COLOR_DIR = BASE / 'output/v6/color'
OUT_DIR = BASE / 'output/yolo_viz'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.5


def main():
    from ultralytics import YOLO

    color_files = sorted(COLOR_DIR.glob('*.png'))
    if not color_files:
        print(f'错误: 在 {COLOR_DIR} 中未找到彩色图像')
        return

    print(f'加载 YOLOv8n-seg...')
    yolo = YOLO('yolov8n-seg.pt')
    print(f'彩色图像: {len(color_files)} 帧 ({color_files[0].parent})')

    detected = 0
    total_mask_px = 0

    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None:
            continue
        h, w = img.shape[:2]

        results = yolo(str(cf), classes=[56], verbose=False)
        r = results[0]

        # 创建可视化图层
        viz = img.copy()
        overlay = np.zeros_like(img)

        if r.boxes is not None and len(r.boxes) > 0:
            conf = r.boxes.conf[0].item()
            if conf >= CONF_THRESHOLD:
                detected += 1

                # 提取 mask
                mask_raw = r.masks.data[0].cpu().numpy()
                mask_resized = cv2.resize(mask_raw, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                mask_px = int(mask_binary.sum())
                total_mask_px += mask_px

                # 半透明绿色覆盖 mask 区域
                overlay[mask_binary > 0] = [0, 255, 0]

                # 检测框
                box = r.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # mask 轮廓
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(viz, contours, -1, (0, 255, 0), 1)

                # 标签
                label = f'chair {conf:.2f}  {mask_px}px'
                cv2.putText(viz, label, (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                cv2.putText(viz, f'low conf {conf:.2f}', (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(viz, 'no chair', (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 混合
        alpha = 0.45
        viz = cv2.addWeighted(viz, 1.0, overlay, alpha, 0)

        # 帧号
        deg = i * 5
        cv2.putText(viz, f'frame {i}  {deg}deg', (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 保存
        out_path = OUT_DIR / f'{i:03d}.png'
        cv2.imwrite(str(out_path), viz)

        if i < 5 or (i + 1) % 18 == 0:
            status = f'conf={conf:.2f} mask={mask_px}px' if r.boxes is not None else 'miss'
            print(f'  [{i+1:2d}/{len(color_files)}] {deg:3d}deg  {status}')

    print(f'\n检出率: {detected}/{len(color_files)} ({detected/len(color_files)*100:.0f}%)')
    if detected > 0:
        print(f'平均 mask 像素: {total_mask_px/detected:.0f}')
    print(f'可视化保存到: {OUT_DIR}/')
    print(f'共 {len(list(OUT_DIR.glob("*.png")))} 张图片')


if __name__ == '__main__':
    main()
