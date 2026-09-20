#!/usr/bin/env python3
"""调试: 看看 detect_center_blob 为什么找不到黑标记"""
import cv2, time, numpy as np
import pyrealsense2 as rs

W, H = 640, 480
DIGITAL_ZOOM = 2.0

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 15)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 15)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)

# crop
crop_w = int(W / DIGITAL_ZOOM)
crop_h = int(H / DIGITAL_ZOOM)
crop_x = (W - crop_w) // 2
crop_y = (H - crop_h) // 2
cW, cH = crop_w, crop_h

print(f'裁剪: {W}x{H} -> {cW}x{cH}')
print('按 Q 退出, 按 S 截图')

for _ in range(30):
    pipe.wait_for_frames()

while True:
    aligned = align.process(pipe.wait_for_frames())
    color = np.asanyarray(aligned.get_color_frame().get_data())
    color_crop = color[crop_y:crop_y + cH, crop_x:crop_x + cW]
    gray = cv2.cvtColor(color_crop, cv2.COLOR_BGR2GRAY)

    # detect_center_blob 逻辑
    th = np.percentile(gray, 20)
    dark = (gray < max(th, 20)).astype(np.uint8) * 255
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)

    cy_img, cx_img = cH / 2, cW / 2
    viz = color_crop.copy()

    best_label, best_score = None, -1
    candidates = []
    for lb in range(1, n_labels):
        area = stats[lb, cv2.CC_STAT_AREA]
        cx_b, cy_b = centroids[lb]
        dist = np.hypot(cx_b - cx_img, cy_b - cy_img)

        candidates.append((lb, area, cx_b, cy_b, dist))

        if area < 20 or area > 5000:
            continue
        if dist > min(cW, cH) * 0.4:
            continue

        mask_lb = (labels == lb)
        mean_brightness = gray[mask_lb].mean()
        score = (255 - mean_brightness) * 0.6 + (1.0 - dist / (min(cW, cH) * 0.4)) * 0.4
        if score > best_score:
            best_score = score
            best_label = lb

    # 画所有候选暗色区域
    for lb, area, cx_b, cy_b, dist in candidates:
        color_flag = (0, 255, 0) if (20 <= area <= 5000 and dist <= min(cW, cH) * 0.4) else (0, 0, 255)
        cv2.circle(viz, (int(cx_b), int(cy_b)), 5, color_flag, -1)
        cv2.putText(viz, f'a{area} d{dist:.0f}', (int(cx_b) + 8, int(cy_b)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_flag, 1)

    # 画中心十字
    cv2.line(viz, (cW // 2, 0), (cW // 2, cH), (255, 255, 0), 1)
    cv2.line(viz, (0, cH // 2), (cW, cH // 2), (255, 255, 0), 1)
    cv2.circle(viz, (cW // 2, cH // 2), int(min(cW, cH) * 0.4), (255, 255, 0), 1)

    status = f'best={best_label} score={best_score:.1f}' if best_label else 'NOT FOUND'
    cv2.putText(viz, f'Dark regions: {n_labels-1}  {status}', (5, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(viz, f'threshold={th:.1f}  center=({cx_img:.0f},{cy_img:.0f})', (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    cv2.imshow('debug', viz)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('/tmp/calib_debug.png', viz)
        print('截图保存 /tmp/calib_debug.png')

pipe.stop()
cv2.destroyAllWindows()
