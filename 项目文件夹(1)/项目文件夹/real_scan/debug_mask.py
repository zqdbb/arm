#!/usr/bin/env python3
"""调试：看深度掩膜、饱和度掩膜、合并掩膜、轮廓到底长什么样"""
import cv2, numpy as np, pyrealsense2 as rs, time

W, H = 1280, 720
ZOOM = 2.0

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
prof = pipe.start(cfg)
align = rs.align(rs.stream.color)
depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

crop_w = int(W / ZOOM)
crop_h = int(H / ZOOM)
x0 = (W - crop_w) // 2
y0 = (H - crop_h) // 2

for _ in range(30):
    pipe.wait_for_frames()

print(f'深度比例: {depth_scale}')
print('按 S 截图保存, 按 Q 退出')

while True:
    aligned = align.process(pipe.wait_for_frames())
    color = np.asanyarray(aligned.get_color_frame().get_data())
    depth = np.asanyarray(aligned.get_depth_frame().get_data())

    color_crop = color[y0:y0+crop_h, x0:x0+crop_w]
    depth_crop = depth[y0:y0+crop_h, x0:x0+crop_w]
    h, w = depth_crop.shape
    cy, cx = h // 2, w // 2

    # 深度掩膜
    z_raw = float(depth_crop[cy, cx])
    z_center = z_raw * depth_scale
    depth_m = depth_crop.astype(np.float64) * depth_scale
    depth_mask = (depth_crop > 0) & (np.abs(depth_m - z_center) < 0.05)

    # 饱和度掩膜
    hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    sat_mask = sat > 25

    # 合并掩膜
    mask_2d = depth_mask & sat_mask

    # 轮廓
    mask_u8 = mask_2d.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # === 可视化 ===
    # 行1: 原图 | 饱和度 | 饱和度掩膜
    # 行2: 深度热力图 | 深度掩膜 | 合并掩膜+轮廓

    sat_color = cv2.applyColorMap(sat, cv2.COLORMAP_HOT)
    depth_viz = cv2.applyColorMap((depth_crop / depth_crop.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)

    dm_viz = cv2.cvtColor(depth_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    sm_viz = cv2.cvtColor(sat_mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    combined_viz = color_crop.copy()
    combined_viz[~mask_2d] = combined_viz[~mask_2d] // 2  # 非掩膜区域变暗

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(combined_viz, [cnt], -1, (0, 255, 0), 2)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(combined_viz, ellipse, (255, 0, 0), 2)

    # 拼接
    row1 = np.hstack([color_crop, sat_color, sm_viz])
    row2 = np.hstack([depth_viz, dm_viz, combined_viz])

    # resize 让它们能显示
    scale = 0.5
    row1_s = cv2.resize(row1, (int(row1.shape[1]*scale), int(row1.shape[0]*scale)))
    row2_s = cv2.resize(row2, (int(row2.shape[1]*scale), int(row2.shape[0]*scale)))

    for i, (name, img) in enumerate([("Original", color_crop), ("Saturation", sat_color), ("Sat Mask", sm_viz),
                                       ("Depth", depth_viz), ("Depth Mask", dm_viz), ("Combined", combined_viz)]):
        cv2.putText(row1_s if i < 3 else row2_s, name, (10 + (i%3)*int(row1_s.shape[1]/3), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    info = f'depth_center={z_center:.2f}m  sat_max={sat.max()}  mask_px={mask_2d.sum()}  contours={len(contours)}'
    display = np.vstack([row1_s, row2_s])
    cv2.putText(display, info, (5, display.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)

    cv2.imshow('debug_mask', display)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('/tmp/debug_mask.png', display)
        print(f'截图保存 /tmp/debug_mask.png  mask={mask_2d.sum()}px  contours={len(contours)}')

pipe.stop()
cv2.destroyAllWindows()
