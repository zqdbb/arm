#!/usr/bin/env python3
"""D405 实时预览：RGB 左 + 深度热力图右."""
import pyrealsense2 as rs
import numpy as np
import cv2

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
s = profile.get_device().first_depth_sensor()
s.set_option(rs.option.visual_preset, 3)

for _ in range(30):
    pipe.wait_for_frames()

print("按 q 退出预览")
while True:
    frames = pipe.wait_for_frames()
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth = np.asanyarray(frames.get_depth_frame().get_data())
    
    valid = depth > 0
    vpct = valid.sum() / depth.size * 100
    
    # 深度热力图 (0-800mm, 蓝=近 红=远)
    d_max = 800
    depth_clipped = np.clip(depth, 0, d_max)
    depth_vis = cv2.applyColorMap((depth_clipped / d_max * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_vis[depth == 0] = 0  # 无效深度=黑色

    # 色标条
    bar = np.zeros((20, depth_vis.shape[1], 3), dtype=np.uint8)
    for x in range(bar.shape[1]):
        val = int(x / bar.shape[1] * 255)
        bar[:, x] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
    cv2.putText(bar, '0mm(近)', (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(bar, f'{d_max}mm(远)', (bar.shape[1]-120, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    depth_vis = np.vstack([depth_vis, bar])

    # RGB 叠加文字
    if valid.sum() > 0:
        d_vals = depth[valid]
        info1 = f'Range: {d_vals.min()}~{int(d_vals.max())}mm Med:{int(np.median(d_vals))}mm'
        info2 = f'Valid: {vpct:.1f}%'
        # 中心十字线深度
        cy, cx = depth.shape[0]//2, depth.shape[1]//2
        dc = depth[cy-10:cy+10, cx-10:cx+10]
        dcv = dc[dc > 0]
        na = "N/A"
        info3 = f"Center: {int(np.median(dcv)) if len(dcv)>0 else na}mm"
    else:
        info1 = 'DEPTH: NONE'
        info2 = '请靠近物体!'
        info3 = ''
    cv2.putText(color, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(color, info2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(color, info3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    
    color_padded = np.vstack([color, np.zeros((20, color.shape[1], 3), dtype=np.uint8)])
    display = np.hstack([color_padded, depth_vis])
    cv2.imshow('D405 | 左=RGB 右=深度热力图(0-800mm) | q退出', display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
pipe.stop()
