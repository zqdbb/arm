#!/usr/bin/env python3
"""2帧融合诊断 — 生成交互式 HTML 网页，动态对比对齐前后."""
import cv2, numpy as np, json
from pathlib import Path

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/capture'

with open(CAPTURE_DIR / 'meta.json') as f:
    meta = json.load(f)

fx, fy = meta['fx'], meta['fy']
ppx, ppy = meta['ppx'], meta['ppy']
W, H = meta['wxH_zoomed']
step_deg = meta['step_deg']
calib = meta['calib']

R_cal = np.array(calib['R'])
t_cal = np.array(calib['t'])
axis_dir = R_cal[2, :].copy()  # 第三行 = 平面法向量
axis_dir = axis_dir / np.linalg.norm(axis_dir)
axis_point = t_cal

from ultralytics import YOLO
yolo = YOLO(str(BASE / 'yolov8n-seg.pt'))

u = np.arange(W); v = np.arange(H)
uu, vv = np.meshgrid(u, v)
ray_x = (uu - ppx) / fx
ray_y = (vv - ppy) / fy

def rodrigues(a, angle):
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def get_pts(idx):
    c = cv2.imread(str(CAPTURE_DIR / 'color' / f'{idx:03d}.jpg'))
    d = cv2.imread(str(CAPTURE_DIR / 'depth' / f'{idx:03d}.png'), cv2.IMREAD_UNCHANGED)
    if c.shape[:2] != (H, W): c = cv2.resize(c, (W, H))
    dm = d.astype(np.float32) * 0.001
    res = yolo(c, verbose=False, classes=[56])
    m = np.zeros((H, W), dtype=bool)
    if res[0].masks is not None:
        best, best_c = None, 0
        for j in range(len(res[0].boxes)):
            if res[0].names[int(res[0].boxes.cls[j])] == 'chair':
                cc = float(res[0].boxes.conf[j])
                if cc > best_c: best_c = cc; best = j
        if best is not None:
            mk = res[0].masks.data[best].cpu().numpy()
            if mk.shape != (H, W): mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
            m = mk > 0.5
    valid = (dm > 0.01) & m
    z = dm[valid]; x = ray_x[valid] * z; y = ray_y[valid] * z
    return np.stack([x, y, z], axis=1)

def make_world(pts, idx, ax, sign):
    theta = np.radians(idx * step_deg)
    R = rodrigues(ax, sign * theta)
    c = pts - axis_point
    return (R @ c.T).T

step = 18
pts0, pts18 = get_pts(0), get_pts(step)

# 测试所有候选轴: R_cal 的三行 + 三列，各两个方向
candidates = {}
for i in range(3):
    for j in range(3):
        ax = R_cal[i, :] if j == 0 else R_cal[:, i]
        for sgn in [-1, +1]:
            for rot_sgn in [-1, +1]:
                ax_n = ax * sgn
                ax_n = ax_n / np.linalg.norm(ax_n)
                w0 = make_world(pts0, 0, ax_n, rot_sgn)
                w18 = make_world(pts18, step, ax_n, rot_sgn)
                d = np.linalg.norm(w0.mean(axis=0) - w18.mean(axis=0)) * 1000
                label = f'R[{i},:]*{sgn} theta*{rot_sgn}' if j == 0 else f'R[:,{i}]*{sgn} theta*{rot_sgn}'
                candidates[label] = (d, ax_n, rot_sgn)

best = min(candidates, key=lambda k: candidates[k][0])
best_d, best_ax, best_sign = candidates[best]
print(f'Frame 0: {len(pts0):,} pts  Frame {step}: {len(pts18):,} pts')
print(f'\n最佳轴: {best}  → 质心距={best_d:.0f}mm')
print(f'axis_dir={best_ax}  sign={best_sign}')

# 打印所有结果
for k, (d, _, _) in sorted(candidates.items(), key=lambda x: x[1][0]):
    if d < 100:
        print(f'  {k}: {d:.0f}mm')

axis_dir = best_ax
sign = best_sign
pts0_w = make_world(pts0, 0, axis_dir, sign); pts18_w = make_world(pts18, step, axis_dir, sign)

# 下采样（网页显示用，每帧取 2000 点）
def sample(pts, n=2000):
    if len(pts) <= n: return pts
    return pts[np.random.choice(len(pts), n, replace=False)]

s0, s18 = sample(pts0), sample(pts18)
s0w, s18w = sample(pts0_w), sample(pts18_w)

print(f'Frame 0:  {len(pts0):,} pts  Frame {step}: {len(pts18):,} pts')
print(f'世界坐标质心距离: {best_d:.0f}mm ', '✓ 对齐正确' if best_d < 20 else '✗ 对齐有问题')

# ── 生成 HTML ──
html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>2帧融合诊断</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ margin:0; font-family:monospace; background:#1a1a1a; color:#ccc; }}
  .plot {{ width:49vw; height:95vh; display:inline-block; }}
  .bar {{ padding:8px 16px; }}
  .bar span {{ color:#0f0; }}
</style>
</head><body>
<div class="bar">
  Frame 0: <span>{len(pts0):,} pts</span> &nbsp;
  Frame {step}: <span>{len(pts18):,} pts</span> &nbsp;
  世界质心距: <span>{best_d:.0f}mm {'✓ 对齐正确' if best_d < 20 else '✗ 对齐有问题'}</span>
  &nbsp; 红=Frame0 &nbsp; 蓝=Frame{step}
</div>
<div id="cam" class="plot"></div>
<div id="world" class="plot"></div>
<script>
function pt(arr) {{ return {{ x: arr.map(p=>p[0]), y: arr.map(p=>p[1]), z: arr.map(p=>p[2]),
  type:'scatter3d', mode:'markers',
  marker:{{ size:2, opacity:0.7 }} }}; }}

var t0 = pt({json.dumps(s0.tolist())});
t0.marker.color = 'red'; t0.name = 'Frame 0';
var t18 = pt({json.dumps(s18.tolist())});
t18.marker.color = 'blue'; t18.name = 'Frame {step}';
var t0w = pt({json.dumps(s0w.tolist())});
t0w.marker.color = 'red'; t0w.name = 'Frame 0';
var t18w = pt({json.dumps(s18w.tolist())});
t18w.marker.color = 'blue'; t18w.name = 'Frame {step}';

var lay = {{
  margin:{{l:0,r:0,t:30,b:0}},
  scene:{{ xaxis:{{title:'X'}}, yaxis:{{title:'Y'}}, zaxis:{{title:'Z'}},
    aspectmode:'data',
    camera:{{eye:{{x:1.5,y:1.5,z:1.5}}}} }},
  showlegend:true, legend:{{x:0,y:1}},
  paper_bgcolor:'#1a1a1a', plot_bgcolor:'#1a1a1a',
  font:{{color:'#ccc'}}
}};
var layCam = JSON.parse(JSON.stringify(lay));
layCam.title = '相机坐标系（未对齐 — 两个角度不重叠）';
var layWorld = JSON.parse(JSON.stringify(lay));
layWorld.title = '世界坐标系（对齐后 — 椅子应该重叠！）';

Plotly.newPlot('cam',    [t0, t18],   layCam);
Plotly.newPlot('world',  [t0w, t18w], layWorld);
</script>
</body></html>'''

out = BASE / 'output/debug_align.html'
out.write_text(html, encoding='utf-8')
print(f'\n→ {out}')
print('用浏览器打开这个文件，左=相机坐标 右=世界坐标')
print('拖拽旋转查看，滚轮缩放')
