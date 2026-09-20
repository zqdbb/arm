#!/usr/bin/env python3
"""
V12 家具重建完整管线
══════════════════════════════════════════════════════════════
  采集 → 自定义模型抠图 → 自动测量 → Hyper3D 生成 → Blender 缩放

使用方式:
  python3 scan_recon_v12.py --name chair       # 椅子
  python3 scan_recon_v12.py --name table       # 桌子
  python3 scan_recon_v12.py --skip-capture     # 跳过采集
  python3 scan_recon_v12.py --blender-host IP  # 远程 Blender

依赖:
  pip install pyrealsense2 ultralytics opencv-python numpy

外部依赖:
  Blender 4.x + BlenderMCP addon（运行在 Blender 中）
  Hyper3D API key（在 Blender addon 面板配置）
  Y200RA60 电动转台 (/dev/ttyUSB0)
  自定义抠图模型: custom_model/weights/best.pt（从 train.zip 解压）
"""

import cv2, json, math, time, os, sys, socket, shutil, argparse
import numpy as np
from pathlib import Path

# ─── 常量 ────────────────────────────────────────────
BASE = Path(__file__).parent
OUT = BASE / 'output/v12/chair'  # 在 main() 中根据 --name 重新赋值
W, H = 640, 480       # D435i 支持的标准分辨率
W_OUT, H_OUT = 500, 500  # 模型训练分辨率（采集后缩放为此尺寸）
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG  # 72
N_AVG = 5
LASER_POWER = 150

# 抠图方案: 默认 SAM（大模型），可用 --no-sam 回退
SAM_MODEL = BASE / 'sam_vit_b_01ec64.pth'
DETECT_MODEL = BASE / 'detect_model/weights/best.pt'  # train37 检测模型
CUSTOM_MODEL = BASE / 'custom_model/weights/best.pt'  # train.zip 分割模型（回退用）

# Blender 连接参数（与 BlenderMCP addon 保持一致）
BLENDER_HOST = os.environ.get('BLENDER_HOST', 'localhost')
BLENDER_PORT = int(os.environ.get('BLENDER_PORT', '9876'))


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 0: Blender TCP 通信                               ║
# ╚══════════════════════════════════════════════════════════╝

class BlenderClient:
    """通过 TCP socket 与 BlenderMCP addon 通信.

    Blender addon 监听 localhost:9876，接收 JSON 命令，
    格式与 blender_mcp Python 包完全相同。
    """

    def __init__(self, host=BLENDER_HOST, port=BLENDER_PORT, timeout=300.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock = None

    def connect(self):
        if self._sock:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        print(f'  已连接 Blender: {self.host}:{self.port}')

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def send(self, cmd_type: str, params: dict = None) -> dict:
        """发送命令到 Blender，返回 result dict."""
        if not self._sock:
            self.connect()

        payload = json.dumps({
            "type": cmd_type,
            "params": params or {}
        }).encode('utf-8')

        self._sock.sendall(payload)

        # 接收完整响应（可能分块）
        chunks = []
        self._sock.settimeout(180.0)
        while True:
            try:
                chunk = self._sock.recv(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                try:
                    data = b''.join(chunks)
                    json.loads(data.decode('utf-8'))
                    break
                except json.JSONDecodeError:
                    continue
            except socket.timeout:
                break

        data = b''.join(chunks)
        if not data:
            raise Exception("Blender 无响应")

        resp = json.loads(data.decode('utf-8'))
        if resp.get("status") == "error":
            raise Exception(resp.get("message", "Blender 错误"))
        return resp.get("result", resp)

    def execute_blender_code(self, code: str) -> str:
        """在 Blender 中执行 Python 代码."""
        result = self.send("execute_code", {"code": code})
        return result.get("result", str(result))

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 1: 采集                                           ║
# ╚══════════════════════════════════════════════════════════╝

def do_capture(calib):
    import pyrealsense2 as rs
    from turntable import TurntableController

    print('\n' + '=' * 60)
    print(f'  Step 1/5  采集: {STEP_DEG}deg × {N_FRAMES} 帧 ({W}×{H} → {W_OUT}×{H_OUT})')
    print('=' * 60)

    tt = TurntableController(port='/dev/ttyUSB0')
    tt.open()
    print('转台已连接')
    tt.move_absolute(0, speed=10000)
    tt.wait_stop(timeout=30)
    print('转台归零')

    # 启动相机
    pipe = rs.pipeline()
    for attempt in range(3):
        try:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
            profile = pipe.start(cfg)
            time.sleep(0.5)
            pipe.wait_for_frames(timeout_ms=5000)
            break
        except RuntimeError as e:
            print(f'  启动尝试 {attempt+1}/3: {e}')
            try:
                pipe.stop()
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.0)
                pipe = rs.pipeline()
    else:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) > 0:
            try:
                devices[0].hardware_reset()
                time.sleep(2.0)
            except Exception as e:
                print(f'  硬件复位失败: {e}')
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        profile = pipe.start(cfg)
        time.sleep(0.5)

    align = rs.align(rs.stream.color)
    ds = profile.get_device().first_depth_sensor()
    if ds.supports(rs.option.laser_power):
        ds.set_option(rs.option.laser_power, LASER_POWER)
    dscale = ds.get_depth_scale()

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # 保存原始内参 + 裁剪缩放后的内参
    crop_size = min(W, H)
    x0 = (W - crop_size) // 2
    scale = W_OUT / crop_size
    calib['fx_full'] = intr.fx * scale
    calib['fy_full'] = intr.fy * scale
    calib['ppx_full'] = (intr.ppx - x0) * scale
    calib['ppy_full'] = (intr.ppy - (H - crop_size) // 2) * scale
    calib['width_full'] = W_OUT
    calib['height_full'] = H_OUT
    calib['depth_scale'] = dscale

    calib_path = OUT / 'calibrate.json'
    with open(calib_path, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'标定已保存: {calib_path}')
    print(f'  fx={intr.fx:.1f} fy={intr.fy:.1f} {intr.width}x{intr.height}')

    (OUT / 'color').mkdir(exist_ok=True)
    (OUT / 'depth').mkdir(exist_ok=True)

    t0 = time.time()
    for i in range(N_FRAMES):
        deg = i * STEP_DEG
        tt.move_absolute(deg, speed=5000)
        tt.wait_stop(timeout=30)

        d_sum = np.zeros((H, W), np.float64)
        color_full = None
        n_good = 0
        for _ in range(N_AVG):
            frames = pipe.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            df = aligned.get_depth_frame()
            cf = aligned.get_color_frame()
            if not df or not cf:
                continue
            d_sum += np.asanyarray(df.get_data()).astype(np.float64)
            if color_full is None:
                color_full = np.asanyarray(cf.get_data())
            n_good += 1

        if n_good == 0:
            print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  无数据')
            continue

        # 直接 resize 完整画面，不裁剪（保留椅子全貌）
        color_out = cv2.resize(color_full, (W_OUT, H_OUT), interpolation=cv2.INTER_LINEAR)
        depth_mm = (d_sum / n_good) * dscale * 1000.0
        depth_out = cv2.resize(depth_mm, (W_OUT, H_OUT), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.png'), color_out)
        cv2.imwrite(str(OUT / 'color' / f'{i:03d}.jpg'), color_out,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUT / 'depth' / f'{i:03d}.png'), depth_out.astype(np.uint16))

        dvals = depth_out[depth_out > 0]
        el = time.time() - t0
        print(f'  [{i+1}/{N_FRAMES}] {deg:3d}deg  '
              f'valid:{len(dvals)//1000}k  med={np.median(dvals):.0f}mm  {el:.0f}s')

    pipe.stop()
    tt.close()
    print(f'采集完成 ({time.time()-t0:.0f}s)\n')


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 2: YOLO mask + 自动测量                           ║
# ╚══════════════════════════════════════════════════════════╝

def measure_chair_from_masks(masks_dict, calib):
    """从 YOLO mask + 深度图推算椅子真实尺寸（像素-距离法）.

    正面帧（0°/180°）的像素宽 → 实际宽度，侧面帧（90°/270°）的像素宽 → 实际深度.
    """
    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    dscale = calib.get('depth_scale', 0.001)
    n_frames = calib.get('n_frames', 72)
    depth_dir = OUT / 'depth'

    # 正面 = 0° 和 180°, 侧面 = 90° 和 270°
    front_indices = [0, n_frames // 2]                     # 0°, 180°
    side_indices = [n_frames // 4, n_frames * 3 // 4]      # 90°, 270°

    def _dist_m(i, ys, xs):
        """返回指定帧 mask 区域的深度中位数（米）."""
        dep = cv2.imread(str(depth_dir / f'{i:03d}.png'), cv2.IMREAD_UNCHANGED)
        if dep is None:
            return None
        dvals = dep[ys, xs].astype(float)
        valid = dvals > 0
        if valid.sum() < 100:
            return None
        dist = float(np.median(dvals[valid])) * dscale
        if dist < 0.1:
            return None
        return dist

    heights_m = []
    widths_m = []
    depths_m = []

    for i, mask in masks_dict.items():
        if mask is None or mask.max() == 0:
            continue
        ys, xs = np.where(mask > 128)
        if len(ys) < 100:
            continue

        dist = _dist_m(i, ys, xs)
        if dist is None:
            continue

        h_px = ys.max() - ys.min()
        w_px = xs.max() - xs.min()
        h_m = h_px * dist / fy
        w_m = w_px * dist / fx
        heights_m.append(h_m)

        # 宽度 / 深度：只在正面/侧面帧用像素宽测量
        if i in front_indices:
            widths_m.append(w_m)
        elif i in side_indices:
            depths_m.append(w_m)

    if not heights_m:
        return None

    heights_m = np.array(heights_m)

    chair_h = float(np.median(heights_m))
    chair_w = float(np.median(widths_m)) if widths_m else float(np.median(heights_m)) * 0.5
    chair_d = float(np.median(depths_m)) if depths_m else float(np.median(heights_m)) * 0.45

    result = {
        'height_m': round(chair_h, 3),
        'width_m': round(chair_w, 3),
        'depth_m': round(chair_d, 3),
        'n_measured': len(heights_m),
        'n_width': len(widths_m),
        'n_depth': len(depths_m),
    }

    print(f'\n  ┌─ 自动测量 (像素-距离法)')
    print(f'  ├─ 高度: {result["height_m"]*100:.1f}cm ({result["n_measured"]} 帧中位数)')
    print(f'  ├─ 宽度: {result["width_m"]*100:.1f}cm (正面帧 {front_indices})')
    print(f'  └─ 深度: {result["depth_m"]*100:.1f}cm (侧面帧 {side_indices})')

    calib['chair_height_m'] = result['height_m']
    calib['chair_width_m'] = result['width_m']
    calib['chair_depth_m'] = result['depth_m']
    return result


def do_yolo_and_measure(calib, use_sam=True):
    """抠图 + 自动测量 + 选 4 正交视角 RGBA 输出.

    use_sam=True:  train37 检测 → SAM 大模型抠图（默认，推荐）
    use_sam=False: 自定义分割模型 train.zip（回退方案）
    """
    from ultralytics import YOLO

    cW = calib.get('width_full', W_OUT)
    cH = calib.get('height_full', H_OUT)
    step = calib.get('step_deg', STEP_DEG)

    color_files = sorted((OUT / 'color').glob('*.png'))
    if len(color_files) < 36:
        print(f'图片不足: {len(color_files)} 张')
        return None, None

    # ── YOLOv8n-seg + SAM 精修 ──
    print('=' * 60)
    print('  Step 2/5  YOLOv8n-seg + SAM 抠图 + 自动测量')
    print('=' * 60)

    yolo_path = str(BASE / 'yolov8n-seg.pt')
    if not Path(yolo_path).exists():
        print(f'YOLO 模型未找到: {yolo_path}')
        sys.exit(1)

    print(f'  加载 YOLO: {yolo_path}')
    yolo_seg = YOLO(yolo_path)

    # 加载 SAM
    sam_path = str(SAM_MODEL)
    sam_ok = Path(sam_path).exists()
    sam_predictor = None
    if sam_ok:
        print(f'  加载 SAM: {sam_path}')
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=sam_path)
        sam_predictor = SamPredictor(sam)
    else:
        print(f'  SAM 未找到，跳过精修: {sam_path}')

    masks_dict = {}
    detected = 0
    view_frames = {0, 18, 36, 54}
    view_data = {}  # (img_rgb, bbox) for SAM refinement

    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None:
            continue

        results = yolo_seg(img, verbose=False, classes=[56])
        r = results[0]

        mask_binary = np.zeros((cH, cW), dtype=np.uint8)
        if r.masks is not None and len(r.boxes) > 0:
            # 筛出所有 chair 检测，排除过大框（>25%面积=背景误检）
            chairs = []
            img_area = cW * cH
            for j in range(len(r.boxes)):
                if r.names[int(r.boxes.cls[j])] == 'chair':
                    box = r.boxes.xyxy[j].cpu().numpy()
                    conf = float(r.boxes.conf[j])
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    area_pct = area / img_area
                    if conf >= 0.25 and area_pct <= 0.25:
                        chairs.append((conf, j, box))

            if chairs:
                # 选置信度最高的框
                chairs.sort(key=lambda c: c[0], reverse=True)
                best_conf, best_idx, best_box = chairs[0]
                detected += 1

                mask_raw = r.masks.data[best_idx].cpu().numpy()
                if mask_raw.shape != (cH, cW):
                    mask_raw = cv2.resize(mask_raw, (cW, cH))
                mask_binary = (mask_raw > 0.5).astype(np.uint8) * 255

                # 视角帧保存数据供 SAM 精修
                if i in view_frames:
                    view_data[i] = (img, best_box)

        masks_dict[i] = mask_binary

        if i < 3 or (i + 1) % 18 == 0:
            mask_k = (mask_binary > 0).sum() // 1000
            print(f'  [{i+1}/{len(color_files)}] {i*step:3d}deg  '
                  f'mask={mask_k}k px  {"✓" if mask_binary.any() else "✗"}')

    # SAM 精修 4 视角帧
    if sam_predictor and view_data:
        print(f'\n  SAM 精修 {len(view_data)} 个视角帧...')
        for idx in sorted(view_data.keys()):
            img, box = view_data[idx]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(img_rgb)
            masks_sam, scores, _ = sam_predictor.predict(
                box=box[None, :], multimask_output=False
            )
            mask_binary = np.zeros((cH, cW), dtype=np.uint8)
            for m, s in zip(masks_sam, scores):
                if s > 0.5:
                    mask_binary[m] = 255
            if mask_binary.any():
                masks_dict[idx] = mask_binary
            print(f'    视角 {idx*step:3d}° (frame {idx:03d}): '
                  f'SAM mask={(mask_binary>0).sum()//1000}k px')

    print(f'\n  检出: {detected}/{len(color_files)}')

    # 自动测量
    chair_dims = measure_chair_from_masks(masks_dict, calib)

    # 选 2 个正交视角（0° / 90°），生成 RGBA 抠图
    indices = [0, 18]
    view_dir = OUT / 'views'
    view_dir.mkdir(exist_ok=True)

    views = []
    for idx in indices:
        src = OUT / 'color' / f'{idx:03d}.png'
        if not src.exists():
            continue

        img = cv2.imread(str(src))
        mask = masks_dict.get(idx)
        has_mask = mask is not None and mask.max() > 0

        rgba_path = view_dir / f'view_{idx:03d}.png'
        if has_mask:
            # 合成白底（Hyper3D/Rodin 训练数据是白底图，非透明背景）
            alpha_f = (mask > 128).astype(np.float32)
            white_bg = np.full_like(img, 255, dtype=np.uint8)
            composited = (img * alpha_f[..., None]
                          + white_bg * (1.0 - alpha_f[..., None])).astype(np.uint8)
            cv2.imwrite(str(rgba_path), composited)
        else:
            cv2.imwrite(str(rgba_path), img)

        deg = idx * step
        views.append({'index': idx, 'degree': deg, 'path': str(rgba_path),
                      'has_mask': has_mask})
        print(f'  视角 {deg:3d}° (frame {idx:03d}) {"✓ 抠图" if has_mask else "✗ 原图"}')

    # 保存测量结果
    if chair_dims:
        with open(OUT / 'chair_dims.json', 'w') as f:
            json.dump(chair_dims, f, indent=2)
        print(f'  尺寸已保存: {OUT / "chair_dims.json"}')

    return masks_dict, views, chair_dims


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 3: Hyper3D 生成 (通过 Blender)                    ║
# ╚══════════════════════════════════════════════════════════╝

def do_hyper3d_generate(blender: BlenderClient, views: list, chair_dims: dict):
    """通过 Blender addon 调用 Hyper3D 生成 3D 模型."""
    import base64

    print('=' * 60)
    print('  Step 3/5  Hyper3D 生成 3D 模型')
    print('=' * 60)

    # 检查 Hyper3D 是否可用
    status = blender.send("get_hyper3d_status")
    print(f'  Hyper3D 状态: {status.get("message", status)}')
    if not status.get("enabled", False):
        raise RuntimeError("Hyper3D 未启用，请在 Blender addon 面板配置 API key")

    # 准备图片
    view_paths = [v['path'] for v in views if v['has_mask']]
    if len(view_paths) < 2:
        # 至少用全部 4 张
        view_paths = [v['path'] for v in views]

    print(f'  提交 {len(view_paths)} 张图片到 Hyper3D...')

    # 编码图片为 base64（MAIN_SITE 模式）
    images = []
    for vp in view_paths:
        with open(vp, 'rb') as f:
            suffix = Path(vp).suffix
            images.append((suffix, base64.b64encode(f.read()).decode('ascii')))

    # 可选 bbox 条件
    bbox = None
    if chair_dims:
        # 归一化到 100
        h, w, d = chair_dims['height_m'], chair_dims['width_m'], chair_dims['depth_m']
        max_dim = max(h, w, d)
        bbox = [int(w / max_dim * 100), int(d / max_dim * 100), int(h / max_dim * 100)]

    result = blender.send("create_rodin_job", {
        "text_prompt": None,
        "images": images,
        "bbox_condition": bbox,
    })

    print(f'  Hyper3D 任务已提交: {json.dumps(result, indent=2)}')

    # 轮询状态（subscription_key 在 result.jobs 里）
    subscription_key = result.get("jobs", {}).get("subscription_key")
    task_uuid = result.get("uuid")  # 注意: 字段名是 "uuid" 不是 "task_uuid"

    print(f'  等待生成完成...')
    for poll_i in range(120):  # 最多等 10 分钟
        time.sleep(5)
        if subscription_key:
            status_resp = blender.send("poll_rodin_job_status",
                                       {"subscription_key": subscription_key})
        else:
            status_resp = blender.send("poll_rodin_job_status",
                                       {"request_id": result.get("request_id")})

        # 解析状态列表
        if isinstance(status_resp, list):
            status_list = [s.get("status", "") if isinstance(s, dict) else str(s)
                          for s in status_resp]
        elif isinstance(status_resp, dict) and "status_list" in status_resp:
            status_list = status_resp["status_list"]
        elif isinstance(status_resp, dict) and "status" in status_resp:
            status_list = [status_resp["status"]]
        else:
            status_list = []

        print(f'  [{poll_i+1}] {json.dumps(status_resp, ensure_ascii=False)[:200]}')

        if status_list:
            if all(s == "Done" for s in status_list):
                print('  生成完成!')
                break
            if any(s == "Failed" for s in status_list):
                raise RuntimeError(f"Hyper3D 生成失败: {status_list}")
            if any(s == "Canceled" for s in status_list):
                raise RuntimeError(f"Hyper3D 生成被取消: {status_list}")
    else:
        raise TimeoutError("Hyper3D 生成超时")

    # 导入前清理旧模型，避免 Blender 自动重命名为 .001
    object_name = "chair_v12"
    print('  清理旧模型...')
    blender.execute_blender_code(f'''
import bpy
# 收集要删的对象名（避免遍历时删除导致 StructRNA 错误）
to_remove = [o.name for o in bpy.data.objects
             if o.type == 'MESH' and o.name.startswith("{object_name}")]
for name in to_remove:
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)
        print(f"    删除: {{name}}")
if not to_remove:
    print("    无旧模型")
print("    旧模型已清理")
''')

    # 导入模型
    print('  导入模型...')
    import_params = {"name": object_name}
    if subscription_key:
        import_params["task_uuid"] = task_uuid
    else:
        import_params["request_id"] = result.get("request_id")
    import_result = blender.send("import_generated_asset", import_params)

    print(f'  导入结果: {json.dumps(import_result, ensure_ascii=False)[:300]}')

    return object_name


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 4: Blender 清理                                   ║
# ╚══════════════════════════════════════════════════════════╝

def _deep_clean_turntable(blender, object_name):
    """检测并切除底部转台底盘（Z聚类法）.

    思路: 转台底盘是模型底部同Z高度的巨大水平面。
    按Z分桶统计水平面总面积 → 面积最大的Z桶 = 底盘 → bisect切除。
    用Z聚类而非BFS，避免沿着共享边爬到腿上。
    """
    print('  切掉转台底盘（Z聚类检测）...')
    code = f'''
import bpy, bmesh

obj = bpy.data.objects.get("{object_name}")
mesh = obj.data
bm = bmesh.new()
bm.from_mesh(mesh)
bm.faces.ensure_lookup_table()

z_vals = [v.co.z for v in bm.verts]
z_min, z_max = min(z_vals), max(z_vals)
z_range = z_max - z_min
z_bottom = z_min + z_range * 0.20
print(f"  Z: {{z_min*100:.1f}} ~ {{z_max*100:.1f}}cm, 底部={{z_bottom*100:.1f}}cm")

# 1. 按Z分桶，统计每个高度层的水平面总面积
n_bins = max(20, int(z_range * 1000 / 2))  # 每2mm一个桶
bin_step = (z_bottom - z_min) / max(n_bins, 1) if z_bottom > z_min else 0.001
if bin_step < 0.0005:
    bin_step = 0.001

# 收集底部区域内所有水平面的 (z, area)
h_faces = []
for f in bm.faces:
    if abs(f.normal.z) > 0.85:
        zf = sum(v.co.z for v in f.verts) / len(f.verts)
        if zf < z_bottom:
            h_faces.append((zf, f.calc_area()))

if len(h_faces) < 5:
    print("  底部水平面不足5个，无底盘，跳过")
    bm.free()
    raise RuntimeError("SKIP_TURNTABLE")

# 按Z分桶
bins = {{}}
for zf, area in h_faces:
    bi = int((zf - z_min) / bin_step)
    bins[bi] = bins.get(bi, 0) + area

# 找面积最大的桶
best_bi = max(bins, key=bins.get)
best_z_center = z_min + (best_bi + 0.5) * bin_step
best_area = bins[best_bi]

# 底盘面积应显著大于其他桶（至少是第二名的2倍）
sorted_areas = sorted(bins.values(), reverse=True)
if len(sorted_areas) >= 2 and sorted_areas[0] < sorted_areas[1] * 2:
    print(f"  底盘面积优势不足 ({{sorted_areas[0]*10000:.0f}} vs {{sorted_areas[1]*10000:.0f}}cm²)，可能无底盘，跳过")
    bm.free()
    raise RuntimeError("SKIP_TURNTABLE")

print(f"  底盘Z桶: Z≈{{best_z_center*100:.1f}}cm, 水平面积={{best_area*10000:.0f}}cm²")

# 2. 在该桶顶部 bisect 切除
cut_z = z_min + (best_bi + 1) * bin_step + 0.0003
geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
result = bmesh.ops.bisect_plane(
    bm, geom=geom, dist=0.0001,
    plane_co=(0, 0, cut_z), plane_no=(0, 0, 1),
    clear_outer=True, clear_inner=False
)
n_removed = len([g for g in result['geom_cut'] if isinstance(g, bmesh.types.BMFace)])
print(f"  bisect 切除 {{n_removed}} 面 (Z < {{cut_z*100:.1f}}cm)")

# 3. 填洞
bm.faces.ensure_lookup_table()
boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
if boundary_edges:
    try:
        bmesh.ops.triangle_fill(bm, edges=boundary_edges)
        print(f"  填补 {{len(boundary_edges)}} 条边界边")
    except Exception:
        pass

bm.to_mesh(mesh); bm.free(); obj.data.update()

zs_final = [v.co.z for v in mesh.vertices]
print(f"  清理后 Z: {{min(zs_final)*100:.1f}} ~ {{max(zs_final)*100:.1f}}cm")
print(f"  顶点: {{len(mesh.vertices)}}, 面: {{len(mesh.polygons)}}")
'''
    return blender.execute_blender_code(code)


def _deep_clean_center_leg(blender, object_name):
    """深度清理: 去除中间多余腿 (仅 --clean 时调用)."""
    print('  [deep] 去除中间多余腿...')
    code = f'''
import bpy, bmesh
obj = bpy.data.objects.get("{object_name}")
mesh = obj.data
bm = bmesh.new()
bm.from_mesh(mesh); bm.faces.ensure_lookup_table()

z_min = min(v.co.z for v in bm.verts)
z_range = max(v.co.z for v in bm.verts) - z_min
z_thr = z_min + z_range*0.15

bv = [v for v in bm.verts if v.co.z < z_thr and v.link_faces]
if bv:
    xs_b = [v.co.x for v in bv]; ys_b = [v.co.y for v in bv]
    cx = (max(xs_b)+min(xs_b))/2; cy = (max(ys_b)+min(ys_b))/2
    x_thr = (max(xs_b)-min(xs_b))*0.20; y_thr = (max(ys_b)-min(ys_b))*0.20
    td = []
    for f in bm.faces:
        xf = sum(v.co.x for v in f.verts)/len(f.verts)
        yf = sum(v.co.y for v in f.verts)/len(f.verts)
        zf = sum(v.co.z for v in f.verts)/len(f.verts)
        if (abs(xf-cx)<x_thr and abs(yf-cy)<y_thr
            and zf<z_thr and abs(f.normal.z)<0.85 and zf>z_min+0.0005):
            td.append(f)
    for f in td: bm.faces.remove(f)
    print(f"  删{{len(td)}}面")

bm.to_mesh(mesh); bm.free(); obj.data.update()
print(f"  完成")
'''
    return blender.execute_blender_code(code)


def do_blender_cleanup(blender: BlenderClient, object_name: str, chair_dims: dict):
    """清理模型: 融合碎片 → 去悬浮 → 翻转修正 → 统一缩放 → 平移.

    分两步执行，避免 Blender StructRNA 引用失效."""
    print('=' * 60)
    print('  Step 4/5  Blender 模型清理（融合 + 去碎片 + 翻转修正 + 缩放）')
    print('=' * 60)

    ch = chair_dims.get('height_m', 0.09)
    obj_path = str((OUT / f'{object_name}.obj').resolve())

    # ── 阶段 A: 找对象 + 删多余 + Remesh 融合 ──
    code_stage_a = f'''
import bpy, bmesh, os

# 1. 找到目标对象
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    candidates = [o for o in bpy.data.objects
                  if o.type == 'MESH' and o.name.startswith("{object_name}")]
    if candidates:
        candidates.sort(key=lambda o: len(o.data.vertices), reverse=True)
        obj = candidates[0]
        obj.name = "{object_name}"
if obj is None:
    obj_path = "{obj_path}"
    if os.path.exists(obj_path):
        print(f"  从磁盘导入: {{obj_path}}")
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj = bpy.context.selected_objects[0]
        obj.name = "{object_name}"
    else:
        raise RuntimeError(f"未找到模型")

print(f"  源对象: {{obj.name}} ({{len(obj.data.vertices)}}v, {{len(obj.data.polygons)}}f)")

# 2. 删除其他 mesh
obj_name = obj.name
names_to_remove = [o.name for o in bpy.data.objects
                   if o.type == 'MESH' and o.name != obj_name]
for n in names_to_remove:
    o = bpy.data.objects.get(n)
    if o:
        bpy.data.objects.remove(o, do_unlink=True)
        print(f"  删除多余: {{n}}")

# 3. 先 Remesh 融合（把邻近的分离部件融合成水密 mesh）
bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

vs = obj.data.vertices
xs = [v.co.x for v in vs]; ys = [v.co.y for v in vs]; zs = [v.co.z for v in vs]
max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
voxel_size = max(max_dim / 200, 0.0005)
print(f"  Remesh: voxel={{voxel_size*100:.3f}}cm  max_dim={{max_dim*100:.1f}}cm")

mod = obj.modifiers.new(name="FuseRemesh", type='REMESH')
mod.mode = 'VOXEL'
mod.voxel_size = voxel_size
bpy.ops.object.modifier_apply(modifier="FuseRemesh")
print(f"  Remesh 后: {{len(obj.data.vertices)}}v")

# 4. 删除不连通的悬浮碎片（通过中心顶点选连通区）
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')

# 找距离几何中心最近的顶点（大概率在主体上）
bm_edit = bmesh.from_edit_mesh(obj.data)
vs_edit = [v for v in bm_edit.verts]
cx = sum(v.co.x for v in vs_edit) / len(vs_edit)
cy = sum(v.co.y for v in vs_edit) / len(vs_edit)
cz = sum(v.co.z for v in vs_edit) / len(vs_edit)
closest = min(vs_edit, key=lambda v: (v.co.x-cx)**2 + (v.co.y-cy)**2 + (v.co.z-cz)**2)
closest.select = True

bpy.ops.mesh.select_linked(delimit=set())
bpy.ops.mesh.select_all(action='INVERT')
# 如果有面被选中，删除它们
n_sel = sum(1 for v in bm_edit.verts if v.select)
if n_sel > 0:
    bpy.ops.mesh.delete(type='VERT')
    print(f"  删除悬浮碎片: {{n_sel}}v")
else:
    print(f"  无悬浮碎片")
bpy.ops.object.mode_set(mode='OBJECT')
print(f"  主体: {{len(obj.data.vertices)}}v")

print("  Stage A 完成")
'''

    result = blender.execute_blender_code(code_stage_a)
    print(f'  {result}')

    # ── 阶段 B: 翻转 + 座面校准 + 平滑 + 缩放 + 平移 ──
    code_stage_b = f'''
import bpy, bmesh, math
from mathutils import Vector, Matrix

obj = bpy.data.objects.get("{object_name}")
if obj is None:
    raise RuntimeError("Stage A 后对象丢失")

bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

# ── 1. 翻转检测：用 Z 密度分布 ──
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()

z_vals = [v.co.z for v in bm.verts]
z_min, z_max = min(z_vals), max(z_vals)
z_span = z_max - z_min

n_bands = 10
band_counts = [0] * n_bands
for z in z_vals:
    bi = int((z - z_min) / z_span * n_bands)
    if bi >= n_bands: bi = n_bands - 1
    if bi < 0: bi = 0
    band_counts[bi] += 1

max_band = band_counts.index(max(band_counts))
print(f"  Z 分布: {{[(i, band_counts[i]) for i in range(n_bands)]}}")
print(f"  最密层: band={{max_band}}/{{n_bands-1}} (0=底面, {{n_bands-1}}=顶面)")

if max_band <= 2 or max_band >= 7:
    print(f"  ! 检测到翻转，修正...")
    for v in bm.verts:
        v.co.z = z_max - (v.co.z - z_min)
    # 重算 Z 范围
    z_vals = [v.co.z for v in bm.verts]
    z_min, z_max = min(z_vals), max(z_vals)
    z_span = z_max - z_min
    print(f"  翻转完成")
else:
    print(f"  朝向正常")

# ── 2. 座面水平校准 ──
# 找所有近水平朝上的面，按 Z 分组，面积最大的 Z 组 = 座面
h_faces = []  # (z_cm, area, normal)
for f in bm.faces:
    if f.normal.z > 0.7:
        z_avg = sum(v.co.z for v in f.verts) / len(f.verts)
        h_faces.append((z_avg, f.calc_area(), f.normal.copy()))

if h_faces:
    # 按每厘米分组
    z_groups = {{}}
    for z, area, n in h_faces:
        bi = int(z * 100)
        if bi not in z_groups:
            z_groups[bi] = {{'area': 0.0, 'normal_sum': Vector((0,0,0))}}
        z_groups[bi]['area'] += area
        z_groups[bi]['normal_sum'] += n * area

    best_bi = max(z_groups, key=lambda k: z_groups[k]['area'])
    seat = z_groups[best_bi]
    avg_n = seat['normal_sum'] / seat['area'] if seat['area'] > 0 else Vector((0,0,1))
    avg_n.normalize()

    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, avg_n.z))))
    print(f"  座面候选: Z≈{{best_bi}}cm, 面积={{seat['area']*10000:.0f}}cm², 倾斜={{tilt_deg:.1f}}°")

    if tilt_deg > 5:
        rot_axis = avg_n.cross(Vector((0, 0, 1)))
        if rot_axis.length > 0.0001:
            rot_axis.normalize()
            # 几何中心
            cx = sum(v.co.x for v in bm.verts) / len(bm.verts)
            cy = sum(v.co.y for v in bm.verts) / len(bm.verts)
            cz = sum(v.co.z for v in bm.verts) / len(bm.verts)
            rot_mat = Matrix.Rotation(math.radians(tilt_deg), 3, rot_axis)
            bmesh.ops.rotate(bm, verts=bm.verts[:],
                             cent=(cx, cy, cz), matrix=rot_mat)
            print(f"  座面已校准（旋转 {{tilt_deg:.1f}}°）")
    else:
        print(f"  座面已水平（倾斜 {{tilt_deg:.1f}}° < 5°）")
else:
    print(f"  未找到座面水平面，跳过校准")

# 写回 mesh，进入编辑模式做平滑
bm.to_mesh(obj.data)
bm.free()
obj.data.update()

# ── 3. 表面平滑去噪 ──
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=3)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"  平滑完成 (factor=0.5, repeat=3)")

# ── 3.5. 去多余腿 ──
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()

zs_leg = [v.co.z for v in bm.verts]
z_lo = min(zs_leg)
z_span_leg = max(zs_leg) - z_lo
z_cut = z_lo + z_span_leg * 0.25

leg_faces = []
for f in bm.faces:
    cx = sum(v.co.x for v in f.verts) / len(f.verts)
    cy = sum(v.co.y for v in f.verts) / len(f.verts)
    cz = sum(v.co.z for v in f.verts) / len(f.verts)
    if cz < z_cut:
        leg_faces.append((f, cx, cy))

print(f"  底部 Z<{{z_cut*100:.1f}}cm: {{len(leg_faces)}} 个面")

if len(leg_faces) >= 10:
    import numpy as np
    pts = np.array([[cx, cy] for _, cx, cy in leg_faces])
    eps = z_span_leg * 0.12
    n_pts = len(pts)
    visited = np.zeros(n_pts, dtype=bool)
    clusters = []
    for pi in range(n_pts):
        if visited[pi]:
            continue
        cl = [pi]; q = [pi]; visited[pi] = True
        while q:
            j = q.pop(0)
            dists = np.sqrt((pts[:, 0] - pts[j, 0])**2 + (pts[:, 1] - pts[j, 1])**2)
            for k in np.where((dists < eps) & ~visited)[0]:
                visited[k] = True; q.append(k); cl.append(k)
        clusters.append(cl)

    clusters.sort(key=len, reverse=True)
    keep_idx = set()
    for cl in clusters[:4]:
        for i in cl:
            keep_idx.add(i)

    to_del = [leg_faces[i][0] for i in range(n_pts) if i not in keep_idx]
    for f in to_del:
        bm.faces.remove(f)
    print(f"  去腿: {{len(clusters)}}簇→保留前4, 删{{len(to_del)}}面")

    bm.faces.ensure_lookup_table()
    be = [e for e in bm.edges if len(e.link_faces) == 1]
    if be:
        try:
            bmesh.ops.triangle_fill(bm, edges=be)
        except Exception:
            pass
else:
    print(f"  底部面不足，跳过去腿")

bm.to_mesh(obj.data); bm.free(); obj.data.update()

# ── 3.6. 靠背矫直 ──
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()

zs_b = [v.co.z for v in bm.verts]
z_lo_b, z_hi_b = min(zs_b), max(zs_b)
z_rng = z_hi_b - z_lo_b
z_back = z_lo_b + z_rng * 0.52
z_seat_lo = z_lo_b + z_rng * 0.20
z_seat_hi = z_lo_b + z_rng * 0.48

# 靠背 X 居中：座面水平校准已处理整体倾斜，此处只做 X 平移对齐
seat_v = [(v.co.x, v.co.y) for v in bm.verts if z_seat_lo < v.co.z < z_seat_hi and v.link_faces]
back_v_all = [(v.co.x, v.co.y, v.co.z) for v in bm.verts if v.co.z > z_back and v.link_faces]

if len(seat_v) > 20 and len(back_v_all) > 20:
    seat_cx = sum(p[0] for p in seat_v) / len(seat_v)
    zs_back = [p[2] for p in back_v_all]
    z_back_lo = min(zs_back)
    back_h = max(zs_back) - z_back_lo

    if back_h > 0.003:
        bm.verts.ensure_lookup_table()
        z_bot_hi = z_back_lo + back_h * 0.25
        back_bot_xy = [(p[0], p[1]) for p in back_v_all if p[2] < z_bot_hi]
        if back_bot_xy:
            back_cx = sum(p[0] for p in back_bot_xy) / len(back_bot_xy)
            sx = seat_cx - back_cx
            if abs(sx) > 0.002:
                for v in bm.verts:
                    if v.co.z > z_back:
                        v.co.x += sx
                print(f"  靠背X居中: {{sx*100:.1f}}cm")
            else:
                print(f"  靠背X已居中 (偏移{{sx*100:.2f}}cm)")
        else:
            print(f"  靠背底部不足")
    else:
        print(f"  靠背太矮")
else:
    print(f"  靠背区域不足")

bm.to_mesh(obj.data); bm.free(); obj.data.update()

# ── 4. 统一缩放 + 平移（重新读 bmesh）──
bm = bmesh.new()
bm.from_mesh(obj.data)
bm.verts.ensure_lookup_table()

zs_cur = [v.co.z for v in bm.verts]
cur_h = max(zs_cur) - min(zs_cur)
target_h = {ch:.4f}
scale = target_h / cur_h if cur_h > 0.0005 else 1.0
print(f"  统一缩放: {{scale:.4f}} (高 {{cur_h*100:.1f}}→{{target_h*100:.1f}}cm)")
for v in bm.verts:
    v.co.x *= scale
    v.co.y *= scale
    v.co.z *= scale

# 平移到底面 Z=0
zs_after = [v.co.z for v in bm.verts]
z_floor = min(zs_after)
for v in bm.verts:
    v.co.z -= z_floor

bm.to_mesh(obj.data); bm.free()
obj.data.update()

# ── 5. 输出 ──
vs_f = obj.data.vertices
xs_f = [v.co.x for v in vs_f]; ys_f = [v.co.y for v in vs_f]; zs_f = [v.co.z for v in vs_f]
print(f"  最终: {{(max(xs_f)-min(xs_f))*100:.1f}}×{{(max(ys_f)-min(ys_f))*100:.1f}}×{{(max(zs_f)-min(zs_f))*100:.1f}}cm")
print(f"  顶点: {{len(vs_f)}}  面: {{len(obj.data.polygons)}}")
print("  Stage B 完成")
'''

    result = blender.execute_blender_code(code_stage_b)
    print(f'  {result}')

    # ── 验证 ──
    code_verify = f'''
import bpy
obj = bpy.data.objects.get("{object_name}")
if obj:
    vs = obj.data.vertices
    xs = [v.co.x for v in vs]; ys = [v.co.y for v in vs]; zs = [v.co.z for v in vs]
    print(f"  V12 完成: {{len(vs)}}v {{len(obj.data.polygons)}}f")
    print(f"  尺寸: {{(max(xs)-min(xs))*100:.1f}}×{{(max(ys)-min(ys))*100:.1f}}×{{(max(zs)-min(zs))*100:.1f}}cm")
    print(f"  Z: {{min(zs)*100:.1f}}~{{max(zs)*100:.1f}}cm")
else:
    print(f"  警告: 对象不存在")
'''
    result = blender.execute_blender_code(code_verify)
    print(f'  {result}')


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 5: 导出                                           ║
# ╚══════════════════════════════════════════════════════════╝

def do_export(blender: BlenderClient, object_name: str, furniture_name: str):
    """导出最终模型."""
    print('=' * 60)
    print('  Step 5/5  导出模型')
    print('=' * 60)

    export_path = str((OUT / f'{furniture_name}_v12.obj').resolve())
    export_path_ply = str((OUT / f'{furniture_name}_v12.ply').resolve())

    code_export = f'''
import bpy

# 确保只选中目标对象
bpy.ops.object.select_all(action='DESELECT')
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    raise RuntimeError("未找到对象: {object_name}")
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# 导出 OBJ
bpy.ops.wm.obj_export(
    filepath="{export_path}",
    export_selected_objects=True,
    export_materials=True,
    export_uv=True,
    export_normals=True,
)

# 导出 PLY
bpy.ops.wm.ply_export(
    filepath="{export_path_ply}",
    export_selected_objects=True,
)

print(f"  已导出: {export_path}")
print(f"  已导出: {export_path_ply}")
'''
    result = blender.execute_blender_code(code_export)
    print(f'  {result}')
    print(f'\n{"=" * 60}')
    print(f'  V12 完成!')
    with open(OUT / 'chair_dims.json') as f:
        dims = json.load(f)
    print(f'  椅子尺寸: {dims["height_m"]*100:.0f}×{dims["width_m"]*100:.0f}×{dims["depth_m"]*100:.0f}cm (高×宽×深)')
    print(f'  输出目录: {OUT}')
    print(f'  {"=" * 60}')


# ╔══════════════════════════════════════════════════════════╗
# ║  Main                                                    ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description='V12 家具重建完整管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 scan_recon_v12.py --name chair           # 椅子
  python3 scan_recon_v12.py --skip-capture         # 跳过采集
  python3 scan_recon_v12.py --blender-host 192.168.1.100
  python3 scan_recon_v12.py --no-blender            # 只做采集+测量
''')
    parser.add_argument('--name', default='chair',
                        help='家具名称，决定输出目录 (默认: chair)')
    parser.add_argument('--no-sam', action='store_true',
                        help='回退到自定义分割模型 (默认使用 SAM)')
    parser.add_argument('--skip-capture', action='store_true',
                        help='跳过采集步骤')
    parser.add_argument('--no-blender', action='store_true',
                        help='跳过所有 Blender 步骤')
    parser.add_argument('--skip-hyper3d', action='store_true',
                        help='跳过 Hyper3D AI 生成（只做清理+导出，需 Blender 中已有模型）')
    parser.add_argument('--blender-host', default=BLENDER_HOST,
                        help=f'Blender 主机地址 (默认: {BLENDER_HOST})')
    parser.add_argument('--blender-port', type=int, default=BLENDER_PORT,
                        help=f'Blender 端口 (默认: {BLENDER_PORT})')
    parser.add_argument('--step-deg', type=int, default=STEP_DEG,
                        help=f'每步角度 (默认: {STEP_DEG})')
    parser.add_argument('--turntable-port', default='/dev/ttyUSB0',
                        help='转台串口 (默认: /dev/ttyUSB0)')
    args = parser.parse_args()

    # 设置输出目录
    global OUT
    OUT = BASE / f'output/v12/{args.name}'
    OUT.mkdir(parents=True, exist_ok=True)

    step_deg = args.step_deg
    n_frames = 360 // step_deg
    blender_host = args.blender_host
    blender_port = args.blender_port

    print('=' * 60)
    print(f'  V12 家具重建管线 [{args.name}]')
    print(f'  自定义抠图模型 | 输出: {OUT}')
    print(f'  {step_deg}deg × {n_frames} 帧 | Blender: {blender_host}:{blender_port}')
    print('=' * 60)

    # ── 加载/初始化标定 ──
    calib_path = OUT / 'calibrate.json'
    ts_calib_path = BASE / 'output/calibrate.json'

    calib = None
    if calib_path.exists():
        with open(calib_path) as f:
            calib = json.load(f)
        if 'fx_full' in calib:
            print(f'已有 V12 标定: {calib_path}')
            if not args.skip_capture:
                resp = input('使用现有标定? [Y/n]: ').strip().lower()
                if resp == 'n':
                    calib_path.unlink()
                    print(f'已删除: {calib_path}')
                    calib = None
        else:
            calib = None

    if calib is None:
        if not args.skip_capture and ts_calib_path.exists():
            # 有转台标定但没 V12 标定 → 先跑标定工具
            resp = input(f'已有转台标定 {ts_calib_path}，是否重新标定? [y/N]: ').strip().lower()
            if resp == 'y':
                ts_calib_path.unlink()
                print('已删除旧转台标定')

        if not ts_calib_path.exists():
            print('\n启动标定工具 turntable_set_v2.py ...')
            print('操作: 空格锁定中心 → 确认椭圆正确 → 按 S 保存 → 按 Q 退出')
            import subprocess
            subprocess.run([sys.executable, str(BASE / 'turntable_set_v2.py')])
            if not ts_calib_path.exists():
                print('标定未保存，退出')
                sys.exit(1)

        with open(ts_calib_path) as f:
            ts_calib = json.load(f)
        print(f'读取转台标定: {ts_calib_path}')
        calib = {
            'R': ts_calib['R'], 't': ts_calib['t'],
            'radius_m': ts_calib['radius_m'],
            'cx': ts_calib.get('cx', 0.0), 'cy': ts_calib.get('cy', 0.0),
            'step_deg': STEP_DEG, 'n_frames': N_FRAMES,
        }

    # ── 采集 ──
    if not args.skip_capture:
        resp = input('\n开始采集? [Y/n]: ').strip().lower()
        if resp != 'n':
            do_capture(calib)
        else:
            print('跳过采集')
    else:
        print('跳过采集 (--skip-capture)')

    # ── 抠图 + 测量 ──
    masks_dict, views, chair_dims = do_yolo_and_measure(calib, use_sam=not args.no_sam)

    if chair_dims is None:
        print('警告: 自动测量失败，使用默认值 9×6×5cm')
        chair_dims = {'height_m': 0.09, 'width_m': 0.06, 'depth_m': 0.05}
        calib['chair_height_m'] = 0.09
        calib['chair_width_m'] = 0.06
        calib['chair_depth_m'] = 0.05

    if args.no_blender:
        print('\n跳过 Blender 步骤 (--no-blender)')
        print(f'数据已保存到: {OUT}')
        return

    # ── Blender 连接 ──
    blender = BlenderClient(host=BLENDER_HOST, port=BLENDER_PORT)
    try:
        blender.connect()

        # ── Hyper3D 生成 ──
        if args.skip_hyper3d:
            object_name = f"{args.name}_v12"
            print(f"\n跳过 Hyper3D 生成 (--skip-hyper3d)，使用已有模型: {object_name}")
        else:
            object_name = do_hyper3d_generate(blender, views, chair_dims)

        # ── 清理 ──
        do_blender_cleanup(blender, object_name, chair_dims)

        # ── 导出 ──
        do_export(blender, object_name, args.name)

        # ── 保存日志 ──
        log = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'V12_HYPER3D',
            'n_frames': N_FRAMES,
            'chair_dims': chair_dims,
            'output_dir': str(OUT),
        }
        with open(OUT / 'log_v12.json', 'w') as f:
            json.dump(log, f, indent=2)

    finally:
        blender.disconnect()


if __name__ == '__main__':
    main()
