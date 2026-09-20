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

import cv2, json, time, os, sys, socket, shutil, argparse, math, subprocess
import numpy as np
from pathlib import Path

# ─── 常量 ────────────────────────────────────────────
BASE = Path(__file__).parent
OUT = BASE / 'output/chair'  # 在 main() 中根据 --name 重新赋值
CAPTURE_ROOT = OUT
W, H = 640, 480       # D435i 支持的标准分辨率
W_OUT, H_OUT = 500, 500  # 模型训练分辨率（采集后缩放为此尺寸）
FPS = 15
STEP_DEG = 5
N_FRAMES = 360 // STEP_DEG  # 72
N_AVG = 5
LASER_POWER = 150

# 默认直接使用 YOLO segmentation；SAM 仅由 --sam-refine 显式启用
SAM_MODEL = BASE / 'sam_vit_b_01ec64.pth'
GENERIC_MODEL = BASE / 'yolov8n-seg.pt'
CUSTOM_MODEL = BASE / 'custom_model/weights/best.pt'

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

def archive_capture_dataset(output_dir, item_type):
    managed = [
        output_dir / 'color',
        output_dir / 'depth',
        output_dir / 'pcd_frames',
        output_dir / 'capture_manifest.json',
        output_dir / f'{item_type}_stitched.ply',
    ]
    existing = [path for path in managed if path.exists()]
    if not existing:
        return None
    archive_dir = output_dir / 'history' / time.strftime('%Y%m%d_%H%M%S')
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(archive_dir / path.name))
    print(f'  旧采集已归档: {archive_dir}')
    return archive_dir


def capture_dataset_exists(output_dir, item_type):
    return any(path.exists() for path in (
        output_dir / 'color',
        output_dir / 'depth',
        output_dir / 'pcd_frames',
        output_dir / 'capture_manifest.json',
        output_dir / f'{item_type}_stitched.ply',
    ))


def run_final_pipeline(item_type, turntable_port, skip_capture=False):
    command = [
        sys.executable,
        str(BASE / 'final.py'),
        '--name', item_type,
        '--no-visualize',
    ]
    if skip_capture:
        command.append('--skip-capture')
    else:
        command.extend(['--save-rgbd', '--turntable-port', turntable_port])
    print(f'  调用 final.py: {" ".join(command)}')
    subprocess.run(command, cwd=str(BASE), check=True)


def _legacy_manifest(output_dir):
    color_files = sorted((output_dir / 'color').glob('*.png'))
    depth_files = sorted((output_dir / 'depth').glob('*.png'))
    depth_by_stem = {p.stem: p for p in depth_files}
    frames = []
    for color_path in color_files:
        depth_path = depth_by_stem.get(color_path.stem)
        if depth_path is None:
            continue
        index = int(color_path.stem)
        frames.append({
            'index': index,
            'angle_deg': index * STEP_DEG,
            'color': str(color_path.relative_to(output_dir)),
            'depth': str(depth_path.relative_to(output_dir)),
            'point_cloud': f'pcd_frames/frame_{index:03d}.ply',
        })
    return {
        'schema_version': 0,
        'complete': bool(frames),
        'step_deg': STEP_DEG,
        'planned_frames': len(frames),
        'depth_scale_m_per_unit': 0.001,
        'color': {'width': W_OUT, 'height': H_OUT},
        'depth': {'width': W_OUT, 'height': H_OUT},
        'frames': frames,
    }


def validate_capture_dataset(output_dir, item_type, allow_legacy=False):
    manifest_path = output_dir / 'capture_manifest.json'
    if manifest_path.exists():
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        if not manifest.get('complete'):
            raise RuntimeError(f'采集未完成: {manifest_path}')
        if manifest.get('name') not in (None, item_type):
            raise RuntimeError(f'采集类别不匹配: {manifest.get("name")} != {item_type}')
    elif allow_legacy:
        manifest = _legacy_manifest(output_dir)
        if not manifest['frames']:
            raise FileNotFoundError(f'未找到 RGB-D 数据: {output_dir}')
        print(f'  使用无 manifest 的旧采集数据: {output_dir}')
    else:
        raise FileNotFoundError(f'采集清单不存在: {manifest_path}')

    valid_frames = []
    for frame in manifest.get('frames', []):
        color_path = output_dir / frame['color']
        depth_path = output_dir / frame['depth']
        point_cloud_path = output_dir / frame.get(
            'point_cloud', f'pcd_frames/frame_{int(frame["index"]):03d}.ply'
        )
        if not color_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f'帧 {frame.get("index")} RGB-D 不完整: {color_path}, {depth_path}'
            )
        if not point_cloud_path.is_file():
            raise FileNotFoundError(
                f'帧 {frame.get("index")} 点云不存在: {point_cloud_path}'
            )
        valid_frames.append(frame)
    planned_frames = int(manifest.get('planned_frames', len(valid_frames)))
    if manifest.get('schema_version', 0) >= 1 and len(valid_frames) != planned_frames:
        raise RuntimeError(
            f'采集帧数不完整: {len(valid_frames)}/{planned_frames}'
        )
    if len(valid_frames) < 2:
        raise RuntimeError(f'有效 RGB-D 帧不足: {len(valid_frames)}')

    stitched_path = output_dir / f'{item_type}_stitched.ply'
    if not stitched_path.is_file():
        raise FileNotFoundError(f'final.py 拼接点云不存在: {stitched_path}')
    manifest['frames'] = valid_frames
    return manifest, stitched_path


def robust_dimensions_from_stitched(stitched_path, lower=2.0, upper=98.0):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(stitched_path))
    points = np.asarray(pcd.points)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 100:
        raise RuntimeError(f'拼接点云有效点不足: {len(points)}')
    low = np.percentile(points, lower, axis=0)
    high = np.percentile(points, upper, axis=0)
    inlier_mask = np.all((points >= low) & (points <= high), axis=1)
    inliers = points[inlier_mask]
    axis_extent = high - low

    xy = inliers[:, :2].astype(np.float32)
    xy_center = np.median(xy, axis=0)
    covariance = np.cov((xy - xy_center).T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    anisotropy = float(eigenvalues[-1] / max(eigenvalues[0], 1e-12))

    rectangle = cv2.minAreaRect(xy.reshape(-1, 1, 2))
    corners = cv2.boxPoints(rectangle).astype(float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    width_edge_index = int(np.argmax(lengths))
    width_axis = edges[width_edge_index] / max(lengths[width_edge_index], 1e-12)
    width_m = float(lengths[width_edge_index])
    depth_m = float(lengths[(width_edge_index + 1) % 4])
    if depth_m > width_m:
        width_m, depth_m = depth_m, width_m
        width_axis = edges[(width_edge_index + 1) % 4]
        width_axis /= max(np.linalg.norm(width_axis), 1e-12)
    yaw_deg = float(np.degrees(np.arctan2(width_axis[1], width_axis[0])))
    yaw_deg = (yaw_deg + 90.0) % 180.0 - 90.0

    extent = np.array([width_m, depth_m, axis_extent[2]])
    dimension_method = 'robust_xy_min_area_rect'
    if np.any(extent < 0.005) or np.any(extent > 2.0):
        raise RuntimeError(f'拼接点云尺寸异常: {extent.tolist()} m')
    result = {
        'width_m': float(extent[0]),
        'depth_m': float(extent[1]),
        'height_m': float(extent[2]),
        'n_points': int(len(points)),
        'percentiles': [lower, upper],
        'dimension_method': dimension_method,
        'xy_anisotropy': anisotropy,
        'xy_yaw_deg': yaw_deg,
        'source': str(stitched_path),
    }
    print(f'  final.py 点云稳健尺寸: '
          f'{result["width_m"]*100:.1f}×{result["depth_m"]*100:.1f}×'
          f'{result["height_m"]*100:.1f}cm (宽×深×高)')
    return result


def get_segmentation_spec(item_type):
    if item_type == 'chair':
        return GENERIC_MODEL, 'chair'
    if item_type == 'table':
        return GENERIC_MODEL, 'dining table'
    if item_type == 'cabinet':
        return CUSTOM_MODEL, 'cabinet'
    raise ValueError(f'不支持的家具类别: {item_type}')


def resolve_class_id(model, target_class):
    names = model.names
    for class_id, class_name in names.items():
        if class_name == target_class:
            return int(class_id)
    raise RuntimeError(f'模型不包含类别 {target_class}: {names}')


def measure_chair_from_masks(masks_dict, calib, bboxes_dict=None):
    """深度图直接分割物体 → 像素-距离法测尺寸.

    转台面 ~ 全图中位数深度。物体比转台面更近，用深度阈值分割。
    正面帧（0°/180°）像素宽 → 实际宽度，侧面帧（90°/270°）像素宽 → 实际深度.
    """
    fx = calib.get('fx_full', 605.0)
    fy = calib.get('fy_full', 604.0)
    dscale = calib.get('depth_scale', 0.001)
    n_frames = calib.get('n_frames', 72)
    depth_dir = OUT / 'depth'

    front_indices = [0, n_frames // 2]
    side_indices = [n_frames // 4, n_frames * 3 // 4]

    # bbox 收缩系数：自定义模型 bbox 偏大约 25%，收缩后更准
    BBOX_SHRINK = 0.75

    heights_m = []
    widths_m = []
    depths_m = []

    for i in range(n_frames):
        dep = cv2.imread(str(depth_dir / f'{i:03d}.png'), cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue
        dep_m = dep.astype(float) * dscale
        valid_all = dep_m[dep > 0]
        if valid_all.size < 1000:
            continue
        base_depth = float(np.median(valid_all))

        # 确定边界：用 bbox，小维度收缩补偿模型 padding
        has_bbox = bboxes_dict and i in bboxes_dict and bboxes_dict[i] is not None
        if has_bbox:
            bx1, by1, bx2, by2 = [int(v) for v in bboxes_dict[i]]
            bw, bh = bx2 - bx1, by2 - by1
            w_px_raw = bw
            w_px_shrunk = bw * BBOX_SHRINK
            h_px = bh * BBOX_SHRINK
        else:
            obj_mask = (dep > 0) & (dep_m < base_depth - 0.04)
            ys, xs = np.where(obj_mask)
            if len(ys) < 100:
                continue
            w_px_raw = xs.max() - xs.min()
            w_px_shrunk = w_px_raw * BBOX_SHRINK
            h_px = ys.max() - ys.min() * BBOX_SHRINK

        h_m = h_px * base_depth / fy
        # 高度仅从正面/背面帧测量（侧面帧的表观高度受 3D 结构影响）
        if i in front_indices:
            heights_m.append(h_m)

        if i in front_indices:
            # 正面帧：bbox 宽度 = 物体真实宽度，不收缩
            widths_m.append(w_px_raw * base_depth / fx)
        elif i in side_indices:
            # 侧面帧：bbox 宽度 = 物体深度，收缩补偿 padding
            depths_m.append(w_px_shrunk * base_depth / fx)

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

    print(f'\n  ┌─ 自动测量 (深度分割法)')
    print(f'  ├─ 高度: {result["height_m"]*100:.1f}cm ({result["n_measured"]} 帧中位数)')
    print(f'  ├─ 宽度: {result["width_m"]*100:.1f}cm (正面帧)')
    print(f'  └─ 深度: {result["depth_m"]*100:.1f}cm (侧面帧)')

    calib['chair_height_m'] = result['height_m']
    calib['chair_width_m'] = result['width_m']
    calib['chair_depth_m'] = result['depth_m']
    return result


def _largest_component(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest] = 255
    return cleaned


def _mask_quality(mask):
    h, w = mask.shape
    area_ratio = float(np.count_nonzero(mask)) / float(h * w)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    border_ratio = float(np.count_nonzero(border)) / max(1, len(border))
    return area_ratio, border_ratio


def _select_view_indices(candidates, targets=(0, 45, 90)):
    selected = []
    for target in targets:
        selected_degrees = [
            c['degree'] for c in candidates if c['index'] in selected
        ]
        remaining = [
            c for c in candidates
            if c['index'] not in selected
            and all(abs(c['degree'] - degree) >= 20 for degree in selected_degrees)
        ]
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda c: (abs(c['degree'] - target), -c['confidence'])
        )
        selected.append(best['index'])
    return selected


def do_yolo_and_measure(manifest, item_type='chair', use_sam=False):
    from ultralytics import YOLO

    print('=' * 60)
    print('  Step 2/5  YOLO 分割抠图')
    print('=' * 60)

    model_path, target_class = get_segmentation_spec(item_type)
    if not model_path.is_file():
        raise FileNotFoundError(f'模型未找到: {model_path}')
    model = YOLO(str(model_path))
    class_id = resolve_class_id(model, target_class)
    print(f'  模型: {model_path.name} | 类别: {target_class} ({class_id})')

    sam_predictor = None
    if use_sam:
        if not SAM_MODEL.is_file():
            raise FileNotFoundError(f'SAM 模型未找到: {SAM_MODEL}')
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_b'](checkpoint=str(SAM_MODEL))
        sam_predictor = SamPredictor(sam)

    masks_dict = {}
    bboxes_dict = {}
    candidates = []
    frame_by_index = {int(f['index']): f for f in manifest['frames']}
    for position, frame in enumerate(manifest['frames']):
        index = int(frame['index'])
        color_path = CAPTURE_ROOT / frame['color']
        img = cv2.imread(str(color_path))
        if img is None:
            raise RuntimeError(f'无法读取彩色图: {color_path}')
        h, w = img.shape[:2]
        result = model(img, verbose=False, classes=[class_id])[0]
        best = None
        if result.masks is not None and len(result.boxes) > 0:
            for detection_index in range(len(result.boxes)):
                if int(result.boxes.cls[detection_index]) != class_id:
                    continue
                confidence = float(result.boxes.conf[detection_index])
                box = result.boxes.xyxy[detection_index].cpu().numpy()
                box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
                area_ratio = box_area / float(w * h)
                if confidence >= 0.20 and 0.002 <= area_ratio <= 0.85:
                    score = confidence - 0.1 * abs(area_ratio - 0.15)
                    if best is None or score > best[0]:
                        best = (score, detection_index, confidence, box)

        mask = np.zeros((h, w), dtype=np.uint8)
        bbox = None
        confidence = 0.0
        if best is not None:
            _, detection_index, confidence, bbox = best
            raw = result.masks.data[detection_index].cpu().numpy()
            if raw.shape != (h, w):
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = _largest_component((raw > 0.5).astype(np.uint8) * 255)

            if sam_predictor is not None:
                sam_predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                sam_masks, scores, _ = sam_predictor.predict(
                    box=np.asarray(bbox)[None, :], multimask_output=False
                )
                if len(sam_masks) and float(scores[0]) > 0.5:
                    sam_mask = _largest_component(sam_masks[0].astype(np.uint8) * 255)
                    intersection = np.count_nonzero((mask > 0) & (sam_mask > 0))
                    union = np.count_nonzero((mask > 0) | (sam_mask > 0))
                    iou = intersection / union if union else 0.0
                    area_ratio = np.count_nonzero(sam_mask) / max(1, np.count_nonzero(mask))
                    if iou >= 0.65 and 0.7 <= area_ratio <= 1.3:
                        mask = sam_mask

        masks_dict[index] = mask
        bboxes_dict[index] = bbox
        area_ratio, border_ratio = _mask_quality(mask)
        degree = float(frame.get('angle_deg', index * manifest.get('step_deg', STEP_DEG)))
        if 0.002 <= area_ratio <= 0.75 and border_ratio <= 0.20:
            candidates.append({
                'index': index,
                'degree': degree,
                'confidence': confidence,
                'area_ratio': area_ratio,
                'border_ratio': border_ratio,
            })
        if position < 3 or (position + 1) % 18 == 0:
            print(f'  [{position+1}/{len(manifest["frames"])}] {degree:5.1f}° '
                  f'conf={confidence:.2f} mask={area_ratio*100:.1f}%')

    print(f'  合格 mask: {len(candidates)}/{len(manifest["frames"])}')
    selected_indices = _select_view_indices(candidates)
    if len(selected_indices) < 2:
        raise RuntimeError(f'合格抠图视角不足: {len(selected_indices)}，不会提交 Hyper3D')

    view_dir = OUT / 'views'
    view_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for index in selected_indices:
        frame = frame_by_index[index]
        src = CAPTURE_ROOT / frame['color']
        img = cv2.imread(str(src))
        mask = masks_dict[index]
        white_bg = np.full_like(img, 255, dtype=np.uint8)
        alpha = (mask > 128).astype(np.float32)[..., None]
        composited = (img * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)
        view_path = view_dir / f'view_{index:03d}.png'
        if not cv2.imwrite(str(view_path), composited):
            raise IOError(f'抠图视图写入失败: {view_path}')
        candidate = next(c for c in candidates if c['index'] == index)
        views.append({
            **candidate,
            'path': str(view_path),
            'has_mask': True,
            'model': str(model_path),
            'target_class': target_class,
        })
        print(f'  视角 {candidate["degree"]:5.1f}° -> {view_path.name}')

    with open(view_dir / 'views_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(views, f, ensure_ascii=False, indent=2)
    return masks_dict, views


def detect_backrest_angle_from_image(side_image_path):
    """从侧面图检测靠背偏离垂直的角度（Hough直线法）。
    返回角度（度），正值=靠背向后倾斜。None=检测失败。"""
    img = cv2.imread(side_image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 聚焦上半部分（靠背区域）
    upper = gray[:int(h * 0.65), :]

    # Canny 边缘检测
    edges = cv2.Canny(upper, 30, 100)

    # Hough 直线检测
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                            minLineLength=min(w, h) * 0.04, maxLineGap=25)

    if lines is None or len(lines) < 5:
        return None

    # 兼容新旧 OpenCV 返回格式: (N,4) 或 (N,1,4)
    if lines.ndim == 3:
        lines = lines[:, 0, :]

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) < 5:
            continue  # 跳过接近水平的线
        # 直线相对垂直方向的角度（水平线=90°, 垂直线=0°）
        angle_from_vertical = abs(np.degrees(np.arctan2(dx, dy)))
        if angle_from_vertical < 40:  # 只取接近垂直的线
            angles.append(angle_from_vertical)

    if len(angles) < 5:
        return None

    # 取中位数，抗噪声
    detected = float(np.median(angles))
    print(f'  侧面图靠背角度检测: {detected:.1f}°（{len(angles)}条线）')
    return detected


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 3: Hyper3D 生成 (通过 Blender)                    ║
# ╚══════════════════════════════════════════════════════════╝

def do_hyper3d_generate(blender: BlenderClient, views: list, object_dims: dict,
                        furniture_name: str):
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
    status_message = str(status.get('message', ''))
    if 'Mode: FAL_AI' in status_message:
        raise RuntimeError(
            '当前本地图片上传只支持 Hyper3D MAIN_SITE；'
            '请在 BlenderMCP 面板切换到 MAIN_SITE'
        )

    view_paths = [v['path'] for v in views if v.get('has_mask')][:3]
    if len(view_paths) < 2:
        raise RuntimeError('有效抠图不足 2 张，不提交 Hyper3D')

    print(f'  提交 {len(view_paths)} 张抠图到 Hyper3D...')

    # 编码图片为 base64（MAIN_SITE 模式）
    images = []
    for vp in view_paths:
        with open(vp, 'rb') as f:
            suffix = Path(vp).suffix
            images.append((suffix, base64.b64encode(f.read()).decode('ascii')))

    # 不使用 bbox 条件（容易压扁有开门等延伸结构的物体）
    bbox = None

    result = blender.send("create_rodin_job", {
        "text_prompt": None,
        "images": images,
        "bbox_condition": bbox,
    })

    print(f'  Hyper3D 任务已提交: {json.dumps(result, indent=2)}')
    if not isinstance(result, dict) or result.get('error'):
        raise RuntimeError(f'Hyper3D 提交失败: {result}')

    subscription_key = result.get("jobs", {}).get("subscription_key")
    request_id = result.get("request_id")
    task_uuid = result.get("uuid")
    if not subscription_key and not request_id:
        raise RuntimeError(f'Hyper3D 返回缺少任务标识: {result}')

    print(f'  等待生成完成...')
    for poll_i in range(120):  # 最多等 10 分钟
        time.sleep(5)
        if subscription_key:
            status_resp = blender.send("poll_rodin_job_status",
                                       {"subscription_key": subscription_key})
        else:
            status_resp = blender.send("poll_rodin_job_status",
                                       {"request_id": request_id})

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
    object_name = f"{furniture_name}_v12"
    print('  清理旧模型...')
    blender.execute_blender_code(f'''
import bpy
# 收集要删的对象名（避免遍历时删除导致 StructRNA 错误）
to_remove = [o.name for o in bpy.data.objects
             if o.type == 'MESH' and (o.name == "{object_name}" or o.name.startswith("{object_name}."))]
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
        import_params["request_id"] = request_id
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


def do_blender_cleanup(blender: BlenderClient, object_name: str, chair_dims: dict, no_scale: bool = False):
    """清理模型: Remesh融合 → 去悬浮碎片 → XYZ缩放 → 平移到底面.

    分两步执行，避免 Blender StructRNA 引用失效."""
    print('=' * 60)
    print('  Step 4/5  Blender 模型清理（Remesh融合 + 去碎片 + 缩放）')
    print('=' * 60)

    ch = chair_dims.get('height_m', 0.09)
    cw = chair_dims.get('width_m', 0.05)
    cd = chair_dims.get('depth_m', 0.04)
    obj_path = str((OUT / f'{object_name}.obj').resolve())

    # ── 阶段 A: 找对象 → Remesh融合 → separate → 保留最大块 ──
    code_stage_a = f'''
import bpy, bmesh, os

# 1. 找到目标对象
obj = bpy.data.objects.get("{object_name}")
if obj is None:
    candidates = [o for o in bpy.data.objects
                  if o.type == 'MESH' and (o.name == "{object_name}" or o.name.startswith("{object_name}."))]
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

# 2. 记录场景中已有的其他 mesh，后续不得删除
protected_mesh_names = {{
    o.name for o in bpy.data.objects
    if o.type == 'MESH' and o != obj
}}

# 3. 应用变换 + 计算模型尺寸
bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

n_orig_v = len(obj.data.vertices)
n_orig_f = len(obj.data.polygons)
print(f"  源对象: {{obj.name}} ({{n_orig_v}}v, {{n_orig_f}}f)")

# 4. 计算 voxel 大小
vs = obj.data.vertices
xs = [v.co.x for v in vs]; ys = [v.co.y for v in vs]; zs = [v.co.z for v in vs]
max_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
voxel_size = max_dim / 200.0
print(f"  max_dim={{max_dim*100:.1f}}cm, voxel={{voxel_size*100:.2f}}cm")

# 5. Voxel Remesh 融合碎片
mod = obj.modifiers.new(name="Remesh", type='REMESH')
mod.mode = 'VOXEL'
mod.voxel_size = voxel_size
bpy.ops.object.modifier_apply(modifier="Remesh")
print(f"  Remesh: {{len(obj.data.vertices)}}v")

# 6. separate_by_loose → 保留最大块
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.separate(type='LOOSE')
bpy.ops.object.mode_set(mode='OBJECT')

# 只收集目标对象和本次 separate 新生成的块
all_pieces = [
    o for o in bpy.data.objects
    if o.type == 'MESH' and o.name not in protected_mesh_names
]
all_pieces.sort(key=lambda o: len(o.data.vertices), reverse=True)
if not all_pieces:
    raise RuntimeError("Remesh 后没有目标网格")

print(f"  分离成 {{len(all_pieces)}} 块")

# 保留目标的最大块，仅删除本次目标产生的其余块
if len(all_pieces) > 1:
    main = all_pieces[0]
    n_removed = sum(len(p.data.vertices) for p in all_pieces[1:])
    for p in all_pieces[1:]:
        bpy.data.objects.remove(p, do_unlink=True)
    main.name = "{object_name}"
    print(f"  删除 {{len(all_pieces)-1}} 块 ({{n_removed}}v, {{100*n_removed/(n_removed+len(main.data.vertices)):.0f}}%)")
else:
    all_pieces[0].name = "{object_name}"
    print(f"  仅1块，无需删除")

obj = bpy.data.objects.get("{object_name}")
print(f"  保留: {{len(obj.data.vertices)}}v, {{len(obj.data.polygons)}}f")



print("  Stage A 完成")
'''

    result = blender.execute_blender_code(code_stage_a)
    print(f'  {result}')

    # ── 阶段 B: XYZ 缩放 + 平移到底面 ──
    code_stage_b = f'''
import bpy, bmesh

no_scale = {no_scale}

obj = bpy.data.objects.get("{object_name}")
if obj is None:
    raise RuntimeError("Stage A 后对象丢失")

bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

print(f"  进入 Stage B: {{len(obj.data.vertices)}}v, {{len(obj.data.polygons)}}f")

# ── XYZ 独立缩放 ──
if not no_scale:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # ── XY 平面旋转对齐：绕 Z 轴转正 bbox 长边到 X 轴 ──
    import math
    vs_co = [v.co for v in bm.verts]
    nv = len(vs_co)
    cx = sum(v.x for v in vs_co) / nv
    cy = sum(v.y for v in vs_co) / nv
    cz = sum(v.z for v in vs_co) / nv
    sxx=syy=sxy=0.0
    for v in vs_co:
        dx,dy = v.x-cx, v.y-cy
        sxx+=dx*dx; syy+=dy*dy; sxy+=dx*dy
    sxx/=nv; syy/=nv; sxy/=nv
    angle = 0.5 * math.atan2(2*sxy, sxx-syy)
    target_angle = 0.0 if {cw:.4f} >= {cd:.4f} else math.pi / 2
    rotation = target_angle - angle
    ca, sa = math.cos(rotation), math.sin(rotation)
    for v in bm.verts:
        dx, dy = v.co.x-cx, v.co.y-cy
        v.co.x = cx + ca*dx - sa*dy
        v.co.y = cy + sa*dx + ca*dy
    bm.to_mesh(obj.data); bm.free(); obj.data.update()
    bm=bmesh.new(); bm.from_mesh(obj.data); bm.verts.ensure_lookup_table()
    print(f"  XY对齐: 旋转 {{math.degrees(rotation):.1f}}°")

    xs_cur = [v.co.x for v in bm.verts]
    ys_cur = [v.co.y for v in bm.verts]
    zs_cur = [v.co.z for v in bm.verts]
    cur_w = max(xs_cur) - min(xs_cur)
    cur_d = max(ys_cur) - min(ys_cur)
    cur_h = max(zs_cur) - min(zs_cur)
    target_w = {cw:.4f}
    target_d = {cd:.4f}
    target_h = {ch:.4f}
    scale_x = target_w / cur_w if cur_w > 0.0005 else 1.0
    scale_y = target_d / cur_d if cur_d > 0.0005 else 1.0
    scale_z = target_h / cur_h if cur_h > 0.0005 else 1.0
    print(f"  各轴缩放: X{{scale_x:.4f}} Y{{scale_y:.4f}} Z{{scale_z:.4f}}")
    print(f"    宽 {{cur_w*100:.1f}}→{{target_w*100:.1f}}cm, 深 {{cur_d*100:.1f}}→{{target_d*100:.1f}}cm, 高 {{cur_h*100:.1f}}→{{target_h*100:.1f}}cm")
    for v in bm.verts:
        v.co.x = (v.co.x - cx) * scale_x
        v.co.y = (v.co.y - cy) * scale_y
        v.co.z = (v.co.z - min(zs_cur)) * scale_z

    bm.to_mesh(obj.data); bm.free()
    obj.data.update()
else:
    print(f"  跳过缩放 (--debug-no-scale)")

# ── 输出 ──
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
    dims_path = OUT / f'{furniture_name}_dims.json'
    if dims_path.exists():
        with open(dims_path) as f:
            dims = json.load(f)
        print(f'  尺寸: {dims["height_m"]*100:.0f}×{dims["width_m"]*100:.0f}×{dims["depth_m"]*100:.0f}cm (高×宽×深)')
    print(f'  输出目录: {OUT}')
    print(f'  {"=" * 60}')


def _write_obj_vertices(source_path, target_path, vertices):
    lines = source_path.read_text(encoding='utf-8').splitlines(keepends=True)
    vertex_index = 0
    output = []
    for line in lines:
        if line.startswith('v '):
            if vertex_index >= len(vertices):
                raise RuntimeError('OBJ 顶点数与 PLY 不一致')
            x, y, z = vertices[vertex_index]
            ending = '\n' if line.endswith('\n') else ''
            output.append(f'v {x:.9g} {y:.9g} {z:.9g}{ending}')
            vertex_index += 1
        else:
            output.append(line)
    if vertex_index != len(vertices):
        raise RuntimeError(
            f'OBJ 顶点数与 PLY 不一致: {vertex_index} != {len(vertices)}'
        )
    target_path.write_text(''.join(output), encoding='utf-8')


def conservative_surface_fit(mesh_path, stitched_path, obj_path=None,
                             yaw_deg=0.0, distance_m=0.005, weight=0.2,
                             max_displacement_m=0.0015):
    import open3d as o3d
    from sklearn.neighbors import NearestNeighbors

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    point_cloud = o3d.io.read_point_cloud(str(stitched_path))
    vertices = np.asarray(mesh.vertices).copy()
    points = np.asarray(point_cloud.points)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points):
        center_xy = np.median(points[:, :2], axis=0)
        theta = np.radians(-yaw_deg)
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        points[:, :2] = (points[:, :2] - center_xy) @ rotation.T
        points[:, :2] -= np.median(points[:, :2], axis=0)
        points[:, 2] -= np.percentile(points[:, 2], 2)
    if len(vertices) == 0 or len(mesh.triangles) == 0 or len(points) < 100:
        raise RuntimeError('表面贴合输入几何无效')

    nearest = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points)
    distances, indices = nearest.kneighbors(vertices)
    distances = distances[:, 0]
    indices = indices[:, 0]
    eligible = distances < distance_m
    displacement = (points[indices] - vertices) * weight
    lengths = np.linalg.norm(displacement, axis=1)
    scale = np.minimum(1.0, max_displacement_m / np.maximum(lengths, 1e-12))
    displacement *= scale[:, None]
    fitted_vertices = vertices.copy()
    fitted_vertices[eligible] += displacement[eligible]

    fitted = o3d.geometry.TriangleMesh(mesh)
    fitted.vertices = o3d.utility.Vector3dVector(fitted_vertices)
    fitted.compute_vertex_normals()
    temp_path = mesh_path.with_name(f'{mesh_path.stem}.surface_fit.tmp{mesh_path.suffix}')
    if not o3d.io.write_triangle_mesh(str(temp_path), fitted):
        raise IOError(f'表面贴合模型写入失败: {temp_path}')

    temp_obj_path = None
    if obj_path is not None:
        temp_obj_path = obj_path.with_name(f'{obj_path.stem}.surface_fit.tmp.obj')
        _write_obj_vertices(obj_path, temp_obj_path, fitted_vertices)

    temp_path.replace(mesh_path)
    if temp_obj_path is not None:
        temp_obj_path.replace(obj_path)
    return {
        'eligible_vertices': int(np.count_nonzero(eligible)),
        'total_vertices': int(len(vertices)),
        'max_displacement_m': float(
            np.linalg.norm(displacement[eligible], axis=1).max()
            if np.any(eligible) else 0.0
        ),
    }


def validate_exported_mesh(mesh_path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(triangles) == 0:
        raise RuntimeError(f'最终文件不是有效三角网格: {mesh_path}')
    extent = np.ptp(vertices, axis=0)
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise RuntimeError(f'最终网格尺寸无效: {extent.tolist()}')
    return {
        'vertices': int(len(vertices)),
        'triangles': int(len(triangles)),
        'extent_m': extent.tolist(),
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  Main                                                    ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description='V12 家具重建完整管线')
    parser.add_argument('--name', choices=['chair', 'table', 'cabinet'],
                        default='chair', help='家具类别 (默认: chair)')
    parser.add_argument('--skip-capture', action='store_true',
                        help='复用已有 RGB-D 和 final.py 拼接点云')
    parser.add_argument('--overwrite-capture', action='store_true',
                        help='归档已有采集数据后重新采集')
    parser.add_argument('--input-dir', type=Path,
                        help='读取旧采集目录；最终结果仍写入 output/<name>')
    parser.add_argument('--no-blender', action='store_true',
                        help='只执行数据校验、抠图和点云尺寸计算')
    parser.add_argument('--skip-hyper3d', action='store_true',
                        help='跳过 Hyper3D，使用 Blender 中已有 <name>_v12 对象')
    parser.add_argument('--sam-refine', action='store_true',
                        help='用 SAM 精修通过一致性检查的 YOLO mask')
    parser.add_argument('--blender-host', default=BLENDER_HOST,
                        help=f'Blender 主机地址 (默认: {BLENDER_HOST})')
    parser.add_argument('--blender-port', type=int, default=BLENDER_PORT,
                        help=f'Blender 端口 (默认: {BLENDER_PORT})')
    parser.add_argument('--turntable-port', default='/dev/ttyUSB0',
                        help='转台串口 (默认: /dev/ttyUSB0)')
    parser.add_argument('--surface-fit', action='store_true',
                        help='实验性地用 stitched 点云做低权重表面贴合')
    parser.add_argument('--debug-no-scale', action='store_true',
                        help='跳过 stitched 点云尺寸缩放')
    parser.add_argument('--debug-raw', action='store_true',
                        help='跳过 Blender 清理，直接导出 Hyper3D 原始模型')
    args = parser.parse_args()

    global OUT, CAPTURE_ROOT
    OUT = BASE / 'output' / args.name
    OUT.mkdir(parents=True, exist_ok=True)
    CAPTURE_ROOT = args.input_dir.resolve() if args.input_dir else OUT

    print('=' * 60)
    print(f'  V12 家具重建管线 [{args.name}]')
    print(f'  采集数据: {CAPTURE_ROOT}')
    print(f'  最终输出: {OUT}')
    print(f'  Blender: {args.blender_host}:{args.blender_port}')
    print('=' * 60)

    if not args.skip_capture:
        if args.input_dir:
            raise ValueError('--input-dir 只能与 --skip-capture 一起使用')
        if capture_dataset_exists(OUT, args.name):
            if not args.overwrite_capture:
                raise FileExistsError(
                    f'{OUT} 已有采集数据；使用 --skip-capture 复用，'
                    '或使用 --overwrite-capture 先归档再重采'
                )
            archive_capture_dataset(OUT, args.name)
        run_final_pipeline(args.name, args.turntable_port, skip_capture=False)
    else:
        print('  跳过硬件采集 (--skip-capture)')
        stitched = CAPTURE_ROOT / f'{args.name}_stitched.ply'
        if CAPTURE_ROOT == OUT and not stitched.is_file():
            pcd_files = list((OUT / 'pcd_frames').glob('frame_*.ply'))
            if pcd_files:
                run_final_pipeline(args.name, args.turntable_port, skip_capture=True)

    manifest, stitched_path = validate_capture_dataset(
        CAPTURE_ROOT, args.name, allow_legacy=bool(args.input_dir)
    )
    object_dims = robust_dimensions_from_stitched(stitched_path)
    dims_path = OUT / f'{args.name}_dims.json'
    with open(dims_path, 'w', encoding='utf-8') as f:
        json.dump(object_dims, f, ensure_ascii=False, indent=2)

    _, views = do_yolo_and_measure(
        manifest, item_type=args.name, use_sam=args.sam_refine
    )

    if args.no_blender:
        print('\n跳过 Blender 步骤 (--no-blender)')
        print(f'数据已保存到: {OUT}')
        return

    blender = BlenderClient(host=args.blender_host, port=args.blender_port)
    try:
        blender.connect()
        if args.skip_hyper3d:
            object_name = f'{args.name}_v12'
            print(f'跳过 Hyper3D，使用 Blender 中已有模型: {object_name}')
        else:
            object_name = do_hyper3d_generate(
                blender, views, object_dims, args.name
            )

        if args.debug_raw:
            print('跳过 Blender 清理 (--debug-raw)')
        else:
            do_blender_cleanup(
                blender, object_name, object_dims,
                no_scale=args.debug_no_scale
            )
        do_export(blender, object_name, args.name)

        mesh_path = OUT / f'{args.name}_v12.ply'
        surface_fit_stats = None
        if args.surface_fit:
            try:
                surface_fit_stats = conservative_surface_fit(
                    mesh_path, stitched_path,
                    obj_path=OUT / f'{args.name}_v12.obj',
                    yaw_deg=object_dims.get('xy_yaw_deg', 0.0)
                )
                print(f'  表面贴合: {surface_fit_stats}')
            except Exception as exc:
                print(f'  表面贴合跳过，保留尺度约束模型: {exc}')
        mesh_stats = validate_exported_mesh(mesh_path)

        log = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'method': 'V12_HYPER3D_FINAL_POINT_CLOUD_CONSTRAINT',
            'name': args.name,
            'capture_root': str(CAPTURE_ROOT),
            'n_frames': len(manifest['frames']),
            'object_dims': object_dims,
            'views': views,
            'surface_fit': surface_fit_stats,
            'mesh': mesh_stats,
            'output_dir': str(OUT),
        }
        with open(OUT / 'log_v12.json', 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    finally:
        blender.disconnect()


if __name__ == '__main__':
    main()
