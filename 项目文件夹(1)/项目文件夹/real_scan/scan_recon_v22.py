#!/usr/bin/env python3
"""
V22 — 3D Gaussian Splatting 重建管线
=====================================
改进（vs V19 Colab 失败）:
  1. YOLO mask 紧裁剪 — 椅子从 4% 像素 → 50%+，消除背景干扰
  2. 白底填充 — 背景一致，零梯度，不吸引高斯球
  3. nerfstudio 后端 — ns-train gaussian-splatting 一键训练，不用手写 CUDA
  4. transforms.json — 用标定数据精确计算每帧相机外参
  5. 位姿用标定旋转轴 — 相机绕转轴 orbit，非简单绕 Y 轴

流程:
  [本地]  python3 scan_recon_v22.py --name chair           → 预处理 + 打包
  [本地]  scp output/v22/chair_v22.zip luoz@192.168.100.55:~/
  [服务器] bash v22_server.sh                               → 安装 + 训练 + 提取
  [本地]  scp luoz@192.168.100.55:~/v22_output/mesh.ply output/v22/chair/
  [本地]  python3 scan_recon_v22.py --name chair --post     → 缩放 + 清理

依赖:
  本地: opencv-python, numpy, pillow
  服务器: RTX 5080, CUDA 12.x, Python 3.10+, nerfstudio, open3d
"""

import cv2, numpy as np, json, sys, argparse, shutil
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / 'output/v22'


def generate_masks(color_dir, n_frames, device='cpu'):
    """用 YOLOv8n-seg 对所有帧生成 chair mask.
    返回 {frame_idx: mask_255} 字典.
    """
    from ultralytics import YOLO

    yolo_path = BASE / 'yolov8n-seg.pt'
    if not yolo_path.exists():
        raise FileNotFoundError(f'YOLO 模型不存在: {yolo_path}')
    yolo = YOLO(str(yolo_path))

    color_files = sorted(Path(color_dir).glob('*.png'))
    if len(color_files) < 36:
        color_files = sorted(Path(color_dir).glob('*.jpg'))

    masks = {}
    detected = 0
    for i, cf in enumerate(color_files):
        if i >= n_frames:
            break
        idx = int(cf.stem)
        img = cv2.imread(str(cf))
        H, W = img.shape[:2]

        result = yolo(img, verbose=False, classes=[56])  # chair class
        mask_binary = np.zeros((H, W), dtype=np.uint8)

        if result[0].masks is not None:
            chairs = []
            for j in range(len(result[0].boxes)):
                if result[0].names[int(result[0].boxes.cls[j])] == 'chair':
                    conf = float(result[0].boxes.conf[j])
                    box = result[0].boxes.xyxy[j].cpu().numpy()
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    area_pct = area / (W * H)
                    if conf >= 0.25 and area_pct <= 0.25:
                        chairs.append((conf, j))

            if chairs:
                chairs.sort(key=lambda c: c[0], reverse=True)
                best_j = chairs[0][1]
                mask_raw = result[0].masks.data[best_j].cpu().numpy()
                if mask_raw.shape != (H, W):
                    mask_raw = cv2.resize(mask_raw, (W, H))
                mask_binary = (mask_raw > 0.5).astype(np.uint8) * 255
                detected += 1

        masks[idx] = mask_binary

        if i < 3 or (i + 1) % 18 == 0:
            mk = (mask_binary > 0).sum() // 1000
            print(f'  [{i+1}/{n_frames}] YOLO mask={mk}k px {"✓" if mask_binary.any() else "✗"}')

    print(f'  检出: {detected}/{len(masks)}')
    return masks

# ╔══════════════════════════════════════════════════════════╗
# ║  Part 1: 预处理 — 裁剪抠图 + 生成位姿                    ║
# ╚══════════════════════════════════════════════════════════╝

def compute_tight_crop(masks_dict, color_dir, padding_ratio=0.3):
    """计算所有帧 mask 的联合包围盒，统一裁剪尺寸."""
    all_bounds = []
    for idx, mask in masks_dict.items():
        ys, xs = np.where(mask > 128)
        if len(xs) < 50:
            continue
        all_bounds.append((xs.min(), ys.min(), xs.max(), ys.max()))

    x1 = min(b[0] for b in all_bounds)
    y1 = min(b[1] for b in all_bounds)
    x2 = max(b[2] for b in all_bounds)
    y2 = max(b[3] for b in all_bounds)

    # 加 padding
    w, h = x2 - x1, y2 - y1
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    img = cv2.imread(str(sorted(color_dir.glob('*.png'))[0]))
    H, W = img.shape[:2]

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y2 = min(H, y2 + pad_y)

    return x1, y1, x2, y2


def prepare_v22(name='chair'):
    """准备 V22 预处理数据: 裁剪图片 + 生成 transforms.json."""
    SRC = BASE / f'output/v12/{name}'
    DST = OUT_DIR / name
    IMG_OUT = DST / 'images'
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    # 加载标定
    with open(SRC / 'calibrate.json') as f:
        calib = json.load(f)
    fx = calib['fx_full']
    fy = calib['fy_full']
    cx = calib['ppx_full']
    cy = calib['ppy_full']
    W, H = calib['width_full'], calib['height_full']

    color_dir = SRC / 'color'
    color_files = sorted(color_dir.glob('*.png'))
    if len(color_files) < 36:
        color_files = sorted(color_dir.glob('*.jpg'))

    # 生成 YOLO masks（如果已有则加载）
    mask_cache = SRC / 'mask_v22'
    if mask_cache.exists():
        print(f'加载已有 masks: {mask_cache}/')
        masks_dict = {}
        for mp in sorted(mask_cache.glob('*.png')):
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 128:
                masks_dict[int(mp.stem)] = mask
    else:
        print('生成 YOLO masks...')
        masks_dict = generate_masks(color_dir, len(color_files))
        mask_cache.mkdir(exist_ok=True)
        for idx, mask in masks_dict.items():
            if mask.any():
                cv2.imwrite(str(mask_cache / f'{idx:03d}.png'), mask)
        print(f'  masks 已保存: {mask_cache}/')

    print(f'标定: {fx:.0f}x{fy:.0f} @ {W}x{H}')
    print(f'Mask 有效帧: {len(masks_dict)}/{len(color_files)}')

    # ── 计算裁剪区域 ──
    x1, y1, x2, y2 = compute_tight_crop(masks_dict, color_dir)
    crop_w, crop_h = x2 - x1, y2 - y1
    new_cx = cx - x1
    new_cy = cy - y1
    print(f'裁剪: ({x1},{y1})-({x2},{y2}) → {crop_w}x{crop_h}')

    # ── 生成裁剪图 + transforms ──
    step_deg = calib['step_deg']
    R_cal = np.array(calib['R'])

    # 旋转轴方向（标定平面法向，世界 Y-up）
    axis = R_cal[2].copy()  # 平面法向 ≈ 转轴方向
    axis = axis / np.linalg.norm(axis)
    print(f'转轴方向: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]')

    # 估计相机距离和高度
    cam_pos_plane = -R_cal.T @ np.array(calib['t'])  # 相机在转台坐标系中的位置
    dist = np.linalg.norm(cam_pos_plane)
    cam_height = cam_pos_plane[1]  # Y ≈ 高度
    print(f'相机: dist={dist:.3f}m, height={cam_height:.3f}m')

    # 使用标定圆心 (3D 射线-平面交点)
    n_axis = R_cal[2].copy()
    # 圆心 = 光心沿视线方向与转台平面的交点 (简化: 用标定的 cx/cy)
    # 实际: center_3d 在转台平面上
    center_3d = np.array([calib['cx'], calib['cy'], 0.0])  # 归一化坐标近似

    frames_out = []
    skipped = 0

    for cf in color_files:
        idx = int(cf.stem)
        if idx not in masks_dict:
            skipped += 1
            continue

        img = cv2.imread(str(cf))
        mask = masks_dict[idx]

        # 裁剪 + 白底
        alpha = (mask > 128).astype(np.float32)
        cropped = img[y1:y2, x1:x2].copy()
        alpha_crop = alpha[y1:y2, x1:x2]

        # 白底合成
        white_bg = np.full_like(cropped, 255, dtype=np.uint8)
        composited = (cropped * alpha_crop[..., None]
                      + white_bg * (1.0 - alpha_crop[..., None])).astype(np.uint8)

        out_name = f'frame_{idx:03d}.png'
        cv2.imwrite(str(IMG_OUT / out_name), composited)

        # ── 相机外参: 相机绕转轴 orbit ──
        theta = np.radians(idx * step_deg)

        # 旋转轴 = 转轴方向（世界坐标）
        # 相机初始位置（frame 0）在转台坐标系
        # 每帧物体绕转轴转 θ，等效相机绕转轴转 -θ
        cam_init = cam_pos_plane.copy()
        # Orbit 旋转: 绕 axis 旋转 -θ
        cos_t = np.cos(-theta)
        sin_t = np.sin(-theta)
        # Rodrigues rotation around axis
        K = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        R_rot = np.eye(3) + sin_t * K + (1 - cos_t) * (K @ K)

        cam_pos = R_rot @ cam_init
        # 相机看向原点附近
        look_at = np.array([0.0, 0.0, 0.0])
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        # 上方向: 世界 Y 轴投影
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        # 相机到世界变换矩阵
        T = np.eye(4)
        T[:3, 0] = right
        T[:3, 1] = up
        T[:3, 2] = forward
        T[:3, 3] = cam_pos

        frames_out.append({
            'file_path': f'images/{out_name}',
            'transform_matrix': [[float(v) for v in row] for row in T],
        })

    # ── 写 transforms.json ──
    transform = {
        'camera_model': 'OPENCV',
        'fl_x': float(fx), 'fl_y': float(fy),
        'cx': float(new_cx), 'cy': float(new_cy),
        'w': int(crop_w), 'h': int(crop_h),
        'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0,
        'frames': frames_out,
    }

    with open(DST / 'transforms.json', 'w') as f:
        json.dump(transform, f, indent=2)

    print(f'\n预处理完成:')
    print(f'  图片: {len(frames_out)} 张 ({crop_w}x{crop_h})')
    print(f'  跳过: {skipped} 帧 (无有效 mask)')
    print(f'  目录: {DST}/')

    # ── 打包 ──
    zip_path = OUT_DIR / f'{name}_v22.zip'
    shutil.make_archive(str(OUT_DIR / f'{name}_v22'), 'zip', DST)
    zip_size = zip_path.stat().st_size / 1024
    print(f'  打包: {zip_path} ({zip_size:.0f} KB)')

    # ── 生成服务器脚本 + 提取 mesh Python 脚本 ──
    server_sh = OUT_DIR / 'v22_server.sh'
    server_sh.write_text(f'''#!/bin/bash
# V22 3DGS 服务器一键脚本
# 用法: bash v22_server.sh
set -e

PROJ=~/v22_{name}
echo "=== V22 3DGS: {name} ==="

# 1. 解压数据
mkdir -p $PROJ
cd $PROJ
unzip -o ~/{name}_v22.zip
echo "数据就绪: $(ls images/ | wc -l) 张图"

# 2. 安装 nerfstudio (首次)
if ! python3 -c "import nerfstudio" 2>/dev/null; then
    echo "安装 nerfstudio..."
    pip install nerfstudio open3d plyfile -q
fi

# 3. 训练 3DGS (高斯泼溅)
echo "开始 3DGS 训练..."
ns-train gaussian-splatting \\
    --data $PROJ \\
    --output-dir $PROJ/output \\
    --max-num-iterations 30000

# 4. 找最新 config
CONFIG=$(ls -t $PROJ/output/*/config.yml 2>/dev/null | head -1)
echo "模型: $CONFIG"

# 5. 渲染 + 提取 mesh
cp ~/v22_extract_mesh.py $PROJ/
python3 $PROJ/v22_extract_mesh.py --config "$CONFIG" --proj "$PROJ"

echo "=== V22 完成 ==="
echo "Mesh: $PROJ/output/mesh.ply"
''')

    # 提取 mesh 的 Python 脚本
    extract_py = OUT_DIR / 'v22_extract_mesh.py'
    extract_py.write_text('''#!/usr/bin/env python3
"""V22: 从训练好的 3DGS 模型渲染深度图 + TSDF 提取 mesh."""
import torch, numpy as np, open3d as o3d, json, argparse, math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='nerfstudio config.yml')
parser.add_argument('--proj', required=True, help='project directory')
parser.add_argument('--voxel', type=float, default=0.002, help='TSDF voxel (m)')
parser.add_argument('--sdf-trunc', type=float, default=0.008)
args = parser.parse_args()

proj = Path(args.proj)
config_path = Path(args.config)

print(f'加载模型: {config_path}')
from nerfstudio.utils.eval_utils import eval_setup
_, pipeline, _, _ = eval_setup(config_path)

with open(proj / 'transforms.json') as f:
    tf = json.load(f)

fx, fy, cx, cy = tf['fl_x'], tf['fl_y'], tf['cx'], tf['cy']
w, h = tf['w'], tf['h']
frames = tf['frames']
n_frames = len(frames)

intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=args.voxel, sdf_trunc=args.sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

print(f'渲染 {n_frames} 个视角...')

for i, frame in enumerate(frames):
    T = np.array(frame['transform_matrix'])
    c2w = T  # camera-to-world

    # 构建 nerfstudio camera
    from nerfstudio.cameras.cameras import Cameras, CameraType
    camera = Cameras(
        camera_to_worlds=torch.tensor([c2w], dtype=torch.float32, device='cuda'),
        fx=torch.tensor([fx], device='cuda'),
        fy=torch.tensor([fy], device='cuda'),
        cx=torch.tensor([cx], device='cuda'),
        cy=torch.tensor([cy], device='cuda'),
        width=torch.tensor([w], device='cuda'),
        height=torch.tensor([h], device='cuda'),
        camera_type=CameraType.PERSPECTIVE,
    )

    with torch.no_grad():
        outputs = pipeline.model.get_outputs(camera)
        rgb = outputs['rgb'].reshape(h, w, 3).cpu().numpy()
        depth = outputs['depth'].reshape(h, w).cpu().numpy()

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    depth_m = depth.squeeze()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image((depth_m * 1000).astype(np.uint16)),
        depth_scale=1000.0, depth_trunc=1.0,
        convert_rgb_to_intensity=False)

    volume.integrate(rgbd, intrinsic, np.linalg.inv(T))

    if i % 20 == 0:
        print(f'  [{i+1}/{n_frames}]')

print('提取 mesh...')
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

# 过滤远距离碎片
verts = np.asarray(mesh.vertices)
dists = np.linalg.norm(verts, axis=1)
keep = dists < 0.15
if keep.sum() > 100:
    keep_idx = np.where(keep)[0]
    tris = np.asarray(mesh.triangles)
    tri_keep = np.all(np.isin(tris, keep_idx), axis=1)
    old2new = -np.ones(len(verts), dtype=int)
    old2new[keep_idx] = np.arange(len(keep_idx))
    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(verts[keep_idx])
    mesh2.triangles = o3d.utility.Vector3iVector(old2new[tris[tri_keep]])
    mesh2.compute_vertex_normals()
    mesh = mesh2

out_path = proj / 'output' / 'mesh.ply'
o3d.io.write_triangle_mesh(str(out_path), mesh)
vf = np.asarray(mesh.vertices)
print(f'Mesh: {len(vf):,} verts, {len(mesh.triangles):,} faces')
print(f'尺寸: X={np.ptp(vf[:,0])*100:.1f} Y={np.ptp(vf[:,1])*100:.1f} Z={np.ptp(vf[:,2])*100:.1f} cm')
print(f'导出: {out_path}')
''')

    print(f'  服务器脚本: {server_sh}')
    print(f'  提取脚本: {extract_py}')
    print(f'\n下一步:')
    print(f'  scp {zip_path} luoz@192.168.100.55:~/')
    print(f'  scp {server_sh} luoz@192.168.100.55:~/')
    print(f'  scp {extract_py} luoz@192.168.100.55:~/')
    print(f'  在服务器: bash v22_server.sh')


# ╔══════════════════════════════════════════════════════════╗
# ║  Part 2: 后处理 — 缩放 + 清理                            ║
# ╚══════════════════════════════════════════════════════════╝

def postprocess_v22(name='chair'):
    """缩放 mesh 到真实尺寸."""
    DST = OUT_DIR / name
    mesh_path = DST / 'mesh_raw.ply'
    if not mesh_path.exists():
        print(f'错误: {mesh_path} 不存在，请先从服务器 scp mesh.ply')
        print(f'  scp luoz@192.168.100.55:~/v22_{name}/output/mesh.ply {mesh_path}')
        sys.exit(1)

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)

    if len(verts) < 100:
        print(f'错误: mesh 顶点太少 ({len(verts)})')
        sys.exit(1)

    # 3DGS 尺度是任意的，需要缩放到真实尺寸
    # 用 chair_dims.json 中的高度做参考
    dims_path = BASE / f'output/v12/{name}/chair_dims.json'
    if dims_path.exists():
        with open(dims_path) as f:
            dims = json.load(f)
        target_h = dims['height_m']  # 实际高度 (m)
    else:
        target_h = 0.091  # 默认 9.1cm

    current_h = verts[:, 1].max() - verts[:, 1].min()  # Y = 高度
    scale = target_h / current_h
    print(f'当前高度: {current_h*100:.1f}cm → 目标: {target_h*100:.1f}cm')
    print(f'缩放系数: {scale:.4f}')

    verts *= scale
    mesh.vertices = o3d.utility.Vector3dVector(verts)

    # 平移到 Y=0
    verts[:, 1] -= verts[:, 1].min()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()

    vf = np.asarray(mesh.vertices)
    print(f'最终: X={np.ptp(vf[:,0])*100:.1f}cm  '
          f'Y={np.ptp(vf[:,1])*100:.1f}cm  '
          f'Z={np.ptp(vf[:,2])*100:.1f}cm')

    out_path = DST / f'{name}_v22.ply'
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    o3d.io.write_triangle_mesh(str(DST / f'{name}_v22.obj'), mesh)
    print(f'导出: {out_path}')


# ╔══════════════════════════════════════════════════════════╗
# ║  Main                                                    ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description='V22 3DGS 重建')
    parser.add_argument('--name', default='chair')
    parser.add_argument('--post', action='store_true',
                        help='后处理: 缩放 mesh 到真实尺寸')
    args = parser.parse_args()

    if args.post:
        postprocess_v22(args.name)
    else:
        prepare_v22(args.name)


if __name__ == '__main__':
    main()
