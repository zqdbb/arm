#!/usr/bin/env python3
"""
准备 3DGS Colab 输入数据:
  直接用原图 (不做 mask), 配合转台标定计算相机位姿.
  训练后用旋转中心过滤背景.

用法:
  python3 prep_for_3dgs.py --name chair
"""

import cv2, numpy as np, json, sys, argparse, struct, shutil
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / 'output/3dgs'


def generate_colmap_binary(calib, img_dir, out_dir):
    """生成 COLMAP 二进制格式的相机模型 (绕过 COLMAP SfM)."""
    fx = calib['fx_full']
    fy = calib['fy_full']
    cx = calib['ppx_full']
    cy = calib['ppy_full']
    w = calib['width_full']
    h = calib['height_full']

    img_files = sorted(Path(img_dir).glob('*.png'))
    if not img_files:
        img_files = sorted(Path(img_dir).glob('*.jpg'))
    n = len(img_files)

    dist = 0.42  # 相机到物体 ~42cm (V12 深度中位数实测)
    step_deg = 5

    # cameras.bin
    with open(out_dir / 'cameras.bin', 'wb') as f:
        f.write(struct.pack('<Q', 1))           # num_cameras (uint64)
        f.write(struct.pack('<Ii', 1, 0))       # camera_id (uint32), model (int32, 0=SIMPLE_PINHOLE)
        f.write(struct.pack('<QQ', w, h))        # width, height (uint64)
        f.write(struct.pack('<ddd', fx, cx, cy))

    # images.bin
    with open(out_dir / 'images.bin', 'wb') as f:
        f.write(struct.pack('<Q', n))
        for i, img_path in enumerate(img_files):
            name = img_path.name
            th = np.radians(i * step_deg)

            # 相机绕 Y 轴环绕 (等效转台旋转)
            cam = np.array([dist * np.sin(th), 0.0, dist * np.cos(th)])
            fwd = -cam / np.linalg.norm(cam)
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(up, fwd)
            if np.linalg.norm(right) < 1e-9:
                right = np.array([1.0, 0.0, 0.0])
            right /= np.linalg.norm(right)
            up = np.cross(fwd, right)

            R = np.vstack([right, up, fwd])
            t = -R @ cam

            # R → quaternion (w,x,y,z)
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / S
                qx = 0.25 * S
                qy = (R[0, 1] + R[1, 0]) / S
                qz = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / S
                qx = (R[0, 1] + R[1, 0]) / S
                qy = 0.25 * S
                qz = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / S
                qx = (R[0, 2] + R[2, 0]) / S
                qy = (R[1, 2] + R[2, 1]) / S
                qz = 0.25 * S

            f.write(struct.pack('<I', i + 1))          # image_id (uint32)
            f.write(struct.pack('<dddd', qw, qx, qy, qz))
            f.write(struct.pack('<ddd', t[0], t[1], t[2]))
            f.write(struct.pack('<I', 1))            # camera_id (uint32)
            f.write(name.encode('utf-8') + b'\x00')
            f.write(struct.pack('<Q', 0))  # no points2D

    # points3D.bin — 球体内随机采样作为初始化点云
    pts = []
    np.random.seed(42)
    for _ in range(1000):
        r = 0.05 * np.random.random() ** 0.33
        th = np.random.random() * 2 * np.pi
        phi = np.random.random() * np.pi
        x = r * np.sin(phi) * np.cos(th)
        y = r * np.sin(phi) * np.sin(th) - 0.02
        z = r * np.cos(phi)
        pts.append((x, y, z, 180, 140, 100))

    with open(out_dir / 'points3D.bin', 'wb') as f:
        f.write(struct.pack('<Q', len(pts)))
        for j, (x, y, z, r, g, b) in enumerate(pts):
            f.write(struct.pack('<Q', j + 1))
            f.write(struct.pack('<ddd', x, y, z))
            f.write(struct.pack('<BBB', r, g, b))
            f.write(struct.pack('<d', 1.0))
            f.write(struct.pack('<Q', 0))

    print(f'  COLMAP model: {n} cameras, {len(pts)} points')
    return n


def main():
    parser = argparse.ArgumentParser(description='准备 3DGS 数据')
    parser.add_argument('--name', default='chair')
    args = parser.parse_args()

    name = args.name
    SRC = BASE / f'output/v12/{name}'
    DST = OUT_DIR / name

    calib_path = SRC / 'calibrate.json'
    if not calib_path.exists():
        print(f'错误: 标定文件不存在 {calib_path}')
        sys.exit(1)
    with open(calib_path) as f:
        calib = json.load(f)

    # ── 复制原图 ──
    img_out = DST / 'input'
    img_out.mkdir(parents=True, exist_ok=True)

    color_dir = SRC / 'color'
    src_files = sorted(color_dir.glob('*.png'))
    if len(src_files) < 36:
        src_files = sorted(color_dir.glob('*.jpg'))

    for cf in src_files:
        shutil.copy2(cf, img_out / cf.name)
    print(f'原图: {len(src_files)} 张 → {img_out}')

    # ── 生成 COLMAP 二进制模型 ──
    sparse_dir = DST / 'sparse' / '0'
    sparse_dir.mkdir(parents=True, exist_ok=True)
    n = generate_colmap_binary(calib, img_out, sparse_dir)

    # ── 打包 ──
    zip_path = str(OUT_DIR / f'{name}_3dgs_v2.zip')
    shutil.make_archive(str(OUT_DIR / f'{name}_3dgs_v2'), 'zip', DST)
    size_kb = Path(zip_path).stat().st_size / 1024
    print(f'\n已打包: {zip_path} ({size_kb:.0f} KB)')
    print(f'\n目录结构:')
    print(f'  input/           — {n} 张原图')
    print(f'  sparse/0/        — COLMAP 二进制 (标定位姿)')
    print(f'\n下一步: 上传 {name}_3dgs_v2.zip 到 Colab')


if __name__ == '__main__':
    main()
