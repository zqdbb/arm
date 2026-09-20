#!/usr/bin/env python3
"""V21: rembg 背景移除 + COLMAP SfM/MVS + Poisson mesh."""
import numpy as np, cv2, json, subprocess, shutil, sys, time
from pathlib import Path

BASE = Path(__file__).parent
CAPTURE_DIR = BASE / 'output/v20_capture'
WORK = BASE / 'output/v21_colmap'
IMAGES = WORK / 'images'
SPARSE = WORK / 'sparse'
DENSE = WORK / 'dense'

# ── Step 1: rembg 背景移除 ──
print('=' * 50)
print('Step 1: rembg background removal')
print('=' * 50)

MASKED_DIR = WORK / 'masked'
MASKED_DIR.mkdir(parents=True, exist_ok=True)

from rembg import remove
from PIL import Image

n_frames = 72
t0 = time.time()
for i in range(n_frames):
    src = CAPTURE_DIR / f'color/{i:03d}.jpg'
    dst = MASKED_DIR / f'{i:03d}.png'
    if dst.exists():
        continue
    img = Image.open(str(src))
    # rembg 移除背景，替换为白色
    out = remove(img, bgcolor=(255, 255, 255, 255))
    out.save(str(dst))
    if (i + 1) % 18 == 0:
        print(f'  [{i+1}/{n_frames}] {time.time()-t0:.0f}s')
print(f'Done: {time.time()-t0:.0f}s')

# ── Step 2: COLMAP workspace ──
print('\n' + '=' * 50)
print('Step 2: COLMAP setup')
print('=' * 50)

IMAGES.mkdir(parents=True, exist_ok=True)
SPARSE.mkdir(parents=True, exist_ok=True)
DENSE.mkdir(parents=True, exist_ok=True)

# Copy masked images
for i in range(n_frames):
    src = MASKED_DIR / f'{i:03d}.png'
    dst = IMAGES / f'{i:03d}.png'
    if not dst.exists():
        shutil.copy2(str(src), str(dst))

# Create database
db_path = WORK / 'database.db'
if db_path.exists():
    db_path.unlink()

# Camera intrinsics (D435i color, 1280x720)
with open(CAPTURE_DIR / 'calib.json') as f:
    cal = json.load(f)

# 彩色相机内参 (全分辨率)
# calibration 是用 640x360 center crop 做的，fx=907.8
# 全分辨率 1280x720 的 fx 相同
fx_c = 907.8
fy_c = 905.7
cx_c = 640.0  # principal point at full res center
cy_c = 360.0

print(f'Camera: 1280x720 fx={fx_c:.1f} fy={fy_c:.1f}')

# ── Step 3: COLMAP feature extraction ──
print('\n' + '=' * 50)
print('Step 3: Feature extraction')
print('=' * 50)

subprocess.run([
    'colmap', 'feature_extractor',
    '--database_path', str(db_path),
    '--image_path', str(IMAGES),
    '--ImageReader.single_camera', '1',
    '--ImageReader.camera_model', 'SIMPLE_RADIAL',
    '--ImageReader.camera_params', f'{fx_c},{cx_c},{cy_c},0.0',
    '--SiftExtraction.max_num_features', '8192',
    '--SiftExtraction.estimate_affine_shape', '1',
    '--SiftExtraction.domain_size_pooling', '1',
], check=True)

# ── Step 4: Exhaustive matching ──
print('\n' + '=' * 50)
print('Step 4: Feature matching')
print('=' * 50)

subprocess.run([
    'colmap', 'exhaustive_matcher',
    '--database_path', str(db_path),
    '--SiftMatching.guided_matching', '1',
], check=True)

# ── Step 5: Sparse reconstruction (SfM) ──
print('\n' + '=' * 50)
print('Step 5: Sparse reconstruction')
print('=' * 50)

SPARSE0 = SPARSE / '0'
SPARSE0.mkdir(parents=True, exist_ok=True)

result = subprocess.run([
    'colmap', 'mapper',
    '--database_path', str(db_path),
    '--image_path', str(IMAGES),
    '--output_path', str(SPARSE),
    '--Mapper.ba_refine_focal_length', '0',
    '--Mapper.ba_refine_extra_params', '0',
], capture_output=True, text=True)

print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr[-500:])
    print('\nSfM failed. Trying with sequential matcher...')
    # Recreate DB
    if db_path.exists():
        db_path.unlink()
    subprocess.run([
        'colmap', 'feature_extractor',
        '--database_path', str(db_path),
        '--image_path', str(IMAGES),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        '--ImageReader.camera_params', f'{fx_c},{cx_c},{cy_c},0.0',
        '--SiftExtraction.max_num_features', '8192',
    ], check=True)
    subprocess.run([
        'colmap', 'sequential_matcher',
        '--database_path', str(db_path),
        '--SiftMatching.guided_matching', '1',
    ], check=True)
    result = subprocess.run([
        'colmap', 'mapper',
        '--database_path', str(db_path),
        '--image_path', str(IMAGES),
        '--output_path', str(SPARSE),
        '--Mapper.ba_refine_focal_length', '0',
        '--Mapper.ba_refine_extra_params', '0',
    ], capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

# Check if reconstruction succeeded
model_dirs = list(SPARSE.glob('*/images.bin')) + list(SPARSE.glob('*/images.txt'))
if not model_dirs:
    print('\nSfM still failed. Trying known-pose approach...')
    # Generate known poses from turntable geometry
    from scipy.spatial.transform import Rotation as R

    cam_dist = 0.42  # estimated
    # Write cameras.txt with known poses
    # ... (complex, skip for now)
    print('Known-pose approach not implemented. Exiting.')
    sys.exit(1)

# Find the largest model
model_dir = max((d.parent for d in SPARSE.glob('*/images.bin')), key=lambda p: len(list(p.glob('*'))))
print(f'\nUsing model: {model_dir}')

# ── Step 6: Image undistortion ──
print('\n' + '=' * 50)
print('Step 6: Image undistortion')
print('=' * 50)

subprocess.run([
    'colmap', 'image_undistorter',
    '--image_path', str(IMAGES),
    '--input_path', str(model_dir),
    '--output_path', str(DENSE),
    '--output_type', 'COLMAP',
], check=True)

# ── Step 7: Dense stereo ──
print('\n' + '=' * 50)
print('Step 7: Patch match stereo')
print('=' * 50)

subprocess.run([
    'colmap', 'patch_match_stereo',
    '--workspace_path', str(DENSE),
    '--workspace_format', 'COLMAP',
    '--PatchMatchStereo.geom_consistency', '1',
    '--PatchMatchStereo.max_image_size', '2000',
], check=True)

# ── Step 8: Stereo fusion ──
print('\n' + '=' * 50)
print('Step 8: Stereo fusion')
print('=' * 50)

subprocess.run([
    'colmap', 'stereo_fusion',
    '--workspace_path', str(DENSE),
    '--workspace_format', 'COLMAP',
    '--input_type', 'geometric',
    '--output_path', str(DENSE / 'fused.ply'),
], check=True)

# ── Step 9: Mesh (Poisson) ──
print('\n' + '=' * 50)
print('Step 9: Poisson mesh')
print('=' * 50)

subprocess.run([
    'colmap', 'poisson_mesher',
    '--input_path', str(DENSE / 'fused.ply'),
    '--output_path', str(DENSE / 'mesh.ply'),
    '--PoissonMeshing.trim', '7',
    '--PoissonMeshing.depth', '10',
], check=True)

# ── Step 10: Clean up with Open3D ──
print('\n' + '=' * 50)
print('Step 10: Post-process mesh')
print('=' * 50)

import open3d as o3d
import trimesh

mesh_o3d = o3d.io.read_triangle_mesh(str(DENSE / 'mesh.ply'))
verts = np.asarray(mesh_o3d.vertices)
print(f'Raw mesh: {len(verts)} verts, {len(mesh_o3d.triangles)} tris')

# Filter to chair region
d_xy = np.linalg.norm(verts[:, [0, 1]], axis=1)
keep = (d_xy < 0.08) & (verts[:, 2] > -0.05) & (verts[:, 2] < 0.10)
k_idx = np.where(keep)[0]

if len(k_idx) > 100:
    tris = np.asarray(mesh_o3d.triangles)
    tk = np.all(np.isin(tris, k_idx), axis=1)
    old = -np.ones(len(verts), dtype=int)
    old[k_idx] = np.arange(len(k_idx))
    mesh_f = o3d.geometry.TriangleMesh()
    mesh_f.vertices = o3d.utility.Vector3dVector(verts[k_idx])
    mesh_f.triangles = o3d.utility.Vector3iVector(old[tris[np.where(tk)[0]]])
    mesh_f.compute_vertex_normals()
else:
    mesh_f = mesh_o3d
    print('WARNING: chair filter too aggressive')

vf = np.asarray(mesh_f.vertices)
print(f'Filtered: {len(vf)} verts, {len(mesh_f.triangles)} tris')
print(f'Size: X={np.ptp(vf[:,0])*1000:.0f} Y={np.ptp(vf[:,1])*1000:.0f} Z={np.ptp(vf[:,2])*1000:.0f} mm')

OUT = BASE / 'output/v21'
OUT.mkdir(parents=True, exist_ok=True)
o3d.io.write_triangle_mesh(str(OUT / 'v21_mesh.ply'), mesh_f)
print(f'\nSaved: {OUT}/v21_mesh.ply')
print('V21 Done.')
