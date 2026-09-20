#!/usr/bin/env python3
"""V13: V12 Hyper3D 模型 + 深度数据校正比例.
   - 加载 V12 生成的椅子 mesh（形状正确）
   - 坐标系转换 (Z-up → Y-up)
   - 缩放到深度点云尺寸
   - 顶点拉向深度点云精修
"""
import numpy as np
from pathlib import Path
import open3d as o3d
import trimesh
import scipy.spatial as spatial

BASE = Path(__file__).parent
OUT_DIR = BASE / 'output/v13'
OUT_DIR.mkdir(parents=True, exist_ok=True)

V12_PATH = BASE / 'output/v12/chair/chair_v12.obj'
DEPTH_PATH = BASE / 'output/fuse/fused_chair_v2.ply'

# ═══════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════
print('=' * 50)
print('Step 1: 加载 V12 模型 + 深度点云')
print('=' * 50)

# V12 模型（OBJ 格式更可靠）
mesh_v12 = trimesh.load(str(V12_PATH), force='mesh')
print(f'V12 mesh: {len(mesh_v12.vertices):,} verts, {len(mesh_v12.faces):,} faces')

# 深度点云
pcd_depth = o3d.io.read_point_cloud(str(DEPTH_PATH))
pts_depth = np.asarray(pcd_depth.points)
print(f'深度点云: {len(pts_depth):,} pts')

# ═══════════════════════════════════════════════
# 2. 坐标系转换: Blender Z-up → 世界 Y-up
# ═══════════════════════════════════════════════
print(f'\n{"=" * 50}')
print('Step 2: 坐标系转换 (Z-up → Y-up)')
print('=' * 50)

v12_v = mesh_v12.vertices.copy()
print(f'V12 原始: X=宽={np.ptp(v12_v[:,0])*1000:.0f}mm  '
      f'Y=深={np.ptp(v12_v[:,1])*1000:.0f}mm  '
      f'Z=高={np.ptp(v12_v[:,2])*1000:.0f}mm')

# Blender (Z-up): X=右, Y=前, Z=上
# 世界 (Y-up):   X=右, Y=上, Z=前(深度)
# 映射: Blender X→World X, Blender Z→World Y, Blender Y→World Z
# 即绕 X 轴旋转 -90°
theta = np.radians(-90)
R_x = np.array([[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])
verts_world = (R_x @ v12_v.T).T

print(f'世界坐标: X=宽={np.ptp(verts_world[:,0])*1000:.0f}mm  '
      f'Y=高={np.ptp(verts_world[:,1])*1000:.0f}mm  '
      f'Z=深={np.ptp(verts_world[:,2])*1000:.0f}mm')

print(f'深度数据: X=宽={np.ptp(pts_depth[:,0])*1000:.0f}mm  '
      f'Y=高={np.ptp(pts_depth[:,1])*1000:.0f}mm  '
      f'Z=深={np.ptp(pts_depth[:,2])*1000:.0f}mm')

# ═══════════════════════════════════════════════
# 3. 缩放到深度数据尺寸
# ═══════════════════════════════════════════════
print(f'\n{"=" * 50}')
print('Step 3: 缩放到深度数据尺寸')
print('=' * 50)

v12_extent = np.ptp(verts_world, axis=0)
depth_extent = np.ptp(pts_depth, axis=0)
scale = depth_extent / v12_extent
print(f'V12尺寸:  X={v12_extent[0]*1000:.0f}  Y={v12_extent[1]*1000:.0f}  Z={v12_extent[2]*1000:.0f} mm')
print(f'深度尺寸: X={depth_extent[0]*1000:.0f}  Y={depth_extent[1]*1000:.0f}  Z={depth_extent[2]*1000:.0f} mm')
print(f'缩放系数: {scale}')

verts_world *= scale
print(f'缩放后:   X={np.ptp(verts_world[:,0])*1000:.0f}  '
      f'Y={np.ptp(verts_world[:,1])*1000:.0f}  '
      f'Z={np.ptp(verts_world[:,2])*1000:.0f} mm')

# 中心对齐
v12_center = verts_world.mean(axis=0)
depth_center = pts_depth.mean(axis=0)
offset = depth_center - v12_center
verts_world += offset
print(f'中心对齐: {offset*1000} mm')

# ═══════════════════════════════════════════════
# 4. 保存转换后的 V12 模型
# ═══════════════════════════════════════════════
mesh_world = o3d.geometry.TriangleMesh()
mesh_world.vertices = o3d.utility.Vector3dVector(verts_world)
mesh_world.triangles = o3d.utility.Vector3iVector(mesh_v12.faces)
mesh_world.compute_vertex_normals()
o3d.io.write_triangle_mesh(str(OUT_DIR / 'v12_world.ply'), mesh_world)
print(f'\n  → {OUT_DIR}/v12_world.ply')

# ═══════════════════════════════════════════════
# 5. 顶点拉向深度点云（精修可见面）
# ═══════════════════════════════════════════════
print(f'\n{"=" * 50}')
print('Step 4: 顶点拉向深度点云')
print('=' * 50)

tree = spatial.cKDTree(pts_depth)
dists, idxs = tree.query(verts_world, k=1)

PULL_THRESH = 0.012  # 12mm
PULL_WEIGHT = 0.6

near_mask = dists < PULL_THRESH
print(f'顶点: {near_mask.sum():,} 在{PULL_THRESH*1000:.0f}mm内 (拉到深度)')
print(f'       {(~near_mask).sum():,} 超出{PULL_THRESH*1000:.0f}mm (保留V12形状)')

verts_final = verts_world.copy()
verts_final[near_mask] = (1 - PULL_WEIGHT) * verts_world[near_mask] + PULL_WEIGHT * pts_depth[idxs[near_mask]]

mesh_final = o3d.geometry.TriangleMesh()
mesh_final.vertices = o3d.utility.Vector3dVector(verts_final)
mesh_final.triangles = o3d.utility.Vector3iVector(mesh_v12.faces)
mesh_final.compute_vertex_normals()

vf = np.asarray(mesh_final.vertices)
print(f'\n最终 mesh: {len(vf):,} verts')
print(f'  尺寸: X={np.ptp(vf[:,0])*1000:.0f}mm  Y={np.ptp(vf[:,1])*1000:.0f}mm  Z={np.ptp(vf[:,2])*1000:.0f}mm')
print(f'  目标: X=43mm  Y=88mm  Z=37mm')

o3d.io.write_triangle_mesh(str(OUT_DIR / 'chair_v13.ply'), mesh_final)
o3d.io.write_triangle_mesh(str(OUT_DIR / 'chair_v13.obj'), mesh_final)
print(f'\n完成 → {OUT_DIR}/chair_v13.ply')
