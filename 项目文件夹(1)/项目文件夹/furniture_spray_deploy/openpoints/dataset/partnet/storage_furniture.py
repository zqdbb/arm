"""
PartNet StorageFurniture semantic segmentation dataset.
Level-3 merged labels, 24 classes.

H5 format: {'data': (N,10000,3), 'label_seg': (N,10000), 'data_num': (N,)}
Labels 0-23 map to PartNet level-3 part IDs:
  0=background, 1=countertop, 2=shelf, 3=frame_vertical_bar, 4=back_panel,
  5=top_panel, 6=vertical_side_panel, 7=frame_horizontal_bar, 8=vertical_front_panel,
  9=bottom_panel, 10=vertical_divider_panel, 11=drawer_back, 12=drawer_bottom,
  13=drawer_side, 14=drawer_front, 15=drawer_handle, 16=base_bottom_panel,
  17=base_side_panel, 18=foot, 19=wheel, 20=caster_stem,
  21=hinge, 22=door_handle, 23=cabinet_door_surface

Door augmentation: during training, door surface + handle points are randomly
rotated around the hinge axis (determined by PCA on hinge points). This forces
the model to learn position-invariant door features, so it generalizes to
open-door cabinets in real scans.
"""

import os
import glob
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


PARTNET_LABEL_NAMES = [
    "background",           # 0
    "countertop",           # 1
    "shelf",                # 2
    "frame_vertical_bar",   # 3
    "back_panel",           # 4
    "top_panel",            # 5
    "vertical_side_panel",  # 6
    "frame_horizontal_bar", # 7
    "vertical_front_panel", # 8
    "bottom_panel",         # 9
    "vertical_divider_panel", # 10
    "drawer_back",          # 11
    "drawer_bottom",        # 12
    "drawer_side",          # 13
    "drawer_front",         # 14
    "drawer_handle",        # 15
    "base_bottom_panel",    # 16
    "base_side_panel",      # 17
    "foot",                 # 18
    "wheel",                # 19
    "caster_stem",          # 20
    "hinge",                # 21
    "door_handle",          # 22
    "cabinet_door_surface", # 23
]

# Semantic groups for spraying
BODY_LABELS = {4, 5, 6, 7, 8, 9, 10}  # back, top, side, front, bottom, divider
DOOR_LABELS = {23}                     # cabinet_door_surface
HINGE_LABELS = {21}                    # hinge
HANDLE_LABELS = {22}                   # door_handle
SHELF_LABELS = {2}                     # shelf
DRAWER_LABELS = {11, 12, 13, 14, 15}  # drawer parts
BASE_LABELS = {16, 17, 18, 19, 20}    # base/foot/wheel


def load_partnet_data(data_root, split_name):
    """Load PartNet H5 files for a given split."""
    import h5py
    files = sorted(glob.glob(os.path.join(data_root, f'*{split_name}*.h5')))
    if not files:
        raise FileNotFoundError(f"No H5 files found for split '{split_name}' in {data_root}")
    all_data, all_seg = [], []
    for h5_name in files:
        with h5py.File(h5_name, 'r') as f:
            all_data.append(f['data'][:].astype(np.float32))
            all_seg.append(f['label_seg'][:].astype(np.int64))
    return np.concatenate(all_data, axis=0), np.concatenate(all_seg, axis=0)


def _cluster_points(pts, threshold=0.08):
    """
    Cluster points by distance using union-find.
    O(n^2) brute-force but fine for < 2000 points.
    Returns list of index lists, each index is into pts.
    """
    n = len(pts)
    if n <= 1:
        return [list(range(n))]
    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pts[i] - pts[j]) < threshold:
                union(i, j)
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def _augment_door_open(pointcloud, seg):
    """
    Randomly rotate door surfaces + handles around their hinge axis.

    Only augments samples that have hinge (21) label points near the door.
    Samples without hinge labels are left unchanged — PCA fallback was tested
    but often picked the wrong axis, causing doors to open through the body.
    """
    door_mask = np.isin(seg, [22, 23])
    if door_mask.sum() < 30:
        return pointcloud

    hinge_mask = seg == 21
    hinge_pts = pointcloud[hinge_mask] if hinge_mask.sum() >= 2 else None
    door_indices = np.where(door_mask)[0]
    door_pts = pointcloud[door_mask]

    door_clusters = _cluster_points(door_pts, threshold=0.12)
    if not door_clusters:
        return pointcloud

    # --- Split wide door clusters (double doors) along X axis ---
    split_clusters = []
    for dc in door_clusters:
        if len(dc) < 50:
            continue
        dc = np.asarray(dc)
        dc_pts = door_pts[dc]
        x_range = dc_pts[:, 0].max() - dc_pts[:, 0].min()
        x_center = 0.5 * (dc_pts[:, 0].min() + dc_pts[:, 0].max())
        # Double-door detection: wide X span, center near x=0
        if x_range > 0.6 and abs(x_center) < 0.3:
            left = dc[dc_pts[:, 0] < 0]
            right = dc[dc_pts[:, 0] >= 0]
            if len(left) >= 30:
                split_clusters.append(left)
            if len(right) >= 30:
                split_clusters.append(right)
        else:
            split_clusters.append(dc)
    door_clusters = split_clusters

    for dc in door_clusters:
        if len(dc) < 50:
            continue
        dc_arr = np.asarray(dc)
        dc_door_pts = door_pts[dc_arr]
        dc_door_center = dc_door_pts.mean(axis=0)
        dc_global = door_indices[dc_arr]

        # --- Internal panel filter ---
        # A real cabinet door is a front-facing thin panel. Use SVD to check:
        # the minimum-variance direction (door normal) should point in Z (forward).
        # Internal shelves/dividers have normals in X or Y.
        dc_centered = dc_door_pts - dc_door_center
        try:
            Ud, Sd, Vtd = np.linalg.svd(dc_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if len(Sd) < 2:
            continue
        # Normal = last principal component (minimum variance)
        normal = Vtd[-1]
        # Door must face forward: |Z component| should dominate
        if abs(normal[2]) < 0.7:  # not a front-facing panel
            continue

        axis = None
        hing_center = None

        # Strategy 1: hinge-based SVD (most accurate)
        if hinge_pts is not None and len(hinge_pts) >= 2:
            hinge_dists = np.linalg.norm(hinge_pts - dc_door_center, axis=1)
            nearby_idx = np.where(hinge_dists < 0.5)[0]
            if len(nearby_idx) >= 4:
                nearby = hinge_pts[nearby_idx]
                hing_center = nearby.mean(axis=0)
                try:
                    U, S, Vt = np.linalg.svd(nearby - hing_center, full_matrices=False)
                except np.linalg.LinAlgError:
                    S = []
                if len(S) >= 2 and (S[1] <= 1e-6 or S[0] / S[1] >= 2.0):
                    axis = Vt[0]

        if axis is None:
            continue

        # Random open angle 0-90 degrees
        angle = np.random.uniform(0, np.pi / 2)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        subset = pointcloud[dc_global]
        subset_c = subset - hing_center
        cross = np.cross(axis, subset_c)
        dot = np.dot(subset_c, axis)[:, None]
        rotated = (subset_c * cos_a
                   + cross * sin_a
                   + axis[None, :] * dot * (1 - cos_a))
        pointcloud[dc_global] = rotated + hing_center

    return pointcloud


@DATASETS.register_module()
class PartNetStorageFurniture(Dataset):
    """
    PartNet StorageFurniture part segmentation dataset.
    Single-category: all samples are StorageFurniture with 24 part labels.

    During training, door parts (label 22, 23) are randomly rotated around
    their hinge axis (label 21) to simulate open-door configurations.
    """
    classes = {"StorageFurniture": 0}
    num_classes = 24

    cls_parts = {"storagefurniture": list(range(24))}
    cls2parts = [list(range(24))]
    cls2partembed = torch.eye(24)

    def __init__(self,
                 data_root='data/sem_seg_h5/StorageFurniture-3',
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 transform=None,
                 **kwargs):
        self.data_root = data_root
        self.num_points = num_points
        self.split = split
        self.transform = transform
        self.class_choice = class_choice

        split_map = {
            'trainval': ['train', 'val'],
            'train': ['train'],
            'val': ['val'],
            'test': ['test'],
        }
        parts = split_map.get(split, [split])

        all_data, all_seg = [], []
        for p in parts:
            try:
                d, s = load_partnet_data(data_root, p)
                all_data.append(d)
                all_seg.append(s)
            except FileNotFoundError:
                logging.warning(f"No {p} data found in {data_root}")

        if not all_data:
            raise FileNotFoundError(f"No data found for split '{split}' in {data_root}")

        self.data = np.concatenate(all_data, axis=0)
        self.seg = np.concatenate(all_seg, axis=0)
        logging.info(f"PartNet {split}: {len(self.data)} samples from {data_root}")

    def __getitem__(self, item):
        full_cloud = self.data[item]   # (N, 3)
        full_seg = self.seg[item]      # (N,)

        if self.split == 'train':
            # --- Smart sampling: always keep ALL hinge points (rare but essential) ---
            hinge_mask = full_seg == 21
            door_mask = np.isin(full_seg, [22, 23])

            hinge_idx = np.where(hinge_mask)[0]
            door_idx = np.where(door_mask)[0]
            other_idx = np.where(~hinge_mask & ~door_mask)[0]

            # Always include all hinge points (typically < 50, essential for augmentation)
            n_hinge = len(hinge_idx)
            # Door + handle: up to 40% of budget
            max_door = min(len(door_idx), int(self.num_points * 0.35))
            n_door = max_door
            if n_door > 0:
                door_idx = np.random.choice(door_idx, n_door, replace=False)
            else:
                door_idx = np.array([], dtype=np.int64)
            # Fill remaining with body/other points
            n_other = self.num_points - n_hinge - n_door
            if n_other > 0 and len(other_idx) > 0:
                other_idx = np.random.choice(other_idx, min(n_other, len(other_idx)), replace=False)
            else:
                other_idx = np.array([], dtype=np.int64)

            selected = np.concatenate([hinge_idx, door_idx, other_idx])
            np.random.shuffle(selected)
            pointcloud = full_cloud[selected].copy()
            seg = full_seg[selected].copy()

            # Door open augmentation
            pointcloud = _augment_door_open(pointcloud, seg)
        else:
            # Val/test: standard random sampling
            n_total = full_cloud.shape[0]
            if n_total >= self.num_points:
                indices = np.random.choice(n_total, self.num_points, replace=False)
            else:
                indices = np.random.choice(n_total, self.num_points, replace=True)
            pointcloud = full_cloud[indices].copy()
            seg = full_seg[indices].copy()

        cls = np.array([0], dtype=np.int64)

        data = {'pos': pointcloud,
                'cls': cls,
                'y': seg}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.data.shape[0]
