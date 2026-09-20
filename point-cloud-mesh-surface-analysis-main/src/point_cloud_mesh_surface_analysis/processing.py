from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .models import PreprocessConfig


def summarize_point_cloud(points: np.ndarray) -> dict[str, object]:
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    extents = max_bound - min_bound
    return {
        "point_count": int(points.shape[0]),
        "min_bound": min_bound.tolist(),
        "max_bound": max_bound.tolist(),
        "extents": extents.tolist(),
        "approx_scale": float(np.linalg.norm(extents)),
    }


def preprocess_point_cloud(
    points: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    if points.shape[0] <= config.nb_neighbors:
        filtered = points.copy()
    else:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=config.nb_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)
        threshold = mean_distances.mean() + (config.std_ratio * mean_distances.std())
        filtered = points[mean_distances <= threshold]

    if filtered.size == 0:
        raise ValueError("Point cloud became empty during outlier removal")

    if config.voxel_size > 0:
        filtered = voxel_downsample(filtered, config.voxel_size)

    if filtered.size == 0:
        raise ValueError("Point cloud became empty during preprocessing")
    return filtered


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return points
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    centroids = np.zeros((unique_keys.shape[0], 3), dtype=float)
    for idx in range(unique_keys.shape[0]):
        centroids[idx] = points[inverse == idx].mean(axis=0)
    return centroids


def detect_dominant_plane(
    points: np.ndarray,
    config: PreprocessConfig,
) -> dict[str, object]:
    total = max(points.shape[0], 1)
    if points.shape[0] < 3:
        return {
            "equation": [0.0, 0.0, 1.0, 0.0],
            "unit_normal": [0.0, 0.0, 1.0],
            "inlier_count": 0,
            "inlier_ratio": 0.0,
        }

    rng = np.random.default_rng(42)
    best_inlier_mask = np.zeros(points.shape[0], dtype=bool)
    best_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=float)

    for _ in range(config.plane_iterations):
        sample_indices = rng.choice(points.shape[0], size=3, replace=False)
        p0, p1, p2 = points[sample_indices]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            continue
        unit_normal = normal / normal_norm
        d = -np.dot(unit_normal, p0)
        distances = np.abs(points @ unit_normal + d)
        inlier_mask = distances <= config.plane_distance_threshold
        if inlier_mask.sum() > best_inlier_mask.sum():
            best_inlier_mask = inlier_mask
            best_plane = np.array([*unit_normal.tolist(), float(d)], dtype=float)

    a, b, c, d = best_plane
    return {
        "equation": [float(a), float(b), float(c), float(d)],
        "unit_normal": [float(a), float(b), float(c)],
        "inlier_count": int(best_inlier_mask.sum()),
        "inlier_ratio": float(best_inlier_mask.sum() / total),
    }


def cluster_summary(
    points: np.ndarray,
    config: PreprocessConfig,
) -> dict[str, object]:
    labels = dbscan(points, config.cluster_eps, config.cluster_min_points)
    if points.size == 0:
        return {"cluster_count": 0, "noise_count": 0, "largest_cluster_size": 0}

    valid = labels[labels >= 0]
    if valid.size == 0:
        return {
            "cluster_count": 0,
            "noise_count": int(np.sum(labels < 0)),
            "largest_cluster_size": 0,
        }

    counts = np.bincount(valid)
    return {
        "cluster_count": int(counts.size),
        "noise_count": int(np.sum(labels < 0)),
        "largest_cluster_size": int(counts.max(initial=0)),
    }


def dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.empty((0,), dtype=int)

    tree = cKDTree(points)
    labels = np.full(points.shape[0], -1, dtype=int)
    visited = np.zeros(points.shape[0], dtype=bool)
    cluster_id = 0

    for index in range(points.shape[0]):
        if visited[index]:
            continue
        visited[index] = True
        neighbors = tree.query_ball_point(points[index], eps)
        if len(neighbors) < min_points:
            labels[index] = -1
            continue

        labels[index] = cluster_id
        seeds = list(neighbors)
        seed_index = 0

        while seed_index < len(seeds):
            neighbor_index = seeds[seed_index]
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = tree.query_ball_point(points[neighbor_index], eps)
                if len(neighbor_neighbors) >= min_points:
                    for candidate in neighbor_neighbors:
                        if candidate not in seeds:
                            seeds.append(candidate)
            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
            seed_index += 1

        cluster_id += 1

    return labels
