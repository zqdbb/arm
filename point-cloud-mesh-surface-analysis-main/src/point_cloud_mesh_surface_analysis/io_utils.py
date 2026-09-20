from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv


def load_point_cloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".xyz":
        points = np.loadtxt(path, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        points = points[:, :3]
    else:
        dataset = pv.read(path)
        points = np.asarray(dataset.points, dtype=float)

    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    if points.shape[1] != 3:
        raise ValueError(f"Point cloud must have 3 columns: {path}")
    return points


def write_point_cloud(path: str | Path, points: np.ndarray) -> None:
    cloud = pv.PolyData(points)
    cloud.save(path)


def write_triangle_mesh(path: str | Path, mesh: pv.PolyData) -> None:
    mesh.save(path)
