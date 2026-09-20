from __future__ import annotations

import numpy as np
import pyvista as pv

from .models import ReconstructionConfig


def reconstruct_mesh(
    points: np.ndarray,
    config: ReconstructionConfig,
) -> pv.PolyData:
    method = config.method.lower()
    cloud = pv.PolyData(points)

    if method == "surface":
        mesh = cloud.reconstruct_surface(
            nbr_sz=config.surface_neighbor_size,
            sample_spacing=config.sample_spacing,
            progress_bar=False,
        )
    elif method == "delaunay3d":
        volume = cloud.delaunay_3d(alpha=config.alpha)
        mesh = volume.extract_surface(algorithm="dataset_surface")
    elif method == "convex_hull":
        mesh = cloud.delaunay_3d(alpha=0.0).extract_surface(algorithm="dataset_surface")
    else:
        raise ValueError(f"Unsupported reconstruction method: {config.method}")

    mesh = mesh.triangulate().clean()

    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError("Mesh reconstruction produced an empty mesh")
    return mesh
