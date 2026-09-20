from __future__ import annotations

import numpy as np
import pyvista as pv
import trimesh


def to_trimesh(mesh: pv.PolyData) -> trimesh.Trimesh:
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.points),
        faces=faces,
        process=True,
        validate=True,
    )


def mesh_metrics(mesh: pv.PolyData) -> dict[str, object]:
    tri_mesh = to_trimesh(mesh)
    tri_mesh.remove_unreferenced_vertices()

    face_adjacency_angles = tri_mesh.face_adjacency_angles
    return {
        "vertex_count": int(len(tri_mesh.vertices)),
        "face_count": int(len(tri_mesh.faces)),
        "surface_area": float(tri_mesh.area),
        "is_watertight": bool(tri_mesh.is_watertight),
        "is_winding_consistent": bool(tri_mesh.is_winding_consistent),
        "body_count": int(tri_mesh.body_count),
        "euler_number": int(tri_mesh.euler_number),
        "bounds": tri_mesh.bounds.tolist() if tri_mesh.bounds is not None else None,
        "extents": tri_mesh.extents.tolist() if tri_mesh.extents is not None else None,
        "volume": float(tri_mesh.volume) if tri_mesh.is_watertight else None,
        "mean_edge_length": float(np.mean(tri_mesh.edges_unique_length))
        if len(tri_mesh.edges_unique_length)
        else 0.0,
        "mean_face_adjacency_angle_rad": float(np.mean(face_adjacency_angles))
        if len(face_adjacency_angles)
        else 0.0,
        "integral_mean_curvature": float(tri_mesh.integral_mean_curvature),
    }
