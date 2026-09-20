from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class PreprocessConfig:
    voxel_size: float = 0.01
    nb_neighbors: int = 20
    std_ratio: float = 2.0
    plane_distance_threshold: float = 0.01
    plane_iterations: int = 1000
    cluster_eps: float = 0.03
    cluster_min_points: int = 15


@dataclass(slots=True)
class ReconstructionConfig:
    method: str = "surface"
    surface_neighbor_size: int = 20
    sample_spacing: float | None = None
    alpha: float = 0.0


@dataclass(slots=True)
class AnalysisResult:
    point_cloud: dict[str, Any]
    dominant_plane: dict[str, Any]
    clusters: dict[str, Any]
    mesh: dict[str, Any]
    files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
