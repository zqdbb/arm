# Architecture

The pipeline is intentionally split into four layers:

1. `io_utils.py` handles file-based point cloud loading and export.
2. `processing.py` handles scan cleanup, downsampling, dominant-plane fitting, and clustering with NumPy and SciPy.
3. `meshing.py` reconstructs a surface with PyVista from the conditioned cloud.
4. `metrics.py` converts the result to Trimesh and calculates mesh quality metrics useful for industrial review.

`pipeline.py` orchestrates those layers and returns a serializable analysis result that the CLI writes to disk.

This keeps the core behaviors locally testable with synthetic geometry instead of requiring large real scan assets in the repository.