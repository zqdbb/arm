from __future__ import annotations

from pathlib import Path


REQUIRED_PATHS = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("pyproject.toml"),
    Path("Makefile"),
    Path("docs/architecture.md"),
    Path("docs/workflow.md"),
    Path("src/point_cloud_mesh_surface_analysis/cli.py"),
    Path("src/point_cloud_mesh_surface_analysis/pipeline.py"),
    Path("tests/test_pipeline.py"),
]


def main() -> int:
    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        raise SystemExit("Missing required paths:\n- " + "\n- ".join(missing))
    print("Scaffold validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
