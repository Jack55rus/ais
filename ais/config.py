from pathlib import Path

from torch import cuda


class CommonConfig:
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    result_dir = project_dir / "results"
    model_dir = project_dir / "models"

    device = "cuda" if cuda.is_available() else "cpu"
