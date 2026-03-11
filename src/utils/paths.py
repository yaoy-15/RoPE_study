from pathlib import Path


def get_project_root() -> Path:
    """Infer project root from this file location: <root>/src/utils/paths.py."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    """Create directory if missing and return the same Path object."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_outputs_dir() -> Path:
    return ensure_dir(get_project_root() / "outputs")


def get_tensors_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "tensors")


def get_figures_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "figures")


def get_reports_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "reports")


def build_tensor_output_dir(model_name: str, prompt_name: str) -> Path:
    """
    Build and create:
    outputs/tensors/{model_name}/{prompt_name}/
    """
    return ensure_dir(get_tensors_dir() / model_name / prompt_name)
