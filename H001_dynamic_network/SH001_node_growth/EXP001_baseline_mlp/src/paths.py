"""Resolve active paths from this experiment, independent of shell cwd."""
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def experiment_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (EXPERIMENT_ROOT / path).resolve()


def checkpoint_directory(run):
    return EXPERIMENT_ROOT / "checkpoints" / Path(run).name


def artifact_path(run, relative):
    """Resolve original manifest keys without rewriting historical records."""
    path = Path(relative)
    if path.parts[0] == "checkpoints":
        return checkpoint_directory(run).joinpath(*path.parts[1:])
    return Path(run) / path
