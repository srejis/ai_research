"""One-time relocation. Refuses overwrite; records every original file mapping."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXP = ROOT / "H001_dynamic_network/SH001_node_growth/EXP001_baseline_mlp"
EXP2 = EXP.parent / "EXP002_single_node"
RUN = "mnist_baseline_seed42_20261022"


def target(relative):
    p = Path(relative)
    if relative == "src/future_expansion.py":
        return EXP2 / "src/future_expansion.py"
    if p.parts[0] in {"src", "configs"}:
        return EXP / p
    if relative == "environment_setup.txt":
        return EXP / p
    if relative == "README.md":
        return EXP / "results" / RUN / "legacy_root_README.md"
    if p.parts[:3] == ("results", RUN, "checkpoints"):
        return EXP / "checkpoints" / RUN / p.name
    if p.parts[0] == "results":
        return EXP / p
    return ROOT / p


def main():
    before = json.loads((HERE / "before.json").read_text(encoding="utf-8"))
    mappings = []
    for entry in before["files"]:
        source = (ROOT / entry["path"]).resolve()
        destination = target(entry["path"]).resolve()
        if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):
            raise ValueError("Move outside workspace")
        if source != destination:
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        mappings.append({**entry, "new_path": destination.relative_to(ROOT).as_posix()})
    # Only empty former directories are removed; no recursive deletion.
    for name in ("src", "configs", "results"):
        parent = ROOT / name
        for directory in sorted((p for p in parent.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            directory.rmdir()
        parent.rmdir()
    (HERE / "mapping.json").write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")
    for exp in (EXP, EXP2):
        for name in ("src", "configs", "checkpoints", "results"):
            (exp / name).mkdir(parents=True, exist_ok=True)
    print("Moved original files without overwrite:", len(mappings))


if __name__ == "__main__":
    main()
