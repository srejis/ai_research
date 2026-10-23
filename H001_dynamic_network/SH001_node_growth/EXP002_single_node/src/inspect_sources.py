"""Check configured source files only. Does not instantiate or train a model."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/sources.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / cfg["baseline_results"] / "summary.json").read_text(encoding="utf-8"))
    for key in ("E_mid", "E_complete", "K"):
        assert cfg[key] == summary[key], key
    for key, relative in cfg["checkpoints"].items():
        path = (ROOT / relative).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg["expected_sha256"][key]
        print(f"{key}: {path} (SHA-256 OK)")
    assert (ROOT / cfg["shared_data"] / "mnist_split_seed42.pt").is_file()
    print("PASS: EXP001 references resolved; no model created, no training executed.")


if __name__ == "__main__":
    main()
