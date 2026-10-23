"""Verify each pre-migration file remains at its mapped location."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED_CHANGED = {
    ".gitignore", "src/train_baseline.py", "src/verify_baseline.py", "src/future_expansion.py"
}


def main():
    mappings = json.loads((HERE / "mapping.json").read_text(encoding="utf-8"))
    checks = []
    for entry in mappings:
        path = ROOT / entry["new_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        equal = digest == entry["sha256"] and path.stat().st_size == entry["size"]
        assert equal or entry["path"] in EXPECTED_CHANGED, entry
        checks.append({**entry, "current_size": path.stat().st_size,
                       "current_sha256": digest, "byte_identical": equal})
    changed = {entry["path"] for entry in checks if not entry["byte_identical"]}
    assert changed == EXPECTED_CHANGED, changed
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p.name != "final_audit.json")
    for path in ROOT.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    for path in ROOT.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    result = {"original_file_count": len(checks), "original_total_bytes": sum(c["size"] for c in checks),
              "mapped_file_count": len(checks), "mapped_current_bytes": sum(c["current_size"] for c in checks),
              "byte_identical_count": sum(c["byte_identical"] for c in checks),
              "intentional_changes": sorted(changed), "checks": checks,
              "current_inventory_excluding_this_output": [{"path": p.relative_to(ROOT).as_posix(), "size": p.stat().st_size} for p in files]}
    (HERE / "final_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(checks)} originals retained; {result['byte_identical_count']} unchanged; 4 intentional code/management edits.")


if __name__ == "__main__":
    main()
