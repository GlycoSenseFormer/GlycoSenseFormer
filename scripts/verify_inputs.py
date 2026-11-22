"""Verify the original conformers, the three example systems and optional checkpoints."""
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(root, relative_path):
    with (root / relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT, help="Directory with archived conformers and MD inputs")
    parser.add_argument("--weights", action="store_true", help="Also verify all archived checkpoints")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    args = parser.parse_args()
    root = args.artifact_root
    if not (root / "data/paper375/manifest.csv").is_file():
        parser.error("Conformer manifest not found. Supply --artifact-root.")
    molecules = records(root, "data/paper375/manifest.csv")
    if len({row["molecule"] for row in molecules}) != 375:
        raise ValueError("Unexpected manuscript conformer membership")
    for row in molecules:
        verify(root / "data/paper375" / row["conformer_file"], row["sha256"])

    systems = [row for row in records(root, "md/systems.csv") if row["molecule_id"] == "aDMan"]
    archives = [row for row in records(root, "md/preparation/manifest.csv") if row["molecule_id"] == "aDMan"]
    for rows in (systems, archives):
        if sorted(row["source"] for row in rows) != ["charmm", "glycam", "gromos"]:
            raise ValueError("Expected one aDMan example from each MD source")
    for row in systems:
        for kind in ("candidate", "selected"):
            verify(root / row[kind + "_path"], row[kind + "_sha256"])
        for name in (row["solute_topology"], row["water_topology"], "md_5ns_water_200cand.tpr"):
            if not (root / row["system_dir"] / name).is_file():
                raise FileNotFoundError(name)
    for row in archives:
        verify(root / row["archive"], row["sha256"])
    print("375 original conformer files and 3 MD examples verified")
    if args.weights:
        weights = records(root, "configs/checkpoint_sha256.csv")
        for row in weights:
            path = args.checkpoint_dir / Path(row["checkpoint"]).relative_to("checkpoints")
            verify(path, row["sha256"])
        print(f"{len(weights)} archived checkpoints verified")


if __name__ == "__main__":
    main()
