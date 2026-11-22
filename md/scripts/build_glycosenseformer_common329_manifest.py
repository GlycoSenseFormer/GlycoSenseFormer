#!/usr/bin/env python3
"""Freeze the strict common-329 conformer-source manifest for GlycoSenseFormer.

The manifest is deliberately driven by the fail-closed alignment364 audit.  It
does not infer GLYCAM provenance by globbing a single directory: each target is
resolved to the exact main or enantiomer-v15 source selected by the audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MD_ROOT = ROOT / "分子动力学模拟"
ALIGNMENT_ROOT = MD_ROOT / "md_alignment364_v1"
AUDIT_CSV = ALIGNMENT_ROOT / "baseline_output_audit.csv"
TARGET_CSV = ALIGNMENT_ROOT / "alignment364_targets.csv"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "glycosenseformer_conformer_source_v1"
    / "manifests"
    / "common329_sources.csv"
)

FROZEN_AUDIT_CSV_SHA256 = "b50eadf37dfcb8a1f4d2d59534d28122a6ff985a47c6dab8897cd00571a3f36f"
FROZEN_TARGET_CSV_SHA256 = "c704c1470606d5ea371eaae3d30b947177c89bc534aed37bfd224e9cf06670b8"

SOURCE_DIRS = {
    "charmm-main": {
        "all": MD_ROOT / "conformations_md_charmm36_exact_water_5ns_200cand_all_seed1",
        "low": MD_ROOT
        / "conformations_md_charmm36_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
    },
    "gromos-main": {
        "all": MD_ROOT / "conformations_md_gromos_exact_water_5ns_200cand_all_seed1",
        "low": MD_ROOT
        / "conformations_md_gromos_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
    },
    "glycam-main": {
        "all": MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_all_seed1",
        "low": MD_ROOT
        / "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1",
    },
    "glycam-enantiomer-v15": {
        "all": MD_ROOT / "conformations_md_glycam06j_enantiomer_v15_all_seed1",
        "low": MD_ROOT / "conformations_md_glycam06j_enantiomer_v15_lowest100_seed1",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_pickle(path: Path, expected_count: int, expected_atoms: int) -> None:
    with path.open("rb") as handle:
        conformers = pickle.load(handle)
    if not isinstance(conformers, (list, tuple)) or len(conformers) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} conformers, observed "
            f"{len(conformers) if hasattr(conformers, '__len__') else type(conformers).__name__}"
        )
    for index, conformer in enumerate(conformers):
        coords = np.asarray(conformer)
        if coords.shape != (expected_atoms, 3):
            raise ValueError(
                f"{path}: conformer {index} shape {coords.shape}, expected {(expected_atoms, 3)}"
            )
        if not np.isfinite(coords).all():
            raise ValueError(f"{path}: conformer {index} contains non-finite coordinates")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stratified_folds(rows: list[dict[str, Any]], n_folds: int, seed: int) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    rng = random.Random(seed)
    assignments: dict[str, int] = {}
    by_class: dict[str, list[str]] = {}
    for row in rows:
        by_class.setdefault(str(row["source_class"]), []).append(str(row["clean_name"]))
    for source_class in sorted(by_class):
        names = sorted(by_class[source_class])
        rng.shuffle(names)
        for index, name in enumerate(names):
            assignments[name] = index % n_folds
    return assignments


def build_rows(
    *,
    n_folds: int = 5,
    fold_seed: int = 20260717,
    validate_payloads: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_audit_sha = sha256_file(AUDIT_CSV)
    observed_target_sha = sha256_file(TARGET_CSV)
    if observed_audit_sha != FROZEN_AUDIT_CSV_SHA256:
        raise ValueError(
            f"alignment audit changed: expected {FROZEN_AUDIT_CSV_SHA256}, "
            f"observed {observed_audit_sha}"
        )
    if observed_target_sha != FROZEN_TARGET_CSV_SHA256:
        raise ValueError(
            f"alignment target manifest changed: expected {FROZEN_TARGET_CSV_SHA256}, "
            f"observed {observed_target_sha}"
        )

    targets = {row["clean_name"]: row for row in read_csv(TARGET_CSV)}
    audit_by_name: dict[str, dict[str, dict[str, str]]] = {}
    for row in read_csv(AUDIT_CSV):
        audit_by_name.setdefault(row["clean_name"], {})[row["forcefield"]] = row

    common_names = sorted(
        name
        for name, forcefields in audit_by_name.items()
        if all(forcefields.get(ff, {}).get("status") == "valid" for ff in ("charmm", "gromos", "glycam"))
    )
    if len(common_names) != 329:
        raise ValueError(f"expected strict common set of 329, observed {len(common_names)}")

    rows: list[dict[str, Any]] = []
    for name in common_names:
        target = targets[name]
        expected_atoms = int(target["n_heavy_atoms"])
        out: dict[str, Any] = {
            "clean_name": name,
            "molecule_name": target["molecule_name"],
            "source_class": target["source_class"],
            "n_heavy_atoms": expected_atoms,
            "source_sdf": target["source_sdf"],
            "source_index": int(target["source_index"]),
        }
        for forcefield in ("charmm", "gromos", "glycam"):
            audit = audit_by_name[name][forcefield]
            source = audit["output_source"]
            if source not in SOURCE_DIRS:
                raise ValueError(f"{name}/{forcefield}: unsupported audited source {source!r}")
            all_path = SOURCE_DIRS[source]["all"] / f"{name}.pickle"
            low_path = SOURCE_DIRS[source]["low"] / f"{name}.pickle"
            if not all_path.is_file() or not low_path.is_file():
                raise FileNotFoundError(f"{name}/{forcefield}: missing {all_path} or {low_path}")
            all_sha = sha256_file(all_path)
            low_sha = sha256_file(low_path)
            if all_sha != audit["all_pickle_sha256"]:
                raise ValueError(f"{name}/{forcefield}: all200 SHA does not match frozen audit")
            if low_sha != audit["low_pickle_sha256"]:
                raise ValueError(f"{name}/{forcefield}: low100 SHA does not match frozen audit")
            if validate_payloads:
                validate_pickle(all_path, 200, expected_atoms)
                validate_pickle(low_path, 100, expected_atoms)
            out.update(
                {
                    f"{forcefield}_output_source": source,
                    f"{forcefield}_all200_path": str(all_path.resolve()),
                    f"{forcefield}_all200_sha256": all_sha,
                    f"{forcefield}_low100_path": str(low_path.resolve()),
                    f"{forcefield}_low100_sha256": low_sha,
                }
            )

        mmff_path = ROOT / "data" / "conformations_paper_100_mmff_seed1" / f"{name}.pickle"
        if not mmff_path.is_file():
            raise FileNotFoundError(f"{name}: missing MMFF baseline {mmff_path}")
        with mmff_path.open("rb") as handle:
            mmff_conformers = pickle.load(handle)
        mmff_count = len(mmff_conformers)
        if mmff_count not in {87, 100}:
            raise ValueError(f"{name}: unexpected MMFF baseline count {mmff_count}")
        for index, conformer in enumerate(mmff_conformers):
            coords = np.asarray(conformer)
            if coords.shape != (expected_atoms, 3) or not np.isfinite(coords).all():
                raise ValueError(f"{name}: invalid MMFF conformer {index}")
        out.update(
            {
                "mmff_paper100_path": str(mmff_path.resolve()),
                "mmff_paper100_sha256": sha256_file(mmff_path),
                "mmff_paper100_count": mmff_count,
            }
        )
        rows.append(out)

    assignments = stratified_folds(rows, n_folds, fold_seed)
    for row in rows:
        row["fold"] = assignments[str(row["clean_name"])]

    class_counts = Counter(str(row["source_class"]) for row in rows)
    fold_counts = Counter(int(row["fold"]) for row in rows)
    glycam_sources = Counter(str(row["glycam_output_source"]) for row in rows)
    mmff_counts = Counter(int(row["mmff_paper100_count"]) for row in rows)
    summary = {
        "schema_version": 1,
        "experiment": "glycosenseformer_conformer_source_v1",
        "molecule_count": len(rows),
        "source_class_counts": dict(sorted(class_counts.items())),
        "fold_count": n_folds,
        "fold_seed": fold_seed,
        "fold_counts": {str(key): value for key, value in sorted(fold_counts.items())},
        "glycam_source_counts": dict(sorted(glycam_sources.items())),
        "mmff_baseline_conformer_counts": {
            str(key): value for key, value in sorted(mmff_counts.items())
        },
        "source_audit_csv": str(AUDIT_CSV.resolve()),
        "source_audit_csv_sha256": observed_audit_sha,
        "source_target_csv": str(TARGET_CSV.resolve()),
        "source_target_csv_sha256": observed_target_sha,
        "payloads_fully_loaded": validate_payloads,
    }
    return rows, summary


def write_manifest(
    output: Path = DEFAULT_OUTPUT,
    *,
    n_folds: int = 5,
    fold_seed: int = 20260717,
    validate_payloads: bool = True,
) -> tuple[Path, Path]:
    rows, summary = build_rows(
        n_folds=n_folds, fold_seed=fold_seed, validate_payloads=validate_payloads
    )
    fields = list(rows[0])
    with tempfile.TemporaryDirectory() as temporary_name:
        staged = Path(temporary_name) / output.name
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        payload = staged.read_bytes()
    _atomic_output = output.resolve()
    atomic_write(_atomic_output, payload)
    summary.update(
        {
            "manifest": str(_atomic_output),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    metadata = output.with_suffix(".json").resolve()
    atomic_write(
        metadata,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return _atomic_output, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260717)
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Verify file hashes but skip full pickle payload loading (not for formal release).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output, metadata = write_manifest(
        args.output,
        n_folds=args.folds,
        fold_seed=args.fold_seed,
        validate_payloads=not args.hash_only,
    )
    print(output)
    print(metadata)


if __name__ == "__main__":
    main()
