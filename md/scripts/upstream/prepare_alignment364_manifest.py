#!/usr/bin/env python3
"""Freeze the 364-molecule, three-force-field alignment target.

The target is the union of the currently accepted CHARMM, GROMOS, and
corrected GLYCAM conformer sets.  The generated manifest records the exact
missing-force-field matrix and is intentionally fail-closed: names must map
one-to-one to the 375-molecule CASPER source corpus and the frozen union must
contain exactly 364 molecules.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_OUTPUT = MD_ROOT / "md_alignment364_v1" / "alignment364_targets.csv"

FORCEFIELD_DIRS = {
    "charmm": (
        "conformations_md_charmm36_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
    ),
    "gromos": (
        "conformations_md_gromos_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
    ),
    "glycam": (
        "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1",
        "conformations_md_glycam06j_enantiomer_v15_lowest100_seed1",
    ),
}

EXPECTED_CORPUS_SIZE = 375
EXPECTED_TARGET_SIZE = 364
EXPECTED_BASELINE_INTERSECTION = 326
EXPECTED_MISSING_RESULTS = {"charmm": 13, "gromos": 19, "glycam": 23}
EXPECTED_ABSENT_ALL = {
    "aDGal1P",
    "aDGlc1P",
    "aDMan1P",
    "aDMan2POMe",
    "aDMan3POMe",
    "aDMan4POMe",
    "aDMan6POMe",
    "aLFuc1P",
    "bDGal1P",
    "bDGlc1P",
    "bLFuc1P",
}


def clean_name(name: str) -> str:
    return name.replace("->", "").replace("(", "").replace(")", "")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pickle_names(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"conformer directory is missing: {directory}")
    return {path.stem for path in directory.glob("*.pickle")}


def forcefield_sets(md_root: Path = MD_ROOT) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for forcefield, directory_names in FORCEFIELD_DIRS.items():
        merged: set[str] = set()
        for directory_name in directory_names:
            merged |= pickle_names(md_root / directory_name)
        result[forcefield] = merged
    return result


def source_records(root: Path = ROOT) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    global_index = 0
    for source_class in ("mono", "dimer", "trimer"):
        source_sdf = root / "data" / "dataset" / f"casper_{source_class}.sdf"
        supplier = Chem.SDMolSupplier(str(source_sdf), removeHs=False)
        for source_index, molecule in enumerate(supplier):
            if molecule is None:
                raise ValueError(f"cannot parse molecule {source_index} from {source_sdf}")
            if not molecule.HasProp("_Name"):
                raise ValueError(f"molecule {source_index} in {source_sdf} has no _Name")
            molecule_name = molecule.GetProp("_Name")
            compact = clean_name(molecule_name)
            if compact in records:
                raise ValueError(f"duplicate compact molecule name in source corpus: {compact}")
            global_index += 1
            records[compact] = {
                "corpus_index": global_index,
                "source_index": source_index,
                "molecule_name": molecule_name,
                "clean_name": compact,
                "source_class": source_class,
                "source_sdf": str(source_sdf.resolve()),
                "source_sdf_sha256": sha256_file(source_sdf),
                "n_heavy_atoms": molecule.GetNumHeavyAtoms(),
            }
    if len(records) != EXPECTED_CORPUS_SIZE:
        raise ValueError(
            f"expected {EXPECTED_CORPUS_SIZE} unique CASPER molecules, found {len(records)}"
        )
    return records


def region_for(name: str, sets: dict[str, set[str]]) -> str:
    present = [forcefield for forcefield in ("charmm", "gromos", "glycam") if name in sets[forcefield]]
    return "+".join(present) if present else "none"


def build_rows(
    md_root: Path = MD_ROOT,
    root: Path = ROOT,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sets = forcefield_sets(md_root)
    records = source_records(root)
    union = set().union(*sets.values())
    intersection = set.intersection(*sets.values())
    corpus = set(records)
    extra = union - corpus
    if extra:
        raise ValueError(f"force-field outputs are absent from CASPER corpus: {sorted(extra)}")
    if len(union) != EXPECTED_TARGET_SIZE:
        raise ValueError(f"expected frozen union of {EXPECTED_TARGET_SIZE}, found {len(union)}")
    absent_all = corpus - union
    if absent_all != EXPECTED_ABSENT_ALL:
        raise ValueError(
            "unexpected molecules absent from all three force fields: "
            f"expected={sorted(EXPECTED_ABSENT_ALL)} observed={sorted(absent_all)}"
        )

    rows: list[dict[str, object]] = []
    missing_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    for priority, name in enumerate(sorted(union), 1):
        record = dict(records[name])
        missing = [forcefield for forcefield in ("charmm", "gromos", "glycam") if name not in sets[forcefield]]
        for forcefield in missing:
            missing_counts[forcefield] += 1
        region = region_for(name, sets)
        region_counts[region] += 1
        record.update(
            {
                "priority": priority,
                "alignment_priority": priority,
                "baseline_region": region,
                "baseline_has_charmm": name in sets["charmm"],
                "baseline_has_gromos": name in sets["gromos"],
                "baseline_has_glycam": name in sets["glycam"],
                "missing_forcefields": ";".join(missing),
                "missing_forcefield_count": len(missing),
                "target_forcefields": "charmm;gromos;glycam",
                "protocol": "explicit_water_5ns_200_candidates_select_100_lowest_sugar_potential",
            }
        )
        rows.append(record)

    if len(intersection) < EXPECTED_BASELINE_INTERSECTION:
        raise ValueError(
            f"intersection regressed below {EXPECTED_BASELINE_INTERSECTION}: {len(intersection)}"
        )
    if dict(missing_counts) != EXPECTED_MISSING_RESULTS:
        raise ValueError(
            f"unexpected missing-result matrix: expected={EXPECTED_MISSING_RESULTS} "
            f"observed={dict(missing_counts)}"
        )
    summary: dict[str, object] = {
        "schema_version": 1,
        "target_name": "three_forcefield_alignment364",
        "target_count": len(union),
        "casper_corpus_count": len(corpus),
        "baseline_forcefield_counts": {key: len(value) for key, value in sets.items()},
        "baseline_intersection_count": len(intersection),
        "baseline_union_count": len(union),
        "baseline_region_counts": dict(sorted(region_counts.items())),
        "missing_result_counts": {
            forcefield: missing_counts[forcefield]
            for forcefield in ("charmm", "gromos", "glycam")
        },
        "missing_result_total": sum(missing_counts.values()),
        "absent_from_all_count": len(absent_all),
        "absent_from_all": sorted(absent_all),
        "release_invariant": "charmm_names == gromos_names == glycam_names == alignment364_manifest",
        "protocol_invariant": {
            "production_ps": 5000,
            "candidate_stride_ps": 25,
            "candidate_count": 200,
            "selected_count": 100,
            "selection": "lowest isolated-sugar potential energy",
            "seed": 1,
        },
    }
    return rows, summary


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_job_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for row in rows:
        missing = [value for value in str(row["missing_forcefields"]).split(";") if value]
        for forcefield in missing:
            job = dict(row)
            job.update(
                {
                    "repair_forcefield": forcefield,
                    "alignment_job_rank": len(jobs) + 1,
                    "alignment_target_count": EXPECTED_TARGET_SIZE,
                }
            )
            jobs.append(job)
    if len(jobs) != sum(EXPECTED_MISSING_RESULTS.values()):
        raise ValueError(f"expected 55 alignment jobs, found {len(jobs)}")
    return jobs


def write_manifest(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path]:
    rows, summary = build_rows()
    fields = list(rows[0])
    with tempfile.TemporaryDirectory() as temporary_name:
        staged = Path(temporary_name) / output.name
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        payload = staged.read_bytes()
    _atomic_write(output, payload)
    summary["manifest"] = str(output.resolve())
    summary["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    jobs = build_job_rows(rows)
    job_output = output.with_name("alignment364_jobs.csv")
    job_fields = list(jobs[0])
    with tempfile.TemporaryDirectory() as temporary_name:
        staged_jobs = Path(temporary_name) / job_output.name
        with staged_jobs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=job_fields)
            writer.writeheader()
            writer.writerows(jobs)
        job_payload = staged_jobs.read_bytes()
    _atomic_write(job_output, job_payload)
    summary["job_manifest"] = str(job_output.resolve())
    summary["job_manifest_sha256"] = hashlib.sha256(job_payload).hexdigest()
    summary["job_count"] = len(jobs)
    metadata = output.with_suffix(".json")
    _atomic_write(
        metadata,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return output, metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    csv_path, json_path = write_manifest(args.output)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
