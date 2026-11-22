#!/usr/bin/env python3
"""Build the fail-closed manifest for sugars missing exactly one MD force field."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_SOURCE = MD_ROOT / "md_all_remaining_dataset_v13" / "md_all_remaining_glycans_v13.csv"
DEFAULT_OUTPUT = MD_ROOT / "third_forcefield_repair_v1" / "third_forcefield_repair_targets.csv"


TARGETS: dict[str, tuple[str, ...]] = {
    "gromos": (
        "aDMan",
        "aLFuc12[aDGal13]bDGal",
        "bDGal14aDXyl",
        "bDGlcNAc",
        "bDGlcNAc13bDGal",
        "bDQui",
        "bLFuc",
        "bLRha",
    ),
    "charmm": (
        "aAbe13[aDGal12]aDManOMe",
        "aDGal2AcOMe",
        "aDGal3AcOMe",
        "aDGal4AcOMe",
        "aDGal6AcOMe",
        "aDGalA14aDGalA",
        "aDGalA14bDGalA",
        "aDGalAOMe",
        "aDGlc11aDGlc",
        "aDGlc11bDGlc",
        "aDGlc2AcOMe",
        "aDGlc3AcOMe",
        "aDGlc4AcOMe",
        "aDGlc6AcOMe",
        "aDGlcA13aLGal",
        "aDGlcA13bLGal",
        "aDGlcAOMe",
        "aLRha2AcOMe",
        "aLRha3AcOMe",
        "aLRha4AcOMe",
        "bDGal2AcOMe",
        "bDGal3AcOMe",
        "bDGal4AcOMe",
        "bDGal6AcOMe",
        "bDGlc11bDGlc",
        "bDGlc2AcOMe",
        "bDGlc3AcOMe",
        "bDGlc4AcOMe",
        "bDGlc6AcOMe",
        "bDGlcAOMe",
    ),
    "glycam": (
        "aDAllNAc",
        "aDGlc3NAc",
        "aDIdo",
        "aDManA",
        "aDQui4NAc",
        "aDRha4NAc",
        "aDRha4NAcOMe",
        "aDTal6dOMe",
        "bDAllNAc",
        "bDGlc3NAc",
        "bDIdo",
        "bDTal6d2NAc4NAc",
    ),
}

INTERSECTION_AFTER_GROUP = {
    "gromos": 304,
    "charmm": 334,
    "glycam": 346,
}


def validate_targets() -> None:
    expected_sizes = {"gromos": 8, "charmm": 30, "glycam": 12}
    observed_sizes = {forcefield: len(names) for forcefield, names in TARGETS.items()}
    if observed_sizes != expected_sizes:
        raise ValueError(f"unexpected target group sizes: {observed_sizes}")
    flattened = [name for names in TARGETS.values() for name in names]
    if len(flattened) != 50 or len(set(flattened)) != 50:
        raise ValueError("repair target list must contain exactly 50 unique clean names")


def build_manifest(source: Path) -> tuple[list[str], list[dict[str, str]]]:
    validate_targets()
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or ())
        source_rows = list(reader)
    if "clean_name" not in source_fields:
        raise ValueError(f"source CSV has no clean_name column: {source}")

    by_clean: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for row in source_rows:
        clean = row["clean_name"]
        if clean in by_clean:
            duplicates.add(clean)
        by_clean[clean] = row
    if duplicates:
        raise ValueError("source CSV contains duplicate clean names: " + ", ".join(sorted(duplicates)))

    missing = sorted(
        clean
        for names in TARGETS.values()
        for clean in names
        if clean not in by_clean
    )
    if missing:
        raise ValueError("repair targets absent from source CSV: " + ", ".join(missing))

    extra_fields = [
        "repair_forcefield",
        "repair_group_size",
        "repair_rank",
        "strict_intersection_after_group",
    ]
    output_rows: list[dict[str, str]] = []
    rank = 0
    for forcefield in ("gromos", "charmm", "glycam"):
        for clean in TARGETS[forcefield]:
            rank += 1
            row = dict(by_clean[clean])
            row.update(
                {
                    "repair_forcefield": forcefield,
                    "repair_group_size": str(len(TARGETS[forcefield])),
                    "repair_rank": str(rank),
                    "strict_intersection_after_group": str(INTERSECTION_AFTER_GROUP[forcefield]),
                }
            )
            output_rows.append(row)
    return [*source_fields, *extra_fields], output_rows


def write_manifest(source: Path, output: Path) -> tuple[Path, Path]:
    fields, rows = build_manifest(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)

    metadata_path = output.with_suffix(".json")
    metadata = {
        "source_csv": str(source.resolve()),
        "manifest_csv": str(output.resolve()),
        "baseline_strict_intersection": 296,
        "target_count": len(rows),
        "groups": {
            forcefield: {
                "count": len(names),
                "strict_intersection_after_group": INTERSECTION_AFTER_GROUP[forcefield],
                "clean_names": list(names),
            }
            for forcefield, names in TARGETS.items()
        },
        "gate": "approved topology + strict preflight + stable smoke MD before production",
    }
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_temporary, metadata_path)
    return output, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    csv_path, json_path = write_manifest(args.source, args.output)
    print(f"wrote 50 repair targets: {csv_path}")
    print(f"wrote repair metadata: {json_path}")


if __name__ == "__main__":
    main()
