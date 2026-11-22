#!/usr/bin/env python3
"""Prepare the exact GLYCAM-supported subset of the frozen alignment-364 set.

The workflow is intentionally fail closed: a target is prepared only when an
exact official GLYCAM06j residue path exists and all topology, strict-grompp,
atom-map, geometry, and three-way chirality checks pass.  Unsupported targets
receive structured blockers and never receive a .top or .gro file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import start_extended24_glycam_topology_dryrun as glycam
from build_glycam_rigid_declashed import prepare_one


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_MANIFEST = MD_ROOT / "md_alignment364_v1" / "alignment364_targets.csv"
DEFAULT_OUTPUT_ROOT = MD_ROOT / "md_alignment364_v1" / "glycam"
MMFF_ROOT = ROOT / "data" / "conformations_paper_100_mmff_seed1"
OFFICIAL_PREP = Path(
    "/home/cns/miniconda3/pkgs/"
    "ambertools-24.8-cuda_None_nompi_py312hbe1986e_101/dat/leap/prep/"
    "GLYCAM_06j-1.prep"
)
OFFICIAL_CODES = Path(
    "/home/cns/miniconda3/pkgs/"
    "ambertools-24.8-cuda_None_nompi_py312hbe1986e_101/AmberTools/src/"
    "cpptraj/dat/GlycamCodes.csv"
)

EXPECTED_CLEAN_NAMES = frozenset(
    {
        "aDAllNAc",
        "aDGlc3NAc",
        "aDIdo",
        "aDManA",
        "aDQui4NAc",
        "aDRha4N",
        "aDRha4NAc",
        "aDRha4NAcOMe",
        "aDRha4NOMe",
        "aDTal6d2NAc4NAc",
        "aDTal6dOMe",
        "aLFuc12bDGal13bD2dGlcOMe",
        "aPar13[aDGal12]aDManOMe",
        "aTyv13[aDGal12]aDManOMe",
        "bDAllNAc",
        "bDGlc2dOMe",
        "bDGlc3NAc",
        "bDGlcp4dOMe",
        "bDIdo",
        "bDManA",
        "bDQui4NAc",
        "bDTal6d2NAc4NAc",
        "bLFuc12bDGal13bD2dGlcOMe",
    }
)
EXACT_SUPPORTED_CLEAN_NAMES = frozenset({"aTyv13[aDGal12]aDManOMe"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_targets(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if "glycam" in row["missing_forcefields"].split(";")
        ]
    observed = {row["clean_name"] for row in rows}
    if observed != EXPECTED_CLEAN_NAMES or len(rows) != 23:
        raise ValueError(
            "frozen missing-GLYCAM identity set changed: "
            f"count={len(rows)}, missing={sorted(EXPECTED_CLEAN_NAMES - observed)}, "
            f"extra={sorted(observed - EXPECTED_CLEAN_NAMES)}"
        )
    for row in rows:
        row["priority"] = row["alignment_priority"]
        row["heavy_atoms"] = row["n_heavy_atoms"]
        mmff = (MMFF_ROOT / f"{row['clean_name']}.pickle").resolve()
        if not mmff.is_file():
            raise FileNotFoundError(f"MMFF conformer file missing: {mmff}")
        row["mmff100_pickle"] = str(mmff)
        source_sdf = Path(row["source_sdf"])
        if not source_sdf.is_file():
            raise FileNotFoundError(f"source SDF missing: {source_sdf}")
        if sha256(source_sdf) != row["source_sdf_sha256"]:
            raise ValueError(f"source SDF hash changed: {source_sdf}")
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def glycam_code_rows() -> dict[str, dict[str, str]]:
    with OFFICIAL_CODES.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        row["Abbreviation"]: row
        for row in rows
        if row["Abbreviation"]
        in {
            "AllNAc",
            "GlcN",
            "QuiNAc",
            "ManA",
            "RhaNAc",
            "Tyv",
            "6dTal",
            "Ido",
            "Abe",
            "Par",
            "Bac",
        }
    }
    return selected


def make_audit(manifest: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    active_templates = glycam.local_glycam_residue_names(glycam.GLYCAM_PREP)
    official_templates = glycam.local_glycam_residue_names(OFFICIAL_PREP)
    if len(active_templates) != 1456 or len(official_templates) != 1456:
        raise ValueError(
            "unexpected GLYCAM06j template count: "
            f"active={len(active_templates)}, official={len(official_templates)}"
        )
    if active_templates != official_templates:
        raise ValueError("active and AmberTools-24.8 GLYCAM template sets differ")

    targets: list[dict[str, Any]] = []
    for row in rows:
        molecule = row["molecule_name"]
        sequence, reason = glycam.resolve_glycam_sequence_with_reason(molecule)
        required = glycam.required_glycam_tokens(molecule, sequence) if sequence else []
        targets.append(
            {
                "priority": int(row["priority"]),
                "molecule_name": molecule,
                "clean_name": row["clean_name"],
                "resolver_status": "exact_official" if sequence else "blocked",
                "sequence": sequence or [],
                "required_residue_tokens": required,
                "missing_active_tokens": (
                    glycam.missing_local_glycam_tokens(molecule, sequence)
                    if sequence
                    else []
                ),
                "branch_acceptor_positions": list(
                    glycam.branch_linkage_positions(molecule) or ()
                ),
                "reason": reason,
                "rejected_near_match_tokens": glycam.GLYCAM06J_REJECTED_NEAR_MATCHES.get(
                    molecule, []
                ),
            }
        )
    resolved = {item["clean_name"] for item in targets if item["resolver_status"] == "exact_official"}
    if resolved != EXACT_SUPPORTED_CLEAN_NAMES:
        raise ValueError(f"exact-supported GLYCAM set changed: {sorted(resolved)}")

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": sha256(manifest),
        "frozen_target_count": len(rows),
        "resolver_summary": {"exact_official": 1, "blocked": 22},
        "policy": {
            "gaff_fallback": "forbidden",
            "near_stereoisomer_template": "forbidden",
            "unvalidated_custom_charge": "forbidden",
            "production_md": "not_started_by_this_workflow",
        },
        "active_library": {
            "tleap": str(glycam.TLEAP),
            "leaprc": str(glycam.GLYCAM_LEAPRC),
            "leaprc_sha256": sha256(glycam.GLYCAM_LEAPRC),
            "prep": str(glycam.GLYCAM_PREP),
            "prep_sha256": sha256(glycam.GLYCAM_PREP),
            "unique_template_count": len(active_templates),
        },
        "official_ambertools_24_8_library": {
            "prep": str(OFFICIAL_PREP),
            "prep_sha256": sha256(OFFICIAL_PREP),
            "unique_template_count": len(official_templates),
            "template_set_identical_to_active": active_templates == official_templates,
            "codes_csv": str(OFFICIAL_CODES),
            "codes_csv_sha256": sha256(OFFICIAL_CODES),
            "relevant_mapping_rows": glycam_code_rows(),
        },
        "derivative_rule_audit": {
            "official_tokens_present": {
                code: code in active_templates for code in ("ACX", "MEX", "SO3")
            },
            "target_applicability": "none",
            "reason": (
                "ACX performs O-acetylation of an existing hydroxyl; it cannot create "
                "Glc3NAc, Qui4NAc, Rha4NAc, or a 2,4-di-N-acetyl amino sugar"
            ),
        },
        "connection_rule_audit": {
            "supported_target": "aTyv13[aDGal12]aDManOMe",
            "sequence": ["OME", "ZMA", "0LA", "0TV"],
            "branch_acceptor_positions": [2, 3],
            "build_mode": "LEaP tail sequence with explicit central O2/O3 tails",
            "all_required_tokens_present": True,
        },
        "targets": targets,
    }


def run_workflow(
    manifest: Path,
    output_root: Path,
    minimum_clearance_angstrom: float,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_targets(manifest)
    derived_csv = output_root / "alignment364_glycam_targets.csv"
    write_csv(derived_csv, rows)
    (output_root / "alignment364_glycam_targets.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audit = make_audit(manifest, rows)
    audit_path = output_root / "alignment364_glycam_template_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    glycam.RUN_ROOT = output_root
    glycam.SAMPLE_CSV = derived_csv
    statuses: list[dict[str, Any]] = []
    for row in rows:
        sequence, reason = glycam.resolve_glycam_sequence_with_reason(row["molecule_name"])
        if sequence is not None:
            try:
                prepared = prepare_one(row, output_root, minimum_clearance_angstrom)
                statuses.append(
                    {
                        "priority": int(row["priority"]),
                        "molecule_name": row["molecule_name"],
                        "clean_name": row["clean_name"],
                        "final_status": "prepared",
                        "glycam_sequence": " ".join(sequence),
                        "reason": "exact official templates; all strict validation gates passed",
                        "run_dir": prepared["output_dir"],
                        "strict_grompp": True,
                        "triple_chirality": True,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                run_dir = output_root / row["clean_name"] / "glycam06j_topology_dryrun"
                run_dir.mkdir(parents=True, exist_ok=True)
                blocker = {
                    "schema_version": 1,
                    "mapping_status": "validation_failed",
                    "molecule_name": row["molecule_name"],
                    "clean_name": row["clean_name"],
                    "glycam_sequence": sequence,
                    "reason": repr(exc),
                    "production_eligibility": "blocked",
                }
                (run_dir / "glycam06j_validation_blocker.json").write_text(
                    json.dumps(blocker, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                statuses.append(
                    {
                        "priority": int(row["priority"]),
                        "molecule_name": row["molecule_name"],
                        "clean_name": row["clean_name"],
                        "final_status": "blocked",
                        "glycam_sequence": " ".join(sequence),
                        "reason": repr(exc),
                        "run_dir": str(run_dir),
                        "strict_grompp": False,
                        "triple_chirality": False,
                    }
                )
        else:
            result = glycam.run_one(row, run_acpype=True)
            run_dir = Path(result.run_dir)
            top_count = len(list(run_dir.rglob("*.top")))
            gro_count = len(list(run_dir.rglob("*.gro")))
            if top_count or gro_count:
                raise ValueError(
                    f"blocked target unexpectedly contains topology artifacts: {run_dir}"
                )
            statuses.append(
                {
                    "priority": int(row["priority"]),
                    "molecule_name": row["molecule_name"],
                    "clean_name": row["clean_name"],
                    "final_status": "blocked",
                    "glycam_sequence": "",
                    "reason": reason,
                    "run_dir": str(run_dir),
                    "strict_grompp": False,
                    "triple_chirality": False,
                }
            )

    prepared_count = sum(row["final_status"] == "prepared" for row in statuses)
    blocked_count = sum(row["final_status"] == "blocked" for row in statuses)
    if (prepared_count, blocked_count) != (1, 22):
        raise ValueError(
            f"unexpected execution result: prepared={prepared_count}, blocked={blocked_count}"
        )
    write_csv(output_root / "alignment364_glycam_status.csv", statuses)
    report = {
        "schema_version": 1,
        "prepared": prepared_count,
        "blocked": blocked_count,
        "production_md_started": False,
        "template_audit": str(audit_path),
        "targets_csv": str(derived_csv),
        "results": statuses,
    }
    (output_root / "alignment364_glycam_status.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Alignment-364 GLYCAM06j preparation",
        "",
        f"- frozen missing-GLYCAM targets: {len(rows)}",
        f"- prepared with exact official templates: {prepared_count}",
        f"- blocked without topology artifacts: {blocked_count}",
        "- 5 ns production MD started: no",
        "- policy: no GAFF fallback, no near-stereoisomer substitution, no unvalidated custom charge",
        "",
        "| priority | target | status | GLYCAM sequence |",
        "|---:|---|---|---|",
    ]
    for row in statuses:
        lines.append(
            f"| {row['priority']} | `{row['clean_name']}` | {row['final_status']} | "
            f"`{row['glycam_sequence']}` |"
        )
    (output_root / "alignment364_glycam_status.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed GLYCAM preparation for the frozen alignment-364 manifest."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--minimum-clearance-angstrom", type=float, default=1.20)
    args = parser.parse_args()
    report = run_workflow(
        args.manifest.resolve(),
        args.output_root.resolve(),
        args.minimum_clearance_angstrom,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
