#!/usr/bin/env python3
"""Build GLYCAM inputs with LEaP-native links and rigid branch de-clashing.

This is an independent, fail-closed builder.  It never edits an existing run
root.  Every job is built in a staging directory and is published only after
topology, geometry, strict-grompp, atom-map, and chirality checks all pass.

The rigid search changes only glycosidic torsions: every atom in a child
residue is rotated about its central-O--child-C1 bond.  Internal bond geometry
and all ring stereocenters therefore remain unchanged by construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from audit_glycam_chirality import audit_chirality
from build_extended24_glycam_atom_index_maps import (
    build_map,
    find_mol,
    parse_gromacs_top_heavy_graph,
    sdf_heavy_graph,
)
from preflight_extended24_exact import (
    InputFiles,
    bond_geometry,
    closest_pair,
    parse_gro,
    parse_topology,
    run_grompp,
)
from repair_glycam_coordinates_from_mmff import (
    assign_mmff_conformer,
    build_full_mapping,
    parse_gromacs_topology,
    write_mapped_gro,
)
from start_extended24_glycam_topology_dryrun import (
    ACPYPE,
    GLYCAM_LEAPRC,
    GLYCAM_PREP,
    TLEAP,
    branch_linkage_positions,
    missing_local_glycam_tokens,
    required_glycam_tokens,
    resolve_glycam_sequence_with_reason,
)


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_SAMPLE_CSV = MD_ROOT / "md_all_remaining_dataset_v13" / "md_all_remaining_glycans_v13.csv"
FORBIDDEN_OUTPUT_ROOTS = {
    (MD_ROOT / "md_runs_all_remaining_v13").resolve(),
    (MD_ROOT / "md_runs_all_remaining_v14_safe").resolve(),
}
TLEAP_ERROR_RE = re.compile(r"Exiting LEaP: Errors\s*=\s*(\d+)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_logged(cmd: list[str], cwd: Path, log_path: Path) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "LC_ALL": "C"},
    )
    log_path.write_text("$ " + " ".join(cmd) + "\n\n" + proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}: {' '.join(cmd)}")
    return proc.stdout


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "priority",
        "molecule_name",
        "clean_name",
        "source_sdf",
        "mmff100_pickle",
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"sample CSV lacks columns: {sorted(missing)}")
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def write_tail_sequence_leap(
    path: Path,
    molecule: str,
    sequence: list[str],
    stem: str,
) -> tuple[str, tuple[int, int] | None]:
    branch_positions = branch_linkage_positions(molecule)
    lines = ["source leaprc.GLYCAM_06j-1", ""]
    if len(sequence) == 3 and sequence[1].startswith("1") and sequence[2].startswith("0"):
        first_residue = "1" + sequence[2][1:]
        second_residue = "0" + sequence[1][1:]
        lines.append(f"mol = sequence {{ {first_residue} {second_residue} }}")
        mode = "anomeric_anomeric_sequence"
    elif branch_positions is not None:
        if len(sequence) != 4 or len(branch_positions) != 2:
            raise ValueError(f"expected two-branch GLYCAM sequence, got {sequence}")
        aglycon, central, *children = sequence
        lines.append(f"mol = sequence {{ {aglycon} {central} }}")
        for position, child in zip(branch_positions, children):
            lines.extend(
                [
                    f"set mol tail mol.2.O{position}",
                    f"mol = sequence {{ mol {child} }}",
                ]
            )
        mode = "branched_tail_sequence"
    else:
        raise ValueError("this repair builder accepts only 1->1 or two-branch glycans")
    lines.extend(
        [
            "check mol",
            "charge mol",
            f"saveamberparm mol {stem}.prmtop {stem}.inpcrd",
            f"savepdb mol {stem}_leap.pdb",
            "quit",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return mode, branch_positions


def parse_pdb_atoms(path: Path) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atoms.append(
            {
                "residue_number": int(line[22:26]),
                "residue": line[17:20].strip(),
                "name": line[12:16].strip(),
            }
        )
    if not atoms:
        raise ValueError(f"no atoms in LEaP PDB: {path}")
    return atoms


def read_inpcrd(path: Path) -> tuple[list[str], np.ndarray, list[float]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"short inpcrd: {path}")
    atom_count = int(lines[1].split()[0])
    values = [float(value) for line in lines[2:] for value in line.split()]
    if len(values) < 3 * atom_count:
        raise ValueError(f"inpcrd has too few coordinates: {path}")
    coordinates = np.asarray(values[: 3 * atom_count], dtype=float).reshape(atom_count, 3)
    if not np.isfinite(coordinates).all():
        raise ValueError(f"non-finite inpcrd coordinate: {path}")
    return lines[:2], coordinates, values[3 * atom_count :]


def write_inpcrd(
    path: Path,
    header: list[str],
    coordinates: np.ndarray,
    trailing_values: list[float],
) -> None:
    values = [float(value) for value in coordinates.reshape(-1)] + trailing_values
    lines = [*header]
    for index in range(0, len(values), 6):
        lines.append("".join(f"{value:12.7f}" for value in values[index : index + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rotate_about_axis(
    coordinates: np.ndarray,
    indices: list[int],
    start: np.ndarray,
    end: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    result = coordinates.copy()
    axis = end - start
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-8:
        raise ValueError("zero-length glycosidic rotation axis")
    unit = axis / norm
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    vectors = coordinates[indices] - start
    rotated = (
        vectors * cosine
        + np.cross(np.broadcast_to(unit, vectors.shape), vectors) * sine
        + np.outer(vectors @ unit, unit) * (1.0 - cosine)
    )
    result[indices] = start + rotated
    return result


def circular_angle(angle: int) -> int:
    return ((angle + 180) % 360) - 180


def branch_clearance_angstrom(
    coordinates: np.ndarray,
    atoms: list[dict[str, Any]],
    branch_positions: tuple[int, int],
) -> float:
    moved_residues = {3, 4}
    glycosidic_pairs: set[frozenset[int]] = set()
    for child_residue, position in zip((3, 4), branch_positions):
        oxygen = next(
            index
            for index, atom in enumerate(atoms)
            if atom["residue_number"] == 2 and atom["name"] == f"O{position}"
        )
        carbon = next(
            index
            for index, atom in enumerate(atoms)
            if atom["residue_number"] == child_residue and atom["name"] == "C1"
        )
        glycosidic_pairs.add(frozenset((oxygen, carbon)))

    minimum = math.inf
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            left_residue = atoms[left]["residue_number"]
            right_residue = atoms[right]["residue_number"]
            if left_residue == right_residue:
                continue
            if left_residue not in moved_residues and right_residue not in moved_residues:
                continue
            if frozenset((left, right)) in glycosidic_pairs:
                continue
            minimum = min(minimum, float(np.linalg.norm(coordinates[left] - coordinates[right])))
    if not math.isfinite(minimum):
        raise ValueError("could not evaluate branch clearance")
    return minimum


def declash_branches(
    coordinates: np.ndarray,
    atoms: list[dict[str, Any]],
    branch_positions: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(atoms) != len(coordinates):
        raise ValueError("PDB/inpcrd atom-count mismatch")
    axes: list[tuple[np.ndarray, np.ndarray]] = []
    residue_indices: list[list[int]] = []
    for child_residue, position in zip((3, 4), branch_positions):
        oxygen = next(
            index
            for index, atom in enumerate(atoms)
            if atom["residue_number"] == 2 and atom["name"] == f"O{position}"
        )
        carbon = next(
            index
            for index, atom in enumerate(atoms)
            if atom["residue_number"] == child_residue and atom["name"] == "C1"
        )
        bond_length = float(np.linalg.norm(coordinates[oxygen] - coordinates[carbon]))
        if not 1.30 <= bond_length <= 1.65:
            raise ValueError(
                f"LEaP tail sequence gave invalid O{position}-C1 bond: {bond_length:.4f} A"
            )
        axes.append((coordinates[oxygen].copy(), coordinates[carbon].copy()))
        residue_indices.append(
            [index for index, atom in enumerate(atoms) if atom["residue_number"] == child_residue]
        )

    cache: dict[tuple[int, int], tuple[float, np.ndarray]] = {}

    def evaluate(angle3: int, angle4: int) -> tuple[float, np.ndarray]:
        key = (circular_angle(angle3), circular_angle(angle4))
        if key in cache:
            return cache[key]
        work = rotate_about_axis(coordinates, residue_indices[0], *axes[0], key[0])
        work = rotate_about_axis(work, residue_indices[1], *axes[1], key[1])
        value = branch_clearance_angstrom(work, atoms, branch_positions)
        cache[key] = (value, work)
        return value, work

    best: tuple[float, int, int, np.ndarray] | None = None

    def consider(angle3: int, angle4: int) -> None:
        nonlocal best
        a3, a4 = circular_angle(angle3), circular_angle(angle4)
        clearance, work = evaluate(a3, a4)
        candidate = (clearance, -(abs(a3) + abs(a4)), -abs(a3), -abs(a4))
        if best is None:
            best = (clearance, a3, a4, work)
            return
        incumbent = (best[0], -(abs(best[1]) + abs(best[2])), -abs(best[1]), -abs(best[2]))
        if candidate > incumbent:
            best = (clearance, a3, a4, work)

    for angle3 in range(-180, 180, 15):
        for angle4 in range(-180, 180, 15):
            consider(angle3, angle4)
    assert best is not None
    coarse_angles = (best[1], best[2])
    for angle3 in range(coarse_angles[0] - 15, coarse_angles[0] + 16):
        for angle4 in range(coarse_angles[1] - 15, coarse_angles[1] + 16):
            consider(angle3, angle4)
    assert best is not None
    return best[3], {
        "angles_degrees": [best[1], best[2]],
        "minimum_inter_residue_clearance_angstrom": best[0],
        "evaluated_angle_pairs": len(cache),
    }


def validate_geometry(gro: Path, top: Path) -> dict[str, Any]:
    atom_count, atoms, _box = parse_gro(gro)
    top_atom_count, bonds = parse_topology(top)
    if atom_count != top_atom_count:
        raise ValueError(f"GRO/TOP atom mismatch: {atom_count} != {top_atom_count}")
    minimum_distance, left, right = closest_pair(atoms)
    if minimum_distance < 0.05:
        raise ValueError(f"atomic overlap {left}--{right}: {minimum_distance:.6f} nm")
    bonded = bond_geometry(atoms, bonds)
    if bonded["failures"]:
        raise ValueError("invalid bond geometry: " + " | ".join(bonded["failures"]))
    return {
        "atom_count": atom_count,
        "minimum_distance_nm": minimum_distance,
        "minimum_distance_pair": [left, right],
        "bond_geometry": bonded,
    }


def run_acpype(
    *,
    work: Path,
    prmtop: Path,
    inpcrd: Path,
    basename: str,
    log_name: str,
) -> tuple[Path, Path]:
    run_logged(
        [str(ACPYPE), "-p", prmtop.name, "-x", inpcrd.name, "-b", basename],
        work,
        work / log_name,
    )
    output = work / f"{basename}.amb2gmx"
    gro = output / f"{basename}_GMX.gro"
    top = output / f"{basename}_GMX.top"
    if not gro.exists() or not top.exists():
        raise RuntimeError(f"ACPYPE outputs missing: {output}")
    return gro, top


def prepare_one(
    row: dict[str, str],
    output_root: Path,
    minimum_clearance_angstrom: float,
) -> dict[str, Any]:
    molecule = row["molecule_name"]
    clean = row["clean_name"]
    sequence, mapping_reason = resolve_glycam_sequence_with_reason(molecule)
    if sequence is None:
        raise ValueError(f"no exact GLYCAM sequence for {molecule}: {mapping_reason}")
    missing_tokens = missing_local_glycam_tokens(molecule, sequence)
    if missing_tokens:
        raise ValueError(
            f"active GLYCAM06j prep lacks required tokens for {molecule}: {missing_tokens}"
        )
    final_dir = output_root / clean / "glycam06j_topology_dryrun"
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing job: {final_dir}")
    staging_parent = output_root / ".staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=clean + "_", dir=staging_parent))
    stem = f"{clean}_GLYCAM06j"
    try:
        mode, branch_positions = write_tail_sequence_leap(
            staging / "build_glycam06j_tail_sequence.leap",
            molecule,
            sequence,
            stem,
        )
        leap_output = run_logged(
            [str(TLEAP), "-f", "build_glycam06j_tail_sequence.leap"],
            staging,
            staging / "tleap.log",
        )
        error_match = TLEAP_ERROR_RE.search(leap_output)
        if not error_match or int(error_match.group(1)) != 0:
            raise RuntimeError("LEaP did not report Errors = 0")
        prmtop = staging / f"{stem}.prmtop"
        inpcrd = staging / f"{stem}.inpcrd"
        pdb = staging / f"{stem}_leap.pdb"
        if not all(path.exists() for path in (prmtop, inpcrd, pdb)):
            raise RuntimeError("LEaP output set is incomplete")

        raw_inpcrd = staging / f"{stem}_tleap_raw.inpcrd"
        shutil.copy2(inpcrd, raw_inpcrd)
        raw_basename = f"{stem}_TLEAP_RAW"
        raw_gro, _raw_top = run_acpype(
            work=staging,
            prmtop=prmtop,
            inpcrd=raw_inpcrd,
            basename=raw_basename,
            log_name="acpype_raw.log",
        )

        header, coordinates, trailing = read_inpcrd(inpcrd)
        pdb_atoms = parse_pdb_atoms(pdb)
        if branch_positions is not None:
            coordinates, rigid = declash_branches(coordinates, pdb_atoms, branch_positions)
            if rigid["minimum_inter_residue_clearance_angstrom"] < minimum_clearance_angstrom:
                raise ValueError(
                    "best rigid branch clearance is too small: "
                    f"{rigid['minimum_inter_residue_clearance_angstrom']:.4f} < "
                    f"{minimum_clearance_angstrom:.4f} A"
                )
            write_inpcrd(inpcrd, header, coordinates, trailing)
        else:
            rigid = {
                "angles_degrees": [],
                "minimum_inter_residue_clearance_angstrom": None,
                "evaluated_angle_pairs": 0,
            }

        final_gro, final_top = run_acpype(
            work=staging,
            prmtop=prmtop,
            inpcrd=inpcrd,
            basename=stem,
            log_name="acpype.log",
        )
        geometry = validate_geometry(final_gro, final_top)

        source_sdf = Path(row["source_sdf"])
        mmff_pickle = Path(row["mmff100_pickle"])
        source_mol = find_mol(source_sdf, molecule)
        with mmff_pickle.open("rb") as handle:
            mmff_conformers = pickle.load(handle)
        if not isinstance(mmff_conformers, list) or not mmff_conformers:
            raise ValueError(f"no MMFF conformers in {mmff_pickle}")
        mmff_mol = assign_mmff_conformer(
            source_mol,
            np.asarray(mmff_conformers[0], dtype=float),
        )

        compatible_raw_gro = final_gro.with_name(final_gro.stem + "_tleap_raw.gro")
        shutil.copy2(raw_gro, compatible_raw_gro)
        full_topology_graph, _topology_atoms = parse_gromacs_topology(final_top)
        full_mapping = build_full_mapping(mmff_mol, full_topology_graph)
        mmff_probe_gro = staging / f"{stem}_MMFF_CHIRAL_PROBE.gro"
        write_mapped_gro(compatible_raw_gro, mmff_probe_gro, mmff_mol, full_mapping)

        chirality_specs = {
            "raw_to_final": (compatible_raw_gro, final_gro),
            "source_mmff_to_raw": (mmff_probe_gro, compatible_raw_gro),
            "source_mmff_to_final": (mmff_probe_gro, final_gro),
        }
        chirality_audits: dict[str, dict[str, Any]] = {}
        chirality_files: dict[str, str] = {}
        for audit_name, (reference, candidate) in chirality_specs.items():
            audit = audit_chirality(
                topology=final_top,
                reference_gro=reference,
                candidate_gro=candidate,
            )
            if not audit["strict_passed"]:
                raise ValueError(
                    f"{audit_name} chirality gate failed: "
                    f"mismatch={audit['counts']['mismatched']} "
                    f"unevaluable={audit['counts']['unevaluable']}"
                )
            audit["topology"] = str(final_top.relative_to(staging))
            audit["reference_gro"] = str(reference.relative_to(staging))
            audit["candidate_gro"] = str(candidate.relative_to(staging))
            audit_path = staging / f"glycam_{audit_name}_chirality_audit.json"
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            chirality_audits[audit_name] = audit
            chirality_files[audit_name] = audit_path.name

        atom_map = build_map(
            sdf_heavy_graph(source_mol),
            parse_gromacs_top_heavy_graph(final_top),
        )
        atom_map_path = staging / "sdf_to_glycam_frame_heavy_index_map.json"
        atom_map_path.write_text(json.dumps(atom_map, indent=2) + "\n", encoding="utf-8")

        grompp = run_grompp(
            "glycam",
            InputFiles(
                gro=final_gro,
                top=final_top,
                atom_map=atom_map_path,
                metadata=None,
                support=(),
            ),
        )
        if grompp["returncode"] != 0 or grompp["warning_count"] != 0:
            raise ValueError(
                f"strict grompp failed: rc={grompp['returncode']} warnings={grompp['warning_count']}"
            )
        charge = grompp["charge"]
        if charge is None or abs(float(charge) - round(float(charge))) > 0.001:
            raise ValueError(f"non-integral total charge: {charge}")

        metadata = {
            "builder": "build_glycam_rigid_declashed.py",
            "priority": int(row["priority"]),
            "molecule_name": molecule,
            "clean_name": clean,
            "glycam_sequence": sequence,
            "glycam_library_evidence": {
                "leaprc": str(GLYCAM_LEAPRC),
                "leaprc_sha256": sha256(GLYCAM_LEAPRC),
                "prep": str(GLYCAM_PREP),
                "prep_sha256": sha256(GLYCAM_PREP),
                "validated_residue_tokens": required_glycam_tokens(molecule, sequence),
            },
            "build_mode": mode,
            "branch_positions": list(branch_positions) if branch_positions else [],
            "rigid_declash": rigid,
            "minimum_required_clearance_angstrom": minimum_clearance_angstrom,
            "geometry": geometry,
            "chirality_audits": {
                name: {
                    "strict_passed": audit["strict_passed"],
                    "counts": audit["counts"],
                    "audit_file": chirality_files[name],
                }
                for name, audit in chirality_audits.items()
            },
            "grompp": grompp,
            "source_sdf": str(source_sdf),
            "source_mmff_pickle": str(mmff_pickle),
            "sha256": {
                "prmtop": sha256(prmtop),
                "raw_inpcrd": sha256(raw_inpcrd),
                "declashed_inpcrd": sha256(inpcrd),
                "gro": sha256(final_gro),
                "top": sha256(final_top),
                "atom_map": sha256(atom_map_path),
            },
        }
        (staging / "glycam_rigid_declash_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        coordinate_metadata = {
            "molecule_name": molecule,
            "method": (
                "corrected GLYCAM D/L residue codes with LEaP tail-sequence "
                "and rigid glycosidic branch de-clashing"
            ),
            "raw_tleap_gro": str(compatible_raw_gro.relative_to(staging)),
            "repaired_gro": str(final_gro.relative_to(staging)),
            "gromacs_top": str(final_top.relative_to(staging)),
            "source_sdf": str(source_sdf),
            "source_mmff_pickle": str(mmff_pickle),
            "chirality_audits": {
                name: {
                    "strict_passed": audit["strict_passed"],
                    "counts": audit["counts"],
                    "audit_file": chirality_files[name],
                }
                for name, audit in chirality_audits.items()
            },
            "input_sha256": {
                "top": sha256(final_top),
                "raw_gro": sha256(compatible_raw_gro),
                "sdf": sha256(source_sdf),
                "mmff_pickle": sha256(mmff_pickle),
            },
            "output_sha256": {
                "gro": sha256(final_gro),
                "atom_map": sha256(atom_map_path),
            },
        }
        (staging / "glycam_coordinate_repair_metadata.json").write_text(
            json.dumps(coordinate_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "glycam06j_topology_metadata.json").write_text(
            json.dumps(
                {
                    "molecule_name": molecule,
                    "clean_name": clean,
                    "priority": int(row["priority"]),
                    "glycam_sequence": sequence,
                    "builder": "build_glycam_rigid_declashed.py",
                    "glycam_library_evidence": {
                        "leaprc": str(GLYCAM_LEAPRC),
                        "leaprc_sha256": sha256(GLYCAM_LEAPRC),
                        "prep": str(GLYCAM_PREP),
                        "prep_sha256": sha256(GLYCAM_PREP),
                        "validated_residue_tokens": required_glycam_tokens(
                            molecule, sequence
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        final_dir.parent.mkdir(parents=True, exist_ok=False)
        staging.rename(final_dir)
        return {
            "priority": int(row["priority"]),
            "molecule_name": molecule,
            "clean_name": clean,
            "status": "prepared",
            "output_dir": str(final_dir),
            "rigid_declash": rigid,
            "minimum_distance_nm": geometry["minimum_distance_nm"],
            "chirality_centers": chirality_audits["source_mmff_to_final"]["counts"][
                "evaluated_matching"
            ],
            "charge": charge,
        }
    except Exception:
        failed = output_root / ".failed" / staging.name
        failed.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            staging.rename(failed)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed GLYCAM 1->1/branched builder with rigid de-clashing."
    )
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE_CSV)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--only",
        action="append",
        required=True,
        help="Clean name or molecule name; repeat for every intended glycan.",
    )
    parser.add_argument(
        "--minimum-clearance-angstrom",
        type=float,
        default=1.20,
        help="Fail if the best inter-residue rigid placement is below this clearance.",
    )
    args = parser.parse_args()
    if args.minimum_clearance_angstrom < 0.5:
        raise SystemExit("--minimum-clearance-angstrom must be at least 0.5")
    if not TLEAP.exists() or not ACPYPE.exists():
        raise SystemExit(f"missing dependency: tleap={TLEAP.exists()} acpype={ACPYPE.exists()}")

    output_root = args.output_root.resolve()
    if output_root in FORBIDDEN_OUTPUT_ROOTS:
        raise SystemExit(f"refusing protected output root: {output_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output root must be absent or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.only)
    rows = [
        row
        for row in read_rows(args.sample_csv)
        if row["clean_name"] in selected or row["molecule_name"] in selected
    ]
    matched = {row["clean_name"] for row in rows} | {row["molecule_name"] for row in rows}
    unmatched = selected - matched
    if unmatched:
        raise SystemExit(f"no sample matched: {sorted(unmatched)}")

    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            result = prepare_one(row, output_root, args.minimum_clearance_angstrom)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as exc:
        report = {
            "status": "failed",
            "error": repr(exc),
            "completed": results,
        }
        (output_root / "glycam_rigid_build_status.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise

    report = {"status": "prepared", "count": len(results), "results": results}
    (output_root / "glycam_rigid_build_status.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
