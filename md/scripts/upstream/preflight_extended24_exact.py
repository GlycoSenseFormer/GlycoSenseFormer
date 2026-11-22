#!/usr/bin/env python3
"""Read-only safety gate for the extended exact-topology MD queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from audit_glycam_chirality import audit_chirality
from charmm_standard_metadata import assess_charmm_standard_metadata
from gromos_curated_gate import load_approved_entry


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
GMX = os.environ.get("GMX", "gmx")
MIN_DISTANCE_NM = float(os.environ.get("MD_PREFLIGHT_MIN_DISTANCE_NM", "0.05"))
CHARGE_TOLERANCE = float(os.environ.get("MD_PREFLIGHT_CHARGE_TOLERANCE", "0.001"))

GROMOS_WARNING = "The GROMOS force fields have been parametrized with a physically"
UNSAFE_NOTE_MARKERS = (
    "not a curated residue template",
    "before production-quality residue/patch mapping",
)
CHARGE_RE = re.compile(r"System has non-zero total charge:\s*([-+0-9.eE]+)")
WARNING_RE = re.compile(r"^WARNING\s+\d+\s+\[", re.MULTILINE)


PREFLIGHT_MDP = """\
integrator              = steep
nsteps                  = 0
emtol                   = 100.0
emstep                  = 0.01
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
coulombtype             = PME
rcoulomb                = 1.2
vdwtype                 = Cut-off
rvdw                    = 1.2
pbc                     = xyz
constraints             = h-bonds
tcoupl                  = no
pcoupl                  = no
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 1
nstlog                  = 1
"""


@dataclass(frozen=True)
class InputFiles:
    gro: Path
    top: Path
    atom_map: Path
    metadata: Path | None
    support: tuple[Path, ...]


def molecule_stem(row: dict[str, str]) -> str:
    return row["molecule_name"].replace("/", "_")


def locate_inputs(
    forcefield: str,
    row: dict[str, str],
    run_root: Path = RUN_ROOT,
    *,
    approved_gromos_manifest: Path | None = None,
) -> InputFiles:
    run_root = run_root.resolve()
    clean = row["clean_name"]
    stem = molecule_stem(row)
    if forcefield == "gromos":
        if approved_gromos_manifest is not None:
            approved = load_approved_entry(
                clean,
                manifest_path=approved_gromos_manifest,
                expected_molecule_name=row["molecule_name"],
                expected_source_class=row.get("source_class"),
            )
            resolved = approved["resolved_inputs"]
            if not isinstance(resolved, dict):
                raise RuntimeError(f"{clean}: approved GROMOS inputs are invalid")
            return InputFiles(
                gro=Path(resolved["processed_gro"]),
                top=Path(resolved["topology"]),
                atom_map=Path(resolved["atom_map"]),
                metadata=Path(resolved["metadata"]),
                support=(Path(resolved["forcefield"]),),
            )
        base = run_root / clean / "gromos_exact_topology_dryrun"
        return InputFiles(
            gro=base / f"{stem}_GROMOS_EXACT_processed.gro",
            top=base / f"{stem}_GROMOS_EXACT.top",
            atom_map=base / "sdf_to_gromos_frame_heavy_index_map.json",
            metadata=base / "gromos_exact_input_metadata.json",
            support=(base / "gromos53a6carbo_exact.ff",),
        )
    if forcefield == "charmm":
        base = run_root / clean / "charmm36_standard_topology_dryrun"
        return InputFiles(
            gro=base / f"{stem}_CHARMM36_STANDARD_processed.gro",
            top=base / f"{stem}_CHARMM36_STANDARD.top",
            atom_map=base / "sdf_to_charmm_frame_heavy_index_map.json",
            metadata=base / "charmm36_standard_input_metadata.json",
            support=(base / "charmm36_standard.ff",),
        )
    if forcefield == "glycam":
        base = run_root / clean / "glycam06j_topology_dryrun"
        acpype = base / f"{clean}_GLYCAM06j.amb2gmx"
        gro = acpype / f"{clean}_GLYCAM06j_GMX.gro"
        return InputFiles(
            gro=gro,
            top=acpype / f"{clean}_GLYCAM06j_GMX.top",
            atom_map=base / "sdf_to_glycam_frame_heavy_index_map.json",
            metadata=base / "glycam06j_topology_metadata.json",
            support=(
                base / "glycam_coordinate_repair_metadata.json",
                gro.with_name(gro.stem + "_tleap_raw.gro"),
            ),
        )
    raise ValueError(f"Unsupported force field: {forcefield}")


def parse_gro(path: Path) -> tuple[int, list[tuple[str, float, float, float]], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("GRO file is too short")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError("invalid GRO atom-count line") from exc
    if len(lines) != atom_count + 3:
        raise ValueError(f"GRO declares {atom_count} atoms but has {len(lines) - 3} atom lines")
    atoms: list[tuple[str, float, float, float]] = []
    for index, line in enumerate(lines[2 : 2 + atom_count], start=1):
        if len(line) < 44:
            raise ValueError(f"atom line {index} is shorter than 44 columns")
        label = f"{index}:{line[5:10].strip()}/{line[10:15].strip()}"
        try:
            xyz = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        except ValueError as exc:
            raise ValueError(f"invalid coordinates on atom line {index}") from exc
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"non-finite coordinates on atom line {index}")
        atoms.append((label, *xyz))
    try:
        box = [float(value) for value in lines[-1].split()]
    except ValueError as exc:
        raise ValueError("invalid GRO box line") from exc
    if len(box) not in (3, 9) or not all(math.isfinite(value) for value in box):
        raise ValueError("GRO box must contain 3 or 9 finite values")
    if any(value <= 0 for value in box[:3]):
        raise ValueError("GRO box diagonal lengths must be positive")
    return atom_count, atoms, box


def closest_pair(atoms: list[tuple[str, float, float, float]]) -> tuple[float, str, str]:
    best = (math.inf, "", "")
    for i, left in enumerate(atoms):
        for right in atoms[i + 1 :]:
            distance = math.sqrt(sum((left[k] - right[k]) ** 2 for k in (1, 2, 3)))
            if distance < best[0]:
                best = (distance, left[0], right[0])
    return best


def parse_topology(path: Path) -> tuple[int, list[tuple[int, int, float | None]]]:
    section = ""
    atom_count = 0
    bonds: list[tuple[int, int, float | None]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and "]" in line:
            section = line[1 : line.index("]")].strip().lower()
            continue
        fields = line.split()
        if section == "atoms":
            try:
                int(fields[0])
            except (IndexError, ValueError):
                continue
            atom_count += 1
        elif section == "bonds" and len(fields) >= 3:
            try:
                left, right, _function = (int(fields[0]), int(fields[1]), int(fields[2]))
            except ValueError:
                continue
            reference = None
            if len(fields) >= 4:
                try:
                    reference = float(fields[3])
                except ValueError:
                    pass
            bonds.append((left, right, reference))
    return atom_count, bonds


def bond_geometry(
    atoms: list[tuple[str, float, float, float]],
    bonds: list[tuple[int, int, float | None]],
) -> dict[str, object]:
    distances: list[float] = []
    failures: list[str] = []
    warnings: list[str] = []
    for left_index, right_index, reference in bonds:
        if not (1 <= left_index <= len(atoms) and 1 <= right_index <= len(atoms)):
            failures.append(f"bond {left_index}-{right_index} indexes outside GRO atom range")
            continue
        left = atoms[left_index - 1]
        right = atoms[right_index - 1]
        distance = math.sqrt(sum((left[k] - right[k]) ** 2 for k in (1, 2, 3)))
        distances.append(distance)
        label = f"{left[0]}--{right[0]}"
        if distance < 0.05 or distance > 0.30:
            failures.append(f"{label} has implausible bond length {distance:.4f} nm")
        elif distance < 0.07 or distance > 0.20:
            warnings.append(f"{label} has unusual bond length {distance:.4f} nm")
        if reference is not None and reference > 0:
            delta = abs(distance - reference)
            if delta > max(0.05, 0.35 * reference):
                failures.append(
                    f"{label} length {distance:.4f} nm differs from r0={reference:.4f} nm"
                )
            elif delta > max(0.03, 0.20 * reference):
                warnings.append(
                    f"{label} length {distance:.4f} nm differs from r0={reference:.4f} nm"
                )
    return {
        "count": len(bonds),
        "minimum_nm": round(min(distances), 6) if distances else None,
        "maximum_nm": round(max(distances), 6) if distances else None,
        "failures": failures,
        "warnings": warnings,
    }


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_directory_tree(path: Path) -> dict[str, object]:
    """Match the fail-closed directory-tree hash used by CHARMM builders."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ValueError(f"force-field directory contains a symlink: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise ValueError(f"force-field directory is empty: {path}")
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def expected_charmm_generator_provenance(policy: object) -> dict[str, object] | None:
    scripts = Path(__file__).resolve().parent
    if policy == "charmm36_standard_patch_expansion_v1":
        entrypoint = scripts / "prepare_charmm_standard_patch_input.py"
        helpers = [scripts / "prepare_charmm_standard_uronic_input.py"]
    elif policy == "charmm36_standard_alignment364_patch_expansion_v1":
        entrypoint = scripts / "prepare_charmm_alignment364_standard_input.py"
        helpers = [
            scripts / "prepare_charmm_standard_patch_input.py",
            scripts / "prepare_charmm_standard_uronic_input.py",
        ]
    elif policy == "charmm36_standard_uronic_residue_v1":
        entrypoint = scripts / "prepare_charmm_standard_uronic_input.py"
        helpers = []
    else:
        return None
    return {
        "entrypoint": entrypoint.name,
        "entrypoint_sha256": sha256_path(entrypoint),
        "imported_helpers": {path.name: sha256_path(path) for path in helpers},
    }


def verify_charmm_artifact_provenance(
    inputs: InputFiles,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Bind downstream preflight to the exact builder-approved artifacts."""

    errors: list[str] = []
    recorded_files = metadata.get("files")
    file_checks: dict[str, object] = {}
    if not isinstance(recorded_files, dict):
        errors.append("CHARMM metadata lacks the generated-file SHA256 table")
    else:
        for label, path in (
            ("processed_gro", inputs.gro),
            ("topology", inputs.top),
            ("atom_map", inputs.atom_map),
        ):
            expected = recorded_files.get(path.name)
            try:
                actual = sha256_path(path)
            except OSError as exc:
                actual = None
                errors.append(f"could not hash CHARMM {label}: {exc}")
            matches = isinstance(expected, str) and actual == expected
            file_checks[label] = {
                "path": str(path),
                "recorded_sha256": expected,
                "actual_sha256": actual,
                "matches": matches,
            }
            if not matches:
                errors.append(f"CHARMM generated-file provenance hash mismatch: {label}")

    forcefield_metadata = metadata.get("forcefield")
    expected_tree: object = None
    if isinstance(forcefield_metadata, dict):
        expected_tree = forcefield_metadata.get("modified_local_tree")
        if expected_tree is None:
            copied = forcefield_metadata.get("self_contained_copy")
            if isinstance(copied, dict):
                expected_tree = copied.get("post_preflight_tree")
    local_forcefields = [path for path in inputs.support if path.name.endswith(".ff")]
    actual_tree: object = None
    if len(local_forcefields) != 1:
        errors.append(
            f"expected exactly one local CHARMM force-field directory; found {len(local_forcefields)}"
        )
    else:
        try:
            actual_tree = sha256_directory_tree(local_forcefields[0])
        except (OSError, ValueError) as exc:
            errors.append(f"could not hash local CHARMM force-field tree: {exc}")
    tree_matches = isinstance(expected_tree, dict) and actual_tree == expected_tree
    if not tree_matches:
        errors.append("CHARMM local force-field tree differs from builder metadata")
    expected_generator = expected_charmm_generator_provenance(
        metadata.get("topology_policy")
    )
    recorded_generator = metadata.get("generator")
    generator_matches = (
        expected_generator is not None and recorded_generator == expected_generator
    )
    if not generator_matches:
        errors.append("CHARMM generator provenance differs from the current approved implementation")
    return {
        "approved": not errors,
        "errors": errors,
        "generated_files": file_checks,
        "forcefield_tree": {
            "path": str(local_forcefields[0]) if len(local_forcefields) == 1 else None,
            "recorded": expected_tree,
            "actual": actual_tree,
            "matches": tree_matches,
        },
        "generator": {
            "recorded": recorded_generator,
            "expected_current": expected_generator,
            "matches": generator_matches,
        },
    }


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def run_grompp(
    forcefield: str,
    inputs: InputFiles,
    *,
    allow_net_charge_warning: bool = False,
    gromos_warning_policy: str = "family_default",
) -> dict[str, object]:
    if gromos_warning_policy not in {"family_default", "zero"}:
        raise ValueError(f"invalid GROMOS warning policy: {gromos_warning_policy!r}")
    with tempfile.TemporaryDirectory(prefix="extended24_preflight_") as temp_name:
        temp = Path(temp_name)
        mdp = temp / "preflight.mdp"
        mdp.write_text(PREFLIGHT_MDP, encoding="utf-8")
        cmd = [
            GMX,
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(inputs.gro),
            "-p",
            str(inputs.top),
            "-o",
            str(temp / "preflight.tpr"),
            "-po",
            str(temp / "mdout.mdp"),
            "-pp",
            str(temp / "processed.top"),
            "-maxwarn",
            (
                "1"
                if (
                    forcefield == "gromos"
                    and gromos_warning_policy == "family_default"
                )
                or allow_net_charge_warning
                else "0"
            ),
        ]
        env = os.environ.copy()
        env["LC_ALL"] = "C"
        proc = subprocess.run(
            cmd,
            cwd=inputs.top.parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        output = proc.stdout
        charge_match = CHARGE_RE.search(output)
        charge = float(charge_match.group(1)) if charge_match else (0.0 if proc.returncode == 0 else None)
        warning_count = len(WARNING_RE.findall(output))
        return {
            "returncode": proc.returncode,
            "charge": charge,
            "warning_count": warning_count,
            "expected_gromos_warning": GROMOS_WARNING in output,
            "expected_net_charge_warning": (
                "You are using Ewald electrostatics in a system with net charge" in output
                and "uniform background charge" in output
            ),
            "output_tail": output[-5000:],
        }


def preflight_job(
    forcefield: str,
    row: dict[str, str],
    run_root: Path = RUN_ROOT,
    *,
    allow_experimental_gromos: bool = False,
    approved_gromos_manifest: Path | None = None,
) -> dict[str, object]:
    inputs = locate_inputs(
        forcefield,
        row,
        run_root,
        approved_gromos_manifest=approved_gromos_manifest,
    )
    errors: list[str] = []
    warnings: list[str] = []
    required = (inputs.gro, inputs.top, inputs.atom_map, *inputs.support)
    if inputs.metadata is not None:
        required = (*required, inputs.metadata)
    missing = [str(path) for path in required if not path.exists()]
    result: dict[str, object] = {
        "forcefield": forcefield,
        "priority": int(row["priority"]),
        "clean_name": row["clean_name"],
        "molecule_name": row["molecule_name"],
        "gro": str(inputs.gro),
        "top": str(inputs.top),
        "errors": errors,
        "warnings": warnings,
    }
    if forcefield == "gromos" and approved_gromos_manifest is not None:
        result["approved_gromos_manifest"] = str(approved_gromos_manifest.resolve())
        result["gromos_warning_policy"] = "zero"
    if missing:
        errors.append("missing required inputs: " + ", ".join(missing))
        result["passed"] = False
        return result

    try:
        atom_count, atoms, box = parse_gro(inputs.gro)
        distance, left, right = closest_pair(atoms)
        result.update(
            {
                "atom_count": atom_count,
                "box": box,
                "minimum_distance_nm": round(distance, 6),
                "minimum_distance_pair": [left, right],
            }
        )
        if distance < MIN_DISTANCE_NM:
            errors.append(
                f"atomic overlap/near-overlap: {left}--{right} is {distance:.6f} nm "
                f"(< {MIN_DISTANCE_NM:.3f} nm)"
            )
        top_atom_count, bonds = parse_topology(inputs.top)
        result["topology_atom_count"] = top_atom_count
        if top_atom_count != atom_count:
            errors.append(
                f"atom-count mismatch: GRO={atom_count}, topology [ atoms ]={top_atom_count}"
            )
        geometry = bond_geometry(atoms, bonds)
        result["bond_geometry"] = geometry
        geometry_failures = [str(value) for value in geometry["failures"]]
        geometry_warnings = [str(value) for value in geometry["warnings"]]
        if geometry_failures:
            errors.append(
                f"{len(geometry_failures)} invalid bond geometries; examples: "
                + "; ".join(geometry_failures[:3])
            )
        if geometry_warnings:
            warnings.append(
                f"{len(geometry_warnings)} unusual bond geometries; examples: "
                + "; ".join(geometry_warnings[:3])
            )
    except (OSError, ValueError) as exc:
        errors.append(f"invalid GRO: {exc}")

    try:
        load_json(inputs.atom_map)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid atom-index map JSON: {exc}")

    metadata: dict[str, object] = {}
    if inputs.metadata is not None:
        try:
            raw_metadata = load_json(inputs.metadata)
            if not isinstance(raw_metadata, dict):
                raise ValueError("metadata root is not an object")
            metadata = raw_metadata
            result["metadata_total_charge"] = metadata.get(
                "observed_topology_charge_e", metadata.get("total_charge")
            )
            note = str(metadata.get("note", ""))
            result["metadata_note"] = note
            if any(marker in note.lower() for marker in UNSAFE_NOTE_MARKERS):
                message = f"topology metadata is explicitly non-production: {note}"
                if forcefield == "gromos" and allow_experimental_gromos:
                    warnings.append(
                        "diagnostic-only experimental GROMOS topology explicitly allowed; "
                        "this does not approve a production 5 ns queue"
                    )
                    result["experimental_topology_warning"] = message
                else:
                    errors.append(message)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid topology metadata JSON: {exc}")

    charmm_policy: dict[str, object] = {
        "approved": False,
        "errors": [],
        "charge_e": None,
        "requires_neutralization": None,
    }
    if forcefield == "charmm":
        charmm_policy = assess_charmm_standard_metadata(
            metadata,
            charge_tolerance=CHARGE_TOLERANCE,
            expected_molecule_name=row["molecule_name"],
            expected_clean_name=row["clean_name"],
            expected_source_class=row.get("source_class"),
        )
        result["charmm_metadata_policy"] = charmm_policy
        if not bool(charmm_policy["approved"]):
            errors.extend(
                f"invalid CHARMM production metadata: {message}"
                for message in charmm_policy["errors"]
            )
        artifact_provenance = verify_charmm_artifact_provenance(inputs, metadata)
        result["charmm_artifact_provenance"] = artifact_provenance
        errors.extend(
            f"invalid CHARMM artifact provenance: {message}"
            for message in artifact_provenance["errors"]
        )
    charmm_production_ready = forcefield == "charmm" and bool(
        charmm_policy["approved"]
    )
    allows_counterion_neutralization = (
        charmm_production_ready
        and charmm_policy.get("requires_neutralization") is True
    )
    result["requires_neutralization"] = bool(allows_counterion_neutralization)
    result["charmm_production_ready"] = bool(charmm_production_ready)

    if forcefield == "glycam":
        coordinate_metadata_path, raw_tleap_gro = inputs.support
        try:
            raw_coordinate_metadata = load_json(coordinate_metadata_path)
            if not isinstance(raw_coordinate_metadata, dict):
                raise ValueError("coordinate-repair metadata root is not an object")
            recorded_inputs = raw_coordinate_metadata.get("input_sha256", {})
            recorded_outputs = raw_coordinate_metadata.get("output_sha256", {})
            if not isinstance(recorded_inputs, dict) or not isinstance(recorded_outputs, dict):
                raise ValueError("coordinate-repair hashes are missing")
            provenance_checks = {
                "top": (inputs.top, recorded_inputs.get("top")),
                "raw_gro": (raw_tleap_gro, recorded_inputs.get("raw_gro")),
                "gro": (inputs.gro, recorded_outputs.get("gro")),
                "atom_map": (inputs.atom_map, recorded_outputs.get("atom_map")),
            }
            provenance_result: dict[str, object] = {}
            for label, (path, recorded_hash) in provenance_checks.items():
                actual_hash = sha256_path(path)
                matches = isinstance(recorded_hash, str) and recorded_hash == actual_hash
                provenance_result[label] = {
                    "path": str(path),
                    "recorded_sha256": recorded_hash,
                    "actual_sha256": actual_hash,
                    "matches": matches,
                }
                if not matches:
                    errors.append(f"GLYCAM coordinate provenance hash mismatch: {label}")
            result["glycam_coordinate_provenance"] = provenance_result

            chirality = audit_chirality(
                topology=inputs.top,
                reference_gro=raw_tleap_gro,
                candidate_gro=inputs.gro,
            )
            result["glycam_chirality"] = chirality
            if not bool(chirality["strict_passed"]):
                counts = chirality["counts"]
                errors.append(
                    "GLYCAM chirality gate failed: "
                    f"mismatched={counts['mismatched']}, unevaluable={counts['unevaluable']}"
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            errors.append(f"invalid GLYCAM coordinate provenance/chirality inputs: {exc}")

    hashed_paths = (inputs.gro, inputs.top, inputs.atom_map, *inputs.support)
    if inputs.metadata is not None:
        hashed_paths = (*hashed_paths, inputs.metadata)
    try:
        result["input_sha256"] = {str(path): sha256_path(path) for path in hashed_paths}
    except OSError as exc:
        errors.append(f"could not hash inputs: {exc}")

    grompp = run_grompp(
        forcefield,
        inputs,
        allow_net_charge_warning=allows_counterion_neutralization,
        gromos_warning_policy=(
            "zero"
            if forcefield == "gromos" and approved_gromos_manifest is not None
            else "family_default"
        ),
    )
    result["grompp"] = grompp
    if int(grompp["returncode"]) != 0:
        errors.append(f"strict grompp failed with return code {grompp['returncode']}")
    warning_count = int(grompp["warning_count"])
    if forcefield == "gromos":
        if approved_gromos_manifest is not None and warning_count:
            errors.append(
                "P3-approved GROMOS topology emitted warnings; zero are allowed"
            )
        elif approved_gromos_manifest is None and (
            warning_count != 1 or not bool(grompp["expected_gromos_warning"])
        ):
            errors.append(
                "grompp warnings differ from the single explicitly allowed GROMOS twin-range warning"
            )
        elif approved_gromos_manifest is None and int(grompp["returncode"]) == 0:
            warnings.append("accepted the single standard GROMOS twin-range force-field warning")
    elif allows_counterion_neutralization:
        if warning_count != 1 or not bool(grompp["expected_net_charge_warning"]):
            errors.append(
                "grompp warnings differ from the single explicitly allowed charged-solute "
                "PME warning"
            )
        elif int(grompp["returncode"]) == 0:
            warnings.append(
                "accepted the single charged-solute PME warning; the water workflow must "
                "add a counterion before EM"
            )
    elif warning_count:
        errors.append(f"grompp emitted {warning_count} warning(s); zero are allowed")

    charge = grompp.get("charge")
    metadata_charge = metadata.get(
        "observed_topology_charge_e", metadata.get("total_charge")
    )
    if charge is None and isinstance(metadata_charge, (int, float)):
        charge = float(metadata_charge)
    result["total_charge"] = charge
    if charge is None:
        errors.append("could not determine system charge")
    else:
        charge_value = float(charge)
        if abs(charge_value - round(charge_value)) > CHARGE_TOLERANCE:
            errors.append(
                f"non-integer system charge {charge_value:+.6f} e "
                f"(tolerance {CHARGE_TOLERANCE:.3f} e)"
            )
        if abs(charge_value) > CHARGE_TOLERANCE and not allows_counterion_neutralization:
            errors.append(
                f"non-neutral system charge {charge_value:+.6f} e; this workflow does not add counterions"
            )
        if isinstance(metadata_charge, (int, float)) and abs(float(metadata_charge) - charge_value) > CHARGE_TOLERANCE:
            errors.append(
                f"metadata charge {float(metadata_charge):+.6f} e disagrees with grompp charge "
                f"{charge_value:+.6f} e"
            )
        expected_charge = metadata.get("expected_solute_charge_e")
        if charmm_production_ready:
            if not isinstance(expected_charge, (int, float)):
                errors.append("production-ready CHARMM metadata lacks expected_solute_charge_e")
            elif abs(float(expected_charge) - charge_value) > CHARGE_TOLERANCE:
                errors.append(
                    f"expected solute charge {float(expected_charge):+.6f} e disagrees with "
                    f"grompp charge {charge_value:+.6f} e"
                )
        if allows_counterion_neutralization:
            if abs(charge_value) <= CHARGE_TOLERANCE:
                errors.append(
                    "metadata requires neutralization but grompp reports a neutral solute"
                )

    result["passed"] = not errors
    return result


def write_reports(results: list[dict[str, object]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "preflight.json"
    csv_path = output_dir / "preflight.csv"
    md_path = output_dir / "preflight.md"
    summary = {
        "generated_at": time.strftime("%F %T %Z"),
        "gmx": GMX,
        "run_root": str(RUN_ROOT),
        "thresholds": {
            "minimum_distance_nm": MIN_DISTANCE_NM,
            "charge_tolerance_e": CHARGE_TOLERANCE,
            "charge_policy": (
                "neutral solute, except explicitly production-ready CHARMM solutes that are "
                "neutralized with a counterion before strict PME EM"
            ),
        },
        "counts": {
            "total": len(results),
            "passed": sum(bool(item["passed"]) for item in results),
            "blocked": sum(not bool(item["passed"]) for item in results),
        },
        "results": results,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "forcefield",
            "priority",
            "clean_name",
            "molecule_name",
            "passed",
            "total_charge",
            "minimum_distance_nm",
            "errors",
            "warnings",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    **{field: item.get(field, "") for field in fields},
                    "errors": " | ".join(str(value) for value in item.get("errors", [])),
                    "warnings": " | ".join(str(value) for value in item.get("warnings", [])),
                }
            )
    counts = summary["counts"]
    lines = [
        "# Extended-24 exact MD preflight",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Total: {counts['total']}; passed: {counts['passed']}; blocked: {counts['blocked']}.",
        "",
        "| Force field | Priority | Molecule | Result | Charge (e) | Min distance (nm) | Main issue |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for item in results:
        issues = item.get("errors", [])
        main_issue = str(issues[0]) if issues else "-"
        main_issue = main_issue.replace("|", "/")
        charge = item.get("total_charge")
        distance = item.get("minimum_distance_nm")
        lines.append(
            f"| {item['forcefield']} | {item['priority']} | {item['clean_name']} | "
            f"{'PASS' if item['passed'] else 'BLOCK'} | "
            f"{charge if charge is not None else '-'} | {distance if distance is not None else '-'} | {main_issue} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def preflight_jobs(
    jobs: list[tuple[str, dict[str, str]]],
    output_dir: Path,
    run_root: Path = RUN_ROOT,
    *,
    allow_experimental_gromos: bool = False,
) -> tuple[list[dict[str, object]], dict[str, Path]]:
    results: list[dict[str, object]] = []
    for index, (forcefield, row) in enumerate(jobs, start=1):
        print(f"[preflight {index}/{len(jobs)}] {forcefield} {row['clean_name']}", flush=True)
        results.append(
            preflight_job(
                forcefield,
                row,
                run_root,
                allow_experimental_gromos=allow_experimental_gromos,
            )
        )
    return results, write_reports(results, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly preflight remaining extended exact MD inputs.")
    parser.add_argument("--forcefields", default="glycam,charmm,gromos")
    parser.add_argument("--only", default="", help="Optional clean_name or molecule_name filter.")
    parser.add_argument(
        "--allow-experimental-gromos",
        action="store_true",
        help=(
            "Downgrade the explicit graph-derived GROMOS metadata block to a warning "
            "for diagnostic preflight only; this does not approve production MD."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(RUN_ROOT / f"preflight_exact_{time.strftime('%Y%m%d_%H%M%S')}"),
    )
    args = parser.parse_args()

    forcefields = [value.strip() for value in args.forcefields.split(",") if value.strip()]
    invalid = sorted(set(forcefields) - {"glycam", "charmm", "gromos"})
    if invalid:
        parser.error(f"invalid forcefields: {', '.join(invalid)}")
    sys.path.insert(0, str(MD_ROOT / "scripts"))
    import schedule_extended24_exact_parallel as scheduler

    jobs = scheduler.build_queue(forcefields)
    if args.only:
        jobs = [
            (forcefield, row)
            for forcefield, row in jobs
            if row["clean_name"] == args.only or row["molecule_name"] == args.only
        ]
    results, paths = preflight_jobs(
        jobs,
        Path(args.output_dir),
        allow_experimental_gromos=args.allow_experimental_gromos,
    )
    passed = sum(bool(item["passed"]) for item in results)
    print(f"Preflight: {passed}/{len(results)} passed", flush=True)
    for label, path in paths.items():
        print(f"{label}: {path}", flush=True)
    raise SystemExit(0 if passed == len(results) else 2)


if __name__ == "__main__":
    main()
