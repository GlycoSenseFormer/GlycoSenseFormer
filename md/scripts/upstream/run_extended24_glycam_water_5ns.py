#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from charmm_canonical_transaction import (
    GLYCAM_SPEC,
    atomic_write_json,
    canonical_json_bytes,
    paths_for as canonical_paths_for,
    publish as publish_canonical_transaction,
    sha256_file,
    target_lock as canonical_target_lock,
    validate_commit as validate_canonical_commit,
)
from run_water_5ns_low_sugar_energy import (
    AMBER_TIP3P_ATOMTYPES,
    TOPDIR,
    ForceFieldJob,
    GpuFatalError,
    extract_solute_frames,
    insert_before_moleculetype,
    insert_before_system,
    run,
    sha256_frame_set,
)


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
SAMPLE_CSV = Path(os.environ.get("MD_SAMPLE_CSV", MD_ROOT / "md_small_sample_dataset_v1" / "md_small_sample_glycans_v1.csv"))
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
ALL_OUTPUT_DIR = Path(
    os.environ.get(
        "MD_GLYCAM_ALL_OUTPUT_DIR",
        MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_all_seed1",
    )
)
LOW_OUTPUT_DIR = Path(
    os.environ.get(
        "MD_GLYCAM_LOW_OUTPUT_DIR",
        MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1",
    )
)
RUN_VARIANT_SUFFIX = os.environ.get("MD_RUN_VARIANT_SUFFIX", "")
if RUN_VARIANT_SUFFIX and not re.fullmatch(r"_[A-Za-z0-9_-]+", RUN_VARIANT_SUFFIX):
    raise ValueError(f"invalid MD_RUN_VARIANT_SUFFIX: {RUN_VARIANT_SUFFIX!r}")
STATUS_CSV = RUN_ROOT / "extended24_glycam06j_water_5ns_status.csv"
STATUS_JSON = RUN_ROOT / "extended24_glycam06j_water_5ns_status.json"
STATUS_MD = RUN_ROOT / "extended24_glycam06j_water_5ns_status.md"
SDF_BY_CLASS = {
    "mono": ROOT / "data" / "dataset" / "casper_mono.sdf",
    "dimer": ROOT / "data" / "dataset" / "casper_dimer.sdf",
    "trimer": ROOT / "data" / "dataset" / "casper_trimer.sdf",
}

RUN_INPUT_EPOCH_SCHEMA = 1
OUTPUT_EPOCH_SCHEMA = 1
CANONICAL_RUN_DIRNAME = "glycam06j_water_5ns_200cand_canonical_v1"


def assert_formal_environment() -> None:
    """Reject diagnostic protocol/output overrides in canonical production."""

    errors: list[str] = []
    if RUN_VARIANT_SUFFIX:
        errors.append(f"MD_RUN_VARIANT_SUFFIX={RUN_VARIANT_SUFFIX!r}")
    for name in sorted(os.environ):
        if (
            name in {
                "MD_DT_PS",
                "MD_STOP_AFTER_STAGE",
                "MD_GLYCAM_ALL_OUTPUT_DIR",
                "MD_GLYCAM_LOW_OUTPUT_DIR",
            }
            or name.startswith("MD_NSTEPS_")
        ):
            errors.append(f"{name}={os.environ[name]!r}")
    if errors:
        raise RuntimeError(
            "formal GLYCAM canonical production rejects diagnostic overrides: "
            + ", ".join(errors)
        )


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_exact_file(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise RuntimeError(f"frozen run input is not a regular file: {destination}")
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(
                f"frozen GLYCAM run input differs from its epoch: {destination}; "
                "use a clean run root"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise RuntimeError(f"source changed while copying frozen input: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_exact_text(path: Path, payload: str, label: str) -> None:
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"frozen {label} differs from its input epoch: {path}")
        return
    _atomic_write_text(path, payload)


def _with_epoch_id(body: dict[str, object]) -> dict[str, object]:
    return {
        **body,
        "epoch_id": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _heavy_atom_count(row: dict[str, str]) -> int:
    raw = row.get("heavy_atoms") or row.get("n_heavy_atoms")
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid heavy-atom count for {row.get('clean_name')}: {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"invalid heavy-atom count for {row.get('clean_name')}: {value}")
    return value


def _bind_run_input_epoch(
    row: dict[str, str],
    *,
    run_dir: Path,
    sources: dict[str, Path],
    destination_names: dict[str, str],
    derived_inputs: dict[str, str],
) -> dict[str, object]:
    input_files = {
        label: {
            "name": destination_names.get(label, path.name),
            "sha256": sha256_file(path),
        }
        for label, path in sorted(sources.items())
    }
    body: dict[str, object] = {
        "schema_version": RUN_INPUT_EPOCH_SCHEMA,
        "forcefield": GLYCAM_SPEC.commit_forcefield,
        "clean_name": row["clean_name"],
        "molecule_name": row["molecule_name"],
        "source_class": row["source_class"],
        "sample_manifest": {
            "path": str(SAMPLE_CSV.resolve()),
            "sha256": sha256_file(SAMPLE_CSV),
        },
        "formal_protocol": {
            "run_variant_suffix": "",
            "integration_timestep_ps": 0.002,
            "nvt_duration_ps": 100.0,
            "npt_duration_ps": 200.0,
            "production_duration_ps": 5000.0,
            "candidate_spacing_ps": 25.0,
            "candidate_count": 200,
            "selected_count": 100,
            "water_model": "AMBER TIP3P",
            "seed": 1,
        },
        "inputs": input_files,
        "derived_input_sha256": dict(sorted(derived_inputs.items())),
    }
    expected = _with_epoch_id(body)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_input_epoch.json"
    if manifest_path.exists():
        try:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid frozen GLYCAM run-input epoch: {manifest_path}") from exc
        if recorded != expected:
            raise RuntimeError(
                f"GLYCAM run-input epoch changed for {row['clean_name']}; "
                "use a new clean run root"
            )
        return expected

    unbound = list(run_dir.iterdir())
    if unbound:
        raise RuntimeError(
            f"{row['clean_name']}: formal run directory contains artifacts without "
            "run_input_epoch.json; refusing checkpoint reuse: "
            + ", ".join(path.name for path in unbound[:8])
        )
    atomic_write_json(manifest_path, expected)
    return expected


def _bind_output_epoch(
    row: dict[str, str],
    job: ForceFieldJob,
    frames_dir: Path,
    selected_dir: Path,
) -> tuple[str, str]:
    frame_paths = sorted(frames_dir.glob("frame*.gro"))
    selected_paths = sorted(selected_dir.glob("*.gro"))
    if len(frame_paths) != 200 or len(selected_paths) != 100:
        raise RuntimeError(
            f"{row['clean_name']}: canonical GLYCAM output requires exactly "
            "200 candidate and 100 selected frames"
        )
    reference_sdf = SDF_BY_CLASS[row["source_class"]]
    required = {
        "run_input_epoch": job.run_dir / "run_input_epoch.json",
        "water_protocol": job.run_dir / "water_md_protocol.json",
        "extraction": job.run_dir / "solute_extraction_manifest.json",
        "energy": job.run_dir / "solute_potential_5ns_water_200cand.xvg",
        "selection_csv": job.run_dir
        / "glycam06j_water_5ns_lowest100_sugar_energy_selected.csv",
        "selection_metadata": job.run_dir
        / "glycam06j_water_5ns_lowest100_sugar_energy_metadata.json",
        "atom_map": Path(job.atom_index_map) if job.atom_index_map is not None else Path(""),
        "reference_sdf": reference_sdf,
    }
    missing = [f"{label}={path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{row['clean_name']}: GLYCAM output epoch lacks required provenance: "
            + ", ".join(missing)
        )
    body: dict[str, object] = {
        "schema_version": OUTPUT_EPOCH_SCHEMA,
        "forcefield": GLYCAM_SPEC.commit_forcefield,
        "clean_name": row["clean_name"],
        "molecule_name": row["molecule_name"],
        "source_class": row["source_class"],
        "n_heavy_atoms": _heavy_atom_count(row),
        "input_sha256": {label: sha256_file(path) for label, path in required.items()},
        "frame_count": len(frame_paths),
        "selected_frame_count": len(selected_paths),
        "frame_set_sha256": sha256_frame_set(frame_paths),
        "selected_frame_set_sha256": sha256_frame_set(selected_paths),
    }
    expected = _with_epoch_id(body)
    manifest_path = job.run_dir / "canonical_output_epoch.json"
    if manifest_path.exists():
        try:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid GLYCAM output epoch: {manifest_path}") from exc
        if recorded != expected:
            raise RuntimeError(
                f"{row['clean_name']}: GLYCAM output provenance changed within one run epoch"
            )
    else:
        atomic_write_json(manifest_path, expected)
    return str(expected["epoch_id"]), sha256_file(manifest_path)


def copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(source, dest)


def read_rows() -> list[dict[str, str]]:
    with SAMPLE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if not row.get("repair_forcefield", "").strip()
        or row["repair_forcefield"].strip() == "glycam"
    ]
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def existing_low_pickle(clean: str) -> Path:
    return canonical_paths_for(MD_ROOT, clean, GLYCAM_SPEC).low_pickle


def existing_all_pickle(clean: str) -> Path:
    return canonical_paths_for(MD_ROOT, clean, GLYCAM_SPEC).all_pickle


def prepare_job(row: dict[str, str]) -> ForceFieldJob:
    clean = row["clean_name"]
    dryrun = RUN_ROOT / clean / "glycam06j_topology_dryrun"
    acpype = dryrun / f"{clean}_GLYCAM06j.amb2gmx"
    run_dir_name = (
        f"glycam06j_water_5ns_200cand{RUN_VARIANT_SUFFIX}"
        if RUN_VARIANT_SUFFIX
        else CANONICAL_RUN_DIRNAME
    )
    run_dir = RUN_ROOT / clean / run_dir_name

    source_gro = acpype / f"{clean}_GLYCAM06j_GMX.gro"
    source_top = acpype / f"{clean}_GLYCAM06j_GMX.top"
    raw_gro = acpype / f"{clean}_GLYCAM06j_GMX_tleap_raw.gro"
    atom_map = dryrun / "sdf_to_glycam_frame_heavy_index_map.json"
    topology_metadata = dryrun / "glycam06j_topology_metadata.json"
    coordinate_metadata = dryrun / "glycam_coordinate_repair_metadata.json"
    chirality_audits = {
        "chirality_raw_to_final": dryrun / "glycam_raw_to_final_chirality_audit.json",
        "chirality_source_to_raw": dryrun / "glycam_source_mmff_to_raw_chirality_audit.json",
        "chirality_source_to_final": dryrun / "glycam_source_mmff_to_final_chirality_audit.json",
    }
    required = (
        source_gro,
        source_top,
        raw_gro,
        atom_map,
        topology_metadata,
        coordinate_metadata,
        *chirality_audits.values(),
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError(f"Missing GLYCAM06j GROMACS topology for {clean}: {acpype}")

    metadata = json.loads(topology_metadata.read_text(encoding="utf-8"))
    if metadata.get("clean_name") != clean or metadata.get("molecule_name") != row["molecule_name"]:
        raise RuntimeError(f"GLYCAM topology metadata identity mismatch: {topology_metadata}")
    sequence = metadata.get("glycam_sequence")
    if not isinstance(sequence, list) or not sequence or not all(
        isinstance(token, str) and token for token in sequence
    ):
        raise RuntimeError(f"GLYCAM topology lacks an exact official residue sequence: {topology_metadata}")
    evidence = metadata.get("glycam_library_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(f"GLYCAM topology lacks official-library evidence: {topology_metadata}")
    for label in ("leaprc", "prep"):
        evidence_path = evidence.get(label)
        evidence_hash = evidence.get(f"{label}_sha256")
        if not isinstance(evidence_path, str) or not isinstance(evidence_hash, str):
            raise RuntimeError(f"GLYCAM evidence is incomplete for {label}: {topology_metadata}")
        library_path = Path(evidence_path)
        if not library_path.is_file() or sha256_file(library_path) != evidence_hash:
            raise RuntimeError(f"GLYCAM official-library evidence changed for {label}: {library_path}")
    validated_tokens = evidence.get("validated_residue_tokens")
    if validated_tokens != sequence:
        raise RuntimeError(f"GLYCAM validated residue tokens differ from sequence: {topology_metadata}")

    for label, audit_path in chirality_audits.items():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        counts = audit.get("counts") if isinstance(audit, dict) else None
        if (
            not isinstance(counts, dict)
            or audit.get("strict_passed") is not True
            or counts.get("mismatched") != 0
            or counts.get("unevaluable") != 0
            or not isinstance(counts.get("evaluated_matching"), int)
            or counts["evaluated_matching"] < 1
        ):
            raise RuntimeError(f"GLYCAM strict chirality audit is not approved ({label}): {audit_path}")

    source_top_text = source_top.read_text(encoding="utf-8")
    prepared_top_text = insert_before_moleculetype(source_top_text, AMBER_TIP3P_ATOMTYPES)
    prepared_top_text = insert_before_system(
        prepared_top_text,
        '; Include local AMBER TIP3P water topology\n#include "amber_tip3p.itp"',
    )
    tip3p_source = TOPDIR / "amber99sb-ildn.ff" / "tip3p.itp"
    posre_sources = sorted(acpype.glob("posre_*.itp"))
    sources = {
        "solute_gro": source_gro,
        "source_topology": source_top,
        "raw_tleap_gro": raw_gro,
        "atom_map": atom_map,
        "topology_metadata": topology_metadata,
        "coordinate_metadata": coordinate_metadata,
        "amber_tip3p_itp": tip3p_source,
        **chirality_audits,
        **{f"posre_{index}": path for index, path in enumerate(posre_sources)},
    }
    derived_inputs = {
        "water_topology": hashlib.sha256(prepared_top_text.encode("utf-8")).hexdigest(),
        "solute_topology": hashlib.sha256(prepared_top_text.encode("utf-8")).hexdigest(),
    }
    epoch = _bind_run_input_epoch(
        row,
        run_dir=run_dir,
        sources=sources,
        destination_names={"amber_tip3p_itp": "amber_tip3p.itp"},
        derived_inputs=derived_inputs,
    )
    epoch_inputs = epoch.get("inputs")
    if not isinstance(epoch_inputs, dict):
        raise RuntimeError(f"invalid GLYCAM run input epoch for {clean}")

    solute_gro = run_dir / source_gro.name
    for label, source in sources.items():
        item = epoch_inputs.get(label)
        if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
            raise RuntimeError(f"invalid frozen GLYCAM source record: {label}")
        destination_name = str(item.get("name", ""))
        if not destination_name or Path(destination_name).name != destination_name:
            raise RuntimeError(f"invalid frozen GLYCAM destination name: {label}")
        _ensure_exact_file(source, run_dir / destination_name, str(item["sha256"]))

    water_top = run_dir / f"{clean}_GLYCAM06j_water.top"
    solute_top = run_dir / f"{clean}_GLYCAM06j_solute_only.top"
    _ensure_exact_text(water_top, prepared_top_text, "GLYCAM water topology")
    _ensure_exact_text(solute_top, prepared_top_text, "GLYCAM solute topology")

    return ForceFieldJob(
        key="glycam06j",
        label="GLYCAM06j",
        run_dir=run_dir,
        solute_gro=solute_gro,
        water_top=water_top,
        solute_top=solute_top,
        water_model="AMBER TIP3P",
        family="amber",
        atom_index_map=run_dir / atom_map.name,
    )


def convert_pickle(row: dict[str, str], job: ForceFieldJob, frames_dir: Path, selected_dir: Path) -> tuple[Path, Path]:
    clean = row["clean_name"]
    molecule = row["molecule_name"]
    reference_sdf = SDF_BY_CLASS[row["source_class"]]
    all_pickle = existing_all_pickle(clean)
    low_pickle = existing_low_pickle(clean)
    epoch_id, epoch_manifest_sha256 = _bind_output_epoch(
        row, job, frames_dir, selected_dir
    )
    attempt_dir = (
        job.run_dir
        / "canonical_publish_staging"
        / epoch_id
        / f"attempt_{os.getpid()}_{time.time_ns()}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    staged: dict[str, tuple[Path, Path]] = {}
    for role, output, source_dir, max_frames in (
        ("all", all_pickle, frames_dir, "200"),
        ("low", low_pickle, selected_dir, "100"),
    ):
        staged_pickle = attempt_dir / f"{role}.pickle"
        cmd = [
            sys.executable,
            str(MD_ROOT / "scripts" / "md_frames_to_conformer_pickle.py"),
            "--frames_glob",
            glob.escape(str(source_dir)) + "/*.gro",
            "--reference_sdf",
            str(reference_sdf),
            "--molecule_name",
            molecule,
            "--output",
            str(staged_pickle),
            "--logical_output",
            str(output),
            "--max_frames",
            max_frames,
            "--atom_index_map",
            str(job.atom_index_map),
            "--epoch_id",
            epoch_id,
            "--epoch_manifest_sha256",
            epoch_manifest_sha256,
            "--artifact_role",
            role,
        ]
        run(cmd, ROOT)
        staged[role] = (
            staged_pickle,
            staged_pickle.with_suffix(staged_pickle.suffix + ".meta.json"),
        )
    publish_canonical_transaction(
        MD_ROOT,
        clean,
        molecule_name=molecule,
        source_class=row["source_class"],
        n_heavy_atoms=_heavy_atom_count(row),
        epoch_id=epoch_id,
        epoch_manifest_sha256=epoch_manifest_sha256,
        staged_all_pickle=staged["all"][0],
        staged_all_meta=staged["all"][1],
        staged_low_pickle=staged["low"][0],
        staged_low_meta=staged["low"][1],
        spec=GLYCAM_SPEC,
    )
    return all_pickle, low_pickle


def process_row(row: dict[str, str], *, setup_only: bool) -> dict[str, object]:
    assert_formal_environment()
    if os.environ.get("MD_PREFLIGHT_PASSED") != "1":
        from preflight_extended24_exact import preflight_job

        check = preflight_job("glycam", row, RUN_ROOT)
        if not check["passed"]:
            raise RuntimeError("GLYCAM preflight blocked this job: " + " | ".join(check["errors"]))
    clean = row["clean_name"]
    started = time.time()
    all_pickle = existing_all_pickle(clean)
    low_pickle = existing_low_pickle(clean)
    with canonical_target_lock(MD_ROOT, clean, GLYCAM_SPEC):
        existing = validate_canonical_commit(
            MD_ROOT,
            clean,
            expected_molecule_name=row["molecule_name"],
            expected_source_class=row["source_class"],
            spec=GLYCAM_SPEC,
        )
        if existing is not None:
            return {
                "priority": int(row["priority"]),
                "molecule_name": row["molecule_name"],
                "clean_name": clean,
                "status": "done_existing",
                "run_dir": "",
                "all_pickle": str(all_pickle),
                "low_pickle": str(low_pickle),
                "canonical_commit": str(
                    canonical_paths_for(MD_ROOT, clean, GLYCAM_SPEC).commit
                ),
                "seconds": round(time.time() - started, 2),
                "message": "validated canonical GLYCAM transaction found",
            }

        job = prepare_job(row)
        if setup_only:
            return {
                "priority": int(row["priority"]),
                "molecule_name": row["molecule_name"],
                "clean_name": clean,
                "status": "setup_ready",
                "run_dir": str(job.run_dir),
                "all_pickle": str(all_pickle),
                "low_pickle": str(low_pickle),
                "seconds": round(time.time() - started, 2),
                "message": "inputs copied and frozen to run_input_epoch.json",
            }

        result = process_job_without_global_pickle(row, job)
        validate_canonical_commit(
            MD_ROOT,
            clean,
            expected_molecule_name=row["molecule_name"],
            expected_source_class=row["source_class"],
            missing_ok=False,
            spec=GLYCAM_SPEC,
        )
        result.update(
            {
                "priority": int(row["priority"]),
                "molecule_name": row["molecule_name"],
                "clean_name": clean,
                "status": "done",
                "canonical_commit": str(
                    canonical_paths_for(MD_ROOT, clean, GLYCAM_SPEC).commit
                ),
                "seconds": round(time.time() - started, 2),
            }
        )
        return result


def process_job_without_global_pickle(row: dict[str, str], job: ForceFieldJob) -> dict[str, object]:
    from run_water_5ns_low_sugar_energy import (
        read_energy_metadata,
        rerun_solute_energy,
        run_water_md,
        select_low_energy,
        selected_indices,
    )

    print(f"\n=== {row['priority']} {row['molecule_name']} / GLYCAM06j ===", flush=True)
    run_water_md(job)
    frames_dir = extract_solute_frames(job)
    energy_xvg = rerun_solute_energy(job, frames_dir)
    selected_dir = select_low_energy(job, frames_dir, energy_xvg)
    all_pickle, low_pickle = convert_pickle(row, job, frames_dir, selected_dir)
    metadata = read_energy_metadata(job)
    return {
        "run_dir": str(job.run_dir),
        "frames_dir": str(frames_dir),
        "selected_dir": str(selected_dir),
        "energy_xvg": str(energy_xvg),
        "candidate_csv": str(job.run_dir / "glycam06j_water_5ns_200_candidate_solute_potential.csv"),
        "selected_csv": str(job.run_dir / "glycam06j_water_5ns_lowest100_sugar_energy_selected.csv"),
        "all_pickle": str(all_pickle),
        "low_pickle": str(low_pickle),
        "n_candidates": metadata["n_candidates"],
        "top_n": metadata["top_n"],
        "lowest_potential_kj_mol": metadata["lowest_potential_kj_mol"],
        "highest_selected_potential_kj_mol": metadata["highest_selected_potential_kj_mol"],
        "highest_candidate_potential_kj_mol": metadata["highest_candidate_potential_kj_mol"],
        "selected_candidate_indices": selected_indices(job),
        "message": "completed",
    }


def write_status(rows: list[dict[str, object]]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with STATUS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    STATUS_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    lines = [
        "# Extended-24 GLYCAM06j water 5 ns status",
        "",
        f"- status counts: {counts}",
        "",
        "| priority | molecule | status | run_dir | low100 pickle |",
        "|---:|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: int(item["priority"])):
        lines.append(
            f"| {row['priority']} | `{row['molecule_name']}` | {row['status']} | `{row.get('run_dir', '')}` | `{row.get('low_pickle', '')}` |"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_status() -> list[dict[str, object]]:
    if not STATUS_JSON.exists():
        return []
    return json.loads(STATUS_JSON.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extended-24 GLYCAM06j explicit-water 5 ns jobs.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare copied inputs/topologies but do not run MD.")
    parser.add_argument("--only", default="", help="Only run one clean_name or molecule_name.")
    parser.add_argument("--max-new", type=int, default=0, help="Maximum non-existing jobs to process; 0 means no limit.")
    args = parser.parse_args()

    rows = read_rows()
    if args.only:
        rows = [row for row in rows if row["clean_name"] == args.only or row["molecule_name"] == args.only]
        if not rows:
            raise SystemExit(f"No sample matched --only {args.only!r}")

    status_by_clean = {str(row["clean_name"]): row for row in load_existing_status()}
    completed = []
    new_count = 0
    for row in rows:
        clean = row["clean_name"]
        previous = status_by_clean.get(clean)
        if previous and previous.get("status") in {"done", "done_existing"}:
            commit = validate_canonical_commit(
                MD_ROOT,
                clean,
                expected_molecule_name=row["molecule_name"],
                expected_source_class=row["source_class"],
                spec=GLYCAM_SPEC,
            )
            if commit is not None:
                completed.append(previous)
                continue
        if args.max_new and new_count >= args.max_new:
            if previous:
                completed.append(previous)
            continue
        try:
            result = process_row(row, setup_only=args.setup_only)
        except GpuFatalError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "priority": int(row["priority"]),
                "molecule_name": row["molecule_name"],
                "clean_name": clean,
                "status": "failed",
                "run_dir": str(RUN_ROOT / clean / CANONICAL_RUN_DIRNAME),
                "all_pickle": str(existing_all_pickle(clean)),
                "low_pickle": str(existing_low_pickle(clean)),
                "seconds": 0.0,
                "message": repr(exc),
            }
        status_by_clean[clean] = result
        completed = list(status_by_clean.values())
        write_status(completed)
        new_count += 1

    write_status(list(status_by_clean.values()))
    print(json.dumps({"status_csv": str(STATUS_CSV), "status_json": str(STATUS_JSON)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
