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
    atomic_write_json,
    canonical_json_bytes,
    paths_for as canonical_paths_for,
    publish as publish_canonical_transaction,
    sha256_file,
    target_lock as canonical_target_lock,
    validate_commit as validate_canonical_commit,
)
from charmm_standard_metadata import assess_charmm_standard_metadata
from preflight_extended24_exact import (
    locate_inputs,
    sha256_directory_tree,
    verify_charmm_artifact_provenance,
)
from run_water_5ns_low_sugar_energy import (
    ForceFieldJob,
    GpuFatalError,
    extract_solute_frames,
    run,
    sha256_frame_set,
)


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
SAMPLE_CSV = Path(os.environ.get("MD_SAMPLE_CSV", MD_ROOT / "md_small_sample_dataset_v1" / "md_small_sample_glycans_v1.csv"))
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
RUN_VARIANT_SUFFIX = os.environ.get("MD_RUN_VARIANT_SUFFIX", "")
if RUN_VARIANT_SUFFIX and not re.fullmatch(r"_[A-Za-z0-9_-]+", RUN_VARIANT_SUFFIX):
    raise ValueError(f"invalid MD_RUN_VARIANT_SUFFIX: {RUN_VARIANT_SUFFIX!r}")
STATUS_CSV = RUN_ROOT / "extended24_charmm36_exact_water_5ns_status.csv"
STATUS_JSON = RUN_ROOT / "extended24_charmm36_exact_water_5ns_status.json"
STATUS_MD = RUN_ROOT / "extended24_charmm36_exact_water_5ns_status.md"
SDF_BY_CLASS = {
    "mono": ROOT / "data" / "dataset" / "casper_mono.sdf",
    "dimer": ROOT / "data" / "dataset" / "casper_dimer.sdf",
    "trimer": ROOT / "data" / "dataset" / "casper_trimer.sdf",
}


def read_rows() -> list[dict[str, str]]:
    with SAMPLE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if not row.get("repair_forcefield", "").strip()
        or row["repair_forcefield"].strip() == "charmm"
    ]
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def existing_low_pickle(clean: str) -> Path:
    return canonical_paths_for(MD_ROOT, clean).low_pickle


def existing_all_pickle(clean: str) -> Path:
    return canonical_paths_for(MD_ROOT, clean).all_pickle


RUN_INPUT_EPOCH_SCHEMA = 1
OUTPUT_EPOCH_SCHEMA = 1


def _heavy_atom_count(row: dict[str, str]) -> int:
    raw = row.get("heavy_atoms") or row.get("n_heavy_atoms")
    try:
        value = int(str(raw))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid heavy-atom count for {row.get('clean_name')}: {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"invalid heavy-atom count for {row.get('clean_name')}: {value}")
    return value


def assert_formal_environment() -> None:
    errors: list[str] = []
    if RUN_VARIANT_SUFFIX:
        errors.append(f"MD_RUN_VARIANT_SUFFIX={RUN_VARIANT_SUFFIX!r}")
    for name in sorted(os.environ):
        if name in {"MD_DT_PS", "MD_STOP_AFTER_STAGE"} or name.startswith("MD_NSTEPS_"):
            errors.append(f"{name}={os.environ[name]!r}")
    if errors:
        raise RuntimeError(
            "formal CHARMM canonical production rejects diagnostic protocol overrides: "
            + ", ".join(errors)
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_exact_file(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise RuntimeError(f"run input is not a regular file: {destination}")
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f"frozen run input differs from its epoch: {destination}; "
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


def _ensure_exact_tree(
    source: Path,
    destination: Path,
    expected_tree: dict[str, object],
) -> None:
    if destination.exists():
        actual_tree = sha256_directory_tree(destination)
        if actual_tree != expected_tree:
            raise RuntimeError(
                f"frozen force-field tree differs from its epoch: {destination}; "
                "use a clean run root"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    temporary_tree = temporary_parent / destination.name
    try:
        shutil.copytree(source, temporary_tree, symlinks=False)
        if sha256_directory_tree(temporary_tree) != expected_tree:
            raise RuntimeError(f"source changed while copying force-field tree: {source}")
        os.replace(temporary_tree, destination)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def _ensure_exact_text(path: Path, text: str, label: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen {label} differs from its input epoch: {path}")
        return
    _atomic_write_text(path, text)


def clean_topology_text(text: str) -> str:
    lines: list[str] = []
    in_molecules = False
    solvent_names = {"SOL", "WAT", "HOH", "NA", "CL", "K", "SOD", "CLA"}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_molecules = stripped.lower() == "[ molecules ]"
            lines.append(line)
            continue
        if in_molecules and stripped and not stripped.startswith(";"):
            first = stripped.split()[0]
            if first in solvent_names:
                continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def molecule_stem(molecule_name: str) -> str:
    return molecule_name.replace("/", "_")


def _with_epoch_id(body: dict[str, object]) -> dict[str, object]:
    epoch_id = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return {**body, "epoch_id": epoch_id}


def _bind_run_input_epoch(
    row: dict[str, str],
    *,
    run_dir: Path,
    source_gro: Path,
    source_top: Path,
    source_ff: Path,
    metadata_path: Path,
    atom_map: Path,
    posre: Path | None,
) -> dict[str, object]:
    input_files: dict[str, object] = {
        "processed_gro": {"name": source_gro.name, "sha256": sha256_file(source_gro)},
        "topology": {"name": source_top.name, "sha256": sha256_file(source_top)},
        "metadata": {"name": metadata_path.name, "sha256": sha256_file(metadata_path)},
        "atom_map": {"name": atom_map.name, "sha256": sha256_file(atom_map)},
        "forcefield": {
            "name": source_ff.name,
            "tree": sha256_directory_tree(source_ff),
        },
        "posre": (
            {"name": posre.name, "sha256": sha256_file(posre)} if posre is not None else None
        ),
    }
    body: dict[str, object] = {
        "schema_version": RUN_INPUT_EPOCH_SCHEMA,
        "forcefield": "charmm36_exact",
        "clean_name": row["clean_name"],
        "molecule_name": row["molecule_name"],
        "source_class": row["source_class"],
        "formal_protocol": {
            "run_variant_suffix": "",
            "integration_timestep_ps": 0.002,
            "nvt_duration_ps": 100.0,
            "npt_duration_ps": 200.0,
            "production_duration_ps": 5000.0,
            "candidate_spacing_ps": 25.0,
            "candidate_count": 200,
        },
        "inputs": input_files,
    }
    expected = _with_epoch_id(body)
    manifest_path = run_dir / "run_input_epoch.json"
    if manifest_path.exists():
        try:
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid frozen run-input epoch: {manifest_path}") from exc
        if recorded != expected:
            raise RuntimeError(
                f"run-input epoch changed for {row['clean_name']}; use a new clean run root"
            )
        return expected

    unbound = [path for path in run_dir.iterdir() if path.name != manifest_path.name]
    if unbound:
        raise RuntimeError(
            f"{row['clean_name']}: run directory contains artifacts without run_input_epoch.json; "
            "refusing to bind legacy TPR/CPT/output files: "
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
            f"{row['clean_name']}: canonical output epoch requires 200 all/100 selected frames"
        )
    reference_sdf = SDF_BY_CLASS[row["source_class"]]
    required = {
        "run_input_epoch": job.run_dir / "run_input_epoch.json",
        "water_protocol": job.run_dir / "water_md_protocol.json",
        "extraction": job.run_dir / "solute_extraction_manifest.json",
        "energy": job.run_dir / "solute_potential_5ns_water_200cand.xvg",
        "selection_csv": job.run_dir / "charmm36_exact_water_5ns_lowest100_sugar_energy_selected.csv",
        "selection_metadata": job.run_dir / "charmm36_exact_water_5ns_lowest100_sugar_energy_metadata.json",
        "atom_map": Path(job.atom_index_map) if job.atom_index_map is not None else Path(""),
        "reference_sdf": reference_sdf,
    }
    energy_method = job.run_dir / "solute_energy_method.json"
    if job.solute_charge or energy_method.exists():
        required["energy_method"] = energy_method
    missing = [f"{label}={path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{row['clean_name']}: output epoch lacks required provenance: " + ", ".join(missing)
        )
    body: dict[str, object] = {
        "schema_version": OUTPUT_EPOCH_SCHEMA,
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
            raise RuntimeError(f"invalid canonical output epoch: {manifest_path}") from exc
        if recorded != expected:
            raise RuntimeError(
                f"{row['clean_name']}: canonical output provenance changed within one run epoch"
            )
    else:
        atomic_write_json(manifest_path, expected)
    return str(expected["epoch_id"]), sha256_file(manifest_path)


def prepare_job(row: dict[str, str]) -> ForceFieldJob:
    clean = row["clean_name"]
    molecule = row["molecule_name"]
    stem = molecule_stem(molecule)
    dryrun = RUN_ROOT / clean / "charmm36_standard_topology_dryrun"
    run_dir = RUN_ROOT / clean / f"charmm36_standard_water_5ns_200cand{RUN_VARIANT_SUFFIX}"

    source_gro = dryrun / f"{stem}_CHARMM36_STANDARD_processed.gro"
    source_top = dryrun / f"{stem}_CHARMM36_STANDARD.top"
    source_ff = dryrun / "charmm36_standard.ff"
    metadata_path = dryrun / "charmm36_standard_input_metadata.json"
    atom_map = dryrun / "sdf_to_charmm_frame_heavy_index_map.json"
    if not all(path.exists() for path in (source_gro, source_top, source_ff, metadata_path, atom_map)):
        raise FileNotFoundError(f"Missing CHARMM standard topology for {clean}: {dryrun}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy = assess_charmm_standard_metadata(
        metadata,
        expected_molecule_name=molecule,
        expected_clean_name=clean,
        expected_source_class=row.get("source_class"),
    )
    if not bool(policy["approved"]):
        raise RuntimeError(
            f"CHARMM standard metadata is not approved for water MD: {metadata_path}; "
            + "; ".join(str(message) for message in policy["errors"])
        )
    artifact_provenance = verify_charmm_artifact_provenance(
        locate_inputs("charmm", row, RUN_ROOT),
        metadata,
    )
    if not bool(artifact_provenance["approved"]):
        raise RuntimeError(
            f"CHARMM artifact provenance is not approved for water MD: {metadata_path}; "
            + "; ".join(str(message) for message in artifact_provenance["errors"])
        )
    charge_e = float(policy["charge_e"])
    solute_charge = int(round(charge_e))

    run_dir.mkdir(parents=True, exist_ok=True)
    source_posre = dryrun / "posre.itp"
    posre = source_posre if source_posre.is_file() else None
    epoch = _bind_run_input_epoch(
        row,
        run_dir=run_dir,
        source_gro=source_gro,
        source_top=source_top,
        source_ff=source_ff,
        metadata_path=metadata_path,
        atom_map=atom_map,
        posre=posre,
    )
    inputs = epoch["inputs"]
    assert isinstance(inputs, dict)

    solute_gro = run_dir / source_gro.name
    _ensure_exact_file(
        source_gro,
        solute_gro,
        str(inputs["processed_gro"]["sha256"]),  # type: ignore[index]
    )
    _ensure_exact_tree(
        source_ff,
        run_dir / source_ff.name,
        inputs["forcefield"]["tree"],  # type: ignore[index]
    )
    _ensure_exact_file(
        metadata_path,
        run_dir / metadata_path.name,
        str(inputs["metadata"]["sha256"]),  # type: ignore[index]
    )
    if posre is not None:
        _ensure_exact_file(
            posre,
            run_dir / posre.name,
            str(inputs["posre"]["sha256"]),  # type: ignore[index]
        )
    _ensure_exact_file(
        atom_map,
        run_dir / atom_map.name,
        str(inputs["atom_map"]["sha256"]),  # type: ignore[index]
    )

    water_top = run_dir / f"{stem}_CHARMM36_STANDARD_water.top"
    solute_top = run_dir / f"{stem}_CHARMM36_STANDARD_solute_only.top"
    clean_topology = clean_topology_text(source_top.read_text(encoding="utf-8"))
    _ensure_exact_text(solute_top, clean_topology, "solute-only topology")
    em_tpr = run_dir / "em_water.tpr"
    if not em_tpr.exists():
        stale_stage_artifacts = [
            run_dir / f"{stage}.{suffix}"
            for stage in ("em_water", "nvt_water", "npt_water", "md_5ns_water_200cand")
            for suffix in ("cpt", "gro", "log", "edr", "trr", "xtc")
            if (run_dir / f"{stage}.{suffix}").exists()
        ]
        if stale_stage_artifacts:
            raise RuntimeError(
                f"{clean}: stage artifacts exist without em_water.tpr; refusing checkpoint reuse: "
                + ", ".join(path.name for path in stale_stage_artifacts[:8])
            )
        _atomic_write_text(water_top, clean_topology)
    elif not water_top.is_file():
        raise RuntimeError(f"{clean}: em_water.tpr exists but its frozen water topology is missing")

    return ForceFieldJob(
        key="charmm36_exact",
        label="CHARMM36 standard carbohydrate residue/patch topology",
        run_dir=run_dir,
        solute_gro=solute_gro,
        water_top=water_top,
        solute_top=solute_top,
        water_model="CHARMM TIP3P",
        family="charmm",
        atom_index_map=run_dir / atom_map.name,
        solute_charge=solute_charge,
        positive_ion="SOD",
        negative_ion="CLA",
    )


def convert_pickle(row: dict[str, str], job: ForceFieldJob, frames_dir: Path, selected_dir: Path) -> tuple[Path, Path]:
    clean = row["clean_name"]
    molecule = row["molecule_name"]
    reference_sdf = SDF_BY_CLASS[row["source_class"]]
    all_pickle = existing_all_pickle(clean)
    low_pickle = existing_low_pickle(clean)
    epoch_id, epoch_manifest_sha256 = _bind_output_epoch(row, job, frames_dir, selected_dir)
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
    )
    return all_pickle, low_pickle


def process_job_transactional(row: dict[str, str], job: ForceFieldJob) -> dict[str, object]:
    from run_water_5ns_low_sugar_energy import (
        read_energy_metadata,
        rerun_solute_energy,
        run_water_md,
        select_low_energy,
        selected_indices,
    )

    print(f"\n=== {row['priority']} {row['molecule_name']} / CHARMM36 exact ===", flush=True)
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
        "candidate_csv": str(job.run_dir / "charmm36_exact_water_5ns_200_candidate_solute_potential.csv"),
        "selected_csv": str(job.run_dir / "charmm36_exact_water_5ns_lowest100_sugar_energy_selected.csv"),
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


def rerun_existing_solute_energy_only(row: dict[str, str]) -> dict[str, object]:
    """Recompute only the solute-energy epoch from an existing 5 ns trajectory.

    This deliberately does not select frames, convert pickles, bind a canonical
    output epoch, or publish a canonical transaction.  The target lock prevents
    a formal publisher for the same glycan from racing this repair operation.
    """

    from run_water_5ns_low_sugar_energy import (
        mdrun_finished,
        rerun_solute_energy,
    )

    assert_formal_environment()
    clean = row["clean_name"]
    with canonical_target_lock(MD_ROOT, clean):
        existing = validate_canonical_commit(
            MD_ROOT,
            clean,
            expected_molecule_name=row["molecule_name"],
            expected_source_class=row["source_class"],
        )
        if existing is not None:
            raise RuntimeError(
                f"{clean}: canonical transaction already exists; refusing to mutate its "
                "source post-processing epoch"
            )
        job = prepare_job(row)
        if not job.solute_charge:
            raise RuntimeError(
                f"{clean}: --energy-rerun-only is the charged-solute repair path; "
                "refusing to alter a neutral-solute post-processing epoch"
            )
        if not mdrun_finished(job, "md_5ns_water_200cand"):
            raise RuntimeError(
                f"{clean}: no complete current 5 ns trajectory; energy-only mode will not "
                "start or resume production MD"
            )
        frames_dir = extract_solute_frames(job)
        energy_xvg = rerun_solute_energy(job, frames_dir)
        method_path = job.run_dir / "solute_energy_method.json"
        method = json.loads(method_path.read_text(encoding="utf-8"))
        return {
            "status": "energy_rerun_complete",
            "clean_name": clean,
            "molecule_name": row["molecule_name"],
            "run_dir": str(job.run_dir),
            "source_production_tpr_sha256": sha256_file(
                job.run_dir / "md_5ns_water_200cand.tpr"
            ),
            "source_production_xtc_sha256": sha256_file(
                job.run_dir / "md_5ns_water_200cand.xtc"
            ),
            "energy_xvg": str(energy_xvg),
            "energy_xvg_sha256": sha256_file(energy_xvg),
            "energy_method": str(method_path),
            "energy_method_sha256": sha256_file(method_path),
            "cutoff_nm": method.get("cutoff_nm"),
            "maximum_pair_distance_nm": method.get("maximum_pair_distance_nm"),
            "minimum_box_length_nm": method.get("minimum_box_length_nm"),
            "minimum_image_margin_nm": method.get("minimum_image_margin_nm"),
            "canonical_publish_performed": False,
        }


def process_row(row: dict[str, str], *, setup_only: bool) -> dict[str, object]:
    assert_formal_environment()
    if os.environ.get("MD_PREFLIGHT_PASSED") != "1":
        from preflight_extended24_exact import preflight_job

        check = preflight_job("charmm", row, RUN_ROOT)
        if not check["passed"]:
            raise RuntimeError("CHARMM preflight blocked this job: " + " | ".join(check["errors"]))
    clean = row["clean_name"]
    started = time.time()
    all_pickle = existing_all_pickle(clean)
    low_pickle = existing_low_pickle(clean)
    with canonical_target_lock(MD_ROOT, clean):
        existing = validate_canonical_commit(
            MD_ROOT,
            clean,
            expected_molecule_name=row["molecule_name"],
            expected_source_class=row["source_class"],
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
                "canonical_commit": str(canonical_paths_for(MD_ROOT, clean).commit),
                "seconds": round(time.time() - started, 2),
                "message": "validated canonical CHARMM transaction found",
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

        result = process_job_transactional(row, job)
        validate_canonical_commit(
            MD_ROOT,
            clean,
            expected_molecule_name=row["molecule_name"],
            expected_source_class=row["source_class"],
            missing_ok=False,
        )
        result.update(
            {
                "priority": int(row["priority"]),
                "molecule_name": row["molecule_name"],
                "clean_name": clean,
                "status": "done",
                "canonical_commit": str(canonical_paths_for(MD_ROOT, clean).commit),
                "seconds": round(time.time() - started, 2),
            }
        )
        return result


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
        "# Extended-24 CHARMM36 exact water 5 ns status",
        "",
        f"- status counts: {counts}",
        "",
        "说明：这是 exact single-residue CHARMM36 topology smoke path，不覆盖原 `CHARMM36-BMM` 锚点结果。",
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
    parser = argparse.ArgumentParser(description="Run extended-24 CHARMM36 exact explicit-water 5 ns jobs.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare copied inputs/topologies but do not run MD.")
    parser.add_argument(
        "--energy-rerun-only",
        action="store_true",
        help=(
            "Recompute solute-energy post-processing from one completed 5 ns trajectory; "
            "requires --only and never publishes canonical outputs."
        ),
    )
    parser.add_argument("--only", default="", help="Only run one clean_name or molecule_name.")
    parser.add_argument("--max-new", type=int, default=0, help="Maximum non-existing jobs to process; 0 means no limit.")
    args = parser.parse_args()

    rows = read_rows()
    if args.only:
        rows = [row for row in rows if row["clean_name"] == args.only or row["molecule_name"] == args.only]
        if not rows:
            raise SystemExit(f"No sample matched --only {args.only!r}")

    if args.energy_rerun_only:
        if args.setup_only or args.max_new:
            raise SystemExit("--energy-rerun-only cannot be combined with --setup-only/--max-new")
        if not args.only or len(rows) != 1:
            raise SystemExit("--energy-rerun-only requires --only matching exactly one sample")
        print(
            json.dumps(
                rerun_existing_solute_energy_only(rows[0]),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

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
            )
            if commit is not None:
                completed.append(previous)
                continue
        if args.max_new and new_count >= args.max_new:
            completed.append(
                {
                    "priority": int(row["priority"]),
                    "molecule_name": row["molecule_name"],
                    "clean_name": clean,
                    "status": "pending",
                    "message": "max-new limit reached",
                }
            )
            continue
        try:
            result = process_row(row, setup_only=args.setup_only)
        except GpuFatalError:
            raise
        except Exception as exc:
            result = {
                "priority": int(row["priority"]),
                "molecule_name": row["molecule_name"],
                "clean_name": clean,
                "status": "failed",
                "message": repr(exc),
            }
        if result["status"] not in {"done_existing", "setup_ready"}:
            new_count += 1
        elif result["status"] == "setup_ready":
            new_count += 1
        completed.append(result)
        write_status(completed)
    write_status(completed)
    counts: dict[str, int] = {}
    for row in completed:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(json.dumps({"counts": counts, "status_csv": str(STATUS_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
