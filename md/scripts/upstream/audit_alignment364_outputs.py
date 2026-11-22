#!/usr/bin/env python3
"""Audit the three-force-field alignment target without trusting file presence.

Legacy outputs predate transactional commits.  This auditor therefore loads
every pickle, verifies metadata/identity/shape/finiteness, checks that the
selected 100 conformers are members of the 200 candidates, and validates a
canonical commit whenever one exists.  Alignment-repair jobs are stricter:
they count only when both production-approved and canonically committed.

The summary verifies the frozen target/job manifests, reports coverage set
intersections, and emits a separate 55-row job ledger.  It never creates or
repairs simulation outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from charmm_canonical_transaction import SPECS, paths_for, sha256_file, validate_commit


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
ALIGNMENT_ROOT = MD_ROOT / "md_alignment364_v1"
DEFAULT_MANIFEST = ALIGNMENT_ROOT / "alignment364_targets.csv"
DEFAULT_JOBS = ALIGNMENT_ROOT / "alignment364_jobs.csv"
DEFAULT_APPROVED = ALIGNMENT_ROOT / "alignment364_approved_jobs.csv"
DEFAULT_OUTPUT = ALIGNMENT_ROOT / "baseline_output_audit.csv"

FORCEFIELDS = ("charmm", "gromos", "glycam")
FROZEN_MANIFEST_SHA256 = "c704c1470606d5ea371eaae3d30b947177c89bc534aed37bfd224e9cf06670b8"
FROZEN_JOBS_SHA256 = "9ca9c18a11cd538e70d68763135aab6325b6a124d2e7b09c72cd08c9d6a2db7d"


@dataclass(frozen=True)
class OutputSource:
    label: str
    all_dirname: str
    low_dirname: str
    canonical_spec_key: str | None


OUTPUT_SOURCES = {
    "charmm": (
        OutputSource(
            "charmm-main",
            "conformations_md_charmm36_exact_water_5ns_200cand_all_seed1",
            "conformations_md_charmm36_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
            "charmm",
        ),
    ),
    "gromos": (
        OutputSource(
            "gromos-main",
            "conformations_md_gromos_exact_water_5ns_200cand_all_seed1",
            "conformations_md_gromos_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
            "gromos",
        ),
    ),
    "glycam": (
        OutputSource(
            "glycam-main",
            "conformations_md_glycam06j_water_5ns_200cand_all_seed1",
            "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1",
            "glycam",
        ),
        OutputSource(
            "glycam-enantiomer-v15",
            "conformations_md_glycam06j_enantiomer_v15_all_seed1",
            "conformations_md_glycam06j_enantiomer_v15_lowest100_seed1",
            None,
        ),
    ),
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 364 or len({row["clean_name"] for row in rows}) != 364:
        raise ValueError(f"alignment manifest is not 364 unique molecules: {path}")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return value


def _missing_forcefields(row: dict[str, str]) -> set[str]:
    value = row.get("missing_forcefields", "")
    return {item for item in value.split(";") if item}


def verify_alignment_inputs(
    manifest_path: Path = DEFAULT_MANIFEST,
    jobs_path: Path = DEFAULT_JOBS,
    approved_path: Path = DEFAULT_APPROVED,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    """Verify the immutable alignment definition and current approval ledger.

    Target and job SHA256 values are compiled into this auditor so editing a
    CSV and its adjacent JSON cannot silently redefine alignment-364.  The
    approval ledger may evolve, but its current hash must match its sidecar and
    every approved key must be a prepared member of the frozen 55-job queue.
    """

    manifest_sha256 = sha256_file(manifest_path)
    jobs_sha256 = sha256_file(jobs_path)
    if manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"frozen alignment manifest SHA256 mismatch: expected "
            f"{FROZEN_MANIFEST_SHA256}, observed {manifest_sha256}"
        )
    if jobs_sha256 != FROZEN_JOBS_SHA256:
        raise ValueError(
            f"frozen alignment job manifest SHA256 mismatch: expected "
            f"{FROZEN_JOBS_SHA256}, observed {jobs_sha256}"
        )

    manifest_rows = read_manifest(manifest_path)
    jobs = _read_csv(jobs_path)
    approved = _read_csv(approved_path)
    target_metadata_path = manifest_path.with_suffix(".json")
    approved_metadata_path = approved_path.with_suffix(".json")
    target_metadata = _read_json_object(target_metadata_path)
    approved_metadata = _read_json_object(approved_metadata_path)

    if target_metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("alignment target sidecar does not match the frozen manifest")
    if target_metadata.get("job_manifest_sha256") != jobs_sha256:
        raise ValueError("alignment target sidecar does not match the frozen job manifest")
    if target_metadata.get("target_count") != 364 or target_metadata.get("job_count") != 55:
        raise ValueError("alignment target sidecar has unexpected target/job counts")
    if len(jobs) != 55 or len({(row["repair_forcefield"], row["clean_name"]) for row in jobs}) != 55:
        raise ValueError("alignment job manifest is not 55 unique forcefield/name keys")

    manifest_by_name = {row["clean_name"]: row for row in manifest_rows}
    expected_job_keys = {
        (forcefield, row["clean_name"])
        for row in manifest_rows
        for forcefield in _missing_forcefields(row)
    }
    job_keys = {(row["repair_forcefield"], row["clean_name"]) for row in jobs}
    if job_keys != expected_job_keys:
        raise ValueError(
            "frozen job keys do not equal the manifest's missing-forcefield keys: "
            f"missing={sorted(expected_job_keys - job_keys)}; "
            f"extra={sorted(job_keys - expected_job_keys)}"
        )
    for row in jobs:
        forcefield = row["repair_forcefield"]
        clean_name = row["clean_name"]
        if forcefield not in FORCEFIELDS or clean_name not in manifest_by_name:
            raise ValueError(f"invalid alignment job identity: {forcefield}/{clean_name}")
        target = manifest_by_name[clean_name]
        if forcefield not in _missing_forcefields(target):
            raise ValueError(f"alignment job is not a frozen coverage gap: {forcefield}/{clean_name}")
        if target.get(f"baseline_has_{forcefield}") != "False":
            raise ValueError(f"alignment job contradicts baseline coverage: {forcefield}/{clean_name}")

    approved_sha256 = sha256_file(approved_path)
    if approved_metadata.get("approved_manifest_sha256") != approved_sha256:
        raise ValueError("approval sidecar SHA256 does not match the approval manifest")
    if approved_metadata.get("source_job_count") != 55:
        raise ValueError("approval sidecar does not refer to the frozen 55-job queue")
    source_hashes = approved_metadata.get("source_sha256")
    if not isinstance(source_hashes, dict) or source_hashes.get(str(jobs_path.resolve())) != jobs_sha256:
        raise ValueError("approval sidecar source hash does not match the frozen job manifest")
    approved_keys = {(row["repair_forcefield"], row["clean_name"]) for row in approved}
    if len(approved_keys) != len(approved) or not approved_keys <= job_keys:
        raise ValueError("approval manifest contains duplicate or non-alignment jobs")
    if any(row.get("parameter_audit_status") != "prepared" for row in approved):
        raise ValueError("approval manifest contains a job without prepared parameter status")
    if approved_metadata.get("approved_job_count") != len(approved):
        raise ValueError("approval sidecar approved-job count mismatch")
    if approved_metadata.get("blocked_job_count") != len(jobs) - len(approved):
        raise ValueError("approval sidecar blocked-job count mismatch")

    integrity: dict[str, object] = {
        "verified": True,
        "target_manifest": {
            "path": str(manifest_path.resolve()),
            "expected_sha256": FROZEN_MANIFEST_SHA256,
            "observed_sha256": manifest_sha256,
            "sidecar_sha256": target_metadata.get("manifest_sha256"),
        },
        "job_manifest": {
            "path": str(jobs_path.resolve()),
            "expected_sha256": FROZEN_JOBS_SHA256,
            "observed_sha256": jobs_sha256,
            "sidecar_sha256": target_metadata.get("job_manifest_sha256"),
        },
        "approval_manifest": {
            "path": str(approved_path.resolve()),
            "observed_sha256": approved_sha256,
            "sidecar_sha256": approved_metadata.get("approved_manifest_sha256"),
            "approved_job_count": len(approved),
        },
    }
    return manifest_rows, jobs, approved, integrity


def metadata_path(pickle_path: Path) -> Path:
    return pickle_path.with_suffix(pickle_path.suffix + ".meta.json")


def _load_metadata(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not an object: {path}")
    return value


def _load_conformers(path: Path, expected_frames: int, expected_heavy: int) -> list[np.ndarray]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list) or len(value) != expected_frames:
        raise ValueError(
            f"{path}: expected list[{expected_frames}], got "
            f"{type(value).__name__}[{len(value) if hasattr(value, '__len__') else '?'}]"
        )
    result: list[np.ndarray] = []
    for index, conformer in enumerate(value):
        array = np.asarray(conformer)
        if array.shape != (expected_heavy, 3):
            raise ValueError(
                f"{path}: frame {index} shape {array.shape}, expected ({expected_heavy}, 3)"
            )
        if not np.issubdtype(array.dtype, np.number) or not bool(np.isfinite(array).all()):
            raise ValueError(f"{path}: frame {index} is non-numeric or non-finite")
        result.append(array)
    return result


def _validate_metadata(
    path: Path,
    *,
    expected_frames: int,
    expected_heavy: int,
    molecule_name: str,
    pickle_path: Path,
) -> dict[str, object]:
    metadata = _load_metadata(path)
    required = {
        "molecule_name": molecule_name,
        "n_frames": expected_frames,
        "n_heavy_atoms": expected_heavy,
        "coordinate_unit": "angstrom",
        "mapping_used": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": metadata.get(key)}
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{path}: metadata mismatch: {mismatches}")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise ValueError(f"{path}: metadata frames list is not length {expected_frames}")
    recorded_hash = metadata.get("pickle_sha256")
    if recorded_hash is not None and recorded_hash != sha256_file(pickle_path):
        raise ValueError(f"{path}: recorded pickle SHA256 mismatch")
    return metadata


def choose_source(md_root: Path, forcefield: str, clean_name: str) -> tuple[OutputSource | None, list[str]]:
    choices: list[OutputSource] = []
    partial: list[str] = []
    for source in OUTPUT_SOURCES[forcefield]:
        all_pickle = md_root / source.all_dirname / f"{clean_name}.pickle"
        low_pickle = md_root / source.low_dirname / f"{clean_name}.pickle"
        present = (all_pickle.exists(), low_pickle.exists())
        if all(present):
            choices.append(source)
        elif any(present):
            partial.append(source.label)
    return (choices[0] if choices else None), [source.label for source in choices[1:]] + partial


def audit_one(
    md_root: Path,
    forcefield: str,
    manifest_row: dict[str, str],
) -> dict[str, object]:
    clean_name = manifest_row["clean_name"]
    molecule_name = manifest_row["molecule_name"]
    source_class = manifest_row["source_class"]
    expected_heavy = int(manifest_row["n_heavy_atoms"])
    source, alternates_or_partial = choose_source(md_root, forcefield, clean_name)
    result: dict[str, object] = {
        "forcefield": forcefield,
        "clean_name": clean_name,
        "molecule_name": molecule_name,
        "source_class": source_class,
        "status": "missing" if source is None else "unchecked",
        "output_source": source.label if source is not None else "",
        "alternate_or_partial_sources": ";".join(alternates_or_partial),
        "provenance_grade": "missing",
        "message": "no all/low output pair" if source is None else "",
    }
    if source is None:
        for candidate in OUTPUT_SOURCES[forcefield]:
            if candidate.canonical_spec_key is None:
                continue
            spec = SPECS[candidate.canonical_spec_key]
            canonical = paths_for(md_root, clean_name, spec)
            if canonical.commit.exists() or any(path.exists() for path in canonical.artifacts):
                try:
                    validate_commit(
                        md_root,
                        clean_name,
                        expected_molecule_name=molecule_name,
                        expected_source_class=source_class,
                        missing_ok=False,
                        spec=spec,
                    )
                    raise ValueError(
                        "canonical commit validated but the canonical output pair was not discoverable"
                    )
                except Exception as exc:
                    result.update(
                        {
                            "status": "invalid",
                            "provenance_grade": "invalid",
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    return result
        if alternates_or_partial:
            result.update(
                {
                    "status": "invalid",
                    "provenance_grade": "invalid",
                    "message": "partial all/low output source without a complete pair: "
                    + ";".join(alternates_or_partial),
                }
            )
        return result

    all_pickle = md_root / source.all_dirname / f"{clean_name}.pickle"
    low_pickle = md_root / source.low_dirname / f"{clean_name}.pickle"
    all_meta = metadata_path(all_pickle)
    low_meta = metadata_path(low_pickle)
    try:
        for artifact in (all_pickle, all_meta, low_pickle, low_meta):
            if not artifact.is_file():
                raise ValueError(f"required artifact is missing: {artifact}")
        all_frames = _load_conformers(all_pickle, 200, expected_heavy)
        low_frames = _load_conformers(low_pickle, 100, expected_heavy)
        all_metadata = _validate_metadata(
            all_meta,
            expected_frames=200,
            expected_heavy=expected_heavy,
            molecule_name=molecule_name,
            pickle_path=all_pickle,
        )
        low_metadata = _validate_metadata(
            low_meta,
            expected_frames=100,
            expected_heavy=expected_heavy,
            molecule_name=molecule_name,
            pickle_path=low_pickle,
        )
        candidate_frames = {np.ascontiguousarray(frame).tobytes() for frame in all_frames}
        if any(np.ascontiguousarray(frame).tobytes() not in candidate_frames for frame in low_frames):
            raise ValueError("selected100 contains a conformer absent from candidate200")

        provenance_grade = "legacy-fully-loaded"
        if source.canonical_spec_key is not None:
            spec = SPECS[source.canonical_spec_key]
            canonical = paths_for(md_root, clean_name, spec)
            if canonical.commit.is_file():
                commit = validate_commit(
                    md_root,
                    clean_name,
                    expected_molecule_name=molecule_name,
                    expected_source_class=source_class,
                    missing_ok=False,
                    spec=spec,
                )
                assert commit is not None
                provenance_grade = "canonical-transaction"
                result.update(
                    {
                        "canonical_commit": str(canonical.commit.resolve()),
                        "canonical_commit_sha256": sha256_file(canonical.commit),
                        "epoch_id": commit["epoch_id"],
                        "epoch_manifest_sha256": commit["epoch_manifest_sha256"],
                    }
                )
            elif any(
                metadata.get("epoch_id") not in (None, "")
                or metadata.get("artifact_role") not in (None, "")
                for metadata in (all_metadata, low_metadata)
            ):
                raise ValueError(
                    "epoch-bound canonical artifacts exist without a commit marker"
                )
        result.update(
            {
                "status": "valid",
                "provenance_grade": provenance_grade,
                "all_pickle_sha256": sha256_file(all_pickle),
                "all_metadata_sha256": sha256_file(all_meta),
                "low_pickle_sha256": sha256_file(low_pickle),
                "low_metadata_sha256": sha256_file(low_meta),
                "message": "validated by full pickle load and metadata/identity checks",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "invalid",
                "provenance_grade": "invalid",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def enforce_alignment_repair_policy(
    results: list[dict[str, object]],
    job_keys: set[tuple[str, str]],
    approved_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    """Reject repair-gap outputs that bypass approval or canonical commit.

    Fully loaded legacy output remains admissible for the pre-existing
    baseline.  It is never admissible for one of the frozen 55 missing
    forcefield/name pairs.
    """

    enforced: list[dict[str, object]] = []
    for original in results:
        row = dict(original)
        key = (str(row["forcefield"]), str(row["clean_name"]))
        reason = ""
        if key in job_keys and row.get("status") == "valid":
            if key not in approved_keys:
                reason = "alignment repair output is not in the production-approved manifest"
            elif row.get("provenance_grade") != "canonical-transaction":
                reason = "alignment repair output lacks a validated canonical transaction"
        if reason:
            row["rejected_provenance_grade"] = row.get("provenance_grade", "")
            row["status"] = "invalid"
            row["provenance_grade"] = "invalid"
            row["message"] = f"repair-policy rejection: {reason}; {row.get('message', '')}"
        enforced.append(row)
    return enforced


def _coverage_summary(
    manifest_rows: list[dict[str, str]],
    results: list[dict[str, object]],
) -> dict[str, object]:
    target_names = {row["clean_name"] for row in manifest_rows}
    valid_names = {
        forcefield: {
            str(row["clean_name"])
            for row in results
            if row["forcefield"] == forcefield and row["status"] == "valid"
        }
        for forcefield in FORCEFIELDS
    }
    exact_regions: Counter[str] = Counter()
    for clean_name in target_names:
        membership = tuple(
            forcefield for forcefield in FORCEFIELDS if clean_name in valid_names[forcefield]
        )
        exact_regions["+".join(membership) if membership else "none"] += 1
    for label in (
        "none",
        "charmm",
        "gromos",
        "glycam",
        "charmm+gromos",
        "charmm+glycam",
        "gromos+glycam",
        "charmm+gromos+glycam",
    ):
        exact_regions.setdefault(label, 0)

    charmm = valid_names["charmm"]
    gromos = valid_names["gromos"]
    glycam = valid_names["glycam"]
    return {
        "forcefield_valid_counts": {
            forcefield: len(valid_names[forcefield]) for forcefield in FORCEFIELDS
        },
        "pairwise_intersection_counts_inclusive": {
            "charmm+gromos": len(charmm & gromos),
            "charmm+glycam": len(charmm & glycam),
            "gromos+glycam": len(gromos & glycam),
        },
        "all_three_intersection_count": len(charmm & gromos & glycam),
        "union_count": len(charmm | gromos | glycam),
        "exact_region_counts": dict(sorted(exact_regions.items())),
        "uncovered_by_all_forcefields_count": len(
            target_names - (charmm | gromos | glycam)
        ),
    }


def _job_ledger(
    jobs: list[dict[str, str]],
    approved_keys: set[tuple[str, str]],
    results: list[dict[str, object]],
) -> dict[str, object]:
    audit_by_key = {
        (str(row["forcefield"]), str(row["clean_name"])): row for row in results
    }
    ledger: list[dict[str, object]] = []
    for job in sorted(jobs, key=lambda row: int(row["alignment_job_rank"])):
        key = (job["repair_forcefield"], job["clean_name"])
        audited = audit_by_key[key]
        approved = key in approved_keys
        if (
            approved
            and audited["status"] == "valid"
            and audited["provenance_grade"] == "canonical-transaction"
        ):
            state = "formally_committed"
        elif not approved:
            state = "parameterization_blocked"
        elif audited["status"] == "missing":
            state = "production_pending"
        else:
            state = "production_invalid"
        ledger.append(
            {
                "alignment_job_rank": int(job["alignment_job_rank"]),
                "forcefield": key[0],
                "clean_name": key[1],
                "molecule_name": job["molecule_name"],
                "source_class": job["source_class"],
                "production_approved": approved,
                "state": state,
                "audit_status": audited["status"],
                "provenance_grade": audited["provenance_grade"],
                "canonical_commit": audited.get("canonical_commit", ""),
                "canonical_commit_sha256": audited.get("canonical_commit_sha256", ""),
                "epoch_id": audited.get("epoch_id", ""),
                "epoch_manifest_sha256": audited.get("epoch_manifest_sha256", ""),
                "message": audited.get("message", ""),
            }
        )
    committed = [row for row in ledger if row["state"] == "formally_committed"]
    remaining = [row for row in ledger if row["state"] != "formally_committed"]
    by_state = {
        state: [row for row in ledger if row["state"] == state]
        for state in (
            "parameterization_blocked",
            "production_pending",
            "production_invalid",
        )
    }
    return {
        "job_count": len(ledger),
        "formally_committed_count": len(committed),
        "remaining_blocked_or_pending_count": len(remaining),
        "state_counts": dict(sorted(Counter(str(row["state"]) for row in ledger).items())),
        "formally_committed": committed,
        "parameterization_blocked": by_state["parameterization_blocked"],
        "production_pending": by_state["production_pending"],
        "production_invalid": by_state["production_invalid"],
        "remaining_blocked_or_pending": remaining,
        "rows": ledger,
    }


def summarize_results(
    manifest_rows: list[dict[str, str]],
    jobs: list[dict[str, str]],
    approved: list[dict[str, str]],
    results: list[dict[str, object]],
    manifest_integrity: dict[str, object],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, object]:
    status_counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("valid", "missing", "invalid")
    }
    per_forcefield = {
        forcefield: {
            status: sum(
                row["forcefield"] == forcefield and row["status"] == status
                for row in results
            )
            for status in ("valid", "missing", "invalid")
        }
        for forcefield in FORCEFIELDS
    }
    per_forcefield["all"] = status_counts
    provenance_counts = {
        forcefield: {
            "canonical_transaction": sum(
                row["forcefield"] == forcefield
                and row["status"] == "valid"
                and row["provenance_grade"] == "canonical-transaction"
                for row in results
            ),
            "legacy_fully_loaded": sum(
                row["forcefield"] == forcefield
                and row["status"] == "valid"
                and row["provenance_grade"] == "legacy-fully-loaded"
                for row in results
            ),
        }
        for forcefield in FORCEFIELDS
    }
    provenance_counts["all"] = {
        key: sum(provenance_counts[forcefield][key] for forcefield in FORCEFIELDS)
        for key in ("canonical_transaction", "legacy_fully_loaded")
    }
    if any(
        provenance_counts[forcefield]["canonical_transaction"]
        + provenance_counts[forcefield]["legacy_fully_loaded"]
        != per_forcefield[forcefield]["valid"]
        for forcefield in FORCEFIELDS
    ):
        raise ValueError("a valid result has an unrecognized provenance grade")

    coverage = _coverage_summary(manifest_rows, results)
    approved_keys = {(row["repair_forcefield"], row["clean_name"]) for row in approved}
    job_ledger = _job_ledger(jobs, approved_keys, results)
    all_forcefields_complete = all(
        per_forcefield[forcefield]["valid"] == len(manifest_rows)
        for forcefield in FORCEFIELDS
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_integrity["target_manifest"]["observed_sha256"],
        "manifest_integrity": manifest_integrity,
        "target_count": len(manifest_rows),
        "row_count": len(results),
        "counts": per_forcefield,
        "coverage": coverage,
        "provenance_counts": provenance_counts,
        "canonical_transaction_count": provenance_counts["all"]["canonical_transaction"],
        "legacy_fully_loaded_count": provenance_counts["all"]["legacy_fully_loaded"],
        "alignment_jobs": job_ledger,
        "release_invariants": {
            "frozen_manifests_verified": manifest_integrity.get("verified") is True,
            "no_invalid_outputs": status_counts["invalid"] == 0,
            "all_forcefields_have_364": all_forcefields_complete,
            "all_55_repairs_formally_committed": job_ledger["formally_committed_count"] == 55,
        },
    }
    summary["release_ready"] = all(summary["release_invariants"].values())
    return summary


def run_audit(
    manifest_path: Path = DEFAULT_MANIFEST,
    md_root: Path = MD_ROOT,
    jobs_path: Path = DEFAULT_JOBS,
    approved_path: Path = DEFAULT_APPROVED,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest_rows, jobs, approved, integrity = verify_alignment_inputs(
        manifest_path, jobs_path, approved_path
    )
    raw_results = [
        audit_one(md_root, forcefield, row)
        for row in manifest_rows
        for forcefield in FORCEFIELDS
    ]
    end_hashes = {
        "target_manifest": sha256_file(manifest_path),
        "job_manifest": sha256_file(jobs_path),
        "approval_manifest": sha256_file(approved_path),
    }
    start_hashes = {
        label: integrity[label]["observed_sha256"] for label in end_hashes
    }
    if end_hashes != start_hashes:
        raise ValueError(
            f"alignment input changed during audit: start={start_hashes}; end={end_hashes}"
        )
    job_keys = {(row["repair_forcefield"], row["clean_name"]) for row in jobs}
    approved_keys = {(row["repair_forcefield"], row["clean_name"]) for row in approved}
    results = enforce_alignment_repair_policy(raw_results, job_keys, approved_keys)
    summary = summarize_results(
        manifest_rows,
        jobs,
        approved,
        results,
        integrity,
        manifest_path=manifest_path,
    )
    return results, summary


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


def write_audit(output: Path = DEFAULT_OUTPUT) -> tuple[Path, Path, dict[str, object]]:
    rows, summary = run_audit()
    fields = sorted({key for row in rows for key in row})
    with tempfile.TemporaryDirectory() as temporary_name:
        staged = Path(temporary_name) / output.name
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        payload = staged.read_bytes()
    _atomic_write(output, payload)
    summary["audit_csv"] = str(output.resolve())
    summary["audit_csv_sha256"] = hashlib.sha256(payload).hexdigest()

    job_rows = summary["alignment_jobs"]["rows"]
    assert isinstance(job_rows, list) and job_rows
    job_output = output.with_name(f"{output.stem}_jobs.csv")
    job_fields = list(job_rows[0])
    with tempfile.TemporaryDirectory() as temporary_name:
        staged_jobs = Path(temporary_name) / job_output.name
        with staged_jobs.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=job_fields)
            writer.writeheader()
            writer.writerows(job_rows)
        job_payload = staged_jobs.read_bytes()
    _atomic_write(job_output, job_payload)
    summary["alignment_jobs"]["audit_csv"] = str(job_output.resolve())
    summary["alignment_jobs"]["audit_csv_sha256"] = hashlib.sha256(job_payload).hexdigest()

    metadata = output.with_suffix(".json")
    _atomic_write(
        metadata,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return output, metadata, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-invalid", action="store_true")
    args = parser.parse_args()
    csv_path, json_path, summary = write_audit(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(csv_path)
    print(json_path)
    print(summary["alignment_jobs"]["audit_csv"])
    if not args.allow_invalid and summary["counts"]["all"]["invalid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
