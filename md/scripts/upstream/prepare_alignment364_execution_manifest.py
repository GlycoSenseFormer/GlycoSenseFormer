#!/usr/bin/env python3
"""Freeze the production-approved subset of the alignment-364 repair queue."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gromos_curated_gate import load_approved_entry  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
ALIGNMENT_ROOT = MD_ROOT / "md_alignment364_v1"
JOBS = ALIGNMENT_ROOT / "alignment364_jobs.csv"
CHARMM_STATUS = ALIGNMENT_ROOT / "charmm" / "alignment364_charmm_topology_status.json"
GLYCAM_STATUS = ALIGNMENT_ROOT / "glycam" / "alignment364_glycam_status.json"
GROMOS_STATUS = ALIGNMENT_ROOT / "gromos" / "curated_audit.json"
GROMOS_P3_APPROVED = (
    ALIGNMENT_ROOT / "gromos" / "approved_p3_production_manifest_v1.json"
)
GROMOS_P4_ROOT = ALIGNMENT_ROOT / "gromos" / "p1_adman"
GROMOS_P3_STATUS = GROMOS_P4_ROOT / "P3_STATUS.json"
GROMOS_P4_STATUS = GROMOS_P4_ROOT / "P4_STATUS.json"
GROMOS_P4_PREFLIGHT = GROMOS_P4_ROOT / "P4_PREFLIGHT.json"
GROMOS_P4_TEST_RESULTS = GROMOS_P4_ROOT / "P4_TEST_RESULTS.json"
GROMOS_P4_FINALIZER = GROMOS_P4_ROOT / "p4_finalize_status.py"
GROMOS_P4_SETUP_STATUS = (
    GROMOS_P4_ROOT / "p4_formal_setup" / "extended24_gromos_exact_water_5ns_status.json"
)
GROMOS_P4_SETUP_EPOCH = (
    GROMOS_P4_ROOT
    / "p4_formal_setup"
    / "aDMan"
    / "gromos_exact_water_5ns_200cand"
    / "run_input_epoch.json"
)
OUTPUT = ALIGNMENT_ROOT / "alignment364_approved_jobs.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _resolved_record_path(record: object, label: str) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"invalid {label} path record")
    path = Path(record["path"])
    return path if path.is_absolute() else MD_ROOT / path


def _require_bound_file(record: object, expected_path: Path, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} record")
    path = _resolved_record_path(record, label)
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"{label} path mismatch: {path} != {expected_path}")
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or expected_sha != sha256_file(path):
        raise ValueError(f"{label} SHA256 mismatch: {path}")


def _validate_gromos_p4_overlay() -> dict[str, object]:
    """Revalidate the full P3/P4 chain before approving only aDMan."""

    approval = load_json(GROMOS_P3_APPROVED)
    if not isinstance(approval, dict):
        raise ValueError("invalid independent GROMOS P3 approval manifest")
    entries = approval.get("entries")
    if (
        approval.get("schema") != "gromos_p3_approved_production_manifest/v1"
        or approval.get("approved_count") != 1
        or approval.get("explicitly_not_approved_count") != 18
        or not isinstance(entries, list)
        or len(entries) != 1
        or entries[0].get("clean_name") != "aDMan"
        or entries[0].get("repair_forcefield") != "gromos"
    ):
        raise ValueError("GROMOS P3 approval must contain only aDMan and reject the other 18")

    # This revalidates manifest_id, generator, P3 semantics, package identity,
    # every frozen input hash, and the force-field directory tree.
    approved_entry = load_approved_entry(
        "aDMan",
        manifest_path=GROMOS_P3_APPROVED,
        expected_molecule_name="aDMan",
        expected_source_class="mono",
    )

    p4 = load_json(GROMOS_P4_STATUS)
    if not isinstance(p4, dict):
        raise ValueError("invalid aDMan P4 status")
    required = {
        "schema": "adman_gromos_production_promotion_status/v1",
        "status": "CURATED_PRODUCTION_INPUT_AND_FORMAL_SETUP_PASS",
        "molecule": "aDMan",
        "forcefield": "GROMOS 56A6@CARBO-R",
        "curated_production_input_validated": True,
        "eligible_for_formal_alignment_manifest_integration": True,
        "formal_alignment_manifest_integrated": False,
        "production_md_authorized_by_curated_gate": True,
        "formal_setup_ready": True,
        "production_md_launch_state": "WAITING_PARENT_GPU_AND_MANIFEST_COORDINATION",
        "five_ns_started": False,
        "canonical_published": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": p4.get(key)}
        for key, expected in required.items()
        if p4.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"aDMan P4 status is not an unlaunched production approval: {mismatches}")

    p4_approval = p4.get("approval")
    _require_bound_file(p4_approval, GROMOS_P3_APPROVED, "P4 approval manifest")
    if not isinstance(p4_approval, dict) or (
        p4_approval.get("manifest_id") != approved_entry.get("approved_manifest_id")
        or p4_approval.get("approved_count") != 1
        or p4_approval.get("explicitly_not_approved_count") != 18
        or p4_approval.get("approval_basis") != "P3_STATUS_gates"
    ):
        raise ValueError("P4 approval record does not match the validated P3 manifest")

    _require_bound_file(p4.get("p3_status"), GROMOS_P3_STATUS, "P4 P3 status")
    package_record = approved_entry.get("package_manifest")
    package_path = _resolved_record_path(package_record, "approved package manifest")
    _require_bound_file(p4.get("curated_package"), package_path, "P4 curated package")
    if not isinstance(p4.get("curated_package"), dict) or (
        p4["curated_package"].get("package_id")  # type: ignore[index]
        != package_record.get("package_id")  # type: ignore[union-attr]
    ):
        raise ValueError("P4 curated package identity mismatch")

    preflight_record = p4.get("formal_preflight")
    preflight_path = GROMOS_P4_PREFLIGHT
    _require_bound_file(preflight_record, preflight_path, "P4 preflight")
    preflight = load_json(preflight_path)
    if (
        not isinstance(preflight, dict)
        or preflight.get("status") != "PASS_MAXWARN_0"
        or not isinstance(preflight_record, dict)
        or preflight_record.get("grompp_returncode") != 0
        or preflight_record.get("grompp_maxwarn") != 0
        or preflight_record.get("grompp_warning_count") != 0
        or preflight_record.get("atom_count") != 17
        or preflight_record.get("topology_atom_count") != 17
        or preflight_record.get("total_charge_e") != 0.0
    ):
        raise ValueError("aDMan P4 strict formal preflight is incomplete")

    setup = p4.get("formal_setup")
    if not isinstance(setup, dict) or setup.get("status") != "setup_ready":
        raise ValueError("aDMan P4 setup-only status is incomplete")
    setup_status_path = GROMOS_P4_SETUP_STATUS
    epoch_path = GROMOS_P4_SETUP_EPOCH
    if (
        Path(str(setup.get("status_path", ""))).resolve() != setup_status_path.resolve()
        or Path(str(setup.get("run_input_epoch_path", ""))).resolve() != epoch_path.resolve()
        or not setup_status_path.is_file()
        or sha256_file(setup_status_path) != setup.get("status_sha256")
        or not epoch_path.is_file()
        or sha256_file(epoch_path) != setup.get("run_input_epoch_sha256")
    ):
        raise ValueError("aDMan P4 setup evidence SHA256 mismatch")
    epoch = load_json(epoch_path)
    if not isinstance(epoch, dict) or epoch.get("epoch_id") != setup.get("run_input_epoch_id"):
        raise ValueError("aDMan P4 setup epoch identity mismatch")
    epoch_body = {key: value for key, value in epoch.items() if key != "epoch_id"}
    epoch_id = hashlib.sha256(
        (json.dumps(epoch_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    if epoch_id != epoch.get("epoch_id"):
        raise ValueError("aDMan P4 setup epoch digest is invalid")
    implementation = setup.get("implementation_sha256")
    if not isinstance(implementation, dict) or len(implementation) != 6:
        raise ValueError("aDMan P4 implementation hash table is incomplete")
    for filename, expected_sha in implementation.items():
        path = SCRIPT_DIR / str(filename)
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"aDMan P4 implementation changed after setup: {path}")

    tests_record = p4.get("regression_tests")
    tests_path = GROMOS_P4_TEST_RESULTS
    _require_bound_file(tests_record, tests_path, "P4 regression tests")
    tests = load_json(tests_path)
    if (
        not isinstance(tests, dict)
        or tests.get("status") != "PASS"
        or tests.get("test_count") != 42
        or tests.get("expected_test_count") != 42
    ):
        raise ValueError("aDMan P4 regression-test evidence is incomplete")
    suites = tests.get("suite_sha256")
    if not isinstance(suites, dict) or not suites:
        raise ValueError("aDMan P4 regression-test suite hashes are missing")
    for raw_path, expected_sha in suites.items():
        path = Path(str(raw_path))
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"aDMan P4 regression-test source changed: {path}")

    expected_gates = {
        "p3_complete_hash_revalidation": "PASS",
        "immutable_curated_input_package": "PASS",
        "only_adman_approved_other18_rejected": "PASS",
        "strict_formal_preflight": "PASS_MAXWARN_0",
        "formal_setup_epoch": "PASS_SETUP_ONLY",
        "protocol_1fs_5ns_200_candidates": "BOUND_NOT_RUN",
        "canonical_transaction": "TESTED_NOT_PUBLISHED",
        "five_ns_production": "NOT_STARTED_WAITING_COORDINATION",
    }
    if p4.get("gates") != expected_gates:
        raise ValueError("aDMan P4 gate table is incomplete or changed")
    _require_bound_file(p4.get("finalizer"), GROMOS_P4_FINALIZER, "P4 finalizer")

    return {
        "clean_name": "aDMan",
        "approved_manifest_id": approved_entry["approved_manifest_id"],
        "approved_manifest_sha256": sha256_file(GROMOS_P3_APPROVED),
        "p3_status_sha256": sha256_file(GROMOS_P3_STATUS),
        "p4_status_sha256": sha256_file(GROMOS_P4_STATUS),
        "package_id": package_record["package_id"],  # type: ignore[index]
    }


def approval_tables() -> tuple[dict[tuple[str, str], str], dict[str, int], dict[str, object]]:
    charmm = load_json(CHARMM_STATUS)
    glycam = load_json(GLYCAM_STATUS)
    gromos = load_json(GROMOS_STATUS)
    if not isinstance(charmm, dict) or not isinstance(charmm.get("rows"), list):
        raise ValueError("invalid CHARMM status")
    if not isinstance(glycam, dict) or not isinstance(glycam.get("results"), list):
        raise ValueError("invalid GLYCAM status")
    if not isinstance(gromos, list):
        raise ValueError("invalid GROMOS status")

    decisions: dict[tuple[str, str], str] = {}
    for row in charmm["rows"]:
        status = str(row.get("status", ""))
        decisions[("charmm", str(row["clean_name"]))] = status
    for row in glycam["results"]:
        status = str(row.get("final_status", ""))
        decisions[("glycam", str(row["clean_name"]))] = status
    for row in gromos:
        status = "prepared" if row.get("production_ready") is True else str(row.get("status", ""))
        decisions[("gromos", str(row["clean_name"]))] = status

    p4_overlay = _validate_gromos_p4_overlay()
    if decisions.get(("gromos", "aDMan")) == "prepared":
        raise ValueError("aDMan must enter through the P3/P4 overlay, not legacy curated_audit")
    decisions[("gromos", "aDMan")] = "prepared"

    counts = {
        forcefield: sum(
            forcefield == key[0] and status == "prepared"
            for key, status in decisions.items()
        )
        for forcefield in ("charmm", "gromos", "glycam")
    }
    if counts != {"charmm": 10, "gromos": 1, "glycam": 1}:
        raise ValueError(f"unexpected production-approved counts: {counts}")
    return decisions, counts, p4_overlay


def build() -> tuple[list[dict[str, str]], dict[str, object]]:
    with JOBS.open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle))
    if len(jobs) != 55:
        raise ValueError(f"expected 55 alignment jobs, got {len(jobs)}")
    if len({(row["repair_forcefield"], row["clean_name"]) for row in jobs}) != 55:
        raise ValueError("alignment jobs are not unique by forcefield/clean_name")
    decisions, approved_counts, p4_overlay = approval_tables()
    job_keys = {(row["repair_forcefield"], row["clean_name"]) for row in jobs}
    if set(decisions) != job_keys:
        raise ValueError(
            f"parameter audit/job key mismatch: missing={sorted(job_keys-set(decisions))}; "
            f"extra={sorted(set(decisions)-job_keys)}"
        )
    approved = [
        {**row, "parameter_audit_status": decisions[(row["repair_forcefield"], row["clean_name"])]}
        for row in jobs
        if decisions[(row["repair_forcefield"], row["clean_name"])] == "prepared"
    ]
    approved.sort(key=lambda row: (row["repair_forcefield"], int(row["priority"])))
    if len(approved) != 12:
        raise ValueError(f"expected 12 approved jobs, got {len(approved)}")

    # Project the coverage only as a post-run expectation.  Actual release
    # counts still come exclusively from the full output auditor.
    names = {row["clean_name"]: row for row in jobs}
    charmm_names = {row["clean_name"] for row in approved if row["repair_forcefield"] == "charmm"}
    gromos_names = {row["clean_name"] for row in approved if row["repair_forcefield"] == "gromos"}
    glycam_names = {row["clean_name"] for row in approved if row["repair_forcefield"] == "glycam"}
    projected_new_triple = sum(
        (
            (row["baseline_has_charmm"] == "True" or clean in charmm_names)
            and (row["baseline_has_gromos"] == "True" or clean in gromos_names)
            and (row["baseline_has_glycam"] == "True" or clean in glycam_names)
        )
        and not (
            row["baseline_has_charmm"] == "True"
            and row["baseline_has_gromos"] == "True"
            and row["baseline_has_glycam"] == "True"
        )
        for clean, row in names.items()
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_job_count": len(jobs),
        "approved_job_count": len(approved),
        "blocked_job_count": len(jobs) - len(approved),
        "approved_counts": approved_counts,
        "projected_forcefield_counts_after_success": {
            "charmm": 351 + approved_counts["charmm"],
            "gromos": 345 + approved_counts["gromos"],
            "glycam": 341 + approved_counts["glycam"],
        },
        "projected_intersection_after_success": 326 + projected_new_triple,
        "gromos_p4_overlay": p4_overlay,
        "source_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in (
                JOBS,
                CHARMM_STATUS,
                GLYCAM_STATUS,
                GROMOS_STATUS,
                GROMOS_P3_APPROVED,
                GROMOS_P3_STATUS,
                GROMOS_P4_STATUS,
            )
        },
        "release_rule": "Only canonical commits that pass audit_alignment364_outputs.py count as completed MD.",
    }
    return approved, summary


def main() -> None:
    approved, summary = build()
    fields = list(approved[0])
    with tempfile.TemporaryDirectory() as temporary_name:
        staged = Path(temporary_name) / OUTPUT.name
        with staged.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(approved)
        payload = staged.read_bytes()
    _atomic_write(OUTPUT, payload)
    summary["approved_manifest"] = str(OUTPUT.resolve())
    summary["approved_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    metadata = OUTPUT.with_suffix(".json")
    _atomic_write(
        metadata,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
