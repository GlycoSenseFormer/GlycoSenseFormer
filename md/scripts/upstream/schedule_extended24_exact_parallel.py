#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from charmm_canonical_transaction import (
    SPECS as CANONICAL_SPECS,
    validate_commit as validate_canonical_commit,
)
from charmm_standard_metadata import assess_charmm_standard_metadata
from preflight_extended24_exact import (
    locate_inputs,
    preflight_jobs,
    verify_charmm_artifact_provenance,
)


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
SAMPLE_CSV = Path(os.environ.get("MD_SAMPLE_CSV", MD_ROOT / "md_small_sample_dataset_v1" / "md_small_sample_glycans_v1.csv"))
WORKER = MD_ROOT / "scripts" / "run_extended24_exact_single_no_status.py"


OUTPUT_DIRS = {
    "glycam": Path(
        os.environ.get(
            "MD_GLYCAM_LOW_OUTPUT_DIR",
            MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1",
        )
    ),
    "charmm": MD_ROOT / "conformations_md_charmm36_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
    "gromos": MD_ROOT / "conformations_md_gromos_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
}

ALL_OUTPUT_DIRS = {
    "glycam": Path(
        os.environ.get(
            "MD_GLYCAM_ALL_OUTPUT_DIR",
            MD_ROOT / "conformations_md_glycam06j_water_5ns_200cand_all_seed1",
        )
    ),
    "charmm": MD_ROOT / "conformations_md_charmm36_exact_water_5ns_200cand_all_seed1",
    "gromos": MD_ROOT / "conformations_md_gromos_exact_water_5ns_200cand_all_seed1",
}


def read_rows() -> list[dict[str, str]]:
    with SAMPLE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def gromos_has_topology(row: dict[str, str]) -> bool:
    stem = row["molecule_name"].replace("/", "_")
    dryrun = RUN_ROOT / row["clean_name"] / "gromos_exact_topology_dryrun"
    required = [
        dryrun / f"{stem}_GROMOS_EXACT_processed.gro",
        dryrun / f"{stem}_GROMOS_EXACT.top",
        dryrun / "gromos53a6carbo_exact.ff",
        dryrun / "sdf_to_gromos_frame_heavy_index_map.json",
    ]
    return all(path.exists() for path in required)


def glycam_has_topology(row: dict[str, str]) -> bool:
    clean = row["clean_name"]
    dryrun = RUN_ROOT / clean / "glycam06j_topology_dryrun"
    acpype = dryrun / f"{clean}_GLYCAM06j.amb2gmx"
    required = [
        acpype / f"{clean}_GLYCAM06j_GMX.gro",
        acpype / f"{clean}_GLYCAM06j_GMX.top",
        acpype / f"{clean}_GLYCAM06j_GMX_tleap_raw.gro",
        dryrun / "sdf_to_glycam_frame_heavy_index_map.json",
        dryrun / "glycam_coordinate_repair_metadata.json",
    ]
    return all(path.exists() for path in required)


def charmm_has_topology(row: dict[str, str]) -> bool:
    stem = row["molecule_name"].replace("/", "_")
    dryrun = RUN_ROOT / row["clean_name"] / "charmm36_standard_topology_dryrun"
    required = [
        dryrun / f"{stem}_CHARMM36_STANDARD_processed.gro",
        dryrun / f"{stem}_CHARMM36_STANDARD.top",
        dryrun / "charmm36_standard.ff",
        dryrun / "sdf_to_charmm_frame_heavy_index_map.json",
        dryrun / "charmm36_standard_input_metadata.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads(required[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    policy = assess_charmm_standard_metadata(
        metadata,
        expected_molecule_name=row["molecule_name"],
        expected_clean_name=row["clean_name"],
        expected_source_class=row.get("source_class"),
    )
    if not bool(policy["approved"]):
        return False
    inputs = locate_inputs("charmm", row, RUN_ROOT)
    return bool(verify_charmm_artifact_provenance(inputs, metadata)["approved"])


def output_exists(forcefield: str, clean: str, row: dict[str, str] | None = None) -> bool:
    return (
        validate_canonical_commit(
            MD_ROOT,
            clean,
            expected_molecule_name=row["molecule_name"] if row is not None else None,
            expected_source_class=row["source_class"] if row is not None else None,
            spec=CANONICAL_SPECS[forcefield],
        )
        is not None
    )


def build_queue(forcefields: list[str]) -> list[tuple[str, dict[str, str]]]:
    rows = read_rows()
    jobs: list[tuple[str, dict[str, str]]] = []
    for forcefield in forcefields:
        for row in rows:
            repair_forcefield = row.get("repair_forcefield", "").strip()
            if repair_forcefield and repair_forcefield != forcefield:
                continue
            clean = row["clean_name"]
            if output_exists(forcefield, clean, row):
                continue
            if forcefield == "glycam" and not glycam_has_topology(row):
                continue
            if forcefield == "charmm" and not charmm_has_topology(row):
                continue
            if forcefield == "gromos" and not gromos_has_topology(row):
                continue
            jobs.append((forcefield, row))
    return jobs


def count_outputs(forcefield: str) -> int:
    transaction_dir = MD_ROOT / CANONICAL_SPECS[forcefield].transaction_dirname
    return len(list(transaction_dir.glob("*.commit.json"))) if transaction_dir.exists() else 0


def write_state(
    state_path: Path,
    *,
    queue: list[tuple[str, dict[str, str]]],
    running: dict[str, dict[str, object]],
    completed: list[dict[str, object]],
    failed: list[dict[str, object]],
    max_parallel: int,
    started_at: str,
    device: str,
    halted_reason: str | None,
    preflight_report: str,
) -> None:
    if halted_reason:
        status = "halted"
    elif running:
        status = "running"
    elif queue:
        status = "pending"
    elif failed:
        status = "finished_with_failures"
    else:
        status = "complete"
    state = {
        "status": status,
        "halted_reason": halted_reason,
        "started_at": started_at,
        "updated_at": time.strftime("%F %T %Z"),
        "max_parallel": max_parallel,
        "device": device,
        "preflight_report": preflight_report,
        "pending": [
            {"forcefield": forcefield, "clean_name": row["clean_name"], "priority": int(row["priority"])}
            for forcefield, row in queue
        ],
        "running": list(running.values()),
        "completed": completed,
        "failed": failed,
        "counts": {
            "glycam_low100": count_outputs("glycam"),
            "charmm_low100": count_outputs("charmm"),
            "gromos_low100": count_outputs("gromos"),
        },
    }
    temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, state_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preflight-gated extended-24 exact MD jobs.")
    parser.add_argument("--forcefields", default="glycam,charmm,gromos")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--allow-shared-gpu-parallel",
        action="store_true",
        help="Explicitly acknowledge the risk of running more than one job on a single GPU.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict the queue to this clean_name or molecule_name; repeat as needed.",
    )
    args = parser.parse_args()

    def request_shutdown(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_shutdown)

    forcefields = [item.strip() for item in args.forcefields.split(",") if item.strip()]
    invalid = sorted(set(forcefields) - {"glycam", "charmm", "gromos"})
    if invalid:
        parser.error(f"invalid forcefields: {', '.join(invalid)}")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if args.device == "gpu" and args.max_parallel > 1 and not args.allow_shared_gpu_parallel:
        parser.error("GPU mode requires --max-parallel 1 unless --allow-shared-gpu-parallel is explicit")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    if "charmm" in forcefields:
        from run_extended24_charmm_exact_water_5ns import assert_formal_environment

        try:
            assert_formal_environment()
        except RuntimeError as exc:
            parser.error(str(exc))
        if args.device == "gpu" and args.max_parallel != 1:
            parser.error(
                "formal CHARMM GPU production requires --max-parallel 1 until "
                "workers have an explicit one-worker/one-GPU mapping"
            )

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (RUN_ROOT / ".extended24_exact_scheduler.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another extended-24 scheduler already holds the run-root lock")

    log_dir = RUN_ROOT / f"parallel_exact_{args.run_id}"
    if log_dir.exists():
        parser.error(f"refusing to reuse existing run directory: {log_dir}")
    log_dir.mkdir(parents=True)
    state_path = log_dir / "state.json"
    started_at = time.strftime("%F %T %Z")

    queue = build_queue(forcefields)
    if args.only:
        selected = set(args.only)
        matched = {
            value
            for _forcefield, row in queue
            for value in (row["clean_name"], row["molecule_name"])
            if value in selected
        }
        unmatched = sorted(selected - matched)
        if unmatched:
            parser.error(
                "--only entries are absent, already complete, or lack an approved topology: "
                + ", ".join(unmatched)
            )
        queue = [
            (forcefield, row)
            for forcefield, row in queue
            if row["clean_name"] in selected or row["molecule_name"] in selected
        ]
    preflight_results, preflight_paths = preflight_jobs(queue, log_dir / "preflight")
    blocked = [item for item in preflight_results if not bool(item["passed"])]
    preflight_report = str(preflight_paths["json"])
    running: dict[str, dict[str, object]] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    completed: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    halted_reason: str | None = None

    def persist_state() -> None:
        write_state(
            state_path,
            queue=queue,
            running=running,
            completed=completed,
            failed=failed,
            max_parallel=args.max_parallel,
            started_at=started_at,
            device=args.device,
            halted_reason=halted_reason,
            preflight_report=preflight_report,
        )

    if blocked:
        halted_reason = f"preflight blocked {len(blocked)} of {len(preflight_results)} jobs; no MD was started"
        persist_state()
        print(halted_reason, file=sys.stderr, flush=True)
        raise SystemExit(2)
    if args.preflight_only:
        halted_reason = "preflight-only requested; no MD was started"
        persist_state()
        print(f"preflight passed for all {len(preflight_results)} jobs", flush=True)
        return

    def start_next() -> None:
        if halted_reason:
            return
        while queue and len(processes) < args.max_parallel:
            forcefield, row = queue.pop(0)
            clean = row["clean_name"]
            job_id = f"{forcefield}_{int(row['priority']):02d}_{clean}"
            log_path = log_dir / f"{job_id}.log"
            result_path = log_dir / f"{job_id}.json"
            cmd = [
                sys.executable,
                str(WORKER),
                "--forcefield",
                forcefield,
                "--only",
                clean,
                "--result-json",
                str(result_path),
            ]
            handle = log_path.open("wb")
            worker_env = os.environ.copy()
            worker_env["MD_DEVICE"] = args.device
            worker_env["MD_PREFLIGHT_PASSED"] = "1"
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=worker_env,
            )
            handle.close()
            processes[job_id] = proc
            running[job_id] = {
                "job_id": job_id,
                "pid": proc.pid,
                "forcefield": forcefield,
                "clean_name": clean,
                "molecule_name": row["molecule_name"],
                "priority": int(row["priority"]),
                "log": str(log_path),
                "result_json": str(result_path),
                "started_at": time.strftime("%F %T %Z"),
            }

    start_next()
    persist_state()

    try:
        while processes:
            time.sleep(args.poll_seconds)
            for job_id, proc in list(processes.items()):
                code = proc.poll()
                if code is None:
                    continue
                info = running.pop(job_id)
                info["returncode"] = code
                info["finished_at"] = time.strftime("%F %T %Z")
                result_path = Path(str(info["result_json"]))
                if result_path.exists():
                    try:
                        info["result"] = json.loads(result_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        info["result"] = {"status": "invalid_result_json"}
                result = info.get("result")
                successful_result = (
                    isinstance(result, dict)
                    and result.get("status") in {"done", "done_existing"}
                    and result.get("clean_name") == info["clean_name"]
                    and result.get("forcefield") == info["forcefield"]
                )
                if code == 0 and successful_result:
                    completed.append(info)
                else:
                    if code == 0:
                        info["returncode"] = 86
                        info["error_kind"] = "invalid_worker_success_result"
                    failed.append(info)
                    if not args.continue_on_error and halted_reason is None:
                        halted_reason = (
                            f"fail-fast after {job_id} returned {code} "
                            "without a validated successful result"
                        )
                processes.pop(job_id)
            start_next()
            persist_state()
    except KeyboardInterrupt:
        halted_reason = "scheduler interrupted; active process groups were sent SIGINT"
        for proc in processes.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        for proc in processes.values():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for job_id, proc in list(processes.items()):
            info = running.pop(job_id)
            info["returncode"] = proc.poll()
            info["finished_at"] = time.strftime("%F %T %Z")
            info["error_kind"] = "scheduler_interrupted"
            result_path = Path(str(info["result_json"]))
            if result_path.exists():
                try:
                    info["result"] = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    info["result"] = {"status": "invalid_result_json"}
            failed.append(info)
            processes.pop(job_id)
        raise SystemExit(130)
    finally:
        persist_state()

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
