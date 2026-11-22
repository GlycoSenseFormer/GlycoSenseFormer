#!/usr/bin/env python3
"""Wait for CHARMM, then smoke and run corrected GLYCAM v15 queues serially."""

from __future__ import annotations

import json
import os
import fcntl
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
RUN_ROOT = MD_ROOT / "md_runs_glycam_enantiomer_v15_safe"
CHARMM_STATE = (
    MD_ROOT
    / "md_runs_all_remaining_v14_safe"
    / "parallel_exact_charmm_standard_v14_safe_resume1_20260712"
    / "state.json"
)
SCOPE_CSV = (
    MD_ROOT
    / "md_runs_all_remaining_v14_safe"
    / "audit_glycam_enantiomer_scope_v1"
    / "glycam_enantiomer_rerun_167.csv"
)
CURRENT_CSV = (
    MD_ROOT
    / "md_all_remaining_dataset_v13"
    / "md_all_remaining_glycans_v13.csv"
)
ALL_OUTPUT_DIR = MD_ROOT / "conformations_md_glycam06j_enantiomer_v15_all_seed1"
LOW_OUTPUT_DIR = MD_ROOT / "conformations_md_glycam06j_enantiomer_v15_lowest100_seed1"
GMX = "/soft/gromacs-2026.1/bin/gmx"
CHAIN_STATE = RUN_ROOT / "post_charmm_glycam_v15_chain_state.json"
CHAIN_LOG_DIR = RUN_ROOT / "post_charmm_chain_logs"
CHAIN_LOCK = RUN_ROOT / ".post_charmm_glycam_v15_chain.lock"
SMOKE_RESULT = RUN_ROOT / "smoke_gpu_corrected_enantiomer_full_v1.json"
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")
GPU_IDLE_POLL_SECONDS = int(os.environ.get("GPU_IDLE_POLL_SECONDS", "30"))
ALLOW_SHARED_GPU = os.environ.get("ALLOW_SHARED_GPU", "0") == "1"


def write_state(status: str, **extra: object) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "updated_at": time.strftime("%F %T %Z"),
        "charmm_state": str(CHARMM_STATE),
        "run_root": str(RUN_ROOT),
        "all_output_dir": str(ALL_OUTPUT_DIR),
        "low_output_dir": str(LOW_OUTPUT_DIR),
        **extra,
    }
    temporary = CHAIN_STATE.with_suffix(CHAIN_STATE.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, CHAIN_STATE)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def wait_for_charmm() -> None:
    while True:
        if not CHARMM_STATE.exists():
            write_state("waiting_for_charmm_state")
            time.sleep(30)
            continue
        state = read_json(CHARMM_STATE)
        status = str(state.get("status", ""))
        if status == "complete":
            write_state("charmm_complete")
            return
        if status in {"halted", "finished_with_failures"}:
            raise RuntimeError(
                "CHARMM prerequisite did not complete cleanly: "
                f"status={status}, reason={state.get('halted_reason')}"
            )
        write_state(
            "waiting_for_charmm",
            charmm_status=status,
            charmm_pending=len(state.get("pending", [])),
            charmm_running=len(state.get("running", [])),
            charmm_completed=len(state.get("completed", [])),
            charmm_failed=len(state.get("failed", [])),
        )
        time.sleep(30)


def common_environment(sample_csv: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MD_RUN_ROOT": str(RUN_ROOT),
            "MD_SAMPLE_CSV": str(sample_csv),
            "MD_GLYCAM_ALL_OUTPUT_DIR": str(ALL_OUTPUT_DIR),
            "MD_GLYCAM_LOW_OUTPUT_DIR": str(LOW_OUTPUT_DIR),
            "GMX": GMX,
            "GMX_CPT_MINUTES": "1",
        }
    )
    return environment


def run_logged(name: str, command: list[str], environment: dict[str, str]) -> None:
    CHAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHAIN_LOG_DIR / f"{name}.log"
    write_state("running_stage", stage=name, log=str(log_path), command=command)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"stage {name} returned {process.returncode}; see {log_path}")


def smoke_passed() -> bool:
    if not SMOKE_RESULT.exists():
        return False
    try:
        return read_json(SMOKE_RESULT).get("status") == "smoke_passed"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def scheduler_complete(run_id: str) -> bool:
    state_path = RUN_ROOT / f"parallel_exact_{run_id}" / "state.json"
    if not state_path.exists():
        return False
    return read_json(state_path).get("status") == "complete"


def scheduler_run_ids(base_run_id: str) -> list[str]:
    """Return the base scheduler run and numbered resume runs in order."""
    run_ids: list[tuple[int, str]] = []
    base_dir = RUN_ROOT / f"parallel_exact_{base_run_id}"
    if base_dir.is_dir():
        run_ids.append((0, base_run_id))
    for path in RUN_ROOT.glob(f"parallel_exact_{base_run_id}_resume*"):
        suffix = path.name.removeprefix(f"parallel_exact_{base_run_id}_resume")
        if path.is_dir() and suffix.isdigit():
            run_ids.append((int(suffix), f"{base_run_id}_resume{suffix}"))
    return [run_id for _number, run_id in sorted(run_ids)]


def completed_scheduler_run_id(base_run_id: str) -> str | None:
    for run_id in scheduler_run_ids(base_run_id):
        if scheduler_complete(run_id):
            return run_id
    return None


def next_scheduler_run_id(base_run_id: str) -> str:
    existing = set(scheduler_run_ids(base_run_id))
    if base_run_id not in existing:
        return base_run_id
    resume_number = 1
    while f"{base_run_id}_resume{resume_number}" in existing:
        resume_number += 1
    return f"{base_run_id}_resume{resume_number}"


def gpu_compute_processes() -> list[str]:
    completed = subprocess.run(
        [
            NVIDIA_SMI,
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot establish GPU availability with {NVIDIA_SMI}: "
            f"{completed.stdout.strip()}"
        )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def wait_for_gpu_idle(stage: str) -> None:
    if GPU_IDLE_POLL_SECONDS < 1:
        raise ValueError("GPU_IDLE_POLL_SECONDS must be at least 1")
    while True:
        processes = gpu_compute_processes()
        if processes and ALLOW_SHARED_GPU:
            write_state(
                "shared_gpu_start",
                next_stage=stage,
                existing_gpu_compute_processes=processes,
            )
            return
        if not processes:
            write_state("gpu_idle", next_stage=stage)
            return
        write_state(
            "waiting_for_gpu_idle",
            next_stage=stage,
            gpu_compute_processes=processes,
            poll_seconds=GPU_IDLE_POLL_SECONDS,
        )
        time.sleep(GPU_IDLE_POLL_SECONDS)


def main() -> None:
    if not Path(GMX).is_file():
        raise FileNotFoundError(GMX)
    for path in (SCOPE_CSV, CURRENT_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = CHAIN_LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another GLYCAM v15 chain process is already active") from exc

    try:
        wait_for_charmm()

        if not smoke_passed():
            smoke_environment = common_environment(CURRENT_CSV)
            smoke_environment.update(
                {
                    "MD_NSTEPS_EM": "5000",
                    "MD_NSTEPS_NVT": "1000",
                    "MD_NSTEPS_NPT": "1000",
                    "MD_NSTEPS_MD": "1000",
                }
            )
            wait_for_gpu_idle("corrected_gpu_smoke")
            run_logged(
                "corrected_gpu_smoke",
                [
                    sys.executable,
                    str(MD_ROOT / "scripts" / "smoke_extended24_water.py"),
                    "--forcefield",
                    "glycam",
                    "--only",
                    "aLRha12[aLRha13]aLRhaOMe",
                    "--through",
                    "md",
                    "--device",
                    "gpu",
                    "--variant",
                    "_smoke_gpu_corrected_enantiomer_full_v15_1",
                    "--result-json",
                    str(SMOKE_RESULT),
                ],
                smoke_environment,
            )

        scope_base_run_id = "glycam_enantiomer_scope166_v15_20260712"
        scope_run_id = completed_scheduler_run_id(scope_base_run_id)
        if scope_run_id is None:
            scope_run_id = next_scheduler_run_id(scope_base_run_id)
            wait_for_gpu_idle("enantiomer_scope166")
            run_logged(
                "enantiomer_scope166",
                [
                    sys.executable,
                    str(MD_ROOT / "scripts" / "schedule_extended24_exact_parallel.py"),
                    "--forcefields",
                    "glycam",
                    "--max-parallel",
                    "1",
                    "--device",
                    "gpu",
                    "--poll-seconds",
                    "10",
                    "--run-id",
                    scope_run_id,
                ],
                common_environment(SCOPE_CSV),
            )

        current_base_run_id = "glycam_current_remaining11_v15_20260712"
        current_run_id = completed_scheduler_run_id(current_base_run_id)
        if current_run_id is None:
            current_run_id = next_scheduler_run_id(current_base_run_id)
            wait_for_gpu_idle("current_remaining11")
            run_logged(
                "current_remaining11",
                [
                    sys.executable,
                    str(MD_ROOT / "scripts" / "schedule_extended24_exact_parallel.py"),
                    "--forcefields",
                    "glycam",
                    "--max-parallel",
                    "1",
                    "--device",
                    "gpu",
                    "--poll-seconds",
                    "10",
                    "--run-id",
                    current_run_id,
                ],
                common_environment(CURRENT_CSV),
            )

        write_state(
            "complete",
            smoke_result=str(SMOKE_RESULT),
            scope_run_id=scope_run_id,
            current_run_id=current_run_id,
        )
    except Exception as exc:
        write_state("blocked", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
