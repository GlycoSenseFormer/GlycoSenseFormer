#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
RUN_ROOT = MD_ROOT / "md_runs_extended24"

RUNNERS = {
    "charmm": MD_ROOT / "scripts" / "run_extended24_charmm_exact_water_5ns.py",
    "gromos": MD_ROOT / "scripts" / "run_extended24_gromos_exact_water_5ns.py",
}


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pids(pids: list[int], poll_seconds: int) -> None:
    remaining = set(pids)
    while remaining:
        finished = {pid for pid in remaining if not process_exists(pid)}
        remaining -= finished
        if remaining:
            print(f"[queue] waiting for active pids: {sorted(remaining)}", flush=True)
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue exact extended-24 MD after current jobs finish.")
    parser.add_argument("--forcefield", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.wait_pid:
        wait_for_pids(args.wait_pid, args.poll_seconds)

    runner = RUNNERS[args.forcefield]
    print(f"[queue] starting remaining {args.forcefield} jobs with {runner}", flush=True)
    completed = subprocess.run([sys.executable, str(runner)], cwd=ROOT)
    print(f"[queue] {args.forcefield} runner exited with {completed.returncode}", flush=True)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
