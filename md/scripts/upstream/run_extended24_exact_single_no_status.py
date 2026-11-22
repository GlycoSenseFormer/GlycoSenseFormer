#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))


def load_runner(forcefield: str):
    if str(MD_ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(MD_ROOT / "scripts"))
    if forcefield == "charmm":
        import run_extended24_charmm_exact_water_5ns as runner

        return runner
    if forcefield == "gromos":
        import run_extended24_gromos_exact_water_5ns as runner

        return runner
    if forcefield == "glycam":
        import run_extended24_glycam_water_5ns as runner

        return runner
    raise ValueError(forcefield)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one extended-24 exact MD job without shared status writes.")
    parser.add_argument("--forcefield", choices=["glycam", "charmm", "gromos"], required=True)
    parser.add_argument("--only", required=True, help="clean_name or molecule_name")
    parser.add_argument("--result-json", default="", help="Optional per-job result JSON path.")
    args = parser.parse_args()

    started = time.time()
    runner = load_runner(args.forcefield)
    rows = [
        row
        for row in runner.read_rows()
        if row["clean_name"] == args.only or row["molecule_name"] == args.only
    ]
    if not rows:
        raise SystemExit(f"No sample matched --only {args.only!r}")
    if len(rows) != 1:
        matches = ", ".join(
            f"{row['clean_name']} ({row['molecule_name']})" for row in rows
        )
        raise SystemExit(f"Ambiguous --only {args.only!r}; matched {len(rows)} rows: {matches}")

    try:
        result = runner.process_row(rows[0], setup_only=False)
    except Exception as exc:
        from run_water_5ns_low_sugar_energy import GpuFatalError

        result = {
            "forcefield": args.forcefield,
            "clean_name": rows[0]["clean_name"],
            "molecule_name": rows[0]["molecule_name"],
            "status": "failed",
            "message": repr(exc),
            "error_kind": "gpu_fatal" if isinstance(exc, GpuFatalError) else "job_failure",
            "traceback": traceback.format_exc(),
        }
    result["forcefield"] = args.forcefield
    result["wall_seconds_total"] = round(time.time() - started, 2)

    if args.result_json:
        result_path = Path(args.result_json)
    else:
        result_path = RUN_ROOT / f"{args.forcefield}_{rows[0]['clean_name']}_single_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temp_result_path = result_path.with_suffix(result_path.suffix + ".tmp")
    temp_result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_result_path, result_path)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if result.get("status") not in {"done", "done_existing"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
