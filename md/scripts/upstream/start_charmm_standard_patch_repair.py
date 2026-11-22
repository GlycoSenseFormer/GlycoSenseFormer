#!/usr/bin/env python3
"""Build and independently preflight the hardened CHARMM standard-patch 30."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import preflight_extended24_exact as preflight
from prepare_charmm_standard_patch_input import (
    DEFAULT_GMX,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOT,
    DEFAULT_SOURCE_FF,
    TARGETS,
    build,
)


STATUS_STEM = "charmm30_hardened_topology_status"


def row_for(target: object, priority: int) -> dict[str, str]:
    return {
        "priority": str(priority),
        "molecule_name": target.raw_name,
        "clean_name": target.clean_name,
        "source_class": target.source_class,
    }


def write_status(run_root: Path, rows: list[dict[str, object]]) -> None:
    summary = {
        "generated_at": time.strftime("%F %T %Z"),
        "run_root": str(run_root),
        "target_count": len(TARGETS),
        "completed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "pending": len(TARGETS) - len(rows),
        "rows": rows,
    }
    json_path = run_root / f"{STATUS_STEM}.json"
    csv_path = run_root / f"{STATUS_STEM}.csv"
    md_path = run_root / f"{STATUS_STEM}.md"
    temp = json_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, json_path)
    fields = [
        "priority",
        "clean_name",
        "molecule_name",
        "source_class",
        "status",
        "charge_e",
        "requires_neutralization",
        "output_dir",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = [
        "# CHARMM30 hardened topology status",
        "",
        f"- Run root: `{run_root}`",
        f"- Passed: {summary['completed']}/{len(TARGETS)}",
        f"- Failed: {summary['failed']}",
        f"- Pending: {summary['pending']}",
        "",
        "| priority | molecule | status | charge (e) | neutralize |",
        "|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | {row['clean_name']} | {row['status']} | "
            f"{row.get('charge_e', '')} | {row.get('requires_neutralization', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--source-ff", default=str(DEFAULT_SOURCE_FF))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gmx", default=DEFAULT_GMX)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Build only this raw or clean target name (repeatable).",
    )
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    preflight.GMX = args.gmx
    selected = [
        target
        for target in TARGETS.values()
        if not args.only
        or target.raw_name in args.only
        or target.clean_name in args.only
    ]
    if args.only and len(selected) != len(set(args.only)):
        unknown = sorted(
            set(args.only)
            - {target.raw_name for target in selected}
            - {target.clean_name for target in selected}
        )
        if unknown:
            parser.error("unknown --only target(s): " + ", ".join(unknown))

    status_rows: list[dict[str, object]] = []
    failures = 0
    for priority, target in enumerate(TARGETS.values(), start=1):
        if target not in selected:
            continue
        queue_row = row_for(target, priority)
        output_dir = run_root / target.clean_name / "charmm36_standard_topology_dryrun"
        status: dict[str, object] = {
            **queue_row,
            "output_dir": str(output_dir),
            "status": "failed",
            "error": "",
        }
        try:
            if not output_dir.exists():
                build_args = argparse.Namespace(
                    molecule_name=target.raw_name,
                    sdf="",
                    run_root=str(run_root),
                    source_ff=args.source_ff,
                    manifest=args.manifest,
                    gmx=args.gmx,
                    seed=args.seed,
                    timeout_seconds=args.timeout_seconds,
                )
                build(build_args)
            report = preflight.preflight_job("charmm", queue_row, run_root)
            if not report["passed"]:
                raise RuntimeError("; ".join(str(error) for error in report["errors"]))
            status.update(
                {
                    "status": "passed",
                    "charge_e": report["total_charge"],
                    "requires_neutralization": report["requires_neutralization"],
                }
            )
            print(f"[passed] {priority:02d}/30 {target.clean_name}", flush=True)
        except Exception as exc:
            failures += 1
            status["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[failed] {priority:02d}/30 {target.clean_name}: {exc}", file=sys.stderr, flush=True)
        status_rows.append(status)
        write_status(run_root, status_rows)
    if failures:
        raise SystemExit(f"{failures} CHARMM standard-patch target(s) failed")


if __name__ == "__main__":
    main()
