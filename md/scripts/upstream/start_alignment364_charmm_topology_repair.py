#!/usr/bin/env python3
"""Build and strictly preflight the frozen alignment-364 CHARMM-missing set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import preflight_extended24_exact as preflight
from prepare_charmm_alignment364_standard_input import (
    DEFAULT_ALIGNMENT_MANIFEST,
    DEFAULT_GMX,
    DEFAULT_MANIFEST,
    DEFAULT_RUN_ROOT,
    DEFAULT_SOURCE_FF,
    OUTPUT_SUBDIR,
    TARGET_BY_CLEAN,
    build,
    load_alignment_rows,
)


STATUS_STEM = "alignment364_charmm_topology_status"


def write_status(run_root: Path, rows: list[dict[str, object]], target_count: int) -> None:
    summary = {
        "generated_at": time.strftime("%F %T %Z"),
        "run_root": str(run_root),
        "target_count": target_count,
        "prepared": sum(row["status"] == "prepared" for row in rows),
        "blocked": sum(row["status"] == "blocked" for row in rows),
        "pending": target_count - len(rows),
        "water_5ns_started": False,
        "rows": rows,
    }
    json_path = run_root / f"{STATUS_STEM}.json"
    csv_path = run_root / f"{STATUS_STEM}.csv"
    md_path = run_root / f"{STATUS_STEM}.md"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, json_path)
    fields = [
        "priority",
        "alignment_priority",
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
        "# Alignment-364 CHARMM topology status",
        "",
        f"- Run root: `{run_root}`",
        f"- Prepared: {summary['prepared']}/{target_count}",
        f"- Blocked: {summary['blocked']}",
        "- 5 ns production MD started: no",
        "",
        "| priority | clean name | status | charge (e) | neutralize |",
        "|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | `{row['clean_name']}` | {row['status']} | "
            f"{row.get('charge_e', '')} | {row.get('requires_neutralization', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--source-ff", default=str(DEFAULT_SOURCE_FF))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--alignment-manifest", default=str(DEFAULT_ALIGNMENT_MANIFEST))
    parser.add_argument("--gmx", default=DEFAULT_GMX)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    alignment_rows = load_alignment_rows(Path(args.alignment_manifest))
    rows = sorted(alignment_rows.values(), key=lambda row: int(row["priority"]))
    if args.only:
        selected = set(args.only)
        known = {
            value
            for row in rows
            for value in (row["clean_name"], row["molecule_name"])
        }
        unknown = sorted(selected - known)
        if unknown:
            parser.error("unknown alignment364 CHARMM target(s): " + ", ".join(unknown))
        rows = [
            row
            for row in rows
            if row["clean_name"] in selected or row["molecule_name"] in selected
        ]

    preflight.GMX = args.gmx
    status_rows: list[dict[str, object]] = []
    failures = 0
    for row in rows:
        target = TARGET_BY_CLEAN[row["clean_name"]]
        output_dir = run_root / target.clean_name / OUTPUT_SUBDIR
        status: dict[str, object] = {
            "priority": int(row["priority"]),
            "alignment_priority": int(row["alignment_priority"]),
            "clean_name": target.clean_name,
            "molecule_name": target.raw_name,
            "source_class": target.source_class,
            "status": "blocked",
            "charge_e": "",
            "requires_neutralization": "",
            "output_dir": str(output_dir),
            "error": "",
        }
        try:
            if not output_dir.exists():
                build(
                    argparse.Namespace(
                        molecule_name=target.raw_name,
                        sdf="",
                        run_root=str(run_root),
                        source_ff=args.source_ff,
                        manifest=args.manifest,
                        alignment_manifest=args.alignment_manifest,
                        gmx=args.gmx,
                        seed=args.seed,
                        timeout_seconds=args.timeout_seconds,
                    )
                )
            report = preflight.preflight_job("charmm", row, run_root)
            if not report["passed"]:
                raise RuntimeError("; ".join(str(error) for error in report["errors"]))
            status.update(
                {
                    "status": "prepared",
                    "charge_e": report["total_charge"],
                    "requires_neutralization": report["requires_neutralization"],
                }
            )
            print(f"[prepared] {target.clean_name}", flush=True)
        except Exception as exc:
            failures += 1
            status["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[blocked] {target.clean_name}: {exc}", file=sys.stderr, flush=True)
        status_rows.append(status)
        write_status(run_root, status_rows, len(rows))
    if failures:
        raise SystemExit(f"{failures} alignment364 CHARMM target(s) blocked")


if __name__ == "__main__":
    main()
