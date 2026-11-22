#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
SAMPLE_CSV = Path(os.environ.get("MD_SAMPLE_CSV", MD_ROOT / "md_small_sample_dataset_v1" / "md_small_sample_glycans_v1.csv"))
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
SOURCE_FF = MD_ROOT / "forcefield_assets" / "gromos56a6carbo" / "gromos53a6carbo.ff"
PREPARE = MD_ROOT / "scripts" / "prepare_gromos_exact_input.py"
STATUS_CSV = RUN_ROOT / "extended24_gromos_exact_topology_status.csv"
STATUS_JSON = RUN_ROOT / "extended24_gromos_exact_topology_status.json"
STATUS_MD = RUN_ROOT / "extended24_gromos_exact_topology_status.md"
SDF_BY_CLASS = {
    "mono": ROOT / "data" / "dataset" / "casper_mono.sdf",
    "dimer": ROOT / "data" / "dataset" / "casper_dimer.sdf",
    "trimer": ROOT / "data" / "dataset" / "casper_trimer.sdf",
}


@dataclass
class Result:
    priority: int
    molecule_name: str
    clean_name: str
    source_class: str
    run_dir: str
    pdb2gmx_status: str
    final_status: str
    message: str


def read_rows() -> list[dict[str, str]]:
    with SAMPLE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["priority"]))
    return rows


def tail(text: str, n: int = 14) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return " | ".join(lines[-n:])


def molecule_stem(molecule_name: str) -> str:
    return molecule_name.replace("/", "_")


def run_one(row: dict[str, str]) -> Result:
    molecule = row["molecule_name"]
    clean = row["clean_name"]
    source_class = row["source_class"]
    priority = int(row["priority"])
    run_dir = RUN_ROOT / clean / "gromos_exact_topology_dryrun"
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = molecule_stem(molecule)
    top = run_dir / f"{stem}_GROMOS_EXACT.top"
    gro = run_dir / f"{stem}_GROMOS_EXACT_processed.gro"
    metadata = run_dir / "gromos_exact_input_metadata.json"
    if top.exists() and gro.exists() and metadata.exists():
        return Result(priority, molecule, clean, source_class, str(run_dir), "done", "prepared", "existing topology")

    cmd = [
        sys.executable,
        str(PREPARE),
        "--sdf",
        str(SDF_BY_CLASS[source_class]),
        "--molecule-name",
        molecule,
        "--source-ff",
        str(SOURCE_FF),
        "--output-dir",
        str(run_dir),
        "--run-pdb2gmx",
    ]
    log_path = run_dir / "prepare_gromos_exact.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode == 0 and top.exists() and gro.exists():
        return Result(priority, molecule, clean, source_class, str(run_dir), "done", "prepared", "topology generated")
    msg = tail(log_path.read_text(encoding="utf-8", errors="replace"))
    status = "skipped" if "skips N-containing/carbonyl" in msg else "failed"
    return Result(priority, molecule, clean, source_class, str(run_dir), status, status, msg)


def write_results(results: list[Result]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [result.__dict__ for result in results]
    with STATUS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    STATUS_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    prepared = sum(1 for result in results if result.final_status == "prepared")
    skipped = sum(1 for result in results if result.final_status == "skipped")
    failed = sum(1 for result in results if result.final_status == "failed")
    lines = [
        "# Extended-24 GROMOS exact topology dry-run status",
        "",
        f"- prepared: {prepared}",
        f"- skipped: {skipped}",
        f"- failed: {failed}",
        "",
        "说明：这是 GROMOS united-atom exact single-residue smoke path；NAc/羰基体系使用 generic GROMOS peptide/ACE-like graph rules 支持，属于批量补齐路线，不等同于 curated residue template。",
        "",
        "| priority | molecule | class | status | run_dir |",
        "|---:|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result.priority} | `{result.molecule_name}` | {result.source_class} | {result.final_status} | `{result.run_dir}` |"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = [run_one(row) for row in read_rows()]
    write_results(results)
    counts: dict[str, int] = {}
    for result in results:
        counts[result.final_status] = counts.get(result.final_status, 0) + 1
    print(json.dumps({"counts": counts, "run_root": str(RUN_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
