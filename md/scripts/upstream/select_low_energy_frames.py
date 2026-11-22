#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


def parse_xvg(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "@")):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                continue
            rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise ValueError(f"No numeric rows found in {path}")
    return rows


def frame_index(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.gro$)", path.name)
    if match is None:
        raise ValueError(f"Could not parse frame index from {path.name}")
    return int(match.group(1))


def load_frames(frames_dir: Path, pattern: str) -> list[Path]:
    frames = sorted(frames_dir.glob(pattern), key=frame_index)
    if not frames:
        raise ValueError(f"No frames matched {frames_dir / pattern}")
    return frames


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = list(path.iterdir())
    if existing and not overwrite:
        raise ValueError(f"{path} is not empty; pass --overwrite to replace its contents")
    if overwrite:
        for item in existing:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def write_candidate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate_index", "time_ps", "potential_kj_mol", "frame_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_selected_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "energy_rank",
                "candidate_index",
                "time_ps",
                "potential_kj_mol",
                "source_frame_path",
                "selected_frame_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select low-potential-energy GROMACS frame files.")
    parser.add_argument("--energy-xvg", required=True, help="gmx energy XVG file with time and Potential columns")
    parser.add_argument("--frames-dir", required=True, help="Directory containing frame0.gro ... files")
    parser.add_argument("--frame-pattern", default="frame*.gro")
    parser.add_argument("--output-dir", required=True, help="Directory for selected frame copies")
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--skip-time-ps", type=float, default=0.0, help="Exclude rows at or before this time")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be positive")

    energy_rows = [(time, energy) for time, energy in parse_xvg(Path(args.energy_xvg)) if time > args.skip_time_ps]
    frames = load_frames(Path(args.frames_dir), args.frame_pattern)
    if len(energy_rows) != len(frames):
        raise ValueError(
            f"Energy rows after time filter ({len(energy_rows)}) do not match frame count ({len(frames)})"
        )
    if args.top_n > len(frames):
        raise ValueError(f"--top-n {args.top_n} exceeds available frame count {len(frames)}")

    candidates: list[dict[str, object]] = []
    for idx, ((time_ps, potential), frame) in enumerate(zip(energy_rows, frames)):
        candidates.append(
            {
                "candidate_index": idx,
                "time_ps": f"{time_ps:.6f}",
                "potential_kj_mol": f"{potential:.6f}",
                "frame_path": str(frame),
            }
        )

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)
    ranked = sorted(
        zip(range(len(frames)), energy_rows, frames),
        key=lambda item: (item[1][1], item[1][0], item[0]),
    )
    selected_rows: list[dict[str, object]] = []
    for rank, (candidate_idx, (time_ps, potential), source_frame) in enumerate(ranked[: args.top_n]):
        selected_frame = output_dir / f"selected_{rank:03d}.gro"
        shutil.copy2(source_frame, selected_frame)
        selected_rows.append(
            {
                "energy_rank": rank,
                "candidate_index": candidate_idx,
                "time_ps": f"{time_ps:.6f}",
                "potential_kj_mol": f"{potential:.6f}",
                "source_frame_path": str(source_frame),
                "selected_frame_path": str(selected_frame),
            }
        )

    candidate_csv = Path(args.candidate_csv)
    selected_csv = Path(args.selected_csv)
    metadata_json = Path(args.metadata_json)
    candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    selected_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    write_candidate_csv(candidate_csv, candidates)
    write_selected_csv(selected_csv, selected_rows)
    metadata = {
        "energy_xvg": args.energy_xvg,
        "frames_dir": args.frames_dir,
        "output_dir": args.output_dir,
        "skip_time_ps": args.skip_time_ps,
        "n_candidates": len(candidates),
        "top_n": args.top_n,
        "lowest_potential_kj_mol": float(ranked[0][1][1]),
        "highest_selected_potential_kj_mol": float(ranked[args.top_n - 1][1][1]),
        "highest_candidate_potential_kj_mol": float(ranked[-1][1][1]),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(selected_csv)


if __name__ == "__main__":
    main()
