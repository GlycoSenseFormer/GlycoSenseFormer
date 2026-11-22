#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


COLORS = {
    "uniform": "#0072B2",
    "low_energy": "#D55E00",
    "discarded": "#9AA0A6",
    "axis": "#222222",
    "grid": "#DDDDDD",
}
ARIAL_FONT_DIR = Path("/usr/share/fonts/truetype/msttcorefonts")
ARIAL_FONT_FILES = (
    "Arial.ttf",
    "arialbd.ttf",
    "ariali.ttf",
    "arialbi.ttf",
    "Arial_Bold.ttf",
    "Arial_Italic.ttf",
    "Arial_Bold_Italic.ttf",
)


def register_arial() -> None:
    for name in ARIAL_FONT_FILES:
        font_path = ARIAL_FONT_DIR / name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def read_nearest_csv(path: Path, value_col: str = "nearest_rmsd_angstrom") -> np.ndarray:
    values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row[value_col]))
    if not values:
        raise ValueError(f"No values found in {path}")
    return np.asarray(values, dtype=float)


def read_energy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    potentials: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            indices.append(int(row["candidate_index"]))
            potentials.append(float(row["potential_kj_mol"]))
    return np.asarray(indices, dtype=int), np.asarray(potentials, dtype=float)


def read_selected_indices(path: Path) -> set[int]:
    selected: set[int] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            selected.add(int(row["candidate_index"]))
    return selected


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["axis"], labelsize=8)
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def add_box_with_points(ax, data: list[np.ndarray], colors: list[str], labels: list[str], ylabel: str) -> None:
    rng = np.random.default_rng(7)
    box = ax.boxplot(
        data,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        boxprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)

    for position, (values, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.normal(0, 0.045, size=values.size)
        ax.scatter(
            np.full(values.size, position) + jitter,
            values,
            s=11,
            color=color,
            alpha=0.46,
            linewidths=0,
        )
        median = float(np.median(values))
        ax.text(
            position,
            median + 0.035,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111111",
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    style_axis(ax)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot evidence for CHARMM low-energy selection vs MMFF94 coverage.")
    parser.add_argument("--uniform-comparison-dir", required=True)
    parser.add_argument("--low-energy-comparison-dir", required=True)
    parser.add_argument("--all-candidates-comparison-dir", required=True)
    parser.add_argument("--candidate-energy-csv", required=True)
    parser.add_argument("--selected-energy-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    uniform_dir = Path(args.uniform_comparison_dir)
    low_dir = Path(args.low_energy_comparison_dir)
    all_dir = Path(args.all_candidates_comparison_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uniform_md_to_mmff = read_nearest_csv(uniform_dir / "rmsd_md_to_nearest_mmff.csv")
    low_md_to_mmff = read_nearest_csv(low_dir / "rmsd_md_to_nearest_mmff.csv")
    uniform_mmff_to_md = read_nearest_csv(uniform_dir / "rmsd_mmff_to_nearest_md.csv")
    low_mmff_to_md = read_nearest_csv(low_dir / "rmsd_mmff_to_nearest_md.csv")
    all_md_to_mmff = read_nearest_csv(all_dir / "rmsd_md_to_nearest_mmff.csv")

    candidate_indices, potentials = read_energy_csv(Path(args.candidate_energy_csv))
    selected = read_selected_indices(Path(args.selected_energy_csv))
    if all_md_to_mmff.size != potentials.size:
        raise ValueError(
            f"All-candidate RMSD count {all_md_to_mmff.size} does not match potential count {potentials.size}"
        )
    selected_mask = np.asarray([int(idx) in selected for idx in candidate_indices], dtype=bool)
    threshold = float(np.max(potentials[selected_mask]))

    register_arial()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.monospace": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.tt": "Arial",
            "mathtext.cal": "Arial",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2), constrained_layout=True)

    add_box_with_points(
        axes[0],
        [uniform_md_to_mmff, low_md_to_mmff],
        [COLORS["uniform"], COLORS["low_energy"]],
        ["Uniform\n100", "Low-energy\n100"],
        "CHARMM to nearest MMFF RMSD (A)",
    )
    axes[0].set_title("A. CHARMM -> MMFF", fontsize=9, loc="left", fontweight="bold")

    add_box_with_points(
        axes[1],
        [uniform_mmff_to_md, low_mmff_to_md],
        [COLORS["uniform"], COLORS["low_energy"]],
        ["Uniform\n100", "Low-energy\n100"],
        "MMFF to nearest CHARMM RMSD (A)",
    )
    axes[1].set_title("B. MMFF -> CHARMM", fontsize=9, loc="left", fontweight="bold")

    ax = axes[2]
    ax.scatter(
        potentials[~selected_mask],
        all_md_to_mmff[~selected_mask],
        s=18,
        color=COLORS["discarded"],
        alpha=0.58,
        linewidths=0,
        label="Discarded 100",
    )
    ax.scatter(
        potentials[selected_mask],
        all_md_to_mmff[selected_mask],
        s=20,
        color=COLORS["low_energy"],
        alpha=0.74,
        linewidths=0,
        label="Selected low-energy 100",
    )
    ax.axvline(threshold, color="#333333", linewidth=1.0, linestyle="--")
    ax.text(
        threshold + 0.9,
        1.66,
        "selection\ncutoff",
        ha="left",
        va="center",
        fontsize=8,
        rotation=90,
        color="#333333",
    )
    ax.set_xlabel("CHARMM Potential (kJ/mol)", fontsize=9)
    ax.set_ylabel("Nearest MMFF RMSD (A)", fontsize=9)
    ax.set_title("C. Potential vs MMFF distance", fontsize=9, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right", handletextpad=0.3)
    style_axis(ax)

    for suffix in ("png", "svg", "pdf"):
        path = out_dir / f"fig_charmm_low_energy_selection_evidence.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(out_dir / "fig_charmm_low_energy_selection_evidence.png")


if __name__ == "__main__":
    main()
