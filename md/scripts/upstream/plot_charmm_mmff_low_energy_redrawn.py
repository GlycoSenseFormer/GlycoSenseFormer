#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


COLORS = {
    "uniform": "#74AFCF",
    "selected": "#E58A45",
    "discarded": "#AEB4B8",
    "ink": "#242424",
    "grid": "#E4E4E4",
}


def register_arial() -> None:
    candidates = (
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
    )
    names = (
        "Arial.ttf",
        "arial.ttf",
        "arialbd.ttf",
        "Arial_Bold.ttf",
        "ariali.ttf",
        "arialbi.ttf",
    )
    for directory in candidates:
        for name in names:
            path = directory / name
            if path.exists():
                font_manager.fontManager.addfont(str(path))


def read_values(path: Path, column: str) -> np.ndarray:
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row[column]))
    if not values:
        raise ValueError(f"No values in {path}")
    return np.asarray(values, dtype=float)


def read_energy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    potentials: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            indices.append(int(row["candidate_index"]))
            potentials.append(float(row["potential_kj_mol"]))
    return np.asarray(indices, dtype=int), np.asarray(potentials, dtype=float)


def read_selected(path: Path) -> set[int]:
    with path.open(newline="") as handle:
        return {int(row["candidate_index"]) for row in csv.DictReader(handle)}


def style_axis(ax, *, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(colors=COLORS["ink"], labelsize=8.5, width=0.8)
    if grid == "y":
        ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.65)
    elif grid == "both":
        ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )


def distribution_panel(ax, uniform: np.ndarray, selected: np.ndarray, ylabel: str, title: str) -> None:
    rng = np.random.default_rng(17)
    data = [uniform, selected]
    colors = [COLORS["uniform"], COLORS["selected"]]
    box = ax.boxplot(
        data,
        positions=[1, 2],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.45},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
        boxprops={"linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    for x, values, color in zip((1, 2), data, colors):
        jitter = rng.normal(0.0, 0.045, size=values.size)
        ax.scatter(
            np.full(values.size, x) + jitter,
            values,
            s=13,
            color=color,
            alpha=0.56,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        median = float(np.median(values))
        ax.annotate(
            f"{median:.3f} Å",
            (x, median),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["ink"],
        )

    ax.set_xticks([1, 2], ["Uniform\n100", "Low-energy\n100"])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    style_axis(ax)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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

    uniform_charmm_to_mmff = read_values(uniform_dir / "rmsd_md_to_nearest_mmff.csv", "nearest_rmsd_angstrom")
    low_charmm_to_mmff = read_values(low_dir / "rmsd_md_to_nearest_mmff.csv", "nearest_rmsd_angstrom")
    uniform_mmff_to_charmm = read_values(uniform_dir / "rmsd_mmff_to_nearest_md.csv", "nearest_rmsd_angstrom")
    low_mmff_to_charmm = read_values(low_dir / "rmsd_mmff_to_nearest_md.csv", "nearest_rmsd_angstrom")
    all_charmm_to_mmff = read_values(all_dir / "rmsd_md_to_nearest_mmff.csv", "nearest_rmsd_angstrom")

    candidate_indices, potentials = read_energy(Path(args.candidate_energy_csv))
    selected_indices = read_selected(Path(args.selected_energy_csv))
    if potentials.size != all_charmm_to_mmff.size:
        raise ValueError("Candidate energy and all-candidate RMSD lengths do not match")
    selected_mask = np.asarray([int(index) in selected_indices for index in candidate_indices], dtype=bool)
    if selected_mask.sum() != 100 or (~selected_mask).sum() != 100:
        raise ValueError("Expected 100 selected and 100 discarded candidates")
    cutoff = float(potentials[selected_mask].max())

    register_arial()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "axes.linewidth": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.35, 7.0))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.88, wspace=0.34, hspace=0.44)
    fig.suptitle(
        "CHARMM36–MMFF94 low-energy conformer selection",
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        "bDGlc14bDGlcOMe: 200 CHARMM36-BMM candidates → lowest-energy 100",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    distribution_panel(
        axes[0, 0],
        uniform_charmm_to_mmff,
        low_charmm_to_mmff,
        "CHARMM → nearest MMFF94 RMSD (Å)",
        "CHARMM conformers covered by MMFF94",
    )
    panel_label(axes[0, 0], "a")

    distribution_panel(
        axes[0, 1],
        uniform_mmff_to_charmm,
        low_mmff_to_charmm,
        "MMFF94 → nearest CHARMM RMSD (Å)",
        "MMFF94 conformers covered by CHARMM",
    )
    panel_label(axes[0, 1], "b")

    ax = axes[1, 0]
    energy_min = float(potentials.min())
    energy_max = float(potentials.max())
    bins = np.linspace(energy_min, energy_max, 21)
    ax.hist(
        potentials[~selected_mask],
        bins=bins,
        color=COLORS["discarded"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.5,
        label="Discarded 100",
    )
    ax.hist(
        potentials[selected_mask],
        bins=bins,
        color=COLORS["selected"],
        alpha=0.86,
        edgecolor="white",
        linewidth=0.5,
        label="Selected low-energy 100",
    )
    ax.axvline(cutoff, color=COLORS["ink"], linewidth=1.1, linestyle="--")
    ax.annotate(
        f"cutoff {cutoff:.1f} kJ mol$^{{-1}}$",
        xy=(cutoff, ax.get_ylim()[1] * 0.78),
        xytext=(8, 0),
        textcoords="offset points",
        rotation=90,
        ha="left",
        va="center",
        fontsize=8,
        color=COLORS["ink"],
    )
    ax.set_xlabel("CHARMM36-BMM potential energy (kJ mol$^{-1}$)", fontsize=9)
    ax.set_ylabel("Candidate conformers", fontsize=9)
    ax.set_title("Energy distribution of 200 candidates", fontsize=10, fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    style_axis(ax)
    panel_label(ax, "c")

    ax = axes[1, 1]
    ax.scatter(
        potentials[~selected_mask],
        all_charmm_to_mmff[~selected_mask],
        s=22,
        color=COLORS["discarded"],
        alpha=0.66,
        linewidths=0,
        label="Discarded 100",
        rasterized=True,
    )
    ax.scatter(
        potentials[selected_mask],
        all_charmm_to_mmff[selected_mask],
        s=24,
        color=COLORS["selected"],
        alpha=0.82,
        linewidths=0,
        label="Selected low-energy 100",
        rasterized=True,
    )
    ax.axvline(cutoff, color=COLORS["ink"], linewidth=1.1, linestyle="--")
    ax.set_xlabel("CHARMM36-BMM potential energy (kJ mol$^{-1}$)", fontsize=9)
    ax.set_ylabel("CHARMM → nearest MMFF94 RMSD (Å)", fontsize=9)
    ax.set_title("Energy versus MMFF94 conformer distance", fontsize=10, fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    style_axis(ax, grid="both")
    panel_label(ax, "d")

    basename = "fig_charmm36_mmff94_low_energy_selection_redrawn"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            out_dir / f"{basename}.{suffix}",
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    summary = {
        "system": "bDGlc14bDGlcOMe",
        "candidate_count": int(potentials.size),
        "selected_count": int(selected_mask.sum()),
        "potential_kj_mol": {
            "minimum": energy_min,
            "selection_cutoff": cutoff,
            "maximum": energy_max,
        },
        "median_nearest_neighbor_rmsd_angstrom": {
            "CHARMM_to_MMFF_uniform100": float(np.median(uniform_charmm_to_mmff)),
            "CHARMM_to_MMFF_low_energy100": float(np.median(low_charmm_to_mmff)),
            "MMFF_to_CHARMM_uniform100": float(np.median(uniform_mmff_to_charmm)),
            "MMFF_to_CHARMM_low_energy100": float(np.median(low_mmff_to_charmm)),
        },
    }
    (out_dir / "figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_dir / f"{basename}.png")


if __name__ == "__main__":
    main()
