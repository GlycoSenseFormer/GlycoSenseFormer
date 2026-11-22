#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


COLORS = {
    "charmm": "#D55E00",
    "mmff": "#0072B2",
    "gray": "#555555",
    "light": "#DDDDDD",
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


def load_conformers(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        conformers = pickle.load(handle)
    arr = np.asarray(conformers, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path} does not contain (n_conformers, n_atoms, 3) coordinates")
    return arr


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    a = coords_a - coords_a.mean(axis=0)
    b = coords_b - coords_b.mean(axis=0)
    covariance = a.T @ b
    v, _, wt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(v @ wt))
    rotation = v @ correction @ wt
    diff = a @ rotation - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def pairwise_rmsd(conformers: np.ndarray) -> np.ndarray:
    n = conformers.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            value = kabsch_rmsd(conformers[i], conformers[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def read_matrix(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            rows.append([float(value) for value in row[1:]])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Could not read matrix from {path}")
    return matrix


def classical_mds(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    centered = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centered @ (distance * distance) @ centered
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    positive = np.maximum(values[:2], 0.0)
    return vectors[:, :2] * np.sqrt(positive)


def nearest_coverage(nearest_values: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    return np.asarray([(nearest_values <= cutoff).mean() for cutoff in cutoffs], dtype=np.float64)


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_axisbelow(True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot partial agreement between CHARMM and MMFF low-energy ensembles.")
    parser.add_argument("--charmm-pickle", required=True)
    parser.add_argument("--mmff-pickle", required=True)
    parser.add_argument("--cross-rmsd-matrix", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    charmm = load_conformers(Path(args.charmm_pickle))
    mmff = load_conformers(Path(args.mmff_pickle))
    cross = read_matrix(Path(args.cross_rmsd_matrix))
    if cross.shape != (charmm.shape[0], mmff.shape[0]):
        raise ValueError(f"Cross matrix shape {cross.shape} does not match conformer counts")

    charmm_to_mmff = cross.min(axis=1)
    mmff_to_charmm = cross.min(axis=0)
    cutoffs = np.linspace(0.1, 1.0, 91)
    charmm_cov = nearest_coverage(charmm_to_mmff, cutoffs)
    mmff_cov = nearest_coverage(mmff_to_charmm, cutoffs)
    threshold = float(args.threshold)
    charmm_at_threshold = float((charmm_to_mmff <= threshold).mean())
    mmff_at_threshold = float((mmff_to_charmm <= threshold).mean())

    charmm_internal = pairwise_rmsd(charmm)
    mmff_internal = pairwise_rmsd(mmff)
    distance = np.block([[charmm_internal, cross], [cross.T, mmff_internal]])
    coords = classical_mds(distance)
    n_charmm = charmm.shape[0]

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
    fig = plt.figure(figsize=(10.6, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    image = ax0.imshow(cross, cmap="viridis_r", vmin=0.0, vmax=1.6, aspect="auto", interpolation="nearest")
    ax0.set_title("A. Pairwise RMSD map", loc="left", fontsize=9, fontweight="bold")
    ax0.set_xlabel("MMFF94 low-energy rank", fontsize=9)
    ax0.set_ylabel("CHARMM low-energy rank", fontsize=9)
    ax0.tick_params(labelsize=8)
    cbar = fig.colorbar(image, ax=ax0, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("RMSD (A)", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(cutoffs, charmm_cov * 100, color=COLORS["charmm"], linewidth=2.0, label="CHARMM covered by MMFF")
    ax1.plot(cutoffs, mmff_cov * 100, color=COLORS["mmff"], linewidth=2.0, label="MMFF covered by CHARMM")
    ax1.axvline(threshold, color=COLORS["gray"], linewidth=1.0, linestyle="--")
    ax1.scatter([threshold], [charmm_at_threshold * 100], color=COLORS["charmm"], s=30, zorder=3)
    ax1.scatter([threshold], [mmff_at_threshold * 100], color=COLORS["mmff"], s=30, zorder=3)
    ax1.text(
        threshold + 0.025,
        52,
        f"{threshold:.1f} A cutoff\nCHARMM {charmm_at_threshold * 100:.0f}%\nMMFF {mmff_at_threshold * 100:.0f}%",
        fontsize=8,
        va="center",
    )
    ax1.set_title("B. Nearest-neighbor coverage", loc="left", fontsize=9, fontweight="bold")
    ax1.set_xlabel("RMSD cutoff (A)", fontsize=9)
    ax1.set_ylabel("Covered conformers (%)", fontsize=9)
    ax1.set_xlim(0.1, 1.0)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", color=COLORS["light"], linewidth=0.6)
    ax1.legend(frameon=False, fontsize=7.5, loc="lower right")
    style_axis(ax1)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(
        coords[:n_charmm, 0],
        coords[:n_charmm, 1],
        s=22,
        color=COLORS["charmm"],
        alpha=0.76,
        linewidths=0,
        label="CHARMM low-energy 100",
    )
    ax2.scatter(
        coords[n_charmm:, 0],
        coords[n_charmm:, 1],
        s=22,
        color=COLORS["mmff"],
        alpha=0.70,
        linewidths=0,
        label="MMFF94 low-energy 100",
    )
    ax2.axhline(0, color=COLORS["light"], linewidth=0.6)
    ax2.axvline(0, color=COLORS["light"], linewidth=0.6)
    ax2.set_title("C. RMSD-MDS conformer space", loc="left", fontsize=9, fontweight="bold")
    ax2.set_xlabel("MDS 1", fontsize=9)
    ax2.set_ylabel("MDS 2", fontsize=9)
    ax2.legend(frameon=False, fontsize=7.5, loc="best")
    style_axis(ax2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        path = out_dir / f"fig_charmm_mmff_low_energy_partial_agreement.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(out_dir / "fig_charmm_mmff_low_energy_partial_agreement.png")


if __name__ == "__main__":
    main()
