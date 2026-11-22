#!/usr/bin/env python3
"""Compact portrait version of the GlycoSenseFormer FPS100 force-field figure.

This version removes the original protocol schematic and 1H representation
schematic, leaving five labeled data panels in a compact paper-style layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments" / "glycosenseformer_conformer_source_v1"
ANALYSIS_ROOT = ROOT / "data/forcefield_analysis"
OUT_DIR = ANALYSIS_ROOT / "forcefield_story_figure_updated_mean_sd"
METRICS_CSV = OUT_DIR / "aggregated_metrics_replaced_seed7.csv"
ENSEMBLE_CSV = ANALYSIS_ROOT / "fps100_forcefield_diversity_v1" / "ensemble_metrics.csv"
PHI_PSI_CSV = ANALYSIS_ROOT / "fps100_forcefield_diversity_v1" / "glycosidic_phi_psi_long.csv"
SOURCE_SUMMARY_CSV = ANALYSIS_ROOT / "fps100_forcefield_diversity_v1" / "source_summary.csv"
OXYGEN_CSV = ANALYSIS_ROOT / "fps100_oxygen_proxy_1h_v1" / "oxygen_proxy_source_summary_with_1h.csv"

SOURCE_ORDER = ["mmff94_fps100", "charmm_fps100", "gromos_fps100", "glycam_fps100"]
SOURCE_LABELS = {
    "mmff94_fps100": "ETKDGv3/MMFF94",
    "charmm_fps100": "CHARMM36",
    "gromos_fps100": "GROMOS",
    "glycam_fps100": "GLYCAM06j",
}

# Palette matched to 稿件/定稿/fig2_fig3_combined_A4_portrait.svg.
COLORS = {
    "mmff94_fps100": "#EEA599",
    "charmm_fps100": "#FAC795",
    "gromos_fps100": "#E3EDE0",
    "glycam_fps100": "#ABD3E1",
}
ACCENTS = {
    "mmff94_fps100": "#EE8F80",
    "charmm_fps100": "#E6B64F",
    "gromos_fps100": "#8FB987",
    "glycam_fps100": "#6E9EB8",
}
GRID = "#CFCFCF"
INK = "#222222"
AXIS = "#000000"
EDGE = "#4A4A4A"
SUBTLE = "#666666"
PHI_PSI_CMAP = LinearSegmentedColormap.from_list(
    "paper_heat",
    ["#FFFFFF", "#ABD3E1", "#E3EDE0", "#FFE9BE", "#FAC795", "#EE8F80"],
)

ARIAL_FONT_FILES = [
    Path("/mnt/c/Windows/Fonts/arial.ttf"),
    Path("/mnt/c/Windows/Fonts/arialbd.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"),
]


def setup_style() -> None:
    for font_path in ARIAL_FONT_FILES:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 8.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.05,
            "axes.edgecolor": AXIS,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.14) -> None:
    ax.text(
        x,
        y,
        label.lower(),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16.5,
        fontweight="bold",
        color=INK,
    )


def ordered(df: pd.DataFrame, source_col: str = "train_source") -> pd.DataFrame:
    out = df.copy()
    out["source_order"] = pd.Categorical(out[source_col], SOURCE_ORDER, ordered=True)
    return out.sort_values("source_order")


def draw_performance(ax_c: plt.Axes, ax_h: plt.Axes, metrics_all: pd.DataFrame) -> None:
    for ax, task, title in [
        (ax_c, "13C", "¹³C prediction"),
        (ax_h, "1H", "¹H prediction"),
    ]:
        subset = ordered(metrics_all[metrics_all["task"] == task])
        x = np.arange(len(subset))
        values = subset["rmse_mean_ppm"].to_numpy()
        errors = subset["rmse_sd_ppm"].to_numpy()
        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=2.5,
            color=[COLORS[s] for s in subset["train_source"]],
            edgecolor=EDGE,
            linewidth=0.9,
            width=0.48,
            error_kw={"ecolor": EDGE, "elinewidth": 0.9, "capthick": 0.9},
        )
        ax.set_title(title + " (mean ± SD)", pad=3.5, fontweight="bold")
        ax.set_ylabel("RMSE (ppm)")
        ax.set_xticks(x)
        ax.set_xticklabels([SOURCE_LABELS[s] for s in subset["train_source"]], rotation=25, ha="right")
        ax.grid(False)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)
    panel_label(ax_c, "A")
    panel_label(ax_h, "B")


def draw_diversity(ax: plt.Axes, ensemble: pd.DataFrame, source_summary: pd.DataFrame) -> None:
    panel_label(ax, "C")
    data = [ensemble.loc[ensemble["source"] == s, "pairwise_rmsd_median"].dropna().to_numpy() for s in SOURCE_ORDER]
    bplot = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.34,
        showfliers=False,
        medianprops={"color": AXIS, "linewidth": 1.1},
        boxprops={"linewidth": 0.9, "color": EDGE},
        whiskerprops={"linewidth": 0.9, "color": EDGE},
        capprops={"linewidth": 0.9, "color": EDGE},
    )
    for patch, source in zip(bplot["boxes"], SOURCE_ORDER):
        patch.set_facecolor(COLORS[source])
        patch.set_alpha(0.95)
    ax.set_title("FPS100 conformer diversity", pad=3.5, fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Median RMSD (Å)", fontsize=9.2, labelpad=2)
    ax.set_xticks(np.arange(1, len(SOURCE_ORDER) + 1))
    ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCE_ORDER], rotation=25, ha="right", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=8.0, width=0.9, length=3.2)
    ax.tick_params(axis="x", width=0.9, length=3.2)
    ax.grid(False)
    ax.set_axisbelow(True)


def draw_phi_psi(fig: plt.Figure, gs_cell, phi_psi: pd.DataFrame) -> None:
    sub = gs_cell.subgridspec(2, 2, wspace=0.16, hspace=0.22)
    first_ax = None
    mesh = None
    axes: list[plt.Axes] = []
    for idx, source in enumerate(SOURCE_ORDER):
        ax = fig.add_subplot(sub[idx // 2, idx % 2])
        axes.append(ax)
        if first_ax is None:
            first_ax = ax
        df = phi_psi[phi_psi["source"] == source]
        mesh = ax.hist2d(
            df["phi_deg"],
            df["psi_deg"],
            bins=72,
            range=[[-180, 180], [-180, 180]],
            norm=LogNorm(vmin=1, vmax=900),
            cmap=PHI_PSI_CMAP,
        )
        ax.set_title(SOURCE_LABELS[source], color=ACCENTS[source], pad=2.5, fontsize=9.0)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-180, 0, 180])
        ax.set_yticks([-180, 0, 180])
        if idx // 2 == 1:
            ax.set_xlabel("phi (deg)")
        else:
            ax.set_xticklabels([])
        if idx % 2 == 0:
            ax.set_ylabel("psi (deg)")
        else:
            ax.set_yticklabels([])
        ax.grid(False)
    assert first_ax is not None and mesh is not None
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y0 = min(box.y0 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    title_y = y1 + 0.028
    fig.text(
        (x0 + x1) / 2,
        title_y,
        "Glycosidic phi/psi density",
        ha="center",
        va="bottom",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        x0 - 0.045,
        title_y,
        "e",
        ha="left",
        va="bottom",
        fontsize=16.5,
        fontweight="bold",
        color=INK,
    )
    cax = fig.add_axes([x1 + 0.018, y0, 0.016, y1 - y0])
    cbar = fig.colorbar(mesh[3], cax=cax)
    cbar.set_ticks([1, 10, 100])
    cbar.set_ticklabels(["1", "10", "100"])
    cbar.set_label("points / bin", fontsize=8.6)
    cbar.ax.tick_params(labelsize=7.6, width=0.8)


def draw_oxygen_proxy(ax: plt.Axes, oxygen: pd.DataFrame) -> None:
    panel_label(ax, "D")
    oxygen = oxygen.copy()
    oxygen["source_order"] = pd.Categorical(oxygen["source"], SOURCE_ORDER, ordered=True)
    oxygen = oxygen.sort_values("source_order")
    x = np.arange(len(oxygen))
    proxy = oxygen["hbearing_o_proxy_contacts_lt_3p5_mean"].to_numpy()
    rmse = oxygen["rmse_mean_ppm"].to_numpy()
    rmse_sd = oxygen["rmse_sd_ppm"].to_numpy()
    bars = ax.bar(
        x,
        proxy,
        color=[COLORS[s] for s in oxygen["source"].astype(str)],
        edgecolor=EDGE,
        linewidth=0.9,
        width=0.48,
        alpha=0.95,
    )
    ax.set_title("O-neighborhood proxy vs ¹H error", pad=3.5, fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Proxy contacts", fontsize=9.2, labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(oxygen["source_label"], rotation=25, ha="right", fontsize=7.6)
    ax.tick_params(axis="y", labelsize=8.0, width=0.9, length=3.2)
    ax.tick_params(axis="x", width=0.9, length=3.2)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(proxy) * 1.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)

    ax2 = ax.twinx()
    ax2.errorbar(
        x,
        rmse,
        yerr=rmse_sd,
        color=EDGE,
        marker="o",
        markersize=3.6,
        linewidth=1.15,
        elinewidth=0.85,
        capsize=2.2,
        capthick=0.85,
    )
    ax2.set_ylabel("¹H RMSE (ppm)", fontsize=9.2, labelpad=5)
    ax2.tick_params(axis="y", labelsize=8.0, width=0.75, length=3.0, pad=3)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.85)
    ax2.spines["top"].set_linewidth(0.85)
    ax2.set_ylim(min(rmse - rmse_sd) - 0.004, max(rmse + rmse_sd) + 0.006)


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(METRICS_CSV)
    metrics_all = metrics[metrics["train_source"].isin(SOURCE_ORDER)].copy()
    ensemble = pd.read_csv(ENSEMBLE_CSV)
    ensemble = ensemble[ensemble["source"].isin(SOURCE_ORDER)].copy()
    source_summary = pd.read_csv(SOURCE_SUMMARY_CSV)
    source_summary = source_summary[source_summary["source"].isin(SOURCE_ORDER)].copy()
    phi_psi = pd.read_csv(PHI_PSI_CSV)
    phi_psi = phi_psi[phi_psi["source"].isin(SOURCE_ORDER)].copy()
    oxygen = pd.read_csv(OXYGEN_CSV)
    oxygen = oxygen[oxygen["source"].isin(SOURCE_ORDER)].copy()
    oxygen = oxygen.drop(columns=[column for column in ["rmse_ppm"] if column in oxygen.columns])
    oxygen = oxygen.merge(
        metrics_all.loc[metrics_all["task"] == "1H", ["train_source", "rmse_mean_ppm", "rmse_sd_ppm"]],
        left_on="source",
        right_on="train_source",
        how="inner",
        validate="one_to_one",
    )

    # Compact portrait layout for a publication-ready multi-panel figure.
    fig = plt.figure(figsize=(7.5, 11.0))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.18, 1.18, 3.00],
        hspace=0.40,
        wspace=0.40,
        left=0.09,
        right=0.91,
        top=0.975,
        bottom=0.055,
    )
    top = gs[0, 0:6].subgridspec(1, 2, wspace=0.42)
    ax_a1 = fig.add_subplot(top[0, 0])
    ax_a2 = fig.add_subplot(top[0, 1])
    draw_performance(ax_a1, ax_a2, metrics_all)

    middle = gs[1, 0:6].subgridspec(1, 2, wspace=0.42)
    ax_b = fig.add_subplot(middle[0, 0])
    draw_diversity(ax_b, ensemble, source_summary)

    ax_d = fig.add_subplot(middle[0, 1])
    draw_oxygen_proxy(ax_d, oxygen)

    draw_phi_psi(fig, gs[2, 0:6], phi_psi)

    out_base = OUT_DIR / "fig_glycosenseformer_fps100_forcefield_story_updated_mean_sd"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"))
    plt.close(fig)

    summary = {
        "output_png": str(out_base.with_suffix(".png")),
        "output_pdf": str(out_base.with_suffix(".pdf")),
        "output_svg": str(out_base.with_suffix(".svg")),
        "layout": "portrait, Arial, equal-sized upper panels, enlarged panel E, centered E title",
        "removed_original_panels": ["A protocol schematic", "F 1H representation schematic"],
        "remaining_panels": [
            "A 13C performance",
            "B 1H performance",
            "C FPS100 diversity",
            "D oxygen-neighborhood proxy vs 1H RMSE",
            "E glycosidic phi/psi density",
        ],
        "n_phi_psi_points": int(len(phi_psi)),
        "n_ensembles": int(len(ensemble)),
        "aggregated_metrics_csv": str(METRICS_CSV),
        "performance_statistic": "mean ± sample SD",
        "13C_seed_basis": {
            "MMFF94": [4, 7, 9],
            "CHARMM36": [4, 11, 9],
            "GROMOS": [4, 7, 9],
            "GLYCAM06j": [4, 11, 9],
        },
        "1H_seed_basis": [4, 7, 9],
    }
    (OUT_DIR / "figure_inputs_summary.json").write_text(pd.Series(summary).to_json(force_ascii=False, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
