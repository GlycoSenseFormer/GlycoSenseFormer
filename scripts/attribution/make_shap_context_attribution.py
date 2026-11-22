#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
FINAL_DIR = PROJECT_ROOT / "outputs/attribution"
PROOF_DIR = PROJECT_ROOT / "data/attribution"
OUT_BASENAME = "fig_shap_context_attribution"

ATOM_ATTR_PATH = PROOF_DIR / "proof_context_ablation_atom_level.csv"
TOKEN_ATTR_PATH = PROOF_DIR / "proof_context_token_ablation_atom_level.csv"
ATTENTION_PATH = PROOF_DIR / "proof_cross_attention_weights_representative_molecule.csv"

EDGE = "#4A4A4A"
GRID = "#E6E6E6"
BLUE = "#6FAFC5"
ORANGE = "#E2A85E"
GREEN = "#9DB89A"
RED = "#C94D4D"
PURPLE = "#8E6BBE"


def register_arial() -> None:
    font_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for name in (
        "Arial.ttf",
        "arialbd.ttf",
        "ariali.ttf",
        "arialbi.ttf",
        "Arial_Bold.ttf",
        "Arial_Italic.ttf",
        "Arial_Bold_Italic.ttf",
    ):
        font_path = font_dir / name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def apply_style() -> None:
    register_arial()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 10.3,
            "axes.linewidth": 1.3,
            "axes.labelsize": 10.8,
            "axes.titlesize": 12.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(EDGE)
        spine.set_linewidth(1.15)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def task_color(task: str) -> str:
    return BLUE if task == "13C" else ORANGE


def motif_for_site(task: str, site: str) -> str:
    if task == "13C":
        if site == "C1":
            return "anomeric carbon"
        if site == "C6":
            return "hydroxymethyl carbon"
        if site in {"C2", "C3", "C4", "C5"}:
            return "ring carbon"
        return "other carbon"
    if site == "H1":
        return "anomeric proton"
    if site == "H6":
        return "hydroxymethyl proton"
    if site in {"H2", "H3", "H4", "H5"}:
        return "ring proton"
    return "other proton"


def add_attribution_columns(atom_df: pd.DataFrame) -> pd.DataFrame:
    out = atom_df.copy()
    out["context_attr_ppm"] = out["max_token_ablation_shift"].astype(float)
    out["no_context_attr_ppm"] = out["prediction_shift_no_context"].astype(float)
    out["motif"] = [motif_for_site(task, site) for task, site in zip(out["task"], out["site"], strict=False)]
    return out


def summarize_sites(atom_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, site, motif), sub in atom_df.groupby(["task", "site", "motif"], observed=True):
        vals = sub["context_attr_ppm"].to_numpy(float)
        rows.append(
            {
                "task": task,
                "site": site,
                "motif": motif,
                "n": len(vals),
                "mean_context_attr_ppm": float(np.mean(vals)),
                "std_context_attr_ppm": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median_context_attr_ppm": float(np.median(vals)),
                "mean_no_context_attr_ppm": float(sub["no_context_attr_ppm"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "site"])


def summarize_tokens(token_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, split, token_index), sub in token_df.groupby(["task", "split", "token_index"], observed=True):
        vals = sub["token_prediction_shift"].to_numpy(float)
        rows.append(
            {
                "task": task,
                "split": split,
                "token_index": int(token_index),
                "n": len(vals),
                "mean_token_attr_ppm": float(np.mean(vals)),
                "std_token_attr_ppm": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median_token_attr_ppm": float(np.median(vals)),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "split", "token_index"])


def motif_enrichment(atom_df: pd.DataFrame, top_fraction: float = 0.2) -> pd.DataFrame:
    rows = []
    for task, sub in atom_df.groupby("task", observed=True):
        threshold = float(sub["context_attr_ppm"].quantile(1.0 - top_fraction))
        high = sub[sub["context_attr_ppm"] >= threshold]
        for motif in sorted(sub["motif"].unique()):
            all_prop = float((sub["motif"] == motif).mean())
            high_prop = float((high["motif"] == motif).mean())
            rows.append(
                {
                    "task": task,
                    "motif": motif,
                    "top_fraction": top_fraction,
                    "threshold_ppm": threshold,
                    "n_all": int((sub["motif"] == motif).sum()),
                    "n_high": int((high["motif"] == motif).sum()),
                    "all_fraction": all_prop,
                    "high_fraction": high_prop,
                    "enrichment_factor": high_prop / all_prop if all_prop > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def representative_atom_table(atom_df: pd.DataFrame, attention_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unique_attention = attention_df[
        ["split", "task", "sample_name", "atom_order", "site_label", "is_glycosidic_linkage_position"]
    ].drop_duplicates()
    for task in ("13C", "1H"):
        tri_samples = unique_attention[(unique_attention["split"] == "tri") & (unique_attention["task"] == task)]
        sample_name = str(tri_samples["sample_name"].iloc[0])
        sub_attr = atom_df[(atom_df["task"] == task) & (atom_df["sample_name"] == sample_name)].copy()
        sub_lab = unique_attention[(unique_attention["task"] == task) & (unique_attention["sample_name"] == sample_name)]
        merged = sub_attr.merge(
            sub_lab,
            left_on=["task", "sample_name", "atom_idx"],
            right_on=["task", "sample_name", "atom_order"],
            how="left",
        )
        merged["site_label"] = merged["site_label"].fillna(merged["site"])
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def draw_representative_panel(ax: plt.Axes, rep: pd.DataFrame, task: str) -> None:
    sub = rep[rep["task"] == task].copy().sort_values("atom_idx")
    x = np.arange(len(sub))
    colors = np.where(sub["is_glycosidic_linkage_position"].fillna(False).to_numpy(bool), RED, task_color(task))
    ax.bar(
        x,
        sub["context_attr_ppm"],
        color=colors,
        edgecolor=EDGE,
        linewidth=0.75,
        width=0.76,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["site_label"], rotation=55, ha="right", fontsize=8.2)
    ax.set_ylabel("Context attribution (ppm)")
    task_label = r"$^{13}$C" if task == "13C" else r"$^1$H"
    ax.set_title(f"Representative {task_label} atom-level attribution", fontweight="bold", pad=8)
    style_axis(ax, "y")


def draw_token_panel(ax: plt.Axes, token_summary: pd.DataFrame) -> None:
    sub = token_summary[token_summary["split"] == "tri"].copy()
    x_base = np.arange(3)
    width = 0.34
    for offset, task in [(-width / 2, "13C"), (width / 2, "1H")]:
        vals = []
        errs = []
        for token_idx in range(3):
            row = sub[(sub["task"] == task) & (sub["token_index"] == token_idx)].iloc[0]
            vals.append(row["mean_token_attr_ppm"])
            errs.append(row["std_token_attr_ppm"])
        ax.bar(
            x_base + offset,
            vals,
            yerr=errs,
            width=width,
            color=task_color(task),
            edgecolor=EDGE,
            linewidth=0.8,
            capsize=2.5,
            label=r"$^{13}$C" if task == "13C" else r"$^1$H",
            zorder=3,
        )
    ax.set_xticks(x_base)
    ax.set_xticklabels(["R1 token", "R2 token", "R3 token"])
    ax.set_ylabel("Mean token attribution (ppm)")
    ax.set_title("Residue-context token attribution in tri set", fontweight="bold", pad=8)
    style_axis(ax, "y")
    ax.legend(loc="upper right", fontsize=9.0)


def draw_site_summary_panel(ax: plt.Axes, site_summary: pd.DataFrame) -> None:
    order = ["C1", "C2", "C3", "C4", "C5", "C6", "H1", "H2", "H3", "H4", "H5", "H6"]
    sub = site_summary.set_index("site").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = [task_color(task) for task in sub["task"]]
    ax.bar(
        x,
        sub["mean_context_attr_ppm"],
        yerr=sub["std_context_attr_ppm"],
        color=colors,
        edgecolor=EDGE,
        linewidth=0.75,
        capsize=2.0,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["site"], rotation=0)
    ax.set_ylabel("Mean context attribution (ppm)")
    ax.set_title("Global atom-site attribution", fontweight="bold", pad=8)
    style_axis(ax, "y")


def draw_enrichment_panel(ax: plt.Axes, enrichment: pd.DataFrame) -> None:
    order = [
        ("13C", "anomeric carbon"),
        ("13C", "ring carbon"),
        ("13C", "hydroxymethyl carbon"),
        ("1H", "anomeric proton"),
        ("1H", "ring proton"),
        ("1H", "hydroxymethyl proton"),
    ]
    rows = []
    for task, motif in order:
        row = enrichment[(enrichment["task"] == task) & (enrichment["motif"] == motif)]
        if not row.empty:
            rows.append(row.iloc[0])
    sub = pd.DataFrame(rows)
    labels = [m.replace(" ", "\n") for m in sub["motif"]]
    colors = [task_color(task) for task in sub["task"]]
    x = np.arange(len(sub))
    ax.bar(
        x,
        sub["enrichment_factor"],
        color=colors,
        edgecolor=EDGE,
        linewidth=0.8,
        zorder=3,
    )
    ax.axhline(1.0, color=EDGE, linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("Enrichment factor")
    ax.set_title("Motif enrichment among top 20% atoms", fontweight="bold", pad=8)
    style_axis(ax, "y")


def build_figure(
    rep: pd.DataFrame,
    token_summary: pd.DataFrame,
    site_summary: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> list[Path]:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.0), constrained_layout=False)
    draw_representative_panel(axes[0, 0], rep, "13C")
    draw_representative_panel(axes[0, 1], rep, "1H")
    draw_token_panel(axes[0, 2], token_summary)
    draw_site_summary_panel(axes[1, 0], site_summary)
    draw_enrichment_panel(axes[1, 1], enrichment)

    axes[1, 2].axis("off")
    legend_handles = [
        Patch(facecolor=BLUE, edgecolor=EDGE, label=r"$^{13}$C attribution"),
        Patch(facecolor=ORANGE, edgecolor=EDGE, label=r"$^1$H attribution"),
        Patch(facecolor=RED, edgecolor=EDGE, label="glycosidic-linkage position"),
    ]
    axes[1, 2].legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        fontsize=10.5,
        borderaxespad=0.0,
    )
    axes[1, 2].text(
        0.0,
        0.50,
        "Attribution score = absolute prediction shift\n"
        "after zeroing residue-context tokens.\n\n"
        "This is a grouped SHAP-style perturbation\n"
        "attribution for the GlycoSense context branch.",
        ha="left",
        va="top",
        fontsize=10.2,
        color="#222222",
        transform=axes[1, 2].transAxes,
    )

    for ax, label in zip(axes.flatten(), "abcdef"):
        panel_label(ax, label)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.13, hspace=0.48, wspace=0.34)
    outputs = []
    for ext in ("png", "pdf", "svg"):
        out = THIS_DIR / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        outputs.append(out)
    plt.close(fig)
    return outputs


def copy_to_final(paths: list[Path]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, FINAL_DIR / path.name)


def main() -> None:
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    atom_df = add_attribution_columns(pd.read_csv(ATOM_ATTR_PATH))
    token_df = pd.read_csv(TOKEN_ATTR_PATH)
    attention_df = pd.read_csv(ATTENTION_PATH)

    site_summary = summarize_sites(atom_df)
    token_summary = summarize_tokens(token_df)
    enrichment = motif_enrichment(atom_df, top_fraction=0.2)
    representative = representative_atom_table(atom_df, attention_df)

    outputs = []
    for name, df in [
        ("shap_context_atom_level.csv", atom_df),
        ("shap_context_site_summary.csv", site_summary),
        ("shap_context_token_summary.csv", token_summary),
        ("shap_context_motif_enrichment.csv", enrichment),
        ("shap_context_representative_atoms.csv", representative),
    ]:
        path = THIS_DIR / name
        df.to_csv(path, index=False)
        outputs.append(path)

    fig_paths = build_figure(representative, token_summary, site_summary, enrichment)
    outputs.extend(fig_paths)
    copy_to_final(outputs)


if __name__ == "__main__":
    main()
