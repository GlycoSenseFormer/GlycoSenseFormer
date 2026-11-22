#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = ROOT / "data/attribution/inputs"
DATASET_DIR = ROOT / "data" / "datasets_glyconmr_len123_independent_validation_clean_split1"
SCATTER_POINTS = FIG_DIR / "fig3" / "fig3_scatter_points_seed0.csv"
SITE_POINTS = FIG_DIR / "figs2" / "figs2_atom_site_error_points.csv"
FIG4_DECOMP = (
    FIG_DIR
    / "fig4"
    / "outputs"
    / "interpretability_decomposition"
    / "fig4_interpretability_decomposition_13C_large_clear.csv"
)

CONTEXT_MODEL = "New cross-attn learnable alpha bs16"
BASELINES = ["EquiformerV3-full bs18", "GeqShift-O3Transformer"]
MODEL_LABELS = {
    CONTEXT_MODEL: "Cross-attn",
    "EquiformerV3-full bs18": "EquiformerV3-full",
    "GeqShift-O3Transformer": "GeqShift",
}
SPLIT_ORDER = ["mono", "di", "tri"]
SPLIT_LABELS = {"mono": "Mono", "di": "Di", "tri": "Tri"}
SITE_ORDER = {
    "13C": [f"C{i}" for i in range(1, 7)],
    "1H": [f"H{i}" for i in range(1, 7)],
}

COLORS = {
    "pink": "#EEA599",
    "yellow": "#FFE9BE",
    "blue": "#92B4C8",
    "green": "#AFCFBE",
    "gray": "#6F6F6F",
    "dark": "#222222",
    "grid": "#E7E7E7",
    "light": "#F7F7F7",
    "bad": "#C66557",
    "good": "#5B9E7D",
}

DEFAULT_RUN_DIR = (
    ROOT
    / "configs/original_runs"
    / "full_100conf_eqv3_glycanaa_cross_attention_identity_residual_fusion_learnable_alpha_fold0_epochs10_bs16_seed0_20260607"
)
ATTENTION_REPRESENTATIVE_NAMES = {
    "di": "glyconmr_godess__labeled_all_1207.pdb",
    "tri": "glyconmr_godess__labeled_all_2119.pdb",
}


def register_arial() -> None:
    font_dir = Path("/tmp/equiformer_arial")
    for name in ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF", "Arialbi.TTF"):
        font_path = font_dir / name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def apply_style() -> None:
    register_arial()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.monospace": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 9.5,
            "axes.linewidth": 1.25,
            "axes.labelsize": 10.2,
            "axes.titlesize": 11.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 3.8,
            "ytick.major.size": 3.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 11) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[sample_idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return mean, float(low), float(high)


def parse_sdf_properties(block: str) -> tuple[str, dict[str, str]]:
    lines = block.strip().splitlines()
    name = lines[0].strip()
    props: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = re.match(r">\s+<([^>]+)>", lines[i])
        if match:
            key = match.group(1)
            values = []
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                values.append(lines[i])
                i += 1
            props[key] = "\n".join(values)
        i += 1
    return name, props


def parse_sdf_block(sample_name: str) -> tuple[list[str], dict[str, str]] | None:
    for sdf_path in sorted(DATASET_DIR.glob("glyconmr_len123_independent_validation_*.sdf")):
        for block in sdf_path.read_text(encoding="utf-8", errors="ignore").split("$$$$"):
            if block.strip().startswith(sample_name):
                lines = block.strip().splitlines()
                _, props = parse_sdf_properties(block)
                return lines, props
    return None


def parse_sdf_bonds(lines: list[str]) -> list[tuple[int, int]]:
    n_atoms = int(lines[3][0:3])
    n_bonds = int(lines[3][3:6])
    bonds = []
    for line in lines[4 + n_atoms : 4 + n_atoms + n_bonds]:
        bonds.append((int(line[0:3]) - 1, int(line[3:6]) - 1))
    return bonds


def site_from_heavy_atom_name(task: str, atom_name: str) -> str | None:
    match = re.search(r"(\d+)$", str(atom_name))
    if not match:
        return None
    position = int(match.group(1))
    if position < 1 or position > 6:
        return None
    return ("C" if task == "13C" else "H") + str(position)


def build_site_mapping() -> dict[tuple[str, str], list[str | None]]:
    mapping: dict[tuple[str, str], list[str | None]] = {}
    for sdf_path in sorted(DATASET_DIR.glob("glyconmr_len123_independent_validation_*.sdf")):
        if sdf_path.name == "glyconmr_len123_independent_validation.sdf":
            continue
        text = sdf_path.read_text(encoding="utf-8", errors="ignore")
        for block in text.split("$$$$"):
            if not block.strip():
                continue
            sample_name, props = parse_sdf_properties(block)
            heavy_atom_names = ast.literal_eval(props.get("heavy_atom_names", "[]"))
            for task in ("13C", "1H"):
                spectrum_text = props.get(f"{task} Spectrum")
                if not spectrum_text:
                    continue
                spectrum = ast.literal_eval(spectrum_text)
                target_atom_indices = sorted(int(idx) for idx in spectrum.keys())
                mapping[(sample_name, task)] = [
                    site_from_heavy_atom_name(task, heavy_atom_names[idx])
                    for idx in target_atom_indices
                ]
    return mapping


def readable_residue_atom_labels(task: str, atom_names: list[str], target_atom_indices: list[int]) -> list[str]:
    residue_counts: dict[str, int] = {}
    labels: list[str] = []
    for fallback_order, atom_idx in enumerate(target_atom_indices, start=1):
        atom_name = str(atom_names[atom_idx])
        match = re.fullmatch(r"C(\d+)", atom_name)
        if not match:
            labels.append(f"Atom {fallback_order}")
            continue
        carbon_site = f"C{match.group(1)}"
        residue_counts[carbon_site] = residue_counts.get(carbon_site, 0) + 1
        residue_label = f"R{residue_counts[carbon_site]}"
        site_label = carbon_site if task == "13C" else f"H{match.group(1)}"
        labels.append(f"{residue_label}-{site_label}")
    return labels


def labels_by_atom_index(task: str, atom_names: list[str], target_atom_indices: list[int]) -> dict[int, str]:
    labels = readable_residue_atom_labels(task, atom_names, target_atom_indices)
    return dict(zip(target_atom_indices, labels, strict=True))


def glycosidic_linkage_labels(sample_name: str, task: str) -> set[str]:
    parsed = parse_sdf_block(sample_name)
    if parsed is None:
        return set()
    lines, props = parsed
    atom_names = ast.literal_eval(props.get("heavy_atom_names", "[]"))
    spectrum_text = props.get(f"{task} Spectrum")
    if not spectrum_text:
        return set()
    target_atom_indices = sorted(int(idx) for idx in ast.literal_eval(spectrum_text).keys())
    atom_to_label = labels_by_atom_index(task, atom_names, target_atom_indices)
    bonds = parse_sdf_bonds(lines)

    neighbors: dict[int, list[int]] = {idx: [] for idx in range(len(atom_names))}
    for a, b in bonds:
        neighbors[a].append(b)
        neighbors[b].append(a)

    linked_carbon_indices: set[int] = set()
    for oxygen_idx, atom_name in enumerate(atom_names):
        if not str(atom_name).startswith("O"):
            continue
        carbon_neighbors = [idx for idx in neighbors[oxygen_idx] if str(atom_names[idx]).startswith("C")]
        if len(carbon_neighbors) != 2:
            continue
        left, right = carbon_neighbors
        left_name = str(atom_names[left])
        right_name = str(atom_names[right])
        is_internal_ring_oxygen = str(atom_name) == "O5" and {left_name, right_name} == {"C1", "C5"}
        if is_internal_ring_oxygen:
            continue
        linked_carbon_indices.update([left, right])

    labels = set()
    for atom_idx in sorted(linked_carbon_indices):
        label = atom_to_label.get(atom_idx)
        if label:
            labels.add(label)
    return labels


def build_atom_label_mapping() -> dict[tuple[str, str], list[str]]:
    mapping: dict[tuple[str, str], list[str]] = {}
    for sdf_path in sorted(DATASET_DIR.glob("glyconmr_len123_independent_validation_*.sdf")):
        if sdf_path.name == "glyconmr_len123_independent_validation.sdf":
            continue
        text = sdf_path.read_text(encoding="utf-8", errors="ignore")
        for block in text.split("$$$$"):
            if not block.strip():
                continue
            sample_name, props = parse_sdf_properties(block)
            heavy_atom_names = ast.literal_eval(props.get("heavy_atom_names", "[]"))
            for task in ("13C", "1H"):
                spectrum_text = props.get(f"{task} Spectrum")
                if not spectrum_text:
                    continue
                spectrum = ast.literal_eval(spectrum_text)
                target_atom_indices = sorted(int(idx) for idx in spectrum.keys())
                mapping[(sample_name, task)] = readable_residue_atom_labels(task, heavy_atom_names, target_atom_indices)
    return mapping


def sdf_sample_names(split: str) -> list[str]:
    suffix = {"mono": "mo", "di": "di", "tri": "tri"}[split]
    sdf_path = DATASET_DIR / f"glyconmr_len123_independent_validation_{suffix}.sdf"
    names = []
    if not sdf_path.exists():
        return names
    for block in sdf_path.read_text(encoding="utf-8", errors="ignore").split("$$$$"):
        if block.strip():
            names.append(block.strip().splitlines()[0].strip())
    return names


def representative_attention_names(label_mapping: dict[tuple[str, str], list[str]]) -> dict[str, str]:
    selected: dict[str, str] = {
        split: name
        for split, name in ATTENTION_REPRESENTATIVE_NAMES.items()
        if (name, "13C") in label_mapping and (name, "1H") in label_mapping
    }
    for split, required_residue, min_labels in [("di", "R2-", 12), ("tri", "R3-", 16)]:
        if split in selected:
            continue
        for name in sdf_sample_names(split):
            labels_c = label_mapping.get((name, "13C"), [])
            labels_h = label_mapping.get((name, "1H"), [])
            if (
                len(labels_c) >= min_labels
                and len(labels_h) >= min_labels
                and any(label.startswith(required_residue) for label in labels_c)
                and any(label.startswith(required_residue) for label in labels_h)
            ):
                selected[split] = name
                break
    return selected


def add_site_column(df: pd.DataFrame) -> pd.DataFrame:
    if "site" in df.columns and df["site"].notna().any():
        return df.copy()
    mapping = build_site_mapping()
    sites = []
    for row in df.itertuples(index=False):
        atom_idx = getattr(row, "atom_idx", None)
        if atom_idx is None:
            atom_idx = getattr(row, "atom_order", None)
        item_sites = mapping.get((row.sample_name, row.task))
        if item_sites is None or atom_idx is None or int(atom_idx) >= len(item_sites):
            sites.append(None)
        else:
            sites.append(item_sites[int(atom_idx)])
    out = df.copy()
    out["site"] = sites
    return out


def paired_context_improvement(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_cols = ["task", "split", "sample_idx", "sample_name", "atom_idx", "site"]
    wide = (
        points.pivot_table(index=key_cols, columns="display", values="abs_error", aggfunc="first")
        .reset_index()
        .dropna(subset=[CONTEXT_MODEL, *BASELINES])
    )

    rows = []
    summary_rows = []
    for baseline in BASELINES:
        delta = wide[baseline] - wide[CONTEXT_MODEL]
        for row, value in zip(wide.to_dict("records"), delta, strict=True):
            rows.append(
                {
                    **{col: row[col] for col in key_cols},
                    "baseline": baseline,
                    "baseline_label": f"{MODEL_LABELS[baseline]} - Cross-attn",
                    "delta_abs_error": float(value),
                    "baseline_abs_error": float(row[baseline]),
                    "context_abs_error": float(row[CONTEXT_MODEL]),
                }
            )

        for task, sites in SITE_ORDER.items():
            for site in sites:
                vals = delta[(wide["task"] == task) & (wide["site"] == site)].to_numpy(float)
                mean, low, high = bootstrap_mean_ci(vals, seed=100 + len(summary_rows))
                summary_rows.append(
                    {
                        "baseline": baseline,
                        "baseline_label": f"{MODEL_LABELS[baseline]} - Cross-attn",
                        "task": task,
                        "site": site,
                        "n": int(len(vals)),
                        "mean_delta_abs_error": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "median_delta_abs_error": float(np.median(vals)) if len(vals) else float("nan"),
                        "positive_fraction": float(np.mean(vals > 0)) if len(vals) else float("nan"),
                    }
                )

    long_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)
    long_df.to_csv(OUT_DIR / "proof_paired_context_error_improvement_points.csv", index=False)
    summary_df.to_csv(OUT_DIR / "proof_paired_context_error_improvement_summary.csv", index=False)
    return long_df, summary_df


def draw_paired_site_improvement(long_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.1), constrained_layout=False)
    panel_labels = ["a", "b", "c", "d"]
    baseline_colors = {
        "EquiformerV3-full bs18": COLORS["pink"],
        "GeqShift-O3Transformer": COLORS["blue"],
    }
    for row_idx, task in enumerate(["13C", "1H"]):
        for col_idx, baseline in enumerate(BASELINES):
            ax = axes[row_idx, col_idx]
            sites = SITE_ORDER[task]
            values = [
                long_df.loc[
                    (long_df["task"] == task)
                    & (long_df["site"] == site)
                    & (long_df["baseline"] == baseline),
                    "delta_abs_error",
                ].to_numpy(float)
                for site in sites
            ]
            color = baseline_colors[baseline]
            box = ax.boxplot(
                values,
                positions=np.arange(1, len(sites) + 1),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": COLORS["dark"], "linewidth": 1.1},
                boxprops={"linewidth": 0.95, "edgecolor": COLORS["dark"]},
                whiskerprops={"linewidth": 0.85, "color": COLORS["dark"]},
                capprops={"linewidth": 0.85, "color": COLORS["dark"]},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)

            sub = summary_df[(summary_df["task"] == task) & (summary_df["baseline"] == baseline)].set_index("site")
            means = sub.loc[sites, "mean_delta_abs_error"].to_numpy(float)
            low = sub.loc[sites, "ci95_low"].to_numpy(float)
            high = sub.loc[sites, "ci95_high"].to_numpy(float)
            x = np.arange(1, len(sites) + 1)
            ax.errorbar(
                x,
                means,
                yerr=np.vstack([means - low, high - means]),
                fmt="o",
                markersize=4.2,
                color=COLORS["dark"],
                ecolor=COLORS["dark"],
                elinewidth=1.0,
                capsize=2.5,
                zorder=4,
            )
            ax.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0, alpha=0.7)
            ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
            ax.set_axisbelow(True)
            ax.set_xticks(x)
            ax.set_xticklabels(sites)
            ax.set_ylabel("Delta |error| (ppm)" if col_idx == 0 else "")
            ax.set_ylim((-1.25, 2.05) if task == "13C" else (-0.095, 0.095))
            task_label = r"$^{13}$C" if task == "13C" else r"$^1$H"
            ax.set_title(f"{task_label}: {MODEL_LABELS[baseline]} - Cross-attn", fontweight="bold", pad=7)
            ax.text(
                -0.16,
                1.05,
                panel_labels[row_idx * 2 + col_idx],
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
    handles = [
        Line2D([0], [0], color=COLORS["dark"], marker="o", linestyle="none", markersize=4.5, label="Mean +/- bootstrap 95% CI"),
        Line2D([0], [0], color=COLORS["dark"], linestyle="--", linewidth=1.0, label="No paired gain"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=9.0)
    fig.suptitle("Paired atom-level evidence for context benefit", fontsize=12.8, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.13, wspace=0.18, hspace=0.34)
    save_figure(fig, "proof_01_paired_site_error_improvement")


def draw_c1_vs_other_sites(summary_df: pd.DataFrame) -> None:
    apply_style()
    rows = []
    for baseline in BASELINES:
        for task in ("13C", "1H"):
            site_col = "C1" if task == "13C" else "H1"
            other_sites = [site for site in SITE_ORDER[task] if site != site_col]
            sub = summary_df[(summary_df["baseline"] == baseline) & (summary_df["task"] == task)]
            c1 = sub[sub["site"] == site_col].iloc[0]
            rows.append(
                {
                    "baseline": baseline,
                    "group": f"{task} {site_col}",
                    "task": task,
                    "mean": c1["mean_delta_abs_error"],
                    "low": c1["ci95_low"],
                    "high": c1["ci95_high"],
                }
            )
            vals = []
            for site in other_sites:
                site_row = sub[sub["site"] == site].iloc[0]
                vals.append(float(site_row["mean_delta_abs_error"]))
            mean = float(np.mean(vals))
            low, high = np.percentile(vals, [2.5, 97.5])
            rows.append(
                {
                    "baseline": baseline,
                    "group": f"{task} other sites",
                    "task": task,
                    "mean": mean,
                    "low": float(low),
                    "high": float(high),
                }
            )
    plot_df = pd.DataFrame(rows)
    plot_df.to_csv(OUT_DIR / "proof_c1_vs_other_sites_context_gain_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.85), constrained_layout=False)
    for ax, baseline, panel in zip(axes, BASELINES, ["a", "b"], strict=True):
        sub = plot_df[plot_df["baseline"] == baseline]
        labels = ["13C C1", "13C other sites", "1H H1", "1H other sites"]
        y = np.arange(len(labels))[::-1]
        color_map = [COLORS["pink"], "#F5CFC9", COLORS["blue"], "#C9DCE8"]
        for i, label in enumerate(labels):
            row = sub[sub["group"] == label].iloc[0]
            yy = y[i]
            ax.barh(yy, row["mean"], height=0.52, color=color_map[i], edgecolor=COLORS["dark"], linewidth=0.7)
            ax.errorbar(
                row["mean"],
                yy,
                xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
                fmt="none",
                color=COLORS["dark"],
                elinewidth=1.0,
                capsize=2.5,
            )
        ax.axvline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.grid(True, axis="x", color=COLORS["grid"], linestyle=":", linewidth=0.85)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean paired Delta |error| (ppm)")
        ax.set_title(f"{MODEL_LABELS[baseline]} - Cross-attn", fontweight="bold", pad=7)
        ax.text(-0.13, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    fig.suptitle("Anomeric-site context gain compared with other ring sites", fontsize=12.4, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.17, wspace=0.28)
    save_figure(fig, "proof_02_c1_vs_other_sites_context_gain")


def split_context_summary(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for task, site in [("13C", "C1"), ("1H", "H1")]:
        for split in SPLIT_ORDER:
            sub = points[(points["task"] == task) & (points["site"] == site) & (points["split"] == split)]
            for model in [CONTEXT_MODEL, *BASELINES]:
                vals = sub.loc[sub["display"] == model, "abs_error"].to_numpy(float)
                mean, low, high = bootstrap_mean_ci(vals, seed=300 + len(rows))
                rows.append(
                    {
                        "task": task,
                        "site": site,
                        "split": split,
                        "display": model,
                        "model_label": MODEL_LABELS[model],
                        "n": int(len(vals)),
                        "mean_abs_error": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    summary = pd.DataFrame(rows)

    adv_rows = []
    for task, site in [("13C", "C1"), ("1H", "H1")]:
        for split in SPLIT_ORDER:
            sub = summary[(summary["task"] == task) & (summary["site"] == site) & (summary["split"] == split)]
            context_mean = float(sub.loc[sub["display"] == CONTEXT_MODEL, "mean_abs_error"].iloc[0])
            baseline_mean = float(sub.loc[sub["display"].isin(BASELINES), "mean_abs_error"].mean())
            n = int(sub.loc[sub["display"] == CONTEXT_MODEL, "n"].iloc[0])
            adv_rows.append(
                {
                    "task": task,
                    "site": site,
                    "split": split,
                    "n": n,
                    "baseline_mean_abs_error": baseline_mean,
                    "context_mean_abs_error": context_mean,
                    "baseline_minus_context": baseline_mean - context_mean,
                }
            )
    advantage = pd.DataFrame(adv_rows)
    summary.to_csv(OUT_DIR / "proof_c1_h1_split_model_summary.csv", index=False)
    advantage.to_csv(OUT_DIR / "proof_c1_h1_split_advantage_summary.csv", index=False)
    return summary, advantage


def draw_context_split(summary: pd.DataFrame, advantage: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.75), constrained_layout=False)
    offsets = {CONTEXT_MODEL: -0.22, "EquiformerV3-full bs18": 0.0, "GeqShift-O3Transformer": 0.22}
    model_colors = {CONTEXT_MODEL: COLORS["pink"], "EquiformerV3-full bs18": COLORS["yellow"], "GeqShift-O3Transformer": COLORS["blue"]}
    panels = [(axes[0], "13C", "C1", "a"), (axes[1], "1H", "H1", "b")]
    for ax, task, site, panel in panels:
        x_base = np.arange(len(SPLIT_ORDER), dtype=float)
        for model in [CONTEXT_MODEL, *BASELINES]:
            sub = (
                summary[(summary["task"] == task) & (summary["site"] == site) & (summary["display"] == model)]
                .set_index("split")
                .loc[SPLIT_ORDER]
            )
            x = x_base + offsets[model]
            y = sub["mean_abs_error"].to_numpy(float)
            low = sub["ci95_low"].to_numpy(float)
            high = sub["ci95_high"].to_numpy(float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - low, high - y]),
                fmt="o",
                markersize=5.0,
                color=model_colors[model],
                markeredgecolor=COLORS["dark"],
                markeredgewidth=0.7,
                ecolor=model_colors[model],
                elinewidth=1.05,
                capsize=2.5,
                label=MODEL_LABELS[model],
                zorder=3,
            )
            ax.plot(x, y, color=model_colors[model], linewidth=1.2, alpha=0.82)
        ax.set_xticks(x_base)
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER])
        ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
        ax.set_axisbelow(True)
        ax.set_ylabel("MAE (ppm)")
        task_label = r"$^{13}$C" if task == "13C" else r"$^1$H"
        ax.set_title(f"{task_label} {site} by glycan-length split", fontweight="bold", pad=7)
        y_top = ax.get_ylim()[1]
        for split_idx, split in enumerate(SPLIT_ORDER):
            row = advantage[(advantage["task"] == task) & (advantage["site"] == site) & (advantage["split"] == split)].iloc[0]
            ax.text(split_idx, y_top * 0.955, f"n={int(row['n'])}", ha="center", va="top", fontsize=8.0, color="#333333")
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=9.0)
    fig.suptitle("Context effect by glycan-length split", fontsize=12.7, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.23, wspace=0.22)
    save_figure(fig, "proof_03_context_strength_c1_h1_by_split")


def load_torch_stack() -> dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    sys.path.append(str(ROOT))
    from module.custom_data import load_train_test_data
    from module.custom_models import build_nmr_model
    from module.custom_tasks.nmr_shift import build_model_kwargs, prepare_dataloader

    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "load_train_test_data": load_train_test_data,
        "build_nmr_model": build_nmr_model,
        "build_model_kwargs": build_model_kwargs,
        "prepare_dataloader": prepare_dataloader,
    }


def target_mask(torch: Any, data: Any) -> Any:
    carbon_mask = data.x[:, 0] == 2.0
    nmr_mask = data.x[:, -1] > -0.5
    return nmr_mask.logical_and(carbon_mask)


def load_args(run_dir: Path) -> argparse.Namespace:
    with (run_dir / "config.json").open() as handle:
        return argparse.Namespace(**json.load(handle))


def checkpoint_path(run_dir: Path, task: str, epoch: int, fold: int) -> Path:
    path = run_dir / "checkpoints" / f"fold_{fold}" / task / f"_checkpoint_epoch_{epoch}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@contextmanager
def cross_attention_patch(model: Any, mode: str, token_index: int | None = None):
    original = model._cross_attention_by_graph

    if mode == "full":
        yield
        return

    def zero_context(*, eq_scalar, mono_tokens, atom_batch, mono_batch):
        return eq_scalar.new_zeros(eq_scalar.shape)

    def zero_token(*, eq_scalar, mono_tokens, atom_batch, mono_batch):
        patched_tokens = mono_tokens.clone()
        if atom_batch is None:
            graph_count = 1
        else:
            graph_count = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        for graph_index in range(graph_count):
            mono_indices = (mono_batch == graph_index).nonzero(as_tuple=False).flatten()
            if token_index is not None and token_index < mono_indices.numel():
                patched_tokens[mono_indices[token_index]] = 0.0
        return original(
            eq_scalar=eq_scalar,
            mono_tokens=patched_tokens,
            atom_batch=atom_batch,
            mono_batch=mono_batch,
        )

    if mode == "no_context":
        model._cross_attention_by_graph = zero_context
    elif mode == "token_zero":
        if token_index is None:
            raise ValueError("token_index is required for token_zero mode.")
        model._cross_attention_by_graph = zero_token
    else:
        raise ValueError(f"Unknown patch mode: {mode}")
    try:
        yield
    finally:
        model._cross_attention_by_graph = original


@contextmanager
def capture_cross_attention_weights(model: Any, storage: list[np.ndarray]):
    original = model._cross_attention_by_graph

    def capture(*, eq_scalar, mono_tokens, atom_batch, mono_batch):
        if atom_batch is None:
            atom_batch = eq_scalar.new_zeros(eq_scalar.shape[0], dtype=mono_batch.dtype)

        context = eq_scalar.new_zeros(eq_scalar.shape)
        graph_count = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        query_all = model.eq_norm(eq_scalar)
        mono_all = model.mono_norm(mono_tokens)
        for graph_index in range(graph_count):
            atom_mask = atom_batch == graph_index
            if not bool(atom_mask.any()):
                continue
            mono_mask = mono_batch == graph_index
            if not bool(mono_mask.any()):
                continue
            query = query_all[atom_mask].unsqueeze(0)
            key_value = mono_all[mono_mask].unsqueeze(0)
            attended, weights = model.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=True,
                average_attn_weights=False,
            )
            context[atom_mask] = attended.squeeze(0)
            # weights: [batch=1, heads, target_atoms, context_tokens]
            storage.append(weights.detach().cpu().numpy()[0].mean(axis=0))
        return context

    model._cross_attention_by_graph = capture
    try:
        yield
    finally:
        model._cross_attention_by_graph = original


def forward_model(model: Any, data: Any) -> Any:
    return model(
        x=data.x[:, 0:2].long(),
        pos=data.pos.float(),
        edge_index=data.edge_index,
        edge_attr=data.edge_attr.long(),
        batch=data.batch,
    )


def load_task_data(stack: dict[str, Any], task: str, mean_batch_size: int) -> tuple[Any, Any, dict[str, Any]]:
    torch = stack["torch"]
    load_train_test_data = stack["load_train_test_data"]
    prepare_dataloader = stack["prepare_dataloader"]
    train_set, test_mo, test_di, test_tri = load_train_test_data(
        DATASET_DIR / f"train_data_{task}.pickle",
        DATASET_DIR / f"test_data_{task}_mo.pickle",
        DATASET_DIR / f"test_data_{task}_di.pickle",
        DATASET_DIR / f"test_data_{task}_tri.pickle",
    )
    nmrs = []
    for data in prepare_dataloader(train_set, mean_batch_size):
        nmrs.append(data.x[:, -1][target_mask(torch, data)])
    values = torch.cat(nmrs)
    return values.mean(), values.std(), {"mono": test_mo, "di": test_di, "tri": test_tri}


def predict_molecule(
    stack: dict[str, Any],
    model: Any,
    loader_dict: dict[str, Any],
    mean: Any,
    std: Any,
    device: Any,
    eval_batch_size: int,
    mode: str,
    token_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    torch = stack["torch"]
    DataLoader = stack["DataLoader"]
    n_confs = int(loader_dict["n_mols"])
    n_nmr = int(loader_dict["n_nmr"])
    loader = DataLoader(loader_dict["loader"].dataset, batch_size=eval_batch_size, shuffle=False)
    true_values = None
    conf_outputs = []
    with torch.no_grad(), cross_attention_patch(model, mode, token_index=token_index):
        for data in loader:
            data = data.to(device)
            mask = target_mask(torch, data)
            nmr_true = data.x[:, -1][mask]
            out = forward_model(model, data)
            out_masked = out[mask] * std + mean
            if true_values is None:
                true_values = nmr_true[:n_nmr].detach().cpu().numpy()
            conf_outputs.append(out_masked.detach().flatten().cpu())
            del data, out, out_masked
    pred = torch.cat(conf_outputs).reshape(n_confs, n_nmr).T.mean(dim=1).numpy()
    if true_values is None:
        true_values = np.empty(0, dtype=float)
    return true_values.astype(float), pred.astype(float)


def unique_site_labels(sites: list[str | None]) -> list[str]:
    counts: dict[str, int] = {}
    out = []
    for idx, site in enumerate(sites):
        label = site or f"atom{idx + 1}"
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1:
            out.append(label)
        else:
            out.append(f"{label}.{counts[label]}")
    return out


def capture_attention_for_molecule(
    stack: dict[str, Any],
    model: Any,
    loader_dict: dict[str, Any],
    task: str,
    device: Any,
    max_conformers: int | None = None,
) -> tuple[np.ndarray, list[str], str, set[str]]:
    torch = stack["torch"]
    DataLoader = stack["DataLoader"]
    dataset = loader_dict["loader"].dataset
    if max_conformers is not None:
        dataset = list(dataset)[:max_conformers]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    sample_name = loader_dict.get("name", "")
    labels = build_atom_label_mapping().get((sample_name, task), [])
    highlighted_labels = glycosidic_linkage_labels(sample_name, task)
    captured = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            storage: list[np.ndarray] = []
            with capture_cross_attention_weights(model, storage):
                _ = forward_model(model, data)
            if not storage:
                del data
                continue
            weights = storage[0]
            mask = target_mask(torch, data).detach().cpu().numpy().astype(bool)
            captured.append(weights[mask])
            del data
    if not captured:
        return np.empty((0, 0)), labels, sample_name, highlighted_labels
    mean_weights = np.mean(np.stack(captured, axis=0), axis=0)
    return mean_weights, labels[: mean_weights.shape[0]], sample_name, highlighted_labels


def run_attention_diagnostic(
    run_dir: Path,
    epoch: int,
    fold: int,
    gpu_id: int,
    mean_batch_size: int,
    sample_name: str | None = None,
) -> pd.DataFrame:
    stack = load_torch_stack()
    torch = stack["torch"]
    build_nmr_model = stack["build_nmr_model"]
    build_model_kwargs = stack["build_model_kwargs"]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    target_name = sample_name
    if target_name is None and FIG4_DECOMP.exists():
        target_name = str(pd.read_csv(FIG4_DECOMP)["sample_name"].iloc[0])

    rows = []
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str], str, set[str]]] = {}
    label_mapping = build_atom_label_mapping()
    preferred_names = representative_attention_names(label_mapping)
    for task in ("13C", "1H"):
        _mean, _std, split_data = load_task_data(stack, task, mean_batch_size)
        args = load_args(run_dir)
        kwargs = build_model_kwargs(args)
        architecture = kwargs.pop("architecture", "equiformer_v3")
        model = build_nmr_model(architecture=architecture, **kwargs).to(device)
        state = torch.load(checkpoint_path(run_dir, task, epoch, fold), map_location="cpu")
        model.load_state_dict(state)
        del state
        model.eval()

        for split in ("di", "tri"):
            molecules = split_data[split]
            selected = None
            preferred_name = preferred_names.get(split)
            if preferred_name:
                for item in molecules:
                    if item.get("name", "") == preferred_name:
                        selected = item
                        break
            if split == "tri" and target_name:
                for item in molecules:
                    if item.get("name", "") == target_name:
                        selected = item
                        break
            if selected is None:
                if split == "tri":
                    selected = molecules[11] if len(molecules) > 11 else molecules[0]
                else:
                    selected = max(
                        molecules,
                        key=lambda item: (
                            any(label.startswith("R2-") for label in label_mapping.get((item.get("name", ""), task), [])),
                            len(label_mapping.get((item.get("name", ""), task), [])),
                        ),
                    )

            matrix, labels, used_name, highlighted_labels = capture_attention_for_molecule(stack, model, selected, task, device)
            matrices[(split, task)] = (matrix, labels, used_name, highlighted_labels)
            for atom_order, label in enumerate(labels):
                for token_idx in range(matrix.shape[1]):
                    rows.append(
                        {
                            "split": split,
                            "task": task,
                            "sample_name": used_name,
                            "atom_order": atom_order,
                            "site_label": label,
                            "is_glycosidic_linkage_position": label in highlighted_labels,
                            "context_token": token_idx + 1,
                            "mean_attention_weight": float(matrix[atom_order, token_idx]),
                        }
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "proof_cross_attention_weights_representative_molecule.csv", index=False)
    draw_attention_heatmap(matrices)
    return out


def run_context_ablation(
    run_dir: Path,
    epoch: int,
    fold: int,
    gpu_id: int,
    mean_batch_size: int,
    eval_batch_size: int,
    max_molecules_per_split: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stack = load_torch_stack()
    torch = stack["torch"]
    build_nmr_model = stack["build_nmr_model"]
    build_model_kwargs = stack["build_model_kwargs"]
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    site_mapping = build_site_mapping()

    rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for task in ("13C", "1H"):
        print(f"Running context ablation for {task}.", flush=True)
        mean, std, split_data = load_task_data(stack, task, mean_batch_size)
        mean = mean.to(device)
        std = std.to(device)
        args = load_args(run_dir)
        kwargs = build_model_kwargs(args)
        architecture = kwargs.pop("architecture", "equiformer_v3")
        model = build_nmr_model(architecture=architecture, **kwargs).to(device)
        state = torch.load(checkpoint_path(run_dir, task, epoch, fold), map_location="cpu")
        model.load_state_dict(state)
        del state
        model.eval()

        for split, molecules in split_data.items():
            expected_tokens = {"mono": 1, "di": 2, "tri": 3}[split]
            selected = molecules[:max_molecules_per_split] if max_molecules_per_split else molecules
            for mol_index, loader_dict in enumerate(selected):
                sample_name = loader_dict.get("name", "")
                true_values, full_pred = predict_molecule(
                    stack, model, loader_dict, mean, std, device, eval_batch_size, mode="full"
                )
                _, no_context_pred = predict_molecule(
                    stack, model, loader_dict, mean, std, device, eval_batch_size, mode="no_context"
                )
                token_preds = []
                for token_index in range(expected_tokens):
                    _, token_pred = predict_molecule(
                        stack,
                        model,
                        loader_dict,
                        mean,
                        std,
                        device,
                        eval_batch_size,
                        mode="token_zero",
                        token_index=token_index,
                    )
                    token_preds.append(token_pred)
                token_stack = np.vstack(token_preds) if token_preds else np.empty((0, len(full_pred)))
                max_token_shift = np.max(np.abs(token_stack - full_pred), axis=0) if len(token_preds) else np.zeros_like(full_pred)
                mean_token_shift = np.mean(np.abs(token_stack - full_pred), axis=0) if len(token_preds) else np.zeros_like(full_pred)

                site_list = site_mapping.get((sample_name, task), [])
                for atom_idx, (true, full, no_ctx, max_shift, mean_shift) in enumerate(
                    zip(true_values, full_pred, no_context_pred, max_token_shift, mean_token_shift, strict=False)
                ):
                    site = site_list[atom_idx] if atom_idx < len(site_list) else None
                    if site is None:
                        continue
                    full_error = float(full - true)
                    no_context_error = float(no_ctx - true)
                    rows.append(
                        {
                            "task": task,
                            "split": split,
                            "sample_idx": mol_index,
                            "sample_name": sample_name,
                            "atom_idx": atom_idx,
                            "site": site,
                            "true": float(true),
                            "full_pred": float(full),
                            "no_context_pred": float(no_ctx),
                            "full_error": full_error,
                            "no_context_error": no_context_error,
                            "full_abs_error": abs(full_error),
                            "no_context_abs_error": abs(no_context_error),
                            "delta_abs_error_no_context_minus_full": abs(no_context_error) - abs(full_error),
                            "prediction_shift_no_context": abs(float(no_ctx - full)),
                            "max_token_ablation_shift": float(max_shift),
                            "mean_token_ablation_shift": float(mean_shift),
                            "n_context_tokens": expected_tokens,
                        }
                    )
                    for token_index, token_pred in enumerate(token_preds):
                        pred = float(token_pred[atom_idx])
                        token_rows.append(
                            {
                                "task": task,
                                "split": split,
                                "sample_idx": mol_index,
                                "sample_name": sample_name,
                                "atom_idx": atom_idx,
                                "site": site,
                                "token_index": token_index,
                                "true": float(true),
                                "full_pred": float(full),
                                "token_zero_pred": pred,
                                "token_prediction_shift": abs(pred - float(full)),
                            }
                        )
            print(f"  {task} {split}: processed {len(selected)} molecules.", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ablation_df = pd.DataFrame(rows)
    token_df = pd.DataFrame(token_rows)
    suffix = "_smoke" if max_molecules_per_split else ""
    ablation_df.to_csv(OUT_DIR / f"proof_context_ablation_atom_level{suffix}.csv", index=False)
    token_df.to_csv(OUT_DIR / f"proof_context_token_ablation_atom_level{suffix}.csv", index=False)
    return ablation_df, token_df


def summarize_ablation(ablation_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for task, sites in SITE_ORDER.items():
        for site in sites:
            vals = ablation_df.loc[(ablation_df["task"] == task) & (ablation_df["site"] == site), "delta_abs_error_no_context_minus_full"].to_numpy(float)
            shifts = ablation_df.loc[(ablation_df["task"] == task) & (ablation_df["site"] == site), "prediction_shift_no_context"].to_numpy(float)
            mean, low, high = bootstrap_mean_ci(vals, seed=501 + len(rows))
            shift_mean, shift_low, shift_high = bootstrap_mean_ci(shifts, seed=701 + len(rows))
            rows.append(
                {
                    "task": task,
                    "site": site,
                    "n": int(len(vals)),
                    "mean_error_gain_full_vs_no_context": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_fraction": float(np.mean(vals > 0)) if len(vals) else float("nan"),
                    "mean_prediction_shift": shift_mean,
                    "shift_ci95_low": shift_low,
                    "shift_ci95_high": shift_high,
                }
            )
    site_summary = pd.DataFrame(rows)

    split_rows = []
    for task, site in [("13C", "C1"), ("1H", "H1")]:
        for split in SPLIT_ORDER:
            sub = ablation_df[(ablation_df["task"] == task) & (ablation_df["site"] == site) & (ablation_df["split"] == split)]
            vals = sub["delta_abs_error_no_context_minus_full"].to_numpy(float)
            shifts = sub["prediction_shift_no_context"].to_numpy(float)
            mean, low, high = bootstrap_mean_ci(vals, seed=901 + len(split_rows))
            shift_mean, shift_low, shift_high = bootstrap_mean_ci(shifts, seed=1001 + len(split_rows))
            split_rows.append(
                {
                    "task": task,
                    "site": site,
                    "split": split,
                    "n": int(len(vals)),
                    "mean_error_gain_full_vs_no_context": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_fraction": float(np.mean(vals > 0)) if len(vals) else float("nan"),
                    "mean_prediction_shift": shift_mean,
                    "shift_ci95_low": shift_low,
                    "shift_ci95_high": shift_high,
                }
            )
    split_summary = pd.DataFrame(split_rows)

    token_rows = []
    for task, sites in SITE_ORDER.items():
        for site in sites:
            sub = ablation_df[(ablation_df["task"] == task) & (ablation_df["site"] == site)]
            vals = sub["max_token_ablation_shift"].to_numpy(float)
            mean, low, high = bootstrap_mean_ci(vals, seed=1201 + len(token_rows))
            token_rows.append(
                {
                    "task": task,
                    "site": site,
                    "n": int(len(vals)),
                    "mean_max_token_prediction_shift": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    token_summary = pd.DataFrame(token_rows)
    site_summary.to_csv(OUT_DIR / "proof_context_ablation_site_summary.csv", index=False)
    split_summary.to_csv(OUT_DIR / "proof_context_ablation_c1_h1_split_summary.csv", index=False)
    token_summary.to_csv(OUT_DIR / "proof_context_token_ablation_site_summary.csv", index=False)
    return site_summary, split_summary, token_summary


def draw_no_context_site_gain(site_summary: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.9), constrained_layout=False)
    for ax, task, panel in [(axes[0], "13C", "a"), (axes[1], "1H", "b")]:
        sites = SITE_ORDER[task]
        sub = site_summary[site_summary["task"] == task].set_index("site").loc[sites]
        x = np.arange(len(sites))
        means = sub["mean_error_gain_full_vs_no_context"].to_numpy(float)
        low = sub["ci95_low"].to_numpy(float)
        high = sub["ci95_high"].to_numpy(float)
        colors = [COLORS["pink"] if site.endswith("1") else COLORS["yellow"] for site in sites] if task == "13C" else [COLORS["blue"] if site.endswith("1") else "#D6E5ED" for site in sites]
        ax.bar(x, means, color=colors, edgecolor=COLORS["dark"], linewidth=0.75, width=0.62)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="none",
            ecolor=COLORS["dark"],
            elinewidth=1.0,
            capsize=2.5,
        )
        ax.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
        ax.set_axisbelow(True)
        ax.set_ylabel("Full-context gain vs no-context (ppm)")
        task_label = r"$^{13}$C" if task == "13C" else r"$^1$H"
        ax.set_title(f"{task_label}: same-model context ablation", fontweight="bold", pad=7)
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    fig.suptitle("Causal ablation: removing cross-attention context", fontsize=12.6, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.16, wspace=0.22)
    save_figure(fig, "proof_04_same_model_no_context_ablation_by_site")


def draw_no_context_split(split_summary: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.05), constrained_layout=False)
    for ax, task, site, panel, color in [
        (axes[0], "13C", "C1", "a", COLORS["pink"]),
        (axes[1], "1H", "H1", "b", COLORS["blue"]),
    ]:
        sub = split_summary[(split_summary["task"] == task) & (split_summary["site"] == site)].set_index("split").loc[SPLIT_ORDER]
        x = np.arange(len(SPLIT_ORDER))
        means = sub["mean_error_gain_full_vs_no_context"].to_numpy(float)
        low = sub["ci95_low"].to_numpy(float)
        high = sub["ci95_high"].to_numpy(float)
        ax.bar(x, means, color=color, alpha=0.62, edgecolor=COLORS["dark"], linewidth=0.75, width=0.58)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="none",
            ecolor=COLORS["dark"],
            elinewidth=1.0,
            capsize=2.5,
        )
        ax.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0, alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER])
        ax.set_ylabel("Full-context gain vs no-context (ppm)")
        ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
        ax.set_axisbelow(True)
        task_label = r"$^{13}$C" if task == "13C" else r"$^1$H"
        ax.set_title(f"{task_label} {site} by split", fontweight="bold", pad=7)
        ymax = ax.get_ylim()[1]
        ymin = ax.get_ylim()[0]
        for idx, split in enumerate(SPLIT_ORDER):
            row = sub.loc[split]
            y_text = ymax - (ymax - ymin) * 0.07
            ax.text(idx, y_text, f"n={int(row['n'])}", ha="center", va="top", fontsize=8.1)
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    fig.suptitle("Does context ablation strengthen with glycan length?", fontsize=12.6, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.17, wspace=0.24)
    save_figure(fig, "proof_05_same_model_no_context_ablation_c1_h1_by_split")


def draw_token_ablation_shift(token_summary: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 3.9), constrained_layout=False)
    for ax, task, panel in [(axes[0], "13C", "a"), (axes[1], "1H", "b")]:
        sites = SITE_ORDER[task]
        sub = token_summary[token_summary["task"] == task].set_index("site").loc[sites]
        x = np.arange(len(sites))
        means = sub["mean_max_token_prediction_shift"].to_numpy(float)
        low = sub["ci95_low"].to_numpy(float)
        high = sub["ci95_high"].to_numpy(float)
        color = COLORS["green"] if task == "13C" else COLORS["blue"]
        ax.bar(x, means, color=color, alpha=0.68, edgecolor=COLORS["dark"], linewidth=0.75, width=0.62)
        ax.errorbar(
            x,
            means,
            yerr=np.vstack([means - low, high - means]),
            fmt="none",
            ecolor=COLORS["dark"],
            elinewidth=1.0,
            capsize=2.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
        ax.set_axisbelow(True)
        ax.set_ylabel("Max token-ablation shift (ppm)")
        task_label = r"$^{13}$C" if task == "13C" else r"$^1$H"
        ax.set_title(f"{task_label}: leave-one-context-token-out", fontweight="bold", pad=7)
        ax.text(-0.12, 1.05, panel, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    fig.suptitle("Prediction sensitivity to individual context tokens", fontsize=12.6, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.16, wspace=0.22)
    save_figure(fig, "proof_06_context_token_ablation_shift_by_site")


def draw_attention_heatmap(matrices: dict[tuple[str, str], tuple[np.ndarray, list[str], str, set[str]]]) -> None:
    if not matrices:
        return
    apply_style()
    cmap = LinearSegmentedColormap.from_list(
        "context_attention",
        ["#FFF8E3", "#E3EDE0", "#ABD3E1", "#92B4C8"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.6), constrained_layout=False)
    panel_map = {("di", "13C"): "a", ("di", "1H"): "b", ("tri", "13C"): "c", ("tri", "1H"): "d"}
    image = None
    for row_idx, split in enumerate(("di", "tri")):
        for col_idx, task in enumerate(("13C", "1H")):
            ax = axes[row_idx, col_idx]
            matrix, labels, sample_name, highlighted_labels = matrices.get((split, task), (np.empty((0, 0)), [], "", set()))
            if matrix.size == 0:
                ax.axis("off")
                continue
            image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
            ax.set_xticks(np.arange(matrix.shape[1]))
            ax.set_xticklabels([f"Token {idx + 1}" for idx in range(matrix.shape[1])])
            ax.set_yticks(np.arange(matrix.shape[0]))
            ax.set_yticklabels(labels, fontsize=7.9)
            for tick in ax.get_yticklabels():
                if tick.get_text() in highlighted_labels:
                    tick.set_color(COLORS["bad"])
                    tick.set_fontweight("bold")
            ax.set_xlabel("Glycan context token")
            ax.set_ylabel("Target atom")
            split_label = "Di-glycan" if split == "di" else "Tri-glycan"
            task_label = "13C" if task == "13C" else "1H"
            ax.set_title(f"{split_label} {task_label}", fontweight="bold", pad=7)
            ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            if matrix.shape[0] <= 24 and matrix.shape[1] <= 4:
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        color = "white" if matrix[i, j] > 0.52 else COLORS["dark"]
                        ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.3, color=color)
            ax.text(
                -0.14,
                1.04,
                panel_map[(split, task)],
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                va="bottom",
            )
            for spine in ax.spines.values():
                spine.set_color(COLORS["dark"])
                spine.set_linewidth(1.1)
            ax.text(
                0.0,
                -0.16,
                sample_name,
                transform=ax.transAxes,
                fontsize=6.9,
                color="#444444",
                ha="left",
                va="top",
            )
    if image is None:
        plt.close(fig)
        return
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
    cbar.set_label("Mean attention weight")
    fig.suptitle("Representative di- and tri-glycan cross-attention maps", fontsize=12.4, fontweight="bold", y=0.992)
    fig.text(
        0.075,
        0.025,
        "Red y-axis labels indicate carbon positions directly involved in glycosidic C-O-C linkages.",
        fontsize=8.2,
        color=COLORS["bad"],
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.075, right=0.90, top=0.925, bottom=0.105, wspace=0.34, hspace=0.38)
    save_figure(fig, "proof_08_representative_cross_attention_weights")


def draw_fig4_case_decomposition() -> None:
    if not FIG4_DECOMP.exists():
        return
    apply_style()
    df = pd.read_csv(FIG4_DECOMP)
    labels = df["Atom label"].astype(str).tolist()
    x = np.arange(len(df))
    base_abs = df["Base abs residual"].to_numpy(float)
    total_abs = df["Total abs residual"].to_numpy(float)
    delta = df["improvement_abs"].to_numpy(float)

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.5), constrained_layout=False, height_ratios=[1.15, 1.0])
    ax = axes[0]
    width = 0.38
    ax.bar(x - width / 2, base_abs, width=width, color=COLORS["yellow"], edgecolor=COLORS["dark"], linewidth=0.65, label="Base |error|")
    ax.bar(x + width / 2, total_abs, width=width, color=COLORS["pink"], edgecolor=COLORS["dark"], linewidth=0.65, label="After context bias |error|")
    ax.set_ylabel("|error| (ppm)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Representative 13C molecule: residual after context-derived correction", fontweight="bold", pad=7)
    ax.text(-0.055, 1.05, "a", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")

    ax2 = axes[1]
    bar_colors = [COLORS["good"] if value > 0 else COLORS["bad"] for value in delta]
    ax2.bar(x, delta, color=bar_colors, edgecolor=COLORS["dark"], linewidth=0.65, width=0.62)
    ax2.axhline(0, color=COLORS["dark"], linestyle="--", linewidth=1.0, alpha=0.75)
    ax2.set_ylabel("Error reduction (ppm)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=0)
    ax2.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.85)
    ax2.set_axisbelow(True)
    improved = int((delta > 0).sum())
    ax2.set_title(f"Direction of correction: improved {improved}/{len(delta)} atoms", fontweight="bold", pad=7)
    ax2.text(-0.055, 1.05, "b", transform=ax2.transAxes, fontsize=16, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.09, hspace=0.36)
    df.to_csv(OUT_DIR / "proof_fig4_representative_context_decomposition.csv", index=False)
    save_figure(fig, "proof_07_representative_molecule_context_decomposition")


def write_readme(
    site_summary: pd.DataFrame | None,
    split_summary: pd.DataFrame | None,
    paired_summary: pd.DataFrame,
    split_advantage: pd.DataFrame,
    ablation_ran: bool,
    attention_ran: bool,
) -> None:
    c1_eq = paired_summary[
        (paired_summary["baseline"] == "EquiformerV3-full bs18")
        & (paired_summary["task"] == "13C")
        & (paired_summary["site"] == "C1")
    ].iloc[0]
    h1_eq = paired_summary[
        (paired_summary["baseline"] == "EquiformerV3-full bs18")
        & (paired_summary["task"] == "1H")
        & (paired_summary["site"] == "H1")
    ].iloc[0]
    c1_geq = paired_summary[
        (paired_summary["baseline"] == "GeqShift-O3Transformer")
        & (paired_summary["task"] == "13C")
        & (paired_summary["site"] == "C1")
    ].iloc[0]
    split_lines = []
    for row in split_advantage.itertuples(index=False):
        split_lines.append(
            f"- {row.task} {row.site} {row.split}: n={int(row.n)}, "
            f"baseline mean - Cross-attn = {row.baseline_minus_context:.4f} ppm"
        )

    ablation_lines = []
    if ablation_ran and site_summary is not None and split_summary is not None:
        c1_ab = site_summary[(site_summary["task"] == "13C") & (site_summary["site"] == "C1")].iloc[0]
        h1_ab = site_summary[(site_summary["task"] == "1H") & (site_summary["site"] == "H1")].iloc[0]
        ablation_lines = [
            f"- Same-model no-context ablation, 13C C1: mean full-context gain = {c1_ab.mean_error_gain_full_vs_no_context:.4f} ppm, "
            f"95% CI [{c1_ab.ci95_low:.4f}, {c1_ab.ci95_high:.4f}], n={int(c1_ab.n)}.",
            f"- Same-model no-context ablation, 1H H1: mean full-context gain = {h1_ab.mean_error_gain_full_vs_no_context:.4f} ppm, "
            f"95% CI [{h1_ab.ci95_low:.4f}, {h1_ab.ci95_high:.4f}], n={int(h1_ab.n)}.",
        ]
        for row in split_summary.itertuples(index=False):
            ablation_lines.append(
                f"- Same-model no-context ablation, {row.task} {row.site} {row.split}: "
                f"mean full-context gain = {row.mean_error_gain_full_vs_no_context:.4f} ppm, n={int(row.n)}."
            )
    else:
        ablation_lines = [
            "- Same-model no-context ablation was not run in this invocation. Run with the `equiformer_v3` conda environment and without `--skip-ablation` to generate proof_04 to proof_06 from real forward passes.",
        ]

    text = "\n".join(
        [
            "# 证明图说明",
            "",
            "这个目录集中放置用于支撑“糖链上下文对糖苷键相关 13C 位点，尤其 C1，影响更明显；对 1H 影响较弱”的补充证据。",
            "",
            "## 已生成图",
            "",
            "- `proof_01_paired_site_error_improvement.*`: 同一原子上比较 baseline 与 Cross-attn 的 |error| 差值，正值表示 Cross-attn 更好。",
            "- `proof_02_c1_vs_other_sites_context_gain.*`: 汇总 C1/H1 与其他环位点的上下文收益差异。",
            "- `proof_03_context_strength_c1_h1_by_split.*`: C1/H1 按 mono/di/tri 的误差分组。这个图展示分组差异，但当前数据不能证明链长单调增强。",
            "- `proof_04_same_model_no_context_ablation_by_site.*`: 同一 Cross-attn 模型中把 cross-attention 上下文置零后的因果消融。",
            "- `proof_05_same_model_no_context_ablation_c1_h1_by_split.*`: no-context 消融按 mono/di/tri 分组。",
            "- `proof_06_context_token_ablation_shift_by_site.*`: leave-one-context-token-out 的预测敏感性。",
            "- `proof_07_representative_molecule_context_decomposition.*`: fig4 代表性分子的上下文残差修正案例。",
            "- `proof_08_representative_cross_attention_weights.*`: 代表性双糖和三糖分子的 cross-attention 权重热图；红色纵坐标标签表示直接参与糖苷 C-O-C 键的碳位点，属于辅助解释，不单独作为因果证据。",
            "",
            "## 关键数值",
            "",
            f"- Paired comparison, 13C C1, EquiformerV3-full - Cross-attn: mean Delta |error| = {c1_eq.mean_delta_abs_error:.4f} ppm, 95% CI [{c1_eq.ci95_low:.4f}, {c1_eq.ci95_high:.4f}], n={int(c1_eq.n)}.",
            f"- Paired comparison, 13C C1, GeqShift - Cross-attn: mean Delta |error| = {c1_geq.mean_delta_abs_error:.4f} ppm, 95% CI [{c1_geq.ci95_low:.4f}, {c1_geq.ci95_high:.4f}], n={int(c1_geq.n)}.",
            f"- Paired comparison, 1H H1, EquiformerV3-full - Cross-attn: mean Delta |error| = {h1_eq.mean_delta_abs_error:.4f} ppm, 95% CI [{h1_eq.ci95_low:.4f}, {h1_eq.ci95_high:.4f}], n={int(h1_eq.n)}.",
            *ablation_lines,
            "",
            "## mono/di/tri 分组",
            "",
            *split_lines,
            "",
            "## 结论边界",
            "",
            "当前全验证集统计能支持：C1 这类 13C 位点比 H1 对上下文模型更敏感。`proof_03` 的 mono/di/tri 分组不支持“糖链越长，上下文作用越强”的单调结论，因为 mono 样本只有 9 个分子且误差受离群点影响大。",
            "",
            "`proof_04` 和 `proof_05` 是更接近因果的同模型 no-context ablation；当前结果仍没有呈现 mono/di/tri 随链长递增，因此不要写成“糖链越长，上下文作用越强”。更稳妥的表述是：上下文扰动对 13C 的预测影响约比 1H 大一个数量级，但链长趋势还需要更强扰动设计或更平衡的数据集。",
            "" if attention_ran else "`proof_08` 未在本次运行生成；运行时不要加 `--skip-attention` 即可生成。",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate context proof and interpretability figures.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--mean-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--force-ablation", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only two molecules per split for checking.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = pd.read_csv(SITE_POINTS)
    points = add_site_column(points)
    points = points.dropna(subset=["site"]).copy()
    if "abs_error" not in points.columns:
        points["abs_error"] = points["error"].abs()

    paired_long, paired_summary = paired_context_improvement(points)
    draw_paired_site_improvement(paired_long, paired_summary)
    draw_c1_vs_other_sites(paired_summary)
    split_summary, split_advantage = split_context_summary(points)
    draw_context_split(split_summary, split_advantage)

    ablation_path = OUT_DIR / ("proof_context_ablation_atom_level_smoke.csv" if args.smoke else "proof_context_ablation_atom_level.csv")
    ablation_ran = False
    site_ablation_summary = None
    split_ablation_summary = None
    if not args.skip_ablation:
        if args.force_ablation or not ablation_path.exists():
            ablation_df, _token_df = run_context_ablation(
                run_dir=Path(args.run_dir),
                epoch=args.epoch,
                fold=args.fold,
                gpu_id=args.gpu_id,
                mean_batch_size=args.mean_batch_size,
                eval_batch_size=args.eval_batch_size,
                max_molecules_per_split=2 if args.smoke else None,
            )
        else:
            ablation_df = pd.read_csv(ablation_path)
        site_ablation_summary, split_ablation_summary, token_summary = summarize_ablation(ablation_df)
        draw_no_context_site_gain(site_ablation_summary)
        draw_no_context_split(split_ablation_summary)
        draw_token_ablation_shift(token_summary)
        ablation_ran = True

    attention_ran = False
    if not args.skip_attention:
        run_attention_diagnostic(
            run_dir=Path(args.run_dir),
            epoch=args.epoch,
            fold=args.fold,
            gpu_id=args.gpu_id,
            mean_batch_size=args.mean_batch_size,
        )
        attention_ran = True

    draw_fig4_case_decomposition()
    write_readme(
        site_summary=site_ablation_summary,
        split_summary=split_ablation_summary,
        paired_summary=paired_summary,
        split_advantage=split_advantage,
        ablation_ran=ablation_ran,
        attention_ran=attention_ran,
    )


if __name__ == "__main__":
    main()
