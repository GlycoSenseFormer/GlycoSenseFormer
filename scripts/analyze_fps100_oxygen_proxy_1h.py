#!/usr/bin/env python3
"""Oxygen-neighborhood proxy analysis for FPS100 force-field ensembles.

This script tests a limited, heavy-atom-only version of the hypothesis that
MMFF94's GlycoSenseFormer 1H advantage is related to oxygen-proximity local
geometry. The FPS100 conformer files in this project contain heavy atoms only,
so the script does not compute real H...O distances. Instead it computes:

* nonbonded O...O contact statistics;
* nearest nonbonded O distance from H-bearing heavy atoms;
* local O density around H-bearing heavy atoms.

The output joins these source-level proxy summaries with the seed-4
GlycoSenseFormer clean_split1 1H metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MMFF_MANIFEST = (
    ROOT
    / "experiments/glycosenseformer_conformer_source_v1/conformer_sources/"
    / "etkdg_mmff_paired_fps100_v1/conformer_source_manifest.csv"
)
DEFAULT_MD_MANIFEST = (
    ROOT
    / "experiments/glycosenseformer_conformer_source_v1/conformer_sources/"
    / "md_fps100/conformer_source_manifest.csv"
)
DEFAULT_METRICS = (
    ROOT
    / "experiments/glycosenseformer_conformer_source_v1/formal/"
    / "full329_mmff94val100_seed4/evaluation_clean_split1/metrics.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments/glycosenseformer_conformer_source_v1/analysis/"
    / "fps100_oxygen_proxy_1h_v1"
)

SOURCE_ORDER = ["mmff94_fps100", "charmm_fps100", "gromos_fps100", "glycam_fps100"]
SOURCE_LABELS = {
    "mmff94_fps100": "MMFF94",
    "charmm_fps100": "CHARMM36",
    "gromos_fps100": "GROMOS",
    "glycam_fps100": "GLYCAM06j",
}
SOURCE_COLORS = {
    "mmff94_fps100": "#4C78A8",
    "charmm_fps100": "#F58518",
    "gromos_fps100": "#54A24B",
    "glycam_fps100": "#B279A2",
}
STRATUM_ALIASES = {
    "mono": "mono",
    "mo": "mono",
    "monomer": "mono",
    "di": "di",
    "dimer": "di",
    "tri": "tri",
    "trimer": "tri",
}


@dataclass(frozen=True)
class SourceRecord:
    molecule_id: str
    stratum: str
    source: str
    conformer_path: Path
    source_sdf: Path
    source_index: int


def ru_maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def normalize_stratum(raw: str) -> str:
    key = (raw or "").strip().lower()
    if key not in STRATUM_ALIASES:
        raise ValueError(f"Unknown stratum: {raw!r}")
    return STRATUM_ALIASES[key]


def resolve_path(raw: str, manifest: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def read_manifest(path: Path, sources: set[str]) -> list[SourceRecord]:
    rows: list[SourceRecord] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source", "").strip()
            if source not in sources:
                continue
            rows.append(
                SourceRecord(
                    molecule_id=(row.get("molecule_id") or row.get("clean_name") or "").strip(),
                    stratum=normalize_stratum(row.get("stratum") or row.get("source_class") or ""),
                    source=source,
                    conformer_path=resolve_path(row["conformer_path"], path),
                    source_sdf=resolve_path(row["source_sdf"], path),
                    source_index=int(row["source_index"]),
                )
            )
    return rows


def load_records(mmff_manifest: Path, md_manifest: Path, limit_molecules: int = 0) -> list[SourceRecord]:
    records = read_manifest(mmff_manifest, {"mmff94_fps100"})
    records.extend(read_manifest(md_manifest, {"charmm_fps100", "gromos_fps100", "glycam_fps100"}))
    by_key = {(r.molecule_id, r.source): r for r in records}
    common = sorted(
        set.intersection(*(set(k[0] for k in by_key if k[1] == source) for source in SOURCE_ORDER))
    )
    if limit_molecules:
        common = common[:limit_molecules]
    return [by_key[(mol, source)] for mol in common for source in SOURCE_ORDER]


def load_conformers(path: Path, max_confs: int) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path}: expected (n_confs, n_atoms, 3), got {arr.shape}")
    if arr.shape[0] < max_confs:
        raise ValueError(f"{path}: has {arr.shape[0]} conformers, need {max_confs}")
    return np.ascontiguousarray(arr[:max_confs])


def load_mol(record: SourceRecord) -> Chem.Mol:
    mol = Chem.SDMolSupplier(str(record.source_sdf), removeHs=False)[record.source_index]
    if mol is None:
        raise ValueError(f"Could not read molecule {record.source_index} from {record.source_sdf}")
    return mol


def bonded_pairs(mol: Chem.Mol) -> set[tuple[int, int]]:
    pairs = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pairs.add((min(a, b), max(a, b)))
    return pairs


def nonbonded_oxygen_pairs(mol: Chem.Mol) -> list[tuple[int, int]]:
    oxy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    bonds = bonded_pairs(mol)
    out = []
    for i, a in enumerate(oxy):
        for b in oxy[i + 1 :]:
            if (min(a, b), max(a, b)) not in bonds:
                out.append((a, b))
    return out


def h_bearing_atoms(mol: Chem.Mol) -> list[int]:
    atoms = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.GetTotalNumHs(includeNeighbors=True) > 0:
            atoms.append(atom.GetIdx())
    return atoms


def h_bearing_to_oxygen_pairs(mol: Chem.Mol) -> list[tuple[int, int]]:
    oxy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    hb = h_bearing_atoms(mol)
    bonds = bonded_pairs(mol)
    pairs = []
    for a in hb:
        for o in oxy:
            if a == o:
                continue
            if (min(a, o), max(a, o)) in bonds:
                continue
            pairs.append((a, o))
    return pairs


def distances_for_pairs(coords: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.empty((coords.shape[0], 0), dtype=np.float64)
    idx_a = np.asarray([p[0] for p in pairs], dtype=np.int64)
    idx_b = np.asarray([p[1] for p in pairs], dtype=np.int64)
    diff = coords[:, idx_a, :] - coords[:, idx_b, :]
    return np.linalg.norm(diff, axis=2)


def safe_stat(arr: np.ndarray, fn: str) -> float:
    if arr.size == 0:
        return float("nan")
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "median":
        return float(np.median(arr))
    if fn == "p10":
        return float(np.quantile(arr, 0.10))
    if fn == "p90":
        return float(np.quantile(arr, 0.90))
    raise ValueError(fn)


def ensemble_proxy_metrics(record: SourceRecord, mol: Chem.Mol, coords: np.ndarray) -> dict[str, Any]:
    oo_pairs = nonbonded_oxygen_pairs(mol)
    hb_o_pairs = h_bearing_to_oxygen_pairs(mol)
    oo_dist = distances_for_pairs(coords, oo_pairs)
    hb_o_dist = distances_for_pairs(coords, hb_o_pairs)

    # Per-conformer nearest and counts.
    min_oo = np.min(oo_dist, axis=1) if oo_dist.shape[1] else np.full(coords.shape[0], np.nan)
    hb_min = np.min(hb_o_dist, axis=1) if hb_o_dist.shape[1] else np.full(coords.shape[0], np.nan)
    thresholds = [2.8, 3.0, 3.2, 3.5, 4.0]
    row: dict[str, Any] = {
        "molecule_id": record.molecule_id,
        "stratum": record.stratum,
        "source": record.source,
        "source_label": SOURCE_LABELS[record.source],
        "n_confs": int(coords.shape[0]),
        "n_atoms": int(coords.shape[1]),
        "n_oxygen_atoms": int(sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms())),
        "n_nonbonded_oo_pairs": int(len(oo_pairs)),
        "n_h_bearing_heavy_atoms": int(len(h_bearing_atoms(mol))),
        "n_hbearing_o_proxy_pairs": int(len(hb_o_pairs)),
        "min_oo_distance_mean": safe_stat(min_oo[np.isfinite(min_oo)], "mean"),
        "min_oo_distance_median": safe_stat(min_oo[np.isfinite(min_oo)], "median"),
        "min_oo_distance_p10": safe_stat(min_oo[np.isfinite(min_oo)], "p10"),
        "hbearing_nearest_o_mean": safe_stat(hb_min[np.isfinite(hb_min)], "mean"),
        "hbearing_nearest_o_median": safe_stat(hb_min[np.isfinite(hb_min)], "median"),
        "hbearing_nearest_o_p10": safe_stat(hb_min[np.isfinite(hb_min)], "p10"),
    }
    for threshold in thresholds:
        key = str(threshold).replace(".", "p")
        if oo_dist.shape[1]:
            counts = (oo_dist < threshold).sum(axis=1)
            row[f"oo_contacts_lt_{key}_mean"] = float(np.mean(counts))
            row[f"oo_any_contact_lt_{key}_fraction"] = float(np.mean(counts > 0))
        else:
            row[f"oo_contacts_lt_{key}_mean"] = float("nan")
            row[f"oo_any_contact_lt_{key}_fraction"] = float("nan")
        if hb_o_dist.shape[1]:
            hb_counts = (hb_o_dist < threshold).sum(axis=1)
            row[f"hbearing_o_proxy_contacts_lt_{key}_mean"] = float(np.mean(hb_counts))
            row[f"hbearing_o_proxy_any_lt_{key}_fraction"] = float(np.mean(hb_counts > 0))
        else:
            row[f"hbearing_o_proxy_contacts_lt_{key}_mean"] = float("nan")
            row[f"hbearing_o_proxy_any_lt_{key}_fraction"] = float("nan")
    return row


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "min_oo_distance_median": "median",
        "min_oo_distance_p10": "median",
        "oo_contacts_lt_3p2_mean": "median",
        "oo_any_contact_lt_3p2_fraction": "median",
        "oo_contacts_lt_3p5_mean": "median",
        "oo_any_contact_lt_3p5_fraction": "median",
        "hbearing_nearest_o_median": "median",
        "hbearing_nearest_o_p10": "median",
        "hbearing_o_proxy_contacts_lt_3p5_mean": "median",
        "hbearing_o_proxy_any_lt_3p5_fraction": "median",
    }
    summary = metrics.groupby(["source", "source_label"], as_index=False).agg(aggregations)
    summary["source"] = pd.Categorical(summary["source"], categories=SOURCE_ORDER, ordered=True)
    return summary.sort_values("source")


def load_1h_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[(df["task"] == "1H") & (df["stratum"] == "all")].copy()
    df["source"] = df["train_source"]
    return df[["source", "micro_mae_ppm", "rmse_ppm", "r2"]]


def plot_proxy(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    x = np.arange(len(summary))
    colors = [SOURCE_COLORS[s] for s in summary["source"].astype(str)]
    labels = summary["source_label"].tolist()
    panels = [
        ("oo_contacts_lt_3p2_mean", "O...O contacts <3.2 Å\nmedian per conformer"),
        ("hbearing_o_proxy_contacts_lt_3p5_mean", "H-bearing atom...O proxy\ncontacts <3.5 Å"),
        ("micro_mae_ppm", "GlycoSenseFormer 1H\nmicro MAE (ppm)"),
    ]
    for ax, (col, ylabel) in zip(axes, panels):
        ax.bar(x, summary[col], color=colors, edgecolor="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D8DEE9", alpha=0.8, linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("Oxygen-neighborhood proxies vs 1H performance (heavy atoms only)", y=1.03)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"fig_oxygen_proxy_vs_1h_mae.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmff-manifest", type=Path, default=DEFAULT_MMFF_MANIFEST)
    parser.add_argument("--md-manifest", type=Path, default=DEFAULT_MD_MANIFEST)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-confs", type=int, default=100)
    parser.add_argument("--limit-molecules", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.mmff_manifest.resolve(), args.md_manifest.resolve(), args.limit_molecules)
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(records, start=1):
        mol = load_mol(record)
        coords = load_conformers(record.conformer_path, args.max_confs)
        if coords.shape[1] != mol.GetNumAtoms():
            raise ValueError(f"Atom count mismatch for {record.molecule_id}/{record.source}")
        rows.append(ensemble_proxy_metrics(record, mol, coords))
        if i % args.progress_every == 0 or i == len(records):
            print(
                f"processed {i}/{len(records)} ensembles; elapsed={time.time()-start:.1f}s; "
                f"ru_maxrss={ru_maxrss_mb():.1f} MiB",
                flush=True,
            )
    write_csv(out_dir / "oxygen_proxy_ensemble_metrics.csv", rows)
    metrics = pd.DataFrame(rows)
    summary = source_summary(metrics)
    perf = load_1h_metrics(args.metrics.resolve())
    merged = summary.merge(perf, on="source", how="left")
    merged.to_csv(out_dir / "oxygen_proxy_source_summary_with_1h.csv", index=False)

    corr_rows = []
    for proxy in [
        "oo_contacts_lt_3p2_mean",
        "oo_contacts_lt_3p5_mean",
        "hbearing_o_proxy_contacts_lt_3p5_mean",
        "min_oo_distance_median",
        "hbearing_nearest_o_median",
    ]:
        coef, pvalue = spearmanr(merged[proxy], merged["micro_mae_ppm"])
        corr_rows.append(
            {
                "proxy": proxy,
                "spearman_rho_vs_1h_mae": coef,
                "pvalue_n4_descriptive_only": pvalue,
            }
        )
    pd.DataFrame(corr_rows).to_csv(out_dir / "oxygen_proxy_source_level_correlations.csv", index=False)
    plot_proxy(merged, out_dir)

    status = {
        "status": "complete",
        "elapsed_seconds": time.time() - start,
        "ru_maxrss_mib": ru_maxrss_mb(),
        "n_ensemble_rows": len(rows),
        "n_sources": len(SOURCE_ORDER),
        "note": "Heavy-atom proxy analysis only; explicit H...O distances are not available in FPS100 pickles.",
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
