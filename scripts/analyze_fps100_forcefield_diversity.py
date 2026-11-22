#!/usr/bin/env python3
"""Analyze FPS100 conformer diversity for four GlycoSenseFormer force-field sources.

Outputs two manuscript-style figures:

* A: within-source FPS100 diversity boxplot using per-molecule pairwise Kabsch RMSD.
* C: glycosidic phi/psi density facets by force-field source.

The implementation is deliberately single-process and streaming: each 100-conformer
ensemble is loaded, summarized, and released before the next ensemble is processed.
RMSD matrices are not written to disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import resource
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MMFF_MANIFEST = (
    ROOT
    / "experiments"
    / "glycosenseformer_conformer_source_v1"
    / "conformer_sources"
    / "etkdg_mmff_paired_fps100_v1"
    / "conformer_source_manifest.csv"
)
DEFAULT_MD_MANIFEST = (
    ROOT
    / "experiments"
    / "glycosenseformer_conformer_source_v1"
    / "conformer_sources"
    / "md_fps100"
    / "conformer_source_manifest.csv"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "experiments"
    / "glycosenseformer_conformer_source_v1"
    / "analysis"
    / "fps100_forcefield_diversity_v1"
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
    expected_conformer_sha256: str | None
    expected_conformer_count: int | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ru_maxrss_mb() -> float:
    """Return max resident set size in MiB on Linux/macOS."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS reports bytes. This workspace is Linux, but keep sane fallback.
    return raw / 1024.0 if raw > 10_000 else raw / (1024.0 * 1024.0)


def normalize_stratum(value: str) -> str:
    key = (value or "").strip().lower()
    if key not in STRATUM_ALIASES:
        raise ValueError(f"Unknown stratum {value!r}")
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
            count_raw = (row.get("expected_conformer_count") or "").strip()
            rows.append(
                SourceRecord(
                    molecule_id=(row.get("molecule_id") or row.get("clean_name") or "").strip(),
                    stratum=normalize_stratum(row.get("stratum") or row.get("source_class") or ""),
                    source=source,
                    conformer_path=resolve_path(row["conformer_path"], path),
                    source_sdf=resolve_path(row["source_sdf"], path),
                    source_index=int(row["source_index"]),
                    expected_conformer_sha256=(row.get("expected_conformer_sha256") or "").strip() or None,
                    expected_conformer_count=int(count_raw) if count_raw else None,
                )
            )
    return rows


def load_records(mmff_manifest: Path, md_manifest: Path) -> list[SourceRecord]:
    records = read_manifest(mmff_manifest, {"mmff94_fps100"})
    records.extend(read_manifest(md_manifest, {"charmm_fps100", "gromos_fps100", "glycam_fps100"}))
    by_key: dict[tuple[str, str], SourceRecord] = {}
    for record in records:
        if not record.molecule_id:
            raise ValueError(f"Empty molecule_id in {record}")
        key = (record.molecule_id, record.source)
        if key in by_key:
            raise ValueError(f"Duplicate source record: {key}")
        by_key[key] = record

    common = sorted(
        set.intersection(*(set(k[0] for k in by_key if k[1] == source) for source in SOURCE_ORDER))
    )
    filtered = [by_key[(molecule_id, source)] for molecule_id in common for source in SOURCE_ORDER]
    expected = len(common) * len(SOURCE_ORDER)
    if len(filtered) != expected:
        raise RuntimeError(f"Expected {expected} source records after intersection, got {len(filtered)}")
    return filtered


def load_conformers(record: SourceRecord, max_confs: int) -> np.ndarray:
    if not record.conformer_path.is_file():
        raise FileNotFoundError(record.conformer_path)
    if record.expected_conformer_sha256:
        actual = sha256_file(record.conformer_path)
        if actual != record.expected_conformer_sha256:
            raise ValueError(
                f"SHA mismatch for {record.conformer_path}: {actual} != {record.expected_conformer_sha256}"
            )
    with record.conformer_path.open("rb") as handle:
        conformers = pickle.load(handle)
    arr = np.asarray(conformers, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{record.conformer_path}: expected (n_confs, n_atoms, 3), got {arr.shape}")
    if record.expected_conformer_count is not None and arr.shape[0] != record.expected_conformer_count:
        raise ValueError(
            f"{record.conformer_path}: count {arr.shape[0]} != expected {record.expected_conformer_count}"
        )
    if arr.shape[0] < max_confs:
        raise ValueError(f"{record.conformer_path}: has {arr.shape[0]} conformers, need {max_confs}")
    return np.ascontiguousarray(arr[:max_confs])


def load_mol(record: SourceRecord) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(record.source_sdf), removeHs=False)
    mol = supplier[record.source_index]
    if mol is None:
        raise ValueError(f"Could not read mol index {record.source_index} from {record.source_sdf}")
    return mol


def kabsch_pairwise_rmsd(conformers: np.ndarray) -> np.ndarray:
    """Vectorized Kabsch RMSD for all pairs in one 100-conformer ensemble."""
    n_conf, n_atoms, _ = conformers.shape
    centered = conformers - conformers.mean(axis=1, keepdims=True)
    sq_norm = np.sum(centered * centered, axis=(1, 2))
    i_idx, j_idx = np.triu_indices(n_conf, k=1)
    p = centered[i_idx]
    q = centered[j_idx]
    cov = np.einsum("bni,bnj->bij", p, q, optimize=True)
    u, s, vt = np.linalg.svd(cov, full_matrices=False)
    det = np.linalg.det(np.matmul(vt.transpose(0, 2, 1), u.transpose(0, 2, 1)))
    trace = s[:, 0] + s[:, 1] + np.sign(det) * s[:, 2]
    rmsd2 = (sq_norm[i_idx] + sq_norm[j_idx] - 2.0 * trace) / float(n_atoms)
    rmsd2 = np.maximum(rmsd2, 0.0)
    return np.sqrt(rmsd2)


def summarize_rmsd(record: SourceRecord, conformers: np.ndarray, pairwise: np.ndarray) -> dict[str, Any]:
    n_conf = conformers.shape[0]
    matrix = np.zeros((n_conf, n_conf), dtype=np.float64)
    iu = np.triu_indices(n_conf, k=1)
    matrix[iu] = pairwise
    matrix[(iu[1], iu[0])] = pairwise
    mean_to_others = matrix.sum(axis=1) / float(n_conf - 1)
    medoid_idx = int(np.argmin(mean_to_others))
    medoid_dist = matrix[medoid_idx, np.arange(n_conf) != medoid_idx]
    return {
        "molecule_id": record.molecule_id,
        "stratum": record.stratum,
        "source": record.source,
        "source_label": SOURCE_LABELS[record.source],
        "n_confs": n_conf,
        "n_atoms": conformers.shape[1],
        "pairwise_rmsd_mean": float(np.mean(pairwise)),
        "pairwise_rmsd_median": float(np.median(pairwise)),
        "pairwise_rmsd_sd": float(np.std(pairwise, ddof=1)),
        "pairwise_rmsd_p10": float(np.quantile(pairwise, 0.10)),
        "pairwise_rmsd_p90": float(np.quantile(pairwise, 0.90)),
        "pairwise_rmsd_p95": float(np.quantile(pairwise, 0.95)),
        "pairwise_rmsd_max": float(np.max(pairwise)),
        "medoid_index": medoid_idx,
        "medoid_rmsd_mean": float(np.mean(medoid_dist)),
        "medoid_rmsd_p90": float(np.quantile(medoid_dist, 0.90)),
        "medoid_rmsd_max": float(np.max(medoid_dist)),
        "near_duplicate_pair_fraction_0p05": float(np.mean(pairwise < 0.05)),
        "near_duplicate_pair_fraction_0p10": float(np.mean(pairwise < 0.10)),
        "near_duplicate_pair_fraction_0p25": float(np.mean(pairwise < 0.25)),
    }


def atom_rings(mol: Chem.Mol) -> list[dict[str, Any]]:
    rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue
        oxy = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8]
        carbons = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6]
        if len(oxy) != 1 or len(carbons) < 4:
            continue
        rings.append({"atoms": set(ring), "oxygen": oxy[0], "carbons": set(carbons)})
    return rings


def ring_for_atom(rings: Sequence[Mapping[str, Any]], atom_idx: int) -> Mapping[str, Any] | None:
    matches = [ring for ring in rings if atom_idx in ring["atoms"]]
    if len(matches) == 1:
        return matches[0]
    return None


def heavy_neighbors(mol: Chem.Mol, atom_idx: int) -> list[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if n.GetAtomicNum() > 1]


def is_anomeric_carbon(mol: Chem.Mol, rings: Sequence[Mapping[str, Any]], atom_idx: int) -> tuple[bool, int | None]:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6:
        return False, None
    ring = ring_for_atom(rings, atom_idx)
    if ring is None:
        return False, None
    ring_o = int(ring["oxygen"])
    if ring_o not in heavy_neighbors(mol, atom_idx):
        return False, None
    # Anomeric carbon should have an exocyclic oxygen neighbor in addition to ring oxygen.
    for nb in heavy_neighbors(mol, atom_idx):
        if nb == ring_o:
            continue
        nb_atom = mol.GetAtomWithIdx(nb)
        if nb_atom.GetAtomicNum() == 8 and nb not in ring["atoms"]:
            return True, ring_o
    return False, None


def acceptor_next_atom(mol: Chem.Mol, rings: Sequence[Mapping[str, Any]], acceptor_c: int, donor_ring: Mapping[str, Any]) -> int | None:
    # Prefer carbon neighbors in the acceptor ring; this gives a deterministic psi
    # definition for ring acceptors.
    acceptor_ring = ring_for_atom(rings, acceptor_c)
    if acceptor_ring is not None and acceptor_ring is not donor_ring:
        candidates = [
            nb
            for nb in heavy_neighbors(mol, acceptor_c)
            if nb in acceptor_ring["atoms"] and mol.GetAtomWithIdx(nb).GetAtomicNum() == 6
        ]
        if candidates:
            return min(candidates)
        candidates = [
            nb
            for nb in heavy_neighbors(mol, acceptor_c)
            if nb in acceptor_ring["atoms"] and nb != int(acceptor_ring["oxygen"])
        ]
        if candidates:
            return min(candidates)
    # 1->6-like exocyclic acceptor: choose the ring carbon attached to acceptor C.
    candidates = []
    for nb in heavy_neighbors(mol, acceptor_c):
        ring = ring_for_atom(rings, nb)
        if ring is not None and ring is not donor_ring:
            candidates.append(nb)
    return min(candidates) if candidates else None


def identify_glycosidic_phi_psi(mol: Chem.Mol) -> list[dict[str, Any]]:
    rings = atom_rings(mol)
    linkages: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for oxygen in mol.GetAtoms():
        if oxygen.GetAtomicNum() != 8 or oxygen.IsInRing():
            continue
        o_idx = oxygen.GetIdx()
        carbon_neighbors = [
            nb.GetIdx() for nb in oxygen.GetNeighbors() if nb.GetAtomicNum() == 6
        ]
        if len(carbon_neighbors) != 2:
            continue
        for anom_c, acceptor_c in (
            (carbon_neighbors[0], carbon_neighbors[1]),
            (carbon_neighbors[1], carbon_neighbors[0]),
        ):
            is_anom, ring_o = is_anomeric_carbon(mol, rings, anom_c)
            if not is_anom or ring_o is None:
                continue
            donor_ring = ring_for_atom(rings, anom_c)
            if donor_ring is None:
                continue
            # Exclude OMe/aglycon cases: acceptor side must be in or attached to a sugar ring
            # that is not the donor ring.
            psi_next = acceptor_next_atom(mol, rings, acceptor_c, donor_ring)
            if psi_next is None:
                continue
            key = tuple(sorted((anom_c, o_idx, acceptor_c)))
            if key in seen:
                continue
            seen.add(key)
            linkages.append(
                {
                    "linkage_id": len(linkages),
                    "ring_o": int(ring_o),
                    "anomeric_c": int(anom_c),
                    "bridge_o": int(o_idx),
                    "acceptor_c": int(acceptor_c),
                    "acceptor_next": int(psi_next),
                }
            )
    return linkages


def dihedral_deg(coords: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    p0, p1, p2, p3 = coords[[a, b, c, d]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm <= 1e-12:
        return float("nan")
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm <= 1e-12 or w_norm <= 1e-12:
        return float("nan")
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def torsion_rows(record: SourceRecord, mol: Chem.Mol, conformers: np.ndarray) -> list[dict[str, Any]]:
    linkages = identify_glycosidic_phi_psi(mol)
    rows: list[dict[str, Any]] = []
    for linkage in linkages:
        for conf_idx, coords in enumerate(conformers):
            phi = dihedral_deg(
                coords,
                linkage["ring_o"],
                linkage["anomeric_c"],
                linkage["bridge_o"],
                linkage["acceptor_c"],
            )
            psi = dihedral_deg(
                coords,
                linkage["anomeric_c"],
                linkage["bridge_o"],
                linkage["acceptor_c"],
                linkage["acceptor_next"],
            )
            if math.isfinite(phi) and math.isfinite(psi):
                rows.append(
                    {
                        "molecule_id": record.molecule_id,
                        "stratum": record.stratum,
                        "source": record.source,
                        "source_label": SOURCE_LABELS[record.source],
                        "conf_idx": conf_idx,
                        "linkage_id": linkage["linkage_id"],
                        "ring_o": linkage["ring_o"],
                        "anomeric_c": linkage["anomeric_c"],
                        "bridge_o": linkage["bridge_o"],
                        "acceptor_c": linkage["acceptor_c"],
                        "acceptor_next": linkage["acceptor_next"],
                        "phi_deg": phi,
                        "psi_deg": psi,
                    }
                )
    return rows


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


def plot_diversity_boxplot(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data = [metrics.loc[metrics["source"] == src, "pairwise_rmsd_median"].to_numpy() for src in SOURCE_ORDER]
    labels = [SOURCE_LABELS[src] for src in SOURCE_ORDER]
    colors = [SOURCE_COLORS[src] for src in SOURCE_ORDER]
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.4},
        boxprops={"linewidth": 1.1, "edgecolor": "#333333"},
        whiskerprops={"linewidth": 1.0, "color": "#333333"},
        capprops={"linewidth": 1.0, "color": "#333333"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    rng = np.random.default_rng(13)
    for i, (src, color) in enumerate(zip(SOURCE_ORDER, colors), start=1):
        y = metrics.loc[metrics["source"] == src, "pairwise_rmsd_median"].to_numpy()
        x = rng.normal(i, 0.045, size=len(y))
        ax.scatter(x, y, s=9, alpha=0.23, color=color, linewidths=0, rasterized=True)
        ax.text(i, np.nanmax(y) * 1.03, f"n={len(y)}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Within-source pairwise Kabsch RMSD median (Å)")
    ax.set_xlabel("FPS100 conformer source")
    ax.set_title("A. FPS100 within-source conformer diversity")
    ax.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"fig_A_fps100_within_source_diversity_boxplot.{ext}", dpi=300)
    plt.close(fig)


def plot_phi_psi_density(torsions: pd.DataFrame, out_dir: Path, bins: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2), sharex=True, sharey=True)
    edges = np.linspace(-180.0, 180.0, bins + 1)
    histograms = {}
    vmax = 1
    for src in SOURCE_ORDER:
        sub = torsions[torsions["source"] == src]
        h, _, _ = np.histogram2d(sub["phi_deg"], sub["psi_deg"], bins=[edges, edges])
        histograms[src] = h
        vmax = max(vmax, int(h.max()))
    for ax, src in zip(axes.ravel(), SOURCE_ORDER):
        h = histograms[src].T
        im = ax.imshow(
            h,
            origin="lower",
            extent=(-180, 180, -180, 180),
            cmap="magma",
            norm=LogNorm(vmin=1, vmax=vmax),
            interpolation="nearest",
            aspect="equal",
        )
        sub = torsions[torsions["source"] == src]
        ax.set_title(f"{SOURCE_LABELS[src]} (n={len(sub):,})", fontsize=10)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.grid(color="white", alpha=0.12, linewidth=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("ψ (degrees)")
    for ax in axes[-1, :]:
        ax.set_xlabel("φ (degrees)")
    fig.suptitle("C. Glycosidic φ/ψ density by FPS100 force-field source", y=0.98)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Conformer-linkage count per bin (log scale)")
    fig.subplots_adjust(top=0.92, right=0.88, wspace=0.10, hspace=0.18)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"fig_C_glycosidic_phi_psi_density_by_forcefield.{ext}", dpi=300)
    plt.close(fig)


def summarize_by_source(metrics: pd.DataFrame, torsions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for src in SOURCE_ORDER:
        sub = metrics[metrics["source"] == src]
        tor = torsions[torsions["source"] == src]
        rows.append(
            {
                "source": src,
                "source_label": SOURCE_LABELS[src],
                "n_ensembles": int(len(sub)),
                "n_molecules": int(sub["molecule_id"].nunique()),
                "n_conformers": int(sub["n_confs"].sum()),
                "pairwise_rmsd_median_of_medians": float(sub["pairwise_rmsd_median"].median()),
                "pairwise_rmsd_mean_of_means": float(sub["pairwise_rmsd_mean"].mean()),
                "pairwise_rmsd_p90_median": float(sub["pairwise_rmsd_p90"].median()),
                "medoid_rmsd_mean_median": float(sub["medoid_rmsd_mean"].median()),
                "n_phi_psi_points": int(len(tor)),
                "n_phi_psi_molecules": int(tor["molecule_id"].nunique()) if len(tor) else 0,
                "n_phi_psi_linkages": int(tor[["molecule_id", "linkage_id"]].drop_duplicates().shape[0])
                if len(tor)
                else 0,
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmff-manifest", type=Path, default=DEFAULT_MMFF_MANIFEST)
    parser.add_argument("--md-manifest", type=Path, default=DEFAULT_MD_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-confs", type=int, default=100)
    parser.add_argument("--density-bins", type=int, default=72)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit-molecules", type=int, default=0, help="Debug limit over common molecule ids.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    tracemalloc.start()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.mmff_manifest.resolve(), args.md_manifest.resolve())
    if args.limit_molecules:
        keep = sorted({r.molecule_id for r in records})[: args.limit_molecules]
        keep_set = set(keep)
        records = [r for r in records if r.molecule_id in keep_set]
    n_common = len({r.molecule_id for r in records})
    if n_common == 0:
        raise RuntimeError("No common molecules selected")

    config = {
        "protocol": "fps100_forcefield_diversity_v1",
        "mmff_manifest": str(args.mmff_manifest.resolve()),
        "md_manifest": str(args.md_manifest.resolve()),
        "out_dir": str(out_dir),
        "sources": SOURCE_ORDER,
        "max_confs": args.max_confs,
        "density_bins": args.density_bins,
        "limit_molecules": args.limit_molecules,
        "n_common_molecules": n_common,
        "n_source_records": len(records),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "rdkit": Chem.rdBase.rdkitVersion,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    ensemble_rows: list[dict[str, Any]] = []
    torsion_long_rows: list[dict[str, Any]] = []
    topology_rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        mol = load_mol(record)
        conformers = load_conformers(record, args.max_confs)
        if conformers.shape[1] != mol.GetNumAtoms():
            raise ValueError(
                f"{record.molecule_id}/{record.source}: conformer atoms {conformers.shape[1]} "
                f"!= SDF atoms {mol.GetNumAtoms()}"
            )
        pairwise = kabsch_pairwise_rmsd(conformers)
        ensemble_rows.append(summarize_rmsd(record, conformers, pairwise))
        linkages = identify_glycosidic_phi_psi(mol)
        topology_rows.append(
            {
                "molecule_id": record.molecule_id,
                "stratum": record.stratum,
                "source": record.source,
                "source_label": SOURCE_LABELS[record.source],
                "n_atoms": mol.GetNumAtoms(),
                "n_rings": len(atom_rings(mol)),
                "n_glycosidic_phi_psi_linkages": len(linkages),
            }
        )
        torsion_long_rows.extend(torsion_rows(record, mol, conformers))
        if idx % args.progress_every == 0 or idx == len(records):
            current, peak = tracemalloc.get_traced_memory()
            elapsed = time.time() - start
            print(
                f"processed {idx}/{len(records)} ensembles; elapsed={elapsed:.1f}s; "
                f"ru_maxrss={ru_maxrss_mb():.1f} MiB; tracemalloc_peak={peak/1024/1024:.1f} MiB",
                flush=True,
            )
        del mol, conformers, pairwise

    write_csv(out_dir / "ensemble_metrics.csv", ensemble_rows)
    write_csv(out_dir / "glycosidic_phi_psi_long.csv", torsion_long_rows)
    write_csv(out_dir / "topology_linkage_counts.csv", topology_rows)

    ensemble_df = pd.DataFrame(ensemble_rows)
    torsion_df = pd.DataFrame(torsion_long_rows)
    source_summary = summarize_by_source(ensemble_df, torsion_df)
    write_csv(out_dir / "source_summary.csv", source_summary)

    plot_diversity_boxplot(ensemble_df, out_dir)
    if torsion_df.empty:
        raise RuntimeError("No glycosidic phi/psi torsions identified; cannot draw density plot")
    plot_phi_psi_density(torsion_df, out_dir, args.density_bins)

    current, peak = tracemalloc.get_traced_memory()
    status = {
        "status": "complete",
        "elapsed_seconds": time.time() - start,
        "ru_maxrss_mib": ru_maxrss_mb(),
        "tracemalloc_current_mib": current / 1024 / 1024,
        "tracemalloc_peak_mib": peak / 1024 / 1024,
        "n_ensemble_rows": len(ensemble_rows),
        "n_torsion_rows": len(torsion_long_rows),
        "n_topology_rows": len(topology_rows),
        "outputs": [
            "ensemble_metrics.csv",
            "glycosidic_phi_psi_long.csv",
            "topology_linkage_counts.csv",
            "source_summary.csv",
            "fig_A_fps100_within_source_diversity_boxplot.png",
            "fig_C_glycosidic_phi_psi_density_by_forcefield.png",
        ],
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
