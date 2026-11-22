#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem


def load_conformers(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        conformers = pickle.load(handle)
    arr = np.asarray(conformers, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{path} does not contain an array-like list of (n_atoms, 3) conformers")
    return arr


def find_molecule(sdf_path: Path, molecule_name: str) -> Chem.Mol:
    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if mol is None:
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name") == molecule_name:
            return mol
    raise ValueError(f"Could not find {molecule_name!r} in {sdf_path}")


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


def pairwise_rmsd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    matrix = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            matrix[i, j] = kabsch_rmsd(a[i], b[j])
    return matrix


def upper_triangle_rmsd(a: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for i in range(a.shape[0]):
        for j in range(i + 1, a.shape[0]):
            values.append(kabsch_rmsd(a[i], a[j]))
    return np.asarray(values, dtype=np.float64)


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: float("nan") for key in ("min", "p25", "mean", "median", "p75", "max", "std")}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def radius_of_gyration(conformers: np.ndarray) -> np.ndarray:
    centered = conformers - conformers.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1))


def dihedral(coords: np.ndarray, atoms: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = (coords[i] for i in atoms)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def find_glycosidic_torsions(mol: Chem.Mol) -> dict[str, tuple[int, int, int, int]]:
    rings = [set(ring) for ring in Chem.GetSymmSSSR(mol)]
    atom_to_rings: dict[int, list[set[int]]] = {}
    for ring in rings:
        for atom_idx in ring:
            atom_to_rings.setdefault(atom_idx, []).append(ring)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.IsInRing():
            continue
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() > 1]
        if len(neighbors) != 2:
            continue
        if not all(mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 for idx in neighbors):
            continue
        if not all(idx in atom_to_rings for idx in neighbors):
            continue

        donor = None
        donor_ring_oxygen = None
        acceptor = None
        for candidate in neighbors:
            for nbr in mol.GetAtomWithIdx(candidate).GetNeighbors():
                if nbr.GetAtomicNum() == 8 and nbr.IsInRing():
                    donor = candidate
                    donor_ring_oxygen = nbr.GetIdx()
                    acceptor = neighbors[0] if neighbors[1] == candidate else neighbors[1]
                    break
            if donor is not None:
                break
        if donor is None or donor_ring_oxygen is None or acceptor is None:
            continue

        acceptor_ring_neighbors = [
            nbr.GetIdx()
            for nbr in mol.GetAtomWithIdx(acceptor).GetNeighbors()
            if nbr.GetAtomicNum() == 6 and nbr.GetIdx() != atom.GetIdx() and nbr.GetIdx() in atom_to_rings
        ]
        torsions = {
            "phi": (donor_ring_oxygen, donor, atom.GetIdx(), acceptor),
        }
        for idx, terminal in enumerate(acceptor_ring_neighbors):
            torsions[f"psi_{idx + 1}"] = (donor, atom.GetIdx(), acceptor, terminal)
        return torsions

    return {}


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["md_index/mmff_index", *range(matrix.shape[1])])
        for i, row in enumerate(matrix):
            writer.writerow([i, *[f"{value:.6f}" for value in row]])


def write_nearest(path: Path, nearest_indices: np.ndarray, nearest_values: np.ndarray, index_name: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([index_name, "nearest_index", "nearest_rmsd_angstrom"])
        for i, (nearest_index, value) in enumerate(zip(nearest_indices, nearest_values)):
            writer.writerow([i, int(nearest_index), f"{value:.6f}"])


def write_series(path: Path, source: str, values: np.ndarray, column: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "index", column])
        for i, value in enumerate(values):
            writer.writerow([source, i, f"{float(value):.6f}"])


def write_torsions(path: Path, torsions: dict[str, tuple[int, int, int, int]], sources: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "index", *torsions.keys()])
        for source, conformers in sources.items():
            for i, coords in enumerate(conformers):
                writer.writerow([source, i, *[f"{dihedral(coords, atoms):.6f}" for atoms in torsions.values()]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two heavy-atom conformer pickle ensembles.")
    parser.add_argument("--md-pickle", required=True)
    parser.add_argument("--mmff-pickle", required=True)
    parser.add_argument("--reference-sdf", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--md-label", default="GAFF2 vacuum MD")
    parser.add_argument("--mmff-label", default="RDKit MMFF94")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md = load_conformers(Path(args.md_pickle))
    mmff = load_conformers(Path(args.mmff_pickle))
    if md.shape[1:] != mmff.shape[1:]:
        raise ValueError(f"Shape mismatch: MD {md.shape}, MMFF {mmff.shape}")

    matrix = pairwise_rmsd(md, mmff)
    md_to_mmff = matrix.min(axis=1)
    md_to_mmff_idx = matrix.argmin(axis=1)
    mmff_to_md = matrix.min(axis=0)
    mmff_to_md_idx = matrix.argmin(axis=0)
    md_internal = upper_triangle_rmsd(md)
    mmff_internal = upper_triangle_rmsd(mmff)
    md_rg = radius_of_gyration(md)
    mmff_rg = radius_of_gyration(mmff)

    mol = find_molecule(Path(args.reference_sdf), args.molecule_name)
    torsions = find_glycosidic_torsions(mol)

    write_matrix(output_dir / "rmsd_matrix_md_vs_mmff.csv", matrix)
    write_nearest(output_dir / "rmsd_md_to_nearest_mmff.csv", md_to_mmff_idx, md_to_mmff, "md_index")
    write_nearest(output_dir / "rmsd_mmff_to_nearest_md.csv", mmff_to_md_idx, mmff_to_md, "mmff_index")
    write_series(output_dir / "radius_of_gyration_md.csv", args.md_label, md_rg, "rg_angstrom")
    write_series(output_dir / "radius_of_gyration_mmff.csv", args.mmff_label, mmff_rg, "rg_angstrom")
    if torsions:
        write_torsions(output_dir / "glycosidic_torsions.csv", torsions, {args.md_label: md, args.mmff_label: mmff})

    summary = {
        "molecule_name": args.molecule_name,
        "md_pickle": args.md_pickle,
        "mmff_pickle": args.mmff_pickle,
        "n_md_conformers": int(md.shape[0]),
        "n_mmff_conformers": int(mmff.shape[0]),
        "n_heavy_atoms": int(md.shape[1]),
        "rmsd_angstrom": {
            "all_md_vs_mmff_pairs": summarize(matrix.ravel()),
            "md_to_nearest_mmff": summarize(md_to_mmff),
            "mmff_to_nearest_md": summarize(mmff_to_md),
            "md_internal_pairwise": summarize(md_internal),
            "mmff_internal_pairwise": summarize(mmff_internal),
        },
        "radius_of_gyration_angstrom": {
            "md": summarize(md_rg),
            "mmff": summarize(mmff_rg),
        },
        "glycosidic_torsion_atom_indices_0based": {key: list(value) for key, value in torsions.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    report = [
        f"# {args.molecule_name}: {args.md_label} vs {args.mmff_label}",
        "",
        "## Inputs",
        "",
        f"- MD conformers: `{args.md_pickle}`",
        f"- MMFF conformers: `{args.mmff_pickle}`",
        f"- Conformer count: MD {md.shape[0]}, MMFF {mmff.shape[0]}",
        f"- Heavy atoms per conformer: {md.shape[1]}",
        "",
        "## RMSD Summary",
        "",
        "| Metric | Min | Median | Mean | P75 | Max | Std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in [
        ("All MD-MMFF pairs", matrix.ravel()),
        ("MD to nearest MMFF", md_to_mmff),
        ("MMFF to nearest MD", mmff_to_md),
        ("MD internal pairwise", md_internal),
        ("MMFF internal pairwise", mmff_internal),
    ]:
        stats = summarize(values)
        report.append(
            f"| {label} | {stats['min']:.3f} | {stats['median']:.3f} | {stats['mean']:.3f} | "
            f"{stats['p75']:.3f} | {stats['max']:.3f} | {stats['std']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Radius Of Gyration",
            "",
            "| Ensemble | Median Å | Mean Å | Std Å |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, values in [(args.md_label, md_rg), (args.mmff_label, mmff_rg)]:
        stats = summarize(values)
        report.append(f"| {label} | {stats['median']:.3f} | {stats['mean']:.3f} | {stats['std']:.3f} |")
    report.extend(["", "## Glycosidic Torsions", ""])
    if torsions:
        for key, atoms in torsions.items():
            report.append(f"- `{key}` atom indices, 0-based SDF order: `{list(atoms)}`")
        report.append("")
        report.append("Raw torsion values are in `glycosidic_torsions.csv`.")
    else:
        report.append("No ring-to-ring glycosidic oxygen was detected automatically.")
    report.extend(
        [
            "",
            "## Notes",
            "",
            "- RMSD values are Kabsch-aligned heavy-atom RMSD in Angstrom.",
            "- This is a technical conformer-ensemble comparison; solvent model, simulation length, and replicate settings should be reported separately for formal force-field comparisons.",
        ]
    )
    report_path = output_dir / "conformer_comparison_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
