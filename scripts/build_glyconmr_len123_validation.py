#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdForceFieldHelpers
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import coalesce, to_undirected


EQUIFORMER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQUIFORMER_ROOT.parent
DEFAULT_OUT_DIR = EQUIFORMER_ROOT / "data" / "datasets_glyconmr_len123_independent_validation"
DEFAULT_GLYCONMR_SDF = PROJECT_ROOT / "dataset" / "glyconmr" / "glyconmr.sdf"
DEFAULT_GLYCONMR_MANIFEST = PROJECT_ROOT / "dataset" / "glyconmr" / "manifest.csv"
DEFAULT_CASPER_SDFS = [
    EQUIFORMER_ROOT / "data" / "dataset" / "casper_mono.sdf",
    EQUIFORMER_ROOT / "data" / "dataset" / "casper_dimer.sdf",
    EQUIFORMER_ROOT / "data" / "dataset" / "casper_trimer.sdf",
]
DEFAULT_TRAIN_DIR = EQUIFORMER_ROOT / "data" / "datasets"

TASKS = ("13C", "1H")
LENGTH_TO_SPLIT = {1: "mo", 2: "di", 3: "tri"}
TYPE_IDS = {"H": 1, "C": 2, "N": 3, "O": 4, "F": 5, "P": 6, "S": 7, "Cl": 8, "Br": 9}
SUPPORTED_BOND_TYPES = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
NON_SUGAR_UNITS = {
    "17hoole",
    "ac",
    "allyl",
    "asn",
    "bu",
    "bz",
    "caf",
    "fer",
    "gallic",
    "gc",
    "mal",
    "me",
    "p",
    "pr",
    "s",
    "ser",
}


def read_manifest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "included":
                rows[(row["source"], row["source_file"])] = row
    return rows


def source_csv_path(row: dict[str, str]) -> Path:
    if row["source"] == "glyconmr_glycosciencedb":
        return PROJECT_ROOT / "GlycoNMR" / "glycosciencedb" / "data" / row["source_file"]
    if row["source"] == "glyconmr_godess":
        return PROJECT_ROOT / "GlycoNMR" / "godess" / "data" / row["source_file"]
    raise ValueError(f"Unsupported GlycoNMR source: {row['source']}")


def glycan_length(row: dict[str, str]) -> tuple[int, list[str]]:
    with source_csv_path(row).open(newline="") as handle:
        atom_rows = list(csv.DictReader(handle))
    if not atom_rows:
        return 0, []
    residue_col = "residual" if "residual" in atom_rows[0] else "Residual_num"
    residue_to_unit: dict[str, str] = {}
    for atom_row in atom_rows:
        residue_to_unit.setdefault(atom_row[residue_col], atom_row.get("reformulated_standard_mono", ""))
    sugar_units = [unit for unit in residue_to_unit.values() if unit not in NON_SUGAR_UNITS]
    return len(sugar_units), sugar_units


def load_mols(path: Path) -> list[Chem.Mol]:
    return [mol for mol in Chem.SDMolSupplier(str(path), removeHs=False) if mol is not None]


def normalized_overlap_smiles(mol: Chem.Mol) -> str:
    """Heavy-atom graph key matching GlycoNMR's all-single-bond SDF conversion."""
    rw_mol = Chem.RWMol(Chem.RemoveHs(mol, sanitize=False))
    for bond in rw_mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw_mol.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    out = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.SanitizeMol(
                out,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass
    return Chem.MolToSmiles(out, canonical=True, isomericSmiles=False)


def generate_conformations(
    mol: Chem.Mol,
    nbr_confs: int,
    oversample: int,
    seed: int,
    num_threads: int,
    optimize_mmff: bool,
) -> tuple[list[list[list[float]]], list[float], str, int]:
    def build_params(prune_rms_thresh: float):
        params = Chem.rdDistGeom.ETKDGv3()
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.pruneRmsThresh = prune_rms_thresh
        params.numThreads = num_threads
        params.enforceChirality = True
        params.randomSeed = seed
        return params

    num_confs = max(1, nbr_confs * oversample)

    def embed_score_candidates(embed_seed: int, prune_rms_thresh: float) -> tuple[list[tuple[list[list[float]], float, str]], str]:
        working = Chem.Mol(mol)
        working.RemoveAllConformers()
        Chem.SanitizeMol(working)
        working = Chem.AddHs(working, addCoords=True)
        params = build_params(prune_rms_thresh)
        params.randomSeed = embed_seed
        conf_ids = list(Chem.rdDistGeom.EmbedMultipleConfs(working, numConfs=num_confs, params=params))
        if not conf_ids:
            return [], "embed_failed"

        forcefield = "none"
        energies: list[float] = []
        if optimize_mmff and AllChem.MMFFHasAllMoleculeParams(working):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                working,
                numThreads=num_threads,
                maxIters=200,
                mmffVariant="MMFF94",
            )
            energies = [float(energy) for _, energy in results]
            forcefield = "MMFF94"
        else:
            ps = AllChem.MMFFGetMoleculeProperties(working, mmffVariant="MMFF94") if optimize_mmff else None
            for conf in working.GetConformers():
                if ps is None:
                    ff = AllChem.UFFGetMoleculeForceField(working, confId=conf.GetId())
                    forcefield = "UFF"
                else:
                    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(working, ps, confId=conf.GetId())
                    forcefield = "MMFF94_partial"
                energies.append(float("inf") if ff is None else float(ff.CalcEnergy()))

        working = Chem.RemoveHs(working)
        candidates = []
        for index, conf in enumerate(working.GetConformers()):
            coords = [
                [
                    float(conf.GetAtomPosition(atom_index).x),
                    float(conf.GetAtomPosition(atom_index).y),
                    float(conf.GetAtomPosition(atom_index).z),
                ]
                for atom_index in range(working.GetNumAtoms())
            ]
            candidates.append((coords, energies[index], forcefield))
        return candidates, forcefield

    all_candidates, forcefield = embed_score_candidates(seed, 0.001)
    if len(all_candidates) < num_confs:
        all_candidates = []
        for attempt in range(max(1, oversample * 20)):
            batch, forcefield = embed_score_candidates(seed + attempt, -1.0)
            all_candidates.extend(batch)
            if len(all_candidates) >= num_confs:
                break
    if not all_candidates:
        return [], [], "embed_failed", 0
    if len(all_candidates) < num_confs:
        return [], [], f"insufficient_candidates:{len(all_candidates)}", len(all_candidates)

    candidate_pool = all_candidates[:num_confs]
    ranked = sorted(candidate_pool, key=lambda item: item[1])[:nbr_confs]
    forcefields = Counter(item[2] for item in ranked)
    forcefield_label = "+".join(f"{key}:{value}" for key, value in sorted(forcefields.items()))
    conformations = [item[0] for item in ranked]
    selected_energies = [item[1] for item in ranked]
    return conformations, selected_energies, forcefield_label, len(candidate_pool)


def target_count(mol: Chem.Mol, prop: str) -> int:
    spec = ast.literal_eval(mol.GetProp(prop))
    carbon_indices = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C"}
    return len(carbon_indices.intersection(spec))


def proportional_counts(
    base_counts: dict[int, int],
    capacities: dict[int, int],
    requested_total: int | None,
) -> dict[int, int]:
    base_total = sum(base_counts.values())
    if base_total <= 0:
        raise ValueError("Cannot allocate from empty base counts.")
    max_total = min(sum(capacities.values()), requested_total or sum(capacities.values()))
    proportions = {length: base_counts[length] / base_total for length in LENGTH_TO_SPLIT}
    for total in range(max_total, 0, -1):
        raw = {length: total * proportions[length] for length in LENGTH_TO_SPLIT}
        counts = {length: int(raw[length]) for length in LENGTH_TO_SPLIT}
        remaining = total - sum(counts.values())
        for _, length in sorted(
            ((raw[length] - counts[length], length) for length in LENGTH_TO_SPLIT),
            reverse=True,
        )[:remaining]:
            counts[length] += 1
        if all(0 < counts[length] <= capacities[length] for length in LENGTH_TO_SPLIT):
            return counts
    raise ValueError(f"Could not allocate validation counts within capacities: {capacities}")


def bond_edge_tensors(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    row: list[int] = []
    col: list[int] = []
    bond_attr: list[int] = []
    for bond in mol.GetBonds():
        row.append(bond.GetBeginAtomIdx())
        col.append(bond.GetEndAtomIdx())
        bond_attr.append(SUPPORTED_BOND_TYPES.get(bond.GetBondType(), 1))

    if row:
        edge_index_b = torch.tensor([row, col], dtype=torch.long)
        edge_attr_b = torch.tensor(bond_attr, dtype=torch.long)
        edge_index_b, edge_attr_b = to_undirected(edge_index_b, edge_attr_b)
    else:
        edge_index_b = torch.empty((2, 0), dtype=torch.long)
        edge_attr_b = torch.empty((0,), dtype=torch.long)

    n_atoms = mol.GetNumAtoms()
    radius_row = []
    radius_col = []
    for src in range(n_atoms):
        for dst in range(n_atoms):
            if src != dst:
                radius_row.append(src)
                radius_col.append(dst)
    edge_index_r = torch.tensor([radius_row, radius_col], dtype=torch.long)
    edge_attr_r = torch.zeros(edge_index_r.size(1), dtype=torch.long)

    edge_index = torch.cat([edge_index_b, edge_index_r], dim=1)
    edge_attr = torch.cat([edge_attr_b, edge_attr_r], dim=0)
    return coalesce(edge_index, edge_attr, reduce="max")


def cutoff_graph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row, col = edge_index
    dist = torch.sqrt(torch.sum((pos[row] - pos[col]) ** 2, dim=1))
    mask = dist <= cutoff
    return to_undirected(edge_index[:, mask], edge_attr[mask], reduce="max")


def mol_to_data(mol: Chem.Mol, task: str, cutoff: float) -> Data:
    prop = "13C Spectrum" if task == "13C" else "1H Spectrum"
    spec = ast.literal_eval(mol.GetProp(prop))
    type_ids: list[float] = []
    num_hs: list[float] = []
    labels: list[float] = []
    for atom in mol.GetAtoms():
        type_ids.append(float(TYPE_IDS[atom.GetSymbol()]))
        num_hs.append(float(atom.GetTotalNumHs()))
        if atom.GetIdx() not in spec:
            labels.append(-1.0)
        elif task == "13C":
            labels.append(float(spec[atom.GetIdx()]))
        else:
            base_shift, correction = spec[atom.GetIdx()]
            labels.append(float(base_shift) + float(correction) / 2.0)

    x = torch.tensor(list(zip(type_ids, num_hs, labels)), dtype=torch.float)
    conf = mol.GetConformer()
    pos = torch.tensor(
        [
            [
                float(conf.GetAtomPosition(idx).x),
                float(conf.GetAtomPosition(idx).y),
                float(conf.GetAtomPosition(idx).z),
            ]
            for idx in range(mol.GetNumAtoms())
        ],
        dtype=torch.float,
    )
    edge_index, edge_attr = bond_edge_tensors(mol)
    edge_index, edge_attr = cutoff_graph(edge_index, edge_attr, pos, cutoff)

    data = Data()
    data.x = x
    data.edge_attr = edge_attr
    data.edge_index = edge_index
    data.pos = pos
    data.name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    data.smiles = Chem.MolToSmiles(mol)
    return data


def mol_with_conformation(mol: Chem.Mol, conformation: list[list[float]]) -> Chem.Mol:
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(out.GetNumAtoms())
    conf.Set3D(True)
    for atom_index, (x, y, z) in enumerate(conformation):
        conf.SetAtomPosition(atom_index, (float(x), float(y), float(z)))
    out.AddConformer(conf)
    return out


def make_test_item(
    mol: Chem.Mol,
    task: str,
    cutoff: float,
    conformations: list[list[list[float]]],
) -> dict[str, Any]:
    data_list = [mol_to_data(mol_with_conformation(mol, conf), task, cutoff) for conf in conformations]
    data = data_list[0]
    mask = (data.x[:, 0] == 2.0).logical_and(data.x[:, -1] > -0.5)
    return {
        "name": data.name,
        "n_atoms": int(data.x.size(0)),
        "n_nmr": int(torch.sum(mask).item()),
        "n_mols": len(data_list),
        "loader": DataLoader(data_list, batch_size=len(data_list), shuffle=False),
    }


def write_pickle(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def replace_symlink(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(os.path.relpath(source, dest.parent), dest)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", r"\|")


def selected_mol_copy(item: dict[str, Any]) -> Chem.Mol:
    mol = Chem.Mol(item["mol"])
    mol.SetProp("validation_split", item["split"])
    mol.SetProp("glycan_length", str(item["glycan_length"]))
    mol.SetProp("independent_validation", "glyconmr_len123_no_casper_overlap")
    return mol


def build_validation(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(Path(args.glyconmr_manifest))
    casper_keys = {
        normalized_overlap_smiles(mol)
        for sdf in args.casper_sdf
        for mol in load_mols(Path(sdf))
    }

    base_counts: Counter[int] = Counter()
    usable_counts: Counter[int] = Counter()
    overlap_counts: Counter[int] = Counter()
    zero_target_counts: Counter[int] = Counter()
    available_by_length: dict[int, list[dict[str, Any]]] = {length: [] for length in LENGTH_TO_SPLIT}

    for mol_index, mol in enumerate(load_mols(Path(args.glyconmr_sdf))):
        source = mol.GetProp("source")
        source_file = mol.GetProp("source_file")
        row = manifest[(source, source_file)]
        length, sugar_units = glycan_length(row)
        if length not in LENGTH_TO_SPLIT:
            continue
        base_counts[length] += 1
        c13_targets = target_count(mol, "13C Spectrum")
        h1_targets = target_count(mol, "1H Spectrum")
        if c13_targets <= 0 or h1_targets <= 0:
            zero_target_counts[length] += 1
            continue
        usable_counts[length] += 1
        overlap_key = normalized_overlap_smiles(mol)
        if overlap_key in casper_keys:
            overlap_counts[length] += 1
            continue
        available_by_length[length].append(
            {
                "mol": mol,
                "source": source,
                "source_file": source_file,
                "source_name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
                "mol_index": mol_index,
                "glycan_length": length,
                "sugar_units": sugar_units,
                "carbon_labels": c13_targets,
                "hydrogen_labels": h1_targets,
                "heavy_atoms": mol.GetNumAtoms(),
                "overlap_key": overlap_key,
                "split": LENGTH_TO_SPLIT[length],
            }
        )

    capacities = {length: len(items) for length, items in available_by_length.items()}
    selected_counts = proportional_counts(dict(base_counts), capacities, args.total)
    selected: list[dict[str, Any]] = []
    for length, items in available_by_length.items():
        items = sorted(items, key=lambda item: (item["source"], item["source_file"], item["mol_index"]))
        selected.extend(rng.sample(items, selected_counts[length]))
    selected.sort(key=lambda item: (item["glycan_length"], item["source"], item["source_file"], item["mol_index"]))

    conformer_counts: Counter[int] = Counter()
    candidate_conformer_counts: Counter[int] = Counter()
    forcefield_counts: Counter[str] = Counter()
    for item in selected:
        conformations, energies, forcefield, candidate_count = generate_conformations(
            item["mol"],
            args.validation_confs,
            args.validation_conformer_oversample,
            args.conformer_seed,
            args.conformer_threads,
            not args.no_mmff_optimize,
        )
        if not conformations:
            raise RuntimeError(f"Failed to generate validation conformers for {item['source_name']}")
        if len(conformations) < args.validation_confs:
            raise RuntimeError(
                f"Only generated {len(conformations)} validation conformers for "
                f"{item['source_name']}; requested {args.validation_confs}."
            )
        item["conformations"] = conformations
        item["conformer_energies"] = energies
        item["candidate_conformers"] = candidate_count
        item["forcefield"] = forcefield
        conformer_counts[len(conformations)] += 1
        candidate_conformer_counts[candidate_count] += 1
        forcefield_counts[forcefield] += 1

    selected_by_split = {split: [] for split in LENGTH_TO_SPLIT.values()}
    for item in selected:
        selected_by_split[item["split"]].append(item)

    for task in TASKS:
        for split, items in selected_by_split.items():
            test_items = [
                make_test_item(item["mol"], task, args.cutoff, item["conformations"])
                for item in items
            ]
            write_pickle(out_dir / f"test_data_{task}_{split}.pickle", test_items)
        replace_symlink(Path(args.train_dir) / f"train_data_{task}.pickle", out_dir / f"train_data_{task}.pickle")

    manifest_rows = []
    for order, item in enumerate(selected):
        manifest_rows.append(
            {
                "selected_order": order,
                "split": item["split"],
                "glycan_length": item["glycan_length"],
                "source": item["source"],
                "source_file": item["source_file"],
                "source_name": item["source_name"],
                "mol_index": item["mol_index"],
                "heavy_atoms": item["heavy_atoms"],
                "carbon_targets": item["carbon_labels"],
                "hydrogen_targets": item["hydrogen_labels"],
                "candidate_conformers": item["candidate_conformers"],
                "conformers": len(item["conformations"]),
                "forcefield": item["forcefield"],
                "min_energy": min(item["conformer_energies"]),
                "max_energy": max(item["conformer_energies"]),
                "sugar_units": " ".join(item["sugar_units"]),
                "overlap_key": item["overlap_key"],
            }
        )
    write_csv(out_dir / "manifest.csv", manifest_rows)

    task_target_totals = {
        "13C": {
            split: sum(int(row["carbon_targets"]) for row in manifest_rows if row["split"] == split)
            for split in LENGTH_TO_SPLIT.values()
        },
        "1H": {
            split: sum(int(row["hydrogen_targets"]) for row in manifest_rows if row["split"] == split)
            for split in LENGTH_TO_SPLIT.values()
        },
    }

    combined_writer = Chem.SDWriter(str(out_dir / "glyconmr_len123_independent_validation.sdf"))
    split_writers = {
        split: Chem.SDWriter(str(out_dir / f"glyconmr_len123_independent_validation_{split}.sdf"))
        for split in LENGTH_TO_SPLIT.values()
    }
    for item in selected:
        mol = selected_mol_copy(item)
        combined_writer.write(mol)
        split_writers[item["split"]].write(mol)
    combined_writer.close()
    for writer in split_writers.values():
        writer.close()

    original_total = sum(base_counts.values())
    selected_total = len(selected)
    summary = {
        "out_dir": str(out_dir),
        "seed": args.seed,
        "selection_rule": "length 1/2/3 GlycoNMR, positive 13C/1H carbon targets, no conservative CASPER heavy-atom graph overlap",
        "overlap_key": "canonical heavy-atom graph SMILES after setting all bonds to single, clearing charge/aromaticity/stereochemistry",
        "non_sugar_units_excluded_from_length": sorted(NON_SUGAR_UNITS),
        "original_length_counts": {str(k): base_counts[k] for k in sorted(base_counts)},
        "original_length_ratios": {
            str(k): base_counts[k] / original_total for k in sorted(base_counts)
        },
        "usable_length_counts_before_overlap": {str(k): usable_counts[k] for k in sorted(base_counts)},
        "zero_target_filtered_counts": {str(k): zero_target_counts[k] for k in sorted(base_counts)},
        "casper_overlap_filtered_counts": {str(k): overlap_counts[k] for k in sorted(base_counts)},
        "available_after_filters": {str(k): capacities[k] for k in sorted(capacities)},
        "selected_counts": {str(k): selected_counts[k] for k in sorted(selected_counts)},
        "selected_ratios": {
            str(k): selected_counts[k] / selected_total for k in sorted(selected_counts)
        },
        "selected_total": selected_total,
        "validation_confs": args.validation_confs,
        "validation_candidate_confs": args.validation_confs * args.validation_conformer_oversample,
        "validation_conformer_oversample": args.validation_conformer_oversample,
        "conformer_seed": args.conformer_seed,
        "conformer_threads": args.conformer_threads,
        "mmff_optimize": not args.no_mmff_optimize,
        "conformer_counts": {str(k): conformer_counts[k] for k in sorted(conformer_counts)},
        "candidate_conformer_counts": {
            str(k): candidate_conformer_counts[k] for k in sorted(candidate_conformer_counts)
        },
        "forcefield_counts": dict(sorted(forcefield_counts.items())),
        "task_target_totals": task_target_totals,
        "task_files": {
            task: {
                split: str(out_dir / f"test_data_{task}_{split}.pickle")
                for split in LENGTH_TO_SPLIT.values()
            }
            for task in TASKS
        },
        "train_symlinks": {
            task: str(out_dir / f"train_data_{task}.pickle")
            for task in TASKS
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# GlycoNMR Length 1/2/3 Independent Validation",
        "",
        f"- selected total: `{selected_total}`",
        f"- source SDF: `{args.glyconmr_sdf}`",
        f"- output dir: `{out_dir}`",
        "- overlap removal: conservative heavy-atom graph key against CASPER mono/di/tri",
        f"- conformers per validation item: `{args.validation_confs}`",
        f"- candidate conformers per validation item: `{args.validation_confs * args.validation_conformer_oversample}`",
        f"- conformer generation: `ETKDGv3`, MMFF94 optimization when parameters are available, lowest-energy `{args.validation_confs}` selected",
        "- train_data files are symlinks to `data/datasets` for normalization/evaluation reuse.",
        "",
        "## Generation",
        "",
        "- default use: `scripts/evaluate_checkpoints.py` defaults to this small independent validation set through `--dataset_dir`.",
        "- command used for the small set:",
        f"  `conda run -n equiformer_v3 python scripts/build_glyconmr_len123_validation.py --out_dir {out_dir} --total {args.total}`",
        f"- selection seed: `{args.seed}`",
        f"- conformer seed: `{args.conformer_seed}`",
        f"- conformer threads: `{args.conformer_threads}`",
        f"- MMFF optimization: `{not args.no_mmff_optimize}`",
        f"- conformer count check: `{dict(sorted(conformer_counts.items()))}`",
        f"- candidate conformer count check: `{dict(sorted(candidate_conformer_counts.items()))}`",
        f"- forcefield check: `{dict(sorted(forcefield_counts.items()))}`",
        "",
        "| Length | Original | Usable | CASPER overlap removed | Available | Selected | Original ratio | Selected ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for length in sorted(LENGTH_TO_SPLIT):
        lines.append(
            f"| {length} | {base_counts[length]} | {usable_counts[length]} | "
            f"{overlap_counts[length]} | {capacities[length]} | {selected_counts[length]} | "
            f"{base_counts[length] / original_total:.4f} | {selected_counts[length] / selected_total:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `manifest.csv`",
            "- `summary.json`",
            "- `glyconmr_len123_independent_validation.sdf`",
            "- `glyconmr_len123_independent_validation_mo.sdf`",
            "- `glyconmr_len123_independent_validation_di.sdf`",
            "- `glyconmr_len123_independent_validation_tri.sdf`",
            "- `test_data_13C_mo.pickle`, `test_data_13C_di.pickle`, `test_data_13C_tri.pickle`",
            "- `test_data_1H_mo.pickle`, `test_data_1H_di.pickle`, `test_data_1H_tri.pickle`",
            "",
            "## Task Targets",
            "",
            "| Task | mo | di | tri | Total |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task in TASKS:
        mo = task_target_totals[task]["mo"]
        di = task_target_totals[task]["di"]
        tri = task_target_totals[task]["tri"]
        lines.append(f"| {task} | {mo} | {di} | {tri} | {mo + di + tri} |")

    lines.extend(
        [
            "",
            "## Selected Records",
            "",
            "| # | Split | Length | Source | Source file | Source name | Mol index | Heavy atoms | 13C targets | 1H targets | Candidate conformers | Selected conformers | Forcefield | Min energy | Max energy | Sugar units |",
            "| ---: | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in manifest_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(row["selected_order"]),
                    md_cell(row["split"]),
                    md_cell(row["glycan_length"]),
                    md_cell(row["source"]),
                    md_cell(row["source_file"]),
                    md_cell(row["source_name"]),
                    md_cell(row["mol_index"]),
                    md_cell(row["heavy_atoms"]),
                    md_cell(row["carbon_targets"]),
                    md_cell(row["hydrogen_targets"]),
                    md_cell(row["candidate_conformers"]),
                    md_cell(row["conformers"]),
                    md_cell(row["forcefield"]),
                    f"{float(row['min_energy']):.6f}",
                    f"{float(row['max_energy']):.6f}",
                    md_cell(row["sugar_units"]),
                ]
            )
            + " |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a GlycoNMR length 1/2/3 independent validation set.")
    parser.add_argument("--glyconmr_sdf", default=str(DEFAULT_GLYCONMR_SDF))
    parser.add_argument("--glyconmr_manifest", default=str(DEFAULT_GLYCONMR_MANIFEST))
    parser.add_argument("--casper_sdf", nargs="+", default=[str(path) for path in DEFAULT_CASPER_SDFS])
    parser.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total", type=int, default=None, help="Optional requested total; default is maximum proportional total.")
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--validation_confs", type=int, default=100)
    parser.add_argument("--validation_conformer_oversample", type=int, default=2)
    parser.add_argument("--conformer_seed", type=int, default=1)
    parser.add_argument("--conformer_threads", type=int, default=10)
    parser.add_argument("--no_mmff_optimize", action="store_true")
    return parser


def main() -> None:
    summary = build_validation(build_parser().parse_args())
    print(f"Wrote {summary['selected_total']} validation molecules to {summary['out_dir']}")
    print(f"Selected counts: {summary['selected_counts']}")


if __name__ == "__main__":
    main()
