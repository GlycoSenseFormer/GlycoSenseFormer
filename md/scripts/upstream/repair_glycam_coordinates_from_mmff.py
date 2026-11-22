#!/usr/bin/env python3
"""Graft validated MMFF heavy-atom geometry onto a GLYCAM/ACPYPE topology.

LEaP can build correct parameters for manually connected branched/1->1
glycans while leaving independently created residues overlapped.  This
module preserves the GLYCAM topology and replaces only the starting
coordinates with an existing MMFF94-optimized conformer from the project.
Hydrogens are rebuilt by RDKit around the fixed heavy-atom geometry.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import shutil
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from rdkit import Chem
from rdkit.Geometry import Point3D


def find_molecule(path: Path, molecule_name: str) -> Chem.Mol:
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name") == molecule_name:
            return mol
    raise ValueError(f"Could not find {molecule_name!r} in {path}")


def infer_element(atom_name: str, mass: float) -> str:
    if mass < 2.0:
        return "H"
    name = atom_name.strip().upper()
    if name.startswith("CL"):
        return "Cl"
    if name.startswith("BR"):
        return "Br"
    return name[:1].title()


def parse_gromacs_topology(path: Path) -> tuple[nx.Graph, dict[int, dict[str, object]]]:
    graph = nx.Graph()
    atoms: dict[int, dict[str, object]] = {}
    bonds: list[tuple[int, int]] = []
    section = ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and "]" in line:
            section = line[1 : line.index("]")].strip().lower()
            continue
        fields = line.split()
        if section == "atoms" and len(fields) >= 8:
            try:
                index = int(fields[0]) - 1
                mass = float(fields[7])
            except ValueError:
                continue
            atoms[index] = {
                "name": fields[4],
                "residue": fields[3],
                "mass": mass,
                "element": infer_element(fields[4], mass),
            }
        elif section == "bonds" and len(fields) >= 2:
            try:
                bonds.append((int(fields[0]) - 1, int(fields[1]) - 1))
            except ValueError:
                continue
    for index, atom in atoms.items():
        graph.add_node(index, element=atom["element"])
    for left, right in bonds:
        if left in atoms and right in atoms:
            graph.add_edge(left, right)
    return graph, atoms


def heavy_graph_from_rdkit(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:
            graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in graph and right in graph:
            graph.add_edge(left, right)
    return graph


def heavy_graph_from_topology(graph: nx.Graph) -> nx.Graph:
    heavy = nx.Graph()
    for index, attributes in graph.nodes(data=True):
        if attributes["element"] != "H":
            heavy.add_node(index, **attributes)
    for left, right in graph.edges():
        if left in heavy and right in heavy:
            heavy.add_edge(left, right)
    return heavy


def assign_mmff_conformer(mol: Chem.Mol, coordinates: np.ndarray) -> Chem.Mol:
    if coordinates.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(
            f"MMFF conformer shape {coordinates.shape} does not match {mol.GetNumAtoms()} SDF atoms"
        )
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    conformer = Chem.Conformer(work.GetNumAtoms())
    conformer.Set3D(True)
    for index, xyz in enumerate(coordinates):
        if not all(math.isfinite(float(value)) for value in xyz):
            raise ValueError(f"non-finite MMFF coordinate at atom {index}")
        conformer.SetAtomPosition(index, Point3D(*(float(value) for value in xyz)))
    work.AddConformer(conformer, assignId=True)
    return Chem.AddHs(work, addCoords=True)


def build_full_mapping(rdkit_mol: Chem.Mol, top_graph: nx.Graph) -> dict[int, int]:
    source_heavy = heavy_graph_from_rdkit(rdkit_mol)
    target_heavy = heavy_graph_from_topology(top_graph)
    if source_heavy.number_of_nodes() != target_heavy.number_of_nodes():
        raise ValueError(
            f"heavy-atom count mismatch: MMFF={source_heavy.number_of_nodes()}, "
            f"GLYCAM={target_heavy.number_of_nodes()}"
        )
    if source_heavy.number_of_edges() != target_heavy.number_of_edges():
        raise ValueError(
            f"heavy-bond count mismatch: MMFF={source_heavy.number_of_edges()}, "
            f"GLYCAM={target_heavy.number_of_edges()}"
        )
    matcher = isomorphism.GraphMatcher(
        source_heavy,
        target_heavy,
        node_match=isomorphism.categorical_node_match("element", None),
    )
    try:
        heavy_mapping = next(matcher.isomorphisms_iter())
    except StopIteration as exc:
        raise ValueError("no heavy-atom graph isomorphism between MMFF and GLYCAM") from exc

    mapping: dict[int, int] = dict(heavy_mapping)
    for source_heavy_index, target_heavy_index in heavy_mapping.items():
        source_hydrogens = sorted(
            neighbor
            for neighbor in (atom.GetIdx() for atom in rdkit_mol.GetAtomWithIdx(source_heavy_index).GetNeighbors())
            if rdkit_mol.GetAtomWithIdx(neighbor).GetAtomicNum() == 1
        )
        target_hydrogens = sorted(
            neighbor
            for neighbor in top_graph.neighbors(target_heavy_index)
            if top_graph.nodes[neighbor]["element"] == "H"
        )
        if len(source_hydrogens) != len(target_hydrogens):
            raise ValueError(
                f"hydrogen-count mismatch at mapped heavy atom {source_heavy_index}->{target_heavy_index}: "
                f"MMFF={len(source_hydrogens)}, GLYCAM={len(target_hydrogens)}"
            )
        mapping.update(zip(source_hydrogens, target_hydrogens))

    if len(mapping) != rdkit_mol.GetNumAtoms() or len(set(mapping.values())) != top_graph.number_of_nodes():
        raise ValueError(
            f"incomplete full-atom mapping: MMFF={len(mapping)}/{rdkit_mol.GetNumAtoms()}, "
            f"GLYCAM unique={len(set(mapping.values()))}/{top_graph.number_of_nodes()}"
        )
    return mapping


def parse_gro(path: Path) -> tuple[list[str], int, list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("GRO file is too short")
    atom_count = int(lines[1].strip())
    if len(lines) != atom_count + 3:
        raise ValueError(f"GRO declares {atom_count} atoms but contains {len(lines) - 3}")
    box = [float(value) for value in lines[-1].split()]
    if len(box) not in (3, 9) or any(value <= 0 for value in box[:3]):
        raise ValueError("invalid GRO box")
    return lines, atom_count, box


def write_mapped_gro(
    source_gro: Path,
    output_gro: Path,
    mol: Chem.Mol,
    mapping: dict[int, int],
) -> None:
    lines, atom_count, box = parse_gro(source_gro)
    if atom_count != mol.GetNumAtoms():
        raise ValueError(f"GRO={atom_count} atoms but mapped MMFF molecule={mol.GetNumAtoms()}")
    conformer = mol.GetConformer()
    coordinates = np.zeros((atom_count, 3), dtype=float)
    for source_index, target_index in mapping.items():
        point = conformer.GetAtomPosition(source_index)
        coordinates[target_index] = (point.x / 10.0, point.y / 10.0, point.z / 10.0)
    coordinates -= coordinates.mean(axis=0)
    coordinates += np.asarray(box[:3], dtype=float) / 2.0
    output_lines = lines[:2]
    for index, original in enumerate(lines[2 : 2 + atom_count]):
        if len(original) < 44:
            raise ValueError(f"short GRO atom line {index + 1}")
        x, y, z = coordinates[index]
        output_lines.append(f"{original[:20]}{x:8.3f}{y:8.3f}{z:8.3f}{original[44:]}")
    output_lines.append(lines[-1])
    output_gro.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repair_coordinates(
    *,
    molecule_name: str,
    source_sdf: Path,
    mmff_pickle: Path,
    gromacs_top: Path,
    gromacs_gro: Path,
    atom_map_output: Path,
    metadata_output: Path,
) -> dict[str, object]:
    source_mol = find_molecule(source_sdf, molecule_name)
    if any(atom.GetAtomicNum() == 1 for atom in source_mol.GetAtoms()):
        raise ValueError("source SDF unexpectedly contains explicit hydrogens")
    with mmff_pickle.open("rb") as handle:
        conformers = pickle.load(handle)
    if not isinstance(conformers, list) or not conformers:
        raise ValueError(f"no MMFF conformers in {mmff_pickle}")
    mmff_mol = assign_mmff_conformer(source_mol, np.asarray(conformers[0], dtype=float))
    top_graph, _top_atoms = parse_gromacs_topology(gromacs_top)
    mapping = build_full_mapping(mmff_mol, top_graph)

    raw_gro = gromacs_gro.with_name(gromacs_gro.stem + "_tleap_raw.gro")
    if not raw_gro.exists():
        shutil.copy2(gromacs_gro, raw_gro)
    write_mapped_gro(raw_gro, gromacs_gro, mmff_mol, mapping)

    target_heavy_order = [
        index for index in sorted(top_graph) if top_graph.nodes[index]["element"] != "H"
    ]
    target_to_heavy_frame = {top_index: frame_index for frame_index, top_index in enumerate(target_heavy_order)}
    sdf_to_frame = [target_to_heavy_frame[mapping[index]] for index in range(source_mol.GetNumAtoms())]
    atom_map_output.write_text(json.dumps(sdf_to_frame, indent=2) + "\n", encoding="utf-8")

    metadata: dict[str, object] = {
        "molecule_name": molecule_name,
        "method": "GLYCAM topology with MMFF94-optimized heavy-atom start geometry",
        "source_sdf": str(source_sdf),
        "source_mmff_pickle": str(mmff_pickle),
        "source_conformer_index": 0,
        "raw_tleap_gro": str(raw_gro),
        "repaired_gro": str(gromacs_gro),
        "gromacs_top": str(gromacs_top),
        "sdf_heavy_atoms": source_mol.GetNumAtoms(),
        "all_atoms": mmff_mol.GetNumAtoms(),
        "input_sha256": {
            "sdf": sha256(source_sdf),
            "mmff_pickle": sha256(mmff_pickle),
            "top": sha256(gromacs_top),
            "raw_gro": sha256(raw_gro),
        },
        "output_sha256": {
            "gro": sha256(gromacs_gro),
            "atom_map": sha256(atom_map_output),
        },
    }
    metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
