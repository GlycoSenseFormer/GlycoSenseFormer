#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import networkx as nx
from networkx.algorithms import isomorphism
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
SAMPLE_CSV = Path(os.environ.get("MD_SAMPLE_CSV", MD_ROOT / "md_small_sample_dataset_v1" / "md_small_sample_glycans_v1.csv"))
RUN_ROOT = Path(os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_extended24"))
SDF_BY_CLASS = {
    "mono": ROOT / "data" / "dataset" / "casper_mono.sdf",
    "dimer": ROOT / "data" / "dataset" / "casper_dimer.sdf",
    "trimer": ROOT / "data" / "dataset" / "casper_trimer.sdf",
}


def find_mol(path: Path, molecule_name: str) -> Chem.Mol:
    for mol in Chem.SDMolSupplier(str(path), removeHs=False):
        if mol is None:
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name") == molecule_name:
            return mol
    raise ValueError(f"Could not find {molecule_name!r} in {path}")


def sdf_heavy_graph(mol: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    rdkit_to_heavy: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        heavy_idx = len(rdkit_to_heavy)
        rdkit_to_heavy[atom.GetIdx()] = heavy_idx
        graph.add_node(heavy_idx, elem=atom.GetSymbol())
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in rdkit_to_heavy and end in rdkit_to_heavy:
            graph.add_edge(rdkit_to_heavy[begin], rdkit_to_heavy[end])
    return graph


def infer_element(atom_name: str, mass: float) -> str:
    if mass < 2.0:
        return "H"
    stripped = atom_name.strip().upper()
    if stripped.startswith("CL"):
        return "Cl"
    if stripped.startswith("BR"):
        return "Br"
    if stripped.startswith("C") or stripped.startswith("CH"):
        return "C"
    if stripped.startswith("O"):
        return "O"
    if stripped.startswith("N"):
        return "N"
    if stripped.startswith("S"):
        return "S"
    if stripped.startswith("P"):
        return "P"
    return stripped[:1].title()


def parse_gromacs_top_heavy_graph(top_path: Path) -> nx.Graph:
    atoms: dict[int, dict[str, object]] = {}
    bonds: list[tuple[int, int]] = []
    section = ""
    for raw in top_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip()
            continue
        parts = line.split()
        if section == "atoms" and len(parts) >= 8 and parts[0].isdigit():
            nr = int(parts[0])
            atom_name = parts[4]
            mass = float(parts[7])
            elem = infer_element(atom_name, mass)
            atoms[nr] = {"elem": elem, "atom_name": atom_name, "mass": mass}
        elif section == "bonds" and len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            bonds.append((int(parts[0]), int(parts[1])))

    top_to_heavy: dict[int, int] = {}
    graph = nx.Graph()
    for nr in sorted(atoms):
        elem = atoms[nr]["elem"]
        if elem == "H":
            continue
        heavy_idx = len(top_to_heavy)
        top_to_heavy[nr] = heavy_idx
        graph.add_node(
            heavy_idx,
            elem=elem,
            atom_name=atoms[nr]["atom_name"],
            top_atom_index=nr,
        )
    for begin, end in bonds:
        if begin in top_to_heavy and end in top_to_heavy:
            graph.add_edge(top_to_heavy[begin], top_to_heavy[end])
    return graph


def build_map(sdf_graph: nx.Graph, gmx_graph: nx.Graph) -> list[int]:
    if sdf_graph.number_of_nodes() != gmx_graph.number_of_nodes():
        raise ValueError(f"node count mismatch: SDF {sdf_graph.number_of_nodes()} vs GMX {gmx_graph.number_of_nodes()}")
    if sdf_graph.number_of_edges() != gmx_graph.number_of_edges():
        raise ValueError(f"edge count mismatch: SDF {sdf_graph.number_of_edges()} vs GMX {gmx_graph.number_of_edges()}")
    matcher = isomorphism.GraphMatcher(
        sdf_graph,
        gmx_graph,
        node_match=isomorphism.categorical_node_match("elem", None),
    )
    try:
        mapping = next(matcher.isomorphisms_iter())
    except StopIteration as exc:
        raise ValueError("no heavy-atom graph isomorphism found") from exc
    return [int(mapping[i]) for i in range(sdf_graph.number_of_nodes())]


def main() -> None:
    rows: list[dict[str, str]]
    with SAMPLE_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["priority"]))

    results = []
    for row in rows:
        molecule = row["molecule_name"]
        clean = row["clean_name"]
        source_class = row["source_class"]
        dryrun_dir = RUN_ROOT / clean / "glycam06j_topology_dryrun"
        top_path = dryrun_dir / f"{clean}_GLYCAM06j.amb2gmx" / f"{clean}_GLYCAM06j_GMX.top"
        map_path = dryrun_dir / "sdf_to_glycam_frame_heavy_index_map.json"
        try:
            mol = find_mol(SDF_BY_CLASS[source_class], molecule)
            mapping = build_map(sdf_heavy_graph(mol), parse_gromacs_top_heavy_graph(top_path))
            map_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
            status = "done"
            message = f"{len(mapping)} heavy atoms"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            message = str(exc)
        results.append(
            {
                "priority": int(row["priority"]),
                "molecule_name": molecule,
                "clean_name": clean,
                "status": status,
                "map_path": str(map_path),
                "message": message,
            }
        )

    out_json = RUN_ROOT / "extended24_glycam06j_atom_index_map_status.json"
    out_csv = RUN_ROOT / "extended24_glycam06j_atom_index_map_status.csv"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(
        json.dumps(
            {
                "done": sum(1 for result in results if result["status"] == "done"),
                "failed": sum(1 for result in results if result["status"] == "failed"),
                "run_root": str(RUN_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
