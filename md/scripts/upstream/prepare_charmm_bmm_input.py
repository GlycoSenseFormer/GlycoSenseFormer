#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


SDF_TO_BMM_HEAVY = {
    4: "1C1",
    2: "1C5",
    3: "1O5",
    5: "1C2",
    6: "1O2",
    7: "1C3",
    8: "1O3",
    9: "1C4",
    10: "1O4",
    1: "1C6",
    0: "1O6",
    13: "2C1",
    12: "2O1",
    15: "2C5",
    14: "2O5",
    22: "2C2",
    23: "2O2",
    20: "2C3",
    21: "2O3",
    18: "2C4",
    19: "2O4",
    16: "2C6",
    17: "2O6",
    11: "M1",
}

HEAVY_TO_H_NAMES = {
    "1C1": ["1H1"],
    "1C5": ["1H5"],
    "1C2": ["1H2"],
    "1O2": ["1HO2"],
    "1C3": ["1H3"],
    "1O3": ["1HO3"],
    "1C4": ["1H4"],
    "1O4": ["1HO4"],
    "1C6": ["1H61", "1H62"],
    "1O6": ["1HO6"],
    "2C1": ["2H1"],
    "2C5": ["2H5"],
    "2C2": ["2H2"],
    "2O2": ["2HO2"],
    "2C3": ["2H3"],
    "2O3": ["2HO3"],
    "2C4": ["2H4"],
    "2C6": ["2H61", "2H62"],
    "2O6": ["2HO6"],
    "M1": ["HM1", "HM2", "HM3"],
}

BMM_ORDER = [
    "1C1", "1H1", "1C5", "1H5", "1O5", "1C2", "1H2", "1O2", "1HO2",
    "1C3", "1H3", "1O3", "1HO3", "1C4", "1H4", "1O4", "1HO4",
    "1C6", "1H61", "1H62", "1O6", "1HO6", "2C1", "2H1", "2O1",
    "2C5", "2H5", "2O5", "2C2", "2H2", "2O2", "2HO2", "2C3",
    "2H3", "2O3", "2HO3", "2C4", "2H4", "2O4", "2C6", "2H61",
    "2H62", "2O6", "2HO6", "M1", "HM1", "HM2", "HM3",
]

BMM_RTP = """

[ BMM ]
; methyl-beta-D-maltoside, local exact-molecule test residue derived from CHARMM36 BOM
  [ atoms ]
      1C1   CC3162  0.2900   1
      1H1     HCA1  0.0900   1
      1C5   CC3163  0.1100   1
      1H5     HCA1  0.0900   1
      1O5   OC3C61 -0.4000   1
      1C2   CC3161  0.1400   2
      1H2     HCA1  0.0900   2
      1O2    OC311 -0.6500   2
     1HO2     HCP1  0.4200   2
      1C3   CC3161  0.1400   3
      1H3     HCA1  0.0900   3
      1O3    OC311 -0.6500   3
     1HO3     HCP1  0.4200   3
      1C4   CC3161  0.1400   4
      1H4     HCA1  0.0900   4
      1O4    OC311 -0.6500   4
     1HO4     HCP1  0.4200   4
      1C6    CC321  0.0500   5
     1H61     HCA2  0.0900   5
     1H62     HCA2  0.0900   5
      1O6    OC311 -0.6500   5
     1HO6     HCP1  0.4200   5
      2C1   CC3162  0.2900   6
      2H1     HCA1  0.0900   6
      2O1    OC301 -0.3600   6
      2C5   CC3163  0.1100   6
      2H5     HCA1  0.0900   6
      2O5   OC3C61 -0.4000   6
      2C2   CC3161  0.1400   7
      2H2     HCA1  0.0900   7
      2O2    OC311 -0.6500   7
     2HO2     HCP1  0.4200   7
      2C3   CC3161  0.1400   8
      2H3     HCA1  0.0900   8
      2O3    OC311 -0.6500   8
     2HO3     HCP1  0.4200   8
      2C4   CC3161  0.0900   9
      2H4     HCA1  0.0900   9
      2O4    OC301 -0.3600   9
      2C6    CC321  0.0500  10
     2H61     HCA2  0.0900  10
     2H62     HCA2  0.0900  10
      2O6    OC311 -0.6500  10
     2HO6     HCP1  0.4200  10
       M1    CC331 -0.0900  11
      HM1     HCA3  0.0900  11
      HM2     HCA3  0.0900  11
      HM3     HCA3  0.0900  11
  [ bonds ]
      1C1   1H1
      1C1   1O5
      1C1   1C2
      1C2   1H2
      1C2   1O2
      1O2  1HO2
      1C2   1C3
      1C3   1H3
      1C3   1O3
      1O3  1HO3
      1C3   1C4
      1C4   1H4
      1C4   1O4
      1O4  1HO4
      1C4   1C5
      1C5   1H5
      1C5   1C6
      1C6  1H61
      1C6  1H62
      1C6   1O6
      1O6  1HO6
      1C5   1O5
      2O4   1C1
      2C1   2O1
      2C1   2H1
      2C1   2O5
      2C1   2C2
      2C2   2H2
      2C2   2O2
      2O2  2HO2
      2C2   2C3
      2C3   2H3
      2C3   2O3
      2O3  2HO3
      2C3   2C4
      2C4   2H4
      2C4   2O4
      2C4   2C5
      2C5   2H5
      2C5   2C6
      2C6  2H61
      2C6  2H62
      2C6   2O6
      2O6  2HO6
      2C5   2O5
      2O1    M1
       M1   HM1
       M1   HM2
       M1   HM3
"""

OLD_TO_SAFE = {
    "1C1": "C1A", "1H1": "H1A", "1C5": "C5A", "1H5": "H5A", "1O5": "O5A",
    "1C2": "C2A", "1H2": "H2A", "1O2": "O2A", "1HO2": "HO2A", "1C3": "C3A",
    "1H3": "H3A", "1O3": "O3A", "1HO3": "HO3A", "1C4": "C4A", "1H4": "H4A",
    "1O4": "O4A", "1HO4": "HO4A", "1C6": "C6A", "1H61": "H61A", "1H62": "H62A",
    "1O6": "O6A", "1HO6": "HO6A",
    "2C1": "C1B", "2H1": "H1B", "2O1": "O1B", "2C5": "C5B", "2H5": "H5B",
    "2O5": "O5B", "2C2": "C2B", "2H2": "H2B", "2O2": "O2B", "2HO2": "HO2B",
    "2C3": "C3B", "2H3": "H3B", "2O3": "O3B", "2HO3": "HO3B", "2C4": "C4B",
    "2H4": "H4B", "2O4": "O4B", "2C6": "C6B", "2H61": "H61B", "2H62": "H62B",
    "2O6": "O6B", "2HO6": "HO6B",
}


def safe_name(name: str) -> str:
    return OLD_TO_SAFE.get(name, name)


def safe_rtp(text: str) -> str:
    for old, new in sorted(OLD_TO_SAFE.items(), key=lambda item: len(item[0]), reverse=True):
        text = re.sub(rf"(?<!\S){re.escape(old)}(?!\S)", new, text)
    return text


SDF_TO_BMM_HEAVY = {idx: safe_name(name) for idx, name in SDF_TO_BMM_HEAVY.items()}
HEAVY_TO_H_NAMES = {
    safe_name(heavy): [safe_name(hydrogen) for hydrogen in hydrogens]
    for heavy, hydrogens in HEAVY_TO_H_NAMES.items()
}
BMM_ORDER = [safe_name(name) for name in BMM_ORDER]
BMM_RTP = safe_rtp(BMM_RTP)


def find_molecule(sdf_path: Path, molecule_name: str) -> Chem.Mol:
    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if mol is None:
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name") == molecule_name:
            return mol
    raise ValueError(f"Could not find {molecule_name!r} in {sdf_path}")


def embed_with_hydrogens(mol: Chem.Mol, seed: int) -> Chem.Mol:
    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    Chem.SanitizeMol(work)
    work = Chem.AddHs(work, addCoords=True)
    params = Chem.rdDistGeom.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True
    status = AllChem.EmbedMolecule(work, params)
    if status != 0:
        raise RuntimeError(f"RDKit EmbedMolecule failed with status {status}")
    if AllChem.MMFFHasAllMoleculeParams(work):
        AllChem.MMFFOptimizeMolecule(work, maxIters=200, mmffVariant="MMFF94")
    else:
        AllChem.UFFOptimizeMolecule(work, maxIters=200)
    return work


def name_atoms(mol: Chem.Mol) -> dict[int, str]:
    atom_names: dict[int, str] = {}
    for sdf_idx, name in SDF_TO_BMM_HEAVY.items():
        atom_names[sdf_idx] = name
        h_neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(sdf_idx).GetNeighbors() if nbr.GetAtomicNum() == 1]
        expected = HEAVY_TO_H_NAMES.get(name, [])
        if len(h_neighbors) != len(expected):
            raise ValueError(f"{name} expected {len(expected)} hydrogens, found {len(h_neighbors)}")
        for h_idx, h_name in zip(sorted(h_neighbors), expected):
            atom_names[h_idx] = h_name
    if len(atom_names) != mol.GetNumAtoms():
        missing = sorted(set(range(mol.GetNumAtoms())) - set(atom_names))
        raise ValueError(f"Missing names for atom indices {missing}")
    return atom_names


def pdb_atom_name(name: str, element: str) -> str:
    if len(name) >= 4:
        return name[:4]
    if len(element) == 1 and not name[0].isdigit():
        return f" {name:<3s}"
    return f"{name:<4s}"


def write_pdb(mol: Chem.Mol, atom_names: dict[int, str], output: Path) -> None:
    conf = mol.GetConformer()
    name_to_idx = {name: idx for idx, name in atom_names.items()}
    lines = ["REMARK bDGlc14bDGlcOMe mapped to local CHARMM36 BMM residue\n"]
    serial = 1
    for name in BMM_ORDER:
        idx = name_to_idx[name]
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        element = atom.GetSymbol().upper()
        atom_field = pdb_atom_name(name, element)
        lines.append(
            f"HETATM{serial:5d} {atom_field:4s} BMM A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {element:>2s}\n"
        )
        serial += 1
    lines.append("TER\nEND\n")
    output.write_text("".join(lines), encoding="utf-8")


def write_gro(mol: Chem.Mol, atom_names: dict[int, str], output: Path) -> None:
    conf = mol.GetConformer()
    name_to_idx = {name: idx for idx, name in atom_names.items()}
    lines = ["bDGlc14bDGlcOMe mapped to local CHARMM36 BMM residue\n", f"{len(BMM_ORDER):5d}\n"]
    for serial, name in enumerate(BMM_ORDER, start=1):
        idx = name_to_idx[name]
        pos = conf.GetAtomPosition(idx)
        lines.append(
            f"{1:5d}{'BMM':<5s}{name:>5s}{serial:5d}"
            f"{pos.x / 10.0:8.3f}{pos.y / 10.0:8.3f}{pos.z / 10.0:8.3f}\n"
        )
    lines.append("   8.00000   8.00000   8.00000\n")
    output.write_text("".join(lines), encoding="utf-8")


def prepare_forcefield(source_ff: Path, dest_ff: Path) -> None:
    dest_ff.mkdir(parents=True, exist_ok=True)
    for source in source_ff.iterdir():
        target = dest_ff / source.name
        if target.exists() or target.is_symlink():
            continue
        if source.name == "carb.rtp":
            shutil.copy2(source, target)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(BMM_RTP)
        else:
            os.symlink(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local CHARMM36 BMM test inputs.")
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--source-ff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_ff = Path(args.source_ff).resolve()
    dest_ff = output_dir / "charmm36_bmm.ff"
    prepare_forcefield(source_ff, dest_ff)

    mol = embed_with_hydrogens(find_molecule(Path(args.sdf), args.molecule_name), args.seed)
    atom_names = name_atoms(mol)
    pdb_out = output_dir / f"{args.molecule_name}_BMM_charmm36_rdkit3d_allatom.pdb"
    gro_out = output_dir / f"{args.molecule_name}_BMM_charmm36_rdkit3d_allatom.gro"
    write_pdb(mol, atom_names, pdb_out)
    write_gro(mol, atom_names, gro_out)

    heavy_order = [SDF_TO_BMM_HEAVY[i] for i in sorted(SDF_TO_BMM_HEAVY)]
    frame_heavy_order = [name for name in BMM_ORDER if not name.startswith(("H", "1H", "2H")) and "HO" not in name]
    frame_index_by_name = {name: idx for idx, name in enumerate(frame_heavy_order)}
    mapping = [frame_index_by_name[SDF_TO_BMM_HEAVY[i]] for i in range(len(SDF_TO_BMM_HEAVY))]
    mapping_out = output_dir / "sdf_to_charmm_frame_heavy_index_map.json"
    mapping_out.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "molecule_name": args.molecule_name,
        "source_sdf": args.sdf,
        "source_forcefield": str(source_ff),
        "local_forcefield": str(dest_ff),
        "custom_residue": "BMM",
        "seed": args.seed,
        "sdf_index_to_bmm_atom_name": {str(k): v for k, v in sorted(SDF_TO_BMM_HEAVY.items())},
        "frame_heavy_order": frame_heavy_order,
        "sdf_to_frame_heavy_index_map": mapping,
        "note": "BMM is a local exact-molecule methyl-beta-D-maltoside residue derived from CHARMM36 BOM by replacing the octyl tail with methyl.",
    }
    (output_dir / "charmm_bmm_input_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(pdb_out)
    print(gro_out)
    print(mapping_out)
    print(dest_ff)


if __name__ == "__main__":
    main()
