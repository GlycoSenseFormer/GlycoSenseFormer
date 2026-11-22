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


SDF_TO_SITE = {
    0: ("GLC0", "O6"),
    1: ("GLC0", "C6"),
    2: ("GLC0", "C5"),
    3: ("GLC0", "O5"),
    4: ("GLC0", "C1"),
    5: ("GLC0", "C2"),
    6: ("GLC0", "O2"),
    7: ("GLC0", "C3"),
    8: ("GLC0", "O3"),
    9: ("GLC0", "C4"),
    10: ("GLC0", "O4"),
    11: ("GCM", "M1"),
    12: ("GCM", "O1"),
    13: ("GCM", "C1"),
    14: ("GCM", "O5"),
    15: ("GCM", "C5"),
    16: ("GCM", "C6"),
    17: ("GCM", "O6"),
    18: ("GCM", "C4"),
    19: ("GLC0", "O1"),
    20: ("GCM", "C3"),
    21: ("GCM", "O3"),
    22: ("GCM", "C2"),
    23: ("GCM", "O2"),
}

HEAVY_TO_H_NAMES = {
    ("GLC0", "O2"): ["HO2"],
    ("GLC0", "O3"): ["HO3"],
    ("GLC0", "O4"): ["HO4"],
    ("GLC0", "O6"): ["HO6"],
    ("GCM", "O2"): ["HO2"],
    ("GCM", "O3"): ["HO3"],
    ("GCM", "O6"): ["HO6"],
}

RESIDUE_ORDER = {
    "GLC0": [
        "C4", "O4", "HO4", "C3", "O3", "HO3", "C2", "O2", "HO2",
        "C6", "O6", "HO6", "C5", "O5", "C1", "O1",
    ],
    "GCM": [
        "C4", "C3", "O3", "HO3", "C2", "O2", "HO2", "C6", "O6",
        "HO6", "C5", "O5", "C1", "O1", "M1",
    ],
}

GCM_RTP = """

[ GCM ]
 [ atoms ]
   C4   CH1R    0.23200     0
   C3   CH1R    0.23200     2
   O3    OA    -0.64200     2
  HO3     H     0.41000     2
   C2   CH1R    0.23200     3
   O2    OA    -0.64200     3
  HO2     H     0.41000     3
   C6   CH2     0.23200     4
   O6    OA    -0.64200     4
  HO6     H     0.41000     4
   C5   CH1R    0.23200     5
   O5    OR    -0.46400     5
   C1   CH1R    0.46400     5
   O1    OE    -0.46400     5
   M1    CH3    0.23200     5

 [ bonds ]
   O2 HO2  gb_1
   O3 HO3  gb_1
   O6 HO6  gb_1
   C4 C3   gb_26
   C4 C5   gb_26
   C2 O2   gb_20
   C2 C3   gb_26
   C2 C1   gb_26
   C3 O3   gb_20
   C6 O6   gb_20
   C6 C5   gb_26
   C5 O5   gb_20
   O5 C1   gb_20
   C1 O1   gb_20
   O1 M1   gb_18

 [ angles ]
    C2   O2   HO2   ga_12
    C3   O3   HO3   ga_12
    C6   O6   HO6   ga_12
    C3   C4   C5    ga_8
    O2   C2   C3    ga_9
    O2   C2   C1    ga_9
    C3   C2   C1    ga_8
    C4   C3   C2    ga_8
    C4   C3   O3    ga_9
    C2   C3   O3    ga_9
    O6   C6   C5    ga_9
    C4   C5   C6    ga_8
    C4   C5   O5    ga_9
    C6   C5   O5    ga_9
    C5   O5   C1    ga_10
    C2   C1   O5    ga_9
    C2   C1   O1    ga_9
    O5   C1   O1    ga_9
    C1   O1   M1    ga_20

 [ impropers ]
   C2    C3    C1   O2    gi_2
   O3    C4    C2   C3    gi_2
   C5    C4    O5   C6    gi_2
   C1    O5    C2   O1    gi_2

 [ dihedrals ]
   HO2  O2   C2  C1   gdc_t3
   C2   C3   O3  HO3  gdc_t3
   C5   C6   O6  HO6  gdc_t3
   C5   C4   C3  C2   gdc_t1
   C3   C4   C5  O5   gdc_t1
   O2   C2   C3  O3   gdc_t12
   C1   C2   C3  C4   gdc_t1
   O2   C2   C1  O5   gdc_t13
   O2   C2   C1  O1   gdc_t12
   C3   C2   C1  O5   gdc_t1
   O6   C6   C5  C4   gdc_t1
   O6   C6   C5  C4   gdc_t10
   O6   C6   C5  O5   gdc_t11
   O6   C6   C5  O5   gdc_t12
   C4   C5   O5  C1   gdc_t2
   C5   O5   C1  C2   gdc_t2
   C5   O5   C1  O1   gdc_t5
   C2   C1   O1  M1   gdc_t7
   O5   C1   O1  M1   gdc_t2
   O5   C1   O1  M1   gdc_t4
   O5   C1   O1  M1   gdc_t6

[ exclusions ]
   C6    O1
   C6    O3
  HO2    C4
  HO2    O5
  HO3    C1
  HO3    C5
"""

GCM_HDB = """
GCM     3
1       2       HO2     O2      C2      C3
1       2       HO3     O3      C3      C2
1       2       HO6     O6      C6      C5
"""


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


def residue_atom_key(residue: str, atom_name: str) -> str:
    return f"{residue}:{atom_name}"


def name_atoms(mol: Chem.Mol) -> dict[int, tuple[str, str]]:
    atom_names: dict[int, tuple[str, str]] = {}
    for sdf_idx, site in SDF_TO_SITE.items():
        atom_names[sdf_idx] = site
        h_neighbors = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(sdf_idx).GetNeighbors() if nbr.GetAtomicNum() == 1]
        expected = HEAVY_TO_H_NAMES.get(site, [])
        if expected and len(h_neighbors) != len(expected):
            raise ValueError(f"{site} expected {len(expected)} polar hydrogens, found {len(h_neighbors)}")
        for h_idx, h_name in zip(sorted(h_neighbors), expected):
            atom_names[h_idx] = (site[0], h_name)
    return atom_names


def pdb_atom_name(name: str, element: str) -> str:
    if len(name) >= 4:
        return name[:4]
    if len(element) == 1 and not name[0].isdigit():
        return f" {name:<3s}"
    return f"{name:<4s}"


def ordered_sites() -> list[tuple[str, str]]:
    sites: list[tuple[str, str]] = []
    for residue in ("GLC0", "GCM"):
        sites.extend((residue, atom_name) for atom_name in RESIDUE_ORDER[residue])
    return sites


def write_pdb(mol: Chem.Mol, atom_names: dict[int, tuple[str, str]], output: Path) -> None:
    conf = mol.GetConformer()
    site_to_idx = {residue_atom_key(*site): idx for idx, site in atom_names.items()}
    lines = ["COMPND    bDGlc14bDGlcOMe mapped to GROMOS 56A_CARBO GLC0-GCM names\n"]
    serial = 1
    residue_ids = {"GLC0": 1, "GCM": 2}
    for residue, atom_name in ordered_sites():
        idx = site_to_idx[residue_atom_key(residue, atom_name)]
        atom = mol.GetAtomWithIdx(idx)
        pos = conf.GetAtomPosition(idx)
        element = atom.GetSymbol().upper()
        atom_field = pdb_atom_name(atom_name, element)
        lines.append(
            f"HETATM{serial:5d} {atom_field:4s} {residue:<4s}{residue_ids[residue]:5d}    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {element:>2s}\n"
        )
        serial += 1
    lines.append("TER\nEND\n")
    output.write_text("".join(lines), encoding="utf-8")


def prepare_forcefield(source_ff: Path, dest_ff: Path) -> None:
    dest_ff.mkdir(parents=True, exist_ok=True)
    for source in source_ff.iterdir():
        target = dest_ff / source.name
        if target.exists() or target.is_symlink():
            continue
        if source.name == "aminoacids.rtp":
            text = source.read_text(encoding="utf-8-sig")
            if "[ GCM ]" not in text:
                text += GCM_RTP
            target.write_text(text, encoding="utf-8")
        elif source.name == "aminoacids.hdb":
            text = source.read_text(encoding="utf-8-sig")
            if re.search(r"^GCM\s", text, flags=re.MULTILINE) is None:
                text += GCM_HDB
            target.write_text(text, encoding="utf-8")
        else:
            os.symlink(source, target)


def write_mapping(output_dir: Path) -> None:
    frame_heavy_order = [
        site for site in ordered_sites() if not site[1].startswith("H")
    ]
    frame_index_by_site = {residue_atom_key(*site): idx for idx, site in enumerate(frame_heavy_order)}
    mapping = [
        frame_index_by_site[residue_atom_key(*SDF_TO_SITE[sdf_idx])]
        for sdf_idx in range(len(SDF_TO_SITE))
    ]
    (output_dir / "sdf_to_gromos_frame_heavy_index_map.json").write_text(
        json.dumps(mapping, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "frame_heavy_order": [list(site) for site in frame_heavy_order],
        "sdf_to_frame_heavy_index_map": mapping,
        "sdf_index_to_gromos_site": {str(k): list(v) for k, v in sorted(SDF_TO_SITE.items())},
        "custom_residue": "GCM",
        "custom_residue_note": (
            "GCM is a local methylated reducing-end glucose patch derived from GLCN. "
            "The rest of the topology uses the downloaded GROMOS 56A_CARBO GLC0/1-4 linkage rules."
        ),
    }
    (output_dir / "gromos56a6carbo_input_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GROMOS 56A_CARBO GLC0-GCM test inputs.")
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--source-ff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_ff = Path(args.source_ff).resolve()
    dest_ff = output_dir / "gromos53a6carbo_gcm.ff"
    prepare_forcefield(source_ff, dest_ff)

    mol = embed_with_hydrogens(find_molecule(Path(args.sdf), args.molecule_name), args.seed)
    atom_names = name_atoms(mol)
    pdb_out = output_dir / f"{args.molecule_name}_GROMOS56A6CARBO_rdkit3d_ua.pdb"
    write_pdb(mol, atom_names, pdb_out)
    write_mapping(output_dir)

    print(pdb_out)
    print(output_dir / "sdf_to_gromos_frame_heavy_index_map.json")
    print(dest_ff)


if __name__ == "__main__":
    main()
