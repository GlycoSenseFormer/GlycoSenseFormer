#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def find_molecule(sdf_path: Path, molecule_name: str) -> Chem.Mol:
    for mol in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if mol is None:
            continue
        if mol.HasProp("_Name") and mol.GetProp("_Name") == molecule_name:
            return mol
    raise ValueError(f"Could not find {molecule_name!r} in {sdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an exact CASPER molecule as a 3D, hydrogenated input for ACPYPE/GAFF."
    )
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol = find_molecule(Path(args.sdf), args.molecule_name)
    heavy_symbols = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]

    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    Chem.SanitizeMol(work)
    work = Chem.AddHs(work, addCoords=True)

    params = Chem.rdDistGeom.ETKDGv3()
    params.randomSeed = args.seed
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

    sdf_out = output_dir / f"{args.molecule_name}_rdkit3d_h.sdf"
    mol_out = output_dir / f"{args.molecule_name}_rdkit3d_h.mol"
    pdb_out = output_dir / f"{args.molecule_name}_rdkit3d_h.pdb"
    Chem.SDWriter(str(sdf_out)).write(work)
    Chem.MolToMolFile(work, str(mol_out))
    Chem.MolToPDBFile(work, str(pdb_out))

    metadata = {
        "molecule_name": args.molecule_name,
        "source_sdf": str(Path(args.sdf)),
        "seed": args.seed,
        "heavy_atom_count": len(heavy_symbols),
        "heavy_symbols_sdf_order": heavy_symbols,
        "note": "RDKit AddHs appends hydrogens after the original SDF atoms, so heavy atom order is preserved.",
    }
    (output_dir / "input_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(sdf_out)
    print(mol_out)
    print(pdb_out)


if __name__ == "__main__":
    main()
