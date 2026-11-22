#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


RESNAME = "XCG"


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
        params.useRandomCoords = True
        params.randomSeed = seed + 1009
        status = AllChem.EmbedMolecule(work, params)
    if status != 0:
        fallback = Chem.Mol(mol)
        Chem.SanitizeMol(fallback)
        if fallback.GetNumConformers() == 0:
            raise RuntimeError(f"RDKit EmbedMolecule failed with status {status}, and SDF has no coordinates")
        work = Chem.AddHs(fallback, addCoords=True)
    if AllChem.MMFFHasAllMoleculeParams(work):
        AllChem.MMFFOptimizeMolecule(work, maxIters=200, mmffVariant="MMFF94")
    else:
        AllChem.UFFOptimizeMolecule(work, maxIters=200)
    return work


def atom_names(mol: Chem.Mol) -> dict[int, str]:
    counters: dict[str, int] = defaultdict(int)
    names: dict[int, str] = {}
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol().upper()
        counters[elem] += 1
        names[atom.GetIdx()] = f"{elem[:1]}{counters[elem]:03d}"[-4:]
    return names


def h_count(atom: Chem.Atom) -> int:
    return sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)


def has_h_neighbor(atom: Chem.Atom) -> bool:
    return any(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors())


def bond_order(atom: Chem.Atom, nbr: Chem.Atom) -> Chem.BondType:
    bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
    if bond is None:
        raise ValueError("missing bond")
    return bond.GetBondType()


def carbonyl_neighbor(atom: Chem.Atom) -> bool:
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        if any(
            other.GetAtomicNum() == 8 and bond_order(nbr, other) == Chem.BondType.DOUBLE
            for other in nbr.GetNeighbors()
        ):
            return True
    return False


def carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        nbr.GetAtomicNum() == 8 and bond_order(atom, nbr) == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def attached_to_n(atom: Chem.Atom) -> bool:
    return any(nbr.GetAtomicNum() == 7 for nbr in atom.GetNeighbors())


def attached_to_nonring_ether(atom: Chem.Atom) -> bool:
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 8 or nbr.IsInRing() or has_h_neighbor(nbr):
            continue
        if bond_order(atom, nbr) == Chem.BondType.SINGLE:
            return True
    return False


def atom_type_charge(atom: Chem.Atom) -> tuple[str, float]:
    num = atom.GetAtomicNum()
    if num == 1:
        parent = atom.GetNeighbors()[0]
        if parent.GetAtomicNum() == 8:
            return "HCP1", 0.42
        if parent.GetAtomicNum() == 7:
            return "HCP1", 0.31
        parent_h = h_count(parent)
        if parent_h >= 3:
            return "HCA3", 0.09
        if parent_h == 2:
            return "HCA2", 0.09
        return "HCA1", 0.09

    if num == 8:
        if any(bond_order(atom, nbr) == Chem.BondType.DOUBLE for nbr in atom.GetNeighbors()):
            return "OC2D1", -0.51
        if atom.IsInRing():
            return "OC3C61", -0.40
        if has_h_neighbor(atom):
            return "OC311", -0.65
        return "OC301", -0.36

    if num == 7:
        return "NC2D1", -0.47

    if num != 6:
        raise ValueError(f"Unsupported atom element for CHARMM exact residue: {atom.GetSymbol()}")

    if carbonyl_carbon(atom):
        return "CC2O1", 0.51

    hnum = h_count(atom)
    if hnum == 3:
        if any(nbr.GetAtomicNum() == 8 for nbr in atom.GetNeighbors()):
            return "CC331", -0.09
        return "CC331", -0.27

    if not atom.IsInRing():
        if hnum == 2 and any(nbr.GetAtomicNum() == 8 for nbr in atom.GetNeighbors()):
            return "CC321", 0.05
        if carbonyl_neighbor(atom):
            return "CC331", -0.27
        if hnum == 2:
            return "CC321", 0.05
        return "CC3161", 0.14

    oxygen_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 8]
    ring_oxygen_neighbors = [nbr for nbr in oxygen_neighbors if nbr.IsInRing()]
    external_oxygen_neighbors = [nbr for nbr in oxygen_neighbors if not nbr.IsInRing()]
    if ring_oxygen_neighbors and attached_to_nonring_ether(atom):
        return "CC3162", 0.29
    if ring_oxygen_neighbors and external_oxygen_neighbors:
        return "CC3162", 0.34
    if ring_oxygen_neighbors and hnum == 2:
        return "CC3263", 0.02
    if ring_oxygen_neighbors:
        return "CC3163", 0.11
    if attached_to_n(atom):
        return "CC3161", 0.07
    if attached_to_nonring_ether(atom):
        return "CC3161", 0.09
    return "CC3161", 0.14


def pdb_atom_name(name: str, element: str) -> str:
    if len(name) >= 4:
        return name[:4]
    if len(element) == 1 and not name[0].isdigit():
        return f" {name:<3s}"
    return f"{name:<4s}"


def write_pdb(mol: Chem.Mol, names: dict[int, str], output: Path, molecule_name: str) -> None:
    conf = mol.GetConformer()
    lines = [f"REMARK {molecule_name} mapped to local CHARMM exact residue {RESNAME}\n"]
    for serial, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        elem = atom.GetSymbol().upper()
        name = names[atom.GetIdx()]
        lines.append(
            f"HETATM{serial:5d} {pdb_atom_name(name, elem):4s} {RESNAME:<4s}A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {elem:>2s}\n"
        )
    lines.append("TER\nEND\n")
    output.write_text("".join(lines), encoding="utf-8")


def write_gro(mol: Chem.Mol, names: dict[int, str], output: Path, molecule_name: str) -> None:
    conf = mol.GetConformer()
    lines = [f"{molecule_name} mapped to local CHARMM exact residue {RESNAME}\n", f"{mol.GetNumAtoms():5d}\n"]
    for serial, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(atom.GetIdx())
        name = names[atom.GetIdx()]
        lines.append(
            f"{1:5d}{RESNAME:<5s}{name:>5s}{serial:5d}"
            f"{pos.x / 10.0:8.3f}{pos.y / 10.0:8.3f}{pos.z / 10.0:8.3f}\n"
        )
    lines.append("   8.00000   8.00000   8.00000\n")
    output.write_text("".join(lines), encoding="utf-8")


def rtp_block(mol: Chem.Mol, names: dict[int, str], molecule_name: str) -> tuple[str, list[dict[str, object]]]:
    typed = []
    lines = [
        "",
        f"[ {RESNAME} ]",
        f"; exact single-residue CHARMM-style topology for {molecule_name}",
        "  [ atoms ]",
    ]
    for atom in mol.GetAtoms():
        atype, charge = atom_type_charge(atom)
        name = names[atom.GetIdx()]
        cgnr = atom.GetIdx() + 1
        lines.append(f"{name:>9s} {atype:>8s} {charge:7.4f} {cgnr:5d}")
        typed.append({"idx": atom.GetIdx(), "name": name, "element": atom.GetSymbol(), "type": atype, "charge": charge})

    lines.append("  [ bonds ]")
    for bond in mol.GetBonds():
        a = names[bond.GetBeginAtomIdx()]
        b = names[bond.GetEndAtomIdx()]
        lines.append(f"{a:>9s} {b:>8s}")

    impropers = []
    for atom in mol.GetAtoms():
        if not carbonyl_carbon(atom):
            continue
        oxy = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 8 and bond_order(atom, nbr) == Chem.BondType.DOUBLE]
        nit = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 7]
        car = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6]
        if oxy and nit and car:
            impropers.append((names[atom.GetIdx()], names[car[0].GetIdx()], names[nit[0].GetIdx()], names[oxy[0].GetIdx()]))
    if impropers:
        lines.append("  [ impropers ]")
        for imp in impropers:
            lines.append(f"{imp[0]:>9s} {imp[1]:>8s} {imp[2]:>8s} {imp[3]:>8s}")

    return "\n".join(lines) + "\n", typed


def prepare_forcefield(source_ff: Path, dest_ff: Path, custom_rtp: str) -> None:
    dest_ff.mkdir(parents=True, exist_ok=True)
    for source in source_ff.iterdir():
        target = dest_ff / source.name
        if target.exists() or target.is_symlink():
            continue
        if source.name == "carb.rtp":
            text = source.read_text(encoding="utf-8")
            if f"[ {RESNAME} ]" not in text:
                text += custom_rtp
            target.write_text(text, encoding="utf-8")
        else:
            os.symlink(source, target)


def run_pdb2gmx(gro: Path, top: Path, processed: Path, ff_choice: str, cwd: Path) -> None:
    cmd = [
        "gmx",
        "pdb2gmx",
        "-f",
        gro.name,
        "-o",
        processed.name,
        "-p",
        top.name,
        "-ff",
        ff_choice,
        "-water",
        "tip3p",
    ]
    log = cwd / "pdb2gmx.log"
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local exact-residue CHARMM topology for a CASPER glycan.")
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--source-ff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-pdb2gmx", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mol = embed_with_hydrogens(find_molecule(Path(args.sdf), args.molecule_name), args.seed)
    names = atom_names(mol)
    custom_rtp, typed_atoms = rtp_block(mol, names, args.molecule_name)

    source_ff = Path(args.source_ff).resolve()
    ff_choice = "charmm36_exact"
    ff_dir_name = f"{ff_choice}.ff"
    dest_ff = output_dir / ff_dir_name
    prepare_forcefield(source_ff, dest_ff, custom_rtp)

    stem = args.molecule_name.replace("/", "_")
    pdb_out = output_dir / f"{stem}_CHARMM_EXACT_rdkit3d_allatom.pdb"
    gro_out = output_dir / f"{stem}_CHARMM_EXACT_rdkit3d_allatom.gro"
    top_out = output_dir / f"{stem}_CHARMM_EXACT.top"
    processed_out = output_dir / f"{stem}_CHARMM_EXACT_processed.gro"
    write_pdb(mol, names, pdb_out, args.molecule_name)
    write_gro(mol, names, gro_out, args.molecule_name)

    heavy_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
    mapping = list(range(heavy_count))
    mapping_out = output_dir / "sdf_to_charmm_frame_heavy_index_map.json"
    mapping_out.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "molecule_name": args.molecule_name,
        "source_sdf": args.sdf,
        "source_forcefield": str(source_ff),
        "local_forcefield": str(dest_ff),
        "custom_residue": RESNAME,
        "seed": args.seed,
        "atom_names": {str(k): v for k, v in sorted(names.items())},
        "typed_atoms": typed_atoms,
        "total_charge": round(sum(float(row["charge"]) for row in typed_atoms), 6),
        "sdf_to_frame_heavy_index_map": mapping,
        "note": "Exact single-residue CHARMM-style topology generated from graph rules and CHARMM36 carbohydrate atom types. Use as a batch topology smoke path before production-quality residue/patch mapping.",
    }
    (output_dir / "charmm_exact_input_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.run_pdb2gmx:
        run_pdb2gmx(gro_out, top_out, processed_out, ff_choice, output_dir)

    print(pdb_out)
    print(gro_out)
    print(mapping_out)
    print(dest_ff)
    if args.run_pdb2gmx:
        print(top_out)
        print(processed_out)


if __name__ == "__main__":
    main()
