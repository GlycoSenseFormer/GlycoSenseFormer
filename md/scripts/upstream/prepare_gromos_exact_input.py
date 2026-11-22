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
from rdkit.Chem import AllChem, rdMolTransforms


RESNAME = "XGO"


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


def include_atom(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 1:
        return True
    return atom.GetNeighbors()[0].GetAtomicNum() in {7, 8}


def atom_names(mol: Chem.Mol) -> dict[int, str]:
    counters: dict[str, int] = defaultdict(int)
    names: dict[int, str] = {}
    for atom in mol.GetAtoms():
        if not include_atom(atom):
            continue
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


def attached_to_nonring_ether(atom: Chem.Atom) -> bool:
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 8 or nbr.IsInRing() or has_h_neighbor(nbr):
            continue
        if bond_order(atom, nbr) == Chem.BondType.SINGLE:
            return True
    return False


def carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        nbr.GetAtomicNum() == 8 and bond_order(atom, nbr) == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def carbonyl_oxygen(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 8 and any(
        nbr.GetAtomicNum() == 6 and bond_order(atom, nbr) == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def amide_nitrogen(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 7:
        return False
    return any(carbonyl_carbon(nbr) for nbr in atom.GetNeighbors())


def methyl_attached_to_carbonyl(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and h_count(atom) >= 3 and any(
        carbonyl_carbon(nbr) for nbr in atom.GetNeighbors()
    )


def attached_to_nitrogen(atom: Chem.Atom) -> bool:
    return any(nbr.GetAtomicNum() == 7 for nbr in atom.GetNeighbors())


def unsupported_atom(atom: Chem.Atom) -> bool:
    num = atom.GetAtomicNum()
    if num in {1, 6, 8}:
        return False
    if num == 7 and amide_nitrogen(atom):
        return False
    return True


def atom_type_charge(atom: Chem.Atom) -> tuple[str, float]:
    num = atom.GetAtomicNum()
    if num == 1:
        parent = atom.GetNeighbors()[0]
        if parent.GetAtomicNum() == 7:
            return "H", 0.310
        if parent.GetAtomicNum() in {7, 8}:
            return "H", 0.410
        raise ValueError("GROMOS exact topology omits nonpolar hydrogens")
    if num == 7:
        if amide_nitrogen(atom):
            return "N", -0.310
        raise ValueError("Only amide nitrogen is supported in GROMOS exact residue")
    if num == 8:
        if carbonyl_oxygen(atom):
            return "O", -0.450
        if atom.IsInRing():
            return "OR", -0.464
        if has_h_neighbor(atom):
            return "OA", -0.642
        return "OE", -0.464
    if num != 6:
        raise ValueError(f"Unsupported atom element for GROMOS exact residue: {atom.GetSymbol()}")

    if carbonyl_carbon(atom):
        return "C", 0.450
    if methyl_attached_to_carbonyl(atom):
        return "CH3", 0.000
    if h_count(atom) >= 3:
        # A deoxy-sugar methyl group is a neutral united atom.  The old
        # graph rule assigned +0.232 e to every methyl, leaving Fuc/Rha/Qui
        # residues with a non-integer +0.232 e net charge.  Retain +0.232 e
        # only for a methyl bound to a heteroatom (for example O-methyl),
        # where it is part of the corresponding polar charge group.
        if any(nbr.GetAtomicNum() in {7, 8} for nbr in atom.GetNeighbors()):
            return "CH3", 0.232
        return "CH3", 0.000
    if h_count(atom) == 2:
        return "CH2", 0.232
    # Replacing a hydroxyl by an acetamido group removes the +0.232/-0.232
    # CH-OH charge-group pair.  The amide fragment below is already neutral,
    # so its ring carbon must not keep the generic hydroxyl-carbon charge.
    # ``attached_to_nitrogen`` existed before but was never used, which made
    # every N-acetylated residue +0.232 e.
    if attached_to_nitrogen(atom):
        return "CH1R", 0.000
    oxygen_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 8]
    has_ring_oxygen = any(nbr.IsInRing() for nbr in oxygen_neighbors)
    has_external_oxygen = any(not nbr.IsInRing() for nbr in oxygen_neighbors)
    if atom.IsInRing() and has_ring_oxygen and has_external_oxygen:
        return "CH1R", 0.464
    if atom.IsInRing() and has_ring_oxygen:
        return "CH1R", 0.232
    if attached_to_nonring_ether(atom):
        return "CH1R", 0.232
    return "CH1R", 0.232


def anomeric_hydroxyl_sites(
    mol: Chem.Mol,
    names: dict[int, str],
) -> dict[frozenset[int], tuple[int, int, int, int]]:
    """Map anomeric C-O bonds to the 56A_CARBO exo-anomeric quartet.

    The free reducing-end templates use both ``gdc_t3`` and ``gdc_t4`` on
    O5-C1-O1-HO1.  Treating HO1 as an ordinary hydroxyl generated two
    unrelated ``gdc_t3`` terms and was the one common topology defect behind
    all eight historical LINCS failures.
    """

    sites: dict[frozenset[int], tuple[int, int, int, int]] = {}
    for carbon in mol.GetAtoms():
        if carbon.GetIdx() not in names or carbon.GetAtomicNum() != 6 or not carbon.IsInRing():
            continue
        ring_oxygen = next(
            (
                nbr
                for nbr in carbon.GetNeighbors()
                if nbr.GetIdx() in names and nbr.GetAtomicNum() == 8 and nbr.IsInRing()
            ),
            None,
        )
        hydroxyl_oxygen = next(
            (
                nbr
                for nbr in carbon.GetNeighbors()
                if nbr.GetIdx() in names
                and nbr.GetAtomicNum() == 8
                and not nbr.IsInRing()
                and has_h_neighbor(nbr)
            ),
            None,
        )
        if ring_oxygen is None or hydroxyl_oxygen is None:
            continue
        hydrogen = next(
            (
                nbr
                for nbr in hydroxyl_oxygen.GetNeighbors()
                if nbr.GetIdx() in names and nbr.GetAtomicNum() == 1
            ),
            None,
        )
        if hydrogen is None:
            continue
        quartet = (
            ring_oxygen.GetIdx(),
            carbon.GetIdx(),
            hydroxyl_oxygen.GetIdx(),
            hydrogen.GetIdx(),
        )
        sites[frozenset((carbon.GetIdx(), hydroxyl_oxygen.GetIdx()))] = quartet
    return sites


def pyranose_position_maps(
    mol: Chem.Mol,
    names: dict[int, str],
) -> list[dict[str, int]]:
    """Identify C1--C6/O1--O5/HO1--HO4 within pyranose rings.

    GROMOS 56A_CARBO residue templates contain carbohydrate-specific
    exclusions that cannot be recovered from the default three-bond graph
    exclusions used by ``pdb2gmx``.  The input atom names here are generated
    serially, so derive the biochemical positions from the molecular graph
    rather than relying on names.

    Only unambiguous six-membered rings containing five carbons and one
    oxygen are returned.  This deliberately avoids inventing rules for
    furanoses or non-carbohydrate rings.
    """

    position_maps: list[dict[str, int]] = []
    seen_rings: set[frozenset[int]] = set()
    for ring_tuple in mol.GetRingInfo().AtomRings():
        ring = frozenset(ring_tuple)
        if ring in seen_rings:
            continue
        seen_rings.add(ring)
        if len(ring) != 6 or not ring.issubset(names):
            continue
        ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        ring_oxygens = [atom for atom in ring_atoms if atom.GetAtomicNum() == 8]
        ring_carbons = [atom for atom in ring_atoms if atom.GetAtomicNum() == 6]
        if len(ring_oxygens) != 1 or len(ring_carbons) != 5:
            continue

        o5 = ring_oxygens[0]
        o5_carbon_neighbors = [
            nbr
            for nbr in o5.GetNeighbors()
            if nbr.GetIdx() in ring and nbr.GetAtomicNum() == 6
        ]
        if len(o5_carbon_neighbors) != 2:
            continue

        # C1 is the O5-adjacent carbon carrying the exocyclic anomeric
        # oxygen.  The other O5-adjacent carbon is C5.
        c1_candidates = []
        for carbon in o5_carbon_neighbors:
            external_oxygens = [
                nbr
                for nbr in carbon.GetNeighbors()
                if nbr.GetIdx() in names
                and nbr.GetIdx() not in ring
                and nbr.GetAtomicNum() == 8
            ]
            if external_oxygens:
                c1_candidates.append(carbon)
        if len(c1_candidates) != 1:
            continue
        c1 = c1_candidates[0]
        c5 = next(
            carbon
            for carbon in o5_carbon_neighbors
            if carbon.GetIdx() != c1.GetIdx()
        )

        # Walk C1-C2-C3-C4-C5 around the carbon side of the ring.
        carbon_path = [c1]
        previous = o5
        current = c1
        while current.GetIdx() != c5.GetIdx() and len(carbon_path) <= 5:
            next_atoms = [
                nbr
                for nbr in current.GetNeighbors()
                if nbr.GetIdx() in ring
                and nbr.GetIdx() != previous.GetIdx()
                and nbr.GetAtomicNum() == 6
            ]
            if len(next_atoms) != 1:
                carbon_path = []
                break
            previous, current = current, next_atoms[0]
            carbon_path.append(current)
        if len(carbon_path) != 5 or carbon_path[-1].GetIdx() != c5.GetIdx():
            continue

        positions = {
            f"C{number}": atom.GetIdx()
            for number, atom in enumerate(carbon_path, start=1)
        }
        positions["O5"] = o5.GetIdx()

        for number, carbon in enumerate(carbon_path[:4], start=1):
            external_heteroatoms = [
                nbr
                for nbr in carbon.GetNeighbors()
                if nbr.GetIdx() in names
                and nbr.GetIdx() not in ring
                and nbr.GetAtomicNum() in {7, 8}
            ]
            if len(external_heteroatoms) != 1:
                continue
            heteroatom = external_heteroatoms[0]
            if heteroatom.GetAtomicNum() == 7:
                positions[f"N{number}"] = heteroatom.GetIdx()
                continue
            oxygen = heteroatom
            positions[f"O{number}"] = oxygen.GetIdx()
            hydroxyl_hydrogens = [
                nbr
                for nbr in oxygen.GetNeighbors()
                if nbr.GetIdx() in names and nbr.GetAtomicNum() == 1
            ]
            if len(hydroxyl_hydrogens) == 1:
                positions[f"HO{number}"] = hydroxyl_hydrogens[0].GetIdx()

        c6_candidates = [
            nbr
            for nbr in c5.GetNeighbors()
            if nbr.GetIdx() in names
            and nbr.GetIdx() not in ring
            and nbr.GetAtomicNum() == 6
        ]
        if len(c6_candidates) == 1:
            c6 = c6_candidates[0]
            positions["C6"] = c6.GetIdx()
            o6_candidates = [
                nbr
                for nbr in c6.GetNeighbors()
                if nbr.GetIdx() in names
                and nbr.GetIdx() != c5.GetIdx()
                and nbr.GetAtomicNum() == 8
            ]
            if len(o6_candidates) == 1:
                o6 = o6_candidates[0]
                positions["O6"] = o6.GetIdx()
                ho6_candidates = [
                    nbr
                    for nbr in o6.GetNeighbors()
                    if nbr.GetIdx() in names and nbr.GetAtomicNum() == 1
                ]
                if len(ho6_candidates) == 1:
                    positions["HO6"] = ho6_candidates[0].GetIdx()
        position_maps.append(positions)
    return position_maps


def carbohydrate_exclusion_pairs(
    mol: Chem.Mol,
    names: dict[int, str],
) -> list[tuple[int, int]]:
    """Return the extra non-bonded exclusions used by 56A_CARBO pyranoses.

    Ten positional rules reproduce the ``[ exclusions ]`` block of the
    curated free GLC templates.  One additional O5--HO1 exclusion prevents
    the explicit full-charge 1-4 pair from collapsing under modern
    PME/Verlet.  Missing atoms (C6 in pentoses, a glycosidically substituted
    hydroxyl, or an N-acetyl substitution) simply omit the corresponding
    pair.
    """

    positional_rules = (
        ("C6", "O1"),
        ("C6", "O3"),
        # With modern PME/Verlet, pdb2gmx otherwise emits an explicit
        # full-charge O5--HO1 1-4 pair.  In all failed graph-derived runs
        # that pair collapsed to 0.000 nm during EM, before NVT started.
        ("HO1", "O5"),
        ("HO1", "C5"),
        ("HO1", "C3"),
        ("HO2", "C4"),
        ("HO2", "O5"),
        ("HO3", "C1"),
        ("HO3", "C5"),
        ("HO4", "C2"),
        ("HO4", "O5"),
    )
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for positions in pyranose_position_maps(mol, names):
        for left_label, right_label in positional_rules:
            if left_label not in positions or right_label not in positions:
                continue
            pair = (positions[left_label], positions[right_label])
            key = tuple(sorted(pair))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
    return pairs


def pyranose_chiral_impropers(
    mol: Chem.Mol,
    names: dict[int, str],
) -> list[tuple[int, int, int, int, str]]:
    """Build 56A_CARBO ``gi_2`` terms while preserving input chirality.

    The GLCB template supplies five chiral impropers.  Its atom ordering
    gives an equilibrium improper near +35.264 degrees.  Apply the same
    positional definitions to each unambiguous pyranose and swap two
    peripheral atoms whenever needed to represent the molecule's actual
    alpha/beta, D/L, or epimeric stereochemistry with the positive ``gi_2``
    convention.
    """

    if mol.GetNumConformers() == 0:
        raise ValueError("pyranose gi_2 generation requires 3D coordinates")
    conf = mol.GetConformer()
    impropers: list[tuple[int, int, int, int, str]] = []
    for positions in pyranose_position_maps(mol, names):
        substituent2 = positions.get("O2", positions.get("N2"))
        candidates: list[tuple[int, int, int, int]] = []
        required_c1 = ("O1", "C2", "O5", "C1")
        if all(label in positions for label in required_c1):
            candidates.append(tuple(positions[label] for label in required_c1))
        if substituent2 is not None:
            candidates.append(
                (
                    positions["C2"],
                    positions["C3"],
                    positions["C1"],
                    substituent2,
                )
            )
        for labels in (
            ("C3", "C2", "C4", "O3"),
            ("O4", "C3", "C5", "C4"),
            ("C5", "C4", "O5", "C6"),
        ):
            if all(label in positions for label in labels):
                candidates.append(tuple(positions[label] for label in labels))

        for quartet in candidates:
            oriented = list(quartet)
            angle = rdMolTransforms.GetDihedralDeg(conf, *oriented)
            if angle < 0.0:
                oriented[1], oriented[2] = oriented[2], oriented[1]
                angle = rdMolTransforms.GetDihedralDeg(conf, *oriented)
            if not (15.0 <= angle <= 60.0):
                raise ValueError(
                    "pyranose gi_2 orientation is inconsistent with a tetrahedral "
                    f"center: quartet={quartet}, angle={angle:.3f} degrees"
                )
            impropers.append((*oriented, "gi_2"))
    return impropers


def pyranose_standard_dihedrals(
    mol: Chem.Mol,
    names: dict[int, str],
) -> tuple[list[tuple[int, int, int, int, str]], set[frozenset[int]]]:
    """Return the curated 56A_CARBO ring torsion network.

    This mirrors the GLCB template's selected proper terms, including the
    oxygen-gauche ``gdc_t10``--``gdc_t13`` corrections.  It is applied only
    to pyranose rings with the standard O1--O4 substitution pattern; rings
    such as GlcNAc remain on the explicit experimental fallback until their
    own residue template is implemented.
    """

    dihedrals: list[tuple[int, int, int, int, str]] = []
    handled_bonds: set[frozenset[int]] = set()
    seen: set[tuple[tuple[int, int, int, int], str]] = set()

    def add(
        positions: dict[str, int],
        labels: tuple[str, str, str, str],
        dtype: str,
    ) -> None:
        if not all(label in positions for label in labels):
            return
        quartet = tuple(positions[label] for label in labels)
        canonical = min(quartet, tuple(reversed(quartet)))
        key = (canonical, dtype)
        if key in seen:
            return
        seen.add(key)
        dihedrals.append((*quartet, dtype))

    for positions in pyranose_position_maps(mol, names):
        if not all(f"O{number}" in positions for number in range(1, 5)):
            continue

        # Selected hydroxyl rotations.  The anomeric OH has the additional
        # exo-anomeric t4 term.
        if "HO1" in positions:
            add(positions, ("O5", "C1", "O1", "HO1"), "gdc_t3")
            add(positions, ("O5", "C1", "O1", "HO1"), "gdc_t4")
            handled_bonds.add(frozenset((positions["C1"], positions["O1"])))
        for labels in (
            ("HO2", "O2", "C2", "C1"),
            ("C2", "C3", "O3", "HO3"),
            ("C3", "C4", "O4", "HO4"),
            ("C5", "C6", "O6", "HO6"),
        ):
            if labels[0] not in positions or labels[3] not in positions:
                continue
            add(positions, labels, "gdc_t3")
            handled_bonds.add(
                frozenset((positions[labels[1]], positions[labels[2]]))
            )

        for labels, dtype in (
            (("O2", "C2", "C3", "O3"), "gdc_t12"),
            (("C1", "C2", "C3", "C4"), "gdc_t1"),
            (("O2", "C2", "C1", "O5"), "gdc_t13"),
            (("O2", "C2", "C1", "O1"), "gdc_t12"),
            (("C3", "C2", "C1", "O5"), "gdc_t1"),
            (("C2", "C3", "C4", "C5"), "gdc_t1"),
            (("O3", "C3", "C4", "O4"), "gdc_t12"),
            (("C3", "C4", "C5", "O5"), "gdc_t1"),
            (("O4", "C4", "C5", "O5"), "gdc_t13"),
            (("O6", "C6", "C5", "C4"), "gdc_t1"),
            (("O6", "C6", "C5", "C4"), "gdc_t10"),
            (("O6", "C6", "C5", "O5"), "gdc_t11"),
            (("O6", "C6", "C5", "O5"), "gdc_t12"),
            (("C4", "C5", "O5", "C1"), "gdc_t2"),
            (("C5", "O5", "C1", "C2"), "gdc_t2"),
            (("C5", "O5", "C1", "O1"), "gdc_t5"),
        ):
            add(positions, labels, dtype)

        for left, right in (
            ("C1", "C2"),
            ("C2", "C3"),
            ("C3", "C4"),
            ("C4", "C5"),
            ("C5", "O5"),
            ("O5", "C1"),
        ):
            handled_bonds.add(frozenset((positions[left], positions[right])))
        if "C6" in positions and "O6" in positions:
            handled_bonds.add(frozenset((positions["C5"], positions["C6"])))
    return dihedrals, handled_bonds


def proper_dihedrals(mol: Chem.Mol, names: dict[int, str]) -> list[tuple[int, int, int, int, str]]:
    seen: set[tuple[int, int, int, int]] = set()
    dihedrals, handled_bonds = pyranose_standard_dihedrals(mol, names)
    for a, b, c, d, _dtype in dihedrals:
        quartet = (a, b, c, d)
        seen.add(min(quartet, tuple(reversed(quartet))))
    anomeric_sites = anomeric_hydroxyl_sites(mol, names)
    for bond in mol.GetBonds():
        b = bond.GetBeginAtom()
        c = bond.GetEndAtom()
        if b.GetIdx() not in names or c.GetIdx() not in names:
            continue
        if frozenset((b.GetIdx(), c.GetIdx())) in handled_bonds:
            continue
        anomeric_quartet = anomeric_sites.get(frozenset((b.GetIdx(), c.GetIdx())))
        if anomeric_quartet is not None:
            dihedrals.append((*anomeric_quartet, "gdc_t3"))
            dihedrals.append((*anomeric_quartet, "gdc_t4"))
            seen.add(min(anomeric_quartet, tuple(reversed(anomeric_quartet))))
            continue
        for a in b.GetNeighbors():
            if a.GetIdx() == c.GetIdx() or a.GetIdx() not in names:
                continue
            for d in c.GetNeighbors():
                if d.GetIdx() == b.GetIdx() or d.GetIdx() == a.GetIdx() or d.GetIdx() not in names:
                    continue
                forward = (a.GetIdx(), b.GetIdx(), c.GetIdx(), d.GetIdx())
                reverse = tuple(reversed(forward))
                key = min(forward, reverse)
                if key in seen:
                    continue
                seen.add(key)
                dtype = "gdc_t3" if 1 in {a.GetAtomicNum(), b.GetAtomicNum(), c.GetAtomicNum(), d.GetAtomicNum()} else "gdc_t1"
                dihedrals.append((*forward, dtype))
    return dihedrals


def gromos_bond_type(a: Chem.Atom, b: Chem.Atom) -> str:
    bond = bond_order(a, b)
    if bond == Chem.BondType.DOUBLE and {a.GetAtomicNum(), b.GetAtomicNum()} == {6, 8}:
        return "gb_5"
    nums = {a.GetAtomicNum(), b.GetAtomicNum()}
    if 1 in nums:
        heavy = a if a.GetAtomicNum() != 1 else b
        return "gb_2" if heavy.GetAtomicNum() == 7 else "gb_1"
    if nums == {6, 7}:
        carbon = a if a.GetAtomicNum() == 6 else b
        return "gb_9" if carbonyl_carbon(carbon) else "gb_21"
    if nums == {6}:
        return "gb_27" if carbonyl_carbon(a) or carbonyl_carbon(b) else "gb_26"
    if nums == {6, 8}:
        carbon = a if a.GetAtomicNum() == 6 else b
        oxygen = b if a.GetAtomicNum() == 6 else a
        if h_count(carbon) >= 3 and not has_h_neighbor(oxygen):
            return "gb_18"
        return "gb_20"
    return "gb_20"


def angle_triples(mol: Chem.Mol, names: dict[int, str]) -> list[tuple[int, int, int, str]]:
    seen: set[tuple[int, int, int]] = set()
    angles: list[tuple[int, int, int, str]] = []
    for center in mol.GetAtoms():
        cidx = center.GetIdx()
        if cidx not in names:
            continue
        neigh = [nbr for nbr in center.GetNeighbors() if nbr.GetIdx() in names]
        for i, left in enumerate(neigh):
            for right in neigh[i + 1 :]:
                forward = (left.GetIdx(), cidx, right.GetIdx())
                reverse = tuple(reversed(forward))
                key = min(forward, reverse)
                if key in seen:
                    continue
                seen.add(key)
                nums = (left.GetAtomicNum(), center.GetAtomicNum(), right.GetAtomicNum())
                if carbonyl_carbon(center):
                    if {left.GetAtomicNum(), right.GetAtomicNum()} == {7, 8}:
                        atype = "ga_33"
                    elif 8 in {left.GetAtomicNum(), right.GetAtomicNum()}:
                        atype = "ga_30"
                    elif 7 in {left.GetAtomicNum(), right.GetAtomicNum()}:
                        atype = "ga_19"
                    else:
                        atype = "ga_15"
                elif center.GetAtomicNum() == 7 and any(carbonyl_carbon(nbr) for nbr in (left, right)):
                    if 1 in nums:
                        atype = "ga_23"
                    else:
                        atype = "ga_31"
                elif 1 in nums:
                    atype = "ga_12"
                elif nums == (6, 6, 6):
                    atype = "ga_8"
                elif center.GetAtomicNum() == 8 and left.GetAtomicNum() == 6 and right.GetAtomicNum() == 6:
                    atype = "ga_20" if (h_count(left) >= 3 or h_count(right) >= 3) else "ga_10"
                else:
                    atype = "ga_9"
                angles.append((*forward, atype))
    return angles


def improper_quartets(mol: Chem.Mol, names: dict[int, str]) -> list[tuple[int, int, int, int, str]]:
    impropers = pyranose_chiral_impropers(mol, names)
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in names:
            continue
        if carbonyl_carbon(atom):
            oxy = [
                nbr
                for nbr in atom.GetNeighbors()
                if nbr.GetIdx() in names and nbr.GetAtomicNum() == 8 and bond_order(atom, nbr) == Chem.BondType.DOUBLE
            ]
            nit = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() in names and nbr.GetAtomicNum() == 7]
            car = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() in names and nbr.GetAtomicNum() == 6]
            if oxy and nit and car:
                impropers.append((atom.GetIdx(), car[0].GetIdx(), nit[0].GetIdx(), oxy[0].GetIdx(), "gi_1"))
        if amide_nitrogen(atom):
            hyd = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() in names and nbr.GetAtomicNum() == 1]
            car = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() in names and nbr.GetAtomicNum() == 6]
            carbonyl = [nbr for nbr in car if carbonyl_carbon(nbr)]
            noncarbonyl = [nbr for nbr in car if not carbonyl_carbon(nbr)]
            if hyd and carbonyl and noncarbonyl:
                impropers.append((atom.GetIdx(), hyd[0].GetIdx(), noncarbonyl[0].GetIdx(), carbonyl[0].GetIdx(), "gi_1"))
    return impropers


def pdb_atom_name(name: str, element: str) -> str:
    if len(name) >= 4:
        return name[:4]
    if len(element) == 1 and not name[0].isdigit():
        return f" {name:<3s}"
    return f"{name:<4s}"


def write_pdb(mol: Chem.Mol, names: dict[int, str], output: Path, molecule_name: str) -> None:
    conf = mol.GetConformer()
    lines = [f"REMARK {molecule_name} mapped to local GROMOS exact residue {RESNAME}\n"]
    serial = 1
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in names:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        elem = atom.GetSymbol().upper()
        name = names[atom.GetIdx()]
        lines.append(
            f"HETATM{serial:5d} {pdb_atom_name(name, elem):4s} {RESNAME:<4s}A   1    "
            f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {elem:>2s}\n"
        )
        serial += 1
    lines.append("TER\nEND\n")
    output.write_text("".join(lines), encoding="utf-8")


def write_gro(mol: Chem.Mol, names: dict[int, str], output: Path, molecule_name: str) -> None:
    conf = mol.GetConformer()
    ordered = [atom for atom in mol.GetAtoms() if atom.GetIdx() in names]
    lines = [f"{molecule_name} mapped to local GROMOS exact residue {RESNAME}\n", f"{len(ordered):5d}\n"]
    for serial, atom in enumerate(ordered, start=1):
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
        f"; exact single-residue GROMOS-style united-atom topology for {molecule_name}",
        " [ atoms ]",
    ]
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in names:
            continue
        atype, charge = atom_type_charge(atom)
        name = names[atom.GetIdx()]
        cgnr = atom.GetIdx() + 1
        lines.append(f"{name:>6s} {atype:>7s} {charge:9.5f} {cgnr:5d}")
        typed.append({"idx": atom.GetIdx(), "name": name, "element": atom.GetSymbol(), "type": atype, "charge": charge})

    lines.append(" [ bonds ]")
    for bond in mol.GetBonds():
        aidx = bond.GetBeginAtomIdx()
        bidx = bond.GetEndAtomIdx()
        if aidx not in names or bidx not in names:
            continue
        btype = gromos_bond_type(mol.GetAtomWithIdx(aidx), mol.GetAtomWithIdx(bidx))
        lines.append(f"{names[aidx]:>6s} {names[bidx]:>7s} {btype:>7s}")
    angles = angle_triples(mol, names)
    if angles:
        lines.append(" [ angles ]")
        for a, b, c, atype in angles:
            lines.append(f"{names[a]:>6s} {names[b]:>7s} {names[c]:>7s} {atype:>7s}")
    impropers = improper_quartets(mol, names)
    if impropers:
        lines.append(" [ impropers ]")
        for a, b, c, d, itype in impropers:
            lines.append(f"{names[a]:>6s} {names[b]:>7s} {names[c]:>7s} {names[d]:>7s} {itype:>7s}")
    dihedrals = proper_dihedrals(mol, names)
    if dihedrals:
        lines.append(" [ dihedrals ]")
        for a, b, c, d, dtype in dihedrals:
            lines.append(f"{names[a]:>6s} {names[b]:>7s} {names[c]:>7s} {names[d]:>7s} {dtype:>8s}")
    exclusions = carbohydrate_exclusion_pairs(mol, names)
    if exclusions:
        lines.append(" [ exclusions ]")
        for a, b in exclusions:
            lines.append(f"{names[a]:>6s} {names[b]:>7s}")
    return "\n".join(lines) + "\n", typed


def prepare_forcefield(source_ff: Path, dest_ff: Path, custom_rtp: str) -> None:
    dest_ff.mkdir(parents=True, exist_ok=True)
    for source in source_ff.iterdir():
        target = dest_ff / source.name
        if target.exists() or target.is_symlink():
            continue
        if source.name == "aminoacids.rtp":
            text = source.read_text(encoding="utf-8-sig")
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
        "spc",
    ]
    log = cwd / "pdb2gmx.log"
    with log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local exact-residue GROMOS topology for a CASPER glycan.")
    parser.add_argument("--sdf", required=True)
    parser.add_argument("--molecule-name", required=True)
    parser.add_argument("--source-ff", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-pdb2gmx", action="store_true")
    args = parser.parse_args()

    source = find_molecule(Path(args.sdf), args.molecule_name)
    unsupported_source = [atom.GetSymbol() for atom in source.GetAtoms() if unsupported_atom(atom)]
    if unsupported_source:
        raise ValueError(f"GROMOS exact smoke unsupported atom types: {unsupported_source}")
    mol = embed_with_hydrogens(source, args.seed)
    unsupported_embedded = [atom.GetSymbol() for atom in mol.GetAtoms() if unsupported_atom(atom)]
    if unsupported_embedded:
        raise ValueError(f"GROMOS exact smoke unsupported atom types: {unsupported_embedded}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = atom_names(mol)
    custom_rtp, typed_atoms = rtp_block(mol, names, args.molecule_name)
    total_charge = round(sum(float(row["charge"]) for row in typed_atoms), 6)
    if abs(total_charge) > 1.0e-6:
        raise ValueError(
            f"GROMOS exact neutral-sugar graph has non-zero total charge {total_charge:+.6f} e"
        )

    ff_choice = "gromos53a6carbo_exact"
    dest_ff = output_dir / f"{ff_choice}.ff"
    prepare_forcefield(Path(args.source_ff).resolve(), dest_ff, custom_rtp)

    stem = args.molecule_name.replace("/", "_")
    pdb_out = output_dir / f"{stem}_GROMOS_EXACT_rdkit3d_unitedatom.pdb"
    gro_out = output_dir / f"{stem}_GROMOS_EXACT_rdkit3d_unitedatom.gro"
    top_out = output_dir / f"{stem}_GROMOS_EXACT.top"
    processed_out = output_dir / f"{stem}_GROMOS_EXACT_processed.gro"
    write_pdb(mol, names, pdb_out, args.molecule_name)
    write_gro(mol, names, gro_out, args.molecule_name)

    frame_heavy_order = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    mapping = list(range(len(frame_heavy_order)))
    mapping_out = output_dir / "sdf_to_gromos_frame_heavy_index_map.json"
    mapping_out.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    metadata = {
        "molecule_name": args.molecule_name,
        "source_sdf": args.sdf,
        "local_forcefield": str(dest_ff),
        "custom_residue": RESNAME,
        "seed": args.seed,
        "atom_names": {str(k): v for k, v in sorted(names.items())},
        "typed_atoms": typed_atoms,
        "total_charge": total_charge,
        "expected_total_charge": 0.0,
        "charge_assignment_version": "neutral-deoxy-nac-v2",
        "carbohydrate_exclusions_version": "56a-carbo-pyranose-modern-pme-v2",
        "carbohydrate_exclusion_count": len(
            carbohydrate_exclusion_pairs(mol, names)
        ),
        "experimental_modern_pme_anomeric_pair_suppression": True,
        "pyranose_improper_version": "56a-carbo-gi2-stereochemical-v1",
        "pyranose_improper_count": len(pyranose_chiral_impropers(mol, names)),
        "pyranose_dihedral_version": "56a-carbo-selected-t1-t13-v1",
        "pyranose_standard_dihedral_count": len(
            pyranose_standard_dihedrals(mol, names)[0]
        ),
        "parameterization_tier": "inferred_custom",
        "topology_provenance": "graph_derived_diagnostic",
        "curated_template_ids": [],
        "curated_source_sha256": None,
        "graph_derived_parameters": True,
        "experimental_gromos": True,
        "strict_validation": {
            "passed": False,
            "checks": {
                "source_identity": True,
                "atom_identity": True,
                "stereochemistry": False,
                "formal_charge": total_charge == 0.0,
                "topology_preprocess": False,
                "energy_minimization": False,
                "short_nvt": False,
            },
            "note": "The builder cannot self-promote graph-derived parameters; downstream smoke results remain diagnostic.",
        },
        "production_ready": False,
        "production_ready_scope": "diagnostic graph-derived topology only",
        "sdf_to_frame_heavy_index_map": mapping,
        "note": "Experimental GROMOS united-atom exact single-residue smoke path. Amide/NAc carbonyl support uses generic GROMOS peptide/ACE-like graph rules and should be treated as batch-enabling, not a curated residue template.",
    }
    (output_dir / "gromos_exact_input_metadata.json").write_text(
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
