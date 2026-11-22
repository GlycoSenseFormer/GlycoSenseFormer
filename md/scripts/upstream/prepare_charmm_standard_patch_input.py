#!/usr/bin/env python3
"""Build the 30 fail-closed CHARMM36 standard-patch repair topologies.

The February-2026 GROMACS port contains the relevant carbohydrate atom and
bonded parameters, but its CHARMM PRES records are not represented as GROMACS
terminal-database patches.  This builder therefore performs a declarative,
auditable expansion of the pinned upstream PRES transformations into one
temporary RTP/HDB residue named ``XCP``.  It never assigns atom types or
charges from a generic graph rule.

Coordinates come from the exact CASPER record, after ETKDGv3/MMFF94
embedding.  The source heavy graph must be isomorphic to the expanded RTP
heavy graph, and canonical stereochemical identity is pinned per target,
before pdb2gmx is allowed to run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import networkx as nx
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from prepare_charmm_standard_uronic_input import (
    copy_pinned_forcefield,
    count_diagnostics,
    embed_mmff,
    find_source_molecule,
    gromacs_version,
    parse_topology_charge,
    preflight_mdp,
    resolve_executable,
    run_logged,
    sha256_directory,
    sha256_file,
    tail,
)


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_SOURCE_FF = (
    MD_ROOT
    / "forcefield_assets"
    / "charmm36"
    / "charmm36-feb2026_cgenff-5.0.ff"
)
DEFAULT_MANIFEST = (
    MD_ROOT
    / "forcefield_assets"
    / "charmm36"
    / "charmm36-feb2026_standard_patch_manifest.json"
)
DEFAULT_RUN_ROOT = Path(
    os.environ.get(
        "MD_RUN_ROOT",
        MD_ROOT / "md_runs_third_forcefield_repair_charmm30_hardened_v4",
    )
)
DEFAULT_GMX = os.environ.get(
    "GMX",
    "/soft/gromacs-2026.1/bin/gmx"
    if Path("/soft/gromacs-2026.1/bin/gmx").exists()
    else "gmx",
)
OUTPUT_SUBDIR = "charmm36_standard_topology_dryrun"
LOCAL_FORCEFIELD_NAME = "charmm36_standard.ff"
LOCAL_RESIDUE = "XCP"
CHARGE_TOLERANCE = 1.0e-6
APPROVED_MANIFEST_SHA256 = "1a37d6e2bada274948f43ded9ace4d431e5c060148622dffda3e9e5fee38fbec"
APPROVED_SOURCE_FF_TREE_SHA256 = "9d942a07f4cbb7c70f8314b6a8f5108bbe9a2f49b4a463477c18acc2656ee92e"


@dataclass(frozen=True)
class RtpAtom:
    name: str
    atom_type: str
    charge: float
    charge_group: int


@dataclass(frozen=True)
class PatchSpec:
    kind: str
    name: str
    unit1: str = ""
    unit2: str = ""
    position: int = 0


@dataclass(frozen=True)
class Target:
    raw_name: str
    clean_name: str
    source_class: str
    bases: tuple[tuple[str, str], ...]
    patches: tuple[PatchSpec, ...]
    expected_isomeric_smiles: str
    expected_formula: str
    expected_charge: int


@dataclass
class Assembly:
    atoms: dict[str, RtpAtom]
    bonds: list[tuple[str, str]]
    impropers: list[tuple[str, str, str, str]]
    hdb: list[list[str]]
    base_residues: list[dict[str, object]]
    applied_patches: list[dict[str, object]]


IDENTITIES: dict[str, tuple[str, str]] = {
    "aAbe(1->3)[aDGal(1->2)]aDManOMe": (
        "CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O[C@H]2O[C@H](C)[C@H](O)C[C@H]2O)[C@@H]1O[C@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O",
        "C19H34O14",
    ),
    "aDGal2AcOMe": ("CO[C@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1OC(C)=O", "C9H16O7"),
    "aDGal3AcOMe": ("CO[C@H]1O[C@H](CO)[C@H](O)[C@H](OC(C)=O)[C@H]1O", "C9H16O7"),
    "aDGal4AcOMe": ("CO[C@H]1O[C@H](CO)[C@H](OC(C)=O)[C@H](O)[C@H]1O", "C9H16O7"),
    "aDGal6AcOMe": ("CO[C@H]1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O", "C9H16O7"),
    "aDGlc2AcOMe": ("CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1OC(C)=O", "C9H16O7"),
    "aDGlc3AcOMe": ("CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](OC(C)=O)[C@H]1O", "C9H16O7"),
    "aDGlc4AcOMe": ("CO[C@H]1O[C@H](CO)[C@@H](OC(C)=O)[C@H](O)[C@H]1O", "C9H16O7"),
    "aDGlc6AcOMe": ("CO[C@H]1O[C@H](COC(C)=O)[C@@H](O)[C@H](O)[C@H]1O", "C9H16O7"),
    "aLRha2AcOMe": ("CO[C@@H]1O[C@@H](C)[C@H](O)[C@@H](O)[C@H]1OC(C)=O", "C9H16O6"),
    "aLRha3AcOMe": ("CO[C@@H]1O[C@@H](C)[C@H](O)[C@@H](OC(C)=O)[C@H]1O", "C9H16O6"),
    "aLRha4AcOMe": ("CO[C@@H]1O[C@@H](C)[C@H](OC(C)=O)[C@@H](O)[C@H]1O", "C9H16O6"),
    "bDGal2AcOMe": ("CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1OC(C)=O", "C9H16O7"),
    "bDGal3AcOMe": ("CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](OC(C)=O)[C@H]1O", "C9H16O7"),
    "bDGal4AcOMe": ("CO[C@@H]1O[C@H](CO)[C@H](OC(C)=O)[C@H](O)[C@H]1O", "C9H16O7"),
    "bDGal6AcOMe": ("CO[C@@H]1O[C@H](COC(C)=O)[C@H](O)[C@H](O)[C@H]1O", "C9H16O7"),
    "bDGlc2AcOMe": ("CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1OC(C)=O", "C9H16O7"),
    "bDGlc3AcOMe": ("CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](OC(C)=O)[C@H]1O", "C9H16O7"),
    "bDGlc4AcOMe": ("CO[C@@H]1O[C@H](CO)[C@@H](OC(C)=O)[C@H](O)[C@H]1O", "C9H16O7"),
    "bDGlc6AcOMe": ("CO[C@@H]1O[C@H](COC(C)=O)[C@@H](O)[C@H](O)[C@H]1O", "C9H16O7"),
    "aDGalAOMe": ("CO[C@H]1O[C@H](C(=O)O)[C@H](O)[C@H](O)[C@H]1O", "C7H12O7"),
    "aDGlcAOMe": ("CO[C@H]1O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]1O", "C7H12O7"),
    "bDGlcAOMe": ("CO[C@@H]1O[C@H](C(=O)O)[C@@H](O)[C@H](O)[C@H]1O", "C7H12O7"),
    "aDGlc11aDGlc": ("OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "C12H22O11"),
    "aDGlc11bDGlc": ("OC[C@H]1O[C@@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "C12H22O11"),
    "bDGlc11bDGlc": ("OC[C@H]1O[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "C12H22O11"),
    "aDGalA14aDGalA": ("O=C(O)[C@H]1O[C@H](O[C@@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2C(=O)O)[C@H](O)[C@@H](O)[C@H]1O", "C12H18O13"),
    "aDGalA14bDGalA": ("O=C(O)[C@H]1O[C@H](O[C@@H]2[C@H](O)[C@@H](O)[C@H](O)O[C@@H]2C(=O)O)[C@H](O)[C@@H](O)[C@H]1O", "C12H18O13"),
    "aDGlcA13aLGal": ("O=C(O)[C@H]1O[C@H](O[C@H]2[C@H](O)[C@H](O)O[C@@H](CO)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "C12H20O12"),
    "aDGlcA13bLGal": ("O=C(O)[C@H]1O[C@H](O[C@@H]2[C@H](O)[C@H](CO)O[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O", "C12H20O12"),
}


def _identity(name: str) -> tuple[str, str]:
    return IDENTITIES[name]


def make_targets() -> dict[str, Target]:
    targets: dict[str, Target] = {}

    mono_specs = {
        "aDGal": ("AGAL", "OMEA", {2: "2BAC", 3: "3BAC", 4: "4AAC", 6: "6TAC"}),
        "aDGlc": ("AGLC", "OMEA", {2: "2BAC", 3: "3BAC", 4: "4BAC", 6: "6TAC"}),
        # alpha-L-rhamnose is 6-deoxy-L-mannose in its 1C4 chair: C2 axial,
        # C3/C4 equatorial.  A/B in the reviewed imlab PRES names denotes
        # substituent geometry, not anomeric alpha/beta nomenclature.
        "aLRha": ("ARHM", "OMEA", {2: "2AAC", 3: "3BAC", 4: "4BAC"}),
        "bDGal": ("BGAL", "OMEB", {2: "2BAC", 3: "3BAC", 4: "4AAC", 6: "6TAC"}),
        "bDGlc": ("BGLC", "OMEB", {2: "2BAC", 3: "3BAC", 4: "4BAC", 6: "6TAC"}),
    }
    for stem, (residue, ome, positions) in mono_specs.items():
        for position, acetyl in positions.items():
            name = f"{stem}{position}AcOMe"
            smiles, formula = _identity(name)
            if name == "aLRha2AcOMe":
                bases = (("ARHMOA", ""),)
                patches = (
                    PatchSpec("preexpanded_o_acetyl", "OR2AC", position=2),
                    PatchSpec("o_methyl", ome),
                )
            else:
                bases = ((residue, ""),)
                patches = (
                    PatchSpec("o_methyl", ome),
                    PatchSpec("o_acetyl", acetyl, position=position),
                )
            targets[name] = Target(
                name,
                name,
                "mono",
                bases,
                patches,
                smiles,
                formula,
                0,
            )

    for name, residue, ome in (
        ("aDGalAOMe", "AGALA", "OMEA"),
        ("aDGlcAOMe", "AGLCA", "OMEA"),
        ("bDGlcAOMe", "BGLCA", "OMEB"),
    ):
        smiles, formula = _identity(name)
        targets[name] = Target(
            name,
            name,
            "mono",
            ((residue, ""),),
            (PatchSpec("o_methyl", ome),),
            smiles,
            formula,
            -1,
        )

    for name, first, second, patch in (
        ("aDGlc11aDGlc", "AGLC", "AGLC", "11aa"),
        ("aDGlc11bDGlc", "AGLC", "BGLC", "11ab"),
        ("bDGlc11bDGlc", "BGLC", "BGLC", "11bb"),
    ):
        smiles, formula = _identity(name)
        targets[name] = Target(
            name,
            name,
            "dimer",
            ((first, "A"), (second, "B")),
            (PatchSpec("link_1_1", patch, "A", "B", 1),),
            smiles,
            formula,
            0,
        )

    for name, acceptor, donor, patch, position, charge in (
        ("aDGalA14aDGalA", "AGALA", "AGALA", "14aa", 4, -2),
        ("aDGalA14bDGalA", "BGALA", "AGALA", "14aa", 4, -2),
        ("aDGlcA13aLGal", "ALGAL", "AGLCA", "13ab", 3, -1),
        ("aDGlcA13bLGal", "BLGAL", "AGLCA", "13ab", 3, -1),
    ):
        smiles, formula = _identity(name)
        targets[name] = Target(
            name,
            name,
            "dimer",
            ((acceptor, "A"), (donor, "B")),
            (PatchSpec("link_1_n", patch, "A", "B", position),),
            smiles,
            formula,
            charge,
        )

    raw = "aAbe(1->3)[aDGal(1->2)]aDManOMe"
    smiles, formula = _identity(raw)
    targets[raw] = Target(
        raw,
        "aAbe13[aDGal12]aDManOMe",
        "trimer",
        (("AMAN", "A"), ("AGAL", "B"), ("ABEQ", "C")),
        (
            PatchSpec("o_methyl", "OMEA", "A"),
            PatchSpec("link_1_n", "12aa", "A", "B", 2),
            PatchSpec("link_1_n", "13ab", "A", "C", 3),
        ),
        smiles,
        formula,
        0,
    )
    if len(targets) != 30 or set(targets) != set(IDENTITIES):
        raise RuntimeError("Internal CHARMM standard-patch allowlist is inconsistent")
    return targets


TARGETS = make_targets()
TARGET_BY_CLEAN = {target.clean_name: target for target in TARGETS.values()}


def atom_element(name: str) -> str:
    stripped = name.lstrip("0123456789")
    if not stripped or stripped[0] not in "CHO":
        raise ValueError(f"Cannot derive supported element from RTP atom name {name!r}")
    return stripped[0]


def unit_atom(prefix: str, name: str) -> str:
    """Create a <=5-character unique name while keeping hydrogens H-first.

    GROMACS and solvation utilities use element-leading atom-name
    conventions.  Insert the A/B/C unit tag after the element letter:
    C1 -> CA1, O1 -> OA1, H1 -> HA1, HO1 -> HAO1.  HDB multi-hydrogen
    expansion remains consistent, e.g. HA6 -> HA61/HA62.
    """

    if not prefix:
        return name
    if prefix not in "ABC" or len(prefix) != 1:
        raise ValueError(f"Unsupported local unit tag {prefix!r}")
    return name[0] + prefix + name[1:]


def local_base_name(name: str, active_prefixes: set[str] | frozenset[str]) -> str:
    """Remove only the local alphabetic unit tag, never chemical name digits."""

    if (
        len(name) > 1
        and name[0] in "CHO"
        and name[1] in active_prefixes
    ):
        return name[0] + name[2:]
    return name


def parse_rtp_residue(path: Path, residue: str) -> tuple[list[RtpAtom], list[tuple[str, str]], list[tuple[str, str, str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.startswith("[") and re.fullmatch(rf"\[\s*{re.escape(residue)}\s*\]", line.strip()):
            start = index + 1
            break
    if start is None:
        raise ValueError(f"RTP residue {residue!r} is absent from {path}")
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("["):
            end = index
            break

    atoms: list[RtpAtom] = []
    bonds: list[tuple[str, str]] = []
    impropers: list[tuple[str, str, str, str]] = []
    section = ""
    allowed_sections = {"atoms", "bonds", "impropers"}
    seen_sections: set[str] = set()
    for line in lines[start:end]:
        stripped = line.split(";", 1)[0].strip()
        if not stripped:
            continue
        section_match = re.fullmatch(r"\[\s*([^]]+)\s*\]", stripped)
        if section_match:
            section = section_match.group(1).strip().lower()
            seen_sections.add(section)
            continue
        fields = stripped.split()
        if section == "atoms":
            if len(fields) < 4:
                raise ValueError(f"Malformed RTP atom line for {residue}: {line!r}")
            atoms.append(RtpAtom(fields[0], fields[1], float(fields[2]), int(fields[3])))
        elif section == "bonds":
            if len(fields) < 2:
                raise ValueError(f"Malformed RTP bond line for {residue}: {line!r}")
            bonds.append((fields[0], fields[1]))
        elif section == "impropers":
            if len(fields) < 4:
                raise ValueError(f"Malformed RTP improper line for {residue}: {line!r}")
            impropers.append(tuple(fields[:4]))
        else:
            raise ValueError(f"Unexpected RTP content outside a section for {residue}: {line!r}")
    unexpected = seen_sections - allowed_sections
    if unexpected or not atoms or not bonds:
        raise ValueError(
            f"Unsupported RTP sections for {residue}: unexpected={sorted(unexpected)}"
        )
    return atoms, bonds, impropers


def parse_hdb_residue(path: Path, residue: str) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = re.compile(rf"^{re.escape(residue)}\s+(\d+)\s*$")
    matches = [(index, int(match.group(1))) for index, line in enumerate(lines) if (match := header.match(line))]
    if len(matches) != 1:
        raise ValueError(f"Expected one HDB entry for {residue!r} in {path}; found {len(matches)}")
    start, count = matches[0]
    recipes = [lines[index].split() for index in range(start + 1, start + 1 + count)]
    if len(recipes) != count or any(len(recipe) < 5 for recipe in recipes):
        raise ValueError(f"Malformed HDB entry for {residue!r}")
    return recipes


def prefix_recipe(recipe: list[str], prefix: str) -> list[str]:
    return recipe[:2] + [unit_atom(prefix, token) for token in recipe[2:]]


def base_assembly(source_ff: Path, target: Target) -> Assembly:
    atoms: dict[str, RtpAtom] = {}
    bonds: list[tuple[str, str]] = []
    impropers: list[tuple[str, str, str, str]] = []
    hdb: list[list[str]] = []
    provenance: list[dict[str, object]] = []
    group_offset = 0
    for residue, prefix in target.bases:
        base_atoms, base_bonds, base_impropers = parse_rtp_residue(source_ff / "carb.rtp", residue)
        base_hdb = parse_hdb_residue(source_ff / "carb.hdb", residue)
        groups = sorted({atom.charge_group for atom in base_atoms})
        group_map = {group: group_offset + index + 1 for index, group in enumerate(groups)}
        for atom in base_atoms:
            name = unit_atom(prefix, atom.name)
            if name in atoms:
                raise ValueError(f"Duplicate expanded atom name {name!r}")
            atoms[name] = RtpAtom(name, atom.atom_type, atom.charge, group_map[atom.charge_group])
        bonds.extend((unit_atom(prefix, first), unit_atom(prefix, second)) for first, second in base_bonds)
        impropers.extend(tuple(unit_atom(prefix, name) for name in row) for row in base_impropers)
        hdb.extend(prefix_recipe(recipe, prefix) for recipe in base_hdb)
        group_offset = max(group_map.values())
        provenance.append(
            {
                "residue": residue,
                "prefix": prefix,
                "atom_count": len(base_atoms),
                "charge_e": sum(atom.charge for atom in base_atoms),
            }
        )
    return Assembly(atoms, bonds, impropers, hdb, provenance, [])


def replace_atom(assembly: Assembly, name: str, atom_type: str, charge: float) -> None:
    old = assembly.atoms.get(name)
    if old is None:
        raise ValueError(f"Patch update refers to missing atom {name!r}")
    assembly.atoms[name] = RtpAtom(name, atom_type, charge, old.charge_group)


def delete_atom(assembly: Assembly, name: str) -> None:
    if name not in assembly.atoms:
        raise ValueError(f"Patch deletion refers to missing atom {name!r}")
    del assembly.atoms[name]
    assembly.bonds = [bond for bond in assembly.bonds if name not in bond]
    assembly.impropers = [row for row in assembly.impropers if name not in row]


def remove_hdb_target(assembly: Assembly, name: str) -> None:
    matches = [recipe for recipe in assembly.hdb if recipe[2] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one HDB recipe generating {name!r}; found {len(matches)}")
    assembly.hdb.remove(matches[0])


def add_atom(assembly: Assembly, name: str, atom_type: str, charge: float, group: int) -> None:
    if name in assembly.atoms:
        raise ValueError(f"Patch atom {name!r} already exists")
    assembly.atoms[name] = RtpAtom(name, atom_type, charge, group)


def add_bond(assembly: Assembly, first: str, second: str) -> None:
    if first not in assembly.atoms or second not in assembly.atoms:
        raise ValueError(f"Patch bond refers to a missing atom: {first!r}-{second!r}")
    key = frozenset((first, second))
    if any(frozenset(row) == key for row in assembly.bonds):
        raise ValueError(f"Patch bond already exists: {first!r}-{second!r}")
    assembly.bonds.append((first, second))


def next_group(assembly: Assembly) -> int:
    return max(atom.charge_group for atom in assembly.atoms.values()) + 1


def manifest_template(manifest: dict[str, object], name: str) -> dict[str, object]:
    templates = manifest.get("templates")
    if not isinstance(templates, dict) or not isinstance(templates.get(name), dict):
        raise ValueError(f"Patch manifest lacks template {name!r}")
    return templates[name]


def apply_o_methyl(assembly: Assembly, patch: PatchSpec, manifest: dict[str, object]) -> None:
    template = manifest_template(manifest, "o_methyl")
    if patch.name not in template.get("source_patches", []):
        raise ValueError(f"O-methyl patch {patch.name!r} is not manifest-approved")
    prefix = patch.unit1
    delete = unit_atom(prefix, str(template["delete_hydrogen"]))
    remove_hdb_target(assembly, delete)
    delete_atom(assembly, delete)
    updates = template["updates"]
    for base_name in ("C1", "O1"):
        atom_type, charge = updates[base_name]
        replace_atom(assembly, unit_atom(prefix, base_name), str(atom_type), float(charge))
    group = next_group(assembly)
    added = template["added_atoms"]
    for base_name in ("CM", "HM1", "HM2", "HM3"):
        atom_type, charge = added[base_name]
        add_atom(assembly, unit_atom(prefix, base_name), str(atom_type), float(charge), group)
    add_bond(assembly, unit_atom(prefix, "O1"), unit_atom(prefix, "CM"))
    for index in (1, 2, 3):
        add_bond(assembly, unit_atom(prefix, "CM"), unit_atom(prefix, f"HM{index}"))
    assembly.hdb.append(
        [
            "3",
            "4",
            unit_atom(prefix, "HM"),
            unit_atom(prefix, "CM"),
            unit_atom(prefix, "O1"),
            unit_atom(prefix, "C1"),
        ]
    )
    assembly.applied_patches.append(
        {"kind": patch.kind, "source_patch": patch.name, "unit_prefix": prefix, "manifest_template": "o_methyl"}
    )


def apply_o_acetyl(assembly: Assembly, patch: PatchSpec, manifest: dict[str, object]) -> None:
    position = patch.position
    template_name = "o_acetyl_primary" if position == 6 else "o_acetyl_secondary"
    template = manifest_template(manifest, template_name)
    if patch.name not in template.get("source_patches", []):
        raise ValueError(f"O-acetyl patch {patch.name!r} is not manifest-approved")
    prefix = patch.unit1
    remove_hdb_target(assembly, unit_atom(prefix, f"HO{position}"))
    delete_atom(assembly, unit_atom(prefix, f"HO{position}"))
    replace_atom(
        assembly,
        unit_atom(prefix, f"C{position}"),
        str(template["carbon_type"]),
        float(template["carbon_charge"]),
    )
    replace_atom(
        assembly,
        unit_atom(prefix, f"O{position}"),
        str(template["oxygen_type"]),
        float(template["oxygen_charge"]),
    )
    group = next_group(assembly)
    source_added = template["acetyl_atoms"]
    names = {
        "CY": unit_atom(prefix, f"CY{position}"),
        "CX": unit_atom(prefix, f"CX{position}"),
        "OX": unit_atom(prefix, f"OX{position}"),
    }
    for generic in ("CY", "CX", "OX"):
        atom_type, charge = source_added[generic]
        add_atom(assembly, names[generic], str(atom_type), float(charge), group)
    for index in (1, 2, 3):
        atom_type, charge = source_added[f"HY{index}"]
        add_atom(
            assembly,
            unit_atom(prefix, f"HY{position}{index}"),
            str(atom_type),
            float(charge),
            group,
        )
    add_bond(assembly, unit_atom(prefix, f"O{position}"), names["CX"])
    add_bond(assembly, names["CX"], names["CY"])
    add_bond(assembly, names["CX"], names["OX"])
    for index in (1, 2, 3):
        add_bond(assembly, names["CY"], unit_atom(prefix, f"HY{position}{index}"))
    assembly.hdb.append(
        [
            "3",
            "4",
            unit_atom(prefix, f"HY{position}"),
            names["CY"],
            names["CX"],
            unit_atom(prefix, f"O{position}"),
        ]
    )
    assembly.applied_patches.append(
        {
            "kind": patch.kind,
            "source_patch": patch.name,
            "unit_prefix": prefix,
            "position": position,
            "manifest_template": template_name,
        }
    )


def validate_preexpanded_o_acetyl(
    assembly: Assembly,
    patch: PatchSpec,
    manifest: dict[str, object],
) -> None:
    template_name = "o_acetyl_rhamnose_c2_preexpanded"
    template = manifest_template(manifest, template_name)
    if patch.name not in template.get("source_patches", []):
        raise ValueError(f"Pre-expanded O-acetyl patch {patch.name!r} is not approved")
    if template.get("source_residue") != "ARHMOA" or patch.position != 2:
        raise ValueError("Pre-expanded OR2AC provenance does not target ARHMOA C2")
    required = template.get("required_atoms")
    if not isinstance(required, dict):
        raise ValueError("OR2AC manifest template lacks its exact required atoms")
    for name, expected in required.items():
        atom = assembly.atoms.get(name)
        if atom is None or not isinstance(expected, list) or len(expected) != 2:
            raise ValueError(f"ARHMOA/OR2AC required atom is absent or malformed: {name}")
        if atom.atom_type != expected[0] or abs(atom.charge - float(expected[1])) > 1.0e-9:
            raise ValueError(
                f"Pinned ARHMOA atom {name} differs from official OR2AC: "
                f"{atom.atom_type}/{atom.charge:+.6f} vs {expected}"
            )
    assembly.applied_patches.append(
        {
            "kind": patch.kind,
            "source_patch": patch.name,
            "source_residue": "ARHMOA",
            "position": 2,
            "manifest_template": template_name,
            "operation": "validated pre-expanded residue; no second transformation applied",
        }
    )


def apply_link(assembly: Assembly, patch: PatchSpec, manifest: dict[str, object]) -> None:
    position = patch.position
    template_name = "link_1_1" if patch.kind == "link_1_1" else "link_1_n_secondary"
    template = manifest_template(manifest, template_name)
    if patch.name not in template.get("source_patches", []):
        raise ValueError(f"Link patch {patch.name!r} is not manifest-approved")
    acceptor = patch.unit1
    donor = patch.unit2
    linker_oxygen = unit_atom(acceptor, f"O{position}")
    acceptor_hydrogen = unit_atom(acceptor, f"HO{position}")
    donor_oxygen = unit_atom(donor, "O1")
    donor_hydrogen = unit_atom(donor, "HO1")
    remove_hdb_target(assembly, acceptor_hydrogen)
    remove_hdb_target(assembly, donor_hydrogen)
    delete_atom(assembly, acceptor_hydrogen)
    delete_atom(assembly, donor_hydrogen)
    delete_atom(assembly, donor_oxygen)
    if patch.kind == "link_1_1":
        carbon_type, carbon_charge = template["acceptor_carbon"]
        oxygen_type, oxygen_charge = template["acceptor_oxygen"]
    else:
        carbon_type, carbon_charge = template["acceptor_carbon"]
        oxygen_type, oxygen_charge = template["acceptor_oxygen"]
    replace_atom(
        assembly,
        unit_atom(acceptor, f"C{position}"),
        str(carbon_type),
        float(carbon_charge),
    )
    replace_atom(assembly, linker_oxygen, str(oxygen_type), float(oxygen_charge))
    donor_type, donor_charge = template["donor_carbon"]
    replace_atom(assembly, unit_atom(donor, "C1"), str(donor_type), float(donor_charge))
    add_bond(assembly, linker_oxygen, unit_atom(donor, "C1"))

    donor_h1 = unit_atom(donor, "H1")
    recipes = [recipe for recipe in assembly.hdb if recipe[2] == donor_h1]
    if len(recipes) != 1 or donor_oxygen not in recipes[0][3:]:
        raise ValueError(f"Could not retarget donor HDB recipe for {donor_h1!r}")
    recipes[0][:] = [linker_oxygen if token == donor_oxygen else token for token in recipes[0]]
    assembly.applied_patches.append(
        {
            "kind": patch.kind,
            "source_patch": patch.name,
            "acceptor_prefix": acceptor,
            "donor_prefix": donor,
            "acceptor_position": position,
            "manifest_template": template_name,
        }
    )


def expanded_hydrogen_names(recipe: list[str]) -> list[str]:
    count = int(recipe[0])
    base = recipe[2]
    return [base] if count == 1 else [base + str(index) for index in range(1, count + 1)]


def validate_assembly(assembly: Assembly, target: Target) -> dict[str, object]:
    if len(assembly.atoms) != len(set(assembly.atoms)):
        raise ValueError("Expanded RTP atom names are not unique")
    too_long = [name for name in assembly.atoms if len(name) > 5]
    if too_long:
        raise ValueError(f"Expanded RTP atom names exceed GRO width: {too_long}")
    for first, second in assembly.bonds:
        if first not in assembly.atoms or second not in assembly.atoms:
            raise ValueError(f"Expanded RTP bond has a missing atom: {first}-{second}")
    for row in assembly.impropers:
        if any(name not in assembly.atoms for name in row):
            raise ValueError(f"Expanded RTP improper has a missing atom: {row}")
    for recipe in assembly.hdb:
        if any(name not in assembly.atoms for name in recipe[3:]):
            raise ValueError(f"Expanded HDB recipe references a missing atom: {recipe}")
    expected_h = sorted(name for name in assembly.atoms if atom_element(name) == "H")
    generated_h = sorted(name for recipe in assembly.hdb for name in expanded_hydrogen_names(recipe))
    if expected_h != generated_h:
        raise ValueError(
            "Expanded HDB does not generate exactly the RTP hydrogens: "
            f"missing={sorted(set(expected_h) - set(generated_h))}, "
            f"extra={sorted(set(generated_h) - set(expected_h))}"
        )
    charge = sum(atom.charge for atom in assembly.atoms.values())
    if abs(charge - target.expected_charge) > CHARGE_TOLERANCE:
        raise ValueError(
            f"Expanded RTP charge {charge:+.6f} e disagrees with expected "
            f"{target.expected_charge:+d} e for {target.clean_name}"
        )
    heavy = [name for name in assembly.atoms if atom_element(name) != "H"]
    return {
        "atom_count": len(assembly.atoms),
        "heavy_atom_count": len(heavy),
        "hydrogen_count": len(expected_h),
        "bond_count": len(assembly.bonds),
        "improper_count": len(assembly.impropers),
        "hdb_recipe_count": len(assembly.hdb),
        "charge_e": charge,
    }


def build_assembly(source_ff: Path, target: Target, manifest: dict[str, object]) -> Assembly:
    assembly = base_assembly(source_ff, target)
    for patch in target.patches:
        if patch.kind == "o_methyl":
            apply_o_methyl(assembly, patch, manifest)
        elif patch.kind == "o_acetyl":
            apply_o_acetyl(assembly, patch, manifest)
        elif patch.kind == "preexpanded_o_acetyl":
            validate_preexpanded_o_acetyl(assembly, patch, manifest)
        elif patch.kind in {"link_1_1", "link_1_n"}:
            apply_link(assembly, patch, manifest)
        else:
            raise ValueError(f"Unsupported declarative patch kind {patch.kind!r}")
    validate_assembly(assembly, target)
    return assembly


def validate_manifest(path: Path, source_ff: Path) -> tuple[dict[str, object], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"CHARMM standard-patch manifest is absent: {path}")
    manifest_sha256 = sha256_file(path)
    if manifest_sha256 != APPROVED_MANIFEST_SHA256:
        raise ValueError(
            "CHARMM standard-patch manifest content is not the reviewed object: "
            f"observed {manifest_sha256}, expected {APPROVED_MANIFEST_SHA256}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported CHARMM standard-patch manifest schema")
    if manifest.get("gromacs_port") != source_ff.name:
        raise ValueError(
            f"Manifest port {manifest.get('gromacs_port')!r} does not match {source_ff.name!r}"
        )
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict) or upstream.get("commit") != "ed3bbdaa4eb0a0152e577480bb0a9d22cd9743c3":
        raise ValueError("Manifest does not pin the reviewed February-2026 upstream commit")
    expected_files = {
        "charmm/toppar/top_all36_carb.rtf": "41749e71731d46d479b041ba8a0877477bb0e46e4264ea25fbee73299eb19a59",
        "charmm/toppar/stream/carb/toppar_all36_carb_imlab.str": "3a5852e14dfe4f5aca4c8c82428c038946da144772da3568a53c7bcbcbe26e35",
    }
    if upstream.get("files") != expected_files:
        raise ValueError("Manifest upstream file hashes differ from the reviewed sources")
    required = ("carb.rtp", "carb.hdb", "forcefield.itp", "ffbonded.itp", "ffnonbonded.itp", "tip3p.itp", "ions.itp")
    missing = [name for name in required if not (source_ff / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Pinned CHARMM force field is incomplete: {missing}")
    source_tree = sha256_directory(source_ff)
    if source_tree["sha256"] != APPROVED_SOURCE_FF_TREE_SHA256:
        raise ValueError(
            "CHARMM source force-field tree is not the reviewed February-2026 tree: "
            f"observed {source_tree['sha256']}, expected {APPROVED_SOURCE_FF_TREE_SHA256}"
        )
    return manifest, {
        "manifest": str(path),
        "manifest_sha256": manifest_sha256,
        "reviewed_upstream": upstream,
        "source_forcefield": str(source_ff),
        "source_forcefield_tree": source_tree,
    }


def validate_source_identity(molecule: Chem.Mol, target: Target) -> dict[str, object]:
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    heavy_count = sum(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms())
    if smiles != target.expected_isomeric_smiles:
        raise ValueError(
            "CASPER stereochemical identity mismatch: "
            f"observed {smiles!r}, expected {target.expected_isomeric_smiles!r}"
        )
    if formula != target.expected_formula:
        raise ValueError(f"CASPER formula mismatch: observed {formula}, expected {target.expected_formula}")
    return {
        "canonical_isomeric_smiles": smiles,
        "formula": formula,
        "heavy_atom_count": heavy_count,
        "chiral_centers": [
            {"atom_index": int(index), "cip": str(cip)}
            for index, cip in Chem.FindMolChiralCenters(
                molecule, includeUnassigned=True, useLegacyImplementation=False
            )
        ],
    }


def assembly_graph(assembly: Assembly) -> nx.Graph:
    graph = nx.Graph()
    for name in assembly.atoms:
        if atom_element(name) != "H":
            graph.add_node(name, element=atom_element(name))
    for first, second in assembly.bonds:
        if first in graph and second in graph:
            graph.add_edge(first, second)
    return graph


def source_graph(molecule: Chem.Mol) -> nx.Graph:
    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() > 1:
            graph.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in molecule.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first in graph and second in graph:
            graph.add_edge(first, second)
    return graph


def carbonyl_constraints(
    molecule: Chem.Mol,
    mapping: dict[int, str],
    active_prefixes: set[str] | frozenset[str],
) -> bool:
    for carbon in molecule.GetAtoms():
        if carbon.GetAtomicNum() != 6:
            continue
        oxygen_bonds = [
            bond
            for bond in carbon.GetBonds()
            if bond.GetOtherAtom(carbon).GetAtomicNum() == 8
        ]
        double = [bond.GetOtherAtom(carbon) for bond in oxygen_bonds if bond.GetBondType() == Chem.BondType.DOUBLE]
        single = [bond.GetOtherAtom(carbon) for bond in oxygen_bonds if bond.GetBondType() == Chem.BondType.SINGLE]
        if len(double) != 1 or len(single) != 1:
            continue
        raw_double_name = mapping[double[0].GetIdx()]
        raw_single_name = mapping[single[0].GetIdx()]
        double_names = {
            raw_double_name,
            local_base_name(raw_double_name, active_prefixes),
        }
        single_names = {
            raw_single_name,
            local_base_name(raw_single_name, active_prefixes),
        }
        single_heavy_degree = sum(
            neighbor.GetAtomicNum() > 1 for neighbor in single[0].GetNeighbors()
        )
        if single_heavy_degree == 1:
            if "O61" not in double_names or "O62" not in single_names:
                return False
        elif not (
            any(name.startswith("OX") or name == "OA2" for name in double_names)
            and bool(single_names & {"O2", "O3", "O4", "O6"})
        ):
            return False
    return True


def anomeric_cip_constraints(
    molecule: Chem.Mol,
    mapping: dict[int, str],
    target: Target | None,
) -> bool:
    """Bind graph-symmetric 1<->1 units to alpha/beta stereochemistry."""

    if target is None or not any(patch.kind == "link_1_1" for patch in target.patches):
        return True
    expected_by_residue = {"AGLC": "R", "BGLC": "S"}
    centers = dict(
        Chem.FindMolChiralCenters(
            molecule,
            includeUnassigned=True,
            useLegacyImplementation=False,
        )
    )
    name_to_source = {name: index for index, name in mapping.items()}
    for residue, prefix in target.bases:
        expected = expected_by_residue.get(residue)
        if expected is None:
            raise ValueError(
                f"No reviewed 1<->1 anomeric-CIP constraint exists for {residue}"
            )
        source_index = name_to_source.get(unit_atom(prefix, "C1"))
        if source_index is None or centers.get(source_index) != expected:
            return False
    return True


def map_source_graph(
    molecule: Chem.Mol,
    assembly: Assembly,
    target: Target | None = None,
) -> tuple[dict[int, str], dict[str, object]]:
    source = source_graph(molecule)
    expanded = assembly_graph(assembly)
    if source.number_of_nodes() != expanded.number_of_nodes() or source.number_of_edges() != expanded.number_of_edges():
        raise ValueError(
            "CASPER and expanded RTP heavy graphs differ in size: "
            f"source={source.number_of_nodes()} atoms/{source.number_of_edges()} bonds, "
            f"RTP={expanded.number_of_nodes()} atoms/{expanded.number_of_edges()} bonds"
        )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source,
        expanded,
        node_match=lambda first, second: first["element"] == second["element"],
    )
    heavy_indices = sorted(source.nodes)
    candidates: list[dict[int, str]] = []
    enumerated = 0
    active_prefixes = frozenset(
        prefix for _residue, prefix in target.bases if prefix
    ) if target is not None else frozenset()
    for candidate in matcher.isomorphisms_iter():
        enumerated += 1
        if enumerated > 100_000:
            raise RuntimeError("Heavy-graph isomorphism became unexpectedly ambiguous")
        if carbonyl_constraints(
            molecule, candidate, active_prefixes
        ) and anomeric_cip_constraints(
            molecule, candidate, target
        ):
            candidates.append(dict(candidate))
    if not candidates:
        raise ValueError(
            "No element-, carbonyl-, and anomer-consistent CASPER-to-RTP graph "
            "isomorphism exists"
        )
    equivalent_unit_automorphisms = False
    if len(candidates) > 1 and target is not None:
        link_1_1 = any(patch.kind == "link_1_1" for patch in target.patches)
        base_residues = [residue for residue, _prefix in target.bases]
        equivalent_unit_automorphisms = link_1_1 and len(set(base_residues)) == 1
        if not equivalent_unit_automorphisms:
            raise ValueError(
                "Multiple stereochemically valid graph mappings remain for non-equivalent "
                "CHARMM units; refusing index-dependent selection"
            )
    candidates.sort(key=lambda row: tuple(row[index] for index in heavy_indices))
    selected = candidates[0]
    return selected, {
        "graph_isomorphisms_enumerated": enumerated,
        "carbonyl_and_anomer_consistent_isomorphisms": len(candidates),
        # Retain the old key for machine consumers while making the stronger
        # condition explicit in the adjacent field.
        "carbonyl_consistent_isomorphisms": len(candidates),
        "equivalent_unit_automorphisms": equivalent_unit_automorphisms,
        "selection": (
            "unique stereochemistry-bound mapping"
            if len(candidates) == 1
            else "one of explicitly equivalent identical-residue unit automorphisms"
        ),
    }


def append_local_residue(forcefield: Path, assembly: Assembly) -> dict[str, object]:
    rtp = forcefield / "carb.rtp"
    hdb = forcefield / "carb.hdb"
    if re.search(rf"^\[\s*{LOCAL_RESIDUE}\s*\]", rtp.read_text(encoding="utf-8"), flags=re.MULTILINE):
        raise ValueError(f"Local RTP residue {LOCAL_RESIDUE} already exists")
    rtp_lines = ["", f"[ {LOCAL_RESIDUE} ]", "  [ atoms ]"]
    for atom in assembly.atoms.values():
        rtp_lines.append(
            f"    {atom.name:<5s} {atom.atom_type:<8s} {atom.charge: .6f} {atom.charge_group:d}"
        )
    rtp_lines.append("  [ bonds ]")
    rtp_lines.extend(f"    {first:<5s} {second:<5s}" for first, second in assembly.bonds)
    if assembly.impropers:
        rtp_lines.append("  [ impropers ]")
        rtp_lines.extend("    " + " ".join(f"{name:<5s}" for name in row) for row in assembly.impropers)
    with rtp.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(rtp_lines) + "\n")

    hdb_lines = ["", f"{LOCAL_RESIDUE:<10s} {len(assembly.hdb)}"]
    hdb_lines.extend(" ".join(recipe) for recipe in assembly.hdb)
    with hdb.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(hdb_lines) + "\n")
    return {
        "residue": LOCAL_RESIDUE,
        "rtp_file": "carb.rtp",
        "hdb_file": "carb.hdb",
        "rtp_sha256_after_append": sha256_file(rtp),
        "hdb_sha256_after_append": sha256_file(hdb),
    }


def write_heavy_gro(
    molecule: Chem.Mol,
    source_heavy_indices: list[int],
    mapping: dict[int, str],
    molecule_name: str,
    output: Path,
) -> None:
    conformer = molecule.GetConformer()
    lines = [
        f"{molecule_name} MMFF94 mapped to CHARMM36 standard patches",
        f"{len(source_heavy_indices):5d}",
    ]
    for serial, atom_index in enumerate(source_heavy_indices, start=1):
        point = conformer.GetAtomPosition(atom_index)
        lines.append(
            f"{1:5d}{LOCAL_RESIDUE:<5s}{mapping[atom_index]:>5s}{serial:5d}"
            f"{point.x / 10.0:8.3f}{point.y / 10.0:8.3f}{point.z / 10.0:8.3f}"
        )
    lines.append("   8.00000   8.00000   8.00000")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_processed_gro(path: Path, assembly: Assembly) -> tuple[list[str], float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Processed GRO is too short: {path}")
    count = int(lines[1].strip())
    atom_lines = lines[2 : 2 + count]
    if len(atom_lines) != count:
        raise ValueError("Processed GRO atom list is truncated")
    names: list[str] = []
    points: list[tuple[float, float, float]] = []
    for line in atom_lines:
        if line[5:10].strip() != LOCAL_RESIDUE:
            raise ValueError(f"Processed GRO contains a non-{LOCAL_RESIDUE} residue")
        name = line[10:15].strip()
        if name not in assembly.atoms:
            raise ValueError(f"Processed GRO contains unexpected atom {name!r}")
        names.append(name)
        point = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Processed GRO contains a non-finite coordinate")
        points.append(point)
    if len(names) != len(set(names)) or set(names) != set(assembly.atoms):
        raise ValueError("Processed GRO atom names are not a bijection with expanded RTP atoms")
    minimum = math.inf
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            minimum = min(
                minimum,
                math.sqrt(sum((points[first][axis] - points[second][axis]) ** 2 for axis in range(3))),
            )
    if minimum < 0.05:
        raise ValueError(f"Processed GRO has an atomic near-overlap: {minimum:.6f} nm")
    heavy_names = [name for name in names if atom_element(name) != "H"]
    expected_heavy = [name for name in assembly.atoms if atom_element(name) != "H"]
    if set(heavy_names) != set(expected_heavy):
        raise ValueError("Processed GRO heavy atoms differ from expanded RTP heavy atoms")
    return heavy_names, minimum


def run_preflight(
    gmx: str,
    cwd: Path,
    topology: Path,
    coordinates: Path,
    environment: dict[str, str],
    timeout_seconds: int,
    charge: int,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for label, coulomb in (("cutoff", "Cut-off"), ("pme", "PME")):
        mdp = cwd / f"strict_solute_{label}.mdp"
        tpr = cwd / f"strict_solute_{label}.tpr"
        log = cwd / f"strict_solute_{label}_grompp.log"
        mdp.write_text(preflight_mdp(coulomb), encoding="utf-8")
        process = run_logged(
            [
                gmx,
                "grompp",
                "-f",
                mdp.name,
                "-c",
                coordinates.name,
                "-p",
                topology.name,
                "-o",
                tpr.name,
                "-maxwarn",
                "0",
            ],
            cwd,
            environment,
            log,
            timeout_seconds,
        )
        warnings = count_diagnostics(process.stdout, "WARNING")
        errors = count_diagnostics(process.stdout, "ERROR")
        missing = "No default" in process.stdout
        if label == "cutoff" or charge == 0:
            if process.returncode != 0 or warnings != 0 or errors != 0 or missing or not tpr.is_file():
                raise RuntimeError(
                    f"Strict {label} grompp failed; zero warnings are allowed.\n{tail(process.stdout)}"
                )
            results[f"strict_{label}"] = {
                "returncode": process.returncode,
                "warnings": warnings,
                "errors": errors,
                "missing_default_parameters": missing,
                "tpr": tpr.name,
                "log": log.name,
            }
        else:
            expected_warning = (
                "system with net charge" in process.stdout
                and "uniform background charge" in process.stdout
                and "Too many warnings (1)" in process.stdout
            )
            if process.returncode != 1 or warnings != 1 or errors != 0 or missing or not expected_warning:
                raise RuntimeError(
                    "Strict charged PME grompp did not fail solely for its one expected "
                    f"net-charge warning.\n{tail(process.stdout)}"
                )
            results["strict_pme"] = {
                "returncode": process.returncode,
                "warnings": warnings,
                "errors": errors,
                "only_expected_net_charge_warning": True,
                "missing_default_parameters": missing,
                "tpr_created": tpr.is_file(),
                "log": log.name,
            }
    return results


def target_from_name(name: str) -> Target:
    if name in TARGETS:
        return TARGETS[name]
    if name in TARGET_BY_CLEAN:
        return TARGET_BY_CLEAN[name]
    raise KeyError(name)


def build(args: argparse.Namespace) -> Path:
    target = target_from_name(args.molecule_name)
    source_ff = Path(args.source_ff).resolve()
    manifest_path = Path(args.manifest).resolve()
    run_root = Path(args.run_root).resolve()
    sdf = Path(args.sdf).resolve() if args.sdf else DATASET_ROOT / f"casper_{target.source_class}.sdf"
    gmx = resolve_executable(args.gmx)
    if not sdf.is_file():
        raise FileNotFoundError(f"CASPER source SDF is absent: {sdf}")
    manifest, manifest_metadata = validate_manifest(manifest_path, source_ff)
    assembly = build_assembly(source_ff, target, manifest)
    assembly_metadata = validate_assembly(assembly, target)
    source = find_source_molecule(sdf, target.raw_name)
    source_metadata = validate_source_identity(source, target)
    optimized, mmff_metadata = embed_mmff(source, args.seed)
    source_to_name, graph_metadata = map_source_graph(optimized, assembly, target)
    source_heavy_indices = [atom.GetIdx() for atom in optimized.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(source_heavy_indices) != assembly_metadata["heavy_atom_count"]:
        raise ValueError("Validated source and RTP heavy-atom counts disagree")

    molecule_dir = run_root / target.clean_name
    output_dir = molecule_dir / OUTPUT_SUBDIR
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    molecule_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    staging = molecule_dir / f".{OUTPUT_SUBDIR}.tmp-{os.getpid()}-{timestamp}"
    staging.mkdir(mode=0o755)
    stem = target.raw_name.replace("/", "_")
    input_gro = staging / f"{stem}_CHARMM36_STANDARD_mmff3d_heavy.gro"
    processed_gro = staging / f"{stem}_CHARMM36_STANDARD_processed.gro"
    topology = staging / f"{stem}_CHARMM36_STANDARD.top"
    mapping_path = staging / "sdf_to_charmm_frame_heavy_index_map.json"
    metadata_path = staging / "charmm36_standard_input_metadata.json"
    pdb2gmx_log = staging / "pdb2gmx.log"
    local_ff = staging / LOCAL_FORCEFIELD_NAME
    try:
        copied = copy_pinned_forcefield(source_ff, local_ff)
        local_residue_metadata = append_local_residue(local_ff, assembly)
        modified_tree_before = sha256_directory(local_ff)
        write_heavy_gro(optimized, source_heavy_indices, source_to_name, target.raw_name, input_gro)
        pdb2gmx_environment = os.environ.copy()
        pdb2gmx_environment["GMXLIB"] = str(staging)
        command = [
            gmx,
            "pdb2gmx",
            "-f",
            input_gro.name,
            "-o",
            processed_gro.name,
            "-p",
            topology.name,
            "-ff",
            LOCAL_FORCEFIELD_NAME.removesuffix(".ff"),
            "-water",
            "tip3p",
            "-ignh",
        ]
        process = run_logged(command, staging, pdb2gmx_environment, pdb2gmx_log, args.timeout_seconds)
        if process.returncode != 0 or not processed_gro.is_file() or not topology.is_file():
            raise RuntimeError(f"pdb2gmx failed:\n{tail(process.stdout)}")
        if f'Total charge {target.expected_charge:.3f} e' not in process.stdout:
            raise RuntimeError(
                "pdb2gmx did not report the independently expected residue charge:\n"
                + tail(process.stdout)
            )
        if source_ff.name in topology.read_text(encoding="utf-8"):
            raise RuntimeError("Generated topology unexpectedly references the external source FF")

        processed_heavy, minimum_distance = parse_processed_gro(processed_gro, assembly)
        frame_index_by_name = {name: index for index, name in enumerate(processed_heavy)}
        source_heavy_names = [source_to_name[index] for index in source_heavy_indices]
        frame_mapping = [frame_index_by_name[name] for name in source_heavy_names]
        if sorted(frame_mapping) != list(range(len(frame_mapping))):
            raise RuntimeError("Generated SDF-to-frame mapping is not bijective")
        mapping_path.write_text(json.dumps(frame_mapping, indent=2) + "\n", encoding="utf-8")

        topology_charge, topology_atom_count = parse_topology_charge(topology)
        if abs(topology_charge - target.expected_charge) > CHARGE_TOLERANCE:
            raise RuntimeError(
                f"Generated topology charge {topology_charge:+.6f} e disagrees with "
                f"expected {target.expected_charge:+d} e"
            )
        self_contained_environment = os.environ.copy()
        self_contained_environment.pop("GMXLIB", None)
        preflight = run_preflight(
            gmx,
            staging,
            topology,
            processed_gro,
            self_contained_environment,
            args.timeout_seconds,
            target.expected_charge,
        )
        modified_tree_after = sha256_directory(local_ff)
        if modified_tree_after != modified_tree_before:
            raise RuntimeError("Local force-field copy changed during strict preflight")
        version = gromacs_version(gmx, self_contained_environment, args.timeout_seconds)

        charged = target.expected_charge != 0
        required_counterions = {
            "count": abs(target.expected_charge),
            "charge_e": -1 if target.expected_charge > 0 else (1 if target.expected_charge < 0 else 0),
            "note": (
                "Add these monovalent counterions after solvation and before strict PME EM/MD."
                if charged
                else "No counterion is required for this neutral solute topology."
            ),
        }
        generated_files = [
            input_gro,
            processed_gro,
            topology,
            mapping_path,
            pdb2gmx_log,
            staging / "posre.itp",
            staging / "strict_solute_cutoff.mdp",
            staging / "strict_solute_cutoff.tpr",
            staging / "strict_solute_cutoff_grompp.log",
            staging / "strict_solute_pme.mdp",
            staging / "strict_solute_pme_grompp.log",
        ]
        if not charged:
            generated_files.append(staging / "strict_solute_pme.tpr")
        missing = [path.name for path in generated_files if not path.is_file()]
        if missing:
            raise RuntimeError(f"Expected generated files are absent: {missing}")
        metadata = {
            "schema_version": 2,
            "topology_policy": "charmm36_standard_patch_expansion_v1",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "production_ready": True,
            "production_ready_scope": "solute topology only",
            "water_md_started": False,
            "requires_solvation": True,
            "requires_neutralization": charged,
            "required_counterions": required_counterions,
            "molecule_name": target.raw_name,
            "clean_name": target.clean_name,
            "source_class": target.source_class,
            "generator": {
                "entrypoint": Path(__file__).name,
                "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
                "imported_helpers": {
                    "prepare_charmm_standard_uronic_input.py": sha256_file(
                        Path(__file__).resolve().with_name(
                            "prepare_charmm_standard_uronic_input.py"
                        )
                    )
                },
            },
            "allowlist_size": len(TARGETS),
            "rtp_residue": LOCAL_RESIDUE,
            "expected_solute_charge_e": target.expected_charge,
            "observed_topology_charge_e": topology_charge,
            "topology_atom_count": topology_atom_count,
            "source": {
                "sdf": str(sdf),
                "sdf_sha256": sha256_file(sdf),
                **source_metadata,
            },
            "conformer": mmff_metadata,
            "standard_patch_expansion": {
                "policy": "declarative expansion of only reviewed upstream CHARMM PRES records",
                "coordinate_policy": manifest["coordinate_policy"],
                "base_residues": assembly.base_residues,
                "applied_patches": assembly.applied_patches,
                "expanded_residue": local_residue_metadata,
                "assembly": assembly_metadata,
                "graph_mapping_validation": graph_metadata,
            },
            "forcefield": {
                **manifest_metadata,
                "initial_self_contained_copy": copied,
                "modified_local_tree": modified_tree_after,
                "external_forcefield_lookup_required": False,
            },
            "gromacs": {"executable": gmx, "version": version},
            "atom_mapping": {
                "file": mapping_path.name,
                "semantics": "zero-based SDF heavy-atom order to processed-frame heavy-atom order",
                "mapping": frame_mapping,
                "sdf_heavy_atom_names": source_heavy_names,
                "processed_heavy_atom_names": processed_heavy,
                "source_atom_index_to_charmm_name": {
                    str(index): source_to_name[index] for index in source_heavy_indices
                },
            },
            "geometry": {"minimum_all_atom_distance_nm": minimum_distance},
            "pdb2gmx": {"command": command, "log": pdb2gmx_log.name},
            "preflight": {
                **preflight,
                "self_contained_forcefield": {
                    "GMXLIB": "unset",
                    "working_directory": ".",
                    "local_directory": LOCAL_FORCEFIELD_NAME,
                },
            },
            "files": {path.name: sha256_file(path) for path in generated_files},
            "rdkit_version": rdBase.rdkitVersion,
            "note": (
                "Topology is built only from pinned CHARMM36 carbohydrate RTP residues and "
                "reviewed standard PRES transformations. Source coordinates do not replace "
                "the force-field charge/type provenance."
            ),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    choices = sorted(set(TARGETS) | set(TARGET_BY_CLEAN))
    parser.add_argument("--molecule-name", required=True, choices=choices)
    parser.add_argument("--sdf", default="", help="Override the target's pinned CASPER source-class SDF")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--source-ff", default=str(DEFAULT_SOURCE_FF))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--gmx", default=DEFAULT_GMX)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.seed <= 2_147_483_647:
        raise ValueError("--seed must be between 0 and 2147483647")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive")
    output = build(args)
    target = target_from_name(args.molecule_name)
    print(
        json.dumps(
            {
                "molecule_name": target.raw_name,
                "clean_name": target.clean_name,
                "production_ready": True,
                "scope": "solute topology only",
                "requires_neutralization": target.expected_charge != 0,
                "output_dir": str(output),
                "metadata": str(output / "charmm36_standard_input_metadata.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
