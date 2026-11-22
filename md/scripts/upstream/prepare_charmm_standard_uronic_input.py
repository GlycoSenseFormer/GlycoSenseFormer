#!/usr/bin/env python3
"""Build fail-closed CHARMM36 topologies for six supported hexuronic acids.

This is intentionally separate from ``prepare_charmm_exact_input.py``.  It
does not infer CHARMM atom types or charges.  Instead, it maps the source
CASPER molecular graph to an explicitly allowlisted CHARMM36 carbohydrate RTP
residue and lets ``gmx pdb2gmx`` generate the topology from that residue.

The generated topology represents a deprotonated hexuronic acid with charge
-1.  It is production-ready as a *solute topology*, but must be solvated and
neutralized with a counterion before PME energy minimization or MD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DEFAULT_SDF = ROOT / "data" / "dataset" / "casper_mono.sdf"
DEFAULT_SOURCE_FF = (
    MD_ROOT
    / "forcefield_assets"
    / "charmm36"
    / "charmm36-feb2026_cgenff-5.0.ff"
)
DEFAULT_RUN_ROOT = Path(
    os.environ.get("MD_RUN_ROOT", MD_ROOT / "md_runs_all_remaining_v13")
)
DEFAULT_GMX = os.environ.get(
    "GMX",
    "/soft/gromacs-2026.1/bin/gmx"
    if Path("/soft/gromacs-2026.1/bin/gmx").exists()
    else "gmx",
)
OUTPUT_SUBDIR = "charmm36_standard_topology_dryrun"
LOCAL_FORCEFIELD_NAME = "charmm36_standard.ff"
EXPECTED_FORMULA = "C6H10O7"
EXPECTED_SOLUTE_CHARGE = -1.0
EXPECTED_HEAVY_NAMES = {
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O61",
    "O62",
}


@dataclass(frozen=True)
class Target:
    clean_name: str
    rtp_residue: str
    expected_isomeric_smiles: str


# Canonical isomeric SMILES are pinned to the source CASPER records.  A source
# record with changed stereochemistry therefore fails before any files are
# promoted into the final output directory.
ALLOWLIST: dict[str, Target] = {
    "bDManA": Target(
        clean_name="bDManA",
        rtp_residue="BMANA",
        expected_isomeric_smiles=(
            "O=C(O)[C@H]1O[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O"
        ),
    ),
    "bDGlcA": Target(
        clean_name="bDGlcA",
        rtp_residue="BGLCA",
        expected_isomeric_smiles=(
            "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        ),
    ),
    "bDGalA": Target(
        clean_name="bDGalA",
        rtp_residue="BGALA",
        expected_isomeric_smiles=(
            "O=C(O)[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O"
        ),
    ),
    "aDManA": Target(
        clean_name="aDManA",
        rtp_residue="AMANA",
        expected_isomeric_smiles=(
            "O=C(O)[C@H]1O[C@H](O)[C@@H](O)[C@@H](O)[C@@H]1O"
        ),
    ),
    "aDGlcA": Target(
        clean_name="aDGlcA",
        rtp_residue="AGLCA",
        expected_isomeric_smiles=(
            "O=C(O)[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
        ),
    ),
    "aDGalA": Target(
        clean_name="aDGalA",
        rtp_residue="AGALA",
        expected_isomeric_smiles=(
            "O=C(O)[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@H]1O"
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> dict[str, object]:
    """Return a deterministic content hash for a real-file directory tree."""

    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for candidate in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if candidate.is_symlink():
            raise ValueError(f"Force-field directory must not contain symlinks: {candidate}")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(path).as_posix()
        size = candidate.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        file_count += 1
        total_bytes += size
    if file_count == 0:
        raise ValueError(f"Force-field directory is empty: {path}")
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def tail(text: str, lines: int = 80) -> str:
    return "\n".join(text.splitlines()[-lines:])


def resolve_executable(command: str) -> str:
    candidate = Path(command)
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise FileNotFoundError(f"GROMACS executable is not executable: {candidate}")
        return str(candidate.resolve())
    found = shutil.which(command)
    if found is None:
        raise FileNotFoundError(f"Could not find executable {command!r} on PATH")
    return str(Path(found).resolve())


def find_source_molecule(sdf_path: Path, molecule_name: str) -> Chem.Mol:
    matches: list[Chem.Mol] = []
    for molecule in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if molecule is None or not molecule.HasProp("_Name"):
            continue
        if molecule.GetProp("_Name") == molecule_name:
            matches.append(molecule)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {molecule_name!r} record in {sdf_path}; "
            f"found {len(matches)}"
        )
    molecule = Chem.Mol(matches[0])
    Chem.SanitizeMol(molecule)
    return molecule


def validate_source_identity(molecule: Chem.Mol, target: Target) -> dict[str, object]:
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    formula = rdMolDescriptors.CalcMolFormula(molecule)
    heavy_count = sum(1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)
    if smiles != target.expected_isomeric_smiles:
        raise ValueError(
            "Source stereochemistry/identity does not match the allowlist: "
            f"observed {smiles!r}, expected {target.expected_isomeric_smiles!r}"
        )
    if formula != EXPECTED_FORMULA:
        raise ValueError(f"Unexpected source formula {formula!r}; expected {EXPECTED_FORMULA!r}")
    if heavy_count != len(EXPECTED_HEAVY_NAMES):
        raise ValueError(
            f"Unexpected heavy-atom count {heavy_count}; expected {len(EXPECTED_HEAVY_NAMES)}"
        )
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


def embed_mmff(molecule: Chem.Mol, seed: int) -> tuple[Chem.Mol, dict[str, object]]:
    work = Chem.Mol(molecule)
    work.RemoveAllConformers()
    work = Chem.AddHs(work, addCoords=False)

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.enforceChirality = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    embed_status = AllChem.EmbedMolecule(work, params)
    used_random_coordinates = False
    if embed_status != 0:
        params.useRandomCoords = True
        params.randomSeed = seed + 1009
        embed_status = AllChem.EmbedMolecule(work, params)
        used_random_coordinates = True
    if embed_status != 0:
        raise RuntimeError(f"ETKDGv3 embedding failed with status {embed_status}")
    if not AllChem.MMFFHasAllMoleculeParams(work):
        raise RuntimeError("MMFF94 parameters are incomplete; UFF fallback is intentionally disabled")

    optimize_status = AllChem.MMFFOptimizeMolecule(
        work, maxIters=1000, mmffVariant="MMFF94"
    )
    if optimize_status != 0:
        raise RuntimeError(
            "MMFF94 optimization did not converge within 1000 iterations "
            f"(status {optimize_status})"
        )
    properties = AllChem.MMFFGetMoleculeProperties(work, mmffVariant="MMFF94")
    if properties is None:
        raise RuntimeError("Could not construct MMFF94 properties after optimization")
    force_field = AllChem.MMFFGetMoleculeForceField(work, properties)
    if force_field is None:
        raise RuntimeError("Could not construct MMFF94 force field after optimization")
    energy_kcal_mol = float(force_field.CalcEnergy())
    if not math.isfinite(energy_kcal_mol):
        raise RuntimeError("MMFF94 returned a non-finite energy")

    conformer = work.GetConformer()
    coordinates = [conformer.GetAtomPosition(index) for index in range(work.GetNumAtoms())]
    if not all(
        math.isfinite(value)
        for point in coordinates
        for value in (point.x, point.y, point.z)
    ):
        raise RuntimeError("Embedded coordinates contain a non-finite value")
    z_span = max(point.z for point in coordinates) - min(point.z for point in coordinates)
    if z_span < 0.05:
        raise RuntimeError(f"Embedded conformer is effectively planar (z span {z_span:.6f} A)")

    return work, {
        "method": "ETKDGv3 followed by MMFF94",
        "seed": seed,
        "embed_status": embed_status,
        "used_random_coordinates": used_random_coordinates,
        "optimization_status": optimize_status,
        "energy_kcal_mol": energy_kcal_mol,
        "z_span_angstrom": float(z_span),
    }


def bond_type(molecule: Chem.Mol, first: Chem.Atom, second: Chem.Atom) -> Chem.BondType:
    bond = molecule.GetBondBetweenAtoms(first.GetIdx(), second.GetIdx())
    if bond is None:
        raise ValueError(f"Atoms {first.GetIdx()} and {second.GetIdx()} are not bonded")
    return bond.GetBondType()


def derive_standard_atom_names(molecule: Chem.Mol) -> dict[int, str]:
    """Map a validated hexuronic-acid graph to CHARMM C/O atom names."""

    Chem.GetSymmSSSR(molecule)
    ring_info = molecule.GetRingInfo()
    carboxyl_candidates: list[tuple[Chem.Atom, Chem.Atom, Chem.Atom, Chem.Atom]] = []
    for carbon in molecule.GetAtoms():
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        oxygen_neighbors = [atom for atom in carbon.GetNeighbors() if atom.GetAtomicNum() == 8]
        carbon_neighbors = [atom for atom in carbon.GetNeighbors() if atom.GetAtomicNum() == 6]
        double_oxygen = [
            atom
            for atom in oxygen_neighbors
            if bond_type(molecule, carbon, atom) == Chem.BondType.DOUBLE
        ]
        single_oxygen = [
            atom
            for atom in oxygen_neighbors
            if bond_type(molecule, carbon, atom) == Chem.BondType.SINGLE
        ]
        if (
            len(oxygen_neighbors) == 2
            and len(carbon_neighbors) == 1
            and len(double_oxygen) == 1
            and len(single_oxygen) == 1
        ):
            carboxyl_candidates.append(
                (carbon, carbon_neighbors[0], double_oxygen[0], single_oxygen[0])
            )
    if len(carboxyl_candidates) != 1:
        raise ValueError(
            f"Expected one carboxyl group; found {len(carboxyl_candidates)} candidates"
        )
    c6, c5, o61, o62 = carboxyl_candidates[0]
    if not c5.IsInRing() or c5.GetAtomicNum() != 6:
        raise ValueError("The carboxyl carbon is not attached to a ring carbon C5")

    rings = [
        tuple(ring)
        for ring in ring_info.AtomRings()
        if c5.GetIdx() in ring
        and len(ring) == 6
        and sum(molecule.GetAtomWithIdx(index).GetAtomicNum() == 8 for index in ring) == 1
        and sum(molecule.GetAtomWithIdx(index).GetAtomicNum() == 6 for index in ring) == 5
    ]
    if len(rings) != 1:
        raise ValueError(f"Expected one six-membered 5C/1O ring; found {len(rings)}")
    ring_indices = set(rings[0])
    o5_candidates = [
        molecule.GetAtomWithIdx(index)
        for index in ring_indices
        if molecule.GetAtomWithIdx(index).GetAtomicNum() == 8
    ]
    if len(o5_candidates) != 1:
        raise ValueError("Could not identify a unique ring oxygen O5")
    o5 = o5_candidates[0]
    c1_candidates = [
        atom
        for atom in o5.GetNeighbors()
        if atom.GetIdx() in ring_indices and atom.GetIdx() != c5.GetIdx()
    ]
    if len(c1_candidates) != 1 or c1_candidates[0].GetAtomicNum() != 6:
        raise ValueError("Could not identify a unique anomeric carbon C1")
    c1 = c1_candidates[0]

    ring_carbons: list[Chem.Atom] = [c1]
    previous = o5
    current = c1
    for _ in range(3):
        next_candidates = [
            atom
            for atom in current.GetNeighbors()
            if atom.GetIdx() in ring_indices
            and atom.GetIdx() != previous.GetIdx()
            and atom.GetAtomicNum() == 6
        ]
        if len(next_candidates) != 1:
            raise ValueError(
                f"Ambiguous ring traversal after atom {current.GetIdx()}: "
                f"{[atom.GetIdx() for atom in next_candidates]}"
            )
        next_atom = next_candidates[0]
        ring_carbons.append(next_atom)
        previous, current = current, next_atom
    c1, c2, c3, c4 = ring_carbons
    if c5.GetIdx() not in {
        atom.GetIdx() for atom in c4.GetNeighbors() if atom.GetIdx() in ring_indices
    }:
        raise ValueError("Ring traversal C1-C2-C3-C4 does not terminate at C5")

    names: dict[int, str] = {
        c1.GetIdx(): "C1",
        c2.GetIdx(): "C2",
        c3.GetIdx(): "C3",
        c4.GetIdx(): "C4",
        c5.GetIdx(): "C5",
        c6.GetIdx(): "C6",
        o5.GetIdx(): "O5",
        o61.GetIdx(): "O61",
        o62.GetIdx(): "O62",
    }
    for position, carbon in enumerate((c1, c2, c3, c4), start=1):
        oxygen_candidates = [
            atom
            for atom in carbon.GetNeighbors()
            if atom.GetAtomicNum() == 8 and atom.GetIdx() not in ring_indices
        ]
        if len(oxygen_candidates) != 1:
            raise ValueError(
                f"Expected one exocyclic oxygen on C{position}; "
                f"found {len(oxygen_candidates)}"
            )
        names[oxygen_candidates[0].GetIdx()] = f"O{position}"

    heavy_indices = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1}
    if set(names) != heavy_indices:
        missing = sorted(heavy_indices - set(names))
        extra = sorted(set(names) - heavy_indices)
        raise ValueError(f"Atom-name mapping is incomplete; missing={missing}, extra={extra}")
    if set(names.values()) != EXPECTED_HEAVY_NAMES:
        raise ValueError(
            f"Mapped names {sorted(names.values())} do not match expected "
            f"{sorted(EXPECTED_HEAVY_NAMES)}"
        )
    return names


def read_rtp_atoms(source_ff: Path, residue: str) -> list[tuple[str, str, float]]:
    rtp_path = source_ff / "carb.rtp"
    text = rtp_path.read_text(encoding="utf-8")
    match = re.search(
        rf"^\[\s*{re.escape(residue)}\s*\]\s*$\n(.*?)(?=^\[\s*\S+\s*\]\s*$|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"RTP residue {residue!r} is absent from {rtp_path}")
    atom_section = re.search(
        r"^\s+\[\s*atoms\s*\]\s*$\n(.*?)(?=^\s+\[|\Z)",
        match.group(1),
        flags=re.MULTILINE | re.DOTALL,
    )
    if atom_section is None:
        raise ValueError(f"RTP residue {residue!r} has no [ atoms ] section")
    atoms: list[tuple[str, str, float]] = []
    for line in atom_section.group(1).splitlines():
        stripped = line.split(";", 1)[0].strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed RTP atom line for {residue}: {line!r}")
        atoms.append((fields[0], fields[1], float(fields[2])))
    if not atoms:
        raise ValueError(f"RTP residue {residue!r} has no atoms")
    return atoms


def validate_forcefield(source_ff: Path, residue: str) -> dict[str, object]:
    if source_ff.name != "charmm36-feb2026_cgenff-5.0.ff":
        raise ValueError(
            "Refusing an unpinned force field directory; expected "
            f"'charmm36-feb2026_cgenff-5.0.ff', got {source_ff.name!r}"
        )
    required = [
        source_ff / "forcefield.itp",
        source_ff / "ffbonded.itp",
        source_ff / "ffnonbonded.itp",
        source_ff / "carb.rtp",
        source_ff / "carb.hdb",
        source_ff / "tip3p.itp",
        source_ff / "ions.itp",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete CHARMM36 force field; missing {missing}")
    atoms = read_rtp_atoms(source_ff, residue)
    charge = sum(charge for _, _, charge in atoms)
    heavy_names = {name for name, _, _ in atoms if not name.upper().startswith("H")}
    if abs(charge - EXPECTED_SOLUTE_CHARGE) > 1e-6:
        raise ValueError(
            f"RTP residue {residue} has charge {charge:+.6f}, "
            f"expected {EXPECTED_SOLUTE_CHARGE:+.6f}"
        )
    if heavy_names != EXPECTED_HEAVY_NAMES:
        raise ValueError(
            f"RTP residue {residue} heavy names differ from the pinned schema: "
            f"{sorted(heavy_names)}"
        )
    hdb_text = (source_ff / "carb.hdb").read_text(encoding="utf-8")
    if re.search(rf"^{re.escape(residue)}\s+\d+\s*$", hdb_text, re.MULTILINE) is None:
        raise ValueError(f"Hydrogen database has no entry for RTP residue {residue}")
    return {
        "directory": str(source_ff),
        "release": "CHARMM36 February 2026, CGenFF 5.0 GROMACS port",
        "rtp_residue": residue,
        "rtp_atom_count": len(atoms),
        "rtp_charge_e": charge,
        "sha256": {path.name: sha256_file(path) for path in required},
    }


def copy_pinned_forcefield(source_ff: Path, destination: Path) -> dict[str, object]:
    if destination.name != LOCAL_FORCEFIELD_NAME:
        raise ValueError(
            f"Local force-field directory must be named {LOCAL_FORCEFIELD_NAME!r}; "
            f"got {destination.name!r}"
        )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite copied force field: {destination}")
    source_tree = sha256_directory(source_ff)
    shutil.copytree(source_ff, destination, copy_function=shutil.copy2, symlinks=False)
    copied_tree = sha256_directory(destination)
    if copied_tree != source_tree:
        raise RuntimeError(
            "Copied force-field tree does not match the pinned source: "
            f"source={source_tree}, copied={copied_tree}"
        )
    return {
        "source_directory": str(source_ff),
        "source_tree": source_tree,
        "copied_directory": destination.name,
        "copied_tree": copied_tree,
        "content_identical": True,
    }


def rewrite_forcefield_includes(
    topology: Path, source_ff: Path, local_forcefield_name: str
) -> dict[str, object]:
    """Rewrite only includes that resolve into the pinned source force field."""

    old_name = source_ff.name
    original = topology.read_text(encoding="utf-8")
    rewritten_lines: list[str] = []
    replacements: list[dict[str, str]] = []
    include_pattern = re.compile(
        r"^(?P<prefix>\s*#include\s+)[\"<](?P<target>[^\">]+)[\">](?P<suffix>.*)$",
        flags=re.MULTILINE,
    )
    for line in original.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        match = include_pattern.match(content)
        if match is None:
            rewritten_lines.append(line)
            continue

        target = match.group("target")
        relative: Path | None = None
        target_path = Path(target)
        if target_path.is_absolute():
            try:
                relative = target_path.relative_to(source_ff)
            except ValueError:
                relative = None
        else:
            marker = old_name + "/"
            if target.startswith(marker):
                relative = Path(target[len(marker) :])
            elif marker in target:
                prefix, remainder = target.split(marker, 1)
                if prefix and not prefix.endswith("/"):
                    raise ValueError(f"Ambiguous force-field include path: {target!r}")
                relative = Path(remainder)

        if relative is None:
            rewritten_lines.append(line)
            continue
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"Unsafe force-field include path: {target!r}")
        source_target = source_ff / relative
        if not source_target.is_file():
            raise FileNotFoundError(
                f"Topology include does not exist in pinned force field: {source_target}"
            )
        new_target = f"{local_forcefield_name}/{relative.as_posix()}"
        rewritten_lines.append(
            f'{match.group("prefix")}"{new_target}"{match.group("suffix")}{newline}'
        )
        replacements.append({"source": target, "local": new_target})

    rewritten = "".join(rewritten_lines)
    required_names = {"forcefield.itp", "tip3p.itp", "ions.itp"}
    rewritten_names = {Path(row["local"]).name for row in replacements}
    if not replacements or not required_names.issubset(rewritten_names):
        raise RuntimeError(
            "Did not rewrite all required CHARMM36 includes: "
            f"rewritten={sorted(rewritten_names)}, required={sorted(required_names)}"
        )
    if old_name in rewritten or str(source_ff) in rewritten:
        raise RuntimeError(
            f"Pinned source force-field path/name remains after include rewrite: {old_name}"
        )
    local_prefix = local_forcefield_name + "/"
    for match in include_pattern.finditer(rewritten):
        target = match.group("target")
        if target.startswith(local_prefix):
            relative = Path(target[len(local_prefix) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe rewritten local include: {target!r}")
    topology.write_text(rewritten, encoding="utf-8")
    return {
        "old_directory_name": old_name,
        "local_directory_name": local_forcefield_name,
        "replacement_count": len(replacements),
        "rewritten_includes": replacements,
        "old_name_absent_after_rewrite": True,
    }


def write_heavy_gro(
    molecule: Chem.Mol,
    source_heavy_indices: list[int],
    atom_names: dict[int, str],
    residue: str,
    molecule_name: str,
    output: Path,
) -> None:
    conformer = molecule.GetConformer()
    lines = [
        f"{molecule_name} MMFF94 3D mapped to CHARMM36 {residue}",
        f"{len(source_heavy_indices):5d}",
    ]
    for serial, atom_index in enumerate(source_heavy_indices, start=1):
        point = conformer.GetAtomPosition(atom_index)
        name = atom_names[atom_index]
        lines.append(
            f"{1:5d}{residue:<5s}{name:>5s}{serial:5d}"
            f"{point.x / 10.0:8.3f}{point.y / 10.0:8.3f}{point.z / 10.0:8.3f}"
        )
    lines.append("   8.00000   8.00000   8.00000")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_logged(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    log_path.write_text(
        "$ " + shlex.join(command) + "\n\n" + process.stdout,
        encoding="utf-8",
    )
    return process


def parse_processed_gro(path: Path, expected_residue: str) -> tuple[list[str], float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Processed GRO is too short: {path}")
    atom_count = int(lines[1].strip())
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"Processed GRO atom count is truncated: {path}")
    heavy_names: list[str] = []
    points: list[tuple[float, float, float]] = []
    for line in atom_lines:
        residue = line[5:10].strip()
        name = line[10:15].strip()
        if residue != expected_residue:
            raise ValueError(
                f"Processed GRO contains residue {residue!r}, expected {expected_residue!r}"
            )
        point = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("Processed GRO contains a non-finite coordinate")
        points.append(point)
        if not name.upper().startswith("H"):
            heavy_names.append(name)
    if set(heavy_names) != EXPECTED_HEAVY_NAMES or len(heavy_names) != len(
        EXPECTED_HEAVY_NAMES
    ):
        raise ValueError(f"Unexpected processed heavy-atom names/order: {heavy_names}")

    minimum_distance = math.inf
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            distance = math.sqrt(
                sum((points[first][axis] - points[second][axis]) ** 2 for axis in range(3))
            )
            minimum_distance = min(minimum_distance, distance)
    if minimum_distance < 0.05:
        raise ValueError(
            f"Processed GRO has an atomic overlap/near-overlap: {minimum_distance:.6f} nm"
        )
    return heavy_names, minimum_distance


def parse_topology_charge(path: Path) -> tuple[float, int]:
    in_atoms = False
    charge = 0.0
    atom_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        section = re.match(r"^\s*\[\s*([^]]+)\s*\]", line)
        if section:
            section_name = section.group(1).strip().lower()
            if in_atoms and section_name != "atoms":
                break
            in_atoms = section_name == "atoms"
            continue
        if not in_atoms:
            continue
        stripped = line.split(";", 1)[0].strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 8:
            raise ValueError(f"Malformed topology [ atoms ] line: {line!r}")
        charge += float(fields[6])
        atom_count += 1
    if atom_count == 0:
        raise ValueError(f"Could not parse a topology [ atoms ] section from {path}")
    return charge, atom_count


def preflight_mdp(coulomb_type: str) -> str:
    return f"""\
integrator               = md
dt                       = 0.001
nsteps                   = 0
continuation             = yes
constraints              = h-bonds
constraint-algorithm     = lincs
cutoff-scheme            = Verlet
nstlist                  = 20
rlist                    = 1.2
coulombtype              = {coulomb_type}
rcoulomb                 = 1.2
vdwtype                  = Cut-off
vdw-modifier             = Force-switch
rvdw-switch              = 1.0
rvdw                     = 1.2
DispCorr                 = no
tcoupl                   = no
pcoupl                   = no
gen-vel                  = no
pbc                      = xyz
nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstenergy                = 0
nstlog                   = 0
"""


def count_diagnostics(output: str, label: str) -> int:
    return len(re.findall(rf"^{label}\s+\d+\s+\[", output, flags=re.MULTILINE))


def run_preflight(
    gmx: str,
    cwd: Path,
    topology: Path,
    coordinates: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> dict[str, object]:
    cutoff_mdp = cwd / "strict_solute_cutoff.mdp"
    cutoff_tpr = cwd / "strict_solute_cutoff.tpr"
    cutoff_log = cwd / "strict_solute_cutoff_grompp.log"
    cutoff_mdp.write_text(preflight_mdp("Cut-off"), encoding="utf-8")
    cutoff = run_logged(
        [
            gmx,
            "grompp",
            "-f",
            cutoff_mdp.name,
            "-c",
            coordinates.name,
            "-p",
            topology.name,
            "-o",
            cutoff_tpr.name,
            "-maxwarn",
            "0",
        ],
        cwd,
        environment,
        cutoff_log,
        timeout_seconds,
    )
    cutoff_warnings = count_diagnostics(cutoff.stdout, "WARNING")
    cutoff_errors = count_diagnostics(cutoff.stdout, "ERROR")
    if (
        cutoff.returncode != 0
        or cutoff_warnings != 0
        or cutoff_errors != 0
        or "No default" in cutoff.stdout
        or not cutoff_tpr.is_file()
    ):
        raise RuntimeError(
            "Strict cut-off solute grompp failed; no warnings or missing parameters "
            f"are allowed.\n{tail(cutoff.stdout)}"
        )

    pme_mdp = cwd / "strict_solute_pme.mdp"
    pme_tpr = cwd / "strict_solute_pme.tpr"
    pme_log = cwd / "strict_solute_pme_grompp.log"
    pme_mdp.write_text(preflight_mdp("PME"), encoding="utf-8")
    pme = run_logged(
        [
            gmx,
            "grompp",
            "-f",
            pme_mdp.name,
            "-c",
            coordinates.name,
            "-p",
            topology.name,
            "-o",
            pme_tpr.name,
            "-maxwarn",
            "0",
        ],
        cwd,
        environment,
        pme_log,
        timeout_seconds,
    )
    pme_warnings = count_diagnostics(pme.stdout, "WARNING")
    pme_errors = count_diagnostics(pme.stdout, "ERROR")
    expected_charge_warning = (
        "system with net charge" in pme.stdout
        and "uniform background charge" in pme.stdout
        and "Too many warnings (1)" in pme.stdout
    )
    if (
        pme.returncode != 1
        or pme_warnings != 1
        or pme_errors != 0
        or not expected_charge_warning
        or "No default" in pme.stdout
    ):
        raise RuntimeError(
            "Strict PME solute grompp did not fail solely for the expected net-charge "
            f"warning.\n{tail(pme.stdout)}"
        )

    return {
        "strict_cutoff": {
            "returncode": cutoff.returncode,
            "warnings": cutoff_warnings,
            "errors": cutoff_errors,
            "missing_default_parameters": False,
            "tpr": cutoff_tpr.name,
            "log": cutoff_log.name,
        },
        "strict_pme_solute": {
            "returncode": pme.returncode,
            "warnings": pme_warnings,
            "errors": pme_errors,
            "only_expected_net_charge_warning": True,
            "missing_default_parameters": False,
            "tpr_created": pme_tpr.is_file(),
            "log": pme_log.name,
        },
    }


def gromacs_version(gmx: str, environment: dict[str, str], timeout_seconds: int) -> str:
    process = subprocess.run(
        [gmx, "--version"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Could not query GROMACS version:\n{tail(process.stdout)}")
    match = re.search(r"^GROMACS version:\s*(.+)$", process.stdout, flags=re.MULTILINE)
    return match.group(1).strip() if match else process.stdout.splitlines()[0].strip()


def build(args: argparse.Namespace) -> Path:
    target = ALLOWLIST[args.molecule_name]
    sdf_path = Path(args.sdf).resolve()
    source_ff = Path(args.source_ff).resolve()
    run_root = Path(args.run_root).resolve()
    gmx = resolve_executable(args.gmx)

    if not sdf_path.is_file():
        raise FileNotFoundError(f"Source SDF does not exist: {sdf_path}")
    forcefield_metadata = validate_forcefield(source_ff, target.rtp_residue)
    source = find_source_molecule(sdf_path, args.molecule_name)
    source_metadata = validate_source_identity(source, target)
    source_heavy_indices = [
        atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    optimized, mmff_metadata = embed_mmff(source, args.seed)
    atom_names = derive_standard_atom_names(optimized)

    molecule_dir = run_root / target.clean_name
    output_dir = molecule_dir / OUTPUT_SUBDIR
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    molecule_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    staging = molecule_dir / f".{OUTPUT_SUBDIR}.tmp-{os.getpid()}-{timestamp}"
    staging.mkdir(mode=0o755)

    stem = args.molecule_name.replace("/", "_")
    input_gro = staging / f"{stem}_CHARMM36_STANDARD_mmff3d_heavy.gro"
    processed_gro = staging / f"{stem}_CHARMM36_STANDARD_processed.gro"
    topology = staging / f"{stem}_CHARMM36_STANDARD.top"
    local_forcefield = staging / LOCAL_FORCEFIELD_NAME
    mapping_path = staging / "sdf_to_charmm_frame_heavy_index_map.json"
    metadata_path = staging / "charmm36_standard_input_metadata.json"
    pdb2gmx_log = staging / "pdb2gmx.log"

    pdb2gmx_environment = os.environ.copy()
    pdb2gmx_environment["GMXLIB"] = str(source_ff.parent)
    self_contained_environment = os.environ.copy()
    self_contained_environment.pop("GMXLIB", None)

    try:
        write_heavy_gro(
            optimized,
            source_heavy_indices,
            atom_names,
            target.rtp_residue,
            args.molecule_name,
            input_gro,
        )
        pdb2gmx_command = [
            gmx,
            "pdb2gmx",
            "-f",
            input_gro.name,
            "-o",
            processed_gro.name,
            "-p",
            topology.name,
            "-ff",
            source_ff.name.removesuffix(".ff"),
            "-water",
            "tip3p",
            "-ignh",
        ]
        pdb2gmx = run_logged(
            pdb2gmx_command,
            staging,
            pdb2gmx_environment,
            pdb2gmx_log,
            args.timeout_seconds,
        )
        if pdb2gmx.returncode != 0 or not processed_gro.is_file() or not topology.is_file():
            raise RuntimeError(f"pdb2gmx failed:\n{tail(pdb2gmx.stdout)}")
        if "Total charge -1.000 e" not in pdb2gmx.stdout:
            raise RuntimeError(
                "pdb2gmx did not report the pinned -1.000 e residue charge:\n"
                + tail(pdb2gmx.stdout)
            )

        copied_forcefield = copy_pinned_forcefield(source_ff, local_forcefield)
        include_rewrite = rewrite_forcefield_includes(
            topology, source_ff, LOCAL_FORCEFIELD_NAME
        )

        processed_heavy_names, minimum_distance = parse_processed_gro(
            processed_gro, target.rtp_residue
        )
        frame_index_by_name = {
            name: index for index, name in enumerate(processed_heavy_names)
        }
        source_heavy_names = [atom_names[index] for index in source_heavy_indices]
        mapping = [frame_index_by_name[name] for name in source_heavy_names]
        if len(set(mapping)) != len(mapping) or sorted(mapping) != list(range(len(mapping))):
            raise RuntimeError(f"Generated heavy-atom mapping is not bijective: {mapping}")
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

        topology_charge, topology_atom_count = parse_topology_charge(topology)
        if abs(topology_charge - EXPECTED_SOLUTE_CHARGE) > 1e-6:
            raise RuntimeError(
                f"Generated topology charge {topology_charge:+.6f} is not "
                f"{EXPECTED_SOLUTE_CHARGE:+.6f}"
            )
        preflight = run_preflight(
            gmx,
            staging,
            topology,
            processed_gro,
            self_contained_environment,
            args.timeout_seconds,
        )
        post_preflight_forcefield_tree = sha256_directory(local_forcefield)
        if post_preflight_forcefield_tree != copied_forcefield["copied_tree"]:
            raise RuntimeError(
                "Local force-field copy changed during self-contained preflight: "
                f"before={copied_forcefield['copied_tree']}, "
                f"after={post_preflight_forcefield_tree}"
            )
        copied_forcefield["verified_after_preflight"] = True
        copied_forcefield["post_preflight_tree"] = post_preflight_forcefield_tree
        preflight["self_contained_forcefield"] = {
            "GMXLIB": "unset",
            "working_directory": ".",
            "local_directory": LOCAL_FORCEFIELD_NAME,
            "external_forcefield_lookup_required": False,
        }
        version = gromacs_version(
            gmx, self_contained_environment, args.timeout_seconds
        )

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
        missing_generated = [path.name for path in generated_files if not path.is_file()]
        if missing_generated:
            raise RuntimeError(f"Expected generated files are missing: {missing_generated}")

        metadata = {
            "schema_version": 2,
            "topology_policy": "charmm36_standard_uronic_residue_v1",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "production_ready": True,
            "production_ready_scope": "solute topology only",
            "water_md_started": False,
            "requires_solvation": True,
            "requires_neutralization": True,
            "required_counterions": {
                "charge_e": 1,
                "count": 1,
                "note": "Add one monovalent cation before strict PME EM/MD; add bulk salt separately if requested.",
            },
            "molecule_name": args.molecule_name,
            "clean_name": target.clean_name,
            "source_class": "mono",
            "generator": {
                "entrypoint": Path(__file__).name,
                "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
                "imported_helpers": {},
            },
            "allowlist_size": len(ALLOWLIST),
            "rtp_residue": target.rtp_residue,
            "expected_solute_charge_e": EXPECTED_SOLUTE_CHARGE,
            "observed_topology_charge_e": topology_charge,
            "topology_atom_count": topology_atom_count,
            "source": {
                "sdf": str(sdf_path),
                "sdf_sha256": sha256_file(sdf_path),
                **source_metadata,
            },
            "conformer": mmff_metadata,
            "forcefield": {
                **forcefield_metadata,
                "self_contained_copy": copied_forcefield,
                "topology_include_rewrite": include_rewrite,
            },
            "gromacs": {"executable": gmx, "version": version},
            "atom_mapping": {
                "file": mapping_path.name,
                "semantics": "zero-based SDF heavy-atom order to processed-frame heavy-atom order",
                "mapping": mapping,
                "sdf_heavy_atom_names": source_heavy_names,
                "processed_heavy_atom_names": processed_heavy_names,
                "source_atom_index_to_charmm_name": {
                    str(index): atom_names[index] for index in source_heavy_indices
                },
            },
            "geometry": {"minimum_all_atom_distance_nm": minimum_distance},
            "pdb2gmx": {"command": pdb2gmx_command, "log": pdb2gmx_log.name},
            "preflight": preflight,
            "files": {
                path.name: sha256_file(path)
                for path in generated_files
            },
            "rdkit_version": rdBase.rdkitVersion,
            "note": (
                "Built only from an explicitly allowlisted CHARMM36 carbohydrate RTP residue. "
                "No graph-based CHARMM atom-type or charge inference was used. The -1 solute "
                "must be neutralized after solvation before PME EM or MD."
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standard CHARMM36 topology for one allowlisted deprotonated "
            "D-hexuronic acid. Unsupported molecules fail closed."
        )
    )
    parser.add_argument(
        "--molecule-name",
        required=True,
        choices=sorted(ALLOWLIST),
        help="Exact CASPER molecule name; only the six listed choices are accepted.",
    )
    parser.add_argument("--sdf", default=str(DEFAULT_SDF))
    parser.add_argument(
        "--run-root",
        default=str(DEFAULT_RUN_ROOT),
        help="Output root (defaults to MD_RUN_ROOT or md_runs_all_remaining_v13).",
    )
    parser.add_argument("--source-ff", default=str(DEFAULT_SOURCE_FF))
    parser.add_argument("--gmx", default=DEFAULT_GMX)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed < 0 or args.seed > 2_147_483_647:
        raise ValueError("--seed must be between 0 and 2147483647")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive")
    output_dir = build(args)
    metadata = output_dir / "charmm36_standard_input_metadata.json"
    print(
        json.dumps(
            {
                "molecule_name": args.molecule_name,
                "production_ready": True,
                "scope": "solute topology only",
                "requires_neutralization": True,
                "output_dir": str(output_dir),
                "metadata": str(metadata),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
