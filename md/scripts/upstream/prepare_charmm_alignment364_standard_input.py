#!/usr/bin/env python3
"""Build the alignment-364 CHARMM subset from pinned official residue/PRES records.

This entrypoint is deliberately separate from the frozen CHARMM30 builder.  It
uses that reviewed builder's parsing, patch, topology, and GROMACS helpers, but
adds only the official February-2026 sulfate and deoxy PRES transformations
needed by the frozen alignment manifest.  Unsupported names fail closed; no
graph-derived atom typing, CGenFF analogy, or guessed charge is permitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import networkx as nx
from rdkit import Chem, rdBase

import prepare_charmm_standard_patch_input as reviewed


ROOT = Path(__file__).resolve().parents[2]
MD_ROOT = ROOT / "分子动力学模拟"
DATASET_ROOT = ROOT / "data" / "dataset"
DEFAULT_SOURCE_FF = reviewed.DEFAULT_SOURCE_FF
DEFAULT_GMX = reviewed.DEFAULT_GMX
DEFAULT_MANIFEST = (
    MD_ROOT
    / "forcefield_assets"
    / "charmm36"
    / "charmm36-feb2026_alignment364_patch_manifest.json"
)
DEFAULT_ALIGNMENT_MANIFEST = MD_ROOT / "md_alignment364_v1" / "alignment364_targets.csv"
DEFAULT_RUN_ROOT = MD_ROOT / "md_alignment364_v1" / "charmm"
OUTPUT_SUBDIR = reviewed.OUTPUT_SUBDIR
LOCAL_FORCEFIELD_NAME = reviewed.LOCAL_FORCEFIELD_NAME
LOCAL_RESIDUE = reviewed.LOCAL_RESIDUE
TOPOLOGY_POLICY = "charmm36_standard_alignment364_patch_expansion_v1"
APPROVED_MANIFEST_SHA256 = "20f29fe96ab93f7faa22258938e185ec1c02149f98a4d75c0ea5ab4c92423b94"
APPROVED_ALIGNMENT_MANIFEST_SHA256 = "c704c1470606d5ea371eaae3d30b947177c89bc534aed37bfd224e9cf06670b8"
APPROVED_SOURCE_FF_TREE_SHA256 = reviewed.APPROVED_SOURCE_FF_TREE_SHA256
APPROVED_UPSTREAM_COMMIT = "ed3bbdaa4eb0a0152e577480bb0a9d22cd9743c3"
APPROVED_UPSTREAM_FILES = {
    "charmm/toppar/top_all36_carb.rtf":
        "41749e71731d46d479b041ba8a0877477bb0e46e4264ea25fbee73299eb19a59",
    "charmm/toppar/stream/carb/toppar_all36_carb_imlab.str":
        "3a5852e14dfe4f5aca4c8c82428c038946da144772da3568a53c7bcbcbe26e35",
}


PatchSpec = reviewed.PatchSpec
Target = reviewed.Target
Assembly = reviewed.Assembly
RtpAtom = reviewed.RtpAtom


IDENTITIES: dict[str, tuple[str, str]] = {
    "aDGal2SOMe": (
        "CO[C@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1OS(=O)(=O)O",
        "C7H14O9S",
    ),
    "aDGal3SOMe": (
        "CO[C@H]1O[C@H](CO)[C@H](O)[C@H](OS(=O)(=O)O)[C@H]1O",
        "C7H14O9S",
    ),
    "aDGal4SOMe": (
        "CO[C@H]1O[C@H](CO)[C@H](OS(=O)(=O)O)[C@H](O)[C@H]1O",
        "C7H14O9S",
    ),
    "aDGal6SOMe": (
        "CO[C@H]1O[C@H](COS(=O)(=O)O)[C@H](O)[C@H](O)[C@H]1O",
        "C7H14O9S",
    ),
    "aDGlc6S": (
        "O=S(=O)(O)OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "C6H12O9S",
    ),
    "bDGlc6S": (
        "O=S(=O)(O)OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "C6H12O9S",
    ),
    "bDGlc2dOMe": (
        "CO[C@H]1C[C@@H](O)[C@H](O)[C@@H](CO)O1",
        "C7H14O5",
    ),
    "bDGlcp4dOMe": (
        "CO[C@@H]1O[C@H](CO)C[C@H](O)[C@H]1O",
        "C7H14O5",
    ),
    "aPar(1->3)[aDGal(1->2)]aDManOMe": (
        "CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O[C@H]2O[C@H](C)[C@@H](O)C[C@H]2O)[C@@H]1O[C@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O",
        "C19H34O14",
    ),
    "aTyv(1->3)[aDGal(1->2)]aDManOMe": (
        "CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O[C@H]2O[C@H](C)[C@@H](O)C[C@@H]2O)[C@@H]1O[C@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O",
        "C19H34O14",
    ),
    "aLFuc(1->2)bDGal(1->3)bD2dGlcOMe": (
        "CO[C@H]1C[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](CO)O1",
        "C19H34O14",
    ),
    "bLFuc(1->2)bDGal(1->3)bD2dGlcOMe": (
        "CO[C@H]1C[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O[C@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](CO)O1",
        "C19H34O14",
    ),
    "aDGalA(1->2)aLRha(1->4)bDGalAOMe": (
        "CO[C@@H]1O[C@H](C(=O)O)[C@H](O[C@@H]2O[C@@H](C)[C@H](O)[C@@H](O)[C@H]2O[C@H]2O[C@H](C(=O)O)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O",
        "C19H30O17",
    ),
}


def _identity(name: str) -> tuple[str, str]:
    return IDENTITIES[name]


def make_targets() -> dict[str, Target]:
    targets: dict[str, Target] = {}
    sulfate_specs = {
        "aDGal2SOMe": ("AGAL", "OMEA", "2BSUF", 2),
        "aDGal3SOMe": ("AGAL", "OMEA", "3BSUF", 3),
        "aDGal4SOMe": ("AGAL", "OMEA", "4ASUF", 4),
        "aDGal6SOMe": ("AGAL", "OMEA", "6TSUF", 6),
        "aDGlc6S": ("AGLC", "", "6TSUF", 6),
        "bDGlc6S": ("BGLC", "", "6TSUF", 6),
    }
    for name, (residue, methyl, sulfate, position) in sulfate_specs.items():
        smiles, formula = _identity(name)
        patches = []
        if methyl:
            patches.append(PatchSpec("o_methyl", methyl))
        patches.append(PatchSpec("o_sulfate", sulfate, position=position))
        targets[name] = Target(
            name, name, "mono", ((residue, ""),), tuple(patches), smiles, formula, -1
        )

    for name, patch, position in (
        ("bDGlc2dOMe", "2DEO", 2),
        ("bDGlcp4dOMe", "4DEO", 4),
    ):
        smiles, formula = _identity(name)
        targets[name] = Target(
            name,
            name,
            "mono",
            (("BGLC", ""),),
            (PatchSpec("o_methyl", "OMEB"), PatchSpec("deoxy", patch, position=position)),
            smiles,
            formula,
            0,
        )

    for raw_name, residue in (
        ("aPar(1->3)[aDGal(1->2)]aDManOMe", "APAR"),
        ("aTyv(1->3)[aDGal(1->2)]aDManOMe", "ATYV"),
    ):
        smiles, formula = _identity(raw_name)
        clean = raw_name.replace("->", "").replace("(", "").replace(")", "")
        targets[raw_name] = Target(
            raw_name,
            clean,
            "trimer",
            (("AMAN", "A"), ("AGAL", "B"), (residue, "C")),
            (
                PatchSpec("o_methyl", "OMEA", "A"),
                PatchSpec("link_1_n", "12aa", "A", "B", 2),
                PatchSpec("link_1_n", "13ab", "A", "C", 3),
            ),
            smiles,
            formula,
            0,
        )

    for raw_name, fucose, link in (
        ("aLFuc(1->2)bDGal(1->3)bD2dGlcOMe", "AFUC", "12ab"),
        ("bLFuc(1->2)bDGal(1->3)bD2dGlcOMe", "BFUC", "12bb"),
    ):
        smiles, formula = _identity(raw_name)
        clean = raw_name.replace("->", "").replace("(", "").replace(")", "")
        targets[raw_name] = Target(
            raw_name,
            clean,
            "trimer",
            (("BGLC", "A"), ("BGAL", "B"), (fucose, "C")),
            (
                PatchSpec("o_methyl", "OMEB", "A"),
                PatchSpec("deoxy", "2DEO", "A", position=2),
                PatchSpec("link_1_n", "13bb", "A", "B", 3),
                PatchSpec("link_1_n", link, "B", "C", 2),
            ),
            smiles,
            formula,
            0,
        )

    raw_name = "aDGalA(1->2)aLRha(1->4)bDGalAOMe"
    smiles, formula = _identity(raw_name)
    targets[raw_name] = Target(
        raw_name,
        "aDGalA12aLRha14bDGalAOMe",
        "trimer",
        (("BGALA", "A"), ("ARHM", "B"), ("AGALA", "C")),
        (
            PatchSpec("o_methyl", "OMEB", "A"),
            PatchSpec("link_1_n", "14aa", "A", "B", 4),
            PatchSpec("link_1_n", "12aa", "B", "C", 2),
        ),
        smiles,
        formula,
        -2,
    )
    if len(targets) != 13 or set(targets) != set(IDENTITIES):
        raise RuntimeError("Internal alignment364 CHARMM allowlist is inconsistent")
    return targets


TARGETS = make_targets()
TARGET_BY_CLEAN = {target.clean_name: target for target in TARGETS.values()}


def atom_element(name: str) -> str:
    stripped = name.lstrip("0123456789")
    if not stripped or stripped[0] not in "CHOS":
        raise ValueError(f"Cannot derive an approved element from RTP atom name {name!r}")
    return stripped[0]


def load_alignment_rows(path: Path = DEFAULT_ALIGNMENT_MANIFEST) -> dict[str, dict[str, str]]:
    path = path.resolve()
    if reviewed.sha256_file(path) != APPROVED_ALIGNMENT_MANIFEST_SHA256:
        raise ValueError("alignment364 target manifest is not the frozen reviewed object")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 364 or len({row["clean_name"] for row in rows}) != 364:
        raise ValueError("alignment364 manifest must contain 364 unique clean names")
    charmm_rows = {
        row["clean_name"]: row
        for row in rows
        if "charmm" in row["missing_forcefields"].split(";")
    }
    if len(charmm_rows) != 13 or set(charmm_rows) != set(TARGET_BY_CLEAN):
        raise ValueError("alignment364 CHARMM-missing set differs from the 13-name allowlist")
    for clean, row in charmm_rows.items():
        target = TARGET_BY_CLEAN[clean]
        if row["molecule_name"] != target.raw_name or row["source_class"] != target.source_class:
            raise ValueError(f"alignment364 identity mismatch for {clean}")
    return charmm_rows


def validate_manifest(path: Path, source_ff: Path) -> tuple[dict[str, object], dict[str, object]]:
    if reviewed.sha256_file(path) != APPROVED_MANIFEST_SHA256:
        raise ValueError("alignment364 CHARMM patch manifest is not the reviewed object")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported alignment364 CHARMM manifest schema")
    if manifest.get("gromacs_port") != source_ff.name:
        raise ValueError("alignment364 CHARMM manifest names a different GROMACS port")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("alignment364 CHARMM manifest lacks upstream provenance")
    if upstream.get("commit") != APPROVED_UPSTREAM_COMMIT or upstream.get("files") != APPROVED_UPSTREAM_FILES:
        raise ValueError("alignment364 CHARMM manifest does not pin the reviewed official sources")
    source_tree = reviewed.sha256_directory(source_ff)
    if source_tree.get("sha256") != APPROVED_SOURCE_FF_TREE_SHA256:
        raise ValueError("CHARMM source force-field tree differs from the reviewed February-2026 tree")
    return manifest, {
        "manifest": str(path),
        "manifest_sha256": APPROVED_MANIFEST_SHA256,
        "reviewed_upstream": upstream,
        "source_forcefield": str(source_ff),
        "source_forcefield_tree": source_tree,
    }


def _source_patch(template: dict[str, object], patch: PatchSpec) -> dict[str, object]:
    sources = template.get("source_patches")
    if not isinstance(sources, dict) or not isinstance(sources.get(patch.name), dict):
        raise ValueError(f"official PRES {patch.name!r} is absent from the reviewed template")
    record = sources[patch.name]
    if record.get("position") != patch.position:
        raise ValueError(f"official PRES {patch.name!r} is not approved at C{patch.position}")
    return record


def apply_o_sulfate(assembly: Assembly, patch: PatchSpec, manifest: dict[str, object]) -> None:
    template_name = "o_sulfate_primary" if patch.position == 6 else "o_sulfate_secondary"
    template = reviewed.manifest_template(manifest, template_name)
    source_record = _source_patch(template, patch)
    prefix = patch.unit1
    position = patch.position
    hydroxyl_h = reviewed.unit_atom(prefix, f"HO{position}")
    reviewed.remove_hdb_target(assembly, hydroxyl_h)
    reviewed.delete_atom(assembly, hydroxyl_h)
    carbon_type, carbon_charge = template["carbon"]
    oxygen_type, oxygen_charge = template["attached_oxygen"]
    reviewed.replace_atom(
        assembly, reviewed.unit_atom(prefix, f"C{position}"), str(carbon_type), float(carbon_charge)
    )
    reviewed.replace_atom(
        assembly, reviewed.unit_atom(prefix, f"O{position}"), str(oxygen_type), float(oxygen_charge)
    )
    group = reviewed.next_group(assembly)
    sulfur_type, sulfur_charge = template["sulfur"]
    terminal_type, terminal_charge = template["terminal_oxygen"]
    sulfur = reviewed.unit_atom(prefix, f"S{position}")
    reviewed.add_atom(assembly, sulfur, str(sulfur_type), float(sulfur_charge), group)
    terminal_names = []
    for suffix in (2, 3, 4):
        name = reviewed.unit_atom(prefix, f"OS{position}{suffix}")
        terminal_names.append(name)
        reviewed.add_atom(assembly, name, str(terminal_type), float(terminal_charge), group)
    reviewed.add_bond(assembly, reviewed.unit_atom(prefix, f"O{position}"), sulfur)
    for name in terminal_names:
        reviewed.add_bond(assembly, sulfur, name)
    assembly.applied_patches.append(
        {
            "kind": patch.kind,
            "source_patch": patch.name,
            "source_file": template["source_file"],
            "unit_prefix": prefix,
            "position": position,
            "geometry": source_record["geometry"],
            "manifest_template": template_name,
            "simulated_charge_delta_e": template["charge_delta_e"],
            "terminal_oxygens_are_officially_equivalent": True,
        }
    )


def apply_deoxy(assembly: Assembly, patch: PatchSpec, manifest: dict[str, object]) -> None:
    template_name = "deoxy_secondary"
    template = reviewed.manifest_template(manifest, template_name)
    source_record = _source_patch(template, patch)
    prefix = patch.unit1
    position = patch.position
    for hydrogen in (f"HO{position}", f"H{position}"):
        reviewed.remove_hdb_target(assembly, reviewed.unit_atom(prefix, hydrogen))
    for name in (f"O{position}", f"HO{position}", f"H{position}"):
        reviewed.delete_atom(assembly, reviewed.unit_atom(prefix, name))
    carbon_name = reviewed.unit_atom(prefix, f"C{position}")
    old_group = assembly.atoms[carbon_name].charge_group
    carbon_type, carbon_charge = template["carbon"]
    hydrogen_type, hydrogen_charge = template["hydrogen"]
    reviewed.replace_atom(assembly, carbon_name, str(carbon_type), float(carbon_charge))
    for index in (1, 2):
        hydrogen = reviewed.unit_atom(prefix, f"H{position}{index}")
        reviewed.add_atom(assembly, hydrogen, str(hydrogen_type), float(hydrogen_charge), old_group)
        reviewed.add_bond(assembly, carbon_name, hydrogen)
    neighbor_positions = source_record["neighbors"]
    if not isinstance(neighbor_positions, list) or len(neighbor_positions) != 2:
        raise ValueError(f"official deoxy PRES {patch.name} lacks two carbon neighbors")
    assembly.hdb.append(
        [
            "2",
            str(template["hdb_method"]),
            reviewed.unit_atom(prefix, f"H{position}"),
            carbon_name,
            reviewed.unit_atom(prefix, f"C{neighbor_positions[0]}"),
            reviewed.unit_atom(prefix, f"C{neighbor_positions[1]}"),
        ]
    )
    assembly.applied_patches.append(
        {
            "kind": patch.kind,
            "source_patch": patch.name,
            "source_file": template["source_file"],
            "unit_prefix": prefix,
            "position": position,
            "manifest_template": template_name,
            "simulated_charge_delta_e": template["charge_delta_e"],
        }
    )


def validate_assembly(assembly: Assembly, target: Target) -> dict[str, object]:
    if len(assembly.atoms) != len(set(assembly.atoms)):
        raise ValueError("expanded RTP atom names are not unique")
    if any(len(name) > 5 for name in assembly.atoms):
        raise ValueError("expanded RTP atom names exceed GRO width")
    for first, second in assembly.bonds:
        if first not in assembly.atoms or second not in assembly.atoms:
            raise ValueError(f"expanded RTP bond references a missing atom: {first}-{second}")
    for improper in assembly.impropers:
        if any(name not in assembly.atoms for name in improper):
            raise ValueError(f"expanded RTP improper references a missing atom: {improper}")
    for recipe in assembly.hdb:
        if any(name not in assembly.atoms for name in recipe[3:]):
            raise ValueError(f"expanded HDB recipe references a missing atom: {recipe}")
    expected_h = sorted(name for name in assembly.atoms if atom_element(name) == "H")
    generated_h = sorted(
        name for recipe in assembly.hdb for name in reviewed.expanded_hydrogen_names(recipe)
    )
    if expected_h != generated_h:
        raise ValueError(
            "expanded HDB does not generate exactly the RTP hydrogens: "
            f"missing={sorted(set(expected_h) - set(generated_h))}, "
            f"extra={sorted(set(generated_h) - set(expected_h))}"
        )
    charge = sum(atom.charge for atom in assembly.atoms.values())
    if abs(charge - target.expected_charge) > reviewed.CHARGE_TOLERANCE:
        raise ValueError(
            f"expanded official CHARMM charge {charge:+.6f} differs from expected "
            f"{target.expected_charge:+d} for {target.clean_name}"
        )
    elements: dict[str, int] = {}
    for name in assembly.atoms:
        element = atom_element(name)
        elements[element] = elements.get(element, 0) + 1
    return {
        "atom_count": len(assembly.atoms),
        "heavy_atom_count": sum(element != "H" for element in map(atom_element, assembly.atoms)),
        "hydrogen_count": len(expected_h),
        "bond_count": len(assembly.bonds),
        "improper_count": len(assembly.impropers),
        "hdb_recipe_count": len(assembly.hdb),
        "element_counts": elements,
        "charge_e": charge,
    }


def build_assembly(source_ff: Path, target: Target, manifest: dict[str, object]) -> Assembly:
    assembly = reviewed.base_assembly(source_ff, target)
    for patch in target.patches:
        if patch.kind == "o_methyl":
            reviewed.apply_o_methyl(assembly, patch, manifest)
        elif patch.kind == "link_1_n":
            reviewed.apply_link(assembly, patch, manifest)
        elif patch.kind == "o_sulfate":
            apply_o_sulfate(assembly, patch, manifest)
        elif patch.kind == "deoxy":
            apply_deoxy(assembly, patch, manifest)
        else:
            raise ValueError(f"unsupported alignment364 declarative patch {patch.kind!r}")
    validate_assembly(assembly, target)
    return assembly


def assembly_graph(assembly: Assembly) -> nx.Graph:
    graph = nx.Graph()
    for name in assembly.atoms:
        if atom_element(name) != "H":
            graph.add_node(name, element=atom_element(name))
    for first, second in assembly.bonds:
        if first in graph and second in graph:
            graph.add_edge(first, second)
    return graph


def sulfate_terminal_groups(graph: nx.Graph) -> list[frozenset[str]]:
    groups = []
    for name, data in graph.nodes(data=True):
        if data["element"] != "S":
            continue
        terminals = frozenset(
            neighbor
            for neighbor in graph.neighbors(name)
            if graph.nodes[neighbor]["element"] == "O" and graph.degree(neighbor) == 1
        )
        if len(terminals) != 3:
            raise ValueError(f"official sulfate {name} does not have three terminal oxygens")
        groups.append(terminals)
    return groups


def mappings_differ_only_by_sulfate_oxygen_permutations(
    first: dict[int, str], second: dict[int, str], groups: list[frozenset[str]]
) -> bool:
    group_by_name = {name: index for index, group in enumerate(groups) for name in group}
    for source_index, first_name in first.items():
        second_name = second[source_index]
        if first_name == second_name:
            continue
        if group_by_name.get(first_name) != group_by_name.get(second_name):
            return False
    return True


def map_source_graph(
    molecule: Chem.Mol, assembly: Assembly, target: Target
) -> tuple[dict[int, str], dict[str, object]]:
    source = reviewed.source_graph(molecule)
    expanded = assembly_graph(assembly)
    if source.number_of_nodes() != expanded.number_of_nodes() or source.number_of_edges() != expanded.number_of_edges():
        raise ValueError(
            "CASPER and official expanded RTP heavy graphs differ: "
            f"source={source.number_of_nodes()} atoms/{source.number_of_edges()} bonds, "
            f"RTP={expanded.number_of_nodes()} atoms/{expanded.number_of_edges()} bonds"
        )
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        source,
        expanded,
        node_match=lambda first, second: first["element"] == second["element"],
    )
    active_prefixes = frozenset(prefix for _residue, prefix in target.bases if prefix)
    candidates = []
    enumerated = 0
    for candidate in matcher.isomorphisms_iter():
        enumerated += 1
        if enumerated > 100_000:
            raise RuntimeError("alignment364 heavy-graph isomorphism is unexpectedly ambiguous")
        if reviewed.carbonyl_constraints(molecule, candidate, active_prefixes):
            candidates.append(dict(candidate))
    if not candidates:
        raise ValueError("no element- and carbonyl-consistent CASPER-to-official-RTP mapping exists")
    heavy_indices = sorted(source.nodes)
    candidates.sort(key=lambda row: tuple(row[index] for index in heavy_indices))
    terminal_groups = sulfate_terminal_groups(expanded)
    selected = candidates[0]
    equivalent_sulfate_automorphisms = len(candidates) > 1 and bool(terminal_groups)
    if len(candidates) > 1 and not all(
        mappings_differ_only_by_sulfate_oxygen_permutations(selected, candidate, terminal_groups)
        for candidate in candidates[1:]
    ):
        raise ValueError(
            "multiple non-equivalent CASPER-to-CHARMM mappings remain; refusing index-dependent selection"
        )
    return selected, {
        "graph_isomorphisms_enumerated": enumerated,
        "carbonyl_consistent_isomorphisms": len(candidates),
        "equivalent_official_sulfate_oxygen_automorphisms": equivalent_sulfate_automorphisms,
        "sulfate_terminal_oxygen_groups": [sorted(group) for group in terminal_groups],
        "selection": (
            "one of official equivalent sulfate-terminal-oxygen permutations"
            if equivalent_sulfate_automorphisms
            else "unique element- and carbonyl-bound mapping"
        ),
    }


def parse_processed_gro(path: Path, assembly: Assembly) -> tuple[list[str], float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"processed GRO is too short: {path}")
    count = int(lines[1].strip())
    atom_lines = lines[2 : 2 + count]
    if len(atom_lines) != count:
        raise ValueError("processed GRO atom list is truncated")
    names = []
    points = []
    for line in atom_lines:
        if line[5:10].strip() != LOCAL_RESIDUE:
            raise ValueError(f"processed GRO contains a non-{LOCAL_RESIDUE} residue")
        name = line[10:15].strip()
        if name not in assembly.atoms:
            raise ValueError(f"processed GRO contains unexpected atom {name!r}")
        names.append(name)
        point = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("processed GRO contains a non-finite coordinate")
        points.append(point)
    if len(names) != len(set(names)) or set(names) != set(assembly.atoms):
        raise ValueError("processed GRO atom names are not a bijection with the official assembly")
    minimum = min(
        math.sqrt(sum((points[first][axis] - points[second][axis]) ** 2 for axis in range(3)))
        for first in range(len(points))
        for second in range(first + 1, len(points))
    )
    if minimum < 0.05:
        raise ValueError(f"processed GRO has an atomic near-overlap: {minimum:.6f} nm")
    heavy_names = [name for name in names if atom_element(name) != "H"]
    expected_heavy = [name for name in assembly.atoms if atom_element(name) != "H"]
    if set(heavy_names) != set(expected_heavy):
        raise ValueError("processed GRO heavy atoms differ from the official assembly")
    return heavy_names, minimum


def target_from_name(name: str) -> Target:
    if name in TARGETS:
        return TARGETS[name]
    if name in TARGET_BY_CLEAN:
        return TARGET_BY_CLEAN[name]
    raise KeyError(name)


def generator_provenance() -> dict[str, object]:
    scripts = Path(__file__).resolve().parent
    helpers = [
        scripts / "prepare_charmm_standard_patch_input.py",
        scripts / "prepare_charmm_standard_uronic_input.py",
    ]
    return {
        "entrypoint": Path(__file__).name,
        "entrypoint_sha256": reviewed.sha256_file(Path(__file__).resolve()),
        "imported_helpers": {path.name: reviewed.sha256_file(path) for path in helpers},
    }


def build(args: argparse.Namespace) -> Path:
    target = target_from_name(args.molecule_name)
    alignment_path = Path(args.alignment_manifest).resolve()
    alignment_rows = load_alignment_rows(alignment_path)
    alignment_row = alignment_rows[target.clean_name]
    source_ff = Path(args.source_ff).resolve()
    manifest_path = Path(args.manifest).resolve()
    run_root = Path(args.run_root).resolve()
    frozen_sdf = Path(alignment_row["source_sdf"]).resolve()
    if args.sdf and Path(args.sdf).resolve() != frozen_sdf:
        raise ValueError("--sdf differs from the frozen alignment364 source")
    sdf = frozen_sdf
    if reviewed.sha256_file(sdf) != alignment_row["source_sdf_sha256"]:
        raise ValueError(f"frozen CASPER SDF hash mismatch for {target.clean_name}")
    gmx = reviewed.resolve_executable(args.gmx)
    manifest, manifest_metadata = validate_manifest(manifest_path, source_ff)
    assembly = build_assembly(source_ff, target, manifest)
    assembly_metadata = validate_assembly(assembly, target)
    source = reviewed.find_source_molecule(sdf, target.raw_name)
    source_metadata = reviewed.validate_source_identity(source, target)
    if int(alignment_row["n_heavy_atoms"]) != source_metadata["heavy_atom_count"]:
        raise ValueError(f"frozen alignment364 heavy-atom count mismatch for {target.clean_name}")
    optimized, mmff_metadata = reviewed.embed_mmff(source, args.seed)
    source_to_name, graph_metadata = map_source_graph(optimized, assembly, target)
    source_heavy_indices = [atom.GetIdx() for atom in optimized.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(source_heavy_indices) != assembly_metadata["heavy_atom_count"]:
        raise ValueError("validated source and official RTP heavy-atom counts disagree")

    molecule_dir = run_root / target.clean_name
    output_dir = molecule_dir / OUTPUT_SUBDIR
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
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
        copied = reviewed.copy_pinned_forcefield(source_ff, local_ff)
        local_residue_metadata = reviewed.append_local_residue(local_ff, assembly)
        modified_tree_before = reviewed.sha256_directory(local_ff)
        reviewed.write_heavy_gro(
            optimized, source_heavy_indices, source_to_name, target.raw_name, input_gro
        )
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
        process = reviewed.run_logged(
            command, staging, pdb2gmx_environment, pdb2gmx_log, args.timeout_seconds
        )
        if process.returncode != 0 or not processed_gro.is_file() or not topology.is_file():
            raise RuntimeError(f"pdb2gmx failed:\n{reviewed.tail(process.stdout)}")
        if f"Total charge {target.expected_charge:.3f} e" not in process.stdout:
            raise RuntimeError(
                "pdb2gmx did not report the independently expected official charge:\n"
                + reviewed.tail(process.stdout)
            )
        if source_ff.name in topology.read_text(encoding="utf-8"):
            raise RuntimeError("generated topology unexpectedly references the external source FF")

        processed_heavy, minimum_distance = parse_processed_gro(processed_gro, assembly)
        frame_index_by_name = {name: index for index, name in enumerate(processed_heavy)}
        source_heavy_names = [source_to_name[index] for index in source_heavy_indices]
        frame_mapping = [frame_index_by_name[name] for name in source_heavy_names]
        if sorted(frame_mapping) != list(range(len(frame_mapping))):
            raise RuntimeError("generated SDF-to-frame mapping is not bijective")
        mapping_path.write_text(json.dumps(frame_mapping, indent=2) + "\n", encoding="utf-8")

        topology_charge, topology_atom_count = reviewed.parse_topology_charge(topology)
        if abs(topology_charge - target.expected_charge) > reviewed.CHARGE_TOLERANCE:
            raise RuntimeError(
                f"generated topology charge {topology_charge:+.6f} differs from expected "
                f"{target.expected_charge:+d}"
            )
        self_contained_environment = os.environ.copy()
        self_contained_environment.pop("GMXLIB", None)
        preflight = reviewed.run_preflight(
            gmx,
            staging,
            topology,
            processed_gro,
            self_contained_environment,
            args.timeout_seconds,
            target.expected_charge,
        )
        modified_tree_after = reviewed.sha256_directory(local_ff)
        if modified_tree_after != modified_tree_before:
            raise RuntimeError("local force-field copy changed during strict preflight")
        version = reviewed.gromacs_version(gmx, self_contained_environment, args.timeout_seconds)

        charged = target.expected_charge != 0
        required_counterions = {
            "count": abs(target.expected_charge),
            "charge_e": -1 if target.expected_charge > 0 else (1 if charged else 0),
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
            raise RuntimeError(f"expected generated files are absent: {missing}")
        metadata = {
            "schema_version": 2,
            "topology_policy": TOPOLOGY_POLICY,
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
            "generator": generator_provenance(),
            "allowlist_size": len(TARGETS),
            "rtp_residue": LOCAL_RESIDUE,
            "expected_solute_charge_e": target.expected_charge,
            "observed_topology_charge_e": topology_charge,
            "topology_atom_count": topology_atom_count,
            "alignment364": {
                "manifest": str(alignment_path),
                "manifest_sha256": APPROVED_ALIGNMENT_MANIFEST_SHA256,
                "row": alignment_row,
            },
            "source": {"sdf": str(sdf), "sdf_sha256": reviewed.sha256_file(sdf), **source_metadata},
            "conformer": mmff_metadata,
            "standard_patch_expansion": {
                "policy": "declarative expansion of only pinned official CHARMM residue/PRES records",
                "coordinate_policy": manifest["coordinate_policy"],
                "microstate_policy": manifest["microstate_policy"],
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
            "files": {path.name: reviewed.sha256_file(path) for path in generated_files},
            "rdkit_version": rdBase.rdkitVersion,
            "note": (
                "Atom types and charges come exclusively from the pinned official CHARMM36 "
                "carbohydrate RTP residues and PRES records; no graph-derived or CGenFF guess is used."
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
    parser.add_argument(
        "--molecule-name", required=True, choices=sorted(set(TARGETS) | set(TARGET_BY_CLEAN))
    )
    parser.add_argument("--sdf", default="", help="Must match the frozen alignment364 source SDF")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--source-ff", default=str(DEFAULT_SOURCE_FF))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--alignment-manifest", default=str(DEFAULT_ALIGNMENT_MANIFEST))
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
