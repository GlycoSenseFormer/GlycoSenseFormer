#!/usr/bin/env python3
"""Read-only local-chirality gate for repaired GLYCAM coordinates.

The reference and candidate GRO files must use the same atom ordering and
topology.  For every tetrahedral carbon that has at least three heavy-atom
neighbours, the script compares the sign of a normalized scalar triple
product.  Neighbours are selected in a stable residue/atom-name order, with
same-residue atoms preferred.  Thus a glycosidic C1 is evaluated from its
intact C2/O5/H1 geometry even when an old manual inter-residue bond left the
external glycosidic oxygen far away.  This detects inversions without
requiring the two conformations to share their global orientation.

Old LEaP ``combine``/manual-``bond`` structures can have invalid coordinates
at an inter-residue bond.  A center whose reference bond geometry is invalid
is reported as unevaluable instead of being called an inversion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from repair_glycam_coordinates_from_mmff import parse_gromacs_topology


def read_gro_coordinates(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"GRO file is too short: {path}")
    try:
        atom_count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"invalid GRO atom count: {path}") from exc
    if len(lines) != atom_count + 3:
        raise ValueError(
            f"GRO declares {atom_count} atoms but contains {len(lines) - 3}: {path}"
        )

    coordinates = np.zeros((atom_count, 3), dtype=float)
    atoms: list[dict[str, Any]] = []
    for index, line in enumerate(lines[2 : 2 + atom_count]):
        if len(line) < 44:
            raise ValueError(f"short GRO atom line {index + 1}: {path}")
        try:
            coordinates[index] = (
                float(line[20:28]),
                float(line[28:36]),
                float(line[36:44]),
            )
            residue_number = int(line[:5])
        except ValueError as exc:
            raise ValueError(f"invalid GRO atom line {index + 1}: {path}") from exc
        atoms.append(
            {
                "index": index,
                "residue_number": residue_number,
                "residue": line[5:10].strip(),
                "name": line[10:15].strip(),
            }
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"non-finite coordinate in {path}")
    return coordinates, atoms


def normalized_signed_volume(
    coordinates: np.ndarray,
    center: int,
    neighbours: list[int],
) -> float:
    if len(neighbours) < 3:
        raise ValueError("three neighbours are required for a signed volume")
    vectors = [coordinates[index] - coordinates[center] for index in neighbours[:3]]
    lengths = [float(np.linalg.norm(vector)) for vector in vectors]
    if any(length <= 0.0 for length in lengths):
        return 0.0
    return float(np.linalg.det(np.stack(vectors)) / math.prod(lengths))


def center_label(
    index: int,
    topology_atoms: dict[int, dict[str, object]],
    gro_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    topology_atom = topology_atoms[index]
    gro_atom = gro_atoms[index]
    return {
        "atom_index_1based": index + 1,
        "residue_number": gro_atom["residue_number"],
        "residue": topology_atom["residue"],
        "atom_name": topology_atom["name"],
    }


def neighbour_label(
    index: int,
    topology_atoms: dict[int, dict[str, object]],
    gro_atoms: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "atom_index_1based": index + 1,
        "residue_number": gro_atoms[index]["residue_number"],
        "residue": topology_atoms[index]["residue"],
        "atom_name": topology_atoms[index]["name"],
    }


def audit_chirality(
    *,
    topology: Path,
    reference_gro: Path,
    candidate_gro: Path,
    minimum_bond_nm: float = 0.07,
    maximum_bond_nm: float = 0.20,
    minimum_normalized_volume: float = 0.05,
) -> dict[str, Any]:
    """Compare local tetrahedral handedness without modifying either input."""
    if not 0.0 < minimum_bond_nm < maximum_bond_nm:
        raise ValueError("bond thresholds must satisfy 0 < minimum < maximum")
    if not 0.0 < minimum_normalized_volume < 1.0:
        raise ValueError("minimum_normalized_volume must be between 0 and 1")

    graph, topology_atoms = parse_gromacs_topology(topology)
    reference, reference_atoms = read_gro_coordinates(reference_gro)
    candidate, candidate_atoms = read_gro_coordinates(candidate_gro)
    atom_count = graph.number_of_nodes()
    if reference.shape != candidate.shape or reference.shape != (atom_count, 3):
        raise ValueError(
            "atom-count mismatch: "
            f"topology={atom_count}, reference={len(reference)}, candidate={len(candidate)}"
        )
    for index in range(atom_count):
        expected = topology_atoms[index]
        for source_name, atoms in (
            ("reference", reference_atoms),
            ("candidate", candidate_atoms),
        ):
            if atoms[index]["name"] != expected["name"] or atoms[index]["residue"] != expected["residue"]:
                raise ValueError(
                    f"{source_name} atom order differs from topology at atom {index + 1}: "
                    f"GRO={atoms[index]['residue']}/{atoms[index]['name']} "
                    f"TOP={expected['residue']}/{expected['name']}"
                )

    evaluated: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    unevaluable: list[dict[str, Any]] = []
    for center in sorted(graph):
        if graph.nodes[center]["element"] != "C" or graph.degree(center) != 4:
            continue
        heavy_neighbours = [
            neighbour
            for neighbour in graph.neighbors(center)
            if graph.nodes[neighbour]["element"] != "H"
        ]
        if len(heavy_neighbours) < 3:
            continue

        label = center_label(center, topology_atoms, reference_atoms)
        neighbours = sorted(
            graph.neighbors(center),
            key=lambda neighbour: (
                reference_atoms[neighbour]["residue_number"]
                != reference_atoms[center]["residue_number"],
                reference_atoms[neighbour]["residue_number"],
                str(topology_atoms[neighbour]["residue"]),
                str(topology_atoms[neighbour]["name"]),
                neighbour,
            ),
        )
        reference_distance = {
            neighbour: float(np.linalg.norm(reference[neighbour] - reference[center]))
            for neighbour in neighbours
        }
        valid_reference_neighbours = [
            neighbour
            for neighbour in neighbours
            if minimum_bond_nm <= reference_distance[neighbour] <= maximum_bond_nm
        ]
        invalid_reference_neighbours = [
            neighbour for neighbour in neighbours if neighbour not in valid_reference_neighbours
        ]
        if len(valid_reference_neighbours) < 3:
            unevaluable.append(
                {
                    **label,
                    "reason": "fewer_than_three_valid_reference_bonds",
                    "valid_reference_bonds": len(valid_reference_neighbours),
                    "invalid_reference_neighbours": [
                        {
                            **neighbour_label(neighbour, topology_atoms, reference_atoms),
                            "distance_nm": reference_distance[neighbour],
                        }
                        for neighbour in invalid_reference_neighbours
                    ],
                }
            )
            continue

        selected_neighbours = valid_reference_neighbours[:3]
        selected_labels = [
            neighbour_label(neighbour, topology_atoms, reference_atoms)
            for neighbour in selected_neighbours
        ]
        reference_volume = normalized_signed_volume(reference, center, selected_neighbours)
        if abs(reference_volume) < minimum_normalized_volume:
            unevaluable.append(
                {
                    **label,
                    "reason": "reference_near_planar",
                    "reference_normalized_volume": reference_volume,
                    "selected_neighbours": selected_labels,
                }
            )
            continue

        candidate_distance = {
            neighbour: float(np.linalg.norm(candidate[neighbour] - candidate[center]))
            for neighbour in neighbours
        }
        candidate_bonds = list(candidate_distance.values())
        candidate_volume = normalized_signed_volume(candidate, center, selected_neighbours)
        record = {
            **label,
            "selected_neighbours": selected_labels,
            "reference_normalized_volume": reference_volume,
            "candidate_normalized_volume": candidate_volume,
            "minimum_candidate_bond_nm": min(candidate_bonds),
            "maximum_candidate_bond_nm": max(candidate_bonds),
            "ignored_invalid_reference_neighbours": [
                {
                    **neighbour_label(neighbour, topology_atoms, reference_atoms),
                    "distance_nm": reference_distance[neighbour],
                }
                for neighbour in invalid_reference_neighbours
            ],
        }
        invalid_candidate_bonds = [
            distance
            for distance in candidate_bonds
            if distance < minimum_bond_nm or distance > maximum_bond_nm
        ]
        if invalid_candidate_bonds:
            mismatches.append({**record, "reason": "candidate_bond_geometry_out_of_range"})
        elif abs(candidate_volume) < minimum_normalized_volume:
            mismatches.append({**record, "reason": "candidate_near_planar"})
        elif reference_volume * candidate_volume < 0.0:
            mismatches.append({**record, "reason": "chirality_inverted"})
        else:
            evaluated.append(record)

    passed = bool(evaluated) and not mismatches
    complete = not unevaluable
    return {
        "passed": passed,
        "complete": complete,
        "strict_passed": passed and complete,
        "topology": str(topology),
        "reference_gro": str(reference_gro),
        "candidate_gro": str(candidate_gro),
        "atom_count": atom_count,
        "thresholds": {
            "minimum_bond_nm": minimum_bond_nm,
            "maximum_bond_nm": maximum_bond_nm,
            "minimum_normalized_volume": minimum_normalized_volume,
        },
        "counts": {
            "evaluated_matching": len(evaluated),
            "mismatched": len(mismatches),
            "unevaluable": len(unevaluable),
        },
        "mismatches": mismatches,
        "unevaluable": unevaluable,
        "evaluated": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only chirality comparison for GLYCAM GRO coordinates."
    )
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--reference-gro", type=Path, required=True)
    parser.add_argument("--candidate-gro", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Also fail when invalid reference geometry leaves any center unevaluable.",
    )
    args = parser.parse_args()
    result = audit_chirality(
        topology=args.topology,
        reference_gro=args.reference_gro,
        candidate_gro=args.candidate_gro,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    gate = result["strict_passed"] if args.require_complete else result["passed"]
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
