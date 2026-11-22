#!/usr/bin/env python3
"""Build matched ETKDGv3-only and MMFF94 conformer ensembles.

The protocol is deliberately fail-closed:

* the ETKDGv3-only branch performs embedding and geometric FPS only; it never
  asks a molecular mechanics force field to optimize or score a conformer;
* the paired branch starts from an exact copy of those embedded conformers and
  is accepted only when RDKit reports complete MMFF94 parameter coverage and
  every conformer converges; there is no fallback force field;
* both branches select conformers using the same deterministic heavy-atom
  Kabsch-RMSD farthest-point algorithm.  MMFF energies are evidence only and
  are not used for selection.

Existing conformer outputs are never overwritten.  A metadata JSON file is the
per-molecule commit marker; ``--resume`` semantics are always enabled by
validating its hashes and reusing a complete bundle.  Partial or invalid bundles
stop rather than being silently replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


MD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MD_ROOT.parent
DEFAULT_NAMES_MANIFEST = (
    REPO_ROOT
    / "experiments"
    / "glycosenseformer_conformer_source_v1"
    / "manifests"
    / "common329_sources.csv"
)
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "dataset"
DEFAULT_OUTPUT_PARENT = MD_ROOT / "glycosenseformer_conformer_sources"

PROTOCOL_ID = "glycosenseformer-etkdgv3-only-paired-mmff94-heavy-fps-v1"
SCHEMA_VERSION = 1
DEFAULT_SEED = 1
DEFAULT_CANDIDATES = 200
DEFAULT_SELECTED = 100
MAX_EMBED_ATTEMPTS = 32
DERIVED_SEED_STRIDE = 104729
MIN_UNIQUE_HEAVY_RMSD_ANGSTROM = 1.0e-4


class BuildError(RuntimeError):
    """Base class for protocol failures."""


class ManifestError(BuildError):
    """The names manifest or its source molecule is invalid."""


class ValidationError(BuildError):
    """Atom, coordinate, stereochemistry, or artifact validation failed."""


class MMFFParameterError(BuildError):
    """MMFF94 does not completely parameterize the molecule."""


class MMFFConvergenceError(BuildError):
    """At least one paired MMFF94 minimization did not converge."""


class OutputConflictError(BuildError):
    """An existing output is partial, invalid, or belongs to another run."""


@dataclass(frozen=True)
class BuildConfig:
    seed: int = DEFAULT_SEED
    n_candidates: int = DEFAULT_CANDIDATES
    n_selected: int = DEFAULT_SELECTED
    mmff_max_iters: int = 1000
    mmff_retries: int = 1

    def validate(self) -> None:
        if not 0 <= self.seed < 2**31 - 1:
            raise ValueError("seed must be in [0, 2^31-2] for the deterministic RDKit seed schedule")
        if self.n_candidates < 1:
            raise ValueError("n_candidates must be positive")
        if not 1 <= self.n_selected <= self.n_candidates:
            raise ValueError("n_selected must be in [1, n_candidates]")
        if self.mmff_max_iters < 1:
            raise ValueError("mmff_max_iters must be positive")
        if self.mmff_retries < 0:
            raise ValueError("mmff_retries cannot be negative")


@dataclass(frozen=True)
class MoleculeJob:
    ordinal: int
    molecule_name: str
    clean_name: str
    source_sdf: str
    source_index: int
    source_sdf_sha256: str
    n_heavy_atoms: int
    molecule_binary: bytes
    names_manifest_sha256: str
    stratum: str = ""
    fold: str = ""


@dataclass(frozen=True)
class ArtifactPaths:
    raw_candidates: Path
    raw_selected: Path
    mmff_candidates: Path
    mmff_selected: Path
    candidates_csv: Path
    metadata: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.raw_candidates,
            self.raw_selected,
            self.mmff_candidates,
            self.mmff_selected,
            self.candidates_csv,
            self.metadata,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def clean_casper_name(name: str) -> str:
    return name.replace("->", "").replace("(", "").replace(")", "")


def validate_clean_name(name: str) -> None:
    if not name or name in {".", ".."}:
        raise ManifestError("clean_name is empty or unsafe")
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ManifestError(f"unsafe clean_name for output path: {name!r}")
    if "\x00" in name:
        raise ManifestError("clean_name contains NUL")


def artifact_paths(output_root: Path, clean_name: str) -> ArtifactPaths:
    validate_clean_name(clean_name)
    return ArtifactPaths(
        raw_candidates=output_root / "etkdgv3_only" / "candidates200" / f"{clean_name}.pickle",
        raw_selected=output_root / "etkdgv3_only" / "fps100" / f"{clean_name}.pickle",
        mmff_candidates=output_root / "mmff94" / "candidates200" / f"{clean_name}.pickle",
        mmff_selected=output_root / "mmff94" / "fps100" / f"{clean_name}.pickle",
        candidates_csv=output_root / "selection_metadata" / f"{clean_name}.candidates.csv",
        metadata=output_root / "selection_metadata" / f"{clean_name}.json",
    )


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    """Return optimal-rotation RMSD for two atom-ordered coordinate arrays."""

    p = np.asarray(reference, dtype=np.float64)
    q = np.asarray(mobile, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or not p.shape[0]:
        raise ValueError(f"Kabsch inputs must have equal nonempty (N, 3) shape: {p.shape}, {q.shape}")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Kabsch inputs contain non-finite coordinates")
    p0 = p - p.mean(axis=0)
    q0 = q - q.mean(axis=0)
    u, _singular, vt = np.linalg.svd(p0.T @ q0)
    if np.linalg.det(u @ vt) < 0.0:
        u[:, -1] *= -1.0
    rotation = u @ vt
    delta = p0 @ rotation - q0
    return float(np.sqrt(np.sum(delta * delta) / p.shape[0]))


def pairwise_kabsch_rmsd(coordinates: Sequence[np.ndarray]) -> np.ndarray:
    if not coordinates:
        raise ValueError("cannot build an RMSD matrix for an empty ensemble")
    arrays = [np.asarray(item, dtype=np.float64) for item in coordinates]
    shape = arrays[0].shape
    if any(item.shape != shape for item in arrays):
        raise ValueError("all conformers must have identical shapes")
    matrix = np.zeros((len(arrays), len(arrays)), dtype=np.float64)
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            value = kabsch_rmsd(arrays[i], arrays[j])
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def deterministic_farthest_point_indices(distance_matrix: np.ndarray, count: int) -> list[int]:
    """Medoid-seeded FPS; all ties resolve to the lower (candidate) index."""

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1] or not distances.shape[0]:
        raise ValueError("distance_matrix must be a nonempty square matrix")
    n_items = distances.shape[0]
    if not 1 <= count <= n_items:
        raise ValueError("FPS count must be in [1, number of candidates]")
    if not np.isfinite(distances).all():
        raise ValueError("distance_matrix contains non-finite values")
    if not np.allclose(distances, distances.T, atol=1e-10, rtol=0.0):
        raise ValueError("distance_matrix is not symmetric")
    if not np.allclose(np.diag(distances), 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("distance_matrix diagonal is not zero")
    if np.any(distances < -1e-12):
        raise ValueError("distance_matrix contains negative distances")

    # Division by n-1 is immaterial for the argmin.  Summation also handles n=1.
    medoid = int(np.argmin(distances.sum(axis=1)))
    selected = [medoid]
    selected_mask = np.zeros(n_items, dtype=bool)
    selected_mask[medoid] = True
    nearest = distances[medoid].copy()
    while len(selected) < count:
        scores = nearest.copy()
        scores[selected_mask] = -math.inf
        next_index = int(np.argmax(scores))
        selected.append(next_index)
        selected_mask[next_index] = True
        nearest = np.minimum(nearest, distances[next_index])
    return selected


def _heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def _heavy_graph_signature(mol: Chem.Mol) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    heavy = _heavy_atom_indices(mol)
    remap = {old: new for new, old in enumerate(heavy)}
    atoms = tuple(
        (
            mol.GetAtomWithIdx(index).GetAtomicNum(),
            mol.GetAtomWithIdx(index).GetFormalCharge(),
            mol.GetAtomWithIdx(index).GetIsotope(),
        )
        for index in heavy
    )
    bonds: list[tuple[Any, ...]] = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin not in remap or end not in remap:
            continue
        left, right = sorted((remap[begin], remap[end]))
        bonds.append((left, right, str(bond.GetBondType()), bool(bond.GetIsAromatic())))
    return atoms, tuple(sorted(bonds))


def _expected_stereo(mol: Chem.Mol) -> dict[int, tuple[int, str]]:
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    expected: dict[int, tuple[int, str]] = {}
    for atom in probe.GetAtoms():
        if atom.GetAtomicNum() <= 1 or atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        expected[atom.GetIdx()] = (int(atom.GetChiralTag()), cip)
    return expected


def _validate_stereo_for_conformer(
    mol: Chem.Mol,
    conf_id: int,
    expected: dict[int, tuple[int, str]],
    label: str,
) -> None:
    if not expected:
        return
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe, confId=conf_id, replaceExistingTags=True)
    for atom_index, (expected_tag, expected_cip) in expected.items():
        atom = probe.GetAtomWithIdx(atom_index)
        observed_tag = int(atom.GetChiralTag())
        observed_cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
        if observed_tag != expected_tag or (expected_cip and observed_cip != expected_cip):
            raise ValidationError(
                f"{label} conformer {conf_id + 1}: stereochemistry changed at atom {atom_index}; "
                f"expected tag/CIP {expected_tag}/{expected_cip!r}, got {observed_tag}/{observed_cip!r}"
            )


def extract_and_validate_heavy_coordinates(
    mol: Chem.Mol,
    *,
    expected_count: int,
    expected_graph: tuple[tuple[Any, ...], tuple[Any, ...]],
    expected_stereo: dict[int, tuple[int, str]],
    label: str,
) -> list[np.ndarray]:
    if _heavy_graph_signature(mol) != expected_graph:
        raise ValidationError(f"{label}: heavy-atom identity/connectivity changed")
    heavy_indices = _heavy_atom_indices(mol)
    if len(mol.GetConformers()) != expected_count:
        raise ValidationError(
            f"{label}: expected {expected_count} conformers, found {len(mol.GetConformers())}"
        )
    coordinates: list[np.ndarray] = []
    for ordinal, conformer in enumerate(mol.GetConformers()):
        array = np.asarray(conformer.GetPositions(), dtype=np.float64)[heavy_indices].copy()
        if array.shape != (len(heavy_indices), 3) or not np.isfinite(array).all():
            raise ValidationError(f"{label} conformer {ordinal + 1}: invalid/non-finite coordinates")
        _validate_stereo_for_conformer(mol, conformer.GetId(), expected_stereo, label)
        coordinates.append(array)
    return coordinates


def prepare_input_molecule(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        raise ValidationError("input molecule is null")
    work = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    Chem.SanitizeMol(work)
    if not work.GetNumAtoms() or not _heavy_atom_indices(work):
        raise ValidationError("input molecule contains no heavy atoms")
    if any(atom.GetAtomicNum() == 1 for atom in work.GetAtoms()):
        raise ValidationError("failed to obtain a heavy-atom-only input molecule")
    Chem.AssignStereochemistry(work, cleanIt=True, force=True)
    return work


def embed_etkdgv3_only(
    mol: Chem.Mol,
    *,
    n_candidates: int,
    seed: int,
) -> tuple[Chem.Mol, list[np.ndarray], tuple[tuple[Any, ...], tuple[Any, ...]], dict[int, tuple[int, str]]]:
    """Embed the force-field-free branch; no MMFF or fallback calls belong here."""

    heavy = prepare_input_molecule(mol)
    expected_graph = _heavy_graph_signature(heavy)
    expected_stereo = _expected_stereo(heavy)
    template = Chem.AddHs(heavy, addCoords=False)
    work = Chem.Mol(template)
    work.RemoveAllConformers()
    accepted_coordinates: list[np.ndarray] = []
    attempt_records: list[dict[str, Any]] = []
    minimum_accepted_separation = math.inf
    minimum_rejected_separation = math.inf

    for attempt in range(MAX_EMBED_ATTEMPTS):
        remaining = n_candidates - len(accepted_coordinates)
        if remaining == 0:
            break
        derived_seed = int((seed + attempt * DERIVED_SEED_STRIDE) % (2**31 - 1))
        batch = _embed_etkdgv3_batch(template, n_conformers=remaining, seed=derived_seed)
        batch_coordinates = extract_and_validate_heavy_coordinates(
            batch,
            expected_count=len(batch.GetConformers()),
            expected_graph=expected_graph,
            expected_stereo=expected_stereo,
            label=f"ETKDGv3-only embedding attempt {attempt + 1}",
        )
        accepted_this_attempt = 0
        duplicates_this_attempt = 0
        accepted_batch_ordinals: list[int] = []
        accepted_candidate_ids: list[int] = []
        for batch_ordinal, (conformer, coordinates) in enumerate(
            zip(batch.GetConformers(), batch_coordinates), start=1
        ):
            if accepted_coordinates:
                nearest = min(kabsch_rmsd(coordinates, prior) for prior in accepted_coordinates)
                if nearest <= MIN_UNIQUE_HEAVY_RMSD_ANGSTROM:
                    duplicates_this_attempt += 1
                    minimum_rejected_separation = min(minimum_rejected_separation, nearest)
                    continue
                minimum_accepted_separation = min(minimum_accepted_separation, nearest)
            work.AddConformer(Chem.Conformer(conformer), assignId=True)
            accepted_coordinates.append(coordinates.copy())
            accepted_this_attempt += 1
            accepted_batch_ordinals.append(batch_ordinal)
            accepted_candidate_ids.append(len(accepted_coordinates))
            if len(accepted_coordinates) == n_candidates:
                break
        attempt_records.append(
            {
                "attempt": attempt + 1,
                "derived_seed": derived_seed,
                "requested": remaining,
                "embedded": len(batch_coordinates),
                "accepted_unique": accepted_this_attempt,
                "accepted_candidate_ids": accepted_candidate_ids,
                "accepted_batch_conformer_ordinals": accepted_batch_ordinals,
                "rejected_numerical_duplicates": duplicates_this_attempt,
                "accepted_total": len(accepted_coordinates),
            }
        )
    if len(accepted_coordinates) != n_candidates:
        raise ValidationError(
            f"ETKDGv3 produced only {len(accepted_coordinates)} unique successful conformers after "
            f"{MAX_EMBED_ATTEMPTS} deterministic attempts; exactly {n_candidates} are required; "
            f"attempts={attempt_records}"
        )
    audit = {
        "max_attempts": MAX_EMBED_ATTEMPTS,
        "attempts_used": len(attempt_records),
        "derived_seed_formula": f"(seed + attempt_index * {DERIVED_SEED_STRIDE}) mod (2^31-1)",
        "unique_heavy_kabsch_rmsd_threshold_angstrom": MIN_UNIQUE_HEAVY_RMSD_ANGSTROM,
        "minimum_accepted_nearest_rmsd_angstrom": (
            None if math.isinf(minimum_accepted_separation) else minimum_accepted_separation
        ),
        "minimum_rejected_duplicate_rmsd_angstrom": (
            None if math.isinf(minimum_rejected_separation) else minimum_rejected_separation
        ),
        "attempts": attempt_records,
    }
    work.SetProp("_GlycoSenseFormerETKDGEmbeddingAudit", json.dumps(audit, sort_keys=True))
    coordinates = extract_and_validate_heavy_coordinates(
        work,
        expected_count=n_candidates,
        expected_graph=expected_graph,
        expected_stereo=expected_stereo,
        label="ETKDGv3-only accepted unique pool",
    )
    return work, coordinates, expected_graph, expected_stereo


def _embed_etkdgv3_batch(template: Chem.Mol, *, n_conformers: int, seed: int) -> Chem.Mol:
    """Return one deterministic batch; callers merge only validated unique successes."""

    batch = Chem.Mol(template)
    batch.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True
    params.pruneRmsThresh = -1.0
    params.numThreads = 1
    params.randomSeed = int(seed)
    params.clearConfs = True
    AllChem.EmbedMultipleConfs(batch, numConfs=n_conformers, params=params)
    return batch


def optimize_mmff94_strict(
    embedded_mol: Chem.Mol,
    *,
    expected_graph: tuple[tuple[Any, ...], tuple[Any, ...]],
    expected_stereo: dict[int, tuple[int, str]],
    max_iters: int,
    retries: int,
) -> tuple[Chem.Mol, list[np.ndarray], list[int], list[float]]:
    """Optimize a copy with MMFF94; incomplete parameters or nonconvergence fail."""

    paired = Chem.Mol(embedded_mol)
    if not AllChem.MMFFHasAllMoleculeParams(paired):
        raise MMFFParameterError("MMFFHasAllMoleculeParams returned false; paired MMFF94 output blocked")

    results = list(
        AllChem.MMFFOptimizeMoleculeConfs(
            paired,
            numThreads=1,
            maxIters=max_iters,
            mmffVariant="MMFF94",
        )
    )
    if len(results) != len(paired.GetConformers()):
        raise MMFFConvergenceError("MMFF94 returned an incomplete optimization result list")
    statuses = [int(item[0]) for item in results]
    energies = [float(item[1]) for item in results]
    for index, status in enumerate(statuses):
        remaining = retries
        while status != 0 and remaining > 0:
            status = int(
                AllChem.MMFFOptimizeMolecule(
                    paired,
                    confId=paired.GetConformer(index).GetId(),
                    maxIters=max_iters,
                    mmffVariant="MMFF94",
                )
            )
            remaining -= 1
        statuses[index] = status

    failed = [index + 1 for index, status in enumerate(statuses) if status != 0]
    if failed:
        preview = ",".join(map(str, failed[:12]))
        suffix = "..." if len(failed) > 12 else ""
        raise MMFFConvergenceError(
            f"MMFF94 did not converge for {len(failed)} conformers: {preview}{suffix}"
        )

    # Re-evaluate energies after any retry, strictly with MMFF94.
    properties = AllChem.MMFFGetMoleculeProperties(paired, mmffVariant="MMFF94")
    if properties is None:
        raise MMFFParameterError("MMFF94 properties unexpectedly unavailable after coverage check")
    energies = []
    for conformer in paired.GetConformers():
        force_field = AllChem.MMFFGetMoleculeForceField(
            paired,
            properties,
            confId=conformer.GetId(),
        )
        if force_field is None:
            raise MMFFParameterError(
                f"MMFF94 force field unavailable for conformer {conformer.GetId() + 1}"
            )
        energy = float(force_field.CalcEnergy())
        if not math.isfinite(energy):
            raise ValidationError(f"MMFF94 conformer {conformer.GetId() + 1}: non-finite energy")
        energies.append(energy)

    coordinates = extract_and_validate_heavy_coordinates(
        paired,
        expected_count=len(embedded_mol.GetConformers()),
        expected_graph=expected_graph,
        expected_stereo=expected_stereo,
        label="paired MMFF94",
    )
    return paired, coordinates, statuses, energies


def fps_summary(distance_matrix: np.ndarray, selected: Sequence[int]) -> dict[str, Any]:
    selected_array = np.asarray(selected, dtype=np.int64)
    nearest = distance_matrix[:, selected_array].min(axis=1)
    triangle = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    medoid = selected[0]
    denominator = max(1, distance_matrix.shape[0] - 1)
    return {
        "selected_candidate_ids": [int(index + 1) for index in selected],
        "medoid_candidate_id": int(medoid + 1),
        "medoid_mean_rmsd_angstrom": float(distance_matrix[medoid].sum() / denominator),
        "all_pair_rmsd_min_angstrom": float(triangle.min()) if triangle.size else 0.0,
        "all_pair_rmsd_mean_angstrom": float(triangle.mean()) if triangle.size else 0.0,
        "all_pair_rmsd_max_angstrom": float(triangle.max()) if triangle.size else 0.0,
        "unselected_to_nearest_selected_max_angstrom": float(nearest.max()),
        "unselected_to_nearest_selected_mean_angstrom": float(nearest.mean()),
    }


def _write_pickle(path: Path, coordinates: Sequence[np.ndarray]) -> None:
    with path.open("wb") as handle:
        pickle.dump([np.asarray(item, dtype=np.float64) for item in coordinates], handle, protocol=4)


CSV_FIELDS = (
    "candidate_id",
    "etkdgv3_only_selected",
    "etkdgv3_only_fps_rank",
    "etkdgv3_only_nearest_selected_rmsd_angstrom",
    "mmff94_selected",
    "mmff94_fps_rank",
    "mmff94_nearest_selected_rmsd_angstrom",
    "mmff94_convergence_status",
    "mmff94_energy_kcal_mol",
    "energy_used_for_selection",
)


def _write_candidates_csv(
    path: Path,
    *,
    raw_distances: np.ndarray,
    raw_selected: Sequence[int],
    mmff_distances: np.ndarray,
    mmff_selected: Sequence[int],
    mmff_statuses: Sequence[int],
    mmff_energies: Sequence[float],
) -> None:
    raw_rank = {candidate: rank for rank, candidate in enumerate(raw_selected, start=1)}
    mmff_rank = {candidate: rank for rank, candidate in enumerate(mmff_selected, start=1)}
    raw_nearest = raw_distances[:, np.asarray(raw_selected)].min(axis=1)
    mmff_nearest = mmff_distances[:, np.asarray(mmff_selected)].min(axis=1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index in range(raw_distances.shape[0]):
            writer.writerow(
                {
                    "candidate_id": index + 1,
                    "etkdgv3_only_selected": str(index in raw_rank),
                    "etkdgv3_only_fps_rank": raw_rank.get(index, ""),
                    "etkdgv3_only_nearest_selected_rmsd_angstrom": f"{raw_nearest[index]:.12g}",
                    "mmff94_selected": str(index in mmff_rank),
                    "mmff94_fps_rank": mmff_rank.get(index, ""),
                    "mmff94_nearest_selected_rmsd_angstrom": f"{mmff_nearest[index]:.12g}",
                    "mmff94_convergence_status": mmff_statuses[index],
                    "mmff94_energy_kcal_mol": f"{mmff_energies[index]:.12g}",
                    "energy_used_for_selection": "False",
                }
            )


def _artifact_record(path: Path, output_root: Path, *, n_conformers: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_to_root(path, output_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if n_conformers is not None:
        record["n_conformers"] = n_conformers
    return record


def _publish_bundle(staging: Path, mapping: Sequence[tuple[Path, Path]]) -> None:
    created: list[Path] = []
    try:
        for source, destination in mapping:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)  # atomic and fails rather than overwriting
            created.append(destination)
    except Exception:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_pickle_coordinates(path: Path, *, count: int, n_atoms: int) -> list[np.ndarray]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise OutputConflictError(f"cannot load existing pickle {path}: {exc}") from exc
    if not isinstance(payload, (list, tuple)) or len(payload) != count:
        raise OutputConflictError(f"{path}: expected {count} conformers")
    arrays = [np.asarray(item) for item in payload]
    if any(item.shape != (n_atoms, 3) or not np.isfinite(item).all() for item in arrays):
        raise OutputConflictError(f"{path}: invalid coordinate shape or non-finite value")
    return arrays


def validate_existing_bundle(
    output_root: Path,
    job: MoleculeJob,
    config: BuildConfig,
) -> dict[str, Any] | None:
    paths = artifact_paths(output_root, job.clean_name)
    existing = [path.exists() for path in paths.all_paths()]
    if not any(existing):
        return None
    if not all(existing):
        missing = [str(path) for path, present in zip(paths.all_paths(), existing) if not present]
        raise OutputConflictError(
            f"{job.clean_name}: partial existing output; refusing overwrite; missing {missing}"
        )
    try:
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OutputConflictError(f"{job.clean_name}: invalid metadata JSON: {exc}") from exc
    expected_config = asdict(config)
    checks = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "clean_name": job.clean_name,
        "molecule_name": job.molecule_name,
    }
    for key, expected in checks.items():
        observed = metadata.get("molecule", {}).get(key) if key in {"clean_name", "molecule_name"} else metadata.get(key)
        if observed != expected:
            raise OutputConflictError(
                f"{job.clean_name}: metadata {key} mismatch ({observed!r} != {expected!r})"
            )
    if metadata.get("config") != expected_config:
        raise OutputConflictError(f"{job.clean_name}: existing bundle uses a different build config")
    source = metadata.get("source", {})
    if source.get("sdf_sha256") != job.source_sdf_sha256 or source.get("source_index") != job.source_index:
        raise OutputConflictError(f"{job.clean_name}: source provenance mismatch")
    if source.get("names_manifest_sha256") != job.names_manifest_sha256:
        raise OutputConflictError(f"{job.clean_name}: frozen names-manifest provenance mismatch")
    molecule_record = metadata.get("molecule", {})
    if molecule_record.get("stratum", "") != job.stratum or str(molecule_record.get("fold", "")) != job.fold:
        raise OutputConflictError(f"{job.clean_name}: stratum/fold provenance mismatch")

    expected_paths = {
        "etkdgv3_only_candidates": (paths.raw_candidates, config.n_candidates),
        "etkdgv3_only_fps": (paths.raw_selected, config.n_selected),
        "mmff94_candidates": (paths.mmff_candidates, config.n_candidates),
        "mmff94_fps": (paths.mmff_selected, config.n_selected),
        "candidates_csv": (paths.candidates_csv, None),
    }
    artifacts = metadata.get("artifacts", {})
    loaded: dict[str, list[np.ndarray]] = {}
    for key, (path, count) in expected_paths.items():
        record = artifacts.get(key, {})
        if record.get("path") != relative_to_root(path, output_root):
            raise OutputConflictError(f"{job.clean_name}: artifact path mismatch for {key}")
        if record.get("sha256") != sha256_file(path) or record.get("size_bytes") != path.stat().st_size:
            raise OutputConflictError(f"{job.clean_name}: hash/size mismatch for {key}")
        if count is not None:
            if record.get("n_conformers") != count:
                raise OutputConflictError(f"{job.clean_name}: conformer count metadata mismatch for {key}")
            loaded[key] = _load_pickle_coordinates(
                path,
                count=count,
                n_atoms=job.n_heavy_atoms,
            )
    for prefix in ("etkdgv3_only", "mmff94"):
        selected_ids = metadata.get("fps", {}).get(prefix, {}).get("selected_candidate_ids", [])
        if len(selected_ids) != config.n_selected or len(set(selected_ids)) != config.n_selected:
            raise OutputConflictError(f"{job.clean_name}: invalid selected candidate IDs for {prefix}")
        candidates = loaded[f"{prefix}_candidates"]
        selected = loaded[f"{prefix}_fps"]
        for position, candidate_id in enumerate(selected_ids):
            if not isinstance(candidate_id, int) or not 1 <= candidate_id <= config.n_candidates:
                raise OutputConflictError(f"{job.clean_name}: out-of-range selected candidate ID")
            if not np.array_equal(selected[position], candidates[candidate_id - 1]):
                raise OutputConflictError(f"{job.clean_name}: selected pickle is not bound to candidates")
    return metadata


def build_molecule_bundle(
    job: MoleculeJob,
    output_root: Path,
    config: BuildConfig,
) -> dict[str, Any]:
    existing = validate_existing_bundle(output_root, job, config)
    paths = artifact_paths(output_root, job.clean_name)
    if existing is not None:
        return result_row(job, paths, status="resumed", metadata=existing)

    molecule = Chem.Mol(job.molecule_binary)
    if molecule is None:
        raise ValidationError(f"{job.clean_name}: cannot reconstruct source molecule")
    embedded, raw_coordinates, expected_graph, expected_stereo = embed_etkdgv3_only(
        molecule,
        n_candidates=config.n_candidates,
        seed=config.seed,
    )
    embedding_audit = json.loads(embedded.GetProp("_GlycoSenseFormerETKDGEmbeddingAudit"))
    if len(raw_coordinates[0]) != job.n_heavy_atoms:
        raise ValidationError(
            f"{job.clean_name}: manifest says {job.n_heavy_atoms} heavy atoms, embedded molecule has "
            f"{len(raw_coordinates[0])}"
        )
    _paired, mmff_coordinates, mmff_statuses, mmff_energies = optimize_mmff94_strict(
        embedded,
        expected_graph=expected_graph,
        expected_stereo=expected_stereo,
        max_iters=config.mmff_max_iters,
        retries=config.mmff_retries,
    )

    raw_distances = pairwise_kabsch_rmsd(raw_coordinates)
    mmff_distances = pairwise_kabsch_rmsd(mmff_coordinates)
    raw_selected = deterministic_farthest_point_indices(raw_distances, config.n_selected)
    mmff_selected = deterministic_farthest_point_indices(mmff_distances, config.n_selected)

    staging = output_root / ".staging" / f"{job.clean_name}-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    staged = {
        "raw_candidates": staging / "etkdgv3_only.candidates.pickle",
        "raw_selected": staging / "etkdgv3_only.fps.pickle",
        "mmff_candidates": staging / "mmff94.candidates.pickle",
        "mmff_selected": staging / "mmff94.fps.pickle",
        "candidates_csv": staging / "candidates.csv",
        "metadata": staging / "metadata.json",
    }
    try:
        _write_pickle(staged["raw_candidates"], raw_coordinates)
        _write_pickle(staged["raw_selected"], [raw_coordinates[index] for index in raw_selected])
        _write_pickle(staged["mmff_candidates"], mmff_coordinates)
        _write_pickle(staged["mmff_selected"], [mmff_coordinates[index] for index in mmff_selected])
        _write_candidates_csv(
            staged["candidates_csv"],
            raw_distances=raw_distances,
            raw_selected=raw_selected,
            mmff_distances=mmff_distances,
            mmff_selected=mmff_selected,
            mmff_statuses=mmff_statuses,
            mmff_energies=mmff_energies,
        )
        script_hash = sha256_file(Path(__file__).resolve())
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": utc_now(),
            "molecule": {
                "molecule_name": job.molecule_name,
                "clean_name": job.clean_name,
                "n_heavy_atoms": job.n_heavy_atoms,
                "isomeric_smiles": Chem.MolToSmiles(prepare_input_molecule(molecule), isomericSmiles=True),
                "specified_chiral_centers": len(expected_stereo),
                "stratum": job.stratum,
                "fold": job.fold,
            },
            "source": {
                "sdf": job.source_sdf,
                "source_index": job.source_index,
                "sdf_sha256": job.source_sdf_sha256,
                "molecule_binary_sha256": sha256_bytes(job.molecule_binary),
                "names_manifest_sha256": job.names_manifest_sha256,
                "builder_script_sha256": script_hash,
            },
            "config": asdict(config),
            "generation": {
                "paired_candidate_ids": list(range(1, config.n_candidates + 1)),
                "etkdgv3_only": {
                    "embedding": "RDKit ETKDGv3",
                    "forcefield_geometry_optimization": False,
                    "forcefield_energy_evaluation": False,
                    "energy_ranking": False,
                    "hydrogens_used_for_embedding": True,
                    "published_atoms": "heavy atoms in source order",
                    "incremental_embedding_audit": embedding_audit,
                },
                "mmff94": {
                    "initial_coordinates": "exact paired copies of ETKDGv3 candidate IDs",
                    "parameter_gate": "MMFFHasAllMoleculeParams == True",
                    "variant": "MMFF94",
                    "fallback_forcefield": None,
                    "all_converged": True,
                    "energy_unit": "kcal/mol",
                    "energy_used_for_fps": False,
                },
            },
            "validation": {
                "finite_coordinates": True,
                "heavy_atom_identity_and_order_preserved": True,
                "heavy_atom_connectivity_preserved": True,
                "specified_stereochemistry_preserved_in_every_conformer": True,
            },
            "fps": {
                "algorithm": "medoid-seeded farthest-point selection",
                "distance": "atom-ordered heavy-atom Kabsch RMSD in angstrom",
                "tie_break": "lower 1-based candidate_id",
                "energy_used": False,
                "etkdgv3_only": fps_summary(raw_distances, raw_selected),
                "mmff94": fps_summary(mmff_distances, mmff_selected),
            },
            "mmff94_energy_kcal_mol": {
                "min": float(min(mmff_energies)),
                "mean": float(np.mean(mmff_energies)),
                "max": float(max(mmff_energies)),
            },
            "artifacts": {
                "etkdgv3_only_candidates": _artifact_record(
                    staged["raw_candidates"], staging, n_conformers=config.n_candidates
                ),
                "etkdgv3_only_fps": _artifact_record(
                    staged["raw_selected"], staging, n_conformers=config.n_selected
                ),
                "mmff94_candidates": _artifact_record(
                    staged["mmff_candidates"], staging, n_conformers=config.n_candidates
                ),
                "mmff94_fps": _artifact_record(
                    staged["mmff_selected"], staging, n_conformers=config.n_selected
                ),
                "candidates_csv": _artifact_record(staged["candidates_csv"], staging),
            },
        }
        # Replace staging-relative evidence paths with immutable output-root paths.
        for key, destination in {
            "etkdgv3_only_candidates": paths.raw_candidates,
            "etkdgv3_only_fps": paths.raw_selected,
            "mmff94_candidates": paths.mmff_candidates,
            "mmff94_fps": paths.mmff_selected,
            "candidates_csv": paths.candidates_csv,
        }.items():
            metadata["artifacts"][key]["path"] = relative_to_root(destination, output_root)
        staged["metadata"].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_bundle(
            staging,
            (
                (staged["raw_candidates"], paths.raw_candidates),
                (staged["raw_selected"], paths.raw_selected),
                (staged["mmff_candidates"], paths.mmff_candidates),
                (staged["mmff_selected"], paths.mmff_selected),
                (staged["candidates_csv"], paths.candidates_csv),
                (staged["metadata"], paths.metadata),  # commit marker is published last
            ),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    validated = validate_existing_bundle(output_root, job, config)
    if validated is None:  # pragma: no cover - defensive
        raise OutputConflictError(f"{job.clean_name}: publication disappeared before validation")
    return result_row(job, paths, status="complete", metadata=validated)


def result_row(
    job: MoleculeJob,
    paths: ArtifactPaths,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "ordinal": job.ordinal,
        "molecule_name": job.molecule_name,
        "clean_name": job.clean_name,
        "status": status,
        "error": error,
        "source_sdf": job.source_sdf,
        "source_index": job.source_index,
        "source_sdf_sha256": job.source_sdf_sha256,
        "n_heavy_atoms": job.n_heavy_atoms,
        "stratum": job.stratum,
        "fold": job.fold,
        "etkdgv3_only_candidates_pickle": str(paths.raw_candidates),
        "etkdgv3_only_fps_pickle": str(paths.raw_selected),
        "mmff94_candidates_pickle": str(paths.mmff_candidates),
        "mmff94_fps_pickle": str(paths.mmff_selected),
        "candidates_csv": str(paths.candidates_csv),
        "metadata_json": str(paths.metadata),
        "metadata_sha256": sha256_file(paths.metadata) if metadata is not None else "",
    }
    return base


RESULT_FIELDS = tuple(
    result_row(
        MoleculeJob(0, "", "placeholder", "", 0, "", 0, b"", ""),
        artifact_paths(Path("."), "placeholder"),
        status="",
    ).keys()
)


def _first_value(row: dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = (row.get(name) or "").strip()
        if value:
            return value
    return ""


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ManifestError(f"names manifest does not exist: {path}")
    if path.suffix.lower() in {".txt", ".list", ".names"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if name and not name.startswith("#"):
                rows.append({"molecule_name": name})
        return rows
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestError(f"CSV manifest has no header: {path}")
        return [{str(key): str(value or "") for key, value in row.items()} for row in reader]


def _resolve_source_path(value: str, manifest: Path) -> Path:
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = (manifest.parent / source).resolve()
    if not source.is_file():
        raise ManifestError(f"source SDF does not exist: {source}")
    return source


def _default_sdf_paths() -> list[Path]:
    paths = [DEFAULT_DATASET_DIR / name for name in ("casper_mono.sdf", "casper_dimer.sdf", "casper_trimer.sdf")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ManifestError(f"default CASPER SDF files are missing: {missing}")
    return paths


def load_jobs(names_manifest: Path, *, limit: int | None = None) -> list[MoleculeJob]:
    rows = _read_manifest_rows(names_manifest)
    if limit is not None:
        if limit < 0:
            raise ManifestError("--limit cannot be negative")
        rows = rows[:limit]
    if not rows:
        raise ManifestError("names manifest selected no molecules")

    supplier_cache: dict[Path, list[Chem.Mol | None]] = {}
    hash_cache: dict[Path, str] = {}
    default_lookup: dict[str, tuple[Path, int, Chem.Mol]] | None = None
    manifest_hash = sha256_file(names_manifest)
    jobs: list[MoleculeJob] = []
    seen: set[str] = set()

    def supplier(source: Path) -> list[Chem.Mol | None]:
        if source not in supplier_cache:
            supplier_cache[source] = list(Chem.SDMolSupplier(str(source), removeHs=False, sanitize=True))
        return supplier_cache[source]

    def make_default_lookup() -> dict[str, tuple[Path, int, Chem.Mol]]:
        nonlocal default_lookup
        if default_lookup is None:
            default_lookup = {}
            for source in _default_sdf_paths():
                for index, molecule in enumerate(supplier(source)):
                    if molecule is None:
                        continue
                    title = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
                    for key in {title, clean_casper_name(title)}:
                        if key and key in default_lookup:
                            raise ManifestError(f"ambiguous default SDF molecule name: {key}")
                        if key:
                            default_lookup[key] = (source, index, molecule)
        return default_lookup

    for ordinal, row in enumerate(rows, start=1):
        molecule_name = _first_value(row, ("molecule_name", "name", "_Name", "title"))
        clean_name = _first_value(row, ("clean_name", "pickle_name", "stem"))
        if not molecule_name and not clean_name:
            raise ManifestError(f"manifest row {ordinal} has no molecule_name or clean_name")
        lookup_name = molecule_name or clean_name
        source_value = _first_value(row, ("source_sdf", "sdf_path", "sdf", "source"))
        index_value = _first_value(row, ("source_index", "sdf_index", "record_index", "index"))

        molecule: Chem.Mol | None
        if source_value:
            source = _resolve_source_path(source_value, names_manifest)
            molecules = supplier(source)
            if index_value:
                try:
                    source_index = int(index_value)
                except ValueError as exc:
                    raise ManifestError(f"row {ordinal}: invalid source index {index_value!r}") from exc
                if not 0 <= source_index < len(molecules):
                    raise ManifestError(f"row {ordinal}: source index {source_index} is out of range")
                molecule = molecules[source_index]
            else:
                matches = []
                for index, candidate in enumerate(molecules):
                    if candidate is None:
                        continue
                    title = candidate.GetProp("_Name").strip() if candidate.HasProp("_Name") else ""
                    if lookup_name in {title, clean_casper_name(title)}:
                        matches.append((index, candidate))
                if len(matches) != 1:
                    raise ManifestError(
                        f"row {ordinal}: expected one {lookup_name!r} match in {source}, found {len(matches)}"
                    )
                source_index, molecule = matches[0]
        else:
            match = make_default_lookup().get(lookup_name)
            if match is None:
                raise ManifestError(f"row {ordinal}: molecule {lookup_name!r} not found in default SDF files")
            source, source_index, molecule = match

        if molecule is None:
            raise ManifestError(f"row {ordinal}: RDKit failed to read molecule at {source}:{source_index}")
        title = molecule.GetProp("_Name").strip() if molecule.HasProp("_Name") else ""
        if not title:
            raise ManifestError(f"row {ordinal}: source molecule has no _Name")
        if molecule_name and title != molecule_name and clean_casper_name(title) != molecule_name:
            raise ManifestError(
                f"row {ordinal}: source title mismatch; manifest {molecule_name!r}, SDF {title!r}"
            )
        molecule_name = molecule_name or title
        clean_name = clean_name or clean_casper_name(title)
        if clean_casper_name(title) != clean_name:
            raise ManifestError(
                f"row {ordinal}: clean_name {clean_name!r} does not match compact SDF title "
                f"{clean_casper_name(title)!r}"
            )
        validate_clean_name(clean_name)
        if clean_name in seen:
            raise ManifestError(f"duplicate clean_name in selected manifest: {clean_name}")
        seen.add(clean_name)

        if source not in hash_cache:
            hash_cache[source] = sha256_file(source)
        source_hash = hash_cache[source]
        expected_hash = _first_value(row, ("source_sdf_sha256", "sdf_sha256"))
        if expected_hash and expected_hash != source_hash:
            raise ManifestError(f"row {ordinal}: source SDF SHA256 mismatch for {source}")
        prepared = prepare_input_molecule(molecule)
        n_heavy = len(_heavy_atom_indices(prepared))
        expected_n_heavy = _first_value(row, ("n_heavy_atoms", "heavy_atom_count", "atom_count"))
        if expected_n_heavy and int(expected_n_heavy) != n_heavy:
            raise ManifestError(
                f"row {ordinal}: heavy-atom count mismatch ({expected_n_heavy} != {n_heavy})"
            )
        binary = molecule.ToBinary(Chem.PropertyPickleOptions.AllProps)
        jobs.append(
            MoleculeJob(
                ordinal=ordinal,
                molecule_name=molecule_name,
                clean_name=clean_name,
                source_sdf=str(source.resolve()),
                source_index=source_index,
                source_sdf_sha256=source_hash,
                n_heavy_atoms=n_heavy,
                molecule_binary=binary,
                names_manifest_sha256=manifest_hash,
                stratum=_first_value(row, ("stratum", "source_class", "class")),
                fold=_first_value(row, ("fold", "split_fold")),
            )
        )
    return jobs


def _failed_result(job: MoleculeJob, output_root: Path, exc: BaseException) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return result_row(
        job,
        artifact_paths(output_root, job.clean_name),
        status="failed",
        error=message,
    )


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_run_manifests(
    output_root: Path,
    *,
    rows: Sequence[dict[str, Any]],
    names_manifest: Path,
    config: BuildConfig,
    workers: int,
    started_at: str,
) -> dict[str, Any]:
    csv_path = output_root / "build_manifest.csv"
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    _atomic_replace_text(csv_path, csv_text)

    long_fields = (
        "molecule_id",
        "molecule_name",
        "clean_name",
        "stratum",
        "source",
        "conformer_path",
        "source_sdf",
        "source_index",
        "fold",
        "expected_conformer_sha256",
        "expected_conformer_count",
        "selection_metadata_path",
        "selection_metadata_sha256",
    )
    long_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] not in {"complete", "resumed"}:
            continue
        metadata_path = Path(str(row["metadata_json"]))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        artifact_by_source = {
            "etkdgv3_only_fps100": metadata["artifacts"]["etkdgv3_only_fps"],
            "mmff94_fps100": metadata["artifacts"]["mmff94_fps"],
        }
        path_by_source = {
            "etkdgv3_only_fps100": row["etkdgv3_only_fps_pickle"],
            "mmff94_fps100": row["mmff94_fps_pickle"],
        }
        for source, artifact in artifact_by_source.items():
            long_rows.append(
                {
                    "molecule_id": row["clean_name"],
                    "molecule_name": row["molecule_name"],
                    "clean_name": row["clean_name"],
                    "stratum": row["stratum"],
                    "source": source,
                    "conformer_path": path_by_source[source],
                    "source_sdf": row["source_sdf"],
                    "source_index": row["source_index"],
                    "fold": row["fold"],
                    "expected_conformer_sha256": artifact["sha256"],
                    "expected_conformer_count": artifact["n_conformers"],
                    "selection_metadata_path": row["metadata_json"],
                    "selection_metadata_sha256": row["metadata_sha256"],
                }
            )
    long_buffer = StringIO()
    long_writer = csv.DictWriter(long_buffer, fieldnames=long_fields, lineterminator="\n")
    long_writer.writeheader()
    long_writer.writerows(long_rows)
    long_path = output_root / "conformer_source_manifest.csv"
    _atomic_replace_text(long_path, long_buffer.getvalue())
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "names_manifest": str(names_manifest.resolve()),
        "names_manifest_sha256": sha256_file(names_manifest),
        "output_root": str(output_root.resolve()),
        "config": asdict(config),
        "workers": workers,
        "status_counts": counts,
        "molecule_count": len(rows),
        "build_manifest_csv": str(csv_path.resolve()),
        "build_manifest_csv_sha256": sha256_file(csv_path),
        "conformer_source_manifest_csv": str(long_path.resolve()),
        "conformer_source_manifest_csv_sha256": sha256_file(long_path),
        "conformer_source_manifest_rows": len(long_rows),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    json_path = output_root / "build_manifest.json"
    _atomic_replace_text(
        json_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def run_jobs(
    jobs: Sequence[MoleculeJob],
    *,
    output_root: Path,
    config: BuildConfig,
    workers: int,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    rows_by_ordinal: dict[int, dict[str, Any]] = {}
    if workers == 1:
        for job in jobs:
            try:
                rows_by_ordinal[job.ordinal] = build_molecule_bundle(job, output_root, config)
            except Exception as exc:
                rows_by_ordinal[job.ordinal] = _failed_result(job, output_root, exc)
                print(f"[failed] {job.clean_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            else:
                print(f"[{rows_by_ordinal[job.ordinal]['status']}] {job.clean_name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_molecule_bundle, job, output_root, config): job for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    rows_by_ordinal[job.ordinal] = future.result()
                except Exception as exc:
                    rows_by_ordinal[job.ordinal] = _failed_result(job, output_root, exc)
                    print(f"[failed] {job.clean_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    if os.environ.get("GLYCOSENSEFORMER_ETKDG_TRACEBACK") == "1":
                        traceback.print_exc()
                else:
                    print(f"[{rows_by_ordinal[job.ordinal]['status']}] {job.clean_name}", flush=True)
    return [rows_by_ordinal[job.ordinal] for job in jobs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build exactly 200 paired ETKDGv3 candidates and select 100 by deterministic "
            "heavy-atom RMSD FPS for both ETKDGv3-only and strict MMFF94 branches."
        )
    )
    parser.add_argument(
        "--names-manifest",
        type=Path,
        default=DEFAULT_NAMES_MANIFEST,
        help=(
            "CSV (alignment364 columns are supported) or one-name-per-line file. "
            f"Default: {DEFAULT_NAMES_MANIFEST}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N manifest rows (pilot/resume helper).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Independent molecule worker processes; RDKit itself remains single-threaded.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="New, isolated output root. Complete hash-validated molecules are resumed.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Fixed ETKDGv3 random seed.")
    parser.add_argument(
        "--mmff-max-iters",
        type=int,
        default=1000,
        help="MMFF94 iterations per attempt; failure after retry blocks the molecule.",
    )
    parser.add_argument(
        "--mmff-retries",
        type=int,
        default=1,
        help="Additional MMFF94 attempts for a non-converged paired candidate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BuildConfig(
        seed=args.seed,
        n_candidates=DEFAULT_CANDIDATES,
        n_selected=DEFAULT_SELECTED,
        mmff_max_iters=args.mmff_max_iters,
        mmff_retries=args.mmff_retries,
    )
    try:
        config.validate()
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        names_manifest = args.names_manifest.resolve()
        output_root = (
            args.output_root.resolve()
            if args.output_root is not None
            else (DEFAULT_OUTPUT_PARENT / f"etkdgv3_only_vs_mmff94_fps_seed{config.seed}").resolve()
        )
        jobs = load_jobs(names_manifest, limit=args.limit)
        started_at = utc_now()
        rows = run_jobs(jobs, output_root=output_root, config=config, workers=args.workers)
        summary = write_run_manifests(
            output_root,
            rows=rows,
            names_manifest=names_manifest,
            config=config,
            workers=args.workers,
            started_at=started_at,
        )
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if summary["status_counts"].get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
