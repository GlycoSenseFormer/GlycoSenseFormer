#!/usr/bin/env python3
"""Select deterministic FPS100 ensembles from three MD all200 sources.

The selector is intentionally read-only with respect to the CHARMM, GROMOS,
and GLYCAM source pickles.  For each source it verifies the hash frozen in the
common-329 manifest, loads exactly 200 finite atom-ordered heavy-atom arrays,
and applies the same Kabsch-RMSD/medoid/FPS implementation used by
``build_glycosenseformer_etkdg_fps.py``.

Each selected pickle is published without overwrite.  Its adjacent metadata
JSON is linked into place last and is therefore the per-source commit marker.
Existing complete bundles are resumed only after the source hash, deterministic
selection, output hash, and every selected coordinate are revalidated.  A
partial or inconsistent bundle fails closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "glycosenseformer_conformer_source_v1"
DEFAULT_MANIFEST = ROOT / "md/systems/example_aDMan.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/rebuilt_fps100"

FORCEFIELDS = ("charmm", "gromos", "glycam")
SOURCE_NAMES = {forcefield: f"{forcefield}_fps100" for forcefield in FORCEFIELDS}
STRATUM_MAP = {
    "mono": "mono",
    "monomer": "mono",
    "di": "di",
    "dimer": "di",
    "tri": "tri",
    "trimer": "tri",
}
PROTOCOL_ID = "glycosenseformer-md-all200-heavy-kabsch-medoid-fps100-v1"
SCHEMA_VERSION = 1
N_CANDIDATES = 200
N_SELECTED = 100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BuildError(RuntimeError):
    """Base class for selection failures."""


class ManifestError(BuildError):
    """The common-source manifest is invalid or incomplete."""


class ValidationError(BuildError):
    """A source pickle or coordinate ensemble is invalid."""


class OutputConflictError(BuildError):
    """An existing destination is partial, stale, or inconsistent."""


@dataclass(frozen=True)
class SourceJob:
    ordinal: int
    molecule_ordinal: int
    molecule_id: str
    molecule_name: str
    stratum: str
    forcefield: str
    source_name: str
    source_pickle: str
    expected_source_sha256: str
    upstream_output_source: str
    n_heavy_atoms: int
    source_sdf: str
    source_sdf_sha256: str
    source_index: int
    fold: str
    manifest_path: str
    manifest_sha256: str


@dataclass(frozen=True)
class ArtifactPaths:
    selected_pickle: Path
    metadata: Path

    def all_paths(self) -> tuple[Path, Path]:
        return self.selected_pickle, self.metadata


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_molecule_id(value: str) -> None:
    if not value or value in {".", ".."}:
        raise ManifestError("molecule_id is empty or unsafe")
    if Path(value).name != value or "/" in value or "\\" in value or "\x00" in value:
        raise ManifestError(f"unsafe molecule_id for output path: {value!r}")


def artifact_paths(output_root: Path, job: SourceJob) -> ArtifactPaths:
    validate_molecule_id(job.molecule_id)
    if job.forcefield not in FORCEFIELDS:
        raise ManifestError(f"unsupported force field: {job.forcefield!r}")
    selected = output_root / job.forcefield / f"{job.molecule_id}.pickle"
    return ArtifactPaths(selected_pickle=selected, metadata=selected.with_suffix(".pickle.meta.json"))


def relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


# These three functions intentionally match build_glycosenseformer_etkdg_fps.py.
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


def fps_summary(distance_matrix: np.ndarray, selected: Sequence[int]) -> dict[str, Any]:
    selected_array = np.asarray(selected, dtype=np.int64)
    nearest = distance_matrix[:, selected_array].min(axis=1)
    triangle = distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)]
    medoid = selected[0]
    denominator = max(1, distance_matrix.shape[0] - 1)
    return {
        "selected_candidate_ids": [int(index + 1) for index in selected],
        "selected_candidate_indices_zero_based": [int(index) for index in selected],
        "medoid_candidate_id": int(medoid + 1),
        "medoid_mean_rmsd_angstrom": float(distance_matrix[medoid].sum() / denominator),
        "all_pair_rmsd_min_angstrom": float(triangle.min()) if triangle.size else 0.0,
        "all_pair_rmsd_mean_angstrom": float(triangle.mean()) if triangle.size else 0.0,
        "all_pair_rmsd_max_angstrom": float(triangle.max()) if triangle.size else 0.0,
        "unselected_to_nearest_selected_max_angstrom": float(nearest.max()),
        "unselected_to_nearest_selected_mean_angstrom": float(nearest.mean()),
    }


def _resolve_path(value: str, manifest: Path, label: str) -> Path:
    if not value.strip():
        raise ManifestError(f"missing {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (manifest.parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise ManifestError(f"{label} is not a file: {path}")
    return path


def _parse_positive_int(value: str, label: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid {label}: {value!r}") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise ManifestError(f"invalid {label}: {parsed}")
    return parsed


def load_jobs(manifest: Path, *, limit: int | None = None) -> list[SourceJob]:
    manifest = manifest.resolve()
    if not manifest.is_file():
        raise ManifestError(f"manifest does not exist: {manifest}")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestError(f"CSV manifest has no header: {manifest}")
        rows = [{str(key): str(value or "") for key, value in row.items()} for row in reader]
    if limit is not None:
        if limit < 0:
            raise ManifestError("--limit cannot be negative")
        rows = rows[:limit]
    if not rows:
        raise ManifestError("manifest selected no molecules")

    required = {
        "clean_name",
        "molecule_name",
        "source_class",
        "n_heavy_atoms",
        "source_sdf",
        "source_index",
        "fold",
        *(f"{forcefield}_output_source" for forcefield in FORCEFIELDS),
        *(f"{forcefield}_all200_path" for forcefield in FORCEFIELDS),
        *(f"{forcefield}_all200_sha256" for forcefield in FORCEFIELDS),
    }
    missing_columns = required - set(reader.fieldnames)
    if missing_columns:
        raise ManifestError(f"manifest lacks required columns: {sorted(missing_columns)}")

    manifest_sha = sha256_file(manifest)
    sdf_hash_cache: dict[Path, str] = {}
    seen: set[str] = set()
    jobs: list[SourceJob] = []
    ordinal = 0
    for molecule_ordinal, row in enumerate(rows, start=1):
        clean = row["clean_name"].strip()
        validate_molecule_id(clean)
        if clean in seen:
            raise ManifestError(f"duplicate clean_name in selected manifest: {clean}")
        seen.add(clean)
        molecule_name = row["molecule_name"].strip()
        if not molecule_name:
            raise ManifestError(f"row {molecule_ordinal}: molecule_name is empty")
        source_class = row["source_class"].strip().lower()
        if source_class not in STRATUM_MAP:
            raise ManifestError(f"row {molecule_ordinal}: unsupported source_class {source_class!r}")
        stratum = STRATUM_MAP[source_class]
        n_heavy = _parse_positive_int(row["n_heavy_atoms"], "n_heavy_atoms")
        source_sdf = _resolve_path(row["source_sdf"], manifest, "source_sdf")
        source_index = _parse_positive_int(row["source_index"], "source_index", allow_zero=True)
        fold = row["fold"].strip()
        try:
            fold_value = int(fold)
        except ValueError as exc:
            raise ManifestError(f"row {molecule_ordinal}: invalid fold {fold!r}") from exc
        if not 0 <= fold_value <= 4:
            raise ManifestError(f"row {molecule_ordinal}: fold must be in [0, 4]")
        if source_sdf not in sdf_hash_cache:
            sdf_hash_cache[source_sdf] = sha256_file(source_sdf)

        for forcefield in FORCEFIELDS:
            ordinal += 1
            source_pickle = _resolve_path(
                row[f"{forcefield}_all200_path"], manifest, f"{forcefield}_all200_path"
            )
            expected_hash = row[f"{forcefield}_all200_sha256"].strip()
            if SHA256_RE.fullmatch(expected_hash) is None:
                raise ManifestError(
                    f"row {molecule_ordinal}: invalid {forcefield}_all200_sha256"
                )
            upstream = row[f"{forcefield}_output_source"].strip()
            if not upstream:
                raise ManifestError(
                    f"row {molecule_ordinal}: missing {forcefield}_output_source"
                )
            jobs.append(
                SourceJob(
                    ordinal=ordinal,
                    molecule_ordinal=molecule_ordinal,
                    molecule_id=clean,
                    molecule_name=molecule_name,
                    stratum=stratum,
                    forcefield=forcefield,
                    source_name=SOURCE_NAMES[forcefield],
                    source_pickle=str(source_pickle),
                    expected_source_sha256=expected_hash,
                    upstream_output_source=upstream,
                    n_heavy_atoms=n_heavy,
                    source_sdf=str(source_sdf),
                    source_sdf_sha256=sdf_hash_cache[source_sdf],
                    source_index=source_index,
                    fold=str(fold_value),
                    manifest_path=str(manifest),
                    manifest_sha256=manifest_sha,
                )
            )
    return jobs


def load_source_candidates(job: SourceJob) -> list[np.ndarray]:
    source = Path(job.source_pickle)
    actual_hash = sha256_file(source)
    if actual_hash != job.expected_source_sha256:
        raise ValidationError(
            f"{job.forcefield}/{job.molecule_id}: source SHA256 mismatch; "
            f"expected {job.expected_source_sha256}, observed {actual_hash}"
        )
    try:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValidationError(f"cannot load source pickle {source}: {exc}") from exc
    if not isinstance(payload, (list, tuple)) or len(payload) != N_CANDIDATES:
        raise ValidationError(
            f"{source}: expected exactly {N_CANDIDATES} conformers"
        )
    candidates: list[np.ndarray] = []
    for index, value in enumerate(payload):
        array = np.asarray(value)
        if (
            array.shape != (job.n_heavy_atoms, 3)
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
        ):
            raise ValidationError(
                f"{source}: candidate {index + 1} is not a real numeric "
                f"({job.n_heavy_atoms}, 3) array"
            )
        converted = np.asarray(array, dtype=np.float64)
        if not np.isfinite(converted).all():
            raise ValidationError(f"{source}: candidate {index + 1} contains non-finite values")
        candidates.append(converted.copy())
    return candidates


def _write_pickle_fsynced(path: Path, coordinates: Sequence[np.ndarray]) -> None:
    with path.open("wb") as handle:
        pickle.dump(
            [np.asarray(item, dtype=np.float64) for item in coordinates],
            handle,
            protocol=4,
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_fsynced(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bundle(staging: Path, mapping: Sequence[tuple[Path, Path]]) -> None:
    created: list[Path] = []
    try:
        for source, destination in mapping:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            created.append(destination)
            _fsync_directory(destination.parent)
    except Exception:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_selected_output(path: Path, job: SourceJob) -> list[np.ndarray]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise OutputConflictError(f"cannot load selected output {path}: {exc}") from exc
    if not isinstance(payload, (list, tuple)) or len(payload) != N_SELECTED:
        raise OutputConflictError(f"{path}: expected {N_SELECTED} selected conformers")
    arrays: list[np.ndarray] = []
    for index, value in enumerate(payload):
        array = np.asarray(value)
        if (
            array.shape != (job.n_heavy_atoms, 3)
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
        ):
            raise OutputConflictError(f"{path}: selected conformer {index + 1} is invalid")
        converted = np.asarray(array, dtype=np.float64)
        if not np.isfinite(converted).all():
            raise OutputConflictError(f"{path}: selected conformer {index + 1} is non-finite")
        arrays.append(converted)
    return arrays


def validate_existing_bundle(
    output_root: Path,
    job: SourceJob,
    candidates: Sequence[np.ndarray],
    selected: Sequence[int],
) -> dict[str, Any] | None:
    paths = artifact_paths(output_root, job)
    existing = [path.exists() for path in paths.all_paths()]
    if not any(existing):
        return None
    if not all(existing):
        missing = [str(path) for path, present in zip(paths.all_paths(), existing) if not present]
        raise OutputConflictError(
            f"{job.forcefield}/{job.molecule_id}: partial existing output; "
            f"refusing overwrite; missing {missing}"
        )
    if paths.selected_pickle.is_symlink() or paths.metadata.is_symlink():
        raise OutputConflictError("selected output bundle contains a symlink")
    try:
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OutputConflictError(f"invalid commit metadata {paths.metadata}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise OutputConflictError(f"commit metadata is not an object: {paths.metadata}")

    expected_scalars = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": job.manifest_sha256,
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise OutputConflictError(
                f"{job.forcefield}/{job.molecule_id}: metadata {key} mismatch"
            )
    molecule = metadata.get("molecule")
    source = metadata.get("source")
    fps = metadata.get("fps")
    artifact = metadata.get("artifact")
    if not all(isinstance(value, dict) for value in (molecule, source, fps, artifact)):
        raise OutputConflictError("commit metadata lacks molecule/source/fps/artifact records")
    expected_molecule = {
        "molecule_id": job.molecule_id,
        "molecule_name": job.molecule_name,
        "stratum": job.stratum,
        "n_heavy_atoms": job.n_heavy_atoms,
        "fold": job.fold,
    }
    if any(molecule.get(key) != value for key, value in expected_molecule.items()):
        raise OutputConflictError(f"{job.forcefield}/{job.molecule_id}: molecule metadata mismatch")
    expected_source = {
        "forcefield": job.forcefield,
        "source_name": job.source_name,
        "upstream_output_source": job.upstream_output_source,
        "all200_path": str(Path(job.source_pickle).resolve()),
        "all200_sha256": job.expected_source_sha256,
        "conformer_count": N_CANDIDATES,
        "source_sdf": str(Path(job.source_sdf).resolve()),
        "source_sdf_sha256": job.source_sdf_sha256,
        "source_index": job.source_index,
    }
    if any(source.get(key) != value for key, value in expected_source.items()):
        raise OutputConflictError(f"{job.forcefield}/{job.molecule_id}: source metadata mismatch")
    expected_ids = [int(index + 1) for index in selected]
    expected_indices = [int(index) for index in selected]
    if (
        fps.get("selected_candidate_ids") != expected_ids
        or fps.get("selected_candidate_indices_zero_based") != expected_indices
        or fps.get("candidate_count") != N_CANDIDATES
        or fps.get("selected_count") != N_SELECTED
        or fps.get("algorithm") != "medoid-seeded farthest-point selection"
        or fps.get("distance") != "atom-ordered heavy-atom Kabsch RMSD in angstrom"
        or fps.get("tie_break") != "lower 1-based candidate_id"
    ):
        raise OutputConflictError(f"{job.forcefield}/{job.molecule_id}: FPS metadata mismatch")
    expected_relative = relative_to_root(paths.selected_pickle, output_root)
    if (
        artifact.get("path") != expected_relative
        or artifact.get("sha256") != sha256_file(paths.selected_pickle)
        or artifact.get("size_bytes") != paths.selected_pickle.stat().st_size
        or artifact.get("n_conformers") != N_SELECTED
    ):
        raise OutputConflictError(f"{job.forcefield}/{job.molecule_id}: output artifact mismatch")
    output = _load_selected_output(paths.selected_pickle, job)
    for position, candidate_index in enumerate(selected):
        if not np.array_equal(output[position], candidates[candidate_index]):
            raise OutputConflictError(
                f"{job.forcefield}/{job.molecule_id}: output position {position + 1} "
                "does not match its selected all200 candidate"
            )
    return metadata


def build_source_bundle(job: SourceJob, output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    paths = artifact_paths(output_root, job)
    source = Path(job.source_pickle).resolve()
    if source == paths.selected_pickle.resolve():
        raise OutputConflictError(
            f"refusing to overwrite source pickle with selected output: {source}"
        )

    candidates = load_source_candidates(job)
    distance_matrix = pairwise_kabsch_rmsd(candidates)
    selected = deterministic_farthest_point_indices(distance_matrix, N_SELECTED)
    existing = validate_existing_bundle(output_root, job, candidates, selected)
    if existing is not None:
        return result_row(job, paths, status="resumed", metadata=existing)

    staging = output_root / ".staging" / (
        f"{job.forcefield}-{job.molecule_id}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    staged_pickle = staging / "fps100.pickle"
    staged_metadata = staging / "fps100.pickle.meta.json"
    try:
        _write_pickle_fsynced(
            staged_pickle,
            [candidates[index] for index in selected],
        )
        summary = fps_summary(distance_matrix, selected)
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": utc_now(),
            "manifest": str(Path(job.manifest_path).resolve()),
            "manifest_sha256": job.manifest_sha256,
            "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            "molecule": {
                "molecule_id": job.molecule_id,
                "molecule_name": job.molecule_name,
                "stratum": job.stratum,
                "n_heavy_atoms": job.n_heavy_atoms,
                "fold": job.fold,
            },
            "source": {
                "forcefield": job.forcefield,
                "source_name": job.source_name,
                "upstream_output_source": job.upstream_output_source,
                "all200_path": str(source),
                "all200_sha256": job.expected_source_sha256,
                "conformer_count": N_CANDIDATES,
                "source_sdf": str(Path(job.source_sdf).resolve()),
                "source_sdf_sha256": job.source_sdf_sha256,
                "source_index": job.source_index,
            },
            "validation": {
                "source_hash_matches_manifest": True,
                "source_is_unchanged": True,
                "candidate_count_exact": True,
                "finite_coordinates": True,
                "atom_order_preserved": True,
                "selected_arrays_match_source_candidates": True,
            },
            "fps": {
                "algorithm": "medoid-seeded farthest-point selection",
                "distance": "atom-ordered heavy-atom Kabsch RMSD in angstrom",
                "tie_break": "lower 1-based candidate_id",
                "candidate_count": N_CANDIDATES,
                "selected_count": N_SELECTED,
                **summary,
            },
            "artifact": {
                "path": relative_to_root(paths.selected_pickle, output_root),
                "sha256": sha256_file(staged_pickle),
                "size_bytes": staged_pickle.stat().st_size,
                "n_conformers": N_SELECTED,
                "coordinate_dtype": "float64",
                "coordinate_unit": "angstrom",
            },
            "commit_marker": True,
        }
        _write_text_fsynced(
            staged_metadata,
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        _publish_bundle(
            staging,
            (
                (staged_pickle, paths.selected_pickle),
                (staged_metadata, paths.metadata),
            ),
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    validated = validate_existing_bundle(output_root, job, candidates, selected)
    if validated is None:  # pragma: no cover - defensive
        raise OutputConflictError("published bundle disappeared before validation")
    return result_row(job, paths, status="complete", metadata=validated)


def result_row(
    job: SourceJob,
    paths: ArtifactPaths,
    *,
    status: str,
    metadata: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "ordinal": job.ordinal,
        "molecule_ordinal": job.molecule_ordinal,
        "molecule_id": job.molecule_id,
        "molecule_name": job.molecule_name,
        "stratum": job.stratum,
        "forcefield": job.forcefield,
        "source": job.source_name,
        "status": status,
        "error": error,
        "source_all200_path": job.source_pickle,
        "source_all200_sha256": job.expected_source_sha256,
        "source_sdf": job.source_sdf,
        "source_index": job.source_index,
        "fold": job.fold,
        "conformer_path": str(paths.selected_pickle),
        "expected_conformer_sha256": (
            str(metadata["artifact"]["sha256"]) if metadata is not None else ""
        ),
        "expected_conformer_count": N_SELECTED if metadata is not None else "",
        "metadata_path": str(paths.metadata),
        "metadata_sha256": sha256_file(paths.metadata) if metadata is not None else "",
    }


RESULT_FIELDS = tuple(
    result_row(
        SourceJob(
            0, 0, "placeholder", "", "mono", "charmm", "charmm_fps100", "", "0" * 64,
            "", 1, "", "0" * 64, 0, "0", "", "0" * 64,
        ),
        ArtifactPaths(Path("placeholder.pickle"), Path("placeholder.pickle.meta.json")),
        status="",
    ).keys()
)


def _failed_result(job: SourceJob, output_root: Path, exc: BaseException) -> dict[str, Any]:
    return result_row(
        job,
        artifact_paths(output_root, job),
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
    )


def run_jobs(
    jobs: Sequence[SourceJob],
    *,
    output_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: dict[int, dict[str, Any]] = {}
    if workers == 1:
        for job in jobs:
            try:
                rows[job.ordinal] = build_source_bundle(job, output_root)
            except Exception as exc:
                rows[job.ordinal] = _failed_result(job, output_root, exc)
                print(
                    f"[failed] {job.forcefield}/{job.molecule_id}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[{rows[job.ordinal]['status']}] {job.forcefield}/{job.molecule_id}",
                    flush=True,
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(build_source_bundle, job, output_root): job for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    rows[job.ordinal] = future.result()
                except Exception as exc:
                    rows[job.ordinal] = _failed_result(job, output_root, exc)
                    print(
                        f"[failed] {job.forcefield}/{job.molecule_id}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if os.environ.get("GLYCOSENSEFORMER_MD_FPS_TRACEBACK") == "1":
                        traceback.print_exc()
                else:
                    print(
                        f"[{rows[job.ordinal]['status']}] {job.forcefield}/{job.molecule_id}",
                        flush=True,
                    )
    return [rows[job.ordinal] for job in jobs]


def _atomic_replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        _write_text_fsynced(temporary, text)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


LONG_FIELDS = (
    "molecule_id",
    "stratum",
    "source",
    "conformer_path",
    "source_sdf",
    "source_index",
    "fold",
    "expected_conformer_sha256",
    "expected_conformer_count",
)


def write_run_manifests(
    output_root: Path,
    *,
    rows: Sequence[dict[str, Any]],
    manifest: Path,
    workers: int,
    started_at: str,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    build_buffer = StringIO()
    writer = csv.DictWriter(build_buffer, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    build_path = output_root / "build_manifest.csv"
    _atomic_replace_text(build_path, build_buffer.getvalue())

    long_rows = [
        {key: row[key] for key in LONG_FIELDS}
        for row in rows
        if row["status"] in {"complete", "resumed"}
    ]
    long_buffer = StringIO()
    long_writer = csv.DictWriter(long_buffer, fieldnames=LONG_FIELDS, lineterminator="\n")
    long_writer.writeheader()
    long_writer.writerows(long_rows)
    long_path = output_root / "conformer_source_manifest.csv"
    _atomic_replace_text(long_path, long_buffer.getvalue())

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "output_root": str(output_root),
        "workers": workers,
        "candidate_count_per_source": N_CANDIDATES,
        "selected_count_per_source": N_SELECTED,
        "status_counts": counts,
        "source_job_count": len(rows),
        "molecule_count": len({str(row["molecule_id"]) for row in rows}),
        "build_manifest_csv": str(build_path),
        "build_manifest_csv_sha256": sha256_file(build_path),
        "conformer_source_manifest_csv": str(long_path),
        "conformer_source_manifest_csv_sha256": sha256_file(long_path),
        "conformer_source_manifest_rows": len(long_rows),
        "builder_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_replace_text(
        output_root / "build_manifest.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify CHARMM/GROMOS/GLYCAM all200 pickles and select deterministic "
            "atom-ordered heavy-atom Kabsch-RMSD FPS100 ensembles."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Frozen common-source CSV. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N molecules (pilot/resume helper).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Independent source worker processes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Isolated destination root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        manifest = args.manifest.resolve()
        output_root = args.output_root.resolve()
        jobs = load_jobs(manifest, limit=args.limit)
        started_at = utc_now()
        rows = run_jobs(jobs, output_root=output_root, workers=args.workers)
        summary = write_run_manifests(
            output_root,
            rows=rows,
            manifest=manifest,
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
