#!/usr/bin/env python3
"""Fail-closed transaction support for canonical force-field conformer outputs.

The module keeps its historical name for compatibility with the completed
CHARMM queue, while the transaction implementation is parameterized for the
CHARMM, GROMOS, and GLYCAM alignment queues.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CanonicalSpec:
    key: str
    commit_forcefield: str
    transaction_dirname: str
    all_dirname: str
    low_dirname: str


CHARMM_SPEC = CanonicalSpec(
    key="charmm",
    commit_forcefield="charmm36_exact",
    transaction_dirname="conformations_md_charmm36_exact_water_5ns_200cand_transactions_seed1",
    all_dirname="conformations_md_charmm36_exact_water_5ns_200cand_all_seed1",
    low_dirname="conformations_md_charmm36_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
)
GROMOS_SPEC = CanonicalSpec(
    key="gromos",
    commit_forcefield="gromos_exact",
    transaction_dirname="conformations_md_gromos_exact_water_5ns_200cand_transactions_seed1",
    all_dirname="conformations_md_gromos_exact_water_5ns_200cand_all_seed1",
    low_dirname="conformations_md_gromos_exact_water_5ns_200cand_lowest100_sugar_energy_seed1",
)
GLYCAM_SPEC = CanonicalSpec(
    key="glycam",
    commit_forcefield="glycam06j",
    transaction_dirname="conformations_md_glycam06j_water_5ns_200cand_transactions_seed1",
    all_dirname="conformations_md_glycam06j_water_5ns_200cand_all_seed1",
    low_dirname="conformations_md_glycam06j_water_5ns_200cand_lowest100_sugar_energy_seed1",
)
SPECS = {spec.key: spec for spec in (CHARMM_SPEC, GROMOS_SPEC, GLYCAM_SPEC)}

# Backwards-compatible constants used by the completed CHARMM scheduler/tests.
TRANSACTION_DIRNAME = CHARMM_SPEC.transaction_dirname
ALL_DIRNAME = CHARMM_SPEC.all_dirname
LOW_DIRNAME = CHARMM_SPEC.low_dirname


class CanonicalCommitError(RuntimeError):
    """Canonical output is incomplete, inconsistent, or has invalid provenance."""


@dataclass(frozen=True)
class CanonicalPaths:
    all_pickle: Path
    all_meta: Path
    low_pickle: Path
    low_meta: Path
    commit: Path
    lock: Path

    @property
    def artifacts(self) -> tuple[Path, ...]:
        return (self.all_pickle, self.all_meta, self.low_pickle, self.low_meta)


def paths_for(
    md_root: Path,
    clean_name: str,
    spec: CanonicalSpec = CHARMM_SPEC,
) -> CanonicalPaths:
    transaction_root = md_root / spec.transaction_dirname
    all_pickle = md_root / spec.all_dirname / f"{clean_name}.pickle"
    low_pickle = md_root / spec.low_dirname / f"{clean_name}.pickle"
    return CanonicalPaths(
        all_pickle=all_pickle,
        all_meta=all_pickle.with_suffix(all_pickle.suffix + ".meta.json"),
        low_pickle=low_pickle,
        low_meta=low_pickle.with_suffix(low_pickle.suffix + ".meta.json"),
        commit=transaction_root / f"{clean_name}.commit.json",
        lock=transaction_root / ".locks" / f"{clean_name}.lock",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def target_lock(
    md_root: Path,
    clean_name: str,
    spec: CanonicalSpec = CHARMM_SPEC,
) -> Iterator[Path]:
    """Serialize a molecule across schedulers, run roots, and direct runners."""

    lock_path = paths_for(md_root, clean_name, spec).lock
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} acquired={time.strftime('%F %T %Z')}\n")
        handle.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanonicalCommitError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CanonicalCommitError(f"{label} is not a JSON object: {path}")
    return value


def _validate_pickle(path: Path, *, expected_frames: int, expected_heavy_atoms: int) -> None:
    try:
        with path.open("rb") as handle:
            conformers = pickle.load(handle)
    except Exception as exc:
        raise CanonicalCommitError(f"cannot load canonical pickle {path}: {exc}") from exc
    if not isinstance(conformers, list) or len(conformers) != expected_frames:
        raise CanonicalCommitError(
            f"{path}: expected a list of {expected_frames} conformers, got "
            f"{type(conformers).__name__} with length "
            f"{len(conformers) if hasattr(conformers, '__len__') else 'unknown'}"
        )
    for index, conformer in enumerate(conformers):
        array = np.asarray(conformer)
        if array.shape != (expected_heavy_atoms, 3):
            raise CanonicalCommitError(
                f"{path}: conformer {index} has shape {array.shape}, expected "
                f"({expected_heavy_atoms}, 3)"
            )
        try:
            finite = np.isfinite(array).all()
        except TypeError as exc:
            raise CanonicalCommitError(f"{path}: conformer {index} is not numeric") from exc
        if not bool(finite):
            raise CanonicalCommitError(f"{path}: conformer {index} contains non-finite values")


def _validate_metadata(
    path: Path,
    *,
    expected_frames: int,
    expected_heavy_atoms: int,
    molecule_name: str,
    epoch_id: str,
    epoch_manifest_sha256: str,
    artifact_role: str,
    pickle_sha256: str,
) -> None:
    metadata = _load_json_object(path, "canonical pickle metadata")
    expected = {
        "n_frames": expected_frames,
        "n_heavy_atoms": expected_heavy_atoms,
        "molecule_name": molecule_name,
        "mapping_used": True,
        "epoch_id": epoch_id,
        "epoch_manifest_sha256": epoch_manifest_sha256,
        "artifact_role": artifact_role,
        "pickle_sha256": pickle_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise CanonicalCommitError(f"{path}: metadata mismatch: {mismatches}")


def _expected_artifact_paths(paths: CanonicalPaths, md_root: Path) -> dict[str, dict[str, object]]:
    return {
        "all": {
            "pickle": paths.all_pickle.relative_to(md_root).as_posix(),
            "metadata": paths.all_meta.relative_to(md_root).as_posix(),
            "n_frames": 200,
        },
        "low": {
            "pickle": paths.low_pickle.relative_to(md_root).as_posix(),
            "metadata": paths.low_meta.relative_to(md_root).as_posix(),
            "n_frames": 100,
        },
    }


def validate_commit(
    md_root: Path,
    clean_name: str,
    *,
    expected_molecule_name: str | None = None,
    expected_source_class: str | None = None,
    missing_ok: bool = True,
    spec: CanonicalSpec = CHARMM_SPEC,
) -> dict[str, object] | None:
    """Load every committed artifact; an uncommitted partial pair is an error."""

    paths = paths_for(md_root, clean_name, spec)
    if not paths.commit.exists():
        orphaned = [path for path in paths.artifacts if path.exists()]
        if orphaned:
            raise CanonicalCommitError(
                f"{clean_name}: canonical artifacts exist without a commit marker: "
                + ", ".join(str(path) for path in orphaned)
            )
        if missing_ok:
            return None
        raise CanonicalCommitError(f"{clean_name}: canonical commit is missing")

    commit = _load_json_object(paths.commit, "canonical commit")
    if commit.get("schema_version") != SCHEMA_VERSION:
        raise CanonicalCommitError(f"{clean_name}: unsupported canonical commit schema")
    if commit.get("clean_name") != clean_name:
        raise CanonicalCommitError(f"{clean_name}: commit identity mismatch")
    if commit.get("forcefield") != spec.commit_forcefield:
        raise CanonicalCommitError(f"{clean_name}: commit forcefield mismatch")
    if expected_molecule_name is not None and commit.get("molecule_name") != expected_molecule_name:
        raise CanonicalCommitError(f"{clean_name}: commit molecule_name mismatch")
    if expected_source_class is not None and commit.get("source_class") != expected_source_class:
        raise CanonicalCommitError(f"{clean_name}: commit source_class mismatch")
    epoch_id = commit.get("epoch_id")
    epoch_manifest_sha256 = commit.get("epoch_manifest_sha256")
    expected_heavy_atoms = commit.get("n_heavy_atoms")
    molecule_name = commit.get("molecule_name")
    if not isinstance(epoch_id, str) or len(epoch_id) != 64:
        raise CanonicalCommitError(f"{clean_name}: invalid epoch_id")
    if not isinstance(epoch_manifest_sha256, str) or len(epoch_manifest_sha256) != 64:
        raise CanonicalCommitError(f"{clean_name}: invalid epoch manifest SHA256")
    if not isinstance(expected_heavy_atoms, int) or expected_heavy_atoms < 1:
        raise CanonicalCommitError(f"{clean_name}: invalid heavy-atom count")
    if not isinstance(molecule_name, str) or not molecule_name:
        raise CanonicalCommitError(f"{clean_name}: invalid molecule name")

    artifact_rows = commit.get("artifacts")
    if not isinstance(artifact_rows, dict):
        raise CanonicalCommitError(f"{clean_name}: commit lacks artifact table")
    expected_paths = _expected_artifact_paths(paths, md_root)
    actual_paths = {
        "all": (paths.all_pickle, paths.all_meta),
        "low": (paths.low_pickle, paths.low_meta),
    }
    for role in ("all", "low"):
        row = artifact_rows.get(role)
        if not isinstance(row, dict):
            raise CanonicalCommitError(f"{clean_name}: commit lacks {role} artifact")
        expected = expected_paths[role]
        for key in ("pickle", "metadata", "n_frames"):
            if row.get(key) != expected[key]:
                raise CanonicalCommitError(f"{clean_name}: {role} {key} does not match canonical path")
        pickle_path, meta_path = actual_paths[role]
        for path in (pickle_path, meta_path):
            if not path.is_file():
                raise CanonicalCommitError(f"{clean_name}: committed artifact is missing: {path}")
        pickle_hash = sha256_file(pickle_path)
        metadata_hash = sha256_file(meta_path)
        if row.get("pickle_sha256") != pickle_hash or row.get("pickle_size") != pickle_path.stat().st_size:
            raise CanonicalCommitError(f"{clean_name}: {role} pickle hash/size mismatch")
        if row.get("metadata_sha256") != metadata_hash or row.get("metadata_size") != meta_path.stat().st_size:
            raise CanonicalCommitError(f"{clean_name}: {role} metadata hash/size mismatch")
        expected_frames = int(expected["n_frames"])
        _validate_pickle(
            pickle_path,
            expected_frames=expected_frames,
            expected_heavy_atoms=expected_heavy_atoms,
        )
        _validate_metadata(
            meta_path,
            expected_frames=expected_frames,
            expected_heavy_atoms=expected_heavy_atoms,
            molecule_name=molecule_name,
            epoch_id=epoch_id,
            epoch_manifest_sha256=epoch_manifest_sha256,
            artifact_role=role,
            pickle_sha256=pickle_hash,
        )
    return commit


def _copy_to_temporary(source: Path, destination: Path, transaction_id: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{transaction_id}.tmp"
    temporary.unlink(missing_ok=True)
    with source.open("rb") as source_handle, temporary.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    return temporary


def publish(
    md_root: Path,
    clean_name: str,
    *,
    molecule_name: str,
    source_class: str,
    n_heavy_atoms: int,
    epoch_id: str,
    epoch_manifest_sha256: str,
    staged_all_pickle: Path,
    staged_all_meta: Path,
    staged_low_pickle: Path,
    staged_low_meta: Path,
    spec: CanonicalSpec = CHARMM_SPEC,
) -> dict[str, object]:
    """Publish four files and atomically place the commit marker last.

    The caller must hold :func:`target_lock`.  A crash before the final commit
    leaves an explicitly uncommitted pair which all readers reject.
    """

    paths = paths_for(md_root, clean_name, spec)
    if validate_commit(
        md_root,
        clean_name,
        expected_molecule_name=molecule_name,
        expected_source_class=source_class,
        spec=spec,
    ) is not None:
        raise CanonicalCommitError(f"{clean_name}: refusing to overwrite a canonical commit")

    staged = {
        "all": (staged_all_pickle, staged_all_meta, 200),
        "low": (staged_low_pickle, staged_low_meta, 100),
    }
    for role, (pickle_path, meta_path, expected_frames) in staged.items():
        pickle_hash = sha256_file(pickle_path)
        _validate_pickle(
            pickle_path,
            expected_frames=expected_frames,
            expected_heavy_atoms=n_heavy_atoms,
        )
        _validate_metadata(
            meta_path,
            expected_frames=expected_frames,
            expected_heavy_atoms=n_heavy_atoms,
            molecule_name=molecule_name,
            epoch_id=epoch_id,
            epoch_manifest_sha256=epoch_manifest_sha256,
            artifact_role=role,
            pickle_sha256=pickle_hash,
        )

    transaction_id = f"{os.getpid()}-{time.time_ns()}"
    destinations = {
        "all_pickle": paths.all_pickle,
        "all_meta": paths.all_meta,
        "low_pickle": paths.low_pickle,
        "low_meta": paths.low_meta,
    }
    sources = {
        "all_pickle": staged_all_pickle,
        "all_meta": staged_all_meta,
        "low_pickle": staged_low_pickle,
        "low_meta": staged_low_meta,
    }
    temporary: dict[str, Path] = {}
    try:
        for label, destination in destinations.items():
            temporary[label] = _copy_to_temporary(sources[label], destination, transaction_id)
        for label, destination in destinations.items():
            os.replace(temporary[label], destination)
            _fsync_directory(destination.parent)

        artifacts = {
            "all": {
                "pickle": paths.all_pickle.relative_to(md_root).as_posix(),
                "pickle_sha256": sha256_file(paths.all_pickle),
                "pickle_size": paths.all_pickle.stat().st_size,
                "metadata": paths.all_meta.relative_to(md_root).as_posix(),
                "metadata_sha256": sha256_file(paths.all_meta),
                "metadata_size": paths.all_meta.stat().st_size,
                "n_frames": 200,
            },
            "low": {
                "pickle": paths.low_pickle.relative_to(md_root).as_posix(),
                "pickle_sha256": sha256_file(paths.low_pickle),
                "pickle_size": paths.low_pickle.stat().st_size,
                "metadata": paths.low_meta.relative_to(md_root).as_posix(),
                "metadata_sha256": sha256_file(paths.low_meta),
                "metadata_size": paths.low_meta.stat().st_size,
                "n_frames": 100,
            },
        }
        commit: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "forcefield": spec.commit_forcefield,
            "clean_name": clean_name,
            "molecule_name": molecule_name,
            "source_class": source_class,
            "n_heavy_atoms": n_heavy_atoms,
            "epoch_id": epoch_id,
            "epoch_manifest_sha256": epoch_manifest_sha256,
            "artifacts": artifacts,
        }
        atomic_write_json(paths.commit, commit)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)

    validated = validate_commit(
        md_root,
        clean_name,
        expected_molecule_name=molecule_name,
        expected_source_class=source_class,
        missing_ok=False,
        spec=spec,
    )
    assert validated is not None
    return validated
