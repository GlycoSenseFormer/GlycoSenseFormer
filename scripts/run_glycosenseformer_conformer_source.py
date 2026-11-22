#!/usr/bin/env python3
"""Train/evaluate GlycoSenseFormer across conformer sources.

The input manifest is deliberately explicit: every molecule/source pair points to
its own conformer pickle and to one labelled PyG template per NMR task.  This
prevents a global conformer directory (or a stale dataset cache) from silently
mixing force fields.

Long-form CSV columns
---------------------
``molecule_id,stratum,source,conformer_path`` plus either
``template_13C_path,template_1H_path`` or ``source_sdf,source_index``.

``stratum`` must be one of ``mono``, ``di`` or ``tri``.  Paths are resolved
relative to the manifest.  A conformer pickle contains an ordered sequence of
``(n_atoms, 3)`` arrays.  The optional ``selection_path`` column points to a
JSON/CSV/pickle sequence of zero-based candidate indices; otherwise the first
``--max-confs`` entries are used.  A template pickle may directly contain a
``torch_geometric.data.Data`` object, a one-element sequence, or a mapping keyed
by ``13C``/``1H``.

The frozen one-row-per-molecule wide manifest is also accepted.  Every nonempty
``*_path`` conformer column is expanded to a source bearing that prefix; select
the intended sources explicitly because a legacy partial source (for example an
87-conformer MMFF entry) will correctly fail the default 100-conformer check.

The default backend is the publication GlycoSenseFormer architecture.  The
``tiny`` backend exists solely for CPU integration tests and requires the
explicit ``--allow-test-backend`` acknowledgement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "src"))
from module.model_names import GLYCOSENSEFORMER

ARCHITECTURE = GLYCOSENSEFORMER
TASKS = ("13C", "1H")
STRATA = ("mono", "di", "tri")
SAFE_SOURCE = re.compile(r"^[A-Za-z0-9_.-]+$")

PUBLICATION_MODEL_CONFIG: dict[str, Any] = {
    "architecture": ARCHITECTURE,
    "eqv3_num_layers": 7,
    "eqv3_num_channels": 128,
    "eqv3_attn_hidden_channels": 32,
    "eqv3_ffn_hidden_channels": 512,
    "eqv3_num_radial_basis": 64,
    "eqv3_lmax": 4,
    "eqv3_mmax": 2,
    "eqv3_max_radius": 6.0,
    "glycanaa_hidden_channels": 128,
    "glycanaa_num_layers": 4,
    "glycanaa_edge_hidden_channels": 64,
    "glycanaa_dropout": 0.1,
    "glycanaa_head_hidden_channels": 128,
    "glycanaa_num_relations": 6,
    "fusion_hidden_channels": 256,
    "fusion_dropout": 0.1,
    "fusion_zero_init": True,
}

PUBLICATION_TRAIN_CONFIG: dict[str, Any] = {
    "epochs": 10,
    "batch_size": 16,
    "lr": 3.0e-4,
    "lr_reduce": 0.8,
    "max_confs": 100,
}


@dataclass(frozen=True)
class ManifestRow:
    molecule_id: str
    stratum: str
    source: str
    conformer_path: Path
    template_13C_path: Path | None = None
    template_1H_path: Path | None = None
    source_sdf: Path | None = None
    source_index: int | None = None
    manifest_fold: int | None = None
    expected_conformer_sha256: str | None = None
    expected_conformer_count: int | None = None
    selection_path: Path | None = None

    def template_path(self, task: str) -> Path | None:
        if task == "13C":
            return self.template_13C_path
        if task == "1H":
            return self.template_1H_path
        raise ValueError(f"Unsupported task: {task}")


@dataclass(frozen=True)
class FoldAssignment:
    molecule_id: str
    stratum: str
    fold: int


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def resolve_manifest_path(manifest: Path, raw: str, field: str) -> Path:
    raw = raw.strip()
    if not raw:
        raise ValueError(f"Manifest field {field!r} cannot be empty")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {field} does not exist: {path}")
    return path


def _manifest_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            value = value.get("rows", value.get("records"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("JSON manifest must be a row list or contain a rows/records list")
        return value
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_manifest(path: Path) -> tuple[list[ManifestRow], str]:
    path = path.resolve()
    raw_hash = sha256_file(path)
    records = _manifest_records(path)
    if not records:
        raise ValueError("Manifest is empty")

    long_base = {
        "molecule_id",
        "stratum",
        "source",
        "conformer_path",
    }
    has_long_templates = {"template_13C_path", "template_1H_path"}.issubset(records[0])
    has_long_sdf = {"source_sdf", "source_index"}.issubset(records[0])
    is_long = long_base.issubset(records[0]) and (has_long_templates or has_long_sdf)
    wide_required = {"clean_name", "source_sdf", "source_index", "fold"}
    if not is_long and not wide_required.issubset(records[0]):
        missing = wide_required.difference(records[0])
        raise ValueError(
            "Manifest must use either the long schema or the frozen wide schema; "
            f"wide columns missing: {', '.join(sorted(missing))}"
        )

    rows: list[ManifestRow] = []
    seen: set[tuple[str, str]] = set()
    molecule_strata: dict[str, str] = {}
    expanded_records: list[dict[str, Any]] = []
    if is_long:
        expanded_records = records
    else:
        ignored_path_fields = {"source_sdf"}
        for record in records:
            conformer_fields = [
                key
                for key, value in record.items()
                if key.endswith("_path")
                and key not in ignored_path_fields
                and not any(
                    marker in key
                    for marker in ("selection", "indices", "metadata", "template", "manifest")
                )
                and str(value or "").strip()
            ]
            if not conformer_fields:
                raise ValueError(f"No conformer *_path fields for {record.get('clean_name')}")
            for field in conformer_fields:
                source = field[: -len("_path")]
                count_raw = str(record.get(f"{source}_count", "") or "").strip()
                expanded_records.append(
                    {
                        "molecule_id": record["clean_name"],
                        "stratum": record.get("stratum", record.get("source_class", "")),
                        "source": source,
                        "conformer_path": record[field],
                        "source_sdf": record["source_sdf"],
                        "source_index": record["source_index"],
                        "fold": record["fold"],
                        "expected_conformer_sha256": record.get(f"{source}_sha256", ""),
                        "expected_conformer_count": int(count_raw) if count_raw else None,
                        "selection_path": record.get(f"{source}_selection_path", ""),
                    }
                )

    stratum_aliases = {
        "mono": "mono",
        "monomer": "mono",
        "di": "di",
        "dimer": "di",
        "tri": "tri",
        "trimer": "tri",
    }
    for record in expanded_records:
        molecule_id = str(record.get("molecule_id", "")).strip()
        source = str(record.get("source", "")).strip()
        stratum_raw = str(record.get("stratum", "")).strip().lower()
        stratum = stratum_aliases.get(stratum_raw, stratum_raw)
        if not molecule_id:
            raise ValueError("molecule_id cannot be empty")
        if stratum not in STRATA:
            raise ValueError(f"Invalid stratum for {molecule_id}: {stratum!r}")
        if not SAFE_SOURCE.fullmatch(source):
            raise ValueError(f"Unsafe/empty source label for {molecule_id}: {source!r}")
        key = (molecule_id, source)
        if key in seen:
            raise ValueError(f"Duplicate molecule/source row: {key}")
        seen.add(key)
        if molecule_id in molecule_strata and molecule_strata[molecule_id] != stratum:
            raise ValueError(f"Molecule {molecule_id} appears in multiple strata")
        molecule_strata[molecule_id] = stratum
        selection_raw = str(record.get("selection_path", "") or "").strip()
        template_13c_raw = str(record.get("template_13C_path", "") or "").strip()
        template_1h_raw = str(record.get("template_1H_path", "") or "").strip()
        source_sdf_raw = str(record.get("source_sdf", "") or "").strip()
        source_index_raw = str(record.get("source_index", "") or "").strip()
        fold_raw = str(record.get("fold", "") or "").strip()
        expected_sha = str(
            record.get(
                "expected_conformer_sha256",
                record.get("conformer_sha256", record.get("sha256", "")),
            )
            or ""
        ).strip().lower()
        expected_count_raw = record.get(
            "expected_conformer_count",
            record.get("conformer_count", record.get("count")),
        )
        if expected_sha and not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise ValueError(f"Invalid expected SHA-256 for {molecule_id}/{source}")
        rows.append(
            ManifestRow(
                molecule_id=molecule_id,
                stratum=stratum,
                source=source,
                conformer_path=resolve_manifest_path(path, str(record["conformer_path"]), "conformer_path"),
                template_13C_path=(
                    resolve_manifest_path(path, template_13c_raw, "template_13C_path")
                    if template_13c_raw
                    else None
                ),
                template_1H_path=(
                    resolve_manifest_path(path, template_1h_raw, "template_1H_path")
                    if template_1h_raw
                    else None
                ),
                source_sdf=(
                    resolve_manifest_path(path, source_sdf_raw, "source_sdf")
                    if source_sdf_raw
                    else None
                ),
                source_index=int(source_index_raw) if source_index_raw else None,
                manifest_fold=int(fold_raw) if fold_raw else None,
                expected_conformer_sha256=expected_sha or None,
                expected_conformer_count=(
                    int(expected_count_raw)
                    if expected_count_raw is not None and str(expected_count_raw).strip()
                    else None
                ),
                selection_path=(resolve_manifest_path(path, selection_raw, "selection_path") if selection_raw else None),
            )
        )
        if (rows[-1].template_13C_path is None or rows[-1].template_1H_path is None) and (
            rows[-1].source_sdf is None or rows[-1].source_index is None
        ):
            raise ValueError(
                f"{molecule_id}/{source} needs template paths or source_sdf/source_index"
            )
    return rows, raw_hash


def select_sources(
    rows: Sequence[ManifestRow],
    train_sources: Sequence[str] | None,
    test_sources: Sequence[str] | None,
) -> tuple[list[str], list[str], list[str]]:
    available = sorted({row.source for row in rows})
    train = list(dict.fromkeys(train_sources or available))
    test = list(dict.fromkeys(test_sources or available))
    unknown = sorted(set(train + test).difference(available))
    if unknown:
        raise ValueError(f"Requested sources absent from manifest: {', '.join(unknown)}")
    selected = sorted(set(train + test))
    if not train or not test:
        raise ValueError("At least one train source and one test source are required")
    return train, test, selected


def strict_common_cohort(rows: Sequence[ManifestRow], sources: Sequence[str]) -> list[ManifestRow]:
    selected = [row for row in rows if row.source in sources]
    by_molecule: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        by_molecule[row.molecule_id].add(row.source)
    expected = set(sources)
    incomplete = {
        molecule_id: sorted(expected.difference(found))
        for molecule_id, found in by_molecule.items()
        if found != expected
    }
    if incomplete:
        sample = "; ".join(f"{key}: missing {','.join(value)}" for key, value in list(sorted(incomplete.items()))[:8])
        raise ValueError(f"Manifest is not a strict common-source cohort ({len(incomplete)} molecules): {sample}")
    if not by_molecule:
        raise ValueError("No molecules remain for selected sources")
    return selected


def limit_cohort(rows: Sequence[ManifestRow], limit: int | None) -> list[ManifestRow]:
    if limit is None:
        return list(rows)
    if limit < 2:
        raise ValueError("--limit must be at least 2")
    strata_ids: dict[str, list[str]] = {stratum: [] for stratum in STRATA}
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: (item.stratum, item.molecule_id)):
        if row.molecule_id not in seen:
            strata_ids[row.stratum].append(row.molecule_id)
            seen.add(row.molecule_id)
    chosen: list[str] = []
    cursor = 0
    while len(chosen) < min(limit, len(seen)):
        added = False
        for stratum in STRATA:
            if cursor < len(strata_ids[stratum]) and len(chosen) < limit:
                chosen.append(strata_ids[stratum][cursor])
                added = True
        if not added:
            break
        cursor += 1
    chosen_set = set(chosen)
    return [row for row in rows if row.molecule_id in chosen_set]


def assign_folds(rows: Sequence[ManifestRow], n_folds: int, split_seed: int) -> list[FoldAssignment]:
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2")
    ids_by_stratum: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ids_by_stratum[row.stratum].add(row.molecule_id)
    assignments: list[FoldAssignment] = []
    for stratum in STRATA:
        ids = sorted(ids_by_stratum[stratum])
        rng = random.Random(f"{split_seed}:{stratum}")
        rng.shuffle(ids)
        for index, molecule_id in enumerate(ids):
            assignments.append(FoldAssignment(molecule_id, stratum, index % n_folds))
    if len({item.molecule_id for item in assignments}) < 2:
        raise ValueError("At least two molecules are required")
    return sorted(assignments, key=lambda item: item.molecule_id)


def manifest_or_generated_folds(
    rows: Sequence[ManifestRow], n_folds: int, split_seed: int
) -> list[FoldAssignment]:
    folds_by_molecule: dict[str, set[int | None]] = defaultdict(set)
    strata: dict[str, str] = {}
    for row in rows:
        folds_by_molecule[row.molecule_id].add(row.manifest_fold)
        strata[row.molecule_id] = row.stratum
    has_manifest_folds = all(values != {None} for values in folds_by_molecule.values())
    if not has_manifest_folds:
        if any(values != {None} for values in folds_by_molecule.values()):
            raise ValueError("Manifest folds are partially specified")
        return assign_folds(rows, n_folds, split_seed)
    assignments: list[FoldAssignment] = []
    for molecule_id, values in sorted(folds_by_molecule.items()):
        if len(values) != 1:
            raise ValueError(f"Molecule {molecule_id} has source-dependent folds: {values}")
        fold = next(iter(values))
        assert fold is not None
        if fold < 0 or fold >= n_folds:
            raise ValueError(f"Manifest fold out of range for {molecule_id}: {fold}")
        assignments.append(FoldAssignment(molecule_id, strata[molecule_id], fold))
    return assignments


def assert_no_fold_leakage(assignments: Sequence[FoldAssignment], fold: int) -> tuple[set[str], set[str]]:
    test_ids = {item.molecule_id for item in assignments if item.fold == fold}
    train_ids = {item.molecule_id for item in assignments if item.fold != fold}
    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise RuntimeError(f"Molecule-level fold leakage: {sorted(overlap)[:5]}")
    if not train_ids or not test_ids:
        raise ValueError(f"Fold {fold} has an empty train or test partition")
    return train_ids, test_ids


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_selection(path: Path | None, n_candidates: int) -> list[int]:
    if path is None:
        return list(range(n_candidates))
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, dict):
            value = value.get("indices", value.get("selected_indices"))
    elif path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        if not csv_rows:
            value = []
        else:
            key = "candidate_index" if "candidate_index" in csv_rows[0] else "index"
            value = [row[key] for row in csv_rows]
    else:
        value = _load_pickle(path)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Selection must be an index sequence: {path}")
    indices = [int(item) for item in value]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Selection contains duplicate indices: {path}")
    if any(index < 0 or index >= n_candidates for index in indices):
        raise ValueError(f"Selection index out of range for {path}")
    return indices


def load_conformers(row: ManifestRow, max_confs: int, n_atoms: int | None = None) -> list[np.ndarray]:
    value = _load_pickle(row.conformer_path)
    if isinstance(value, dict):
        value = value.get("conformations", value.get("coords"))
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f"Conformer pickle is not a sequence: {row.conformer_path}")
    if row.expected_conformer_count is not None and len(value) != int(row.expected_conformer_count):
        raise ValueError(
            f"Conformer count mismatch for {row.molecule_id}/{row.source}: "
            f"expected {row.expected_conformer_count}, got {len(value)}"
        )
    indices = load_selection(row.selection_path, len(value))
    if len(indices) < max_confs:
        raise ValueError(
            f"{row.molecule_id}/{row.source} has only {len(indices)} selected conformers; "
            f"{max_confs} required"
        )
    conformers: list[np.ndarray] = []
    for index in indices[:max_confs]:
        coords = np.asarray(value[index], dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Invalid coordinate shape {coords.shape} in {row.conformer_path}")
        if n_atoms is not None and coords.shape[0] != n_atoms:
            raise ValueError(
                f"Atom count mismatch for {row.molecule_id}/{row.source}: "
                f"template={n_atoms}, coordinates={coords.shape[0]}"
            )
        if not np.isfinite(coords).all():
            raise ValueError(f"Non-finite coordinates in {row.conformer_path}, candidate {index}")
        conformers.append(coords)
    return conformers


def load_template(path: Path, task: str) -> Any:
    value = _load_pickle(path)
    if isinstance(value, Mapping):
        for key in (task, f"template_{task}", "data"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Template sequence must have exactly one item: {path}")
        value = value[0]
    required = ("x", "edge_index", "edge_attr")
    if not all(hasattr(value, key) for key in required):
        raise ValueError(f"Template is not a PyG-like Data object: {path}")
    if value.x.ndim != 2 or value.x.shape[1] < 3:
        raise ValueError(f"Template x must have [atom_type, hydrogen_count, label]: {path}")
    return value


def clean_molecule_name(name: str) -> str:
    return name.replace("->", "").replace("(", "").replace(")", "")


@lru_cache(maxsize=None)
def template_from_sdf(source_sdf: str, source_index: int, task: str) -> Any:
    from rdkit import Chem
    from scripts.build_glyconmr_len123_validation import mol_to_data

    supplier = Chem.SDMolSupplier(source_sdf, removeHs=True)
    if source_index < 0 or source_index >= len(supplier):
        raise IndexError(f"source_index {source_index} out of range for {source_sdf}")
    molecule = supplier[source_index]
    if molecule is None:
        raise ValueError(f"RDKit failed to read index {source_index} from {source_sdf}")
    return mol_to_data(
        molecule,
        task,
        float(PUBLICATION_MODEL_CONFIG["eqv3_max_radius"]),
    )


def load_row_template(row: ManifestRow, task: str) -> Any:
    path = row.template_path(task)
    if path is not None:
        return load_template(path, task)
    assert row.source_sdf is not None and row.source_index is not None
    template = template_from_sdf(str(row.source_sdf), row.source_index, task)
    template_name = getattr(template, "name", "")
    if clean_molecule_name(str(template_name)) != row.molecule_id:
        raise ValueError(
            f"SDF identity mismatch at {row.source_sdf}[{row.source_index}]: "
            f"expected {row.molecule_id}, got {template_name}"
        )
    return template


def target_mask(data: Any) -> Any:
    return (data.x[:, 0] == 2).logical_and(data.x[:, -1] > -0.5)


def template_signature(data: Any) -> str:
    import torch

    payload = {
        "x": torch.as_tensor(data.x).detach().cpu().tolist(),
        "edge_index": torch.as_tensor(data.edge_index).detach().cpu().tolist(),
        "edge_attr": torch.as_tensor(data.edge_attr).detach().cpu().tolist(),
    }
    return sha256_bytes(canonical_json(payload))


def rebuild_graph(data: Any, coords: np.ndarray, cutoff: float) -> Any:
    import torch
    from torch_geometric.utils import coalesce

    graph = data.clone()
    graph.pos = torch.as_tensor(coords, dtype=torch.float32)
    n_atoms = int(graph.x.shape[0])
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long)
    edge_attr = torch.as_tensor(graph.edge_attr, dtype=torch.long).flatten()
    bond_mask = edge_attr > 0
    bond_index = edge_index[:, bond_mask]
    bond_attr = edge_attr[bond_mask]

    atom_ids = torch.arange(n_atoms, dtype=torch.long)
    row = atom_ids.repeat_interleave(n_atoms)
    col = atom_ids.repeat(n_atoms)
    nonself = row != col
    radius_index = torch.stack((row[nonself], col[nonself]), dim=0)
    radius_attr = torch.zeros(radius_index.shape[1], dtype=torch.long)
    merged_index = torch.cat((bond_index, radius_index), dim=1)
    merged_attr = torch.cat((bond_attr, radius_attr), dim=0)
    merged_index, merged_attr = coalesce(merged_index, merged_attr, num_nodes=n_atoms, reduce="max")
    src, dst = merged_index
    distances = torch.linalg.vector_norm(graph.pos[src] - graph.pos[dst], dim=1)
    keep = distances <= cutoff
    graph.edge_index = merged_index[:, keep]
    graph.edge_attr = merged_attr[keep]
    return graph


def row_lookup(rows: Sequence[ManifestRow]) -> dict[tuple[str, str], ManifestRow]:
    return {(row.molecule_id, row.source): row for row in rows}


def validate_templates_and_conformers(
    rows: Sequence[ManifestRow], tasks: Sequence[str], max_confs: int
) -> dict[tuple[str, str], str]:
    signatures: dict[tuple[str, str], str] = {}
    reference: dict[tuple[str, str], str] = {}
    for row in rows:
        for task in tasks:
            template = load_row_template(row, task)
            if int(target_mask(template).sum()) <= 0:
                raise ValueError(f"No {task} targets for {row.molecule_id}")
            signature = template_signature(template)
            key = (row.molecule_id, task)
            if key in reference and reference[key] != signature:
                raise ValueError(
                    f"Topology/labels differ across sources for {row.molecule_id}/{task}; "
                    "cross-source evaluation would not be paired"
                )
            reference[key] = signature
            signatures[(row.molecule_id, task)] = signature
            load_conformers(row, max_confs, int(template.x.shape[0]))
    return signatures


def input_fingerprints(rows: Sequence[ManifestRow]) -> dict[str, str]:
    paths = {
        path
        for row in rows
        for path in (
            row.conformer_path,
            row.template_13C_path,
            row.template_1H_path,
            row.source_sdf,
            row.selection_path,
        )
        if path is not None
    }
    hashes = {str(path): sha256_file(path) for path in sorted(paths)}
    for row in rows:
        actual = hashes[str(row.conformer_path)]
        if row.expected_conformer_sha256 and actual != row.expected_conformer_sha256:
            raise ValueError(
                f"Conformer SHA-256 mismatch for {row.molecule_id}/{row.source}: "
                f"expected {row.expected_conformer_sha256}, got {actual}"
            )
    return hashes


def cache_key(
    *,
    input_hash: str,
    source: str,
    task: str,
    fold: int,
    partition: str,
    max_confs: int,
    cutoff: float,
    molecule_ids: Sequence[str],
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "schema": 1,
                "input_hash": input_hash,
                "source": source,
                "task": task,
                "fold": fold,
                "partition": partition,
                "max_confs": max_confs,
                "cutoff": cutoff,
                "molecule_ids": sorted(molecule_ids),
            }
        )
    )


def build_graphs(
    *,
    rows_by_key: Mapping[tuple[str, str], ManifestRow],
    molecule_ids: Sequence[str],
    source: str,
    task: str,
    max_confs: int,
    cutoff: float,
) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for molecule_id in sorted(molecule_ids):
        row = rows_by_key[(molecule_id, source)]
        template = load_row_template(row, task)
        conformers = load_conformers(row, max_confs, int(template.x.shape[0]))
        graphs: list[Any] = []
        for conformer_index, coords in enumerate(conformers):
            graph = rebuild_graph(template, coords, cutoff)
            graph.molecule_id = molecule_id
            graph.conformer_index = conformer_index
            graph.conformer_source = source
            graphs.append(graph)
        result[molecule_id] = graphs
    return result


def load_or_build_graphs(
    *,
    cache_root: Path,
    input_hash: str,
    rows_by_key: Mapping[tuple[str, str], ManifestRow],
    molecule_ids: Sequence[str],
    source: str,
    task: str,
    fold: int,
    partition: str,
    max_confs: int,
    cutoff: float,
) -> dict[str, list[Any]]:
    import torch

    key = cache_key(
        input_hash=input_hash,
        source=source,
        task=task,
        fold=fold,
        partition=partition,
        max_confs=max_confs,
        cutoff=cutoff,
        molecule_ids=molecule_ids,
    )
    cache_path = cache_root / source / f"fold_{fold}" / task / f"{partition}_{key}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("cache_key") != key or payload.get("source") != source:
            raise RuntimeError(f"Source-isolated cache metadata mismatch: {cache_path}")
        return payload["graphs"]
    graphs = build_graphs(
        rows_by_key=rows_by_key,
        molecule_ids=molecule_ids,
        source=source,
        task=task,
        max_confs=max_confs,
        cutoff=cutoff,
    )
    atomic_torch_save(cache_path, {"cache_key": key, "source": source, "graphs": graphs})
    return graphs


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(backend: str, equiformer_v3_root: Path) -> Any:
    import torch

    if backend == "tiny":
        class _TinyNMR(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.type_embedding = torch.nn.Embedding(16, 8)
                self.h_embedding = torch.nn.Embedding(8, 4)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(15, 24),
                    torch.nn.SiLU(),
                    torch.nn.Linear(24, 1),
                )

            def forward(self, x, pos, edge_index, edge_attr, batch):
                del edge_index, edge_attr
                counts = torch.bincount(batch, minlength=int(batch.max().item()) + 1).clamp_min(1)
                centers = torch.zeros((counts.numel(), 3), device=pos.device, dtype=pos.dtype)
                centers.index_add_(0, batch, pos)
                centers = centers / counts[:, None]
                local = pos - centers[batch]
                features = torch.cat(
                    (
                        self.type_embedding(x[:, 0].clamp(0, 15)),
                        self.h_embedding(x[:, 1].clamp(0, 7)),
                        local,
                    ),
                    dim=1,
                )
                return self.mlp(features)

        return _TinyNMR()

    from module.custom_models import build_nmr_model

    kwargs = dict(PUBLICATION_MODEL_CONFIG)
    architecture = kwargs.pop("architecture")
    kwargs["equiformer_v3_root"] = str(equiformer_v3_root)
    return build_nmr_model(architecture=architecture, **kwargs)


def forward_model(model: Any, data: Any) -> Any:
    return model(
        x=data.x[:, 0:2].long(),
        pos=data.pos.float(),
        edge_index=data.edge_index,
        edge_attr=data.edge_attr.long(),
        batch=data.batch,
    ).flatten()


def unique_target_stats(graphs: Mapping[str, Sequence[Any]]) -> tuple[float, float]:
    import torch

    labels = []
    for molecule_id in sorted(graphs):
        graph = graphs[molecule_id][0]
        labels.append(graph.x[:, -1][target_mask(graph)].float())
    merged = torch.cat(labels)
    mean = float(merged.mean())
    std = float(merged.std(unbiased=False))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Training target standard deviation must be positive")
    return mean, std


def checkpoint_identity(
    *,
    config_hash: str,
    manifest_hash: str,
    input_hash: str,
    train_source: str,
    fold: int,
    task: str,
    backend: str,
) -> dict[str, Any]:
    return {
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "input_hash": input_hash,
        "train_source": train_source,
        "fold": fold,
        "task": task,
        "backend": backend,
    }


def train_one(
    *,
    model: Any,
    train_graphs: Mapping[str, Sequence[Any]],
    device: Any,
    epochs: int,
    batch_size: int,
    lr: float,
    lr_reduce: float,
    checkpoint_dir: Path,
    identity: Mapping[str, Any],
    resume: bool,
    seed: int,
) -> tuple[Any, float, float, list[dict[str, Any]]]:
    import torch
    from torch_geometric.loader import DataLoader

    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_reduce)
    mean, std = unique_target_stats(train_graphs)
    start_epoch = 0
    losses: list[dict[str, Any]] = []
    latest = checkpoint_dir / "latest.pt"
    if resume and latest.is_file():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        got_identity = payload.get("identity")
        if got_identity != dict(identity):
            raise RuntimeError(f"Checkpoint identity mismatch at {latest}")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["next_epoch"])
        mean = float(payload["target_mean"])
        std = float(payload["target_std"])
        losses = list(payload.get("losses", []))

    flat_graphs = [graph for molecule_id in sorted(train_graphs) for graph in train_graphs[molecule_id]]
    criterion = torch.nn.L1Loss()
    for epoch in range(start_epoch, epochs):
        # Epoch-derived seeds make an interrupted/resumed run follow exactly the
        # same shuffle/dropout stream as an uninterrupted run.
        set_seed(seed + epoch)
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        loader = DataLoader(
            flat_graphs,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        model.train()
        batch_losses: list[float] = []
        epoch_started = time.perf_counter()
        n_batches = len(loader)
        print(
            json.dumps(
                {
                    "event": "train_epoch_started",
                    "train_source": identity.get("train_source"),
                    "task": identity.get("task"),
                    "epoch": epoch,
                    "n_batches": n_batches,
                    "batch_size": batch_size,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        for batch_index, data in enumerate(loader):
            batch_started = time.perf_counter()
            data = data.to(device)
            mask = target_mask(data)
            targets = (data.x[:, -1][mask].float() - mean) / std
            optimizer.zero_grad(set_to_none=True)
            prediction = forward_model(model, data)[mask]
            loss = criterion(prediction, targets)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            batch_losses.append(loss_value)
            if batch_index == 0 or (batch_index + 1) % 100 == 0 or batch_index + 1 == n_batches:
                print(
                    json.dumps(
                        {
                            "event": "train_batch_progress",
                            "train_source": identity.get("train_source"),
                            "task": identity.get("task"),
                            "epoch": epoch,
                            "batch": batch_index + 1,
                            "n_batches": n_batches,
                            "batch_seconds": time.perf_counter() - batch_started,
                            "epoch_seconds": time.perf_counter() - epoch_started,
                            "loss_normalized_mae": loss_value,
                            "num_graphs": int(data.num_graphs),
                            "num_nodes": int(data.num_nodes),
                            "num_edges": int(data.edge_index.shape[1]),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
        epoch_loss = float(np.mean(batch_losses))
        print(
            json.dumps(
                {
                    "event": "train_epoch_completed",
                    "train_source": identity.get("train_source"),
                    "task": identity.get("task"),
                    "epoch": epoch,
                    "n_batches": n_batches,
                    "epoch_seconds": time.perf_counter() - epoch_started,
                    "loss_normalized_mae": epoch_loss,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        losses.append(
            {
                **dict(identity),
                "epoch": epoch,
                "loss_normalized_mae": epoch_loss,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step()
        payload = {
            "schema_version": 1,
            "identity": dict(identity),
            "next_epoch": epoch + 1,
            "target_mean": mean,
            "target_std": std,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "losses": losses,
        }
        epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        atomic_torch_save(epoch_path, payload)
        atomic_torch_save(latest, payload)
    return model, mean, std, losses


def predict_molecule(
    *,
    model: Any,
    graphs: Sequence[Any],
    device: Any,
    mean: float,
    std: float,
    eval_batch_size: int,
) -> tuple[list[int], list[float], list[float]]:
    import torch
    from torch_geometric.loader import DataLoader

    if not graphs:
        raise ValueError("Cannot evaluate an empty conformer ensemble")
    reference_mask = target_mask(graphs[0])
    atom_indices = torch.where(reference_mask)[0].tolist()
    true = graphs[0].x[:, -1][reference_mask].float().detach().cpu()
    prediction_sum = torch.zeros_like(true)
    n_seen = 0
    model.eval()
    loader = DataLoader(list(graphs), batch_size=min(eval_batch_size, len(graphs)), shuffle=False)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            mask = target_mask(data)
            batch_prediction = forward_model(model, data)[mask] * std + mean
            n_graphs = int(data.num_graphs)
            if batch_prediction.numel() != n_graphs * true.numel():
                raise RuntimeError("Target atom count changed across conformers")
            # Equal arithmetic mean: every selected conformer has exactly 1/N weight.
            prediction_sum += batch_prediction.reshape(n_graphs, true.numel()).sum(dim=0).cpu()
            n_seen += n_graphs
    prediction = prediction_sum / n_seen
    return atom_indices, true.tolist(), prediction.tolist()


def prediction_rows(
    *,
    model: Any,
    graphs: Mapping[str, Sequence[Any]],
    assignments: Mapping[str, FoldAssignment],
    fold: int,
    task: str,
    train_source: str,
    test_source: str,
    device: Any,
    mean: float,
    std: float,
    eval_batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for molecule_id in sorted(graphs):
        atom_indices, true, prediction = predict_molecule(
            model=model,
            graphs=graphs[molecule_id],
            device=device,
            mean=mean,
            std=std,
            eval_batch_size=eval_batch_size,
        )
        n_confs = len(graphs[molecule_id])
        for atom_index, target, pred in zip(atom_indices, true, prediction):
            rows.append(
                {
                    "fold": fold,
                    "task": task,
                    "train_source": train_source,
                    "test_source": test_source,
                    "molecule_id": molecule_id,
                    "stratum": assignments[molecule_id].stratum,
                    "atom_index": atom_index,
                    "true_ppm": target,
                    "pred_ppm": pred,
                    "residual_ppm": pred - target,
                    "n_confs": n_confs,
                    "conformer_weight": 1.0 / n_confs,
                }
            )
    return rows


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def metrics_for_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    true = np.asarray([float(row["true_ppm"]) for row in rows], dtype=float)
    pred = np.asarray([float(row["pred_ppm"]) for row in rows], dtype=float)
    residual = pred - true
    absolute = np.abs(residual)
    molecule_abs: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, absolute):
        molecule_abs[str(row["molecule_id"])].append(float(value))
    molecule_mae = np.asarray([np.mean(values) for values in molecule_abs.values()])
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float("nan") if denominator == 0 else 1.0 - float(np.sum(residual**2)) / denominator
    return {
        "n_molecules": len(molecule_abs),
        "n_atoms": len(rows),
        "macro_molecule_mae_ppm": float(np.mean(molecule_mae)),
        "micro_mae_ppm": float(np.mean(absolute)),
        "rmse_ppm": float(np.sqrt(np.mean(residual**2))),
        "r2": r2,
        "median_absolute_error_ppm": float(np.median(absolute)),
        "p90_absolute_error_ppm": _quantile(absolute, 0.90),
        "p95_absolute_error_ppm": _quantile(absolute, 0.95),
    }


def aggregate_metrics(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        base = (row["fold"], row["task"], row["train_source"], row["test_source"])
        grouped[(*base, "all")].append(row)
        grouped[(*base, row["stratum"])].append(row)
    output: list[dict[str, Any]] = []
    for (fold, task, train_source, test_source, stratum), rows in sorted(grouped.items()):
        output.append(
            {
                "fold": fold,
                "task": task,
                "train_source": train_source,
                "test_source": test_source,
                "stratum": stratum,
                **metrics_for_group(rows),
            }
        )
    return output


def parse_folds(raw: Sequence[int] | None, n_folds: int) -> list[int]:
    folds = list(range(n_folds)) if raw is None else list(dict.fromkeys(raw))
    if not folds or any(fold < 0 or fold >= n_folds for fold in folds):
        raise ValueError(f"Folds must be in [0, {n_folds - 1}]")
    return folds


def resolved_device(raw: str) -> Any:
    import torch

    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Source-isolated GlycoSenseFormer conformer comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-sources", nargs="+", default=None)
    parser.add_argument("--test-sources", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=20260717)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=PUBLICATION_TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=PUBLICATION_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=PUBLICATION_TRAIN_CONFIG["lr"])
    parser.add_argument("--lr-reduce", type=float, default=PUBLICATION_TRAIN_CONFIG["lr_reduce"])
    parser.add_argument("--max-confs", type=int, default=PUBLICATION_TRAIN_CONFIG["max_confs"])
    parser.add_argument("--limit", type=int, default=None, help="Balanced molecule cap for a pilot")
    parser.add_argument("--pilot", action="store_true", help="Cap epochs=3, conformers=20, molecules=20")
    parser.add_argument("--backend", choices=("official", "tiny"), default="official")
    parser.add_argument("--allow-test-backend", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--equiformer-v3-root", type=Path, default=ROOT / "src/equiformer_v3")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate and snapshot only; do not train")
    return parser


def effective_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.pilot:
        args.epochs = min(args.epochs, 3)
        args.max_confs = min(args.max_confs, 20)
        args.limit = min(args.limit, 20) if args.limit is not None else 20
    if args.backend == "tiny" and not args.allow_test_backend:
        raise ValueError("--backend tiny is test-only; add --allow-test-backend explicitly")
    for key in ("epochs", "batch_size", "eval_batch_size", "max_confs"):
        if getattr(args, key) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.lr <= 0 or not 0 < args.lr_reduce <= 1:
        raise ValueError("--lr must be positive and --lr-reduce must be in (0, 1]")
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.equiformer_v3_root = args.equiformer_v3_root.resolve()
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args = effective_args(args)
    rows, manifest_hash = load_manifest(args.manifest)
    train_sources, test_sources, selected_sources = select_sources(
        rows, args.train_sources, args.test_sources
    )
    rows = strict_common_cohort(rows, selected_sources)
    rows = limit_cohort(rows, args.limit)
    assignments_list = manifest_or_generated_folds(rows, args.n_folds, args.split_seed)
    assignments = {item.molecule_id: item for item in assignments_list}
    folds = parse_folds(args.folds, args.n_folds)
    for fold in folds:
        assert_no_fold_leakage(assignments_list, fold)

    validate_templates_and_conformers(rows, args.tasks, args.max_confs)
    fingerprints = input_fingerprints(rows)
    input_hash = sha256_bytes(canonical_json(fingerprints))
    cohort = sorted(assignments)
    cohort_hash = sha256_bytes(canonical_json([asdict(assignments[key]) for key in cohort]))
    resolved_config = {
        "schema_version": 1,
        "scientific_backend": args.backend == "official",
        "backend": args.backend,
        "architecture": ARCHITECTURE,
        "publication_model_config": PUBLICATION_MODEL_CONFIG,
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "lr_reduce": args.lr_reduce,
            "seed": args.seed,
            "max_confs": args.max_confs,
            "ensemble_aggregation": "equal_arithmetic_mean",
        },
        "data": {
            "manifest": str(args.manifest),
            "manifest_hash": manifest_hash,
            "input_hash": input_hash,
            "cohort_hash": cohort_hash,
            "n_molecules": len(cohort),
            "train_sources": train_sources,
            "test_sources": test_sources,
            "tasks": list(args.tasks),
            "n_folds": args.n_folds,
            "folds": folds,
            "split_seed": args.split_seed,
            "cutoff_angstrom": PUBLICATION_MODEL_CONFIG["eqv3_max_radius"],
        },
        "runtime": {
            "device": str(args.device),
            "equiformer_v3_root": str(args.equiformer_v3_root),
            "resume": args.resume,
            "pilot": args.pilot,
            "limit": args.limit,
        },
    }
    config_hash = sha256_bytes(canonical_json(resolved_config))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.out_dir / "config.json",
        {
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "input_hash": input_hash,
            "cohort_hash": cohort_hash,
            "resolved_config": resolved_config,
        },
    )
    (args.out_dir / "manifest.sha256").write_text(f"{manifest_hash}  {args.manifest}\n")
    atomic_json(args.out_dir / "input_fingerprints.json", fingerprints)
    write_csv(
        args.out_dir / "fold_assignments.csv",
        [asdict(item) for item in assignments_list],
        ("molecule_id", "stratum", "fold"),
    )
    snapshot_rows = []
    for row in sorted(rows, key=lambda item: (item.molecule_id, item.source)):
        snapshot_rows.append(
            {
                "molecule_id": row.molecule_id,
                "stratum": row.stratum,
                "source": row.source,
                "conformer_path": str(row.conformer_path),
                "template_13C_path": str(row.template_13C_path or ""),
                "template_1H_path": str(row.template_1H_path or ""),
                "source_sdf": str(row.source_sdf or ""),
                "source_index": "" if row.source_index is None else row.source_index,
                "selection_path": str(row.selection_path or ""),
            }
        )
    write_csv(
        args.out_dir / "manifest_snapshot.csv",
        snapshot_rows,
        (
            "molecule_id",
            "stratum",
            "source",
            "conformer_path",
            "template_13C_path",
            "template_1H_path",
            "source_sdf",
            "source_index",
            "selection_path",
        ),
    )
    if args.dry_run:
        result = {
            "status": "validated",
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "input_hash": input_hash,
            "cohort_hash": cohort_hash,
            "n_molecules": len(cohort),
        }
        atomic_json(args.out_dir / "run_status.json", result)
        return result

    device = resolved_device(args.device)
    rows_by_key = row_lookup(rows)
    all_losses: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    cache_root = args.out_dir / "cache" / input_hash
    for fold in folds:
        train_ids, test_ids = assert_no_fold_leakage(assignments_list, fold)
        for task in args.tasks:
            for train_source in train_sources:
                train_graphs = load_or_build_graphs(
                    cache_root=cache_root,
                    input_hash=input_hash,
                    rows_by_key=rows_by_key,
                    molecule_ids=sorted(train_ids),
                    source=train_source,
                    task=task,
                    fold=fold,
                    partition="train",
                    max_confs=args.max_confs,
                    cutoff=PUBLICATION_MODEL_CONFIG["eqv3_max_radius"],
                )
                identity = checkpoint_identity(
                    config_hash=config_hash,
                    manifest_hash=manifest_hash,
                    input_hash=input_hash,
                    train_source=train_source,
                    fold=fold,
                    task=task,
                    backend=args.backend,
                )
                checkpoint_dir = (
                    args.out_dir / "checkpoints" / train_source / f"fold_{fold}" / task
                )
                # All conformer sources for a given formal seed start from the
                # same model initialization, making the source contrast paired.
                set_seed(args.seed)
                model, mean, std, losses = train_one(
                    model=build_model(args.backend, args.equiformer_v3_root),
                    train_graphs=train_graphs,
                    device=device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_reduce=args.lr_reduce,
                    checkpoint_dir=checkpoint_dir,
                    identity=identity,
                    resume=args.resume,
                    seed=args.seed,
                )
                all_losses.extend(losses)
                for test_source in test_sources:
                    test_graphs = load_or_build_graphs(
                        cache_root=cache_root,
                        input_hash=input_hash,
                        rows_by_key=rows_by_key,
                        molecule_ids=sorted(test_ids),
                        source=test_source,
                        task=task,
                        fold=fold,
                        partition="test",
                        max_confs=args.max_confs,
                        cutoff=PUBLICATION_MODEL_CONFIG["eqv3_max_radius"],
                    )
                    all_predictions.extend(
                        prediction_rows(
                            model=model,
                            graphs=test_graphs,
                            assignments=assignments,
                            fold=fold,
                            task=task,
                            train_source=train_source,
                            test_source=test_source,
                            device=device,
                            mean=mean,
                            std=std,
                            eval_batch_size=args.eval_batch_size,
                        )
                    )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    loss_fields = (
        "config_hash",
        "manifest_hash",
        "input_hash",
        "train_source",
        "fold",
        "task",
        "backend",
        "epoch",
        "loss_normalized_mae",
        "lr",
    )
    prediction_fields = (
        "config_hash",
        "manifest_hash",
        "input_hash",
        "fold",
        "task",
        "train_source",
        "test_source",
        "molecule_id",
        "stratum",
        "atom_index",
        "true_ppm",
        "pred_ppm",
        "residual_ppm",
        "n_confs",
        "conformer_weight",
    )
    # Resumed checkpoints return their full loss history; deduplicate deterministically.
    unique_losses = {
        (row["fold"], row["task"], row["train_source"], row["epoch"]): row
        for row in all_losses
    }
    losses_sorted = [unique_losses[key] for key in sorted(unique_losses)]
    predictions_sorted = sorted(
        all_predictions,
        key=lambda row: (
            row["fold"],
            row["task"],
            row["train_source"],
            row["test_source"],
            row["molecule_id"],
            row["atom_index"],
        ),
    )
    for row in predictions_sorted:
        row["config_hash"] = config_hash
        row["manifest_hash"] = manifest_hash
        row["input_hash"] = input_hash
    metrics = aggregate_metrics(predictions_sorted)
    for row in metrics:
        row["config_hash"] = config_hash
        row["manifest_hash"] = manifest_hash
        row["input_hash"] = input_hash
    metric_fields = (
        "config_hash",
        "manifest_hash",
        "input_hash",
        "fold",
        "task",
        "train_source",
        "test_source",
        "stratum",
        "n_molecules",
        "n_atoms",
        "macro_molecule_mae_ppm",
        "micro_mae_ppm",
        "rmse_ppm",
        "r2",
        "median_absolute_error_ppm",
        "p90_absolute_error_ppm",
        "p95_absolute_error_ppm",
    )
    write_csv(args.out_dir / "loss.csv", losses_sorted, loss_fields)
    write_csv(args.out_dir / "predictions.csv", predictions_sorted, prediction_fields)
    write_csv(args.out_dir / "metrics.csv", metrics, metric_fields)
    # JSON cannot represent NaN; keep undefined R2 explicit as null.
    json_metrics = [
        {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in row.items()}
        for row in metrics
    ]
    atomic_json(args.out_dir / "metrics.json", json_metrics)
    result = {
        "status": "complete",
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "input_hash": input_hash,
        "cohort_hash": cohort_hash,
        "n_molecules": len(cohort),
        "n_losses": len(losses_sorted),
        "n_predictions": len(predictions_sorted),
        "n_metric_rows": len(metrics),
        "scientific_backend": args.backend == "official",
    }
    atomic_json(args.out_dir / "run_status.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
