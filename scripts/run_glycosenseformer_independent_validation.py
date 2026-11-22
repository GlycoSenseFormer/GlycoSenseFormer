#!/usr/bin/env python3
"""Train GlycoSenseFormer on all 329 source-matched sugars and validate externally.

This is the formal conformer-source protocol.  Each requested conformer source
trains a separate model on the same 329 labelled CASPER molecules.  There is no
cross-validation split: all 329 molecules are used for optimization for 10
epochs with 100 conformers per molecule.  The resulting model is evaluated on
the frozen 100-molecule GlycoNMR independent set, where every validation
molecule has 100 MMFF94 conformers.

The older ``run_glycosenseformer_conformer_source.py`` retains the paired
cross-source/pilot machinery and supplies the shared graph, model, training,
checkpoint, prediction, and metric primitives used here.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_glycosenseformer_conformer_source as core


EXPERIMENT_ROOT = ROOT / "experiments" / "glycosenseformer_conformer_source_v1"
DEFAULT_TRAIN_MANIFESTS = (
    ROOT / "data/mmff329/manifest.csv",
    ROOT / "md/fps100/manifest.csv",
)
DEFAULT_VALIDATION_DATASET = (
    ROOT / "data/forcefield_training_validation"
)
FORMAL_SOURCES = (
    "mmff94_fps100",
    "charmm_fps100",
    "gromos_fps100",
    "glycam_fps100",
)
VALIDATION_SOURCE = "independent_mmff94_100"
SPLIT_TO_FILE = {"mono": "mo", "di": "di", "tri": "tri"}
FORMAL_TRAIN_MOLECULES = 329
FORMAL_VALIDATION_MOLECULES = 100
FORMAL_CONFORMERS = 100
FORMAL_EPOCHS = 10


def load_training_manifests(
    paths: Sequence[Path],
) -> tuple[list[core.ManifestRow], str, list[dict[str, str]]]:
    if not paths:
        raise ValueError("At least one training manifest is required")
    rows: list[core.ManifestRow] = []
    seen: set[tuple[str, str]] = set()
    manifest_records: list[dict[str, str]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        loaded, digest = core.load_manifest(path)
        manifest_records.append({"path": str(path), "sha256": digest})
        for row in loaded:
            key = (row.molecule_id, row.source)
            if key in seen:
                raise ValueError(f"Duplicate molecule/source across training manifests: {key}")
            seen.add(key)
            rows.append(row)
    combined_hash = core.sha256_bytes(core.canonical_json(manifest_records))
    return rows, combined_hash, manifest_records


def validation_paths(dataset_dir: Path, tasks: Sequence[str]) -> list[Path]:
    paths = [dataset_dir / "summary.json", dataset_dir / "manifest.csv"]
    for task in tasks:
        for file_split in SPLIT_TO_FILE.values():
            paths.append(dataset_dir / f"test_data_{task}_{file_split}.pickle")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Independent-validation files are missing: {missing}")
    return paths


def load_and_audit_validation(
    dataset_dir: Path,
    tasks: Sequence[str],
    *,
    expected_molecules: int = FORMAL_VALIDATION_MOLECULES,
    expected_conformers: int = FORMAL_CONFORMERS,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, str]]:
    dataset_dir = dataset_dir.resolve()
    paths = validation_paths(dataset_dir, tasks)
    summary = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))
    if int(summary.get("selected_total", -1)) != expected_molecules:
        raise ValueError(
            "Independent validation molecule count mismatch: "
            f"expected {expected_molecules}, got {summary.get('selected_total')}"
        )
    if int(summary.get("validation_confs", -1)) != expected_conformers:
        raise ValueError(
            "Independent validation conformer count mismatch: "
            f"expected {expected_conformers}, got {summary.get('validation_confs')}"
        )
    forcefield_counts = summary.get("forcefield_counts", {})
    expected_forcefield_key = f"MMFF94:{expected_conformers}"
    if int(forcefield_counts.get(expected_forcefield_key, -1)) != expected_molecules:
        raise ValueError(
            "Independent validation is not the frozen MMFF94 ensemble: "
            f"expected {expected_forcefield_key} for {expected_molecules} molecules"
        )

    datasets: dict[str, dict[str, list[dict[str, Any]]]] = {}
    reference_names: set[str] | None = None
    for task in tasks:
        datasets[task] = {}
        task_names: set[str] = set()
        for stratum, file_split in SPLIT_TO_FILE.items():
            path = dataset_dir / f"test_data_{task}_{file_split}.pickle"
            with path.open("rb") as handle:
                items = pickle.load(handle)
            if not isinstance(items, list):
                raise ValueError(f"Validation pickle must contain a list: {path}")
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(f"Validation item must be a mapping: {path}")
                name = str(item.get("name", "")).strip()
                n_confs = int(item.get("n_mols", -1))
                n_nmr = int(item.get("n_nmr", -1))
                loader = item.get("loader")
                graphs = getattr(loader, "dataset", None)
                if not name or name in task_names:
                    raise ValueError(f"Empty or duplicate validation molecule name: {name!r}")
                if n_confs != expected_conformers or graphs is None or len(graphs) != expected_conformers:
                    raise ValueError(
                        f"{task}/{name}: expected {expected_conformers} conformers, "
                        f"metadata={n_confs}, dataset={None if graphs is None else len(graphs)}"
                    )
                if n_nmr <= 0:
                    raise ValueError(f"{task}/{name}: no labelled NMR targets")
                task_names.add(name)
            datasets[task][stratum] = items
        if len(task_names) != expected_molecules:
            raise ValueError(
                f"{task}: expected {expected_molecules} independent molecules, got {len(task_names)}"
            )
        if reference_names is not None and task_names != reference_names:
            raise ValueError("13C and 1H independent-validation molecule identities differ")
        reference_names = task_names

    fingerprints = {str(path.resolve()): core.sha256_file(path) for path in sorted(paths)}
    return datasets, fingerprints


def independent_prediction_rows(
    *,
    model: Any,
    validation: Mapping[str, Sequence[Mapping[str, Any]]],
    task: str,
    train_source: str,
    device: Any,
    mean: float,
    std: float,
    eval_batch_size: int,
    validation_limit: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = 0
    for stratum in core.STRATA:
        for item in validation[stratum]:
            if validation_limit is not None and seen >= validation_limit:
                return rows
            graphs = list(item["loader"].dataset)
            atom_indices, true, prediction = core.predict_molecule(
                model=model,
                graphs=graphs,
                device=device,
                mean=mean,
                std=std,
                eval_batch_size=eval_batch_size,
            )
            molecule_id = str(item["name"])
            n_confs = len(graphs)
            for atom_index, target, pred in zip(atom_indices, true, prediction):
                rows.append(
                    {
                        "task": task,
                        "train_source": train_source,
                        "test_source": VALIDATION_SOURCE,
                        "molecule_id": molecule_id,
                        "stratum": stratum,
                        "atom_index": atom_index,
                        "true_ppm": target,
                        "pred_ppm": pred,
                        "residual_ppm": pred - target,
                        "n_confs": n_confs,
                        "conformer_weight": 1.0 / n_confs,
                    }
                )
            seen += 1
    return rows


def aggregate_independent_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        base = (row["task"], row["train_source"], row["test_source"])
        grouped[(*base, "all")].append(row)
        grouped[(*base, row["stratum"])].append(row)
    output: list[dict[str, Any]] = []
    for (task, train_source, test_source, stratum), rows in sorted(grouped.items()):
        output.append(
            {
                "task": task,
                "train_source": train_source,
                "test_source": test_source,
                "stratum": stratum,
                **core.metrics_for_group(rows),
            }
        )
    return output


def load_or_build_full_train_graphs(
    *,
    cache_root: Path,
    input_hash: str,
    rows_by_key: Mapping[tuple[str, str], core.ManifestRow],
    molecule_ids: Sequence[str],
    source: str,
    task: str,
    max_confs: int,
    cutoff: float,
) -> dict[str, list[Any]]:
    import torch

    key = core.sha256_bytes(
        core.canonical_json(
            {
                "schema": 1,
                "protocol": "full_train_no_fold",
                "input_hash": input_hash,
                "source": source,
                "task": task,
                "max_confs": max_confs,
                "cutoff": cutoff,
                "molecule_ids": sorted(molecule_ids),
            }
        )
    )
    cache_path = cache_root / source / task / f"full_train_{key}.pt"
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            payload.get("cache_key") != key
            or payload.get("source") != source
            or payload.get("partition") != "full_train_no_fold"
        ):
            raise RuntimeError(f"Full-training cache metadata mismatch: {cache_path}")
        return payload["graphs"]
    graphs = core.build_graphs(
        rows_by_key=rows_by_key,
        molecule_ids=molecule_ids,
        source=source,
        task=task,
        max_confs=max_confs,
        cutoff=cutoff,
    )
    core.atomic_torch_save(
        cache_path,
        {
            "cache_key": key,
            "source": source,
            "partition": "full_train_no_fold",
            "graphs": graphs,
        },
    )
    return graphs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train on all common329 molecules and validate on MMFF94-100 GlycoNMR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifests", nargs="+", type=Path, default=list(DEFAULT_TRAIN_MANIFESTS))
    parser.add_argument("--validation-dataset-dir", type=Path, default=DEFAULT_VALIDATION_DATASET)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-sources", nargs="+", default=list(FORMAL_SOURCES))
    parser.add_argument("--tasks", nargs="+", choices=core.TASKS, default=list(core.TASKS))
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=core.PUBLICATION_TRAIN_CONFIG["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=core.PUBLICATION_TRAIN_CONFIG["lr"])
    parser.add_argument("--lr-reduce", type=float, default=core.PUBLICATION_TRAIN_CONFIG["lr_reduce"])
    parser.add_argument("--max-confs", type=int, default=FORMAL_CONFORMERS)
    parser.add_argument("--backend", choices=("official", "tiny"), default="official")
    parser.add_argument("--allow-test-backend", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--equiformer-v3-root", type=Path, default=ROOT / "src/equiformer_v3")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Smoke only: 20 train molecules, 20 conformers, 3 epochs, 6 validation molecules")
    parser.add_argument("--limit", type=int, default=None, help="Training molecule cap; formal runs forbid this")
    parser.add_argument("--validation-limit", type=int, default=None, help="Validation molecule cap; formal runs forbid this")
    return parser


def effective_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.pilot:
        args.epochs = min(args.epochs, 3)
        args.max_confs = min(args.max_confs, 20)
        args.limit = min(args.limit, 20) if args.limit is not None else 20
        args.validation_limit = (
            min(args.validation_limit, 6) if args.validation_limit is not None else 6
        )
    if args.backend == "tiny" and not args.allow_test_backend:
        raise ValueError("--backend tiny is test-only; add --allow-test-backend explicitly")
    for key in ("epochs", "batch_size", "eval_batch_size", "max_confs"):
        if int(getattr(args, key)) <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least 2")
    if args.validation_limit is not None and args.validation_limit < 1:
        raise ValueError("--validation-limit must be positive")
    if args.lr <= 0 or not 0 < args.lr_reduce <= 1:
        raise ValueError("--lr must be positive and --lr-reduce must be in (0, 1]")
    args.manifests = [path.resolve() for path in args.manifests]
    args.validation_dataset_dir = args.validation_dataset_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.equiformer_v3_root = args.equiformer_v3_root.resolve()
    return args


def assert_formal_protocol(
    args: argparse.Namespace,
    rows: Sequence[core.ManifestRow],
    sources: Sequence[str],
) -> None:
    if args.pilot or args.backend == "tiny":
        return
    molecules = {row.molecule_id for row in rows}
    if tuple(sources) != FORMAL_SOURCES:
        raise ValueError(f"Formal run requires sources in this order: {FORMAL_SOURCES}")
    if len(molecules) != FORMAL_TRAIN_MOLECULES:
        raise ValueError(
            f"Formal run requires {FORMAL_TRAIN_MOLECULES} common training molecules; "
            f"got {len(molecules)}"
        )
    if args.epochs != FORMAL_EPOCHS or args.max_confs != FORMAL_CONFORMERS:
        raise ValueError(
            f"Formal run requires epochs={FORMAL_EPOCHS} and max_confs={FORMAL_CONFORMERS}"
        )
    if args.limit is not None or args.validation_limit is not None:
        raise ValueError("Formal run forbids training or validation molecule limits")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    args = effective_args(args)
    all_rows, manifest_hash, manifest_records = load_training_manifests(args.manifests)
    sources = list(dict.fromkeys(args.train_sources))
    unknown = sorted(set(sources).difference({row.source for row in all_rows}))
    if unknown:
        raise ValueError(f"Requested training sources absent from manifests: {', '.join(unknown)}")
    rows = core.strict_common_cohort(all_rows, sources)
    assert_formal_protocol(args, rows, sources)
    rows = core.limit_cohort(rows, args.limit)
    molecule_ids = sorted({row.molecule_id for row in rows})

    core.validate_templates_and_conformers(rows, args.tasks, args.max_confs)
    train_fingerprints = core.input_fingerprints(rows)
    validation_data, validation_fingerprints = load_and_audit_validation(
        args.validation_dataset_dir,
        args.tasks,
    )
    train_input_hash = core.sha256_bytes(core.canonical_json(train_fingerprints))
    validation_hash = core.sha256_bytes(core.canonical_json(validation_fingerprints))
    combined_input_hash = core.sha256_bytes(
        core.canonical_json(
            {"training": train_input_hash, "validation": validation_hash}
        )
    )
    resolved_config = {
        "schema_version": 1,
        "protocol": "common329_full_train_independent_mmff94_100_validation",
        "scientific_backend": args.backend == "official",
        "backend": args.backend,
        "architecture": core.ARCHITECTURE,
        "publication_model_config": core.PUBLICATION_MODEL_CONFIG,
        "train": {
            "molecule_count": len(molecule_ids),
            "split": "none_all_molecules_train",
            "sources": sources,
            "tasks": list(args.tasks),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_reduce": args.lr_reduce,
            "seed": args.seed,
            "max_confs": args.max_confs,
            "ensemble_aggregation": "equal_arithmetic_mean",
        },
        "validation": {
            "dataset_dir": str(args.validation_dataset_dir),
            "source": VALIDATION_SOURCE,
            "molecule_count": FORMAL_VALIDATION_MOLECULES,
            "conformers_per_molecule": FORMAL_CONFORMERS,
            "validation_limit": args.validation_limit,
        },
        "inputs": {
            "manifests": manifest_records,
            "manifest_hash": manifest_hash,
            "training_input_hash": train_input_hash,
            "validation_hash": validation_hash,
            "combined_input_hash": combined_input_hash,
        },
        "runtime": {
            "device": args.device,
            "equiformer_v3_root": str(args.equiformer_v3_root),
            "resume": args.resume,
            "pilot": args.pilot,
            "limit": args.limit,
        },
    }
    config_hash = core.sha256_bytes(core.canonical_json(resolved_config))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    core.atomic_json(
        args.out_dir / "config.json",
        {"config_hash": config_hash, "resolved_config": resolved_config},
    )
    core.atomic_json(args.out_dir / "training_input_fingerprints.json", train_fingerprints)
    core.atomic_json(args.out_dir / "validation_input_fingerprints.json", validation_fingerprints)
    snapshot = [
        {
            "molecule_id": row.molecule_id,
            "stratum": row.stratum,
            "source": row.source,
            "conformer_path": str(row.conformer_path),
            "source_sdf": str(row.source_sdf or ""),
            "source_index": "" if row.source_index is None else row.source_index,
            "expected_conformer_sha256": row.expected_conformer_sha256 or "",
            "expected_conformer_count": row.expected_conformer_count or "",
        }
        for row in sorted(rows, key=lambda item: (item.molecule_id, item.source))
    ]
    core.write_csv(
        args.out_dir / "training_manifest_snapshot.csv",
        snapshot,
        (
            "molecule_id",
            "stratum",
            "source",
            "conformer_path",
            "source_sdf",
            "source_index",
            "expected_conformer_sha256",
            "expected_conformer_count",
        ),
    )
    if args.dry_run:
        result = {
            "status": "validated",
            "protocol": resolved_config["protocol"],
            "config_hash": config_hash,
            "n_train_molecules": len(molecule_ids),
            "n_train_sources": len(sources),
            "n_validation_molecules": FORMAL_VALIDATION_MOLECULES,
        }
        core.atomic_json(args.out_dir / "run_status.json", result)
        return result

    device = core.resolved_device(args.device)
    rows_by_key = core.row_lookup(rows)
    cache_root = args.out_dir / "cache"
    all_losses: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    work_items = [(train_source, task) for train_source in sources for task in args.tasks]
    # A restarted formal run should resume unfinished optimization before spending
    # hours re-evaluating source/task models whose final epoch is already durable.
    # Python's stable sort preserves the configured scientific order within the
    # unfinished and completed groups, so this changes only recovery latency.
    final_epoch_name = f"epoch_{args.epochs - 1:03d}.pt"
    work_items.sort(
        key=lambda item: (
            args.out_dir / "checkpoints" / item[0] / item[1] / final_epoch_name
        ).is_file()
    )
    print(
        json.dumps(
            {
                "event": "resume_work_queue",
                "items": [
                    {
                        "train_source": train_source,
                        "task": task,
                        "final_checkpoint_present": (
                            args.out_dir
                            / "checkpoints"
                            / train_source
                            / task
                            / final_epoch_name
                        ).is_file(),
                    }
                    for train_source, task in work_items
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    for train_source, task in work_items:
        train_graphs = load_or_build_full_train_graphs(
            cache_root=cache_root,
            input_hash=combined_input_hash,
            rows_by_key=rows_by_key,
            molecule_ids=molecule_ids,
            source=train_source,
            task=task,
            max_confs=args.max_confs,
            cutoff=core.PUBLICATION_MODEL_CONFIG["eqv3_max_radius"],
        )
        identity = {
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "input_hash": combined_input_hash,
            "protocol": resolved_config["protocol"],
            "train_source": train_source,
            "task": task,
            "backend": args.backend,
        }
        # Paired initialization: every source/task run with the same seed
        # starts from the same deterministic parameter initialization.
        core.set_seed(args.seed)
        model, mean, std, losses = core.train_one(
            model=core.build_model(args.backend, args.equiformer_v3_root),
            train_graphs=train_graphs,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_reduce=args.lr_reduce,
            checkpoint_dir=args.out_dir / "checkpoints" / train_source / task,
            identity=identity,
            resume=args.resume,
            seed=args.seed,
        )
        all_losses.extend(losses)
        all_predictions.extend(
            independent_prediction_rows(
                model=model,
                validation=validation_data[task],
                task=task,
                train_source=train_source,
                device=device,
                mean=mean,
                std=std,
                eval_batch_size=args.eval_batch_size,
                validation_limit=args.validation_limit,
            )
        )
        del model, train_graphs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated_before = int(torch.cuda.memory_allocated(device))
            reserved_before = int(torch.cuda.memory_reserved(device))
            torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "event": "cuda_task_cleanup",
                        "train_source": train_source,
                        "task": task,
                        "allocated_bytes": allocated_before,
                        "reserved_bytes_before_empty_cache": reserved_before,
                        "reserved_bytes_after_empty_cache": int(
                            torch.cuda.memory_reserved(device)
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    unique_losses = {
        (row["train_source"], row["task"], row["epoch"]): row for row in all_losses
    }
    losses = [unique_losses[key] for key in sorted(unique_losses)]
    predictions = sorted(
        all_predictions,
        key=lambda row: (
            row["task"],
            row["train_source"],
            row["molecule_id"],
            row["atom_index"],
        ),
    )
    metrics = aggregate_independent_metrics(predictions)
    loss_fields = (
        "config_hash",
        "manifest_hash",
        "input_hash",
        "protocol",
        "train_source",
        "task",
        "backend",
        "epoch",
        "loss_normalized_mae",
        "lr",
    )
    prediction_fields = (
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
    metric_fields = (
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
    core.write_csv(args.out_dir / "loss.csv", losses, loss_fields)
    core.write_csv(args.out_dir / "independent_predictions.csv", predictions, prediction_fields)
    core.write_csv(args.out_dir / "independent_metrics.csv", metrics, metric_fields)
    json_metrics = [
        {
            key: (None if isinstance(value, float) and math.isnan(value) else value)
            for key, value in row.items()
        }
        for row in metrics
    ]
    core.atomic_json(args.out_dir / "independent_metrics.json", json_metrics)
    result = {
        "status": "complete",
        "protocol": resolved_config["protocol"],
        "config_hash": config_hash,
        "n_train_molecules": len(molecule_ids),
        "n_train_sources": len(sources),
        "n_validation_molecules": (
            args.validation_limit or FORMAL_VALIDATION_MOLECULES
        ),
        "n_losses": len(losses),
        "n_predictions": len(predictions),
        "n_metric_rows": len(metrics),
        "scientific_backend": args.backend == "official",
    }
    core.atomic_json(args.out_dir / "run_status.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
