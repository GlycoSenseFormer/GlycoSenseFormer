#!/usr/bin/env python3
"""Evaluate formal force-field checkpoints on the manuscript clean split.

This evaluator bridges the formal ``epoch_009.pt`` checkpoint format used by
``run_glycosenseformer_independent_validation.py`` and the one-conformer
``clean_split1`` validation pickles used by the manuscript.  It intentionally
evaluates only the four classical force-field sources and reports atom-level
micro metrics alongside molecule-macro MAE and error-tail statistics.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_glycosenseformer_conformer_source as core


DEFAULT_RUN_DIR = (
    ROOT
    / "experiments"
    / "glycosenseformer_conformer_source_v1"
    / "formal"
    / "full329_mmff94val100_seed4"
)
DEFAULT_DATASET_DIR = (
    ROOT / "data" / "datasets_glyconmr_len123_independent_validation_clean_split1"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "evaluation_clean_split1"
SOURCES = (
    "mmff94_fps100",
    "charmm_fps100",
    "gromos_fps100",
    "glycam_fps100",
)
TASKS = ("13C", "1H")
STRATUM_FILES = {"mono": "mo", "di": "di", "tri": "tri"}
EXPECTED_MOLECULES = {"mono": 9, "di": 44, "tri": 47}
TEST_SOURCE = "clean_split1_single_conformer"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate formal GlycoSenseFormer force-field checkpoints on clean_split1"
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "checkpoints/forcefield329/seed4")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/forcefield_eval")
    parser.add_argument("--equiformer-v3-root", type=Path, default=ROOT / "src/equiformer_v3")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def clean_item_graph(item: Mapping[str, Any], *, task: str, stratum: str) -> Any:
    graphs = getattr(item.get("loader"), "dataset", None)
    n_mols = int(item.get("n_mols", -1))
    n_nmr = int(item.get("n_nmr", -1))
    if graphs is None or n_mols != 1 or len(graphs) != 1:
        raise ValueError(
            f"{task}/{stratum}/{item.get('name')}: clean_split1 must contain exactly one "
            f"conformer, metadata={n_mols}, dataset={None if graphs is None else len(graphs)}"
        )
    graph = graphs[0]
    got_nmr = int(core.target_mask(graph).sum())
    if got_nmr <= 0 or got_nmr != n_nmr:
        raise ValueError(
            f"{task}/{stratum}/{item.get('name')}: target count mismatch, "
            f"metadata={n_nmr}, graph={got_nmr}"
        )
    return graph


def load_clean_validation(
    dataset_dir: Path, tasks: Sequence[str]
) -> tuple[dict[str, dict[str, list[tuple[str, Any]]]], dict[str, str]]:
    dataset_dir = dataset_dir.resolve()
    required = [dataset_dir / "summary.json", dataset_dir / "manifest.csv"]
    validation: dict[str, dict[str, list[tuple[str, Any]]]] = {}
    reference: dict[str, set[str]] | None = None
    for task in tasks:
        validation[task] = {}
        task_names: dict[str, set[str]] = {}
        for stratum, suffix in STRATUM_FILES.items():
            path = dataset_dir / f"test_data_{task}_{suffix}.pickle"
            required.append(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as handle:
                items = pickle.load(handle)
            if not isinstance(items, list) or len(items) != EXPECTED_MOLECULES[stratum]:
                raise ValueError(
                    f"{path}: expected {EXPECTED_MOLECULES[stratum]} molecules, "
                    f"got {None if not isinstance(items, list) else len(items)}"
                )
            records: list[tuple[str, Any]] = []
            names: set[str] = set()
            for item in items:
                if not isinstance(item, Mapping):
                    raise TypeError(f"{path}: validation item is not a mapping")
                name = str(item.get("name", "")).strip()
                if not name or name in names:
                    raise ValueError(f"{path}: empty or duplicate molecule name {name!r}")
                names.add(name)
                records.append((name, clean_item_graph(item, task=task, stratum=stratum)))
            validation[task][stratum] = records
            task_names[stratum] = names
        if reference is not None and task_names != reference:
            raise ValueError("13C and 1H clean_split1 molecule identities differ")
        reference = task_names
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing clean_split1 files: {missing}")
    fingerprints = {str(path): core.sha256_file(path) for path in sorted(set(required))}
    return validation, fingerprints


def checkpoint_path(run_dir: Path, source: str, task: str) -> Path:
    path = run_dir / "checkpoints" / source / task / "epoch_009.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_record(path: Path, source: str, task: str) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = payload.get("identity", {})
    if identity.get("train_source") != source or identity.get("task") != task:
        raise ValueError(f"Checkpoint identity mismatch: {path}: {identity}")
    if int(payload.get("next_epoch", -1)) != 10:
        raise ValueError(f"Checkpoint is not the completed 10-epoch model: {path}")
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError(f"Checkpoint lacks model_state: {path}")
    mean = float(payload.get("target_mean", float("nan")))
    std = float(payload.get("target_std", float("nan")))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid target normalization in {path}: mean={mean}, std={std}")
    return {
        "path": str(path.resolve()),
        "sha256": core.sha256_file(path),
        "identity": dict(identity),
        "target_mean": mean,
        "target_std": std,
    }


def evaluate_records(
    *,
    model: Any,
    records_by_stratum: Mapping[str, Sequence[tuple[str, Any]]],
    source: str,
    task: str,
    device: Any,
    mean: float,
    std: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    from torch_geometric.loader import DataLoader

    predictions: list[dict[str, Any]] = []
    model.eval()
    for stratum in STRATUM_FILES:
        records = list(records_by_stratum[stratum])
        graphs = [graph for _, graph in records]
        offset = 0
        with torch.no_grad():
            for data in DataLoader(graphs, batch_size=batch_size, shuffle=False):
                data = data.to(device)
                node_predictions = core.forward_model(model, data) * std + mean
                n_graphs = int(data.num_graphs)
                for local_graph in range(n_graphs):
                    start = int(data.ptr[local_graph])
                    stop = int(data.ptr[local_graph + 1])
                    local_x = data.x[start:stop]
                    mask = (local_x[:, 0] == 2).logical_and(local_x[:, -1] > -0.5)
                    atom_indices = torch.where(mask)[0].tolist()
                    true = local_x[:, -1][mask].detach().cpu().tolist()
                    pred = node_predictions[start:stop][mask].detach().cpu().tolist()
                    molecule_id = records[offset + local_graph][0]
                    for atom_index, target, prediction in zip(atom_indices, true, pred):
                        predictions.append(
                            {
                                "task": task,
                                "train_source": source,
                                "test_source": TEST_SOURCE,
                                "molecule_id": molecule_id,
                                "stratum": stratum,
                                "atom_index": atom_index,
                                "true_ppm": target,
                                "pred_ppm": prediction,
                                "residual_ppm": prediction - target,
                                "n_confs": 1,
                                "conformer_weight": 1.0,
                            }
                        )
                offset += n_graphs
        if offset != len(records):
            raise RuntimeError(f"Incomplete evaluation for {source}/{task}/{stratum}")
    return predictions


def aggregate_metrics(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        base = (str(row["task"]), str(row["train_source"]))
        grouped[(*base, "all")].append(row)
        grouped[(*base, str(row["stratum"]))].append(row)
    metrics = []
    for (task, source, stratum), rows in sorted(grouped.items()):
        metrics.append(
            {
                "task": task,
                "train_source": source,
                "test_source": TEST_SOURCE,
                "stratum": stratum,
                **core.metrics_for_group(rows),
            }
        )
    return metrics


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def write_summary(
    path: Path,
    metrics: Sequence[Mapping[str, Any]],
    config_hash: str,
    sources: Sequence[str],
    tasks: Sequence[str],
) -> None:
    source_names = {
        "mmff94_fps100": "MMFF94",
        "charmm_fps100": "CHARMM36",
        "gromos_fps100": "GROMOS",
        "glycam_fps100": "GLYCAM06j",
    }
    lookup = {
        (str(row["task"]), str(row["train_source"]), str(row["stratum"])): row
        for row in metrics
    }
    lines = [
        "# 四种力场 checkpoint 在 clean_split1 上的统一复评",
        "",
        "- 模型：GlycoSenseFormer formal seed 4 epoch 9",
        "- 验证集：clean_split1（100 个分子，每分子 1 个构象）",
        "- 主要口径：原子级 micro MAE、RMSE、R²",
        "- ETKDGv3-only：未纳入",
        f"- config hash：`{config_hash}`",
        "",
    ]
    for task in tasks:
        lines.extend(
            [
                f"## {task}",
                "",
                "| 力场 | 分层 | n分子 | n原子 | micro MAE | RMSE | R² | macro分子MAE |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for source in sources:
            for stratum in ("mono", "di", "tri", "all"):
                row = lookup[(task, source, stratum)]
                lines.append(
                    f"| {source_names[source]} | {stratum} | {int(row['n_molecules'])} | "
                    f"{int(row['n_atoms'])} | {float(row['micro_mae_ppm']):.6f} | "
                    f"{float(row['rmse_ppm']):.6f} | {float(row['r2']):.6f} | "
                    f"{float(row['macro_molecule_mae_ppm']):.6f} |"
                )
        lines.append("")
    atomic_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    run_dir = args.run_dir.resolve()
    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir.resolve()
    equiformer_v3_root = args.equiformer_v3_root.resolve()
    validation, validation_fingerprints = load_clean_validation(dataset_dir, args.tasks)
    checkpoint_records = []
    for source in args.sources:
        for task in args.tasks:
            checkpoint_records.append(
                checkpoint_record(checkpoint_path(run_dir, source, task), source, task)
            )
    config = {
        "schema_version": 1,
        "protocol": "formal_forcefields_seed4_epoch9_on_clean_split1_single_conformer",
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "architecture": core.PUBLICATION_MODEL_CONFIG["architecture"],
        "sources": list(args.sources),
        "tasks": list(args.tasks),
        "excluded_source": "etkdgv3_only_fps100",
        "eval_batch_size": args.eval_batch_size,
        "equiformer_v3_root": str(equiformer_v3_root),
        "evaluator_script": str(Path(__file__).resolve()),
        "evaluator_script_sha256": core.sha256_file(Path(__file__).resolve()),
        "validation_fingerprints": validation_fingerprints,
        "checkpoints": checkpoint_records,
    }
    config_hash = core.sha256_bytes(core.canonical_json(config))
    if args.dry_run:
        result = {
            "status": "validated",
            "config_hash": config_hash,
            "sources": list(args.sources),
            "tasks": list(args.tasks),
            "molecules": sum(EXPECTED_MOLECULES.values()),
            "checkpoint_count": len(checkpoint_records),
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return result

    device = core.resolved_device(args.device)
    all_predictions: list[dict[str, Any]] = []
    for checkpoint in checkpoint_records:
        source = str(checkpoint["identity"]["train_source"])
        task = str(checkpoint["identity"]["task"])
        print(
            json.dumps(
                {"event": "evaluation_started", "train_source": source, "task": task},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        model = core.build_model("official", equiformer_v3_root)
        model.load_state_dict(payload["model_state"])
        model = model.to(device)
        all_predictions.extend(
            evaluate_records(
                model=model,
                records_by_stratum=validation[task],
                source=source,
                task=task,
                device=device,
                mean=float(payload["target_mean"]),
                std=float(payload["target_std"]),
                batch_size=args.eval_batch_size,
            )
        )
        del model, payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {"event": "evaluation_completed", "train_source": source, "task": task},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    all_predictions.sort(
        key=lambda row: (
            row["task"],
            row["train_source"],
            row["stratum"],
            row["molecule_id"],
            row["atom_index"],
        )
    )
    metrics = aggregate_metrics(all_predictions)
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
    core.atomic_json(out_dir / "config.json", {"config_hash": config_hash, **config})
    core.write_csv(out_dir / "predictions.csv", all_predictions, prediction_fields)
    core.write_csv(out_dir / "metrics.csv", metrics, metric_fields)
    core.atomic_json(out_dir / "metrics.json", metrics)
    write_summary(
        out_dir / "summary.md",
        metrics,
        config_hash,
        args.sources,
        args.tasks,
    )
    result = {
        "status": "complete",
        "config_hash": config_hash,
        "n_sources": len(args.sources),
        "n_tasks": len(args.tasks),
        "n_predictions": len(all_predictions),
        "n_metric_rows": len(metrics),
        "dataset": TEST_SOURCE,
    }
    core.atomic_json(out_dir / "run_status.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
