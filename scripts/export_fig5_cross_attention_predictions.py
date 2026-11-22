#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from module.custom_data import load_train_test_data
from module.custom_models import build_nmr_model
from module.custom_tasks.nmr_shift import build_model_kwargs, prepare_dataloader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets_glyconmr_len123_independent_validation_clean_split1"
DEFAULT_FIG_DIR = ROOT / "data/cases"
DEFAULT_SELECTION_CSV = DEFAULT_FIG_DIR / "fig5_representative_validation_molecules_selected.csv"
DEFAULT_OUT_CSV = DEFAULT_FIG_DIR / "fig5_cross_attention_seed0_2_3_prediction_values.csv"
DEFAULT_OUT_MD = DEFAULT_FIG_DIR / "fig5_cross_attention_seed0_2_3_prediction_values.md"

MODEL_GROUP = "New cross-attn learnable alpha bs16"
MODEL_NAME = "EquiformerV3-GlycanAAIdentityResidualCrossAttentionFusion"

RUN_DIR_BY_SEED = {
    0: "configs/original_runs/full_100conf_eqv3_glycanaa_cross_attention_identity_residual_fusion_learnable_alpha_fold0_epochs10_bs16_seed0_20260607",
    2: "configs/original_runs/full_100conf_eqv3_glycanaa_cross_attention_identity_residual_fusion_learnable_alpha_fold0_epochs10_bs16_seed2_20260608",
    3: "configs/original_runs/full_100conf_eqv3_glycanaa_cross_attention_identity_residual_fusion_learnable_alpha_fold0_epochs10_bs16_seed3_20260608",
}

TRUE_MD_BY_PANEL = {
    "S1": "S1_alpha_D_ManpNAc_true_1H_13C_shifts.md",
    "S2": "S2_alpha_D_Ribf_true_1H_13C_shifts.md",
    "S3": "S3_beta_D_Glcp_GODESS_2028_true_1H_13C_shifts.md",
    "S4": "S4_beta_D_Glcp_1-4_beta_D_Glcp_true_1H_13C_shifts.md",
    "S5": "S5_beta_D_GlcpNAc_1-3_beta_D_Galp_true_1H_13C_shifts.md",
    "S6": "S6_beta_D_Xylp_1-3_beta_L_Arap_true_1H_13C_shifts.md",
    "S7": "S7_beta_D_Xylp_1-4_alpha_L_Fucp_OMe_true_1H_13C_shifts.md",
    "S8": "S8_alpha_D_Glcp_1-2_beta_D_Fruf_true_1H_13C_shifts.md",
    "S9": "S9_Galb3_Fuca4_GlcNAcb_true_1H_13C_shifts.md",
    "S10": "S10_beta_D_Galp_1-4_D_Manp_1-4_D_Manp_OMe_true_1H_13C_shifts.md",
    "S11": "S11_D_Glcp_1-3_D_Xylp_1-3_D_Xylp_true_1H_13C_shifts.md",
    "S12": "S12_alpha_L_Fucp_1-N2_beta_D_Galp_1-3_D_GlcN_true_1H_13C_shifts.md",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Fig.5 selected molecule predictions for the cross-attention model.")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--selection_csv", default=str(DEFAULT_SELECTION_CSV))
    parser.add_argument("--out_csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out_md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mean_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_args(run_dir: Path) -> argparse.Namespace:
    with (run_dir / "config.json").open() as handle:
        return argparse.Namespace(**json.load(handle))


def checkpoint_path(run_dir: Path, fold: int, task: str, epoch: int) -> Path:
    path = run_dir / "checkpoints" / f"fold_{fold}" / task / f"_checkpoint_epoch_{epoch}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def target_mask(data) -> torch.Tensor:
    carbon_mask = data.x[:, 0] == 2.0
    nmr_mask = data.x[:, -1] > -0.5
    return nmr_mask.logical_and(carbon_mask)


def forward_model(model: torch.nn.Module, data) -> torch.Tensor:
    return model(
        x=data.x[:, 0:2].long(),
        pos=data.pos.float(),
        edge_index=data.edge_index,
        edge_attr=data.edge_attr.long(),
        batch=data.batch,
    )


def load_validation_task(dataset_dir: Path, task: str, mean_batch_size: int, device: torch.device):
    train_set, test_mo, test_di, test_tri = load_train_test_data(
        dataset_dir / f"train_data_{task}.pickle",
        dataset_dir / f"test_data_{task}_mo.pickle",
        dataset_dir / f"test_data_{task}_di.pickle",
        dataset_dir / f"test_data_{task}_tri.pickle",
    )
    values = []
    for data in prepare_dataloader(train_set, mean_batch_size):
        values.append(data.x[:, -1][target_mask(data)])
    all_values = torch.cat(values)
    return all_values.mean().to(device), all_values.std().to(device), {"mono": test_mo, "di": test_di, "tri": test_tri}


def predict_molecule(model: torch.nn.Module, loader_dict, mean: torch.Tensor, std: torch.Tensor, device: torch.device, eval_batch_size: int):
    n_confs = int(loader_dict["n_mols"])
    n_nmr = int(loader_dict["n_nmr"])
    loader = DataLoader(loader_dict["loader"].dataset, batch_size=eval_batch_size, shuffle=False)
    true_values = None
    conf_outputs = []
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            mask = target_mask(data)
            nmr_true = data.x[:, -1][mask]
            out = forward_model(model, data)
            out_masked = out[mask] * std + mean
            if true_values is None:
                true_values = nmr_true[:n_nmr].detach().cpu()
            conf_outputs.append(out_masked.detach().flatten().cpu())
            del data, out, out_masked
    if true_values is None:
        return [], []
    pred_values = torch.cat(conf_outputs).reshape(n_confs, n_nmr).T.mean(dim=1).detach().cpu()
    return true_values.tolist(), pred_values.tolist()


def load_model(run_dir: Path, task: str, fold: int, epoch: int, device: torch.device) -> tuple[torch.nn.Module, Path]:
    args = load_args(run_dir)
    model_kwargs = build_model_kwargs(args)
    architecture = model_kwargs.pop("architecture", "equiformer_v3")
    model = build_nmr_model(architecture=architecture, **model_kwargs).to(device)
    checkpoint = checkpoint_path(run_dir, fold, task, epoch)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    return model, checkpoint


def split_sdf_path(dataset_dir: Path, split: str) -> Path:
    suffix = {"mono": "mo", "di": "di", "tri": "tri"}[split]
    return dataset_dir / f"glyconmr_len123_independent_validation_{suffix}.sdf"


def load_target_labels(dataset_dir: Path, selections: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    by_split: dict[str, set[int]] = defaultdict(set)
    for row in selections:
        by_split[row["split"]].add(int(row["sample_idx"]))

    labels: dict[tuple[str, str], list[dict[str, object]]] = {}
    for split, indices in by_split.items():
        supplier = Chem.SDMolSupplier(str(split_sdf_path(dataset_dir, split)), removeHs=False)
        for sample_idx in sorted(indices):
            mol = supplier[sample_idx]
            if mol is None:
                raise RuntimeError(f"Could not read SDF molecule {split} index {sample_idx}")
            heavy_names = ast.literal_eval(mol.GetProp("heavy_atom_names"))
            residues = ast.literal_eval(mol.GetProp("heavy_residues"))
            for task in ("13C", "1H"):
                spectrum = ast.literal_eval(mol.GetProp(f"{task} Spectrum"))
                task_labels = []
                for order, (heavy_idx, true_ppm) in enumerate(spectrum.items()):
                    heavy_idx = int(heavy_idx)
                    atom_name = heavy_names[heavy_idx]
                    residue = residues[heavy_idx]
                    target_label = f"{residue}:{atom_name}" if residue else atom_name
                    if task == "1H":
                        target_label = f"{target_label} attached H"
                    task_labels.append(
                        {
                            "atom_order": order,
                            "heavy_atom_index": heavy_idx,
                            "atom_name": atom_name,
                            "residue": residue,
                            "target_label": target_label,
                            "sdf_true_ppm": float(true_ppm[0] if isinstance(true_ppm, list) else true_ppm),
                            "sdf_name": mol.GetProp("_Name"),
                        }
                    )
                labels[(split, str(sample_idx), task)] = task_labels
    return labels


def export_predictions(args: argparse.Namespace) -> list[dict[str, object]]:
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    selection_csv = Path(args.selection_csv)
    if not selection_csv.is_absolute():
        selection_csv = ROOT / selection_csv
    selections = sorted(read_csv(selection_csv), key=lambda r: int(r["panel"].lstrip("S")))
    labels = load_target_labels(dataset_dir, selections)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)

    validation_cache = {}
    rows: list[dict[str, object]] = []
    for task in ("13C", "1H"):
        print(f"Loading validation data for {task}", flush=True)
        mean, std, split_data = load_validation_task(dataset_dir, task, args.mean_batch_size, device)
        validation_cache[task] = (mean, std, split_data)
        for seed, run_dir_text in RUN_DIR_BY_SEED.items():
            run_dir = ROOT / run_dir_text
            print(f"Predicting {task}: seed {seed}", flush=True)
            model, checkpoint = load_model(run_dir, task, args.fold, args.epoch, device)
            for selection in selections:
                split = selection["split"]
                sample_idx = int(selection["sample_idx"])
                loader_dict = split_data[split][sample_idx]
                true_values, pred_values = predict_molecule(model, loader_dict, mean, std, device, args.eval_batch_size)
                task_labels = labels[(split, str(sample_idx), task)]
                if len(true_values) != len(task_labels):
                    raise RuntimeError(
                        f"Target count mismatch for {selection['panel']} {task}: "
                        f"pred {len(true_values)} labels {len(task_labels)}"
                    )
                for order, (true_ppm, pred_ppm) in enumerate(zip(true_values, pred_values)):
                    label = task_labels[order]
                    rows.append(
                        {
                            "panel": selection["panel"],
                            "display_name": selection["display_name"],
                            "source_name": selection["source_name"],
                            "split": split,
                            "sample_idx": sample_idx,
                            "task": task,
                            "seed": seed,
                            "atom_order": order,
                            "heavy_atom_index": label["heavy_atom_index"],
                            "atom_name": label["atom_name"],
                            "residue": label["residue"],
                            "target_label": label["target_label"],
                            "true_ppm": float(true_ppm),
                            "pred_ppm": float(pred_ppm),
                            "residual_ppm": float(pred_ppm - true_ppm),
                            "model_group": MODEL_GROUP,
                            "model_name": MODEL_NAME,
                            "checkpoint": str(checkpoint.relative_to(ROOT)),
                            "true_md": TRUE_MD_BY_PANEL.get(selection["panel"], ""),
                            "sdf_name": label["sdf_name"],
                        }
                    )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows


def fmt(value: float) -> str:
    return f"{value:.3f}"


def mae(values: list[float]) -> float:
    return sum(abs(value) for value in values) / len(values) if values else float("nan")


def markdown_link(label: str, path: Path) -> str:
    return f"[{label}]({path})"


def build_markdown(rows: list[dict[str, object]], out_md: Path) -> str:
    fig_dir = out_md.parent
    panel_order = sorted({str(row["panel"]) for row in rows}, key=lambda panel: int(panel.lstrip("S")))
    seeds = sorted({int(row["seed"]) for row in rows})
    lines = [
        "# EquiformerV3-GlycanAA Cross-attention seed0/2/3 prediction values",
        "",
        f"- Model group: `{MODEL_GROUP}`",
        f"- Model class: `{MODEL_NAME}`",
        "- Dataset: `data/datasets_glyconmr_len123_independent_validation_clean_split1`",
        "- Checkpoint: fold 0, epoch 9; predictions are averaged over the conformer ensemble used by validation evaluation.",
        "- Unit: ppm. `err` means `prediction - true`.",
        "- Only assigned validation targets are listed. Rows marked `未赋值` in the individual true-shift files are not model targets and therefore have no prediction.",
        "",
        "## Per-panel MAE summary",
        "",
        "| Panel | Molecule | split/index | 13C n | 13C MAE s0 | 13C MAE s2 | 13C MAE s3 | 1H n | 1H MAE s0 | 1H MAE s2 | 1H MAE s3 | True-shift table |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    by_panel_task_seed: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    first_by_panel: dict[str, dict[str, object]] = {}
    for row in rows:
        by_panel_task_seed[(str(row["panel"]), str(row["task"]), int(row["seed"]))].append(row)
        first_by_panel.setdefault(str(row["panel"]), row)

    for panel in panel_order:
        first = first_by_panel[panel]
        true_md = str(first.get("true_md", ""))
        true_md_text = markdown_link(true_md, fig_dir / true_md) if true_md else ""
        values = []
        for task in ("13C", "1H"):
            task_rows = by_panel_task_seed[(panel, task, seeds[0])]
            values.append(str(len(task_rows)))
            for seed in seeds:
                err = [float(row["residual_ppm"]) for row in by_panel_task_seed[(panel, task, seed)]]
                values.append(fmt(mae(err)))
        lines.append(
            f"| {panel} | {first['display_name']} | {first['split']}/{first['sample_idx']} | "
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | "
            f"{values[4]} | {values[5]} | {values[6]} | {values[7]} | {true_md_text} |"
        )

    for panel in panel_order:
        first = first_by_panel[panel]
        true_md = str(first.get("true_md", ""))
        lines.extend(
            [
                "",
                f"## {panel} {first['display_name']}",
                "",
                f"- Validation item: `{first['split']}/{first['sample_idx']}`; SDF name: `{first['sdf_name']}`",
                f"- True-shift table: `{true_md}`",
            ]
        )
        for task in ("13C", "1H"):
            lines.extend(
                [
                    "",
                    f"### {task}",
                    "",
                    "| Order | Heavy atom index | Target label | True | Pred s0 | Err s0 | Pred s2 | Err s2 | Pred s3 | Err s3 | Mean pred | Mean err |",
                    "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            order_rows = by_panel_task_seed[(panel, task, seeds[0])]
            for base_row in sorted(order_rows, key=lambda row: int(row["atom_order"])):
                order = int(base_row["atom_order"])
                seed_rows = {
                    seed: next(row for row in by_panel_task_seed[(panel, task, seed)] if int(row["atom_order"]) == order)
                    for seed in seeds
                }
                pred_values = [float(seed_rows[seed]["pred_ppm"]) for seed in seeds]
                err_values = [float(seed_rows[seed]["residual_ppm"]) for seed in seeds]
                mean_pred = sum(pred_values) / len(pred_values)
                mean_err = sum(err_values) / len(err_values)
                lines.append(
                    f"| {order} | {base_row['heavy_atom_index']} | {base_row['target_label']} | {fmt(float(base_row['true_ppm']))} | "
                    f"{fmt(pred_values[0])} | {fmt(err_values[0])} | "
                    f"{fmt(pred_values[1])} | {fmt(err_values[1])} | "
                    f"{fmt(pred_values[2])} | {fmt(err_values[2])} | "
                    f"{fmt(mean_pred)} | {fmt(mean_err)} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md

    rows = export_predictions(args)
    rows = sorted(rows, key=lambda r: (int(str(r["panel"]).lstrip("S")), str(r["task"]), int(r["atom_order"]), int(r["seed"])))
    write_csv(out_csv, rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(rows, out_md), encoding="utf-8")
    print(f"Wrote {out_csv}", flush=True)
    print(f"Wrote {out_md}", flush=True)


if __name__ == "__main__":
    main()
