from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from module.model_names import GLYCOSENSEFORMER, canonical_model_name
DATA_DIR = ROOT / "data"
RESULT_RE = re.compile(
    r"Test MAE is (?P<mae>[-+0-9.eE]+)(?:,| and) RMSE is (?P<rmse>[-+0-9.eE]+)"
    r"(?:, and R2 is (?P<r2>[-+0-9.eE]+|nan))? for (?P<split>[^.]+)\."
)


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with log_path.open("w") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError(f"Failed to capture command output: {' '.join(cmd)}")
        for line in proc.stdout:
            lines.append(line)
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        returncode = proc.wait()
    text = "".join(lines)
    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {' '.join(cmd)}\nSee {log_path}")
    return text


def parse_metrics(text: str, task: str, fold: int, args: argparse.Namespace) -> list[dict]:
    rows = []
    for match in RESULT_RE.finditer(text):
        rows.append(
            {
                "model": args.model_name,
                "architecture": args.architecture,
                "task": task,
                "fold": fold,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lr_reduce": args.lr_reduce,
                "seed": args.seed,
                "mono_confs": args.mono_confs,
                "di_confs": args.di_confs,
                "tri_confs": args.tri_confs,
                "conformer_oversample": args.conformer_oversample,
                "conformer_dir": args.conformer_dir,
                "mmff_optimize": not args.no_mmff_optimize,
                "split": match.group("split"),
                "mae_ppm": float(match.group("mae")),
                "rmse_ppm": float(match.group("rmse")),
                "r2": float(match.group("r2")) if match.group("r2") is not None else float("nan"),
            }
        )
    expected = {"mono", "di", "tri"}
    got = {row["split"] for row in rows}
    if got != expected:
        raise RuntimeError(f"Expected metrics for {expected}, got {got}")
    return rows


def move_checkpoints(out_dir: Path, fold: int, task: str) -> None:
    ckpt_dir = out_dir / "checkpoints" / f"fold_{fold}" / task
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(ROOT.glob("_checkpoint_epoch_*.pkl")):
        shutil.move(str(path), str(ckpt_dir / path.name))


def clear_root_checkpoints() -> None:
    for path in ROOT.glob("_checkpoint_epoch_*.pkl"):
        path.unlink()


def resolve_dataset_dir(dataset_dir: str | Path) -> Path:
    path = Path(dataset_dir)
    if not path.is_absolute():
        path = ROOT / path
    return path


def required_dataset_paths(dataset_dir: str | Path) -> list[Path]:
    dataset_path = resolve_dataset_dir(dataset_dir)
    return [
        dataset_path / f"train_data_{task}.pickle"
        for task in ("13C", "1H")
    ] + [
        dataset_path / f"test_data_{task}_{split}.pickle"
        for task in ("13C", "1H")
        for split in ("mo", "di", "tri")
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_metrics(out_dir: Path) -> list[dict]:
    for name in ("metrics_partial.csv", "metrics.csv"):
        path = out_dir / name
        if path.exists():
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["fold"] = int(row["fold"])
                row["epochs"] = int(row["epochs"])
                row["batch_size"] = int(row["batch_size"])
                row["lr"] = float(row["lr"])
                if "lr_reduce" in row and row["lr_reduce"] != "":
                    row["lr_reduce"] = float(row["lr_reduce"])
                row["seed"] = int(row["seed"]) if "seed" in row and row["seed"] != "" else 0
                for key in ("mono_confs", "di_confs", "tri_confs", "conformer_oversample"):
                    if key in row and row[key] != "":
                        row[key] = int(row[key])
                row["mae_ppm"] = float(row["mae_ppm"])
                row["rmse_ppm"] = float(row["rmse_ppm"])
                row["r2"] = float(row["r2"]) if "r2" in row and row["r2"] != "" else float("nan")
            return rows
    return []


def fold_complete(rows: list[dict], fold: int) -> bool:
    expected = {(task, split) for task in ("13C", "1H") for split in ("mono", "di", "tri")}
    got = {(row["task"], row["split"]) for row in rows if int(row["fold"]) == fold}
    return expected.issubset(got)


def task_complete(rows: list[dict], fold: int, task: str) -> bool:
    expected = {"mono", "di", "tri"}
    got = {row["split"] for row in rows if int(row["fold"]) == fold and row["task"] == task}
    return expected.issubset(got)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["split"])].append(row)
    summary = []
    for (task, split), group in sorted(groups.items()):
        maes = [float(row["mae_ppm"]) for row in group]
        rmses = [float(row["rmse_ppm"]) for row in group]
        r2s = [float(row["r2"]) for row in group]
        summary.append(
            {
                "task": task,
                "split": split,
                "n_folds": len(group),
                "mae_mean_ppm": mean(maes),
                "mae_std_ppm": sample_std(maes),
                "rmse_mean_ppm": mean(rmses),
                "rmse_std_ppm": sample_std(rmses),
                "r2_mean": mean(r2s),
                "r2_std": sample_std(r2s),
            }
        )
    return summary


def write_markdown(
    out_dir: Path,
    metric_rows: list[dict],
    summary_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        f"# {args.model_name} 10-fold 训练结果",
        "",
        "本报告由 `scripts/reproduce_10fold.py` 自动生成。",
        "",
        "## 配置",
        "",
        f"- architecture: `{args.architecture}`",
        f"- folds: `{args.start_fold}` 到 `{args.end_fold - 1}`",
        f"- epochs: `{args.epochs}`",
        f"- batch size: `{args.batch_size}`",
        f"- learning rate: `{args.lr}`",
        f"- learning rate reduce: `{args.lr_reduce}`",
        f"- seed: `{args.seed}`",
        f"- conformers: mono `{args.mono_confs}`, di `{args.di_confs}`, tri `{args.tri_confs}`",
        f"- conformer oversample: `{args.conformer_oversample}`",
        f"- conformer seed: `{args.conformer_seed}`",
        f"- conformer dir: `{args.conformer_dir}`",
        f"- MMFF optimize: `{not args.no_mmff_optimize}`",
        f"- skip data generation: `{args.skip_data}`",
        f"- dataset dir: `{args.dataset_dir}`",
    ]
    if args.architecture == "glycangt_atom":
        lines.extend(
            [
                f"- GlycanGT root: `{args.glycangt_root}`",
                f"- GlycanGT config: `{args.glycangt_config_name}`",
                f"- GlycanGT distance bins: `{args.glycangt_distance_bins}`",
                f"- GlycanGT distance cutoff: `{args.glycangt_distance_cutoff}`",
                f"- GlycanGT freeze encoder: `{args.glycangt_freeze_encoder}`",
            ]
        )
    if args.architecture == "glycanaa_atom":
        lines.append("- GlycanAA atom-conformer: `atom type + H count` node embedding with compositional relation graph convolution.")
    if args.architecture == "glycanaa_hetero_atom":
        lines.append("- GlycanAA hetero atom-conformer: inferred monosaccharide residue nodes plus atom nodes.")
    if args.architecture == "eqv3_glycanaa_hetero_fusion":
        lines.append("- Fusion: Equiformer v3 scalar atom features plus GlycanAA hetero atom-conformer residue features, trained with a gated residual head.")
    if args.architecture == "eqv3_glycanaa_cross_attention_fusion":
        lines.append("- Cross-attention fusion: Equiformer v3 atom scalar features query inferred GlycanAA monosaccharide tokens, then the fused scalar features are used for direct NMR prediction.")
    if args.architecture == "eqv3_glycanaa_cross_attention_residual_fusion":
        lines.append("- Residual cross-attention fusion: Equiformer v3 atom scalar features query inferred GlycanAA monosaccharide tokens, then a small residual adapter is added before direct NMR prediction.")
    if canonical_model_name(args.architecture) == GLYCOSENSEFORMER:
        lines.append("- Identity residual cross-attention fusion: Equiformer v3 atom scalar features query inferred GlycanAA monosaccharide tokens, then a small no-norm residual adapter is added before direct NMR prediction.")
    if args.architecture == "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_fusion":
        lines.append("- Zero-mono-token identity residual cross-attention ablation: GlycanAA inferred monosaccharide token content is zeroed while the single-direction attention path and residual adapter are kept.")
    if args.architecture == "eqv3_glycanaa_cross_attention_identity_residual_bidirectional_fusion":
        lines.append("- Identity residual bidirectional cross-attention fusion: Equiformer v3 atom scalar features attend to GlycanAA monosaccharide tokens, GlycanAA tokens attend back to Equiformer atom features, and the bidirectional context is injected through a small no-norm residual adapter before direct NMR prediction.")
    if args.architecture == "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_bidirectional_fusion":
        lines.append("- Zero-mono-token bidirectional cross-attention ablation: GlycanAA inferred monosaccharide token content is zeroed while the bidirectional attention path is kept as a placeholder.")
    if args.architecture == "eqv3_glycanaa_cross_attention_identity_residual_mono_context_fusion":
        lines.append("- Identity residual mono-context cross-attention fusion: Equiformer v3 atom scalar features query atom-aligned GlycanAA monosaccharide context, then a small no-norm residual adapter is added before direct NMR prediction.")
    if args.architecture == "eqv3_glycanaa_gated_residual_fusion":
        lines.append("- Gated residual fusion: `h_fused = h_Eq + sigmoid(MLP([h_Eq, h_mono])) * phi(h_mono)`, followed by the Equiformer NMR head.")
    if args.architecture == "eqv3_glycanaa_atom_uncertainty_gated_fusion":
        lines.append("- Atom-aware uncertainty gated fusion: Equiformer scalar features are corrected by GlycanAA monosaccharide context with a gate conditioned on atom type, H count, degree, and learned uncertainty.")
    if args.architecture == "eqv3_glycanaa_linkage_aware_gated_fusion":
        lines.append("- Linkage-aware gated fusion: GlycanAA monosaccharide context is gated with bridge/linkage-local atom features.")
    if args.architecture == "eqv3_glycan_chain_atom_context":
        lines.append("- Glycan-chain atom context: inferred monosaccharide chain topology and bridge-oxygen features are encoded as atom-level context and gated into Equiformer v3 scalar features.")
    if args.architecture == "eqv3_glycan_chain_residual_bias":
        lines.append("- Glycan-chain residual bias: Equiformer v3 prediction is kept as the base, while inferred chain topology and GlycanAA monosaccharide context add a small gated residual correction.")
    if args.architecture == "eqv3_glycan_chain_feature_adapter":
        lines.append("- Glycan-chain feature adapter: inferred chain topology and GlycanAA monosaccharide context add a small zero-initialized gated residual directly to Equiformer v3 scalar atom features.")
    if args.architecture == "eqv3_glycan_chain_feature_adapter_residual_bias":
        lines.append("- Glycan-chain feature adapter residual bias: a small gated chain adapter is injected into Equiformer v3 scalar atom features, with an additional conservative chain residual bias at the prediction level.")
    if args.architecture == "eqv3_glycan_chain_tiny_feature_residual_bias":
        lines.append("- Tiny glycan-chain feature adapter residual bias: an ultra-small fixed chain adapter is injected into Equiformer v3 scalar atom features, with a conservative residual bias used as the main correction path.")
    if args.architecture == "eqv3_glycan_chain_anchored_feature_residual_bias":
        lines.append("- Anchored glycan-chain feature residual bias: Equiformer v3 prediction is kept as the base, while a chain-aware feature branch contributes only a bounded prediction delta plus a conservative residual bias.")
    if args.architecture == "eqv3_glycan_chain_stable_residual_bias":
        lines.append("- Stable glycan-chain residual bias: same conservative chain residual correction as V5A, but removes the order-dependent normalized monosaccharide index from the explicit chain features.")
    if args.architecture == "eqv3_glycan_chain_stable_chain_only_residual_bias":
        lines.append("- Stable chain-only residual bias: uses explicit inferred chain topology without GlycanAA monosaccharide context to test whether chain features are independently useful.")
    if args.architecture == "eqv3_glycan_chain_stable_hard_carbon_residual_bias":
        lines.append("- Stable hard-carbon residual bias: V6C plus a bounded linkage-local carbon branch focused on hard disaccharide carbon corrections.")
    if args.architecture == "eqv3_glycan_chain_stable_film_residual_bias":
        lines.append("- Stable chain FiLM residual bias: uses explicit chain context and GlycanAA context to conservatively FiLM-modulate Equiformer v3 scalar features, then anchors the prediction to the base output.")
    if args.architecture == "eqv3_glycan_chain_stable_scalar_concat_fusion":
        lines.append("- Stable scalar concat fusion: Equiformer v3 l=0 scalar features, stable explicit chain context, and GlycanAA monosaccharide context are concatenated and fused with an MLP before the NMR head.")
    if args.architecture == "eqv3_glycan_chain_stable_scalar_residual_concat_fusion":
        lines.append("- Stable scalar residual concat fusion: Equiformer v3 l=0 scalar features are kept as an identity path, with a small zero-initialized chain/GlycanAA scalar residual added before the NMR head.")
    if args.architecture == "eqv3_glycanaa_mono_bias":
        lines.append("- Residue-context bias: Equiformer v3 prediction plus a small MLP bias from GlycanAA inferred monosaccharide context.")
    lines.extend(
        [
            "- 数据生成命令使用本仓库 `data/create_data.py`。",
            "- 训练命令使用本仓库 `scripts/train_nmr.py`。",
            "",
            "## 10 折汇总",
            "",
            "| Task | Split | Folds | MAE mean | MAE std | RMSE mean | RMSE std | R2 mean | R2 std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['task']} | {row['split']} | {row['n_folds']} | "
            f"{row['mae_mean_ppm']:.6f} | {row['mae_std_ppm']:.6f} | "
            f"{row['rmse_mean_ppm']:.6f} | {row['rmse_std_ppm']:.6f} | "
            f"{row['r2_mean']:.6f} | {row['r2_std']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 单折明细",
            "",
            "| Task | Fold | Seed | Split | MAE ppm | RMSE ppm | R2 |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['task']} | {row['fold']} | {row['seed']} | {row['split']} | "
            f"{row['mae_ppm']:.6f} | {row['rmse_ppm']:.6f} | {row['r2']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "该脚本按本地仓库当前可运行代码进行 10-fold 训练；本报告中的构象数量见配置区。",
            "",
        ]
    )
    (out_dir / "reproduction_10fold_report.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 10-fold Equiformer v3 NMR reproduction")
    parser.add_argument("--out_dir", default="reproduce_results/tenfold_equiformer_v3_full_epochs1_lr3e-4_bs16")
    parser.add_argument("--model_name", default="EquiformerV3-full")
    parser.add_argument(
        "--architecture",
        type=canonical_model_name,
        default="equiformer_v3",
        choices=[
            "equiformer_v3",
            "gcn",
            "gat",
            "mpnn",
            "rgcn",
            "compgcn",
            "geqshift",
            "glycanaa_atom",
            "glycanaa_hetero_atom",
            "glycangt_atom",
            "eqv3_glycanaa_hetero_fusion",
            "eqv3_glycanaa_cross_attention_fusion",
            "eqv3_glycanaa_cross_attention_residual_fusion",
            GLYCOSENSEFORMER,
            "eqv3_glycanaa_cross_attention_zero_mono_token_residual_fusion",
            "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_fusion",
            "eqv3_glycanaa_cross_attention_bidirectional_residual_fusion",
            "eqv3_glycanaa_cross_attention_identity_residual_bidirectional_fusion",
            "eqv3_glycanaa_cross_attention_zero_mono_token_bidirectional_residual_fusion",
            "eqv3_glycanaa_cross_attention_identity_residual_zero_mono_token_bidirectional_fusion",
            "eqv3_glycanaa_cross_attention_mono_context_residual_fusion",
            "eqv3_glycanaa_cross_attention_identity_residual_mono_context_fusion",
            "eqv3_glycanaa_gated_residual_fusion",
            "eqv3_glycanaa_atom_uncertainty_gated_fusion",
            "eqv3_glycanaa_linkage_aware_gated_fusion",
            "eqv3_glycan_chain_atom_context",
            "eqv3_glycan_chain_residual_bias",
            "eqv3_glycan_chain_feature_adapter",
            "eqv3_glycan_chain_feature_adapter_residual_bias",
            "eqv3_glycan_chain_tiny_feature_residual_bias",
            "eqv3_glycan_chain_anchored_feature_residual_bias",
            "eqv3_glycan_chain_stable_residual_bias",
            "eqv3_glycan_chain_stable_chain_only_residual_bias",
            "eqv3_glycan_chain_stable_hard_carbon_residual_bias",
            "eqv3_glycan_chain_stable_film_residual_bias",
            "eqv3_glycan_chain_stable_scalar_concat_fusion",
            "eqv3_glycan_chain_stable_scalar_residual_concat_fusion",
            "eqv3_glycanaa_mono_bias",
        ],
    )
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--end_fold", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_reduce", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--equiformer_v3_root", default="/home/cns/糖NMR/equiformer_v3")
    parser.add_argument("--eqv3_num_layers", type=int, default=7)
    parser.add_argument("--eqv3_num_channels", type=int, default=128)
    parser.add_argument("--eqv3_attn_hidden_channels", type=int, default=32)
    parser.add_argument("--eqv3_ffn_hidden_channels", type=int, default=512)
    parser.add_argument("--eqv3_num_radial_basis", type=int, default=64)
    parser.add_argument("--eqv3_lmax", type=int, default=4)
    parser.add_argument("--eqv3_mmax", type=int, default=2)
    parser.add_argument("--eqv3_max_radius", type=float, default=6.0)
    parser.add_argument("--baseline_hidden_channels", type=int, default=128)
    parser.add_argument("--baseline_num_layers", type=int, default=4)
    parser.add_argument("--baseline_dropout", type=float, default=0.1)
    parser.add_argument("--baseline_edge_hidden_channels", type=int, default=64)
    parser.add_argument("--baseline_head_hidden_channels", type=int, default=128)
    parser.add_argument("--baseline_attention_heads", type=int, default=4)
    parser.add_argument("--baseline_num_relations", type=int, default=5)
    parser.add_argument("--geqshift_n_input", type=int, default=128)
    parser.add_argument("--geqshift_n_node_attr", type=int, default=128)
    parser.add_argument("--geqshift_n_output", type=int, default=128)
    parser.add_argument("--geqshift_n_layers", type=int, default=7)
    parser.add_argument("--glycangt_root", default="/home/cns/糖NMR/GlycanGT")
    parser.add_argument("--glycangt_config_name", default="ss", choices=["ss", "small", "medium", "large"])
    parser.add_argument("--glycangt_distance_bins", type=int, default=16)
    parser.add_argument("--glycangt_distance_cutoff", type=float, default=6.0)
    parser.add_argument("--glycangt_freeze_encoder", action="store_true")
    parser.add_argument("--fusion_hidden_channels", type=int, default=256)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--no_fusion_zero_init", action="store_true")
    parser.add_argument("--bias_hidden_channels", type=int, default=128)
    parser.add_argument("--bias_dropout", type=float, default=0.1)
    parser.add_argument("--no_bias_zero_init", action="store_true")
    parser.add_argument("--mono_confs", type=int, default=10)
    parser.add_argument("--di_confs", type=int, default=1)
    parser.add_argument("--tri_confs", type=int, default=1)
    parser.add_argument("--force_conformers", action="store_true")
    parser.add_argument("--conformer_oversample", type=int, default=2)
    parser.add_argument("--conformer_seed", type=int, default=1)
    parser.add_argument("--conformer_threads", type=int, default=10)
    parser.add_argument("--conformer_dir", default="conformations")
    parser.add_argument("--no_mmff_optimize", action="store_true")
    parser.add_argument("--skip_data", action="store_true", help="Skip create_data.py and reuse existing data/datasets pickles")
    parser.add_argument("--dataset_dir", default="data/datasets", help="Directory containing train/test pickle files")
    return parser


def run_reproduction(args: argparse.Namespace) -> None:
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    python = sys.executable
    metric_rows: list[dict] = read_existing_metrics(out_dir)

    for fold in range(args.start_fold, args.end_fold):
        if fold_complete(metric_rows, fold):
            print(f"Fold {fold} already complete; skipping.", flush=True)
            continue
        print(f"Starting fold {fold}.", flush=True)
        if args.skip_data:
            missing = [path for path in required_dataset_paths(args.dataset_dir) if not path.exists()]
            if missing:
                missing_text = ", ".join(str(path) for path in missing)
                raise FileNotFoundError(f"--skip_data was set, but required dataset pickles are missing: {missing_text}")
            print(f"Skipping create_data.py and reusing existing pickles from {args.dataset_dir}.", flush=True)
        else:
            data_log = out_dir / "logs" / f"fold_{fold}" / "create_data.log"
            data_cmd = [
                python,
                "-u",
                "create_data.py",
                "--data_path_mo",
                "dataset/casper_mono.sdf",
                "--data_path_di",
                "dataset/casper_dimer.sdf",
                "--data_path_tri",
                "dataset/casper_trimer.sdf",
                "--fold",
                str(fold),
                "--mono_confs",
                str(args.mono_confs),
                "--di_confs",
                str(args.di_confs),
                "--tri_confs",
                str(args.tri_confs),
                "--conformer_oversample",
                str(args.conformer_oversample),
                "--conformer_seed",
                str(args.conformer_seed),
                "--conformer_threads",
                str(args.conformer_threads),
                "--conformer_dir",
                args.conformer_dir,
            ]
            if args.force_conformers:
                data_cmd.append("--force_conformers")
            if args.no_mmff_optimize:
                data_cmd.append("--no_mmff_optimize")
            run_command(data_cmd, DATA_DIR, data_log)

        for task in ("13C", "1H"):
            if task_complete(metric_rows, fold, task):
                print(f"Fold {fold} {task} already complete; skipping.", flush=True)
                continue
            dataset_dir = str(Path(args.dataset_dir))
            train_path = str(Path(dataset_dir) / f"train_data_{task}.pickle")
            test_mo = str(Path(dataset_dir) / f"test_data_{task}_mo.pickle")
            test_di = str(Path(dataset_dir) / f"test_data_{task}_di.pickle")
            test_tri = str(Path(dataset_dir) / f"test_data_{task}_tri.pickle")
            train_log = out_dir / "logs" / f"fold_{fold}" / f"train_{task}.log"
            clear_root_checkpoints()
            train_cmd = [
                python,
                "-u",
                "scripts/train_nmr.py",
                "--train_path",
                train_path,
                "--test_path_mo",
                test_mo,
                "--test_path_di",
                test_di,
                "--test_path_tri",
                test_tri,
                "--batch_size",
                str(args.batch_size),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--lr_reduce",
                str(args.lr_reduce),
                "--seed",
                str(args.seed),
                "--architecture",
                args.architecture,
                "--equiformer_v3_root",
                args.equiformer_v3_root,
                "--eqv3_num_layers",
                str(args.eqv3_num_layers),
                "--eqv3_num_channels",
                str(args.eqv3_num_channels),
                "--eqv3_attn_hidden_channels",
                str(args.eqv3_attn_hidden_channels),
                "--eqv3_ffn_hidden_channels",
                str(args.eqv3_ffn_hidden_channels),
                "--eqv3_num_radial_basis",
                str(args.eqv3_num_radial_basis),
                "--eqv3_lmax",
                str(args.eqv3_lmax),
                "--eqv3_mmax",
                str(args.eqv3_mmax),
                "--eqv3_max_radius",
                str(args.eqv3_max_radius),
                "--baseline_hidden_channels",
                str(args.baseline_hidden_channels),
                "--baseline_num_layers",
                str(args.baseline_num_layers),
                "--baseline_dropout",
                str(args.baseline_dropout),
                "--baseline_edge_hidden_channels",
                str(args.baseline_edge_hidden_channels),
                "--baseline_head_hidden_channels",
                str(args.baseline_head_hidden_channels),
                "--baseline_attention_heads",
                str(args.baseline_attention_heads),
                "--baseline_num_relations",
                str(args.baseline_num_relations),
                "--geqshift_n_input",
                str(args.geqshift_n_input),
                "--geqshift_n_node_attr",
                str(args.geqshift_n_node_attr),
                "--geqshift_n_output",
                str(args.geqshift_n_output),
                "--geqshift_n_layers",
                str(args.geqshift_n_layers),
                "--glycangt_root",
                args.glycangt_root,
                "--glycangt_config_name",
                args.glycangt_config_name,
                "--glycangt_distance_bins",
                str(args.glycangt_distance_bins),
                "--glycangt_distance_cutoff",
                str(args.glycangt_distance_cutoff),
                "--fusion_hidden_channels",
                str(args.fusion_hidden_channels),
                "--fusion_dropout",
                str(args.fusion_dropout),
                "--bias_hidden_channels",
                str(args.bias_hidden_channels),
                "--bias_dropout",
                str(args.bias_dropout),
            ]
            if args.glycangt_freeze_encoder:
                train_cmd.append("--glycangt_freeze_encoder")
            if args.no_fusion_zero_init:
                train_cmd.append("--no_fusion_zero_init")
            if args.no_bias_zero_init:
                train_cmd.append("--no_bias_zero_init")
            text = run_command(train_cmd, ROOT, train_log)
            metric_rows.extend(parse_metrics(text, task, fold, args))
            move_checkpoints(out_dir, fold, task)

        write_csv(out_dir / "metrics_partial.csv", metric_rows)
        print(f"Finished fold {fold}.", flush=True)

    summary_rows = summarize(metric_rows)
    write_csv(out_dir / "metrics.csv", metric_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_markdown(out_dir, metric_rows, summary_rows, args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_reproduction(args)
