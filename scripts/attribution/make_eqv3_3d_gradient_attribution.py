#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from PIL import Image
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

try:
    import torch
    from torch_geometric.loader import DataLoader
except ModuleNotFoundError:
    torch = None
    DataLoader = None


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR = ROOT / "outputs/attribution"
DATASET_DIR = ROOT / "data" / "datasets_glyconmr_len123_independent_validation_clean_split1"
RUN_DIR = (
    ROOT
    / "configs/original_runs"
    / "full_100conf_eqv3_glycanaa_cross_attention_identity_residual_fusion_fixed_token_split_fold0_epochs10_bs16_seed4_20260617"
)
OUT_BASENAME = "fig_eqv3_3d_gradient_attribution"

EDGE = "#3F474A"
TEXT = "#1F2426"
GRID = "#E4E7E8"
BLUE = "#5B9DB8"
ORANGE = "#DFA85A"
GREEN = "#8FAE91"
RED = "#C94D4D"
PURPLE = "#8E6BBE"
PANEL_BG = "#FBFBF8"

sys.path.append(str(ROOT))
try:
    from module.custom_data import load_train_test_data  # noqa: E402
    from module.custom_models import build_nmr_model  # noqa: E402
    from module.custom_tasks.nmr_shift import build_model_kwargs  # noqa: E402
except ModuleNotFoundError:
    load_train_test_data = None
    build_nmr_model = None
    build_model_kwargs = None


@dataclass(frozen=True)
class TaskData:
    train_set: object
    split_data: dict[str, object]
    mean: torch.Tensor
    std: torch.Tensor


def register_arial() -> None:
    font_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for name in (
        "Arial.ttf",
        "arialbd.ttf",
        "ariali.ttf",
        "arialbi.ttf",
        "Arial_Bold.ttf",
        "Arial_Italic.ttf",
        "Arial_Bold_Italic.ttf",
    ):
        font_path = font_dir / name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def apply_style() -> None:
    register_arial()
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 10.2,
            "axes.linewidth": 1.15,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def target_mask(data) -> torch.Tensor:
    return (data.x[:, 0] == 2.0).logical_and(data.x[:, -1] > -0.5)


def site_from_atom_name(task: str, atom_name: str) -> str | None:
    text = str(atom_name)
    if len(text) < 2 or text[0] != "C":
        return None
    suffix = "".join(ch for ch in text[1:] if ch.isdigit())
    if not suffix:
        return None
    pos = int(suffix)
    if not 1 <= pos <= 6:
        return None
    return ("C" if task == "13C" else "H") + str(pos)


def read_sdf_site_maps() -> dict[tuple[str, str], dict[int, str]]:
    maps: dict[tuple[str, str], dict[int, str]] = {}
    for sdf_path in sorted(DATASET_DIR.glob("glyconmr_len123_independent_validation_*.sdf")):
        if sdf_path.name == "glyconmr_len123_independent_validation.sdf":
            continue
        text = sdf_path.read_text(encoding="utf-8", errors="ignore")
        for block in text.split("$$$$"):
            if not block.strip():
                continue
            lines = block.strip().splitlines()
            name = lines[0].strip()
            props: dict[str, str] = {}
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.startswith(">") and "<" in line and ">" in line:
                    key = line.split("<", 1)[1].split(">", 1)[0]
                    i += 1
                    values = []
                    while i < len(lines) and lines[i].strip() != "":
                        values.append(lines[i])
                        i += 1
                    props[key] = "\n".join(values)
                i += 1
            atom_names = ast.literal_eval(props.get("heavy_atom_names", "[]"))
            for task, prop in (("13C", "13C Spectrum"), ("1H", "1H Spectrum")):
                if prop not in props:
                    continue
                spectrum = ast.literal_eval(props[prop])
                maps[(name, task)] = {
                    int(atom_idx): site
                    for atom_idx in spectrum
                    if (site := site_from_atom_name(task, atom_names[int(atom_idx)])) is not None
                }
    return maps


def load_args(run_dir: Path) -> argparse.Namespace:
    with (run_dir / "config.json").open() as handle:
        return argparse.Namespace(**json.load(handle))


def checkpoint_path(run_dir: Path, task: str, epoch: int) -> Path:
    path = run_dir / "checkpoints" / "fold_0" / task / f"_checkpoint_epoch_{epoch}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def derive_mean_std(train_set, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    values = []
    for data in loader:
        values.append(data.x[:, -1][target_mask(data)])
    joined = torch.cat(values)
    return joined.mean(), joined.std()


def load_task_data(task: str, batch_size: int) -> TaskData:
    train_set, test_mo, test_di, test_tri = load_train_test_data(
        DATASET_DIR / f"train_data_{task}.pickle",
        DATASET_DIR / f"test_data_{task}_mo.pickle",
        DATASET_DIR / f"test_data_{task}_di.pickle",
        DATASET_DIR / f"test_data_{task}_tri.pickle",
    )
    mean, std = derive_mean_std(train_set, batch_size)
    return TaskData(
        train_set=train_set,
        split_data={"mono": test_mo, "di": test_di, "tri": test_tri},
        mean=mean,
        std=std,
    )


def build_model(run_dir: Path, task: str, epoch: int, device: torch.device) -> torch.nn.Module:
    args = load_args(run_dir)
    model_kwargs = build_model_kwargs(args)
    architecture = model_kwargs.pop("architecture", "equiformer_v3")
    model = build_nmr_model(architecture=architecture, **model_kwargs).to(device)
    state = torch.load(checkpoint_path(run_dir, task, epoch), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def equiformer_only_prediction(model: torch.nn.Module, data, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    x = data.x[:, 0:2].long()
    eq_scalar = model.equiformer.encode(
        x=x,
        pos=pos,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr.long(),
        batch=batch,
    )
    features = model.equiformer.make_head_features(eq_scalar, x)
    return model.equiformer.nmr_head(features)


def covalent_adjacency(data) -> dict[int, list[int]]:
    adjacency = {idx: [] for idx in range(int(data.x.shape[0]))}
    edge_index = data.edge_index.detach().cpu()
    edge_attr = data.edge_attr.detach().cpu()
    for col in range(edge_index.shape[1]):
        if int(edge_attr[col].item()) <= 0:
            continue
        src = int(edge_index[0, col].item())
        dst = int(edge_index[1, col].item())
        if dst not in adjacency[src]:
            adjacency[src].append(dst)
    return adjacency


def graph_distances(adjacency: dict[int, list[int]], start: int) -> dict[int, int]:
    out = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt in out:
                continue
            out[nxt] = out[node] + 1
            queue.append(nxt)
    return out


def bridge_like_atoms(data, adjacency: dict[int, list[int]]) -> set[int]:
    atom_types = data.x[:, 0].detach().cpu().numpy()
    atoms: set[int] = set()
    for idx, atom_type in enumerate(atom_types):
        if int(atom_type) != 4:
            continue
        carbon_neighbors = [nbr for nbr in adjacency[idx] if int(atom_types[nbr]) == 2]
        if len(carbon_neighbors) >= 2:
            atoms.add(idx)
            atoms.update(carbon_neighbors)
    return atoms


def local_category(
    influenced_idx: int,
    target_idx: int,
    graph_distance: int | None,
    atom_type: int,
    distance_angstrom: float,
    bridge_atoms: set[int],
) -> str:
    if influenced_idx == target_idx:
        return "target atom"
    if graph_distance == 1 and atom_type == 4:
        return "bonded oxygen"
    if graph_distance == 1:
        return "one-bond carbon"
    if influenced_idx in bridge_atoms:
        return "C-O-C bridge neighborhood"
    if graph_distance == 2:
        return "two-bond neighborhood"
    if distance_angstrom <= 4.0:
        return "spatially nearby nonbonded"
    return "distal atoms"


def motif_for_site(task: str, site: str | None) -> str:
    if site is None:
        return "unassigned"
    if task == "13C":
        if site == "C1":
            return "anomeric carbon"
        if site == "C6":
            return "hydroxymethyl carbon"
        return "ring carbon"
    if site == "H1":
        return "anomeric proton"
    if site == "H6":
        return "hydroxymethyl proton"
    return "ring proton"


def choose_indices(length: int, max_count: int | None) -> list[int]:
    if max_count is None or length <= max_count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, max_count, dtype=int).tolist()))


def gradient_rows_for_graph(
    model: torch.nn.Module,
    data,
    task: str,
    split: str,
    sample_name: str,
    mean: torch.Tensor,
    std: torch.Tensor,
    site_map: dict[int, str],
    max_targets_per_molecule: int | None,
    device: torch.device,
) -> list[dict]:
    data = data.to(device)
    pos = data.pos.float().detach().clone().to(device).requires_grad_(True)
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)
    mask = target_mask(data)
    target_atom_indices = torch.where(mask)[0]
    selected_orders = choose_indices(int(target_atom_indices.numel()), max_targets_per_molecule)

    adjacency = covalent_adjacency(data)
    bridge_atoms = bridge_like_atoms(data, adjacency)
    atom_types = data.x[:, 0].detach().cpu().numpy().astype(int)
    coord = data.pos.detach().cpu().numpy()

    out = equiformer_only_prediction(model, data, pos, batch)
    preds = out[mask].flatten() * std.to(device) + mean.to(device)
    rows: list[dict] = []
    for target_order in selected_orders:
        target_atom_idx = int(target_atom_indices[target_order].item())
        target_site = site_map.get(target_atom_idx)
        target_motif = motif_for_site(task, target_site)
        if pos.grad is not None:
            pos.grad.zero_()
        grad = torch.autograd.grad(preds[target_order], pos, retain_graph=True, create_graph=False)[0]
        grad_norm = grad.detach().norm(dim=1).cpu().numpy()
        distances = graph_distances(adjacency, target_atom_idx)
        xyz_target = coord[target_atom_idx]
        total_grad = float(np.sum(grad_norm))
        local_grad = float(np.sum([grad_norm[idx] for idx, dist in distances.items() if dist <= 2]))
        for influenced_idx, value in enumerate(grad_norm):
            graph_distance = distances.get(influenced_idx)
            euc = float(np.linalg.norm(coord[influenced_idx] - xyz_target))
            category = local_category(
                influenced_idx=influenced_idx,
                target_idx=target_atom_idx,
                graph_distance=graph_distance,
                atom_type=int(atom_types[influenced_idx]),
                distance_angstrom=euc,
                bridge_atoms=bridge_atoms,
            )
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "sample_name": sample_name,
                    "target_order": target_order,
                    "target_atom_idx": target_atom_idx,
                    "target_site": target_site or "",
                    "target_motif": target_motif,
                    "influenced_atom_idx": influenced_idx,
                    "influenced_atom_type": "C" if int(atom_types[influenced_idx]) == 2 else "O",
                    "graph_distance": "" if graph_distance is None else graph_distance,
                    "euclidean_distance_angstrom": euc,
                    "category": category,
                    "grad_norm_ppm_per_angstrom": float(value),
                    "target_total_grad_norm": total_grad,
                    "target_local_2hop_grad_fraction": local_grad / total_grad if total_grad > 0 else np.nan,
                }
            )
    return rows


def compute_attribution(args: argparse.Namespace) -> pd.DataFrame:
    if torch is None or DataLoader is None or load_train_test_data is None or build_nmr_model is None or build_model_kwargs is None:
        raise ModuleNotFoundError(
            "Computing coordinate gradients requires torch, torch_geometric, and project model modules. "
            "Use build_figure(...) with existing CSV files when only redrawing the figure."
        )
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_id)
    site_maps = read_sdf_site_maps()
    all_rows: list[dict] = []
    for task in args.tasks:
        task_data = load_task_data(task, args.mean_batch_size)
        model = build_model(Path(args.run_dir), task, args.epoch, device)
        mean = task_data.mean
        std = task_data.std
        for split in args.splits:
            items = task_data.split_data[split]
            selected_items = items[: args.max_molecules_per_split] if args.max_molecules_per_split else items
            for item in selected_items:
                dataset = item["loader"].dataset
                data = dataset[0]
                sample_name = item["name"]
                all_rows.extend(
                    gradient_rows_for_graph(
                        model=model,
                        data=data,
                        task=task,
                        split=split,
                        sample_name=sample_name,
                        mean=mean,
                        std=std,
                        site_map=site_maps.get((sample_name, task), {}),
                        max_targets_per_molecule=args.max_targets_per_molecule,
                        device=device,
                    )
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(all_rows)


def summarize_category(atom_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_keys = ["task", "split", "sample_name", "target_order", "target_atom_idx"]
    for keys, target_df in atom_df.groupby(target_keys, observed=True):
        total = float(target_df["grad_norm_ppm_per_angstrom"].sum())
        for category, sub in target_df.groupby("category", observed=True):
            value = float(sub["grad_norm_ppm_per_angstrom"].sum())
            rows.append(
                {
                    **dict(zip(target_keys, keys, strict=True)),
                    "target_site": target_df["target_site"].iloc[0],
                    "target_motif": target_df["target_motif"].iloc[0],
                    "category": category,
                    "category_grad_norm": value,
                    "category_grad_fraction": value / total if total > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_targets(atom_df: pd.DataFrame) -> pd.DataFrame:
    target_keys = ["task", "split", "sample_name", "target_order", "target_atom_idx", "target_site", "target_motif"]
    rows = []
    for keys, target_df in atom_df.groupby(target_keys, observed=True):
        total = float(target_df["grad_norm_ppm_per_angstrom"].sum())
        self_grad = float(target_df.loc[target_df["category"] == "target atom", "grad_norm_ppm_per_angstrom"].sum())
        local_2hop = float(
            target_df.loc[
                target_df["graph_distance"].replace("", np.nan).astype(float).le(2),
                "grad_norm_ppm_per_angstrom",
            ].sum()
        )
        rows.append(
            {
                **dict(zip(target_keys, keys, strict=True)),
                "n_influenced_atoms": int(len(target_df)),
                "total_grad_norm_ppm_per_angstrom": total,
                "self_grad_norm_ppm_per_angstrom": self_grad,
                "self_grad_fraction": self_grad / total if total > 0 else np.nan,
                "local_2hop_grad_fraction": local_2hop / total if total > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(atom_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    category_df = summarize_category(atom_df)
    target_df = summarize_targets(atom_df)
    outputs = {
        "eqv3_3d_gradient_atom_level.csv": atom_df,
        "eqv3_3d_gradient_category_summary.csv": category_df,
        "eqv3_3d_gradient_target_summary.csv": target_df,
    }
    for name, df in outputs.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        shutil.copy2(path, FINAL_DIR / name)
    return category_df, target_df


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color=TEXT,
    )


def setup_panel(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(1.25)
        spine.set_linestyle("-")


def model_ring_coords(cx: float, cy: float, radius: float = 1.0) -> dict[str, tuple[float, float]]:
    labels = ["O5", "C1", "C2", "C3", "C4", "C5"]
    angles = np.deg2rad([125, 55, -5, -65, -125, 185])
    return {
        label: (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
        for label, angle in zip(labels, angles, strict=True)
    }


def draw_model_bond(ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float], lw: float = 2.0) -> None:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=EDGE, lw=lw, solid_capstyle="round", zorder=1)


def rdkit_model_from_layout(
    atom_specs: list[tuple[str, str]],
    bonds: list[tuple[str, str]],
    coords: dict[str, tuple[float, float]],
) -> tuple[Chem.Mol, dict[str, int]]:
    mol = Chem.RWMol()
    label_to_idx = {}
    for label, element in atom_specs:
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        label_to_idx[label] = mol.AddAtom(atom)
    for left, right in bonds:
        mol.AddBond(label_to_idx[left], label_to_idx[right], Chem.BondType.SINGLE)
    out = mol.GetMol()
    out.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(len(atom_specs))
    for label, _element in atom_specs:
        x, y = coords[label]
        conf.SetAtomPosition(label_to_idx[label], Point3D(float(x), float(y), 0.0))
    out.AddConformer(conf)
    return out, label_to_idx


def draw_model_atom(
    ax: plt.Axes,
    xy: tuple[float, float],
    label: str,
    color: str,
    radius: float = 0.18,
    text_color: str = TEXT,
    z: int = 4,
) -> None:
    ax.add_patch(Circle(xy, radius=radius, facecolor=color, edgecolor=EDGE, linewidth=1.15, zorder=z))
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=8.4, fontweight="bold", color=text_color, zorder=z + 1)


def draw_model_halo(
    ax: plt.Axes,
    xy: tuple[float, float],
    radius: float,
    color: str,
    alpha: float = 0.75,
    lw: float = 2.0,
) -> None:
    ax.add_patch(Circle(xy, radius=radius, facecolor="none", edgecolor=color, linewidth=lw, alpha=alpha, zorder=2))


def draw_model_disaccharide_backbone(
    ax: plt.Axes,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]], tuple[float, float], tuple[float, float]]:
    left = model_ring_coords(1.45, 2.05, 0.82)
    right = model_ring_coords(4.08, 2.00, 0.82)
    bridge_o = (2.65, 2.43)
    h1 = (left["C1"][0] + 0.22, left["C1"][1] + 0.49)
    coords = {f"L_{label}": xy for label, xy in left.items()}
    coords.update({f"R_{label}": xy for label, xy in right.items()})
    coords["O_bridge"] = bridge_o
    coords["H1_star"] = h1
    atom_specs = [(key, "O" if key.endswith("O5") or key == "O_bridge" else "C") for key in coords]
    atom_specs[-1] = ("H1_star", "H")
    bonds = []
    order = ["O5", "C1", "C2", "C3", "C4", "C5", "O5"]
    for prefix in ("L", "R"):
        bonds.extend((f"{prefix}_{a}", f"{prefix}_{b}") for a, b in zip(order[:-1], order[1:], strict=True))
    bonds.extend(
        [
            ("L_C1", "O_bridge"),
            ("O_bridge", "R_C4"),
            ("L_C1", "H1_star"),
        ]
    )
    mol, label_to_idx = rdkit_model_from_layout(atom_specs, bonds, coords)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    for bond in mol.GetBonds():
        left_label = idx_to_label[bond.GetBeginAtomIdx()]
        right_label = idx_to_label[bond.GetEndAtomIdx()]
        lw = 1.4 if "H1_star" in {left_label, right_label} else 2.0
        draw_model_bond(ax, coords[left_label], coords[right_label], lw=lw)
    return left, right, bridge_o, h1


def build_rdkit_disaccharide_model() -> tuple[Chem.Mol, dict[str, int]]:
    left = model_ring_coords(1.45, 2.05, 0.82)
    right = model_ring_coords(4.08, 2.00, 0.82)
    bridge_o = (2.65, 2.43)
    h1 = (left["C1"][0] + 0.22, left["C1"][1] + 0.49)
    coords = {f"L_{label}": xy for label, xy in left.items()}
    coords.update({f"R_{label}": xy for label, xy in right.items()})
    coords["O_bridge"] = bridge_o
    coords["H1_star"] = h1
    atom_specs = [(key, "O" if key.endswith("O5") or key == "O_bridge" else "C") for key in coords]
    atom_specs[-1] = ("H1_star", "H")
    bonds = []
    order = ["O5", "C1", "C2", "C3", "C4", "C5", "O5"]
    for prefix in ("L", "R"):
        bonds.extend((f"{prefix}_{a}", f"{prefix}_{b}") for a, b in zip(order[:-1], order[1:], strict=True))
    bonds.extend(
        [
            ("L_C1", "O_bridge"),
            ("O_bridge", "R_C4"),
            ("L_C1", "H1_star"),
        ]
    )
    return rdkit_model_from_layout(atom_specs, bonds, coords)


def build_rdkit_target_motif_model() -> tuple[Chem.Mol, dict[str, int]]:
    coords = {
        "O5": (0.43, 0.66),
        "C1": (0.57, 0.64),
        "C2": (0.66, 0.50),
        "C3": (0.58, 0.34),
        "C4": (0.41, 0.31),
        "C5": (0.32, 0.47),
        "C6": (0.20, 0.58),
        "H1": (0.67, 0.74),
        "H2": (0.80, 0.50),
        "H3": (0.65, 0.20),
        "H4": (0.35, 0.17),
        "H5": (0.19, 0.42),
        "H6": (0.10, 0.65),
    }
    atom_specs = [
        ("O5", "O"),
        ("C1", "C"),
        ("C2", "C"),
        ("C3", "C"),
        ("C4", "C"),
        ("C5", "C"),
        ("C6", "C"),
        ("H1", "H"),
        ("H2", "H"),
        ("H3", "H"),
        ("H4", "H"),
        ("H5", "H"),
        ("H6", "H"),
    ]
    bonds = [
        ("O5", "C1"),
        ("C1", "C2"),
        ("C2", "C3"),
        ("C3", "C4"),
        ("C4", "C5"),
        ("C5", "O5"),
        ("C5", "C6"),
        ("C1", "H1"),
        ("C2", "H2"),
        ("C3", "H3"),
        ("C4", "H4"),
        ("C5", "H5"),
        ("C6", "H6"),
    ]
    return rdkit_model_from_layout(atom_specs, bonds, coords)


def render_rdkit_model_image(
    mol: Chem.Mol,
    label_to_idx: dict[str, int],
    atom_labels: dict[str, str],
    atom_colors: dict[str, str],
    atom_radii: dict[str, float],
    width: int = 1200,
    height: int = 740,
) -> np.ndarray:
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.04
    opts.bondLineWidth = 3
    opts.minFontSize = 18
    opts.maxFontSize = 34
    for label, idx in label_to_idx.items():
        opts.atomLabels[idx] = atom_labels.get(label, label)
    highlight_atoms = [label_to_idx[label] for label in atom_colors]
    highlight_colors = {label_to_idx[label]: to_rgb(color) for label, color in atom_colors.items()}
    highlight_radii = {label_to_idx[label]: atom_radii[label] for label in atom_colors}
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=[],
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA"))


def draw_protocol(ax: plt.Axes) -> None:
    setup_panel(ax)
    ax.set_title("Target C/H motif map for panel c", fontweight="bold", pad=9)
    panel_label(ax, "a")

    mol, label_to_idx = build_rdkit_target_motif_model()
    anomeric_color = RED
    ring_color = BLUE
    hydroxymethyl_color = ORANGE
    rdkit_image = render_rdkit_model_image(
        mol,
        label_to_idx,
        atom_labels={
            "O5": "",
            "C1": "C1",
            "C2": "C2",
            "C3": "C3",
            "C4": "C4",
            "C5": "C5",
            "C6": "C6",
            "H1": "H1",
            "H2": "H2",
            "H3": "H3",
            "H4": "H4",
            "H5": "H5",
            "H6": "H6",
        },
        atom_colors={
            "C1": anomeric_color,
            "H1": anomeric_color,
            "C2": ring_color,
            "H2": ring_color,
            "C3": ring_color,
            "H3": ring_color,
            "C4": ring_color,
            "H4": ring_color,
            "C5": ring_color,
            "H5": ring_color,
            "C6": hydroxymethyl_color,
            "H6": hydroxymethyl_color,
        },
        atom_radii={
            "C1": 0.18,
            "H1": 0.14,
            "C2": 0.14,
            "H2": 0.11,
            "C3": 0.14,
            "H3": 0.11,
            "C4": 0.14,
            "H4": 0.11,
            "C5": 0.14,
            "H5": 0.11,
            "C6": 0.18,
            "H6": 0.14,
        },
    )
    ax.imshow(rdkit_image, extent=(0.06, 0.76, 0.25, 0.88), aspect="auto", zorder=2)

    ax.text(
        0.73,
        0.74,
        "anomeric C/H\nC1, H1",
        ha="left",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color=anomeric_color,
    )
    ax.text(
        0.73,
        0.55,
        "ring C/H\nC2-C5, H2-H5",
        ha="left",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color=ring_color,
        linespacing=1.18,
    )
    ax.text(
        0.73,
        0.35,
        "hydroxymethyl C/H\nC6, H6",
        ha="left",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color=hydroxymethyl_color,
        linespacing=1.18,
    )
    legend = [
        (0.11, "anomeric C/H", anomeric_color),
        (0.39, "ring C/H", ring_color),
        (0.62, "hydroxymethyl C/H", hydroxymethyl_color),
    ]
    for x, label, color in legend:
        ax.add_patch(Circle((x, 0.10), 0.018, facecolor=color, edgecolor=EDGE, linewidth=0.9))
        ax.text(x + 0.026, 0.10, label, ha="left", va="center", fontsize=8.1, color=TEXT)
    ax.text(
        0.09,
        0.93,
        "Motif classes match the target labels summarized in panel c.",
        ha="left",
        va="center",
        fontsize=8.4,
        color="#4A4F51",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_category_panel(ax: plt.Axes, category_df: pd.DataFrame) -> None:
    setup_panel(ax)
    panel_label(ax, "b")
    order = [
        "target atom",
        "bonded oxygen",
        "one-bond carbon",
        "two-bond neighborhood",
        "C-O-C bridge neighborhood",
        "spatially nearby nonbonded",
        "distal atoms",
    ]
    labels = ["target", "bonded O", "1-bond C", "2-bond", "C-O-C / ring-O\nneighborhood", "nearby\nnonbonded", "distal"]
    x = np.arange(len(order))
    width = 0.36
    for offset, task, color in [(-width / 2, "13C", BLUE), (width / 2, "1H", ORANGE)]:
        values = []
        errors = []
        for category in order:
            vals = category_df[(category_df["task"] == task) & (category_df["category"] == category)][
                "category_grad_fraction"
            ].to_numpy(float)
            values.append(float(np.nanmean(vals)) if len(vals) else 0.0)
            errors.append(float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        ax.bar(x + offset, values, width=width, yerr=errors, color=color, edgecolor=EDGE, linewidth=0.8, capsize=2.0, label=task)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("Gradient mass fraction")
    ax.set_title("Where 3D coordinate sensitivity is concentrated", fontweight="bold", pad=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9.2)


def draw_motif_panel(ax: plt.Axes, target_df: pd.DataFrame) -> None:
    setup_panel(ax)
    panel_label(ax, "c")
    motif_order = ["anomeric carbon", "ring carbon", "hydroxymethyl carbon", "anomeric proton", "ring proton", "hydroxymethyl proton"]
    friendly = {
        "anomeric carbon": "anomeric C",
        "ring carbon": "ring C",
        "hydroxymethyl carbon": "hydroxymethyl C",
        "anomeric proton": "anomeric H",
        "ring proton": "ring H",
        "hydroxymethyl proton": "hydroxymethyl H",
    }
    rows = []
    task_means = target_df.groupby("task", observed=True)["total_grad_norm_ppm_per_angstrom"].mean().to_dict()
    for motif in motif_order:
        sub = target_df[target_df["target_motif"] == motif]
        vals = sub["total_grad_norm_ppm_per_angstrom"].to_numpy(float)
        if len(vals):
            task = str(sub["task"].iloc[0])
            normalized = vals / float(task_means[task])
            rows.append(
                (
                    friendly[motif],
                    task,
                    float(np.nanmean(normalized)),
                    float(np.nanstd(normalized, ddof=1) / np.sqrt(len(normalized))) if len(normalized) > 1 else 0.0,
                )
            )
    y = np.arange(len(rows))
    ax.axvline(1.0, color="#6B6F70", linestyle="--", linewidth=1.0)
    for yi, (label, task, value, error) in zip(y, rows):
        color = BLUE if task == "13C" else ORANGE
        ax.plot([1.0, value], [yi, yi], color=color, linewidth=5.0, alpha=0.45, solid_capstyle="round")
        ax.errorbar(value, yi, xerr=error, fmt="o", color=color, ecolor=EDGE, elinewidth=1.0, capsize=2.0, markersize=7.2, markeredgecolor=EDGE)
        ax.text(value + 0.06, yi, f"{value:.2f}x", ha="left", va="center", fontsize=8.8, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlabel("Relative sensitivity within each nucleus")
    ax.set_title("Target motif sensitivity of the 3D branch", fontweight="bold", pad=9)
    ax.set_xlim(0, max(2.6, max(row[2] + row[3] for row in rows) + 0.35))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_distance_panel(ax: plt.Axes, atom_df: pd.DataFrame) -> None:
    setup_panel(ax)
    panel_label(ax, "d")
    bins = [0, 2, 4, 6, 100]
    labels = ["0-2", "2-4", "4-6", ">6"]
    rows = []
    target_keys = ["task", "sample_name", "target_order", "target_atom_idx"]
    df = atom_df.copy()
    df["distance_bin"] = pd.cut(df["euclidean_distance_angstrom"], bins=bins, labels=labels, right=False)
    for keys, sub in df.groupby(target_keys, observed=True):
        total = float(sub["grad_norm_ppm_per_angstrom"].sum())
        for label in labels:
            value = float(sub.loc[sub["distance_bin"] == label, "grad_norm_ppm_per_angstrom"].sum())
            rows.append({"task": keys[0], "distance_bin": label, "fraction": value / total if total > 0 else np.nan})
    dist_df = pd.DataFrame(rows)
    x = np.arange(len(labels))
    width = 0.36
    for offset, task, color in [(-width / 2, "13C", BLUE), (width / 2, "1H", ORANGE)]:
        vals = []
        errs = []
        for label in labels:
            arr = dist_df[(dist_df["task"] == task) & (dist_df["distance_bin"] == label)]["fraction"].to_numpy(float)
            vals.append(float(np.nanmean(arr)) if len(arr) else 0.0)
            errs.append(float(np.nanstd(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)
        ax.plot(x + offset, vals, marker="o", color=color, linewidth=2.1, markersize=6.0, label=task)
        ax.fill_between(x + offset, np.array(vals) - np.array(errs), np.array(vals) + np.array(errs), color=color, alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Distance from target site (Å)")
    ax.set_ylabel("Gradient mass fraction")
    ax.set_title("Spatial range of 3D sensitivity", fontweight="bold", pad=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9.2)


def build_figure(atom_df: pd.DataFrame, category_df: pd.DataFrame, target_df: pd.DataFrame) -> list[Path]:
    apply_style()
    fig = plt.figure(figsize=(13.8, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.35], height_ratios=[1.0, 1.02], wspace=0.25, hspace=0.32)
    draw_protocol(fig.add_subplot(gs[0, 0]))
    draw_category_panel(fig.add_subplot(gs[0, 1]), category_df)
    draw_motif_panel(fig.add_subplot(gs[1, 0]), target_df)
    draw_distance_panel(fig.add_subplot(gs[1, 1]), atom_df)
    outputs = []
    for ext in ("png", "pdf", "svg"):
        path = OUT_DIR / f"{OUT_BASENAME}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight")
        outputs.append(path)
        shutil.copy2(path, FINAL_DIR / path.name)
    plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MMFF-only EquiformerV3 3D coordinate-gradient attribution.")
    parser.add_argument("--run_dir", default=str(RUN_DIR))
    parser.add_argument("--tasks", nargs="+", default=["13C", "1H"], choices=["13C", "1H"])
    parser.add_argument("--splits", nargs="+", default=["mono", "di", "tri"], choices=["mono", "di", "tri"])
    parser.add_argument("--epoch", type=int, default=9)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mean_batch_size", type=int, default=32)
    parser.add_argument("--max_molecules_per_split", type=int, default=16)
    parser.add_argument("--max_targets_per_molecule", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atom_df = compute_attribution(args)
    category_df, target_df = write_outputs(atom_df)
    build_figure(atom_df, category_df, target_df)
    print(f"Wrote {FINAL_DIR / (OUT_BASENAME + '.pdf')}")
    print(f"Wrote {FINAL_DIR / 'eqv3_3d_gradient_atom_level.csv'}")


if __name__ == "__main__":
    main()
