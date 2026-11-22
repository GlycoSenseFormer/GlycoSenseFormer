#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "data/cases"
PRED_CSV = FIG_DIR / "fig5_cross_attention_seed0_2_3_prediction_values.csv"
OUT_MD = FIG_DIR / "fig5_cross_attention_seed0_2_3_predictions_by_figure_number.md"

TRUE_FILES = {
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

RESIDUE_ALIASES = [
    ("GlcpNAc", "glcnac"),
    ("GlcNAc", "glcnac"),
    ("GlcN", "glcn"),
    ("Glcp", "glc"),
    ("Galp", "gal"),
    ("Fucp", "fuc"),
    ("Xylp", "xyl"),
    ("Arap", "ara"),
    ("Fruf", "fru"),
    ("Manp", "man"),
]


def read_prediction_rows() -> list[dict[str, str]]:
    with PRED_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_main_table(path: Path) -> tuple[str, list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    title = next((line.removeprefix("#").strip() for line in lines if line.startswith("# ")), path.stem)
    header_idx = next(
        idx
        for idx, line in enumerate(lines)
        if line.startswith("| 编号 | 图中基团 | 对应原子/氢标签 |")
    )
    rows = []
    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            break
        cells = split_md_row(line)
        if len(cells) < 6 or not cells[0].isdigit():
            break
        rows.append(
            {
                "编号": cells[0],
                "图中基团": cells[1],
                "对应原子/氢标签": cells[2],
                "δ13C true (ppm)": cells[3],
                "δ1H true (ppm)": cells[4],
                "备注": cells[5],
            }
        )
    return title, rows


def is_missing_true(cell: str) -> bool:
    return cell in {"—", "-", "未赋值", ""}


def extract_eval_true(cell: str, note: str) -> float | None:
    if is_missing_true(cell):
        return None
    if "/" in cell:
        for pattern in (r"均值为\s*`?([0-9]+(?:\.[0-9]+)?)", r"评估用\s*`?([0-9]+(?:\.[0-9]+)?)"):
            match = re.search(pattern, note)
            if match:
                return float(match.group(1))
    match = re.search(r"[0-9]+(?:\.[0-9]+)?", cell)
    return float(match.group(0)) if match else None


def extract_atom_name(label: str) -> str | None:
    matches = re.findall(r"\bC\d+(?:_[A-Za-z0-9]+)?\b", label)
    return matches[0] if matches else None


def extract_residue(label: str) -> str | None:
    for text, residue in RESIDUE_ALIASES:
        if text in label:
            return residue
    return None


def row_key(row: dict[str, str]) -> tuple[int, int]:
    return int(row["atom_order"]), int(row["seed"])


def fmt(value: float) -> str:
    return f"{value:.3f}"


def prediction_cells(
    panel: str,
    task: str,
    table_row: dict[str, str],
    pred_by_panel_task_seed0: dict[tuple[str, str], list[dict[str, str]]],
    pred_by_panel_task_order_seed: dict[tuple[str, str, int, int], dict[str, str]],
) -> tuple[list[str], bool]:
    true_col = "δ13C true (ppm)" if task == "13C" else "δ1H true (ppm)"
    true_text = table_row[true_col]
    true_value = extract_eval_true(true_text, table_row["备注"])
    if true_value is None:
        fill = "—" if true_text == "—" else "未预测"
        return [fill, fill, fill, fill], False

    atom_name = extract_atom_name(table_row["对应原子/氢标签"])
    if atom_name is None:
        raise RuntimeError(f"{panel} {task} row {table_row['编号']} has true value but no atom name: {table_row}")

    residue = extract_residue(table_row["对应原子/氢标签"])
    candidates = [
        row
        for row in pred_by_panel_task_seed0[(panel, task)]
        if row["atom_name"] == atom_name and (residue is None or row["residue"] == residue)
    ]
    if not candidates:
        if "未纳入" in table_row["备注"] or "不作为" in table_row["备注"]:
            return ["未预测", "未预测", "未预测", "未预测"], False
        raise RuntimeError(f"{panel} {task} row {table_row['编号']} has no candidate for {atom_name}/{residue}")

    candidates.sort(key=lambda row: abs(float(row["true_ppm"]) - true_value))
    best = candidates[0]
    delta = abs(float(best["true_ppm"]) - true_value)
    if delta > 0.08:
        raise RuntimeError(
            f"{panel} {task} row {table_row['编号']} true mismatch: "
            f"table={true_value} best={best['true_ppm']} label={table_row['对应原子/氢标签']}"
        )

    order = int(best["atom_order"])
    preds = []
    for seed in (0, 2, 3):
        pred_row = pred_by_panel_task_order_seed[(panel, task, order, seed)]
        preds.append(float(pred_row["pred_ppm"]))
    mean_pred = sum(preds) / len(preds)
    return [fmt(preds[0]), fmt(preds[1]), fmt(preds[2]), fmt(mean_pred)], True


def build_markdown() -> str:
    prediction_rows = read_prediction_rows()
    pred_by_panel_task_seed0: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    pred_by_panel_task_order_seed: dict[tuple[str, str, int, int], dict[str, str]] = {}
    first_row_by_panel: dict[str, dict[str, str]] = {}
    for row in prediction_rows:
        panel = row["panel"]
        task = row["task"]
        seed = int(row["seed"])
        order = int(row["atom_order"])
        if seed == 0:
            pred_by_panel_task_seed0[(panel, task)].append(row)
        pred_by_panel_task_order_seed[(panel, task, order, seed)] = row
        first_row_by_panel.setdefault(panel, row)

    lines = [
        "# Fig.5 S1-S12 Cross-attention seed0/2/3 预测值（按图中编号）",
        "",
        "模型：`EquiformerV3-GlycanAAIdentityResidualCrossAttentionFusion` / `New cross-attn learnable alpha bs16`。",
        "",
        "说明：",
        "",
        "- 本文件把 seed0/2/3 的预测值映射回 S1-S12 true-shift 表中的图中编号。",
        "- `pred mean` 是 seed0、seed2、seed3 三个 checkpoint 预测值的算术平均。",
        "- `未预测` 表示原始真实位移为 `未赋值`，或虽有原始值但未纳入 SDF 评估靶点；`—` 表示该原子本身不对应当前核种的 1H/13C 靶点。",
        "- CH2/CH3 等多个氢合并的位点，按验证集评估时使用的合并靶值匹配预测；原始氢的说明保留在备注列。",
        "",
    ]

    matched_counts = {"13C": 0, "1H": 0}
    for panel in sorted(TRUE_FILES, key=lambda name: int(name.removeprefix("S"))):
        title, table_rows = parse_main_table(FIG_DIR / TRUE_FILES[panel])
        first = first_row_by_panel[panel]
        lines.extend(
            [
                f"## {panel} {first['display_name']}",
                "",
                f"- True-shift table: `{TRUE_FILES[panel]}`",
                f"- Validation item: `{first['split']}/{first['sample_idx']}`; SDF name: `{first['sdf_name']}`",
                "",
                "| 编号 | 图中基团 | 对应原子/氢标签 | δ13C true | 13C pred s0 | 13C pred s2 | 13C pred s3 | 13C pred mean | δ1H true | 1H pred s0 | 1H pred s2 | 1H pred s3 | 1H pred mean | 备注 |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in table_rows:
            c_cells, c_matched = prediction_cells(
                panel,
                "13C",
                row,
                pred_by_panel_task_seed0,
                pred_by_panel_task_order_seed,
            )
            h_cells, h_matched = prediction_cells(
                panel,
                "1H",
                row,
                pred_by_panel_task_seed0,
                pred_by_panel_task_order_seed,
            )
            matched_counts["13C"] += int(c_matched)
            matched_counts["1H"] += int(h_matched)
            lines.append(
                f"| {row['编号']} | {row['图中基团']} | {row['对应原子/氢标签']} | "
                f"{row['δ13C true (ppm)']} | {c_cells[0]} | {c_cells[1]} | {c_cells[2]} | {c_cells[3]} | "
                f"{row['δ1H true (ppm)']} | {h_cells[0]} | {h_cells[1]} | {h_cells[2]} | {h_cells[3]} | "
                f"{row['备注']} |"
            )
        lines.append("")

    expected_counts = {
        "13C": len({(row["panel"], row["atom_order"]) for row in prediction_rows if row["task"] == "13C"}),
        "1H": len({(row["panel"], row["atom_order"]) for row in prediction_rows if row["task"] == "1H"}),
    }
    if matched_counts != expected_counts:
        raise RuntimeError(f"Matched counts {matched_counts} do not equal expected counts {expected_counts}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUT_MD.write_text(build_markdown(), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
