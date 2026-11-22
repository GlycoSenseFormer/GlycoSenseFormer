from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        if path.suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(handle)
        return json.load(handle)


def add_equiformer_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--architecture",
        default="equiformer_v3",
        choices=[
            "equiformer_v3",
            "gcn",
            "gin",
            "gat",
            "mpnn",
            "painn",
            "egnn",
            "rgcn",
            "compgcn",
            "geqshift",
            "glycanaa_atom",
            "glycanaa_hetero_atom",
            "glycangt_atom",
            "eqv3_glycanaa_hetero_fusion",
            "eqv3_glycanaa_cross_attention_fusion",
            "eqv3_glycanaa_cross_attention_residual_fusion",
            "eqv3_glycanaa_cross_attention_identity_residual_fusion",
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
        help="NMR model architecture",
    )
    parser.add_argument("--equiformer_v3_root", default=None, type=str, help="Path to the Equiformer v3 repository")
    parser.add_argument("--eqv3_num_layers", default=7, type=int, help="Equiformer v3 transformer blocks")
    parser.add_argument("--eqv3_num_channels", default=128, type=int, help="Equiformer v3 spherical channels")
    parser.add_argument("--eqv3_attn_hidden_channels", default=32, type=int, help="Equiformer v3 attention hidden channels")
    parser.add_argument("--eqv3_ffn_hidden_channels", default=512, type=int, help="Equiformer v3 feed-forward hidden channels")
    parser.add_argument("--eqv3_num_radial_basis", default=64, type=int, help="Equiformer v3 radial basis size")
    parser.add_argument("--eqv3_lmax", default=4, type=int, help="Equiformer v3 maximum spherical harmonic degree")
    parser.add_argument("--eqv3_mmax", default=2, type=int, help="Equiformer v3 maximum spherical harmonic order")
    parser.add_argument("--eqv3_max_radius", default=6.0, type=float, help="Equiformer v3 graph cutoff radius")
    parser.add_argument("--baseline_hidden_channels", default=128, type=int, help="Baseline hidden channels")
    parser.add_argument("--baseline_num_layers", default=4, type=int, help="Baseline message passing layers")
    parser.add_argument("--baseline_dropout", default=0.1, type=float, help="Baseline dropout")
    parser.add_argument("--baseline_edge_hidden_channels", default=64, type=int, help="MPNN edge network hidden channels")
    parser.add_argument("--baseline_head_hidden_channels", default=128, type=int, help="Baseline head hidden channels")
    parser.add_argument("--baseline_attention_heads", default=4, type=int, help="GAT attention heads")
    parser.add_argument("--baseline_num_relations", default=5, type=int, help="R-GCN / CompGCN edge relation count")
    parser.add_argument("--geqshift_n_input", default=128, type=int, help="GeqShift input embedding channels")
    parser.add_argument("--geqshift_n_node_attr", default=128, type=int, help="GeqShift node attribute channels")
    parser.add_argument("--geqshift_n_output", default=128, type=int, help="GeqShift output scalar channels")
    parser.add_argument("--geqshift_n_layers", default=7, type=int, help="GeqShift O3Transformer layers")
    parser.add_argument("--glycangt_root", default="/home/cns/糖NMR/GlycanGT", type=str, help="Path to the GlycanGT repository")
    parser.add_argument("--glycangt_config_name", default="ss", type=str, choices=["ss", "small", "medium", "large"], help="GlycanGT TokenGT config preset")
    parser.add_argument("--glycangt_distance_bins", default=16, type=int, help="Distance buckets for GlycanGT atom-conformer edge tokens")
    parser.add_argument("--glycangt_distance_cutoff", default=6.0, type=float, help="Distance cutoff used for GlycanGT atom-conformer buckets")
    parser.add_argument("--glycangt_freeze_encoder", action="store_true", help="Freeze the TokenGT encoder and train only the NMR head")
    parser.add_argument("--fusion_hidden_channels", default=256, type=int, help="Fusion residual head hidden channels")
    parser.add_argument("--fusion_dropout", default=0.1, type=float, help="Fusion residual head dropout")
    parser.add_argument("--no_fusion_zero_init", action="store_true", help="Disable zero initialization of the fusion residual head")
    parser.add_argument("--bias_hidden_channels", default=128, type=int, help="Residue-context bias head hidden channels")
    parser.add_argument("--bias_dropout", default=0.1, type=float, help="Residue-context bias head dropout")
    parser.add_argument("--no_bias_zero_init", action="store_true", help="Disable zero initialization of the residue-context bias head")
