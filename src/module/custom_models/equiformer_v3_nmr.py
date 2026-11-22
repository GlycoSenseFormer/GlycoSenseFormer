from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


DEFAULT_EQUIFORMER_V3_ROOT = Path(__file__).resolve().parents[2] / "equiformer_v3"
LEGACY_EQUIFORMER_V3_ROOT = Path("/home/cns/糖NMR/equiformer_v3")


def _add_equiformer_v3_to_path(equiformer_v3_root: str | os.PathLike[str] | None) -> None:
    root = Path(
        equiformer_v3_root
        or os.environ.get("EQUIFORMER_V3_ROOT", DEFAULT_EQUIFORMER_V3_ROOT)
    ).expanduser()
    if root == LEGACY_EQUIFORMER_V3_ROOT:
        root = Path(os.environ.get("EQUIFORMER_V3_ROOT", DEFAULT_EQUIFORMER_V3_ROOT)).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"Equiformer v3 root was not found: {root}. "
            "Set EQUIFORMER_V3_ROOT to the repository path."
        )
    fairchem_src = str(root / "src")
    if fairchem_src not in sys.path:
        sys.path.insert(0, fairchem_src)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


GEQSHIFT_TYPE_TO_ATOMIC_NUMBER = torch.tensor(
    [
        0,
        1,
        6,
        7,
        8,
        9,
        15,
        16,
        17,
        35,
    ],
    dtype=torch.long,
)


def _load_equiformer_v3_model(equiformer_v3_root: str | os.PathLike[str] | None):
    _add_equiformer_v3_to_path(equiformer_v3_root)
    from experimental.models.equiformer_v3.equiformer_v3 import EquiformerV3_OC

    return EquiformerV3_OC


class EquiformerV3NMR(torch.nn.Module):
    """GeqShift-compatible per-atom NMR head on top of Equiformer v3."""

    def __init__(
        self,
        equiformer_v3_root: str | os.PathLike[str] | None = None,
        max_radius: float = 6.0,
        num_radial_basis: int = 64,
        max_num_elements: int = 128,
        num_layers: int = 7,
        num_channels: int = 128,
        attn_hidden_channels: int = 32,
        num_heads: int = 8,
        attn_alpha_channels: int = 64,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "merge_layer_norm",
        lmax: int = 4,
        mmax: int = 2,
        attn_grid_resolution_list: list[int] | None = None,
        ffn_grid_resolution_list: list[int] | None = None,
        edge_channels: int = 128,
        h_count_channels: int = 16,
        atom_type_channels: int = 16,
        head_hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        equiformer_v3_cls = _load_equiformer_v3_model(equiformer_v3_root)

        attn_grid_resolution_list = attn_grid_resolution_list or [14, 8]
        ffn_grid_resolution_list = ffn_grid_resolution_list or [14, 14]

        self.backbone = equiformer_v3_cls(
            use_pbc=False,
            use_pbc_single=False,
            otf_graph=False,
            regress_forces=False,
            regress_stress=False,
            direct_prediction=True,
            max_neighbors=300,
            max_radius=max_radius,
            num_radial_basis=num_radial_basis,
            max_num_elements=max_num_elements,
            num_layers=num_layers,
            num_channels=num_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,
            norm_type=norm_type,
            lmax=lmax,
            mmax=mmax,
            attn_grid_resolution_list=attn_grid_resolution_list,
            ffn_grid_resolution_list=ffn_grid_resolution_list,
            edge_channels=edge_channels,
            use_atom_edge_embedding=True,
            use_envelope=True,
            attn_activation="sep-merge_gates2_swiglu",
            use_attn_renorm=True,
            use_add_merge=False,
            use_rad_l_parametrization=True,
            softcap=None,
            attn_eps=1e-16,
            ffn_activation="sep-merge_gates2_swiglu",
            use_grid_mlp=True,
            use_gate_force_head=False,
            alpha_drop=0.0,
            attn_mask_rate=0.0,
            attn_weights_drop=0.1,
            value_drop=0.0,
            drop_path_rate=0.05,
            proj_drop=0.0,
            ffn_drop=0.0,
            use_head_reg=False,
            gradient_checkpointing_block_list=[0] * num_layers,
            avg_degree=23.395238876342773,
            enforce_max_neighbors_strictly=True,
        )

        self.register_buffer(
            "type_to_atomic_number",
            GEQSHIFT_TYPE_TO_ATOMIC_NUMBER.clone(),
            persistent=False,
        )
        self.atom_type_embedding = torch.nn.Embedding(
            len(GEQSHIFT_TYPE_TO_ATOMIC_NUMBER),
            atom_type_channels,
        )
        self.h_count_embedding = torch.nn.Embedding(8, h_count_channels)

        head_in_channels = num_channels + atom_type_channels + h_count_channels
        self.scalar_channels = num_channels
        self.head_in_channels = head_in_channels
        self.nmr_head = torch.nn.Sequential(
            torch.nn.LayerNorm(head_in_channels),
            torch.nn.Linear(head_in_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def _atomic_numbers_from_geqshift_types(self, x: torch.Tensor) -> torch.Tensor:
        atom_types = x[:, 0].long()
        if torch.any(atom_types < 0) or torch.any(atom_types >= self.type_to_atomic_number.numel()):
            bad_types = atom_types[
                (atom_types < 0) | (atom_types >= self.type_to_atomic_number.numel())
            ].detach().cpu().unique().tolist()
            raise ValueError(f"Unsupported GeqShift atom type ids: {bad_types}")
        return self.type_to_atomic_number[atom_types]

    def make_head_features(self, scalar_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        atom_types = x[:, 0].long()
        h_counts = x[:, 1].long().clamp(min=0, max=self.h_count_embedding.num_embeddings - 1)
        return torch.cat(
            [
                scalar_features,
                self.atom_type_embedding(atom_types),
                self.h_count_embedding(h_counts),
            ],
            dim=1,
        )

    @staticmethod
    def _edge_geometry(pos: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_src, edge_dst = edge_index
        edge_vec = pos[edge_src] - pos[edge_dst]
        edge_distance = torch.linalg.vector_norm(edge_vec, dim=1)

        zero_mask = edge_distance < 1e-7
        if torch.any(zero_mask):
            edge_vec = edge_vec.clone()
            edge_vec[zero_mask] = edge_vec.new_tensor([1e-3, 0.0, 0.0])
            edge_distance = torch.linalg.vector_norm(edge_vec, dim=1)

        return edge_distance, edge_vec

    def encode(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_attr, norm_batch

        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=torch.long)
        if edge_index.numel() == 0:
            raise ValueError("EquiformerV3NMR requires at least one edge in edge_index.")

        atomic_numbers = self._atomic_numbers_from_geqshift_types(x)
        edge_distance, edge_vec = self._edge_geometry(pos, edge_index)

        self.backbone.dtype = pos.dtype
        self.backbone.device = pos.device

        source_atomic_numbers = atomic_numbers[edge_index[0]]
        target_atomic_numbers = atomic_numbers[edge_index[1]]
        edge_distance, edge_envelope_weight = self.backbone._forward_edge(
            edge_distance,
            edge_vec,
        )
        node_features = self.backbone._forward_embedding(
            atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
        )
        scalar_features, _ = self.backbone._forward_blocks(
            node_features,
            source_atomic_numbers,
            target_atomic_numbers,
            edge_distance,
            edge_index,
            edge_envelope_weight,
            batch,
        )
        return scalar_features

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scalar_features = self.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        head_features = self.make_head_features(scalar_features, x)
        return self.nmr_head(head_features)
