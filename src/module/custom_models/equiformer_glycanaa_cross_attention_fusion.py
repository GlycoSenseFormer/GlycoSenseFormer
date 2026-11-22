from __future__ import annotations

import os

import torch

from .equiformer_v3_nmr import EquiformerV3NMR
from .glycanaa_atom import GlycanAAHeteroAtomConformerNMR


class EquiformerV3GlycanAACrossAttentionFusionNMR(torch.nn.Module):
    """Feature-level Equiformer/GlycanAA cross-attention fusion with direct prediction."""

    def __init__(
        self,
        equiformer_v3_root: str | os.PathLike[str] | None = None,
        eqv3_num_layers: int = 7,
        eqv3_num_channels: int = 128,
        eqv3_attn_hidden_channels: int = 32,
        eqv3_ffn_hidden_channels: int = 512,
        eqv3_num_radial_basis: int = 64,
        eqv3_lmax: int = 4,
        eqv3_mmax: int = 2,
        eqv3_max_radius: float = 6.0,
        glycanaa_hidden_channels: int = 128,
        glycanaa_num_layers: int = 4,
        glycanaa_edge_hidden_channels: int = 64,
        glycanaa_dropout: float = 0.1,
        glycanaa_head_hidden_channels: int = 128,
        glycanaa_num_relations: int = 6,
        fusion_hidden_channels: int = 256,
        fusion_dropout: float = 0.1,
        fusion_zero_init: bool = False,
        cross_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        del fusion_zero_init
        self.equiformer = EquiformerV3NMR(
            equiformer_v3_root=equiformer_v3_root,
            max_radius=eqv3_max_radius,
            num_radial_basis=eqv3_num_radial_basis,
            num_layers=eqv3_num_layers,
            num_channels=eqv3_num_channels,
            attn_hidden_channels=eqv3_attn_hidden_channels,
            ffn_hidden_channels=eqv3_ffn_hidden_channels,
            lmax=eqv3_lmax,
            mmax=eqv3_mmax,
        )
        self.glycanaa = GlycanAAHeteroAtomConformerNMR(
            hidden_channels=glycanaa_hidden_channels,
            num_layers=glycanaa_num_layers,
            edge_hidden_channels=glycanaa_edge_hidden_channels,
            dropout=glycanaa_dropout,
            head_hidden_channels=glycanaa_head_hidden_channels,
            num_relations=glycanaa_num_relations,
        )

        eq_channels = self.equiformer.scalar_channels
        if eq_channels % cross_attention_heads != 0:
            raise ValueError(
                f"cross_attention_heads={cross_attention_heads} must divide eq_channels={eq_channels}."
            )
        self.eq_norm = torch.nn.LayerNorm(eq_channels)
        self.mono_norm = torch.nn.LayerNorm(glycanaa_hidden_channels)
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=eq_channels,
            num_heads=cross_attention_heads,
            dropout=fusion_dropout,
            batch_first=True,
            kdim=glycanaa_hidden_channels,
            vdim=glycanaa_hidden_channels,
        )
        self.scalar_fusion = torch.nn.Sequential(
            torch.nn.LayerNorm(eq_channels * 2),
            torch.nn.Linear(eq_channels * 2, fusion_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(fusion_dropout),
            torch.nn.Linear(fusion_hidden_channels, eq_channels),
            torch.nn.LayerNorm(eq_channels),
        )

    def _cross_attention_by_graph(
        self,
        eq_scalar: torch.Tensor,
        mono_tokens: torch.Tensor,
        atom_batch: torch.Tensor | None,
        mono_batch: torch.Tensor,
    ) -> torch.Tensor:
        if atom_batch is None:
            atom_batch = eq_scalar.new_zeros(eq_scalar.shape[0], dtype=torch.long)

        context = eq_scalar.new_zeros(eq_scalar.shape)
        graph_count = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        query_all = self.eq_norm(eq_scalar)
        mono_all = self.mono_norm(mono_tokens)
        for graph_index in range(graph_count):
            atom_mask = atom_batch == graph_index
            if not torch.any(atom_mask):
                continue
            mono_mask = mono_batch == graph_index
            if not torch.any(mono_mask):
                continue
            query = query_all[atom_mask].unsqueeze(0)
            key_value = mono_all[mono_mask].unsqueeze(0)
            attended, _ = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=False,
            )
            context[atom_mask] = attended.squeeze(0)
        return context

    def fuse_scalar(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eq_scalar = self.equiformer.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_tokens, mono_batch = self.glycanaa.encode_mono_tokens(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        attn_context = self._cross_attention_by_graph(
            eq_scalar=eq_scalar,
            mono_tokens=mono_tokens,
            atom_batch=batch,
            mono_batch=mono_batch,
        )
        fused_scalar = self.scalar_fusion(torch.cat([eq_scalar, attn_context], dim=1))
        return fused_scalar, attn_context

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused_scalar, _ = self.fuse_scalar(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        return self.equiformer.nmr_head(self.equiformer.make_head_features(fused_scalar, x))


class EquiformerV3GlycanAAResidualCrossAttentionFusionNMR(
    EquiformerV3GlycanAACrossAttentionFusionNMR
):
    """Small residual cross-attention feature adapter with direct prediction."""

    def __init__(
        self,
        equiformer_v3_root: str | os.PathLike[str] | None = None,
        eqv3_num_layers: int = 7,
        eqv3_num_channels: int = 128,
        eqv3_attn_hidden_channels: int = 32,
        eqv3_ffn_hidden_channels: int = 512,
        eqv3_num_radial_basis: int = 64,
        eqv3_lmax: int = 4,
        eqv3_mmax: int = 2,
        eqv3_max_radius: float = 6.0,
        glycanaa_hidden_channels: int = 128,
        glycanaa_num_layers: int = 4,
        glycanaa_edge_hidden_channels: int = 64,
        glycanaa_dropout: float = 0.1,
        glycanaa_head_hidden_channels: int = 128,
        glycanaa_num_relations: int = 6,
        fusion_hidden_channels: int = 256,
        fusion_dropout: float = 0.1,
        fusion_zero_init: bool = True,
        cross_attention_heads: int = 4,
        scalar_alpha_init: float = 0.03,
        scalar_alpha_learnable: bool = True,
    ) -> None:
        super().__init__(
            equiformer_v3_root=equiformer_v3_root,
            eqv3_num_layers=eqv3_num_layers,
            eqv3_num_channels=eqv3_num_channels,
            eqv3_attn_hidden_channels=eqv3_attn_hidden_channels,
            eqv3_ffn_hidden_channels=eqv3_ffn_hidden_channels,
            eqv3_num_radial_basis=eqv3_num_radial_basis,
            eqv3_lmax=eqv3_lmax,
            eqv3_mmax=eqv3_mmax,
            eqv3_max_radius=eqv3_max_radius,
            glycanaa_hidden_channels=glycanaa_hidden_channels,
            glycanaa_num_layers=glycanaa_num_layers,
            glycanaa_edge_hidden_channels=glycanaa_edge_hidden_channels,
            glycanaa_dropout=glycanaa_dropout,
            glycanaa_head_hidden_channels=glycanaa_head_hidden_channels,
            glycanaa_num_relations=glycanaa_num_relations,
            fusion_hidden_channels=fusion_hidden_channels,
            fusion_dropout=fusion_dropout,
            fusion_zero_init=fusion_zero_init,
            cross_attention_heads=cross_attention_heads,
        )
        eq_channels = self.equiformer.scalar_channels
        self.attn_adapter = torch.nn.Sequential(
            torch.nn.LayerNorm(eq_channels * 2),
            torch.nn.Linear(eq_channels * 2, fusion_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(fusion_dropout),
            torch.nn.Linear(fusion_hidden_channels, eq_channels),
        )
        self.residual_norm = torch.nn.LayerNorm(eq_channels)
        scalar_alpha = torch.tensor(float(scalar_alpha_init))
        if scalar_alpha_learnable:
            self.scalar_alpha = torch.nn.Parameter(scalar_alpha)
        else:
            self.register_buffer("scalar_alpha", scalar_alpha)
        if fusion_zero_init:
            torch.nn.init.zeros_(self.attn_adapter[-1].weight)
            torch.nn.init.zeros_(self.attn_adapter[-1].bias)

    def fuse_scalar(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eq_scalar = self.equiformer.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_tokens, mono_batch = self.glycanaa.encode_mono_tokens(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        attn_context = self._cross_attention_by_graph(
            eq_scalar=eq_scalar,
            mono_tokens=mono_tokens,
            atom_batch=batch,
            mono_batch=mono_batch,
        )
        adapter_input = torch.cat([eq_scalar, attn_context], dim=1)
        adapter_delta = self.attn_adapter(adapter_input)
        fused_scalar = self.residual_norm(eq_scalar + self.scalar_alpha * adapter_delta)
        return fused_scalar, attn_context


class EquiformerV3GlycanAAIdentityResidualCrossAttentionFusionNMR(
    EquiformerV3GlycanAAResidualCrossAttentionFusionNMR
):
    """Small residual cross-attention adapter without post-residual normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_norm = torch.nn.Identity()


class EquiformerV3GlycanAAZeroMonoTokenResidualCrossAttentionFusionNMR(
    EquiformerV3GlycanAAResidualCrossAttentionFusionNMR
):
    """Single-direction cross-attention ablation with GlycanAA mono token content zeroed."""

    def fuse_scalar(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eq_scalar = self.equiformer.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_tokens, mono_batch = self.glycanaa.encode_mono_tokens(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_tokens = torch.zeros_like(mono_tokens)
        attn_context = self._cross_attention_by_graph(
            eq_scalar=eq_scalar,
            mono_tokens=mono_tokens,
            atom_batch=batch,
            mono_batch=mono_batch,
        )
        adapter_input = torch.cat([eq_scalar, attn_context], dim=1)
        adapter_delta = self.attn_adapter(adapter_input)
        fused_scalar = self.residual_norm(eq_scalar + self.scalar_alpha * adapter_delta)
        return fused_scalar, attn_context


class EquiformerV3GlycanAAIdentityResidualZeroMonoTokenCrossAttentionFusionNMR(
    EquiformerV3GlycanAAZeroMonoTokenResidualCrossAttentionFusionNMR
):
    """Zero-mono-token single-direction cross-attention ablation without post-residual normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_norm = torch.nn.Identity()


class EquiformerV3GlycanAABidirectionalResidualCrossAttentionFusionNMR(
    EquiformerV3GlycanAAResidualCrossAttentionFusionNMR
):
    """Residual adapter with bidirectional Equiformer/GlycanAA feature attention."""

    def __init__(
        self,
        *args,
        fusion_hidden_channels: int = 256,
        fusion_dropout: float = 0.1,
        cross_attention_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            fusion_hidden_channels=fusion_hidden_channels,
            fusion_dropout=fusion_dropout,
            cross_attention_heads=cross_attention_heads,
            **kwargs,
        )
        eq_channels = self.equiformer.scalar_channels
        mono_channels = int(self.mono_norm.normalized_shape[0])
        if mono_channels % cross_attention_heads != 0:
            raise ValueError(
                f"cross_attention_heads={cross_attention_heads} must divide mono_channels={mono_channels}."
            )
        self.reverse_cross_attention = torch.nn.MultiheadAttention(
            embed_dim=mono_channels,
            num_heads=cross_attention_heads,
            dropout=fusion_dropout,
            batch_first=True,
            kdim=eq_channels,
            vdim=eq_channels,
        )
        self.bidirectional_context_mix = torch.nn.Sequential(
            torch.nn.LayerNorm(eq_channels * 2),
            torch.nn.Linear(eq_channels * 2, fusion_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(fusion_dropout),
            torch.nn.Linear(fusion_hidden_channels, eq_channels),
        )

    def _cross_attention_by_graph(
        self,
        eq_scalar: torch.Tensor,
        mono_tokens: torch.Tensor,
        atom_batch: torch.Tensor | None,
        mono_batch: torch.Tensor,
    ) -> torch.Tensor:
        if atom_batch is None:
            atom_batch = eq_scalar.new_zeros(eq_scalar.shape[0], dtype=torch.long)

        context = eq_scalar.new_zeros(eq_scalar.shape)
        graph_count = int(atom_batch.max().item()) + 1 if atom_batch.numel() else 1
        query_all = self.eq_norm(eq_scalar)
        mono_all = self.mono_norm(mono_tokens)
        for graph_index in range(graph_count):
            atom_mask = atom_batch == graph_index
            if not torch.any(atom_mask):
                continue
            mono_mask = mono_batch == graph_index
            if not torch.any(mono_mask):
                continue

            atom_query = query_all[atom_mask].unsqueeze(0)
            mono_key_value = mono_all[mono_mask].unsqueeze(0)

            atom_from_mono, _ = self.cross_attention(
                query=atom_query,
                key=mono_key_value,
                value=mono_key_value,
                need_weights=False,
            )
            mono_from_atom, _ = self.reverse_cross_attention(
                query=mono_key_value,
                key=atom_query,
                value=atom_query,
                need_weights=False,
            )
            atom_from_updated_mono, _ = self.cross_attention(
                query=atom_query,
                key=mono_key_value + mono_from_atom,
                value=mono_key_value + mono_from_atom,
                need_weights=False,
            )
            mixed_context = self.bidirectional_context_mix(
                torch.cat([atom_from_mono.squeeze(0), atom_from_updated_mono.squeeze(0)], dim=1)
            )
            context[atom_mask] = mixed_context
        return context


class EquiformerV3GlycanAAIdentityResidualBidirectionalCrossAttentionFusionNMR(
    EquiformerV3GlycanAABidirectionalResidualCrossAttentionFusionNMR
):
    """Bidirectional residual cross-attention adapter without post-residual normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_norm = torch.nn.Identity()


class EquiformerV3GlycanAAZeroMonoTokenBidirectionalResidualCrossAttentionFusionNMR(
    EquiformerV3GlycanAABidirectionalResidualCrossAttentionFusionNMR
):
    """Bidirectional attention ablation with GlycanAA mono token content zeroed."""

    def fuse_scalar(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eq_scalar = self.equiformer.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_tokens, mono_batch = self.glycanaa.encode_mono_tokens(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_tokens = torch.zeros_like(mono_tokens)
        attn_context = self._cross_attention_by_graph(
            eq_scalar=eq_scalar,
            mono_tokens=mono_tokens,
            atom_batch=batch,
            mono_batch=mono_batch,
        )
        adapter_input = torch.cat([eq_scalar, attn_context], dim=1)
        adapter_delta = self.attn_adapter(adapter_input)
        fused_scalar = self.residual_norm(eq_scalar + self.scalar_alpha * adapter_delta)
        return fused_scalar, attn_context


class EquiformerV3GlycanAAIdentityResidualZeroMonoTokenBidirectionalCrossAttentionFusionNMR(
    EquiformerV3GlycanAAZeroMonoTokenBidirectionalResidualCrossAttentionFusionNMR
):
    """Zero-mono-token bidirectional ablation without post-residual normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_norm = torch.nn.Identity()


class EquiformerV3GlycanAAMonoContextResidualCrossAttentionFusionNMR(
    EquiformerV3GlycanAAResidualCrossAttentionFusionNMR
):
    """Residual cross-attention using atom-aligned GlycanAA monosaccharide context as K/V."""

    def fuse_scalar(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eq_scalar = self.equiformer.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_context = self.glycanaa.encode_mono_context(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        attn_context = self._cross_attention_by_graph(
            eq_scalar=eq_scalar,
            mono_tokens=mono_context,
            atom_batch=batch,
            mono_batch=batch if batch is not None else x.new_zeros(x.shape[0], dtype=torch.long),
        )
        adapter_input = torch.cat([eq_scalar, attn_context], dim=1)
        adapter_delta = self.attn_adapter(adapter_input)
        fused_scalar = self.residual_norm(eq_scalar + self.scalar_alpha * adapter_delta)
        return fused_scalar, attn_context


class EquiformerV3GlycanAAIdentityResidualMonoContextCrossAttentionFusionNMR(
    EquiformerV3GlycanAAMonoContextResidualCrossAttentionFusionNMR
):
    """Mono-context residual cross-attention adapter without post-residual normalization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_norm = torch.nn.Identity()
