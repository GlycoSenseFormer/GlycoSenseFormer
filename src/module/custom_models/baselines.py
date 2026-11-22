from __future__ import annotations

import torch
from e3nn import o3
from torch_geometric.nn import GATConv, GCNConv, GINConv, NNConv, RGCNConv

from model.model import O3Transformer
from model.norms import EquivariantLayerNorm


class AtomFeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        max_atom_type: int = 16,
        max_h_count: int = 8,
    ) -> None:
        super().__init__()
        atom_channels = hidden_channels // 2
        h_channels = hidden_channels - atom_channels
        self.atom_embedding = torch.nn.Embedding(max_atom_type, atom_channels)
        self.h_embedding = torch.nn.Embedding(max_h_count, h_channels)
        self.norm = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atom_type = x[:, 0].long().clamp(min=0, max=self.atom_embedding.num_embeddings - 1)
        h_count = x[:, 1].long().clamp(min=0, max=self.h_embedding.num_embeddings - 1)
        return self.norm(torch.cat([self.atom_embedding(atom_type), self.h_embedding(h_count)], dim=1))


def edge_type_ids(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    num_relations: int,
) -> torch.Tensor:
    if edge_attr is None:
        return edge_index.new_zeros(edge_index.shape[1])
    return edge_attr.long().reshape(-1).clamp(min=0, max=num_relations - 1)


def bond_distance_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
) -> torch.Tensor:
    src, dst = edge_index
    distance = torch.linalg.vector_norm(pos[src] - pos[dst], dim=1, keepdim=True)
    if edge_attr is None:
        bond_flag = distance.new_zeros((edge_index.shape[1], 1))
    else:
        bond_flag = edge_attr.float().reshape(-1, 1)
    return torch.cat([bond_flag, distance], dim=1)


def mean_aggregate(messages: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    output = messages.new_zeros((num_nodes, messages.shape[1]))
    output.index_add_(0, dst, messages)
    counts = messages.new_zeros((num_nodes, 1))
    counts.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=messages.dtype, device=messages.device))
    return output / counts.clamp_min(1.0)


def vector_mean_aggregate(messages: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    channels = messages.shape[1]
    flat = mean_aggregate(messages.reshape(messages.shape[0], channels * 3), dst, num_nodes)
    return flat.reshape(num_nodes, channels, 3)


class GCNNMR(torch.nn.Module):
    """GCN baseline with the same per-atom output contract as EquiformerV3NMR."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del pos, edge_attr, batch, norm_batch
        h = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            update = self.activation(norm(conv(h, edge_index)))
            h = h + self.dropout(update)
        return self.head(h)


class MPNNNMR(torch.nn.Module):
    """NNConv message-passing baseline using bond flags and inter-atomic distances."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 4,
        edge_hidden_channels: int = 64,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            edge_network = torch.nn.Sequential(
                torch.nn.Linear(2, edge_hidden_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(edge_hidden_channels, hidden_channels * hidden_channels),
            )
            self.convs.append(NNConv(hidden_channels, hidden_channels, edge_network, aggr="mean"))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch, norm_batch
        h = self.encoder(x)
        edge_features = bond_distance_features(pos, edge_index, edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            update = self.activation(norm(conv(h, edge_index, edge_features)))
            h = h + self.dropout(update)
        return self.head(h)


class GINNMR(torch.nn.Module):
    """GIN baseline using atom identity features and the molecular graph."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(torch.nn.LayerNorm(hidden_channels))
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del pos, edge_attr, batch, norm_batch
        h = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            update = self.activation(norm(conv(h, edge_index)))
            h = h + self.dropout(update)
        return self.head(h)


class EGNNLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        edge_hidden_channels: int,
        dropout: float,
        coord_update_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.coord_update_scale = coord_update_scale
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2 + 2, edge_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(edge_hidden_channels, edge_hidden_channels),
            torch.nn.SiLU(),
        )
        self.coord_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_hidden_channels, edge_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(edge_hidden_channels, 1),
        )
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels + edge_hidden_channels, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        coord_diff = pos[dst] - pos[src]
        radial = torch.sum(coord_diff * coord_diff, dim=1, keepdim=True)
        if edge_attr is None:
            bond_flag = radial.new_zeros((edge_index.shape[1], 1))
        else:
            bond_flag = edge_attr.float().reshape(-1, 1)
        messages = self.edge_mlp(torch.cat([h[src], h[dst], bond_flag, radial], dim=1))
        coord_weight = self.coord_mlp(messages).clamp(min=-10.0, max=10.0)
        pos_update = mean_aggregate(coord_diff * coord_weight, dst, h.shape[0])
        pos = pos + self.coord_update_scale * pos_update
        aggregated = mean_aggregate(messages, dst, h.shape[0])
        update = self.node_mlp(torch.cat([h, aggregated], dim=1))
        h = self.norm(h + self.dropout(update))
        return h, pos


class EGNNNMR(torch.nn.Module):
    """EGNN-style 3D message-passing baseline for per-atom NMR prediction."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        edge_hidden_channels: int = 64,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
        coord_update_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.layers = torch.nn.ModuleList(
            [
                EGNNLayer(
                    hidden_channels=hidden_channels,
                    edge_hidden_channels=edge_hidden_channels,
                    dropout=dropout,
                    coord_update_scale=coord_update_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch, norm_batch
        h = self.encoder(x)
        pos_work = pos.float()
        for layer in self.layers:
            h, pos_work = layer(h, pos_work, edge_index, edge_attr)
        return self.head(h)


class PaiNNLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        edge_hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.filter_net = torch.nn.Sequential(
            torch.nn.Linear(2, edge_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(edge_hidden_channels, hidden_channels * 3),
        )
        self.scalar_message = torch.nn.Linear(hidden_channels, hidden_channels)
        self.scalar_update = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.vector_gate = torch.nn.Linear(hidden_channels, hidden_channels)
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        edge_features = bond_distance_features(pos, edge_index, edge_attr)
        distance = edge_features[:, 1:2].clamp_min(1.0e-6)
        direction = (pos[src] - pos[dst]) / distance
        scalar_filter, vector_filter, direction_filter = self.filter_net(edge_features).chunk(3, dim=1)

        scalar_messages = scalar_filter * self.scalar_message(h[src])
        vector_messages = (
            vector_filter.unsqueeze(-1) * v[src]
            + direction_filter.unsqueeze(-1) * direction.unsqueeze(1)
        )

        scalar_agg = mean_aggregate(scalar_messages, dst, h.shape[0])
        vector_agg = vector_mean_aggregate(vector_messages, dst, h.shape[0])
        v = v + vector_agg

        vector_norm = torch.linalg.vector_norm(v, dim=2)
        update = self.scalar_update(torch.cat([scalar_agg, vector_norm], dim=1))
        h = self.norm(h + self.dropout(update))
        v = v * torch.sigmoid(self.vector_gate(h)).unsqueeze(-1)
        return h, v


class PaiNNNMR(torch.nn.Module):
    """PaiNN-style scalar/vector message-passing baseline."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        edge_hidden_channels: int = 64,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.layers = torch.nn.ModuleList(
            [
                PaiNNLayer(
                    hidden_channels=hidden_channels,
                    edge_hidden_channels=edge_hidden_channels,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch, norm_batch
        h = self.encoder(x)
        v = pos.new_zeros((h.shape[0], h.shape[1], 3))
        for layer in self.layers:
            h, v = layer(h, v, pos.float(), edge_index, edge_attr)
        return self.head(h)


class GATNMR(torch.nn.Module):
    """Graph attention baseline using bond flags and distances as edge features."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads for GATNMR")
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList(
            [
                GATConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=2,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del batch, norm_batch
        h = self.encoder(x)
        edge_features = bond_distance_features(pos, edge_index, edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            update = self.activation(norm(conv(h, edge_index, edge_features)))
            h = h + self.dropout(update)
        return self.head(h)


class RGCNNMR(torch.nn.Module):
    """Relational GCN baseline over non-bond and bond-type edge relations."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
        num_relations: int = 5,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList(
            [
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations=num_relations,
                    num_bases=min(num_relations, 4),
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del pos, batch, norm_batch
        h = self.encoder(x)
        relations = edge_type_ids(edge_index, edge_attr, self.num_relations)
        for conv, norm in zip(self.convs, self.norms):
            update = self.activation(norm(conv(h, edge_index, relations)))
            h = h + self.dropout(update)
        return self.head(h)


class CompGCNLayer(torch.nn.Module):
    """Small compositional relation layer for the CompGCN baseline."""

    def __init__(self, hidden_channels: int, num_relations: int) -> None:
        super().__init__()
        self.relation_embedding = torch.nn.Embedding(num_relations, hidden_channels)
        self.message = torch.nn.Linear(hidden_channels, hidden_channels)
        self.self_update = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        relation = self.relation_embedding(relations)
        messages = self.message(h[src] + relation)
        aggregated = mean_aggregate(messages, dst, h.shape[0])
        return self.self_update(h) + aggregated


class CompGCNNMR(torch.nn.Module):
    """CompGCN-style relation-composition baseline for per-atom NMR prediction."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
        num_relations: int = 5,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.convs = torch.nn.ModuleList(
            [CompGCNLayer(hidden_channels, num_relations) for _ in range(num_layers)]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.Linear(hidden_channels, head_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del pos, batch, norm_batch
        h = self.encoder(x)
        relations = edge_type_ids(edge_index, edge_attr, self.num_relations)
        for conv, norm in zip(self.convs, self.norms):
            update = self.activation(norm(conv(h, edge_index, relations)))
            h = h + self.dropout(update)
        return self.head(h)


class GeqShiftNMR(torch.nn.Module):
    """Original GeqShift O3Transformer baseline with the current training contract."""

    def __init__(
        self,
        n_input: int = 128,
        n_node_attr: int = 128,
        n_output: int = 128,
        n_layers: int = 7,
    ) -> None:
        super().__init__()
        self.model = O3Transformer(
            norm=EquivariantLayerNorm,
            n_input=n_input,
            n_node_attr=n_node_attr,
            n_output=n_output,
            irreps_hidden=o3.Irreps("64x0e + 32x1o + 8x2e"),
            n_layers=n_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
