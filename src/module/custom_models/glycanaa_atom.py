from __future__ import annotations

import torch

from .baselines import AtomFeatureEncoder, bond_distance_features, edge_type_ids, mean_aggregate


class GlycanAAAtomConformerLayer(torch.nn.Module):
    """GlycanAA-style compositional relation layer for PyG atom-conformer graphs."""

    def __init__(
        self,
        hidden_channels: int,
        num_relations: int,
        edge_hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.relation = torch.nn.Embedding(num_relations, hidden_channels)
        self.edge_linear = torch.nn.Sequential(
            torch.nn.Linear(2, edge_hidden_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(edge_hidden_channels, hidden_channels),
        )
        self.message_linear = torch.nn.Linear(hidden_channels, hidden_channels)
        self.self_loop = torch.nn.Linear(hidden_channels, hidden_channels)
        self.norm = torch.nn.LayerNorm(hidden_channels)
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        src, dst = edge_index
        relations = edge_type_ids(edge_index, edge_attr, self.num_relations)
        relation_feature = self.relation(relations)
        edge_feature = bond_distance_features(pos, edge_index, edge_attr)
        message = h[src] * relation_feature + self.edge_linear(edge_feature)
        update = mean_aggregate(message, dst, h.shape[0])
        out = self.message_linear(update) + self.self_loop(h)
        out = self.activation(self.norm(out))
        return h + self.dropout(out)


class GlycanAAAtomConformerNMR(torch.nn.Module):
    """GlycanAA-inspired baseline over the same conformer atom graphs as GeqShift."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        edge_hidden_channels: int = 64,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
        num_relations: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.layers = torch.nn.ModuleList(
            [
                GlycanAAAtomConformerLayer(
                    hidden_channels=hidden_channels,
                    num_relations=num_relations,
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
            torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def encode(
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
        for layer in self.layers:
            h = layer(h, pos, edge_index, edge_attr)
        return h

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        return self.head(h)


def _components_without_nodes(num_nodes: int, adjacency: list[set[int]], excluded: set[int]) -> tuple[list[list[int]], dict[int, int]]:
    seen = set(excluded)
    components = []
    comp_id = {}
    for node in range(num_nodes):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        index = len(components)
        for member in component:
            comp_id[member] = index
        components.append(component)
    return components, comp_id


def _is_sugar_like_component(component: list[int], atom_types: list[int]) -> bool:
    carbon_count = sum(1 for atom_index in component if atom_types[atom_index] == 2)
    oxygen_count = sum(1 for atom_index in component if atom_types[atom_index] == 4)
    heavy_count = sum(1 for atom_index in component if atom_types[atom_index] > 1)
    return carbon_count >= 4 and oxygen_count >= 2 and heavy_count >= 6


def _assign_nonsugar_component(
    component: list[int],
    comp_id: dict[int, int],
    sugar_comp_to_mono: dict[int, int],
    adjacency: list[set[int]],
) -> int:
    neighbor_monos = []
    for atom_index in component:
        for neighbor in adjacency[atom_index]:
            neighbor_comp = comp_id.get(neighbor)
            if neighbor_comp in sugar_comp_to_mono:
                neighbor_monos.append(sugar_comp_to_mono[neighbor_comp])
    if neighbor_monos:
        return min(neighbor_monos)
    return 0


def _infer_residue_membership(
    x_local: torch.Tensor,
    edge_index_local: torch.Tensor,
    edge_attr_local: torch.Tensor | None,
) -> tuple[list[list[int]], list[tuple[int, int]], list[tuple[int, int]]]:
    num_nodes = int(x_local.shape[0])
    adjacency = [set() for _ in range(num_nodes)]
    if edge_attr_local is None:
        bond_mask = torch.ones(edge_index_local.shape[1], dtype=torch.bool, device=edge_index_local.device)
    else:
        bond_mask = edge_attr_local.reshape(-1) > 0
    for src, dst in edge_index_local[:, bond_mask].detach().cpu().t().tolist():
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    atom_types = x_local[:, 0].detach().cpu().long().tolist()
    bridge_oxygens = set()
    bridge_links = {}
    for atom_index, atom_type in enumerate(atom_types):
        if atom_type != 4:
            continue
        carbon_neighbors = [neighbor for neighbor in adjacency[atom_index] if atom_types[neighbor] == 2]
        if len(carbon_neighbors) < 2:
            continue
        components_without_oxygen, comp_id_without_oxygen = _components_without_nodes(
            num_nodes,
            adjacency,
            {atom_index},
        )
        neighbor_components = sorted(
            {comp_id_without_oxygen[neighbor] for neighbor in carbon_neighbors if neighbor in comp_id_without_oxygen}
        )
        sugar_neighbor_components = [
            comp_index
            for comp_index in neighbor_components
            if _is_sugar_like_component(components_without_oxygen[comp_index], atom_types)
        ]
        if len(sugar_neighbor_components) >= 2:
            bridge_oxygens.add(atom_index)
            bridge_links[atom_index] = carbon_neighbors

    raw_components, raw_comp_id = _components_without_nodes(num_nodes, adjacency, bridge_oxygens)
    if not raw_components:
        raw_components = [[node for node in range(num_nodes)]]
        raw_comp_id = {node: 0 for node in range(num_nodes)}

    sugar_component_indices = [
        comp_index
        for comp_index, component in enumerate(raw_components)
        if _is_sugar_like_component(component, atom_types)
    ]
    if sugar_component_indices:
        sugar_comp_to_mono = {
            comp_index: mono_index for mono_index, comp_index in enumerate(sugar_component_indices)
        }
        components = [list(raw_components[comp_index]) for comp_index in sugar_component_indices]
        comp_to_mono = dict(sugar_comp_to_mono)
        for comp_index, component in enumerate(raw_components):
            if comp_index in sugar_comp_to_mono:
                continue
            mono_index = _assign_nonsugar_component(
                component,
                raw_comp_id,
                sugar_comp_to_mono,
                adjacency,
            )
            comp_to_mono[comp_index] = mono_index
            components[mono_index].extend(component)
        comp_id = {
            atom_index: comp_to_mono[comp_index]
            for atom_index, comp_index in raw_comp_id.items()
        }
    else:
        components = [[node for node in range(num_nodes)]]
        comp_id = {node: 0 for node in range(num_nodes)}

    atom_to_mono = []
    for mono_index, component in enumerate(components):
        for atom_index in component:
            atom_to_mono.append((atom_index, mono_index))

    mono_edges = set()
    for oxygen_index, carbon_neighbors in bridge_links.items():
        neighbor_components = sorted({comp_id[neighbor] for neighbor in carbon_neighbors if neighbor in comp_id})
        if not neighbor_components:
            nearest = 0
            atom_to_mono.append((oxygen_index, nearest))
            continue
        for mono_index in neighbor_components:
            atom_to_mono.append((oxygen_index, mono_index))
        for src in neighbor_components:
            for dst in neighbor_components:
                if src != dst:
                    mono_edges.add((src, dst))

    if not atom_to_mono:
        atom_to_mono = [(node, 0) for node in range(num_nodes)]
    return components, atom_to_mono, sorted(mono_edges)


class GlycanAAHeteroAtomConformerLayer(torch.nn.Module):
    """Single layer over atom nodes and inferred monosaccharide residue nodes."""

    def __init__(
        self,
        hidden_channels: int,
        num_relations: int,
        edge_hidden_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.atom_layer = GlycanAAAtomConformerLayer(hidden_channels, num_relations, edge_hidden_channels, dropout)
        self.atom_to_mono = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mono_self = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mono_to_mono = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mono_to_atom = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mono_norm = torch.nn.LayerNorm(hidden_channels)
        self.atom_norm = torch.nn.LayerNorm(hidden_channels)
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        atom_h: torch.Tensor,
        mono_h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        atom_to_mono_edge: torch.Tensor,
        mono_to_atom_edge: torch.Tensor,
        mono_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_h = self.atom_layer(atom_h, pos, edge_index, edge_attr)

        atom_src, mono_dst = atom_to_mono_edge
        mono_update = mean_aggregate(self.atom_to_mono(atom_h[atom_src]), mono_dst, mono_h.shape[0])
        mono_update = mono_update + self.mono_self(mono_h)
        if mono_edge_index.numel() > 0:
            mono_src, mono_dst_edges = mono_edge_index
            mono_message = self.mono_to_mono(mono_h[mono_src])
            mono_update = mono_update + mean_aggregate(mono_message, mono_dst_edges, mono_h.shape[0])
        mono_h = mono_h + self.dropout(self.activation(self.mono_norm(mono_update)))

        mono_src, atom_dst = mono_to_atom_edge
        atom_update = mean_aggregate(self.mono_to_atom(mono_h[mono_src]), atom_dst, atom_h.shape[0])
        atom_h = atom_h + self.dropout(self.activation(self.atom_norm(atom_update)))
        return atom_h, mono_h


class GlycanAAHeteroAtomConformerNMR(torch.nn.Module):
    """Monosaccharide-atom heterogeneous conformer graph baseline."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 4,
        edge_hidden_channels: int = 64,
        dropout: float = 0.1,
        head_hidden_channels: int = 128,
        num_relations: int = 6,
    ) -> None:
        super().__init__()
        self.encoder = AtomFeatureEncoder(hidden_channels)
        self.layers = torch.nn.ModuleList(
            [
                GlycanAAHeteroAtomConformerLayer(
                    hidden_channels=hidden_channels,
                    num_relations=num_relations,
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
            torch.nn.Dropout(dropout),
            torch.nn.Linear(head_hidden_channels, 1),
        )

    def _build_hetero_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=torch.long)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        atom_to_mono_pairs = []
        mono_to_atom_pairs = []
        mono_pairs = []
        mono_offset = 0
        for graph_index in range(graph_count):
            node_indices = torch.nonzero(batch == graph_index, as_tuple=False).flatten()
            if node_indices.numel() == 0:
                continue
            start = int(node_indices[0].item())
            end = int(node_indices[-1].item()) + 1
            edge_mask = (edge_index[0] >= start) & (edge_index[0] < end)
            local_edge_index = edge_index[:, edge_mask] - start
            local_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
            components, atom_to_mono, mono_edges = _infer_residue_membership(
                x[start:end],
                local_edge_index,
                local_edge_attr,
            )
            for atom_local, mono_local in atom_to_mono:
                atom_global = start + atom_local
                mono_global = mono_offset + mono_local
                atom_to_mono_pairs.append((atom_global, mono_global))
                mono_to_atom_pairs.append((mono_global, atom_global))
            for mono_src, mono_dst in mono_edges:
                mono_pairs.append((mono_offset + mono_src, mono_offset + mono_dst))
            mono_offset += max(1, len(components))

        device = x.device
        if atom_to_mono_pairs:
            atom_to_mono_edge = torch.tensor(atom_to_mono_pairs, dtype=torch.long, device=device).t().contiguous()
            mono_to_atom_edge = torch.tensor(mono_to_atom_pairs, dtype=torch.long, device=device).t().contiguous()
        else:
            atom_to_mono_edge = torch.empty((2, 0), dtype=torch.long, device=device)
            mono_to_atom_edge = torch.empty((2, 0), dtype=torch.long, device=device)
        if mono_pairs:
            mono_edge_index = torch.tensor(mono_pairs, dtype=torch.long, device=device).t().contiguous()
        else:
            mono_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        return atom_to_mono_edge, mono_to_atom_edge, mono_edge_index, max(1, mono_offset)

    def _encode_hetero(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del norm_batch
        if batch is None:
            batch = x.new_zeros(x.shape[0], dtype=torch.long)
        atom_h = self.encoder(x)
        atom_to_mono_edge, mono_to_atom_edge, mono_edge_index, num_mono = self._build_hetero_edges(
            x,
            edge_index,
            edge_attr,
            batch,
        )
        atom_src, mono_dst = atom_to_mono_edge
        mono_h = mean_aggregate(atom_h[atom_src], mono_dst, num_mono)
        for layer in self.layers:
            atom_h, mono_h = layer(
                atom_h,
                mono_h,
                pos,
                edge_index,
                edge_attr,
                atom_to_mono_edge,
                mono_to_atom_edge,
                mono_edge_index,
            )
        mono_batch = x.new_zeros(num_mono, dtype=torch.long)
        if atom_to_mono_edge.numel() > 0:
            atom_src, mono_dst = atom_to_mono_edge
            mono_batch[mono_dst] = batch[atom_src]
        return atom_h, mono_h, mono_to_atom_edge, mono_batch

    def encode_with_mono_context(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        atom_h, mono_h, mono_to_atom_edge, _ = self._encode_hetero(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        mono_src, atom_dst = mono_to_atom_edge
        mono_context = mean_aggregate(mono_h[mono_src], atom_dst, atom_h.shape[0])
        return atom_h, mono_context

    def encode_mono_tokens(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, mono_h, _, mono_batch = self._encode_hetero(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        return mono_h, mono_batch

    def encode(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        atom_h, _ = self.encode_with_mono_context(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        return atom_h

    def encode_mono_context(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, mono_context = self.encode_with_mono_context(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        return mono_context

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        norm_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        atom_h = self.encode(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            norm_batch=norm_batch,
        )
        return self.head(atom_h)
