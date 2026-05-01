"""
FedIPE client model.

Each platform owns:
- a private explicit adapter
- a private low-level graph layer

All platforms federate:
- the explicit projector
- high-level graph layers
- the implicit alignment/projector
- the final prediction head
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from data.federated_platforms import PlatformGraphData


SHARED_PARAMETER_PREFIXES = (
    "explicit_projector.",
    "shared_graph_encoder.",
    "implicit_projector.",
    "subgraph_head.",
    "predictor.",
)


@dataclass
class ClientForwardOutput:
    predictions: torch.Tensor
    explicit_embedding: torch.Tensor
    implicit_embedding: torch.Tensor
    anchor_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    subgraph_embeddings: torch.Tensor
    structural_attributes: torch.Tensor


def _safe_diameter(graph: nx.Graph) -> float:
    if graph.number_of_nodes() <= 1:
        return 0.0
    try:
        if nx.is_connected(graph):
            return float(nx.diameter(graph))
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc).copy()
        return float(nx.diameter(subgraph)) if subgraph.number_of_nodes() > 1 else 0.0
    except Exception:
        return 0.0


def _graph_centralization(graph: nx.Graph) -> float:
    n_nodes = graph.number_of_nodes()
    if n_nodes <= 2:
        return 0.0
    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)
    max_deg = degrees.max() if len(degrees) > 0 else 0.0
    numerator = float(np.maximum(max_deg - degrees, 0.0).sum())
    denominator = float((n_nodes - 1) * (n_nodes - 2))
    return numerator / denominator if denominator > 0 else 0.0


def structural_attributes(graph: nx.Graph) -> torch.Tensor:
    if graph.number_of_nodes() == 0:
        return torch.zeros(5, dtype=torch.float32)
    undirected = nx.Graph(graph)
    avg_degree = 0.0
    if undirected.number_of_nodes() > 0:
        avg_degree = float(np.mean([deg for _, deg in undirected.degree()]))
    clustering = float(nx.average_clustering(undirected)) if undirected.number_of_nodes() > 1 else 0.0
    diameter = _safe_diameter(undirected)
    centralization = _graph_centralization(undirected)
    modularity = 0.0
    try:
        if undirected.number_of_edges() > 0 and undirected.number_of_nodes() > 2:
            communities = list(nx.community.greedy_modularity_communities(undirected))
            if len(communities) > 1:
                modularity = float(nx.community.modularity(undirected, communities))
    except Exception:
        modularity = 0.0
    return torch.tensor(
        [clustering, np.log1p(diameter), avg_degree, modularity, centralization],
        dtype=torch.float32,
    )


class PrivateExplicitAdapter(nn.Module):
    def __init__(self, input_dim: int, core_dim: int) -> None:
        super().__init__()
        hidden_dim = max(32, input_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, core_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedExplicitProjector(nn.Module):
    def __init__(self, core_dim: int, unified_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(core_dim, unified_dim),
            nn.LayerNorm(unified_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrivateGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv = SAGEConv(node_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x, edge_index))


class SharedGraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, unified_dim: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, unified_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class SharedImplicitProjector(nn.Module):
    def __init__(self, unified_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(unified_dim * 3, unified_dim),
            nn.ReLU(),
            nn.LayerNorm(unified_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedPredictor(nn.Module):
    def __init__(self, unified_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(unified_dim * 2, unified_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(unified_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FedIPEClientModel(nn.Module):
    def __init__(
        self,
        explicit_input_dim: int,
        node_feature_dim: int,
        core_dim: int = 10,
        hidden_dim: int = 64,
        unified_dim: int = 128,
    ) -> None:
        super().__init__()
        self.core_dim = core_dim
        self.unified_dim = unified_dim
        self.explicit_adapter = PrivateExplicitAdapter(explicit_input_dim, core_dim)
        self.explicit_projector = SharedExplicitProjector(core_dim, unified_dim)
        self.private_graph_encoder = PrivateGraphEncoder(node_feature_dim, hidden_dim)
        self.shared_graph_encoder = SharedGraphEncoder(hidden_dim, unified_dim)
        self.implicit_projector = SharedImplicitProjector(unified_dim)
        self.subgraph_head = nn.Sequential(
            nn.Linear(unified_dim, unified_dim),
            nn.ReLU(),
            nn.LayerNorm(unified_dim),
        )
        self.predictor = SharedPredictor(unified_dim)

    def shared_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if key.startswith(SHARED_PARAMETER_PREFIXES)
        }

    def load_shared_state_dict(self, shared_state: Dict[str, torch.Tensor]) -> None:
        own_state = self.state_dict()
        for key, value in shared_state.items():
            if key in own_state:
                own_state[key].copy_(value.to(device=own_state[key].device, dtype=own_state[key].dtype))

    def _aggregate_edge_features(self, node_repr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = node_repr.size(0)
        if edge_index.numel() == 0:
            return torch.zeros_like(node_repr)
        src, dst = edge_index
        edge_repr = node_repr[src] * node_repr[dst]
        aggr = torch.zeros_like(node_repr)
        counts = torch.zeros(num_nodes, 1, device=node_repr.device, dtype=node_repr.dtype)
        aggr.index_add_(0, src, edge_repr)
        aggr.index_add_(0, dst, edge_repr)
        ones = torch.ones((src.numel(), 1), device=node_repr.device, dtype=node_repr.dtype)
        counts.index_add_(0, src, ones)
        counts.index_add_(0, dst, ones)
        return aggr / counts.clamp(min=1.0)

    def _ego_context(
        self,
        node_repr: torch.Tensor,
        platform_data: PlatformGraphData,
    ) -> torch.Tensor:
        contexts: List[torch.Tensor] = []
        for node_id in platform_data.target_node_ids:
            ego = nx.ego_graph(platform_data.nx_graph, node_id, radius=2)
            indices = [platform_data.node_id_to_idx[n] for n in ego.nodes if n in platform_data.node_id_to_idx]
            if not indices:
                contexts.append(torch.zeros(self.unified_dim, device=node_repr.device))
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=node_repr.device)
            contexts.append(node_repr.index_select(0, idx_tensor).mean(dim=0))
        return torch.stack(contexts, dim=0)

    def _sample_subgraph_bank(
        self,
        node_repr: torch.Tensor,
        platform_data: PlatformGraphData,
        sample_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_targets = platform_data.num_targets
        if num_targets == 0:
            empty = torch.zeros((0, self.unified_dim), device=node_repr.device)
            return empty, torch.zeros((0, 5), device=node_repr.device)
        sample_size = min(sample_size, num_targets)
        step = max(1, num_targets // sample_size)
        sampled = platform_data.target_node_ids[::step][:sample_size]
        embeddings: List[torch.Tensor] = []
        attrs: List[torch.Tensor] = []
        for node_id in sampled:
            ego = nx.ego_graph(platform_data.nx_graph, node_id, radius=2)
            indices = [platform_data.node_id_to_idx[n] for n in ego.nodes if n in platform_data.node_id_to_idx]
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=node_repr.device)
                emb = node_repr.index_select(0, idx_tensor).mean(dim=0)
            else:
                emb = torch.zeros(self.unified_dim, device=node_repr.device)
            embeddings.append(self.subgraph_head(emb))
            attrs.append(structural_attributes(ego).to(node_repr.device))
        return torch.stack(embeddings, dim=0), torch.stack(attrs, dim=0)

    def structural_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        attrs: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if embeddings.size(0) < 3:
            return embeddings.new_zeros(())
        attr_norm = (attrs - attrs.mean(dim=0, keepdim=True)) / (attrs.std(dim=0, keepdim=True) + 1e-6)
        dist = torch.cdist(attr_norm, attr_norm, p=2)
        non_diag = ~torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
        threshold = torch.quantile(dist[non_diag], 0.35)
        positive_mask = (dist <= threshold) & non_diag

        emb_norm = torch.nn.functional.normalize(embeddings, dim=1)
        sim = emb_norm @ emb_norm.t() / temperature
        exp_sim = torch.exp(sim) * non_diag

        losses: List[torch.Tensor] = []
        for row in range(sim.size(0)):
            pos_count = positive_mask[row].sum()
            if pos_count == 0:
                continue
            numerator = (exp_sim[row] * positive_mask[row]).sum()
            denominator = exp_sim[row].sum().clamp(min=1e-6)
            losses.append(-torch.log(numerator / denominator + 1e-6))
        if not losses:
            return embeddings.new_zeros(())
        return torch.stack(losses).mean()

    def forward(
        self,
        platform_data: PlatformGraphData,
        temperature: float = 0.1,
        subgraph_sample_size: int = 48,
    ) -> ClientForwardOutput:
        device = next(self.parameters()).device
        explicit_local = platform_data.explicit_local.to(device)
        core_targets = platform_data.core_targets.to(device)
        node_features = platform_data.node_features.to(device)
        edge_index = platform_data.edge_index.to(device)
        target_idx = platform_data.target_node_indices.to(device)

        core_logits = self.explicit_adapter(explicit_local)
        core_repr = torch.sigmoid(core_logits)
        explicit_embedding = self.explicit_projector(core_repr)
        anchor_loss = torch.nn.functional.mse_loss(core_repr, core_targets)

        private_node_repr = self.private_graph_encoder(node_features, edge_index)
        shared_node_repr = self.shared_graph_encoder(private_node_repr, edge_index)
        target_node_repr = shared_node_repr.index_select(0, target_idx)
        edge_context = self._aggregate_edge_features(shared_node_repr, edge_index).index_select(0, target_idx)
        subgraph_context = self._ego_context(shared_node_repr, platform_data)
        implicit_embedding = self.implicit_projector(
            torch.cat([target_node_repr, edge_context, subgraph_context], dim=1)
        )

        subgraph_embeddings, structural_attrs = self._sample_subgraph_bank(
            shared_node_repr,
            platform_data,
            sample_size=subgraph_sample_size,
        )
        contrastive_loss = self.structural_contrastive_loss(
            embeddings=subgraph_embeddings,
            attrs=structural_attrs,
            temperature=temperature,
        )

        predictions = self.predictor(torch.cat([explicit_embedding, implicit_embedding], dim=1)).squeeze(-1)
        return ClientForwardOutput(
            predictions=predictions,
            explicit_embedding=explicit_embedding,
            implicit_embedding=implicit_embedding,
            anchor_loss=anchor_loss,
            contrastive_loss=contrastive_loss,
            subgraph_embeddings=subgraph_embeddings,
            structural_attributes=structural_attrs,
        )


__all__ = [
    "ClientForwardOutput",
    "FedIPEClientModel",
    "SHARED_PARAMETER_PREFIXES",
    "structural_attributes",
]
