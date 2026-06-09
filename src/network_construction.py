"""Construct NetworkX graphs from dependence matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx

from src.config import EDGE_DENSITY


def adjacency_from_fixed_density(
    score_matrix: np.ndarray,
    density: float = EDGE_DENSITY,
    use_absolute: bool = True,
) -> np.ndarray:
    """Build a symmetric binary adjacency matrix using the top off-diagonal links."""

    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("score_matrix must be square.")
    if not 0 <= density <= 1:
        raise ValueError("density must lie in [0, 1].")

    n_nodes = scores.shape[0]
    if n_nodes < 2 or density == 0:
        return np.zeros((n_nodes, n_nodes), dtype=np.uint8)

    working = np.abs(scores) if use_absolute else scores.copy()
    working = np.nan_to_num(working, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    np.fill_diagonal(working, -np.inf)

    row_idx, col_idx = np.triu_indices(n_nodes, k=1)
    edge_scores = working[row_idx, col_idx]
    n_possible = edge_scores.size
    n_edges = int(np.floor(density * n_possible))
    if density > 0 and n_edges == 0:
        n_edges = 1
    n_edges = min(n_edges, n_possible)

    if n_edges == n_possible:
        chosen = np.arange(n_possible)
    else:
        partition = np.argpartition(edge_scores, -n_edges)[-n_edges:]
        chosen = partition[np.argsort(edge_scores[partition])[::-1]]

    A = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    A[row_idx[chosen], col_idx[chosen]] = 1
    A[col_idx[chosen], row_idx[chosen]] = 1
    np.fill_diagonal(A, 0)
    return A


def build_graph_from_adjacency(
    A: np.ndarray,
    node_metadata: pd.DataFrame | None = None,
) -> nx.Graph:
    """Build an undirected NetworkX graph and attach node metadata."""

    adjacency = np.asarray(A)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("A must be a square adjacency matrix.")

    G = nx.from_numpy_array(adjacency)
    G.remove_edges_from(nx.selfloop_edges(G))

    if node_metadata is not None:
        if len(node_metadata) != adjacency.shape[0]:
            raise ValueError("node_metadata rows must match the size of A.")
        for _, row in node_metadata.iterrows():
            node_id = int(row["node_id"])
            attrs = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in row.to_dict().items()
                if key != "node_id"
            }
            G.nodes[node_id].update(attrs)

    return G

