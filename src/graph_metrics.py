"""Graph-theoretic metrics for climate networks."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


def compute_basic_metrics(G: nx.Graph) -> dict[str, float | int]:
    """Compute global graph metrics for an undirected network."""

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = dict(G.degree())

    if n_nodes == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "edge_density": 0.0,
            "average_degree": 0.0,
            "average_clustering": np.nan,
            "largest_connected_component_size": 0,
            "average_shortest_path_length_lcc": np.nan,
        }

    components = list(nx.connected_components(G))
    largest_component = max(components, key=len) if components else set()
    lcc = G.subgraph(largest_component)
    if lcc.number_of_nodes() > 1:
        path_length = nx.average_shortest_path_length(lcc)
    else:
        path_length = 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "edge_density": nx.density(G),
        "average_degree": float(np.mean(list(degrees.values()))) if degrees else 0.0,
        "average_clustering": nx.average_clustering(G),
        "largest_connected_component_size": lcc.number_of_nodes(),
        "average_shortest_path_length_lcc": path_length,
    }


def compute_node_metrics(
    G: nx.Graph,
    approximate_betweenness_threshold: int = 1000,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute node-level graph metrics.

    Betweenness centrality is approximated when the graph is larger than the
    configured threshold.
    """

    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return pd.DataFrame(
            columns=[
                "node_id",
                "degree",
                "degree_centrality",
                "clustering",
                "betweenness_centrality",
            ]
        )

    degree = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G)

    if n_nodes > approximate_betweenness_threshold:
        k = min(approximate_betweenness_threshold, n_nodes)
        betweenness = nx.betweenness_centrality(G, k=k, seed=random_seed)
    else:
        betweenness = nx.betweenness_centrality(G)

    rows = []
    for node in G.nodes:
        attrs = dict(G.nodes[node])
        rows.append(
            {
                "node_id": node,
                "degree": degree[node],
                "degree_centrality": degree_centrality[node],
                "clustering": clustering[node],
                "betweenness_centrality": betweenness[node],
                **attrs,
            }
        )
    return pd.DataFrame(rows).sort_values("node_id").reset_index(drop=True)


def save_node_metrics(metrics_df: pd.DataFrame, path: str | Path) -> Path:
    """Save node metrics as CSV or Parquet based on the file extension."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        metrics_df.to_parquet(output_path, index=False)
    else:
        metrics_df.to_csv(output_path, index=False)
    return output_path

