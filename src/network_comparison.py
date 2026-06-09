"""Compare climate networks constructed from alternative preprocessing choices."""

from __future__ import annotations

import networkx as nx
import numpy as np

from src.visualisation import compute_link_lengths_km


def edge_set(G: nx.Graph) -> set[frozenset[int]]:
    """Return an undirected edge set with order-independent edge keys."""

    return {frozenset((int(source), int(target))) for source, target in G.edges}


def edge_overlap(G_a: nx.Graph, G_b: nx.Graph) -> dict[str, float | int]:
    """Compute overlap and Jaccard similarity between two undirected graphs."""

    edges_a = edge_set(G_a)
    edges_b = edge_set(G_b)
    intersection = edges_a & edges_b
    union = edges_a | edges_b

    return {
        "edges_a": len(edges_a),
        "edges_b": len(edges_b),
        "shared_edges": len(intersection),
        "union_edges": len(union),
        "jaccard_similarity": len(intersection) / len(union) if union else np.nan,
    }


def degree_correlation(G_a: nx.Graph, G_b: nx.Graph) -> float:
    """Compute Pearson correlation between node-degree sequences."""

    nodes = sorted(set(G_a.nodes) | set(G_b.nodes))
    degree_a = np.asarray([G_a.degree(node) for node in nodes], dtype=float)
    degree_b = np.asarray([G_b.degree(node) for node in nodes], dtype=float)

    if degree_a.size < 2:
        return np.nan
    if np.isclose(degree_a.std(), 0.0) or np.isclose(degree_b.std(), 0.0):
        return np.nan
    return float(np.corrcoef(degree_a, degree_b)[0, 1])


def link_length_summary(G: nx.Graph) -> dict[str, float | int]:
    """Summarise the graph's great-circle link-length distribution."""

    lengths = compute_link_lengths_km(G)
    if lengths.size == 0:
        return {
            "n_links_with_lengths": 0,
            "mean_km": np.nan,
            "median_km": np.nan,
            "std_km": np.nan,
            "q10_km": np.nan,
            "q90_km": np.nan,
        }

    return {
        "n_links_with_lengths": int(lengths.size),
        "mean_km": float(np.mean(lengths)),
        "median_km": float(np.median(lengths)),
        "std_km": float(np.std(lengths)),
        "q10_km": float(np.quantile(lengths, 0.10)),
        "q90_km": float(np.quantile(lengths, 0.90)),
    }


def compare_networks(
    G_a: nx.Graph,
    G_b: nx.Graph,
    label_a: str = "network_a",
    label_b: str = "network_b",
) -> dict[str, object]:
    """Compare two networks using edge, degree, clustering, and distance metrics."""

    return {
        "label_a": label_a,
        "label_b": label_b,
        "edge_overlap": edge_overlap(G_a, G_b),
        "degree_correlation": degree_correlation(G_a, G_b),
        "global_clustering": {
            label_a: nx.average_clustering(G_a) if G_a.number_of_nodes() else np.nan,
            label_b: nx.average_clustering(G_b) if G_b.number_of_nodes() else np.nan,
        },
        "link_length_distribution": {
            label_a: link_length_summary(G_a),
            label_b: link_length_summary(G_b),
        },
    }

