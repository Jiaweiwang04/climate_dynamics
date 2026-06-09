"""Run the default CRUTEM3 Pearson climate-network workflow from the repo root."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DETREND,
    EDGE_DENSITY,
    END_YEAR,
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    START_YEAR,
)
from src.data_loader import download_crutem3, inspect_dataset, load_crutem3
from src.dependence import compute_pearson_matrix
from src.graph_metrics import compute_basic_metrics, compute_node_metrics
from src.graph_metrics import save_node_metrics
from src.network_comparison import compare_networks
from src.network_construction import adjacency_from_fixed_density
from src.network_construction import build_graph_from_adjacency
from src.preprocessing import extract_temperature_array, fill_or_drop_missing_values
from src.preprocessing import filter_valid_gridpoints, optionally_detrend_timeseries
from src.preprocessing import select_time_period, stack_gridpoints
from src.visualisation import plot_degree_histogram, plot_degree_map
from src.visualisation import plot_link_length_distribution
from src.visualisation import plot_link_length_distributions
from src.visualisation import plot_missing_data_map


def _write_json(payload: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True))
    return path


def _save_and_close(fig: plt.Figure) -> None:
    plt.close(fig)


def preprocess_default_data() -> tuple[np.ndarray, pd.DataFrame]:
    """Load CRUTEM3 and produce the filled default node-time matrix."""

    download_crutem3()
    ds = load_crutem3(download_if_missing=False)
    inspect_dataset(ds)

    ds_period = select_time_period(ds, START_YEAR, END_YEAR)
    da = extract_temperature_array(ds_period)
    X_raw, node_metadata = stack_gridpoints(da)
    X_filtered, node_metadata = filter_valid_gridpoints(X_raw, node_metadata)
    X_filled = fill_or_drop_missing_values(X_filtered, strategy="mean")

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INTERIM_DATA_DIR / "X_preprocessed.npy", X_filled)
    node_metadata.to_parquet(INTERIM_DATA_DIR / "node_metadata.parquet", index=False)

    fig, _ = plot_missing_data_map(
        node_metadata,
        save_path=FIGURES_DIR / "retained_gridpoints_missing_fraction.png",
    )
    _save_and_close(fig)

    ds.close()
    return X_filled, node_metadata


def build_and_save_pearson_network(
    X: np.ndarray,
    node_metadata: pd.DataFrame,
    *,
    detrend: bool,
    prefix: str,
) -> tuple[nx.Graph, dict[str, object]]:
    """Construct and save a Pearson climate network."""

    X_ready = optionally_detrend_timeseries(X, detrend=detrend)
    corr = compute_pearson_matrix(X_ready)
    adjacency = adjacency_from_fixed_density(
        corr,
        density=EDGE_DENSITY,
        use_absolute=True,
    )
    graph = build_graph_from_adjacency(adjacency, node_metadata)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / f"{prefix}_correlation.npy", corr)
    np.save(PROCESSED_DATA_DIR / f"{prefix}_adjacency.npy", adjacency)
    nx.write_graphml(graph, PROCESSED_DATA_DIR / f"{prefix}_network.graphml")

    global_metrics = compute_basic_metrics(graph)
    node_metrics = compute_node_metrics(graph)
    _write_json(global_metrics, PROCESSED_DATA_DIR / f"{prefix}_global_metrics.json")
    save_node_metrics(
        node_metrics,
        PROCESSED_DATA_DIR / f"{prefix}_node_metrics.parquet",
    )

    fig, _ = plot_degree_map(
        node_metrics,
        save_path=FIGURES_DIR / f"{prefix}_degree_map.png",
    )
    _save_and_close(fig)
    fig, _ = plot_degree_histogram(
        node_metrics,
        save_path=FIGURES_DIR / f"{prefix}_degree_histogram.png",
    )
    _save_and_close(fig)
    fig, _ = plot_link_length_distribution(
        graph,
        save_path=FIGURES_DIR / f"{prefix}_link_length_distribution.png",
    )
    _save_and_close(fig)

    return graph, global_metrics


def main() -> None:
    """Run the full default workflow and the detrending comparison."""

    for directory in (INTERIM_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    X, node_metadata = preprocess_default_data()

    graph_raw, raw_metrics = build_and_save_pearson_network(
        X,
        node_metadata,
        detrend=DETREND,
        prefix="pearson",
    )
    graph_detrended, detrended_metrics = build_and_save_pearson_network(
        X,
        node_metadata,
        detrend=True,
        prefix="pearson_detrended",
    )

    comparison = compare_networks(
        graph_raw,
        graph_detrended,
        label_a="without_detrending",
        label_b="with_linear_detrending",
    )
    comparison["global_metrics"] = {
        "without_detrending": raw_metrics,
        "with_linear_detrending": detrended_metrics,
    }
    _write_json(comparison, PROCESSED_DATA_DIR / "detrending_comparison.json")

    fig, _ = plot_link_length_distributions(
        {
            "without detrending": graph_raw,
            "with linear detrending": graph_detrended,
        },
        save_path=FIGURES_DIR / "detrending_link_length_comparison.png",
    )
    _save_and_close(fig)

    print("End-to-end workflow complete.")
    print(f"Nodes: {graph_raw.number_of_nodes()}")
    print(f"Pearson edges: {graph_raw.number_of_edges()}")
    print(f"Detrended Pearson edges: {graph_detrended.number_of_edges()}")
    print(
        "Detrending Jaccard similarity: "
        f"{comparison['edge_overlap']['jaccard_similarity']:.4f}"
    )


if __name__ == "__main__":
    main()
