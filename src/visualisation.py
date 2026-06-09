"""Plain matplotlib visualisations for the prototype climate networks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from src.data_loader import identify_temperature_variable
from src.preprocessing import find_time_dim


def _save_if_requested(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_global_mean_timeseries(
    ds_or_da: xr.Dataset | xr.DataArray,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the spatial mean anomaly through time."""

    da = ds_or_da[identify_temperature_variable(ds_or_da)] if isinstance(ds_or_da, xr.Dataset) else ds_or_da
    time_dim = find_time_dim(da)
    spatial_dims = [dim for dim in da.dims if dim != time_dim]
    series = da.mean(dim=spatial_dims, skipna=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series[time_dim].values, series.values, linewidth=1.2)
    ax.set_title("CRUTEM3 global land temperature anomaly")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly")
    ax.grid(True, alpha=0.3)
    _save_if_requested(fig, save_path)
    return fig, ax


def plot_missing_data_map(
    lat_lon: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot retained grid points coloured by missing-data fraction when available."""

    color = lat_lon["missing_fraction"] if "missing_fraction" in lat_lon else np.zeros(len(lat_lon))
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(lat_lon["lon"], lat_lon["lat"], c=color, s=12, cmap="viridis")
    ax.set_title("Retained grid points and missing-data fraction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.colorbar(scatter, ax=ax, label="Missing fraction")
    ax.grid(True, alpha=0.25)
    _save_if_requested(fig, save_path)
    return fig, ax


def plot_degree_map(
    metrics_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot node degree on a longitude-latitude scatter map."""

    return plot_metric_map(metrics_df, metric="degree", save_path=save_path)


def plot_metric_map(
    metrics_df: pd.DataFrame,
    metric: str,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot any node metric with latitude/longitude metadata."""

    required = {"lat", "lon", metric}
    missing = required.difference(metrics_df.columns)
    if missing:
        raise KeyError(f"metrics_df is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        metrics_df["lon"],
        metrics_df["lat"],
        c=metrics_df[metric],
        s=16,
        cmap="plasma",
    )
    ax.set_title(f"Node {metric}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.colorbar(scatter, ax=ax, label=metric)
    ax.grid(True, alpha=0.25)
    _save_if_requested(fig, save_path)
    return fig, ax


def plot_degree_histogram(
    metrics_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of node degrees."""

    if "degree" not in metrics_df:
        raise KeyError("metrics_df must contain a 'degree' column.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(metrics_df["degree"], bins=30, color="#4C78A8", edgecolor="white")
    ax.set_title("Degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Number of nodes")
    ax.grid(True, axis="y", alpha=0.25)
    _save_if_requested(fig, save_path)
    return fig, ax


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * radius_km * np.arcsin(np.sqrt(a)))


def plot_link_length_distribution(
    G: nx.Graph,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot great-circle lengths of graph edges using node lat/lon attributes."""

    lengths = []
    for source, target in G.edges:
        source_attrs = G.nodes[source]
        target_attrs = G.nodes[target]
        if {"lat", "lon"}.issubset(source_attrs) and {"lat", "lon"}.issubset(target_attrs):
            lengths.append(
                _haversine_km(
                    source_attrs["lat"],
                    source_attrs["lon"],
                    target_attrs["lat"],
                    target_attrs["lon"],
                )
            )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=40, color="#59A14F", edgecolor="white")
    ax.set_title("Link length distribution")
    ax.set_xlabel("Great-circle distance (km)")
    ax.set_ylabel("Number of links")
    ax.grid(True, axis="y", alpha=0.25)
    _save_if_requested(fig, save_path)
    return fig, ax
