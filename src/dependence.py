"""Dependence measures for climate network construction."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import mutual_info_score
from tqdm import tqdm


def compute_pearson_matrix(X: np.ndarray) -> np.ndarray:
    """Compute an N x N Pearson correlation matrix for X shaped (time, nodes)."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (time, nodes); got {X.shape}")
    if X.shape[1] < 2:
        raise ValueError("At least two nodes are required to compute correlations.")

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 0.0)
    return corr


def _bin_column(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=int)

    clean = values.copy()
    fill_value = np.nanmean(clean[finite])
    clean[~finite] = fill_value

    if np.allclose(clean, clean[0]):
        return np.zeros(clean.shape, dtype=int)

    edges = np.histogram_bin_edges(clean, bins=n_bins)
    return np.digitize(clean, edges[1:-1], right=False)


def compute_mutual_information_matrix(X: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Compute a simple binned mutual information matrix.

    This estimator is intended as a prototype extension. It loops over all
    node pairs and is therefore expensive for large grids.
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (time, nodes); got {X.shape}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    n_nodes = X.shape[1]
    if n_nodes > 500:
        warnings.warn(
            "Mutual information scales as O(nodes^2). Consider subsetting nodes first.",
            RuntimeWarning,
            stacklevel=2,
        )

    binned = np.column_stack([_bin_column(X[:, idx], n_bins) for idx in range(n_nodes)])
    mi = np.zeros((n_nodes, n_nodes), dtype=float)

    for i in tqdm(range(n_nodes), desc="Mutual information rows"):
        for j in range(i + 1, n_nodes):
            value = mutual_info_score(binned[:, i], binned[:, j])
            mi[i, j] = value
            mi[j, i] = value

    np.fill_diagonal(mi, 0.0)
    return mi

