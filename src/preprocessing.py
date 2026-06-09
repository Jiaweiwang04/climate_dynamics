"""Preprocessing utilities for gridded climate anomaly fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

from src.config import DETREND, END_YEAR, MAX_MISSING_FRACTION, START_YEAR
from src.data_loader import identify_temperature_variable


def _find_dim_name(
    names: list[str] | tuple[str, ...],
    candidates: tuple[str, ...],
) -> str:
    lower_map = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    for name in names:
        lower = name.lower()
        if any(candidate in lower for candidate in candidates):
            return name
    raise ValueError(f"Could not find any of these dimensions: {candidates}")


def find_time_dim(ds_or_da: xr.Dataset | xr.DataArray) -> str:
    """Infer the time dimension name, including CRUTEM3's short name ``t``."""

    names = tuple(ds_or_da.coords) + tuple(ds_or_da.dims)
    try:
        return _find_dim_name(names, ("time", "t"))
    except ValueError:
        pass

    for name in names:
        coord = ds_or_da.coords.get(name)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            return name

    raise ValueError("Could not infer a time dimension.")


def select_time_period(
    ds: xr.Dataset,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> xr.Dataset:
    """Select a calendar-year range from the dataset."""

    time_name = find_time_dim(ds)
    return ds.sel({time_name: slice(f"{start_year}-01-01", f"{end_year}-12-31")})


def extract_temperature_array(
    ds: xr.Dataset,
    variable_name: str | None = None,
) -> xr.DataArray:
    """Extract the temperature anomaly DataArray from the dataset."""

    selected_name = identify_temperature_variable(ds, variable_name)
    da = ds[selected_name]

    squeeze_dims = [dim for dim, size in da.sizes.items() if size == 1]
    if squeeze_dims:
        da = da.squeeze(dim=squeeze_dims, drop=True)

    return da


def stack_gridpoints(da: xr.DataArray) -> tuple[np.ndarray, pd.DataFrame]:
    """Stack a time-lat-lon anomaly field into an array with shape (time, nodes)."""

    time_dim = find_time_dim(da)
    lat_dim = _find_dim_name(da.dims, ("lat", "latitude"))
    lon_dim = _find_dim_name(da.dims, ("lon", "longitude"))

    extra_dims = [dim for dim in da.dims if dim not in {time_dim, lat_dim, lon_dim}]
    if extra_dims:
        raise ValueError(
            "Expected a 3D time-lat-lon field after squeezing. "
            f"Unexpected dimensions: {extra_dims}"
        )

    stacked = da.transpose(time_dim, lat_dim, lon_dim).stack(node=(lat_dim, lon_dim))
    X = stacked.values.astype(float)

    lat_lon = pd.DataFrame(
        {
            "node_id": np.arange(stacked.sizes["node"], dtype=int),
            "lat": stacked[lat_dim].values.astype(float),
            "lon": stacked[lon_dim].values.astype(float),
        }
    )
    return X, lat_lon


def filter_valid_gridpoints(
    X: np.ndarray,
    lat_lon: pd.DataFrame,
    max_missing_fraction: float = MAX_MISSING_FRACTION,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Drop grid points with more missing values than the configured threshold."""

    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (time, nodes); got {X.shape}")
    if len(lat_lon) != X.shape[1]:
        raise ValueError("lat_lon rows must match the number of columns in X.")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must lie in [0, 1].")

    missing_fraction = np.mean(~np.isfinite(X), axis=0)
    valid = missing_fraction <= max_missing_fraction
    valid &= missing_fraction < 1.0

    filtered_metadata = lat_lon.loc[valid].copy()
    filtered_metadata["original_node_id"] = filtered_metadata["node_id"].to_numpy()
    filtered_metadata["missing_fraction"] = missing_fraction[valid]
    filtered_metadata["node_id"] = np.arange(valid.sum(), dtype=int)

    return X[:, valid], filtered_metadata.reset_index(drop=True)


def fill_or_drop_missing_values(X: np.ndarray, strategy: str = "mean") -> np.ndarray:
    """Fill remaining missing values by column mean or linear interpolation."""

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (time, nodes); got {X.shape}")
    if not np.isnan(X).any() and np.isfinite(X).all():
        return X.copy()

    if strategy == "mean":
        filled = X.copy()
        col_means = np.nanmean(np.where(np.isfinite(filled), filled, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        missing_rows, missing_cols = np.where(~np.isfinite(filled))
        filled[missing_rows, missing_cols] = col_means[missing_cols]
        return filled

    if strategy == "linear":
        frame = pd.DataFrame(X)
        frame = frame.interpolate(method="linear", axis=0, limit_direction="both")
        frame = frame.fillna(frame.mean()).fillna(0.0)
        return frame.to_numpy(dtype=float)

    if strategy == "drop":
        return X[np.isfinite(X).all(axis=1), :]

    raise ValueError("strategy must be one of: 'mean', 'linear', 'drop'.")


def optionally_detrend_timeseries(
    X: np.ndarray,
    detrend: bool = DETREND,
) -> np.ndarray:
    """Optionally remove a linear trend from each node time series."""

    X = np.asarray(X, dtype=float)
    if not detrend:
        return X.copy()
    return signal.detrend(X, axis=0, type="linear")
