"""Download and inspect the CRUTEM3 NetCDF dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import xarray as xr
from tqdm import tqdm

from src.config import DATA_URL, RAW_DATA_PATH, TEMPERATURE_VARIABLE


def download_crutem3(
    url: str = DATA_URL,
    path: str | Path = RAW_DATA_PATH,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download the CRUTEM3 NetCDF file if it is not already present."""

    target = Path(path)
    if target.exists() and not overwrite:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".part")

    with urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with tmp_path.open("wb") as output:
            progress = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {target.name}",
            )
            try:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    output.write(chunk)
                    progress.update(len(chunk))
            finally:
                progress.close()

    tmp_path.replace(target)
    return target


def load_crutem3(
    path: str | Path = RAW_DATA_PATH,
    download_if_missing: bool = True,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Open the CRUTEM3 dataset with xarray."""

    data_path = Path(path)
    if not data_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"{data_path} does not exist. Run download_crutem3() first."
            )
        download_crutem3(path=data_path)

    return xr.open_dataset(data_path, chunks=chunks)


def identify_temperature_variable(
    ds: xr.Dataset,
    variable_name: str | None = TEMPERATURE_VARIABLE,
) -> str:
    """Infer the main temperature anomaly variable unless explicitly provided."""

    if variable_name is not None:
        if variable_name not in ds.data_vars:
            available = ", ".join(ds.data_vars)
            raise KeyError(
                f"Variable {variable_name!r} was not found. "
                f"Available variables: {available}"
            )
        return variable_name

    candidate_words = ("temperature", "temp", "anom", "anomaly", "tas", "tem")
    scored: list[tuple[int, str]] = []
    for name, da in ds.data_vars.items():
        lower_name = name.lower()
        attrs = " ".join(str(value).lower() for value in da.attrs.values())
        text = f"{lower_name} {attrs}"
        score = sum(word in text for word in candidate_words)

        dims = {dim.lower() for dim in da.dims}
        has_time = any(dim == "t" or "time" in dim for dim in dims)
        has_lat = any(dim in {"lat", "latitude"} or "lat" in dim for dim in dims)
        has_lon = any(dim in {"lon", "longitude"} or "lon" in dim for dim in dims)
        if has_time and has_lat and has_lon:
            score += 3
        if da.ndim >= 3:
            score += 1
        scored.append((score, name))

    if not scored:
        raise ValueError("No data variables were found in the dataset.")

    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    if best_score <= 0:
        available = ", ".join(ds.data_vars)
        raise ValueError(
            "Could not infer a temperature anomaly variable. "
            f"Set TEMPERATURE_VARIABLE manually. Available variables: {available}"
        )

    return best_name


def inspect_dataset(ds: xr.Dataset) -> str:
    """Print and return a compact textual summary of the dataset."""

    lines = [
        "Dimensions:",
        *[f"  {name}: {size}" for name, size in ds.sizes.items()],
        "",
        "Coordinates:",
        *[
            f"  {name}: dims={coord.dims}, shape={coord.shape}"
            for name, coord in ds.coords.items()
        ],
        "",
        "Data variables:",
    ]
    for name, da in ds.data_vars.items():
        units = da.attrs.get("units", "")
        long_name = da.attrs.get("long_name", da.attrs.get("standard_name", ""))
        description = ", ".join(part for part in [long_name, units] if part)
        if description:
            description = f" ({description})"
        lines.append(f"  {name}: dims={da.dims}, shape={da.shape}{description}")

    try:
        variable = identify_temperature_variable(ds)
        lines.extend(["", f"Inferred temperature anomaly variable: {variable}"])
    except Exception as exc:  # pragma: no cover - used for interactive inspection.
        lines.extend(["", f"Could not infer temperature variable: {exc}"])

    summary = "\n".join(lines)
    print(summary)
    return summary
