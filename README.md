# Complex Network Analysis of CRUTEM3 Land Temperature Anomalies

First working prototype for a mathematics summer project in climate dynamics, inspired by Donges et al. (2009), "Complex networks in climate dynamics: comparing linear and nonlinear network construction methods".

The project builds climate networks from gridded CRUTEM3 land temperature anomaly time series. It is not a machine-learning prediction project.

## What Is Implemented

- Download/load the CRUTEM3 NetCDF best estimate anomaly file.
- Open and inspect the dataset with `xarray`.
- Select a manageable time period, defaulting to 1950-2000.
- Convert gridded anomaly fields into node time series with shape `(time, nodes)`.
- Drop grid points with too many missing values.
- Fill remaining missing values by temporal mean or linear interpolation.
- Optionally detrend each node time series.
- Compute a Pearson correlation dependence matrix.
- Construct an unweighted, undirected fixed-density network using the strongest absolute correlations.
- Compute basic global and node-level graph metrics.
- Save intermediate arrays, metadata tables, metrics, and figures.
- Provide an optional binned mutual information extension for smaller node subsets.

## Data

CRUTEM3 NetCDF best estimate temperature anomaly file:

<https://www.metoffice.gov.uk/hadobs/crutem3/data/CRUTEM3.nc>

Dataset page:

<https://www.metoffice.gov.uk/hadobs/crutem3/index.html>

Download page:

<https://www.metoffice.gov.uk/hadobs/crutem3/data/download.html>

The downloaded NetCDF file is saved to `data/raw/CRUTEM3.nc`. Data files and generated figures are ignored by git.

## Project Layout

```text
.
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_load_and_explore_crutem3.ipynb
│   ├── 02_construct_pearson_network.ipynb
│   ├── 03_graph_metrics_analysis.ipynb
│   └── 04_mutual_information_extension.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dependence.py
│   ├── network_construction.py
│   ├── graph_metrics.py
│   └── visualisation.py
├── figures/
├── reports/
│   └── notes.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run notebooks from the repository root so imports like `from src.config import ...` resolve correctly.

## Suggested Workflow

1. Open `notebooks/01_load_and_explore_crutem3.ipynb` to download/load CRUTEM3 and inspect the dataset.
2. Run `notebooks/02_construct_pearson_network.ipynb` to preprocess the anomaly field and build the Pearson climate network.
3. Run `notebooks/03_graph_metrics_analysis.ipynb` to compute graph measures and save plots.
4. Optionally run `notebooks/04_mutual_information_extension.ipynb` on a reduced node subset.

## Configuration

Default parameters live in `src/config.py`:

- `DATA_URL`
- `RAW_DATA_PATH`
- `START_YEAR`
- `END_YEAR`
- `MAX_MISSING_FRACTION`
- `EDGE_DENSITY`
- `DETREND`
- `RANDOM_SEED`

Set `TEMPERATURE_VARIABLE` manually in `src/config.py` if automatic variable detection fails for a different NetCDF file.

## Outputs

Typical generated outputs:

- `data/interim/X_preprocessed.npy`
- `data/interim/node_metadata.parquet`
- `data/processed/pearson_correlation.npy`
- `data/processed/pearson_adjacency.npy`
- `data/processed/pearson_network.graphml`
- `data/processed/pearson_node_metrics.parquet`
- `data/processed/pearson_global_metrics.json`
- figures in `figures/`

## Method Notes

For the first prototype, Pearson correlation is the main network construction method. Edges are selected by fixed density from the strongest off-diagonal absolute correlations, self-loops are removed, and the graph is treated as unweighted and undirected.

The mutual information implementation uses a simple binned estimator. It is included to mirror the linear/nonlinear comparison in the reference paper, but it is intentionally kept as an optional extension because pairwise MI over all grid points can be slow.
