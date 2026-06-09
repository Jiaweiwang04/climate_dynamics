# Complex Network Analysis of CRUTEM3 Land Temperature Anomalies

First working prototype for a mathematics summer project in climate dynamics, inspired by Donges et al. (2009), "Complex networks in climate dynamics: comparing linear and nonlinear network construction methods".

The project builds climate networks from gridded CRUTEM3 land temperature anomaly time series. It is not a machine-learning prediction project.

## Current Status

The prototype now runs as a reproducible end-to-end workflow from the repository root:

- CRUTEM3 NetCDF data can be downloaded or loaded from `data/raw/CRUTEM3.nc`.
- The CRUTEM3 time coordinate `t` and temperature variable `temp` are handled automatically.
- The default analysis period is 1950-2000.
- Grid cells with too many missing values are removed, and remaining missing values are filled by temporal means.
- A Pearson correlation matrix, fixed-density adjacency matrix, NetworkX graph, graph metrics, and figures are generated.
- A detrending comparison is implemented for Pearson networks with and without node-wise linear detrending.
- A binned mutual information network is available as an optional subset experiment.

Latest verified default run:

- retained nodes: 723;
- Pearson edges at `EDGE_DENSITY = 0.01`: 2610;
- detrended Pearson edges: 2610;
- edge Jaccard similarity between non-detrended and detrended Pearson networks: approximately 0.957.

## Data

CRUTEM3 NetCDF best estimate temperature anomaly file:

<https://www.metoffice.gov.uk/hadobs/crutem3/data/CRUTEM3.nc>

Dataset page:

<https://www.metoffice.gov.uk/hadobs/crutem3/index.html>

Download page:

<https://www.metoffice.gov.uk/hadobs/crutem3/data/download.html>

The downloaded file is saved as `data/raw/CRUTEM3.nc`. Data products and generated figures are ignored by git.

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
├── scripts/
│   └── run_end_to_end.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── dependence.py
│   ├── network_construction.py
│   ├── network_comparison.py
│   ├── graph_metrics.py
│   └── visualisation.py
├── figures/
├── reports/
│   └── notes.md
├── requirements.txt
└── README.md
```

## Environment Setup

Recommended conda setup:

```bash
conda create -n climate-dynamics -c conda-forge python=3.11
conda activate climate-dynamics
conda install -c conda-forge numpy pandas xarray netcdf4 scipy scikit-learn networkx matplotlib tqdm pyarrow notebook ipykernel
python -m ipykernel install --user --name climate-dynamics --display-name "Python (climate-dynamics)"
```

Alternative pip setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install notebook ipykernel
python -m ipykernel install --user --name climate-dynamics --display-name "Python (climate-dynamics)"
```

Run all commands from the repository root so imports like `from src.config import ...` resolve correctly.

## Compile and Import Checks

```bash
python -m py_compile src/*.py scripts/run_end_to_end.py
python -c "import src.config, src.data_loader, src.preprocessing, src.dependence, src.network_construction, src.network_comparison, src.graph_metrics, src.visualisation; print('imports ok')"
```

## Download Data

```bash
python -c "from src.data_loader import download_crutem3; print(download_crutem3())"
```

If `data/raw/CRUTEM3.nc` already exists, the downloader leaves it in place.

## Run the Full Pipeline

```bash
python scripts/run_end_to_end.py
```

This script runs the default Pearson workflow and the detrending comparison. It writes intermediate arrays, processed graph products, metrics, and figures.

## Run the Notebooks

```bash
conda activate climate-dynamics
jupyter notebook
```

Then run notebooks in order:

1. `notebooks/01_load_and_explore_crutem3.ipynb`
2. `notebooks/02_construct_pearson_network.ipynb`
3. `notebooks/03_graph_metrics_analysis.ipynb`
4. `notebooks/04_mutual_information_extension.ipynb` optional

In VS Code, select the `Python (climate-dynamics)` kernel. If a notebook appears to use stale code after edits, restart the kernel and run all cells from the top.

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

## Main Outputs

Typical generated outputs:

- `data/interim/X_preprocessed.npy`
- `data/interim/node_metadata.parquet`
- `data/processed/pearson_correlation.npy`
- `data/processed/pearson_adjacency.npy`
- `data/processed/pearson_network.graphml`
- `data/processed/pearson_global_metrics.json`
- `data/processed/pearson_node_metrics.parquet`
- `data/processed/pearson_detrended_correlation.npy`
- `data/processed/pearson_detrended_adjacency.npy`
- `data/processed/pearson_detrended_network.graphml`
- `data/processed/detrending_comparison.json`
- figures in `figures/`

## Method Notes

Pearson correlation is the main prototype network construction method. Edges are selected at fixed density from the strongest off-diagonal absolute correlations, self-loops are removed, and the graph is treated as unweighted and undirected.

The mutual information implementation uses a simple binned estimator. It is included to mirror the linear/nonlinear comparison idea in the reference paper, but it is intentionally run on a smaller subset because pairwise MI over all grid points can be slow.
