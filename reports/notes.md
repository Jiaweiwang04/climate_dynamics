# Research Notes

## Project Aim

This project constructs climate networks from gridded CRUTEM3 land temperature anomaly time series. The goal is mathematical and diagnostic: represent relationships between spatial climate anomaly records as a graph and study the resulting network structure.

This is not a prediction project. No LightGBM, deep learning, or machine-learning forecast model is used.

## CRUTEM3 Data

CRUTEM3 is a gridded land temperature anomaly dataset from the Met Office Hadley Centre. The prototype uses the NetCDF best estimate file `CRUTEM3.nc`.

In the downloaded file used here:

- the time coordinate is named `t`;
- the anomaly variable is named `temp`;
- the spatial grid uses `latitude` and `longitude`;
- the full file spans 1850-01 to 2014-05;
- the default prototype period is 1950-2000.

The raw data are monthly anomalies. The first prototype uses these monthly values directly. Later versions may remove the seasonal cycle more carefully if needed, but the supplied anomaly field is already the intended starting point.

## Climate Network Definition

A climate network treats spatial grid points as graph nodes. A link between two nodes is added when the corresponding time series are sufficiently dependent under a chosen dependence measure.

For this prototype:

- node: one retained CRUTEM3 latitude-longitude grid cell;
- node time series: monthly anomaly sequence over 1950-2000;
- edge score: Pearson correlation or optional mutual information;
- edge selection: fixed edge density thresholding;
- graph type: unweighted, undirected, no self-loops.

The result is a mathematical object that can be studied with graph-theoretic measures.

## Preprocessing Choices

The code stacks the gridded data into a matrix `X` with shape `(time, nodes)`.

The current missing-data handling is simple:

- drop nodes whose missing fraction exceeds `MAX_MISSING_FRACTION`;
- fill remaining missing entries by the node's temporal mean;
- optionally apply a linear detrend to each node time series.

This is intentionally conservative for a first working implementation. More careful treatments could include seasonal stratification, area weighting, interpolation checks, or sensitivity tests for missing-data thresholds.

## Pearson Correlation Network

The main network uses Pearson correlation between each pair of grid-point anomaly time series. The resulting matrix is symmetric with zero diagonal.

Pearson correlation captures linear co-variability. In climate-network terms, a strong positive or negative correlation suggests that two regions have anomaly series that tend to vary together or oppositely over the selected time window.

The prototype uses absolute correlation for edge ranking. This keeps both strong positive and strong negative relationships, which is useful for an initial network view but should be interpreted carefully because the sign is discarded in the binary adjacency matrix.

## Mutual Information Network

Mutual information is included as an optional nonlinear extension. It measures statistical dependence beyond strictly linear correlation.

The current estimator is deliberately simple:

- each node time series is discretised into bins;
- pairwise mutual information is computed from binned values;
- a fixed-density graph is built from the largest MI scores.

This estimator is computationally expensive because it scales quadratically with the number of nodes. The notebook therefore samples a smaller subset. A more serious nonlinear analysis would test sensitivity to the number of bins, sample size, missing-data handling, and temporal dependence.

## Fixed Edge Density Thresholding

The prototype uses fixed edge density rather than a fixed correlation threshold. This means the graph keeps a specified proportion of all possible off-diagonal links, for example the top 1% by absolute correlation.

This has two advantages:

- networks constructed under different preprocessing choices have the same number of edges;
- graph metrics are less confounded by trivial density differences.

The drawback is that a fixed density forces a graph even if the dependence distribution is weak. Results should therefore be compared across several densities in later work.

## Why Detrending Matters

Temperature anomaly series may share long-term trends. Pearson correlation can be strongly influenced by such trends, especially when many locations warm over the same period.

Detrending each node time series removes a best-fit linear trend before computing dependence. This asks a different question: which locations co-vary after removing the simplest long-term trend?

The current code compares Pearson networks with and without detrending using:

- edge overlap;
- Jaccard similarity;
- degree correlation;
- global clustering;
- link-length distribution summaries.

If detrending substantially lowers edge overlap, then the original network may be strongly shaped by shared trends. If degree correlation remains high, the same broad regions may remain important even though individual links change.

## Graph Metrics

Global metrics currently computed:

- number of nodes;
- number of edges;
- edge density;
- average degree;
- average clustering;
- largest connected component size;
- average shortest path length on the largest connected component.

Node-level metrics currently computed:

- degree;
- degree centrality;
- clustering coefficient;
- betweenness centrality.

Betweenness centrality can be expensive for large graphs. The implementation approximates it when the graph exceeds the configured threshold.

## Current Limitations

The prototype is intentionally rough. Important limitations include:

- no area weighting by latitude;
- no explicit removal of autocorrelation effects;
- no statistical significance testing or surrogate-data testing;
- no correction for spatial distance or shared large-scale modes;
- no seasonal or annual aggregation comparison;
- binary unweighted networks discard dependence magnitude after thresholding;
- absolute Pearson thresholding discards correlation sign;
- mutual information uses a basic binned estimator;
- only a single default period and edge density are used in the first run.

## Next Steps

Useful next extensions:

- run sensitivity tests over edge densities such as 0.005, 0.01, and 0.02;
- compare periods, for example 1900-1950 and 1950-2000;
- add latitude area weighting for spatial summaries;
- compare signed positive-only and negative-only Pearson networks;
- test detrended versus non-detrended networks more systematically;
- add surrogate testing for significance of correlations;
- compare monthly and annual anomaly networks;
- improve mutual information estimation and runtime;
- relate high-degree or high-betweenness regions to known climate variability patterns.

