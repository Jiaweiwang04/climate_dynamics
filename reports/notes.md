# Notes

This is a first working prototype for constructing climate networks from CRUTEM3 land temperature anomaly fields.

The current implementation follows the simplest reproducible route:

- monthly anomalies are used directly;
- grid points with too many missing values are dropped;
- remaining missing values are filled by temporal mean by default;
- Pearson correlation is the main dependence measure;
- an unweighted, undirected graph is formed by keeping the strongest fixed-density links;
- mutual information is included as an optional smaller-scale extension.

Initial defaults use 1950-2000, 25% maximum missing values, and 1% edge density. These are intentionally conservative for a first prototype and should be varied during analysis.

The mutual information estimator is binned and quadratic in the number of nodes, so it should be run on a subset of grid points unless the grid has already been reduced.

