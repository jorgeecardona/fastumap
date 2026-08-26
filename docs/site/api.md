# API reference

Generated from the source. Everything here is exported at the top level: `from fastumap import ...`.

## Projection

::: fastumap.projection
    options:
      members:
        - umap_project
        - spectral_project
        - fit
        - transform
        - UMAPModel

## Clustering

::: fastumap.clustering
    options:
      members:
        - cluster
        - kmeans
        - dbscan
        - hdbscan
        - choose_k
        - assign_clusters
        - ClusterResult

## Metrics

::: fastumap.metrics
    options:
      members:
        - neighbor_overlap
        - global_distance_correlation
        - random_layout

## Environment

::: fastumap.layout
    options:
      members:
        - accelerator_active

::: fastumap.neighbors
    options:
      members:
        - ann_available
