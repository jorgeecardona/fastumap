# Clustering

`cluster()` does **not** compute a 2-D layout first. That layout is an iterative SGD and takes
minutes at a million points; clustering does not need it.

--8<-- "README.md:clustering"

## Next

- Wall-clock and inertia for the k-means kernel: [Benchmark reports](benchmarks.md).
- The compiled k-means path: [The native accelerator](accelerator.md).
- Full signatures: [API reference](api.md).
