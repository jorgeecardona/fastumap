# Projection

`umap_project` reduces `(n, d)` to any output dimension: 2-D and 3-D for visualisation, around
10-D as a preprocessing step before [clustering](clustering.md).

--8<-- "README.md:usage"

## Supervised projection

--8<-- "README.md:supervised"

## Next

- Placing *new* points into an existing layout without refitting is [In a server](server.md).
- Above ~20k points the neighbour search dominates — see [Large inputs](large-inputs.md).
- Full signatures are in the [API reference](api.md).
