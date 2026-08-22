"""Thread safety: the library is called from asyncio.to_thread, so concurrent
projections in different threads must not interfere with each other.

Each projection holds its own local RNG (np.random.default_rng(random_state)) and no
module-global mutable state, so a projection run concurrently with others must produce
exactly the same result as when it runs alone. Different seeds per thread mean any shared
mutable state would corrupt the outputs and fail the equality check.
"""

from __future__ import annotations

import concurrent.futures

import numpy as np

from fastumap import spectral_project, umap_project
from fixtures import gaussian_clusters


def test_concurrent_umap_projections_match_single_threaded() -> None:
    cases = [(gaussian_clusters(n=250, seed=s)[0], s) for s in range(4)]
    references = [umap_project(x, 2, random_state=seed, n_epochs=100) for x, seed in cases]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(umap_project, x, 2, random_state=seed, n_epochs=100) for x, seed in cases
        ]
        results = [f.result() for f in futures]

    for reference, result in zip(references, results, strict=True):
        assert np.array_equal(reference, result)


def test_concurrent_spectral_projections_match_single_threaded() -> None:
    cases = [(gaussian_clusters(n=200, seed=s + 10)[0], s + 10) for s in range(4)]
    references = [spectral_project(x, 2, random_state=seed) for x, seed in cases]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(spectral_project, x, 2, random_state=seed) for x, seed in cases]
        results = [f.result() for f in futures]

    for reference, result in zip(references, results, strict=True):
        assert np.array_equal(reference, result)
