"""UMAPModel persistence contract (GitLab #4): a fitted model pickles, so a server can
fit once and every worker (or a worker after restart) reuses it instead of refitting.
"""

from __future__ import annotations

import pickle

import numpy as np

from fastumap import fit, transform
from fixtures import gaussian_clusters


def test_model_train_is_float32() -> None:
    # train is cached as float32 to halve the model footprint (Lambda/ECS); transform
    # upcasts to float64 for the distance compute, so quality is unaffected.
    x, _ = gaussian_clusters(n=200, dim=32, seed=0)
    model = fit(x, 2, random_state=0, n_epochs=50)
    assert model.train.dtype == np.float32


def test_umapmodel_pickle_round_trip() -> None:
    x, _ = gaussian_clusters(n=300, dim=32, seed=0)
    model = fit(x, 2, random_state=0, n_epochs=100)
    restored = pickle.loads(pickle.dumps(model))

    # a restored model transforms identically to the original — the whole point of caching
    new, _ = gaussian_clusters(n=40, dim=32, seed=1)
    assert np.array_equal(transform(model, new), transform(restored, new))

    # the fitted state survives the round trip
    assert np.array_equal(model.embedding, restored.embedding)
    assert np.array_equal(model.train, restored.train)
    assert (model.a, model.b, model.n_neighbors, model.metric) == (
        restored.a,
        restored.b,
        restored.n_neighbors,
        restored.metric,
    )
