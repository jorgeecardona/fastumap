"""BLAS thread cap for constrained CPU quotas (Fargate/ECS oversubscription)."""

import os

import numpy as np
import pytest
import threadpoolctl

import fastumap._threads as T
from fastumap import umap_project


def test_limit_is_none_or_positive_int() -> None:
    lim = T.blas_thread_limit()
    assert lim is None or (isinstance(lim, int) and lim >= 1)


def test_context_manager_is_safe() -> None:
    with T.limit_blas_to_quota():
        pass  # must not raise whether or not a cap applies on this host


def test_invisible_quota_does_not_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Fargate case (GitLab #10): the CPU quota is metered outside the container's cgroup, so
    # it's invisible, and the micro-VM reports only the small task core count. With no visible
    # quota and cpu_count == affinity, nothing is capped and BLAS is left alone.
    def two_cores(_pid: int) -> set[int]:
        return {0, 1}

    monkeypatch.setattr(T, "_cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr(T.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(T.os, "sched_getaffinity", two_cores)
    assert T.blas_thread_limit() is None


def test_fractional_quota_floors_to_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    if (os.cpu_count() or 1) <= 1:
        pytest.skip("needs a multi-core host to be 'constrained' below")
    monkeypatch.setattr(T, "_cgroup_cpu_quota", lambda: 0.5)  # simulate 0.5 vCPU
    assert T.blas_thread_limit() == 1


def test_cap_actually_limits_blas(monkeypatch: pytest.MonkeyPatch) -> None:
    if (os.cpu_count() or 1) <= 1:
        pytest.skip("needs a multi-core host")
    monkeypatch.setattr(T, "_cgroup_cpu_quota", lambda: 1.0)  # cap to one thread
    with T.limit_blas_to_quota():
        blas = [i for i in threadpoolctl.threadpool_info() if i.get("user_api") == "blas"]
        for info in blas:  # every BLAS backend present must report the cap
            assert info["num_threads"] == 1


def test_cap_does_not_change_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cap changes only thread count, never results. Force it on and confirm the layout is
    # bit-identical to the same seed without it.
    x = np.random.default_rng(0).standard_normal((200, 32))
    baseline = umap_project(x, 2, random_state=7)
    monkeypatch.setattr(T, "_cgroup_cpu_quota", lambda: 1.0)
    capped = umap_project(x, 2, random_state=7)
    assert np.array_equal(baseline, capped)
