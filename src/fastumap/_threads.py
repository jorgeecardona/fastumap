"""Cap BLAS threads to the container's CPU allotment, where that allotment is visible.

Two things can be true at once: a cgroup CFS quota is visible inside the container, *and* the
container still reports the host's full core count. When that happens, NumPy's BLAS sizes its
thread pool to those cores and oversubscribes the quota. Under concurrent load — several workers
each spawning that many threads against a fraction of a vCPU — the threads thrash. Measured under
``docker --cpus=0.5`` on a 16-core host with 4 worker processes: per-call time was ~70 s, versus
~42 s with threads pinned (~1.6x). Reading the quota and limiting the pool to it removes that
penalty. This fires where the quota is visible in-container: ``docker --cpus``, Kubernetes / EKS
CPU limits, EC2 with cgroup limits.

It does NOT fire on AWS Fargate (GitLab #10). Fargate meters CPU outside the container's cgroup
namespace (``cpu.cfs_quota_us`` is ``-1``, no ``cpu.max``), so the quota is invisible here. And
the task runs in a micro-VM. Its ``os.cpu_count()`` already reports the small task core count,
not a host's dozens, so BLAS does not oversubscribe there in the first place. Wherever no quota
is visible (a notebook, a full server, Fargate) nothing is capped and BLAS is left alone.

``threadpoolctl`` is imported lazily — only when a cap actually applies. So ``import fastumap``
and uncapped runs never touch it.
"""

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable, Generator
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _cgroup_cpu_quota() -> float | None:
    """Effective CPU count from the cgroup CFS quota, or None if unlimited/unreadable."""
    try:  # cgroup v2: "<quota> <period>" or "max <period>"
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            q, p = float(quota), float(period)
            if q > 0 and p > 0:
                return q / p
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        if q > 0 and p > 0:
            return q / p
    except (OSError, ValueError):
        pass
    return None


def blas_thread_limit() -> int | None:
    """BLAS thread count for a constrained CPU allotment, or None to leave BLAS alone.

    Returns a cap only when the process is limited below the host core count. The limit can come
    from a cgroup CFS quota (Fargate/ECS/Lambda) or a cpuset affinity smaller than the host. A
    fractional quota floors to one thread. On an unconstrained host it returns None.
    """
    host = os.cpu_count() or 1
    limit = host
    # os.sched_getaffinity is Linux-only (absent on macOS / Windows); getattr keeps this both
    # runtime-safe and type-clean on those platforms.
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        limit = min(limit, len(getaffinity(0)))  # cpuset restriction (Linux only)
    quota = _cgroup_cpu_quota()
    if quota is not None:
        limit = min(limit, max(1, int(quota)))
    return limit if limit < host else None


@contextlib.contextmanager
def limit_blas_to_quota() -> Generator[None, None, None]:
    """Cap BLAS threads to the CPU allotment within this block; no-op on an unconstrained host."""
    limit = blas_thread_limit()
    if limit is None:
        yield
        return
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=limit, user_api="blas"):
        yield


def with_blas_limit(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorator: run ``fn`` under :func:`limit_blas_to_quota` (preserves its signature)."""

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with limit_blas_to_quota():
            return fn(*args, **kwargs)

    return wrapper
