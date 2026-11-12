---
title: Mini-batch k-means fails on well-separated clusters (4.8x worse inertia)
status: done
fixed_in: 0.3.1
area: capability
owner: null
filed: 2026-08-26
scope: M
verify: test
---

Split out of the mini-batch k-means issue (closed in 3d8842c), where this was measured but
deliberately not fixed. It is recorded there and pinned by
tests/test_minibatch_kmeans.py:test_minibatch_struggles_on_well_separated_blobs.

MEASURED: on 20000 x 32 with 16 Gaussian blobs whose centres are 8 sigma apart, mini-batch
(`kmeans(x, 16, batch_size=2000)`) returns inertia 3.06e6 against full Lloyd's 6.41e5 — 4.8x worse,
from the SAME k-means++ seeding, which Lloyd turns into a perfect recovery.

The failure is structural, not noise. Cluster sizes come back as [588, 653, 1188, ..., 2501]
against the true [1188, ..., 1309]: two centroids settle inside one blob (588+653) while a
neighbouring blob absorbs both its own points and theirs (2501). The running-mean update
(`c += (1/v[c]) * (x - c)`, src/fastumap/clustering.py routes to minibatch_kmeans_impl in
rust/src/lib.rs) has no force that can walk a centroid back across an empty gap, where Lloyd's
recompute-from-scratch does it in one iteration.

TRIED and did NOT help, so do not repeat them:
- initialising the per-centroid count to 1 instead of 0, to stop the first point assigned to a
  centroid teleporting it exactly onto that point (the seeded position is thrown away otherwise).
  Objective did not move.
- raising iterations from 100 to 400, 1000, 4000 at batch 2000, and varying batch 200/1000/5000
  at 1000 iterations. Inertia was 3.0604e6 in every combination — it converges to the same bad
  optimum rather than converging slowly.

INFERRED fixes worth trying, in order of expected value:
1. `n_init`: run the mini-batch phase 3 times from different seeds on a subsample, keep the lowest
   inertia. What scikit-learn's MiniBatchKMeans does.
2. Reassign centroids whose count stays far below the mean to the point furthest from its centroid,
   periodically during the run (sklearn's `reassignment_ratio`).
3. A short Lloyd polish is NOT a fix — Lloyd preserves this local optimum once reached; it only
   helps if the centroid placement is already basically right.

NOT urgent: real, overlapping data does not trigger it. MEASURED at 1.013x Lloyd on MNIST
60000 x 784 (k=64) and 1.010x on SIFT1M 1M x 128 (k=64), both well inside the 5% bar. That is why
the auto-switch above 200k rows is on and why this is a follow-up rather than a blocker.

Done when: on the well-separated fixture above, mini-batch inertia is within ~5% of full Lloyd,
the MNIST and SIFT1M ratios have not regressed, and
test_minibatch_struggles_on_well_separated_blobs is replaced by a parity assertion.

## Update 2026-08-27 (v0.2.29) — remedy landed as `n_init`; default unchanged, so NOT fully done

Fix #1 from above (n_init) is implemented on `kmeans` and `cluster(method="kmeans")`: restart from
`n_init` seeds, keep the lowest inertia. MEASURED on the fixture: n_init sweep 1→4.78x, 3→3.08x,
5→1.00x. So **n_init=5 recovers full-Lloyd parity** — a different seed escapes the bad optimum.

Fix #2 (reassignment) was reconsidered and NOT used: the failure has no empty centroids to reassign
(the two-in-one-blob centroids are moderately populated, 588/653), so a low-count reassignment rule
would not trigger here.

The test is updated: `test_minibatch_struggles_on_well_separated_blobs_but_n_init_recovers` asserts
both the default single-run gap AND that n_init=5 reaches parity.

STILL OPEN, because the "Done when" above asks for the DEFAULT to be within ~5%, and it is not: the
default stays a single run on purpose. Making n_init=5 the default would fix the adversarial case
but multiply mini-batch runtime by 5x on EVERY call, including the common case (real embeddings are
already ~1.01x, so they would pay 5x for no gain, and mini-batch exists to be fast on constrained
CPU). The decision — keep the fast opt-in default, or change the default and revise "Done when" — is
Jorge's. MNIST/SIFT1M ratios are unchanged (default path untouched).
## Done 2026-11-12

Fixed by greedy k-means++ seeding in the Rust kernel (#30). On the same 20000x32, 16-blob data the
default single mini-batch run now reaches full-Lloyd inertia. Pinned by
`test_minibatch_matches_lloyd_on_well_separated_blobs`.
