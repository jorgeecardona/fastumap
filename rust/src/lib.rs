//! Native kernels for fastumap: the SGD layout walk and k-means.
//!
//! The SGD is the *in-place* (Hogwild) attract/repel walk, the same one umap-learn's numba kernel
//! does: later edges in an epoch see earlier moves. That is faster and more faithful to umap-learn
//! than the pure-numpy snapshot update.
//!
//! The heavy loop runs with the GIL released (`py.allow_threads`), so concurrent
//! `asyncio.to_thread` calls run in parallel. Nothing here touches Python objects, and the borrow
//! checker guarantees no data races.
//!
//! Determinism: a seeded SplitMix64 stream drives negative sampling, so a given seed is
//! reproducible. Not bit-identical to the numpy path — different RNG and update order (the same
//! reason fastumap and umap-learn differ).

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use rayon::prelude::*;

const GRAD_CLIP: f64 = 4.0;

#[inline]
fn clamp(x: f64) -> f64 {
    x.clamp(-GRAD_CLIP, GRAD_CLIP)
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Optimise `embedding` in place-style (a copy is returned) with the attract/repel SGD.
///
/// Mirrors `optimize_layout` in the pure-numpy path, but does the true in-place walk.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn optimize_layout<'py>(
    py: Python<'py>,
    embedding: PyReadonlyArray2<'py, f64>,
    head: PyReadonlyArray1<'py, i64>,
    tail: PyReadonlyArray1<'py, i64>,
    epochs_per_sample: PyReadonlyArray1<'py, f64>,
    n_epochs: usize,
    a: f64,
    b: f64,
    gamma: f64,
    negative_sample_rate: usize,
    initial_alpha: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = embedding.shape();
    let n = shape[0];
    let dim = shape[1];

    // Copy the coordinates into an owned buffer so the SGD can run without the GIL and
    // without borrowing a numpy array mutably.
    let mut emb: Vec<f64> = embedding.as_slice()?.to_vec();
    let head = head.as_slice()?;
    let tail = tail.as_slice()?;
    let eps = epochs_per_sample.as_slice()?;
    let n_edges = eps.len();

    let neg_rate = negative_sample_rate.max(1) as f64;
    let eps_neg: Vec<f64> = eps.iter().map(|&e| e / neg_rate).collect();
    let mut epoch_of_next: Vec<f64> = eps.to_vec();
    let mut epoch_of_next_neg: Vec<f64> = eps_neg.clone();

    py.allow_threads(|| {
        let mut rng = seed ^ 0x123456789ABCDEF;
        for epoch in 0..n_epochs {
            let alpha = initial_alpha * (1.0 - epoch as f64 / n_epochs as f64);
            let epoch_f = epoch as f64;
            for i in 0..n_edges {
                if eps[i] <= 0.0 || epoch_of_next[i] > epoch_f {
                    continue;
                }
                let j = head[i] as usize;
                let k = tail[i] as usize;
                let (jb, kb) = (j * dim, k * dim);

                // Attraction: both ends move.
                let mut d2 = 0.0;
                for d in 0..dim {
                    let diff = emb[jb + d] - emb[kb + d];
                    d2 += diff * diff;
                }
                let coeff = if d2 > 0.0 {
                    -2.0 * a * b * d2.powf(b - 1.0) / (1.0 + a * d2.powf(b))
                } else {
                    0.0
                };
                for d in 0..dim {
                    let grad = clamp(coeff * (emb[jb + d] - emb[kb + d])) * alpha;
                    emb[jb + d] += grad;
                    emb[kb + d] -= grad;
                }
                epoch_of_next[i] += eps[i];

                // Repulsion: per-edge negative count from the second schedule; only head moves.
                let n_neg = ((epoch_f - epoch_of_next_neg[i]) / eps_neg[i]).max(0.0) as usize;
                for _ in 0..n_neg {
                    let kk = (splitmix64(&mut rng) % n as u64) as usize;
                    if kk == j {
                        continue;
                    }
                    let kkb = kk * dim;
                    let mut d2n = 0.0;
                    for d in 0..dim {
                        let diff = emb[jb + d] - emb[kkb + d];
                        d2n += diff * diff;
                    }
                    let coeff_n = if d2n > 0.0 {
                        2.0 * gamma * b / ((0.001 + d2n) * (1.0 + a * d2n.powf(b)))
                    } else {
                        0.0
                    };
                    for d in 0..dim {
                        let grad = if d2n > 0.0 {
                            clamp(coeff_n * (emb[jb + d] - emb[kkb + d]))
                        } else {
                            GRAD_CLIP
                        } * alpha;
                        emb[jb + d] += grad;
                    }
                }
                epoch_of_next_neg[i] += n_neg as f64 * eps_neg[i];
            }
        }
    });

    let arr = numpy::ndarray::Array2::from_shape_vec((n, dim), emb)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

/// k-means (k-means++ seed + Lloyd), the compiled counterpart of `clustering.kmeans`.
///
/// The assignment step is the O(n*k*d) wall. We compute it as one GEMM: `G = X @ centroidsᵀ`
/// (matrixmultiply's SIMD kernel). The nearest centroid is then argmin over `‖c‖² − 2·G[i,·]` —
/// the `‖x‖²` term is constant per row, so it drops out. That argmin runs over rows in parallel
/// with rayon.
///
/// All in f32 for SIMD width — clustering tolerates it. Deterministic given `seed` (not identical
/// to the numpy path: different init RNG and f32). Returns the `(n,)` integer labels.
#[pyfunction]
fn kmeans<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    n_clusters: usize,
    n_iter: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let shape = data.shape();
    let n = shape[0];
    let dim = shape[1];
    let x: Vec<f32> = data.as_slice()?.iter().map(|&v| v as f32).collect(); // row-major (n, dim)
    let labels = py.allow_threads(|| kmeans_impl(&x, n, dim, n_clusters, n_iter, seed));
    Ok(numpy::ndarray::Array1::from_vec(labels).into_pyarray_bound(py))
}

/// k-means++ seeding, shared by full Lloyd and mini-batch.
///
/// Full k-means++ is O(n*k*d) — the fixed cost that dominates at 1M. Seed it from a random
/// subsample instead (what faiss/sklearn do at scale); the sample captures the distribution, then
/// the iterations refine against the real data.
fn kmeanspp_seed(x: &[f32], n: usize, dim: usize, k: usize, rng: &mut u64) -> Vec<f32> {
    let dist2 = |a: &[f32], b: &[f32]| -> f32 {
        let mut s = 0.0f32;
        for d in 0..dim {
            let diff = a[d] - b[d];
            s += diff * diff;
        }
        s
    };
    let mut centroids = vec![0.0f32; k * dim];
    // Full k-means++ is O(n*k*d) — the fixed cost that dominates at 1M. Seed it from a random
    // subsample instead (what faiss/sklearn do at scale); the sample captures the distribution,
    // then Lloyd below refines against all n points.
    let sn = n.min(100_000);
    let sx: Vec<f32> = if sn < n {
        let mut xs = vec![0.0f32; sn * dim];
        for s in 0..sn {
            let i = (splitmix64(rng) % n as u64) as usize;
            xs[s * dim..s * dim + dim].copy_from_slice(&x[i * dim..i * dim + dim]);
        }
        xs
    } else {
        x.to_vec()
    };
    let first = (splitmix64(rng) % sn as u64) as usize;
    centroids[0..dim].copy_from_slice(&sx[first * dim..first * dim + dim]);
    let mut closest: Vec<f32> = (0..sn)
        .into_par_iter()
        .map(|i| dist2(&sx[i * dim..i * dim + dim], &centroids[0..dim]))
        .collect();
    // Greedy k-means++ (Arthur & Vassilvitskii 2007, scikit-learn's default): draw `trials`
    // candidates per step and keep the one that lowers the total squared distance most. Plain
    // k-means++ (one draw) often puts two seeds in one cluster on high-dimensional data, where
    // between-cluster and within-cluster distances are close; that start is a local optimum that
    // makes choose_k's inertia curve rise with k (#30).
    let trials = 2 + (k as f64).ln() as usize;
    for c in 1..k {
        // Sequential f64 sum, not a parallel f32 reduce: this total feeds a threshold walk that
        // picks a discrete seed index, so an order-dependent low-order-bit change (which a parallel
        // f32 reduce produces when the core count changes the reduction tree) would select a
        // different seed on a different machine and diverge the entire run. A fixed-order f64 sum
        // is thread-count-independent, and the walk below accumulates in f64 to match it.
        let total: f64 = closest.iter().map(|&d| d as f64).sum();
        let mut best: Option<(f64, usize, Vec<f32>)> = None;
        for _ in 0..trials {
            let target = (splitmix64(rng) as f64 / u64::MAX as f64) * total;
            let mut chosen = sn - 1;
            if total > 0.0 {
                let mut acc = 0.0f64;
                for (i, &d2) in closest.iter().enumerate() {
                    acc += d2 as f64;
                    if acc >= target {
                        chosen = i;
                        break;
                    }
                }
            } else {
                chosen = (splitmix64(rng) % sn as u64) as usize;
            }
            let cand = &sx[chosen * dim..chosen * dim + dim];
            let new_closest: Vec<f32> = (0..sn)
                .into_par_iter()
                .map(|i| dist2(&sx[i * dim..i * dim + dim], cand).min(closest[i]))
                .collect();
            // Fixed-order f64 sum again, so the winner is the same at any thread count.
            let potential: f64 = new_closest.iter().map(|&d| d as f64).sum();
            let better = match &best {
                None => true,
                Some((p, idx, _)) => potential < *p || (potential == *p && chosen < *idx),
            };
            if better {
                best = Some((potential, chosen, new_closest));
            }
        }
        let (_, chosen, new_closest) = best.expect("trials >= 2");
        centroids[c * dim..c * dim + dim].copy_from_slice(&sx[chosen * dim..chosen * dim + dim]);
        closest = new_closest;
    }
    centroids
}

/// Nearest centroid for every row of `x` (an `(rows, dim)` block), as one GEMM plus an argmin.
///
/// `G = X @ centroidsᵀ` (matrixmultiply's SIMD kernel); the nearest centroid is the argmin over
/// `‖c‖² − 2·G[i,·]`, since the `‖x‖²` term is constant per row and drops out. `g` is caller-owned
/// scratch of at least `rows * k`, so a loop can reuse one allocation.
fn assign_gemm(
    x: &[f32],
    rows: usize,
    dim: usize,
    centroids: &[f32],
    k: usize,
    g: &mut [f32],
) -> Vec<i64> {
    let mut c_sq = vec![0.0f32; k];
    for j in 0..k {
        let mut s = 0.0f32;
        for d in 0..dim {
            let v = centroids[j * dim + d];
            s += v * v;
        }
        c_sq[j] = s;
    }
    // centroids is (k×dim) row-major; viewing it as B with row-stride 1, col-stride dim gives
    // Cᵀ (dim×k) without a copy. matrixmultiply is single-threaded, so split X into per-thread
    // row-blocks and GEMM each independently.
    let n_threads = rayon::current_num_threads().max(1);
    let block = rows.div_ceil(n_threads).max(1);
    g[..rows * k]
        .par_chunks_mut(block * k)
        .enumerate()
        .for_each(|(bi, gchunk)| {
            let start = bi * block;
            let chunk_rows = gchunk.len() / k;
            if chunk_rows == 0 {
                return;
            }
            unsafe {
                matrixmultiply::sgemm(
                    chunk_rows,
                    dim,
                    k,
                    1.0,
                    x.as_ptr().add(start * dim),
                    dim as isize,
                    1,
                    centroids.as_ptr(),
                    1,
                    dim as isize,
                    0.0,
                    gchunk.as_mut_ptr(),
                    k as isize,
                    1,
                );
            }
        });
    (0..rows)
        .into_par_iter()
        .map(|i| {
            let row = &g[i * k..i * k + k];
            let mut best = 0usize;
            let mut best_v = f32::INFINITY;
            for j in 0..k {
                let val = c_sq[j] - 2.0 * row[j]; // = ‖x-c_j‖² minus the per-row constant ‖x‖²
                if val < best_v {
                    best_v = val;
                    best = j;
                }
            }
            best as i64
        })
        .collect()
}

fn kmeans_impl(x: &[f32], n: usize, dim: usize, k: usize, n_iter: usize, seed: u64) -> Vec<i64> {
    let mut rng = seed ^ 0x9E3779B97F4A7C15;
    let mut centroids = kmeanspp_seed(x, n, dim, k, &mut rng);

    // ─── Lloyd iterations (GEMM assignment) ───
    let mut labels = vec![0i64; n];
    let mut g = vec![0.0f32; n * k]; // G = X @ centroidsᵀ, reused each iteration
    for _ in 0..n_iter {
        let next = assign_gemm(x, n, dim, &centroids, k, &mut g);
        // Converged once almost no point changes cluster. Exact equality never triggers at 1M
        // (a handful always flip), so allow a 0.1% tolerance — real data stops well before n_iter.
        let changed = next.iter().zip(labels.iter()).filter(|(a, b)| a != b).count();
        labels = next;
        // Centroid means as a parallel fold-reduce: each thread accumulates its own (sums,
        // counts), then they're merged — avoids a data race on shared centroids.
        //
        // Sums accumulate in f64, not f32. The reduction tree's shape depends on the rayon thread
        // count, and f32 addition is not associative, so an f32 accumulator makes the centroids
        // (and thus a few labels) differ in low-order bits between machines with different core
        // counts. Widening the running total to f64 keeps the accumulation error far below the
        // final f32 rounding, so `(sum/count) as f32` lands on the same bits regardless of how the
        // partial sums were split — the "bit-identical across machines" guarantee the README makes.
        let (sums, counts) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![0.0f64; k * dim], vec![0usize; k]),
                |(mut s, mut cnt), i| {
                    let c = labels[i] as usize;
                    cnt[c] += 1;
                    let xi = &x[i * dim..i * dim + dim];
                    let sc = &mut s[c * dim..c * dim + dim];
                    for d in 0..dim {
                        sc[d] += xi[d] as f64;
                    }
                    (s, cnt)
                },
            )
            .reduce(
                || (vec![0.0f64; k * dim], vec![0usize; k]),
                |(mut s1, mut c1), (s2, c2)| {
                    for i in 0..k * dim {
                        s1[i] += s2[i];
                    }
                    for i in 0..k {
                        c1[i] += c2[i];
                    }
                    (s1, c1)
                },
            );
        let empty: Vec<usize> = (0..k).filter(|&c| counts[c] == 0).collect();
        // Stop only when labels are stable AND every cluster has points. An empty cluster left in
        // place makes inertia rise with k, so choose_k's elbow follows the seed (#30).
        if changed <= n / 1000 && empty.is_empty() {
            break;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0f64 / counts[c] as f64;
                for d in 0..dim {
                    centroids[c * dim + d] = (sums[c * dim + d] * inv) as f32;
                }
            }
        }
        if !empty.is_empty() {
            reseed_empty(x, dim, &labels, &mut centroids, &empty);
        }
    }
    labels
}

/// Move each empty centroid onto a point far from its own centroid.
///
/// The standard Lloyd repair (scikit-learn does the same): the points worst served by the current
/// centroids become the new ones. Picks the `empty.len()` points with the largest squared distance
/// to their assigned centroid, ties broken by the lower index, so it is deterministic.
fn reseed_empty(x: &[f32], dim: usize, labels: &[i64], centroids: &mut [f32], empty: &[usize]) {
    let n = labels.len();
    let dist: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let c = labels[i] as usize;
            let xi = &x[i * dim..i * dim + dim];
            let ci = &centroids[c * dim..c * dim + dim];
            let mut s = 0.0f64;
            for d in 0..dim {
                let diff = (xi[d] - ci[d]) as f64;
                s += diff * diff;
            }
            s
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| dist[b].total_cmp(&dist[a]).then(a.cmp(&b)));
    for (slot, &c) in empty.iter().enumerate() {
        let i = order[slot.min(n - 1)];
        centroids[c * dim..c * dim + dim].copy_from_slice(&x[i * dim..i * dim + dim]);
    }
}

/// Mini-batch k-means (Sculley 2010), for when CPU is the constraint rather than memory.
///
/// Full Lloyd touches all n rows on every iteration: at 1M x 128 with k=64 that is ~154
/// core-seconds, which is seconds on a 16-core box but minutes on a 0.5-2 vCPU container. This
/// samples `batch` rows per iteration instead and moves only the centroids those rows land on, so
/// per-iteration work is O(batch*k*d) rather than O(n*k*d).
///
/// Each sampled point pulls its centroid toward itself by a per-centroid learning rate 1/v[c],
/// where v[c] counts how many points that centroid has ever absorbed. The rate decays on its own,
/// so the centroids settle without a schedule to tune.
///
/// One full assignment pass at the end produces the labels — unavoidable, since every row needs
/// one. So the total cost is roughly two full passes instead of `n_iter` of them.
///
/// Deterministic given `seed`: the batch is drawn from the same seeded SplitMix64 stream, and the
/// within-batch update order is fixed. Thread count does not change the result.
fn minibatch_kmeans_impl(
    x: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    n_iter: usize,
    batch: usize,
    seed: u64,
) -> Vec<i64> {
    let mut rng = seed ^ 0x9E3779B97F4A7C15;
    let mut centroids = kmeanspp_seed(x, n, dim, k, &mut rng);
    let batch = batch.min(n).max(1);

    // v[c] starts at 1, not 0. With v[c]=0 the first point to reach a centroid gives eta=1 and
    // teleports it exactly onto that one point, throwing away the k-means++ placement that put it
    // there — which is how two centroids end up inside one blob while another blob goes uncovered.
    // Starting at 1 treats the seed as one prior observation, so the first step is a half-step.
    let mut counts = vec![1.0f32; k]; // v[c]: points this centroid has absorbed, plus the seed
    let mut xb = vec![0.0f32; batch * dim];
    let mut g = vec![0.0f32; batch * k];
    for _ in 0..n_iter {
        // Draw the batch with replacement. Sampling into a contiguous buffer keeps the GEMM
        // below on one dense block instead of gathering rows inside it.
        for b in 0..batch {
            let i = (splitmix64(&mut rng) % n as u64) as usize;
            xb[b * dim..b * dim + dim].copy_from_slice(&x[i * dim..i * dim + dim]);
        }
        let assigned = assign_gemm(&xb, batch, dim, &centroids, k, &mut g);
        // Sequential on purpose: the learning rate for a point depends on how many points its
        // centroid has already taken in THIS pass, so the order is part of the algorithm. It is
        // also cheap — O(batch*dim), against the GEMM's O(batch*k*dim).
        for (b, &c) in assigned.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1.0;
            let eta = 1.0 / counts[c];
            let (xi, cen) = (&xb[b * dim..b * dim + dim], &mut centroids[c * dim..c * dim + dim]);
            for d in 0..dim {
                cen[d] += eta * (xi[d] - cen[d]);
            }
        }
    }

    // Label every row against the settled centroids: one full pass, blocked so the scratch
    // buffer stays bounded rather than allocating n*k floats (at 1M x 64 that alone is 256 MB).
    let block = 65_536.min(n).max(1);
    let mut gfull = vec![0.0f32; block * k];
    let mut labels = vec![0i64; n];
    let mut start = 0usize;
    while start < n {
        let rows = block.min(n - start);
        let part = assign_gemm(
            &x[start * dim..(start + rows) * dim],
            rows,
            dim,
            &centroids,
            k,
            &mut gfull,
        );
        labels[start..start + rows].copy_from_slice(&part);
        start += rows;
    }
    labels
}

/// Mini-batch k-means: the CPU-constrained counterpart of `kmeans`. See `minibatch_kmeans_impl`.
#[pyfunction]
fn minibatch_kmeans<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    n_clusters: usize,
    n_iter: usize,
    batch_size: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let shape = data.shape();
    let n = shape[0];
    let dim = shape[1];
    let x: Vec<f32> = data.as_slice()?.iter().map(|&v| v as f32).collect();
    let labels = py.allow_threads(|| {
        minibatch_kmeans_impl(&x, n, dim, n_clusters, n_iter, batch_size, seed)
    });
    Ok(numpy::ndarray::Array1::from_vec(labels).into_pyarray_bound(py))
}

#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_layout, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    m.add_function(wrap_pyfunction!(minibatch_kmeans, m)?)?;
    Ok(())
}
