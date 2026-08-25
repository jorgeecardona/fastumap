//! Native SGD layout kernel for fastumap.
//!
//! This is the *in-place* (Hogwild) attract/repel walk that umap-learn's numba kernel does
//! — later edges in an epoch see earlier moves — which is both faster and more faithful to
//! umap-learn than the pure-numpy snapshot update. The heavy loop runs with the GIL
//! released (`py.allow_threads`), so concurrent `asyncio.to_thread` calls run in parallel;
//! nothing here touches Python objects, and the borrow checker guarantees no data races.
//!
//! Determinism: a seeded SplitMix64 stream drives negative sampling, so a given seed is
//! reproducible (not bit-identical to the numpy path — different RNG and update order, the
//! same reason fastumap and umap-learn differ).
//!
//! NOTE: unbuilt scaffold. The first `maturin develop` may need minor PyO3 0.22 API tweaks.

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
/// The assignment step is the O(n*k*d) wall. It's computed as a single GEMM
/// `G = X @ centroidsᵀ` via matrixmultiply's SIMD kernel (nearest centroid = argmin over
/// `‖c‖² − 2·G[i,·]`, since `‖x‖²` is constant per row), then argmin parallelised over rows with
/// rayon. All in f32 for SIMD width — clustering tolerates it. Deterministic given `seed` (not
/// identical to the numpy path: different init RNG and f32). Returns the `(n,)` integer labels.
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

fn kmeans_impl(x: &[f32], n: usize, dim: usize, k: usize, n_iter: usize, seed: u64) -> Vec<i64> {
    let dist2 = |a: &[f32], b: &[f32]| -> f32 {
        let mut s = 0.0f32;
        for d in 0..dim {
            let diff = a[d] - b[d];
            s += diff * diff;
        }
        s
    };

    // ─── k-means++ seeding ───
    let mut rng = seed ^ 0x9E3779B97F4A7C15;
    let mut centroids = vec![0.0f32; k * dim];
    // Full k-means++ is O(n*k*d) — the fixed cost that dominates at 1M. Seed it from a random
    // subsample instead (what faiss/sklearn do at scale); the sample captures the distribution,
    // then Lloyd below refines against all n points.
    let sn = n.min(100_000);
    let sx: Vec<f32> = if sn < n {
        let mut xs = vec![0.0f32; sn * dim];
        for s in 0..sn {
            let i = (splitmix64(&mut rng) % n as u64) as usize;
            xs[s * dim..s * dim + dim].copy_from_slice(&x[i * dim..i * dim + dim]);
        }
        xs
    } else {
        x.to_vec()
    };
    let first = (splitmix64(&mut rng) % sn as u64) as usize;
    centroids[0..dim].copy_from_slice(&sx[first * dim..first * dim + dim]);
    let mut closest: Vec<f32> = (0..sn)
        .into_par_iter()
        .map(|i| dist2(&sx[i * dim..i * dim + dim], &centroids[0..dim]))
        .collect();
    for c in 1..k {
        let total: f32 = closest.par_iter().sum();
        let target = (splitmix64(&mut rng) as f64 / u64::MAX as f64) as f32 * total;
        let mut chosen = sn - 1;
        if total > 0.0 {
            let mut acc = 0.0f32;
            for (i, &d2) in closest.iter().enumerate() {
                acc += d2;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
        } else {
            chosen = (splitmix64(&mut rng) % sn as u64) as usize;
        }
        centroids[c * dim..c * dim + dim].copy_from_slice(&sx[chosen * dim..chosen * dim + dim]);
        let cen = centroids[c * dim..c * dim + dim].to_vec();
        closest.par_iter_mut().enumerate().for_each(|(i, cl)| {
            let d2 = dist2(&sx[i * dim..i * dim + dim], &cen);
            if d2 < *cl {
                *cl = d2;
            }
        });
    }

    // ─── Lloyd iterations (GEMM assignment) ───
    let mut labels = vec![0i64; n];
    let mut g = vec![0.0f32; n * k]; // G = X @ centroidsᵀ, reused each iteration
    for _ in 0..n_iter {
        let mut c_sq = vec![0.0f32; k];
        for j in 0..k {
            let mut s = 0.0f32;
            for d in 0..dim {
                let v = centroids[j * dim + d];
                s += v * v;
            }
            c_sq[j] = s;
        }
        // G (n×k) = X (n×dim) @ centroidsᵀ. centroids is (k×dim) row-major; viewing it as B with
        // row-stride 1, col-stride dim gives Cᵀ (dim×k) without a copy. matrixmultiply is
        // single-threaded, so split X into per-thread row-blocks and GEMM each independently.
        let n_threads = rayon::current_num_threads().max(1);
        let block = n.div_ceil(n_threads).max(1);
        g.par_chunks_mut(block * k).enumerate().for_each(|(bi, gchunk)| {
            let start = bi * block;
            let rows = gchunk.len() / k;
            if rows == 0 {
                return;
            }
            unsafe {
                matrixmultiply::sgemm(
                    rows,
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
        let next: Vec<i64> = (0..n)
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
            .collect();
        // Converged once almost no point changes cluster. Exact equality never triggers at 1M
        // (a handful always flip), so allow a 0.1% tolerance — real data stops well before n_iter.
        let changed = next.iter().zip(labels.iter()).filter(|(a, b)| a != b).count();
        labels = next;
        if changed <= n / 1000 {
            break;
        }
        // Centroid means as a parallel fold-reduce: each thread accumulates its own (sums,
        // counts), then they're merged — avoids a data race on shared centroids.
        let (sums, counts) = (0..n)
            .into_par_iter()
            .fold(
                || (vec![0.0f32; k * dim], vec![0usize; k]),
                |(mut s, mut cnt), i| {
                    let c = labels[i] as usize;
                    cnt[c] += 1;
                    let xi = &x[i * dim..i * dim + dim];
                    let sc = &mut s[c * dim..c * dim + dim];
                    for d in 0..dim {
                        sc[d] += xi[d];
                    }
                    (s, cnt)
                },
            )
            .reduce(
                || (vec![0.0f32; k * dim], vec![0usize; k]),
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
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0f32 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }
    }
    labels
}

#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_layout, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans, m)?)?;
    Ok(())
}
