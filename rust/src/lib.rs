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

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

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

#[pymodule]
fn fastumap_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize_layout, m)?)?;
    Ok(())
}
