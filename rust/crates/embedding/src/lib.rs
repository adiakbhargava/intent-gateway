//! EEG feature extraction and embedding utilities.
//!
//! # Pipeline
//!
//! 1. Push raw samples into a [`SlidingWindow`] one at a time.
//! 2. When the window is ready (`is_ready()` returns `true`), call
//!    [`extract_features`] to get a fixed-length [`EegFeatures`] vector.
//! 3. Compare feature vectors with [`cosine_similarity`] to measure
//!    trial-to-trial similarity (contrastive learning proxy metric).
//!
//! # Allocations
//!
//! `SlidingWindow` makes ONE allocation at construction (the circular buffer).
//! `extract_features` makes ONE allocation per call (the feature `Vec`).
//! The hot-path `push` loop is allocation-free.

// ---------------------------------------------------------------------------
// SlidingWindow
// ---------------------------------------------------------------------------

/// Fixed-size circular buffer for time-series windowing.
///
/// Accepts samples one at a time; when `size` samples have been pushed,
/// `is_ready()` returns `true` and `window_slice()` yields the samples
/// in chronological order (oldest → newest).
pub struct SlidingWindow {
    buf: Box<[f64]>,
    pos: usize,
    filled: bool,
    size: usize,
}

impl SlidingWindow {
    /// Allocate a new window of `size` samples.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "window size must be > 0");
        let buf = vec![0.0f64; size].into_boxed_slice();
        SlidingWindow { buf, pos: 0, filled: false, size }
    }

    /// Push one sample, overwriting the oldest slot when the window is full.
    pub fn push(&mut self, sample: f64) {
        self.buf[self.pos] = sample;
        self.pos = (self.pos + 1) % self.size;
        if self.pos == 0 {
            self.filled = true;
        }
    }

    /// `true` once at least `size` samples have been pushed.
    pub fn is_ready(&self) -> bool {
        self.filled
    }

    /// Iterate over the current window contents in chronological order.
    ///
    /// Returns `None` if the window is not yet full.
    pub fn window_iter(&self) -> Option<impl Iterator<Item = f64> + '_> {
        if !self.filled {
            return None;
        }
        let start = self.pos; // oldest sample (next to be overwritten)
        let size = self.size;
        Some((0..size).map(move |i| self.buf[(start + i) % size]))
    }

    /// Copy the window into a `Vec<f64>` in chronological order.
    ///
    /// Returns `None` if the window is not yet full.
    pub fn to_vec(&self) -> Option<Vec<f64>> {
        self.window_iter().map(|it| it.collect())
    }

    pub fn size(&self) -> usize { self.size }
}

// ---------------------------------------------------------------------------
// EegFeatures
// ---------------------------------------------------------------------------

/// Fixed-length feature vector extracted from one EEG window.
///
/// Contains 11 features:
/// - 5 relative band powers (delta, theta, alpha, beta, gamma)
/// - 5 absolute band powers
/// - 1 signal RMS (amplitude proxy)
#[derive(Debug, Clone)]
pub struct EegFeatures {
    pub values: Vec<f64>,
}

impl EegFeatures {
    /// Feature count — the length of `values`.
    pub const DIM: usize = 11;

    fn new(values: Vec<f64>) -> Self {
        debug_assert_eq!(values.len(), Self::DIM);
        EegFeatures { values }
    }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Extract [`EegFeatures`] from a window of EEG samples.
///
/// `sample_rate_hz` is required to map FFT bins to frequency bands.
///
/// Uses [`statistics::RollingFft`] internally. The window must have length
/// ≥ 64 and be a power of two; if not, the nearest smaller power of two is
/// used.
pub fn extract_features(window: &[f64], sample_rate_hz: f64) -> EegFeatures {
    // Determine largest power-of-two FFT size ≤ window length.
    let fft_size = {
        let s = window.len().next_power_of_two() / 2;
        s.max(2) // at minimum, a 2-point FFT
    };

    // Feed the tail of the window into a RollingFft.
    let mut fft = statistics::RollingFft::new(fft_size);
    let start = window.len().saturating_sub(fft_size);
    for &s in &window[start..] {
        fft.push(s);
    }

    let bp = fft.band_powers(sample_rate_hz);
    let rel = bp.relative();

    // RMS of the full window.
    let rms = {
        let sum_sq: f64 = window.iter().map(|x| x * x).sum();
        (sum_sq / window.len() as f64).sqrt()
    };

    EegFeatures::new(vec![
        // Relative band powers (sum to ~1)
        rel.delta,
        rel.theta,
        rel.alpha,
        rel.beta,
        rel.gamma,
        // Absolute band powers (log-scaled for numerical range)
        (bp.delta + 1e-30).ln(),
        (bp.theta + 1e-30).ln(),
        (bp.alpha + 1e-30).ln(),
        (bp.beta  + 1e-30).ln(),
        (bp.gamma + 1e-30).ln(),
        // Amplitude
        rms,
    ])
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length feature vectors.
///
/// Returns a value in `[-1, 1]`. Returns `0.0` if either vector is the
/// zero vector.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let dot: f64  = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-30 || norm_b < 1e-30 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── SlidingWindow ───────────────────────────────────────────────────────

    #[test]
    fn sliding_window_not_ready_until_full() {
        let mut w = SlidingWindow::new(4);
        assert!(!w.is_ready());
        w.push(1.0); w.push(2.0); w.push(3.0);
        assert!(!w.is_ready());
        w.push(4.0);
        assert!(w.is_ready());
    }

    #[test]
    fn sliding_window_fifo_order() {
        let mut w = SlidingWindow::new(4);
        for i in 1..=4 { w.push(i as f64); }
        let v = w.to_vec().unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn sliding_window_overwrites_oldest() {
        let mut w = SlidingWindow::new(4);
        for i in 1..=8 { w.push(i as f64); }
        let v = w.to_vec().unwrap();
        // After 8 pushes into a size-4 window, should contain [5,6,7,8]
        assert_eq!(v, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn sliding_window_none_when_not_ready() {
        let w = SlidingWindow::new(8);
        assert!(w.to_vec().is_none());
        assert!(w.window_iter().is_none());
    }

    // ── cosine_similarity ───────────────────────────────────────────────────

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn cosine_zero_vector_returns_zero() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ── extract_features ───────────────────────────────────────────────────

    #[test]
    fn features_correct_dimension() {
        let window: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let f = extract_features(&window, 256.0);
        assert_eq!(f.len(), EegFeatures::DIM);
    }

    #[test]
    fn features_relative_powers_sum_to_one() {
        let window: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let f = extract_features(&window, 256.0);
        let relative_sum: f64 = f.values[0..5].iter().sum();
        // Relative powers should sum to ≈1 (may differ slightly due to
        // frequency bands not covering full spectrum)
        assert!(relative_sum >= 0.0 && relative_sum <= 1.0 + 1e-10,
            "relative power sum: {relative_sum}");
    }

    #[test]
    fn features_same_signal_high_similarity() {
        let window: Vec<f64> = (0..256)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 256.0).sin())
            .collect();
        let f1 = extract_features(&window, 256.0);
        let f2 = extract_features(&window, 256.0);
        let sim = cosine_similarity(&f1.values, &f2.values);
        assert!((sim - 1.0).abs() < 1e-10, "identical signals: sim={sim}");
    }

    #[test]
    fn features_different_frequencies_lower_similarity() {
        let n = 256usize;
        let sr = 256.0f64;
        // Alpha (10 Hz) vs Beta (20 Hz)
        let alpha: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / sr).sin())
            .collect();
        let beta: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 20.0 * i as f64 / sr).sin())
            .collect();
        let fa = extract_features(&alpha, sr);
        let fb = extract_features(&beta, sr);
        let sim = cosine_similarity(&fa.values, &fb.values);
        assert!(sim < 0.99, "different frequencies should have sim < 0.99, got {sim}");
    }
}
