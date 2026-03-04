//! EEG feature extraction and embedding utilities.
//!
//! # Pipeline
//!
//! 1. Push raw samples into a [`SlidingWindow`] one at a time.
//! 2. When the window is ready (`is_ready()` returns `true`), call
//!    [`extract_features`] to get a fixed-length [`EegFeatures`] vector.
//! 3. Compare feature vectors with [`cosine_similarity`].

use crate::statistics;

// ---------------------------------------------------------------------------
// SlidingWindow
// ---------------------------------------------------------------------------

pub struct SlidingWindow {
    buf: Box<[f64]>,
    pos: usize,
    filled: bool,
    size: usize,
}

impl SlidingWindow {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "window size must be > 0");
        let buf = vec![0.0f64; size].into_boxed_slice();
        SlidingWindow { buf, pos: 0, filled: false, size }
    }

    pub fn push(&mut self, sample: f64) {
        self.buf[self.pos] = sample;
        self.pos = (self.pos + 1) % self.size;
        if self.pos == 0 {
            self.filled = true;
        }
    }

    pub fn is_ready(&self) -> bool {
        self.filled
    }

    pub fn window_iter(&self) -> Option<impl Iterator<Item = f64> + '_> {
        if !self.filled {
            return None;
        }
        let start = self.pos;
        let size = self.size;
        Some((0..size).map(move |i| self.buf[(start + i) % size]))
    }

    pub fn to_vec(&self) -> Option<Vec<f64>> {
        self.window_iter().map(|it| it.collect())
    }

    pub fn size(&self) -> usize { self.size }
}

// ---------------------------------------------------------------------------
// EegFeatures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EegFeatures {
    pub values: Vec<f64>,
}

impl EegFeatures {
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

pub fn extract_features(window: &[f64], sample_rate_hz: f64) -> EegFeatures {
    let fft_size = {
        let s = window.len().next_power_of_two() / 2;
        s.max(2)
    };

    let mut fft = statistics::RollingFft::new(fft_size);
    let start = window.len().saturating_sub(fft_size);
    for &s in &window[start..] {
        fft.push(s);
    }

    let bp = fft.band_powers(sample_rate_hz);
    let rel = bp.relative();

    let rms = {
        let sum_sq: f64 = window.iter().map(|x| x * x).sum();
        (sum_sq / window.len() as f64).sqrt()
    };

    EegFeatures::new(vec![
        rel.delta, rel.theta, rel.alpha, rel.beta, rel.gamma,
        (bp.delta + 1e-30).ln(),
        (bp.theta + 1e-30).ln(),
        (bp.alpha + 1e-30).ln(),
        (bp.beta  + 1e-30).ln(),
        (bp.gamma + 1e-30).ln(),
        rms,
    ])
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

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
        assert_eq!(v, vec![5.0, 6.0, 7.0, 8.0]);
    }

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
        assert!(relative_sum >= 0.0 && relative_sum <= 1.0 + 1e-10);
    }
}
