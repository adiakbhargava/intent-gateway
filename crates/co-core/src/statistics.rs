//! Online statistics for real-time neural telemetry.
//!
//! All structures use fixed-size stack/heap allocations (no dynamic growth)
//! so the hot-path measurement loop is allocation-free after initialization.

// ---------------------------------------------------------------------------
// WelfordState: numerically stable online mean/variance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WelfordState {
    count: u64,
    mean: f64,
    m2: f64,
}

impl WelfordState {
    pub fn new() -> Self {
        WelfordState { count: 0, mean: 0.0, m2: 0.0 }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    pub fn count(&self) -> u64 { self.count }
    pub fn mean(&self) -> f64 { self.mean }

    pub fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 }
    }

    pub fn std_dev(&self) -> f64 { self.variance().sqrt() }
}

impl Default for WelfordState {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// LatencyHistogram: logarithmic-bucket latency distribution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    buckets: [u64; 64],
    min_ns: u64,
    max_ns: u64,
    count: u64,
    sum_ns: u64,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        LatencyHistogram {
            buckets: [0; 64],
            min_ns: u64::MAX,
            max_ns: 0,
            count: 0,
            sum_ns: 0,
        }
    }

    fn bucket_for(ns: u64) -> usize {
        if ns == 0 {
            0
        } else {
            (63 - ns.leading_zeros() as usize).min(63)
        }
    }

    pub fn record(&mut self, latency_ns: u64) {
        let b = Self::bucket_for(latency_ns);
        self.buckets[b] += 1;
        self.count += 1;
        self.sum_ns = self.sum_ns.saturating_add(latency_ns);
        if latency_ns < self.min_ns { self.min_ns = latency_ns; }
        if latency_ns > self.max_ns { self.max_ns = latency_ns; }
    }

    pub fn percentile(&self, p: f64) -> u64 {
        if self.count == 0 { return 0; }
        let target = ((self.count as f64 * p / 100.0).ceil() as u64).max(1);
        let mut cumulative = 0u64;
        for (i, &cnt) in self.buckets.iter().enumerate() {
            cumulative += cnt;
            if cumulative >= target {
                return 1u64 << i;
            }
        }
        self.max_ns
    }

    pub fn mean_ns(&self) -> u64 {
        self.sum_ns.checked_div(self.count).unwrap_or(0)
    }

    pub fn min_ns(&self) -> u64 { if self.count == 0 { 0 } else { self.min_ns } }
    pub fn max_ns(&self) -> u64 { self.max_ns }
    pub fn count(&self) -> u64 { self.count }
}

impl Default for LatencyHistogram {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// JitterTracker: inter-arrival interval variance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct JitterTracker {
    last_timestamp_ns: Option<u64>,
    interval_stats: WelfordState,
}

impl JitterTracker {
    pub fn new() -> Self {
        JitterTracker { last_timestamp_ns: None, interval_stats: WelfordState::new() }
    }

    pub fn record(&mut self, timestamp_ns: u64) {
        if let Some(last) = self.last_timestamp_ns {
            let interval = timestamp_ns.saturating_sub(last) as f64;
            self.interval_stats.update(interval);
        }
        self.last_timestamp_ns = Some(timestamp_ns);
    }

    pub fn jitter_std_dev_ns(&self) -> f64 { self.interval_stats.std_dev() }
    pub fn mean_interval_ns(&self) -> f64 { self.interval_stats.mean() }
    pub fn sample_count(&self) -> u64 { self.interval_stats.count() }
}

impl Default for JitterTracker {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// RollingFft: Cooley-Tukey FFT over a sliding sample window
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct BandPowers {
    pub delta: f64,
    pub theta: f64,
    pub alpha: f64,
    pub beta:  f64,
    pub gamma: f64,
    pub total: f64,
}

impl BandPowers {
    pub fn relative(&self) -> BandPowers {
        if self.total < 1e-30 {
            return BandPowers::default();
        }
        BandPowers {
            delta: self.delta / self.total,
            theta: self.theta / self.total,
            alpha: self.alpha / self.total,
            beta:  self.beta  / self.total,
            gamma: self.gamma / self.total,
            total: 1.0,
        }
    }
}

pub struct RollingFft {
    window: Box<[f64]>,
    pos: usize,
    size: usize,
}

impl RollingFft {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two() && size > 0, "FFT size must be a non-zero power of two");
        let window = vec![0.0f64; size].into_boxed_slice();
        RollingFft { window, pos: 0, size }
    }

    pub fn push(&mut self, sample: f64) {
        self.window[self.pos] = sample;
        self.pos = (self.pos + 1) % self.size;
    }

    pub fn compute_power_spectrum(&self) -> Vec<f64> {
        let n = self.size;
        let mut re: Vec<f64> = (0..n).map(|i| self.window[(self.pos + i) % n]).collect();

        let nm1 = (n - 1) as f64;
        for (i, x) in re.iter_mut().enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / nm1).cos());
            *x *= w;
        }

        let mut im = vec![0.0f64; n];
        fft_inplace(&mut re, &mut im);

        let scale = 1.0 / (n as f64 * n as f64);
        (0..n / 2).map(|k| (re[k] * re[k] + im[k] * im[k]) * scale).collect()
    }

    pub fn band_powers(&self, sample_rate_hz: f64) -> BandPowers {
        let spectrum = self.compute_power_spectrum();
        let hz_per_bin = sample_rate_hz / self.size as f64;

        let bin_range = |lo_hz: f64, hi_hz: f64| -> f64 {
            let lo = (lo_hz / hz_per_bin).round() as usize;
            let hi = ((hi_hz / hz_per_bin).round() as usize).min(spectrum.len());
            if lo >= hi { 0.0 } else { spectrum[lo..hi].iter().sum() }
        };

        let delta = bin_range(0.5,  4.0);
        let theta = bin_range(4.0,  8.0);
        let alpha = bin_range(8.0, 13.0);
        let beta  = bin_range(13.0, 30.0);
        let gamma = bin_range(30.0, 100.0);
        let total = spectrum.iter().sum();

        BandPowers { delta, theta, alpha, beta, gamma, total }
    }

    pub fn size(&self) -> usize { self.size }
}

fn fft_inplace(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (w_re, w_im) = (ang.cos(), ang.sin());

        let mut i = 0;
        while i < n {
            let (mut t_re, mut t_im) = (1.0f64, 0.0f64);
            for jj in 0..len / 2 {
                let u_re = re[i + jj];
                let u_im = im[i + jj];
                let v_re = re[i + jj + len / 2] * t_re - im[i + jj + len / 2] * t_im;
                let v_im = re[i + jj + len / 2] * t_im + im[i + jj + len / 2] * t_re;
                re[i + jj]           = u_re + v_re;
                im[i + jj]           = u_im + v_im;
                re[i + jj + len / 2] = u_re - v_re;
                im[i + jj + len / 2] = u_im - v_im;
                let next_re = t_re * w_re - t_im * w_im;
                let next_im = t_re * w_im + t_im * w_re;
                t_re = next_re;
                t_im = next_im;
            }
            i += len;
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_mean_and_variance() {
        let mut s = WelfordState::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.update(v);
        }
        assert!((s.mean() - 5.0).abs() < 1e-10);
        assert!((s.variance() - 32.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn welford_single_value() {
        let mut s = WelfordState::new();
        s.update(42.0);
        assert_eq!(s.mean(), 42.0);
        assert_eq!(s.variance(), 0.0);
        assert_eq!(s.count(), 1);
    }

    #[test]
    fn welford_empty() {
        let s = WelfordState::new();
        assert_eq!(s.count(), 0);
        assert_eq!(s.mean(), 0.0);
        assert_eq!(s.variance(), 0.0);
    }

    #[test]
    fn histogram_basic_percentiles() {
        let mut h = LatencyHistogram::new();
        for _ in 0..900 { h.record(1_000); }
        for _ in 0..99  { h.record(1_000_000); }
        h.record(1_000_000_000);
        assert!(h.percentile(50.0) <= 2_048);
        assert!(h.percentile(99.0) >= 512_000);
        assert_eq!(h.count(), 1000);
    }

    #[test]
    fn histogram_min_max() {
        let mut h = LatencyHistogram::new();
        h.record(100);
        h.record(500);
        h.record(1000);
        assert_eq!(h.min_ns(), 100);
        assert_eq!(h.max_ns(), 1000);
    }

    #[test]
    fn histogram_empty_returns_zero() {
        let h = LatencyHistogram::new();
        assert_eq!(h.percentile(99.0), 0);
        assert_eq!(h.mean_ns(), 0);
        assert_eq!(h.min_ns(), 0);
    }

    #[test]
    fn jitter_constant_interval() {
        let mut jt = JitterTracker::new();
        for i in 0..1000u64 {
            jt.record(i * 1_000_000);
        }
        assert!(jt.jitter_std_dev_ns() < 1.0);
        assert!((jt.mean_interval_ns() - 1_000_000.0).abs() < 1.0);
    }

    #[test]
    fn jitter_variable_interval() {
        let mut jt = JitterTracker::new();
        let mut t = 0u64;
        for i in 0..1000u64 {
            t += 1_000_000 + (i % 10) * 100_000;
            jt.record(t);
        }
        assert!(jt.jitter_std_dev_ns() > 0.0);
        assert_eq!(jt.sample_count(), 999);
    }

    #[test]
    fn fft_dc_signal() {
        let mut fft = RollingFft::new(64);
        for _ in 0..64 { fft.push(1.0); }
        let spectrum = fft.compute_power_spectrum();
        assert!(spectrum[0] > 0.0);
        let dc = spectrum[0];
        for &p in &spectrum[1..] {
            assert!(p < dc);
        }
    }

    #[test]
    fn fft_known_frequency() {
        let n = 256usize;
        let mut fft = RollingFft::new(n);
        for i in 0..n {
            let v = (2.0 * std::f64::consts::PI * i as f64 / 4.0).sin();
            fft.push(v);
        }
        let spectrum = fft.compute_power_spectrum();
        let peak_bin = spectrum.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!((peak_bin as i64 - n as i64 / 4).abs() <= 2);
    }

    #[test]
    fn fft_band_powers_alpha_dominant() {
        let n = 256usize;
        let sample_rate = 256.0f64;
        let freq = 10.0f64;
        let mut fft = RollingFft::new(n);
        for i in 0..n {
            fft.push((2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin());
        }
        let bp = fft.band_powers(sample_rate).relative();
        assert!(bp.alpha > bp.delta && bp.alpha > bp.theta && bp.alpha > bp.beta);
    }
}
