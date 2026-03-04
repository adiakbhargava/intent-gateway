//! Fault injection framework for resilience testing.

// ---------------------------------------------------------------------------
// Xorshift64
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift64 seed must be non-zero");
        Xorshift64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_u64_mod(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    pub fn roll(&mut self, rate: f64) -> bool {
        if rate <= 0.0 { return false; }
        if rate >= 1.0 { return true; }
        let threshold = (rate * u64::MAX as f64) as u64;
        self.next_u64() < threshold
    }
}

// ---------------------------------------------------------------------------
// PacketDropper
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PacketDropper {
    drop_rate: f64,
    rng: Xorshift64,
    total_seen: u64,
    total_dropped: u64,
}

impl PacketDropper {
    pub fn new(drop_rate: f64, seed: u64) -> Self {
        PacketDropper {
            drop_rate: drop_rate.clamp(0.0, 1.0),
            rng: Xorshift64::new(seed),
            total_seen: 0,
            total_dropped: 0,
        }
    }

    pub fn should_drop(&mut self) -> bool {
        self.total_seen += 1;
        let drop = self.rng.roll(self.drop_rate);
        if drop { self.total_dropped += 1; }
        drop
    }

    pub fn drop_rate(&self) -> f64 { self.drop_rate }
    pub fn total_seen(&self) -> u64 { self.total_seen }
    pub fn total_dropped(&self) -> u64 { self.total_dropped }

    pub fn observed_drop_fraction(&self) -> f64 {
        if self.total_seen == 0 { 0.0 }
        else { self.total_dropped as f64 / self.total_seen as f64 }
    }
}

// ---------------------------------------------------------------------------
// PacketCorruptor
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct PacketCorruptor {
    corrupt_rate: f64,
    bytes_per_corrupt: usize,
    rng: Xorshift64,
    total_packets: u64,
    total_corrupted: u64,
}

impl PacketCorruptor {
    pub fn new(corrupt_rate: f64, bytes_per_corrupt: usize, seed: u64) -> Self {
        PacketCorruptor {
            corrupt_rate: corrupt_rate.clamp(0.0, 1.0),
            bytes_per_corrupt,
            rng: Xorshift64::new(seed),
            total_packets: 0,
            total_corrupted: 0,
        }
    }

    pub fn maybe_corrupt(&mut self, buf: &mut [u8]) -> bool {
        self.total_packets += 1;
        if buf.is_empty() || !self.rng.roll(self.corrupt_rate) {
            return false;
        }
        self.total_corrupted += 1;
        for _ in 0..self.bytes_per_corrupt {
            let pos = self.rng.next_u64_mod(buf.len() as u64) as usize;
            buf[pos] ^= 0xFF;
        }
        true
    }

    pub fn total_packets(&self) -> u64 { self.total_packets }
    pub fn total_corrupted(&self) -> u64 { self.total_corrupted }
}

// ---------------------------------------------------------------------------
// BackpressureStrategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureStrategy {
    Block,
    Drop,
    Yield,
}

impl BackpressureStrategy {
    pub fn apply<F>(&self, mut try_push: F) -> bool
    where
        F: FnMut() -> Result<(), ()>,
    {
        match self {
            BackpressureStrategy::Block => {
                loop {
                    if try_push().is_ok() { return true; }
                    std::hint::spin_loop();
                }
            }
            BackpressureStrategy::Drop => {
                try_push().is_ok()
            }
            BackpressureStrategy::Yield => {
                loop {
                    if try_push().is_ok() { return true; }
                    std::thread::yield_now();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FaultMetrics
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct FaultMetrics {
    pub packets_sent: u64,
    pub packets_dropped: u64,
    pub packets_corrupted: u64,
    pub decode_errors: u64,
    pub total_duration_ns: u64,
}

impl FaultMetrics {
    pub fn new() -> Self { FaultMetrics::default() }

    pub fn throughput_pps(&self) -> f64 {
        if self.total_duration_ns == 0 { return 0.0; }
        self.packets_sent as f64 / (self.total_duration_ns as f64 * 1e-9)
    }

    pub fn loss_fraction(&self) -> f64 {
        if self.packets_sent == 0 { return 0.0; }
        (self.packets_dropped + self.decode_errors) as f64 / self.packets_sent as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_produces_distinct_values() {
        let mut rng = Xorshift64::new(12345);
        let vals: Vec<u64> = (0..100).map(|_| rng.next_u64()).collect();
        let unique: std::collections::HashSet<_> = vals.iter().collect();
        assert_eq!(unique.len(), 100);
    }

    #[test]
    fn xorshift_roll_never() {
        let mut rng = Xorshift64::new(1);
        for _ in 0..1000 {
            assert!(!rng.roll(0.0));
        }
    }

    #[test]
    fn xorshift_roll_always() {
        let mut rng = Xorshift64::new(1);
        for _ in 0..1000 {
            assert!(rng.roll(1.0));
        }
    }

    #[test]
    fn dropper_zero_rate_drops_nothing() {
        let mut d = PacketDropper::new(0.0, 1);
        for _ in 0..1000 {
            assert!(!d.should_drop());
        }
        assert_eq!(d.total_dropped(), 0);
    }

    #[test]
    fn dropper_full_rate_drops_everything() {
        let mut d = PacketDropper::new(1.0, 1);
        for _ in 0..100 {
            assert!(d.should_drop());
        }
        assert_eq!(d.total_dropped(), 100);
    }

    #[test]
    fn corruptor_zero_rate_leaves_buffer_unchanged() {
        let mut c = PacketCorruptor::new(0.0, 1, 1);
        let original = vec![0xAAu8; 64];
        let mut buf = original.clone();
        for _ in 0..100 {
            c.maybe_corrupt(&mut buf);
        }
        assert_eq!(buf, original);
    }

    #[test]
    fn backpressure_drop_returns_false_on_full() {
        let pushed = BackpressureStrategy::Drop.apply(|| Err(()));
        assert!(!pushed);
    }

    #[test]
    fn backpressure_block_retries_until_success() {
        let mut attempts = 0usize;
        let pushed = BackpressureStrategy::Block.apply(|| {
            attempts += 1;
            if attempts >= 5 { Ok(()) } else { Err(()) }
        });
        assert!(pushed);
        assert_eq!(attempts, 5);
    }

    #[test]
    fn metrics_loss_fraction() {
        let m = FaultMetrics {
            packets_sent: 1000,
            packets_dropped: 50,
            decode_errors: 10,
            ..Default::default()
        };
        assert!((m.loss_fraction() - 0.06).abs() < 1e-10);
    }
}
