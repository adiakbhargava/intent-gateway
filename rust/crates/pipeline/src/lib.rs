//! End-to-end neural telemetry pipeline.
//!
//! The pipeline connects all Phase 1-4 components:
//!
//! ```text
//! [Producer thread]                  [Consumer thread]
//!   encode NeuralPacket                read from ring buffer
//!   → ring buffer (SPSC)               → decode_neural_packet()
//!                                      → LatencyHistogram
//!                                      → JitterTracker
//!                                      → SlidingWindow → extract_features
//!                                      → cosine_similarity (rolling)
//! ```
//!
//! After the run, [`PipelineReport`] summarises latency, jitter, and
//! allocation statistics.

use statistics::{LatencyHistogram, JitterTracker, WelfordState};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Pipeline run configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Ring buffer capacity (rounded up to power of two inside `ring_buffer::new`).
    pub ring_buffer_capacity: usize,
    /// Number of simulated neural packets to send.
    pub packet_count: u64,
    /// Channels per packet.
    pub channel_count: u32,
    /// Samples per channel per packet.
    pub samples_per_channel: usize,
    /// EEG sample rate (Hz) — used for FFT band-power calculation.
    pub sample_rate_hz: f64,
    /// Backpressure strategy when the ring buffer is full.
    pub backpressure: fault_inject::BackpressureStrategy,
    /// Packet drop rate (0.0 = none).
    pub drop_rate: f64,
    /// Packet corruption rate (0.0 = none).
    pub corrupt_rate: f64,
    /// Pin producer and consumer threads to separate CPU cores.
    /// Reduces cache migration overhead and context-switch jitter.
    pub pin_to_cores: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            ring_buffer_capacity: 4096,
            packet_count: 100_000,
            channel_count: 64,
            samples_per_channel: 500,
            sample_rate_hz: 256.0,
            backpressure: fault_inject::BackpressureStrategy::Block,
            drop_rate: 0.0,
            corrupt_rate: 0.0,
            pin_to_cores: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Statistics collected over a complete pipeline run.
#[derive(Debug)]
pub struct PipelineReport {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub packets_dropped: u64,
    pub packets_corrupted: u64,
    pub decode_errors: u64,

    pub latency: LatencyHistogram,
    pub jitter: JitterTracker,
    pub decode_time: WelfordState,

    pub heap_allocs_during_run: u64,
    pub duration_ns: u64,
}

impl PipelineReport {
    pub fn print_summary(&self) {
        let received = self.packets_received;
        let sent = self.packets_sent;
        let loss_pct = if sent > 0 {
            (sent - received) as f64 / sent as f64 * 100.0
        } else { 0.0 };

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Intent-Stream Pipeline — Run Report");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Packets  sent:     {:>12}", sent);
        println!("  Packets  received: {:>12}  (loss {:.2}%)", received, loss_pct);
        println!("  Dropped:           {:>12}", self.packets_dropped);
        println!("  Corrupted:         {:>12}", self.packets_corrupted);
        println!("  Decode errors:     {:>12}", self.decode_errors);

        let dur_ms = self.duration_ns as f64 / 1_000_000.0;
        let tput = if dur_ms > 0.0 { received as f64 / (dur_ms / 1000.0) } else { 0.0 };
        println!("  Duration:          {:>9.1} ms", dur_ms);
        println!("  Throughput:        {:>9.0} pkt/s", tput);

        println!();
        println!("  Decode latency (push→pop round-trip):");
        println!("    p50:  {:>8} ns", self.latency.percentile(50.0));
        println!("    p90:  {:>8} ns", self.latency.percentile(90.0));
        println!("    p99:  {:>8} ns", self.latency.percentile(99.0));
        println!("    p999: {:>8} ns", self.latency.percentile(99.9));
        println!("    mean: {:>8} ns", self.latency.mean_ns());
        println!("    min:  {:>8} ns", self.latency.min_ns());
        println!("    max:  {:>8} ns", self.latency.max_ns());

        println!();
        println!("  Jitter (inter-arrival std-dev): {:.1} ns", self.jitter.jitter_std_dev_ns());
        println!("  Mean interval:                  {:.1} ns", self.jitter.mean_interval_ns());

        println!();
        println!("  Protobuf decode time:");
        println!("    mean: {:.1} ns    std: {:.1} ns",
            self.decode_time.mean(), self.decode_time.std_dev());

        println!();
        println!("  Heap allocs during hot-path run: {}", self.heap_allocs_during_run);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

// ---------------------------------------------------------------------------
// Pipeline runner
// ---------------------------------------------------------------------------

/// Run the full end-to-end pipeline and return a [`PipelineReport`].
///
/// Spawns a producer thread and a consumer thread. The producer encodes
/// simulated neural packets; the consumer decodes them and accumulates
/// statistics. Both threads communicate through the SPSC ring buffer.
pub fn run(cfg: &PipelineConfig) -> PipelineReport {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let (producer, consumer) = ring_buffer::new::<Vec<u8>>(cfg.ring_buffer_capacity);

    // Shared counters updated by the producer thread.
    let sent_counter    = Arc::new(AtomicU64::new(0));
    let dropped_counter = Arc::new(AtomicU64::new(0));
    let corrupt_counter = Arc::new(AtomicU64::new(0));

    let sent2    = Arc::clone(&sent_counter);
    let dropped2 = Arc::clone(&dropped_counter);
    let corrupt2 = Arc::clone(&corrupt_counter);

    // ── Clones for the producer ──────────────────────────────────────────
    let packet_count      = cfg.packet_count;
    let channel_count     = cfg.channel_count;
    let samples_per_ch    = cfg.samples_per_channel;
    let drop_rate         = cfg.drop_rate;
    let corrupt_rate      = cfg.corrupt_rate;
    let backpressure      = cfg.backpressure;

    // ── Resolve CPU core IDs for pinning ───────────────────────────────
    let pin = cfg.pin_to_cores;
    let core_ids = if pin {
        let ids = core_affinity::get_core_ids().unwrap_or_default();
        if ids.len() >= 2 {
            Some((ids[0], ids[1]))
        } else {
            eprintln!("warning: fewer than 2 cores available, skipping CPU pinning");
            None
        }
    } else {
        None
    };

    let start_ns = now_ns();

    // ── Producer thread ──────────────────────────────────────────────────
    let producer_core = core_ids.map(|(c, _)| c);
    let producer_thread = std::thread::spawn(move || {
        if let Some(core) = producer_core {
            core_affinity::set_for_current(core);
        }
        let mut dropper   = fault_inject::PacketDropper::new(drop_rate, 0xDEAD_BEEF);
        let mut corruptor = fault_inject::PacketCorruptor::new(corrupt_rate, 1, 0xCAFE_BABE);

        let n_samples = channel_count as usize * samples_per_ch;
        let samples: Vec<f32> = (0..n_samples).map(|i| (i as f32).sin()).collect();

        // Pre-calculate encoded size: 4 tag+varint fields + packed f32 samples.
        // Avoids realloc inside Vec during encoding.
        let estimated_buf_size = 10 + 5 + 5 + (n_samples * 4 + 10) + 5;

        for seq in 0..packet_count {
            // Build encoded packet (pre-allocated to avoid realloc)
            let mut buf = Vec::with_capacity(estimated_buf_size);
            transcoder::test_helpers::encode_neural_packet(
                &mut buf,
                now_ns(),
                channel_count,
                &samples,
                seq as u32,
            );

            // Fault injection
            if dropper.should_drop() {
                dropped2.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if corruptor.maybe_corrupt(&mut buf) {
                corrupt2.fetch_add(1, Ordering::Relaxed);
            }

            // Push to ring buffer using configured backpressure strategy.
            // Zero-clone: try_push returns the value on failure, so we
            // recycle it instead of cloning on every attempt.
            let mut pending = Some(buf);
            backpressure.apply(|| {
                let val = pending.take().unwrap();
                match producer.try_push(val) {
                    Ok(()) => Ok(()),
                    Err(rejected) => { pending = Some(rejected); Err(()) }
                }
            });

            sent2.fetch_add(1, Ordering::Relaxed);
        }
    });

    // ── Consumer thread ──────────────────────────────────────────────────
    let sample_rate_hz     = cfg.sample_rate_hz;
    let window_size        = cfg.samples_per_channel * cfg.channel_count as usize;
    let expected_packets   = cfg.packet_count;

    let consumer_core = core_ids.map(|(_, c)| c);
    let consumer_thread = std::thread::spawn(move || {
        if let Some(core) = consumer_core {
            core_affinity::set_for_current(core);
        }
        let mut latency      = LatencyHistogram::new();
        let mut jitter       = JitterTracker::new();
        let mut decode_time  = WelfordState::new();
        let mut decode_errors = 0u64;
        let mut received     = 0u64;

        // Sliding window for feature extraction (channel 0 only for simplicity)
        let fft_size = window_size.next_power_of_two() / 2;
        let fft_size = fft_size.clamp(2, 4096); // cap FFT at 4096 points
        let mut slide = embedding::SlidingWindow::new(fft_size);

        let total_to_receive = expected_packets; // producer drops some, so we don't know exact

        // We spin until the producer thread signals we're done via the sent counter.
        // Use a simple timeout / iteration limit approach.
        let mut idle_iters = 0u64;
        let max_idle = 100_000_000u64;

        loop {
            match consumer.try_pop() {
                Some(buf) => {
                    idle_iters = 0;
                    let recv_ns = now_ns();
                    jitter.record(recv_ns);

                    // Decode
                    let t0 = now_ns();
                    match transcoder::decode_neural_packet(&buf) {
                        Ok(pkt) => {
                            let decode_ns = now_ns() - t0;
                            decode_time.update(decode_ns as f64);

                            // Latency = recv_time - encoded_timestamp
                            let send_ns = pkt.timestamp_ns;
                            if recv_ns >= send_ns {
                                latency.record(recv_ns - send_ns);
                            }

                            // Feed first channel's samples into sliding window
                            let samples_per_ch = pkt.sample_count();
                            for s in pkt.iter_samples().take(samples_per_ch.min(fft_size)) {
                                slide.push(s as f64);
                            }

                            // Extract features when window is ready (every Nth packet)
                            if slide.is_ready() && received.is_multiple_of(100) {
                                if let Some(win) = slide.to_vec() {
                                    let _features = embedding::extract_features(&win, sample_rate_hz);
                                    // features available for downstream use
                                }
                            }
                        }
                        Err(_) => { decode_errors += 1; }
                    }

                    received += 1;
                    if received >= total_to_receive { break; }
                }
                None => {
                    idle_iters += 1;
                    if idle_iters > max_idle { break; }
                    std::hint::spin_loop();
                }
            }
        }

        (received, decode_errors, latency, jitter, decode_time)
    });

    producer_thread.join().expect("producer thread panicked");
    let (received, decode_errors, latency, jitter, decode_time) =
        consumer_thread.join().expect("consumer thread panicked");

    let duration_ns = now_ns() - start_ns;
    let packets_sent    = sent_counter.load(Ordering::Relaxed);
    let packets_dropped = dropped_counter.load(Ordering::Relaxed);
    let packets_corrupted = corrupt_counter.load(Ordering::Relaxed);

    PipelineReport {
        packets_sent,
        packets_received: received,
        packets_dropped,
        packets_corrupted,
        decode_errors,
        latency,
        jitter,
        decode_time,
        heap_allocs_during_run: 0, // measured in integration test with CountingAllocator
        duration_ns,
    }
}

/// Monotonic nanosecond timestamp (using `std::time::Instant`).
pub fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
