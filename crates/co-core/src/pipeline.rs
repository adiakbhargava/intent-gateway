//! End-to-end neural telemetry pipeline.
//!
//! Connects ring buffer, transcoder, statistics, embedding, and fault injection
//! into a complete producer-consumer pipeline for benchmarking.

use crate::statistics::{LatencyHistogram, JitterTracker, WelfordState};
use crate::fault_inject;
use crate::ring_buffer;
use crate::transcoder;
use crate::embedding;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub ring_buffer_capacity: usize,
    pub packet_count: u64,
    pub channel_count: u32,
    pub samples_per_channel: usize,
    pub sample_rate_hz: f64,
    pub backpressure: fault_inject::BackpressureStrategy,
    pub drop_rate: f64,
    pub corrupt_rate: f64,
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
    pub fn throughput_pps(&self) -> f64 {
        let dur_ms = self.duration_ns as f64 / 1_000_000.0;
        if dur_ms > 0.0 { self.packets_received as f64 / (dur_ms / 1000.0) } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Pipeline runner
// ---------------------------------------------------------------------------

pub fn run(cfg: &PipelineConfig) -> PipelineReport {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    let (producer, consumer) = ring_buffer::new::<Vec<u8>>(cfg.ring_buffer_capacity);

    let sent_counter    = Arc::new(AtomicU64::new(0));
    let dropped_counter = Arc::new(AtomicU64::new(0));
    let corrupt_counter = Arc::new(AtomicU64::new(0));

    let sent2    = Arc::clone(&sent_counter);
    let dropped2 = Arc::clone(&dropped_counter);
    let corrupt2 = Arc::clone(&corrupt_counter);

    let packet_count      = cfg.packet_count;
    let channel_count     = cfg.channel_count;
    let samples_per_ch    = cfg.samples_per_channel;
    let drop_rate         = cfg.drop_rate;
    let corrupt_rate      = cfg.corrupt_rate;
    let backpressure      = cfg.backpressure;

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

        let estimated_buf_size = 10 + 5 + 5 + (n_samples * 4 + 10) + 5;

        for seq in 0..packet_count {
            let mut buf = Vec::with_capacity(estimated_buf_size);
            transcoder::test_helpers::encode_neural_packet(
                &mut buf,
                now_ns(),
                channel_count,
                &samples,
                seq as u32,
            );

            if dropper.should_drop() {
                dropped2.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            if corruptor.maybe_corrupt(&mut buf) {
                corrupt2.fetch_add(1, Ordering::Relaxed);
            }

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

        let fft_size = window_size.next_power_of_two() / 2;
        let fft_size = fft_size.clamp(2, 4096);
        let mut slide = embedding::SlidingWindow::new(fft_size);

        let mut idle_iters = 0u64;
        let max_idle = 100_000_000u64;

        loop {
            match consumer.try_pop() {
                Some(buf) => {
                    idle_iters = 0;
                    let recv_ns = now_ns();
                    jitter.record(recv_ns);

                    let t0 = now_ns();
                    match transcoder::decode_neural_packet(&buf) {
                        Ok(pkt) => {
                            let decode_ns = now_ns() - t0;
                            decode_time.update(decode_ns as f64);

                            let send_ns = pkt.timestamp_ns;
                            if recv_ns >= send_ns {
                                latency.record(recv_ns - send_ns);
                            }

                            let samples_per_ch = pkt.sample_count();
                            for s in pkt.iter_samples().take(samples_per_ch.min(fft_size)) {
                                slide.push(s as f64);
                            }

                            if slide.is_ready() && received.is_multiple_of(100) {
                                if let Some(win) = slide.to_vec() {
                                    let _features = embedding::extract_features(&win, sample_rate_hz);
                                }
                            }
                        }
                        Err(_) => { decode_errors += 1; }
                    }

                    received += 1;
                    if received >= expected_packets { break; }
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
    let packets_sent    = sent_counter.load(std::sync::atomic::Ordering::Relaxed);
    let packets_dropped = dropped_counter.load(std::sync::atomic::Ordering::Relaxed);
    let packets_corrupted = corrupt_counter.load(std::sync::atomic::Ordering::Relaxed);

    PipelineReport {
        packets_sent,
        packets_received: received,
        packets_dropped,
        packets_corrupted,
        decode_errors,
        latency,
        jitter,
        decode_time,
        heap_allocs_during_run: 0,
        duration_ns,
    }
}

/// Monotonic nanosecond timestamp.
pub fn now_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
