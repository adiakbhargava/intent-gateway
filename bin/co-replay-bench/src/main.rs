//! Replay benchmark: reads a .corec file through the full pipeline
//! (fusion + feature extraction + stub inference) and reports latency/throughput.

use clap::Parser;
use std::time::Instant;

use co_core::embedding;
use co_core::statistics::{LatencyHistogram, WelfordState};
use co_fusion::FusionEngine;
use co_ingest::{EegSource, GazeSource};
use co_inference::{StubClassifier, InferenceAccumulator};
use co_replay::{ReplayEegSource, ReplayGazeSource};

#[derive(Parser)]
#[command(name = "co-replay-bench")]
#[command(about = "Benchmark the pipeline on real EEG+gaze data from .corec files")]
struct Args {
    /// Path to .corec file
    #[arg(long)]
    input: String,

    /// Number of EEG frames to process (0 = all)
    #[arg(long, default_value = "0")]
    limit: u64,
}

fn main() {
    let args = Args::parse();

    // Open both sources from the same .corec file
    let mut eeg_src = ReplayEegSource::open(&args.input)
        .expect("failed to open .corec for EEG");
    let mut gaze_src = ReplayGazeSource::open(&args.input)
        .expect("failed to open .corec for gaze");

    let header = eeg_src.header().clone();
    let n_ch = header.eeg_channels as usize;
    let total_eeg = header.num_eeg_frames;
    let total_gaze = header.num_gaze_frames;
    let eeg_rate = header.sample_rate_hz;
    let gaze_rate = header.gaze_rate_hz;

    println!("Dataset: {} EEG channels, {} EEG frames @ {} Hz, {} gaze frames @ {} Hz",
             n_ch, total_eeg, eeg_rate, total_gaze, gaze_rate);
    println!("Duration: {:.1}s", total_eeg as f64 / eeg_rate);
    println!();

    // Pipeline components
    let mut fusion = FusionEngine::new(10.0, 256);
    let classifier = StubClassifier;
    let mut accumulator = InferenceAccumulator::new(n_ch, 3, 500, 100);

    // Sliding window for feature extraction
    let fft_size = (n_ch).next_power_of_two().clamp(2, 4096);
    let mut slide = embedding::SlidingWindow::new(fft_size);

    // Timing
    let mut fusion_latency = LatencyHistogram::new();
    let mut feature_latency = LatencyHistogram::new();
    let mut inference_latency = LatencyHistogram::new();
    let mut total_latency = WelfordState::new();

    let limit = if args.limit > 0 { args.limit } else { total_eeg as u64 };

    // Pre-buffer all gaze frames (in a real system these arrive async)
    let gaze_start = Instant::now();
    let mut gaze_count = 0u64;
    while let Some(gaze) = gaze_src.next_gaze_frame() {
        fusion.push_gaze(gaze);
        gaze_count += 1;
    }
    let gaze_load_ms = gaze_start.elapsed().as_secs_f64() * 1000.0;
    println!("Gaze buffer: loaded {} frames in {:.1}ms", gaze_count, gaze_load_ms);

    // Process EEG frames through pipeline
    let mut processed = 0u64;
    let mut fused_count = 0u64;
    let mut features_computed = 0u64;
    let mut inferences_run = 0u64;

    let pipeline_start = Instant::now();

    while let Some(eeg) = eeg_src.next_eeg_frame() {
        let frame_start = Instant::now();

        // Fusion
        let t0 = Instant::now();
        let fused = fusion.fuse(eeg);
        let fusion_ns = t0.elapsed().as_nanos() as u64;
        fusion_latency.record(fusion_ns);

        if let Some(fused) = fused {
            fused_count += 1;

            // Feature extraction (every 100 frames)
            for &s in fused.eeg.channels.iter().take(fft_size) {
                slide.push(s as f64);
            }

            if slide.is_ready() && fused_count.is_multiple_of(100) {
                let t1 = Instant::now();
                if let Some(win) = slide.to_vec() {
                    let _features = embedding::extract_features(&win, eeg_rate);
                    features_computed += 1;
                }
                let feat_ns = t1.elapsed().as_nanos() as u64;
                feature_latency.record(feat_ns);
            }

            // Inference accumulation
            accumulator.push(
                &fused.eeg.channels,
                fused.gaze.x,
                fused.gaze.y,
                fused.gaze.pupil_diameter,
            );

            if accumulator.is_ready() {
                let t2 = Instant::now();
                let _pred = classifier.predict(
                    accumulator.eeg_window(),
                    accumulator.gaze_window(),
                );
                let inf_ns = t2.elapsed().as_nanos() as u64;
                inference_latency.record(inf_ns);
                inferences_run += 1;
            }
        }

        let frame_ns = frame_start.elapsed().as_nanos() as u64;
        total_latency.update(frame_ns as f64);

        processed += 1;
        if processed >= limit {
            break;
        }
    }

    let pipeline_elapsed = pipeline_start.elapsed();
    let pipeline_ms = pipeline_elapsed.as_secs_f64() * 1000.0;
    let throughput = processed as f64 / pipeline_elapsed.as_secs_f64();

    // Report
    println!();
    println!("=== Pipeline Results (Real EEG Data) ===");
    println!();
    println!("Frames processed:    {}", processed);
    println!("Frames fused:        {} ({:.1}%)", fused_count,
             fused_count as f64 / processed as f64 * 100.0);
    println!("Features extracted:  {}", features_computed);
    println!("Inferences run:      {}", inferences_run);
    println!("Degraded fraction:   {:.2}%", fusion.degraded_fraction() * 100.0);
    println!("Mean alignment:      {:.0} ns", fusion.mean_alignment_ns());
    println!();
    println!("--- Timing ---");
    println!("Total wall clock:    {:.1} ms", pipeline_ms);
    println!("Throughput:          {:.0} frames/sec ({:.1}x real-time @ {} Hz)",
             throughput, throughput / eeg_rate, eeg_rate);
    println!();
    println!("Per-frame total:     mean={:.0}ns  σ={:.0}ns",
             total_latency.mean(), total_latency.std_dev());
    println!();
    println!("Fusion latency:      p50={:.0}ns  p99={:.0}ns  max={:.0}ns",
             fusion_latency.percentile(50.0),
             fusion_latency.percentile(99.0),
             fusion_latency.max_ns());
    if feature_latency.count() > 0 {
        println!("Feature extraction:  p50={:.0}ns  p99={:.0}ns  max={:.0}ns",
                 feature_latency.percentile(50.0),
                 feature_latency.percentile(99.0),
                 feature_latency.max_ns());
    }
    if inference_latency.count() > 0 {
        println!("Inference (stub):    p50={:.0}ns  p99={:.0}ns  max={:.0}ns",
                 inference_latency.percentile(50.0),
                 inference_latency.percentile(99.0),
                 inference_latency.max_ns());
    }
}
