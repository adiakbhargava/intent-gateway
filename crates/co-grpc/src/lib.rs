//! gRPC streaming server for the c_o Neural Gateway.
//!
//! Implements the IntentStream service defined in co.proto:
//! - `Subscribe`: server-streaming RPC that pushes FusedPackets
//! - `Audit`: unary RPC that runs a latency benchmark

pub mod co_systems {
    include!(concat!(env!("OUT_DIR"), "/co_systems.rs"));
}

use co_systems::intent_stream_server::IntentStream;
use co_systems::{
    SubscribeRequest, FusedPacket,
    EmbeddingRequest, EmbeddingPacket,
    AuditRequest, AuditReport,
};

use co_core::pipeline::{PipelineConfig, self};
use co_fusion::FusionEngine;
use co_ingest::{EegSource, GazeSource, SimulatedEegSource, SimulatedGazeSource};
use co_inference::{StubClassifier, IntentClassifier, InferenceAccumulator};
use co_replay::{ReplayEegSource, ReplayGazeSource};

use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tonic::{Request, Response, Status};
use std::pin::Pin;
use tokio_stream::Stream;

// ---------------------------------------------------------------------------
// CoGateway server
// ---------------------------------------------------------------------------

pub struct CoGateway {
    tx: broadcast::Sender<FusedPacket>,
}

impl CoGateway {
    pub fn new(tx: broadcast::Sender<FusedPacket>) -> Self {
        CoGateway { tx }
    }

    pub fn sender(&self) -> broadcast::Sender<FusedPacket> {
        self.tx.clone()
    }
}

type SubscribeStream = Pin<Box<dyn Stream<Item = Result<FusedPacket, Status>> + Send>>;
type StreamEmbeddingsStream = Pin<Box<dyn Stream<Item = Result<EmbeddingPacket, Status>> + Send>>;

#[tonic::async_trait]
impl IntentStream for CoGateway {
    type SubscribeStream = SubscribeStream;
    type StreamEmbeddingsStream = StreamEmbeddingsStream;

    async fn subscribe(
        &self,
        _req: Request<SubscribeRequest>,
    ) -> Result<Response<Self::SubscribeStream>, Status> {
        let rx = self.tx.subscribe();
        let stream = BroadcastStream::new(rx);

        let mapped = tokio_stream::StreamExt::map(stream, |result: Result<FusedPacket, _>| {
            result.map_err(|e| Status::internal(format!("broadcast error: {e}")))
        });

        Ok(Response::new(Box::pin(mapped)))
    }

    async fn stream_embeddings(
        &self,
        _req: Request<EmbeddingRequest>,
    ) -> Result<Response<Self::StreamEmbeddingsStream>, Status> {
        let rx = self.tx.subscribe();
        let stream = BroadcastStream::new(rx);

        // Map FusedPacket → EmbeddingPacket (strip classification, keep features)
        let mapped = tokio_stream::StreamExt::map(stream, |result: Result<FusedPacket, _>| {
            match result {
                Ok(pkt) => Ok(EmbeddingPacket {
                    timestamp_ns: pkt.timestamp_ns,
                    sequence_id: pkt.sequence_id,
                    eeg_features: pkt.features,
                    gaze_x: pkt.gaze_x,
                    gaze_y: pkt.gaze_y,
                    pupil_diameter: pkt.pupil_diameter,
                    alignment_offset_ns: pkt.alignment_offset_ns,
                    raw_eeg: pkt.eeg_channels,
                }),
                Err(e) => Err(Status::internal(format!("broadcast error: {e}"))),
            }
        });

        Ok(Response::new(Box::pin(mapped)))
    }

    async fn audit(
        &self,
        req: Request<AuditRequest>,
    ) -> Result<Response<AuditReport>, Status> {
        let audit_req = req.into_inner();

        let cfg = PipelineConfig {
            packet_count: if audit_req.packet_count > 0 { audit_req.packet_count } else { 10_000 },
            drop_rate: audit_req.drop_rate,
            corrupt_rate: audit_req.corrupt_rate,
            pin_to_cores: audit_req.pin_cores,
            ..PipelineConfig::default()
        };

        let report = tokio::task::spawn_blocking(move || {
            pipeline::run(&cfg)
        })
        .await
        .map_err(|e| Status::internal(format!("pipeline error: {e}")))?;

        let duration_ms = report.duration_ns as f64 / 1_000_000.0;

        Ok(Response::new(AuditReport {
            packets_sent: report.packets_sent,
            packets_received: report.packets_received,
            packets_dropped: report.packets_dropped,
            decode_errors: report.decode_errors,
            duration_ms,
            throughput_pps: report.throughput_pps(),
            p50_ns: report.latency.percentile(50.0),
            p90_ns: report.latency.percentile(90.0),
            p99_ns: report.latency.percentile(99.0),
            p999_ns: report.latency.percentile(99.9),
            min_ns: report.latency.min_ns(),
            max_ns: report.latency.max_ns(),
            mean_ns: report.latency.mean_ns(),
            jitter_std_dev_ns: report.jitter.jitter_std_dev_ns(),
            mean_interval_ns: report.jitter.mean_interval_ns(),
            heap_allocs: report.heap_allocs_during_run,
        }))
    }
}

// ---------------------------------------------------------------------------
// Pipeline ingestion task
// ---------------------------------------------------------------------------

/// Spawns the ingestion pipeline that generates simulated data,
/// fuses it, and broadcasts FusedPackets to all subscribers.
pub fn spawn_ingest_pipeline(
    tx: broadcast::Sender<FusedPacket>,
    channel_count: u32,
    sample_rate_hz: f64,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        let mut eeg_src = SimulatedEegSource::new(channel_count, sample_rate_hz, 1);
        let mut gaze_src = SimulatedGazeSource::new(60.0);
        let mut fusion = FusionEngine::new(10.0, 256);
        let classifier = StubClassifier;
        let mut accumulator = InferenceAccumulator::new(
            channel_count as usize, 3, 500, 100,
        );
        let mut sequence = 0u32;

        // Sample interval for EEG in nanoseconds
        let eeg_interval_ns = (1_000_000_000.0 / sample_rate_hz) as u64;
        // Gaze at 60 Hz
        let gaze_interval_ns = (1_000_000_000.0 / 60.0) as u64;
        let mut next_gaze_ns = 0u64;

        loop {
            let now = pipeline::now_ns();

            // Push gaze frames at their rate
            if now >= next_gaze_ns {
                let gaze = gaze_src.next_frame();
                fusion.push_gaze(gaze);
                next_gaze_ns = now + gaze_interval_ns;
            }

            // Generate and fuse EEG frame
            let eeg = eeg_src.next_frame();
            if let Some(fused) = fusion.fuse(eeg) {
                // Accumulate for inference
                accumulator.push(
                    &fused.eeg.channels,
                    fused.gaze.x,
                    fused.gaze.y,
                    fused.gaze.pupil_diameter,
                );

                let (confidence, classification) = if accumulator.is_ready() {
                    let pred = classifier.predict(
                        accumulator.eeg_window(),
                        accumulator.gaze_window(),
                    );
                    (pred.confidence, pred.classification)
                } else {
                    (0.0, "observe")
                };

                let packet = FusedPacket {
                    timestamp_ns: fused.timestamp_ns,
                    sequence_id: sequence,
                    eeg_channels: fused.eeg.channels,
                    gaze_x: fused.gaze.x,
                    gaze_y: fused.gaze.y,
                    pupil_diameter: fused.gaze.pupil_diameter,
                    alignment_offset_ns: fused.alignment_offset_ns,
                    features: vec![],
                    confidence,
                    classification: classification.to_string(),
                };

                // Broadcast — ignore error if no subscribers
                let _ = tx.send(packet);
                sequence = sequence.wrapping_add(1);
            }

            // Sleep for one EEG sample interval
            std::thread::sleep(std::time::Duration::from_nanos(eeg_interval_ns));
        }
    })
}

// ---------------------------------------------------------------------------
// Replay pipeline — replays .corec files with optional ONNX model inference
// ---------------------------------------------------------------------------

/// Spawns a replay pipeline that reads EEG+gaze data from a `.corec` file,
/// fuses it, optionally runs ONNX model inference, and broadcasts
/// FusedPackets with latency metrics.
pub fn spawn_replay_pipeline(
    tx: broadcast::Sender<FusedPacket>,
    replay_file: String,
    model_path: Option<String>,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn_blocking(move || {
        // Open replay sources
        let mut eeg_src = ReplayEegSource::open(&replay_file)
            .unwrap_or_else(|e| panic!("failed to open replay file for EEG: {e}"));
        let mut gaze_src = ReplayGazeSource::open(&replay_file)
            .unwrap_or_else(|e| panic!("failed to open replay file for gaze: {e}"));

        let header = eeg_src.header().clone();
        let n_ch = header.eeg_channels as usize;

        eprintln!("Replay source: {}", replay_file);
        eprintln!("  EEG:  {}ch @ {} Hz, {} frames",
                  header.eeg_channels, header.sample_rate_hz, header.num_eeg_frames);
        eprintln!("  Gaze: @ {} Hz, {} frames",
                  header.gaze_rate_hz, header.num_gaze_frames);

        // Fusion engine
        let mut fusion = FusionEngine::new(10.0, header.sample_rate_hz as usize);

        // Load ONNX model or fall back to stub
        let mut onnx_classifier: Option<IntentClassifier> = model_path
            .as_ref()
            .and_then(|p| {
                match IntentClassifier::load(p) {
                    Ok(c) => {
                        eprintln!("  Model: {} (ONNX, {}ch × {} samples)",
                                  p, c.n_eeg_channels(), c.n_samples());
                        Some(c)
                    }
                    Err(e) => {
                        eprintln!("  Model: failed to load {} — {e}, using stub", p);
                        None
                    }
                }
            });

        if onnx_classifier.is_none() {
            eprintln!("  Model: StubClassifier (no ONNX model loaded)");
        }

        let stub = StubClassifier;
        let mut accumulator = InferenceAccumulator::new(n_ch, 3, 500, 100);
        let mut sequence = 0u32;

        // Latency tracking
        let pipeline_start = std::time::Instant::now();
        let mut total_eeg_frames = 0u64;
        let mut total_fused = 0u64;
        let mut inference_count = 0u64;
        let mut total_inference_ns = 0u64;
        let mut intent_count = 0u64;
        let mut observe_count = 0u64;
        let mut fusion_latency_ns_sum = 0u64;

        // Interleave gaze pushes with EEG processing based on timestamp ratio
        let gaze_per_eeg = header.gaze_rate_hz / header.sample_rate_hz;
        let mut gaze_accumulator = 0.0f64;

        // Process all frames as fast as possible (no sleep — benchmark mode)
        loop {
            // Push gaze frames proportional to sample rate ratio
            gaze_accumulator += gaze_per_eeg;
            while gaze_accumulator >= 1.0 {
                if let Some(gaze) = gaze_src.next_gaze_frame() {
                    fusion.push_gaze(gaze);
                }
                gaze_accumulator -= 1.0;
            }

            // Get next EEG frame
            let eeg = match eeg_src.next_eeg_frame() {
                Some(f) => f,
                None => break, // End of file
            };

            total_eeg_frames += 1;

            let fuse_start = std::time::Instant::now();
            if let Some(fused) = fusion.fuse(eeg) {
                let fuse_elapsed = fuse_start.elapsed().as_nanos() as u64;
                fusion_latency_ns_sum += fuse_elapsed;
                total_fused += 1;

                // Accumulate for inference
                accumulator.push(
                    &fused.eeg.channels,
                    fused.gaze.x,
                    fused.gaze.y,
                    fused.gaze.pupil_diameter,
                );

                let (confidence, classification) = if accumulator.is_ready() {
                    if let Some(ref mut model) = onnx_classifier {
                        match model.predict(
                            accumulator.eeg_window(),
                            accumulator.gaze_window(),
                        ) {
                            Ok(pred) => {
                                inference_count += 1;
                                total_inference_ns += pred.inference_ns;
                                if pred.classification == "intent" {
                                    intent_count += 1;
                                } else {
                                    observe_count += 1;
                                }
                                (pred.confidence, pred.classification)
                            }
                            Err(_) => {
                                observe_count += 1;
                                (0.5, "observe")
                            }
                        }
                    } else {
                        let pred = stub.predict(
                            accumulator.eeg_window(),
                            accumulator.gaze_window(),
                        );
                        observe_count += 1;
                        (pred.confidence, pred.classification)
                    }
                } else {
                    (0.0, "accumulating")
                };

                let packet = FusedPacket {
                    timestamp_ns: fused.timestamp_ns,
                    sequence_id: sequence,
                    eeg_channels: fused.eeg.channels,
                    gaze_x: fused.gaze.x,
                    gaze_y: fused.gaze.y,
                    pupil_diameter: fused.gaze.pupil_diameter,
                    alignment_offset_ns: fused.alignment_offset_ns,
                    features: vec![],
                    confidence,
                    classification: classification.to_string(),
                };

                let _ = tx.send(packet);
                sequence = sequence.wrapping_add(1);
            }
        }

        // Report
        let elapsed = pipeline_start.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        let throughput = (total_eeg_frames * 1000).checked_div(elapsed_ms).unwrap_or(0);
        let realtime_factor = if elapsed_ms > 0 {
            let data_duration_ms = (header.num_eeg_frames as f64
                / header.sample_rate_hz * 1000.0) as u64;
            data_duration_ms as f64 / elapsed_ms as f64
        } else {
            0.0
        };
        let avg_fusion_ns = fusion_latency_ns_sum.checked_div(total_fused).unwrap_or(0);
        let avg_inference_us = total_inference_ns
            .checked_div(inference_count)
            .unwrap_or(0)
            / 1000;

        eprintln!();
        eprintln!("╔══════════════════════════════════════════════════════════╗");
        eprintln!("║            REPLAY PIPELINE COMPLETE                      ║");
        eprintln!("╠══════════════════════════════════════════════════════════╣");
        eprintln!("║  EEG frames read:     {:>10}                        ║", total_eeg_frames);
        eprintln!("║  Fused packets:       {:>10}                        ║", total_fused);
        eprintln!("║  Inferences:          {:>10}                        ║", inference_count);
        eprintln!("║    Intent:            {:>10}                        ║", intent_count);
        eprintln!("║    Observe:           {:>10}                        ║", observe_count);
        eprintln!("║  Elapsed:             {:>10} ms                     ║", elapsed_ms);
        eprintln!("║  Throughput:          {:>10} frames/sec              ║", throughput);
        eprintln!("║  Real-time factor:    {:>10.1}x                       ║", realtime_factor);
        eprintln!("║  Avg fusion latency:  {:>10} ns                     ║", avg_fusion_ns);
        eprintln!("║  Avg inference:       {:>10} µs                     ║", avg_inference_us);
        if onnx_classifier.is_some() {
            eprintln!("║  Model:               ONNX (real inference)              ║");
        } else {
            eprintln!("║  Model:               Stub (no ONNX loaded)              ║");
        }
        eprintln!("╚══════════════════════════════════════════════════════════╝");
    })
}
