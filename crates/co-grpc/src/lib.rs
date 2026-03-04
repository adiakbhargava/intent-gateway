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
use co_ingest::{SimulatedEegSource, SimulatedGazeSource};
use co_inference::{StubClassifier, InferenceAccumulator};

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
