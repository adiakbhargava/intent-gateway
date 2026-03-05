//! c_o Neural Gateway — main server binary.
//!
//! Starts the gRPC server and ingestion pipeline.
//! Supports two modes:
//!   - Live (default): generates simulated EEG+gaze data
//!   - Replay: reads a .corec file with optional ONNX model inference

use clap::Parser;
use co_grpc::co_systems::intent_stream_server::IntentStreamServer;
use co_grpc::co_systems::FusedPacket;
use co_grpc::CoGateway;
use tokio::sync::broadcast;
use tonic::transport::Server;
use std::path::Path;

#[derive(Parser)]
#[command(name = "co-gateway", about = "c_o Neural Gateway server")]
struct Args {
    /// gRPC listen address
    #[arg(long, default_value = "[::1]:50051")]
    addr: String,

    /// ONNX model path (uses stub classifier if file not found)
    #[arg(long, default_value = "models/conv1d_fused.onnx")]
    model: String,

    /// Replay a .corec file instead of generating simulated data.
    /// Processes all frames at maximum speed, then keeps server alive
    /// for audit RPCs.
    #[arg(long)]
    replay_file: Option<String>,

    /// Number of EEG channels (ignored in replay mode — read from file)
    #[arg(long, default_value_t = 64)]
    channels: u32,

    /// EEG sample rate in Hz (ignored in replay mode — read from file)
    #[arg(long, default_value_t = 256.0)]
    sample_rate: f64,

    /// Pin threads to separate CPU cores
    #[arg(long)]
    pin_cores: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let addr = args.addr.parse()?;

    // Broadcast channel for streaming FusedPackets to subscribers
    let (tx, _rx) = broadcast::channel::<FusedPacket>(4096);

    let gateway = CoGateway::new(tx.clone());

    eprintln!("c_o Neural Gateway starting...");
    eprintln!("  addr:        {}", args.addr);

    // Choose pipeline mode
    if let Some(ref replay_file) = args.replay_file {
        eprintln!("  mode:        REPLAY");
        eprintln!("  replay_file: {}", replay_file);

        // Resolve model path — use it only if the file exists
        let model_path = if Path::new(&args.model).exists() {
            eprintln!("  model:       {} (ONNX)", args.model);
            Some(args.model.clone())
        } else {
            eprintln!("  model:       StubClassifier (no ONNX file at {})", args.model);
            None
        };

        let _replay_handle = co_grpc::spawn_replay_pipeline(
            tx,
            replay_file.clone(),
            model_path,
        );
    } else {
        eprintln!("  mode:        SIMULATED");
        eprintln!("  model:       {} (not loaded in simulated mode)", args.model);
        eprintln!("  channels:    {}", args.channels);
        eprintln!("  sample_rate: {} Hz", args.sample_rate);
        eprintln!("  pin_cores:   {}", args.pin_cores);

        let _ingest_handle = co_grpc::spawn_ingest_pipeline(
            tx,
            args.channels,
            args.sample_rate,
        );
    }

    eprintln!("Listening on {}", addr);

    Server::builder()
        .add_service(IntentStreamServer::new(gateway))
        .serve(addr)
        .await?;

    Ok(())
}
