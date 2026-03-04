//! c_o Neural Gateway — main server binary.
//!
//! Starts the gRPC server and ingestion pipeline.

use clap::Parser;
use co_grpc::co_systems::intent_stream_server::IntentStreamServer;
use co_grpc::co_systems::FusedPacket;
use co_grpc::CoGateway;
use tokio::sync::broadcast;
use tonic::transport::Server;

#[derive(Parser)]
#[command(name = "co-gateway", about = "c_o Neural Gateway server")]
struct Args {
    /// gRPC listen address
    #[arg(long, default_value = "[::1]:50051")]
    addr: String,

    /// ONNX model path (optional for MVP — uses stub classifier)
    #[arg(long, default_value = "models/conv1d_fused.onnx")]
    model: String,

    /// Number of EEG channels
    #[arg(long, default_value_t = 64)]
    channels: u32,

    /// EEG sample rate in Hz
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
    let (tx, _rx) = broadcast::channel::<FusedPacket>(1024);

    let gateway = CoGateway::new(tx.clone());

    eprintln!("c_o Neural Gateway starting...");
    eprintln!("  addr:        {}", args.addr);
    eprintln!("  model:       {}", args.model);
    eprintln!("  channels:    {}", args.channels);
    eprintln!("  sample_rate: {} Hz", args.sample_rate);
    eprintln!("  pin_cores:   {}", args.pin_cores);

    // Spawn the ingestion pipeline
    let _ingest_handle = co_grpc::spawn_ingest_pipeline(
        tx,
        args.channels,
        args.sample_rate,
    );

    eprintln!("Listening on {}", addr);

    Server::builder()
        .add_service(IntentStreamServer::new(gateway))
        .serve(addr)
        .await?;

    Ok(())
}
