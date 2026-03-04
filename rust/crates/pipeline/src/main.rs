//! Intent-Stream Pipeline — end-to-end entry point.
//!
//! Runs the full neural telemetry pipeline with configurable packet count and
//! optional fault injection, then prints a latency/throughput report.
//!
//! # Usage
//!
//! Default run (100k packets, no faults):
//!   cargo run -p pipeline --release
//!
//! Short smoke test:
//!   cargo run -p pipeline --release -- --packets 1000
//!
//! With fault injection:
//!   cargo run -p pipeline --release -- --packets 50000 --drop-rate 0.01 --corrupt-rate 0.005

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut cfg = pipeline::PipelineConfig::default();

    // Minimal flag parser — avoids pulling in a CLI crate.
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--packets" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.packet_count = v.parse().expect("--packets: expected u64");
                }
            }
            "--drop-rate" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.drop_rate = v.parse().expect("--drop-rate: expected f64 in [0,1]");
                }
            }
            "--corrupt-rate" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.corrupt_rate = v.parse().expect("--corrupt-rate: expected f64 in [0,1]");
                }
            }
            "--capacity" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.ring_buffer_capacity = v.parse().expect("--capacity: expected usize");
                }
            }
            "--channels" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.channel_count = v.parse().expect("--channels: expected u32");
                }
            }
            "--samples" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.samples_per_channel = v.parse().expect("--samples: expected usize");
                }
            }
            "--backpressure" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    cfg.backpressure = match v.as_str() {
                        "block" => fault_inject::BackpressureStrategy::Block,
                        "drop"  => fault_inject::BackpressureStrategy::Drop,
                        "yield" => fault_inject::BackpressureStrategy::Yield,
                        other   => panic!("unknown backpressure strategy: {other}"),
                    };
                }
            }
            "--pin-cores" => {
                cfg.pin_to_cores = true;
            }
            "--help" | "-h" => {
                println!("intent-stream-pipeline");
                println!();
                println!("Options:");
                println!("  --packets N          Number of packets to send (default: 100000)");
                println!("  --channels N         Channels per packet (default: 64)");
                println!("  --samples N          Samples per channel (default: 500)");
                println!("  --capacity N         Ring buffer capacity (default: 4096)");
                println!("  --drop-rate F        Packet drop probability 0..1 (default: 0.0)");
                println!("  --corrupt-rate F     Packet corruption probability 0..1 (default: 0.0)");
                println!("  --backpressure MODE  block|drop|yield (default: block)");
                println!("  --pin-cores          Pin producer/consumer to separate CPU cores");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Print configuration
    println!("Configuration:");
    println!("  packets:      {}", cfg.packet_count);
    println!("  channels:     {}", cfg.channel_count);
    println!("  samples/ch:   {}", cfg.samples_per_channel);
    println!("  ring buf cap: {}", cfg.ring_buffer_capacity);
    println!("  drop rate:    {:.3}", cfg.drop_rate);
    println!("  corrupt rate: {:.3}", cfg.corrupt_rate);
    println!("  backpressure: {:?}", cfg.backpressure);
    println!("  pin-cores:    {}", cfg.pin_to_cores);
    println!();

    let report = pipeline::run(&cfg);
    report.print_summary();
}
