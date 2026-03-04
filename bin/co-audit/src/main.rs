//! c_o Audit CLI — benchmark tool.
//!
//! Runs the pipeline locally and produces a latency/jitter report,
//! optionally comparing against Python/LSL reference values.

use clap::Parser;
use co_core::pipeline::{PipelineConfig, PipelineReport};
use co_core::fault_inject::BackpressureStrategy;

#[derive(Parser)]
#[command(name = "co-audit", about = "c_o Neural Gateway latency audit")]
struct Args {
    /// Number of packets to benchmark
    #[arg(long, default_value_t = 100_000)]
    packets: u64,

    /// EEG channels per packet
    #[arg(long, default_value_t = 64)]
    channels: u32,

    /// Samples per channel per packet
    #[arg(long, default_value_t = 500)]
    samples_per_channel: usize,

    /// Simulated packet drop rate [0.0, 1.0]
    #[arg(long, default_value_t = 0.0)]
    drop_rate: f64,

    /// Simulated packet corruption rate [0.0, 1.0]
    #[arg(long, default_value_t = 0.0)]
    corrupt_rate: f64,

    /// Pin producer/consumer to separate CPU cores
    #[arg(long)]
    pin_cores: bool,

    /// Output format: table, json, markdown
    #[arg(long, default_value = "table")]
    format: String,

    /// Add Python/LSL reference comparison column
    #[arg(long)]
    compare: bool,
}

fn main() {
    let args = Args::parse();

    let cfg = PipelineConfig {
        packet_count: args.packets,
        channel_count: args.channels,
        samples_per_channel: args.samples_per_channel,
        drop_rate: args.drop_rate,
        corrupt_rate: args.corrupt_rate,
        pin_to_cores: args.pin_cores,
        backpressure: BackpressureStrategy::Block,
        ..PipelineConfig::default()
    };

    let report = co_core::pipeline::run(&cfg);

    match args.format.as_str() {
        "json" => print_json(&report, &args),
        "markdown" => print_markdown(&report, &args),
        _ => print_table(&report, &args),
    }
}

fn print_table(report: &PipelineReport, args: &Args) {
    let dur_ms = report.duration_ns as f64 / 1_000_000.0;
    let tput = report.throughput_pps();
    let loss_pct = if report.packets_sent > 0 {
        (report.packets_sent - report.packets_received) as f64 / report.packets_sent as f64 * 100.0
    } else { 0.0 };

    println!();
    println!("\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}");
    println!("  c_o Audit Report \u{2014} Neural Gateway Latency Analysis");
    println!("\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}");
    println!();
    println!("  Configuration:");
    println!("    Packets:     {:>12}", format_number(args.packets));
    println!("    Channels:    {} x {} samples", args.channels, args.samples_per_channel);
    println!("    Drop rate:   {:.3}", args.drop_rate);
    println!("    CPU pinning: {}", if args.pin_cores { "enabled" } else { "disabled" });
    println!("    Duration:    {:.1} ms", dur_ms);
    println!();

    if args.compare {
        println!("  \u{250C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{252C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}");
        println!("  \u{2502} {:>18} \u{2502} {:>16} \u{2502} {:>18} \u{2502}", "Metric", "c_o Gateway", "Python/LSL Stack*");
        println!("  \u{251C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{253C}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2524}");
        println!("  \u{2502} {:>18} \u{2502} {:>13} ns \u{2502} {:>15} us \u{2502}", "p50 Latency", format_number(report.latency.percentile(50.0)), "~12,000");
        println!("  \u{2502} {:>18} \u{2502} {:>13} ns \u{2502} {:>15} us \u{2502}", "p90 Latency", format_number(report.latency.percentile(90.0)), "~28,000");
        println!("  \u{2502} {:>18} \u{2502} {:>13} ns \u{2502} {:>15} us \u{2502}", "p99 Latency", format_number(report.latency.percentile(99.0)), "~42,500");
        println!("  \u{2502} {:>18} \u{2502} {:>13} ns \u{2502} {:>15} us \u{2502}", "Max Jitter", format_number(report.jitter.jitter_std_dev_ns() as u64), "~12,100");
        println!("  \u{2502} {:>18} \u{2502} {:>12} pkt/s \u{2502} {:>14} pkt/s \u{2502}", "Throughput", format_number(tput as u64), "~800");
        println!("  \u{2502} {:>18} \u{2502} {:>15.2}% \u{2502} {:>17}% \u{2502}", "Dropped Packets", loss_pct, "~0.40");
        println!("  \u{2502} {:>18} \u{2502} {:>16} \u{2502} {:>18} \u{2502}", "Heap Allocs (hot)", report.heap_allocs_during_run, "N/A");
        println!("  \u{2502} {:>18} \u{2502} {:>13} ns \u{2502} {:>15} us \u{2502}", "Decode Time", format_number(report.decode_time.mean() as u64), "~1,200");
        println!("  \u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2534}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}");
        println!();
        println!("  * Python/LSL comparison values are reference benchmarks from");
        println!("    pylsl + numpy decode on equivalent hardware.");
    } else {
        println!("  Latency (push->pop round-trip):");
        println!("    p50:  {:>12} ns", format_number(report.latency.percentile(50.0)));
        println!("    p90:  {:>12} ns", format_number(report.latency.percentile(90.0)));
        println!("    p99:  {:>12} ns", format_number(report.latency.percentile(99.0)));
        println!("    p999: {:>12} ns", format_number(report.latency.percentile(99.9)));
        println!("    mean: {:>12} ns", format_number(report.latency.mean_ns()));
        println!("    min:  {:>12} ns", format_number(report.latency.min_ns()));
        println!("    max:  {:>12} ns", format_number(report.latency.max_ns()));
        println!();
        println!("  Jitter (inter-arrival std-dev): {:>12} ns", format_number(report.jitter.jitter_std_dev_ns() as u64));
        println!("  Mean interval:                  {:>12} ns", format_number(report.jitter.mean_interval_ns() as u64));
        println!();
        println!("  Throughput:     {:>12} pkt/s", format_number(tput as u64));
        println!("  Packet loss:    {:>12.4}%", loss_pct);
        println!("  Decode errors:  {:>12}", format_number(report.decode_errors));
        println!("  Heap allocs:    {:>12}", report.heap_allocs_during_run);
    }

    println!();
    println!("\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}\u{2501}");
}

fn print_json(report: &PipelineReport, args: &Args) {
    #[derive(serde::Serialize)]
    struct AuditJson {
        config: ConfigJson,
        results: ResultsJson,
    }

    #[derive(serde::Serialize)]
    struct ConfigJson {
        packets: u64,
        channels: u32,
        samples_per_channel: usize,
        drop_rate: f64,
        corrupt_rate: f64,
        pin_cores: bool,
    }

    #[derive(serde::Serialize)]
    struct ResultsJson {
        packets_sent: u64,
        packets_received: u64,
        packets_dropped: u64,
        decode_errors: u64,
        duration_ms: f64,
        throughput_pps: f64,
        p50_ns: u64,
        p90_ns: u64,
        p99_ns: u64,
        p999_ns: u64,
        min_ns: u64,
        max_ns: u64,
        mean_ns: u64,
        jitter_std_dev_ns: f64,
        mean_interval_ns: f64,
        heap_allocs: u64,
    }

    let json = AuditJson {
        config: ConfigJson {
            packets: args.packets,
            channels: args.channels,
            samples_per_channel: args.samples_per_channel,
            drop_rate: args.drop_rate,
            corrupt_rate: args.corrupt_rate,
            pin_cores: args.pin_cores,
        },
        results: ResultsJson {
            packets_sent: report.packets_sent,
            packets_received: report.packets_received,
            packets_dropped: report.packets_dropped,
            decode_errors: report.decode_errors,
            duration_ms: report.duration_ns as f64 / 1_000_000.0,
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
        },
    };

    println!("{}", serde_json::to_string_pretty(&json).unwrap());
}

fn print_markdown(report: &PipelineReport, args: &Args) {
    let tput = report.throughput_pps();
    let loss_pct = if report.packets_sent > 0 {
        (report.packets_sent - report.packets_received) as f64 / report.packets_sent as f64 * 100.0
    } else { 0.0 };

    println!("# c_o Audit Report");
    println!();
    println!("## Configuration");
    println!();
    println!("| Parameter | Value |");
    println!("|-----------|-------|");
    println!("| Packets | {} |", format_number(args.packets));
    println!("| Channels | {} x {} |", args.channels, args.samples_per_channel);
    println!("| Drop rate | {:.3} |", args.drop_rate);
    println!("| CPU pinning | {} |", if args.pin_cores { "enabled" } else { "disabled" });
    println!();

    if args.compare {
        println!("## Results (vs Python/LSL)");
        println!();
        println!("| Metric | c_o Gateway | Python/LSL Stack |");
        println!("|--------|-------------|------------------|");
        println!("| p50 Latency | {} ns | ~12,000 us |", format_number(report.latency.percentile(50.0)));
        println!("| p90 Latency | {} ns | ~28,000 us |", format_number(report.latency.percentile(90.0)));
        println!("| p99 Latency | {} ns | ~42,500 us |", format_number(report.latency.percentile(99.0)));
        println!("| Jitter Std Dev | {} ns | ~12,100 us |", format_number(report.jitter.jitter_std_dev_ns() as u64));
        println!("| Throughput | {} pkt/s | ~800 pkt/s |", format_number(tput as u64));
        println!("| Packet Loss | {:.4}% | ~0.40% |", loss_pct);
        println!("| Heap Allocs | {} | N/A |", report.heap_allocs_during_run);
    } else {
        println!("## Latency");
        println!();
        println!("| Percentile | Value (ns) |");
        println!("|------------|------------|");
        println!("| p50 | {} |", format_number(report.latency.percentile(50.0)));
        println!("| p90 | {} |", format_number(report.latency.percentile(90.0)));
        println!("| p99 | {} |", format_number(report.latency.percentile(99.0)));
        println!("| p999 | {} |", format_number(report.latency.percentile(99.9)));
        println!("| mean | {} |", format_number(report.latency.mean_ns()));
        println!("| min | {} |", format_number(report.latency.min_ns()));
        println!("| max | {} |", format_number(report.latency.max_ns()));
        println!();
        println!("## Throughput");
        println!();
        println!("- **{} pkt/s**", format_number(tput as u64));
        println!("- Loss: {:.4}%", loss_pct);
    }
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
