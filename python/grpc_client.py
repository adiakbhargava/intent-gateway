#!/usr/bin/env python3
"""gRPC client for the co-gateway IntentStream service.

Connects to the gateway, subscribes to the FusedPacket stream, and
measures end-to-end latency from packet timestamp to receipt time.
Produces a summary report suitable for README documentation.

Usage:
    # Basic subscription (collect 100 packets)
    python python/grpc_client.py

    # Collect more packets with custom address
    python python/grpc_client.py --addr localhost:50051 --max-packets 500

    # Run audit benchmark
    python python/grpc_client.py --audit --packet-count 10000

    # JSON output for automated processing
    python python/grpc_client.py --json
"""

import argparse
import json
import sys
import time
from statistics import mean, median, stdev

try:
    import grpc
except ImportError:
    print("ERROR: pip install grpcio grpcio-tools", file=sys.stderr)
    sys.exit(1)


def generate_stubs():
    """Generate Python gRPC stubs from co.proto if not already present."""
    from pathlib import Path
    import subprocess

    proto_dir = Path(__file__).parent.parent / "proto"
    proto_file = proto_dir / "co.proto"
    out_dir = Path(__file__).parent

    pb2_file = out_dir / "co_pb2.py"
    pb2_grpc_file = out_dir / "co_pb2_grpc.py"

    if pb2_file.exists() and pb2_grpc_file.exists():
        return

    print(f"Generating gRPC stubs from {proto_file}...")
    subprocess.run([
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ], check=True)
    print("Stubs generated.")


def subscribe(addr: str, max_packets: int, timeout_s: float = 30.0):
    """Subscribe to FusedPacket stream and collect latency stats."""
    generate_stubs()

    # Import generated stubs
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import co_pb2
    import co_pb2_grpc

    channel = grpc.insecure_channel(addr)
    stub = co_pb2_grpc.IntentStreamStub(channel)

    request = co_pb2.SubscribeRequest(
        channel_count=128,
        sample_rate_hz=500.0,
        include_raw_eeg=False,
        include_features=True,
    )

    packets = []
    classifications = {"intent": 0, "observe": 0}
    confidences = []
    latencies_us = []

    print(f"Connecting to {addr}...")
    print(f"Collecting up to {max_packets} packets (timeout: {timeout_s}s)...")

    start_time = time.time()

    try:
        response_stream = stub.Subscribe(request, timeout=timeout_s)

        for i, packet in enumerate(response_stream):
            receive_ns = time.time_ns()

            # Compute transit latency (packet timestamp → receipt)
            if packet.timestamp_ns > 0:
                latency_ns = receive_ns - packet.timestamp_ns
                latencies_us.append(latency_ns / 1000.0)

            classification = packet.classification or "observe"
            classifications[classification] = classifications.get(classification, 0) + 1
            confidences.append(packet.confidence)

            packets.append({
                "seq": packet.sequence_id,
                "timestamp_ns": packet.timestamp_ns,
                "confidence": round(packet.confidence, 4),
                "classification": classification,
                "gaze_x": round(packet.gaze_x, 3),
                "gaze_y": round(packet.gaze_y, 3),
                "n_eeg_ch": len(packet.eeg_channels),
            })

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{max_packets}] "
                      f"rate={rate:.0f} pkt/s, "
                      f"last={classification} ({packet.confidence:.3f})")

            if i + 1 >= max_packets:
                break

    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            print(f"\nTimeout after {timeout_s}s — collected {len(packets)} packets")
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            print(f"\nERROR: Cannot connect to {addr}. Is the gateway running?")
            sys.exit(1)
        else:
            print(f"\ngRPC error: {e.code()} — {e.details()}")
            if not packets:
                sys.exit(1)

    elapsed = time.time() - start_time

    return {
        "packets_received": len(packets),
        "elapsed_s": round(elapsed, 3),
        "throughput_pps": round(len(packets) / elapsed, 1) if elapsed > 0 else 0,
        "classifications": classifications,
        "confidence_mean": round(mean(confidences), 4) if confidences else 0,
        "confidence_std": round(stdev(confidences), 4) if len(confidences) > 1 else 0,
        "latency_us": {
            "mean": round(mean(latencies_us), 1) if latencies_us else 0,
            "median": round(median(latencies_us), 1) if latencies_us else 0,
            "p95": round(sorted(latencies_us)[int(len(latencies_us) * 0.95)], 1)
                   if len(latencies_us) > 20 else 0,
            "p99": round(sorted(latencies_us)[int(len(latencies_us) * 0.99)], 1)
                   if len(latencies_us) > 100 else 0,
            "min": round(min(latencies_us), 1) if latencies_us else 0,
            "max": round(max(latencies_us), 1) if latencies_us else 0,
        },
    }


def run_audit(addr: str, packet_count: int, drop_rate: float, corrupt_rate: float):
    """Run the Audit RPC to get a latency benchmark report."""
    generate_stubs()

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    import co_pb2
    import co_pb2_grpc

    channel = grpc.insecure_channel(addr)
    stub = co_pb2_grpc.IntentStreamStub(channel)

    request = co_pb2.AuditRequest(
        packet_count=packet_count,
        drop_rate=drop_rate,
        corrupt_rate=corrupt_rate,
        pin_cores=False,
    )

    print(f"Running audit benchmark ({packet_count} packets)...")
    report = stub.Audit(request, timeout=60.0)

    return {
        "packets_sent": report.packets_sent,
        "packets_received": report.packets_received,
        "packets_dropped": report.packets_dropped,
        "decode_errors": report.decode_errors,
        "duration_ms": round(report.duration_ms, 2),
        "throughput_pps": round(report.throughput_pps, 0),
        "latency_ns": {
            "p50": report.p50_ns,
            "p90": report.p90_ns,
            "p99": report.p99_ns,
            "p999": report.p999_ns,
            "min": report.min_ns,
            "max": report.max_ns,
            "mean": report.mean_ns,
        },
        "jitter_std_dev_ns": round(report.jitter_std_dev_ns, 1),
        "heap_allocs": report.heap_allocs,
    }


def print_report(result: dict, mode: str):
    """Print a formatted report."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              INTENT GATEWAY — END-TO-END REPORT             ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    if mode == "subscribe":
        r = result
        print(f"║  Packets received:  {r['packets_received']:>8}                           ║")
        print(f"║  Elapsed:           {r['elapsed_s']:>8.1f} s                            ║")
        print(f"║  Throughput:        {r['throughput_pps']:>8.0f} packets/sec               ║")
        print(f"║                                                              ║")
        print(f"║  Classifications:                                            ║")
        for cls, count in r["classifications"].items():
            pct = count / r["packets_received"] * 100 if r["packets_received"] > 0 else 0
            print(f"║    {cls:>10}:     {count:>6} ({pct:>5.1f}%)                        ║")
        print(f"║                                                              ║")
        print(f"║  Confidence:        {r['confidence_mean']:.4f} ± {r['confidence_std']:.4f}                      ║")
        print(f"║                                                              ║")
        print(f"║  Transit Latency (packet → client):                          ║")
        lat = r["latency_us"]
        print(f"║    Mean:            {lat['mean']:>10.1f} µs                          ║")
        print(f"║    Median:          {lat['median']:>10.1f} µs                          ║")
        print(f"║    P95:             {lat['p95']:>10.1f} µs                          ║")
        print(f"║    P99:             {lat['p99']:>10.1f} µs                          ║")
        print(f"║    Min:             {lat['min']:>10.1f} µs                          ║")
        print(f"║    Max:             {lat['max']:>10.1f} µs                          ║")

    elif mode == "audit":
        r = result
        print(f"║  Packets sent:      {r['packets_sent']:>12}                       ║")
        print(f"║  Packets received:  {r['packets_received']:>12}                       ║")
        print(f"║  Packets dropped:   {r['packets_dropped']:>12}                       ║")
        print(f"║  Decode errors:     {r['decode_errors']:>12}                       ║")
        print(f"║  Duration:          {r['duration_ms']:>12.2f} ms                    ║")
        print(f"║  Throughput:        {r['throughput_pps']:>12.0f} pkt/s                ║")
        print(f"║  Heap allocations:  {r['heap_allocs']:>12}                       ║")
        print(f"║                                                              ║")
        print(f"║  Pipeline Latency (per-packet):                              ║")
        lat = r["latency_ns"]
        print(f"║    P50:             {lat['p50']:>10} ns                          ║")
        print(f"║    P90:             {lat['p90']:>10} ns                          ║")
        print(f"║    P99:             {lat['p99']:>10} ns                          ║")
        print(f"║    P999:            {lat['p999']:>10} ns                          ║")
        print(f"║    Min:             {lat['min']:>10} ns                          ║")
        print(f"║    Max:             {lat['max']:>10} ns                          ║")

    print("╚══════════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(
        description="gRPC client for intent-gateway IntentStream service")
    parser.add_argument(
        "--addr", default="[::1]:50051",
        help="Gateway gRPC address (default: [::1]:50051)")
    parser.add_argument(
        "--max-packets", type=int, default=500,
        help="Maximum packets to collect (default: 500)")
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Subscription timeout in seconds (default: 30)")
    parser.add_argument(
        "--audit", action="store_true",
        help="Run Audit RPC instead of Subscribe")
    parser.add_argument(
        "--packet-count", type=int, default=10000,
        help="Packets for audit benchmark (default: 10000)")
    parser.add_argument(
        "--drop-rate", type=float, default=0.0,
        help="Simulated drop rate for audit (default: 0.0)")
    parser.add_argument(
        "--corrupt-rate", type=float, default=0.0,
        help="Simulated corruption rate for audit (default: 0.0)")
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON")

    args = parser.parse_args()

    if args.audit:
        result = run_audit(args.addr, args.packet_count,
                          args.drop_rate, args.corrupt_rate)
        mode = "audit"
    else:
        result = subscribe(args.addr, args.max_packets, args.timeout)
        mode = "subscribe"

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, mode)


if __name__ == "__main__":
    main()
