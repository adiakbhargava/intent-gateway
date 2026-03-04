# Intent Gateway

**Real-time Rust gateway for multi-modal neural intent detection: EEG + gaze ingestion, timestamp fusion, ONNX inference, and gRPC streaming.**

```
EEG (128ch @ 256 Hz) ──┐
                        ├─ Timestamp Alignment ─ Feature Extraction ─ ONNX Inference ─ gRPC Stream
Gaze (x,y,pupil @ 60 Hz)┘   (nearest-neighbor)    (11-dim embedding)   (Conv1D fusion)   (fan-out)
```

Processes 65 subjects of real EEG+gaze data at **671K-1.8M frames/sec** (1,341x-14,025x real-time) on a single core with zero hot-path heap allocations.

> **Companion repository:** [spn-gaze-intent-research](https://github.com/REPLACE/spn-gaze-intent-research) — offline Python pipeline validating the scientific thesis (SPN+gaze fusion for intent decoding).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           intent-gateway                                │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ co-ingest│    │ co-fusion│    │ co-core  │    │co-infer- │          │
│  │          │    │          │    │          │    │  ence    │          │
│  │ EegSource├───▶│ Nearest- ├───▶│ FFT Band ├───▶│ ONNX RT  │          │
│  │ GazeSrc  │    │ Neighbor │    │ Powers   │    │ Conv1D   │          │
│  └──────────┘    │ Alignment│    │ 11-dim   │    └────┬─────┘          │
│                  └──────────┘    └──────────┘         │                │
│  ┌──────────┐                                         ▼                │
│  │ co-replay│    ┌──────────┐                   ┌──────────┐          │
│  │          │    │ co-core  │                   │ co-grpc  │          │
│  │ .corec   │    │          │                   │          │          │
│  │ reader   │    │ RingBuf  │                   │ Subscribe│          │
│  └──────────┘    │ Transcode│                   │ Embeddings          │
│                  │ Stats    │                   │ Audit    │          │
│                  │ FaultInj │                   └──────────┘          │
│                  └──────────┘                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Workspace Structure

```
intent-gateway/
├── proto/co.proto              # gRPC service definition
├── crates/
│   ├── co-core/                # Pipeline primitives (~2,100 LOC)
│   │   ├── ring_buffer.rs      #   Lock-free SPSC ring buffer (cache-line padded)
│   │   ├── transcoder.rs       #   Zero-allocation protobuf decoder
│   │   ├── statistics.rs       #   Welford, histograms, FFT, band powers
│   │   ├── embedding.rs        #   EEG feature extraction (11-dim)
│   │   ├── fault_inject.rs     #   Packet dropper/corruptor, backpressure
│   │   ├── allocator.rs        #   Counting allocator (zero-alloc proof)
│   │   └── pipeline.rs         #   End-to-end benchmark runner
│   ├── co-fusion/              # EEG/gaze timestamp alignment engine
│   ├── co-inference/           # ONNX Runtime wrapper + sliding window
│   ├── co-ingest/              # Hardware source traits + simulated adapters
│   ├── co-replay/              # Dataset replay (.corec binary format)
│   └── co-grpc/                # tonic 0.14 streaming server
├── bin/
│   ├── co-gateway/             # Main server binary
│   ├── co-audit/               # CLI benchmark tool
│   └── co-replay-bench/        # Replay benchmark (real data through full pipeline)
├── python/
│   ├── export_onnx.py          # Export Conv1DFusion model to ONNX
│   ├── validate_onnx.py        # Compare PyTorch vs ONNX Runtime outputs
│   ├── convert_dataset.py      # Convert EEGEyeNet/EEGET-ALS/EEGET-RSOD to .corec
│   ├── convert_openneuro.py    # Convert OpenNeuro BIDS .edf+physio to .corec
│   └── requirements.txt
├── rust/                       # Original Rust prototype (standalone crates)
│   ├── crates/                 #   allocator, ring-buffer, transcoder, statistics,
│   │                           #   embedding, fault-inject, pipeline
│   ├── artifacts/              #   Flamegraphs, benchmark data, fuzz results
│   └── docs/
│       └── real-time-analysis.md
└── run_benchmarks.py           # Batch benchmark runner for all .corec files
```

The `rust/` directory contains the original standalone prototype that preceded the workspace reorganization into `co-*` crates. It includes performance artifacts (flamegraphs, allocation proofs) and the real-time systems analysis document.

## Prerequisites

- **Rust nightly** (uses `is_multiple_of`)
- **[protoc](https://github.com/protocolbuffers/protobuf/releases)** on `PATH` (or set `PROTOC` env var) — required for `co-grpc`
- **ONNX Runtime** — pulled automatically via the `ort` crate
- **Python 3.10+** — for dataset conversion and ONNX export only

## Build & Test

```bash
# Full workspace (requires protoc)
cargo +nightly build --workspace
cargo +nightly test --workspace

# Without gRPC (no protoc required)
cargo +nightly check -p co-core -p co-fusion -p co-inference -p co-ingest -p co-replay -p co-audit -p co-replay-bench

# Prototype workspace (stable Rust)
cd rust && cargo test --workspace

# Lint
cargo +nightly clippy --workspace
```

## Test Results

### Main Workspace (co-* crates)

**74 passed, 0 failed** across 5 crates:

| Crate | Tests | Coverage |
|-------|-------|----------|
| co-core | 53 | Ring buffer, transcoder, statistics, embedding, fault injection, allocator |
| co-fusion | 7 | Timestamp alignment, degradation detection, tolerance |
| co-inference | 5 | Sliding window, model loading, inference pipeline |
| co-ingest | 4 | Source traits, simulated adapters |
| co-replay | 5 | .corec parsing, replay fidelity |
| co-grpc | N/A | Requires `protoc` at build time |

### Prototype Workspace (rust/)

**76 passed, 1 ignored, 0 failed** across 7 crates:

| Crate | Tests | Notes |
|-------|-------|-------|
| allocator | 3 | Counting allocator, reset, zero-alloc proof |
| ring-buffer | 12 | SPSC correctness, concurrency, wraparound (1 stress test ignored) |
| transcoder | 21 | Zero-copy decode, fuzz regressions, malformed input |
| statistics | 14 | FFT, band powers, Welford, jitter, histograms |
| embedding | 12 | Sliding window, cosine similarity, feature extraction |
| fault-inject | 14 | Drop/corrupt rates, backpressure strategies |

## Running the Server

```bash
# Start the gateway
cargo +nightly run --bin co-gateway -- --addr 0.0.0.0:50051 --channels 128 --sample-rate 256

# With a trained ONNX model
cargo +nightly run --bin co-gateway -- --model python/conv1d_fusion.onnx

# Run the internal benchmark
cargo +nightly run --bin co-audit -- --packets 100000 --channels 64
cargo +nightly run --bin co-audit -- --packets 100000 --format json
```

## Dataset Replay

Convert public EEG+eye-tracking datasets to `.corec` binary format, then replay through the full pipeline:

```bash
# Convert datasets
cd python && pip install -r requirements.txt
python convert_dataset.py als --input /path/to/participant/ --output als_p01.corec
python convert_dataset.py rsod --input /path/to/participant/ --output rsod_p01.corec
python convert_openneuro.py --edf data/eeg.edf --physio data/physio.tsv --output sub001.corec

# Replay benchmark
cargo +nightly run --release --bin co-replay-bench -- --input subject.corec
```

### Supported Datasets

| Dataset | Subjects | EEG | Eye Tracking | Source |
|---------|----------|-----|--------------|--------|
| [EEGEyeNet](https://openneuro.org/datasets/ds005872) | 1 (BIDS) | 129ch @ 500 Hz | 500 Hz co-registered | OpenNeuro |
| [EEGET-ALS](https://springernature.figshare.com/articles/dataset/EEGET-ALS_Dataset/24485689) | 26 (6 ALS + 20 healthy) | 32ch @ 128 Hz | 30 Hz Tobii | Figshare |
| [EEGET-RSOD](https://figshare.com/articles/dataset/EEGET-RSOD/26943565) | 38 | 32ch @ 500 Hz | 250 Hz SMI RED250 | Figshare |

## ONNX Model Export

```bash
cd python
pip install -r requirements.txt
python export_onnx.py                         # Writes conv1d_fusion.onnx
python validate_onnx.py conv1d_fusion.onnx    # Validates against PyTorch
```

## Performance

Benchmarked on **65 subjects across 3 public datasets** (109.5 minutes of real neural recordings). Single core, single-threaded, no GPU.

| Dataset | Subjects | Throughput | Real-Time Factor |
|---------|----------|-----------|------------------|
| EEGET-RSOD | 38 | 948K-1,589K fps | 1,897x-3,178x |
| EEGET-ALS | 26 | 883K-1,795K fps | 6,900x-14,025x |
| EEGEyeNet | 1 | 671K fps | 1,341x |

### Latency Breakdown

| Stage | p50 | p99 |
|-------|-----|-----|
| Protobuf decode | ~165 ns | - |
| Timestamp fusion | sub-us | - |
| Feature extraction (11-dim) | 512-2,048 ns | - |
| Full pipeline decode | ~16 us | ~262 us |
| Jitter (std-dev) | ~23 us | - |

### Real-Time Guarantees

- **Zero hot-path heap allocations** — verified by `CountingAllocator` integration test
- **Lock-free ring buffer** — atomic loads/stores only, no mutexes, no syscalls, no priority inversion
- **Zero-copy protobuf decode** — `NeuralPacket.samples` borrows directly from encoded buffer
- **Fuzz-tested decoder** — 584k+ cargo-fuzz iterations; found and fixed a usize overflow
- **Optional CPU affinity** — thread-to-core pinning reduces cache migration and scheduling jitter

See `rust/docs/real-time-analysis.md` for detailed analysis of where this sits on the real-time spectrum and the path to hard-RT guarantees.

## gRPC API

Defined in `proto/co.proto`:

| RPC | Type | Description |
|-----|------|-------------|
| `Subscribe` | server-streaming | Fused packets: EEG channels, gaze, features, confidence, classification |
| `StreamEmbeddings` | server-streaming | Raw 11-dim feature vectors for downstream models |
| `Audit` | unary | Run internal benchmark and return `AuditReport` |

### FusedPacket Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp_ns` | `uint64` | EEG-anchored nanosecond timestamp |
| `sequence_id` | `uint64` | Monotonic packet counter |
| `eeg_channels` | `repeated float` | Raw EEG channel values |
| `gaze_x`, `gaze_y` | `float` | Aligned gaze coordinates |
| `pupil_diameter` | `float` | Pupil diameter (mm) |
| `alignment_offset_ns` | `int64` | EEG-gaze temporal offset |
| `features` | `repeated float` | 11-dim: 5 relative + 5 absolute band powers + RMS |
| `confidence` | `float` | 0.0 (observe) to 1.0 (intent) |
| `classification` | `string` | `"intent"` or `"observe"` |

## Known Limitations

### co-grpc Build Dependency
The `co-grpc` crate requires `protoc` (the Protocol Buffers compiler) at build time. Without it, the full workspace fails to compile. All other crates build and test independently. Install from [protobuf releases](https://github.com/protocolbuffers/protobuf/releases) or via your package manager.

### EEG Robustness Assumptions
The pipeline currently processes raw EEG samples without preprocessing (no bandpass filter, no notch filter, no common average reference). Feature extraction operates on unfiltered data. For clinical-grade accuracy, preprocessing must be added either in-pipeline or upstream. The companion research repo's preprocessing (1-45 Hz bandpass, 50/60 Hz notch, CAR) has not yet been ported.

### Calibration Requirements
The ONNX inference model currently ships with random weights (proof-of-concept). A trained model from the research pipeline is required for meaningful intent classification. The `python/export_onnx.py` script exports the trained Conv1D fusion model, but training must be done in the research repo first.

### Streaming vs. Epoch Mismatch
This gateway processes continuous frames. The research pipeline validates intent detection using epoch-based (event-locked) analysis, specifically the SPN signal which requires a -500ms to 0ms pre-stimulus window. Bridging this gap requires an epoch segmentation layer that is not yet implemented. See the companion research repo's roadmap for the integration plan.

## Roadmap

### Real Hardware Validation
- **BrainFlow bridge** for `co-ingest` — supports 20+ EEG boards (OpenBCI, Muse, Neurosity) via C++ FFI
- **Tobii Stream Engine** or **Pupil Labs Neon** bridge for eye tracking
- **LSL ingress adapter** (`co-lsl`) for compatibility with lab streaming setups

### Production Observability
- **Structured logging** with `tracing` — replace `eprintln!` with JSON-structured log output
- **Graceful degradation metrics** — expose `co-fusion` alignment stats (degraded fraction, mean offset) via Prometheus or the Audit RPC
- **Replay recording** — `--record` flag for offline debugging and model retraining from production data

### Adaptive Thresholds
- **Confidence calibration** — temperature scaling or Platt scaling on model outputs
- **Abstention handling** — when confidence is below threshold, emit `"uncertain"` instead of forcing a classification
- **Online normalization** — per-session feature normalization to handle electrode impedance drift

### Epoch Segmentation (Research Integration)
- **`co-epoch` crate** — event-triggered windowing that segments continuous streams into epoch-based windows
- **`.npz` export** — write epochs in NumPy format compatible with the research pipeline's loaders
- This enables closed-loop validation: live data through the gateway, epoch extraction, research pipeline feature extraction and SPN validation

### Scale
- **Hot-swappable ONNX models** — watch directory for new `.onnx` files, reload without restart
- **WebSocket adapter** — browser-based UIs without gRPC client library
- **Edge deployment** — cross-compile for `aarch64` (RPi 5, Jetson), target <50MB RAM at 256 Hz
- **Multi-node fan-out** — NATS or Redis Streams for distributed consumption

## License

MIT
