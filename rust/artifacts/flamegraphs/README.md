# Flamegraphs

CPU flamegraph SVGs showing where time is spent in the pipeline.

## Regenerate (Linux/macOS only)

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Generate (requires perf on Linux or dtrace on macOS)
cd rust
cargo flamegraph -p pipeline -- --packets 100000
mv flamegraph.svg artifacts/flamegraphs/pipeline.svg
```

## Note

`cargo flamegraph` requires:
- **Linux**: `perf` (install via `linux-tools-generic`)
- **macOS**: `dtrace` (built in, but requires SIP disabled for profiling)
- **Windows**: Not natively supported. Use:
  - [Windows Performance Recorder](https://learn.microsoft.com/en-us/windows-hardware/test/wpt/) + [FlameGraph scripts](https://github.com/brendangregg/FlameGraph)
  - Or Intel VTune with its built-in flame graph view

## Expected Hot Spots

1. `ring_buffer::buffer::Producer::try_push` (atomic stores)
2. `ring_buffer::buffer::Consumer::try_pop` (atomic loads)
3. `transcoder::Decoder::decode_varint` (inner loop)
4. `std::hint::spin_loop` (backpressure wait)
