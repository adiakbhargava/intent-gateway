# Benchmark Artifacts

Criterion HTML reports for all benchmark suites.

## Regenerate

```bash
cd rust
cargo bench -p ring-buffer
cargo bench -p transcoder
cargo bench -p statistics
```

Reports land in `target/criterion/`. Copy them here:
```bash
cp -r target/criterion/* artifacts/benchmarks/
```

## Key Results (measured on this machine)

| Benchmark | Value |
|-----------|-------|
| single_push_pop_latency | ~2.9 ns |
| throughput/256 | ~329 Melem/s |
| throughput/4096 | ~335 Melem/s |
| vs_crossbeam ring_buffer | ~4.6 ns |
| vs_crossbeam crossbeam | ~35.6 ns |
| ring_buffer advantage | **7.7x faster** |
| decode_typical (64ch x 500) | ~65 ns |
| decode_1k_varints | measured |
