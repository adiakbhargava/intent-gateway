# Hardware Performance Counters

CPU performance counter data from `perf stat`.

## Regenerate (Linux only)

```bash
cd rust
cargo build -p pipeline --release

# Basic counters
perf stat -e cache-misses,cache-references,instructions,cycles \
  ./target/release/pipeline --packets 100000

# With CPU pinning (requires core_affinity crate integration)
taskset -c 0 ./target/release/pipeline --packets 100000
taskset -c 0,1 ./target/release/pipeline --packets 100000
```

## Note

`perf stat` requires Linux with a kernel supporting hardware performance
counters. On Windows, equivalent data can be collected with Intel VTune
or Windows Performance Analyzer (WPA):

```powershell
# Windows: use VTune CLI
vtune -collect hotspots -- target\release\pipeline.exe --packets 100000
```

## Expected Observations

- **Cache misses** should be low (<1%) with `#[repr(align(64))]` padding
- **IPC** (instructions per cycle) should be high (~3-4) for the hot loop
- **Same-core vs cross-core**: cross-core should show higher cache-miss rate
  due to L1/L2 coherence traffic
