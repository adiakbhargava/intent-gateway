# Latency Histogram Data

End-to-end pipeline latency distributions.

## Regenerate

```bash
cd rust
cargo run -p pipeline --release -- --packets 100000
cargo run -p pipeline --release -- --packets 50000 --drop-rate 0.01 --corrupt-rate 0.005
```

## Sample Run (10 000 packets, 1% drop, 0.5% corrupt)

```
Packets sent:     9915
Packets received: 9915  (loss 0.00% of sent)
Dropped:          85    (by fault injector)
Corrupted:        51
Decode errors:    0

Latency (push-to-pop round-trip):
  p50:     32,768 ns    (~33 us)
  p90:    262,144 ns    (~262 us)
  p99:    262,144 ns    (~262 us)
  p999:   524,288 ns    (~524 us)
  mean:   121,047 ns    (~121 us)
  min:     42,100 ns    (~42 us)
  max:  1,375,900 ns    (~1.4 ms)

Jitter std-dev:  119,693 ns
Mean interval:   125,687 ns
Decode time:     424 ns mean (497 ns std)
```
