# Stress Test Results

10 million message concurrent correctness test.

## Regenerate

```bash
cd rust
cargo test -p ring-buffer --release -- --ignored --nocapture
```

## Result

```
running 1 test
test buffer::tests::test_stress_10m ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 10 filtered out;
             finished in 0.15s
```

- **10,000,000 messages** sent producer -> consumer
- Buffer capacity: 4096
- Every message verified in-order: `assert_eq!(v, expected)`
- **Zero data corruption**
- Completed in **0.15 seconds** (release mode)
- Effective throughput: ~66.7M messages/second
