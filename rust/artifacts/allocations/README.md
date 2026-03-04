# Zero-Allocation Proof

Verifies that the ring buffer hot path (push/pop/batch/utility methods) makes
exactly zero heap allocations after the initial slot-array construction.

## Regenerate

```bash
cd rust
cargo test -p ring-buffer --test zero_alloc -- --nocapture
```

## Method

`CountingAllocator` is installed as `#[global_allocator]` in the integration
test binary. After `ring_buffer::new()` returns, `reset_counters()` zeros the
count. The hot-path loop then runs 10 000 push/pop cycles, 100 batch rounds,
and all utility methods. The test asserts `allocation_count() == 0`.

## Result

```
test ring_buffer_hot_path_makes_zero_allocations ... ok
```

Three sub-cases verified:
1. `try_push` / `try_pop` over 10 000 iterations: **0 allocations**
2. `push_slice` / `pop_slice` over 100 batch rounds: **0 allocations**
3. `capacity()`, `len()`, `is_full()`, `is_empty()`: **0 allocations**
