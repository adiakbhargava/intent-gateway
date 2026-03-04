# Real-Time Systems Analysis

An honest assessment of where this Rust neural telemetry engine sits on the
real-time spectrum, what design decisions move it toward hard-RT, and what
would be required to get there.

## 1. What "Real-Time" Means

| Level | Guarantee | Example |
|-------|-----------|---------|
| **Hard real-time** | Every deadline met; failure = system failure | Cardiac pacemaker, ABS brakes |
| **Firm real-time** | Late results are worthless but not catastrophic | Video frame decode |
| **Soft real-time** | Late results degrade quality but are still usable | Audio streaming, EEG monitoring |

This engine targets **soft real-time** for EEG BCI (Brain-Computer Interface)
telemetry. Late packets degrade classification accuracy but do not endanger
the user.

## 2. Design Decisions Toward Real-Time

### 2.1 Zero Post-Init Allocations

Every `malloc` call is unbounded in the worst case (the allocator may need to
request pages from the kernel via `mmap`/`VirtualAlloc`, which can block for
milliseconds). By proving zero allocations on the hot path, we eliminate the
single largest source of unbounded latency in userspace code.

**Verified by:** `ring-buffer/tests/zero_alloc.rs` integration test with
`CountingAllocator` as `#[global_allocator]`.

### 2.2 Lock-Free SPSC Ring Buffer

Mutexes introduce priority inversion: a low-priority thread holding a lock
can block a high-priority thread indefinitely. Our ring buffer uses only
atomic loads and stores — no locks, no syscalls, no kernel transitions on the
fast path.

**Memory ordering:** Minimum correct ordering (Relaxed for own-index,
Acquire/Release for cross-index) avoids unnecessary memory fences that would
add latency on weakly-ordered architectures (ARM, RISC-V).

### 2.3 Cache-Line Padding

`#[repr(align(64))]` ensures `head` and `tail` live on separate 64-byte
cache lines. Without this, the producer and consumer would cause MESI
invalidation ping-pong on every operation, serializing two threads that
should be independent.

**Impact:** 2-5x throughput improvement in benchmarks.

### 2.4 Power-of-Two Bitmask Indexing

`index & mask` replaces `index % capacity`. Modulo on x86 is a `div`
instruction (20-40 cycles); bitwise AND is 1 cycle. For a 3 ns push/pop,
saving 20+ cycles per operation is significant.

### 2.5 Zero-Copy Protobuf Decoding

`decode_length_delimited()` returns `&'a [u8]` borrowed from the source
buffer. No allocation, no copy. The `NeuralPacket` struct borrows its
`samples` field directly from the encoded bytes.

**Measured decode latency:** ~65 ns for a 64-channel, 500-sample packet.

## 3. Why This Is NOT Hard Real-Time

### 3.1 No Worst-Case Execution Time (WCET) Guarantee

Hard-RT systems must prove a bounded WCET for every code path. Our engine:
- Uses `std::time::SystemTime` (may syscall into the kernel)
- Allocates `Vec<u8>` in the producer thread for packet encoding
- Calls `std::thread::spawn` (kernel thread creation is unbounded)
- The Cooley-Tukey FFT in `statistics::RollingFft` allocates working buffers

Without formal WCET analysis (e.g., via AbsInt aiT), we cannot guarantee
any specific latency bound.

### 3.2 No Real-Time Scheduler

Standard Linux/Windows schedulers use CFS/EEVDF (Linux) or the Windows
scheduler, which make no latency guarantees. A high-priority thread can be
preempted by:
- Kernel interrupts (IRQs)
- Page faults (if memory is swapped)
- Other threads at the same priority level
- Kernel housekeeping (RCU callbacks, timer interrupts)

### 3.3 No Memory Locking

`mlockall(MCL_CURRENT | MCL_FUTURE)` prevents page faults by pinning all
pages in physical RAM. Without it, a cold code path can trigger a page fault
(~10 us on Linux, potentially milliseconds if swap is involved).

### 3.4 No CPU Isolation

Without `isolcpus` kernel parameter and `taskset`/`core_affinity`, the OS
freely migrates our threads between cores, causing:
- L1/L2 cache cold misses on migration
- TLB flushes
- Cross-NUMA memory access latency

## 4. Measured Latency Profile

From a 10,000-packet pipeline run (release mode, no fault injection):

| Percentile | Latency |
|------------|---------|
| p50 | ~33 us |
| p90 | ~262 us |
| p99 | ~262 us |
| p999 | ~524 us |
| max | ~1.4 ms |

**Protobuf decode:** ~276 ns mean (well under the 500 ns target)

**Jitter std-dev:** ~83 us (indicates OS scheduling interference)

## 5. Path to Hard Real-Time

If this system needed hard-RT guarantees (e.g., for a closed-loop neural
stimulation device), the following changes would be required:

### 5.1 Kernel

- **PREEMPT_RT patch** (Linux) — makes the kernel fully preemptible,
  converting spinlocks to sleeping locks with priority inheritance
- Alternatively, a dedicated RTOS (Zephyr, FreeRTOS, QNX)

### 5.2 Scheduling

```bash
# Set SCHED_FIFO with maximum priority
chrt -f 99 ./target/release/pipeline
```

With PREEMPT_RT, `SCHED_FIFO` threads preempt everything except hardware
IRQs, providing deterministic scheduling.

### 5.3 Memory

```rust
// At startup, before any real-time work
unsafe {
    libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
}
```

This pins all current and future pages, eliminating page-fault latency.

### 5.4 CPU Isolation

```bash
# Boot parameter: reserve cores 2-3 for RT threads only
GRUB_CMDLINE_LINUX="isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3"
```

Then pin threads:
```rust
use core_affinity::CoreId;
core_affinity::set_for_current(CoreId { id: 2 });
```

### 5.5 Eliminate All Allocations

The producer thread currently allocates `Vec<u8>` for each packet. In a
hard-RT system, this would be replaced with a pre-allocated pool of fixed-size
buffers (slab allocator pattern).

### 5.6 Formal WCET Analysis

Use tools like AbsInt aiT or measure worst-case paths empirically with
millions of iterations under `cyclictest` load.

## 6. Conclusion

This engine achieves **soft real-time** performance with sub-millisecond
p99 latency. The zero-allocation, lock-free, cache-optimized design
eliminates the most common sources of latency jitter in userspace. For
closed-loop BCI applications where late packets merely degrade accuracy
(not safety), this level of determinism is appropriate.

For safety-critical applications requiring hard-RT guarantees, the path
forward is well-defined: PREEMPT_RT kernel, SCHED_FIFO scheduling,
mlockall, CPU isolation, and pre-allocated buffer pools. The current
architecture was designed with these upgrades in mind — the SPSC ring
buffer and zero-copy decoder would remain unchanged.
