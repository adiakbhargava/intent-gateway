//! Integration test: prove the ring buffer hot path makes zero heap allocations.
//!
//! # Why a single test function?
//!
//! The `CountingAllocator` is a global counter shared by every thread in the
//! process. Rust runs tests concurrently by default; if three separate
//! `#[test]` functions are scheduled at the same time, allocations from one
//! bleed into another's measurement window and produce false failures.
//!
//! Putting all cases in one `#[test]` function means this is the only test
//! running in the binary, so the measurement windows are clean.
//!
//! # Measurement pattern
//!
//! 1. Call `ring_buffer::new()` — this does ONE allocation (the slot array).
//! 2. Call `reset_counters()` — zero the baseline after construction.
//! 3. Run the hot path.
//! 4. Assert `allocation_count() == 0`.

use allocator::CountingAllocator;

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

#[test]
fn ring_buffer_hot_path_makes_zero_allocations() {
    // ── Case 1: single-item try_push / try_pop ────────────────────────────
    {
        let (producer, consumer) = ring_buffer::new::<u64>(1024);
        // Reset AFTER new() so the one-time slot-array allocation is excluded.
        allocator::reset_counters();

        for i in 0..10_000u64 {
            while producer.try_push(i).is_err() {
                std::hint::spin_loop();
            }
            let _ = consumer.try_pop();
        }

        assert_eq!(
            allocator::allocation_count(),
            0,
            "try_push / try_pop must not heap-allocate"
        );
    }

    // ── Case 2: batch push_slice / pop_slice ──────────────────────────────
    {
        let (producer, consumer) = ring_buffer::new::<u32>(256);
        // Stack-allocated arrays — no heap involved on our side.
        let input = [1u32; 64];
        let mut output = [0u32; 64];

        // new() above allocated the slot array; reset before the hot path.
        allocator::reset_counters();

        for _ in 0..100 {
            producer.push_slice(&input);
            consumer.pop_slice(&mut output);
        }

        assert_eq!(
            allocator::allocation_count(),
            0,
            "push_slice / pop_slice must not heap-allocate"
        );
    }

    // ── Case 3: utility methods (capacity, len, is_empty, is_full) ────────
    {
        let (producer, consumer) = ring_buffer::new::<u8>(8);

        allocator::reset_counters();

        // All of these are atomic loads / simple arithmetic — no allocation.
        let _ = producer.capacity();
        let _ = producer.len();
        let _ = producer.is_full();
        let _ = consumer.capacity();
        let _ = consumer.len();
        let _ = consumer.is_empty();

        assert_eq!(
            allocator::allocation_count(),
            0,
            "utility methods must not heap-allocate"
        );
    }
}
