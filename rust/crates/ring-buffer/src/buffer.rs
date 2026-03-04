//! Lock-free Single-Producer Single-Consumer (SPSC) ring buffer.
//!
//! # Design
//!
//! The buffer is split into a `Producer` and `Consumer` half, enforcing
//! the SPSC invariant at compile time — the producer cannot pop, the
//! consumer cannot push, and neither can be cloned across threads.
//!
//! # Memory ordering
//!
//! Each method documents its ordering choices. The short version:
//! - Own-index loads: `Relaxed` (only we write our own index)
//! - Cross-index loads: `Acquire` (we need to see the other side's writes)
//! - Own-index stores: `Release` (make our buffer writes visible before the index update)
//!
//! # Zero allocations
//!
//! After `new()` returns, every push and pop is allocation-free. The only
//! heap allocation is the initial slot array inside `Inner::new()`.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Cache-line padding
// ---------------------------------------------------------------------------

/// Pads `T` to a full 64-byte cache line to prevent false sharing.
///
/// Without this, the producer's `head` index and the consumer's `tail`
/// index would likely land on the same cache line. When one core writes
/// to any byte in a cache line, every other core holding that line gets
/// an invalidation signal — even if they're touching different bytes.
/// This is false sharing, and it serialises two threads that should be
/// independent. Padding ensures head and tail are on separate lines.
#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

// ---------------------------------------------------------------------------
// Inner shared state
// ---------------------------------------------------------------------------

/// The shared state behind both `Producer` and `Consumer`.
///
/// Both halves hold an `Arc<Inner<T>>`. The reference count keeps the
/// buffer alive for as long as either half exists.
struct Inner<T> {
    /// Heap-allocated slot array.  This is the ONE heap allocation.
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    /// Number of slots. Always a power of two so we can use bitmask indexing.
    capacity: usize,
    /// `capacity - 1`. `index & mask` is equivalent to `index % capacity`
    /// but takes one cycle instead of 20-40.
    mask: usize,
    /// Producer's write position — only ever written by the producer.
    head: CachePadded<AtomicUsize>,
    /// Consumer's read position — only ever written by the consumer.
    tail: CachePadded<AtomicUsize>,
}

// SAFETY: `Inner<T>` is safe to send and share between threads because:
// - `head` is only written by the producer thread
// - `tail` is only written by the consumer thread
// - Acquire/Release ordering ensures buffer slot accesses are correctly
//   sequenced relative to index updates
// - `T: Send` ensures the values themselves are safe to transfer between threads
unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
    fn new(min_capacity: usize) -> Self {
        // Round up to the next power of two, minimum 2.
        // Minimum 2: a capacity-1 ring buffer is degenerate (always "full").
        // next_power_of_two(0) panics, next_power_of_two(1) == 1 (too small).
        let capacity = min_capacity.next_power_of_two().max(2);
        let mask = capacity - 1;

        // Allocate `capacity` uninitialized slots.
        // This is the ONE heap allocation the ring buffer ever makes.
        let buffer: Vec<UnsafeCell<MaybeUninit<T>>> = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect();

        Inner {
            // into_boxed_slice() drops the Vec's hidden `capacity` field
            // (the growth capacity, not our slot count). The buffer is
            // fixed-size, so we don't need Vec's reallocation machinery.
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask,
            head: CachePadded { value: AtomicUsize::new(0) },
            tail: CachePadded { value: AtomicUsize::new(0) },
        }
    }
}

/// Drop any values still in the buffer when the last reference is released.
///
/// Without this, dropping a `RingBuffer<String>` with items in it would
/// leak the strings' heap memory. We iterate the live range [tail, head)
/// and call `assume_init_drop()` on each slot.
impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        // `get_mut()` gives a plain `&mut usize` — no atomic overhead.
        // Safe because we have `&mut self`, so no other thread can exist.
        let head = *self.head.value.get_mut();
        let tail = *self.tail.value.get_mut();

        for i in tail..head {
            let slot = i & self.mask;
            // SAFETY: slots in [tail, head) were written by the producer
            // and not yet consumed. We have exclusive access (no Arc clones
            // remain — this is the Drop impl, called when Arc count hits 0).
            unsafe {
                (*self.buffer[slot].get()).assume_init_drop();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The producing half of a ring buffer. Can only push values.
pub struct Producer<T> {
    inner: Arc<Inner<T>>,
}

/// The consuming half of a ring buffer. Can only pop values.
pub struct Consumer<T> {
    inner: Arc<Inner<T>>,
}

/// Create a new SPSC ring buffer with at least `capacity` slots.
///
/// The actual capacity is rounded up to the next power of two.
/// Returns a `(Producer, Consumer)` pair that can be sent to separate threads.
///
/// # Example
/// ```rust
/// let (producer, consumer) = ring_buffer::new::<u32>(1024);
/// ```
pub fn new<T>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let inner = Arc::new(Inner::new(capacity));
    (
        Producer { inner: Arc::clone(&inner) },
        Consumer { inner },
    )
}

// ---------------------------------------------------------------------------
// Producer implementation
// ---------------------------------------------------------------------------

impl<T> Producer<T> {
    /// Push a single value into the ring buffer.
    ///
    /// Returns `Ok(())` on success or `Err(value)` if the buffer is full.
    /// The caller decides the backpressure strategy: spin, drop, or yield.
    ///
    /// # Memory ordering
    /// - `head` load: `Relaxed` — only we write head, so we always see our own value.
    /// - `tail` load: `Acquire` — we need to see the consumer's latest tail update
    ///   *and* all reads the consumer performed before that update. Without Acquire,
    ///   the CPU could let us see `tail` updated before the consumer finishes reading
    ///   the slot at `tail - 1`, causing us to overwrite a slot still being read.
    /// - `head` store: `Release` — ensures the buffer write above this store is
    ///   committed before the consumer can see the updated head. Without Release,
    ///   the consumer might read the new head and see uninitialized data in the slot.
    pub fn try_push(&self, value: T) -> Result<(), T> {
        // Relaxed: only we write head. No synchronization needed for our own writes.
        let head = self.inner.head.value.load(Ordering::Relaxed);
        // Acquire: synchronize with the consumer's Release store of tail.
        let tail = self.inner.tail.value.load(Ordering::Acquire);

        // Buffer is full when we're exactly `capacity` positions ahead of the consumer.
        // wrapping_sub handles the case where head has wrapped past usize::MAX.
        if head.wrapping_sub(tail) >= self.inner.capacity {
            return Err(value);
        }

        let slot = head & self.inner.mask;
        // SAFETY:
        // - The full check above guarantees this slot is not currently held by the consumer.
        // - We are the only writer (SPSC invariant).
        // - UnsafeCell allows mutation through the shared Arc reference.
        // - MaybeUninit::write does not read the old value (no drop), which is correct
        //   because the slot may be uninitialized.
        unsafe {
            (*self.inner.buffer[slot].get()).write(value);
        }

        // Release: make the buffer write above visible to the consumer before they
        // can see this updated head. The consumer's Acquire load of head synchronizes
        // with this Release store.
        self.inner.head.value.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Returns the number of slots in the ring buffer.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Returns the approximate number of items currently in the buffer.
    ///
    /// This is a snapshot — the value may be stale by the time the caller uses it.
    /// Both loads are `Relaxed` because this is purely informational.
    pub fn len(&self) -> usize {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the buffer is currently empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the buffer is currently full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }
}

impl<T: Copy> Producer<T> {
    /// Push a batch of values into the ring buffer.
    ///
    /// Returns the number of values actually pushed (≤ `values.len()`).
    /// Uses a single Acquire load and single Release store for the entire
    /// batch, making per-element atomic cost O(1/batch_size).
    ///
    /// Requires `T: Copy` because each element is copied from the slice.
    pub fn push_slice(&self, values: &[T]) -> usize {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Acquire);

        let available = self.inner.capacity - head.wrapping_sub(tail);
        let count = values.len().min(available);

        for (i, value) in values[..count].iter().enumerate() {
            let slot = (head + i) & self.inner.mask;
            // SAFETY: same as try_push. Each slot in [head, head+count) is free.
            unsafe {
                (*self.inner.buffer[slot].get()).write(*value);
            }
        }

        // Single Release store covers all `count` writes above.
        self.inner.head.value.store(head.wrapping_add(count), Ordering::Release);
        count
    }
}

// ---------------------------------------------------------------------------
// Consumer implementation
// ---------------------------------------------------------------------------

impl<T> Consumer<T> {
    /// Pop a single value from the ring buffer.
    ///
    /// Returns `Some(value)` or `None` if the buffer is empty.
    ///
    /// # Memory ordering
    /// Mirror of `try_push`:
    /// - `tail` load: `Relaxed` — only we write tail.
    /// - `head` load: `Acquire` — synchronize with the producer's Release store,
    ///   ensuring we see the data written before the producer updated head.
    /// - `tail` store: `Release` — tell the producer the slot is free, ensuring
    ///   our read of the slot is fully visible before the producer can reuse it.
    pub fn try_pop(&self) -> Option<T> {
        // Relaxed: only we write tail.
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        // Acquire: synchronize with the producer's Release store of head.
        let head = self.inner.head.value.load(Ordering::Acquire);

        if tail == head {
            return None;
        }

        let slot = tail & self.inner.mask;
        // SAFETY:
        // - tail != head means the slot was written by the producer.
        // - The Acquire load of head ensures we see the producer's write.
        // - assume_init_read() does a bitwise copy without dropping. The
        //   slot is not cleared — it will be overwritten on the next push.
        //   No double-drop risk because the consumer advances tail immediately
        //   and the SPSC protocol prevents re-reading the same slot.
        let value = unsafe { (*self.inner.buffer[slot].get()).assume_init_read() };

        // Release: signal to the producer that this slot is now free.
        self.inner.tail.value.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Returns the number of slots in the ring buffer.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Returns the approximate number of items currently in the buffer.
    pub fn len(&self) -> usize {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// Returns `true` if the buffer is currently empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Copy> Consumer<T> {
    /// Pop a batch of values from the ring buffer into `output`.
    ///
    /// Returns the number of values actually popped (≤ `output.len()`).
    /// Uses a single Acquire load and single Release store for the batch.
    pub fn pop_slice(&self, output: &mut [T]) -> usize {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Acquire);

        let available = head.wrapping_sub(tail);
        let count = output.len().min(available);

        for (i, dest) in output[..count].iter_mut().enumerate() {
            let slot = (tail + i) & self.inner.mask;
            // SAFETY: slots in [tail, tail+count) were written by the producer
            // and the Acquire load of head ensures their writes are visible.
            *dest = unsafe { (*self.inner.buffer[slot].get()).assume_init_read() };
        }

        self.inner.tail.value.store(tail.wrapping_add(count), Ordering::Release);
        count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Basic operation ────────────────────────────────────────────────────

    #[test]
    fn test_push_pop_basic() {
        let (producer, consumer) = new::<u32>(8);
        for i in 0..5u32 {
            assert!(producer.try_push(i).is_ok(), "push {i} failed");
        }
        for i in 0..5u32 {
            assert_eq!(consumer.try_pop(), Some(i), "expected {i}");
        }
        assert_eq!(consumer.try_pop(), None, "buffer should be empty");
    }

    // ── Boundary: full buffer ──────────────────────────────────────────────

    #[test]
    fn test_full_buffer() {
        let (producer, consumer) = new::<u32>(4);
        // All 4 slots should be fillable (wrapping_sub scheme uses the full capacity)
        for i in 0..4u32 {
            assert!(producer.try_push(i).is_ok(), "slot {i} should be free");
        }
        // Now full
        assert!(producer.try_push(99).is_err(), "should be full");
        // Pop one, then push should succeed
        assert_eq!(consumer.try_pop(), Some(0));
        assert!(producer.try_push(99).is_ok(), "slot freed after pop");
    }

    // ── Boundary: empty buffer ─────────────────────────────────────────────

    #[test]
    fn test_empty_pop() {
        let (_producer, consumer) = new::<u32>(8);
        assert_eq!(consumer.try_pop(), None);
    }

    // ── Constructor: power-of-two rounding ────────────────────────────────

    #[test]
    fn test_power_of_two_rounding() {
        let (p, _) = new::<u32>(100);
        assert_eq!(p.capacity(), 128);

        let (p, _) = new::<u32>(1);
        assert_eq!(p.capacity(), 2);

        let (p, _) = new::<u32>(8);
        assert_eq!(p.capacity(), 8); // exact power of two unchanged
    }

    // ── Wraparound: index arithmetic across usize wraps ───────────────────

    #[test]
    fn test_wraparound() {
        let (producer, consumer) = new::<u64>(4);
        for i in 0..1000u64 {
            while producer.try_push(i).is_err() {}
            assert_eq!(consumer.try_pop(), Some(i));
        }
    }

    // ── Batch: happy path ─────────────────────────────────────────────────

    #[test]
    fn test_batch_push_pop() {
        let (producer, consumer) = new::<u32>(16);
        let input: Vec<u32> = (0..10).collect();
        let pushed = producer.push_slice(&input);
        assert_eq!(pushed, 10);

        let mut output = vec![0u32; 10];
        let popped = consumer.pop_slice(&mut output);
        assert_eq!(popped, 10);
        assert_eq!(output, input);
    }

    // ── Batch: partial fill ───────────────────────────────────────────────

    #[test]
    fn test_batch_partial() {
        let (producer, consumer) = new::<u32>(4);
        // Try to push 10 but only 4 slots available
        let input: Vec<u32> = (0..10).collect();
        let pushed = producer.push_slice(&input);
        assert_eq!(pushed, 4, "should only push 4 (capacity)");

        // Try to pop 10 but only 4 available
        let mut output = vec![0u32; 10];
        let popped = consumer.pop_slice(&mut output);
        assert_eq!(popped, 4, "should only pop 4");
        assert_eq!(&output[..4], &[0u32, 1, 2, 3]);
    }

    // ── Concurrent correctness ────────────────────────────────────────────

    #[test]
    fn test_concurrent_correctness() {
        use std::thread;

        const N: u64 = 1_000_000;
        let (producer, consumer) = new::<u64>(1024);

        let producer_thread = thread::spawn(move || {
            for i in 0..N {
                // Spin until space is available (backpressure via busy-wait)
                while producer.try_push(i).is_err() {
                    std::hint::spin_loop();
                }
            }
        });

        let consumer_thread = thread::spawn(move || {
            let mut received = Vec::with_capacity(N as usize);
            while received.len() < N as usize {
                match consumer.try_pop() {
                    Some(v) => received.push(v),
                    None => std::hint::spin_loop(),
                }
            }
            received
        });

        producer_thread.join().expect("producer panicked");
        let received = consumer_thread.join().expect("consumer panicked");

        assert_eq!(received.len(), N as usize);
        for (i, &v) in received.iter().enumerate() {
            assert_eq!(v, i as u64, "value mismatch at position {i}");
        }
    }

    // ── Stress test: 10M messages ─────────────────────────────────────────
    // Marked #[ignore] so it doesn't slow down the normal test suite.
    // Run with: cargo test -p ring-buffer -- --ignored

    #[test]
    #[ignore]
    fn test_stress_10m() {
        use std::thread;

        const N: u64 = 10_000_000;
        let (producer, consumer) = new::<u64>(4096);

        let producer_thread = thread::spawn(move || {
            for i in 0..N {
                while producer.try_push(i).is_err() {
                    std::hint::spin_loop();
                }
            }
        });

        let consumer_thread = thread::spawn(move || {
            let mut count = 0u64;
            let mut expected = 0u64;
            while count < N {
                match consumer.try_pop() {
                    Some(v) => {
                        assert_eq!(v, expected, "corruption at message {count}");
                        expected += 1;
                        count += 1;
                    }
                    None => std::hint::spin_loop(),
                }
            }
        });

        producer_thread.join().expect("producer panicked");
        consumer_thread.join().expect("consumer panicked");
    }

    // ── Utility methods ───────────────────────────────────────────────────

    #[test]
    fn test_len_and_empty() {
        let (producer, consumer) = new::<u32>(8);
        assert!(consumer.is_empty());
        producer.try_push(1).unwrap();
        producer.try_push(2).unwrap();
        assert_eq!(producer.len(), 2);
        assert_eq!(consumer.len(), 2);
        consumer.try_pop();
        assert_eq!(consumer.len(), 1);
    }

    #[test]
    fn test_is_full() {
        let (producer, _consumer) = new::<u32>(4);
        assert!(!producer.is_full());
        for i in 0..4 {
            producer.try_push(i).unwrap();
        }
        assert!(producer.is_full());
    }
}
