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

#[repr(align(64))]
struct CachePadded<T> {
    value: T,
}

// ---------------------------------------------------------------------------
// Inner shared state
// ---------------------------------------------------------------------------

struct Inner<T> {
    buffer: Box<[UnsafeCell<MaybeUninit<T>>]>,
    capacity: usize,
    mask: usize,
    head: CachePadded<AtomicUsize>,
    tail: CachePadded<AtomicUsize>,
}

unsafe impl<T: Send> Send for Inner<T> {}
unsafe impl<T: Send> Sync for Inner<T> {}

impl<T> Inner<T> {
    fn new(min_capacity: usize) -> Self {
        let capacity = min_capacity.next_power_of_two().max(2);
        let mask = capacity - 1;

        let buffer: Vec<UnsafeCell<MaybeUninit<T>>> = (0..capacity)
            .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
            .collect();

        Inner {
            buffer: buffer.into_boxed_slice(),
            capacity,
            mask,
            head: CachePadded { value: AtomicUsize::new(0) },
            tail: CachePadded { value: AtomicUsize::new(0) },
        }
    }
}

impl<T> Drop for Inner<T> {
    fn drop(&mut self) {
        let head = *self.head.value.get_mut();
        let tail = *self.tail.value.get_mut();

        for i in tail..head {
            let slot = i & self.mask;
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
    pub fn try_push(&self, value: T) -> Result<(), T> {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.inner.capacity {
            return Err(value);
        }

        let slot = head & self.inner.mask;
        unsafe {
            (*self.inner.buffer[slot].get()).write(value);
        }

        self.inner.head.value.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }
}

impl<T: Copy> Producer<T> {
    pub fn push_slice(&self, values: &[T]) -> usize {
        let head = self.inner.head.value.load(Ordering::Relaxed);
        let tail = self.inner.tail.value.load(Ordering::Acquire);

        let available = self.inner.capacity - head.wrapping_sub(tail);
        let count = values.len().min(available);

        for (i, value) in values[..count].iter().enumerate() {
            let slot = (head + i) & self.inner.mask;
            unsafe {
                (*self.inner.buffer[slot].get()).write(*value);
            }
        }

        self.inner.head.value.store(head.wrapping_add(count), Ordering::Release);
        count
    }
}

// ---------------------------------------------------------------------------
// Consumer implementation
// ---------------------------------------------------------------------------

impl<T> Consumer<T> {
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Acquire);

        if tail == head {
            return None;
        }

        let slot = tail & self.inner.mask;
        let value = unsafe { (*self.inner.buffer[slot].get()).assume_init_read() };

        self.inner.tail.value.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T: Copy> Consumer<T> {
    pub fn pop_slice(&self, output: &mut [T]) -> usize {
        let tail = self.inner.tail.value.load(Ordering::Relaxed);
        let head = self.inner.head.value.load(Ordering::Acquire);

        let available = head.wrapping_sub(tail);
        let count = output.len().min(available);

        for (i, dest) in output[..count].iter_mut().enumerate() {
            let slot = (tail + i) & self.inner.mask;
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

    #[test]
    fn test_full_buffer() {
        let (producer, consumer) = new::<u32>(4);
        for i in 0..4u32 {
            assert!(producer.try_push(i).is_ok(), "slot {i} should be free");
        }
        assert!(producer.try_push(99).is_err(), "should be full");
        assert_eq!(consumer.try_pop(), Some(0));
        assert!(producer.try_push(99).is_ok(), "slot freed after pop");
    }

    #[test]
    fn test_empty_pop() {
        let (_producer, consumer) = new::<u32>(8);
        assert_eq!(consumer.try_pop(), None);
    }

    #[test]
    fn test_power_of_two_rounding() {
        let (p, _) = new::<u32>(100);
        assert_eq!(p.capacity(), 128);

        let (p, _) = new::<u32>(1);
        assert_eq!(p.capacity(), 2);

        let (p, _) = new::<u32>(8);
        assert_eq!(p.capacity(), 8);
    }

    #[test]
    fn test_wraparound() {
        let (producer, consumer) = new::<u64>(4);
        for i in 0..1000u64 {
            while producer.try_push(i).is_err() {}
            assert_eq!(consumer.try_pop(), Some(i));
        }
    }

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

    #[test]
    fn test_batch_partial() {
        let (producer, consumer) = new::<u32>(4);
        let input: Vec<u32> = (0..10).collect();
        let pushed = producer.push_slice(&input);
        assert_eq!(pushed, 4, "should only push 4 (capacity)");

        let mut output = vec![0u32; 10];
        let popped = consumer.pop_slice(&mut output);
        assert_eq!(popped, 4, "should only pop 4");
        assert_eq!(&output[..4], &[0u32, 1, 2, 3]);
    }

    #[test]
    fn test_concurrent_correctness() {
        use std::thread;

        const N: u64 = 1_000_000;
        let (producer, consumer) = new::<u64>(1024);

        let producer_thread = thread::spawn(move || {
            for i in 0..N {
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
