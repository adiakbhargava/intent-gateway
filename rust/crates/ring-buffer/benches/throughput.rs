//! Criterion benchmarks for the SPSC ring buffer.
//!
//! Run with:
//!   cargo bench -p ring-buffer
//!   cargo bench -p ring-buffer -- single_push_pop_latency   # one group
//!
//! HTML reports land in target/criterion/.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ring_buffer::new;

// ---------------------------------------------------------------------------
// 1. Single push-pop round-trip latency
//
// Measures the cost of ONE try_push + ONE try_pop in the common case where
// the buffer has space and an item is available. This is the baseline latency
// for the hot path.
// ---------------------------------------------------------------------------

fn bench_single_latency(c: &mut Criterion) {
    let (producer, consumer) = new::<u64>(1024);

    c.bench_function("single_push_pop_latency", |b| {
        b.iter(|| {
            // black_box prevents the optimizer from eliminating the push/pop.
            producer.try_push(std::hint::black_box(42u64)).ok();
            std::hint::black_box(consumer.try_pop());
        });
    });
}

// ---------------------------------------------------------------------------
// 2. Throughput as a function of ring buffer capacity
//
// Tests whether cache effects from larger slot arrays affect per-operation
// cost. Smaller buffers stay hot in L1/L2; larger ones may miss.
// ---------------------------------------------------------------------------

fn bench_throughput_by_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_by_capacity");
    group.throughput(Throughput::Elements(1));

    for &cap in &[16usize, 64, 256, 1024, 4096, 16384] {
        let (producer, consumer) = new::<u64>(cap);

        group.bench_with_input(BenchmarkId::from_parameter(cap), &cap, |b, _| {
            b.iter(|| {
                producer.try_push(std::hint::black_box(1u64)).ok();
                std::hint::black_box(consumer.try_pop());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Batch push/pop throughput (T: Copy)
//
// push_slice / pop_slice amortize atomic overhead across multiple items.
// This benchmark shows the benefit of batching and how it scales with batch
// size.
// ---------------------------------------------------------------------------

fn bench_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_throughput");

    for &batch_size in &[8usize, 32, 128, 512] {
        let (producer, consumer) = new::<u64>(4096);
        let input = vec![0u64; batch_size];
        let mut output = vec![0u64; batch_size];

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    producer.push_slice(std::hint::black_box(input.as_slice()));
                    consumer.pop_slice(std::hint::black_box(output.as_mut_slice()));
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Head-to-head vs crossbeam bounded channel
//
// Crossbeam's bounded channel is the standard high-performance reference.
// Both are tested single-threaded (no context switching) so we measure pure
// data-structure overhead.
// ---------------------------------------------------------------------------

fn bench_vs_crossbeam(c: &mut Criterion) {
    let mut group = c.benchmark_group("vs_crossbeam");
    group.throughput(Throughput::Elements(1));

    // Our ring buffer
    let (producer, consumer) = new::<u64>(1024);
    group.bench_function("ring_buffer", |b| {
        b.iter(|| {
            producer.try_push(std::hint::black_box(42u64)).ok();
            std::hint::black_box(consumer.try_pop());
        });
    });

    // crossbeam bounded channel (same capacity)
    let (tx, rx) = crossbeam::channel::bounded::<u64>(1024);
    group.bench_function("crossbeam_bounded", |b| {
        b.iter(|| {
            tx.try_send(std::hint::black_box(42u64)).ok();
            std::hint::black_box(rx.try_recv().ok());
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Concurrent throughput (producer and consumer on separate threads)
//
// Measures real-world throughput when the producer and consumer are on
// different CPU cores. Thread-spawn overhead is amortised over N messages
// per iteration (N = 1 000 so Criterion's timing is reasonable).
// ---------------------------------------------------------------------------

fn bench_concurrent_throughput(c: &mut Criterion) {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    const MESSAGES_PER_ITER: u64 = 1_000;

    let mut group = c.benchmark_group("concurrent_throughput");
    group.throughput(Throughput::Elements(MESSAGES_PER_ITER));

    group.bench_function("ring_buffer_1k_messages", |b| {
        b.iter(|| {
            let (producer, consumer) = new::<u64>(1024);
            let done = Arc::new(AtomicBool::new(false));
            let done2 = Arc::clone(&done);

            let producer_thread = std::thread::spawn(move || {
                for i in 0..MESSAGES_PER_ITER {
                    while producer.try_push(i).is_err() {
                        std::hint::spin_loop();
                    }
                }
            });

            let consumer_thread = std::thread::spawn(move || {
                let mut received = 0u64;
                while received < MESSAGES_PER_ITER {
                    if consumer.try_pop().is_some() {
                        received += 1;
                    } else {
                        std::hint::spin_loop();
                    }
                }
                done2.store(true, Ordering::Relaxed);
            });

            producer_thread.join().unwrap();
            consumer_thread.join().unwrap();

            // Prevent the compiler from optimizing away the done flag.
            std::hint::black_box(done.load(Ordering::Relaxed));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_single_latency,
    bench_throughput_by_capacity,
    bench_batch_throughput,
    bench_vs_crossbeam,
    bench_concurrent_throughput,
);
criterion_main!(benches);
