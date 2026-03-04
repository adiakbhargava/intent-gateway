//! Criterion benchmarks for the statistics crate.
//!
//! Run with: cargo bench -p statistics

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use statistics::{WelfordState, LatencyHistogram, JitterTracker, RollingFft};

// ---------------------------------------------------------------------------
// 1. Welford online update throughput
// ---------------------------------------------------------------------------

fn bench_welford(c: &mut Criterion) {
    let mut group = c.benchmark_group("welford_update");

    for &n in &[1_000usize, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut s = WelfordState::new();
                for i in 0..n {
                    s.update(std::hint::black_box(i as f64));
                }
                std::hint::black_box(s.mean())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. LatencyHistogram record throughput
// ---------------------------------------------------------------------------

fn bench_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_record");
    group.throughput(Throughput::Elements(100_000));

    group.bench_function("record_100k", |b| {
        b.iter(|| {
            let mut h = LatencyHistogram::new();
            for i in 0u64..100_000 {
                h.record(std::hint::black_box(i * 100 + 1));
            }
            std::hint::black_box(h.percentile(99.0))
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. FFT computation time vs window size
// ---------------------------------------------------------------------------

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_compute");

    for &size in &[64usize, 256, 1024, 4096] {
        let mut fft = RollingFft::new(size);
        for i in 0..size {
            fft.push((i as f64 * 0.1).sin());
        }
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                std::hint::black_box(fft.compute_power_spectrum())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. JitterTracker record throughput
// ---------------------------------------------------------------------------

fn bench_jitter(c: &mut Criterion) {
    c.bench_function("jitter_record_1M", |b| {
        b.iter(|| {
            let mut jt = JitterTracker::new();
            for i in 0u64..1_000_000 {
                jt.record(std::hint::black_box(i * 1_000_000));
            }
            std::hint::black_box(jt.jitter_std_dev_ns())
        });
    });
}

criterion_group!(benches, bench_welford, bench_histogram, bench_fft, bench_jitter);
criterion_main!(benches);
