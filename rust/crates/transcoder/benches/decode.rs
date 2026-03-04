//! Criterion benchmarks for the zero-allocation protobuf decoder.
//!
//! Run with: cargo bench -p transcoder

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use transcoder::test_helpers::encode_neural_packet;
use transcoder::decode_neural_packet;

fn make_packet(n_samples: usize) -> Vec<u8> {
    let samples: Vec<f32> = (0..n_samples).map(|i| i as f32 * 0.001).collect();
    let mut buf = Vec::new();
    encode_neural_packet(&mut buf, 1_234_567_890, 64, &samples, 42);
    buf
}

// ---------------------------------------------------------------------------
// 1. Round-trip latency for typical EEG packet (64ch × 500 samples = 32 000 f32)
// ---------------------------------------------------------------------------

fn bench_decode_typical(c: &mut Criterion) {
    let buf = make_packet(32_000);
    c.bench_function("decode_typical_64ch_500samples", |b| {
        b.iter(|| {
            let pkt = decode_neural_packet(std::hint::black_box(&buf)).unwrap();
            std::hint::black_box(pkt.sample_count())
        });
    });
}

// ---------------------------------------------------------------------------
// 2. Decode latency vs packet size
// ---------------------------------------------------------------------------

fn bench_decode_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_by_sample_count");

    for &n in &[64usize, 512, 4096, 32_000] {
        let buf = make_packet(n);
        group.throughput(Throughput::Bytes(buf.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let pkt = decode_neural_packet(std::hint::black_box(&buf)).unwrap();
                std::hint::black_box(pkt.sample_count())
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Varint decode throughput (hot path within decode_neural_packet)
// ---------------------------------------------------------------------------

fn bench_varint_sequence(c: &mut Criterion) {
    // Build a buffer of 1 000 packed varints
    let mut buf = Vec::with_capacity(10_000);
    for i in 0u64..1_000 {
        let mut v = i * 12345 + 1;
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 { buf.push(b | 0x80); } else { buf.push(b); break; }
        }
    }
    let buf = buf;

    c.bench_function("decode_1k_varints", |b| {
        b.iter(|| {
            let mut dec = transcoder::Decoder::new(std::hint::black_box(&buf));
            let mut sum = 0u64;
            while !dec.is_empty() {
                if let Ok(v) = dec.decode_varint() { sum = sum.wrapping_add(v); }
                else { break; }
            }
            std::hint::black_box(sum)
        });
    });
}

criterion_group!(benches, bench_decode_typical, bench_decode_by_size, bench_varint_sequence);
criterion_main!(benches);
