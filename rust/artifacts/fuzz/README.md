# Fuzz Testing Results

Coverage-guided fuzz testing of the protobuf decoder.

## Regenerate (requires nightly Rust)

```bash
# Install cargo-fuzz (one-time)
rustup install nightly
cargo +nightly install cargo-fuzz

# Create fuzz target (if not already present)
cd rust/crates/transcoder
cargo +nightly fuzz init

# Run the fuzzer
cargo +nightly fuzz run decode_packet -- -max_len=4096 -runs=1000000
```

## Fuzz Target

```rust
// fuzz/fuzz_targets/decode_packet.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Must never panic on any input
    let _ = transcoder::decode_neural_packet(data);
});
```

## Expected Outcome

- **No panics** on any input up to 4096 bytes
- All error paths return `Err(DecodeError::...)` instead of panicking
- Crashing inputs (if any) are saved to `fuzz/artifacts/decode_packet/`
  and should be converted to regression tests

## Note

`cargo fuzz` requires:
- **Nightly Rust** (`rustup run nightly`)
- **libfuzzer** (bundled with nightly on Linux/macOS)
- **Windows**: `cargo fuzz` does not support Windows. Run on WSL or a Linux CI machine.
