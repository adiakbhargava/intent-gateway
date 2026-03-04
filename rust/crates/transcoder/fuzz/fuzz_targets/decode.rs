//! Fuzz target for the zero-copy protobuf decoder.
//!
//! Feeds arbitrary byte sequences into `decode_neural_packet` to verify it
//! never panics — only returns `Ok(NeuralPacket)` or `Err(DecodeError)`.
//!
//! Run:  cargo +nightly fuzz run decode -- -max_len=131072
//! From: rust/crates/transcoder/

#![no_main]

use libfuzzer_sys::fuzz_target;
use transcoder::decode_neural_packet;

fuzz_target!(|data: &[u8]| {
    // This must NEVER panic.  Every input must resolve to Ok or Err.
    let _ = decode_neural_packet(data);
});
