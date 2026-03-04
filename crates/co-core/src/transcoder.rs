//! Zero-allocation protobuf decoder for neural telemetry packets.
//!
//! # Wire format
//!
//! Each field is encoded as `[tag varint][value]`.
//! Tag encoding: `tag = (field_number << 3) | wire_type`
//!
//! Wire types used here:
//! - `0` Varint  - `2` Length-delimited  - `5` Fixed 32-bit
//!
//! # Zero allocations
//!
//! `decode_length_delimited` returns `&'a [u8]` borrowed from the original
//! buffer — no `Vec` is created. `NeuralPacket` inherits that lifetime.

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DecodeError {
    UnexpectedEof,
    VarintOverflow,
    InvalidWireType(u8),
    InvalidLength,
    InvalidFieldNumber,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnexpectedEof => write!(f, "unexpected end of input"),
            DecodeError::VarintOverflow => write!(f, "varint overflows u64 (>10 bytes)"),
            DecodeError::InvalidWireType(wt) => write!(f, "unknown wire type {wt}"),
            DecodeError::InvalidLength => write!(f, "length-delimited field extends past buffer"),
            DecodeError::InvalidFieldNumber => write!(f, "field number 0 is reserved"),
        }
    }
}

impl std::error::Error for DecodeError {}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

pub struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Decoder { buf, pos: 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub fn decode_varint(&mut self) -> Result<u64, DecodeError> {
        let mut result = 0u64;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.buf.len() {
                return Err(DecodeError::UnexpectedEof);
            }
            if shift >= 70 {
                return Err(DecodeError::VarintOverflow);
            }
            let byte = self.buf[self.pos];
            self.pos += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
        }
    }

    pub fn decode_tag(&mut self) -> Result<(u32, u8), DecodeError> {
        let tag = self.decode_varint()?;
        let wire_type = (tag & 0x07) as u8;
        let field_number = (tag >> 3) as u32;
        if field_number == 0 {
            return Err(DecodeError::InvalidFieldNumber);
        }
        Ok((field_number, wire_type))
    }

    pub fn decode_fixed32(&mut self) -> Result<u32, DecodeError> {
        if self.pos + 4 > self.buf.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let b = &self.buf[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn decode_float(&mut self) -> Result<f32, DecodeError> {
        self.decode_fixed32().map(f32::from_bits)
    }

    pub fn decode_fixed64(&mut self) -> Result<u64, DecodeError> {
        if self.pos + 8 > self.buf.len() {
            return Err(DecodeError::UnexpectedEof);
        }
        let b = &self.buf[self.pos..self.pos + 8];
        self.pos += 8;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    pub fn decode_length_delimited(&mut self) -> Result<&'a [u8], DecodeError> {
        let len = self.decode_varint()? as usize;
        let end = self.pos.checked_add(len).ok_or(DecodeError::InvalidLength)?;
        if end > self.buf.len() {
            return Err(DecodeError::InvalidLength);
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    pub fn skip_field(&mut self, wire_type: u8) -> Result<(), DecodeError> {
        match wire_type {
            0 => { self.decode_varint()?; Ok(()) }
            1 => { self.decode_fixed64()?; Ok(()) }
            2 => { self.decode_length_delimited()?; Ok(()) }
            5 => { self.decode_fixed32()?; Ok(()) }
            wt => Err(DecodeError::InvalidWireType(wt)),
        }
    }
}

// ---------------------------------------------------------------------------
// NeuralPacket
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub struct NeuralPacket<'a> {
    pub timestamp_ns: u64,
    pub channel_count: u32,
    pub samples: &'a [u8],
    pub sequence_id: u32,
}

impl<'a> NeuralPacket<'a> {
    pub fn sample_count(&self) -> usize {
        self.samples.len() / 4
    }

    pub fn iter_samples(&self) -> impl Iterator<Item = f32> + '_ {
        self.samples.chunks_exact(4).map(|b| {
            f32::from_le_bytes([b[0], b[1], b[2], b[3]])
        })
    }
}

pub fn decode_neural_packet(buf: &[u8]) -> Result<NeuralPacket<'_>, DecodeError> {
    let mut dec = Decoder::new(buf);
    let mut timestamp_ns = 0u64;
    let mut channel_count = 0u32;
    let mut samples: &[u8] = &buf[..0];
    let mut sequence_id = 0u32;

    while !dec.is_empty() {
        let (field, wire_type) = dec.decode_tag()?;
        match (field, wire_type) {
            (1, 0) => timestamp_ns   = dec.decode_varint()?,
            (2, 0) => channel_count  = dec.decode_varint()? as u32,
            (3, 2) => samples        = dec.decode_length_delimited()?,
            (4, 0) => sequence_id   = dec.decode_varint()? as u32,
            (_, wt) => dec.skip_field(wt)?,
        }
    }

    Ok(NeuralPacket { timestamp_ns, channel_count, samples, sequence_id })
}

// ---------------------------------------------------------------------------
// Encoding helpers (tests + benches)
// ---------------------------------------------------------------------------

pub mod test_helpers {
    pub fn encode_varint(buf: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut b = (v & 0x7F) as u8;
            v >>= 7;
            if v != 0 { b |= 0x80; }
            buf.push(b);
            if v == 0 { break; }
        }
    }

    pub fn encode_tag(buf: &mut Vec<u8>, field: u32, wire: u8) {
        encode_varint(buf, ((field as u64) << 3) | wire as u64);
    }

    pub fn encode_neural_packet(
        buf: &mut Vec<u8>,
        timestamp_ns: u64,
        channel_count: u32,
        samples: &[f32],
        sequence_id: u32,
    ) {
        encode_tag(buf, 1, 0); encode_varint(buf, timestamp_ns);
        encode_tag(buf, 2, 0); encode_varint(buf, channel_count as u64);
        if !samples.is_empty() {
            encode_tag(buf, 3, 2);
            encode_varint(buf, (samples.len() * 4) as u64);
            for &s in samples { buf.extend_from_slice(&s.to_le_bytes()); }
        }
        encode_tag(buf, 4, 0); encode_varint(buf, sequence_id as u64);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use test_helpers::*;

    #[test] fn varint_single_byte() { assert_eq!(Decoder::new(&[0x01]).decode_varint(), Ok(1)); }
    #[test] fn varint_300() { assert_eq!(Decoder::new(&[0xAC, 0x02]).decode_varint(), Ok(300)); }
    #[test] fn varint_zero() { assert_eq!(Decoder::new(&[0x00]).decode_varint(), Ok(0)); }

    #[test]
    fn varint_u64_max() {
        let buf = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0x01];
        assert_eq!(Decoder::new(&buf).decode_varint(), Ok(u64::MAX));
    }

    #[test]
    fn varint_overflow() {
        assert_eq!(Decoder::new(&[0xFF; 11]).decode_varint(), Err(DecodeError::VarintOverflow));
    }

    #[test]
    fn varint_eof() {
        assert_eq!(Decoder::new(&[0x80]).decode_varint(), Err(DecodeError::UnexpectedEof));
    }

    #[test]
    fn fixed32_le() {
        assert_eq!(Decoder::new(&[0x01,0x00,0x00,0x00]).decode_fixed32(), Ok(1));
    }

    #[test]
    fn float_roundtrip() {
        let f = std::f32::consts::PI;
        let mut buf = Vec::new();
        buf.extend_from_slice(&f.to_le_bytes());
        assert!((Decoder::new(&buf).decode_float().unwrap() - f).abs() < 1e-7);
    }

    #[test]
    fn fixed64_roundtrip() {
        let v = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&v.to_le_bytes());
        assert_eq!(Decoder::new(&buf).decode_fixed64(), Ok(v));
    }

    #[test]
    fn length_delimited_zero_copy() {
        let payload = b"hello";
        let mut buf = Vec::new();
        encode_varint(&mut buf, payload.len() as u64);
        buf.extend_from_slice(payload);
        assert_eq!(Decoder::new(&buf).decode_length_delimited().unwrap(), payload.as_slice());
    }

    #[test]
    fn decode_full_packet() {
        let samples = [1.0f32, 2.0, 3.0, 4.0];
        let mut buf = Vec::new();
        encode_neural_packet(&mut buf, 1_000_000_000, 64, &samples, 42);
        let pkt = decode_neural_packet(&buf).unwrap();
        assert_eq!(pkt.timestamp_ns, 1_000_000_000);
        assert_eq!(pkt.channel_count, 64);
        assert_eq!(pkt.sequence_id, 42);
        assert_eq!(pkt.sample_count(), 4);
        let got: Vec<f32> = pkt.iter_samples().collect();
        assert_eq!(got, samples.to_vec());
    }

    #[test]
    fn decode_empty_packet() {
        let pkt = decode_neural_packet(&[]).unwrap();
        assert_eq!(pkt.timestamp_ns, 0);
        assert_eq!(pkt.samples.len(), 0);
    }

    #[test]
    fn decode_skips_unknown_fields() {
        let mut buf = Vec::new();
        encode_tag(&mut buf, 1, 0); encode_varint(&mut buf, 99);
        encode_tag(&mut buf, 99, 0); encode_varint(&mut buf, 12345);
        encode_tag(&mut buf, 4, 0); encode_varint(&mut buf, 7);
        let pkt = decode_neural_packet(&buf).unwrap();
        assert_eq!(pkt.timestamp_ns, 99);
        assert_eq!(pkt.sequence_id, 7);
    }

    #[test]
    fn samples_are_zero_copy() {
        let samples = [0.5f32, 1.5];
        let mut buf = Vec::new();
        encode_neural_packet(&mut buf, 0, 0, &samples, 0);
        let pkt = decode_neural_packet(&buf).unwrap();
        assert!(buf.as_ptr_range().contains(&pkt.samples.as_ptr()));
    }

    #[test]
    fn fuzz_regression_huge_length_overflow() {
        let data: &[u8] = &[
            0x0a, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0x1f, 0xff, 0xff, 0xff,
        ];
        assert_eq!(decode_neural_packet(data), Err(DecodeError::InvalidLength));
    }
}
