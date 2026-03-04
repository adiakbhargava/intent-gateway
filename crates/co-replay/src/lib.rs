//! Dataset replay source for co-gateway.
//!
//! Reads `.corec` binary files produced by `python/convert_dataset.py` and
//! replays them through the pipeline as if they were live hardware streams.
//! Supports EEGEyeNet and EEGET-ALS datasets.

use co_fusion::{EegFrame, GazeFrame};
use co_ingest::{EegSource, GazeSource};

use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::Path;

// ---------------------------------------------------------------------------
// Binary format constants
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = *b"CREC";
const HEADER_SIZE: usize = 40;  // 4+4+8+4+8+4+4 = 36, padded to 40

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CorecHeader {
    pub sample_rate_hz: f64,
    pub eeg_channels: u32,
    pub gaze_rate_hz: f64,
    pub num_eeg_frames: u32,
    pub num_gaze_frames: u32,
}

impl CorecHeader {
    fn read_from(reader: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        reader.read_exact(&mut buf)?;

        if buf[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid magic: expected CREC, got {:?}", &buf[0..4]),
            ));
        }

        let _version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let sample_rate_hz = f64::from_le_bytes(buf[8..16].try_into().unwrap());
        let eeg_channels = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let gaze_rate_hz = f64::from_le_bytes(buf[20..28].try_into().unwrap());
        let num_eeg_frames = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let num_gaze_frames = u32::from_le_bytes(buf[32..36].try_into().unwrap());

        Ok(CorecHeader {
            sample_rate_hz,
            eeg_channels,
            gaze_rate_hz,
            num_eeg_frames,
            num_gaze_frames,
        })
    }
}

// ---------------------------------------------------------------------------
// ReplayEegSource
// ---------------------------------------------------------------------------

/// Replays EEG frames from a `.corec` file.
pub struct ReplayEegSource {
    reader: BufReader<File>,
    header: CorecHeader,
    frames_read: u32,
    sequence: u32,
    /// Bytes per EEG frame: 8 (timestamp) + 4 * channels
    frame_bytes: usize,
}

impl ReplayEegSource {
    /// Open a `.corec` file for EEG replay.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let header = CorecHeader::read_from(&mut file)?;
        let frame_bytes = 8 + 4 * header.eeg_channels as usize;

        Ok(ReplayEegSource {
            reader: file,
            header,
            frames_read: 0,
            sequence: 0,
            frame_bytes,
        })
    }

    pub fn header(&self) -> &CorecHeader { &self.header }

    /// Read one EEG frame from the file.
    fn read_frame(&mut self) -> io::Result<Option<EegFrame>> {
        if self.frames_read >= self.header.num_eeg_frames {
            return Ok(None);
        }

        let mut buf = vec![0u8; self.frame_bytes];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let timestamp_ns = u64::from_le_bytes(buf[0..8].try_into().unwrap());

        let n_ch = self.header.eeg_channels as usize;
        let mut channels = Vec::with_capacity(n_ch);
        for i in 0..n_ch {
            let offset = 8 + i * 4;
            let val = f32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
            channels.push(val);
        }

        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        self.frames_read += 1;

        Ok(Some(EegFrame {
            timestamp_ns,
            channels,
            sequence_id: seq,
        }))
    }
}

impl EegSource for ReplayEegSource {
    fn next_eeg_frame(&mut self) -> Option<EegFrame> {
        self.read_frame().ok().flatten()
    }

    fn sample_rate_hz(&self) -> f64 {
        self.header.sample_rate_hz
    }

    fn channel_count(&self) -> u32 {
        self.header.eeg_channels
    }
}

// ---------------------------------------------------------------------------
// ReplayGazeSource
// ---------------------------------------------------------------------------

/// Replays gaze frames from a `.corec` file.
pub struct ReplayGazeSource {
    reader: BufReader<File>,
    header: CorecHeader,
    frames_read: u32,
}

impl ReplayGazeSource {
    /// Open a `.corec` file for gaze replay.
    ///
    /// Seeks past the header and EEG section to the gaze data.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let header = CorecHeader::read_from(&mut file)?;

        // Skip past all EEG frames to reach gaze section
        let eeg_frame_bytes = 8 + 4 * header.eeg_channels as usize;
        let eeg_section_bytes = eeg_frame_bytes * header.num_eeg_frames as usize;

        // Read and discard EEG section
        let mut skip = vec![0u8; eeg_section_bytes.min(65536)];
        let mut remaining = eeg_section_bytes;
        while remaining > 0 {
            let chunk = remaining.min(skip.len());
            file.read_exact(&mut skip[..chunk])?;
            remaining -= chunk;
        }

        Ok(ReplayGazeSource {
            reader: file,
            header,
            frames_read: 0,
        })
    }

    pub fn header(&self) -> &CorecHeader { &self.header }

    /// Read one gaze frame from the file.
    fn read_frame(&mut self) -> io::Result<Option<GazeFrame>> {
        if self.frames_read >= self.header.num_gaze_frames {
            return Ok(None);
        }

        // 8 (timestamp) + 4*3 (x, y, pupil) = 20 bytes
        let mut buf = [0u8; 20];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let timestamp_ns = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let x = f32::from_le_bytes(buf[8..12].try_into().unwrap());
        let y = f32::from_le_bytes(buf[12..16].try_into().unwrap());
        let pupil_diameter = f32::from_le_bytes(buf[16..20].try_into().unwrap());

        self.frames_read += 1;

        Ok(Some(GazeFrame {
            timestamp_ns,
            x,
            y,
            pupil_diameter,
        }))
    }
}

impl GazeSource for ReplayGazeSource {
    fn next_gaze_frame(&mut self) -> Option<GazeFrame> {
        self.read_frame().ok().flatten()
    }

    fn gaze_rate_hz(&self) -> f64 {
        self.header.gaze_rate_hz
    }
}

// ---------------------------------------------------------------------------
// Utility: write a .corec file from in-memory data (for testing)
// ---------------------------------------------------------------------------

/// Write a `.corec` file from in-memory frames. Used for testing.
pub fn write_corec(
    path: impl AsRef<Path>,
    sample_rate_hz: f64,
    gaze_rate_hz: f64,
    eeg_frames: &[EegFrame],
    gaze_frames: &[GazeFrame],
) -> io::Result<()> {
    use std::io::Write;
    let mut f = File::create(path)?;

    let eeg_channels = eeg_frames.first().map(|f| f.channels.len() as u32).unwrap_or(0);

    // Header (40 bytes)
    f.write_all(&MAGIC)?;
    f.write_all(&1u32.to_le_bytes())?;                              // version
    f.write_all(&sample_rate_hz.to_le_bytes())?;                    // sample_rate_hz
    f.write_all(&eeg_channels.to_le_bytes())?;                      // eeg_channels
    f.write_all(&gaze_rate_hz.to_le_bytes())?;                      // gaze_rate_hz
    f.write_all(&(eeg_frames.len() as u32).to_le_bytes())?;         // num_eeg_frames
    f.write_all(&(gaze_frames.len() as u32).to_le_bytes())?;        // num_gaze_frames
    f.write_all(&[0u8; 4])?;                                        // padding to 40

    // EEG frames
    for frame in eeg_frames {
        f.write_all(&frame.timestamp_ns.to_le_bytes())?;
        for &ch in &frame.channels {
            f.write_all(&ch.to_le_bytes())?;
        }
    }

    // Gaze frames
    for frame in gaze_frames {
        f.write_all(&frame.timestamp_ns.to_le_bytes())?;
        f.write_all(&frame.x.to_le_bytes())?;
        f.write_all(&frame.y.to_le_bytes())?;
        f.write_all(&frame.pupil_diameter.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn make_test_eeg(n: u32, channels: u32) -> Vec<EegFrame> {
        (0..n).map(|i| EegFrame {
            timestamp_ns: i as u64 * 3_906_250, // 256 Hz
            channels: (0..channels).map(|c| (i * channels + c) as f32 * 0.1).collect(),
            sequence_id: i,
        }).collect()
    }

    fn make_test_gaze(n: u32) -> Vec<GazeFrame> {
        (0..n).map(|i| GazeFrame {
            timestamp_ns: i as u64 * 16_666_667, // 60 Hz
            x: i as f32 * 0.5,
            y: i as f32 * -0.3,
            pupil_diameter: 3.0 + i as f32 * 0.01,
        }).collect()
    }

    #[test]
    fn test_roundtrip_eeg() {
        let dir = env::temp_dir().join("co_replay_test_eeg.corec");
        let eeg = make_test_eeg(100, 32);
        let gaze = make_test_gaze(25);

        write_corec(&dir, 256.0, 60.0, &eeg, &gaze).unwrap();

        let mut src = ReplayEegSource::open(&dir).unwrap();
        assert_eq!(src.header().eeg_channels, 32);
        assert_eq!(src.header().num_eeg_frames, 100);
        assert!((src.sample_rate_hz() - 256.0).abs() < 1e-6);

        for i in 0..100 {
            let frame = src.next_eeg_frame().expect("expected frame");
            assert_eq!(frame.sequence_id, i);
            assert_eq!(frame.channels.len(), 32);
            assert!((frame.channels[0] - (i * 32) as f32 * 0.1).abs() < 1e-5);
        }

        assert!(src.next_eeg_frame().is_none());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn test_roundtrip_gaze() {
        let dir = env::temp_dir().join("co_replay_test_gaze.corec");
        let eeg = make_test_eeg(50, 4);
        let gaze = make_test_gaze(20);

        write_corec(&dir, 256.0, 60.0, &eeg, &gaze).unwrap();

        let mut src = ReplayGazeSource::open(&dir).unwrap();
        assert_eq!(src.header().num_gaze_frames, 20);
        assert!((src.gaze_rate_hz() - 60.0).abs() < 1e-6);

        for i in 0..20 {
            let frame = src.next_gaze_frame().expect("expected frame");
            assert!((frame.x - i as f32 * 0.5).abs() < 1e-5);
            assert!((frame.y - i as f32 * -0.3).abs() < 1e-5);
        }

        assert!(src.next_gaze_frame().is_none());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn test_header_validation() {
        let dir = env::temp_dir().join("co_replay_test_bad.corec");
        std::fs::write(&dir, b"NOPE_not_a_corec_file").unwrap();

        let result = ReplayEegSource::open(&dir);
        assert!(result.is_err());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn test_eeg_source_trait() {
        let dir = env::temp_dir().join("co_replay_test_trait.corec");
        let eeg = make_test_eeg(10, 8);
        let gaze = make_test_gaze(5);
        write_corec(&dir, 128.0, 30.0, &eeg, &gaze).unwrap();

        let mut src = ReplayEegSource::open(&dir).unwrap();
        assert_eq!(src.channel_count(), 8);

        let mut count = 0;
        while src.next_eeg_frame().is_some() {
            count += 1;
        }
        assert_eq!(count, 10);
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn test_empty_file() {
        let dir = env::temp_dir().join("co_replay_test_empty.corec");
        let eeg: Vec<EegFrame> = vec![];
        let gaze: Vec<GazeFrame> = vec![];
        write_corec(&dir, 256.0, 60.0, &eeg, &gaze).unwrap();

        let mut eeg_src = ReplayEegSource::open(&dir).unwrap();
        assert!(eeg_src.next_eeg_frame().is_none());

        let mut gaze_src = ReplayGazeSource::open(&dir).unwrap();
        assert!(gaze_src.next_gaze_frame().is_none());

        std::fs::remove_file(&dir).ok();
    }
}
