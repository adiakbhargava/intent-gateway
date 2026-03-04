//! Multi-modal timestamp alignment engine.
//!
//! Fuses EEG and gaze data streams using EEG-anchored nearest-neighbor
//! alignment. EEG is the reference clock (higher temporal precision for
//! neural events). For each EEG frame, the engine finds the gaze frame
//! with the closest timestamp.

use std::collections::VecDeque;
use co_core::statistics::WelfordState;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A timestamped EEG sample batch from one acquisition cycle.
#[derive(Debug, Clone)]
pub struct EegFrame {
    pub timestamp_ns: u64,
    pub channels: Vec<f32>,
    pub sequence_id: u32,
}

/// A timestamped gaze sample.
#[derive(Debug, Clone)]
pub struct GazeFrame {
    pub timestamp_ns: u64,
    pub x: f32,
    pub y: f32,
    pub pupil_diameter: f32,
}

/// Aligned multi-modal frame — the output of fusion.
#[derive(Debug, Clone)]
pub struct FusedFrame {
    pub timestamp_ns: u64,
    pub eeg: EegFrame,
    pub gaze: GazeFrame,
    pub alignment_offset_ns: i64,
    pub sequence_id: u32,
}

// ---------------------------------------------------------------------------
// FusionEngine
// ---------------------------------------------------------------------------

pub struct FusionEngine {
    max_alignment_ns: u64,
    gaze_buffer: VecDeque<GazeFrame>,
    gaze_buffer_cap: usize,
    alignment_stats: WelfordState,
    frames_fused: u64,
    frames_degraded: u64,
}

impl FusionEngine {
    pub fn new(max_alignment_ms: f64, gaze_buffer_cap: usize) -> Self {
        FusionEngine {
            max_alignment_ns: (max_alignment_ms * 1_000_000.0) as u64,
            gaze_buffer: VecDeque::with_capacity(gaze_buffer_cap),
            gaze_buffer_cap,
            alignment_stats: WelfordState::new(),
            frames_fused: 0,
            frames_degraded: 0,
        }
    }

    /// Buffer an incoming gaze frame.
    pub fn push_gaze(&mut self, frame: GazeFrame) {
        if self.gaze_buffer.len() >= self.gaze_buffer_cap {
            self.gaze_buffer.pop_front();
        }
        self.gaze_buffer.push_back(frame);
    }

    /// Fuse an EEG frame with the nearest buffered gaze frame.
    /// Returns None if no gaze data is available yet.
    pub fn fuse(&mut self, eeg: EegFrame) -> Option<FusedFrame> {
        let (idx, offset) = self.find_nearest_gaze(eeg.timestamp_ns)?;

        let gaze = self.gaze_buffer[idx].clone();
        let abs_offset = offset.unsigned_abs();

        self.alignment_stats.update(abs_offset as f64);
        self.frames_fused += 1;

        if abs_offset > self.max_alignment_ns {
            self.frames_degraded += 1;
        }

        let sequence_id = eeg.sequence_id;
        Some(FusedFrame {
            timestamp_ns: eeg.timestamp_ns,
            eeg,
            gaze,
            alignment_offset_ns: offset,
            sequence_id,
        })
    }

    /// Mean alignment offset (for monitoring).
    pub fn mean_alignment_ns(&self) -> f64 {
        self.alignment_stats.mean()
    }

    /// Fraction of frames with alignment > max_alignment_ns.
    pub fn degraded_fraction(&self) -> f64 {
        if self.frames_fused == 0 {
            0.0
        } else {
            self.frames_degraded as f64 / self.frames_fused as f64
        }
    }

    pub fn frames_fused(&self) -> u64 {
        self.frames_fused
    }

    /// Find the gaze frame with the timestamp closest to `target_ns`.
    /// Returns `(index, signed_offset_ns)` where offset = gaze_ts - target_ts.
    fn find_nearest_gaze(&self, target_ns: u64) -> Option<(usize, i64)> {
        if self.gaze_buffer.is_empty() {
            return None;
        }

        // Binary search for the partition point where gaze timestamps
        // transition from <= target to > target.
        let partition = self.gaze_buffer.partition_point(|g| g.timestamp_ns <= target_ns);

        let mut best_idx = 0usize;
        let mut best_offset = i64::MAX;

        // Check candidate before partition point
        if partition > 0 {
            let idx = partition - 1;
            let offset = target_ns as i64 - self.gaze_buffer[idx].timestamp_ns as i64;
            if offset.abs() < best_offset.abs() {
                best_idx = idx;
                best_offset = -offset; // gaze_ts - target_ts
            }
        }

        // Check candidate at partition point
        if partition < self.gaze_buffer.len() {
            let offset = self.gaze_buffer[partition].timestamp_ns as i64 - target_ns as i64;
            if offset.abs() < best_offset.abs() {
                best_idx = partition;
                best_offset = offset;
            }
        }

        Some((best_idx, best_offset))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_eeg(ts: u64, seq: u32) -> EegFrame {
        EegFrame {
            timestamp_ns: ts,
            channels: vec![0.0; 64],
            sequence_id: seq,
        }
    }

    fn make_gaze(ts: u64) -> GazeFrame {
        GazeFrame {
            timestamp_ns: ts,
            x: 0.0,
            y: 0.0,
            pupil_diameter: 3.0,
        }
    }

    #[test]
    fn test_fuse_exact_match() {
        let mut engine = FusionEngine::new(10.0, 100);
        engine.push_gaze(make_gaze(1_000_000));
        let fused = engine.fuse(make_eeg(1_000_000, 0)).unwrap();
        assert_eq!(fused.alignment_offset_ns, 0);
    }

    #[test]
    fn test_fuse_nearest_neighbor() {
        let mut engine = FusionEngine::new(10.0, 100);
        engine.push_gaze(make_gaze(500_000));   // 500us before
        engine.push_gaze(make_gaze(1_500_000)); // 500us after
        let fused = engine.fuse(make_eeg(1_000_000, 0)).unwrap();
        // Should pick whichever is closest (both are 500us away, picks first)
        assert!(fused.alignment_offset_ns.unsigned_abs() <= 500_000);
    }

    #[test]
    fn test_fuse_no_gaze_returns_none() {
        let mut engine = FusionEngine::new(10.0, 100);
        assert!(engine.fuse(make_eeg(1_000_000, 0)).is_none());
    }

    #[test]
    fn test_fuse_degraded_tracking() {
        let mut engine = FusionEngine::new(0.001, 100); // max 1us
        engine.push_gaze(make_gaze(0));
        // EEG is 10ms after gaze — way beyond 1us max
        engine.fuse(make_eeg(10_000_000, 0)).unwrap();
        assert!((engine.degraded_fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_alignment_stats() {
        let mut engine = FusionEngine::new(100.0, 100);
        for i in 0..10u64 {
            engine.push_gaze(make_gaze(i * 1_000_000));
            engine.fuse(make_eeg(i * 1_000_000 + 100, i as u32));
        }
        assert!(engine.mean_alignment_ns() > 0.0);
        assert_eq!(engine.frames_fused(), 10);
    }

    #[test]
    fn test_gaze_buffer_eviction() {
        let mut engine = FusionEngine::new(10.0, 4);
        for i in 0..10u64 {
            engine.push_gaze(make_gaze(i * 1_000_000));
        }
        // Buffer should only hold 4 most recent
        assert_eq!(engine.gaze_buffer.len(), 4);
        assert_eq!(engine.gaze_buffer[0].timestamp_ns, 6_000_000);
    }

    #[test]
    fn test_monotonic_timestamps() {
        let mut engine = FusionEngine::new(10.0, 100);
        engine.push_gaze(make_gaze(1_000_000));
        engine.push_gaze(make_gaze(2_000_000));
        engine.push_gaze(make_gaze(3_000_000));

        let fused = engine.fuse(make_eeg(2_500_000, 0)).unwrap();
        // Should pick either 2_000_000 or 3_000_000 (both 500us away)
        assert!(fused.alignment_offset_ns.unsigned_abs() <= 500_000);
    }
}
