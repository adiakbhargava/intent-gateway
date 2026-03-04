//! Hardware source traits and simulated adapters.
//!
//! Defines the `EegSource` and `GazeSource` traits that abstract over
//! real hardware and simulated/replay data sources. Includes simulated
//! implementations for development and testing.

use co_fusion::{EegFrame, GazeFrame};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Hardware source traits
// ---------------------------------------------------------------------------

/// Trait for any EEG data source (simulated, replay, or live hardware).
pub trait EegSource {
    /// Returns the next EEG frame, or `None` if the source is exhausted.
    fn next_eeg_frame(&mut self) -> Option<EegFrame>;
    /// Sampling rate in Hz.
    fn sample_rate_hz(&self) -> f64;
    /// Number of EEG channels.
    fn channel_count(&self) -> u32;
}

/// Trait for any gaze/eye-tracking data source.
pub trait GazeSource {
    /// Returns the next gaze frame, or `None` if the source is exhausted.
    fn next_gaze_frame(&mut self) -> Option<GazeFrame>;
    /// Sampling rate in Hz.
    fn gaze_rate_hz(&self) -> f64;
}

// ---------------------------------------------------------------------------
// SimulatedEegSource
// ---------------------------------------------------------------------------

pub struct SimulatedEegSource {
    channel_count: u32,
    sample_rate_hz: f64,
    samples_per_frame: usize,
    sequence: u32,
    phase: f64,
}

impl SimulatedEegSource {
    pub fn new(channel_count: u32, sample_rate_hz: f64, samples_per_frame: usize) -> Self {
        SimulatedEegSource {
            channel_count,
            sample_rate_hz,
            samples_per_frame,
            sequence: 0,
            phase: 0.0,
        }
    }

    /// Generates alpha (10 Hz) + beta (20 Hz) sinusoids with Gaussian noise.
    pub fn next_frame(&mut self) -> EegFrame {
        let dt = 1.0 / self.sample_rate_hz;
        let mut channels = Vec::with_capacity(self.channel_count as usize);

        for ch in 0..self.channel_count {
            // Each channel gets a slightly different phase offset
            let ch_offset = ch as f64 * 0.1;
            let t = self.phase + ch_offset;

            // Alpha (10 Hz) + Beta (20 Hz) + noise
            let alpha = (2.0 * std::f64::consts::PI * 10.0 * t).sin() * 20.0;
            let beta = (2.0 * std::f64::consts::PI * 20.0 * t).sin() * 10.0;
            let noise = pseudo_gaussian(self.sequence as u64 * 1000 + ch as u64) * 5.0;

            channels.push((alpha + beta + noise) as f32);
        }

        self.phase += dt * self.samples_per_frame as f64;
        let timestamp_ns = now_ns();
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);

        EegFrame {
            timestamp_ns,
            channels,
            sequence_id: seq,
        }
    }

    pub fn channel_count(&self) -> u32 { self.channel_count }
    pub fn sample_rate_hz(&self) -> f64 { self.sample_rate_hz }
}

impl EegSource for SimulatedEegSource {
    fn next_eeg_frame(&mut self) -> Option<EegFrame> {
        Some(self.next_frame())
    }
    fn sample_rate_hz(&self) -> f64 { self.sample_rate_hz }
    fn channel_count(&self) -> u32 { self.channel_count }
}

// ---------------------------------------------------------------------------
// SimulatedGazeSource
// ---------------------------------------------------------------------------

pub struct SimulatedGazeSource {
    sample_rate_hz: f64,
    sequence: u32,
    fixation_x: f32,
    fixation_y: f32,
    saccade_countdown: u32,
}

impl SimulatedGazeSource {
    pub fn new(sample_rate_hz: f64) -> Self {
        SimulatedGazeSource {
            sample_rate_hz,
            sequence: 0,
            fixation_x: 0.0,
            fixation_y: 0.0,
            saccade_countdown: 60, // ~1 second at 60 Hz
        }
    }

    /// Simulates prosaccade task: fixation periods + saccadic jumps.
    pub fn next_frame(&mut self) -> GazeFrame {
        self.saccade_countdown = self.saccade_countdown.saturating_sub(1);

        if self.saccade_countdown == 0 {
            // Saccadic jump to new fixation point
            let seed = self.sequence as u64 * 7919 + 13;
            self.fixation_x = (pseudo_gaussian(seed) * 15.0) as f32;
            self.fixation_y = (pseudo_gaussian(seed + 1) * 10.0) as f32;
            // Next saccade in 30-90 frames (~0.5-1.5 seconds at 60 Hz)
            self.saccade_countdown = 30 + (xorshift(seed + 2) % 60) as u32;
        }

        // Add microsaccade noise around fixation point
        let noise_x = pseudo_gaussian(self.sequence as u64 * 31 + 7) * 0.1;
        let noise_y = pseudo_gaussian(self.sequence as u64 * 37 + 11) * 0.1;

        let timestamp_ns = now_ns();
        self.sequence = self.sequence.wrapping_add(1);

        GazeFrame {
            timestamp_ns,
            x: self.fixation_x + noise_x as f32,
            y: self.fixation_y + noise_y as f32,
            pupil_diameter: 3.0 + (pseudo_gaussian(self.sequence as u64) * 0.5) as f32,
        }
    }

    pub fn sample_rate_hz(&self) -> f64 { self.sample_rate_hz }
}

impl GazeSource for SimulatedGazeSource {
    fn next_gaze_frame(&mut self) -> Option<GazeFrame> {
        Some(self.next_frame())
    }
    fn gaze_rate_hz(&self) -> f64 { self.sample_rate_hz }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Simple xorshift64 for deterministic pseudo-random values.
fn xorshift(mut seed: u64) -> u64 {
    if seed == 0 { seed = 1; }
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    seed
}

/// Approximate Gaussian using Box-Muller with xorshift.
fn pseudo_gaussian(seed: u64) -> f64 {
    let u1 = (xorshift(seed) as f64) / (u64::MAX as f64);
    let u2 = (xorshift(seed.wrapping_add(1)) as f64) / (u64::MAX as f64);
    let u1 = u1.max(1e-30); // avoid log(0)
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_source_produces_frames() {
        let mut src = SimulatedEegSource::new(64, 256.0, 1);
        let frame = src.next_frame();
        assert_eq!(frame.channels.len(), 64);
        assert_eq!(frame.sequence_id, 0);
        assert!(frame.timestamp_ns > 0);
    }

    #[test]
    fn test_eeg_source_sequential_ids() {
        let mut src = SimulatedEegSource::new(4, 256.0, 1);
        for i in 0..10 {
            let frame = src.next_frame();
            assert_eq!(frame.sequence_id, i);
        }
    }

    #[test]
    fn test_gaze_source_produces_frames() {
        let mut src = SimulatedGazeSource::new(60.0);
        let frame = src.next_frame();
        assert!(frame.timestamp_ns > 0);
        assert!(frame.pupil_diameter > 0.0);
    }

    #[test]
    fn test_gaze_source_saccade_occurs() {
        let mut src = SimulatedGazeSource::new(60.0);
        let mut positions = Vec::new();
        for _ in 0..200 {
            let frame = src.next_frame();
            positions.push((frame.x, frame.y));
        }
        // Should have at least one saccade in 200 frames
        let distinct_fixations: std::collections::HashSet<_> = positions.iter()
            .map(|(x, y)| ((*x * 10.0) as i32, (*y * 10.0) as i32))
            .collect();
        assert!(distinct_fixations.len() > 1, "should have saccadic movement");
    }
}
