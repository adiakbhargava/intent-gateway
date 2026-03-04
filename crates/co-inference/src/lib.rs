//! ONNX Runtime model wrapper for intent classification.
//!
//! Wraps an ONNX model that takes EEG + gaze windows and produces
//! an intent confidence score.

use std::path::Path;
use std::time::Instant;
use ort::session::Session;

// ---------------------------------------------------------------------------
// IntentClassifier
// ---------------------------------------------------------------------------

pub struct IntentClassifier {
    session: Session,
    n_eeg_channels: usize,
    n_samples: usize,
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub confidence: f32,
    pub classification: &'static str,
    pub inference_ns: u64,
}

impl IntentClassifier {
    /// Load an ONNX model from disk.
    pub fn load(model_path: impl AsRef<Path>) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        Ok(IntentClassifier {
            session,
            n_eeg_channels: 128,
            n_samples: 500,
        })
    }

    /// Run inference on EEG and gaze windows.
    ///
    /// `eeg_window`: &[f32] of length `n_eeg_channels * n_samples` (128 * 500)
    /// `gaze_window`: &[f32] of length `3 * n_samples` (3 * 500)
    pub fn predict(&mut self, eeg_window: &[f32], gaze_window: &[f32]) -> ort::Result<Prediction> {
        let start = Instant::now();

        let eeg_shape = [1usize, self.n_eeg_channels, self.n_samples];
        let eeg_value = ort::value::Tensor::from_array(
            (eeg_shape.as_slice(), eeg_window.to_vec())
        )?;

        let gaze_shape = [1usize, 3, self.n_samples];
        let gaze_value = ort::value::Tensor::from_array(
            (gaze_shape.as_slice(), gaze_window.to_vec())
        )?;

        let outputs = self.session.run(ort::inputs![eeg_value, gaze_value])?;

        let (_, raw_data) = outputs[0].try_extract_tensor::<f32>()?;
        let logit = raw_data[0];

        let confidence = sigmoid(logit);
        let classification = if confidence >= 0.5 { "intent" } else { "observe" };

        let inference_ns = start.elapsed().as_nanos() as u64;

        Ok(Prediction {
            confidence,
            classification,
            inference_ns,
        })
    }

    pub fn n_eeg_channels(&self) -> usize { self.n_eeg_channels }
    pub fn n_samples(&self) -> usize { self.n_samples }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// InferenceAccumulator
// ---------------------------------------------------------------------------

/// Accumulates fused frames into sliding windows for model inference.
pub struct InferenceAccumulator {
    eeg_ring: Vec<f32>,
    gaze_ring: Vec<f32>,
    eeg_capacity: usize,
    gaze_capacity: usize,
    pos: usize,
    filled: bool,
    stride: usize,
    frame_count: usize,
}

impl InferenceAccumulator {
    pub fn new(n_eeg_ch: usize, n_gaze_ch: usize, n_samples: usize, stride: usize) -> Self {
        let eeg_capacity = n_eeg_ch * n_samples;
        let gaze_capacity = n_gaze_ch * n_samples;
        InferenceAccumulator {
            eeg_ring: vec![0.0f32; eeg_capacity],
            gaze_ring: vec![0.0f32; gaze_capacity],
            eeg_capacity,
            gaze_capacity,
            pos: 0,
            filled: false,
            stride,
            frame_count: 0,
        }
    }

    /// Push one timestep of EEG channels and gaze data.
    pub fn push(&mut self, eeg: &[f32], gaze_x: f32, gaze_y: f32, pupil: f32) {
        // Write EEG channels at current position
        let n_ch = eeg.len();
        for (c, &val) in eeg.iter().enumerate() {
            let idx = self.pos * n_ch + c;
            if idx < self.eeg_capacity {
                self.eeg_ring[idx] = val;
            }
        }

        // Write gaze at current position
        let gaze_idx = self.pos * 3;
        if gaze_idx + 2 < self.gaze_capacity {
            self.gaze_ring[gaze_idx] = gaze_x;
            self.gaze_ring[gaze_idx + 1] = gaze_y;
            self.gaze_ring[gaze_idx + 2] = pupil;
        }

        let n_samples = self.eeg_capacity / n_ch.max(1);
        self.pos += 1;
        if self.pos >= n_samples {
            self.pos = 0;
            self.filled = true;
        }

        self.frame_count += 1;
    }

    /// Returns true when enough frames have been accumulated and
    /// the stride interval has been reached.
    pub fn is_ready(&self) -> bool {
        self.filled && self.frame_count.is_multiple_of(self.stride)
    }

    pub fn eeg_window(&self) -> &[f32] {
        &self.eeg_ring
    }

    pub fn gaze_window(&self) -> &[f32] {
        &self.gaze_ring
    }
}

// ---------------------------------------------------------------------------
// Stub classifier for when no ONNX model is available
// ---------------------------------------------------------------------------

/// A stub classifier that returns random-ish predictions without ONNX.
/// Used for testing the pipeline without a trained model.
pub struct StubClassifier;

impl StubClassifier {
    pub fn predict(&self, _eeg: &[f32], _gaze: &[f32]) -> Prediction {
        Prediction {
            confidence: 0.5,
            classification: "observe",
            inference_ns: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_accumulator_not_ready() {
        let mut acc = InferenceAccumulator::new(128, 3, 500, 100);
        for _ in 0..499 {
            acc.push(&vec![0.0f32; 128], 0.0, 0.0, 3.0);
        }
        assert!(!acc.filled);
    }

    #[test]
    fn test_accumulator_ready_after_fill() {
        let mut acc = InferenceAccumulator::new(4, 3, 8, 1);
        for _ in 0..8 {
            acc.push(&vec![0.0f32; 4], 0.0, 0.0, 3.0);
        }
        assert!(acc.filled);
    }

    #[test]
    fn test_accumulator_stride() {
        let mut acc = InferenceAccumulator::new(4, 3, 8, 4);
        // Fill the window
        for _ in 0..8 {
            acc.push(&vec![0.0f32; 4], 0.0, 0.0, 3.0);
        }
        // After 8 pushes (frame_count=8), stride=4 → 8%4==0 → ready
        assert!(acc.is_ready());

        // Push one more — frame_count=9 → 9%4!=0 → not ready
        acc.push(&vec![0.0f32; 4], 0.0, 0.0, 3.0);
        assert!(!acc.is_ready());
    }

    #[test]
    fn test_stub_classifier() {
        let stub = StubClassifier;
        let pred = stub.predict(&[0.0; 64000], &[0.0; 1500]);
        assert_eq!(pred.classification, "observe");
        assert!((pred.confidence - 0.5).abs() < 1e-6);
    }
}
