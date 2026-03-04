"""Validate ONNX model against PyTorch model output.

Runs N random inputs through both PyTorch and ONNX Runtime and
asserts numerical equivalence.

Usage:
    python python/validate_onnx.py --onnx models/conv1d_fused.onnx --n-tests 100
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort

# Import from export script
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from export_onnx import Conv1DFusion

import torch


def validate(onnx_path: str, n_tests: int = 100, n_eeg_ch: int = 128,
             n_gaze_ch: int = 3, n_samples: int = 500):
    """Compare PyTorch and ONNX Runtime outputs."""

    model = Conv1DFusion(n_eeg_ch, n_gaze_ch, n_samples)
    model.eval()

    session = ort.InferenceSession(onnx_path)

    max_diff = 0.0

    for i in range(n_tests):
        eeg = np.random.randn(1, n_eeg_ch, n_samples).astype(np.float32)
        gaze = np.random.randn(1, n_gaze_ch, n_samples).astype(np.float32)

        # PyTorch
        with torch.no_grad():
            pt_out = model(
                torch.from_numpy(eeg),
                torch.from_numpy(gaze),
            ).numpy()

        # ONNX Runtime
        onnx_out = session.run(None, {"eeg": eeg, "gaze": gaze})[0]

        diff = np.abs(pt_out - onnx_out).max()
        max_diff = max(max_diff, diff)

        np.testing.assert_allclose(pt_out, onnx_out, rtol=1e-4, atol=1e-5,
                                   err_msg=f"Mismatch at test {i}")

    print(f"Validation passed: {n_tests} tests, max diff = {max_diff:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX vs PyTorch")
    parser.add_argument("--onnx", type=str, default="models/conv1d_fused.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--n-tests", type=int, default=100)
    args = parser.parse_args()

    validate(args.onnx, args.n_tests)


if __name__ == "__main__":
    main()
