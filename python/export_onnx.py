"""Export PyTorch Conv1DFusion model to ONNX format.

Usage:
    python python/export_onnx.py --model-path path/to/weights.pt --output models/conv1d_fused.onnx

If no model-path is provided, exports a randomly-initialized model
(useful for testing the Rust inference wrapper).
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


class Conv1DFusion(nn.Module):
    """EEG + Gaze fusion model using 1D convolutions."""

    def __init__(self, n_eeg_channels=128, n_gaze_channels=3, n_samples=500):
        super().__init__()

        # EEG branch
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(n_eeg_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Gaze branch
        self.gaze_conv = nn.Sequential(
            nn.Conv1d(n_gaze_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, eeg, gaze):
        eeg_feat = self.eeg_conv(eeg).squeeze(-1)   # (batch, 64)
        gaze_feat = self.gaze_conv(gaze).squeeze(-1) # (batch, 64)
        fused = torch.cat([eeg_feat, gaze_feat], dim=1)  # (batch, 128)
        return self.head(fused)  # (batch, 1)


def main():
    parser = argparse.ArgumentParser(description="Export Conv1DFusion to ONNX")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained PyTorch weights (.pt)")
    parser.add_argument("--output", type=str, default="models/conv1d_fused.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--n-eeg-channels", type=int, default=128)
    parser.add_argument("--n-gaze-channels", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    model = Conv1DFusion(args.n_eeg_channels, args.n_gaze_channels, args.n_samples)

    if args.model_path:
        state = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded weights from {args.model_path}")
    else:
        print("No weights provided — exporting randomly-initialized model")

    model.eval()

    dummy_eeg = torch.randn(1, args.n_eeg_channels, args.n_samples)
    dummy_gaze = torch.randn(1, args.n_gaze_channels, args.n_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_eeg, dummy_gaze),
        str(output_path),
        input_names=["eeg", "gaze"],
        output_names=["logit"],
        dynamic_axes={
            "eeg": {0: "batch"},
            "gaze": {0: "batch"},
            "logit": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"  EEG input:  (batch, {args.n_eeg_channels}, {args.n_samples})")
    print(f"  Gaze input: (batch, {args.n_gaze_channels}, {args.n_samples})")
    print(f"  Output:     (batch, 1)")


if __name__ == "__main__":
    main()
