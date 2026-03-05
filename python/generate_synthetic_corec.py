#!/usr/bin/env python3
"""Generate a synthetic .corec file for gateway end-to-end testing.

Creates synthetic EEG (128ch @ 500 Hz) + gaze (60 Hz) data that models
the Midas touch problem: both intent and observe trials produce saccades
toward the target, with intent encoded in SPN-like EEG modulation and
subtle gaze kinematic differences.

Usage:
    python python/generate_synthetic_corec.py --output data/synthetic_test.corec
    python python/generate_synthetic_corec.py --output data/synthetic_test.corec --n-trials 40 --duration 2.0
"""

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = b"CREC"
VERSION = 1


def generate_synthetic_corec(
    output_path: str,
    n_trials: int = 20,
    trial_duration_s: float = 1.0,
    n_eeg_channels: int = 128,
    eeg_rate_hz: float = 500.0,
    gaze_rate_hz: float = 60.0,
    seed: int = 42,
) -> dict:
    """Generate synthetic EEG+gaze data and write to .corec format.

    Returns a dict with metadata about the generated file.
    """
    rng = np.random.RandomState(seed)

    samples_per_trial = int(trial_duration_s * eeg_rate_hz)
    gaze_per_trial = int(trial_duration_s * gaze_rate_hz)

    total_eeg_samples = n_trials * samples_per_trial
    total_gaze_samples = n_trials * gaze_per_trial

    # Pre-allocate arrays
    eeg_data = np.zeros((total_eeg_samples, n_eeg_channels), dtype=np.float32)
    eeg_timestamps = np.zeros(total_eeg_samples, dtype=np.uint64)
    gaze_x = np.zeros(total_gaze_samples, dtype=np.float32)
    gaze_y = np.zeros(total_gaze_samples, dtype=np.float32)
    gaze_pupil = np.zeros(total_gaze_samples, dtype=np.float32)
    gaze_timestamps = np.zeros(total_gaze_samples, dtype=np.uint64)

    eeg_dt_ns = int(1e9 / eeg_rate_hz)
    gaze_dt_ns = int(1e9 / gaze_rate_hz)

    labels = []

    for trial in range(n_trials):
        is_intent = trial % 2 == 0  # Alternate intent/observe
        labels.append(1 if is_intent else 0)

        eeg_start = trial * samples_per_trial
        gaze_start = trial * gaze_per_trial
        trial_start_ns = trial * int(trial_duration_s * 1e9)

        # --- EEG generation ---
        t = np.arange(samples_per_trial) / eeg_rate_hz

        for ch in range(n_eeg_channels):
            # 1/f background noise
            noise = rng.randn(samples_per_trial).astype(np.float32) * 15.0

            # Alpha rhythm (8-12 Hz) — present in both conditions
            alpha_freq = 10.0 + rng.randn() * 0.5
            alpha = np.sin(2 * np.pi * alpha_freq * t) * 20.0

            # Beta rhythm (13-30 Hz)
            beta_freq = 20.0 + rng.randn() * 2.0
            beta = np.sin(2 * np.pi * beta_freq * t) * 8.0

            signal = noise + alpha + beta

            # SPN signature for intent trials (occipitoparietal channels: ~80-110)
            if is_intent and 80 <= ch <= 110:
                # Pre-stimulus negativity: ramp from 0 to -8µV in last 250ms
                spn_window = int(0.5 * eeg_rate_hz)  # 250 samples at 500Hz
                spn = np.zeros(samples_per_trial)
                spn[-spn_window:] = np.linspace(0, -8.0, spn_window)
                signal += spn

                # ERD: reduced alpha power in last 300ms
                erd_start = samples_per_trial - int(0.3 * eeg_rate_hz)
                signal[erd_start:] *= 0.7

            eeg_data[eeg_start:eeg_start + samples_per_trial, ch] = signal.astype(np.float32)

        # EEG timestamps
        eeg_timestamps[eeg_start:eeg_start + samples_per_trial] = (
            trial_start_ns + np.arange(samples_per_trial, dtype=np.uint64) * eeg_dt_ns
        )

        # --- Gaze generation ---
        # Both conditions: saccade toward target, then fixation (Midas touch)
        target_x = rng.uniform(-10, 10)
        target_y = rng.uniform(-7, 7)

        for g in range(gaze_per_trial):
            gi = gaze_start + g
            t_frac = g / gaze_per_trial

            if t_frac < 0.15:
                # Initial fixation at center
                gaze_x[gi] = rng.randn() * 0.3
                gaze_y[gi] = rng.randn() * 0.3
            elif t_frac < 0.25:
                # Saccade to target (both conditions)
                progress = (t_frac - 0.15) / 0.1
                gaze_x[gi] = progress * target_x + rng.randn() * 0.2
                gaze_y[gi] = progress * target_y + rng.randn() * 0.2
            else:
                # Fixation on target
                jitter = 0.15 if is_intent else 0.25  # Intent: tighter fixation
                gaze_x[gi] = target_x + rng.randn() * jitter
                gaze_y[gi] = target_y + rng.randn() * jitter

            # Pupil: slightly dilated for intent (cognitive load)
            base_pupil = 3.5 if is_intent else 3.2
            gaze_pupil[gi] = base_pupil + rng.randn() * 0.3

            gaze_timestamps[gi] = trial_start_ns + g * gaze_dt_ns

    # Write .corec file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Header (40 bytes)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<d", eeg_rate_hz))
        f.write(struct.pack("<I", n_eeg_channels))
        f.write(struct.pack("<d", gaze_rate_hz))
        f.write(struct.pack("<I", total_eeg_samples))
        f.write(struct.pack("<I", total_gaze_samples))
        f.write(b"\x00" * 4)  # padding

        # EEG section
        for i in range(total_eeg_samples):
            f.write(struct.pack("<Q", int(eeg_timestamps[i])))
            f.write(eeg_data[i].tobytes())

        # Gaze section
        for i in range(total_gaze_samples):
            f.write(struct.pack("<Q", int(gaze_timestamps[i])))
            f.write(struct.pack("<fff",
                                float(gaze_x[i]),
                                float(gaze_y[i]),
                                float(gaze_pupil[i])))

    file_size = output_path.stat().st_size
    size_mb = file_size / (1024 * 1024)

    meta = {
        "file": str(output_path),
        "size_mb": round(size_mb, 2),
        "n_trials": n_trials,
        "labels": labels,
        "n_intent": sum(labels),
        "n_observe": len(labels) - sum(labels),
        "eeg_channels": n_eeg_channels,
        "eeg_rate_hz": eeg_rate_hz,
        "gaze_rate_hz": gaze_rate_hz,
        "total_eeg_frames": total_eeg_samples,
        "total_gaze_frames": total_gaze_samples,
        "duration_s": n_trials * trial_duration_s,
    }

    print(f"Generated {output_path}")
    print(f"  Trials:      {n_trials} ({meta['n_intent']} intent, {meta['n_observe']} observe)")
    print(f"  EEG:         {n_eeg_channels}ch @ {eeg_rate_hz} Hz, {total_eeg_samples} frames")
    print(f"  Gaze:        3ch @ {gaze_rate_hz} Hz, {total_gaze_samples} frames")
    print(f"  Duration:    {meta['duration_s']:.1f}s")
    print(f"  File size:   {size_mb:.2f} MB")

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic .corec file for gateway testing")
    parser.add_argument(
        "--output", type=str, default="data/synthetic_test.corec",
        help="Output .corec file path")
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of trials (alternating intent/observe)")
    parser.add_argument(
        "--duration", type=float, default=1.0,
        help="Duration per trial in seconds")
    parser.add_argument(
        "--channels", type=int, default=128,
        help="Number of EEG channels")
    parser.add_argument(
        "--eeg-rate", type=float, default=500.0,
        help="EEG sample rate in Hz")
    parser.add_argument(
        "--gaze-rate", type=float, default=60.0,
        help="Gaze sample rate in Hz")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility")

    args = parser.parse_args()

    generate_synthetic_corec(
        output_path=args.output,
        n_trials=args.n_trials,
        trial_duration_s=args.duration,
        n_eeg_channels=args.channels,
        eeg_rate_hz=args.eeg_rate,
        gaze_rate_hz=args.gaze_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
