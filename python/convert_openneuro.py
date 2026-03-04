#!/usr/bin/env python3
"""Convert EEGEyeNet OpenNeuro BIDS data (EDF + physio TSV) to .corec format.

Usage:
  python convert_openneuro.py --edf data/eeg.edf --physio data/physio.tsv --output sub001.corec
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"CREC"
VERSION = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edf", required=True, help="Path to .edf EEG file")
    parser.add_argument("--physio", required=True, help="Path to physio.tsv (eye tracking)")
    parser.add_argument("--output", required=True, help="Output .corec file")
    parser.add_argument("--max-seconds", type=float, default=None, help="Crop to N seconds")
    args = parser.parse_args()

    try:
        import mne
    except ImportError:
        print("ERROR: pip install mne", file=sys.stderr)
        sys.exit(1)

    # Load EEG from EDF
    print(f"Loading EEG from {args.edf}...")
    raw = mne.io.read_raw_edf(args.edf, preload=True, verbose=False)
    if args.max_seconds:
        raw.crop(tmax=args.max_seconds)

    sfreq = raw.info["sfreq"]
    eeg_data = raw.get_data()  # (n_channels, n_samples)
    n_ch, n_samples = eeg_data.shape
    print(f"EEG: {n_ch} channels, {n_samples} samples @ {sfreq} Hz "
          f"({n_samples / sfreq:.1f}s)")

    # Load eye tracking from physio.tsv
    print(f"Loading eye tracking from {args.physio}...")
    physio = np.loadtxt(args.physio, skiprows=1, delimiter='\t')
    # Columns: time, L-GAZE-X, L-GAZE-Y
    gaze_time = physio[:, 0]  # seconds
    gaze_x = physio[:, 1].astype(np.float32)
    gaze_y = physio[:, 2].astype(np.float32)

    if args.max_seconds:
        mask = gaze_time <= args.max_seconds
        gaze_time = gaze_time[mask]
        gaze_x = gaze_x[mask]
        gaze_y = gaze_y[mask]

    n_gaze = len(gaze_time)
    gaze_rate = 1.0 / np.median(np.diff(gaze_time)) if n_gaze > 1 else sfreq
    pupil = np.full(n_gaze, 3.0, dtype=np.float32)  # no pupil in this dataset

    print(f"Gaze: {n_gaze} samples @ {gaze_rate:.1f} Hz")

    # Generate timestamps in nanoseconds
    dt_ns = int(1e9 / sfreq)
    eeg_timestamps = np.arange(n_samples, dtype=np.uint64) * dt_ns
    gaze_timestamps = (gaze_time * 1e9).astype(np.uint64)

    # Transpose EEG to (n_samples, n_channels)
    eeg_data_t = eeg_data.T.astype(np.float32)

    # Write .corec
    print(f"Writing {args.output}...")
    with open(args.output, "wb") as f:
        # Header (40 bytes)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<d", sfreq))
        f.write(struct.pack("<I", n_ch))
        f.write(struct.pack("<d", gaze_rate))
        f.write(struct.pack("<I", n_samples))
        f.write(struct.pack("<I", n_gaze))
        f.write(b"\x00" * 4)  # padding to 40 bytes

        # EEG frames
        for i in range(n_samples):
            f.write(struct.pack("<Q", int(eeg_timestamps[i])))
            f.write(eeg_data_t[i].tobytes())

        # Gaze frames
        for i in range(n_gaze):
            f.write(struct.pack("<Q", int(gaze_timestamps[i])))
            f.write(struct.pack("<fff", float(gaze_x[i]), float(gaze_y[i]),
                                float(pupil[i])))

    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Done: {size_mb:.1f} MB "
          f"({n_ch}ch x {n_samples} EEG + {n_gaze} gaze frames)")


if __name__ == "__main__":
    main()
