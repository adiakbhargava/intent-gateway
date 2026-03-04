#!/usr/bin/env python3
"""Convert EEGEyeNet and EEGET-ALS datasets to co-replay binary format.

Binary format (.corec):
  Header (40 bytes, 36 data + 4 padding):
    magic:           4 bytes  b"CREC"
    version:         u32 LE
    sample_rate_hz:  f64 LE
    eeg_channels:    u32 LE
    gaze_rate_hz:    f64 LE
    num_eeg_frames:  u32 LE
    num_gaze_frames: u32 LE

  EEG section:
    For each frame:
      timestamp_ns:  u64 LE
      channels:      [f32 LE; eeg_channels]

  Gaze section (immediately after EEG):
    For each frame:
      timestamp_ns:  u64 LE
      x:             f32 LE
      y:             f32 LE
      pupil_diameter: f32 LE

Usage:
  # EEGEyeNet (OpenNeuro BIDS format, .set/.fdt files)
  python convert_dataset.py eegeyenet --input /path/to/sub-001/ --output sub001.corec

  # EEGEyeNet preprocessed (.npz from OSF direct download)
  python convert_dataset.py eegeyenet-npz --input Position_task.npz --output position.corec

  # EEGET-ALS (Figshare download, .edf + .csv files)
  python convert_dataset.py als --input /path/to/participant/scenario/ --output als_p01_s01.corec
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"CREC"
VERSION = 1


def write_header(f, sample_rate_hz, eeg_channels, gaze_rate_hz,
                 num_eeg_frames, num_gaze_frames):
    """Write 40-byte binary header (36 bytes data + 4 bytes padding)."""
    f.write(MAGIC)
    f.write(struct.pack("<I", VERSION))
    f.write(struct.pack("<d", sample_rate_hz))
    f.write(struct.pack("<I", eeg_channels))
    f.write(struct.pack("<d", gaze_rate_hz))
    f.write(struct.pack("<I", num_eeg_frames))
    f.write(struct.pack("<I", num_gaze_frames))
    f.write(b"\x00" * 4)  # padding to 40 bytes


def write_eeg_frames(f, timestamps_ns, data):
    """Write EEG frames: [u64 timestamp, f32*channels] per frame."""
    for i in range(len(timestamps_ns)):
        f.write(struct.pack("<Q", int(timestamps_ns[i])))
        row = data[i].astype(np.float32)
        f.write(row.tobytes())


def write_gaze_frames(f, timestamps_ns, x, y, pupil):
    """Write gaze frames: [u64 timestamp, f32 x, f32 y, f32 pupil] per frame."""
    for i in range(len(timestamps_ns)):
        f.write(struct.pack("<Q", int(timestamps_ns[i])))
        f.write(struct.pack("<fff",
                            float(x[i]), float(y[i]), float(pupil[i])))


# ---------------------------------------------------------------------------
# EEGEyeNet (OpenNeuro BIDS .set/.fdt)
# ---------------------------------------------------------------------------

def convert_eegeyenet_bids(input_path, output_path, max_seconds=None):
    """Convert EEGEyeNet BIDS files using MNE-Python."""
    try:
        import mne
    except ImportError:
        print("ERROR: pip install mne", file=sys.stderr)
        sys.exit(1)

    input_path = Path(input_path)

    # Find .set files
    set_files = sorted(input_path.rglob("*.set"))
    if not set_files:
        print(f"ERROR: No .set files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(set_files)} .set files, using first: {set_files[0]}")

    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)

    if max_seconds:
        raw.crop(tmax=max_seconds)

    sfreq = raw.info["sfreq"]

    # Separate EEG and eye-tracking channels
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    misc_picks = mne.pick_types(raw.info, misc=True)

    eeg_data = raw.get_data(picks=eeg_picks)  # (n_eeg_ch, n_samples)
    n_eeg_ch, n_samples = eeg_data.shape

    # Try to find gaze channels
    ch_names = [raw.ch_names[i].lower() for i in misc_picks]
    gaze_x_idx = None
    gaze_y_idx = None
    pupil_idx = None

    for i, name in enumerate(ch_names):
        if "gaze" in name and ("x" in name or "left" in name):
            gaze_x_idx = i
        elif "gaze" in name and ("y" in name or "right" in name):
            gaze_y_idx = i
        elif "pupil" in name or "area" in name:
            pupil_idx = i

    has_gaze = gaze_x_idx is not None and gaze_y_idx is not None
    misc_data = raw.get_data(picks=misc_picks) if has_gaze else None

    # Generate timestamps (ns since epoch)
    base_ns = 0
    dt_ns = int(1e9 / sfreq)
    eeg_timestamps = np.arange(n_samples, dtype=np.uint64) * dt_ns + base_ns

    # EEG: transpose to (n_samples, n_channels)
    eeg_data_t = eeg_data.T.astype(np.float32)

    # Gaze: same rate as EEG in co-registered data
    if has_gaze:
        gaze_x = misc_data[gaze_x_idx].astype(np.float32)
        gaze_y = misc_data[gaze_y_idx].astype(np.float32)
        pupil = misc_data[pupil_idx].astype(np.float32) if pupil_idx else \
            np.full(n_samples, 3.0, dtype=np.float32)
        gaze_timestamps = eeg_timestamps.copy()
        n_gaze = n_samples
        gaze_rate = sfreq
    else:
        # No gaze channels found — generate placeholder at 60 Hz
        gaze_rate = 60.0
        gaze_dt_ns = int(1e9 / gaze_rate)
        total_duration_ns = n_samples * dt_ns
        n_gaze = int(total_duration_ns / gaze_dt_ns)
        gaze_timestamps = np.arange(n_gaze, dtype=np.uint64) * gaze_dt_ns
        gaze_x = np.zeros(n_gaze, dtype=np.float32)
        gaze_y = np.zeros(n_gaze, dtype=np.float32)
        pupil = np.full(n_gaze, 3.0, dtype=np.float32)

    print(f"EEG: {n_eeg_ch} channels, {n_samples} samples @ {sfreq} Hz")
    print(f"Gaze: {n_gaze} samples @ {gaze_rate} Hz (detected={has_gaze})")

    with open(output_path, "wb") as f:
        write_header(f, sfreq, n_eeg_ch, gaze_rate, n_samples, n_gaze)
        write_eeg_frames(f, eeg_timestamps, eeg_data_t)
        write_gaze_frames(f, gaze_timestamps, gaze_x, gaze_y, pupil)

    print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# EEGEyeNet preprocessed (.npz from OSF)
# ---------------------------------------------------------------------------

def convert_eegeyenet_npz(input_path, output_path):
    """Convert EEGEyeNet preprocessed .npz files."""
    data = np.load(input_path, allow_pickle=True)

    print("Keys in .npz:", list(data.keys()))

    # Typical keys: 'EEG', 'labels', 'info' or similar
    # The EEG key contains (n_trials, n_channels, n_timepoints)
    eeg_key = None
    for k in data.keys():
        if "eeg" in k.lower() or data[k].ndim == 3:
            eeg_key = k
            break

    if eeg_key is None:
        # Fall back to first 3D array
        for k in data.keys():
            if data[k].ndim >= 2:
                eeg_key = k
                break

    if eeg_key is None:
        print("ERROR: Could not find EEG data in .npz", file=sys.stderr)
        sys.exit(1)

    eeg_raw = data[eeg_key]
    print(f"EEG array '{eeg_key}': shape {eeg_raw.shape}")

    if eeg_raw.ndim == 3:
        # (n_trials, n_channels, n_timepoints) -> flatten trials
        n_trials, n_ch, n_tp = eeg_raw.shape
        eeg_flat = eeg_raw.transpose(0, 2, 1).reshape(-1, n_ch)
        n_samples = eeg_flat.shape[0]
    elif eeg_raw.ndim == 2:
        eeg_flat = eeg_raw
        n_ch = eeg_flat.shape[1]
        n_samples = eeg_flat.shape[0]
    else:
        print(f"ERROR: Unexpected EEG shape: {eeg_raw.shape}", file=sys.stderr)
        sys.exit(1)

    sfreq = 128.0  # EEGEyeNet default
    dt_ns = int(1e9 / sfreq)
    timestamps = np.arange(n_samples, dtype=np.uint64) * dt_ns

    # Generate synthetic gaze at 60 Hz (eye tracking data in labels, not .npz EEG)
    gaze_rate = 60.0
    total_ns = n_samples * dt_ns
    n_gaze = int(total_ns / int(1e9 / gaze_rate))
    gaze_ts = np.arange(n_gaze, dtype=np.uint64) * int(1e9 / gaze_rate)

    # Try to extract gaze from labels
    label_key = None
    for k in data.keys():
        if "label" in k.lower() or "position" in k.lower():
            label_key = k
            break

    if label_key is not None and data[label_key].ndim >= 1:
        labels = data[label_key]
        print(f"Labels '{label_key}': shape {labels.shape}")
        # Resample labels to gaze rate
        if labels.ndim == 2 and labels.shape[1] >= 2:
            indices = np.linspace(0, len(labels) - 1, n_gaze).astype(int)
            gaze_x = labels[indices, 0].astype(np.float32)
            gaze_y = labels[indices, 1].astype(np.float32)
        else:
            gaze_x = np.zeros(n_gaze, dtype=np.float32)
            gaze_y = np.zeros(n_gaze, dtype=np.float32)
    else:
        gaze_x = np.zeros(n_gaze, dtype=np.float32)
        gaze_y = np.zeros(n_gaze, dtype=np.float32)

    pupil = np.full(n_gaze, 3.0, dtype=np.float32)

    print(f"EEG: {n_ch} channels, {n_samples} samples @ {sfreq} Hz")
    print(f"Gaze: {n_gaze} samples @ {gaze_rate} Hz")

    with open(output_path, "wb") as f:
        write_header(f, sfreq, n_ch, gaze_rate, n_samples, n_gaze)
        write_eeg_frames(f, timestamps, eeg_flat.astype(np.float32))
        write_gaze_frames(f, gaze_ts, gaze_x, gaze_y, pupil)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Written to {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# EEGET-ALS (.edf + .csv)
# ---------------------------------------------------------------------------

def convert_als(input_path, output_path):
    """Convert EEGET-ALS scenario folder (.edf EEG + .csv eye tracking)."""
    try:
        import mne
    except ImportError:
        print("ERROR: pip install mne", file=sys.stderr)
        sys.exit(1)

    input_path = Path(input_path)

    # Find .edf and .csv files
    edf_files = sorted(input_path.rglob("*.edf"))
    csv_files = sorted(input_path.rglob("*.csv"))

    if not edf_files:
        print(f"ERROR: No .edf files found in {input_path}", file=sys.stderr)
        sys.exit(1)
    if not csv_files:
        print(f"ERROR: No .csv files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"EEG: {edf_files[0]}")
    print(f"ET:  {csv_files[0]}")

    # Load EEG
    raw = mne.io.read_raw_edf(str(edf_files[0]), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]  # 128 Hz for Emotiv EPOC Flex
    eeg_data = raw.get_data()  # (n_channels, n_samples)
    n_ch, n_samples = eeg_data.shape
    eeg_data_t = eeg_data.T.astype(np.float32)

    # Generate EEG timestamps
    dt_ns = int(1e9 / sfreq)
    eeg_timestamps = np.arange(n_samples, dtype=np.uint64) * dt_ns

    # Load eye tracking CSV
    import csv
    gaze_records = []
    with open(csv_files[0], "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            try:
                ts = float(row[0])
                x = float(row[1])
                y = float(row[2])
                gaze_records.append((ts, x, y))
            except (ValueError, IndexError):
                continue

    if not gaze_records:
        print("WARNING: No valid gaze records found, generating placeholder")
        gaze_rate = 30.0
        n_gaze = int(n_samples * gaze_rate / sfreq)
        gaze_ts = np.arange(n_gaze, dtype=np.uint64) * int(1e9 / gaze_rate)
        gaze_x = np.zeros(n_gaze, dtype=np.float32)
        gaze_y = np.zeros(n_gaze, dtype=np.float32)
    else:
        gaze_rate = 30.0  # Tobii Eye Tracker 4C at 30 Hz
        gaze_arr = np.array(gaze_records)
        # Convert Unix timestamps to nanoseconds relative to start
        ts_start = gaze_arr[0, 0]
        gaze_ts = ((gaze_arr[:, 0] - ts_start) * 1e9).astype(np.uint64)
        gaze_x = gaze_arr[:, 1].astype(np.float32)
        gaze_y = gaze_arr[:, 2].astype(np.float32)
        n_gaze = len(gaze_records)

    pupil = np.full(n_gaze, 3.0, dtype=np.float32)

    print(f"EEG: {n_ch} channels, {n_samples} samples @ {sfreq} Hz")
    print(f"Gaze: {n_gaze} samples @ {gaze_rate} Hz")

    with open(output_path, "wb") as f:
        write_header(f, sfreq, n_ch, gaze_rate, n_samples, n_gaze)
        write_eeg_frames(f, eeg_timestamps, eeg_data_t)
        write_gaze_frames(f, gaze_ts, gaze_x, gaze_y, pupil)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Written to {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# EEGET-RSOD (.edf EEG + .txt eye tracking, 32ch @ 500Hz + 250Hz gaze)
# ---------------------------------------------------------------------------

def convert_rsod(input_path, output_path, max_seconds=None):
    """Convert EEGET-RSOD participant folder (.edf EEG + .txt eye tracking).

    Expected structure:
      <participant_id>/
        *.edf           # 32-channel EEG @ 500 Hz (NE Enobio 32)
        *.txt           # Eye tracking @ 250 Hz (SMI RED250)

    The eye-tracking TXT files have columns including:
      Time, Type, ..., L POR X [px], L POR Y [px], R POR X [px], R POR Y [px],
      L Mapped Diameter [mm], R Mapped Diameter [mm], ...
    """
    try:
        import mne
    except ImportError:
        print("ERROR: pip install mne", file=sys.stderr)
        sys.exit(1)

    input_path = Path(input_path)

    # Find .edf and .txt files
    edf_files = sorted(input_path.rglob("*.edf"))
    txt_files = sorted(input_path.rglob("*.txt"))

    if not edf_files:
        print(f"ERROR: No .edf files found in {input_path}", file=sys.stderr)
        sys.exit(1)

    # Pick the first EDF and a matching TXT
    edf_file = edf_files[0]
    print(f"EEG: {edf_file}")

    # Load EEG
    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
    if max_seconds:
        raw.crop(tmax=max_seconds)

    sfreq = raw.info["sfreq"]
    eeg_data = raw.get_data()  # (n_channels, n_samples)
    n_ch, n_samples = eeg_data.shape
    eeg_data_t = eeg_data.T.astype(np.float32)

    # Generate EEG timestamps
    dt_ns = int(1e9 / sfreq)
    eeg_timestamps = np.arange(n_samples, dtype=np.uint64) * dt_ns

    print(f"EEG: {n_ch} channels, {n_samples} samples @ {sfreq} Hz "
          f"({n_samples / sfreq:.1f}s)")

    # Load eye tracking
    gaze_x_list = []
    gaze_y_list = []
    pupil_list = []
    gaze_ts_list = []

    if txt_files:
        txt_file = txt_files[0]
        print(f"Eye tracking: {txt_file}")

        with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
            header_line = None
            for line in f:
                line = line.strip()
                if not line or line.startswith("##"):
                    continue
                # First non-comment, non-empty line is the header
                if header_line is None:
                    header_line = line
                    # SMI BeGaze exports use comma-separated format
                    cols = header_line.split(",")
                    # Find relevant column indices
                    time_col = None
                    type_col = None
                    lx_col = None
                    ly_col = None
                    rx_col = None
                    ry_col = None
                    lpupil_col = None

                    for i, col in enumerate(cols):
                        cl = col.strip().lower()
                        if cl == "time":
                            time_col = i
                        elif cl == "type":
                            type_col = i
                        elif "l por x" in cl:
                            lx_col = i
                        elif "l por y" in cl:
                            ly_col = i
                        elif "r por x" in cl:
                            rx_col = i
                        elif "r por y" in cl:
                            ry_col = i
                        elif "l pupil diameter" in cl:
                            lpupil_col = i

                    if time_col is None:
                        print("WARNING: Could not find Time column in eye tracking")
                        break
                    print(f"  Columns found: time={time_col}, type={type_col}, "
                          f"lx={lx_col}, ly={ly_col}, pupil={lpupil_col}")
                    continue

                # Data line (comma-separated)
                parts = line.split(",")
                if type_col is not None and len(parts) > type_col:
                    if parts[type_col].strip().upper() != "SMP":
                        continue

                try:
                    ts_us = float(parts[time_col])
                    ts_ns = int(ts_us * 1000)  # microseconds to nanoseconds

                    # Prefer left eye POR, fallback to right
                    x = 0.0
                    y = 0.0
                    pupil = 3.0

                    if lx_col is not None and len(parts) > lx_col:
                        val = parts[lx_col].strip()
                        if val and val != "-" and val != ".":
                            x = float(val)
                    elif rx_col is not None and len(parts) > rx_col:
                        val = parts[rx_col].strip()
                        if val and val != "-" and val != ".":
                            x = float(val)

                    if ly_col is not None and len(parts) > ly_col:
                        val = parts[ly_col].strip()
                        if val and val != "-" and val != ".":
                            y = float(val)
                    elif ry_col is not None and len(parts) > ry_col:
                        val = parts[ry_col].strip()
                        if val and val != "-" and val != ".":
                            y = float(val)

                    if lpupil_col is not None and len(parts) > lpupil_col:
                        val = parts[lpupil_col].strip()
                        if val and val != "-" and val != ".":
                            pupil = float(val)

                    gaze_ts_list.append(ts_ns)
                    gaze_x_list.append(x)
                    gaze_y_list.append(y)
                    pupil_list.append(pupil)
                except (ValueError, IndexError):
                    continue

    if gaze_ts_list:
        gaze_timestamps = np.array(gaze_ts_list, dtype=np.uint64)
        # Make timestamps relative to first sample
        gaze_timestamps -= gaze_timestamps[0]
        gaze_x = np.array(gaze_x_list, dtype=np.float32)
        gaze_y = np.array(gaze_y_list, dtype=np.float32)
        pupil_arr = np.array(pupil_list, dtype=np.float32)
        n_gaze = len(gaze_ts_list)
        gaze_rate = 250.0  # SMI RED250 nominal rate

        if max_seconds:
            max_ns = int(max_seconds * 1e9)
            mask = gaze_timestamps <= max_ns
            gaze_timestamps = gaze_timestamps[mask]
            gaze_x = gaze_x[mask]
            gaze_y = gaze_y[mask]
            pupil_arr = pupil_arr[mask]
            n_gaze = len(gaze_timestamps)
    else:
        print("WARNING: No eye tracking data found, generating placeholder")
        gaze_rate = 250.0
        total_ns = n_samples * dt_ns
        n_gaze = int(total_ns / int(1e9 / gaze_rate))
        gaze_timestamps = np.arange(n_gaze, dtype=np.uint64) * int(1e9 / gaze_rate)
        gaze_x = np.zeros(n_gaze, dtype=np.float32)
        gaze_y = np.zeros(n_gaze, dtype=np.float32)
        pupil_arr = np.full(n_gaze, 3.0, dtype=np.float32)

    print(f"Gaze: {n_gaze} samples @ {gaze_rate} Hz")

    with open(output_path, "wb") as f:
        write_header(f, sfreq, n_ch, gaze_rate, n_samples, n_gaze)
        write_eeg_frames(f, eeg_timestamps, eeg_data_t)
        write_gaze_frames(f, gaze_timestamps, gaze_x, gaze_y, pupil_arr)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Written to {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert EEG+gaze datasets to co-replay binary format (.corec)")
    sub = parser.add_subparsers(dest="command", required=True)

    # EEGEyeNet BIDS
    p1 = sub.add_parser("eegeyenet", help="EEGEyeNet BIDS .set/.fdt files")
    p1.add_argument("--input", required=True, help="Path to subject folder")
    p1.add_argument("--output", required=True, help="Output .corec file")
    p1.add_argument("--max-seconds", type=float, default=None,
                    help="Crop recording to N seconds")

    # EEGEyeNet preprocessed
    p2 = sub.add_parser("eegeyenet-npz", help="EEGEyeNet preprocessed .npz")
    p2.add_argument("--input", required=True, help="Path to .npz file")
    p2.add_argument("--output", required=True, help="Output .corec file")

    # EEGET-ALS
    p3 = sub.add_parser("als", help="EEGET-ALS .edf + .csv files")
    p3.add_argument("--input", required=True, help="Path to scenario folder")
    p3.add_argument("--output", required=True, help="Output .corec file")

    # EEGET-RSOD
    p4 = sub.add_parser("rsod", help="EEGET-RSOD .edf + .txt files (32ch @ 500Hz)")
    p4.add_argument("--input", required=True,
                    help="Path to participant folder containing .edf and .txt")
    p4.add_argument("--output", required=True, help="Output .corec file")
    p4.add_argument("--max-seconds", type=float, default=None,
                    help="Crop recording to N seconds")

    args = parser.parse_args()

    if args.command == "eegeyenet":
        convert_eegeyenet_bids(args.input, args.output, args.max_seconds)
    elif args.command == "eegeyenet-npz":
        convert_eegeyenet_npz(args.input, args.output)
    elif args.command == "als":
        convert_als(args.input, args.output)
    elif args.command == "rsod":
        convert_rsod(args.input, args.output, args.max_seconds)


if __name__ == "__main__":
    main()
