#!/usr/bin/env python3
"""Batch benchmark runner for co-replay-bench."""
import subprocess
import re
import json
import glob
import os

results = []
BENCH_EXE = os.path.abspath(os.path.join("target", "release", "co-replay-bench.exe"))

# All .corec files except the short sub-EP10 version
files = sorted(glob.glob("data/*.corec"))
files = [f for f in files if not f.endswith("sub-EP10.corec")]

print(f"Running benchmarks on {len(files)} subjects...")

for i, f in enumerate(files):
    name = os.path.basename(f).replace(".corec", "")
    if name == "sub-EP10-full":
        name_display = "sub-EP10"
    else:
        name_display = name

    if name.startswith("rsod-"):
        dataset = "EEGET-RSOD"
        subj = name.replace("rsod-", "")
    elif name.startswith("als-"):
        dataset = "EEGET-ALS"
        subj = name.replace("als-", "")
    else:
        dataset = "EEGEyeNet"
        subj = name_display

    print(f"[{i+1}/{len(files)}] {dataset} {subj}...", end=" ", flush=True)

    try:
        r = subprocess.run(
            [BENCH_EXE, "--input", os.path.abspath(f)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        continue

    if r.returncode != 0:
        print(f"ERROR: {r.stderr[:200]}")
        continue

    out = r.stdout + r.stderr

    throughput = ""
    rt_factor = ""
    fusion_p50 = ""
    channels = ""
    rate = ""
    duration = ""
    feature_p50 = ""

    for line in out.split("\n"):
        if "Throughput" in line:
            m = re.search(r"([\d,]+)\s*frames/sec", line)
            if m:
                throughput = m.group(1).replace(",", "")
        if "real-time" in line.lower() or "Real-time" in line:
            m = re.search(r"([\d,.]+)x", line)
            if m:
                rt_factor = m.group(1).replace(",", "")
        if "Fusion" in line and "p50" in line:
            m = re.search(r"p50[:\s]+([\d.]+)\s*(ns|us|µs|ms)", line)
            if m:
                fusion_p50 = f"{m.group(1)} {m.group(2)}"
        if "Feature" in line and "p50" in line:
            m = re.search(r"p50[:\s]+([\d.]+)\s*(ns|us|µs|ms)", line)
            if m:
                feature_p50 = f"{m.group(1)} {m.group(2)}"
        if "channels" in line.lower() and "@" in line:
            m = re.search(r"(\d+)\s*ch.*?@\s*([\d.]+)\s*Hz", line)
            if m:
                channels = m.group(1)
                rate = m.group(2)
        if "uration" in line:
            m = re.search(r"([\d.]+)\s*s", line)
            if m:
                duration = m.group(1)

    result = {
        "subject": subj,
        "dataset": dataset,
        "channels": channels,
        "rate": rate,
        "duration": duration,
        "throughput": throughput,
        "rt_factor": rt_factor,
        "fusion_p50": fusion_p50,
        "feature_p50": feature_p50,
    }
    results.append(result)

    tp_k = int(throughput) // 1000 if throughput.isdigit() else throughput
    print(f"{tp_k}K fps, {rt_factor}x RT")

with open("data/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. {len(results)} subjects benchmarked.")
print(f"Results saved to data/benchmark_results.json")
