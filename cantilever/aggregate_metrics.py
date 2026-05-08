"""Aggregate VI metrics across multiple result folders (v4, v5, v6)."""

import os
import glob
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = [
    os.path.join(RESULTS_DIR, "results_data_no_ic_v15"),
    os.path.join(RESULTS_DIR, "results_data_no_ic_v16"),
    os.path.join(RESULTS_DIR, "results_data_no_ic_v17"),
]

METRICS = [
    # "mae",
    # "rmse",
    "distance_to_boundary",
    # "distance_from_boundary",
    # "boundary_width",
    "width_vs_noise_ratio",
    "fraction_outside",
]


def parse_metrics(filepath):
    """Parse metric values from a vi_metrics.txt file."""
    metrics = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key in METRICS:
                metrics[key] = float(value.strip())
    return metrics


NOISE_TYPES = ["normal", "uniform", "beta"]


def get_noise_type(filepath):
    """Extract noise type from filename."""
    basename = os.path.basename(filepath)
    for noise in NOISE_TYPES:
        if f"_{noise}_" in basename:
            return noise
    return "unknown"


def main():
    # Collect metric values per noise type
    all_metrics = {noise: {m: [] for m in METRICS} for noise in NOISE_TYPES}

    for folder in FOLDERS:
        files = sorted(glob.glob(os.path.join(folder, "*_vi_metrics.txt")))
        for filepath in files:
            noise = get_noise_type(filepath)
            if noise not in all_metrics:
                continue
            metrics = parse_metrics(filepath)
            for m in METRICS:
                if m in metrics:
                    all_metrics[noise][m].append(metrics[m])

    # Compute and print mean/std per noise type
    for noise in NOISE_TYPES:
        print(f"\nNoise type: {noise}")
        print(f"{'Metric':<25} {'Mean':>12} {'Std':>12}  {'N':>4}")
        print("-" * 57)
        for m in METRICS:
            values = np.array(all_metrics[noise][m])
            if len(values) == 0:
                print(f"{m:<25} {'N/A':>12} {'N/A':>12}  {0:>4}")
            else:
                print(f"{m:<25} {values.mean():>12.6f} {values.std():>12.6f}  {len(values):>4}")


if __name__ == "__main__":
    main()
