"""Aggregate VI metrics across multiple result folders (v4, v5, v6)."""

import os
import glob
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = [
    os.path.join(RESULTS_DIR, "results_data_no_ic_v4"),
    os.path.join(RESULTS_DIR, "results_data_no_ic_v5"),
    os.path.join(RESULTS_DIR, "results_data_no_ic_v6"),
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


def main():
    # Collect all metric values
    all_metrics = {m: [] for m in METRICS}

    for folder in FOLDERS:
        files = sorted(glob.glob(os.path.join(folder, "*_vi_metrics.txt")))
        for filepath in files:
            metrics = parse_metrics(filepath)
            for m in METRICS:
                if m in metrics:
                    all_metrics[m].append(metrics[m])

    # Compute and print mean/std
    print(f"{'Metric':<25} {'Mean':>12} {'Std':>12}  {'N':>4}")
    print("-" * 57)
    for m in METRICS:
        values = np.array(all_metrics[m])
        print(f"{m:<25} {values.mean():>12.6f} {values.std():>12.6f}  {len(values):>4}")

    print(f"\nTotal files processed: {sum(len(v) for v in all_metrics.values()) // len(METRICS)}")


if __name__ == "__main__":
    main()
