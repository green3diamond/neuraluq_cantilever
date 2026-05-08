#!/usr/bin/env python
"""
Grouped bar chart comparison of cantilever VI experiments across runs v15, v16, v17.

Produces two plots (one per mode M2/M3) showing individual run metric values
as bars, grouped by metric -> noise_type -> noise_level.
W/N ratio is recomputed using the scaling: boundary_width / (max|U| * noise_amplitude * 5).
"""

import os
import re
import glob
import sys

import numpy as np
# Compatibility shim: .npy files saved with numpy 2.x reference numpy._core
# which doesn't exist in numpy <2.0. Add module aliases so pickle can resolve them.
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

LOG_PLOTS = not True

FOLDERS = [
    os.path.join(SCRIPT_DIR, "results_data_no_ic_v15"),
    os.path.join(SCRIPT_DIR, "results_data_no_ic_v16"),
    os.path.join(SCRIPT_DIR, "results_data_no_ic_v17"),
]

METRICS = ["distance_to_boundary", "distance_from_boundary", "fraction_outside"]

METRIC_TITLES = {
    "distance_to_boundary": "Dist2Bnd",
    "width_vs_noise_ratio": "W/N ratio",
    "distance_from_boundary": "Overshoot",
    "fraction_outside": "FracOut",
}
NOISE_TYPES = ["normal", "uniform", "beta"]
MODES = [2, 3]
NOISE_LEVEL_COLORS = {
    0.01: "#1f77b4",
    0.03: "#ff7f0e",
    0.05: "#2ca02c",
}
if LOG_PLOTS:
    METRIC_YLIMS = {
        # log
        "distance_to_boundary": (1e-4, 1e1),
        "distance_from_boundary": (5e-3, 1e0),
        "width_vs_noise_ratio": (1e-1, 1e2),
        "fraction_outside":     (5e-2, 1e0),
    }
else:
    METRIC_YLIMS = {
        # linear
        "distance_to_boundary": (1e-4, 1e-2),
        "distance_from_boundary": (1e-4, 1e-1),
        "width_vs_noise_ratio": (1e-1, 1.5e1),
        "fraction_outside":     (1e-2, 4e-1),
    }

plt.rcParams.update({
    'font.size':        20,
    'axes.labelsize':   20,
    'xtick.labelsize':  20,
    'ytick.labelsize':  20,
    'legend.fontsize':  20,
    'axes.titlesize':   22,
    'figure.titlesize': 22,
})


PARSE_KEYS = [
    "data_file", "noise_type", "noise_amplitude",
    "distance_to_boundary", "distance_from_boundary",
    "boundary_width", "width_vs_noise_ratio", "fraction_outside",
]
FLOAT_KEYS = {
    "noise_amplitude", "distance_to_boundary", "distance_from_boundary",
    "boundary_width", "width_vs_noise_ratio", "fraction_outside",
}


def parse_metrics(filepath):
    """Parse key-value pairs from a vi_metrics.txt file."""
    metrics = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in PARSE_KEYS:
                metrics[key] = float(value) if key in FLOAT_KEYS else value
    return metrics


def extract_mode(filepath):
    """Extract mode number (2 or 3) from filename like test_NT01_M2_noise001_normal."""
    m = re.search(r"_M(\d+)_", os.path.basename(filepath))
    return int(m.group(1)) if m else None


def recompute_wn_ratio(metrics, script_dir):
    """Recompute width_vs_noise_ratio using 28-style scaling."""
    data_file = metrics.get("data_file", "")
    if not data_file:
        return metrics.get("width_vs_noise_ratio", float("nan"))

    if not os.path.isabs(data_file):
        data_file = os.path.join(script_dir, data_file)

    if not os.path.exists(data_file):
        return metrics.get("width_vs_noise_ratio", float("nan"))

    data = np.load(data_file, allow_pickle=True).item()
    max_abs_target = np.max(np.abs(data["U"]))
    noise_amplitude = metrics.get("noise_amplitude", 0)
    boundary_width = metrics.get("boundary_width", 0)

    denominator = max_abs_target * noise_amplitude * 5
    if denominator <= 0:
        return float("inf")
    return boundary_width / denominator


def load_all_metrics():
    """Load metrics from all vi_metrics.txt files across all folders.

    Returns list of dicts with keys: mode, noise_type, noise_level, metric, value.
    """
    rows = []
    for folder in FOLDERS:
        files = sorted(glob.glob(os.path.join(folder, "*_vi_metrics.txt")))
        for filepath in files:
            mode = extract_mode(filepath)
            if mode not in MODES:
                continue

            metrics = parse_metrics(filepath)
            noise_type = metrics.get("noise_type", "unknown")
            if noise_type not in NOISE_TYPES:
                continue

            noise_level = metrics.get("noise_amplitude", 0)

            # Recompute W/N ratio with 28-style scaling
            wn_ratio = recompute_wn_ratio(metrics, SCRIPT_DIR)

            for metric_name in METRICS:
                if metric_name == "width_vs_noise_ratio":
                    value = wn_ratio
                else:
                    value = metrics.get(metric_name)
                if value is not None:
                    rows.append({
                        "mode": mode,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "metric": metric_name,
                        "value": value,
                    })
    return rows


def plot_grouped_bars(rows, metrics, mode, title, filename):
    """Plot one bar per individual run, one subplot per metric.

    Grouping: metric (subplot) -> noise_type -> noise_level (runs as bars).
    Bars colored by noise_level.
    """
    from collections import defaultdict

    mode_rows = [r for r in rows if r["mode"] == mode]
    noise_levels = sorted({r["noise_level"] for r in mode_rows})

    # (metric, noise_type, noise_level) -> list of values
    lookup = defaultdict(list)
    for r in mode_rows:
        lookup[(r["metric"], r["noise_type"], r["noise_level"])].append(r["value"])

    bar_width = 0.6
    experiment_gap = 0.3
    noise_level_gap = 0.6
    noise_type_gap = 1.8

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    # fig.suptitle(title)

    for ax, metric in zip(axes, metrics):
        x_pos = 0.0
        nt_label_positions = []

        for nt in NOISE_TYPES:
            nt_start = x_pos
            for nl in noise_levels:
                nl_start = x_pos
                values = lookup.get((metric, nt, nl), [])
                color = NOISE_LEVEL_COLORS.get(nl, "#333333")
                for v in sorted(values):
                    ax.bar(x_pos, v, width=bar_width * 0.9, color=color,
                           edgecolor="black", linewidth=0.4)
                    x_pos += bar_width + experiment_gap
                # Remove trailing experiment gap
                if x_pos > nl_start:
                    x_pos -= experiment_gap
                if x_pos > nl_start:
                    x_pos += noise_level_gap
            # Remove trailing noise_level gap
            if x_pos > nt_start:
                x_pos -= noise_level_gap
            nt_center = (nt_start + x_pos) / 2.0
            nt_label_positions.append((nt_center, nt))
            x_pos += noise_type_gap

        ax.set_xticks([p for p, _ in nt_label_positions])
        ax.set_xticklabels([lbl for _, lbl in nt_label_positions])
        ax.set_title(METRIC_TITLES.get(metric, metric))
        ax.set_ylabel("Value")
        if LOG_PLOTS:
            ax.set_yscale("log")
        if metric in METRIC_YLIMS:
            ax.set_ylim(METRIC_YLIMS[metric])
        ax.grid(axis="y", alpha=0.3)

        # Vertical separators between noise_type groups
        for i in range(len(nt_label_positions) - 1):
            x1 = nt_label_positions[i][0]
            x2 = nt_label_positions[i + 1][0]
            sep_x = (x1 + x2) / 2.0
            ax.axvline(sep_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Shared legend at the bottom
    handles = [
        mpatches.Patch(
            facecolor=NOISE_LEVEL_COLORS.get(nl, "#333333"),
            edgecolor="black",
            label=f"{nl*5:.2f}",
        )
        for nl in noise_levels
    ]
    # Include title as a blank patch so it appears inline to the left
    title_handle = mpatches.Patch(facecolor="none", edgecolor="none", label="Noise Level:")
    fig.legend(
        handles=[title_handle] + handles,
        loc="lower center", ncol=len(noise_levels) + 1,
        bbox_to_anchor=(0.5, -0.15),
        handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
    )

    fig.subplots_adjust(bottom=0.18, left=0.06, right=0.98, wspace=0.35)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved {filename}")
    return fig


def main():
    rows = load_all_metrics()
    print(f"Loaded {len(rows)} metric values from {len(FOLDERS)} folders")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            print(f"No data for mode {mode}, skipping.")
            continue
        plot_grouped_bars(
            rows, METRICS, mode,
            title=f"Mode {mode} — VI Metric Comparison",
            filename=os.path.join(RESULTS_DIR, f"grouped_bars_M{mode}.pdf"),
        )


if __name__ == "__main__":
    main()
