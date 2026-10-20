#!/usr/bin/env python3
"""Compare GPU memory consumption across speculative decoding methods.

Reads JSON reports produced by profile_memory.py and generates:
  - A formatted comparison table (console)
  - A stacked bar chart (matplotlib, saved as PNG)

Usage:
    uv run read_memory_results.py /path/to/memory_profile/
    uv run read_memory_results.py nospec_memory.json dflash_memory.json eagle3_memory.json whiflash_entropy_memory.json whiflash_neural_memory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def format_bytes(b: int) -> str:
    """Human-readable byte string."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b / 1024**2:.2f} MB"
    elif b < 1024**3:
        return f"{b / 1024**3:.2f} GB"
    else:
        return f"{b / 1024**3:.2f} GB"


def format_gb(b: int) -> str:
    """Format bytes as GB with 2 decimal places."""
    return f"{b / 1024**3:.2f}"


def load_report(path: str) -> dict:
    """Load a memory report JSON file."""
    with open(path) as f:
        return json.load(f)


def print_comparison_table(reports: list[dict]) -> None:
    """Print a formatted comparison table to console."""
    # Column widths
    method_w = 22
    col_w = 12

    # Header
    header = f"{'Method':{method_w}}"
    metrics = [
        ("Peak GPU", "peak_gpu_memory_bytes"),
        ("Weights", "weights_total_bytes"),
        ("KV+Buffers", "kv_and_buffers_bytes"),
        ("Overhead", "overhead_vs_target_only_bytes"),
    ]
    for label, _ in metrics:
        header += f" | {label:>{col_w}}"

    # Add component columns
    comp_keys = []
    for report in reports:
        for k in report.get("component_breakdown_bytes", {}):
            if k not in comp_keys:
                comp_keys.append(k)

    for k in comp_keys:
        label = k.replace("_weights", "").replace("_", " ")
        header += f" | {label:>{col_w}}"

    print("\n" + "=" * len(header))
    print("  MEMORY COMPARISON ACROSS METHODS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for report in reports:
        row = f"{report['method']:{method_w}}"
        for _, key in metrics:
            val = report.get(key, 0)
            row += f" | {format_gb(val):>{col_w}}"

        comp = report.get("component_breakdown_bytes", {})
        for k in comp_keys:
            val = comp.get(k, 0)
            label = k.replace("_weights", "").replace("_", " ")
            row += f" | {format_gb(val):>{col_w}}"

        print(row)

    print("-" * len(header))

    # Overhead as percentage of target
    if len(reports) >= 1:
        target_mem = None
        for report in reports:
            # Find the target-only baseline from the first report's snapshots
            for snap in report.get("snapshots", []):
                if snap["label"] == "after_target_load":
                    target_mem = snap["gpu_allocated_bytes"]
                    break
            if target_mem is not None:
                break

        if target_mem is not None:
            print(f"\n  Target-only baseline: {format_bytes(target_mem)}")
            print(f"  {'Method':{method_w}} | {'Overhead':>{col_w}} | {'% of Target':>{col_w}}")
            print("  " + "-" * (method_w + 3 + col_w + 3 + col_w))
            for report in reports:
                overhead = report.get("overhead_vs_target_only_bytes", 0)
                pct = (overhead / target_mem * 100) if target_mem > 0 else 0
                print(f"  {report['method']:{method_w}} | {format_bytes(overhead):>{col_w}} | {pct:>{col_w}.1f}%")

    # Lifecycle snapshots comparison
    print("\n  Lifecycle memory snapshots (GB):")
    all_labels = []
    for report in reports:
        for snap in report.get("snapshots", []):
            if snap["label"] not in all_labels:
                all_labels.append(snap["label"])

    # Header
    snap_header = f"  {'Phase':{30}}"
    for report in reports:
        snap_header += f" | {report['method']:>{col_w}}"
    print(snap_header)
    print("  " + "-" * (30 + 3 + (col_w + 3) * len(reports)))

    for label in all_labels:
        row = f"  {label:30}"
        for report in reports:
            val = 0
            for snap in report.get("snapshots", []):
                if snap["label"] == label:
                    val = snap["gpu_allocated_bytes"]
                    break
            row += f" | {format_gb(val):>{col_w}}"
        print(row)

    # Peak row
    peak_row = f"  {'PEAK (generation)':30}"
    for report in reports:
        peak_row += f" | {format_gb(report['peak_gpu_memory_bytes']):>{col_w}}"
    print(peak_row)

    # NOSPEC breakdown: target model weights vs KV-cache
    nospec_reports = [r for r in reports if r["method"] == "NOSPEC"]
    if nospec_reports:
        print("\n  NOSPEC Memory Breakdown:")
        print("  " + "-" * 60)
        for report in nospec_reports:
            target_weights = report.get("component_breakdown_bytes", {}).get("target_weights", 0)
            peak_mem = report.get("peak_gpu_memory_bytes", 0)
            kv_cache = peak_mem - target_weights
            model_tag = report.get("model", "unknown")
            max_tokens = report.get("max_new_tokens", "?")
            num_output = report.get("num_output_tokens", "?")
            print(f"    Model: {model_tag}")
            print(f"    Max new tokens: {max_tokens}, Actual output tokens: {num_output}")
            print(
                f"    Target model weights:       {format_bytes(target_weights):>12s}  ({format_gb(target_weights)} GB)"
            )
            print(f"    KV-cache + buffers:          {format_bytes(kv_cache):>12s}  ({format_gb(kv_cache)} GB)")
            print(f"    Peak GPU memory:             {format_bytes(peak_mem):>12s}  ({format_gb(peak_mem)} GB)")
            print()

    print()


def plot_comparison(reports: list[dict], output_path: str) -> None:
    """Generate a stacked bar chart comparing memory across methods."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping chart generation.")
        return

    methods = [r["method"] for r in reports]
    n = len(methods)

    # Stacked bar components
    comp_keys = ["target_weights"]
    for r in reports:
        for k in r.get("component_breakdown_bytes", {}):
            if k not in comp_keys:
                comp_keys.append(k)

    # Convert to GB
    data = {}
    for k in comp_keys:
        data[k] = []
        for r in reports:
            val = r.get("component_breakdown_bytes", {}).get(k, 0)
            data[k].append(val / 1024**3)

    # Add KV+buffers as the residual
    for r in reports:
        kv_buf = r.get("kv_and_buffers_bytes", 0) - r.get("weights_total_bytes", 0)
        if "kv_and_buffers" not in data:
            data["kv_and_buffers"] = []
        data["kv_and_buffers"].append(max(0, kv_buf) / 1024**3)

    # Colors
    colors = {
        "target_weights": "#4C72B0",
        "dflash_weights": "#55A868",
        "eagle3_weights": "#C44E52",
        "router_weights": "#8172B2",
        "kv_and_buffers": "#CCB974",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n)
    width = 0.5
    bottom = np.zeros(n)

    for k in comp_keys + ["kv_and_buffers"]:
        if k not in data:
            continue
        vals = data[k]
        label = k.replace("_weights", " weights").replace("_", " ").title()
        color = colors.get(k, None)
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color)
        bottom += np.array(vals)

    ax.set_ylabel("GPU Memory (GB)")
    ax.set_title("GPU Memory Consumption by Method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.legend(loc="upper left")

    # Add total labels on top of each bar
    for i, total in enumerate(bottom):
        ax.text(i, total + 0.1, f"{total:.2f} GB", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare memory profiles across methods")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSON report files or a directory containing *_memory.json files",
    )
    parser.add_argument(
        "--output-chart",
        type=str,
        default="memory_comparison.png",
        help="Output path for the comparison chart (PNG). Default: memory_comparison.png",
    )

    args = parser.parse_args()

    # Collect JSON files
    json_files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            json_files.extend(sorted(p.glob("*_memory.json")))
        elif p.is_file() and p.suffix == ".json":
            json_files.append(p)
        else:
            print(f"Warning: skipping {inp} (not a file or directory)")

    if not json_files:
        print("Error: no JSON report files found.")
        sys.exit(1)

    # Load reports
    reports = []
    for f in json_files:
        try:
            report = load_report(str(f))
            reports.append(report)
            print(f"Loaded: {f} ({report['method']})")
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")

    if not reports:
        print("Error: no valid reports loaded.")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(reports)

    # Generate chart
    plot_comparison(reports, args.output_chart)


if __name__ == "__main__":
    main()
