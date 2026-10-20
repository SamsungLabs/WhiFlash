#!/usr/bin/env python3
"""Analyze WhiFlash switch overhead from benchmark JSONL results.

Reads switch_events from a transformers.jsonl file (or all such files in a
directory) and plots t_switch_ms against each influencing factor.

Usage:
    # Single JSONL file
    uv run analysis/switch_overhead.py <path_to_jsonl>

    # All JSONL files in a directory (recursively)
    uv run analysis/switch_overhead.py <path_to_directory>

    # As a module
    uv run -m analysis.switch_overhead <path_to_directory>

Examples:
    uv run analysis/switch_overhead.py logs/17apr2026/sdpa/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/WHIFLASH-sysprompt-llama3/gsm8k/entropy_1.5/transformers.jsonl
    uv run analysis/switch_overhead.py logs/17apr2026/sdpa/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/WHIFLASH-sysprompt-llama3
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def find_jsonl_files(path: str) -> list[Path]:
    """Find all transformers.jsonl files from a path (file or directory)."""
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(p.rglob("transformers.jsonl"))


def load_switch_events(jsonl_path: str, dataset_name: str | None = None) -> list[dict]:
    """Load all switch_events from a JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            for turn in sample.get("turns", []):
                for evt in turn.get("switch_events", []):
                    evt["question_id"] = sample.get("question_id")
                    if dataset_name is not None:
                        evt["dataset"] = dataset_name
                    records.append(evt)
    return records


def load_all_switch_events(path: str) -> dict[str, pd.DataFrame]:
    """Load switch_events from a single JSONL or all JSONL files in a directory.

    Returns dict mapping direction -> DataFrame.
    """
    jsonl_files = find_jsonl_files(path)
    if not jsonl_files:
        print(f"No transformers.jsonl files found in {path}")
        sys.exit(1)

    all_records: list[dict] = []
    for jf in jsonl_files:
        parts = jf.parts
        dataset_name = None
        for i, part in enumerate(parts):
            if part.startswith("entropy_"):
                if i >= 2:
                    dataset_name = parts[i - 1]
                break
        if dataset_name is None:
            dataset_name = jf.parent.parent.name

        records = load_switch_events(str(jf), dataset_name)
        if records:
            print(f"  Loaded {len(records)} switch events from {jf.parent} (dataset={dataset_name})")
            all_records.extend(records)

    if not all_records:
        print("No switch_events found in any JSONL file.")
        sys.exit(0)

    # Separate by direction (backward compat: no direction field = dflash_to_eagle3)
    by_direction: dict[str, list[dict]] = {}
    for rec in all_records:
        direction = rec.get("direction", "dflash_to_eagle3")
        by_direction.setdefault(direction, []).append(rec)

    result = {}
    for direction, records in by_direction.items():
        df = pd.DataFrame(records)
        # Remove outliers using IQR method on t_switch_ms
        q1 = df["t_switch_ms"].quantile(0.25)
        q3 = df["t_switch_ms"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 5 * iqr
        n_before = len(df)
        df = df[df["t_switch_ms"] <= upper_bound].copy()
        n_removed = n_before - len(df)
        if n_removed > 0:
            print(f"  [{direction}] Removed {n_removed} outliers (t_switch_ms > {upper_bound:.1f}ms, 5×IQR)")
        result[direction] = df

    return result


def plot_correlation_by_dataset(
    ax, df: pd.DataFrame, x_col: str, xlabel: str, ylabel: str = "Switching Time [ms]", show_regression: bool = True
):
    """Scatter colored by dataset + optional overall regression."""
    datasets = sorted(df["dataset"].unique())
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(len(datasets), 1))

    for i, ds in enumerate(datasets):
        mask = df["dataset"] == ds
        ds_df = df[mask]
        ax.scatter(ds_df[x_col], ds_df["t_switch_ms"], alpha=0.4, s=20, edgecolors="none", label=ds, color=cmap(i))

    x = df[x_col].values
    y = df["t_switch_ms"].values
    valid = np.isfinite(x) & np.isfinite(y)

    if show_regression and valid.sum() > 2:
        slope, intercept, r_value, _, _ = stats.linregress(x[valid], y[valid])
        x_min, x_max = x[valid].min(), x[valid].max()
        xs = np.linspace(x_min, x_max, 200)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.5, alpha=0.7)
        pearson_r2 = r_value**2
        margin_x = (x_max - x_min) * 0.02
        y_pred = slope * xs + intercept
        y_min = min(y[valid].min(), y_pred.min())
        y_max = max(y[valid].max(), y_pred.max())
        margin_y = (y_max - y_min) * 0.05
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)
    else:
        pearson_r2 = float("nan")
        if valid.sum() > 2:
            margin_x = (x[valid].max() - x[valid].min()) * 0.02
            margin_y = (y[valid].max() - y[valid].min()) * 0.05
            ax.set_xlim(x[valid].min() - margin_x, x[valid].max() + margin_x)
            ax.set_ylim(y[valid].min() - margin_y, y[valid].max() + margin_y)

    if valid.sum() > 2:
        spearman_rho, _ = stats.spearmanr(x[valid], y[valid])
    else:
        spearman_rho = float("nan")

    n_total = len(df)
    mean_total = df["t_switch_ms"].mean()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_regression:
        ax.set_title(
            f"R²={pearson_r2:.3f}  ρ={spearman_rho:.3f}\nswitching count={n_total}, avg switching time={mean_total:.1f}ms"
        )
    else:
        ax.set_title(f"ρ={spearman_rho:.3f}\nswitching count={n_total}, avg switching time={mean_total:.1f}ms")


# Direction-specific configuration: x-axis columns and labels
DIRECTION_CONFIG = {
    "dflash_to_eagle3": {
        "label": "DFlash → EAGLE3",
        "plots": [
            ("num_backlog_tokens", "Num Backlog Tokens", True),
            ("ctx_len", "EAGLE3 KV Cache Length", False),
        ],
    },
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    input_p = Path(input_path)
    is_dir = input_p.is_dir()

    print(f"Loading from: {input_path} ({'directory' if is_dir else 'file'})")
    direction_dfs = load_all_switch_events(input_path)

    # Print stats for each direction
    for direction, df in direction_dfs.items():
        config = DIRECTION_CONFIG.get(direction, {})
        label = config.get("label", direction)
        print(f"\n{'='*60}")
        print(f"  {label} ({len(df)} events)")
        print(f"{'='*60}")
        print("\nSummary statistics:")
        stat_cols = [c for c in df.columns if c not in ("question_id", "dataset", "direction", "num_chunks")]
        print(df[stat_cols].describe().to_string())

        n_datasets = df["dataset"].nunique()
        if n_datasets > 1:
            # Build a clean summary table with readable column names
            col_rename = {
                "t_switch_ms": "switch time mean (ms)",
                "num_backlog_tokens": "num backlog tokens mean",
                "ctx_len": "current context length (kv cache) mean",
            }
            agg_dict = {"t_switch_ms": ["count", "mean"]}
            for col in stat_cols:
                if col != "t_switch_ms":
                    agg_dict[col] = "mean"
            summary = df.groupby("dataset").agg(agg_dict).round(2)
            # Flatten and rename columns
            new_cols = []
            for col, stat in summary.columns:
                if col == "t_switch_ms" and stat == "count":
                    new_cols.append("switch count")
                elif col == "t_switch_ms" and stat == "mean":
                    new_cols.append(col_rename.get(col, col))
                else:
                    new_cols.append(col_rename.get(col, col))
            summary.columns = new_cols

            # Add total row
            total_means = {"switch count": len(df), "switch time mean (ms)": round(df["t_switch_ms"].mean(), 2)}
            for col in stat_cols:
                if col != "t_switch_ms" and col in col_rename:
                    total_means[col_rename[col]] = round(df[col].mean(), 2)

            total_df = pd.DataFrame(total_means, index=pd.Index(["TOTAL"]))
            summary = pd.concat([summary, total_df])

            print("\nPer-dataset summary:")
            print(summary.to_string())

        print("\nCorrelation with t_switch_ms:")
        for col in stat_cols:
            if col == "t_switch_ms":
                continue
            r, pval = stats.pearsonr(df[col], df["t_switch_ms"])
            rho, p_s = stats.spearmanr(df[col], df["t_switch_ms"])
            print(f"  {col:30s}  Pearson r={r:.4f} (p={pval:.2e})  Spearman ρ={rho:.4f} (p={p_s:.2e})")

    # Plot: one figure per direction, saved to separate files
    if len(direction_dfs) == 0:
        print("No switch events to plot.")
        return

    out_dir = input_p if is_dir else input_p.parent

    for direction, config in DIRECTION_CONFIG.items():
        df = direction_dfs.get(direction)
        label = config.get("label", direction)
        plots = config.get("plots", [])

        n_cols = len(plots) if plots else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
        if n_cols == 1:
            axes = [axes]
        title_path = input_path if not is_dir else f"{input_path}/"
        fig.suptitle(f"Switch Overhead: {label}\n{title_path}", fontsize=13)

        if df is not None and len(df) > 0:
            for col_idx, (x_col, xlabel, show_reg) in enumerate(plots):
                ax = axes[col_idx]
                if x_col in df.columns:
                    plot_correlation_by_dataset(ax, df, x_col, xlabel, show_regression=show_reg)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Column '{x_col}'\nnot found",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="gray",
                    )
                    ax.set_title(f"{xlabel} (no data)")

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(
                    handles,
                    labels,
                    loc="lower right",
                    ncol=2,
                    fontsize=8,
                    framealpha=0.9,
                    edgecolor="gray",
                )
        else:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "No data available\n(run benchmarks with updated code)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                )

        plt.tight_layout()
        out_path = out_dir / f"switch_overhead_{direction}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
