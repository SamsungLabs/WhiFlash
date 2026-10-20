#!/usr/bin/env python3
"""
Script to find and organize benchmark results from nested log directories
and print out a column of results ready to be pasted to spreadsheet.

Supports multiple metrics (--metric):
  - throughput:  [Decoding] Average Throughput (tokens/s)
  - acceptance:  Average Acceptance length

Supports multiple model types (--model-type):
  - qwen8b:   Qwen3-8B experiments
  - qwen4b:   Qwen3-4B experiments
  - llama8b:  Llama-3.1-8B experiments

Supports multiple modes:
1. Default mode: Dataset is the parent directory of transformers.log

2. Config subfolder mode (--config-subfolder / auto-activated by --config or --sweep-table)

3. Sweep table mode (--sweep-table): Print a multi-column table for a router sweep.
   Rows = datasets, columns sorted by (variation, budget, threshold).

4. Full table mode (--full-table): Print the complete results table with columns:
   NOSPEC | EAGLE3 | DFLASH | DDTree budget sweep | WhiFlash entropy sweep | WhiFlash neural sweep
   For throughput it reports speedup over NOSPEC by default (--no-speedup for tok/s).

"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from calibration_select_thresholds import CATEGORY, CATS, calibration_qids
from calibration_select_thresholds import metric as turns_metric
from calibration_select_thresholds import router_matches_pair

# Define the desired order (case-insensitive matching)
DATASET_ORDER = [
    "mt-bench",
    "alpaca",
    "ultrachat-test",
    "wildchat",
    "gsm8k",
    "math500",
    "aime24",
    "aime25",
    "humaneval",
    "mbpp",
    "livecodebench",
    "swe-bench",
    "tau-bench",
    "agentbench",
    "agent-instruct",
]

# Default threshold values for comprehensive evaluation
ENTROPY_THRESHOLDS = [0.00001, 0.001, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 5.0]
NEURAL_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
ORACLE_VARIANTS = ["accept", "utility"]

# WhiFlash drafter-pair variations
WHIFLASH_VARIATIONS = ("df-e3", "ddt-e3", "df-ddt")


# Neural routers per model_type, matched against the config tag. The version suffix is pinned.
NEURAL_ROUTER_MODELS = {
    "qwen8b": [
        "Qwen3-8B_dolci_v1",
        "Qwen3-8B_dolci_eagle-vs-ddtree_b32_v1",
        "Qwen3-8B_dolci_eagle-vs-ddtree_b64_v1",
        "Qwen3-8B_dolci_eagle-vs-ddtree_b128_v1",
        "Qwen3-8B_dolci_eagle-vs-ddtree_b256_v1",
        "Qwen3-8B_dolci_eagle-vs-ddtree_b512_v1",
    ],
    "qwen4b": [
        "Qwen3-4B_dolci_v1",
        "Qwen3-4B_dolci_eagle-vs-ddtree_b32_v1",
        "Qwen3-4B_dolci_eagle-vs-ddtree_b64_v1",
        "Qwen3-4B_dolci_eagle-vs-ddtree_b128_v1",
        "Qwen3-4B_dolci_eagle-vs-ddtree_b256_v1",
        "Qwen3-4B_dolci_eagle-vs-ddtree_b512_v1",
    ],
    "llama8b": [
        "Llama-3.1-8B-Instruct_dolci_v1",
        "Llama-3.1-8B-Instruct_dolci_eagle-vs-ddtree_b32_v1",
        "Llama-3.1-8B-Instruct_dolci_eagle-vs-ddtree_b64_v1",
        "Llama-3.1-8B-Instruct_dolci_eagle-vs-ddtree_b128_v1",
        "Llama-3.1-8B-Instruct_dolci_eagle-vs-ddtree_b256_v1",
        "Llama-3.1-8B-Instruct_dolci_eagle-vs-ddtree_b512_v1",
    ],
}

# Path patterns to match for each model_type (used to filter log directories).
MODEL_TYPE_PATTERNS = {
    "qwen8b": "Qwen3-8B",
    "qwen4b": "Qwen3-4B",
    "llama8b": "LLaMA",  # matches LLaMA3.1-8B and LLaMA-8B in draft names
}

# Metric definitions: name -> (regex pattern, column header)
METRIC_DEFS = {
    "throughput": (
        r"\[Decoding\] Average Throughput: ([\d.]+) tokens/s",
        "Throughput(tokens/s)",
    ),
    "acceptance": (
        r"Average Acceptance length: ([\d.]+)",
        "Acceptance_Length",
    ),
}


def _classify_neural_config(config_name, model_type=None):
    """Classify a neural config by its router model for a given model_type.

    Matched on the full versioned name, so older router generations are excluded.

    Args:
        config_name: e.g., "neural_df-e3_Qwen3-8B_dolci_v1_0.5", or for a published router
                     "neural_df-e3_Qwen3-8B-NeuralRouter-df-e3_0.5"
        model_type: If provided, only check routers for this model type.
                    If None, check all known router models across all model types.

    Returns a key like 'WHIFLASH_neural_ultrachat' or 'WHIFLASH_neural_dolci',
    or None if the config doesn't match any known router.
    """
    if model_type is not None:
        router_models = NEURAL_ROUTER_MODELS.get(model_type, [])
    else:
        # Flatten all router models from all model types
        router_models = []
        for models in NEURAL_ROUTER_MODELS.values():
            router_models.extend(models)

    for router_model in router_models:
        if router_model in config_name:
            # Short tag = token after the model name: "Qwen3-8B_dolci_v1" -> "dolci"
            tag = router_model.split("_")[1]
            return f"WHIFLASH_neural_{tag}"
    # Published routers ("neural_df-e3_Qwen3-8B-NeuralRouter-df-e3_1.0", as written by
    # scripts/release/benchmark/benchmark.sh) are the dolci-trained ones.
    if "-NeuralRouter-" in config_name:
        if model_type is None or MODEL_TYPE_PATTERNS.get(model_type, "") + "-NeuralRouter-" in config_name:
            return "WHIFLASH_neural_dolci"
    return None


def _path_matches_model_type(path, model_type):
    """Check if a file path matches the given model_type pattern."""
    pattern = MODEL_TYPE_PATTERNS.get(model_type, "")
    return pattern in path


def find_metric_in_logs(logs_dir, metric="throughput", config_subfolder=False, config_filter=None, model_type=None):
    """
    Recursively find transformers.log files and extract a metric.

    Args:
        logs_dir: Root directory to search
        metric: Metric to extract ("throughput" or "acceptance")
        config_subfolder: If True, expect config subfolders under each dataset
        config_filter: If provided, only return results from this config folder
        model_type: If provided, only include results from paths matching this model type

    Returns:
        If config_subfolder=False: dict mapping dataset -> metric value
        If config_subfolder=True: dict mapping config -> {dataset -> metric value}
    """
    results = {} if not config_subfolder else defaultdict(dict)
    pattern = METRIC_DEFS[metric][0]

    for root, dirs, files in os.walk(logs_dir):
        for filename in files:
            if filename == "transformers.log":
                filepath = os.path.join(root, filename)

                # Filter by model_type if specified
                if model_type and not _path_matches_model_type(filepath, model_type):
                    continue

                if config_subfolder:
                    config_name = os.path.basename(root)
                    dataset = os.path.basename(os.path.dirname(root)).lower()
                    if config_filter and config_name != config_filter:
                        continue
                else:
                    dataset = os.path.basename(root).lower()
                    config_name = None

                try:
                    with open(filepath) as f:
                        content = f.read()
                        match = re.search(pattern, content)
                        if match:
                            value = float(match.group(1))
                            if config_subfolder:
                                results[config_name][dataset] = value
                            else:
                                results[dataset] = value
                except OSError as e:
                    print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)

    return results


def _eval_split_value(run_dir, dataset, metric):
    """Metric over a run's eval samples: all of a --split eval run, the samples outside the calibration split
    for a full run, None for a calibration run."""
    cfg, jsonl = Path(run_dir) / "config.json", Path(run_dir) / "transformers.jsonl"
    if not jsonl.is_file():
        return None
    run_split = json.loads(cfg.read_text())["args"].get("split", "all") if cfg.is_file() else "all"
    if run_split == "calibration":
        return None
    recs = [json.loads(line) for line in open(jsonl)]
    if run_split == "all":
        calibration = calibration_qids(dataset, len(recs))
        recs = [r for r in recs if r["question_id"] not in calibration]
    return turns_metric([t for r in recs for t in r["turns"]], "tps" if metric == "throughput" else "mat")


def find_all_results(logs_dir, metric="throughput", model_type=None, split="all"):
    """
    Walk the entire logs directory and classify results by algorithm type.

    Args:
        logs_dir: Root directory to search
        metric: Metric to extract
        model_type: If provided, only include results from paths matching this model type

    Returns a dict with structure:
    {
        "NOSPEC": {dataset: value, ...},
        "EAGLE3": {dataset: value, ...},
        "DFLASH": {dataset: value, ...},
        "WHIFLASH_entropy": {config: {dataset: value, ...}, ...},
        "WHIFLASH_neural_ultrachat": {config: {dataset: value, ...}, ...},
        "WHIFLASH_neural_dolci": {config: {dataset: value, ...}, ...},
        "WHIFLASH_oracle": {config: {dataset: value, ...}, ...},
    }
    """
    results = {
        "NOSPEC": {},
        "EAGLE3": {},
        "DFLASH": {},
        "DDTREE": defaultdict(dict),  # config = "budget_<N>"
        "WHIFLASH_entropy": defaultdict(dict),
        "WHIFLASH_oracle": defaultdict(dict),
    }
    pattern = METRIC_DEFS[metric][0]

    for root, dirs, files in os.walk(logs_dir):
        for filename in files:
            if filename != "transformers.log":
                continue

            filepath = os.path.join(root, filename)

            # Filter by model_type if specified
            if model_type and not _path_matches_model_type(filepath, model_type):
                continue

            parts = root.split(os.sep)

            algo = None
            config_name = None
            dataset = None

            for i, part in enumerate(parts):
                if "-sysprompt-" in part:
                    algo_part = part.split("-sysprompt-")[0]
                    if algo_part.startswith("WHIFLASH"):
                        algo = "WHIFLASH"
                    elif algo_part.startswith("NOSPEC"):
                        algo = "NOSPEC"
                    elif algo_part.startswith("EAGLE3"):
                        algo = "EAGLE3"
                    elif algo_part.startswith("DFLASH"):
                        algo = "DFLASH"
                    else:
                        algo = algo_part

                    remaining_parts = parts[i + 1 :]

                    # WHIFLASH and DDTREE store results in a config subfolder
                    # (CFG_TAG / budget_<N>); the others put logs directly under dataset/.
                    if algo in ("WHIFLASH", "DDTREE") and len(remaining_parts) >= 2:
                        dataset = remaining_parts[0].lower()
                        config_name = remaining_parts[1]
                    elif len(remaining_parts) >= 1:
                        dataset = remaining_parts[0].lower()
                        config_name = None
                    break

            if algo is None or dataset is None:
                continue

            try:
                with open(filepath) as f:
                    content = f.read()
                    match = re.search(pattern, content)
                    if match:
                        value = float(match.group(1))
                        if split == "eval":
                            value = _eval_split_value(root, dataset, metric)
                            if value is None:
                                continue

                        if algo in ("NOSPEC", "EAGLE3", "DFLASH"):
                            results[algo][dataset] = value
                        elif algo == "DDTREE" and config_name:
                            results["DDTREE"][config_name][dataset] = value
                        elif algo == "WHIFLASH" and config_name:
                            if config_name.startswith("entropy_"):
                                results["WHIFLASH_entropy"][config_name][dataset] = value
                            elif config_name.startswith("neural_"):
                                neural_key = _classify_neural_config(config_name, model_type)
                                if neural_key is not None:
                                    if neural_key not in results:
                                        results[neural_key] = defaultdict(dict)
                                    results[neural_key][config_name][dataset] = value
                            elif config_name.startswith("oracle_"):
                                results["WHIFLASH_oracle"][config_name][dataset] = value
                            else:
                                key = f"WHIFLASH_{config_name}"
                                if key not in results:
                                    results[key] = defaultdict(dict)
                                results[key][config_name][dataset] = value
            except OSError as e:
                print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)

    return results


def _extract_threshold(config_name):
    """Extract numeric threshold value from config name like 'entropy_0.5' or 'neural_Qwen3-8B_router_v1_1.0'."""
    parts = config_name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return float(parts[1])
        except ValueError:
            return None
    return None


def _sort_configs_by_threshold(config_dict):
    """Sort config names by their threshold value."""
    configs = list(config_dict.keys())
    threshold_pairs = []
    for c in configs:
        t = _extract_threshold(c)
        if t is not None:
            threshold_pairs.append((t, c))
        else:
            threshold_pairs.append((float("inf"), c))
    threshold_pairs.sort(key=lambda x: x[0])
    return [c for _, c in threshold_pairs]


def _threshold_str(config_name):
    """Raw threshold token (last underscore segment), preserving its literal form."""
    return config_name.rsplit("_", 1)[-1]


def _parse_variation(config_name):
    """Drafter variation (df-e3/ddt-e3/df-ddt) from a WhiFlash config tag, else None.

    Tag format: <router>_<variation>[_b<budget>]_..._<threshold>. The variation is the
    token right after the router prefix (variations contain '-', never '_').
    """
    parts = config_name.split("_")
    if len(parts) >= 2 and parts[1] in WHIFLASH_VARIATIONS:
        return parts[1]
    return None


def _parse_budget(config_name):
    """DDTree node budget from a '_b<N>' segment, else None."""
    m = re.search(r"_b(\d+)(?:_|$)", config_name)
    return int(m.group(1)) if m else None


def _whiflash_sort_key(config_name):
    """Order WhiFlash configs by (variation, budget, threshold) for stable columns."""
    var = _parse_variation(config_name)
    var_idx = WHIFLASH_VARIATIONS.index(var) if var in WHIFLASH_VARIATIONS else -1
    budget = _parse_budget(config_name)
    thr = _extract_threshold(config_name)
    return (var_idx, budget if budget is not None else -1, thr if thr is not None else float("inf"))


def _whiflash_header(config_name, label):
    """Unambiguous column header: <label>[_<variation>][_b<budget>]_<threshold>."""
    out = [label]
    var = _parse_variation(config_name)
    budget = _parse_budget(config_name)
    if var is not None:
        out.append(var)
    if budget is not None:
        out.append(f"b{budget}")
    thr = _extract_threshold(config_name)
    out.append(_threshold_str(config_name) if thr is not None else config_name)
    return "_".join(out)


def _filter_whiflash_configs(configs, variation=None, budget=None):
    """Filter a {config_name: data} dict by variation and/or budget.

    A budget filter keeps no-budget configs too (e.g. df-e3 has no budget), so a
    fixed-budget view still includes the df-e3 baseline alongside the DDTree runs.
    """
    out = {}
    for c, data in configs.items():
        if variation is not None and _parse_variation(c) != variation:
            continue
        if budget is not None:
            b = _parse_budget(c)
            if b is not None and b != budget:
                continue
        out[c] = data
    return out


def _whiflash_columns(configs, label):
    """(header, data) columns for WhiFlash configs: one per (variation, budget, threshold) for a sweep,
    merged into one per (variation, budget) for a calibrated run (each dataset ran at one threshold)."""
    groups = defaultdict(list)
    for config in sorted(configs, key=_whiflash_sort_key):
        groups[(_parse_variation(config), _parse_budget(config))].append(config)
    columns = []
    for (var, budget), group in groups.items():
        if len(group) > 1 and all(sum(d in configs[c] for c in group) <= 1 for d in DATASET_ORDER):
            header = "_".join([label] + ([var] if var else []) + ([f"b{budget}"] if budget is not None else []))
            columns.append((header, {d: v for c in group for d, v in configs[c].items()}))
        else:
            columns.extend((_whiflash_header(c, label), configs[c]) for c in group)
    return columns


def print_sweep_table(results, sweep_prefix, metric="throughput"):
    """Print a multi-column sweep table for a specific router type."""
    matching_configs = {k: v for k, v in results.items() if k.startswith(sweep_prefix)}

    if not matching_configs:
        print(f"No configs found matching prefix '{sweep_prefix}'.", file=sys.stderr)
        return

    # Group by (variation, budget, threshold); degrades to threshold-only for
    # non-WhiFlash configs (e.g. budget_<N>, oracle_<type>).
    sorted_configs = sorted(matching_configs, key=_whiflash_sort_key)

    header = "Dataset " + " ".join(sorted_configs)
    print(header)
    print("-" * (len(header) + 10))

    for dataset in DATASET_ORDER:
        values = []
        for config in sorted_configs:
            val = matching_configs[config].get(dataset)
            values.append(f"{val:.2f}" if val is not None else "N/A")
        print(f"{dataset} " + " ".join(values))

    print("\n--- Values only (for spreadsheet) ---")
    for dataset in DATASET_ORDER:
        values = []
        for config in sorted_configs:
            val = matching_configs[config].get(dataset)
            values.append(f"{val:.2f}" if val is not None else "N/A")
        print(" ".join(values))


MODEL_NAMES = {"qwen8b": "Qwen3-8B", "qwen4b": "Qwen3-4B", "llama8b": "Llama-3.1-8B-Instruct"}


def _apply_thresholds(configs, table, model_type):
    """Keep, per dataset, only the neural config at the threshold `table` assigns to its category and pair."""
    out = {}
    for config, data in configs.items():
        var, budget, thr = _parse_variation(config) or "df-e3", _parse_budget(config), _extract_threshold(config)
        router = re.match(r"neural_(?:(?:df-e3|ddt-e3|df-ddt)_(?:b\d+_)?)?(.+)_[^_]+$", config).group(1)
        if not router_matches_pair(router, var, str(budget) if budget is not None else None):
            continue
        pair = f"{var}-b{budget}" if budget is not None else var
        keep = {
            d: v
            for d, v in data.items()
            if float(table.get(f"{MODEL_NAMES[model_type]}:{CATEGORY.get(d)}:{pair}", "nan")) == thr
        }
        if keep:
            out[config] = keep
    return out


def print_full_table(
    logs_dir,
    metric="throughput",
    model_type=None,
    variation=None,
    budget=None,
    speedup=True,
    split="all",
    thresholds=None,
):
    """Print the complete results table with columns:
    NOSPEC | EAGLE3 | DFLASH | DDTree budget sweep | WhiFlash entropy sweep | WhiFlash neural sweep

    WhiFlash entropy/neural columns are emitted per (variation, budget, threshold) with
    unambiguous headers (e.g. ent_ddt-e3_b128_0.5), or one per (variation, budget) for a calibrated
    run where each dataset ran at a single threshold (e.g. dolci_ddt-e3_b128). --variation / --budget
    narrow the view; by default all variations and budgets are shown. split="eval" reports each run on its
    eval samples only; thresholds ({"<model>:<category>:<pair>": thr}) keeps one threshold per category.

    With speedup (the default for throughput), every column is divided by NOSPEC's throughput on the same
    dataset, and the NOSPEC column itself is dropped.
    """
    all_results = find_all_results(logs_dir, metric=metric, model_type=model_type, split=split)

    columns = []

    # NOSPEC
    if all_results["NOSPEC"]:
        columns.append(("NOSPEC", all_results["NOSPEC"]))

    # EAGLE3
    if all_results["EAGLE3"]:
        columns.append(("EAGLE3", all_results["EAGLE3"]))

    # DFLASH
    if all_results["DFLASH"]:
        columns.append(("DFLASH", all_results["DFLASH"]))

    # Standalone DDTree (tree-budget sweep; config = "budget_<N>")
    ddtree_configs = all_results.get("DDTREE", {})
    if ddtree_configs:
        for config in sorted(ddtree_configs, key=lambda c: _extract_threshold(c) or 0):
            b = _extract_threshold(config)
            header = f"ddt_b{int(b)}" if b is not None else f"ddt_{config}"
            columns.append((header, ddtree_configs[config]))

    # WhiFlash entropy sweep (per variation/budget/threshold)
    entropy_configs = _filter_whiflash_configs(all_results.get("WHIFLASH_entropy", {}), variation, budget)
    columns.extend(_whiflash_columns(entropy_configs, "ent"))

    # WhiFlash neural sweep (per router model, then variation/budget/threshold)
    neural_keys = sorted(k for k in all_results if k.startswith("WHIFLASH_neural_"))
    for neural_key in neural_keys:
        neural_configs = _filter_whiflash_configs(all_results[neural_key], variation, budget)
        if thresholds is not None:
            neural_configs = _apply_thresholds(neural_configs, thresholds, model_type)
        if not neural_configs:
            continue
        columns.extend(_whiflash_columns(neural_configs, neural_key.replace("WHIFLASH_neural_", "")))

    # WhiFlash Oracle-Token (based on accept_length and utility)
    oracle_configs = all_results.get("WHIFLASH_oracle", {})
    if oracle_configs:
        sorted_oracle = _sort_configs_by_threshold(oracle_configs)
        for config in sorted_oracle:
            threshold = _extract_threshold(config)
            header = f"{threshold}" if threshold is not None else f"{config}"
            columns.append((header, oracle_configs[config]))

    if not columns:
        print("No results found.", file=sys.stderr)
        return

    if speedup and metric == "throughput":
        nospec = all_results["NOSPEC"]
        if nospec:
            columns = [
                (h, {d: v / nospec[d] for d, v in data.items() if d in nospec}) for h, data in columns if h != "NOSPEC"
            ]
            print("# Speedup over NOSPEC (throughput ratio per dataset); --no-speedup for tok/s")
        else:
            print("Warning: no NOSPEC runs found, printing throughput instead of speedup.", file=sys.stderr)

    # Rows: datasets, then category means and the mean over all datasets (as in the README tables).
    rows = [(d, [data.get(d) for _, data in columns]) for d in DATASET_ORDER]
    groups = [(f"[{c}]", [d for d in DATASET_ORDER if CATEGORY[d] == c]) for c in CATS] + [("[avg]", DATASET_ORDER)]
    for name, dss in groups:
        means = []
        for _, data in columns:
            vals = [data[d] for d in dss if d in data]
            means.append(sum(vals) / len(vals) if vals else None)
        rows.append((name, means))

    header = "Dataset " + " ".join(h for h, _ in columns)
    print(header)
    print("-" * (len(header) + 10))
    for name, vals in rows:
        print(f"{name} " + " ".join(f"{v:.2f}" if v is not None else "N/A" for v in vals))

    print("\n--- Values only (for spreadsheet) ---")
    for _, vals in rows:
        print(" ".join(f"{v:.2f}" if v is not None else "N/A" for v in vals))


def print_results(results, config_subfolder=False, metric="throughput"):
    """Print results in vertical format for easy spreadsheet pasting."""
    metric_header = METRIC_DEFS[metric][1]

    if config_subfolder:
        configs = sorted(results.keys())
        for config in configs:
            print(f"\n# {config}")
            print(f"Dataset {metric_header}")
            print("-" * 30)
            for dataset in DATASET_ORDER:
                if dataset in results[config]:
                    print(f"{dataset} {results[config][dataset]}")
                else:
                    print(f"{dataset} N/A")
            print("\n--- Values only (for spreadsheet) ---")
            values = [str(results[config].get(ds, "N/A")) for ds in DATASET_ORDER]
            print("\n".join(values))
    else:
        print(f"Dataset {metric_header}")
        print("-" * 30)
        for dataset in DATASET_ORDER:
            if dataset in results:
                print(f"{dataset} {results[dataset]}")
            else:
                print(f"{dataset} N/A")
        print("\n--- Values only (for spreadsheet) ---")
        values = [str(results.get(ds, "N/A")) for ds in DATASET_ORDER]
        print("\n".join(values))


def list_available_configs(logs_dir):
    """List all available config subfolders in the logs directory."""
    configs = set()
    for root, dirs, files in os.walk(logs_dir):
        for d in dirs:
            potential_log = os.path.join(root, d, "transformers.log")
            if os.path.exists(potential_log):
                configs.add(d)
    return sorted(configs)


def main():
    parser = argparse.ArgumentParser(description="Find and organize benchmark results from nested log directories.")
    parser.add_argument(
        "logs_dir", nargs="?", default=".", help="Root logs directory to search (default: current directory)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=list(METRIC_DEFS.keys()),
        default="throughput",
        help="Metric to extract from log files (default: throughput)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(NEURAL_ROUTER_MODELS.keys()),
        default=None,
        help="Model type to filter results by: 'qwen8b' (Qwen3-8B), 'qwen4b' (Qwen3-4B), "
        "'llama8b' (Llama-3.1-8B). Filters log paths to only include matching models.",
    )
    parser.add_argument(
        "--config-subfolder",
        action="store_true",
        help="Expect config subfolders under each dataset directory " "(e.g., dataset/entropy_0.5/transformers.log)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Specific config subfolder to read (e.g., 'entropy_0.5'). " "Automatically enables config-subfolder mode.",
    )
    parser.add_argument(
        "--list-configs", action="store_true", help="List available config subfolders in the logs directory"
    )
    parser.add_argument(
        "--sweep-table",
        action="store_true",
        help="Print a multi-column sweep table (rows=datasets, columns=thresholds). "
        "Requires --sweep-prefix. Automatically enables config-subfolder mode.",
    )
    parser.add_argument(
        "--sweep-prefix",
        type=str,
        default=None,
        help="Prefix for sweep table config filtering (e.g., 'entropy' or 'neural'). " "Only used with --sweep-table.",
    )
    parser.add_argument(
        "--full-table",
        action="store_true",
        help="Print the complete results table: NOSPEC | EAGLE3 | DFLASH | DDTree budget sweep | "
        "WhiFlash entropy sweep | WhiFlash neural sweep. Rows = datasets, columns = algorithms/configs. "
        "By default shows ALL WhiFlash variations and DDTree budgets.",
    )
    parser.add_argument(
        "--speedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --full-table and --metric throughput: report speedup over NOSPEC (default) "
        "instead of tok/s (--no-speedup).",
    )
    parser.add_argument(
        "--split",
        choices=["all", "eval"],
        default="all",
        help="With --full-table: 'eval' reports each run on its eval samples only (the samples outside "
        "benchmark.py's --split calibration draw); 'all' (default) uses the logged values.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=None,
        help='With --full-table: file holding a ["<model>:<category>:<pair>"]="<thr>" table (benchmark.sh, or '
        "calibration_select_thresholds.py output); keeps one WhiFlash-Neural threshold per category. "
        "Needs --model-type.",
    )
    parser.add_argument(
        "--variation",
        type=str,
        choices=list(WHIFLASH_VARIATIONS),
        default=None,
        help="Restrict WhiFlash columns to one drafter variation (df-e3 / ddt-e3 / df-ddt). "
        "Only affects --full-table; default shows all.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Restrict WhiFlash DDTree columns to one tree budget (e.g. 64). Keeps no-budget "
        "configs (df-e3) too. Only affects --full-table; default shows all budgets.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"Error: {args.logs_dir} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    if args.list_configs:
        configs = list_available_configs(args.logs_dir)
        if configs:
            print("Available config subfolders:")
            for c in configs:
                print(f"  {c}")
        else:
            print("No config subfolders found.")
        sys.exit(0)

    # Auto-activate config-subfolder mode when --config or --sweep-table is given
    if args.config or args.sweep_table:
        args.config_subfolder = True

    # Validate --sweep-table requirements
    if args.sweep_table and not args.sweep_prefix:
        print("Error: --sweep-table requires --sweep-prefix (e.g., 'entropy' or 'neural')", file=sys.stderr)
        sys.exit(1)

    if args.thresholds and not args.model_type:
        print("Error: --thresholds needs --model-type (the table keys include the model name).", file=sys.stderr)
        sys.exit(1)

    # Full table mode
    if args.full_table:
        print_full_table(
            args.logs_dir,
            metric=args.metric,
            model_type=args.model_type,
            variation=args.variation,
            budget=args.budget,
            speedup=args.speedup,
            split=args.split,
            thresholds=dict(re.findall(r'\["([^"]+:[^"]+:[^"]+)"\]="([^"]+)"', args.thresholds.read_text()))
            if args.thresholds
            else None,
        )
        return

    # Find and parse results
    results = find_metric_in_logs(
        args.logs_dir,
        metric=args.metric,
        config_subfolder=args.config_subfolder,
        config_filter=args.config,
        model_type=args.model_type,
    )

    if not results:
        print("No results found.", file=sys.stderr)
        if args.config:
            print(f"No results found for config '{args.config}'.", file=sys.stderr)
            print("Use --list-configs to see available options.", file=sys.stderr)
        sys.exit(1)

    # Sweep table mode
    if args.sweep_table:
        print_sweep_table(results, args.sweep_prefix, metric=args.metric)
        return

    # Default: print formatted output
    print_results(results, config_subfolder=args.config_subfolder, metric=args.metric)


if __name__ == "__main__":
    main()
