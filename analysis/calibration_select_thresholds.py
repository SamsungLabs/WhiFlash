"""Pick one WhiFlash-Neural threshold per task category from calibration runs (benchmark.py --split
calibration, or full sweeps with --from-full-runs) and print the table to paste into
scripts/release/benchmark/benchmark.sh.

Usage:
    uv run python analysis/calibration_select_thresholds.py logs/calibration_<ddMonYYYY>/sdpa [--metric tps|mat]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
from pathlib import Path

CATEGORY = {
    "mt-bench": "chat",
    "alpaca": "chat",
    "ultrachat-test": "chat",
    "wildchat": "chat",
    "gsm8k": "math",
    "math500": "math",
    "aime24": "math",
    "aime25": "math",
    "humaneval": "coding",
    "mbpp": "coding",
    "livecodebench": "coding",
    "swe-bench": "agentic",
    "tau-bench": "agentic",
    "agentbench": "agentic",
    "agent-instruct": "agentic",
}
CATS = ["chat", "math", "coding", "agentic"]
# --from-full-runs: full dataset sizes, to rebuild benchmark.py's sample order, and calibration samples per task.
DATASET_SIZES = {
    "mt-bench": 80,
    "alpaca": 52002,
    "ultrachat-test": 51414,
    "wildchat": 189830,
    "gsm8k": 1319,
    "math500": 500,
    "aime24": 30,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 257,
    "livecodebench": 1055,
    "swe-bench": 300,
    "tau-bench": 162,
    "agentbench": 103,
    "agent-instruct": 80,
}
CALIBRATION_SAMPLES = {"aime24": 5, "aime25": 5}  # 10 otherwise


def calibration_qids(task: str, n: int) -> set[int]:
    """question_ids that `--split calibration` runs, for a task run with n samples."""
    import numpy as np

    from data import calibration_split

    size = DATASET_SIZES[task]
    order = np.random.default_rng(0).permutation(size)[:n].tolist() if size > n else list(range(n))
    return {order[i] for i in calibration_split(n, CALIBRATION_SAMPLES.get(task, 10))[0]}


def router_matches_pair(router: str, drafters: str, budget: str | None) -> bool:
    """Whether a router was trained for this drafter pair (full sweeps may cross them)."""
    tag = re.search(r"NeuralRouter-([a-z0-9-]+?)(?:-b(\d+))?$", router)
    if tag:
        return tag.group(1) == drafters and tag.group(2) == budget
    if "eagle-vs-ddtree" in router:
        return drafters == "ddt-e3" and f"_b{budget}_" in router
    return drafters == "df-e3"


def metric(turns: list, name: str) -> float:
    """benchmark.py's logged throughput (tps) or acceptance length (mat) over all turns."""
    if name == "tps":
        return len(turns) / sum(t["time_per_output_token"] for t in turns)
    return st.fmean(t["avg_accept_length"] for t in turns)


def collect(root: Path, metric_name: str, from_full_runs: bool = False):
    """-> {(model, router checkpoint, pair tag): {task: {threshold: metric}}}"""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for cfg_path in root.rglob("config.json"):
        log = cfg_path.parent / "transformers.log"
        if not log.is_file() or "records to" not in log.read_text(errors="ignore"):
            continue  # unfinished run
        args = json.loads(cfg_path.read_text())["args"]
        task, split = args["dataset"], args.get("split", "all")
        if (
            args["speculative_algorithm"] != "WHIFLASH"
            or args["whiflash_router"] != "neural"
            or task not in CATEGORY
            or split not in ("calibration", "all" if from_full_runs else None)
        ):
            continue
        router = Path(args["whiflash_neural_router_path"]).stem
        drafters, budget = args["whiflash_drafters"], args.get("ddtree_tree_budget")
        recs = [json.loads(line) for line in open(cfg_path.parent / "transformers.jsonl")]
        if split == "all":  # full sweep: keep only the samples --split calibration would run
            if not router_matches_pair(router, drafters, str(budget) if budget else None):
                continue
            keep = calibration_qids(task, len(recs))
            recs = [r for r in recs if r["question_id"] in keep]
        turns = [t for r in recs for t in r["turns"]]
        model = Path(args["model_name_or_path"]).name
        pair = f"{drafters}-b{budget}" if "ddt" in drafters and budget else drafters
        out[(model, router, pair)][task][args["whiflash_neural_router_threshold"]] = metric(turns, metric_name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="log root holding the calibration runs")
    ap.add_argument("--metric", choices=["tps", "mat"], default="tps")
    ap.add_argument(
        "--from-full-runs",
        action="store_true",
        help="also use full sweeps (--split all), restricted to their calibration samples",
    )
    args = ap.parse_args()

    runs = collect(args.root, args.metric, args.from_full_runs)
    if not runs:
        raise SystemExit(f"No finished WhiFlash-Neural calibration runs under {args.root}.")
    print(f"declare -A WHIFLASH_THRESHOLDS_{args.metric.upper()}=(")
    for (model, router, pair), by_task in sorted(runs.items()):
        grid = sorted(set.intersection(*(set(v) for v in by_task.values())))
        print(f"    # {router}")
        for c in CATS:
            tasks = [t for t in by_task if CATEGORY[t] == c]
            if tasks:
                best = max(grid, key=lambda thr: st.fmean(by_task[t][thr] for t in tasks))
                print(f'    ["{model}:{c}:{pair}"]="{best}"')
    print(")")


if __name__ == "__main__":
    main()
