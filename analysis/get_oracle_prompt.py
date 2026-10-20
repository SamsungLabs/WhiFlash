"""Compute Oracle-Prompt TPS / accept-length from EAGLE3 + DFlash benchmark JSONLs.

Oracle-Prompt — per-prompt picks the single drafter (EAGLE3 or DFlash) with
higher TPS for that prompt. Upper bound on prompt-level routing (HedgeSpec
class). Derivable post-hoc from two single-drafter runs.

(Oracle-Token is computed elsewhere from a real WHIFLASH+oracle run's
transformers.log — not handled here.)

Usage:
    uv run analysis/get_oracle_prompt.py \\
        --eagle3 logs/.../EAGLE3/.../transformers.jsonl \\
        --dflash logs/.../DFLASH/.../transformers.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_jsonl_turns(path: Path) -> dict[tuple[str, int], dict]:
    """Read benchmark.py output, return {(question_id, turn_index): turn_dict}."""
    out: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["question_id"]
            for turn in rec["turns"]:
                key = (qid, int(turn["turn_index"]))
                out[key] = turn
    return out


def per_method_aggregate(turns: dict[tuple[str, int], dict], keys: list[tuple[str, int]]) -> dict[str, float]:
    """Aggregate TPS and accept_length across the given keys.

    TPS aggregation matches benchmark.py's `[Decoding] Average Throughput`:
        agg_tps = 1.0 / mean(time_per_output_token)
    which is the harmonic mean of per-turn TPS values (NOT the arithmetic
    mean) — that's the standard speedup denominator in spec-decoding papers.
    """
    mean_tpot = mean(turns[k]["time_per_output_token"] for k in keys)
    return {
        "mean_tps": 1.0 / mean_tpot if mean_tpot > 0 else 0.0,
        "mean_accept_length": mean(turns[k]["avg_accept_length"] for k in keys),
        "n_turns": len(keys),
    }


def oracle_prompt(
    eagle3: dict[tuple[str, int], dict],
    dflash: dict[tuple[str, int], dict],
    keys: list[tuple[str, int]],
) -> tuple[dict[str, float], dict[str, int]]:
    """Per-key pick the higher-TPS drafter; aggregate with harmonic-mean TPS.

    Selection is per-turn argmax over `tps` (which drafter is faster for that
    prompt — equivalent to argmin over `time_per_output_token`). Aggregation
    uses 1 / mean(chosen_tpot) to match benchmark.py's reporting formula.
    """
    chosen_tpot: list[float] = []
    chosen_acc: list[float] = []
    selection: dict[str, int] = defaultdict(int)
    for k in keys:
        e_tps = eagle3[k]["tps"]
        d_tps = dflash[k]["tps"]
        if e_tps >= d_tps:
            chosen_tpot.append(eagle3[k]["time_per_output_token"])
            chosen_acc.append(eagle3[k]["avg_accept_length"])
            selection["EAGLE3"] += 1
        else:
            chosen_tpot.append(dflash[k]["time_per_output_token"])
            chosen_acc.append(dflash[k]["avg_accept_length"])
            selection["DFlash"] += 1
    mean_tpot = mean(chosen_tpot)
    return (
        {
            "mean_tps": 1.0 / mean_tpot if mean_tpot > 0 else 0.0,
            "mean_accept_length": mean(chosen_acc),
            "n_turns": len(keys),
        },
        dict(selection),
    )


def fmt_row(label: str, stats: dict, selection: str = "—") -> str:
    return f"  {label:<18} {stats['mean_tps']:>9.2f}  " f"{stats['mean_accept_length']:>13.3f}  {selection:>20}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eagle3", type=Path, required=True, help="EAGLE3-only transformers.jsonl")
    p.add_argument("--dflash", type=Path, required=True, help="DFlash-only transformers.jsonl")
    p.add_argument(
        "--per-question",
        action="store_true",
        help="Aggregate per question_id (averaging across multi-turn) before computing "
        "Oracle-Prompt. Default: aggregate per (question_id, turn_index) — each turn "
        "is treated as one prompt, matching how spec-decoding papers typically report.",
    )
    args = p.parse_args()

    eagle3 = load_jsonl_turns(args.eagle3)
    dflash = load_jsonl_turns(args.dflash)

    # Intersect keys.
    keys_e = set(eagle3.keys())
    keys_d = set(dflash.keys())
    common = sorted(keys_e & keys_d)
    dropped_e = len(keys_e - keys_d)
    dropped_d = len(keys_d - keys_e)
    if dropped_e or dropped_d:
        print(f"WARNING: dropped {dropped_e} EAGLE3-only and {dropped_d} DFlash-only turns.")

    if args.per_question:
        # Aggregate per question by averaging TPS / accept across turns first,
        # then compute Oracle-Prompt at question granularity.
        def avg_by_qid(turns: dict[tuple[str, int], dict]) -> dict[tuple[str, int], dict]:
            by_qid: dict[str, list[dict]] = defaultdict(list)
            for (qid, _), t in turns.items():
                by_qid[qid].append(t)
            out: dict[tuple[str, int], dict] = {}
            for qid, ts in by_qid.items():
                mean_tpot = mean(t["time_per_output_token"] for t in ts)
                # Synthetic turn_index=0 so the rest of the pipeline reuses
                # the same (qid, turn_index) → turn dict shape.
                out[(qid, 0)] = {
                    "time_per_output_token": mean_tpot,
                    "tps": 1.0 / mean_tpot if mean_tpot > 0 else 0.0,
                    "avg_accept_length": mean(t["avg_accept_length"] for t in ts),
                }
            return out

        eagle3 = avg_by_qid({k: v for k, v in eagle3.items() if k in common})
        dflash = avg_by_qid({k: v for k, v in dflash.items() if k in common})
        common = sorted(set(eagle3.keys()) & set(dflash.keys()))

    # Per-method aggregates on the common set.
    e_stats = per_method_aggregate(eagle3, common)
    d_stats = per_method_aggregate(dflash, common)
    op_stats, op_sel = oracle_prompt(eagle3, dflash, common)

    granularity = "question" if args.per_question else "turn"
    print(f"\n=== Oracle-Prompt analysis (granularity={granularity}, n={len(common)}) ===\n")
    print(f"  {'Method':<18} {'mean TPS':>9}  {'mean accept_len':>13}  {'selection':>20}")
    print(f"  {'-'*18} {'-'*9}  {'-'*13}  {'-'*20}")
    print(fmt_row("EAGLE3", e_stats))
    print(fmt_row("DFlash", d_stats))
    op_sel_str = f"E:{op_sel.get('EAGLE3', 0)} / D:{op_sel.get('DFlash', 0)}"
    print(fmt_row("Oracle-Prompt", op_stats, op_sel_str))

    print("\n  Speedups (TPS ratio):")
    print(f"    Oracle-Prompt / EAGLE3 : {op_stats['mean_tps'] / e_stats['mean_tps']:.3f}x")
    print(f"    Oracle-Prompt / DFlash : {op_stats['mean_tps'] / d_stats['mean_tps']:.3f}x")
    print()


if __name__ == "__main__":
    main()
