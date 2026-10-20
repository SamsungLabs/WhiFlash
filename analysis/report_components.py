"""Aggregate component-profile JSONs into CSV, markdown, and plots.

Walks ``--root`` for ``components.json`` files (expected layout:
``<root>/<model_name>/<attn_impl>/components.json``) and emits:

  - ``<out>/all_results.csv``      long-format table for ad-hoc analysis
  - ``<out>/summary.md``           paper-ready markdown tables
  - ``<out>/plots/<model>/<attn>/<metric>.png``    one PNG per metric story

The script tolerates partial runs: any optional section in the JSON
(``e2d_switch_dflash_round``, ``flush_baseline_no_logits_to_keep``, ``prefill``)
shows up only if at least one input file contains it.

Usage:
    uv run analysis/report_components.py \\
        --root results/components \\
        --out figures/components_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the repo root is importable (kept for parity with profile_components.py).
_DFLASH_ROOT = Path(__file__).resolve().parent.parent
if str(_DFLASH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DFLASH_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

###############################################################################
# JSON ingest
###############################################################################


def collect_runs(root: Path) -> list[tuple[str, str, dict]]:
    """Walk ``root`` and return a list of ``(model_name, attn_impl, data)``.

    Expects layout ``<root>/<model>/<attn>/components.json``. Falls back to
    using the JSON's own ``model_name`` / ``attn_impl`` fields when the path
    structure differs.
    """
    runs = []
    for path in sorted(Path(root).rglob("components.json")):
        with path.open() as f:
            data = json.load(f)
        # Prefer path-derived names so multiple files distinguish cleanly.
        try:
            model = path.parent.parent.name
            attn = path.parent.name
        except Exception:
            model = data.get("model_name", "unknown")
            attn = data.get("attn_impl", "unknown")
        runs.append((model, attn, data))
    return runs


###############################################################################
# Long-format DataFrame
###############################################################################


def build_long_df(runs: list[tuple[str, str, dict]]) -> pd.DataFrame:
    """One row per (model, attn, kv_len, component, key) measurement.

    ``key`` is the secondary sweep dim: backlog size N for flush, block_size
    for dflash_round / e2d_switch, None for eagle3_round, input_len for prefill.
    """
    rows = []

    for model, attn, data in runs:
        # Per-kv-len components --------------------------------------------
        for kv_str, r in data.get("per_kv_len", {}).items():
            kv = int(kv_str)
            T_e = r.get("T_round_eagle3_ms", float("nan"))

            # eagle3_round per hyperparam config (key = "d{D}_k{K}_n{N}").
            # Older runs may have stored a flat dict (no config sweep); detect
            # that case and treat the whole thing as a single 'default' config.
            eagle3_round_by_cfg = r.get("eagle3_round", {})
            if "median_ms" in eagle3_round_by_cfg:  # legacy flat-dict shape
                eagle3_round_by_cfg = {"default": eagle3_round_by_cfg}
            for cfg_key, stats in eagle3_round_by_cfg.items():
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "eagle3_round",
                        "key": cfg_key,
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "ratio_to_eagle3": 1.0,
                    }
                )

            # eagle3_draft per hyperparam config (key = "d{D}_k{K}_n{N}")
            eagle3_draft_by_cfg = r.get("eagle3_draft", {})
            if "median_ms" in eagle3_draft_by_cfg:
                eagle3_draft_by_cfg = {"default": eagle3_draft_by_cfg}
            for cfg_key, stats in eagle3_draft_by_cfg.items():
                round_med = (
                    eagle3_round_by_cfg.get(cfg_key, {}).get("median_ms")
                    if isinstance(eagle3_round_by_cfg.get(cfg_key, {}), dict)
                    else None
                )
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "eagle3_draft",
                        "key": cfg_key,
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "ratio_to_eagle3": stats.get("median_ms", float("nan")) / T_e if T_e else float("nan"),
                        "draft_over_round": (
                            stats.get("median_ms", float("nan")) / round_med if round_med else float("nan")
                        ),
                    }
                )

            # dflash_round (key = block_size)
            for bs_str, stats in r.get("dflash_round", {}).items():
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "dflash_round",
                        "key": int(bs_str),
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "ratio_to_eagle3": stats.get("median_ms", float("nan")) / T_e if T_e else float("nan"),
                    }
                )

            # dflash_draft — DFlash drafter forward only (key = block_size)
            for bs_str, stats in r.get("dflash_draft", {}).items():
                round_med = r.get("dflash_round", {}).get(bs_str, {}).get("median_ms")
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "dflash_draft",
                        "key": int(bs_str),
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "ratio_to_eagle3": stats.get("median_ms", float("nan")) / T_e if T_e else float("nan"),
                        "draft_over_round": (
                            stats.get("median_ms", float("nan")) / round_med if round_med else float("nan")
                        ),
                    }
                )

            # flush (key = N)
            for n_str, stats in r.get("flush", {}).items():
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "flush",
                        "key": int(n_str),
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "ratio_to_eagle3": stats.get("median_ms", float("nan")) / T_e if T_e else float("nan"),
                    }
                )

            # E→D switch DFlash round (key = block_size)
            for bs_str, stats in r.get("e2d_switch_dflash_round", {}).items():
                steady = r.get("dflash_round", {}).get(bs_str, {}).get("median_ms")
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "e2d_switch",
                        "key": int(bs_str),
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "ratio_to_eagle3": stats.get("median_ms", float("nan")) / T_e if T_e else float("nan"),
                        "switch_over_steady": (
                            stats.get("median_ms", float("nan")) / steady if steady else float("nan")
                        ),
                    }
                )

            # Ablation: flush without logits_to_keep (key = N)
            for n_str, stats in r.get("flush_baseline_no_logits_to_keep", {}).items():
                patched = r.get("flush", {}).get(n_str, {}).get("median_ms")
                rows.append(
                    {
                        "model": model,
                        "attn": attn,
                        "kv_len": kv,
                        "component": "flush_baseline",
                        "key": int(n_str),
                        "median_ms": stats.get("median_ms"),
                        "mean_ms": stats.get("mean_ms"),
                        "std_ms": stats.get("std_ms"),
                        "speedup_vs_baseline": (
                            stats.get("median_ms", float("nan")) / patched if patched else float("nan")
                        ),
                    }
                )

        # Prefill (component-level, keyed by input_len) --------------------
        for kv_str, r in data.get("prefill", {}).items():
            if r.get("oom"):
                continue
            n = r.get("actual_input_len", int(kv_str))
            rows.append(
                {
                    "model": model,
                    "attn": attn,
                    "kv_len": int(kv_str),
                    "component": "prefill",
                    "key": n,
                    "median_ms": r.get("median_ms"),
                    "mean_ms": r.get("mean_ms"),
                    "std_ms": r.get("std_ms"),
                    "tokens_per_s": (n / (r["median_ms"] / 1000) if r.get("median_ms") else float("nan")),
                }
            )

    return pd.DataFrame(rows)


###############################################################################
# Markdown helpers (no tabulate dep)
###############################################################################


def _md_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Hand-rolled markdown table; avoids the optional `tabulate` dependency."""
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    cols_str = [str(c) for c in cols]

    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return "—"
            return float_fmt.format(v)
        return str(v)

    header = "| " + " | ".join(cols_str) + " |"
    sep = "| " + " | ".join("---" for _ in cols_str) + " |"
    body = "\n".join("| " + " | ".join(fmt(row[c]) for c in cols) + " |" for _, row in df.iterrows())
    return f"{header}\n{sep}\n{body}\n"


def _pivot(df: pd.DataFrame, index: str, columns: str, values: str) -> pd.DataFrame:
    if df.empty:
        return df
    p = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    p = p.reset_index()
    p.columns.name = None
    return p


def write_markdown(df: pd.DataFrame, runs: list[tuple[str, str, dict]], out: Path) -> None:
    """Per (model, attn) markdown section with pivot tables."""
    lines = ["# WhiFlash component profile\n"]

    for model, attn, data in runs:
        lines.append(f"\n## {model} · attn={attn}\n")
        cfg = data.get("config", {})
        lines.append(
            f"- EAGLE3: steps={cfg.get('eagle_num_steps')}, "
            f"top_k={cfg.get('eagle_topk')}, "
            f"num_draft_tokens={cfg.get('eagle_num_draft_tokens')}\n"
            f"- block_sizes: {cfg.get('block_sizes')}\n"
            f"- backlog_sizes: {cfg.get('backlog_sizes')}\n"
            f"- base_kv_lens: {cfg.get('base_kv_lens')}\n"
        )
        sub = df[(df.model == model) & (df.attn == attn)]
        if sub.empty:
            lines.append("\n_(no data)_\n")
            continue

        # ── Per-round cost summary (the routing-relevant metric) ──
        # Each (drafter, config) gets its own column. Values are full-round
        # median ms (drafter forward + target verify), since both differ
        # across drafters (different n_verify and different attn kernel).
        er_all = sub[sub.component == "eagle3_round"]
        dr_all = sub[sub.component == "dflash_round"]
        if not er_all.empty or not dr_all.empty:
            er_pivot = _pivot(er_all, "kv_len", "key", "median_ms")
            if er_pivot is not None and not er_pivot.empty:
                er_pivot = er_pivot.rename(columns={c: f"EAGLE3 {c}" for c in er_pivot.columns if c != "kv_len"})
            dr_pivot = _pivot(dr_all, "kv_len", "key", "median_ms")
            if dr_pivot is not None and not dr_pivot.empty:
                dr_pivot = dr_pivot.rename(columns={c: f"DFlash B={c}" for c in dr_pivot.columns if c != "kv_len"})
            summary = (
                er_pivot.merge(dr_pivot, on="kv_len", how="outer").sort_values("kv_len")
                if er_pivot is not None and dr_pivot is not None
                else (er_pivot if dr_pivot is None else dr_pivot)
            )
            lines.append(
                "\n### Per-round cost (median ms): kv_len × (drafter, config) — *the routing-relevant metric*\n"
                "*Round = drafter forward + target verify. Both sub-costs differ across drafters: "
                "EAGLE3 verify processes ~ndt+1 tokens with a 4D tree mask (SDPA only); "
                "DFlash verify processes block_size tokens with FA2-friendly mask.*\n"
            )
            lines.append(_md_table(summary))

        # ── Drafter-only decomposition (sub-component, useful for *why* rounds differ) ──
        ed_all = sub[sub.component == "eagle3_draft"]
        dd_all = sub[sub.component == "dflash_draft"]
        if not ed_all.empty or not dd_all.empty:
            ed_pivot = _pivot(ed_all, "kv_len", "key", "median_ms")
            if ed_pivot is not None and not ed_pivot.empty:
                ed_pivot = ed_pivot.rename(columns={c: f"EAGLE3 {c}" for c in ed_pivot.columns if c != "kv_len"})
            dd_pivot = _pivot(dd_all, "kv_len", "key", "median_ms")
            if dd_pivot is not None and not dd_pivot.empty:
                dd_pivot = dd_pivot.rename(columns={c: f"DFlash B={c}" for c in dd_pivot.columns if c != "kv_len"})
            summary = (
                ed_pivot.merge(dd_pivot, on="kv_len", how="outer").sort_values("kv_len")
                if ed_pivot is not None and dd_pivot is not None
                else (ed_pivot if dd_pivot is None else dd_pivot)
            )
            lines.append(
                "\n### Drafter-only sub-cost (median ms): kv_len × (drafter, config) — sub-component breakdown\n"
            )
            lines.append(_md_table(summary))

        # eagle3_round vs kv_len pivot (kv × config)
        e = sub[sub.component == "eagle3_round"]
        if not e.empty:
            lines.append("\n### `_eagle3_round` median latency (ms): kv_len × eagle config\n")
            lines.append(_md_table(_pivot(e, "kv_len", "key", "median_ms")))

        # dflash_round pivot: kv_len × block_size
        d = sub[sub.component == "dflash_round"]
        if not d.empty:
            lines.append("\n### `_dflash_round` median latency (ms): kv_len × block_size\n")
            lines.append(_md_table(_pivot(d, "kv_len", "key", "median_ms")))
            lines.append("\n### `_dflash_round` ratio to `_eagle3_round`: kv_len × block_size\n")
            lines.append(_md_table(_pivot(d, "kv_len", "key", "ratio_to_eagle3")))

        # dflash_draft pivot: drafter-only latency and draft/round fraction
        dd = sub[sub.component == "dflash_draft"]
        if not dd.empty:
            lines.append("\n### `dflash_draft` (drafter forward only) median (ms): kv_len × block_size\n")
            lines.append(_md_table(_pivot(dd, "kv_len", "key", "median_ms")))
            lines.append("\n### Draft fraction of DFlash round (draft/round): kv_len × block_size\n")
            lines.append(_md_table(_pivot(dd, "kv_len", "key", "draft_over_round")))

        # flush pivot: kv_len × N (median_ms and ratio)
        f = sub[sub.component == "flush"]
        if not f.empty:
            lines.append("\n### `_flush_eagle3_backlog` median latency (ms): kv_len × N\n")
            lines.append(_md_table(_pivot(f, "kv_len", "key", "median_ms")))
            lines.append("\n### Flush as fraction of `_eagle3_round`: kv_len × N\n")
            lines.append(_md_table(_pivot(f, "kv_len", "key", "ratio_to_eagle3")))

        # E→D switch ratio: kv_len × block_size
        e2d = sub[sub.component == "e2d_switch"]
        if not e2d.empty:
            lines.append("\n### Post-E→D-switch DFlash round / steady ratio: kv_len × block_size\n")
            lines.append(_md_table(_pivot(e2d, "kv_len", "key", "switch_over_steady")))

        # logits_to_keep speedup
        ab = sub[sub.component == "flush_baseline"]
        if not ab.empty:
            lines.append("\n### `logits_to_keep` flush speedup (baseline / patched): kv_len × N\n")
            lines.append(_md_table(_pivot(ab, "kv_len", "key", "speedup_vs_baseline")))

        # Prefill
        pf = sub[sub.component == "prefill"]
        if not pf.empty:
            lines.append("\n### Target prefill latency (ms) and throughput (tok/s)\n")
            cols = (
                pf[["key", "median_ms", "tokens_per_s"]].rename(columns={"key": "input_len"}).sort_values("input_len")
            )
            lines.append(_md_table(cols))

    out.write_text("".join(lines))


###############################################################################
# Plots
###############################################################################


def _fig_path(out: Path, model: str, attn: str, name: str) -> Path:
    p = out / "plots" / model / attn
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.png"


def make_plots(df: pd.DataFrame, runs: list[tuple[str, str, dict]], out: Path) -> None:
    if df.empty:
        return

    for model, attn, _ in runs:
        sub = df[(df.model == model) & (df.attn == attn)]
        if sub.empty:
            continue

        # 0a. Per-round cost vs kv_len — HEADLINE plot for routing.
        #     Round = drafter + verify. Both sub-costs differ across drafters
        #     (different n_verify, different attn kernel) so per-round is what
        #     determines cost-per-accepted-token in the actual decode loop.
        er = sub[sub.component == "eagle3_round"].sort_values(["key", "kv_len"])
        dr = sub[sub.component == "dflash_round"].sort_values(["key", "kv_len"])
        if not er.empty or not dr.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ci = 0
            for cfg, g in er.groupby("key"):
                ax.plot(g["kv_len"], g["median_ms"], marker="s", linewidth=2.0, color=f"C{ci}", label=f"EAGLE3 ({cfg})")
                ci += 1
            for bs, g in dr.groupby("key"):
                ax.plot(
                    g["kv_len"], g["median_ms"], marker="o", linewidth=2.0, color=f"C{ci}", label=f"DFlash (B={bs})"
                )
                ci += 1
            ax.set_xscale("log")
            ax.set_xlabel("base_kv_len (tokens)")
            ax.set_ylabel("per-round median latency (ms)")
            ax.set_title(f"{model} · {attn} — PER-ROUND COST vs context length")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            fig.savefig(_fig_path(out, model, attn, "round_cost_vs_kv"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 0b. Drafter-only sub-cost vs kv_len — decomposition / explainer.
        ed = sub[sub.component == "eagle3_draft"].sort_values(["key", "kv_len"])
        dd = sub[sub.component == "dflash_draft"].sort_values(["key", "kv_len"])
        if not ed.empty or not dd.empty:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ci = 0
            for cfg, g in ed.groupby("key"):
                ax.plot(
                    g["kv_len"],
                    g["median_ms"],
                    marker="s",
                    linewidth=1.8,
                    color=f"C{ci}",
                    label=f"EAGLE3 draft ({cfg})",
                )
                ci += 1
            for bs, g in dd.groupby("key"):
                ax.plot(
                    g["kv_len"],
                    g["median_ms"],
                    marker="o",
                    linewidth=1.8,
                    color=f"C{ci}",
                    label=f"DFlash draft (B={bs})",
                )
                ci += 1
            ax.set_xscale("log")
            ax.set_xlabel("base_kv_len (tokens)")
            ax.set_ylabel("drafter-only median latency (ms)")
            ax.set_title(f"{model} · {attn} — drafter-only sub-cost vs context length")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            fig.savefig(_fig_path(out, model, attn, "drafter_cost_vs_kv"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 1. Flush ms vs N, line per kv_len
        f = sub[sub.component == "flush"].sort_values(["kv_len", "key"])
        if not f.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for kv, g in f.groupby("kv_len"):
                ax.plot(g["key"], g["median_ms"], marker="o", label=f"kv={kv}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Backlog size N (tokens)")
            ax.set_ylabel("`_flush_eagle3_backlog` median (ms)")
            ax.set_title(f"{model} · {attn} — D→E switching cost vs N")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.savefig(_fig_path(out, model, attn, "flush_vs_N"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 1b. Flush as % of eagle3 round
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for kv, g in f.groupby("kv_len"):
                ax.plot(g["key"], g["ratio_to_eagle3"] * 100, marker="o", label=f"kv={kv}")
            ax.set_xscale("log")
            ax.set_xlabel("Backlog size N (tokens)")
            ax.set_ylabel("Flush cost as % of `_eagle3_round`")
            ax.set_title(f"{model} · {attn} — switching overhead %")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.savefig(_fig_path(out, model, attn, "flush_pct_eagle3_vs_N"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # (Plots 0a/0b above already cover round and drafter-only curves cleanly.)

        # 3. E→D switch ratio
        e2d = sub[sub.component == "e2d_switch"].sort_values(["key", "kv_len"])
        if not e2d.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for bs, g in e2d.groupby("key"):
                ax.plot(g["kv_len"], g["switch_over_steady"], marker="o", label=f"B={bs}")
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("base_kv_len (tokens)")
            ax.set_ylabel("post-E→D-switch / steady DFlash round (×)")
            ax.set_title(f"{model} · {attn} — E→D switching cost asymmetry")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            fig.savefig(_fig_path(out, model, attn, "e2d_switch_ratio_vs_kv"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 4. logits_to_keep speedup
        ab = sub[sub.component == "flush_baseline"].sort_values(["kv_len", "key"])
        if not ab.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for kv, g in ab.groupby("kv_len"):
                ax.plot(g["key"], g["speedup_vs_baseline"], marker="o", label=f"kv={kv}")
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("Backlog size N (tokens)")
            ax.set_ylabel("logits_to_keep speedup (×)")
            ax.set_title(f"{model} · {attn} — flush speedup from logits_to_keep")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            fig.savefig(_fig_path(out, model, attn, "logits_to_keep_speedup_vs_N"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # 5. Prefill latency
        pf = sub[sub.component == "prefill"].sort_values("key")
        if not pf.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(pf["key"], pf["median_ms"], marker="o")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("input length (tokens)")
            ax.set_ylabel("target prefill median (ms)")
            ax.set_title(f"{model} · {attn} — prefill latency vs input length")
            ax.grid(True, which="both", alpha=0.3)
            fig.savefig(_fig_path(out, model, attn, "prefill_vs_input_len"), dpi=150, bbox_inches="tight")
            plt.close(fig)


###############################################################################
# Entry point
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="Directory tree containing components.json files.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for CSV / markdown / plots.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output.")
    parser.add_argument("--no-markdown", action="store_true", help="Skip markdown output.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(args.root)
    if not runs:
        print(f"No components.json files under {args.root}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(runs)} run(s):")
    for model, attn, _ in runs:
        print(f"  - {model} / {attn}")

    df = build_long_df(runs)

    if not args.no_csv:
        csv_path = args.out / "all_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    if not args.no_markdown:
        md_path = args.out / "summary.md"
        write_markdown(df, runs, md_path)
        print(f"Wrote {md_path}")

    if not args.no_plots:
        make_plots(df, runs, args.out)
        n_plots = len(list((args.out / "plots").rglob("*.png"))) if (args.out / "plots").exists() else 0
        print(f"Wrote {n_plots} plot(s) under {args.out / 'plots'}")


if __name__ == "__main__":
    main()
