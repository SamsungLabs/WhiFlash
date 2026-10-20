# Benchmark scripts

Run from the repository root. Jobs run in parallel, one per GPU; set `CUDA_VISIBLE_DEVICES` to
choose the GPUs (e.g. to skip busy ones, or `=0` to run one job at a time).

**Benchmark** every method (NoSpec, EAGLE-3, DFlash, DDTree, WhiFlash-Neural) and tabulate the speedup over NoSpec:

```bash
bash scripts/release/benchmark/benchmark.sh
uv run python analysis/read_benchmark_results.py logs/<ddMonYYYY>_split-eval/sdpa --full-table --model-type qwen8b
```

**Re-calibrate** the WhiFlash thresholds (e.g. for other hardware), then paste the printed table into `benchmark.sh`:

```bash
bash scripts/release/benchmark/calibrate_thresholds.sh
uv run python analysis/calibration_select_thresholds.py logs/calibration_<ddMonYYYY>/sdpa
```

Thresholds are calibrated per task category on 10 random samples of each task (5 for AIME), and every method is
evaluated on the remaining samples. `WHIFLASH_SPLIT=all` evaluates on all samples instead.
