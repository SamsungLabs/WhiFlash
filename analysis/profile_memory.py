#!/usr/bin/env python3
"""Profile GPU memory consumption of speculative decoding methods.

Measures overall GPU memory at each lifecycle boundary for:
  - NOSPEC (target-only autoregressive baseline)
  - DFlash (target + DFlashDraftModel)
  - EAGLE3 (target + Eagle3DraftModel)
  - WHIFLASH-entropy (target + DFlashDraftModel + Eagle3DraftModel + EntropyThresholdRouter)
  - WHIFLASH-neural (target + DFlashDraftModel + Eagle3DraftModel + NeuralRouter)

Usage:
    uv run profile_memory.py \
        --speculative-algorithm NOSPEC \
        --model-name-or-path Qwen/Qwen3-8B

    uv run profile_memory.py \
        --speculative-algorithm DFLASH \
        --model-name-or-path Qwen/Qwen3-8B \
        --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16

    uv run profile_memory.py \
        --speculative-algorithm EAGLE3 \
        --model-name-or-path Qwen/Qwen3-8B \
        --draft-name-or-path WhiFlash/Qwen3-8B-EAGLE3

    uv run profile_memory.py \
        --speculative-algorithm WHIFLASH \
        --whiflash-router entropy \
        --model-name-or-path Qwen/Qwen3-8B \
        --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
        --whiflash-eagle-draft WhiFlash/Qwen3-8B-EAGLE3

    uv run profile_memory.py \
        --speculative-algorithm WHIFLASH \
        --whiflash-router neural \
        --whiflash-neural-router-path WhiFlash/Qwen3-8B-NeuralRouter-df-e3 \
        --model-name-or-path Qwen/Qwen3-8B \
        --draft-name-or-path z-lab/Qwen3-8B-DFlash-b16 \
        --whiflash-eagle-draft WhiFlash/Qwen3-8B-EAGLE3

Output:
    - JSON report: <output_prefix>_memory.json
    - Console summary table
"""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generation import EntropyThresholdRouter, NeuralRouter, dflash_generate, eagle3_generate, whiflash_generate
from model import DFlashDraftModel, Eagle3DraftModel

##############################################################################
# Memory measurement helpers
##############################################################################


def measure_model_weights(model: torch.nn.Module) -> int:
    """Total bytes for all parameters and buffers."""
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total


def measure_unique_weights(models: list[torch.nn.Module]) -> int:
    """Total bytes for all parameters/buffers, deduplicating shared tensors.

    Eagle3DraftModel shares embed_tokens and lm_head with the target model,
    so naive per-model summation double-counts them. This function tracks
    unique tensor data pointers to avoid that.
    """
    seen: set[int] = set()
    total = 0
    for model in models:
        if model is None:
            continue
        for p in model.parameters():
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += p.nelement() * p.element_size()
        for b in model.buffers():
            ptr = b.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += b.nelement() * b.element_size()
    return total


def measure_kv_cache_memory(cache) -> int:
    """Total bytes for all key + value tensors in a DynamicCache."""
    total = 0
    if hasattr(cache, "key_cache"):
        for layer_idx in range(len(cache.key_cache)):
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
            total += k.nelement() * k.element_size()
            total += v.nelement() * v.element_size()
    elif hasattr(cache, "layers"):
        for layer in cache.layers:
            total += layer.keys.nelement() * layer.keys.element_size()
            total += layer.values.nelement() * layer.values.element_size()
    return total


def snapshot_gpu_memory(device: torch.device, label: str) -> dict:
    """Capture current GPU memory state."""
    return {
        "label": label,
        "gpu_allocated_bytes": torch.cuda.memory_allocated(device),
        "gpu_reserved_bytes": torch.cuda.memory_reserved(device),
        "gpu_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


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


##############################################################################
# Main
##############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GPU memory of speculative decoding methods")

    # Algorithm selection
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        required=True,
        choices=["DFLASH", "EAGLE3", "WHIFLASH", "NOSPEC"],
    )

    # Model paths
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--draft-name-or-path",
        type=str,
        default=None,
        help="Draft model path. For WHIFLASH, this is the DFlash draft path.",
    )
    parser.add_argument(
        "--whiflash-eagle-draft",
        type=str,
        default=None,
        help="EAGLE3 draft model path (WHIFLASH only).",
    )

    # Router (WHIFLASH only)
    parser.add_argument(
        "--whiflash-router",
        type=str,
        default="entropy",
        choices=["entropy", "neural"],
        help="WHIFLASH router type. Default 'entropy'.",
    )
    parser.add_argument(
        "--whiflash-neural-router-path",
        type=str,
        default=None,
        help="Path to neural router model (safetensors). Required if --whiflash-router is 'neural'.",
    )
    parser.add_argument(
        "--whiflash-neural-router-threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--whiflash-entropy-threshold",
        type=float,
        default=2.0,
    )

    # Generation
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max tokens to generate (keep small for memory profiling)"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eagle-num-steps", type=int, default=5)
    parser.add_argument("--eagle-topk", type=int, default=4)
    parser.add_argument("--eagle-num-draft-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Prompt for profiling run",
    )

    # Output
    parser.add_argument("--output-prefix", type=str, default="profile_memory", help="Prefix for output files")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["auto", "flash_attention_2", "fa2", "sdpa"])

    args = parser.parse_args()

    # Validate
    if args.speculative_algorithm == "DFLASH" and args.draft_name_or_path is None:
        parser.error("--draft-name-or-path is required for DFLASH")
    if args.speculative_algorithm == "EAGLE3" and args.draft_name_or_path is None:
        parser.error("--draft-name-or-path is required for EAGLE3")
    if args.speculative_algorithm == "WHIFLASH":
        if args.draft_name_or_path is None:
            parser.error("--draft-name-or-path (DFlash draft) is required for WHIFLASH")
        if args.whiflash_eagle_draft is None:
            parser.error("--whiflash-eagle-draft is required for WHIFLASH")
        if args.whiflash_router == "neural" and args.whiflash_neural_router_path is None:
            parser.error("--whiflash-neural-router-path is required when --whiflash-router is 'neural'")

    # ── Device setup ──────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    attn_impl = args.attn_impl
    if attn_impl == "fa2":
        attn_impl = "flash_attention_2"

    # ── Memory tracking ────────────────────────────────────────────────────
    snapshots = []
    component_breakdown = {}

    # Phase A: Baseline
    torch.cuda.init()  # the memory-stats calls below fail with "Invalid device argument" before CUDA is initialised
    torch.cuda.reset_peak_memory_stats(device)
    snapshots.append(snapshot_gpu_memory(device, "baseline"))

    # ── Phase B: Load target model ─────────────────────────────────────────
    print("Loading target model...")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    component_breakdown["target_weights"] = measure_model_weights(target)
    snapshots.append(snapshot_gpu_memory(device, "after_target_load"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ── Phase C: Load draft model(s) ──────────────────────────────────────
    dflash_draft = None
    eagle3_draft = None

    if args.speculative_algorithm == "DFLASH":
        print("Loading DFlash draft model...")
        dflash_draft = (
            DFlashDraftModel.from_pretrained(
                args.draft_name_or_path,
                attn_implementation=attn_impl,
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        component_breakdown["dflash_weights"] = measure_model_weights(dflash_draft)
        snapshots.append(snapshot_gpu_memory(device, "after_draft_load"))

    elif args.speculative_algorithm == "EAGLE3":
        print("Loading EAGLE3 draft model...")
        eagle3_draft = (
            Eagle3DraftModel.from_pretrained(
                args.draft_name_or_path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        eagle3_draft.set_target_model(target)
        component_breakdown["eagle3_weights"] = measure_model_weights(eagle3_draft)
        snapshots.append(snapshot_gpu_memory(device, "after_draft_load"))

    elif args.speculative_algorithm == "NOSPEC":
        # No draft model needed — autoregressive target-only baseline
        snapshots.append(snapshot_gpu_memory(device, "after_draft_load"))

    elif args.speculative_algorithm == "WHIFLASH":
        print("Loading DFlash draft model...")
        dflash_draft = (
            DFlashDraftModel.from_pretrained(
                args.draft_name_or_path,
                attn_implementation=attn_impl,
                dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        component_breakdown["dflash_weights"] = measure_model_weights(dflash_draft)
        snapshots.append(snapshot_gpu_memory(device, "after_dflash_load"))

        print("Loading EAGLE3 draft model...")
        eagle3_draft = (
            Eagle3DraftModel.from_pretrained(
                args.whiflash_eagle_draft,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        eagle3_draft.set_target_model(target)
        component_breakdown["eagle3_weights"] = measure_model_weights(eagle3_draft)
        snapshots.append(snapshot_gpu_memory(device, "after_draft_load"))

    # ── Phase D: Load router (WHIFLASH only) ──────────────────────────────
    router = None
    if args.speculative_algorithm == "WHIFLASH":
        if args.whiflash_router == "entropy":
            router = EntropyThresholdRouter(threshold=args.whiflash_entropy_threshold)
            component_breakdown["router_weights"] = 0
        elif args.whiflash_router == "neural":
            router = NeuralRouter(
                model_path=args.whiflash_neural_router_path,
                threshold=args.whiflash_neural_router_threshold,
                device=device,
            )
            component_breakdown["router_weights"] = measure_model_weights(router.model)
        snapshots.append(snapshot_gpu_memory(device, "after_router_load"))

    # ── Prepare input ─────────────────────────────────────────────────────
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    # ── Phase E: Prefill + short generation ────────────────────────────────
    # We run the full generation and track peak memory.
    # Reset peak stats before generation to isolate generation-phase memory.
    torch.cuda.reset_peak_memory_stats(device)
    snapshots.append(snapshot_gpu_memory(device, "before_generation"))

    print(f"Running {args.speculative_algorithm} generation ({args.max_new_tokens} tokens)...")

    with torch.inference_mode():
        if args.speculative_algorithm == "NOSPEC":
            result = dflash_generate(
                model=target,
                target=target,
                input_ids=input_ids,
                mask_token_id=-1,
                max_new_tokens=args.max_new_tokens,
                block_size=1,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

        elif args.speculative_algorithm == "DFLASH":
            block_size = args.block_size if args.block_size is not None else dflash_draft.block_size
            result = dflash_generate(
                model=dflash_draft,
                target=target,
                input_ids=input_ids,
                mask_token_id=dflash_draft.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

        elif args.speculative_algorithm == "EAGLE3":
            result = eagle3_generate(
                model=eagle3_draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                num_steps=args.eagle_num_steps,
                top_k=args.eagle_topk,
                num_draft_tokens=args.eagle_num_draft_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
            )

        elif args.speculative_algorithm == "WHIFLASH":
            block_size = args.block_size if args.block_size is not None else dflash_draft.block_size
            result = whiflash_generate(
                eagle3_model=eagle3_draft,
                dflash_model=dflash_draft,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                router=router,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                eagle3_num_steps=args.eagle_num_steps,
                eagle3_top_k=args.eagle_topk,
                eagle3_num_draft_tokens=args.eagle_num_draft_tokens,
                dflash_block_size=block_size,
                lazy_catchup=True,
                log_timing=False,
            )

    # Phase F: Peak during generation
    peak_during_gen = torch.cuda.max_memory_allocated(device)
    snapshots.append(snapshot_gpu_memory(device, "after_generation"))

    # ── Compute derived metrics ────────────────────────────────────────────
    # Memory deltas between phases
    memory_deltas = {}
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]["gpu_allocated_bytes"]
        curr = snapshots[i]["gpu_allocated_bytes"]
        memory_deltas[f"{snapshots[i-1]['label']}→{snapshots[i]['label']}"] = curr - prev

    # Total overhead vs target-only
    target_only_mem = snapshots[1]["gpu_allocated_bytes"]  # after target load
    peak_mem = peak_during_gen
    overhead_vs_target = peak_mem - target_only_mem

    # Weights total (deduplicated — Eagle3DraftModel shares embed/lm_head with target)
    all_models = [target, dflash_draft, eagle3_draft]
    if router is not None and hasattr(router, "model"):
        all_models.append(router.model)
    weights_total = measure_unique_weights(all_models)

    # KV + buffers = peak - weights (deduplicated)
    kv_and_buffers = peak_mem - weights_total

    # ── Build report ───────────────────────────────────────────────────────
    method_label = args.speculative_algorithm
    if args.speculative_algorithm == "WHIFLASH":
        method_label += f"-{args.whiflash_router}"

    report = {
        "method": method_label,
        "model": args.model_name_or_path,
        "max_new_tokens": args.max_new_tokens,
        "num_output_tokens": result.num_output_tokens,
        "snapshots": snapshots,
        "memory_deltas": memory_deltas,
        "component_breakdown_bytes": component_breakdown,
        "peak_gpu_memory_bytes": peak_mem,
        "overhead_vs_target_only_bytes": overhead_vs_target,
        "weights_total_bytes": weights_total,
        "kv_and_buffers_bytes": kv_and_buffers,
    }

    # ── Export JSON ─────────────────────────────────────────────────────────
    json_path = f"{args.output_prefix}_memory.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMemory report exported to: {json_path}")

    # ── Console report ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  MEMORY PROFILING REPORT — {method_label}")
    print("=" * 80)

    print(f"\n  Generated tokens:  {result.num_output_tokens}")

    print("\n  Lifecycle memory snapshots (gpu_allocated):")
    print("  " + "-" * 60)
    for snap in snapshots:
        print(f"    {snap['label']:30s}  {format_bytes(snap['gpu_allocated_bytes']):>12s}")

    print(f"\n    {'Peak during generation':30s}  {format_bytes(peak_mem):>12s}")

    print("\n  Memory deltas between phases:")
    print("  " + "-" * 60)
    for label, delta in memory_deltas.items():
        sign = "+" if delta >= 0 else ""
        print(f"    {label:40s}  {sign}{format_bytes(delta):>12s}")

    print("\n  Component breakdown (weights):")
    print("  " + "-" * 60)
    for name, size in component_breakdown.items():
        print(f"    {name:30s}  {format_bytes(size):>12s}")
    print(f"    {'TOTAL WEIGHTS':30s}  {format_bytes(weights_total):>12s}")

    print("\n  Summary:")
    print("  " + "-" * 60)
    print(f"    {'Peak GPU memory':30s}  {format_bytes(peak_mem):>12s}")
    print(f"    {'Total weights':30s}  {format_bytes(weights_total):>12s}")
    print(f"    {'KV cache + buffers (derived)':30s}  {format_bytes(kv_and_buffers):>12s}")
    print(f"    {'Overhead vs target-only':30s}  {format_bytes(overhead_vs_target):>12s}")

    if args.speculative_algorithm == "WHIFLASH" and hasattr(result, "drafter_choices"):
        choices = result.drafter_choices
        n = len(choices)
        if n > 0:
            n_e = choices.count("eagle3")
            n_d = choices.count("dflash")
            print(f"\n  Drafter mix: eagle3 {n_e/n*100:.1f}% / dflash {n_d/n*100:.1f}%")

    print()


if __name__ == "__main__":
    main()
