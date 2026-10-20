import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import distributed as dist
from data import calibration_split, load_and_process_dataset, load_latency_profile, load_system_prompt
from generation import (
    Drafter,
    EntropyThresholdRouter,
    NeuralRouter,
    OracleRouter,
    ddtree_generate,
    dflash_generate,
    eagle3_generate,
    maybe_enable_cpp_compact,
    whiflash_generate,
)
from model import DFlashDraftModel, Eagle3DraftModel, has_flash_attn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument(
        "--draft-name-or-path",
        type=str,
        default=None,
        help="Draft model path. For WHIFLASH, this is the DFlash draft path. "
        "Optional only for --speculative-algorithm NOSPEC.",
    )
    parser.add_argument("--speculative-algorithm", type=str, default="DFLASH")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for target and drafters. Default is 'bfloat16'. Use 'float32' to match Oracle traces exactly.",
    )

    # DFlash argument
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="DFlash: tokens generated per draft forward. " "Defaults to the draft model's native block size.",
    )

    # DDTree arguments
    parser.add_argument(
        "--ddtree-tree-budget",
        type=int,
        default=None,
        help="DDTree: max number of tree nodes to expand per draft round. Defaults to (block_size - 1).",
    )
    parser.add_argument(
        "--ddtree-enable-cpp-compact-cache",
        action="store_true",
        help="DDTree: enable the inline C++ KV-cache tail compactor. Python fallback is used.",
    )
    parser.add_argument(
        "--profile-gen-stages",
        action="store_true",
        help="Enable the per-stage generation timing breakdown.",
    )

    # EAGLE3 arguments
    parser.add_argument(
        "--eagle-num-steps",
        type=int,
        default=8,
        help="EAGLE: number of draft forward passes (tree depth). Default 8 (match default value in the EAGLE3 paper).",
    )
    parser.add_argument(
        "--eagle-topk",
        type=int,
        default=10,
        help="EAGLE: branching factor per step/depth. Default 10 (match default value in the EAGLE3 paper).",
    )
    parser.add_argument(
        "--eagle-num-draft-tokens",
        type=int,
        default=60,
        help="EAGLE: global candidate budget fed to verifier. Default 60 (match default value in the EAGLE3 paper).",
    )

    # WhiFlash arguments
    parser.add_argument(
        "--whiflash-eagle-draft",
        type=str,
        default=None,
        help="WHIFLASH: EAGLE draft model path. " "--draft-name-or-path is used for DFlash draft path.",
    )
    parser.add_argument(
        "--whiflash-drafters",
        type=str,
        default="df-e3",
        choices=["df-e3", "ddt-e3", "df-ddt"],
        help="WHIFLASH: which two drafters to route between "
        "'df-e3' = EAGLE3 vs DFlash (default), "
        "'ddt-e3' = EAGLE3 vs DDTree, 'df-ddt' = DFlash vs DDTree.",
    )
    parser.add_argument(
        "--whiflash-router",
        type=str,
        default="entropy",
        choices=[
            "entropy",
            "neural",
            "oracle",
        ],
        help="WHIFLASH: routing policy. Default 'entropy'.",
    )
    parser.add_argument(
        "--whiflash-neural-router-path",
        type=str,
        default=None,
        help="WHIFLASH: router checkpoint — a Hub repo id (e.g. 'WhiFlash/Qwen3-8B-NeuralRouter-df-e3'), "
        "a local directory, or an exported .safetensors file. Required if --whiflash-router is 'neural'.",
    )
    parser.add_argument(
        "--whiflash-neural-router-threshold",
        type=float,
        default=0.0,
        help="WHIFLASH: threshold for neural router logit. Default 0.0.",
    )
    parser.add_argument(
        "--whiflash-entropy-threshold",
        type=float,
        default=2.0,
        help="WHIFLASH: H_t threshold (entropy router). Default 2.0.",
    )

    # Latency profile (used by the utility-based oracle router)
    parser.add_argument(
        "--whiflash-latency-config",
        type=str,
        default="configs/whiflash_latency_profile.json",
        help="WHIFLASH: path to JSON profile with per_round_latency_seconds and "
        "switching_cost_seconds. Required for --whiflash-oracle-router-utility-based.",
    )

    # OracleRouter (--whiflash-router oracle)
    parser.add_argument("--whiflash-oracle-router-utility-based", action="store_true")
    parser.add_argument(
        "--whiflash-oracle-utility-mode",
        type=str,
        default="full",
        choices=["full", "partial"],
        help="OracleRouter utility mode: 'full' (default) incorporates switching cost, "
        "'partial' uses acceptance/latency only.",
    )

    parser.add_argument(
        "--whiflash-no-lazy-catchup",
        action="store_true",
        help="WHIFLASH: disable lazy catchup. Default (omit flag) catches up the "
        "inactive drafter only on switch; setting this updates both KVs every round.",
    )
    parser.add_argument(
        "--whiflash-log-timing",
        action="store_true",
        help="WHIFLASH: log switch-overhead timing (adds cuda.synchronize overhead; "
        "disable for accurate throughput/acceptance-rate benchmarks).",
    )

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "calibration", "eval"],
        help="Which part of the selected samples to run: 'calibration' = --calibration-samples samples drawn at random "
        "with a fixed seed (for router-threshold calibration), 'eval' = the remaining ones, disjoint from 'calibration'.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Number of samples reserved for calibration. Required unless --split all.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384)

    # Common arguments
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write JSONL results. If not set, no JSONL is written.",
    )
    parser.add_argument("--num-choices", type=int, default=1)

    # Inference behaviour arguments
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "fa2", "sdpa"],
        help="Attention backend for target and DFlash draft. 'auto' (default) "
        "uses FA2 if installed, else SDPA. EAGLE3 drafts always use SDPA "
        "(custom 4D masks unsupported by FA2).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="default",
        help="System prompt preset name from configs/system_prompts.yaml (e.g. 'llama3'). "
        "If set, prepended as a system message to each conversation.",
    )
    parser.add_argument(
        "--structure-window",
        type=int,
        default=32,
        help="Number of recent output tokens decoded for context-structure scoring.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in chat template (e.g. Qwen3, Llama3, and other reasoning models).",
    )
    parser.add_argument("--per-token-log", action="store_true")
    parser.add_argument(
        "--dump-hidden-states",
        action="store_true",
        help="WHIFLASH + per-token-log only: dump raw target hidden states per "
        "turn as float16 .npz files under <output_dir>/target_hidden_states/. "
        "Each file contains two arrays — dflash, eagle3 — both shaped "
        "[num_steps, 4, hidden_dim] (3 layers from the drafter's "
        "target_layer_ids + avg). Layer IDs are recorded in config.json.",
    )
    parser.add_argument("--skip-nospec", action="store_true")

    args = parser.parse_args()

    if args.system_prompt is not None:
        args.system_prompt = load_system_prompt(args.system_prompt)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    # EAGLE3 and DDTREE verify passes a 4D tree mask which FA2 cannot support
    # WHIFLASH ensures the consistent kernel use between target and draft models, as well as
    # switches per-round inside whiflash_generate.
    if args.attn_impl == "auto":
        use_fa2 = has_flash_attn() and args.speculative_algorithm not in ("EAGLE3", "DDTREE")
        attn_impl = "flash_attention_2" if use_fa2 else "sdpa"
    else:
        attn_impl = args.attn_impl
        if attn_impl.lower() == "fa2":
            attn_impl = "flash_attention_2"
        if attn_impl == "flash_attention_2" and not has_flash_attn():
            raise RuntimeError("--attn-impl flash_attention_2 requested but flash_attn is not installed.")
        if attn_impl == "flash_attention_2" and args.speculative_algorithm in ("EAGLE3", "DDTREE"):
            raise RuntimeError(
                f"{args.speculative_algorithm} verify passes a 4D mask which FA2 cannot support."
                "Re-run with --attn-impl sdpa or auto."
            )

    ######################################################
    # Model Init
    ######################################################
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=args.dtype,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if args.speculative_algorithm == "DFLASH":
        draft_model = (
            DFlashDraftModel.from_pretrained(
                args.draft_name_or_path,
                attn_implementation=attn_impl,
                dtype=args.dtype,
            )
            .to(device)
            .eval()
        )
        block_size = args.block_size if args.block_size is not None else draft_model.block_size
    elif args.speculative_algorithm == "DDTREE":
        # DDTREE reuses the DFlash drafter; only the verification (tree vs linear) differs.
        draft_model = (
            DFlashDraftModel.from_pretrained(
                args.draft_name_or_path,
                attn_implementation=attn_impl,
                dtype=args.dtype,
            )
            .to(device)
            .eval()
        )
        block_size = args.block_size if args.block_size is not None else draft_model.block_size
        maybe_enable_cpp_compact(args.ddtree_enable_cpp_compact_cache)
    elif args.speculative_algorithm == "EAGLE3":
        draft_model = (
            Eagle3DraftModel.from_pretrained(
                args.draft_name_or_path,
                dtype=args.dtype,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        draft_model.set_target_model(target)
        block_size = args.eagle_num_draft_tokens  # unified key for response dict
    elif args.speculative_algorithm == "WHIFLASH":
        if args.draft_name_or_path is None:
            raise ValueError(
                "--draft-name-or-path (DFlash draft path) is required when --speculative-algorithm WHIFLASH"
            )
        whiflash_drafters = {
            "df-e3": (Drafter.DFLASH, Drafter.EAGLE3),
            "ddt-e3": (Drafter.DDTREE, Drafter.EAGLE3),
            "df-ddt": (Drafter.DFLASH, Drafter.DDTREE),
        }[args.whiflash_drafters]
        whiflash_eagle3_in_pair = Drafter.EAGLE3 in whiflash_drafters
        if whiflash_eagle3_in_pair and args.whiflash_eagle_draft is None:
            raise ValueError(
                "--whiflash-eagle-draft is required when EAGLE3 is in the WHIFLASH pair "
                f"(--whiflash-drafters {args.whiflash_drafters})."
            )
        if args.whiflash_router == "oracle" and args.whiflash_drafters != "df-e3":
            raise ValueError(
                "--whiflash-router oracle measures EAGLE3 vs DFlash and is only supported "
                "with --whiflash-drafters df-e3."
            )

        dflash_draft_model = (
            DFlashDraftModel.from_pretrained(
                args.draft_name_or_path,
                attn_implementation=attn_impl,
                dtype=args.dtype,
            )
            .to(device)
            .eval()
        )
        if whiflash_eagle3_in_pair:
            eagle3_draft_model = (
                Eagle3DraftModel.from_pretrained(
                    args.whiflash_eagle_draft,
                    dtype=args.dtype,
                    attn_implementation="sdpa",
                )
                .to(device)
                .eval()
            )
            eagle3_draft_model.set_target_model(target)
        else:
            eagle3_draft_model = None  # DFlash-vs-DDTree: no EAGLE3 drafter needed
        block_size = args.block_size if args.block_size is not None else dflash_draft_model.block_size
        draft_model = None  # not used directly; eagle3_draft_model / dflash_draft_model used below
        if args.whiflash_router == "entropy":
            whiflash_router = EntropyThresholdRouter(threshold=args.whiflash_entropy_threshold)
        elif args.whiflash_router == "neural":
            if args.whiflash_neural_router_path is None:
                raise ValueError("--whiflash-neural-router-path is required when --whiflash-router is 'neural'")
            whiflash_router = NeuralRouter(
                model_path=args.whiflash_neural_router_path,
                threshold=args.whiflash_neural_router_threshold,
                device=device,
            )
        elif args.whiflash_router == "oracle":
            if args.whiflash_oracle_router_utility_based:
                if args.whiflash_latency_config is None:
                    raise ValueError(
                        "--whiflash-latency-config is required when " "--whiflash-oracle-router-utility-based is set"
                    )
                profile = load_latency_profile(args.whiflash_latency_config, args.model_name_or_path)
                switching_cost_raw = profile.get("switching_cost_seconds", {})
                switching_cost = {tuple(k.split("->")): float(v) for k, v in switching_cost_raw.items()}
                whiflash_router = OracleRouter(
                    utility_based=True,
                    utility_mode=args.whiflash_oracle_utility_mode,
                    per_round_latency=profile["per_round_latency_seconds"],
                    switching_cost=switching_cost,
                )
            else:
                whiflash_router = OracleRouter(utility_based=False)
    elif args.speculative_algorithm == "NOSPEC":
        block_size = 1

    # Persist args + values resolved at runtime for reproducibility.
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        runtime_params = {
            "block_size": block_size,
            "attn_impl": attn_impl,
        }
        if args.speculative_algorithm == "WHIFLASH":
            runtime_params["whiflash_drafters"] = args.whiflash_drafters
            runtime_params["dflash_target_layer_ids"] = list(dflash_draft_model.target_layer_ids)
            if eagle3_draft_model is not None:
                runtime_params["eagle3_target_layer_ids"] = list(eagle3_draft_model.target_layer_ids)
        with open(out_dir / "config.json", "w") as f:
            json.dump(
                {
                    "args": vars(args),
                    "runtime_params": runtime_params,
                },
                f,
                indent=4,
            )

    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    # Calibration / evaluation split: router thresholds are tuned on --calibration-samples randomly chosen
    if args.split != "all":
        if args.calibration_samples is None:
            raise ValueError(f"--split {args.split} needs --calibration-samples.")
        calibration, held_out = calibration_split(len(dataset), args.calibration_samples)
        dataset = dataset.select(calibration if args.split == "calibration" else held_out)

    # running instance method used by both warmup and main measurement stages.
    def _run_instance(instance, idx):
        inst_response = []
        for i in range(args.num_choices):
            messages = []
            if args.system_prompt is not None:
                messages.append({"role": "system", "content": args.system_prompt})
            for turn_index, user_content in enumerate(instance["turns"]):
                messages.append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking
                )
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

                response = {}
                if args.speculative_algorithm == "NOSPEC" or args.skip_nospec:
                    block_sizes = [block_size]
                else:
                    block_sizes = [1, block_size]

                for bs in block_sizes:
                    torch.manual_seed(i)
                    if bs == 1:  # no_spec baseline for measuring speedup
                        response[bs] = dflash_generate(
                            model=target,
                            target=target,
                            input_ids=input_ids,
                            mask_token_id=-1,
                            max_new_tokens=args.max_new_tokens,
                            block_size=bs,
                            stop_token_ids=[tokenizer.eos_token_id],
                            temperature=args.temperature,
                        )
                    else:  # actual speculative algorithm runs here
                        if args.speculative_algorithm == "DFLASH":
                            response[bs] = dflash_generate(
                                model=draft_model,
                                target=target,
                                input_ids=input_ids,
                                mask_token_id=draft_model.mask_token_id,
                                max_new_tokens=args.max_new_tokens,
                                block_size=bs,
                                stop_token_ids=[tokenizer.eos_token_id],
                                temperature=args.temperature,
                                per_token_log=args.per_token_log,
                                tokenizer=tokenizer if args.per_token_log else None,
                                structure_window=args.structure_window,
                            )
                        elif args.speculative_algorithm == "DDTREE":
                            response[bs] = ddtree_generate(
                                model=draft_model,
                                target=target,
                                input_ids=input_ids,
                                mask_token_id=draft_model.mask_token_id,
                                max_new_tokens=args.max_new_tokens,
                                block_size=bs,
                                stop_token_ids=[tokenizer.eos_token_id],
                                temperature=args.temperature,
                                tree_budget=args.ddtree_tree_budget,
                                log_timing=args.profile_gen_stages,
                                per_token_log=args.per_token_log,
                            )
                        elif args.speculative_algorithm == "EAGLE3":
                            response[bs] = eagle3_generate(
                                model=draft_model,
                                target=target,
                                input_ids=input_ids,
                                max_new_tokens=args.max_new_tokens,
                                num_steps=args.eagle_num_steps,
                                top_k=args.eagle_topk,
                                num_draft_tokens=args.eagle_num_draft_tokens,
                                stop_token_ids=[tokenizer.eos_token_id],
                                temperature=args.temperature,
                                per_token_log=args.per_token_log,
                                tokenizer=tokenizer if args.per_token_log else None,
                                structure_window=args.structure_window,
                            )
                        elif args.speculative_algorithm == "WHIFLASH":
                            if args.whiflash_router == "oracle":
                                # For oracle router, we first run per-token for both drafters to get accepted length -> drafter_seq
                                res_ = whiflash_generate(
                                    eagle3_model=eagle3_draft_model,
                                    dflash_model=dflash_draft_model,
                                    target=target,
                                    input_ids=input_ids,
                                    max_new_tokens=args.max_new_tokens,
                                    router=None,
                                    stop_token_ids=[tokenizer.eos_token_id],
                                    temperature=args.temperature,
                                    eagle3_num_steps=args.eagle_num_steps,
                                    eagle3_top_k=args.eagle_topk,
                                    eagle3_num_draft_tokens=args.eagle_num_draft_tokens,
                                    dflash_block_size=block_size,
                                    lazy_catchup=not args.whiflash_no_lazy_catchup,
                                    log_timing=args.whiflash_log_timing,
                                    per_token_log=True,
                                    tokenizer=tokenizer,
                                    structure_window=args.structure_window,
                                    dump_hidden_states=args.dump_hidden_states,
                                )

                                # process drafter sequence per-token into selection of drafter per-round
                                whiflash_router.process_drafter_seq(
                                    dflash_acceptance_lengths=res_.dflash_acceptance_lengths,
                                    eagle3_acceptance_lengths=res_.eagle3_acceptance_lengths,
                                )

                            response[bs] = whiflash_generate(
                                eagle3_model=eagle3_draft_model,
                                dflash_model=dflash_draft_model,
                                target=target,
                                input_ids=input_ids,
                                max_new_tokens=args.max_new_tokens,
                                router=whiflash_router,
                                stop_token_ids=[tokenizer.eos_token_id],
                                temperature=args.temperature,
                                eagle3_num_steps=args.eagle_num_steps,
                                eagle3_top_k=args.eagle_topk,
                                eagle3_num_draft_tokens=args.eagle_num_draft_tokens,
                                dflash_block_size=block_size,
                                ddtree_tree_budget=args.ddtree_tree_budget,
                                drafters=whiflash_drafters,
                                lazy_catchup=not args.whiflash_no_lazy_catchup,
                                log_timing=args.whiflash_log_timing,
                                per_token_log=args.per_token_log if args.whiflash_router != "oracle" else False,
                                tokenizer=tokenizer,
                                structure_window=args.structure_window,
                                dump_hidden_states=args.dump_hidden_states,
                            )

                spec_response = response[block_size]
                generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                messages.append({"role": "assistant", "content": output_text})
                response["question_id"] = instance["question_id"]
                response["seed"] = i
                response["turn_index"] = turn_index
                response["output_text"] = output_text

                # Write target hidden states tensors
                if (
                    args.dump_hidden_states
                    and args.output_dir is not None
                    and hasattr(spec_response, "dflash_target_hidden_states")
                ):
                    target_hidden_states_dir = Path(args.output_dir) / "target_hidden_states"
                    target_hidden_states_dir.mkdir(parents=True, exist_ok=True)
                    target_hidden_states_filename = f"{instance['question_id']}_turn{turn_index}_seed{i}.npz"
                    np.savez_compressed(
                        target_hidden_states_dir / target_hidden_states_filename,
                        dflash=spec_response.dflash_target_hidden_states.float().numpy(),
                        eagle3=spec_response.eagle3_target_hidden_states.float().numpy(),
                    )
                    spec_response.target_hidden_states_path = target_hidden_states_filename
                    delattr(spec_response, "dflash_target_hidden_states")
                    delattr(spec_response, "eagle3_target_hidden_states")

                inst_response.append(response)
        return inst_response

    # Warmup
    NUM_WARMUP = 3
    for idx in tqdm(range(dist.rank(), NUM_WARMUP * dist.size(), dist.size()), disable=not dist.is_main()):
        _run_instance(dataset[idx % len(dataset)], idx)  # wraps for splits smaller than the warmup

    ######################################################
    # Main Measurement Start
    ######################################################
    responses = []
    for idx in tqdm(range(dist.rank(), len(dataset), dist.size()), disable=not dist.is_main()):
        responses.extend(_run_instance(dataset[idx], idx))

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    ######################################################
    # Post Process
    ######################################################
    ttft = np.mean([r[block_size].time_to_first_token for r in responses])
    print(f"[Prefill] Average Time-To-First-Token: {ttft:.4f} s")

    prefill_tps = np.mean([r[block_size].num_input_tokens / r[block_size].time_to_first_token for r in responses])
    print(f"[Prefill] Average Throughput: {prefill_tps:.2f} tokens/s")

    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"[Decoding] Average Time Per Output Token: {tb:.4f} s/token")

    print(f"[Decoding] Average Throughput: {1.0 / tb:.2f} tokens/s")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if not args.skip_nospec and args.speculative_algorithm != "NOSPEC":
        t1 = np.mean([r[1].time_per_output_token for r in responses])
        print(f"Decoding speedup: {t1 / tb:.2f}")

    if args.speculative_algorithm == "WHIFLASH":
        choices = list(chain(*[r[block_size].drafter_choices for r in responses]))
        n = len(choices)
        if n > 0:
            # Transitions are computed over whatever drafters actually appear.
            # Any 2-drafter pairing works (eagle3/dflash, dflash/ddtree, ddtree/eagle3) without a hardcoded table.
            counts = Counter(choices)
            print("Drafter mix: " + " / ".join(f"{name} {counts[name] / n * 100:.1f}%" for name in sorted(counts)))

            transitions: dict[tuple[str, str], int] = defaultdict(int)
            for prev, cur in zip(choices[:-1], choices[1:]):
                transitions[(prev, cur)] += 1
            m = max(sum(transitions.values()), 1)
            if transitions:
                print(
                    "Transitions: "
                    + " / ".join(
                        f"{prev}→{cur} {cnt / m * 100:.1f}%" for (prev, cur), cnt in sorted(transitions.items())
                    )
                )

        all_switch_events = list(
            chain(*[r[block_size].switch_events for r in responses if hasattr(r[block_size], "switch_events")])
        )
        if all_switch_events:
            avg_t = np.mean([e["t_switch_ms"] for e in all_switch_events])
            total_switch_time = sum(e["t_switch_ms"] for e in all_switch_events)
            avg_n = np.mean([e["num_backlog_tokens"] for e in all_switch_events])
            avg_ctx = np.mean([e["ctx_len"] for e in all_switch_events])
            print(
                f"Switch events: {len(all_switch_events)} switches into EAGLE-3 (backlog catch-up), "
                f"avg t_switch={avg_t:.2f}ms, total={total_switch_time:.1f}ms, "
                f"avg backlog={avg_n:.1f}, avg ctx={avg_ctx:.0f}"
            )

    # Write JSONL results
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_dir / "transformers.jsonl"

        samples: dict = defaultdict(list)
        for resp_idx, resp in enumerate(responses):
            spec_response = resp[block_size]

            tps = 1.0 / spec_response.time_per_output_token if spec_response.time_per_output_token > 0 else 0.0
            total_latency = (
                spec_response.time_to_first_token
                + spec_response.time_per_output_token * spec_response.num_output_tokens
            )
            avg_accept = float(np.mean(spec_response.acceptance_lengths)) if spec_response.acceptance_lengths else 0.0
            samples[resp["question_id"]].append(
                {
                    "seed": resp["seed"],
                    "turn_index": resp["turn_index"],
                    "num_input_tokens": spec_response.num_input_tokens,
                    "num_output_tokens": spec_response.num_output_tokens,
                    "time_to_first_token": round(spec_response.time_to_first_token, 6),
                    "time_per_output_token": round(spec_response.time_per_output_token, 6),
                    "tps": round(tps, 6),
                    "avg_accept_length": round(avg_accept, 6),
                    "total_latency": round(total_latency, 6),
                    "output_text": resp["output_text"],
                    "accept_lengths": spec_response.acceptance_lengths,
                    **(
                        {"entropies": [round(n, 6) for n in spec_response.entropies]}
                        if hasattr(spec_response, "entropies") and spec_response.entropies
                        else {}
                    ),
                    **(
                        {"hidden_norms": [round(n, 6) for n in spec_response.hidden_norms]}
                        if hasattr(spec_response, "hidden_norms") and spec_response.hidden_norms
                        else {}
                    ),
                    **(
                        {
                            "dflash_target_hidden_norms": [
                                [round(n, 6) for n in step] for step in spec_response.dflash_target_hidden_norms
                            ]
                        }
                        if hasattr(spec_response, "dflash_target_hidden_norms")
                        and spec_response.dflash_target_hidden_norms
                        else {}
                    ),
                    **(
                        {
                            "eagle3_target_hidden_norms": [
                                [round(n, 6) for n in step] for step in spec_response.eagle3_target_hidden_norms
                            ]
                        }
                        if hasattr(spec_response, "eagle3_target_hidden_norms")
                        and spec_response.eagle3_target_hidden_norms
                        else {}
                    ),
                    **(
                        {"target_hidden_states_path": spec_response.target_hidden_states_path}
                        if hasattr(spec_response, "target_hidden_states_path")
                        else {}
                    ),
                    **(
                        {"structure_scores": [round(s, 6) for s in spec_response.structure_scores]}
                        if hasattr(spec_response, "structure_scores") and spec_response.structure_scores
                        else {}
                    ),
                    **(
                        {"eagle3_acceptance_lengths": spec_response.eagle3_acceptance_lengths}
                        if hasattr(spec_response, "eagle3_acceptance_lengths")
                        and spec_response.eagle3_acceptance_lengths
                        else {}
                    ),
                    **(
                        {"dflash_acceptance_lengths": spec_response.dflash_acceptance_lengths}
                        if hasattr(spec_response, "dflash_acceptance_lengths")
                        and spec_response.dflash_acceptance_lengths
                        else {}
                    ),
                    **(
                        {"drafter_choices": spec_response.drafter_choices}
                        if hasattr(spec_response, "drafter_choices")
                        else {}
                    ),
                    **(
                        {"switch_events": spec_response.switch_events}
                        if hasattr(spec_response, "switch_events") and spec_response.switch_events
                        else {}
                    ),
                    # DDTREE per-stage timing breakdown — only present when run with
                    # --profile-gen-stages (otherwise stage_times is None and is omitted).
                    **(
                        {"stage_times": {k: round(v, 6) for k, v in spec_response.stage_times.items()}}
                        if getattr(spec_response, "stage_times", None)
                        else {}
                    ),
                    **(
                        {"decode_rounds": spec_response.decode_rounds}
                        if getattr(spec_response, "decode_rounds", None)
                        else {}
                    ),
                }
            )

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for question_id in sorted(samples.keys()):
                turns = sorted(samples[question_id], key=lambda t: t["turn_index"])
                record = {
                    "question_id": question_id,
                    "dataset": args.dataset,
                    "target_model": args.model_name_or_path,
                    "drafter": args.draft_name_or_path,
                    "backend": "transformers",
                    "turns": turns,
                }
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(responses)} records to {jsonl_path}")


if __name__ == "__main__":
    main()
