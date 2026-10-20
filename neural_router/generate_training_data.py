import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from loguru import logger
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, PreTrainedTokenizerBase, set_seed

from data import load_and_process_dataset
from generation.ddtree_generate import ddtree_generate
from generation.dflash_generate import dflash_generate
from generation.eagle3_generate import eagle3_generate
from model import DFlashDraftModel, Eagle3DraftModel


@dataclass
class Arguments:
    target_model_path: str  # HF path/name of the target model being accelerated (also provides the tokenizer)
    dataset: str  # dataset key for load_and_process_dataset (e.g. "dolci", "dolci-test" for the test split)
    output_dir: str  # where per-example {example_id}.safetensors shards are written
    drafters: list[str] = field(default_factory=lambda: ["eagle3", "dflash"])
    dflash_model_path: str | None = None  # required if any drafter uses the DFlash draft model
    eagle_model_path: str | None = None  # required if 'eagle3' is in --drafters
    hidden_from: str = "eagle3"  # whose final_hidden_states to store as the router input
    tree_budgets: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])  # ddtree only
    # Incremental mode: when set to an existing dataset dir (which stores output_ids), ADD the
    # --drafters to it instead of building from scratch — each drafter is gated against the stored
    # output_ids (exact), examples without a baseline shard are skipped, and only the new drafters'
    # acceptance-length columns are written (hidden/output_ids already live in the base). Stitched
    # at train time via train.py --extra-train-dirs. Unset = build a self-contained dataset.
    baseline_dir: str | None = None
    eagle_num_steps: int = 8
    eagle_topk: int = 10
    eagle_num_draft_tokens: int = 60
    dtype: str = "float32"
    num_examples: int | None = None  # cap on dataset examples to process (None = all)
    max_new_tokens: int = 1024
    temperature: float = 0.0
    enable_thinking: bool = False
    seed: int = 42


class ResponseMismatchException(Exception):
    pass


def _dflash(target, drafts, input_ids, eos, args):
    model = drafts["dflash"]
    r = dflash_generate(
        model=model,
        target=target,
        input_ids=input_ids,
        mask_token_id=model.mask_token_id,
        max_new_tokens=args.max_new_tokens,
        block_size=model.block_size,
        stop_token_ids=[eos],
        temperature=args.temperature,
        per_token_log=True,
    )
    return {"dflash_acceptance_lengths": r.acceptance_lengths}, r.output_ids, r.final_hidden_states


def _eagle(target, drafts, input_ids, eos, args):
    r = eagle3_generate(
        model=drafts["eagle3"],
        target=target,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        num_steps=args.eagle_num_steps,
        top_k=args.eagle_topk,
        num_draft_tokens=args.eagle_num_draft_tokens,
        stop_token_ids=[eos],
        temperature=args.temperature,
        per_token_log=True,
    )
    return {"eagle_acceptance_lengths": r.acceptance_lengths}, r.output_ids, r.final_hidden_states


def _ddtree(target, drafts, input_ids, eos, args):
    model = drafts["dflash"]  # DDTree reuses the DFlash draft model; only verification differs.
    r = ddtree_generate(
        model=model,
        target=target,
        input_ids=input_ids,
        mask_token_id=model.mask_token_id,
        max_new_tokens=args.max_new_tokens,
        block_size=model.block_size,
        stop_token_ids=[eos],
        temperature=args.temperature,
        per_token_log=True,
        per_token_log_budgets=args.tree_budgets,  # every tree size from one max-budget verify
    )
    cols = {f"ddtree_b{b}_acceptance_lengths": al for b, al in r.acceptance_lengths_by_budget.items()}
    return cols, r.output_ids, r.final_hidden_states


@dataclass
class DrafterSpec:
    draft_model: str  # which draft model to load: "dflash" or "eagle3"
    # runs the drafter (oracle) for one example -> (columns, output_ids, final_hidden_states)
    run: Callable[..., tuple[dict[str, list[int]], torch.Tensor, list]]


DRAFTERS: dict[str, DrafterSpec] = {
    "eagle3": DrafterSpec(draft_model="eagle3", run=_eagle),
    "dflash": DrafterSpec(draft_model="dflash", run=_dflash),
    "ddtree": DrafterSpec(draft_model="dflash", run=_ddtree),
    # A new drafter is one more entry here (draft model + its per-token-log run adapter).
}


def _generate_training_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    target_model: AutoModelForCausalLM,
    drafts: dict[str, Any],
    drafters: list[str],
    hidden_src: str,
    args: Arguments,
    baseline_output_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["turns"][0]}],
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    ).to(target_model.device)
    eos = tokenizer.eos_token_id

    columns: dict[str, list[int]] = {}
    output_ids_by_drafter: dict[str, torch.Tensor] = {}
    hidden: list | None = None
    for name in drafters:
        cols, output_ids, final_hidden = DRAFTERS[name].run(target_model, drafts, input_ids, eos, args)
        columns.update(cols)
        output_ids_by_drafter[name] = output_ids
        if name == hidden_src:
            hidden = final_hidden

    al_columns: dict[str, torch.Tensor] = {k: torch.tensor(v, dtype=torch.uint8) for k, v in columns.items()}

    if baseline_output_ids is not None:
        # Incremental: each drafter must reproduce the EXACT stored target-greedy tokens, else its
        # columns would not align with the base — drop the example. Write only the new AL columns
        # (hidden / output_ids already live in the base dataset).
        for name in drafters:
            if not torch.equal(
                output_ids_by_drafter[name][0].to(dtype=baseline_output_ids.dtype).cpu(), baseline_output_ids
            ):
                raise ResponseMismatchException
        min_length = min(len(v) for v in al_columns.values())
        return {k: v[:min_length] for k, v in al_columns.items()}

    # Unified (from scratch): every drafter must produce the same sequence as the first — drop the
    # example otherwise. Store all AL columns + the router-input hidden + the token sequence.
    reference = output_ids_by_drafter[drafters[0]]
    for name in drafters[1:]:
        if not torch.equal(output_ids_by_drafter[name], reference):
            raise ResponseMismatchException

    result = dict(al_columns)
    result["final_hidden_states"] = torch.cat(hidden).to(dtype=torch.bfloat16)
    min_length = min(len(v) for v in result.values())
    out = {k: v[:min_length] for k, v in result.items()}
    # Store the target-greedy token sequence so a new drafter can be added later by re-checking
    # exact token identity against it — no need to re-run these drafters.
    out["output_ids"] = reference[0].to(dtype=torch.int32)
    return out


def _load_drafts(drafters: list[str], args: Arguments, target_model, device) -> dict[str, Any]:
    kinds = {DRAFTERS[name].draft_model for name in drafters}
    drafts: dict[str, Any] = {}
    if "dflash" in kinds:
        if args.dflash_model_path is None:
            raise ValueError("--dflash-model-path is required for drafters that use the DFlash draft model.")
        drafts["dflash"] = DFlashDraftModel.from_pretrained(
            args.dflash_model_path, dtype=target_model.dtype, device_map=device
        )
    if "eagle3" in kinds:
        if args.eagle_model_path is None:
            raise ValueError("--eagle-model-path is required when 'eagle3' is in --drafters.")
        eagle_model = Eagle3DraftModel.from_pretrained(
            args.eagle_model_path, dtype=target_model.dtype, device_map=device
        )
        eagle_model.set_target_model(target_model)
        drafts["eagle3"] = eagle_model
    return drafts


@torch.inference_mode()
def generate_training_data(args: Arguments) -> None:
    unknown = [d for d in args.drafters if d not in DRAFTERS]
    if unknown:
        raise ValueError(f"Unknown drafter(s) {unknown}. Registered: {sorted(DRAFTERS)}")
    incremental = args.baseline_dir is not None
    # From scratch needs >=2 drafters to form a pair; incremental can add a single new drafter
    # (it pairs against columns already in the baseline dataset).
    if not incremental and len(args.drafters) < 2:
        raise ValueError("--drafters needs at least two drafters to form a router pair (or set --baseline-dir).")
    hidden_src = args.hidden_from if args.hidden_from in args.drafters else args.drafters[0]

    set_seed(args.seed)

    num_local = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank_in_launch = int(os.environ.get("RANK", "0"))
    # Multi-node: launch per node with NODE_RANK/NUM_NODES so each covers a disjoint slice.
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    num_nodes = int(os.environ.get("NUM_NODES", "1"))
    world_size = num_local * num_nodes
    global_rank = node_rank * num_local + rank_in_launch

    if num_local > 1:
        dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model_path, dtype=args.dtype, device_map=device)
    drafts = _load_drafts(args.drafters, args, target_model, device)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    dataset = load_and_process_dataset(args.dataset)

    total_examples = len(dataset) if args.num_examples is None else args.num_examples
    indices = list(range(global_rank, total_examples, world_size))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_dir) if incremental else None

    if global_rank == 0:
        mode = f"incremental (add to baseline={baseline_dir.name})" if incremental else "from scratch"
        logger.info(
            f"generate_training_data [{mode}] | target={Path(args.target_model_path).name} "
            f"dataset={args.dataset} dtype={args.dtype} | drafters={args.drafters} hidden_from={hidden_src}"
        )
        logger.info(
            f"  GPUs: {world_size} ({num_nodes} node(s) x {num_local}) | total_examples={total_examples} | "
            f"output_dir={output_dir}"
        )

    num_saved = 0
    for i in tqdm(indices, desc=f"rank {global_rank}/{world_size}", total=len(indices), disable=rank_in_launch != 0):
        example = dataset[i]
        example_id = example["question_id"]

        output_path = output_dir / f"{example_id}.safetensors"
        if output_path.exists():
            continue

        baseline_output_ids = None
        if baseline_dir is not None:
            baseline_path = baseline_dir / f"{example_id}.safetensors"
            if not baseline_path.exists():
                continue  # only add the drafter where the base dataset has the example
            baseline_output_ids = load_file(baseline_path).get("output_ids")
            if baseline_output_ids is None:
                raise KeyError(
                    f"{baseline_path} has no 'output_ids' — the base dataset must be built by the current "
                    "generate_training_data.py (which stores it) before adding a drafter."
                )

        try:
            training_example = _generate_training_example(
                example, tokenizer, target_model, drafts, args.drafters, hidden_src, args, baseline_output_ids
            )
        except ResponseMismatchException:
            logger.info(f"Response mismatch across drafters. Skipping example: {example_id}")
            continue

        save_file(training_example, output_path)
        num_saved += 1

    logger.info(f"rank {global_rank}: saved {num_saved} examples")

    if num_local > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)  # type: ignore[arg-type]
    (args,) = parser.parse_args_into_dataclasses()
    generate_training_data(args)
