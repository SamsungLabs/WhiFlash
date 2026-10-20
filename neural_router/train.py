import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import EvalPrediction, HfArgumentParser, Trainer, TrainingArguments, set_seed

from .modeling import RouterModel


@dataclass
class Arguments:
    train_dir: str  # dataset dir (generate_training_data.py output) holding the train shards
    test_dir: str | None = None  # optional held-out dataset dir for a final test-set evaluation
    num_train_examples: int = 8_388_608  # number of per-token rows to load for training
    num_eval_examples: int = 8192  # per-token rows held out from train_dir for in-loop eval
    num_test_examples: int = 131_072  # per-token rows from test_dir for the final evaluation
    hidden_size_ratio: float = 0.125  # router MLP hidden size = ratio * input (target hidden) size
    dropout: float = 0.01
    # A router predicts `AL[label_pos_col] - AL[label_neg_col]` per token. Acceptance
    # lengths are named by drafter identity; columns may be spread across `train_dir`
    # plus any `extra_train_dirs`, merged by matching shard filename (the per-token
    # traces all follow the same target-greedy sequence, so they align). This is fully
    # drafter-agnostic — to route against a new drafter, generate its column and name it
    # here. Defaults reproduce the EAGLE3-vs-DFlash router. Examples:
    #   eagle vs ddtree@128: pos=eagle_acceptance_lengths,  neg=ddtree_b128_acceptance_lengths
    #   dflash vs ddtree@128: pos=ddtree_b128_acceptance_lengths, neg=dflash_acceptance_lengths
    label_pos_col: str = "eagle_acceptance_lengths"
    label_neg_col: str = "dflash_acceptance_lengths"
    extra_train_dirs: list[str] = field(default_factory=list)
    extra_test_dirs: list[str] = field(default_factory=list)


@dataclass
class HFTrainingArguments(TrainingArguments):
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 8192
    per_device_eval_batch_size: int = 8192
    optim: str = "adamw_torch_fused"
    learning_rate: float = 1e-3
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    eval_strategy: str = "steps"
    eval_steps: float | None = 10
    save_strategy: str = "no"
    save_total_limit: int | None = 1
    logging_steps: float = 10
    logging_first_step: bool = True
    bf16: bool = True
    label_names: list[str] | None = field(default_factory=lambda: ["labels"])
    report_to: None | str | list[str] = "tensorboard"


class RouterDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        paths: Iterator[Path],
        num_examples: int,
        label_pos_col: str,
        label_neg_col: str,
        extra_dirs: list[str] | None = None,
    ) -> None:
        self._num_examples = num_examples
        self._label_pos_col = label_pos_col
        self._label_neg_col = label_neg_col
        self._data = self._load_data(paths, num_examples, label_pos_col, label_neg_col, extra_dirs or [])

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "final_hidden_states": self._data["final_hidden_states"][i],
            "labels": self._data[self._label_pos_col][i].float() - self._data[self._label_neg_col][i].float(),
        }

    @staticmethod
    def _load_data(
        paths: Iterator[Path],
        num_examples: int,
        label_pos_col: str,
        label_neg_col: str,
        extra_dirs: list[str],
    ) -> dict[str, torch.Tensor]:
        # Only the hidden state and the two label columns are needed. A column may live
        # in the primary shard or in a same-named shard under any extra trace dir (e.g.
        # a newly-added drafter), merged here by filename.
        keep = {"final_hidden_states", label_pos_col, label_neg_col}
        extra_paths = [Path(d) for d in extra_dirs]

        data: dict[str, torch.Tensor] = {}
        progress = tqdm(total=num_examples, desc="Loading")
        i = 0

        for path in paths:
            if i >= num_examples:
                break

            shard = load_file(path)
            for extra in extra_paths:
                if keep <= shard.keys():
                    break
                shard.update(load_file(extra / path.name))

            missing = keep - shard.keys()
            if missing:
                raise KeyError(f"{path}: missing columns {sorted(missing)} (searched extra_dirs={extra_dirs})")

            shard = {k: shard[k] for k in keep}
            # Columns may have been produced by separate runs (e.g. the primary shard
            # truncated to min(dflash, eagle) while a drafter's trace ran to full
            # length); all follow the identical target-greedy sequence, so truncating to
            # the common length keeps them aligned.
            shard_size = min(len(v) for v in shard.values())

            if not data:
                data = {k: torch.empty(num_examples, *v.shape[1:], dtype=v.dtype) for k, v in shard.items()}

            num_next_examples = min(shard_size, num_examples - i)
            for k, v in shard.items():
                data[k][i : i + num_next_examples] = v[:num_next_examples]

            i += num_next_examples
            progress.update(num_next_examples)

        progress.close()

        if i < num_examples:
            raise ValueError(f"Insufficient data for {num_examples:,} examples")

        return data

    @classmethod
    def from_path(
        cls,
        path: str,
        num_examples: int,
        label_pos_col: str,
        label_neg_col: str,
        extra_dirs: list[str] | None = None,
    ) -> "RouterDataset":
        return cls(
            iter(cls._shard_paths(path)),
            num_examples=num_examples,
            label_pos_col=label_pos_col,
            label_neg_col=label_neg_col,
            extra_dirs=extra_dirs,
        )

    @classmethod
    def train_eval_split(
        cls,
        path: str,
        num_eval_examples: int,
        num_train_examples: int,
        seed: int,
        label_pos_col: str,
        label_neg_col: str,
        extra_dirs: list[str] | None = None,
    ) -> tuple["RouterDataset", "RouterDataset"]:
        paths = cls._shard_paths(path)
        random.Random(seed).shuffle(paths)
        iterator = iter(paths)

        # Shared iterator to ensure no overlap between train/eval.
        common = dict(label_pos_col=label_pos_col, label_neg_col=label_neg_col, extra_dirs=extra_dirs)
        eval_dataset = cls(iterator, num_examples=num_eval_examples, **common)
        train_dataset = cls(iterator, num_examples=num_train_examples, **common)
        return train_dataset, eval_dataset

    @staticmethod
    def _shard_paths(path: str) -> list[Path]:
        return sorted(Path(path).glob("*.safetensors"))


def compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    predictions = (logits > 0).astype(np.float32)
    binary_labels = (labels > 0).astype(np.float32)
    return (predictions == binary_labels).mean().item()


def compute_regret(logits: np.ndarray, labels: np.ndarray) -> float:
    return (
        np.where(
            logits > 0,
            np.maximum(-labels, 0),
            np.maximum(labels, 0),
        )
        .mean()
        .item()
    )


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    if not isinstance(logits, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("Expected `np.ndarray`")

    return {
        "accuracy": compute_accuracy(logits, labels),
        "regret": compute_regret(logits, labels),
    }


def main(args: Arguments, hf_args: HFTrainingArguments) -> None:
    set_seed(hf_args.seed)

    train_dataset, eval_dataset = RouterDataset.train_eval_split(
        args.train_dir,
        num_eval_examples=args.num_eval_examples,
        num_train_examples=args.num_train_examples,
        seed=hf_args.seed,
        label_pos_col=args.label_pos_col,
        label_neg_col=args.label_neg_col,
        extra_dirs=args.extra_train_dirs,
    )

    test_dataset = (
        RouterDataset.from_path(
            args.test_dir,
            num_examples=args.num_test_examples,
            label_pos_col=args.label_pos_col,
            label_neg_col=args.label_neg_col,
            extra_dirs=args.extra_test_dirs,
        )
        if args.test_dir is not None
        else None
    )

    input_size = train_dataset[0]["final_hidden_states"].shape[-1]
    hidden_size = int(input_size * args.hidden_size_ratio)
    logger.info(f"Router input_size: {input_size}, hidden_size: {hidden_size}")

    model = RouterModel(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=args.dropout,
    )
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if test_dataset:
        trainer.evaluate(test_dataset, metric_key_prefix="test")

    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((Arguments, HFTrainingArguments))  # type: ignore
    args, hf_args = parser.parse_args_into_dataclasses()

    main(args, hf_args)
