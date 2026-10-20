import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset

from .agentic_prompts import (
    DB_SYSTEM,
    KG_INSTRUCTIONS,
    KG_USER_FMT,
    OS_SYSTEM,
    OS_USER_FMT,
    TAU_AGENT_INSTRUCTION,
    TAU_SYSTEM_TEMPLATE,
    db_user_msg,
)
from .preprocessing import deduplicate

############################################
# Misc utilities
############################################


def load_system_prompt(name: str) -> str:
    """Load a system prompt by name from configs/system_prompts.yaml."""
    prompts_path = Path(__file__).parent.parent / "configs" / "system_prompts.yaml"
    with open(prompts_path) as f:
        prompts = yaml.safe_load(f)
    if name not in prompts:
        raise ValueError(f"Unknown system prompt '{name}'. Available: {list(prompts)}")
    return prompts[name]


############################################
# Latency profile utilities
############################################


def load_latency_profile(path: str, target_model: str) -> dict[str, Any]:
    """Load a latency profile from json file given target model name."""
    with open(path) as f:
        data = json.load(f)

    # Multi-model format: pick the profile matching target_model
    profiles = data.get("profiles", [])
    for p in profiles:
        if p.get("target_model") == target_model:
            return p

    available = [p.get("target_model", "<unknown>") for p in profiles]
    raise ValueError(
        f"No latency profile found for target_model={target_model!r} in {path}. " f"Available profiles: {available}"
    )


############################################
# Dataset utilities
############################################


def calibration_split(num_samples: int, calibration_samples: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Split sample positions 0..num_samples-1 into (calibration, evaluation) position lists."""
    if not 0 < calibration_samples < num_samples:
        raise ValueError(f"calibration_samples must be in (0, {num_samples}), got {calibration_samples}.")
    chosen = set(np.random.default_rng(seed).permutation(num_samples)[:calibration_samples].tolist())
    calibration = [i for i in range(num_samples) if i in chosen]
    return calibration, [i for i in range(num_samples) if i not in chosen]


def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    # Chat datasets
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(
            lambda x: {
                "formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x["input"] else x["instruction"])
            }
        )
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name in ["mt-bench", "mt_bench"]:
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "ultrachat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft+train_gen")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "ultrachat-test":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft+test_gen")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "wildchat":
        # WildChat (allenai/WildChat). Single train split.
        # https://huggingface.co/datasets/allenai/WildChat
        # Filters (each defends a specific noise pattern observed in samples):
        #   - language == "English" / toxic == False / first turn is user
        #   - 30 <= len(prompt) <= 1500 chars:
        #         lower bound removes trivial very short prompts;
        #         upper bound removes the long-tail noise:
        #         prompt-generator templates (3-5k chars, often
        #         near-duplicated), raw HTML/data dumps.
        dataset = load_dataset("allenai/WildChat", split="train")
        dataset = dataset.filter(
            lambda x: x["language"] == "English"
            and not x["toxic"]
            and len(x["conversation"]) > 0
            and x["conversation"][0]["role"] == "user"
            and 30 <= len(x["conversation"][0]["content"]) <= 1500
        )
        dataset = dataset.map(lambda x: {"turns": [x["conversation"][0]["content"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "dolci":
        dataset = load_dolci()["train"]
        dataset = dataset.map(lambda x: {"turns": [x["messages"][0]["content"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "dolci-test":
        dataset = load_dolci()["test"]
        dataset = dataset.map(lambda x: {"turns": [x["messages"][0]["content"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]

        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"

        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]}, remove_columns=dataset.column_names, features=target_features
        )
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    # Agentic datasets
    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "tau-bench":
        # tau-bench (sierra-research/tau2-bench): airline (49) + retail (113), both after
        # deduplication on (domain, reason_for_call) = 162.
        # https://github.com/sierra-research/tau2-bench
        # Prompt construction (single user turn):
        #   TAU_SYSTEM_TEMPLATE(agent_instruction, policy.md[domain])
        #   + "\n\n"
        #   + reason_for_call [+ " " + known_info]
        base = Path(__file__).parent.parent / "data_files" / "agentic" / "tau_bench"
        policies = {d: (base / f"{d}_policy.md").read_text() for d in ["airline", "retail"]}
        records = []
        seen = set()
        for domain in ["airline", "retail"]:
            system_msg = TAU_SYSTEM_TEMPLATE.format(
                agent_instruction=TAU_AGENT_INSTRUCTION,
                domain_policy=policies[domain],
            )
            with open(base / f"{domain}_tasks.json") as f:
                for t in json.load(f):
                    instr = t["user_scenario"]["instructions"]
                    key = (domain, instr["reason_for_call"])
                    if key in seen:
                        continue
                    seen.add(key)
                    user_parts = [instr["reason_for_call"]]
                    if instr.get("known_info"):
                        user_parts.append(instr["known_info"])
                    user_msg = " ".join(user_parts)
                    records.append({"turns": [system_msg + "\n\n" + user_msg]})
        dataset = Dataset.from_list(records)
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "agentbench":
        # AgentBench (THUDM/AgentBench): os_interaction (23, after deduplication)
        # + knowledgegraph (20) + dbbench (60) = 103 tasks.
        # https://github.com/THUDM/AgentBench
        # Each per-task prompt is system_message + "\n\n" + user_message,
        # concatenated into a single user turn (same pattern as tau-bench).
        base = Path(__file__).parent.parent / "data_files" / "agentic" / "agentbench"
        records = []

        # OS: per task.py L530 the user message is
        #   "Now, I will start a new problem in a new OS. My problem is:\n\n{description}"
        with open(base / "os_dev.json") as f:
            seen_os = set()
            for t in json.load(f):
                desc = t["description"]
                if desc in seen_os:
                    continue
                seen_os.add(desc)
                prompt = OS_SYSTEM + "\n\n" + OS_USER_FMT.format(description=desc)
                records.append({"turns": [prompt]})

        # KG: max_round=15 matches the kg-std default in AgentBench's
        # configs/tasks/kg.yaml (upstream, not in this repo).
        kg_system = KG_INSTRUCTIONS.format(max_round=15)
        with open(base / "kg_dev.json") as f:
            for t in json.load(f):
                entities = ", ".join(t.get("entities", {}).keys())
                prompt = kg_system + "\n\n" + KG_USER_FMT.format(question=t["question"], entities=entities)
                records.append({"turns": [prompt]})

        # DB: user message construction is delegated to db_user_msg() which
        # mirrors AgentBench's src/server/tasks/dbbench/task.py L108–L114
        with open(base / "db_dev.jsonl") as f:
            for line in f:
                t = json.loads(line)
                prompt = DB_SYSTEM + "\n\n" + db_user_msg(t)
                records.append({"turns": [prompt]})

        dataset = Dataset.from_list(records)
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)

    elif data_name == "agent-instruct":
        # AgentInstruct (THUDM/AgentInstruct): 4 splits × 20 = 80 tasks.
        # https://huggingface.co/datasets/THUDM/AgentInstruct
        #   db        — turn 0 = agent instruction, turn 2 = real task.
        #   alfworld  — same: turn 0 = agent instruction, turn 2 = task.
        #   webshop   — same: turn 0 = agent instruction, turn 2 = task.
        #   mind2web  — turn 0 contains agent instruction + task already
        #               (only 2 turns total).
        splits_instr_then_task = ["db", "alfworld", "webshop"]
        splits_combined_turn0 = ["mind2web"]
        records = []

        for split in splits_instr_then_task:
            ds = load_dataset("THUDM/AgentInstruct", split=split)
            seen: set = set()
            for ex in ds:
                if len(seen) >= 20:
                    break
                convs = ex["conversations"]
                if not (len(convs) >= 3 and convs[0]["from"] == "human" and convs[2]["from"] == "human"):
                    continue
                task_text = convs[2]["value"]
                if task_text in seen:
                    continue
                seen.add(task_text)
                records.append({"turns": [convs[0]["value"] + "\n" + task_text]})

        for split in splits_combined_turn0:
            ds = load_dataset("THUDM/AgentInstruct", split=split)
            seen = set()
            for ex in ds:
                if len(seen) >= 20:
                    break
                convs = ex["conversations"]
                if not (convs and convs[0]["from"] == "human"):
                    continue
                task_text = convs[0]["value"]
                if task_text in seen:
                    continue
                seen.add(task_text)
                records.append({"turns": [task_text]})

        dataset = Dataset.from_list(records)
        dataset = dataset.map(lambda x, idx: {**x, "question_id": idx}, with_indices=True)
    else:
        raise ValueError(f"Unknown dataset {data_name!r}.")

    return dataset


def load_dolci(test_size: float = 0.1, seed: int = 42, max_length: int = 4096) -> DatasetDict:
    dataset = load_dataset("allenai/Dolci-Instruct-SFT-No-Tools", split="train")

    def get_prompt(example: dict[str, Any]) -> str:
        return example["messages"][0]["content"]

    dataset = dataset.filter(lambda example: len(get_prompt(example)) <= max_length)
    dataset = deduplicate(dataset, key_extractor=get_prompt)

    return dataset.train_test_split(test_size=test_size, seed=seed)
