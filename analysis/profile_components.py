"""Component-level latency profiling for WhiFlash.

Measures the absolute cost of:
  - ``_eagle3_round``  — one EAGLE3 speculation round (steady state)
  - ``_dflash_round``  — one DFlash speculation round, swept over block_size
  - ``_flush_eagle3_backlog`` — EAGLE3 catch-up cost, swept over backlog size N
  - (optional) target prefill — swept over input length

Sweeps each component over ``--base-kv-lens`` so the kv_len-axis behaviour of
each cost shows up directly. Long-context support uses forward-hook-based
hidden-state capture (default on) so prefill memory scales with the few layers
EAGLE3+DFlash actually consume rather than the full 33-layer transformer.

Distinct from the end-to-end ``benchmark*.py`` scripts: those measure
user-visible TTFT/TPOT on real workloads; this profiler isolates per-component
GPU cost as input to the routing/utility analysis and to the paper's
switching-cost story. Argument names mirror ``benchmark.py`` for consistency.

Usage (from the repo root, or via scripts/release/profile/profile_components.sh):

    uv run analysis/profile_components.py \\
        --model-name-or-path meta-llama/Llama-3.1-8B-Instruct \\
        --whiflash-eagle-draft lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \\
        --draft-name-or-path z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat \\
        --base-kv-lens 512 1024 2048 4096 8192 16384 32768 \\
        --backlog-sizes 1 16 32 64 128 256 512 1024 \\
        --measure-e2d-switch --ablate-logits-to-keep --profile-prefill \\
        --output-dir results/components/llama3.1-8b
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from data import load_warmup_corpus_tokens, slice_warmup_prompt
from generation.eagle3_generate import topk_tree_draft
from generation.whiflash_generate import _dflash_round, _eagle3_round, _flush_eagle3_backlog, _GenState
from model import (
    DFlashDraftModel,
    Eagle3DraftModel,
    extract_context_feature,
    has_flash_attn,
    sample,
    set_attn_implementation,
)


@torch.inference_mode()
def build_prefill_state(
    target: AutoModelForCausalLM,
    eagle3_model: Eagle3DraftModel,
    dflash_model: DFlashDraftModel,
    tokenizer: AutoTokenizer,
    prompt: str | None,
    *,
    eagle3_num_steps: int,
    eagle3_top_k: int,
    eagle3_num_draft_tokens: int,
    max_block_size: int,
    max_new_tokens: int = 256,
    device: torch.device,
    input_ids: torch.Tensor | None = None,
    use_hidden_state_hooks: bool = True,
) -> _GenState:
    """Run a real prefill and assemble a ``_GenState`` matching ``whiflash_generate``.

    Pass ``prompt`` (string) for tokenize-then-prefill, or ``input_ids``
    (1-D or 2-D LongTensor) to skip the tokenize step. The latter is used by
    the corpus-warmup path which pre-tokenizes once and slices per kv_len.

    ``use_hidden_state_hooks`` (default True) replaces the standard
    ``output_hidden_states=True`` path with forward hooks on the specific
    layers EAGLE3 + DFlash actually consume. Saves ~26GB at N=128K — required
    to fit long contexts on 40GB GPUs. Set False for the bench/baseline path.
    """
    if input_ids is None:
        if prompt is None:
            raise ValueError("must pass either prompt or input_ids")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
    num_input_tokens = input_ids.shape[1]
    buf_len = num_input_tokens + max_new_tokens + max(eagle3_num_draft_tokens, max_block_size) + 1
    mask_token_id = dflash_model.mask_token_id

    output_ids = torch.full((1, buf_len), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, :num_input_tokens] = input_ids
    global_pos_ids = torch.arange(buf_len, device=device).unsqueeze(0)

    if use_hidden_state_hooks:
        # Capture only the layers EAGLE3 + DFlash + the last layer (needed for
        # current_hidden_norm via hidden_states_proxy[-1]). Drops the full-tuple
        # memory cost of output_hidden_states=True at long N.
        last_layer_idx = target.config.num_hidden_layers - 1
        needed_layer_ids = list(eagle3_model.target_layer_ids) + list(dflash_model.target_layer_ids) + [last_layer_idx]
        with capture_target_hiddens(target, needed_layer_ids) as captured:
            target_output = target(input_ids, use_cache=True)
        hidden_states_proxy = captured
    else:
        target_output = target(input_ids, use_cache=True, output_hidden_states=True)
        hidden_states_proxy = target_output.hidden_states

    target_kv = target_output.past_key_values
    first_token = sample(target_output.logits[:, -1:, :], 0.0)
    output_ids[:, num_input_tokens] = first_token.squeeze()

    aux_hidden_prefill = eagle3_model.extract_hidden_states(hidden_states_proxy)
    dflash_ctx_hidden_prefill = extract_context_feature(
        hidden_states_proxy,
        dflash_model.target_layer_ids,
    )
    eagle3_prefill_tokens = torch.cat([input_ids[:, 1:], first_token], dim=1)

    return _GenState(
        eagle3_model=eagle3_model,
        dflash_model=dflash_model,
        target=target,
        output_ids=output_ids,
        global_pos_ids=global_pos_ids,
        device=device,
        target_kv=target_kv,
        eagle3_stable_kv=DynamicCache(),
        eagle3_token=first_token,
        eagle3_root_logits=None,
        eagle3_root_hidden=None,
        dflash_kv=DynamicCache(),
        dflash_target_hidden=dflash_ctx_hidden_prefill,
        eagle3_backlog_tokens=[eagle3_prefill_tokens],
        eagle3_backlog_hiddens=[aux_hidden_prefill],
        eagle3_backlog_start_pos=0,
        pos=num_input_tokens + 1,
        num_input_tokens=num_input_tokens,
        eagle3_num_steps=eagle3_num_steps,
        eagle3_top_k=eagle3_top_k,
        eagle3_num_draft_tokens=eagle3_num_draft_tokens,
        dflash_block_size=max_block_size,
        temperature=0.0,
        current_logits=target_output.logits[0, -1, :],
        current_hidden_states=hidden_states_proxy[-1][0, -1:, :],
    )


@torch.inference_mode()
def warm_eagle3(gen_state: _GenState, target_pos: int, min_rounds: int = 3) -> int:
    """Run real EAGLE3 rounds until ``gen_state.pos >= target_pos`` and at least
    ``min_rounds`` rounds have executed. Returns the number of rounds run.

    The flush-after-each-round pattern matches the production decode loop in
    ``whiflash_generate`` (which always flushes before the next ``_eagle3_round``).
    Without it, ``eagle3_stable_kv`` would stay at ``num_input_tokens`` while
    ``pos`` advances — and ``bench_flush`` would then measure flush cost at
    the wrong (much smaller) kv_len.

    After return:
      - ``eagle3_stable_kv`` length ≈ ``gen_state.pos - 1`` (production-aligned)
      - backlog empty, ``root_logits`` / ``root_hidden`` fresh
      - ``dflash_target_hidden`` has accumulated all accepted-token hidden
        states since prefill (size ≈ gen_state.pos - 1), ``dflash_kv`` empty —
        models an E→D switch with maximum idleness.
    """
    if gen_state.eagle3_backlog_tokens:
        _flush_eagle3_backlog(gen_state)
    rounds = 0
    while gen_state.pos < target_pos or rounds < min_rounds:
        _eagle3_round(gen_state)
        # Match production: flush each round's accepted tokens so stable_kv
        # advances with pos and root_* are fresh for the next round.
        _flush_eagle3_backlog(gen_state)
        rounds += 1
    return rounds


@torch.inference_mode()
def warm_dflash(gen_state: _GenState, block_size: int, target_pos: int, min_rounds: int = 3) -> int:
    """Run real DFlash rounds until ``gen_state.pos >= target_pos`` and at least
    ``min_rounds`` rounds have executed. Returns the number of rounds run.

    After return: DFlash invariant ``dflash_kv_len + target_hidden.shape[1] =
    dflash_start`` holds, ``target_hidden`` size = last round's accepted count.
    """
    gen_state.dflash_block_size = block_size
    rounds = 0
    while gen_state.pos < target_pos or rounds < min_rounds:
        _dflash_round(gen_state)
        rounds += 1
    return rounds


@contextlib.contextmanager
def disable_logits_to_keep(model):
    """Temporarily force ``model.forward`` to ignore the ``logits_to_keep`` kwarg,
    restoring the pre-patch behaviour (full sequence through MLP + lm_head).

    Used by ``--ablate-logits-to-keep`` to A/B the optimization in one run.
    """
    orig = model.forward

    def patched(*args, **kwargs):
        kwargs.pop("logits_to_keep", None)
        return orig(*args, **kwargs)

    model.forward = patched
    try:
        yield
    finally:
        model.forward = orig


@contextlib.contextmanager
def capture_target_hiddens(target, layer_ids):
    """Hook-based capture of specific decoder layers' outputs.

    Replaces ``output_hidden_states=True`` for the duration of the wrapped
    forward. Yields a list-of-tensors-or-None of length ``n_target_layers + 1``
    matching HF's ``hidden_states`` indexing convention (position 0 = embedding
    output, position i+1 = decoder layer i's output). Only positions in
    ``layer_ids`` are populated; the rest are ``None``.

    Memory: holds only ``len(layer_ids)`` tensors instead of all 33+, so
    prefill at long contexts (≥64K tokens) becomes feasible on 40GB GPUs.
    Per-token saved: ``(n_target_layers - len(layer_ids)) × hidden_size × 2 B``.
    For Llama-3.1-8B with 6 needed layers vs 33: ~22 KB/token saved → ~3 GB at
    128K tokens.

    Caller invokes the target *without* ``output_hidden_states=True`` and
    passes the yielded list anywhere ``target_output.hidden_states`` would
    have been used (downstream code only indexes specific positions).
    """
    n_layers = target.config.num_hidden_layers
    captured: list = [None] * (n_layers + 1)
    handles = []

    decoder_layers = target.model.layers  # standard for Llama, Qwen, etc.

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # LlamaDecoderLayer returns a bare tensor or a tuple (hidden, ...).
            hidden = out[0] if isinstance(out, tuple) else out
            captured[layer_idx + 1] = hidden

        return hook

    for idx in sorted(set(layer_ids)):
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"layer_idx {idx} out of range (n_layers={n_layers})")
        handles.append(decoder_layers[idx].register_forward_hook(make_hook(idx)))

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _save_kv_refs(cache: DynamicCache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Save per-layer K, V tensor REFERENCES (no tensor copy).

    The bench op may replace each layer's K, V (via index_select / crop /
    update — all of which create new tensors and rebind the cache's internal
    list); the original tensor objects are still alive, held by this list,
    for restoration. This avoids the VRAM cost of deep-cloning the cache,
    which would double memory at large kv_len.
    """
    if hasattr(cache, "key_cache"):
        return [(cache.key_cache[i], cache.value_cache[i]) for i in range(len(cache.key_cache))]
    layers = getattr(cache, "layers", [])
    return [(lyr.keys, lyr.values) for lyr in layers]


def _restore_kv_refs(cache: DynamicCache, saved: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Restore per-layer K, V refs saved by ``_save_kv_refs``.

    Truncates any extra layers added by the op (common case: cache growing
    from empty during the round). Raises if the op shrunk layer count below
    the saved snapshot — that doesn't happen in our paths.
    """
    if hasattr(cache, "key_cache"):
        del cache.key_cache[len(saved) :]
        del cache.value_cache[len(saved) :]
        for i, (k, v) in enumerate(saved):
            if i < len(cache.key_cache):
                cache.key_cache[i] = k
                cache.value_cache[i] = v
            else:
                cache.key_cache.append(k)
                cache.value_cache.append(v)
    else:
        layers = cache.layers
        del layers[len(saved) :]
        for i, (k, v) in enumerate(saved):
            if i >= len(layers):
                raise RuntimeError("DynamicCache layer count shrunk during op — cannot restore.")
            layers[i].keys = k
            layers[i].values = v


@contextlib.contextmanager
def _snapshot_state(gen_state: _GenState):
    """Save mutable ``_GenState`` on entry, restore on exit.

    Replaces the previous deep-clone approach. The big win: KV caches save
    only per-layer tensor REFERENCES (not contents), which avoids doubling
    VRAM at long kv_len (target_kv is 32 GB at kv=32K on Llama-8B; a clone
    wouldn't fit on a 40 GB GPU). Only ``output_ids`` is cloned (it's a
    small buffer mutated in-place by token writes); list fields are
    shallow-copied since the op mutates them via ``.append()`` / ``.clear()``.

    Used by the bench helpers in place of ``copy.deepcopy`` so the same
    ``base_state`` can be reused across all timed iterations safely.
    """
    # KV caches — ref-only (the bulk of state memory)
    s_target_kv = _save_kv_refs(gen_state.target_kv)
    s_stable_kv = _save_kv_refs(gen_state.eagle3_stable_kv)
    s_dflash_kv = _save_kv_refs(gen_state.dflash_kv)

    # output_ids buffer — small (~kbs), mutated in-place by token writes
    s_output_ids = gen_state.output_ids.clone()

    # Tensor refs (replaced by op-side assignment, not in-place)
    s_dth = gen_state.dflash_target_hidden
    s_root_logits = gen_state.eagle3_root_logits
    s_root_hidden = gen_state.eagle3_root_hidden
    s_eagle3_token = gen_state.eagle3_token

    # Scalars
    s_pos = gen_state.pos
    s_block_size = gen_state.dflash_block_size
    s_eagle3_num_steps = gen_state.eagle3_num_steps
    s_eagle3_top_k = gen_state.eagle3_top_k
    s_eagle3_num_draft_tokens = gen_state.eagle3_num_draft_tokens
    s_backlog_start = gen_state.eagle3_backlog_start_pos
    s_cur_logits = gen_state.current_logits
    s_cur_hidden = gen_state.current_hidden_states

    # Lists — shallow copies (op mutates in-place via append/clear)
    s_backlog_tokens = list(gen_state.eagle3_backlog_tokens)
    s_backlog_hiddens = list(gen_state.eagle3_backlog_hiddens)
    s_eagle3_hist = list(gen_state.eagle3_accept_hist)
    s_dflash_hist = list(gen_state.dflash_accept_hist)
    s_entropies = list(gen_state.entropies)
    s_acceptance = list(gen_state.acceptance_lengths)
    s_choices = list(gen_state.drafter_choices)
    s_switch_events = list(gen_state.switch_events)
    s_hidden_norms = list(gen_state.hidden_norms)
    s_dflash_norms = list(gen_state.dflash_target_hidden_norms)
    s_eagle3_norms = list(gen_state.eagle3_target_hidden_norms)
    s_structure_scores = list(gen_state.structure_scores)
    s_dflash_buf = list(gen_state.dflash_hidden_states_buffer)
    s_eagle3_buf = list(gen_state.eagle3_hidden_states_buffer)

    try:
        yield
    finally:
        _restore_kv_refs(gen_state.target_kv, s_target_kv)
        _restore_kv_refs(gen_state.eagle3_stable_kv, s_stable_kv)
        _restore_kv_refs(gen_state.dflash_kv, s_dflash_kv)
        gen_state.output_ids = s_output_ids
        gen_state.dflash_target_hidden = s_dth
        gen_state.eagle3_root_logits = s_root_logits
        gen_state.eagle3_root_hidden = s_root_hidden
        gen_state.eagle3_token = s_eagle3_token
        gen_state.pos = s_pos
        gen_state.dflash_block_size = s_block_size
        gen_state.eagle3_num_steps = s_eagle3_num_steps
        gen_state.eagle3_top_k = s_eagle3_top_k
        gen_state.eagle3_num_draft_tokens = s_eagle3_num_draft_tokens
        gen_state.eagle3_backlog_start_pos = s_backlog_start
        gen_state.current_logits = s_cur_logits
        gen_state.current_hidden_states = s_cur_hidden
        gen_state.eagle3_backlog_tokens = s_backlog_tokens
        gen_state.eagle3_backlog_hiddens = s_backlog_hiddens
        gen_state.eagle3_accept_hist = s_eagle3_hist
        gen_state.dflash_accept_hist = s_dflash_hist
        gen_state.entropies = s_entropies
        gen_state.acceptance_lengths = s_acceptance
        gen_state.drafter_choices = s_choices
        gen_state.switch_events = s_switch_events
        gen_state.hidden_norms = s_hidden_norms
        gen_state.dflash_target_hidden_norms = s_dflash_norms
        gen_state.eagle3_target_hidden_norms = s_eagle3_norms
        gen_state.structure_scores = s_structure_scores
        gen_state.dflash_hidden_states_buffer = s_dflash_buf
        gen_state.eagle3_hidden_states_buffer = s_eagle3_buf


def time_callable(setup_fn, op_fn, n_warmup: int = 5, n_repeats: int = 30) -> list[float]:
    """Time ``op_fn(state)`` where state is created by ``setup_fn()`` before timing.

    The setup is excluded from measurement — we sync between setup and op.
    """
    for _ in range(n_warmup):
        state = setup_fn()
        op_fn(state)
    times = []
    for _ in range(n_repeats):
        state = setup_fn()
        torch.cuda.synchronize()  # complete any GPU work from setup (e.g. deepcopy clones)
        t0 = time.perf_counter()
        op_fn(state)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def summarize(times_s: list[float]) -> dict[str, float]:
    arr = np.array(times_s) * 1000.0  # ms
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p05_ms": float(np.percentile(arr, 5)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n_samples": len(arr),
    }


@torch.inference_mode()
def bench_eagle3_round(
    base_state: _GenState,
    n_repeats: int,
    *,
    num_steps: int | None = None,
    top_k: int | None = None,
    num_draft_tokens: int | None = None,
) -> dict:
    """Time ``_eagle3_round`` from a steady-state EAGLE3 snapshot.

    The optional ``num_steps`` / ``top_k`` / ``num_draft_tokens`` override the
    state's current EAGLE3 hyperparameters for the duration of the bench. The
    snapshot context manager restores them after each iteration.

    Reuses ``base_state`` across iterations via snapshot+restore — avoids
    cloning the full KV cache each iteration (which doubles VRAM).
    """

    def setup():
        return base_state

    def op(gen_state):
        with _snapshot_state(gen_state):
            if num_steps is not None:
                gen_state.eagle3_num_steps = num_steps
            if top_k is not None:
                gen_state.eagle3_top_k = top_k
            if num_draft_tokens is not None:
                gen_state.eagle3_num_draft_tokens = num_draft_tokens
            _eagle3_round(gen_state)

    return summarize(time_callable(setup, op, n_warmup=5, n_repeats=n_repeats))


@torch.inference_mode()
def bench_eagle3_draft(
    base_state: _GenState,
    n_repeats: int,
    *,
    num_steps: int | None = None,
    top_k: int | None = None,
    num_draft_tokens: int | None = None,
) -> dict:
    """Time only the EAGLE3 tree-draft phase (``topk_tree_draft``).

    Isolates the drafter cost from the target-verify cost. Note: total round
    cost (drafter + verify) is what matters for routing, since EAGLE3 verify
    handles ~``num_draft_tokens+1`` tokens via a 4D tree mask (SDPA only)
    whereas DFlash verify handles ``block_size`` tokens — different verify
    cost as well. This sub-bench just decomposes the round into its parts.

    ``stable_kv`` is NOT mutated — ``topk_tree_draft`` creates a temporary
    ``tree_kv`` for expansion and discards it. Wrapped in ``_snapshot_state``
    so the optional config override is restored after each iteration.
    """

    def setup():
        return base_state

    def op(gen_state):
        with _snapshot_state(gen_state):
            if num_steps is not None:
                gen_state.eagle3_num_steps = num_steps
            if top_k is not None:
                gen_state.eagle3_top_k = top_k
            if num_draft_tokens is not None:
                gen_state.eagle3_num_draft_tokens = num_draft_tokens
            topk_tree_draft(
                gen_state.eagle3_model,
                gen_state.eagle3_stable_kv,
                gen_state.eagle3_root_logits,
                gen_state.eagle3_root_hidden,
                gen_state.eagle3_token,
                top_k=gen_state.eagle3_top_k,
                depth=gen_state.eagle3_num_steps,
                total_tokens=gen_state.eagle3_num_draft_tokens,
                device=gen_state.device,
            )

    return summarize(time_callable(setup, op, n_warmup=5, n_repeats=n_repeats))


@torch.inference_mode()
def bench_dflash_round(base_state: _GenState, block_size: int, n_repeats: int) -> dict:
    """Time ``_dflash_round`` at a given block_size from a DFlash steady-state snapshot.

    ``base_state`` must already be in DFlash steady state for ``block_size``
    (the invariant ``dflash_kv_len + target_hidden.shape[1] = pos - 1``).
    """

    def setup():
        return base_state

    def op(gen_state):
        with _snapshot_state(gen_state):
            gen_state.dflash_block_size = block_size
            _dflash_round(gen_state)

    return summarize(time_callable(setup, op, n_warmup=5, n_repeats=n_repeats))


@torch.inference_mode()
def bench_dflash_draft(base_state: _GenState, block_size: int, n_repeats: int) -> dict:
    """Time only the DFlash draft forward (no target verify, no accept).

    Mirrors the draft-side block of ``_dflash_round``: dflash_model forward
    over ``target_hidden + noise_embedding`` followed by the kv crop. That's
    everything before the target verify. Useful for separating the drafter
    cost from the verify cost in the per-kv_len breakdown.
    """

    def setup():
        return base_state

    def op(gen_state):
        with _snapshot_state(gen_state):
            gen_state.dflash_block_size = block_size
            dflash_start = gen_state.pos - 1
            block_output_ids = gen_state.output_ids[:, dflash_start : dflash_start + block_size].clone()
            noise_embedding = gen_state.target.model.embed_tokens(block_output_ids)
            kv_start = gen_state.dflash_kv.get_seq_length()
            _ = gen_state.target.lm_head(
                gen_state.dflash_model(
                    target_hidden=gen_state.dflash_target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=gen_state.global_pos_ids[:, kv_start : dflash_start + block_size],
                    past_key_values=gen_state.dflash_kv,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            gen_state.dflash_kv.crop(dflash_start)

    return summarize(time_callable(setup, op, n_warmup=5, n_repeats=n_repeats))


@torch.inference_mode()
def bench_prefill(
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    n_warmup: int = 2,
    n_repeats: int = 5,
    output_hidden_states: bool = True,
) -> dict:
    """Time target prefill at a fixed input length.

    By default uses ``output_hidden_states=True`` (representative of what
    speculative decoding actually pays during the user-visible TTFT, since
    EAGLE3+DFlash both need the auxiliary hiddens). Set to False for the
    inference-only baseline.

    Note: each call allocates a fresh KV cache. Prior iterations' outputs go
    out of scope at the loop boundary, so memory peaks at one prefill at a
    time even at long N.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    for _ in range(n_warmup):
        _ = target(input_ids, use_cache=True, output_hidden_states=output_hidden_states)

    times = []
    for _ in range(n_repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = target(input_ids, use_cache=True, output_hidden_states=output_hidden_states)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return summarize(times)


@torch.inference_mode()
def bench_flush(
    base_state: _GenState,
    backlog_size: int,
    n_repeats: int,
) -> dict:
    """Time ``_flush_eagle3_backlog`` with a synthetic backlog of ``backlog_size``.

    ``base_state`` should have an empty backlog and populated ``eagle3_stable_kv``
    (the post-warmup snapshot from ``warm_eagle3``). Synthetic tokens/hiddens
    with the right shape/dtype are pre-allocated once outside the timing loop —
    values don't affect latency, only shapes do.
    """
    device = base_state.device
    dtype = base_state.eagle3_model.dtype
    vocab = base_state.eagle3_model.config.vocab_size
    # EAGLE3 flush input is the concatenated aux-hidden — matches ``fc.in_features``.
    target_dim = base_state.eagle3_model.fc.in_features

    # Allocate the synthetic backlog ONCE — reused across iterations and warmup.
    syn_tokens = torch.randint(0, vocab, (1, backlog_size), device=device)
    syn_hiddens = torch.randn(1, backlog_size, target_dim, device=device, dtype=dtype)

    def setup():
        return base_state

    def op(gen_state):
        with _snapshot_state(gen_state):
            gen_state.eagle3_backlog_tokens = [syn_tokens]
            gen_state.eagle3_backlog_hiddens = [syn_hiddens]
            gen_state.eagle3_backlog_start_pos = gen_state.pos
            _flush_eagle3_backlog(gen_state)

    return summarize(time_callable(setup, op, n_warmup=5, n_repeats=n_repeats))


def main() -> None:
    parser = argparse.ArgumentParser()
    # Names mirror benchmark.py for consistency across the codebase.
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Target model path (HF hub or local).")
    parser.add_argument(
        "--draft-name-or-path",
        type=str,
        required=True,
        help="DFlash draft model path. Mirrors benchmark.py " "(in WHIFLASH context, this is the DFlash draft).",
    )
    parser.add_argument(
        "--whiflash-eagle-draft", type=str, required=True, help="EAGLE3 draft model path. Mirrors benchmark.py."
    )
    parser.add_argument(
        "--eagle-num-steps", type=int, default=8, help="EAGLE3 tree depth. Default 8 (matches benchmark.py default)."
    )
    parser.add_argument(
        "--eagle-topk",
        type=int,
        default=10,
        help="EAGLE3 branching factor per depth. Default 10 " "(matches benchmark.py default).",
    )
    parser.add_argument(
        "--eagle-num-draft-tokens",
        type=int,
        default=60,
        help="EAGLE3 global candidate budget. Default 60 " "(matches benchmark.py default).",
    )
    parser.add_argument(
        "--eagle-configs",
        type=str,
        nargs="+",
        default=None,
        help="EAGLE3 hyperparameter sweep — list of 'depth:topk:ndt' triples "
        "(e.g. --eagle-configs 5:4:8 8:10:60). Each config produces its own "
        "_eagle3_round / _eagle3_draft measurement at every base_kv_len. "
        "If unset, falls back to the singleton from --eagle-num-steps / "
        "--eagle-topk / --eagle-num-draft-tokens.",
    )
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="DFlash block sizes to sweep. (Plural here vs --block-size "
        "in benchmark.py, since this script sweeps over multiple.)",
    )
    parser.add_argument(
        "--backlog-sizes",
        type=int,
        nargs="+",
        default=[1, 16, 32, 64, 128, 256, 512, 1024],
        help="Backlog sizes N for flush benchmark.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how quicksort works in Python, and " "then write a small example with sample input.",
    )
    parser.add_argument(
        "--n-warm-rounds",
        type=int,
        default=3,
        help="Minimum number of real rounds to run before snapshotting "
        "(applied on top of the warm-to-kv-len target).",
    )
    parser.add_argument(
        "--base-kv-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
        help="Target generation positions to warm to before benching. "
        "Each value produces a full set of (eagle3_round, dflash_round, "
        "flush) measurements. Captures the kv_len-axis of switching cost. "
        "Default sweeps 512 → 32768; long-context tail (65536, 131072) "
        "requires --use-hidden-state-hooks (default on) to fit on 40GB.",
    )
    parser.add_argument(
        "--measure-e2d-switch",
        action="store_true",
        help="Also measure DFlash round latency in the post-E→D-switch state "
        "(EAGLE3 warmup, no DFlash warmup, target_hidden ≈ base_kv_len). "
        "Quantifies the asymmetric counterpart to the D→E flush cost.",
    )
    parser.add_argument(
        "--warmup-mode",
        choices=["corpus", "fixed"],
        default="corpus",
        help="How to drive the prefill to the target kv_len. "
        "'corpus' (default) uses a long pre-cached text and slices per "
        "kv_len so prefill (cheap) reaches target_pos in one forward. "
        "'fixed' uses --prompt and runs many decode rounds (slow, "
        "and acceptance collapses past natural EOS — leave only as a "
        "fallback for environments without `datasets`).",
    )
    parser.add_argument(
        "--warmup-corpus-path",
        type=str,
        default=None,
        help="Override the cached warmup-corpus text file. "
        "Default: assets/warmup_corpus.txt (auto-built on first use).",
    )
    parser.add_argument(
        "--use-hidden-state-hooks",
        dest="use_hidden_state_hooks",
        action="store_true",
        default=True,
        help="(default) Use forward hooks to capture only the layers "
        "EAGLE3+DFlash consume during prefill. Required for long contexts.",
    )
    parser.add_argument(
        "--no-hidden-state-hooks",
        dest="use_hidden_state_hooks",
        action="store_false",
        help="Disable hook-based capture; fall back to output_hidden_states=True. "
        "OOMs at kv_len ≥ ~64K on 40GB GPUs but matches the path used by "
        "end-to-end benchmark.py. Useful for sanity-checking the hooks.",
    )
    parser.add_argument(
        "--profile-prefill",
        action="store_true",
        help="Also profile target prefill latency at each base_kv_len. "
        "Uses output_hidden_states=True (the baseline path) so the "
        "reported numbers reflect what end-to-end TTFT actually pays.",
    )
    parser.add_argument(
        "--prefill-n-repeats",
        type=int,
        default=5,
        help="Repeats for prefill profiling (kept low because each forward " "is expensive at long N). Default 5.",
    )
    parser.add_argument("--n-repeats", type=int, default=30, help="Timing repeats per configuration.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write components.json.")
    parser.add_argument("--model-name", type=str, default=None, help="Short label for the output file.")
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "fa2", "sdpa"],
        help="Attention backend for target and DFlash draft. 'auto' (default) "
        "uses FA2 if installed, else SDPA. EAGLE3 drafts always use SDPA "
        "(custom 4D masks unsupported by FA2). Mirrors benchmark.py.",
    )
    parser.add_argument(
        "--ablate-logits-to-keep",
        action="store_true",
        help="Run the flush benchmark twice per N: once with the "
        "logits_to_keep optimization (default) and once with it "
        "monkey-patched off (baseline). Reports speedup side-by-side.",
    )
    args = parser.parse_args()

    # Parse --eagle-configs (list of "d:k:n") into tuples; fall back to the
    # singleton from --eagle-num-steps/--eagle-topk/--eagle-num-draft-tokens.
    if args.eagle_configs:
        eagle_configs: list[tuple[int, int, int]] = []
        for cfg_str in args.eagle_configs:
            try:
                d, k, n = (int(x) for x in cfg_str.split(":"))
            except ValueError as e:
                raise ValueError(f"--eagle-configs entry '{cfg_str}' is not 'depth:topk:ndt'") from e
            eagle_configs.append((d, k, n))
    else:
        eagle_configs = [(args.eagle_num_steps, args.eagle_topk, args.eagle_num_draft_tokens)]

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Mirror benchmark.py's --attn-impl handling, including 'fa2' alias.
    if args.attn_impl == "auto":
        attn_impl = "flash_attention_2" if has_flash_attn() else "sdpa"
    else:
        attn_impl = args.attn_impl
        if attn_impl.lower() == "fa2":
            attn_impl = "flash_attention_2"
        if attn_impl == "flash_attention_2" and not has_flash_attn():
            raise RuntimeError("--attn-impl flash_attention_2 requested but flash_attn is not installed.")
    if attn_impl == "sdpa":
        logger.warning("Using SDPA. Absolute times will be higher than on a FlashAttention2 build.")

    # EAGLE3 uses 4D tree masks incompatible with FA2; SDPA is the only option.
    eagle3_attn_impl = "sdpa"

    logger.info(f"Loading target: {args.model_name_or_path} (attn={attn_impl})")
    target = (
        AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    logger.info(f"Loading EAGLE3 draft: {args.whiflash_eagle_draft} (attn={eagle3_attn_impl})")
    eagle3_model = (
        Eagle3DraftModel.from_pretrained(
            args.whiflash_eagle_draft,
            dtype=torch.bfloat16,
            attn_implementation=eagle3_attn_impl,
        )
        .to(device)
        .eval()
    )
    eagle3_model.set_target_model(target)

    logger.info(f"Loading DFlash draft: {args.draft_name_or_path} (attn={attn_impl})")
    dflash_model = (
        DFlashDraftModel.from_pretrained(
            args.draft_name_or_path,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    max_block_size = max(args.block_sizes)
    # Buffer must accommodate the largest base_kv_len + headroom for the bench
    # rounds (each accepts up to ~max_block_size tokens; a few hundred should suffice).
    max_kv_len = max(args.base_kv_lens)
    max_new_tokens_for_buf = max_kv_len + 256

    # --- Pre-load warmup corpus if requested -------------------------------
    # Corpus mode: tokenize ~1.1M-token cached text once, slice per kv_len.
    # Fixed mode: fall back to the short --prompt and rely on decode warmup
    # (much slower; only useful when `datasets` is unavailable).
    warmup_corpus_ids: torch.Tensor | None = None
    if args.warmup_mode == "corpus":
        logger.info("Loading warmup corpus (auto-builds on first use)…")
        load_kwargs = {"min_tokens": max_kv_len + 100}
        if args.warmup_corpus_path is not None:
            load_kwargs["cache_path"] = args.warmup_corpus_path
        warmup_corpus_ids = load_warmup_corpus_tokens(tokenizer, **load_kwargs)
        logger.info(f"  corpus length: {warmup_corpus_ids.shape[0]} tokens")

    # #######################################################################
    # Per-base_kv_len sweep. For each target kv_len:
    #   1. EAGLE3-warm to target_pos        -> snapshot s_e
    #   2. Bench _eagle3_round on s_e
    #   3. Bench _flush_eagle3_backlog (+ optional ablation) on s_e
    #   4. (optional) Bench _dflash_round on s_e         -> E→D switch upper bound
    #   5. Per block_size: DFlash-warm fresh state -> bench _dflash_round (steady state)
    # #######################################################################
    per_kv_results: dict[int, dict] = {}

    for target_kv_len in args.base_kv_lens:
        logger.info("=" * 60)
        logger.info(f"base_kv_len target = {target_kv_len}")
        logger.info("=" * 60)

        # Per-kv prompt: in corpus mode, slice to (target - margin) tokens so
        # prefill alone reaches the target pos in one cheap forward. In fixed
        # mode, use the short --prompt and rely on the decode warmup loop.
        if warmup_corpus_ids is not None:
            prompt_ids = slice_warmup_prompt(warmup_corpus_ids, target_kv_len, margin=10)
            prompt_str: str | None = None
        else:
            prompt_ids = None
            prompt_str = args.prompt

        # --- EAGLE3-warmed snapshot ----------------------------------------
        logger.info("Building EAGLE3-warmed snapshot…")
        eagle3_state = build_prefill_state(
            target,
            eagle3_model,
            dflash_model,
            tokenizer,
            prompt_str,
            eagle3_num_steps=args.eagle_num_steps,
            eagle3_top_k=args.eagle_topk,
            eagle3_num_draft_tokens=args.eagle_num_draft_tokens,
            max_block_size=max_block_size,
            max_new_tokens=max_new_tokens_for_buf,
            device=device,
            input_ids=prompt_ids,
            use_hidden_state_hooks=args.use_hidden_state_hooks,
        )
        set_attn_implementation(target, eagle3_attn_impl)
        n_warm_e = warm_eagle3(eagle3_state, target_pos=target_kv_len, min_rounds=args.n_warm_rounds)
        actual_kv_len_e = eagle3_state.eagle3_stable_kv.get_seq_length()
        logger.info(
            f"  EAGLE3 warmed: {n_warm_e} rounds, pos={eagle3_state.pos}, "
            f"eagle3_stable_kv_len={actual_kv_len_e}, "
            f"dflash_target_hidden={eagle3_state.dflash_target_hidden.shape[1]}"
        )

        # --- _eagle3_round + _eagle3_draft, swept over hyperparam configs ---
        # Each config = (depth, top_k, num_draft_tokens). Per-round cost depends
        # on all three (depth × top_k = drafter forwards; ndt = verify size).
        eagle3_round_results: dict[str, dict] = {}
        eagle3_draft_results: dict[str, dict] = {}
        for d, k, n_dt in eagle_configs:
            cfg_key = f"d{d}_k{k}_n{n_dt}"
            logger.info(f"Benchmarking _eagle3_round + _eagle3_draft (config={cfg_key})…")
            eagle3_round_results[cfg_key] = bench_eagle3_round(
                eagle3_state,
                n_repeats=args.n_repeats,
                num_steps=d,
                top_k=k,
                num_draft_tokens=n_dt,
            )
            eagle3_draft_results[cfg_key] = bench_eagle3_draft(
                eagle3_state,
                n_repeats=args.n_repeats,
                num_steps=d,
                top_k=k,
                num_draft_tokens=n_dt,
            )
        # Pick a reference config for ratios — first entry (sorted) is canonical.
        primary_cfg = next(iter(eagle3_round_results))
        eagle3_stats = eagle3_round_results[primary_cfg]
        T_round_e = eagle3_stats["median_ms"]

        # --- _flush_eagle3_backlog (with optional ablation) ----------------
        logger.info("Benchmarking _flush_eagle3_backlog over backlog sizes…")
        flush_results: dict[int, dict] = {}
        flush_results_baseline: dict[int, dict] = {}
        for N in args.backlog_sizes:
            logger.info(f"  N={N} (patched)")
            flush_results[N] = bench_flush(eagle3_state, backlog_size=N, n_repeats=args.n_repeats)
            if args.ablate_logits_to_keep:
                logger.info(f"  N={N} (baseline: logits_to_keep off)")
                with disable_logits_to_keep(eagle3_model):
                    flush_results_baseline[N] = bench_flush(
                        eagle3_state,
                        backlog_size=N,
                        n_repeats=args.n_repeats,
                    )

        # --- E→D switch DFlash round (eagle3_state has target_hidden ≈ pos-1)
        e2d_switch_results: dict[int, dict] = {}
        if args.measure_e2d_switch:
            logger.info(
                "Benchmarking _dflash_round in post-E→D-switch state "
                f"(target_hidden={eagle3_state.dflash_target_hidden.shape[1]}, "
                "dflash_kv empty)…"
            )
            set_attn_implementation(target, attn_impl)
            for B in args.block_sizes:
                logger.info(f"  block_size={B}")
                e2d_switch_results[B] = bench_dflash_round(
                    eagle3_state,
                    block_size=B,
                    n_repeats=args.n_repeats,
                )

        # --- DFlash-warmed snapshot per block_size ------------------------
        logger.info("Benchmarking _dflash_round at multiple block sizes (steady state)…")
        set_attn_implementation(target, attn_impl)
        dflash_results: dict[int, dict] = {}
        dflash_draft_results: dict[int, dict] = {}
        for B in args.block_sizes:
            logger.info(f"  block_size={B}: DFlash-warming to pos≥{target_kv_len}…")
            dflash_state = build_prefill_state(
                target,
                eagle3_model,
                dflash_model,
                tokenizer,
                prompt_str,
                eagle3_num_steps=args.eagle_num_steps,
                eagle3_top_k=args.eagle_topk,
                eagle3_num_draft_tokens=args.eagle_num_draft_tokens,
                max_block_size=max_block_size,
                max_new_tokens=max_new_tokens_for_buf,
                device=device,
                input_ids=prompt_ids,
            )
            # DFlash warmup expects backlog drained — flush prefill seed first.
            if dflash_state.eagle3_backlog_tokens:
                _flush_eagle3_backlog(dflash_state)
            warm_dflash(dflash_state, block_size=B, target_pos=target_kv_len, min_rounds=args.n_warm_rounds)
            dflash_results[B] = bench_dflash_round(dflash_state, block_size=B, n_repeats=args.n_repeats)
            # Draft-only sub-bench on the same DFlash-warmed state.
            dflash_draft_results[B] = bench_dflash_draft(dflash_state, block_size=B, n_repeats=args.n_repeats)

        per_kv_results[target_kv_len] = {
            "actual_pos": eagle3_state.pos,
            "eagle3_stable_kv_len": actual_kv_len_e,
            "n_warm_rounds_eagle3": n_warm_e,
            # EAGLE3 round + draft per hyperparam config (key = "d{D}_k{K}_n{N}")
            "eagle3_round": eagle3_round_results,
            "eagle3_draft": eagle3_draft_results,
            "eagle3_primary_config": primary_cfg,
            "T_round_eagle3_ms": T_round_e,  # median for primary_cfg, kept for legacy ratios
            "flush": {str(N): stats for N, stats in flush_results.items()},
            "dflash_round": {str(B): stats for B, stats in dflash_results.items()},
            "dflash_draft": {str(B): stats for B, stats in dflash_draft_results.items()},
            "flush_over_eagle3_ratio": {str(N): flush_results[N]["median_ms"] / T_round_e for N in args.backlog_sizes},
            "dflash_over_eagle3_ratio": {str(B): dflash_results[B]["median_ms"] / T_round_e for B in args.block_sizes},
            # Per-config draft fraction: how much of the EAGLE3 round is the draft phase.
            "eagle3_draft_fraction": {
                cfg: (eagle3_draft_results[cfg]["median_ms"] / eagle3_round_results[cfg]["median_ms"])
                for cfg in eagle3_round_results
                if eagle3_round_results[cfg]["median_ms"] > 0
            },
            "dflash_draft_fraction": {
                str(B): dflash_draft_results[B]["median_ms"] / dflash_results[B]["median_ms"]
                for B in args.block_sizes
                if dflash_results[B]["median_ms"] > 0
            },
            # Verify-side cost (round - draft). NOTE: NOT identical across drafters —
            # EAGLE3 verify processes ~ndt+1 tokens with a 4D tree mask (SDPA only)
            # whereas DFlash verify processes block_size tokens with FA2-friendly mask.
            "eagle3_verify_ms": {
                cfg: (eagle3_round_results[cfg]["median_ms"] - eagle3_draft_results[cfg]["median_ms"])
                for cfg in eagle3_round_results
            },
            "dflash_verify_ms": {
                str(B): dflash_results[B]["median_ms"] - dflash_draft_results[B]["median_ms"] for B in args.block_sizes
            },
            **(
                {
                    "e2d_switch_dflash_round": {str(B): stats for B, stats in e2d_switch_results.items()},
                    # Switching cost as ratio over steady-state DFlash round.
                    # >1 means the post-switch first round is more expensive.
                    "e2d_switch_over_steady_ratio": {
                        str(B): e2d_switch_results[B]["median_ms"] / dflash_results[B]["median_ms"]
                        for B in args.block_sizes
                    },
                }
                if args.measure_e2d_switch
                else {}
            ),
            **(
                {
                    "flush_baseline_no_logits_to_keep": {str(N): stats for N, stats in flush_results_baseline.items()},
                    "flush_logits_to_keep_speedup": {
                        str(N): flush_results_baseline[N]["median_ms"] / flush_results[N]["median_ms"]
                        for N in args.backlog_sizes
                    },
                }
                if args.ablate_logits_to_keep
                else {}
            ),
        }

    # #######################################################################
    # Prefill profiling (optional). Uses output_hidden_states=True (no hooks)
    # since this is meant to reflect what end-to-end TTFT actually pays.
    # #######################################################################
    prefill_results: dict[int, dict] = {}
    if args.profile_prefill:
        logger.info("=" * 60)
        logger.info("Profiling target prefill (output_hidden_states=True)…")
        logger.info("=" * 60)
        # Match the kernel used by the spec algorithm. Use FA2 if available.
        set_attn_implementation(target, attn_impl)
        for target_kv_len in args.base_kv_lens:
            if warmup_corpus_ids is not None:
                prefill_input = (
                    slice_warmup_prompt(
                        warmup_corpus_ids,
                        target_kv_len,
                        margin=10,
                    )
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                # Fixed-prompt mode: the prompt may be shorter than target;
                # report what we measured at the actual length.
                prefill_input = tokenizer(
                    args.prompt,
                    return_tensors="pt",
                ).input_ids.to(device)
            actual_len = prefill_input.shape[1]
            logger.info(f"  N={actual_len} (target_kv={target_kv_len})")
            try:
                prefill_results[target_kv_len] = bench_prefill(
                    target,
                    prefill_input,
                    n_warmup=2,
                    n_repeats=args.prefill_n_repeats,
                    output_hidden_states=True,
                )
                prefill_results[target_kv_len]["actual_input_len"] = actual_len
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"  OOM at N={actual_len}: {e}; skipping remainder")
                prefill_results[target_kv_len] = {
                    "actual_input_len": actual_len,
                    "oom": True,
                }
                torch.cuda.empty_cache()
                break

    # #######################################################################
    # Report
    # #######################################################################
    print("\n" + "=" * 72)
    print("WhiFlash component-level latency profile")
    print(f"  Target:       {args.model_name_or_path}")
    print(f"  EAGLE3 draft: {args.whiflash_eagle_draft}")
    print(f"  DFlash draft: {args.draft_name_or_path}")
    print(f"  Attention:    {attn_impl}")
    print(f"  base_kv_lens: {args.base_kv_lens}")
    print("=" * 72)

    for target_kv_len in args.base_kv_lens:
        r = per_kv_results[target_kv_len]
        eagle3_round_results = r["eagle3_round"]
        eagle3_draft_results = r["eagle3_draft"]
        dflash_results = {int(k): v for k, v in r["dflash_round"].items()}
        dflash_draft_results = {int(k): v for k, v in r["dflash_draft"].items()}
        flush_results = {int(k): v for k, v in r["flush"].items()}

        print(
            f"\n--- base_kv_len target={target_kv_len} "
            f"(actual eagle3_kv={r['eagle3_stable_kv_len']}, pos={r['actual_pos']}) ---"
        )

        # Per-round cost — THE routing-relevant metric. Round = drafter + verify;
        # both differ across drafters (EAGLE3 verify n_verify ≈ ndt+1 with 4D
        # mask + SDPA, DFlash verify n_verify = block_size with FA2-friendly mask).
        print("\n  ── PER-ROUND COST (the routing-relevant cost) ──")
        print(
            f"    {'drafter (config)':>26}  {'round (ms)':>11}  "
            f"{'draft (ms)':>11}  {'verify (ms)':>12}  {'draft%':>7}"
        )
        for cfg, round_stats in eagle3_round_results.items():
            r_med = round_stats["median_ms"]
            d_med = eagle3_draft_results[cfg]["median_ms"]
            v_ms = r_med - d_med
            print(
                f"    {f'EAGLE3 ({cfg})':>26}  {r_med:>11.3f}  "
                f"{d_med:>11.3f}  {v_ms:>12.3f}  {d_med / r_med * 100:>6.1f}%"
            )
        for B in args.block_sizes:
            med = dflash_results[B]["median_ms"]
            d_med = dflash_draft_results[B]["median_ms"]
            v_ms = med - d_med
            print(
                f"    {f'DFlash (B={B})':>26}  {med:>11.3f}  "
                f"{d_med:>11.3f}  {v_ms:>12.3f}  {d_med / med * 100:>6.1f}%"
            )

        if args.measure_e2d_switch:
            e2d = {int(k): v for k, v in r["e2d_switch_dflash_round"].items()}
            print(f"\n  _dflash_round (post-E→D switch, target_hidden≈{r['actual_pos']-1})")
            print(f"    {'block_size':>11}  {'switch (ms)':>12}  " f"{'steady (ms)':>12}  {'switch/steady':>14}")
            for B in args.block_sizes:
                sw = e2d[B]["median_ms"]
                st = dflash_results[B]["median_ms"]
                print(f"    {B:>11d}  {sw:>12.3f}  {st:>12.3f}  {sw/st:>13.2f}x")

        print("\n  _flush_eagle3_backlog (sweep backlog size N)")
        if args.ablate_logits_to_keep:
            baseline = {int(k): v for k, v in r["flush_baseline_no_logits_to_keep"].items()}
            print(
                f"    {'N':>5}  {'patched (ms)':>13}  {'baseline (ms)':>14}  "
                f"{'speedup':>8}  {'% of eagle3_round':>18}"
            )
            for N in args.backlog_sizes:
                m = flush_results[N]["median_ms"]
                b = baseline[N]["median_ms"]
                speedup = b / m if m > 0 else float("inf")
                print(f"    {N:>5d}  {m:>13.3f}  {b:>14.3f}  " f"{speedup:>7.2f}x  {m / T_round_e * 100:>17.2f}%")
        else:
            print(f"    {'N':>5}  {'median (ms)':>12}  {'% of eagle3_round':>20}")
            for N in args.backlog_sizes:
                med = flush_results[N]["median_ms"]
                print(f"    {N:>5d}  {med:>12.3f}  {med / T_round_e * 100:>19.2f}%")

    if args.profile_prefill and prefill_results:
        print("\n--- target prefill (output_hidden_states=True) ---")
        print(f"  {'input_len':>9}  {'median (ms)':>12}  {'mean ± std':>16}  " f"{'tokens/s':>10}")
        for kv in args.base_kv_lens:
            if kv not in prefill_results:
                continue
            r = prefill_results[kv]
            if r.get("oom"):
                print(f"  {r['actual_input_len']:>9d}  {'OOM':>12}  " f"{'(skipped)':>16}  {'-':>10}")
                continue
            n = r["actual_input_len"]
            med = r["median_ms"]
            mean, std = r["mean_ms"], r["std_ms"]
            tps = n / (med / 1000) if med > 0 else 0.0
            print(f"  {n:>9d}  {med:>12.2f}  {mean:>7.2f} ± {std:>5.2f}  " f"{tps:>10.1f}")

    # #######################################################################
    # Write JSON
    # #######################################################################
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "components.json"

        record = {
            "model_name": args.model_name or args.model_name_or_path,
            "target_model": args.model_name_or_path,
            "eagle3_draft": args.whiflash_eagle_draft,
            "dflash_draft": args.draft_name_or_path,
            "attn_impl": attn_impl,
            "config": {
                "eagle_num_steps": args.eagle_num_steps,
                "eagle_topk": args.eagle_topk,
                "eagle_num_draft_tokens": args.eagle_num_draft_tokens,
                "eagle_configs": [f"d{d}_k{k}_n{n}" for d, k, n in eagle_configs],
                "block_sizes": args.block_sizes,
                "backlog_sizes": args.backlog_sizes,
                "base_kv_lens": args.base_kv_lens,
                "n_warm_rounds": args.n_warm_rounds,
                "n_repeats": args.n_repeats,
                "measure_e2d_switch": args.measure_e2d_switch,
                "ablate_logits_to_keep": args.ablate_logits_to_keep,
                "use_hidden_state_hooks": args.use_hidden_state_hooks,
                "profile_prefill": args.profile_prefill,
            },
            # 2D characterisation: top-level key is base_kv_len; each entry
            # contains eagle3_round, dflash_round (per block_size), flush
            # (per backlog size), ratios, and (optional) e2d_switch + ablation.
            "per_kv_len": {str(kv): r for kv, r in per_kv_results.items()},
            **({"prefill": {str(kv): r for kv, r in prefill_results.items()}} if args.profile_prefill else {}),
        }

        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nWrote results to {out_path}")


if __name__ == "__main__":
    main()
