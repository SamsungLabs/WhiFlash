"""WhiFlash step-level routing: lazy KV catch-up on drafter switch.

Only the active drafter's KV cache is updated each round.  The inactive
drafter accumulates state for later catch-up.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import SimpleNamespace
from typing import Protocol

import torch
from transformers import DynamicCache, PreTrainedModel

from model import DFlashDraftModel, cuda_time, extract_context_feature, sample, set_attn_implementation
from model.eagle3 import Eagle3DraftModel
from neural_router.modeling import RouterModel

from .ddtree_generate import build_ddtree_tree, follow_verified_tree
from .eagle3_generate import _build_causal_mask, _get_kv, _num_layers, _set_kv, evaluate_posterior, topk_tree_draft
from .text_structure_score import compute_text_structure_score

##############################################################################
# Router interface
##############################################################################


@dataclass
class RouterState:
    """Observable state passed to the router each speculation round.

    To avoid unnecessary work during the decode loop, routers are responsible for
    computing their own derived features (e.g. entropy).

    Fields:
      - logits: last target logits
      - eagle3/dflash_acceptance_history: raw per-step acceptance lengths
      - hidden_states: last-layer hidden state
    """

    logits: torch.Tensor | None
    eagle3_acceptance_history: list[int]
    dflash_acceptance_history: list[int]
    hidden_states: torch.Tensor | None = None


class Drafter(StrEnum):
    DFLASH = "dflash"
    EAGLE3 = "eagle3"
    DDTREE = "ddtree"


@dataclass
class RouterOutput:
    result: Drafter
    entropy: float | None = None


class Router(Protocol):
    def route(self, state: RouterState) -> RouterOutput:
        ...


class EntropyThresholdRouter:
    """H_t < threshold → DFlash, else EAGLE3.

    Default drafter (on round 0 or when the signal is unavailable) is DFlash:
    E→D switching is free, but D→E pays a backlog-proportional flush, so
    starting in DFlash is the lower-regret choice.
    """

    def __init__(self, threshold: float = 2.0, default: Drafter = Drafter.DFLASH):
        self.threshold = threshold
        self.default = default

    def route(self, state: RouterState) -> RouterOutput:
        if state.logits is None:
            return RouterOutput(result=self.default)

        entropy = _compute_entropy(state.logits)
        return RouterOutput(
            result=Drafter.DFLASH if entropy < self.threshold else Drafter.EAGLE3,
            entropy=entropy,
        )


class OracleRouter:
    """Oracle Router that switches between Eagle3 and DFlash based on:
    - acceptance length (default), or
    - throughput/utility when utility_based=True:
        - utility_mode="full" (default): acceptance / (latency + switching_cost),
          incorporates switching overhead from switching_cost_seconds.
        - utility_mode="partial": acceptance / latency only (no switching cost).
    """

    def __init__(
        self,
        default: Drafter = Drafter.DFLASH,
        utility_based: bool = False,
        utility_mode: str = "full",
        per_round_latency: dict[str, float] | None = None,
        switching_cost: dict[tuple[str, str], float] | None = None,
    ):
        self.drafter_seq: list[str] = []
        self.step: int = 0
        self.default = default
        self.utility_based = utility_based
        self.utility_mode = utility_mode
        self.per_round_latency = per_round_latency or {Drafter.EAGLE3: 1.0, Drafter.DFLASH: 1.0}
        self.switching_cost = switching_cost or {}

    def _compute_utility(self, drafter: Drafter, acc: int, prev_drafter: Drafter | None) -> float:
        """Compute utility for *drafter* given acceptance *acc* and previous drafter."""
        lat = self.per_round_latency.get(drafter, 1.0)
        if lat <= 0:
            return 0.0
        if self.utility_mode == "full" and prev_drafter is not None and prev_drafter != drafter:
            sw = self.switching_cost.get((prev_drafter, drafter), 0.0)
            return acc / (lat + sw)
        return acc / lat

    def process_drafter_seq(self, dflash_acceptance_lengths, eagle3_acceptance_lengths):
        """Convert per-token acceptance to per-round drafter decisions.

        When utility_based=True, compare utility (acceptance / effective latency)
        instead of raw acceptance.  Ties are broken in favour of *default*.
        """
        self.drafter_seq = []
        i = 0
        n = len(eagle3_acceptance_lengths)

        while i < n:
            e_acc = eagle3_acceptance_lengths[i]
            d_acc = dflash_acceptance_lengths[i]

            if self.utility_based:
                prev = self.drafter_seq[-1] if self.drafter_seq else None
                e_util = self._compute_utility(Drafter.EAGLE3, e_acc, prev)
                d_util = self._compute_utility(Drafter.DFLASH, d_acc, prev)

                if e_util > d_util:
                    self.drafter_seq.append(Drafter.EAGLE3)
                    jump = e_acc
                elif d_util > e_util:
                    self.drafter_seq.append(Drafter.DFLASH)
                    jump = d_acc
                else:
                    # Tie: go with default
                    self.drafter_seq.append(self.default)
                    jump = e_acc
            else:
                # Original: compare raw acceptance
                if e_acc > d_acc:
                    self.drafter_seq.append(Drafter.EAGLE3)
                    jump = e_acc
                elif d_acc > e_acc:
                    self.drafter_seq.append(Drafter.DFLASH)
                    jump = d_acc
                else:
                    # Tie: go with default
                    self.drafter_seq.append(self.default)
                    jump = e_acc

            # Jump forward by acceptance length (at least 1)
            i += max(jump, 1)

        self.step = 0

    def route(self, state: RouterState) -> RouterOutput:
        if self.step >= len(self.drafter_seq):
            # Fallback: return last known drafter or default
            return RouterOutput(result=self.drafter_seq[-1] if self.drafter_seq else Drafter.DFLASH)

        self.step += 1
        return RouterOutput(result=self.drafter_seq[self.step - 1])


class NeuralRouter:
    """Neural router using target hidden states."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.0,
        device: torch.device | str | None = None,
    ):
        self.model = RouterModel.from_pretrained(model_path, device=device)
        self.threshold = threshold

    @torch.inference_mode()
    def route(self, state: RouterState) -> RouterOutput:
        output = self.model(state.hidden_states)
        return RouterOutput(result=Drafter.EAGLE3 if output.logits.item() > self.threshold else Drafter.DFLASH)


##############################################################################
# Mutable generation state
##############################################################################


@dataclass
class _GenState:
    """All mutable state for one whiflash_generate call.

    Passed to round helpers by reference; mutated in place.
    """

    # Models (read-only references).
    # eagle3_model may be None if it's DFlash-vs-DDTree variation.
    eagle3_model: Eagle3DraftModel | None
    dflash_model: DFlashDraftModel
    target: PreTrainedModel

    # Buffers
    output_ids: torch.Tensor  # [1, buf_len]
    global_pos_ids: torch.Tensor  # [1, buf_len]
    device: torch.device

    # Target KV (shared)
    target_kv: DynamicCache

    # EAGLE3 state (new all-tensor API: stable_kv + root logits/hidden from draft forward)
    eagle3_stable_kv: DynamicCache
    eagle3_token: torch.Tensor  # [1, 1]
    eagle3_root_logits: torch.Tensor | None  # [1, vocab] — from draft model forward
    eagle3_root_hidden: torch.Tensor | None  # [1, 1, H] — from draft model forward

    # DFlash state — no backlog needed, target_hidden accumulates naturally
    dflash_kv: DynamicCache
    dflash_target_hidden: torch.Tensor  # [1, ctx_len, H_dflash]

    # EAGLE3 backlog (flushed on switch to EAGLE3)
    eagle3_backlog_tokens: list[torch.Tensor] = field(default_factory=list)
    eagle3_backlog_hiddens: list[torch.Tensor] = field(default_factory=list)
    eagle3_backlog_start_pos: int | None = None

    # Position
    pos: int = 0
    num_input_tokens: int = 0

    # EAGLE3 params
    eagle3_num_steps: int = 5
    eagle3_top_k: int = 4
    eagle3_num_draft_tokens: int = 8

    # DFlash params
    dflash_block_size: int = 16

    # DDTree params
    # tree depth is still capped by dflash_block_size - 1.
    ddtree_tree_budget: int = 15

    # Whether EAGLE3 is one of the active drafters. When False (DFlash-vs-DDTree),
    # the EAGLE3 backlog accumulation and stable_kv flushes are skipped entirely.
    eagle3_in_pair: bool = True

    # Temperature
    temperature: float = 0.0

    # Per-round results
    acceptance_lengths: list[int] = field(default_factory=list)
    last_accept_count: int = 0  # used in DDTree only
    drafter_choices: list[str] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    eagle3_accept_hist: list[int] = field(default_factory=list)
    dflash_accept_hist: list[int] = field(default_factory=list)
    current_logits: torch.Tensor | None = None

    # Neural router inputs
    current_hidden_states: torch.Tensor | None = None

    # Switch-cost logging (DFlash→EAGLE3 flush events)
    switch_events: list[dict] = field(default_factory=list)

    # Timing control: when False, skip cuda_time() calls
    log_timing: bool = False

    # (Optional) routing-data collection (per_token_log mode / Oracle-only).
    hidden_norms: list[int] = field(default_factory=list)
    dflash_target_hidden_norms: list[int] = field(default_factory=list)
    eagle3_target_hidden_norms: list[int] = field(default_factory=list)
    structure_scores: list[int] = field(default_factory=list)

    dump_hidden_states: bool = False
    # Each buffer entry is [4, hidden_dim] (3 layers + avg) at the AR root pos.
    dflash_hidden_states_buffer: list[torch.Tensor] = field(default_factory=list)
    eagle3_hidden_states_buffer: list[torch.Tensor] = field(default_factory=list)

    tokenizer: object | None = None
    structure_window: int = 32  # number of recent tokens to decode


##############################################################################
# Helpers
##############################################################################


def _compute_entropy(logits_1d: torch.Tensor) -> float:
    """Shannon entropy of a 1-D logit vector (nats)."""
    probs = torch.softmax(logits_1d.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-9)).sum().item()


def _flush_eagle3_backlog(gen_state: _GenState) -> None:
    """Catch up EAGLE3 stable_kv from accumulated backlog.

    After flush, gen_state.eagle3_root_logits and gen_state.eagle3_root_hidden are set
    from the last position of the draft model forward.
    """
    if not gen_state.eagle3_backlog_tokens:
        return

    start_t = cuda_time() if gen_state.log_timing else 0.0
    tokens = torch.cat(gen_state.eagle3_backlog_tokens, dim=1)
    hiddens = torch.cat(gen_state.eagle3_backlog_hiddens, dim=1)
    num_backlog_tokens = tokens.shape[1]
    positions = torch.arange(
        gen_state.eagle3_backlog_start_pos,
        gen_state.eagle3_backlog_start_pos + num_backlog_tokens,
        device=gen_state.device,
    ).unsqueeze(0)
    if num_backlog_tokens == 1:
        draft_logits, draft_hidden = gen_state.eagle3_model(
            tokens,
            hiddens,
            positions,
            gen_state.eagle3_stable_kv,
            logits_to_keep=1,
        )
    else:
        ctx_len = gen_state.eagle3_stable_kv.get_seq_length()
        draft_logits, draft_hidden = gen_state.eagle3_model(
            tokens,
            hiddens,
            positions,
            gen_state.eagle3_stable_kv,
            attention_mask=_build_causal_mask(
                ctx_len, num_backlog_tokens, gen_state.device, gen_state.eagle3_model.dtype
            ),
            logits_to_keep=1,
        )
    gen_state.eagle3_root_logits = draft_logits[:, -1, :]
    gen_state.eagle3_root_hidden = draft_hidden[:, -1:, :]

    gen_state.eagle3_backlog_tokens.clear()
    gen_state.eagle3_backlog_hiddens.clear()
    gen_state.eagle3_backlog_start_pos = None

    if gen_state.log_timing:
        t_switch = cuda_time() - start_t
        ctx_len = gen_state.eagle3_stable_kv.get_seq_length()
        gen_state.switch_events.append(
            {
                "direction": "dflash_to_eagle3",
                "num_backlog_tokens": num_backlog_tokens,
                "ctx_len": ctx_len,
                "t_switch_ms": round(t_switch * 1000, 2),
            }
        )


def _flush_dflash_accumulated(gen_state: _GenState) -> None:
    """Eager mode: push all but the last target_hidden entry into DFlash KV.

    After this, dflash_target_hidden has exactly 1 entry and the invariant
    dflash_kv_len + 1 = dflash_start holds.
    """
    num_accumulated = gen_state.dflash_target_hidden.shape[1]
    if num_accumulated <= 1:
        return
    to_flush = gen_state.dflash_target_hidden[:, :-1, :]
    N = to_flush.shape[1]
    kv_start = gen_state.dflash_kv.get_seq_length()
    tokens = gen_state.output_ids[:, kv_start : kv_start + N]
    noise_embedding = gen_state.target.model.embed_tokens(tokens)
    # K = cat(target_hidden[N], noise[N]) = 2N entries, needs 2N positions
    pos_ids = torch.arange(kv_start, kv_start + 2 * N, device=gen_state.device).unsqueeze(0)
    gen_state.dflash_model(
        target_hidden=to_flush,
        noise_embedding=noise_embedding,
        position_ids=pos_ids,
        past_key_values=gen_state.dflash_kv,
        use_cache=True,
        is_causal=False,
    )
    gen_state.dflash_kv.crop(kv_start + N)  # keep only target_hidden-derived entries
    gen_state.dflash_target_hidden = gen_state.dflash_target_hidden[:, -1:, :]


##############################################################################
# Per-round functions
##############################################################################


def _eagle3_round(gen_state: _GenState) -> None:
    """One EAGLE3 speculation round: tree draft → verify → accept → update.

    Uses the all-tensor tree drafting API (topk_tree_draft + evaluate_posterior).
    """
    old_pos = gen_state.pos
    context_kv_len = gen_state.pos - 1

    # 1. Tree draft (all-tensor, stable_kv not modified)
    draft_tokens, retrieve_indices, tree_mask, tree_pos_ids = topk_tree_draft(
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

    # 2. Build verification mask (extend tree_mask with context portion)
    n_verify = draft_tokens.shape[1]
    total_kv = context_kv_len + n_verify
    verify_mask = torch.full(
        (1, 1, n_verify, total_kv),
        float("-inf"),
        device=gen_state.device,
        dtype=gen_state.target.dtype,
    )
    verify_mask[:, :, :, :context_kv_len] = 0.0
    verify_mask[:, :, :, context_kv_len:] = tree_mask.to(gen_state.target.dtype)
    verify_positions = (tree_pos_ids + (gen_state.pos - 1)).unsqueeze(0)

    # 3. Target verification
    target_output = gen_state.target(
        draft_tokens,
        past_key_values=gen_state.target_kv,
        attention_mask=verify_mask,
        position_ids=verify_positions,
        use_cache=True,
        output_hidden_states=True,
    )

    # 4. Acceptance (vectorised)
    padding = torch.full((1, 1), -1, dtype=torch.long, device=gen_state.device)
    draft_tokens_padded = torch.cat([draft_tokens, padding], dim=1)
    candidates = draft_tokens_padded[0, retrieve_indices]
    logits_indexed = target_output.logits[0, retrieve_indices]
    best_candidate, accept_len, bonus_logits = evaluate_posterior(
        logits_indexed,
        candidates,
        gen_state.temperature,
    )
    if gen_state.temperature < 1e-5:
        bonus_token = sample(bonus_logits.unsqueeze(0), gen_state.temperature)
    else:
        bonus_token = torch.multinomial(bonus_logits[0], num_samples=1)[None]

    best_path = retrieve_indices[best_candidate]

    # 5. Write accepted tokens
    for i in range(accept_len):
        gen_state.output_ids[0, gen_state.pos + i] = draft_tokens[0, best_path[i + 1]]
    gen_state.output_ids[0, gen_state.pos + accept_len] = bonus_token.squeeze()

    # 6. Trim target KV: keep context + accepted path
    accepted_indices = [best_path[i].item() for i in range(accept_len + 1)]
    keep = list(range(context_kv_len)) + [context_kv_len + i for i in accepted_indices]
    keep_t = torch.tensor(keep, dtype=torch.long, device=gen_state.device)
    for li in range(_num_layers(gen_state.target_kv)):
        k, v = _get_kv(gen_state.target_kv, li)
        _set_kv(gen_state.target_kv, li, k.index_select(2, keep_t), v.index_select(2, keep_t))

    gen_state.pos += accept_len + 1
    gen_state.eagle3_accept_hist.append(accept_len + 1)

    # 7. Append accepted tokens to EAGLE3 backlog (deferred stable_kv update).
    #    The draft forward that refreshes root_logits/root_hidden is folded into
    #    the next EAGLE3 round's flush.
    new_aux = gen_state.eagle3_model.extract_hidden_states(target_output.hidden_states)
    n = accept_len + 1
    upd_hidden = torch.cat(
        [new_aux[:, best_path[i] : best_path[i] + 1, :] for i in range(n)],
        dim=1,
    )
    shifted_ids = [draft_tokens[0, best_path[i + 1]].item() for i in range(accept_len)]
    shifted_ids.append(bonus_token[0, 0].item())
    upd_tokens = torch.tensor([shifted_ids], device=gen_state.device)
    gen_state.eagle3_backlog_tokens.append(upd_tokens)
    gen_state.eagle3_backlog_hiddens.append(upd_hidden)
    if gen_state.eagle3_backlog_start_pos is None:
        gen_state.eagle3_backlog_start_pos = old_pos - 1
    gen_state.eagle3_token = bonus_token

    # 8. Accumulate DFlash context (inactive — no flush needed)
    dflash_ctx = extract_context_feature(
        target_output.hidden_states,
        gen_state.dflash_model.target_layer_ids,
    )
    accepted_path_indices = [best_path[i] for i in range(n)]
    gen_state.dflash_target_hidden = torch.cat(
        [gen_state.dflash_target_hidden, dflash_ctx[:, accepted_path_indices, :]],
        dim=1,
    )

    # 9. Store inputs for routing
    last_v_idx = best_path[accept_len]
    gen_state.current_logits = target_output.logits[0, last_v_idx, :]
    gen_state.current_hidden_states = target_output.hidden_states[-1][0, last_v_idx : last_v_idx + 1, :]


def _dflash_round(gen_state: _GenState) -> None:
    """One DFlash speculation round: block draft → verify → accept → update.

    Matches the reference spec_generate pattern:
      1. Draft forward includes target_hidden as K context
      2. crop(dflash_start) keeps target_hidden entries, discards noise
      3. target_hidden = accepted hidden states (carried to next round)
    No explicit step-4 KV update — the invariant
    ``dflash_kv_len + target_hidden.shape[1] = dflash_start`` makes it automatic.
    """
    dflash_start = gen_state.pos - 1  # block[0] = last accepted token

    # 1. DFlash draft
    block_output_ids = gen_state.output_ids[:, dflash_start : dflash_start + gen_state.dflash_block_size].clone()
    noise_embedding = gen_state.target.model.embed_tokens(block_output_ids)
    kv_start = gen_state.dflash_kv.get_seq_length()
    draft_logits = gen_state.target.lm_head(
        gen_state.dflash_model(
            target_hidden=gen_state.dflash_target_hidden,
            noise_embedding=noise_embedding,
            position_ids=gen_state.global_pos_ids[:, kv_start : dflash_start + gen_state.dflash_block_size],
            past_key_values=gen_state.dflash_kv,
            use_cache=True,
            is_causal=False,
        )[:, -gen_state.dflash_block_size + 1 :, :]
    )
    gen_state.dflash_kv.crop(dflash_start)
    block_output_ids[:, 1:] = sample(draft_logits)

    # 2. Target verify
    block_position_ids = gen_state.global_pos_ids[:, dflash_start : dflash_start + gen_state.dflash_block_size]
    target_output = gen_state.target(
        block_output_ids,
        position_ids=block_position_ids,
        past_key_values=gen_state.target_kv,
        use_cache=True,
        output_hidden_states=True,
    )

    # 3. Accept
    posterior = sample(target_output.logits, gen_state.temperature)
    acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    n_accepted = acceptance_length + 1
    gen_state.output_ids[:, dflash_start : dflash_start + n_accepted] = block_output_ids[:, :n_accepted]
    gen_state.output_ids[:, dflash_start + n_accepted] = posterior[0, acceptance_length]
    gen_state.target_kv.crop(dflash_start + n_accepted)
    gen_state.pos = dflash_start + n_accepted + 1
    gen_state.dflash_accept_hist.append(n_accepted)

    # 4. Carry target_hidden for next round (matches reference pattern)
    dflash_ctx = extract_context_feature(
        target_output.hidden_states,
        gen_state.dflash_model.target_layer_ids,
    )
    gen_state.dflash_target_hidden = dflash_ctx[:, :n_accepted, :]

    # 5. Accumulate EAGLE3 backlog (inactive, shifted for EAGLE3 alignment:
    #    hidden[i] paired with token[i+1]). Skipped when EAGLE3 is not an active
    #    drafter (DFlash-vs-DDTree variation) — no partner to catch up.
    if gen_state.eagle3_in_pair:
        new_aux = gen_state.eagle3_model.extract_hidden_states(target_output.hidden_states)
        shifted_tokens = torch.cat(
            [
                gen_state.output_ids[:, dflash_start + 1 : dflash_start + n_accepted],
                posterior[:, acceptance_length : acceptance_length + 1],
            ],
            dim=1,
        )
        gen_state.eagle3_backlog_tokens.append(shifted_tokens)
        gen_state.eagle3_backlog_hiddens.append(new_aux[:, :n_accepted, :])
        if gen_state.eagle3_backlog_start_pos is None:
            gen_state.eagle3_backlog_start_pos = dflash_start
        gen_state.eagle3_token = posterior[:, acceptance_length : acceptance_length + 1]

    # 6. Store inputs for routing
    gen_state.current_logits = target_output.logits[0, acceptance_length, :]
    gen_state.current_hidden_states = target_output.hidden_states[-1][0, acceptance_length : acceptance_length + 1, :]


def _ddtree_round(gen_state: _GenState) -> None:
    """One DDTree speculation round: DFlash block draft → heap-built draft tree →
    one-pass tree verify (4D mask) → follow accepted path → compact KV → update.

    Shares the DFlash drafter, KV cache and target_hidden carry with _dflash_round
    (identical step 1), so the two interchange seamlessly within a generation. The
    only differences are tree-shaped verification and an index_select target-KV
    compaction (accepted path is non-contiguous), like _eagle3_round. The DFlash
    state invariant (dflash_kv_len + target_hidden.shape[1] = dflash_start) is
    preserved, so DDTree rounds chain with DFlash rounds without catch-up.
    """
    dflash_start = gen_state.pos - 1  # block[0] = last accepted token (the tree root)
    context_kv_len = dflash_start  # target_kv holds exactly the context at round entry

    # 1. DFlash draft (identical to _dflash_round step 1) — produces per-position
    #    top-k logits over the draft horizon (block_size - 1 positions).
    block_output_ids = gen_state.output_ids[:, dflash_start : dflash_start + gen_state.dflash_block_size].clone()
    noise_embedding = gen_state.target.model.embed_tokens(block_output_ids)
    kv_start = gen_state.dflash_kv.get_seq_length()
    draft_logits = gen_state.target.lm_head(
        gen_state.dflash_model(
            target_hidden=gen_state.dflash_target_hidden,
            noise_embedding=noise_embedding,
            position_ids=gen_state.global_pos_ids[:, kv_start : dflash_start + gen_state.dflash_block_size],
            past_key_values=gen_state.dflash_kv,
            use_cache=True,
            is_causal=False,
        )[:, -gen_state.dflash_block_size + 1 :, :]
    )
    gen_state.dflash_kv.crop(dflash_start)
    root_token = block_output_ids[:, :1]

    # 2. Build the draft tree from per-position logits (heap, bounded by budget).
    node_token_ids, node_depths, parents, child_maps, visibility_cpu, _ = build_ddtree_tree(
        draft_logits[0], gen_state.ddtree_tree_budget, log_timing=False
    )
    current_length = 1 + int(node_token_ids.numel())  # root + tree nodes

    # 3. Compile verify inputs (fresh allocation, matching _eagle3_round's style).
    verify_input_ids = torch.empty((1, current_length), dtype=torch.long, device=gen_state.device)
    verify_input_ids[0, 0] = root_token[0, 0]
    verify_position_ids = torch.empty((1, current_length), dtype=torch.long, device=gen_state.device)
    verify_position_ids[0, 0] = dflash_start
    if current_length > 1:
        verify_input_ids[0, 1:] = node_token_ids.to(gen_state.device)
        verify_position_ids[0, 1:] = node_depths.to(gen_state.device) + dflash_start

    # 4D additive tree mask: context fully visible (0), tree visibility from the
    # build, everything else -inf.
    total_kv = context_kv_len + current_length
    verify_mask = torch.full(
        (1, 1, current_length, total_kv),
        float("-inf"),
        device=gen_state.device,
        dtype=gen_state.target.dtype,
    )
    verify_mask[:, :, :, :context_kv_len] = 0.0
    verify_mask[0, 0, :, context_kv_len:].masked_fill_(visibility_cpu.to(gen_state.device), 0.0)

    # 4. Target verification (one forward over the whole tree).
    target_output = gen_state.target(
        verify_input_ids,
        position_ids=verify_position_ids,
        attention_mask=verify_mask,
        past_key_values=gen_state.target_kv,
        use_cache=True,
        output_hidden_states=True,
    )

    # 5. Follow the longest accepted root→leaf path; bonus = target prediction at leaf.
    posterior = sample(target_output.logits, gen_state.temperature)
    accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
    n_accepted = len(accepted_indices)  # path length incl. root, excl. bonus
    accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=gen_state.device)
    accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)  # [1, n_accepted]
    next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=gen_state.device)

    # 6. Write accepted path + bonus token.
    gen_state.output_ids[:, dflash_start : dflash_start + n_accepted] = accepted_tokens
    gen_state.output_ids[:, dflash_start + n_accepted] = next_token

    # 7. Compact target KV: keep context + accepted tree nodes (non-contiguous).
    keep = list(range(context_kv_len)) + [context_kv_len + i for i in accepted_indices]
    keep_t = torch.tensor(keep, dtype=torch.long, device=gen_state.device)
    for li in range(_num_layers(gen_state.target_kv)):
        k, v = _get_kv(gen_state.target_kv, li)
        _set_kv(gen_state.target_kv, li, k.index_select(2, keep_t), v.index_select(2, keep_t))

    gen_state.pos = dflash_start + n_accepted + 1
    gen_state.last_accept_count = n_accepted  # main loop appends to the slot-keyed history

    # 8. Carry DFlash target_hidden for next round = accepted-path hidden states
    #    (mirrors _dflash_round's dflash_ctx[:, :n_accepted], but path-indexed).
    dflash_ctx = extract_context_feature(target_output.hidden_states, gen_state.dflash_model.target_layer_ids)
    gen_state.dflash_target_hidden = dflash_ctx.index_select(1, accepted_index_tensor)

    # 9. Accumulate EAGLE3 backlog (inactive partner), shifted hidden[i]↔token[i+1].
    #    Skipped in the DFlash-vs-DDTree variation (no EAGLE3 partner).
    if gen_state.eagle3_in_pair:
        new_aux = gen_state.eagle3_model.extract_hidden_states(target_output.hidden_states)
        path_aux = new_aux.index_select(1, accepted_index_tensor)  # [1, n_accepted, H]
        shifted_tokens = torch.cat([accepted_tokens[:, 1:], next_token_t], dim=1)  # [1, n_accepted]
        gen_state.eagle3_backlog_tokens.append(shifted_tokens)
        gen_state.eagle3_backlog_hiddens.append(path_aux)
        if gen_state.eagle3_backlog_start_pos is None:
            gen_state.eagle3_backlog_start_pos = dflash_start
        gen_state.eagle3_token = next_token_t

    # 10. Store inputs for routing (logits/hidden at the last accepted node).
    last_idx = accepted_indices[-1]
    gen_state.current_logits = target_output.logits[0, last_idx, :]
    gen_state.current_hidden_states = target_output.hidden_states[-1][0, last_idx : last_idx + 1, :]


##############################################################################
# Per-token oracle round (per_token_log mode)
##############################################################################


def _oracle_round(gen_state: _GenState) -> None:
    """Per-token-log round: measure both drafters on identical context, advance by 1 token.

    Records counterfactual acceptance lengths for EAGLE3 and DFlash at the same
    target-AR position, then commits a single token using the target's
    next-token prediction at the root. State updates mirror a normal
    n_accepted=1 round so subsequent rounds see consistent caches.

    Target KV evolution within the round:
      enter:    pos - 1 entries
      after E3: pos - 1 + n_verify_e  → cropped back to pos - 1
      after DF: pos - 1 + block_size  → cropped to pos (keeps current_token's KV)
    """

    # Compute context-structure score from recent output tokens
    if gen_state.tokenizer is not None:
        win_start = max(gen_state.num_input_tokens, gen_state.pos - gen_state.structure_window)
        recent_ids = gen_state.output_ids[0, win_start : gen_state.pos].tolist()
        recent_text = gen_state.tokenizer.decode(recent_ids, skip_special_tokens=True)
        structure_score = compute_text_structure_score(recent_text)

    ctx_kv_len = gen_state.pos - 1

    # ── EAGLE3 measurement ─────────────────────────────────────────────────
    set_attn_implementation(gen_state.target, gen_state.eagle3_model.config._attn_implementation)
    _flush_eagle3_backlog(gen_state)

    draft_tokens_e, retrieve_indices, tree_mask, tree_pos_ids = topk_tree_draft(
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

    n_verify_e = draft_tokens_e.shape[1]
    total_kv = ctx_kv_len + n_verify_e
    verify_mask = torch.full(
        (1, 1, n_verify_e, total_kv),
        float("-inf"),
        device=gen_state.device,
        dtype=gen_state.target.dtype,
    )
    verify_mask[:, :, :, :ctx_kv_len] = 0.0
    verify_mask[:, :, :, ctx_kv_len:] = tree_mask.to(gen_state.target.dtype)
    verify_positions = (tree_pos_ids + (gen_state.pos - 1)).unsqueeze(0)

    target_out_e = gen_state.target(
        draft_tokens_e,
        past_key_values=gen_state.target_kv,
        attention_mask=verify_mask,
        position_ids=verify_positions,
        use_cache=True,
        output_hidden_states=True,
    )

    padding = torch.full((1, 1), -1, dtype=torch.long, device=gen_state.device)
    draft_tokens_padded = torch.cat([draft_tokens_e, padding], dim=1)
    candidates = draft_tokens_padded[0, retrieve_indices]
    logits_indexed = target_out_e.logits[0, retrieve_indices]
    best_candidate, eagle_accept_len, _ = evaluate_posterior(
        logits_indexed,
        candidates,
        gen_state.temperature,
    )
    gen_state.eagle3_accept_hist.append(eagle_accept_len + 1)
    best_path = retrieve_indices[best_candidate]

    # Capture the root-position aux hidden for the EAGLE3 backlog before the
    # next verify so we don't carry target_out_e past the crop.
    new_aux_e = gen_state.eagle3_model.extract_hidden_states(target_out_e.hidden_states)
    root_aux = new_aux_e[:, best_path[0] : best_path[0] + 1, :]

    # Drop EAGLE3 verify entries before DFlash measurement.
    gen_state.target_kv.crop(ctx_kv_len)

    # ── DFlash measurement ──────────────────────────────────────────────────
    set_attn_implementation(gen_state.target, gen_state.dflash_model.config._attn_implementation)

    dflash_start = gen_state.pos - 1
    block_output_ids = gen_state.output_ids[:, dflash_start : dflash_start + gen_state.dflash_block_size].clone()
    noise_embedding = gen_state.target.model.embed_tokens(block_output_ids)
    kv_start = gen_state.dflash_kv.get_seq_length()
    draft_logits = gen_state.target.lm_head(
        gen_state.dflash_model(
            target_hidden=gen_state.dflash_target_hidden,
            noise_embedding=noise_embedding,
            position_ids=gen_state.global_pos_ids[:, kv_start : dflash_start + gen_state.dflash_block_size],
            past_key_values=gen_state.dflash_kv,
            use_cache=True,
            is_causal=False,
        )[:, -gen_state.dflash_block_size + 1 :, :]
    )
    gen_state.dflash_kv.crop(dflash_start)
    block_output_ids[:, 1:] = sample(draft_logits)

    block_position_ids = gen_state.global_pos_ids[:, dflash_start : dflash_start + gen_state.dflash_block_size]
    target_out_d = gen_state.target(
        block_output_ids,
        position_ids=block_position_ids,
        past_key_values=gen_state.target_kv,
        use_cache=True,
        output_hidden_states=True,
    )

    posterior = sample(target_out_d.logits, gen_state.temperature)
    dflash_accept_len = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
    gen_state.dflash_accept_hist.append(dflash_accept_len + 1)

    # ── Commit exactly 1 token (target's AR prediction at the root) ────────
    # Mirrors a normal n_accepted=1 path: keep current_token at output_ids[pos-1],
    # write next_token at output_ids[pos], crop target_kv to context+root.
    n_accepted = 1
    gen_state.output_ids[:, dflash_start : dflash_start + n_accepted] = block_output_ids[:, :n_accepted]
    gen_state.output_ids[:, dflash_start + n_accepted] = posterior[0, 0]
    gen_state.target_kv.crop(dflash_start + n_accepted)
    gen_state.pos = dflash_start + n_accepted + 1

    # DFlash state for next round (1 entry, mirrors normal n_accepted=1 path).
    dflash_ctx = extract_context_feature(
        target_out_d.hidden_states,
        gen_state.dflash_model.target_layer_ids,
    )
    gen_state.dflash_target_hidden = dflash_ctx[:, :n_accepted, :]

    # EAGLE3 backlog: append (next_token, current_token's aux) — shifted alignment
    # hidden[i] ↔ embed(token[i+1]).
    next_token = posterior[:, 0:1]
    gen_state.eagle3_backlog_tokens.append(next_token)
    gen_state.eagle3_backlog_hiddens.append(root_aux)
    if gen_state.eagle3_backlog_start_pos is None:
        gen_state.eagle3_backlog_start_pos = dflash_start
    gen_state.eagle3_token = next_token

    # Per-layer target hidden states at the AR root position (sequence pos - 1).
    # Recorded for both drafters' layer sets (which overlap but are not equal)
    # so the per-step features capture each drafter's conditioning signal.
    def _extract_layer_features(layer_ids: list[int]) -> tuple[list[float], torch.Tensor | None]:
        layer_hiddens = [target_out_d.hidden_states[lid + 1][0, 0, :] for lid in layer_ids]
        avg_hidden = torch.stack(layer_hiddens).mean(dim=0)
        norms = [h.float().norm().item() for h in layer_hiddens] + [avg_hidden.float().norm().item()]
        stacked = (
            torch.stack(layer_hiddens + [avg_hidden], dim=0).detach().cpu() if gen_state.dump_hidden_states else None
        )
        return norms, stacked

    d_norms, d_stack = _extract_layer_features(gen_state.dflash_model.target_layer_ids)
    e_norms, e_stack = _extract_layer_features(gen_state.eagle3_model.target_layer_ids)

    # Bookkeeping
    # Per-token features (recorded at root position post-verify).
    gen_state.acceptance_lengths.append(gen_state.pos - (ctx_kv_len + 1))
    gen_state.entropies.append(_compute_entropy(target_out_d.logits[0, 0, :]))
    gen_state.hidden_norms.append(target_out_d.hidden_states[-1][0, 0, :].float().norm().item())
    if gen_state.tokenizer is not None:
        gen_state.structure_scores.append(structure_score)

    gen_state.dflash_target_hidden_norms.append(d_norms)
    gen_state.eagle3_target_hidden_norms.append(e_norms)
    if gen_state.dump_hidden_states:
        gen_state.dflash_hidden_states_buffer.append(d_stack)
        gen_state.eagle3_hidden_states_buffer.append(e_stack)


##############################################################################
# Main entry point
##############################################################################


@torch.inference_mode()
def whiflash_generate(
    eagle3_model: Eagle3DraftModel | None,
    dflash_model: DFlashDraftModel,
    target: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    router: Router,
    stop_token_ids: list[int] | None = None,
    temperature: float = 0.0,
    # EAGLE3 params
    eagle3_num_steps: int = 5,
    eagle3_top_k: int = 4,
    eagle3_num_draft_tokens: int = 8,
    # DFlash params
    dflash_block_size: int = 16,
    # DDTree params
    ddtree_tree_budget: int | None = None,
    drafters: tuple[Drafter, Drafter] = (Drafter.DFLASH, Drafter.EAGLE3),
    # Catch-up strategy
    lazy_catchup: bool = True,
    # Switch-overhead logging (adds cuda.synchronize overhead; disable for throughput benchmarks)
    log_timing: bool = False,
    # Oracle data-collection mode: 1 target token / step, both drafters measured.
    per_token_log: bool = False,
    tokenizer: object | None = None,
    structure_window: int = 32,
    # Raw target hidden states sidecar for representation-level routing experiments.
    # Only takes effect when per_token_log=True.
    dump_hidden_states: bool = False,
) -> SimpleNamespace:
    """Speculative generation with step-level routing between EAGLE3 and DFlash.

    Args:
        lazy_catchup: If True (default), the inactive drafter's KV is only
            updated on switch via a single batched forward.  If False,
            both KV caches are updated every round (eager mode).
        per_token_log: Oracle mode - advance 1 target token / step via _oracle_round.
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    device = input_ids.device

    # Abstract-slot drafter pairing validation.
    valid_pairs = {
        (Drafter.DFLASH, Drafter.EAGLE3),
        (Drafter.DDTREE, Drafter.EAGLE3),
        (Drafter.DFLASH, Drafter.DDTREE),
    }
    if tuple(drafters) not in valid_pairs:
        raise ValueError(
            f"Unsupported drafters {tuple(drafters)!r}; expected one of {sorted(valid_pairs)}. "
            "drafters = (concrete for router's DFLASH decision, concrete for its EAGLE3 decision)."
        )
    eagle3_in_pair = Drafter.EAGLE3 in drafters
    if eagle3_in_pair and eagle3_model is None:
        raise ValueError("eagle3_model is required when EAGLE3 is one of the drafters.")
    if per_token_log and not eagle3_in_pair:
        raise ValueError("per_token_log (oracle) mode measures EAGLE3 vs DFlash and requires EAGLE3 in the pair.")
    ddtree_budget = ddtree_tree_budget if ddtree_tree_budget is not None else dflash_block_size - 1

    # Output buffer
    mask_token_id = dflash_model.mask_token_id
    buf_len = max_length + max(eagle3_num_draft_tokens, dflash_block_size) + 1
    output_ids = torch.full((1, buf_len), mask_token_id, dtype=torch.long, device=device)
    output_ids[:, :num_input_tokens] = input_ids
    global_pos_ids = torch.arange(buf_len, device=device).unsqueeze(0)

    # ── Prefill (target only) ────────────────────────────────────────────────
    prefill_start = cuda_time()
    target_output = target(
        input_ids,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    target_kv = target_output.past_key_values
    first_token = sample(target_output.logits[:, -1:, :], temperature)
    output_ids[:, num_input_tokens] = first_token.squeeze()

    dflash_ctx_hidden_prefill = extract_context_feature(
        target_output.hidden_states,
        dflash_model.target_layer_ids,
    )

    # ── Build state ──────────────────────────────────────────────────────────
    # DFlash: target_hidden = full prefill context, kv empty.
    #   Invariant: 0 + num_input_tokens = num_input_tokens = dflash_start ✓
    # EAGLE3: backlog seeded with prefill (shifted hidden[i]↔token[i+1]) — catches
    #   up on first use. Skipped entirely when EAGLE3 is not an active drafter.
    if eagle3_in_pair:
        aux_hidden_prefill = eagle3_model.extract_hidden_states(target_output.hidden_states)
        eagle3_prefill_tokens = torch.cat([input_ids[:, 1:], first_token], dim=1)
        eagle3_backlog_tokens_init = [eagle3_prefill_tokens]
        eagle3_backlog_hiddens_init = [aux_hidden_prefill]
        eagle3_backlog_start_pos_init = 0
    else:
        eagle3_backlog_tokens_init = []
        eagle3_backlog_hiddens_init = []
        eagle3_backlog_start_pos_init = None

    gen_state = _GenState(
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
        eagle3_backlog_tokens=eagle3_backlog_tokens_init,
        eagle3_backlog_hiddens=eagle3_backlog_hiddens_init,
        eagle3_backlog_start_pos=eagle3_backlog_start_pos_init,
        pos=num_input_tokens + 1,
        num_input_tokens=num_input_tokens,
        eagle3_num_steps=eagle3_num_steps,
        eagle3_top_k=eagle3_top_k,
        eagle3_num_draft_tokens=eagle3_num_draft_tokens,
        dflash_block_size=dflash_block_size,
        ddtree_tree_budget=ddtree_budget,
        eagle3_in_pair=eagle3_in_pair,
        temperature=temperature,
        current_logits=target_output.logits[0, -1, :],
        current_hidden_states=target_output.hidden_states[-1][0, -1:, :],
        log_timing=log_timing,
        tokenizer=tokenizer,
        structure_window=structure_window,
        dump_hidden_states=dump_hidden_states and per_token_log,
    )

    # Drafter prefill: flush seeded EAGLE3 backlog so stable_kv +
    # root_logits/root_hidden are ready before decode begins, matching original EAGLE and eagle3_generate.py.
    if eagle3_in_pair:
        _flush_eagle3_backlog(gen_state)
    time_to_first_token = cuda_time() - prefill_start

    # ── Decode loop ──────────────────────────────────────────────────────────
    decode_start = cuda_time()

    while gen_state.pos < max_length:
        old_pos = gen_state.pos

        if per_token_log:
            _oracle_round(gen_state)
        else:
            state = RouterState(
                logits=gen_state.current_logits,
                eagle3_acceptance_history=gen_state.eagle3_accept_hist,
                dflash_acceptance_history=gen_state.dflash_accept_hist,
                hidden_states=gen_state.current_hidden_states,
            )
            router_output = router.route(state)

            # The router emits an abstract *slot* (DFLASH or EAGLE3); `drafters` maps
            # each slot to the concrete drafter that actually runs this round.
            routed_slot = router_output.result
            drafter = drafters[0] if routed_slot == Drafter.DFLASH else drafters[1]
            gen_state.drafter_choices.append(drafter.value)

            # Per-round target kernel: EAGLE3 and DDTree verify with a 4D tree mask, which is
            # FA2-incompatible, so force SDPA on those rounds; on DFlash rounds, mirror whatever
            # kernel the DFlash draft was loaded with. Skip the switch when already set.
            if drafter == Drafter.DFLASH:
                active_attn_implementation = gen_state.dflash_model.config._attn_implementation
            else:
                active_attn_implementation = "sdpa"

            if (
                isinstance(active_attn_implementation, str)
                and gen_state.target.config._attn_implementation != active_attn_implementation
            ):
                set_attn_implementation(gen_state.target, active_attn_implementation)

            # Pre-draft catch-up: EAGLE3 needs explicit flush; DFlash/DDTree automatic
            if drafter == Drafter.EAGLE3:
                _flush_eagle3_backlog(gen_state)

            # Draft + verify
            if drafter == Drafter.EAGLE3:
                _eagle3_round(gen_state)
            elif drafter == Drafter.DDTREE:
                _ddtree_round(gen_state)
            else:
                _dflash_round(gen_state)

            if gen_state.pos == -1:  # tree_draft produced no nodes
                break

            # As DDTree rounds don't append to an acceptance history themselves; do it here.
            if drafter == Drafter.DDTREE:
                hist = gen_state.eagle3_accept_hist if routed_slot == Drafter.EAGLE3 else gen_state.dflash_accept_hist
                hist.append(gen_state.last_accept_count)

            # Eager mode: flush inactive drafter's backlog/accumulated state. Only meaningful when EAGLE3 is the inactive drafter.
            # In DFlash-vs-DDTree both share the DFlash cache, so nothing to catch up.
            if not lazy_catchup and gen_state.eagle3_in_pair:
                _flush_eagle3_backlog(gen_state)
                _flush_dflash_accumulated(gen_state)

            # Bookkeeping
            gen_state.acceptance_lengths.append(gen_state.pos - old_pos)
            if router_output.entropy is not None:
                gen_state.entropies.append(router_output.entropy)

        # Stopping Criteria
        if stop_token_ids is not None and any(
            sid in gen_state.output_ids[0, num_input_tokens : gen_state.pos] for sid in stop_token_ids
        ):
            break

    # ── Post-process ─────────────────────────────────────────────────────────
    output_ids = gen_state.output_ids[:, : min(gen_state.pos, max_length)]
    if stop_token_ids is not None:
        stop_ids_t = torch.tensor(stop_token_ids, device=device)
        gen = output_ids[0, num_input_tokens:]
        idx = torch.isin(gen, stop_ids_t).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + idx[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    result = SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=gen_state.acceptance_lengths,
        drafter_choices=gen_state.drafter_choices,
        entropies=gen_state.entropies,
        switch_events=gen_state.switch_events,
    )
    if per_token_log:
        result.eagle3_acceptance_lengths = gen_state.eagle3_accept_hist
        result.dflash_acceptance_lengths = gen_state.dflash_accept_hist
        result.hidden_norms = gen_state.hidden_norms
        result.dflash_target_hidden_norms = gen_state.dflash_target_hidden_norms
        result.eagle3_target_hidden_norms = gen_state.eagle3_target_hidden_norms
        result.structure_scores = gen_state.structure_scores
        if dump_hidden_states and gen_state.dflash_hidden_states_buffer:
            result.dflash_target_hidden_states = torch.stack(
                gen_state.dflash_hidden_states_buffer,
                dim=0,
            )
            result.eagle3_target_hidden_states = torch.stack(
                gen_state.eagle3_hidden_states_buffer,
                dim=0,
            )
    return result
