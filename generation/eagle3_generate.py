"""EAGLE-3 speculative generation with tree-structured drafting.

Faithful reproduction of the EAGLE-3 algorithm on the Transformers backend
for apple-to-apple comparison with DFlash. Mirrors the tree drafting approach
from SafeAILab/EAGLE (topK_genrate) and SGLang's select_top_k_tokens.

Algorithm per speculation round:
  1. Tree draft: beam-search expansion (top-k at each depth, pruned globally)
  2. Verify: single target forward with tree attention mask
  3. Accept: find longest accepted path, trim KV caches

Returns the same SimpleNamespace format as dflash_generate() for unified analysis.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import Eagle3DraftModel, cuda_time, sample

from .text_structure_score import compute_text_structure_score

############################################################################
# KV cache helpers
############################################################################


def _num_layers(cache: DynamicCache) -> int:
    """Number of populated layers, compatible across transformers versions."""
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(getattr(cache, "layers", []))


def _get_kv(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (key, value) tensors for a layer, compatible across transformers versions."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    return cache.layers[layer_idx].keys, cache.layers[layer_idx].values


def _set_kv(cache: DynamicCache, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    """Set (key, value) tensors for a layer in-place, compatible across transformers versions."""
    if hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = k
        cache.value_cache[layer_idx] = v
    else:
        cache.layers[layer_idx].keys = k
        cache.layers[layer_idx].values = v


def clone_kv_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-clone a DynamicCache (copies underlying tensors)."""
    new = DynamicCache()
    for layer_idx in range(_num_layers(cache)):
        k, v = _get_kv(cache, layer_idx)
        new.update(k.clone(), v.clone(), layer_idx)
    return new


def _build_causal_mask(
    context_len: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Build [1, 1, seq_len, context_len+seq_len] attention mask.

    New tokens attend to all context positions and causally to each other.
    0.0 = attend, -inf = masked.
    """
    mask = torch.full((1, 1, seq_len, context_len + seq_len), float("-inf"), device=device, dtype=dtype)
    mask[:, :, :, :context_len] = 0.0
    mask[:, :, :, context_len:] = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1
    )
    return mask


############################################################################
# Reference-style all-tensor tree drafting (port of cnets.py topK_genrate)
############################################################################


@torch.no_grad()
def topk_tree_draft(
    model: Eagle3DraftModel,
    stable_kv: DynamicCache,
    root_logits: torch.Tensor,  # [1, vocab]
    root_hidden: torch.Tensor,  # [1, 1, hidden_size]
    sample_token: torch.Tensor,  # [1, 1]
    top_k: int = 4,
    depth: int = 5,
    total_tokens: int = 8,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-tensor tree drafting (port of reference topK_genrate).

    Performs beam-search expansion from root_logits, using stable_kv as the
    context KV. Creates a temporary tree_kv (reference copy of stable_kv) for
    expansion — stable_kv is never modified.

    Returns:
        draft_tokens: [1, total_tokens+1] token IDs (sample_token + tree, target vocab).
        retrieve_indices: [num_leaves, max_depth+1] paths for acceptance (padded with -1).
        tree_mask: [1, 1, total_tokens+1, total_tokens+1] float 0/-inf mask for target.
        tree_position_ids: [total_tokens+1] relative position offsets (0 = root).
    """
    hot = model.hot_token_id
    if hot is not None:
        hot = hot.to(device)

    # Create tree_kv from stable_kv (reference copy — stable_kv not modified).
    # DynamicCache.update() will torch.cat to create NEW tensors, so the
    # original stable_kv tensors are never overwritten.
    tree_kv = DynamicCache()
    for layer_idx in range(_num_layers(stable_kv)):
        k, v = _get_kv(stable_kv, layer_idx)
        tree_kv.update(k, v, layer_idx)
    base_kv_len = tree_kv.get_seq_length()

    # --- Root top-k selection ---
    log_probs = torch.log_softmax(root_logits[0], dim=-1)  # [vocab]
    topk_vals, topk_ids = torch.topk(log_probs, top_k)  # [top_k] each
    target_tokens_root = hot[topk_ids] if hot is not None else topk_ids

    # Flat candidate tracking (all candidates across all depths)
    all_scores = [topk_vals]  # each element: [n_candidates_at_depth]
    all_tokens = [target_tokens_root]
    all_parents = [torch.full((top_k,), -1, dtype=torch.long, device=device)]
    all_depths = [torch.zeros(top_k, dtype=torch.long, device=device)]

    # Live state for beam expansion
    live_scores = topk_vals
    live_input_ids = target_tokens_root[None]  # [1, top_k]
    live_hidden = root_hidden.expand(1, top_k, -1).contiguous()  # [1, top_k, H]
    live_global_idx = torch.arange(top_k, device=device)
    offset = top_k

    # Incremental tree mask (reference: self.tree_mask in cnets.py topK_genrate).
    # Format: 1 = attend, 0 = mask. Shape grows each step: [1, 1, top_k, step*top_k+top_k].
    # At each depth, covers only the tree portion of KV (not context — context is always visible).
    tree_mask_init = torch.eye(top_k, device=device)[None, None]  # [1, 1, top_k, top_k]
    draft_tree_mask = tree_mask_init  # step 0: each token sees itself only

    # Pre-allocate attention mask buffer for the largest step (last step).
    # Max kv_len = base_kv_len + (depth-1)*top_k + top_k = base_kv_len + depth*top_k
    max_kv_len = base_kv_len + depth * top_k
    attn_mask_buf = torch.zeros(1, 1, top_k, max_kv_len, dtype=model.dtype, device=device)

    # --- Tree expansion loop ---
    for step in range(depth):
        pos_ids = torch.full(
            (1, top_k),
            base_kv_len + step,
            dtype=torch.long,
            device=device,
        )

        # Reuse pre-allocated mask buffer, sliced to current kv_len.
        # Context portion (0..base_kv_len-1): already 0.0 from initialization.
        # Tree portion: overwrite with converted draft_tree_mask.
        total_kv = base_kv_len + step * top_k + top_k
        attn_mask = attn_mask_buf[:, :, :, :total_kv]
        tree_cols = draft_tree_mask.shape[3]  # step*top_k + top_k
        attn_mask[:, :, :, base_kv_len : base_kv_len + tree_cols] = torch.where(
            draft_tree_mask.bool(),
            0.0,
            float("-inf"),
        ).to(model.dtype)

        logits_out, out_hidden = model(
            live_input_ids,
            live_hidden,
            pos_ids,
            tree_kv,
            attention_mask=attn_mask,
        )
        # tree_kv grows by top_k; stable_kv unaffected

        log_probs = torch.log_softmax(logits_out[0], dim=-1)  # [top_k, vocab]
        topk_vals, topk_ids = torch.topk(log_probs, top_k, dim=-1)  # [top_k, top_k]
        target_ids_all = hot[topk_ids] if hot is not None else topk_ids

        cu_scores = topk_vals + live_scores[:, None]  # [top_k, top_k]

        # Record all top_k^2 candidates from this depth
        all_scores.append(cu_scores.view(-1))
        all_tokens.append(target_ids_all.view(-1))
        parent_idx = live_global_idx.unsqueeze(1).expand(-1, top_k).reshape(-1)
        all_parents.append(parent_idx)
        all_depths.append(torch.full((top_k * top_k,), step + 1, dtype=torch.long, device=device))

        # Global top-k pruning for next depth
        flat_cu = cu_scores.view(-1)
        winner_vals, winner_flat = torch.topk(flat_cu, top_k)
        out_ids = winner_flat // top_k

        live_scores = winner_vals
        winner_draft_ids = topk_ids.view(-1)[winner_flat]
        live_input_ids = (hot[winner_draft_ids] if hot is not None else winner_draft_ids)[None]
        live_hidden = out_hidden[:, out_ids]
        live_global_idx = offset + winner_flat
        offset += top_k * top_k

        # Update tree mask: winners inherit parent's row, append eye for new tokens
        # Reference: tree_mask = cat(tree_mask[:,:,out_ids], tree_mask_init, dim=3)
        draft_tree_mask = torch.cat(
            (draft_tree_mask[:, :, out_ids], tree_mask_init),
            dim=3,
        )

    # --- Tree construction (vectorised, port of reference) ---
    all_scores_t = torch.cat(all_scores)
    all_tokens_t = torch.cat(all_tokens)
    all_parents_t = torch.cat(all_parents)
    all_depths_t = torch.cat(all_depths)

    n_select = min(total_tokens, len(all_scores_t))
    _, top_idx = torch.topk(all_scores_t, n_select)
    top_idx = torch.sort(top_idx).values  # preserve depth order

    sel_tokens = all_tokens_t[top_idx]
    sel_parents = all_parents_t[top_idx]
    sel_depths = all_depths_t[top_idx]
    N = len(sel_tokens)

    # draft_tokens: [1, N+1] = [sample_token, selected_tree_tokens]
    draft_tokens = torch.cat([sample_token.view(-1), sel_tokens])[None]

    # Map parent global indices → positions in draft_tokens (0 = root)
    mask_index = torch.zeros(N, dtype=torch.long, device=device)
    non_root = sel_parents >= 0
    if non_root.any():
        mask_index[non_root] = torch.searchsorted(top_idx, sel_parents[non_root]) + 1

    # Build verification tree mask: [N+1, N+1]
    total_new = N + 1
    tree_mask = torch.eye(total_new, dtype=torch.bool, device=device)
    tree_mask[:, 0] = True
    mask_index_list = mask_index.tolist()
    for i in range(N):
        tree_mask[i + 1] |= tree_mask[mask_index_list[i]]

    # Position IDs: root=0, depth d → d+1
    tree_position_ids = torch.zeros(total_new, dtype=torch.long, device=device)
    tree_position_ids[1:] = sel_depths + 1

    tree_mask_float = torch.where(tree_mask, 0.0, float("-inf")).to(model.dtype)[None, None]

    # Build retrieve_indices (root-to-leaf paths, padded with -1)
    children_set = set(mask_index[non_root].tolist()) if non_root.any() else set()
    children_set.add(0)  # root is always a parent
    max_depth = tree_position_ids.max().item()

    retrieve_indices_list: list[list[int]] = []
    for i in range(1, total_new):
        if i not in children_set:
            path: list[int] = []
            cur = i
            while cur > 0:
                path.append(cur)
                cur = mask_index_list[cur - 1]
            path.append(0)
            path.reverse()
            while len(path) < max_depth + 1:
                path.append(-1)
            retrieve_indices_list.append(path)

    retrieve_indices_list.sort()
    retrieve_indices = torch.tensor(retrieve_indices_list, dtype=torch.long, device=device)

    return draft_tokens, retrieve_indices, tree_mask_float, tree_position_ids


############################################################################
# Vectorised acceptance (port of reference evaluate_posterior)
############################################################################


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[int, int, torch.Tensor]:
    """Acceptance check for speculative decoding (greedy or sampling).

    Port of the reference evaluate_posterior (EAGLE/eagle/model/utils.py).

    Args:
        logits: [num_paths, path_len, vocab] target logits indexed by retrieve_indices.
        candidates: [num_paths, path_len] token IDs along each path.
        temperature: Sampling temperature. 0 = greedy (argmax matching).

    Returns:
        best_candidate: Index of the path with longest acceptance.
        accept_length: Number of accepted draft tokens (0 = none accepted, bonus only).
        bonus_logits: [1, vocab] target logits at the last accepted position
            (greedy) or adjusted probability distribution (sampling).
    """
    if temperature < 1e-5:
        # --- Greedy: accept if draft matches argmax ---
        posterior_mask = (candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)).int()
        accept_lengths = torch.cumprod(posterior_mask, dim=1).sum(dim=1)
        accept_length = accept_lengths.max().item()

        if accept_length == 0:
            best_candidate = 0
        else:
            best_candidate = accept_lengths.argmax().item()

        bonus_logits = logits[best_candidate, accept_length : accept_length + 1]
        return best_candidate, accept_length, bonus_logits

    # --- Sampling: rejection-based acceptance (reference evaluate_posterior) ---
    # Walk positions along paths. At each position, find the first path that
    # shares the accepted prefix so far, then accept the next draft token with
    # probability p(x) (target prob). If rejected, resample from adjusted dist.
    accept_length = 1  # 1-indexed in the loop (position 0 = root, always accepted)
    accept_cand = candidates[0, :1]
    best_candidate = 0
    adjusted = False

    for i in range(1, candidates.shape[1]):
        if i != accept_length:
            break

        # Find paths that match the accepted prefix so far
        is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
        fi = torch.nonzero(is_eq, as_tuple=True)[0][0]

        # Target distribution at this position
        gt_logits = logits[fi, i - 1 : i] / temperature
        gtp = torch.softmax(gt_logits[0], dim=-1)

        adjusted = False
        seen = set()
        for j in range(candidates.shape[0]):
            if not is_eq[j]:
                continue
            x = candidates[j, i]
            xi = x.item()
            if xi in seen or xi == -1:
                continue
            seen.add(xi)

            # Accept with probability p(x) / q(x), where q(x) = 1 (deterministic draft)
            r = torch.rand(1, device=logits.device).item()
            px = gtp[xi].item()
            if r <= px:
                accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                accept_length += 1
                best_candidate = j
                break
            else:
                # Rejection: zero out this token and renormalize
                gtp[xi] = 0
                remaining = gtp.sum()
                if remaining > 0:
                    gtp = gtp / remaining
                adjusted = True

    # Bonus token distribution
    if adjusted and accept_length != candidates.shape[1]:
        bonus_logits = gtp.unsqueeze(0)  # already a probability distribution
    else:
        gt_logits = logits[best_candidate, accept_length - 1 : accept_length] / temperature
        bonus_logits = torch.softmax(gt_logits, dim=-1)  # [1, vocab]

    return best_candidate, accept_length - 1, bonus_logits


############################################################################
# Main generation function
############################################################################


@torch.inference_mode()
def eagle3_generate(
    model: Eagle3DraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    num_steps: int = 5,
    top_k: int = 4,
    num_draft_tokens: int = 8,
    stop_token_ids: list[int] | None = None,
    temperature: float = 0.0,
    per_token_log: bool = False,
    tokenizer: object | None = None,
    structure_window: int = 32,
) -> SimpleNamespace:
    """Speculative generation using EAGLE-3 tree-structured drafting.

    Args:
        model: EAGLE-3 draft model (shares embed/head with target).
        target: Target model for verification.
        input_ids: Input token IDs [1, seq_len].
        max_new_tokens: Maximum new tokens to generate.
        num_steps: Number of draft forward passes = tree depth.
        top_k: Branching factor per depth.
        num_draft_tokens: Global candidate budget fed to verifier.
        stop_token_ids: Stop token IDs.
        temperature: Sampling temperature (0 = greedy).
        per_token_log: Oracle mode — advance 1 token/step, record counterfactual acceptance.

    Returns:
        SimpleNamespace with output_ids, timing info, acceptance_lengths.
    """
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    device = input_ids.device

    output_ids = input_ids.new_full((1, max_length + num_draft_tokens + 1), 0)
    output_ids[:, :num_input_tokens] = input_ids

    # === PREFILL ===
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

    # Extract auxiliary hidden states for EAGLE-3
    aux_hidden = model.extract_hidden_states(target_output.hidden_states)

    # Build stable_kv (draft model persistent KV) over full context.
    # EAGLE3 alignment: hidden[i] paired with embed(token[i+1]).
    stable_kv = DynamicCache()
    draft_prefill_ids = torch.cat([input_ids[:, 1:], first_token], dim=1)
    prefill_pos_ids = torch.arange(num_input_tokens, device=device).unsqueeze(0)
    draft_logits, draft_hidden = model(
        draft_prefill_ids,
        aux_hidden,
        prefill_pos_ids,
        stable_kv,
        attention_mask=_build_causal_mask(0, num_input_tokens, device, model.dtype),
        logits_to_keep=1,
    )
    # draft_logits[:, -1] = root logits for first tree draft
    # draft_hidden[:, -1:] = root pre-norm hidden for tree expansion

    time_to_first_token = cuda_time() - prefill_start

    # === DECODE LOOP ===
    pos = num_input_tokens + 1
    acceptance_lengths = []
    entropies = []
    hidden_norms = []
    structure_scores = []
    final_hidden_states = []

    current_token = first_token
    root_logits = draft_logits[:, -1, :]  # [1, vocab]
    root_hidden = draft_hidden[:, -1:, :]  # [1, 1, hidden_size]
    current_final_hidden = target_output.hidden_states[-1][:, -1, :].clone()

    # Pending stable_kv update — deferred to the start of the next iteration
    # so we skip the draft forward when EOS is detected after acceptance.
    # None on first iteration (prefill already produced root_logits/root_hidden).
    pending_update = None  # tuple (new_token_ids, new_hidden, new_positions) or None

    decode_start = cuda_time()
    while pos < max_length:
        # 0. APPLY DEFERRED STABLE_KV UPDATE (skipped on first iteration)
        if pending_update is not None:
            upd_tokens, upd_hidden, upd_positions = pending_update
            n = upd_tokens.shape[1]
            if n == 1:
                draft_logits_new, draft_hidden_new = model(
                    upd_tokens,
                    upd_hidden,
                    upd_positions,
                    stable_kv,
                    logits_to_keep=1,
                )
            else:
                ctx = stable_kv.get_seq_length()
                mask = _build_causal_mask(ctx, n, device, model.dtype)
                draft_logits_new, draft_hidden_new = model(
                    upd_tokens,
                    upd_hidden,
                    upd_positions,
                    stable_kv,
                    attention_mask=mask,
                    logits_to_keep=1,
                )
            root_logits = draft_logits_new[:, -1, :]
            root_hidden = draft_hidden_new[:, -1:, :]
            pending_update = None

        context_kv_len = pos - 1

        # 1. TREE DRAFT (all-tensor, stable_kv not modified)
        draft_tokens, retrieve_indices, tree_mask, tree_pos_ids = topk_tree_draft(
            model,
            stable_kv,
            root_logits,
            root_hidden,
            current_token,
            top_k=top_k,
            depth=num_steps,
            total_tokens=num_draft_tokens,
            device=device,
        )

        # 2. BUILD VERIFICATION MASK (extend tree_mask with context portion)
        n_verify = draft_tokens.shape[1]
        total_kv = context_kv_len + n_verify
        verify_mask = torch.full(
            (1, 1, n_verify, total_kv),
            float("-inf"),
            device=device,
            dtype=target.dtype,
        )
        verify_mask[:, :, :, :context_kv_len] = 0.0
        verify_mask[:, :, :, context_kv_len:] = tree_mask.to(target.dtype)

        verify_positions = (tree_pos_ids + (pos - 1)).unsqueeze(0)

        # 3. TARGET VERIFICATION
        target_output = target(
            draft_tokens,
            past_key_values=target_kv,
            attention_mask=verify_mask,
            position_ids=verify_positions,
            use_cache=True,
            output_hidden_states=True,
        )

        # 4. ACCEPTANCE (vectorised)
        padding = torch.full((1, 1), -1, dtype=torch.long, device=device)
        draft_tokens_padded = torch.cat([draft_tokens, padding], dim=1)
        candidates = draft_tokens_padded[0, retrieve_indices]
        logits_indexed = target_output.logits[0, retrieve_indices]
        best_candidate, accept_len, bonus_logits = evaluate_posterior(
            logits_indexed,
            candidates,
            temperature,
        )
        if temperature < 1e-5:
            bonus_token = sample(bonus_logits.unsqueeze(0), temperature)
        else:
            # bonus_logits is already a probability distribution from rejection sampling
            bonus_token = torch.multinomial(bonus_logits[0], num_samples=1)[None]

        # 5. RECORD ACCEPTANCE
        acceptance_lengths.append(accept_len + 1)

        new_aux = model.extract_hidden_states(target_output.hidden_states)
        best_path = retrieve_indices[best_candidate]  # [max_depth+1], may have -1 padding

        if per_token_log:
            # Oracle mode: record counterfactual, advance by 1
            logits_root = target_output.logits[0, best_path[0], :].float()
            probs = torch.softmax(logits_root, dim=-1)
            entropies.append(-(probs * torch.log(probs + 1e-9)).sum().item())

            # Hidden-state L2 norm from target last layer at root position
            hidden_norms.append(target_output.hidden_states[-1][0, best_path[0], :].float().norm().item())

            final_hidden_states.append(current_final_hidden)

            # Context-structure score from recent output tokens
            if tokenizer is not None:
                win_start = max(num_input_tokens, pos - structure_window)
                recent_text = tokenizer.decode(
                    output_ids[0, win_start:pos].tolist(),
                    skip_special_tokens=True,
                )
                structure_scores.append(compute_text_structure_score(recent_text))

            bonus_oracle = sample(target_output.logits[:, best_path[0] : best_path[0] + 1, :], temperature)
            output_ids[0, pos] = bonus_oracle.squeeze()

            # Trim target KV: keep context + root only
            keep = list(range(context_kv_len)) + [context_kv_len + best_path[0].item()]
            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
            for li in range(_num_layers(target_kv)):
                k, v = _get_kv(target_kv, li)
                _set_kv(target_kv, li, k.index_select(2, keep_t), v.index_select(2, keep_t))

            # Defer stable_kv update to next iteration
            root_aux = new_aux[:, best_path[0] : best_path[0] + 1, :]
            pending_update = (
                bonus_oracle,
                root_aux,
                torch.tensor([[pos - 1]], device=device),
            )
            current_final_hidden = target_output.hidden_states[-1][:, best_path[0], :].clone()

            pos += 1
            current_token = bonus_oracle
        else:
            # Normal speculation — advance by accept_len + 1
            for i in range(accept_len):
                output_ids[0, pos + i] = draft_tokens[0, best_path[i + 1]]
            output_ids[0, pos + accept_len] = bonus_token.squeeze()

            # Trim target KV: keep context + accepted path
            accepted_indices = [best_path[i].item() for i in range(accept_len + 1)]
            keep = list(range(context_kv_len)) + [context_kv_len + i for i in accepted_indices]
            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
            for li in range(_num_layers(target_kv)):
                k, v = _get_kv(target_kv, li)
                _set_kv(target_kv, li, k.index_select(2, keep_t), v.index_select(2, keep_t))

            # Defer stable_kv update to next iteration
            n = accept_len + 1
            new_hidden = torch.cat(
                [new_aux[:, best_path[i] : best_path[i] + 1, :] for i in range(n)],
                dim=1,
            )
            shifted_ids = [draft_tokens[0, best_path[i + 1]].item() for i in range(accept_len)]
            shifted_ids.append(bonus_token[0, 0].item())
            new_token_ids = torch.tensor([shifted_ids], device=device)
            new_positions = torch.arange(pos - 1, pos - 1 + n, device=device).unsqueeze(0)

            pending_update = (new_token_ids, new_hidden, new_positions)

            pos += accept_len + 1
            current_token = bonus_token

        # Check for stop tokens
        if stop_token_ids is not None and any(
            stop_id in output_ids[0, num_input_tokens:pos] for stop_id in stop_token_ids
        ):
            break

    # === POST-PROCESS ===
    output_ids = output_ids[:, : min(pos, max_length)]

    if stop_token_ids is not None:
        stop_ids_tensor = torch.tensor(stop_token_ids, device=device)
        gen_tokens = output_ids[0, num_input_tokens:]
        stop_indices = torch.isin(gen_tokens, stop_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        entropies=entropies,
        hidden_norms=hidden_norms,
        structure_scores=structure_scores,
        final_hidden_states=final_hidden_states,
    )
