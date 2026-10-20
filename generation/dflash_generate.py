"""DFlash speculative generation (block-diffusion drafting).

Adapted from https://github.com/z-lab/dflash (MIT License; see LICENSES/LICENSE_DFLASH).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, cuda_time, extract_context_feature, sample

from .text_structure_score import compute_text_structure_score


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    per_token_log: bool = False,
    tokenizer: object | None = None,
    structure_window: int = 32,
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if (block_size > 1 or per_token_log) else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    final_hidden_states = [] if per_token_log else None
    current_final_hidden = output.hidden_states[-1][:, -1, :].clone() if per_token_log else None

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    entropies = []  # per-token entropy (per_token_log only)
    hidden_norms = []  # per-token hidden-state L2 norm (per_token_log only)
    structure_scores = []  # per-token context-structure score (per_token_log + tokenizer only)
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )[:, -block_size + 1 :, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if (block_size > 1 or per_token_log) else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        acceptance_lengths.append(acceptance_length + 1)

        if per_token_log:
            # Entropy from target logits at position 0 (last verified token from previous step)
            target_logits = output.logits[0, 0, :]
            probs = torch.softmax(target_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum().item()
            entropies.append(entropy)

            # Hidden-state L2 norm from target last layer at position 0
            hidden_norms.append(output.hidden_states[-1][0, 0, :].float().norm().item())
            final_hidden_states.append(current_final_hidden)

            # Context-structure score from recent output tokens
            if tokenizer is not None:
                win_start = max(num_input_tokens, start - structure_window)
                recent_text = tokenizer.decode(
                    output_ids[0, win_start : start + 1].tolist(),
                    skip_special_tokens=True,
                )
                structure_scores.append(compute_text_structure_score(recent_text))

            # Advance by 1 token only, i.e., acceptance_length is 0
            output_ids[:, start : start + 1] = block_output_ids[:, :1]
            output_ids[:, start + 1] = posterior[:, 0]
            start += 1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :1, :]
            current_final_hidden = output.hidden_states[-1][:, 0, :].clone()
        else:
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
                    :, : acceptance_length + 1, :
                ]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        final_hidden_states=final_hidden_states,
        entropies=entropies,
        hidden_norms=hidden_norms,
        structure_scores=structure_scores,
    )
