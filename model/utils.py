# build_target_layer_ids, extract_context_feature, sample and apply_rotary_pos_emb are adapted from
# DFlash (https://github.com/z-lab/dflash, MIT; see LICENSES/LICENSE_DFLASH).
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import rotate_half

############################################
# Misc utilities
############################################


def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def has_flash_attn() -> bool:
    """Whether the flash_attn package is importable in this env."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


############################################
# Model utilities
############################################


def _validate_impl(impl: str) -> None:
    if impl == "eager":
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    registered = (
        set(ALL_ATTENTION_FUNCTIONS.keys())
        if hasattr(ALL_ATTENTION_FUNCTIONS, "keys")
        else set(ALL_ATTENTION_FUNCTIONS._global_mapping.keys())
    )
    if impl not in registered:
        raise RuntimeError(f"attn_implementation '{impl}' is not registered. " f"Available: {sorted(registered)}.")


def set_attn_implementation(model: nn.Module, impl: str) -> None:
    """Mutate `_attn_implementation` on model.config and all submodule configs.

    Validates that `impl` is registered in transformers' attention dispatch
    table before mutating, so misconfigurations fail here instead of deep in
    the first forward.
    """
    _validate_impl(impl)
    seen: set[int] = set()
    root_cfg = getattr(model, "config", None)
    if root_cfg is not None:
        root_cfg._attn_implementation = impl
        seen.add(id(root_cfg))
    for sub in model.modules():
        sub_cfg = getattr(sub, "config", None)
        if sub_cfg is not None and id(sub_cfg) not in seen:
            sub_cfg._attn_implementation = impl
            seen.add(id(sub_cfg))


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k.

    Args:
        q, k: [batch, heads, seq_len, head_dim]
        cos, sin: [batch, seq_len, head_dim] — position-aligned, no head dim yet
        unsqueeze_dim: dimension to unsqueeze for head broadcast (default 1)
    """
    cos = cos.unsqueeze(unsqueeze_dim)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]
    return target_layer_ids


def build_eagle3_target_layer_ids(num_target_layers: int, num_aux_layers: int) -> list[int]:
    """Return layer indices for EAGLE3 aux hidden state extraction.

    Matches the SafeAILab/EAGLE reference exactly: captures hidden states *before*
    running layers at indices [2, N//2, N-3], which are the outputs of layers
    [1, N//2-1, N-4]. The num_aux_layers == 3 case is hardcoded in the reference.
    """
    if num_aux_layers == 1:
        return [num_target_layers // 2]
    if num_aux_layers == 3:
        return [1, num_target_layers // 2 - 1, num_target_layers - 4]
    # General evenly-spaced fallback for non-standard checkpoints
    start = 1
    end = num_target_layers - 4
    span = end - start
    return [int(round(start + (i * span) / (num_aux_layers - 1))) for i in range(num_aux_layers)]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)
