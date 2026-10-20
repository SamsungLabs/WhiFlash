# EAGLE-3 draft model (https://github.com/SafeAILab/EAGLE, Apache-2.0; see LICENSES/LICENSE_EAGLE), built on the
# Llama modeling code of Hugging Face Transformers (Apache-2.0; see LICENSES/LICENSE_TRANSFORMERS).
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    GradientCheckpointingLayer,
    LlamaAttention,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from .utils import build_eagle3_target_layer_ids

# Default number of auxiliary hidden state layers (matches SGLang's hardcoded default)
_DEFAULT_NUM_AUX_LAYERS = 3


class Eagle3Attention(LlamaAttention):
    """EAGLE3 attention: identical to LlamaAttention except QKV input is 2*hidden_size
    (concat of normalized input_embeds + normalized hidden states)."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        self.is_causal = False  # tree mask passed explicitly

        # Override projections: input is 2*hidden_size instead of hidden_size
        self.q_proj = nn.Linear(2 * hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden_size, num_kv_heads * self.head_dim, bias=False)


class Eagle3DecoderLayer(GradientCheckpointingLayer):
    """Single decoder layer with modified attention for EAGLE3.

    Key difference from LlamaDecoderLayer: takes an extra `input_embeds` input.
    QKV input is concat([input_layernorm(input_embeds), hidden_norm(hidden_states)]),
    giving 2*hidden_size input to Eagle3Attention.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Eagle3Attention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        logits_to_keep: int = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        # Self Attention — QKV input is concat([normed embeds, normed hidden]) (EAGLE3-specific)
        hidden_states, _ = self.self_attn(
            hidden_states=torch.cat([self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)], dim=-1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # If only the last K tokens are consumed, drop the prefix here so
        # MLP + downstream norm/lm_head skip compute of the last K tokens.
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Eagle3DraftModel(LlamaPreTrainedModel):
    """EAGLE3 draft model for transformers-level speculative decoding.

    Inherits LlamaPreTrainedModel for flash_attn/sdpa/attention_backend support.
    EAGLE3-specific fields like draft_vocab_size are stored as dynamic attributes
    by HF's PretrainedConfig.

    Usage:
        eagle3 = Eagle3DraftModel.from_pretrained("path/to/eagle3", target_model=target)
        # eagle3 shares embed_tokens and lm_head with target_model
    """

    # All EAGLE3 checkpoints use model_type="llama" -> AutoConfig gives LlamaConfig
    config_class = AutoConfig
    # Weights sit directly on self (fc, midlayer, norm, ...), not under self.model
    base_model_prefix = ""
    _no_split_modules = ["Eagle3DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = ["t2d"]

    def __init__(self, config):
        super().__init__(config)

        hidden_size = config.hidden_size
        num_aux_layers = getattr(config, "num_aux_layers", _DEFAULT_NUM_AUX_LAYERS)

        # Project concatenated auxiliary hidden states
        self.fc = nn.Linear(hidden_size * num_aux_layers, hidden_size, bias=False)

        # Single decoder layer (layer_idx=0: EAGLE3 always uses one layer)
        self.midlayer = Eagle3DecoderLayer(config, layer_idx=0)

        self.norm = LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

        # LM head: reduced-vocab from checkpoint or full-vocab from target
        draft_vocab_size = getattr(config, "draft_vocab_size", None)
        if draft_vocab_size is not None:
            # Checkpoint has its own reduced-vocab lm_head + d2t mapping
            self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)
            self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long), persistent=True)
            self.load_lm_head_from_target = False
        else:
            # Full-vocab: HF loads checkpoint's lm_head, set_target_model overwrites it
            self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)
            self.load_lm_head_from_target = True

        # hot_token_id: maps draft vocab indices -> target vocab IDs (computed from d2t after loading)
        self.hot_token_id: torch.Tensor | None = None

        # Set after construction via set_target_model()
        self.embed_tokens: nn.Embedding | None = None
        self.target_layer_ids: list[int] = []

        self.post_init()

    def set_target_model(self, target_model: nn.Module) -> None:
        """Share embedding (and optionally lm_head) from the target model.

        Must be called after from_pretrained() before any forward pass.
        Computes target_layer_ids from target model depth and num_aux_layers config.
        """
        num_aux = getattr(self.config, "num_aux_layers", _DEFAULT_NUM_AUX_LAYERS)
        n_target_layers = target_model.config.num_hidden_layers
        self.target_layer_ids = build_eagle3_target_layer_ids(n_target_layers, num_aux)
        self.embed_tokens = target_model.model.embed_tokens
        if self.load_lm_head_from_target:
            self.lm_head = target_model.lm_head

    def get_hot_token_id(self) -> torch.Tensor | None:
        """Return draft-to-target token ID mapping, or None if using full vocab."""
        return self.hot_token_id

    def extract_hidden_states(self, target_hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Extract and concatenate hidden states from target model's specified layers.

        Args:
            target_hidden_states: Tuple of hidden states from target model
                (output_hidden_states=True gives (embed, layer0, layer1, ..., layerN)).
                Index offset +1 because element 0 is the embedding output.
        """
        selected = [target_hidden_states[lid + 1] for lid in self.target_layer_ids]
        return torch.cat(selected, dim=-1)  # [batch, seq_len, hidden_size * num_aux_layers]

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the EAGLE3 draft model.

        Args:
            input_ids: Token IDs to embed [batch, seq_len].
            hidden_states: Projected hidden states from target model [batch, seq_len, hidden_size*num_aux].
            position_ids: Position IDs [batch, seq_len].
            past_key_values: KV cache for autoregressive decoding.
            attention_mask: Optional attention mask.
            logits_to_keep: If > 0, only the last K positions are run through
                MLP + norm + lm_head; output shape becomes [batch, K, ...].

        Returns:
            Tuple of (logits, hidden_states):
                logits: [batch, seq_len, vocab_size]
                hidden_states: [batch, seq_len, hidden_size] — pre-norm, for feeding back
                    to next draft step (the decoder layer applies its own norms internally).
        """
        input_embeds = self.embed_tokens(input_ids)

        # Project concatenated hidden states if needed
        if hidden_states.shape[-1] != self.config.hidden_size:
            hidden_states = self.fc(hidden_states)

        position_embeddings = self.rotary_emb(input_embeds, position_ids)

        hidden_states = self.midlayer(
            hidden_states=hidden_states,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            logits_to_keep=logits_to_keep,
        )

        normed = self.norm(hidden_states)
        logits = self.lm_head(normed)
        return logits, hidden_states

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> Eagle3DraftModel:
        """Load EAGLE3 draft model from HuggingFace checkpoint.

        Call set_target_model(target) separately after loading to share
        embed_tokens and lm_head with the target model.

        Args:
            pretrained_model_name_or_path: HF model ID or local path.
            device: Device to load model onto.
            dtype: Data type for model parameters.
            **kwargs: Additional arguments passed to PreTrainedModel.from_pretrained().
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, torch_dtype=dtype, **kwargs
        )
        # Compute hot_token_id from d2t (draft-to-target delta mapping)
        if hasattr(model, "d2t") and model.d2t is not None and model.d2t.any():
            model.hot_token_id = model.d2t + torch.arange(len(model.d2t), device=model.d2t.device)
        return model
