import glob
import json
import os

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HFValidationError, RepositoryNotFoundError
from safetensors.torch import load_file
from torch import nn
from transformers.utils.generic import ModelOutput

# Layout of a published router repo. Training exports a bare `<run_name>_<version>.safetensors`
# instead (see scripts/release/train/export_router_models.sh); both are accepted below.
ROUTER_WEIGHTS_NAME = "router.safetensors"
ROUTER_CONFIG_NAME = "config.json"


def _load_json(path: str | None) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _resolve(ref: str, **hub_kwargs) -> tuple[str, dict]:
    """Locate router weights + config for a local file, a local directory, or a Hub repo id."""
    if os.path.isfile(ref):
        return ref, _load_json(f"{os.path.splitext(ref)[0]}.json")

    if os.path.isdir(ref):
        weights = os.path.join(ref, ROUTER_WEIGHTS_NAME)
        if not os.path.isfile(weights):
            found = sorted(glob.glob(os.path.join(ref, "*.safetensors")))
            if len(found) != 1:
                raise FileNotFoundError(
                    f"Expected '{ROUTER_WEIGHTS_NAME}' or one '*.safetensors' in '{ref}', found {len(found)}."
                )
            weights = found[0]
        return weights, _load_json(os.path.join(ref, ROUTER_CONFIG_NAME))

    try:
        weights = hf_hub_download(ref, ROUTER_WEIGHTS_NAME, **hub_kwargs)
    except (HFValidationError, RepositoryNotFoundError, EntryNotFoundError) as e:
        raise FileNotFoundError(
            f"Router '{ref}' is not a local file or directory, and fetching it from the Hub failed "
            f"({type(e).__name__}). Pass an exported .safetensors or a repo id such as "
            "'WhiFlash/Qwen3-8B-NeuralRouter-df-e3'."
        ) from e
    try:
        return weights, _load_json(hf_hub_download(ref, ROUTER_CONFIG_NAME, **hub_kwargs))
    except EntryNotFoundError:
        return weights, {}


class RouterModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.01):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        # Provenance of a loaded checkpoint (target model, drafter pair, tree budget); empty for a
        # fresh model or a checkpoint published without a config.
        self.config: dict = {}

    def forward(
        self,
        final_hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ModelOutput:
        logits = self.model(final_hidden_states).squeeze(-1)
        loss = nn.functional.mse_loss(logits, labels) if labels is not None else None

        return ModelOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = torch.bfloat16,
        **hub_kwargs,
    ) -> "RouterModel":
        """Load from a local `.safetensors` file, a local directory, or a Hub repo id."""
        weights_file, config = _resolve(path, **hub_kwargs)
        state_dict = load_file(weights_file)

        model = cls(state_dict["model.0.weight"].shape[1], state_dict["model.3.weight"].shape[1])
        model.load_state_dict(state_dict)
        model.config = config
        model.eval()

        return model.to(device=device, dtype=dtype)
