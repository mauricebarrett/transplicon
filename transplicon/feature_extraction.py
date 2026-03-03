"""Feature extraction from DNA language models.

Tokenisation, embedding, and feature extraction (mean-pooled per-sequence vectors).
Supports transformers models (nucleotide-transformer, DNABERT-2) and evo2.

Layer=final hidden state
Pooling=mean-pool
"""

from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_IDS = {
    "DNABERT-2-117M": "zhihan1996/DNABERT-2-117M",
    "nucleotide-transformer-v2-50m-multi-species": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "nucleotide-transformer-v2-250m-multi-species": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "NTv3-100M-pre": "InstaDeepAI/NTv3_100M_pre",
    "NTv3-650M-pre": "InstaDeepAI/NTv3_650M_pre",
    "NTv3-650M-post": "InstaDeepAI/NTv3_650M_post",
}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tokenize_transformers(
    sequences: list[str],
    sequence_ids: list[str],
    tokenizer: AutoTokenizer,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Tokenize with transformers, pad to max length across dataset."""
    encoding = tokenizer(
        sequences,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    return encoding["input_ids"], encoding["attention_mask"], sequence_ids





def get_nth_layer(layer_idx: int):
    """Strategy to extract a specific layer index."""
    def selector(hidden_states: tuple) -> torch.Tensor:
        return hidden_states[layer_idx] if isinstance(hidden_states, (tuple, list)) else hidden_states
    return selector


def get_ntv3_bottleneck(hidden_states: tuple) -> torch.Tensor:
    """Strategy for U-Net like architectures (NTv3): extract the shortest sequence layer."""
    if not isinstance(hidden_states, (tuple, list)):
        return hidden_states
    seq_lengths = [h.shape[1] for h in hidden_states]
    bottleneck_idx = seq_lengths.index(min(seq_lengths))
    return hidden_states[bottleneck_idx]


def _embed_transformers(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model: AutoModelForMaskedLM,
    device: torch.device,
    batch_size: int,
    layer_selector_fn,
) -> torch.Tensor:
    """Run frozen backbone, return selected hidden states."""
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()

    n_seqs = input_ids.shape[0]
    hidden_states_list = []

    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            batch_mask = attention_mask[i : i + batch_size].to(device)

            outputs = model(
                batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = outputs.hidden_states
            hidden = layer_selector_fn(hs)
            hidden_states_list.append(hidden.cpu())

    return torch.cat(hidden_states_list, dim=0)





import torch.nn.functional as F

def _mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool over non-padded tokens. Adapts mask to sequence length."""
    seq_len = hidden_states.shape[1]
    
    # Adapt the mask to this specific layer's sequence length
    float_mask = attention_mask.unsqueeze(1).float()
    adapted_mask = F.adaptive_max_pool1d(float_mask, output_size=seq_len)
    adapted_mask = adapted_mask.squeeze(1).round().unsqueeze(-1)
    
    # Mean-pool with the adapted mask
    denom = adapted_mask.sum(dim=1).clamp(min=1e-9)
    pooled = (hidden_states * adapted_mask).sum(dim=1) / denom
    
    return pooled


def run_feature_extraction(
    seq_meta: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    layer: str = "last",
    batch_size: int = 64,
) -> None:
    """Run tokenisation, embedding, and feature extraction. Write tokenised.pt,
    embeddings.pt, and features.pt to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_id = MODEL_IDS.get(model_name, model_name)

    sequences = seq_meta["sequence"].tolist()
    sequence_ids = seq_meta["sequence_id"].tolist()
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    input_ids, attention_mask, sequence_ids = _tokenize_transformers(
        sequences, sequence_ids, tokenizer
    )

    torch.save(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence_ids": sequence_ids,
        },
        output_dir / "tokenised.pt",
    )

    model = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True
    )
    
    # Choose layer extraction strategy based on model architecture and argument
    if layer == "bottleneck":
        selector_fn = get_ntv3_bottleneck
    elif layer == "last":
        selector_fn = get_nth_layer(-1)
    else:
        selector_fn = get_nth_layer(int(layer))

    hidden_states = _embed_transformers(
        input_ids, attention_mask, model, device, batch_size,
        layer_selector_fn=selector_fn
    )

    torch.save(
        {
            "hidden_states": hidden_states,
            "sequence_ids": sequence_ids,
        },
        output_dir / "embeddings.pt",
    )

    features = _mean_pool(hidden_states, attention_mask)

    torch.save(
        {
            "features": features,
            "sequence_ids": sequence_ids,
        },
        output_dir / "features.pt",
    )
