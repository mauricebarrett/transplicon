"""Prediction module for transplicon.

Load a trained model and predict KO probabilities for new sequences.

Usage (Python API)::

    from transplicon.predict import load_model, predict_fasta

    model = load_model("/path/to/model_dir")
    df = predict_fasta(model, "sequences.fasta")

Usage (CLI)::

    transplicon -i sequences.fasta -m /path/to/model_dir -o predictions.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import mappy
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

from transplicon.feature_extraction import (
    MODEL_IDS,
    _get_device,
    _mean_pool,
    _tokenize_transformers,
    get_nth_layer,
    get_ntv3_bottleneck,
)
from transplicon.head import KOHead


@dataclass
class TranspliconModel:
    """Container for everything needed to run predictions."""

    head: KOHead
    backbone: AutoModelForMaskedLM
    tokenizer: AutoTokenizer
    ko_columns: list[str]
    model_name: str
    layer: str
    device: torch.device = field(default_factory=_get_device)


def load_model(model_dir: str | Path) -> TranspliconModel:
    """Load a trained transplicon model from *model_dir*.

    The directory must contain:
    - ``head.pt``            – MLP head weights
    - ``head_config.json``   – head architecture (input_dim, n_kos, hidden_dims)
    - ``ko_columns.txt``     – KO column names (one per line)
    - ``model_info.json``    – DNA LM name and layer used during training
    """
    model_dir = Path(model_dir)

    with open(model_dir / "head_config.json") as f:
        head_config = json.load(f)
    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)
    with open(model_dir / "ko_columns.txt") as f:
        ko_columns = [line.strip() for line in f if line.strip()]

    device = _get_device()

    # Load MLP head
    head = KOHead(
        head_config["input_dim"],
        head_config["n_kos"],
        hidden_dims=head_config["hidden_dims"],
    )
    head.load_state_dict(torch.load(model_dir / "head.pt", weights_only=True))
    head = head.to(device)
    head.eval()

    # Load DNA language model + tokenizer
    model_name = model_info["model_name"]
    layer = model_info["layer"]
    model_id = MODEL_IDS.get(model_name, model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True
    )
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)
    backbone.eval()

    return TranspliconModel(
        head=head,
        backbone=backbone,
        tokenizer=tokenizer,
        ko_columns=ko_columns,
        model_name=model_name,
        layer=layer,
        device=device,
    )


def predict(
    model: TranspliconModel,
    sequences: list[tuple[str, str]],
    batch_size: int = 64,
) -> pd.DataFrame:
    """Predict KO probabilities for a list of *(id, sequence)* tuples.

    Returns a DataFrame with sequence IDs as index and KO names as columns.
    Values are probabilities in [0, 1].
    """
    if not sequences:
        return pd.DataFrame(columns=model.ko_columns)

    seq_ids = [s[0] for s in sequences]
    seqs = [s[1] for s in sequences]

    # Tokenize
    input_ids, attention_mask, _ = _tokenize_transformers(
        seqs, seq_ids, model.tokenizer
    )

    # Choose layer selector
    if model.layer == "bottleneck":
        selector_fn = get_ntv3_bottleneck
    elif model.layer == "last":
        selector_fn = get_nth_layer(-1)
    else:
        selector_fn = get_nth_layer(int(model.layer))

    # Extract features in batches
    n_seqs = input_ids.shape[0]
    hidden_list: list[torch.Tensor] = []

    with torch.no_grad():
        for i in range(0, n_seqs, batch_size):
            batch_ids = input_ids[i : i + batch_size].to(model.device)
            batch_mask = attention_mask[i : i + batch_size].to(model.device)

            outputs = model.backbone(
                batch_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = selector_fn(outputs.hidden_states)
            hidden_list.append(hidden.cpu())

    hidden_states = torch.cat(hidden_list, dim=0)
    features = _mean_pool(hidden_states, attention_mask)

    # Run MLP head
    with torch.no_grad():
        logits = model.head(features.to(model.device))
        probs = torch.sigmoid(logits).cpu()

    return pd.DataFrame(
        probs.numpy(),
        index=seq_ids,
        columns=model.ko_columns,
    )


def predict_fasta(
    model: TranspliconModel,
    fasta_path: str | Path,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Read a FASTA file and predict KO probabilities for each sequence."""
    sequences: list[tuple[str, str]] = []
    for name, seq, _ in mappy.fastx_read(str(fasta_path)):
        sequences.append((name, seq))

    if not sequences:
        print(f"Warning: no sequences found in {fasta_path}", file=sys.stderr)

    return predict(model, sequences, batch_size=batch_size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transplicon",
        description="Predict KO functional profiles from amplicon sequences",
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="FASTA file of query sequences",
    )
    parser.add_argument(
        "-m", "--model-dir", type=Path, required=True,
        help="Directory containing trained model artifacts "
             "(head.pt, head_config.json, ko_columns.txt, model_info.json)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output CSV path (default: stdout)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for inference (default: 64)",
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> None:
    """Console entry point for ``transplicon``."""
    args = _parse_args(argv)

    print(f"Loading model from {args.model_dir} ...", file=sys.stderr)
    model = load_model(args.model_dir)
    print(
        f"Model loaded: {model.model_name} (layer={model.layer}), "
        f"{len(model.ko_columns)} KOs",
        file=sys.stderr,
    )

    print(f"Predicting on {args.input} ...", file=sys.stderr)
    df = predict_fasta(model, args.input, batch_size=args.batch_size)

    if args.output is not None:
        df.to_csv(args.output)
        print(f"Predictions written to {args.output}", file=sys.stderr)
    else:
        df.to_csv(sys.stdout)


if __name__ == "__main__":
    cli()
