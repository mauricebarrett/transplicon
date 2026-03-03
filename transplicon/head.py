"""Train an MLP head on extracted features to predict KO probabilities."""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr


WEIGHT_CAP = 25.0


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class KOHead(nn.Module):
    """MLP head: features -> logits (no sigmoid; BCEWithLogitsLoss applies it)."""

    def __init__(
        self,
        input_dim: int,
        n_kos: int,
        hidden_dims: list[int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_kos))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeaturesKODataset(Dataset):
    """Dataset of (features, labels, sample_weight) for a given split."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> None:
        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "sample_weight": self.sample_weights[idx],
        }


def _compute_sample_weights(num_genomes_annotated: torch.Tensor) -> torch.Tensor:
    """sqrt + cap + mean-normalize."""
    raw = torch.sqrt(num_genomes_annotated.clamp(min=1.0, max=WEIGHT_CAP))
    return raw / raw.mean().clamp(min=1e-9)


def train_head(
    features_path: Path,
    ko_matrix_path: Path,
    seq_meta_path: Path,
    output_dir: Path,
    model_name: str,
    layer: str,
    hidden_dims: list[int] | None = None,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> KOHead:
    """Train MLP head on features. Save head.pt, head_config.json, ko_columns.txt."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(features_path, weights_only=True)
    features = data["features"]
    sequence_ids = data["sequence_ids"]

    seq_meta = pd.read_csv(seq_meta_path)
    ko_matrix = pd.read_csv(ko_matrix_path, index_col="sequence_id")

    seq_meta = seq_meta.set_index("sequence_id")
    seq_meta = seq_meta.loc[sequence_ids]
    labels = torch.tensor(
        ko_matrix.loc[sequence_ids].values,
        dtype=torch.float32,
    )
    num_annotated = torch.tensor(
        seq_meta["num_genomes_annotated"].values,
        dtype=torch.float32,
    )
    sample_weights = _compute_sample_weights(num_annotated)
    splits = seq_meta["split"].values

    input_dim = features.shape[1]
    n_kos = labels.shape[1]
    hidden_dims = hidden_dims or [512, 256]

    device = _get_device()

    train_mask = splits == "train"
    val_mask = splits == "val"

    train_ds = FeaturesKODataset(
        features[train_mask],
        labels[train_mask],
        sample_weights[train_mask],
    )
    val_ds = FeaturesKODataset(
        features[val_mask],
        labels[val_mask],
        sample_weights[val_mask],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    head = KOHead(input_dim, n_kos, hidden_dims=hidden_dims).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            feat = batch["features"].to(device)
            lab = batch["labels"].to(device)
            weight = batch["sample_weight"].to(device)

            logits = head(feat)
            loss_per_element = criterion(logits, lab)
            loss_per_seq = loss_per_element.mean(dim=1)
            loss = (loss_per_seq * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(lab)
            n_train += len(lab)

        head.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(device)
                lab = batch["labels"].to(device)
                weight = batch["sample_weight"].to(device)

                logits = head(feat)
                loss_per_element = criterion(logits, lab)
                loss_per_seq = loss_per_element.mean(dim=1)
                loss = (loss_per_seq * weight).mean()

                val_loss += loss.item() * len(lab)
                n_val += len(lab)

        train_loss /= n_train
        val_loss /= n_val

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(head.state_dict(), output_dir / "head.pt")
            marker = " *"

        print(f"  Epoch {epoch + 1:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}{marker}")

    print(f"\n  Best val loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")

    config = {
        "input_dim": input_dim,
        "n_kos": n_kos,
        "hidden_dims": hidden_dims,
    }
    with open(output_dir / "head_config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "ko_columns.txt", "w") as f:
        f.write("\n".join(ko_matrix.columns))

    with open(output_dir / "model_info.json", "w") as f:
        json.dump({"model_name": model_name, "layer": layer}, f, indent=2)

    head.load_state_dict(torch.load(output_dir / "head.pt", weights_only=True))
    return head


def evaluate_head(
    features_path: Path,
    ko_matrix_path: Path,
    seq_meta_path: Path,
    head_path: Path,
) -> dict:
    """Evaluate head on test split. Print test BCE loss and mean Spearman."""
    data = torch.load(features_path, weights_only=True)
    features = data["features"]
    sequence_ids = data["sequence_ids"]

    seq_meta = pd.read_csv(seq_meta_path)
    ko_matrix = pd.read_csv(ko_matrix_path, index_col="sequence_id")

    seq_meta = seq_meta.set_index("sequence_id")
    seq_meta = seq_meta.loc[sequence_ids]
    labels = ko_matrix.loc[sequence_ids].values
    splits = seq_meta["split"].values

    test_mask = splits == "test"
    if not test_mask.any():
        print("  No test sequences — skipping evaluation.")
        return {}

    with open(head_path.parent / "head_config.json") as f:
        config = json.load(f)

    device = _get_device()
    head = KOHead(
        config["input_dim"],
        config["n_kos"],
        hidden_dims=config["hidden_dims"],
    )
    head.load_state_dict(torch.load(head_path, weights_only=True))
    head = head.to(device)
    head.eval()

    feat_test = features[test_mask].to(device)
    lab_test = torch.tensor(labels[test_mask], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = head(feat_test)
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        test_loss = criterion(logits, lab_test).item()

    preds = torch.sigmoid(logits).cpu().numpy()
    labels_arr = labels[test_mask]

    ko_columns = list(ko_matrix.columns)
    spearman_scores = {}
    for i, ko in enumerate(ko_columns):
        if labels_arr[:, i].std() > 0 and preds[:, i].std() > 0:
            rho, _ = spearmanr(labels_arr[:, i], preds[:, i])
            spearman_scores[ko] = rho

    mean_spearman = float(sum(spearman_scores.values()) / len(spearman_scores)) if spearman_scores else 0.0

    print(f"  Test BCE loss: {test_loss:.6f}")
    print(f"  Test Spearman (mean over {len(spearman_scores)} KOs): {mean_spearman:.4f}")

    return {
        "test_loss": test_loss,
        "mean_spearman": mean_spearman,
        "per_ko_spearman": spearman_scores,
    }
