"""Build normalised training tables by joining dereplicated 16S sequences
with genome KO annotations.

Inputs
------
genome_ko_summary.csv
    Long-format: one row per (genome, KO) pair.
    Columns: genome, ko

streptomycetales_16s_dereplicated.csv
    Long-format: one row per (sequence, genome) pair.
    Columns: sequence, genome, num_genomes,
             domain, phylum, class, order, family, genus, species

Outputs (written to ``output_dir``)
-----------------------------------
sequence_metadata.csv
    One row per unique dereplicated 16S sequence.
    Columns: sequence_id, sequence, domain, phylum, class, order, family,
             genus, species, num_genomes, num_genomes_annotated, num_kos, split

ko_matrix.csv
    Wide-format table – one row per sequence, one column per KO.
    Values are KO probabilities (0.0 where absent).
"""

from pathlib import Path
import random

import pandas as pd

META_COLS = [
    "sequence_id", "sequence",
    "domain", "phylum", "class", "order", "family", "genus", "species",
    "num_genomes", "num_genomes_annotated", "num_kos",
]


def build_training_tables(
    ko_summary_path: Path,
    dereplicated_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join dereplicated 16S sequences with KO annotations and write outputs.

    Parameters
    ----------
    ko_summary_path
        Long-format CSV with columns: genome, ko.
    dereplicated_path
        Long-format dereplicated 16S CSV with columns: sequence, genome,
        num_genomes, domain, phylum, class, order, family, genus, species.
    output_dir
        Directory to write sequence_metadata.csv and ko_matrix.csv.

    Returns
    -------
    seq_meta : pd.DataFrame
        Sequence metadata table.
    ko_matrix : pd.DataFrame
        Wide-format KO probability matrix indexed by sequence_id.
    """
    ko_df = pd.read_csv(ko_summary_path)
    derep_df = pd.read_csv(dereplicated_path)

    n_sequences = derep_df["sequence"].nunique()
    print("Loading data ...")
    print(f"  {ko_df['genome'].nunique()} genomes, {ko_df['ko'].nunique()} unique KOs")
    print(f"  {n_sequences} dereplicated 16S sequences\n")

    merged = derep_df.merge(ko_df, on="genome", how="inner")

    seq_genome_counts = (
        merged.groupby("sequence")["genome"]
        .nunique()
        .rename("num_genomes_annotated")
    )

    ko_counts = (
        merged.groupby(["sequence", "ko"])["genome"]
        .nunique()
        .rename("num_genomes_with_ko")
        .reset_index()
    )

    ko_counts = ko_counts.merge(seq_genome_counts, on="sequence")
    ko_counts["probability"] = (ko_counts["num_genomes_with_ko"] / ko_counts["num_genomes_annotated"]).round(6)

    # Collapse to one row per sequence for metadata
    seq_info = derep_df.drop(columns=["genome"]).drop_duplicates(subset="sequence")
    seq_meta = seq_info[seq_info["sequence"].isin(seq_genome_counts.index)].copy()
    seq_meta = seq_meta.merge(seq_genome_counts, on="sequence")
    seq_meta["num_kos"] = seq_meta["sequence"].map(
        ko_counts.groupby("sequence")["ko"].nunique()
    )

    seq_meta = seq_meta.sort_values("num_genomes", ascending=False).reset_index(drop=True)
    width = len(str(len(seq_meta)))
    seq_meta.insert(0, "sequence_id", [f"seq_{i:0{width}d}" for i in range(len(seq_meta))])

    dropped = n_sequences - len(seq_meta)
    print(f"Building tables ...")
    print(f"  {len(seq_meta)} sequences retained ({dropped} dropped — no annotated genomes)")
    print(f"  {ko_counts['ko'].nunique()} unique KOs")
    print(f"  {len(ko_counts)} (sequence, KO) label rows\n")

    # Map sequence IDs into ko_counts and pivot to wide KO matrix
    seq_id_map = dict(zip(seq_meta["sequence"], seq_meta["sequence_id"]))
    ko_counts["sequence_id"] = ko_counts["sequence"].map(seq_id_map)

    ko_matrix = (
        ko_counts.pivot(index="sequence_id", columns="ko", values="probability")
        .fillna(0.0)
        .reindex(seq_meta["sequence_id"])
    )
    ko_matrix.columns.name = None
    ko_matrix.index.name = "sequence_id"

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "sequence_metadata.csv"
    seq_meta[META_COLS].to_csv(meta_path, index=False)
    print(f"Saving ...")
    print(f"  {meta_path}  ({len(seq_meta)} sequences)")

    matrix_path = output_dir / "ko_matrix.csv"
    ko_matrix.to_csv(matrix_path)
    print(f"  {matrix_path}  ({ko_matrix.shape[0]} rows x {ko_matrix.shape[1]} KOs)")

    return seq_meta[META_COLS], ko_matrix


def filter_ko_matrix(
    seq_meta: pd.DataFrame,
    ko_matrix: pd.DataFrame,
    min_ko_prevalence_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rare KOs based on percentage prevalence across sequences.

    Parameters
    ----------
    seq_meta
        Sequence metadata DataFrame.
    ko_matrix
        Wide-format KO probability matrix indexed by sequence_id.
    min_ko_prevalence_pct
        Drop KOs present in fewer than this percentage of sequences (0–100).

    Returns
    -------
    seq_meta : pd.DataFrame
        Filtered metadata with updated ``num_kos``.
    ko_matrix : pd.DataFrame
        Filtered KO matrix.
    """
    n_kos_before = ko_matrix.shape[1]
    n_seqs = ko_matrix.shape[0]

    ko_prevalence_pct = (ko_matrix > 0).sum(axis=0) / n_seqs * 100
    kept_kos = ko_prevalence_pct[ko_prevalence_pct >= min_ko_prevalence_pct].index
    ko_matrix = ko_matrix[kept_kos]

    seq_meta = seq_meta.copy()
    kos_per_seq = (ko_matrix > 0).sum(axis=1)
    seq_meta["num_kos"] = seq_meta["sequence_id"].map(kos_per_seq)

    print(f"  KOs: {n_kos_before} -> {ko_matrix.shape[1]} (dropped {n_kos_before - ko_matrix.shape[1]} with prevalence < {min_ko_prevalence_pct:.1f}%)")

    return seq_meta, ko_matrix


def assign_splits(
    seq_meta: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test splits by genus.

    Entire genera are kept together. Genera are ordered by size (largest first),
    then assigned to splits while preserving target split proportions by number
    of genera.

    Parameters
    ----------
    seq_meta
        Sequence metadata DataFrame with a ``genus`` column.
    val_fraction
        Fraction of genera for validation.
    test_fraction
        Fraction of genera for test.
    seed
        Random seed for reproducible splitting.

    Returns
    -------
    pd.DataFrame
        Copy of *seq_meta* with an added ``split`` column.
    """
    seq_meta = seq_meta.copy()

    genera = seq_meta["genus"].unique().tolist()
    genus_sizes = seq_meta.groupby("genus").size().to_dict()
    rng = random.Random(seed)
    # Shuffle first so equal-sized genera are ordered reproducibly by seed.
    rng.shuffle(genera)
    genera.sort(key=lambda g: genus_sizes[g], reverse=True)

    train_fraction = 1.0 - val_fraction - test_fraction
    fractions = {"train": train_fraction, "test": test_fraction, "val": val_fraction}
    assigned = {"train": 0, "test": 0, "val": 0}

    split_map: dict[str, str] = {}
    # Priority on ties: train first, then val, then test.
    tie_break = {"train": 2, "val": 1, "test": 0}
    for g in genera:
        # Assign next largest genus to the split that's most under target ratio.
        # Minimizing assigned/fraction keeps long-run proportions near 8:1:1.
        next_split = min(
            ("train", "val", "test"),
            key=lambda s: (
                assigned[s] / fractions[s] if fractions[s] > 0 else float("inf"),
                -tie_break[s],
            ),
        )
        split_map[g] = next_split
        assigned[next_split] += 1

    seq_meta["split"] = seq_meta["genus"].map(split_map)

    for s in ("train", "val", "test"):
        n = (seq_meta["split"] == s).sum()
        print(f"  {s}: {n} sequences")

    return seq_meta
