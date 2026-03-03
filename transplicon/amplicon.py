"""Parse vsearch dereplication output into a CSV mapping unique 16S sequences
to their source genomes and taxonomies."""

import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
import mappy


_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
_PREFIX_RE = r"^[dpcofgs]__"


def load_taxonomy(tsv_path: str | Path) -> dict[str, str]:
    """Load taxonomy TSV (accession<TAB>taxonomy) into a dict."""
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["accession", "taxonomy"])
    return dict(zip(df["accession"], df["taxonomy"]))


def reformat_taxonomy(taxonomy_df: pd.DataFrame) -> pd.DataFrame:
    """Split GTDB taxonomy strings into separate rank columns.

    Expects a ``taxonomy`` column containing semicolon-delimited GTDB strings
    like ``d__Bacteria;p__Actinobacteriota;...;s__Streptomyces coelicolor``.

    Adds columns: domain, phylum, class, order, family, genus, species.
    Rank prefixes (``d__``, ``p__``, etc.) are stripped and missing/empty
    ranks are filled with ``unclassified``.
    """
    taxonomy_df[_RANKS] = taxonomy_df["taxonomy"].str.split(";", expand=True)

    for col in _RANKS:
        taxonomy_df[col] = (
            taxonomy_df[col]
            .str.strip()
            .str.replace(_PREFIX_RE, "", regex=True)
            .replace({"": "unclassified", "None": "unclassified", "Unassigned": "unclassified"})
            .fillna("unclassified")
        )

    return taxonomy_df


def parse_uc(uc_path: str | Path) -> dict[str, set[str]]:
    """Parse vsearch UC file and return {centroid_id: set of genome accessions}."""
    clusters: dict[str, set[str]] = defaultdict(set)

    with open(uc_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.strip().split("\t")
            record_type = fields[0]
            query_label = fields[8]

            if record_type == "S":
                centroid = query_label
                clusters[centroid].add(centroid)
            elif record_type == "H":
                centroid = fields[9]
                clusters[centroid].add(query_label)

    return clusters


def read_fasta(fasta_path: str | Path) -> dict[str, str]:
    """Read FASTA into {header: sequence} dict."""
    

    return {
        name.split(";size=")[0]: seq
        for name, seq, _ in mappy.fastx_read(str(fasta_path))
    }


def _resolve_taxonomy(genome_accessions: set[str], taxonomy: dict[str, str]) -> dict[str, str]:
    """Resolve per-rank taxonomy for a cluster of genomes.

    For each rank, collects the unique values across all genomes in the cluster.
    Multiple values are joined with ``|``.
    """
    tax_strings = [taxonomy[g] for g in genome_accessions if g in taxonomy]
    if not tax_strings:
        return {rank: "unclassified" for rank in _RANKS}

    tax_df = reformat_taxonomy(pd.DataFrame({"taxonomy": tax_strings}))

    result = {}
    for rank in _RANKS:
        unique_vals = sorted(tax_df[rank].unique())
        result[rank] = "|".join(unique_vals)
    return result


def parse_derep(
    uc_path: Path,
    fasta_path: Path,
    taxonomy_path: Path,
    output_path: Path,
) -> None:
    """Parse vsearch dereplication outputs and write a long-format CSV.

    Writes one row per (sequence, genome) pair. Taxonomy is resolved
    per cluster (unique values joined with ``|`` when genera disagree)
    and repeated on every row for that cluster.

    Parameters
    ----------
    uc_path
        UC cluster file produced by ``vsearch --derep_fulllength``.
    fasta_path
        Dereplicated FASTA from vsearch.
    taxonomy_path
        TSV mapping genome accession to GTDB taxonomy string.
    output_path
        Long-format CSV with columns: sequence, genome, num_genomes,
        domain, phylum, class, order, family, genus, species.
    """
    clusters = parse_uc(uc_path)
    sequences = read_fasta(fasta_path)
    taxonomy = load_taxonomy(taxonomy_path)

    n_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "genome", "num_genomes"] + _RANKS)

        for centroid, genomes in sorted(clusters.items(), key=lambda x: -len(x[1])):
            seq = sequences.get(centroid, "")
            ranks = _resolve_taxonomy(genomes, taxonomy)
            rank_vals = [ranks[r] for r in _RANKS]
            for genome in sorted(genomes):
                writer.writerow([seq, genome, len(genomes), *rank_vals])
                n_rows += 1

    print(f"  Wrote {n_rows} rows ({len(clusters)} sequences, {n_rows - len(clusters)} duplicate genomes) to {output_path}")
