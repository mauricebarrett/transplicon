#!/usr/bin/env python3
"""Download GTDB metadata, filter by taxon, and sample N genomes per species.

Replaces gtt-get-accessions-from-GTDB with direct GTDB metadata access and
per-species sampling (by genome quality when available).
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

GTDB_METADATA_URLS = {
    "bacteria": "https://data.gtdb.ecogenomic.org/releases/latest/bac120_metadata.tsv.gz",
    "archaea": "https://data.gtdb.ecogenomic.org/releases/latest/ar53_metadata.tsv.gz",
}

RANK_PREFIXES = ("d__", "p__", "c__", "o__", "f__", "g__", "s__")


def download_metadata(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Using cached metadata: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GTDB metadata from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def load_metadata(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, sep="\t", compression=compression, low_memory=False)


def filter_by_taxon(df: pd.DataFrame, taxon: str) -> pd.DataFrame:
    """Filter to rows whose gtdb_taxonomy contains the taxon.

    Tries the raw string first, then prepends each rank prefix (d__ through s__)
    until a match is found.
    """
    col = df["gtdb_taxonomy"]
    mask = col.str.contains(taxon, case=False, na=False)
    if mask.sum() == 0:
        for prefix in RANK_PREFIXES:
            mask = col.str.contains(f"{prefix}{taxon}", case=False, na=False)
            if mask.sum() > 0:
                break
    filtered = df[mask]
    print(f"Found {len(filtered)} genomes matching '{taxon}'")
    return filtered


def _species_from_taxonomy(taxonomy: str) -> str:
    for part in taxonomy.split(";"):
        part = part.strip()
        if part.startswith("s__"):
            return part
    return "s__unknown"


def sample_per_species(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Select up to *n* genomes per species.

    The GTDB representative genome for each species is always included.
    When n > 1, the remaining slots are filled with the highest-quality
    non-representative genomes (ranked by completeness − 5 × contamination).
    """
    has_rep_col = "gtdb_representative" in df.columns
    has_quality = {"checkm_completeness", "checkm_contamination"} <= set(df.columns)

    df = df.copy()
    df["_species"] = df["gtdb_taxonomy"].apply(_species_from_taxonomy)

    if has_quality:
        df["_quality"] = df["checkm_completeness"] - 5 * df["checkm_contamination"]

    # Always take the representative genome for every species
    if has_rep_col:
        reps = df[df["gtdb_representative"] == "t"]
    else:
        reps = df.drop_duplicates(subset="_species", keep="first")

    if n <= 1:
        sampled = reps
    else:
        # Fill remaining slots with top non-representative genomes by quality
        non_reps = df[~df.index.isin(reps.index)]
        if has_quality:
            non_reps = non_reps.sort_values("_quality", ascending=False)
        extras = non_reps.groupby("_species", sort=False).head(n - 1)
        sampled = pd.concat([reps, extras]).sort_index().reset_index(drop=True)

    n_species = sampled["_species"].nunique()
    print(f"Selected {len(sampled)} genomes from {n_species} species (≤{n} per species)")
    sampled = sampled.drop(columns=["_species", "_quality"], errors="ignore")
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample GTDB genomes per species",
    )
    parser.add_argument("--taxon", required=True)
    parser.add_argument(
        "--genomes-per-species", type=int, default=1,
        help="Max genomes per species; the GTDB representative is always "
             "included, extra slots filled by quality (default: 1)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Cache dir for raw GTDB metadata (default: same as output-dir)",
    )
    parser.add_argument(
        "--domain", choices=["bacteria", "archaea", "both"], default="bacteria",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    domains = ["bacteria", "archaea"] if args.domain == "both" else [args.domain]
    frames = []
    for domain in domains:
        url = GTDB_METADATA_URLS[domain]
        cache_path = cache_dir / url.rsplit("/", 1)[-1]
        download_metadata(url, cache_path)
        frames.append(load_metadata(cache_path))
    meta = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    meta = filter_by_taxon(meta, args.taxon)
    if meta.empty:
        print(f"Error: no genomes found for taxon '{args.taxon}'", file=sys.stderr)
        sys.exit(1)

    meta = sample_per_species(meta, args.genomes_per_species)

    # Write accession list (GCA/GCF ids for NCBI datasets download)
    acc_col = "ncbi_genbank_assembly_accession"
    accessions = meta[acc_col].dropna()
    accessions = accessions[accessions != "none"]

    accessions_path = args.output_dir / "gtdb_rep_accessions.txt"
    accessions.to_csv(accessions_path, index=False, header=False)

    metadata_path = args.output_dir / "gtdb_rep_metadata.tsv"
    meta.to_csv(metadata_path, sep="\t", index=False)

    print(f"Wrote {len(accessions)} accessions → {accessions_path}")
    print(f"Wrote metadata → {metadata_path}")


if __name__ == "__main__":
    main()
