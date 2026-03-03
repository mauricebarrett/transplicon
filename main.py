#!/usr/bin/env python3
"""Transplicon pipeline orchestrator.

Usage:
    pixi run python main.py -d /home/mossy/projects/transplicon -g /path/to/gtdb/metadata
"""


import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

from transplicon.amplicon import parse_derep
from transplicon.annotation import summarise_ko
from transplicon.feature_extraction import run_feature_extraction
from transplicon.head import evaluate_head, train_head
from transplicon.prepare_input import assign_splits, build_training_tables, filter_ko_matrix

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transplicon pipeline orchestrator")
    parser.add_argument("-d", "--project-dir", type=Path, required=True,
                        help="Root project data directory")
    parser.add_argument("-t", "--taxa", type=str, required=True,
                        help="GTDB taxon name")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model name: DNABERT-2-117M, nucleotide-transformer-v2-50m-multi-species, "
                        "nucleotide-transformer-v2-250m-multi-species, NTv3-100M-pre, "
                        "NTv3-650M-pre, NTv3-650M-post")
    parser.add_argument("-l", "--layer", type=str, default="last",
                        help="Which layer to extract. Can be an integer, 'last' for final layer, "
                        "or 'bottleneck' for U-Net models (default: last)")
    parser.add_argument("-n", "--genomes-per-species", type=int, default=1,
                        help="Max genomes per species; representative always included, "
                        "extras filled by quality (default: 1)")
    parser.add_argument("-g", "--gtdb-metadata", type=Path, required=True,
                        help="Directory containing GTDB metadata TSVs "
                        "(bac120_metadata.tsv.gz / ar53_metadata.tsv.gz); "
                        "files are downloaded here if not already present")
    parser.add_argument("-p", "--min-ko-prevalence", type=float, required=True,
                        help="Drop KOs present in fewer than this %% of sequences")
    return parser.parse_args()


def run_script(script: Path, extra_args: list[str] | None = None) -> None:
    if script.suffix == ".py":
        cmd = [sys.executable, str(script)]
    else:
        cmd = ["bash", str(script)]

    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: {script.name} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir
    taxa = args.taxa
    taxa_lower = taxa.lower()
    model_name = args.model

    os.environ["PROJECT_DIR"] = str(project_dir)
    os.environ["TAXA"] = taxa
    os.environ["GENOMES_PER_SPECIES"] = str(args.genomes_per_species)
    os.environ["GTDB_METADATA"] = str(args.gtdb_metadata)
    data_dir = project_dir / "data"

    # Step 1a: Gather genomic data
    if (data_dir / "gtdb" / "genomes" / "genome_manifest.txt").exists():
        print("Skipping gather-genomic-data: outputs already exist")
    else:
        print("Step 1a: Gathering genomic data...")
        run_script(SCRIPTS_DIR / "001-gather-data" / "fetch_genomic_data.sh")
        print("Step 1a complete: genomic data gathered")

    # Step 1b: Gather amplicon data
    amplicon_dir = data_dir / "amplicon"
    if (amplicon_dir / f"{taxa_lower}_16s_sequences.fna").exists():
        print("Skipping gather-amplicon-data: outputs already exist")
    else:
        print("Step 1b: Gathering amplicon data...")
        run_script(SCRIPTS_DIR / "001-gather-data" / "fetch_amplicon_data.sh")
        print("Step 1b complete: amplicon data gathered")

    # Step 2: Prepare data (annotate, summarise KOs, dereplicate 16S)
    if (data_dir / "annotation" / "genome_ko_summary.csv").exists() \
       and (amplicon_dir / f"{taxa_lower}_16s_dereplicated.csv").exists():
        print("Skipping prepare-data: outputs already exist")
    else:
        print("Step 2: Preparing data...")
        step_dir = SCRIPTS_DIR / "002-prepare-data"
        run_script(step_dir / "run_annotation.sh")
        summarise_ko(
            deepkoala_dir=data_dir / "annotation" / "deepkoala",
            output_path=data_dir / "annotation" / "genome_ko_summary.csv",
        )
        run_script(step_dir / "dereplicate_16s_data.sh")

        parse_derep(
            uc_path=amplicon_dir / f"{taxa_lower}_16s_derep.uc",
            fasta_path=amplicon_dir / f"{taxa_lower}_16s_derep.fna",
            taxonomy_path=amplicon_dir / f"{taxa_lower}_16s_taxonomy.tsv",
            output_path=amplicon_dir / f"{taxa_lower}_16s_dereplicated.csv",
        )
        print("Step 2 complete: data prepared")

    # Step 3: Build training data
    output_dir = data_dir / "training"
    if (output_dir / "sequence_metadata.csv").exists() \
       and (output_dir / "ko_matrix.csv").exists():
        print("Skipping build-training-data: outputs already exist")
    else:
        print("Step 3: Building training data...")
        ko_summary = data_dir / "annotation" / "genome_ko_summary.csv"
        derep_csv = amplicon_dir / f"{taxa_lower}_16s_dereplicated.csv"

        for f in (ko_summary, derep_csv):
            if not f.exists():
                print(f"Error: required input {f} not found — earlier step may have failed", file=sys.stderr)
                sys.exit(1)

        seq_meta, ko_matrix = build_training_tables(ko_summary, derep_csv, output_dir)

        print("Filtering ...")
        seq_meta, ko_matrix = filter_ko_matrix(seq_meta, ko_matrix, min_ko_prevalence_pct=args.min_ko_prevalence)

        print("Splitting by genus ...")
        seq_meta = assign_splits(seq_meta)

        seq_meta.to_csv(output_dir / "sequence_metadata.csv", index=False)
        ko_matrix.to_csv(output_dir / "ko_matrix.csv")
        print("Step 3 complete: training data built")

    # Step 4: Feature extraction
    features_dir = output_dir / "features"
    if (features_dir / "features.pt").exists():
        print("Skipping feature-extraction: outputs already exist")
    else:
        print("Step 4: Feature extraction...")

        seq_meta = pd.read_csv(output_dir / "sequence_metadata.csv")
        run_feature_extraction(
            seq_meta=seq_meta,
            output_dir=features_dir,
            model_name=model_name,
            layer=args.layer,
        )
        print("Step 4 complete: features extracted")

    # Step 5: Train head
    head_dir = output_dir / "head"
    if (head_dir / "head.pt").exists():
        print("Skipping train-head: checkpoint already exists")
    else:
        print("Step 5: Training head...")
        train_head(
            features_path=features_dir / "features.pt",
            ko_matrix_path=output_dir / "ko_matrix.csv",
            seq_meta_path=output_dir / "sequence_metadata.csv",
            output_dir=head_dir,
            model_name=model_name,
            layer=args.layer,
        )
        print("Step 5 complete: head trained")

    if (head_dir / "head.pt").exists():
        print("\nEvaluating on test set ...")
        evaluate_head(
            features_path=features_dir / "features.pt",
            ko_matrix_path=output_dir / "ko_matrix.csv",
            seq_meta_path=output_dir / "sequence_metadata.csv",
            head_path=head_dir / "head.pt",
        )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
