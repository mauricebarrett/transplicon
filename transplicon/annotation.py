"""Summarise per-genome DeepKoala annotation CSVs into a long-format
genome-to-KO mapping."""

import csv
import sys
from pathlib import Path


def summarise_ko(deepkoala_dir: Path, output_path: Path) -> None:
    """Read per-genome DeepKoala CSVs and write a long-format summary.

    Keeps only confident annotations (``annotate == '*'``) and writes one
    row per (genome, KO) pair.

    Parameters
    ----------
    deepkoala_dir
        Directory containing one sub-directory per genome, each with a
        DeepKoala CSV result.
    output_path
        Destination CSV with columns: genome, ko.
    """
    if not deepkoala_dir.is_dir():
        print(f"Error: {deepkoala_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[str, str]] = []
    for genome_dir in sorted(deepkoala_dir.iterdir()):
        if not genome_dir.is_dir():
            continue

        accession = genome_dir.name

        csv_files = list(genome_dir.glob("*.csv"))
        if not csv_files:
            continue

        kos: set[str] = set()
        with open(csv_files[0]) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("annotate", "").strip() == "*":
                    kos.add(row["predict_label"])

        for ko in sorted(kos):
            rows.append((accession, ko))
        print(f"  {accession}: {len(kos)} confident KOs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["genome", "ko"])
        writer.writerows(rows)

    n_genomes = len({r[0] for r in rows})
    n_kos = len({r[1] for r in rows})
    print(f"\n  Wrote {len(rows)} rows ({n_genomes} genomes, {n_kos} unique KOs) to {output_path}")
