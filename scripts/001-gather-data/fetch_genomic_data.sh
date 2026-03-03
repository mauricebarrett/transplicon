#!/bin/bash
set -euo pipefail

project_dir=${PROJECT_DIR:?PROJECT_DIR must be set}
taxa=${TAXA:?TAXA must be set}
genomes_per_species=${GENOMES_PER_SPECIES:-1}
gtdb_cache_dir=${GTDB_METADATA:?GTDB_METADATA must be set}
data_dir=$project_dir/data
accessions_dir=$data_dir/gtdb/accessions
genome_dir=$data_dir/gtdb/genomes
script_dir=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$accessions_dir" "$genome_dir"

# Sample GTDB genomes: download metadata, filter by taxon, cap per species
pixi run python "$script_dir/sample_gtdb_genomes.py" \
    --taxon "$taxa" \
    --output-dir "$accessions_dir" \
    --cache-dir "$gtdb_cache_dir" \
    --genomes-per-species "$genomes_per_species"

# Download genomes via NCBI datasets in batches to avoid connection drops
cd "$genome_dir"
batch_size=200
accession_file="$accessions_dir/gtdb_rep_accessions.txt"
total=$(wc -l < "$accession_file")
batch_dir="$genome_dir/_batches"
mkdir -p "$batch_dir"

split -l "$batch_size" -d -a 4 "$accession_file" "$batch_dir/batch_"

for batch_file in "$batch_dir"/batch_*; do
    batch_name=$(basename "$batch_file")
    batch_count=$(wc -l < "$batch_file")
    echo "Downloading batch $batch_name ($batch_count accessions)..."

    for attempt in 1 2 3; do
        if pixi run datasets download genome accession \
            --inputfile "$batch_file" \
            --dehydrated \
            --filename "$genome_dir/${batch_name}.zip"; then
            break
        fi
        echo "  Download attempt $attempt failed, retrying in 10s..."
        sleep 10
    done

    unzip -o "$genome_dir/${batch_name}.zip" -d "$genome_dir/${batch_name}_data"

    for attempt in 1 2 3; do
        if pixi run datasets rehydrate --directory "$genome_dir/${batch_name}_data"; then
            break
        fi
        echo "  Rehydrate attempt $attempt failed, retrying in 10s..."
        sleep 10
    done

    mv "$genome_dir/${batch_name}_data"/ncbi_dataset/data/GC*/GC* "$genome_dir"/ 2>/dev/null || true
    rm -rf "$genome_dir/${batch_name}_data" "$genome_dir/${batch_name}.zip"
done

rm -rf "$batch_dir"
echo "Finished downloading $total genomes in batches of $batch_size"

# Normalise filenames to bare accessions (GCA_000001234.1.fna)
for fasta in "$genome_dir"/*.fna; do
    file_name=$(basename "$fasta")
    if [[ "$file_name" =~ (GC[AF]_[0-9]+\.[0-9]+) ]]; then
        cleaned="${BASH_REMATCH[1]}"
        mv "$fasta" "$genome_dir/$cleaned.fna"
    fi
done

# Rename genomes from GCA/GCF to GTDB accession (first column, RS_/GB_ stripped)
# so that genome names match the 16S amplicon accessions from GTDB SSU data.
metadata_tsv="$accessions_dir/gtdb_rep_metadata.tsv"
if [[ -f "$metadata_tsv" ]]; then
    renamed=0
    while read -r old_name new_name; do
        if [[ -f "$genome_dir/$old_name.fna" ]]; then
            mv "$genome_dir/$old_name.fna" "$genome_dir/$new_name.fna"
            ((renamed++)) || true
        fi
    done < <(awk -F'\t' '
        NR == 1 {
            for (i = 1; i <= NF; i++)
                if ($i == "ncbi_genbank_assembly_accession") gca_col = i
            next
        }
        {
            acc = $1; sub(/^RS_/, "", acc); sub(/^GB_/, "", acc)
            gca = $gca_col
            if (gca != "" && gca != "none" && gca != acc)
                print gca, acc
        }
    ' "$metadata_tsv")
    echo "Renamed $renamed genomes from GenBank to GTDB accession format"
fi

printf '%s\n' "$genome_dir"/*.fna > "$genome_dir/genome_manifest.txt"
echo "Downloaded $(wc -l < "$genome_dir/genome_manifest.txt") genomes"
