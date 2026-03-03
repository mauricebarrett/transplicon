#!/bin/bash
set -euo pipefail

project_dir=${PROJECT_DIR:?PROJECT_DIR must be set}
taxa=${TAXA:?TAXA must be set}
taxa_lower=${taxa,,}
data_dir=$project_dir/data
gtdb_dir=$data_dir/gtdb
accessions_dir=$gtdb_dir/accessions
amplicon_dir=$data_dir/amplicon

mkdir -p "$amplicon_dir"

wget -P "$amplicon_dir" -nc https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/genomic_files_all/ssu_all.fna.gz
gunzip -kf "$amplicon_dir/ssu_all.fna.gz"

seqkit seq \
	-j 14 \
	-m 1100 \
	-M 1800 \
	-g \
	-o "$amplicon_dir/ssu_all_filtered.fna" \
	"$amplicon_dir/ssu_all.fna"

seqkit seq -w 0 "$amplicon_dir/ssu_all_filtered.fna" > "$amplicon_dir/ssu_all_filtered_nowrap.fna"

awk '{print $1}' "$accessions_dir/gtdb_rep_metadata.tsv" | \
	tail -n +2 \
	> "$amplicon_dir/${taxa_lower}_genome_accessions.txt"

rg -A 1 -f "$amplicon_dir/${taxa_lower}_genome_accessions.txt" \
    "$amplicon_dir/ssu_all_filtered_nowrap.fna" \
    | rg -v '^--$' \
    > "$amplicon_dir/${taxa_lower}_16s_sequences_raw.fna"

seqkit replace -p '^(RS_|GB_)(GC[AF]_[0-9]+\.[0-9]+)~\S*(.*)' -r '${2}${3}' \
    "$amplicon_dir/${taxa_lower}_16s_sequences_raw.fna" \
    > "$amplicon_dir/${taxa_lower}_16s_sequences.fna"

rg '^>' "$amplicon_dir/${taxa_lower}_16s_sequences.fna" |
    tr -d '>' |
    sed 's/ d__/\td__/' |
    sed 's/\[.*//' |
    sort -u \
    > "$amplicon_dir/${taxa_lower}_16s_taxonomy.tsv"
