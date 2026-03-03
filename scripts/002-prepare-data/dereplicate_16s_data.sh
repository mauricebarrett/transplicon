#!/bin/bash
set -euo pipefail

project_dir=${PROJECT_DIR:?PROJECT_DIR must be set}
taxa=${TAXA:?TAXA must be set}
taxa_lower=${taxa,,}
data_dir=$project_dir/data
amplicon_dir=$data_dir/amplicon
input_fasta=$amplicon_dir/${taxa_lower}_16s_sequences.fna

derep_fasta=$amplicon_dir/${taxa_lower}_16s_derep.fna
derep_uc=$amplicon_dir/${taxa_lower}_16s_derep.uc

pixi run vsearch \
    --derep_fulllength "$input_fasta" \
    --output "$derep_fasta" \
    --uc "$derep_uc" \
    --sizeout
