#!/bin/bash
set -euo pipefail

project_dir=${PROJECT_DIR:?PROJECT_DIR must be set}
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

nf_dir="$project_dir/nextflow"
mkdir -p "$nf_dir/logs"

cd "$nf_dir"

pixi run nextflow -log "$nf_dir/logs/nextflow.log" \
    run "$script_dir/annotate.nf" \
    -c "$repo_root/nextflow.config" \
    --project_dir "$project_dir" \
    -work-dir "$nf_dir/work" \
    -resume

echo "annotation_complete" > "$project_dir/data/annotation/.done"
