#!/usr/bin/env nextflow

params.project_dir      = "${projectDir}"
params.genome_dir       = "${params.project_dir}/data/gtdb/genomes"
params.pyrodigal_outdir = "${params.project_dir}/data/annotation/pyrodigal"
params.deepkoala_outdir = "${params.project_dir}/data/annotation/deepkoala"
params.deepkoala_path   = "/home/mossy/tools/deepkoala"
params.pyrodigal_cpus   = 4
params.deepkoala_cpus   = 8
params.deepkoala_batch  = 128
params.deepkoala_max_forks = 2

process PYRODIGAL {
    tag "${genome_name}"
    publishDir "${params.pyrodigal_outdir}/${genome_name}", mode: 'copy'
    cpus params.pyrodigal_cpus

    input:
    tuple val(genome_name), path(fasta)

    output:
    tuple val(genome_name), path("${genome_name}.faa"), path("${genome_name}.ffn"), path("${genome_name}.gff")

    script:
    """
    pyrodigal \
        -j ${task.cpus} \
        -i ${fasta} \
        -a ${genome_name}.faa \
        -d ${genome_name}.ffn \
        -o ${genome_name}.gff
    """
}

process DEEPKOALA {
    tag "${genome_name}"
    publishDir "${params.deepkoala_outdir}/${genome_name}", mode: 'copy'
    cpus params.deepkoala_cpus
    maxForks params.deepkoala_max_forks
    maxRetries 3
    errorStrategy { task.attempt <= maxRetries ? 'retry' : 'ignore' }

    input:
    tuple val(genome_name), path(faa), path(ffn), path(gff)

    output:
    tuple val(genome_name), path("${genome_name}.csv")

    script:
    """
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONPATH=${params.deepkoala_path} \
    ${params.deepkoala_path}/deepkoala_env/bin/python -m deepkoala.cli \
        -i ${faa} \
        -o ${genome_name}.csv \
        -m full \
        -bs ${params.deepkoala_batch} \
        -nw ${task.cpus} \
        --detail
    """
}

workflow {
    genomes = Channel
        .fromPath("${params.genome_dir}/*.fna")
        .map { fasta -> tuple(fasta.baseName, fasta) }

    PYRODIGAL(genomes)
    DEEPKOALA(PYRODIGAL.out)
}
