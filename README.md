# transplicon

## Introduction
Transplicon is designed for functional inference from amplicon data using genomic transformer models. The name is derived from **trans**former and am**plicon**.

Currently the tool is in alpha and only supports Actinomycetes.

Furthermore it only supports 16S amplicon data and the prediction of KEGG Orthology (KO) numbers.

The tool plans to expand to more taxa, more amplicon data types, and more functional inference targets.

Functional inference
- KEGG Orthology (KO)
- eggNOG
- KEGG Pathways
- Metacyc Pathways

Amplicon data
- 16S
- 18S–ITS–28S operon




## Installation

### Prerequisites

- **Linux** (currently the only supported platform)
- **NVIDIA GPU** with CUDA 12.0+ (required for model inference)
- **[pixi](https://pixi.sh)** package manager

### Setup

Clone the repository and install all dependencies:

```bash
git clone https://github.com/mauricebarrett/transplicon.git
cd transplicon
pixi install
```

This will set up a reproducible environment with all required dependencies including PyTorch (GPU), transformers, and bioinformatics tools (VSEARCH, pyrodigal, Nextflow, etc.).

## Usage

### Predicting KO profiles from amplicon sequences

Once you have a trained model, use the `transplicon` CLI to predict KO profiles from a FASTA file:

```bash
pixi run transplicon -i sequences.fasta -m /path/to/model_dir -o predictions.csv
```

| Argument | Description |
|---|---|
| `-i`, `--input` | FASTA file of query sequences |
| `-m`, `--model-dir` | Directory containing trained model artifacts (`head.pt`, `head_config.json`, `ko_columns.txt`, `model_info.json`) |
| `-o`, `--output` | Output CSV path (default: stdout) |
| `--batch-size` | Batch size for inference (default: 64) |

### Python API

You can also use Transplicon directly in Python:

```python
from transplicon.predict import load_model, predict_fasta

model = load_model("/path/to/model_dir")
df = predict_fasta(model, "sequences.fasta")
print(df.head())
```

The output is a DataFrame with sequence IDs as rows and KO numbers as columns, with values representing predicted probabilities.




## How it works
Transplicon is based on a multilayer perceptron (MLP) that predicts KO numbers from amplicon data. The MLP is trained on features extracted by passing amplicon sequences through a pre-trained genomic language model (GLM).

## Data Preparation
Central to data preparation is paired genomic and amplicon data. Since the tool is currently in alpha, data is only trained on a chosen taxa (Actinomycetes).

Genomic data is downloaded from NCBI RefSeq based on the GTDB metadata. These genomes are annotated using pyrodigal to identify ORFs. Deepkoala is used to annotate these ORFs with KO numbers.

16S amplicon data is gathered from the GTDB SSU rRNA data (ssu_all.fna.gz). Amplicon data is dereplicated (100% identity clustering) using VSEARCH. The genomes from which the dereplicated cluster of sequences was derived from is recorded.

The probability of a sequence being assigned to a KO is calculated based on the frequency of the KO in the genomes from which the sequence was derived.

## Feature Extraction
Features are extracted by passing the amplicon sequences through a pre-trained genomic language model (GLM).

First the amplicon sequences are tokenized using the GLM tokenizer. The tokenized sequences are embedded using the GLM. The embeddings are then extracted from the specified layer. These embeddings are then used as features for the MLP.


## Training
The MLP is trained on the features extracted from the amplicon sequences and the KO probability data.



## Running model training

```bash
pixi run python main.py \
    -d /path/to/project_dir \
    -g /path/to/gtdb_metadata \
    -m nucleotide-transformer-v2-250m-multi-species \
    -t Actinomycetota \
    -n 4 \
    -p 10
```

| Argument | Description |
|---|---|
| `-d`, `--project-dir` | Root project data directory |
| `-g`, `--gtdb-metadata` | Directory containing GTDB metadata TSVs (`bac120_metadata.tsv.gz` / `ar53_metadata.tsv.gz`) |
| `-m`, `--model` | GLM model name (e.g. `nucleotide-transformer-v2-250m-multi-species`, `DNABERT-2-117M`, `NTv3-650M-pre`) |
| `-t`, `--taxa` | GTDB taxon name to train on |
| `-l`, `--layer` | Which layer to extract features from: integer, `last`, or `bottleneck` (default: `last`) |
| `-n`, `--genomes-per-species` | Max genomes per species (default: 1) |
| `-p`, `--min-ko-prevalence` | Drop KOs present in fewer than this % of sequences |



