LEADER Pipeline
Base on this: /home/hiro/Downloads/Workspace/drug_rec/LEADER-pytorch
Do not wrap the code from other folder, must create code from scratch
This repository reproduces the LEADER medication recommendation framework and adapts it to run efficiently on GPU clusters (e.g., RTX 5060 Ti and NVIDIA H100 nodes).

The goal is to provide a reproducible pipeline including configuration management, training scripts, and cluster job submission.

Repository Structure
leader_pipeline/
│
├── configs/
│   data path, log output, model output
│   
│   
│
├── data/
│   
│
├── models/
│   └── leader.py        # LEADER model implementation
│
├── scripts/
│   ├── train.py         # Training entry point
│   ├── evaluate.py      # Evaluation pipeline
│   └── preprocess.py    # Data preprocessing
│
├── slurm/
│   └── job.slurm        # Cluster training script
│-- requirements.txt
└── README.md
Pipeline Overview

The pipeline contains three main stages.

1. Configuration

All paths and hyperparameters are defined in configuration files.

Example:

configs/data.yaml

mimic3_path: /mnt/data/mimiciii
mimic4_path: /mnt/data/mimiciv
output_dir: /mnt/data/leader_outputs

configs/model.yaml

model_name: LEADER
hidden_size: 256
num_layers: 2
dropout: 0.3

configs/train.yaml

batch_size: 32
learning_rate: 1e-4
epochs: 50
device: cuda
2. Training

Run locally:

python scripts/train.py \
    --data configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml
3. Evaluation
python scripts/evaluate.py

Metrics used:

Jaccard

F1 Score

PRAUC

DDI Rate

Average number of predicted medications

GPU Cluster Support

This repository supports execution on GPU clusters using SLURM.

Tested environments:

RTX 5060 Ti nodes

NVIDIA H100 80GB nodes

Running on Cluster (SLURM)

Create a job script:

slurm/job.slurm

#!/bin/bash

#SBATCH --job-name=leader_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

#SBATCH --partition=main
#SBATCH --time=24:00:00

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

#SBATCH --gres=gpu:1

echo "Start time: $(date)"
echo "Running on node: $(hostname)"

# Activate environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate leader

# Run training
python scripts/train.py

Submit the job:

sbatch slurm/job.slurm

Check job status:

squeue -u $USER

Cancel job:

scancel JOB_ID
Cluster Best Practices
Store large data on shared storage
/mnt/data/

Avoid storing large datasets in $HOME.

Debug with short jobs first

Before launching long jobs:

#SBATCH --time=00:10:00
Interactive GPU session (optional)
salloc --time=01:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=16G

Then run:

python scripts/train.py