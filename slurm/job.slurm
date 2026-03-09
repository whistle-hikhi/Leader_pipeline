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
echo "Job ID: $SLURM_JOB_ID"

# Activate conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate leader

# Move to project root
cd $HOME/Downloads/Workspace/Leader_pipeline

# Create log directory if it doesn't exist
mkdir -p logs

# Optional: run preprocessing first if processed data doesn't exist
# python scripts/preprocess.py --data configs/data.yaml

# Run training
python scripts/train.py \
    --data  configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml

echo "End time: $(date)"
