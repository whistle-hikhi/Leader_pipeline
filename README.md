# LEADER Pipeline

Reproducible implementation of the LEADER medication recommendation framework, adapted for GPU clusters (RTX 5060 Ti, NVIDIA H100).

## Repository Structure

```
Leader_pipeline/
├── configs/
│   ├── data.yaml         # Data paths (MIMIC-III/IV, mapping files, output dir)
│   ├── model.yaml        # Model hyperparameters
│   └── train.yaml        # Training hyperparameters
├── data/                 # Processed data written here by preprocess.py
├── llm/
│   ├── llama.py          # LlamaForMedRec — LLaMA backbone with medication cls head
│   └── lora_cls.py       # LoRA helpers: create / save / load (includes cls_head)
├── models/
│   └── leader.py         # LEADER student model
├── resources/
│   └── llama-7b/         # Place downloaded LLaMA-7B weights here
├── scripts/
│   ├── preprocess.py     # Raw MIMIC -> processed JSONL + vocab + DDI matrices
│   ├── finetune_llm.py   # Fine-tune LLaMA teacher with LoRA
│   ├── train.py          # Train LEADER student (with optional KD from LLM)
│   └── evaluate.py       # Evaluation pipeline
├── slurm/
│   └── job.slurm         # SLURM cluster job script
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `rdkit`, `dill`, `jsonlines`, `tqdm`, `PyYAML`.

LLM teacher dependencies (only needed when `distill: true`): `transformers`, `peft`, `accelerate`.

## Configuration

All paths and hyperparameters are controlled through three YAML files.

**`configs/data.yaml`** — set your data paths:

```yaml
dataset: mimic3            # "mimic3" or "mimic4"
mimic3_path: /path/to/mimiciii/1.4
mimic4_path: /path/to/mimiciv/3.1/hosp
ndc2rxcui_path: /path/to/ndc2RXCUI.txt
rxcui2atc4_path: /path/to/RXCUI2atc4.csv
drug_atc_path:   /path/to/drug-atc.csv
drugbank_info_path: /path/to/drugbank_drugs_info.csv
ddi_file_path:   /path/to/drug-DDI.csv
output_dir: /path/to/Leader_pipeline/data
log_dir:    /path/to/Leader_pipeline/logs
```

**`configs/model.yaml`** — key defaults:

```yaml
hidden_size: 64
num_trm_layers: 1
num_attention_heads: 4
profile: false      # set true to use patient demographics as prompts
distill: false      # set true to enable LLM knowledge distillation
align: false        # set true to enable profile-medication alignment loss

# LLM teacher (only used when distill: true)
llm_path: ""        # path to base LLaMA-7B weights
peft_path: ""       # path to fine-tuned LoRA checkpoint (output of finetune_llm.py)
max_source_length: 1024
offline: false      # true = use pre-computed hidden states from JSONL (no teacher GPU needed)
```

**`configs/train.yaml`** — key defaults:

```yaml
batch_size: 128
learning_rate: 5.0e-4
epochs: 30
patience: 10
threshold: 0.3
```

## Running the Pipeline

### 1. Preprocess

Reads raw MIMIC CSVs and writes vocabulary, JSONL splits, and DDI matrices to `output_dir`.

```bash
python scripts/preprocess.py --data configs/data.yaml
```

To use MIMIC-IV instead:

```bash
python scripts/preprocess.py --data configs/data.yaml --dataset mimic4
```

Outputs written to `data/`:
- `voc_final.pkl` — diagnosis / procedure / medication vocabularies
- `train_mimic3.json`, `val_mimic3.json`, `test_mimic3.json` — JSONL splits (80 / 10 / 10)
- `profile_dict.json` — patient profile feature tokenizer
- `ddi_A_final.pkl` — DDI adjacency matrix
- `ehr_adj_final.pkl` — EHR co-occurrence matrix
- `ddi_mask_H.pkl` — drug-fragment mask
- `atc3toSMILES.pkl` — ATC3 -> SMILES mapping

### 2. (Optional) Fine-tune LLaMA teacher

Skip this step if you only want to train the LEADER student without distillation.

First, place LLaMA-7B weights in `resources/llama-7b/` (see [Downloading LLaMA-7B](#downloading-llama-7b) below), then run:

```bash
python scripts/finetune_llm.py \
    --data       configs/data.yaml \
    --model      configs/model.yaml \
    --train      configs/train.yaml \
    --llm_path   /mnt/data/hungnh2/leader/resources/llama-7b \
    --output_dir /mnt/data/hungnh2/leader/resources/llm_teacher \
    --epochs     3 \
    --batch_size 4 \
    --lr         2e-4
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--llm_path` | *(required)* | Path to base LLaMA-7B weights |
| `--output_dir` | `saved/llm_teacher` | Where to save LoRA checkpoints |
| `--lora_rank` | `8` | LoRA rank |
| `--lora_alpha` | `32.0` | LoRA alpha |
| `--trainable` | `q_proj,v_proj` | Modules to apply LoRA to |
| `--epochs` | `3` | Number of fine-tuning epochs |
| `--batch_size` | `4` | Batch size (keep low for 7B model) |
| `--save_steps` | `500` | Save a checkpoint every N steps |
| `--gpu_id` | `0` | GPU to use |

The best checkpoint (by Jaccard on validation set) is saved to `resources/llm_teacher/best/`.

### 3. Train LEADER student

**Without distillation** (default):

```bash
python scripts/train.py \
    --data  configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml
```

**With LLM knowledge distillation** (online — teacher runs each batch):

Set in `configs/model.yaml`:
```yaml
distill: true
llm_path: resources/llama-7b
peft_path: resources/llm_teacher/best
offline: false
```

Then run:
```bash
python scripts/train.py \
    --data  configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml
```

**With LLM knowledge distillation** (offline — uses pre-computed hidden states):

First generate teacher predictions on the training set (run `finetune_llm.py` with test mode and save `hidden_states` into JSONL), then set `offline: true` in `configs/model.yaml`.

The best checkpoint (by validation PRAUC) is saved to `data/checkpoints/pytorch_model.bin`.
Training logs and a results JSON are written to `logs/`.

To train on MIMIC-IV:

```bash
python scripts/train.py \
    --data  configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml \
    --dataset mimic4
```

To run test-only with an existing checkpoint:

```bash
python scripts/train.py \
    --data  configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml \
    --test_only
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --data  configs/data.yaml \
    --model configs/model.yaml \
    --train configs/train.yaml
```

Reports the following metrics on the test set:

| Metric | Description |
|--------|-------------|
| Jaccard | Set similarity between predicted and ground-truth medications |
| F1 | Harmonic mean of precision and recall |
| PRAUC | Precision-recall AUC |
| DDI rate | Drug-drug interaction rate in predictions |
| avg_meds | Average number of predicted medications per visit |

Metrics are reported overall and split by single-visit vs multi-visit patients.

To evaluate on the validation set:

```bash
python scripts/evaluate.py ... --split val
```

To use a custom checkpoint:

```bash
python scripts/evaluate.py ... --checkpoint /path/to/pytorch_model.bin
```

## Downloading LLaMA-7B

The LLM teacher uses `huggyllama/llama-7b` (community upload of Meta LLaMA-1 7B, no access token required).

```bash
conda activate leader
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='huggyllama/llama-7b',
    local_dir='resources/llama-7b',
    local_dir_use_symlinks=False,
)
"
```

This downloads ~13 GB. After completion, `resources/llama-7b/` should contain:
```
config.json
tokenizer.model
tokenizer_config.json
pytorch_model-00001-of-00002.bin
pytorch_model-00002-of-00002.bin
pytorch_model.bin.index.json
```

## GPU Cluster (SLURM)

Edit `slurm/job.slurm` to set your partition and paths, then submit:

```bash
sbatch slurm/job.slurm
```

Monitor the job:

```bash
squeue -u $USER
```

Cancel the job:

```bash
scancel JOB_ID
```

For a short debug run before committing to 24 hours:

```bash
#SBATCH --time=00:10:00
```

For an interactive GPU session:

```bash
salloc --time=01:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=16G
python scripts/train.py --data configs/data.yaml --model configs/model.yaml --train configs/train.yaml
```

## Cluster Best Practices

- Store large datasets on shared storage (`/mnt/data/`) — avoid `$HOME`
- Run a short debug job before submitting long runs
- Use `logs/%j.out` and `logs/%j.err` for per-job output files

```bash
srun --pty --time=01:00:00 --ntasks=1 --cpus-per-task=4 --mem=8G --gres=gpu:1 bash
```