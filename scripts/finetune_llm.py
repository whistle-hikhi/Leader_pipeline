"""
Fine-tune LLaMA as a medication recommendation teacher model.

This produces a LoRA checkpoint that can later be used as a frozen teacher
during LEADER student distillation (scripts/train.py with distill: true).

Usage:
  python scripts/finetune_llm.py \
      --data   configs/data.yaml \
      --model  configs/model.yaml \
      --train  configs/train.yaml \
      --llm_path  /path/to/llama-7b \
      --output_dir saved/llm_teacher

After training the best checkpoint is at:
  <output_dir>/checkpoint-<step>/   (LoRA weights + cls_head.bin)

Pass that path as peft_path in model.yaml for distillation.
"""

import argparse
import json
import logging
import os
import sys
import time

import dill
import jsonlines
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llama import LlamaForMedRec
from llm.lora_cls import create_lora_model, save_lora_model, load_lora_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*paths) -> dict:
    cfg = {}
    for p in paths:
        if p:
            cfg.update(load_yaml(p))
    return cfg


# ---------------------------------------------------------------------------
# Vocabulary (mirrors train.py)
# ---------------------------------------------------------------------------

class Voc:
    def __init__(self):
        self.idx2word: dict = {}
        self.word2idx: dict = {}

    def add_sentence(self, tokens):
        for t in tokens:
            if t not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[t] = idx
                self.idx2word[idx] = t


class EHRTokenizer:
    SPECIAL = ("[PAD]", "[CLS]", "[MASK]")

    def __init__(self, voc_path: str):
        self.vocab = Voc()
        self.vocab.add_sentence(self.SPECIAL)
        with open(voc_path, "rb") as f:
            voc_dict = dill.load(f)
        self.diag_voc = voc_dict["diag_voc"]
        self.med_voc  = voc_dict["med_voc"]
        self.pro_voc  = voc_dict["pro_voc"]
        self.vocab.add_sentence(self.med_voc.word2idx)
        self.vocab.add_sentence(self.diag_voc.word2idx)
        self.vocab.add_sentence(self.pro_voc.word2idx)

    def convert_med_tokens_to_ids(self, tokens):
        return [self.med_voc.word2idx[t] for t in tokens if t in self.med_voc.word2idx]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(record: dict) -> str:
    """Convert a structured patient record into a text prompt for the LLM."""
    diag_visits  = record["records"]["diagnosis"]
    proc_visits  = record["records"]["procedure"]
    med_visits   = record["records"]["medication"]
    n_hist = len(diag_visits) - 1  # all visits except the current one

    lines = ["Patient medical history:"]
    for i in range(n_hist):
        lines.append(f"Visit {i + 1}:")
        lines.append(f"  Diagnoses:   {', '.join(diag_visits[i])}")
        lines.append(f"  Procedures:  {', '.join(proc_visits[i])}")
        lines.append(f"  Medications: {', '.join(med_visits[i])}")

    lines.append("Current visit:")
    lines.append(f"  Diagnoses:  {', '.join(diag_visits[-1])}")
    lines.append(f"  Procedures: {', '.join(proc_visits[-1])}")
    lines.append("Please recommend medications for the current visit.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LLMMedRecDataset(Dataset):
    """Tokenises patient records for LLM fine-tuning (classification mode)."""

    def __init__(self, records: list, ehr_tokenizer: EHRTokenizer,
                 llm_tokenizer, max_source_length: int = 1024):
        self.records = records
        self.ehr_tokenizer = ehr_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.max_source_length = max_source_length
        self.med_voc_size = len(ehr_tokenizer.med_voc.word2idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # tokenise prompt
        prompt = build_prompt(rec)
        ids = self.llm_tokenizer.encode(
            text=prompt, add_special_tokens=False,
            truncation=True, max_length=self.max_source_length - 1,
        )
        input_ids = ids + [self.llm_tokenizer.eos_token_id]
        # pad to fixed length
        pad_len = self.max_source_length - len(input_ids)
        input_ids = input_ids + [self.llm_tokenizer.pad_token_id] * pad_len

        # multi-hot medication label
        label_ids = self.ehr_tokenizer.convert_med_tokens_to_ids(rec["drug_code"])
        label = np.zeros(self.med_voc_size, dtype=np.float32)
        for i in label_ids:
            label[i] = 1.0

        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(label, dtype=torch.float))


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels    = torch.stack([b[1] for b in batch])
    attention_mask = (input_ids != batch[0][0].new_zeros(1)).long()
    # recompute attention mask per sample
    pad_id = input_ids[0][-1].item()  # may be pad; just use non-zero mask
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_llm(model, loader, device, threshold=0.3, logger=None):
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(out.logits).cpu().numpy()
            all_preds.append(probs)
            all_trues.append(labels.numpy())

    y_prob = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_trues, axis=0)
    y_pred = (y_prob >= threshold).astype(float)

    jaccard, f1 = [], []
    for i in range(len(y_true)):
        gt  = set(np.where(y_true[i] == 1)[0])
        out = set(np.where(y_pred[i] == 1)[0])
        inter = gt & out
        union = gt | out
        jaccard.append(len(inter) / len(union) if union else 0)
        p = len(inter) / len(out)   if out else 0
        r = len(inter) / len(gt)    if gt  else 0
        f1.append(2 * p * r / (p + r) if (p + r) else 0)

    metrics = {"jaccard": float(np.mean(jaccard)), "f1": float(np.mean(f1))}
    if logger:
        for k, v in metrics.items():
            logger.info("  %-10s : %.4f", k, v)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA teacher for medication recommendation")
    parser.add_argument("--data",        default="configs/data.yaml")
    parser.add_argument("--model",       default="configs/model.yaml")
    parser.add_argument("--train",       default="configs/train.yaml")
    parser.add_argument("--llm_path",    required=True,
                        help="Path to base LLaMA weights (e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--output_dir",  default="saved/llm_teacher")
    parser.add_argument("--lora_rank",   type=int,   default=8)
    parser.add_argument("--lora_alpha",  type=float, default=32.0)
    parser.add_argument("--lora_dropout",type=float, default=0.1)
    parser.add_argument("--trainable",   default="q_proj,v_proj",
                        help="Comma-separated LLaMA module names to apply LoRA to")
    parser.add_argument("--max_source_length", type=int, default=1024)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--save_steps",  type=int,   default=500,
                        help="Save a checkpoint every N steps")
    parser.add_argument("--gpu_id",      type=int,   default=0)
    parser.add_argument("--peft_path",   default=None,
                        help="Resume from existing LoRA checkpoint")
    args = parser.parse_args()

    cfg = merge_configs(args.data, args.model, args.train)
    data_dir = cfg["output_dir"]

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "finetune.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("finetune_llm")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- tokenizers ----
    ehr_tokenizer = EHRTokenizer(os.path.join(data_dir, "voc_final.pkl"))
    med_voc_size  = len(ehr_tokenizer.med_voc.word2idx)
    logger.info("Med vocab size: %d", med_voc_size)

    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    llm_tokenizer.pad_token = llm_tokenizer.unk_token
    llm_tokenizer.padding_side = "right"

    # ---- load model ----
    logger.info("Loading base LLaMA from %s ...", args.llm_path)
    base_model = LlamaForMedRec.from_pretrained(
        args.llm_path,
        med_voc=med_voc_size,
    ).half().to(device)

    if args.peft_path is not None:
        logger.info("Resuming from LoRA checkpoint: %s", args.peft_path)
        model = load_lora_model(base_model, args.peft_path, is_trainable=True)
    else:
        model = create_lora_model(base_model,
                                  lora_rank=args.lora_rank,
                                  lora_alpha=args.lora_alpha,
                                  trainable=args.trainable,
                                  lora_dropout=args.lora_dropout)

    # ensure cls_head is trainable
    for name, param in model.named_parameters():
        if "cls_head" in name:
            param.requires_grad = True
    model.print_trainable_parameters()

    # ---- datasets ----
    tag = cfg.get("dataset", "mimic3")

    def load_split(fname):
        with jsonlines.open(os.path.join(data_dir, fname)) as f:
            return list(f)

    train_records = load_split(f"train_{tag}.json")
    val_records   = load_split(f"val_{tag}.json")
    logger.info("Train: %d  Val: %d", len(train_records), len(val_records))

    ds_kwargs = dict(ehr_tokenizer=ehr_tokenizer,
                     llm_tokenizer=llm_tokenizer,
                     max_source_length=args.max_source_length)
    train_ds = LLMMedRecDataset(train_records, **ds_kwargs)
    val_ds   = LLMMedRecDataset(val_records,   **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    # ---- optimiser ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # ---- training loop ----
    global_step = 0
    best_jaccard = 0.0
    loss_fct = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        epoch_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        for batch in prog:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            out  = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fct(out.logits.float(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1
            prog.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                save_lora_model(model, ckpt_dir)
                logger.info("Saved checkpoint -> %s", ckpt_dir)

        logger.info("Epoch %d  avg_loss=%.4f", epoch, epoch_loss / len(train_loader))

        metrics = evaluate_llm(model, val_loader, device, logger=logger)
        if metrics["jaccard"] > best_jaccard:
            best_jaccard = metrics["jaccard"]
            best_dir = os.path.join(args.output_dir, "best")
            save_lora_model(model, best_dir)
            logger.info("New best jaccard=%.4f  saved -> %s", best_jaccard, best_dir)

    logger.info("Training done. Best jaccard: %.4f", best_jaccard)
    logger.info("Use --peft_path %s for distillation.", os.path.join(args.output_dir, "best"))


if __name__ == "__main__":
    main()
