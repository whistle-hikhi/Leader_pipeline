"""
Training entry point for the LEADER pipeline.

Usage:
  python scripts/train.py \
      --data   configs/data.yaml \
      --model  configs/model.yaml \
      --train  configs/train.yaml

All three config files are merged into a single namespace before training.
Run python scripts/train.py --help for the full list of CLI overrides.
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time

import dill
import jsonlines
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.leader import LEADER

# LLM teacher imports (only needed when distill=True)
try:
    from transformers import AutoTokenizer
    from llm.llama import LlamaForMedRec
    from llm.lora_cls import load_lora_model
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*yamls: str) -> dict:
    cfg = {}
    for y in yamls:
        if y:
            cfg.update(load_yaml(y))
    return cfg


# ---------------------------------------------------------------------------
# Vocabulary / tokenizer (mirrors generators/data.py)
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

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.word2idx[t] for t in tokens]

    def convert_med_tokens_to_ids(self, tokens):
        return [self.med_voc.word2idx[t] for t in tokens]


# ---------------------------------------------------------------------------
# Prompt builder (for LLM teacher)
# ---------------------------------------------------------------------------

def build_prompt(record: dict) -> str:
    """Convert a structured patient record into a text prompt for the LLM."""
    diag_visits = record["records"]["diagnosis"]
    proc_visits = record["records"]["procedure"]
    med_visits  = record["records"]["medication"]
    n_hist = len(diag_visits) - 1  # visits before the current one

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

class MedRecDataset(Dataset):
    VAR_NAMES = ["diag_seq", "proc_seq", "med_seq", "seq_mask",
                 "labels", "multi_label", "profile"]

    def __init__(self, records: list, tokenizer: EHRTokenizer,
                 profile_tokenizer: dict, max_seq_len: int, max_record_num: int,
                 filter_single: bool = False):
        self.records = records
        self.tokenizer = tokenizer
        self.profile_tokenizer = profile_tokenizer
        self.max_seq_len = max_seq_len
        self.max_record_num = max_record_num
        self.var_name = self.VAR_NAMES

        if filter_single:
            self.records = [r for r in records
                            if len(r["records"]["medication"]) > 1]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        adm = self.records[item]
        med_seq  = adm["records"]["medication"]
        diag_seq = adm["records"]["diagnosis"]
        proc_seq = adm["records"]["procedure"]

        profile = [
            self.profile_tokenizer["word2idx"][col].get(str(v), 0)
            for col, v in adm["profile"].items()
        ]

        # medication label
        label_ids = self.tokenizer.convert_med_tokens_to_ids(adm["drug_code"])
        label = np.zeros(len(self.tokenizer.med_voc.word2idx))
        multi_label = np.full(len(self.tokenizer.med_voc.word2idx), -1)
        for i, idx in enumerate(label_ids):
            label[idx] = 1
            multi_label[i] = idx

        pad_ids = self.tokenizer.convert_tokens_to_ids(
            ["[PAD]"] * self.max_seq_len)

        def pad_seq(seq):
            out = self.tokenizer.convert_tokens_to_ids(
                (seq + ["[PAD]"] * self.max_seq_len)[:self.max_seq_len])
            return out

        def pad_record(seqs, max_r):
            padded = [pad_seq(s) for s in seqs]
            n_pad = 0
            while len(padded) < max_r:
                padded.append(pad_ids)
                n_pad += 1
            if len(padded) > max_r:
                padded = padded[:max_r]
            return padded, n_pad

        med_seq  = med_seq[:-1]   # remove current admission (it is the label)
        med_padded,  _       = pad_record(med_seq,  self.max_record_num)
        diag_padded, pad_num = pad_record(diag_seq, self.max_record_num)
        proc_padded, _       = pad_record(proc_seq, self.max_record_num)

        mask = np.ones(self.max_record_num)
        if pad_num:
            mask[-pad_num:] = 0

        return (np.array(diag_padded, dtype=int),
                np.array(proc_padded, dtype=int),
                np.array(med_padded,  dtype=int),
                mask.astype(int),
                label.astype(float),
                multi_label.astype(int),
                np.array(profile, dtype=int))


class DistillMedRecDataset(MedRecDataset):
    """Extends MedRecDataset with tokenised LLM input_ids for online distillation."""

    VAR_NAMES = ["diag_seq", "proc_seq", "med_seq", "seq_mask",
                 "labels", "multi_label", "profile", "input_ids"]

    def __init__(self, records, tokenizer, profile_tokenizer,
                 max_seq_len, max_record_num, llm_tokenizer,
                 max_source_length: int = 1024, filter_single: bool = False):
        super().__init__(records, tokenizer, profile_tokenizer,
                         max_seq_len, max_record_num, filter_single)
        self.llm_tokenizer = llm_tokenizer
        self.max_source_length = max_source_length
        self.var_name = self.VAR_NAMES

    def __getitem__(self, item):
        base = super().__getitem__(item)
        rec = self.records[item]
        prompt = build_prompt(rec)
        ids = self.llm_tokenizer.encode(text=prompt, add_special_tokens=False)
        ids = ids[: self.max_source_length - 1]
        input_ids = ids + [self.llm_tokenizer.eos_token_id]
        pad_len = self.max_source_length - len(input_ids)
        input_ids = input_ids + [self.llm_tokenizer.pad_token_id] * pad_len
        return base + (np.array(input_ids, dtype=int),)


class OfflineDistillMedRecDataset(DistillMedRecDataset):
    """Extends DistillMedRecDataset with pre-computed LLM hidden_states and logits."""

    VAR_NAMES = ["diag_seq", "proc_seq", "med_seq", "seq_mask",
                 "labels", "multi_label", "profile", "input_ids",
                 "hidden_states", "logits"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var_name = self.VAR_NAMES

    def __getitem__(self, item):
        base = super().__getitem__(item)
        rec = self.records[item]
        hidden_states = np.array(rec["hidden_states"], dtype=np.float32)
        logits = np.array(rec["logits"], dtype=np.float32)
        return base + (hidden_states, logits)


def read_jsonlines(path: str) -> list:
    with jsonlines.open(path, "r") as f:
        return list(f)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ddi_rate(pred_labels: list, ddi_adj: np.ndarray) -> float:
    all_cnt = dd_cnt = 0
    for adm in pred_labels:
        meds = adm[0]
        for i, mi in enumerate(meds):
            for j, mj in enumerate(meds):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_adj[mi, mj] == 1 or ddi_adj[mj, mi] == 1:
                    dd_cnt += 1
    return dd_cnt / all_cnt if all_cnt else 0.0


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    from sklearn.metrics import average_precision_score

    jaccard, prec, recall, f1 = [], [], [], []
    prauc = []
    for b in range(len(y_true)):
        gt  = set(np.where(y_true[b] == 1)[0])
        out = set(np.where(y_pred[b] == 1)[0])
        inter = gt & out
        union = gt | out
        jaccard.append(len(inter) / len(union) if union else 0)
        prec.append(len(inter) / len(out) if out else 0)
        recall.append(len(inter) / len(gt) if gt else 0)
        p_r = prec[-1] + recall[-1]
        f1.append(2 * prec[-1] * recall[-1] / p_r if p_r else 0)
        prauc.append(average_precision_score(y_true[b], y_prob[b], average="macro"))

    return (np.mean(jaccard), np.mean(prauc),
            np.mean(prec), np.mean(recall), np.mean(f1))


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int, path: str):
        os.makedirs(path, exist_ok=True)
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.ckpt_path = os.path.join(path, "pytorch_model.bin")

    def __call__(self, score: float, epoch: int, model: nn.Module):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), self.ckpt_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict, tokenizer: EHRTokenizer,
                profile_tokenizer: dict, device: torch.device) -> LEADER:
    use_profile = cfg.get("profile", False)
    return LEADER(
        vocab_size=len(tokenizer.vocab.word2idx),
        med_voc_size=len(tokenizer.med_voc.word2idx),
        device=device,
        hidden_size=cfg.get("hidden_size", 64),
        num_trm_layers=cfg.get("num_trm_layers", 1),
        num_heads=cfg.get("num_attention_heads", 4),
        intermediate_size=cfg.get("intermediate_size", 64),
        hidden_dropout=cfg.get("hidden_dropout_prob", 0.4),
        attn_dropout=cfg.get("attention_probs_dropout_prob", 0.1),
        prompt_num=cfg.get("prompt_num", 1),
        profile_tokenizer=profile_tokenizer if use_profile else None,
        distill=cfg.get("distill", False),
        d_loss=cfg.get("d_loss", "mse"),
        alpha=cfg.get("alpha", 0.1),
        temperature=cfg.get("temperature", 5.0),
        align=cfg.get("align", False),
        align_weight=cfg.get("align_weight", 0.1),
        ml_weight=cfg.get("ml_weight", 0.05),
    ).to(device)


def prepare_inputs(batch, var_names: list, device: torch.device) -> dict:
    return {name: t.to(device) for name, t in zip(var_names, batch)}


def evaluate(model: LEADER, loader: DataLoader, var_names: list,
             device: torch.device, ddi_adj: np.ndarray,
             threshold: float, logger: logging.Logger) -> dict:
    model.eval()
    y_probs, y_trues, seq_lens = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = tuple(t.to(device) for t in batch)
        inputs = prepare_inputs(batch, var_names, device)
        with torch.no_grad():
            output = model(**inputs)
        prob = torch.sigmoid(output).cpu().numpy()
        y_probs.append(prob)
        y_trues.append(inputs["labels"].cpu().numpy())
        seq_lens.append(inputs["seq_mask"].sum(dim=1).cpu().numpy())

    y_prob = np.concatenate(y_probs, axis=0)
    y_true = np.concatenate(y_trues, axis=0)

    y_pred = (y_prob >= threshold).astype(float)
    ja, prauc, avg_p, avg_r, avg_f1 = multilabel_metrics(y_true, y_pred, y_prob)

    pred_labels = [[list(np.where(p == 1)[0])] for p in y_pred]
    ddi = ddi_rate(pred_labels, ddi_adj)

    avg_meds = y_pred.sum(axis=1).mean()

    metrics = {
        "jaccard": ja, "prauc": prauc, "f1": avg_f1,
        "precision": avg_p, "recall": avg_r,
        "ddi": ddi, "avg_meds": avg_meds,
    }
    for k, v in metrics.items():
        logger.info("  %-12s : %.4f" % (k, v))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train LEADER medication recommendation model")
    parser.add_argument("--data",  default="configs/data.yaml")
    parser.add_argument("--model", default="configs/model.yaml")
    parser.add_argument("--train", default="configs/train.yaml")
    parser.add_argument("--dataset", default=None, choices=["mimic3", "mimic4"],
                        help="Override dataset in data config")
    parser.add_argument("--tag", default=None, help="Data file tag")
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    cfg = merge_configs(args.data, args.model, args.train)
    dataset = args.dataset or cfg.get("dataset", "mimic3")
    tag = args.tag or dataset

    set_seed(cfg.get("seed", 42))

    # ---- logging ----
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"train_{dataset}.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("leader")
    logger.info("Config: %s", json.dumps(cfg, indent=2, default=str))

    # ---- device ----
    gpu_id = cfg.get("gpu_id", 0)
    use_cuda = not cfg.get("no_cuda", False) and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    logger.info("Using device: %s", device)

    # ---- data ----
    data_dir = cfg["output_dir"]
    voc_path  = os.path.join(data_dir, "voc_final.pkl")
    prof_path = os.path.join(data_dir, "profile_dict.json")
    ddi_path  = os.path.join(data_dir, "ddi_A_final.pkl")

    tokenizer = EHRTokenizer(voc_path)
    with open(prof_path) as f:
        profile_tokenizer = json.load(f)
    with open(ddi_path, "rb") as f:
        ddi_adj = dill.load(f)

    train_data = read_jsonlines(os.path.join(data_dir, f"train_{tag}.json"))
    val_data   = read_jsonlines(os.path.join(data_dir, f"val_{tag}.json"))
    test_data  = read_jsonlines(os.path.join(data_dir, f"test_{tag}.json"))
    logger.info("Dataset splits — train: %d  val: %d  test: %d",
                len(train_data), len(val_data), len(test_data))

    # ---- distillation setup ----
    use_distill = cfg.get("distill", False)
    offline     = cfg.get("offline", False)
    teacher     = None

    ds_kwargs = dict(
        tokenizer=tokenizer,
        profile_tokenizer=profile_tokenizer,
        max_seq_len=cfg.get("max_seq_length", 100),
        max_record_num=cfg.get("max_record_num", 10),
        filter_single=cfg.get("filter", False),
    )

    if use_distill:
        if not _LLM_AVAILABLE:
            raise RuntimeError(
                "distill=True requires 'transformers' and 'peft'. "
                "Install them: pip install transformers peft accelerate"
            )
        llm_path = cfg.get("llm_path", "")
        peft_path = cfg.get("peft_path", "")
        max_source_length = cfg.get("max_source_length", 1024)

        llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        llm_tokenizer.pad_token = llm_tokenizer.unk_token
        llm_tokenizer.padding_side = "right"

        distill_ds_kwargs = dict(**ds_kwargs,
                                 llm_tokenizer=llm_tokenizer,
                                 max_source_length=max_source_length)

        if offline:
            logger.info("Offline distillation: loading pre-computed hidden states from data files")
            train_ds = OfflineDistillMedRecDataset(train_data, **distill_ds_kwargs)
        else:
            logger.info("Online distillation: loading LLM teacher from %s (LoRA: %s)",
                        llm_path, peft_path)
            base = LlamaForMedRec.from_pretrained(
                llm_path, med_voc=len(tokenizer.med_voc.word2idx),
            ).half().to(device)
            teacher = load_lora_model(base, peft_path, is_trainable=False)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            logger.info("Teacher loaded and frozen.")
            train_ds = DistillMedRecDataset(train_data, **distill_ds_kwargs)

        val_ds  = DistillMedRecDataset(val_data,  **distill_ds_kwargs)
        test_ds = DistillMedRecDataset(test_data, **distill_ds_kwargs)
    else:
        train_ds = MedRecDataset(train_data, **ds_kwargs)
        val_ds   = MedRecDataset(val_data,   **ds_kwargs)
        test_ds  = MedRecDataset(test_data,  **ds_kwargs)

    bs = cfg.get("batch_size", 128)
    nw = cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds),
                              batch_size=bs, num_workers=nw)
    val_loader   = DataLoader(val_ds,   sampler=SequentialSampler(val_ds),
                              batch_size=100, num_workers=nw)
    test_loader  = DataLoader(test_ds,  sampler=SequentialSampler(test_ds),
                              batch_size=100, num_workers=nw)

    # ---- model ----
    model = build_model(cfg, tokenizer, profile_tokenizer, device)
    logger.info("Model parameters: %d",
                sum(p.numel() for p in model.parameters()))

    ckpt_dir = cfg.get("ckpt_dir", os.path.join(cfg["output_dir"], "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.test_only:
        ckpt = os.path.join(ckpt_dir, "pytorch_model.bin")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info("=== Test results ===")
        evaluate(model, test_loader, test_ds.var_name, device, ddi_adj,
                 cfg.get("threshold", 0.3), logger)
        return

    # ---- optimiser + scheduler ----
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("learning_rate", 5e-4),
        weight_decay=cfg.get("l2", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.get("lr_dc_step", 1000),
        gamma=cfg.get("lr_dc", 1.0) if cfg.get("lr_dc", 0) > 0 else 1.0,
    )
    stopper = EarlyStopping(
        patience=cfg.get("patience", 10),
        path=ckpt_dir,
    )

    # ---- training loop ----
    best_metrics = None
    for epoch in trange(int(cfg.get("epochs", 30)), desc="Epoch"):
        model.train()
        total_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in prog:
            batch = tuple(t.to(device) for t in batch)
            inputs = prepare_inputs(batch, train_ds.var_name, device)

            # attach teacher output for distillation
            if use_distill:
                if offline:
                    # hidden_states and logits already loaded from data
                    inputs.pop("input_ids", None)  # not used in offline mode
                    inputs["llm_output"] = {
                        "hidden_states": inputs.pop("hidden_states"),
                        "logits": inputs.pop("logits"),
                    }
                else:
                    with torch.no_grad():
                        teacher_out = teacher(input_ids=inputs.pop("input_ids"))
                    inputs["llm_output"] = {
                        "hidden_states": teacher_out.hidden_states,
                        "logits": teacher_out.logits,
                    }
            elif "input_ids" in inputs:
                inputs.pop("input_ids")

            loss = model.get_loss(**inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prog.set_postfix(loss=f"{total_loss / (prog.n + 1):.4f}")

        scheduler.step()

        logger.info("\n--- Epoch %d  train_loss=%.4f ---", epoch, total_loss / len(train_loader))
        metrics = evaluate(model, val_loader, val_ds.var_name, device, ddi_adj,
                           cfg.get("threshold", 0.3), logger)
        stopper(metrics["prauc"], epoch, model)
        if stopper.early_stop:
            logger.info("Early stopping at epoch %d", epoch)
            break
        if best_metrics is None or metrics["prauc"] > best_metrics["prauc"]:
            best_metrics = metrics

    logger.info("Best epoch: %d", stopper.best_epoch)
    logger.info("Best val metrics: %s", best_metrics)

    # ---- test with best checkpoint ----
    model.load_state_dict(torch.load(stopper.ckpt_path, map_location=device))
    logger.info("\n=== Test results ===")
    test_metrics = evaluate(model, test_loader, test_ds.var_name, device, ddi_adj,
                            cfg.get("threshold", 0.3), logger)

    # save results json
    result_path = os.path.join(cfg.get("log_dir", "logs"), f"results_{dataset}.json")
    with open(result_path, "w") as f:
        json.dump({"best_epoch": stopper.best_epoch,
                   "val": best_metrics, "test": test_metrics}, f, indent=2)
    logger.info("Results saved to %s", result_path)


if __name__ == "__main__":
    main()
