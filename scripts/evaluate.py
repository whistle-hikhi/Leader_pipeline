"""
Evaluation script for the LEADER pipeline.

Loads a trained checkpoint and computes:
  - Jaccard similarity
  - F1 score
  - PRAUC (precision-recall AUC)
  - DDI rate
  - Average number of predicted medications

Reports overall metrics and per-group metrics (single-visit vs multi-visit).

Usage:
  python scripts/evaluate.py \
      --data   configs/data.yaml \
      --model  configs/model.yaml \
      --train  configs/train.yaml \
      [--checkpoint path/to/pytorch_model.bin] \
      [--dataset mimic3|mimic4] \
      [--threshold 0.3]
"""

import argparse
import json
import logging
import os
import sys

import dill
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train import (
    EHRTokenizer,
    MedRecDataset,
    build_model,
    ddi_rate,
    merge_configs,
    multilabel_metrics,
    prepare_inputs,
    read_jsonlines,
)


# ---------------------------------------------------------------------------
# Extended metrics: single-visit and multi-visit groups
# ---------------------------------------------------------------------------

def evaluate_full(model, loader, var_names, device, ddi_adj, threshold, logger):
    model.eval()
    y_probs, y_trues, seq_lens = [], [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = tuple(t.to(device) for t in batch)
        inputs = prepare_inputs(batch, var_names, device)
        with torch.no_grad():
            output = model(**inputs)
        y_probs.append(torch.sigmoid(output).cpu().numpy())
        y_trues.append(inputs["labels"].cpu().numpy())
        seq_lens.append(inputs["seq_mask"].sum(dim=1).cpu().numpy())

    y_prob = np.concatenate(y_probs, axis=0)
    y_true = np.concatenate(y_trues, axis=0)
    seq_len = np.concatenate(seq_lens, axis=0)

    y_pred = (y_prob >= threshold).astype(float)

    pred_labels = [[list(np.where(p == 1)[0])] for p in y_pred]
    ddi = ddi_rate(pred_labels, ddi_adj)
    avg_meds = y_pred.sum(axis=1).mean()

    ja, prauc, avg_p, avg_r, avg_f1 = multilabel_metrics(y_true, y_pred, y_prob)

    # single-visit (seq_len == 1) vs multi-visit (seq_len > 1)
    single_idx = seq_len == 1
    multi_idx  = seq_len != 1

    def group_metrics(idx, tag):
        if idx.sum() == 0:
            return {}
        gja, gprauc, gp, gr, gf1 = multilabel_metrics(
            y_true[idx], y_pred[idx], y_prob[idx])
        gpred_labels = [[list(np.where(p == 1)[0])] for p in y_pred[idx]]
        gddi = ddi_rate(gpred_labels, ddi_adj)
        return {
            f"{tag}-jaccard": gja, f"{tag}-prauc": gprauc,
            f"{tag}-f1": gf1, f"{tag}-ddi": gddi,
        }

    results = {
        "jaccard": ja, "prauc": prauc, "f1": avg_f1,
        "precision": avg_p, "recall": avg_r,
        "ddi": ddi, "avg_meds": avg_meds,
    }
    results.update(group_metrics(single_idx, "single"))
    results.update(group_metrics(multi_idx,  "multi"))

    logger.info("\n%-20s  %-10s" % ("Metric", "Value"))
    logger.info("-" * 35)
    for k, v in results.items():
        logger.info("%-20s  %.4f" % (k, v))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained LEADER checkpoint")
    parser.add_argument("--data",       default="configs/data.yaml")
    parser.add_argument("--model",      default="configs/model.yaml")
    parser.add_argument("--train",      default="configs/train.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to pytorch_model.bin (default: output_dir/pytorch_model.bin)")
    parser.add_argument("--dataset",    default=None, choices=["mimic3", "mimic4"])
    parser.add_argument("--tag",        default=None, help="Data file tag")
    parser.add_argument("--threshold",  default=None, type=float,
                        help="Prediction threshold (default: from train config)")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"],
                        help="Which data split to evaluate on")
    args = parser.parse_args()

    cfg = merge_configs(args.data, args.model, args.train)
    dataset = args.dataset or cfg.get("dataset", "mimic3")
    tag = args.tag or dataset
    threshold = args.threshold or cfg.get("threshold", 0.3)

    # ---- logging ----
    log_dir = cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"eval_{dataset}.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("leader_eval")

    # ---- device ----
    gpu_id = cfg.get("gpu_id", 0)
    use_cuda = not cfg.get("no_cuda", False) and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")
    logger.info("Device: %s  |  threshold: %.3f", device, threshold)

    # ---- data ----
    data_dir = cfg["output_dir"]
    tokenizer = EHRTokenizer(os.path.join(data_dir, "voc_final.pkl"))
    with open(os.path.join(data_dir, "profile_dict.json")) as f:
        profile_tokenizer = json.load(f)
    with open(os.path.join(data_dir, "ddi_A_final.pkl"), "rb") as f:
        ddi_adj = dill.load(f)

    split_file = os.path.join(data_dir, f"{args.split}_{tag}.json")
    records = read_jsonlines(split_file)
    logger.info("Loaded %d records from %s", len(records), split_file)

    ds = MedRecDataset(
        records, tokenizer=tokenizer,
        profile_tokenizer=profile_tokenizer,
        max_seq_len=cfg.get("max_seq_length", 100),
        max_record_num=cfg.get("max_record_num", 10),
        filter_single=cfg.get("filter", False),
    )
    loader = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=100,
                        num_workers=cfg.get("num_workers", 0))

    # ---- model ----
    model = build_model(cfg, tokenizer, profile_tokenizer, device)

    ckpt_path = args.checkpoint or os.path.join(
        cfg.get("output_dir", "data/checkpoints"), "pytorch_model.bin")
    if not os.path.isfile(ckpt_path):
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    state = torch.load(ckpt_path, map_location=device)
    # filter any keys that no longer exist in the model (e.g. distill projector)
    state = {k: v for k, v in state.items() if k in model.state_dict()}
    model.load_state_dict(state)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    # ---- evaluate ----
    results = evaluate_full(model, loader, ds.var_name, device,
                             ddi_adj, threshold, logger)

    # ---- save ----
    out_path = os.path.join(log_dir, f"eval_{dataset}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
