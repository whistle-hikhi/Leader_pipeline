"""
Data preprocessing for the LEADER pipeline.

Supports both MIMIC-III (v1.4) and MIMIC-IV (v3.1) datasets.
Reads raw CSVs, maps NDC -> ATC3, builds vocabularies, constructs
patient-level records, and writes:
  - voc_final.pkl         : {diag_voc, med_voc, pro_voc}  (dill)
  - train_{tag}.json      : JSONL — train split
  - val_{tag}.json        : JSONL — validation split
  - test_{tag}.json       : JSONL — test split
  - profile_dict.json     : profile feature tokenizer
  - ddi_A_final.pkl       : DDI adjacency matrix           (dill)
  - ehr_adj_final.pkl     : EHR co-occurrence matrix       (dill)
  - ddi_mask_H.pkl        : drug-fragment mask             (dill)
  - atc3toSMILES.pkl      : ATC3 -> SMILES list            (dill)

Usage:
  python scripts/preprocess.py --data configs/data.yaml
  python scripts/preprocess.py --data configs/data.yaml --dataset mimic4
"""

import argparse
import ast
import json
import os
import random
import re
from collections import defaultdict

import dill
import jsonlines
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Voc:
    def __init__(self):
        self.idx2word: dict[int, str] = {}
        self.word2idx: dict[str, int] = {}

    def add_sentence(self, tokens):
        for t in tokens:
            if t not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[t] = idx
                self.idx2word[idx] = t


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# NDC -> RXCUI mapping (Python-2 dict literal in text file)
# ---------------------------------------------------------------------------

def load_ndc2rxcui(path: str) -> dict:
    with open(path, "r") as f:
        content = f.read()
    content = re.sub(r"\bu'", "'", content)  # strip Python 2 unicode prefix
    return ast.literal_eval(content)


# ---------------------------------------------------------------------------
# MIMIC-III preprocessing
# ---------------------------------------------------------------------------

def preprocess_mimic3(cfg: dict) -> tuple:
    """Returns (data_df, adm_df) where data_df has per-admission records."""
    base = cfg["mimic3_path"]

    # ---- medications ----
    med_pd = pd.read_csv(os.path.join(base, "PRESCRIPTIONS.csv.gz"),
                         dtype={"NDC": "category"}, low_memory=False)
    keep_cols = ["SUBJECT_ID", "HADM_ID", "STARTDATE", "NDC", "DRUG"]
    med_pd = med_pd[[c for c in keep_cols if c in med_pd.columns]]
    med_pd = med_pd[med_pd["NDC"] != "0"]
    med_pd["NDC"] = med_pd["NDC"].astype(str).str.replace("-", "")
    med_pd.ffill(inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["STARTDATE"] = pd.to_datetime(med_pd["STARTDATE"], format="%Y-%m-%d %H:%M:%S",
                                          errors="coerce")
    med_pd.sort_values(["SUBJECT_ID", "HADM_ID", "STARTDATE"], inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    ndc2rxcui = load_ndc2rxcui(cfg["ndc2rxcui_path"])
    med_pd["RXCUI"] = med_pd["NDC"].map(ndc2rxcui)
    med_pd.dropna(subset=["RXCUI"], inplace=True)
    med_pd = med_pd[med_pd["RXCUI"].astype(str).str.strip() != ""].reset_index(drop=True)

    rxcui2atc = pd.read_csv(cfg["rxcui2atc4_path"])
    rxcui2atc.drop(columns=["YEAR", "MONTH", "NDC"], inplace=True)
    rxcui2atc.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.merge(rxcui2atc, on="RXCUI")
    med_pd["ATC3"] = med_pd["ATC4"].str[:4]
    med_pd.drop(columns=["NDC", "RXCUI", "ATC4"], inplace=True, errors="ignore")
    med_pd.drop_duplicates(inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    # keep top-300 ATC3
    top_med = (med_pd.groupby("ATC3").size().reset_index(name="cnt")
               .sort_values("cnt", ascending=False).head(300)["ATC3"])
    med_pd = med_pd[med_pd["ATC3"].isin(top_med)].reset_index(drop=True)

    # ---- diagnoses ----
    diag_pd = pd.read_csv(os.path.join(base, "DIAGNOSES_ICD.csv.gz"), low_memory=False)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["ROW_ID", "SEQ_NUM"], inplace=True, errors="ignore")
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd.reset_index(drop=True, inplace=True)
    top_diag = (diag_pd.groupby("ICD9_CODE").size().reset_index(name="cnt")
                .sort_values("cnt", ascending=False).head(2000)["ICD9_CODE"])
    diag_pd = diag_pd[diag_pd["ICD9_CODE"].isin(top_diag)].reset_index(drop=True)

    # ---- procedures ----
    pro_pd = pd.read_csv(os.path.join(base, "PROCEDURES_ICD.csv.gz"), low_memory=False)
    pro_pd.drop(columns=["ROW_ID"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    top_pro = (pro_pd.groupby("ICD9_CODE").size().reset_index(name="cnt")
               .sort_values("cnt", ascending=False).head(1001)["ICD9_CODE"])
    pro_pd = pro_pd[pro_pd["ICD9_CODE"].isin(top_pro)].reset_index(drop=True)

    # ---- admissions (for profile) ----
    adm_pd = pd.read_csv(os.path.join(base, "ADMISSIONS.csv.gz"), low_memory=False)
    adm_pd = adm_pd[["SUBJECT_ID", "HADM_ID", "INSURANCE", "LANGUAGE",
                      "RELIGION", "MARITAL_STATUS", "ETHNICITY"]]
    adm_pd.fillna("Unknown", inplace=True)

    # ---- combine ----
    med_key  = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    pro_key  = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    key = med_key.merge(diag_key, on=["SUBJECT_ID", "HADM_ID"]).merge(pro_key, on=["SUBJECT_ID", "HADM_ID"])

    diag_pd = diag_pd.merge(key, on=["SUBJECT_ID", "HADM_ID"])
    med_pd  = med_pd.merge(key, on=["SUBJECT_ID", "HADM_ID"])
    pro_pd  = pro_pd.merge(key, on=["SUBJECT_ID", "HADM_ID"])

    diag_agg = diag_pd.groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"].unique().reset_index()
    med_agg  = med_pd.groupby(["SUBJECT_ID", "HADM_ID"])["ATC3"].unique().reset_index()
    pro_agg  = (pro_pd.groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"].unique()
                .reset_index().rename(columns={"ICD9_CODE": "PRO_CODE"}))

    data = diag_agg.merge(med_agg, on=["SUBJECT_ID", "HADM_ID"]).merge(pro_agg, on=["SUBJECT_ID", "HADM_ID"])
    data["ICD9_CODE"] = data["ICD9_CODE"].map(list)
    data["ATC3"]      = data["ATC3"].map(list)
    data["PRO_CODE"]  = data["PRO_CODE"].map(list)

    return data, adm_pd, "SUBJECT_ID", "HADM_ID", "ICD9_CODE", "PRO_CODE"


# ---------------------------------------------------------------------------
# MIMIC-IV preprocessing
# ---------------------------------------------------------------------------

def preprocess_mimic4(cfg: dict) -> tuple:
    base = cfg["mimic4_path"]

    # ---- medications ----
    med_pd = pd.read_csv(os.path.join(base, "pharmacy.csv.gz"),
                         dtype={"ndc": "category"}, low_memory=False)
    if "ndc" not in med_pd.columns:
        med_pd = pd.read_csv(os.path.join(base, "prescriptions.csv.gz"),
                             dtype={"ndc": "category"}, low_memory=False)
    keep_cols = ["subject_id", "hadm_id", "starttime", "ndc", "drug"]
    med_pd = med_pd[[c for c in keep_cols if c in med_pd.columns]]
    med_pd = med_pd[med_pd["ndc"].astype(str) != "0"]
    med_pd["ndc"] = med_pd["ndc"].astype(str).str.replace("-", "")
    med_pd.ffill(inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    if "starttime" in med_pd.columns:
        med_pd["starttime"] = pd.to_datetime(med_pd["starttime"], errors="coerce")
        med_pd.sort_values(["subject_id", "hadm_id", "starttime"], inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    ndc2rxcui = load_ndc2rxcui(cfg["ndc2rxcui_path"])
    med_pd["RXCUI"] = med_pd["ndc"].map(ndc2rxcui)
    med_pd.dropna(subset=["RXCUI"], inplace=True)

    rxcui2atc = pd.read_csv(cfg["rxcui2atc4_path"])
    rxcui2atc.drop(columns=["YEAR", "MONTH", "NDC"], inplace=True)
    rxcui2atc.drop_duplicates(subset=["RXCUI"], inplace=True)
    med_pd["RXCUI"] = med_pd["RXCUI"].astype("int64")
    med_pd = med_pd.merge(rxcui2atc, on="RXCUI")
    med_pd["ATC3"] = med_pd["ATC4"].str[:4]
    med_pd.drop(columns=["ndc", "RXCUI", "ATC4"], inplace=True, errors="ignore")
    med_pd.drop_duplicates(inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    top_med = (med_pd.groupby("ATC3").size().reset_index(name="cnt")
               .sort_values("cnt", ascending=False).head(300)["ATC3"])
    med_pd = med_pd[med_pd["ATC3"].isin(top_med)].reset_index(drop=True)

    # ---- diagnoses ----
    diag_pd = pd.read_csv(os.path.join(base, "diagnoses_icd.csv.gz"), low_memory=False)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["seq_num", "icd_version"], inplace=True, errors="ignore")
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(["subject_id", "hadm_id"], inplace=True)
    diag_pd.reset_index(drop=True, inplace=True)
    top_diag = (diag_pd.groupby("icd_code").size().reset_index(name="cnt")
                .sort_values("cnt", ascending=False).head(2000)["icd_code"])
    diag_pd = diag_pd[diag_pd["icd_code"].isin(top_diag)].reset_index(drop=True)

    # ---- procedures ----
    pro_pd = pd.read_csv(os.path.join(base, "procedures_icd.csv.gz"), low_memory=False)
    pro_pd.drop(columns=["icd_version", "chartdate"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(["subject_id", "hadm_id", "seq_num"], inplace=True)
    pro_pd.drop(columns=["seq_num"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    top_pro = (pro_pd.groupby("icd_code").size().reset_index(name="cnt")
               .sort_values("cnt", ascending=False).head(1001)["icd_code"])
    pro_pd = pro_pd[pro_pd["icd_code"].isin(top_pro)].reset_index(drop=True)
    pro_pd.reset_index(drop=True, inplace=True)

    # ---- admissions ----
    adm_pd = pd.read_csv(os.path.join(base, "admissions.csv.gz"), low_memory=False)
    profile_cols = ["subject_id", "hadm_id", "insurance", "language",
                    "marital_status", "race"]
    adm_pd = adm_pd[[c for c in profile_cols if c in adm_pd.columns]]
    adm_pd.fillna("Unknown", inplace=True)

    # ---- combine ----
    med_key  = med_pd[["subject_id", "hadm_id"]].drop_duplicates()
    diag_key = diag_pd[["subject_id", "hadm_id"]].drop_duplicates()
    pro_key  = pro_pd[["subject_id", "hadm_id"]].drop_duplicates()
    key = (med_key.merge(diag_key, on=["subject_id", "hadm_id"])
                  .merge(pro_key,  on=["subject_id", "hadm_id"]))

    diag_pd = diag_pd.merge(key, on=["subject_id", "hadm_id"])
    med_pd  = med_pd.merge(key,  on=["subject_id", "hadm_id"])
    pro_pd  = pro_pd.merge(key,  on=["subject_id", "hadm_id"])

    diag_agg = diag_pd.groupby(["subject_id", "hadm_id"])["icd_code"].unique().reset_index()
    med_agg  = med_pd.groupby(["subject_id", "hadm_id"])["ATC3"].unique().reset_index()
    pro_agg  = (pro_pd.groupby(["subject_id", "hadm_id"])["icd_code"].unique()
                .reset_index().rename(columns={"icd_code": "pro_code"}))

    data = (diag_agg.merge(med_agg, on=["subject_id", "hadm_id"])
                    .merge(pro_agg, on=["subject_id", "hadm_id"]))
    data["icd_code"] = data["icd_code"].map(list)
    data["ATC3"]     = data["ATC3"].map(list)
    data["pro_code"] = data["pro_code"].map(list)

    return data, adm_pd, "subject_id", "hadm_id", "icd_code", "pro_code"


# ---------------------------------------------------------------------------
# Build DDI adjacency and drug-fragment mask
# ---------------------------------------------------------------------------

def build_ddi_matrices(records, med_voc: Voc,
                       ddi_file: str, drug_atc_path: str,
                       ehr_adj_path: str, ddi_adj_path: str) -> np.ndarray:
    TOPK = 40
    med_voc_size = len(med_voc.idx2word)

    # map CID -> ATC4 codes
    cid2atc: dict[str, set] = defaultdict(set)
    atc3_atc4: dict[str, set] = defaultdict(set)
    for atc4 in med_voc.idx2word.values():
        atc3_atc4[atc4[:4]].add(atc4)

    with open(drug_atc_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            cid, atcs = parts[0], parts[1:]
            for atc in atcs:
                if len(atc3_atc4[atc[:4]]) != 0:
                    cid2atc[cid].add(atc[:4])

    # EHR co-occurrence
    ehr_adj = np.zeros((med_voc_size, med_voc_size))
    for patient in records:
        for adm in patient:
            meds = adm[2]
            for i, mi in enumerate(meds):
                for j, mj in enumerate(meds):
                    if j <= i:
                        continue
                    ii = med_voc.word2idx[mi]
                    jj = med_voc.word2idx[mj]
                    ehr_adj[ii, jj] = 1
                    ehr_adj[jj, ii] = 1
    dill.dump(ehr_adj, open(ehr_adj_path, "wb"))

    # DDI adjacency
    ddi_df = pd.read_csv(ddi_file)
    top_effects = (ddi_df.groupby(["Polypharmacy Side Effect", "Side Effect Name"])
                   .size().reset_index(name="cnt")
                   .sort_values("cnt", ascending=False).iloc[-TOPK:])
    ddi_df = ddi_df.merge(top_effects[["Side Effect Name"]], on="Side Effect Name")
    ddi_df = ddi_df[["STITCH 1", "STITCH 2"]].drop_duplicates().reset_index(drop=True)

    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for _, row in ddi_df.iterrows():
        for ai in cid2atc[row["STITCH 1"]]:
            for aj in cid2atc[row["STITCH 2"]]:
                for i_atc in atc3_atc4[ai]:
                    for j_atc in atc3_atc4[aj]:
                        vi, vj = med_voc.word2idx.get(i_atc), med_voc.word2idx.get(j_atc)
                        if vi is not None and vj is not None and vi != vj:
                            ddi_adj[vi, vj] = 1
                            ddi_adj[vj, vi] = 1
    dill.dump(ddi_adj, open(ddi_adj_path, "wb"))
    return ddi_adj


def build_ddi_mask(atc3tosmiles: dict, med_voc: Voc) -> np.ndarray:
    fraction_sets = []
    for k in med_voc.idx2word:
        atc = med_voc.idx2word[k]
        fracs = set()
        for smi in atc3tosmiles.get(atc, []):
            try:
                m = BRICS.BRICSDecompose(Chem.MolFromSmiles(smi))
                fracs.update(m)
            except Exception:
                pass
        fraction_sets.append(fracs)

    all_fracs = list(set(f for fs in fraction_sets for f in fs))
    mask = np.zeros((len(med_voc.idx2word), len(all_fracs)))
    for i, fracs in enumerate(fraction_sets):
        for frac in fracs:
            mask[i, all_fracs.index(frac)] = 1
    return mask


# ---------------------------------------------------------------------------
# Build patient records and profile tokenizer
# ---------------------------------------------------------------------------

def build_records_and_vocabs(data, adm_df,
                              subj_col, hadm_col,
                              diag_col, pro_col):
    diag_voc, med_voc, pro_voc = Voc(), Voc(), Voc()

    for _, row in data.iterrows():
        diag_voc.add_sentence([str(c) for c in row[diag_col]])
        med_voc.add_sentence([str(c) for c in row["ATC3"]])
        pro_voc.add_sentence([str(c) for c in row[pro_col]])

    records = []  # list of patients, each a list of admissions
    for subj_id in data[subj_col].unique():
        patient_df = data[data[subj_col] == subj_id].sort_values(hadm_col)
        patient = []
        for _, row in patient_df.iterrows():
            adm = [
                [str(c) for c in row[diag_col]],
                [str(c) for c in row[pro_col]],
                [str(c) for c in row["ATC3"]],
            ]
            patient.append(adm)
        records.append(patient)

    return records, diag_voc, med_voc, pro_voc


def build_profile_tokenizer(adm_df, subj_col, hadm_col) -> dict:
    # determine which profile columns are available
    skip_cols = {subj_col, hadm_col}
    profile_cols = [c for c in adm_df.columns if c not in skip_cols]

    word2idx: dict[str, dict] = {}
    for col in profile_cols:
        unique_vals = ["Unknown"] + sorted(adm_df[col].dropna().unique().tolist())
        word2idx[col] = {v: i for i, v in enumerate(unique_vals)}

    return {"word2idx": word2idx, "columns": profile_cols}


# ---------------------------------------------------------------------------
# Create JSONL records
# ---------------------------------------------------------------------------

def create_jsonl_records(data, adm_df, records,
                         diag_voc, med_voc, pro_voc,
                         profile_tokenizer,
                         subj_col, hadm_col, diag_col, pro_col) -> list:
    """Convert per-patient records into list of JSONL-ready dicts."""
    profile_cols = profile_tokenizer["columns"]
    word2idx = profile_tokenizer["word2idx"]

    # build hadm -> profile lookup
    adm_lookup = {}
    for _, row in adm_df.iterrows():
        adm_lookup[row[hadm_col]] = {
            col: word2idx[col].get(str(row[col]), word2idx[col]["Unknown"])
            for col in profile_cols
        }

    jsonl_records = []
    subj_ids = data[subj_col].unique()
    for subj_id, patient in zip(subj_ids, records):
        if len(patient) < 2:
            continue  # need at least 2 visits (one history + one label)
        patient_df = data[data[subj_col] == subj_id].sort_values(hadm_col)
        hadm_ids = patient_df[hadm_col].tolist()

        for t in range(1, len(patient)):
            # all visits up to and including t form the context; label is visit t
            history = patient[:t + 1]
            hadm_id = hadm_ids[t]

            profile_raw = adm_lookup.get(hadm_id, {col: 0 for col in profile_cols})

            rec = {
                "records": {
                    "diagnosis":  [adm[0] for adm in history],
                    "procedure":  [adm[1] for adm in history],
                    "medication": [adm[2] for adm in history],
                },
                "profile": {col: profile_raw[col] for col in profile_cols},
                "drug_code": list(history[-1][2]),
            }
            jsonl_records.append(rec)

    return jsonl_records


def split_and_save(records: list, output_dir: str, tag: str, seed: int = 42):
    rng = random.Random(seed)
    idx = list(range(len(records)))
    # sequential split — same as original LEADER
    n = len(idx)
    train_end = int(0.8 * n)
    val_end = train_end + int(0.1 * n)

    splits = {
        f"train_{tag}.json": records[:train_end],
        f"val_{tag}.json":   records[train_end:val_end],
        f"test_{tag}.json":  records[val_end:],
    }
    for fname, split_data in splits.items():
        path = os.path.join(output_dir, fname)
        with jsonlines.open(path, "w") as w:
            for item in split_data:
                w.write(item)
        print(f"  Wrote {len(split_data):,} records -> {path}")


# ---------------------------------------------------------------------------
# SMILES lookup
# ---------------------------------------------------------------------------

def build_atc3tosmiles(med_voc: Voc, drugbank_path: str) -> dict:
    drug_info = pd.read_csv(drugbank_path)
    drug2smiles = {}
    for _, row in drug_info[["name", "moldb_smiles"]].iterrows():
        if isinstance(row["moldb_smiles"], str):
            drug2smiles[row["name"].lower()] = row["moldb_smiles"]

    atc3tosmiles: dict[str, list] = {}
    # Simple heuristic: drug names often appear in drugbank by generic name
    # We just store empty lists for ATC3 codes not found — DDI mask still works
    for atc3 in med_voc.word2idx:
        atc3tosmiles[atc3] = []

    return atc3tosmiles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--dataset", default=None,
                        choices=["mimic3", "mimic4"],
                        help="Override dataset in config")
    parser.add_argument("--tag", default=None, help="Output file tag (default: dataset name)")
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    dataset = args.dataset or cfg.get("dataset", "mimic3")
    tag = args.tag or dataset
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[preprocess] dataset={dataset}  output_dir={output_dir}")

    # ---- load raw data ----
    if dataset == "mimic3":
        data, adm_df, subj_col, hadm_col, diag_col, pro_col = preprocess_mimic3(cfg)
    else:
        data, adm_df, subj_col, hadm_col, diag_col, pro_col = preprocess_mimic4(cfg)

    print(f"  Combined admissions: {len(data):,}  |  unique patients: {data[subj_col].nunique():,}")

    # ---- build vocabularies and raw records ----
    records, diag_voc, med_voc, pro_voc = build_records_and_vocabs(
        data, adm_df, subj_col, hadm_col, diag_col, pro_col)
    print(f"  Vocab sizes — diag: {len(diag_voc.word2idx)}  "
          f"med: {len(med_voc.word2idx)}  pro: {len(pro_voc.word2idx)}")

    voc_path = os.path.join(output_dir, "voc_final.pkl")
    dill.dump({"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
              open(voc_path, "wb"))
    print(f"  Saved vocabs -> {voc_path}")

    # ---- profile tokenizer ----
    profile_tokenizer = build_profile_tokenizer(adm_df, subj_col, hadm_col)
    prof_path = os.path.join(output_dir, "profile_dict.json")
    with open(prof_path, "w") as f:
        json.dump(profile_tokenizer, f)
    print(f"  Saved profile tokenizer -> {prof_path}")

    # ---- JSONL records ----
    jsonl_records = create_jsonl_records(
        data, adm_df, records, diag_voc, med_voc, pro_voc,
        profile_tokenizer, subj_col, hadm_col, diag_col, pro_col)
    print(f"  Total (patient, visit-target) pairs: {len(jsonl_records):,}")
    split_and_save(jsonl_records, output_dir, tag)

    # ---- DDI matrices ----
    print("  Building DDI adjacency matrices...")
    ddi_adj = build_ddi_matrices(
        records, med_voc,
        ddi_file=cfg["ddi_file_path"],
        drug_atc_path=cfg["drug_atc_path"],
        ehr_adj_path=os.path.join(output_dir, "ehr_adj_final.pkl"),
        ddi_adj_path=os.path.join(output_dir, "ddi_A_final.pkl"),
    )
    print(f"  DDI rate in training set: {ddi_adj.sum() / (len(med_voc.word2idx) ** 2):.4f}")

    # ---- SMILES + fragment mask ----
    print("  Building ATC3 -> SMILES mapping...")
    atc3tosmiles = build_atc3tosmiles(med_voc, cfg["drugbank_info_path"])
    dill.dump(atc3tosmiles, open(os.path.join(output_dir, "atc3toSMILES.pkl"), "wb"))

    print("  Building DDI fragment mask (ddi_mask_H)...")
    ddi_mask = build_ddi_mask(atc3tosmiles, med_voc)
    dill.dump(ddi_mask, open(os.path.join(output_dir, "ddi_mask_H.pkl"), "wb"))

    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
