"""
Data preprocessing for the LEADER pipeline.

Follows the LEADER-pytorch reference pipeline (data/mimic3/construction.ipynb + utils.py)
and produces JSONL records with the same schema as that notebook:

  {
    "input":      <str>   natural-language prompt (history + current visit context),
    "target":     <str>   comma-separated drug class names (last visit, prediction target),
    "subject_id": <int>   patient identifier,
    "drug_code":  <list>  ATC3 codes for last visit (prediction target),
    "records": {
        "diagnosis":  [[ICD9, ...], ...],   # all visits (incl. target)
        "procedure":  [[ICD9, ...], ...],
        "medication": [[ATC3, ...], ...]
    },
    "profile": {                            # raw strings from ADMISSIONS, first visit
        "INSURANCE": <str>, "LANGUAGE": <str>,
        "RELIGION": <str>, "MARITAL_STATUS": <str>, "ETHNICITY": <str>
    }
  }

Pipeline steps (matching construction.ipynb):
  1. Load/clean PRESCRIPTIONS; filter to multi-visit patients; map NDC->ATC3 (top-300)
  2. Build ATC3->SMILES (via DRUG column + drugbank); filter to ATC3 codes with SMILES
  3. Load/clean DIAGNOSES_ICD (top-2000); PROCEDURES_ICD (no frequency filter)
  4. Inner-join all three; aggregate per admission
  5. Join ADMISSIONS for profile columns
  6. Load ICD->description and ATC3->class-name decoders
  7. Build one JSONL record per patient; split 80/10/10
  8. Build vocab, EHR adj, DDI adj, BRICS mask

Usage:
  python scripts/preprocess.py --data configs/data.yaml
  python scripts/preprocess.py --data configs/data.yaml --dataset mimic4
"""

import argparse
import ast
import json
import os
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
    content = re.sub(r"\bu'", "'", content)
    return ast.literal_eval(content)


# ---------------------------------------------------------------------------
# Code -> description decoders
# ---------------------------------------------------------------------------

def load_atc3_names(atc_names_path: str) -> dict:
    """Load WHO ATC-DDD CSV; return dict ATC3(4-char) -> lowercase class name."""
    df = pd.read_csv(atc_names_path)
    df = df[df["atc_code"].str.len() == 4].copy()
    df["atc_name"] = df["atc_name"].str.lower()
    return dict(zip(df["atc_code"].values, df["atc_name"].values))


def load_icd_descriptions(mimic3_base: str) -> tuple[dict, dict]:
    """Return (icd2diag, icd2proc) short-title dicts from MIMIC-III D_ICD_*.csv files."""
    diag_df = pd.read_csv(os.path.join(mimic3_base, "D_ICD_DIAGNOSES.csv.gz"),
                          low_memory=False)
    proc_df = pd.read_csv(os.path.join(mimic3_base, "D_ICD_PROCEDURES.csv.gz"),
                          low_memory=False)
    icd2diag = dict(zip(diag_df["ICD9_CODE"].astype(str).values,
                        diag_df["SHORT_TITLE"].values))
    icd2proc = dict(zip(proc_df["ICD9_CODE"].astype(str).values,
                        proc_df["SHORT_TITLE"].values))
    return icd2diag, icd2proc


def decode(code_list: list, decoder: dict) -> list:
    """Map codes to descriptions; silently skip codes not in decoder."""
    result = []
    for code in code_list:
        desc = decoder.get(str(code))
        if desc is not None:
            result.append(desc)
    return result


def concat_str(str_list: list) -> str:
    """Join list of strings with ', '."""
    if not str_list:
        return ""
    return ", ".join(str_list)


# ---------------------------------------------------------------------------
# Prompt templates (from construction.ipynb cell-37)
# ---------------------------------------------------------------------------

MAIN_TEMPLATE = (
    "The patient has <VISIT_NUM> times ICU visits. \n "
    "<HISTORY>"
    "In this visit, he has diagnosis: <DIAGNOSIS>; procedures: <PROCEDURE>. "
    "Then, the patient should be prescribed: "
)
HIST_TEMPLATE = (
    "In <VISIT_NO> visit, the patient had diagnosis: <DIAGNOSIS>; "
    "procedures: <PROCEDURE>. The patient was prescribed drugs: <MEDICATION>. \n"
)


# ---------------------------------------------------------------------------
# Multi-visit filter
# ---------------------------------------------------------------------------

def process_visit_lg2(med_pd: pd.DataFrame, subj_col: str, hadm_col: str):
    """Return index of subject IDs that have ≥2 distinct admissions."""
    counts = med_pd[[subj_col, hadm_col]].groupby(subj_col)[hadm_col].nunique()
    return counts[counts > 1].index


# ---------------------------------------------------------------------------
# ATC3 -> drug name -> SMILES mapping
# ---------------------------------------------------------------------------

def atc3_to_drug_map(med_pd: pd.DataFrame, atc_col: str, drug_col: str) -> dict:
    """Build dict: ATC3 -> set of drug names from the DRUG column."""
    mapping: dict[str, set] = defaultdict(set)
    for atc3, drug in med_pd[[atc_col, drug_col]].values:
        mapping[atc3].add(drug)
    return dict(mapping)


def build_atc3tosmiles(atc3_to_drug: dict, druginfo: pd.DataFrame) -> dict:
    """Map ATC3 codes -> list of SMILES (up to 3) via drug names."""
    drug2smiles: dict[str, str] = {}
    for _, row in druginfo[["name", "moldb_smiles"]].iterrows():
        if isinstance(row["moldb_smiles"], str):
            drug2smiles[row["name"].lower()] = row["moldb_smiles"]

    atc3tosmiles: dict[str, list] = {}
    for atc3, drugs in atc3_to_drug.items():
        smiles_list = []
        for drug in drugs:
            smi = drug2smiles.get(str(drug).lower())
            if smi:
                smiles_list.append(smi)
        if smiles_list:
            atc3tosmiles[atc3] = smiles_list[:3]

    return atc3tosmiles


# ---------------------------------------------------------------------------
# MIMIC-III preprocessing
# ---------------------------------------------------------------------------

def preprocess_mimic3(cfg: dict) -> tuple:
    """
    Returns (data_df, subj_col, hadm_col, diag_col, pro_col, atc3tosmiles).
    data_df has ADMISSIONS profile columns merged in (INSURANCE, LANGUAGE, etc.).
    Follows construction.ipynb exactly.
    """
    base = cfg["mimic3_path"]

    # ---- medications ----
    med_pd = pd.read_csv(os.path.join(base, "PRESCRIPTIONS.csv.gz"),
                         dtype={"NDC": "category"}, low_memory=False)
    keep_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "NDC", "DRUG"]
    med_pd = med_pd[[c for c in keep_cols if c in med_pd.columns]]
    med_pd = med_pd[med_pd["NDC"] != "0"]
    med_pd["NDC"] = med_pd["NDC"].astype(str).str.replace("-", "")
    med_pd.ffill(inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd["STARTDATE"] = pd.to_datetime(med_pd["STARTDATE"],
                                          format="%Y-%m-%d %H:%M:%S", errors="coerce")
    sort_cols = ["SUBJECT_ID", "HADM_ID", "STARTDATE"]
    if "ICUSTAY_ID" in med_pd.columns:
        med_pd["ICUSTAY_ID"] = med_pd["ICUSTAY_ID"].astype("int64")
        sort_cols = ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE"]
    med_pd.sort_values(sort_cols, inplace=True)
    med_pd.drop(columns=["ICUSTAY_ID"], inplace=True, errors="ignore")
    med_pd.drop_duplicates(inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    # multi-visit filter (before NDC mapping, matching reference order)
    multi_ids = process_visit_lg2(med_pd, "SUBJECT_ID", "HADM_ID")
    med_pd = med_pd[med_pd["SUBJECT_ID"].isin(multi_ids)].reset_index(drop=True)

    # NDC -> RXCUI -> ATC3
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

    # top-300 ATC3
    top_med = (med_pd.groupby("ATC3").size().reset_index(name="cnt")
               .sort_values("cnt", ascending=False).iloc[:300]["ATC3"])
    med_pd = med_pd[med_pd["ATC3"].isin(top_med)].reset_index(drop=True)

    # ATC3 -> SMILES; filter to ATC3 codes with SMILES
    atc3_drug = atc3_to_drug_map(med_pd, atc_col="ATC3", drug_col="DRUG")
    druginfo = pd.read_csv(cfg["drugbank_info_path"])
    atc3tosmiles = build_atc3tosmiles(atc3_drug, druginfo)
    med_pd = med_pd[med_pd["ATC3"].isin(atc3tosmiles)].reset_index(drop=True)

    med_pd.drop(columns=["DRUG", "STARTDATE"], inplace=True, errors="ignore")
    med_pd.drop_duplicates(inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    # ---- diagnoses (top-2000) ----
    diag_pd = pd.read_csv(os.path.join(base, "DIAGNOSES_ICD.csv.gz"), low_memory=False)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["ROW_ID", "SEQ_NUM"], inplace=True, errors="ignore")
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(["SUBJECT_ID", "HADM_ID"], inplace=True)
    diag_pd.reset_index(drop=True, inplace=True)
    top_diag = (diag_pd.groupby("ICD9_CODE").size().reset_index(name="cnt")
                .sort_values("cnt", ascending=False).iloc[:2000]["ICD9_CODE"])
    diag_pd = diag_pd[diag_pd["ICD9_CODE"].isin(top_diag)].reset_index(drop=True)

    # ---- procedures (NOT frequency-filtered, matches reference) ----
    pro_pd = pd.read_csv(os.path.join(base, "PROCEDURES_ICD.csv.gz"), low_memory=False)
    pro_pd.drop(columns=["ROW_ID"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], inplace=True)
    pro_pd.drop(columns=["SEQ_NUM"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    # ---- admissions (profile) ----
    adm_pd = pd.read_csv(os.path.join(base, "ADMISSIONS.csv.gz"), low_memory=False)
    profile_cols = ["HADM_ID", "INSURANCE", "LANGUAGE", "RELIGION",
                    "MARITAL_STATUS", "ETHNICITY"]
    adm_pd = adm_pd[[c for c in profile_cols if c in adm_pd.columns]]
    adm_pd.fillna("unknown", inplace=True)

    # ---- combine (inner join) ----
    med_key  = med_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    diag_key = diag_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    pro_key  = pro_pd[["SUBJECT_ID", "HADM_ID"]].drop_duplicates()
    key = (med_key.merge(diag_key, on=["SUBJECT_ID", "HADM_ID"])
                  .merge(pro_key,  on=["SUBJECT_ID", "HADM_ID"]))

    diag_pd = diag_pd.merge(key, on=["SUBJECT_ID", "HADM_ID"])
    med_pd  = med_pd.merge(key, on=["SUBJECT_ID", "HADM_ID"])
    pro_pd  = pro_pd.merge(key, on=["SUBJECT_ID", "HADM_ID"])

    diag_agg = diag_pd.groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"].unique().reset_index()
    med_agg  = med_pd.groupby(["SUBJECT_ID", "HADM_ID"])["ATC3"].unique().reset_index()
    pro_agg  = (pro_pd.groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_CODE"].unique()
                .reset_index().rename(columns={"ICD9_CODE": "PRO_CODE"}))

    data = (diag_agg.merge(med_agg, on=["SUBJECT_ID", "HADM_ID"])
                    .merge(pro_agg, on=["SUBJECT_ID", "HADM_ID"]))
    data["ICD9_CODE"] = data["ICD9_CODE"].map(list)
    data["ATC3"]      = data["ATC3"].map(list)
    data["PRO_CODE"]  = data["PRO_CODE"].map(list)

    # merge profile into data (left join on HADM_ID, like get_side in notebook)
    data = data.merge(adm_pd, on="HADM_ID", how="left")
    data.fillna("unknown", inplace=True)

    return data, "SUBJECT_ID", "HADM_ID", "ICD9_CODE", "PRO_CODE", atc3tosmiles


# ---------------------------------------------------------------------------
# MIMIC-IV preprocessing
# ---------------------------------------------------------------------------

def preprocess_mimic4(cfg: dict) -> tuple:
    """
    Returns (data_df, subj_col, hadm_col, diag_col, pro_col, atc3tosmiles).
    Mirrors the MIMIC-III pipeline for MIMIC-IV column names.
    """
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

    multi_ids = process_visit_lg2(med_pd, "subject_id", "hadm_id")
    med_pd = med_pd[med_pd["subject_id"].isin(multi_ids)].reset_index(drop=True)

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
               .sort_values("cnt", ascending=False).iloc[:300]["ATC3"])
    med_pd = med_pd[med_pd["ATC3"].isin(top_med)].reset_index(drop=True)

    drug_col = "drug" if "drug" in med_pd.columns else None
    if drug_col:
        atc3_drug = atc3_to_drug_map(med_pd, atc_col="ATC3", drug_col=drug_col)
        druginfo = pd.read_csv(cfg["drugbank_info_path"])
        atc3tosmiles = build_atc3tosmiles(atc3_drug, druginfo)
        med_pd = med_pd[med_pd["ATC3"].isin(atc3tosmiles)].reset_index(drop=True)
    else:
        atc3tosmiles = {}

    med_pd.drop(columns=["drug", "starttime"], inplace=True, errors="ignore")
    med_pd.drop_duplicates(inplace=True)
    med_pd.reset_index(drop=True, inplace=True)

    # ---- diagnoses (top-2000) ----
    diag_pd = pd.read_csv(os.path.join(base, "diagnoses_icd.csv.gz"), low_memory=False)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=["seq_num", "icd_version"], inplace=True, errors="ignore")
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(["subject_id", "hadm_id"], inplace=True)
    diag_pd.reset_index(drop=True, inplace=True)
    top_diag = (diag_pd.groupby("icd_code").size().reset_index(name="cnt")
                .sort_values("cnt", ascending=False).iloc[:2000]["icd_code"])
    diag_pd = diag_pd[diag_pd["icd_code"].isin(top_diag)].reset_index(drop=True)

    # ---- procedures (NOT frequency-filtered) ----
    pro_pd = pd.read_csv(os.path.join(base, "procedures_icd.csv.gz"), low_memory=False)
    pro_pd.drop(columns=["icd_version", "chartdate"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(["subject_id", "hadm_id", "seq_num"], inplace=True)
    pro_pd.drop(columns=["seq_num"], inplace=True, errors="ignore")
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    # ---- admissions (profile) ----
    adm_pd = pd.read_csv(os.path.join(base, "admissions.csv.gz"), low_memory=False)
    profile_cols = ["hadm_id", "insurance", "language", "marital_status", "race"]
    adm_pd = adm_pd[[c for c in profile_cols if c in adm_pd.columns]]
    adm_pd.fillna("unknown", inplace=True)

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

    data = data.merge(adm_pd, on="hadm_id", how="left")
    data.fillna("unknown", inplace=True)

    return data, "subject_id", "hadm_id", "icd_code", "pro_code", atc3tosmiles


# ---------------------------------------------------------------------------
# Build vocabularies and patient records
# ---------------------------------------------------------------------------

def build_records_and_vocabs(data, subj_col, hadm_col, diag_col, pro_col):
    diag_voc, med_voc, pro_voc = Voc(), Voc(), Voc()

    for _, row in data.iterrows():
        diag_voc.add_sentence([str(c) for c in row[diag_col]])
        med_voc.add_sentence([str(c) for c in row["ATC3"]])
        pro_voc.add_sentence([str(c) for c in row[pro_col]])

    records = []
    for subj_id in data[subj_col].unique():
        patient_df = data[data[subj_col] == subj_id].sort_values(hadm_col)
        patient = []
        for _, row in patient_df.iterrows():
            patient.append([
                [str(c) for c in row[diag_col]],
                [str(c) for c in row[pro_col]],
                [str(c) for c in row["ATC3"]],
            ])
        records.append(patient)

    return records, diag_voc, med_voc, pro_voc


# ---------------------------------------------------------------------------
# Build DDI adjacency and drug-fragment mask
# ---------------------------------------------------------------------------

def build_ddi_matrices(records, med_voc: Voc,
                       ddi_file: str, drug_atc_path: str,
                       ehr_adj_path: str, ddi_adj_path: str) -> np.ndarray:
    TOPK = 40
    med_voc_size = len(med_voc.idx2word)

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

    # DDI adjacency (bottom-TOPK by frequency, matching reference .iloc[-TOPK:])
    ddi_df = pd.read_csv(ddi_file)
    ddi_most = (ddi_df.groupby(["Polypharmacy Side Effect", "Side Effect Name"])
                .size().reset_index(name="cnt")
                .sort_values("cnt", ascending=False)
                .reset_index(drop=True))
    ddi_most = ddi_most.iloc[-TOPK:]
    ddi_df = (ddi_df.merge(ddi_most[["Side Effect Name"]], on="Side Effect Name")
              [["STITCH 1", "STITCH 2"]].drop_duplicates().reset_index(drop=True))

    ddi_adj = np.zeros((med_voc_size, med_voc_size))
    for _, row in ddi_df.iterrows():
        for ai in cid2atc[row["STITCH 1"]]:
            for aj in cid2atc[row["STITCH 2"]]:
                for i_atc in atc3_atc4[ai]:
                    for j_atc in atc3_atc4[aj]:
                        vi = med_voc.word2idx.get(i_atc)
                        vj = med_voc.word2idx.get(j_atc)
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
# Create JSONL records — ONE per patient, matching construction.ipynb cell-40
# ---------------------------------------------------------------------------

def _get_profile_cols(data, subj_col, hadm_col, diag_col, pro_col):
    """Return list of profile column names present in data."""
    skip = {subj_col, hadm_col, diag_col, pro_col, "ATC3"}
    return [c for c in data.columns if c not in skip]


def create_jsonl_records(data, records,
                         subj_col, hadm_col, diag_col, pro_col,
                         icd2diag: dict, icd2proc: dict,
                         atc3_names: dict) -> list:
    """
    One JSONL record per patient, matching construction.ipynb output schema:
      - input:      natural language prompt
      - target:     comma-separated drug class names (last visit)
      - subject_id: int
      - drug_code:  ATC3 list (last visit)
      - records:    all visits {diagnosis, procedure, medication}
      - profile:    raw profile strings from first admission
    """
    profile_cols = _get_profile_cols(data, subj_col, hadm_col, diag_col, pro_col)

    jsonl_records = []
    subj_ids = list(data[subj_col].unique())

    for subj_id, patient in zip(subj_ids, records):
        item_df = data[data[subj_col] == subj_id].sort_values(hadm_col)
        visit_num = len(item_df) - 1  # number of historical visits (excl. current)

        # profile from first admission (item_df.iloc[0])
        first_row = item_df.iloc[0]
        patient_profile = {col: str(first_row[col]) for col in profile_cols}

        # build per-visit prompt strings (history only, current visit popped)
        hist_strings = []
        last_drug_names = last_diag_names = last_proc_names = ""
        for visit_no, (_, row) in enumerate(item_df.iterrows()):
            drug_names = concat_str(decode(row["ATC3"], atc3_names))
            diag_names = concat_str(decode(row[diag_col], icd2diag))
            proc_names = concat_str(decode(row[pro_col], icd2proc))
            hist_str = (HIST_TEMPLATE
                        .replace("<VISIT_NO>", str(visit_no + 1))
                        .replace("<DIAGNOSIS>", diag_names)
                        .replace("<PROCEDURE>", proc_names)
                        .replace("<MEDICATION>", drug_names))
            hist_strings.append(hist_str)
            # keep last visit values for current-visit slot in main template
            last_drug_names = drug_names
            last_diag_names = diag_names
            last_proc_names = proc_names

        # pop the last visit from history (it becomes the current/target visit)
        hist_strings.pop()

        # cap history at 3 visits (matching reference)
        if len(hist_strings) > 3:
            hist_strings = hist_strings[-3:]

        hist_str_combined = "".join(hist_strings)
        input_str = (MAIN_TEMPLATE
                     .replace("<VISIT_NUM>", str(visit_num))
                     .replace("<HISTORY>", hist_str_combined)
                     .replace("<DIAGNOSIS>", last_diag_names)
                     .replace("<PROCEDURE>", last_proc_names))

        # drug_code and target from last visit
        last_row = item_df.iloc[-1]
        drug_code = [str(x) for x in last_row["ATC3"]]
        target = last_drug_names

        # records: all visits (incl. target visit)
        hist = {"diagnosis": [], "procedure": [], "medication": []}
        for _, row in item_df.iterrows():
            hist["diagnosis"].append([str(x) for x in row[diag_col]])
            hist["procedure"].append([str(x) for x in row[pro_col]])
            hist["medication"].append([str(x) for x in row["ATC3"]])

        jsonl_records.append({
            "input":      input_str,
            "target":     target,
            "subject_id": int(subj_id),
            "drug_code":  drug_code,
            "records":    hist,
            "profile":    patient_profile,
        })

    return jsonl_records


def split_and_save(records: list, output_dir: str, tag: str):
    n = len(records)
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
# Profile tokenizer (saved separately for downstream model use)
# ---------------------------------------------------------------------------

def build_profile_tokenizer(data, subj_col, hadm_col, diag_col, pro_col) -> dict:
    profile_cols = _get_profile_cols(data, subj_col, hadm_col, diag_col, pro_col)
    word2idx: dict[str, dict] = {}
    for col in profile_cols:
        unique_vals = ["unknown"] + sorted(
            v for v in data[col].dropna().unique() if v != "unknown")
        word2idx[col] = {v: i for i, v in enumerate(unique_vals)}
    return {"word2idx": word2idx, "columns": profile_cols}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--dataset", default=None, choices=["mimic3", "mimic4"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.data)
    dataset = args.dataset or cfg.get("dataset", "mimic3")
    tag = args.tag or dataset
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[preprocess] dataset={dataset}  output_dir={output_dir}")

    # ---- load and clean raw data ----
    if dataset == "mimic3":
        data, subj_col, hadm_col, diag_col, pro_col, atc3tosmiles = preprocess_mimic3(cfg)
    else:
        data, subj_col, hadm_col, diag_col, pro_col, atc3tosmiles = preprocess_mimic4(cfg)

    print(f"  Admissions: {len(data):,}  |  patients: {data[subj_col].nunique():,}  "
          f"|  ATC3 with SMILES: {len(atc3tosmiles)}")

    # ---- code -> description decoders ----
    atc3_names = load_atc3_names(cfg["atc_names_path"])
    if dataset == "mimic3":
        icd2diag, icd2proc = load_icd_descriptions(cfg["mimic3_path"])
    else:
        # MIMIC-IV: ICD descriptions not available in same format; use empty dicts
        icd2diag, icd2proc = {}, {}

    # ---- build vocabularies and raw records ----
    records, diag_voc, med_voc, pro_voc = build_records_and_vocabs(
        data, subj_col, hadm_col, diag_col, pro_col)
    print(f"  Vocab — diag: {len(diag_voc.word2idx)}  "
          f"med: {len(med_voc.word2idx)}  pro: {len(pro_voc.word2idx)}")

    voc_path = os.path.join(output_dir, "voc_final.pkl")
    dill.dump({"diag_voc": diag_voc, "med_voc": med_voc, "pro_voc": pro_voc},
              open(voc_path, "wb"))
    print(f"  Saved vocabs -> {voc_path}")

    # ---- profile tokenizer (for downstream model use) ----
    profile_tokenizer = build_profile_tokenizer(data, subj_col, hadm_col, diag_col, pro_col)
    prof_path = os.path.join(output_dir, "profile_dict.json")
    with open(prof_path, "w") as f:
        json.dump(profile_tokenizer, f)
    print(f"  Saved profile tokenizer -> {prof_path}")

    # ---- JSONL records (one per patient, matching notebook schema) ----
    jsonl_records = create_jsonl_records(
        data, records, subj_col, hadm_col, diag_col, pro_col,
        icd2diag, icd2proc, atc3_names)
    print(f"  Patients -> JSONL records: {len(jsonl_records):,}")
    split_and_save(jsonl_records, output_dir, tag)

    # ---- ATC3 -> SMILES ----
    dill.dump(atc3tosmiles, open(os.path.join(output_dir, "atc3toSMILES.pkl"), "wb"))
    print(f"  Saved atc3toSMILES ({len(atc3tosmiles)} entries)")

    # ---- DDI matrices ----
    print("  Building DDI / EHR adjacency matrices...")
    ddi_adj = build_ddi_matrices(
        records, med_voc,
        ddi_file=cfg["ddi_file_path"],
        drug_atc_path=cfg["drug_atc_path"],
        ehr_adj_path=os.path.join(output_dir, "ehr_adj_final.pkl"),
        ddi_adj_path=os.path.join(output_dir, "ddi_A_final.pkl"),
    )
    print(f"  DDI rate: {ddi_adj.sum() / (len(med_voc.word2idx) ** 2):.4f}")

    # ---- BRICS fragment mask ----
    print("  Building ddi_mask_H...")
    ddi_mask = build_ddi_mask(atc3tosmiles, med_voc)
    dill.dump(ddi_mask, open(os.path.join(output_dir, "ddi_mask_H.pkl"), "wb"))

    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
