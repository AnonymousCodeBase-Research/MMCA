# script.py — DeepER grid runner (TRAIN/VALID per-epoch logs, TEST summary)
# Checkpoints: runs/checkpoints/<dataset_slug>/idx<fold>/<run_id>_best.{pth,json}
# Results CSV: runs/results.csv (appends all rows)

from __future__ import annotations
import os, re, json, csv, time, random, hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

from load_embeddings import load_embeddings as fast_load_embeddings
from DeepER import DeepERModel  # keep your DeepER implementation

def refit_on_trainvalid_and_test(dataset: str,
                                 fold: int,
                                 cfg: Dict[str, any],
                                 best_epoch: int,
                                 best_thr: float,
                                 embedding,
                                 vocab,
                                 MAX_LEN: int,
                                 RESULTS_CSV: str,
                                 CKPT_ROOT: str,
                                 cols: List[str],
                                 ds_slug: str,
                                 te_df: pd.DataFrame):
    """
    Merge TRAIN+VALID for this fold, refit for 'best_epoch' with best cfg, test once,
    and write a single 'test_refit' row.
    """
    # Reload splits to build the merged TRAIN+VALID quickly
    tr_df, va_df, _ = load_splits(dataset, fold)
    tr_df = ensure_left_right_and_label(tr_df)
    va_df = ensure_left_right_and_label(va_df)
    tv_df = pd.concat([tr_df, va_df], ignore_index=True)

    # Build datasets / loaders
    tv_ds = ERPairDataset(tv_df, vocab, MAX_LEN)
    te_ds = ERPairDataset(te_df, vocab, MAX_LEN)
    num_attrs = len(tv_ds.attr_names)

    tv_loader = DataLoader(tv_ds, batch_size=cfg["batch"], shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=cfg["batch"], shuffle=False)

    # Model with chosen hyperparams
    model = DeepERModel(
        embedding=embedding,
        num_attrs=num_attrs,
        encoder=cfg["e"], sim=cfg["s"], head=cfg["h"],
        enc_hidden=cfg["enc_hidden"], enc_layers=cfg["enc_layers"],
        sim_hidden=cfg["sim_hidden"], dropout=cfg["dropout"]
    ).to(device)

    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["wd"]
    )
    crit = nn.BCELoss()

    # Refit exactly 'best_epoch' epochs (no peeking)
    for _ in range(best_epoch):
        _ = train_epoch(model, tv_loader, optim, crit)

    # Save a refit checkpoint (optional but useful)
    ckpt_dir = ensure_dir(os.path.join(CKPT_ROOT, ds_slug, f"idx{fold}"))
    refit_pth = os.path.join(ckpt_dir, f"{make_run_id(dataset, fold, cfg)}_refit_best.pth")
    torch.save(model.state_dict(), refit_pth)

    # Final TEST at locked threshold
    te_scores, te_y = forward_scores(model, te_loader)
    te_prec, te_rec, te_f1, te_acc, te_pra, te_loss = metrics_at(te_scores, te_y, thr=best_thr)

    # Write test_refit row
    run_id = make_run_id(dataset, fold, cfg)
    row = base_row(dataset, "test_refit", fold, run_id, cfg)
    row.update(dict(epoch=best_epoch, epochs_trained=best_epoch, threshold=best_thr,
                    precision=te_prec*100, recall=te_rec*100, f1=te_f1*100,
                    accuracy=te_acc*100, pr_auc=te_pra, loss=te_loss,
                    checkpoint_path=os.path.relpath(refit_pth).replace("\\","/")))
    append_row(RESULTS_CSV, cols, row)


# -----------------------------
# Repro & device
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Datasets to run
# -----------------------------
DATASETS = [
    "datasets/PII_Textual/CC_Textual/", "datasets/PII_Textual/Senior_Textual/", "datasets/PII_Textual/SSN_Textual/"
]

# -----------------------------
# Results & checkpoints layout
# -----------------------------
RESULTS_CSV  = os.path.join("runs","results.csv")
CKPT_ROOT    = os.path.join("runs","checkpoints")

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def dataset_slug(p: str) -> str:
    return p.strip("/").replace("/", "_")

def make_run_id(dataset: str, fold: int, cfg: Dict[str, any]) -> str:
    blob = json.dumps(cfg, sort_keys=True)
    h = hashlib.md5(blob.encode()).hexdigest()[:6]
    return f"{dataset_slug(dataset)}_{fold}_{h}"

# -----------------------------
# Tokenization & preprocessing
# -----------------------------
PAD, UNK = "<pad>", "<unk>"

def tokenize(s: str) -> List[str]:
    return str(s).lower().split()

def preprocess_series(col: pd.Series) -> List[List[str]]:
    return [tokenize(x) for x in col.astype(str).tolist()]

def parse_attributes(raw_text_tokens: List[str]) -> Dict[str, List[str]]:
    return {"text": raw_text_tokens}  # single attr; extend if needed

# -----------------------------
# Vocab
# -----------------------------
def build_vocab(all_token_lists: List[List[str]], min_freq: int = 1) -> Dict[str,int]:
    counter = Counter()
    for toks in all_token_lists: counter.update(toks)
    vocab = {PAD:0, UNK:1}
    for w,c in counter.items():
        if c >= min_freq and w not in vocab: vocab[w] = len(vocab)
    return vocab

def numericalize(tokens: List[str], vocab: Dict[str,int], max_len: int) -> Tuple[List[int], int]:
    ids = [vocab.get(t, vocab[UNK]) for t in tokens][:max_len]
    L = len(ids)
    if L < max_len: ids += [vocab[PAD]]*(max_len-L)
    return ids, L

# -----------------------------
# Robust schema handling
# -----------------------------
LABEL_CANDIDATES = ["label","gold","y","match","is_match","target"]
LEFT_SINGLE_CANDS  = ["left_","left","left_text","l_text","text_left"]
RIGHT_SINGLE_CANDS = ["right_","right","right_text","r_text","text_right"]
EXCLUDE_PATTERNS = [r"(^|_)id($|_)", r"(^|_)pair(_|$)", r"(^|_)index($|_)",
                    r"(^|_)label($|_)", r"(^|_)match($|_)", r"(^|_)is_match($|_)", r"(^|_)target($|_)"]

def _find_label_column(df: pd.DataFrame) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in lower: return lower[cand]
    for c in df.columns:
        s = df[c].dropna().astype(str).str.lower()
        if s.isin(["0","1","true","false","yes","no"]).all(): return c
    raise KeyError(f"Could not infer label column. Available: {list(df.columns)}")

def _first_present(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand in lower: return lower[cand]
    return None

def _matches_any(patterns: List[str], col: str) -> bool:
    return any(re.search(p, col, flags=re.IGNORECASE) for p in patterns)

def _concat_group(df: pd.DataFrame, prefixes: List[str]) -> Optional[pd.Series]:
    chosen = []
    for c in df.columns:
        lc = c.lower()
        if any(lc.startswith(p) for p in prefixes) and not _matches_any(EXCLUDE_PATTERNS, lc):
            chosen.append(c)
    if not chosen: return None
    chosen = sorted(chosen)
    return df[chosen].astype(str).apply(lambda r: " ".join([x for x in r.values if x and x!="nan"]), axis=1)

def ensure_left_right_and_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ycol = _find_label_column(out)
    y = out[ycol]
    if y.dtype == bool: out["label"] = y.astype(int)
    else:
        s = y.astype(str).str.strip().str.lower()
        m = {"1":1,"0":0,"true":1,"false":0,"yes":1,"no":0}
        out["label"] = s.map(m) if s.isin(m.keys()).all() else pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
    left_single  = _first_present(out, LEFT_SINGLE_CANDS)
    right_single = _first_present(out, RIGHT_SINGLE_CANDS)
    left_group   = _concat_group(out,  ["left_","l_"])
    right_group  = _concat_group(out, ["right_","r_"])
    if "left_" not in out.columns:
        if left_single is not None: out["left_"] = out[left_single].astype(str)
        elif left_group is not None: out["left_"] = left_group
        else: raise KeyError("Could not infer left_ column")
    if "right_" not in out.columns:
        if right_single is not None: out["right_"] = out[right_single].astype(str)
        elif right_group is not None: out["right_"] = right_group
        else: raise KeyError("Could not infer right_ column")
    return out[["left_","right_","label"]]

# -----------------------------
# Dataset
# -----------------------------
class ERPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Dict[str,int], max_len: int):
        df = ensure_left_right_and_label(df)
        self.vocab, self.max_len = vocab, max_len
        self.left_tokens  = preprocess_series(df["left_"])
        self.right_tokens = preprocess_series(df["right_"])
        self.labels       = df["label"].astype(int).tolist()
        self.left_attrs  = [parse_attributes(toks) for toks in self.left_tokens]
        self.right_attrs = [parse_attributes(toks) for toks in self.right_tokens]
        names = set()
        for d in self.left_attrs + self.right_attrs: names |= set(d.keys())
        self.attr_names = sorted(list(names))
        self.left_attr_ids, self.left_attr_lens = [], []
        self.right_attr_ids, self.right_attr_lens = [], []
        for i in range(len(self.labels)):
            L_ids, L_len, R_ids, R_len = [], [], [], []
            for a in self.attr_names:
                li, ll = numericalize(self.left_attrs[i].get(a, []),  vocab, max_len)
                ri, rl = numericalize(self.right_attrs[i].get(a, []), vocab, max_len)
                L_ids.append(li); L_len.append(ll)
                R_ids.append(ri); R_len.append(rl)
            self.left_attr_ids.append(L_ids);  self.left_attr_lens.append(L_len)
            self.right_attr_ids.append(R_ids); self.right_attr_lens.append(R_len)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        la = torch.tensor(self.left_attr_ids[idx],  dtype=torch.long)
        ll = torch.tensor(self.left_attr_lens[idx], dtype=torch.long)
        ra = torch.tensor(self.right_attr_ids[idx], dtype=torch.long)
        rl = torch.tensor(self.right_attr_lens[idx],dtype=torch.long)
        y  = torch.tensor(self.labels[idx], dtype=torch.float)
        return la, ll, ra, rl, y

# -----------------------------
# IO helpers
# -----------------------------
def load_splits(base: str, fold: int):
    tr = pd.read_csv(os.path.join(base, f"train_{fold}.csv"))
    va = pd.read_csv(os.path.join(base, f"valid_{fold}.csv"))
    te = pd.read_csv(os.path.join(base, f"test_{fold}.csv"))
    return tr, va, te

def ensure_results_csv(path: str):
    cols = [
        "dataset","split","index","run_id","timestamp",
        "batch_size","lr","enc_hidden","enc_layers",
        "epoch","epochs_trained","threshold",
        "precision","recall","f1","accuracy","pr_auc","loss",
        "checkpoint_path"
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    return cols

def append_row(path: str, cols: List[str], row: Dict[str, any]):
    with open(path,"a",newline="",encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=cols).writerow(row)

def base_row(dataset, split, idx, run_id, cfg) -> Dict[str, any]:
    return dict(
        dataset=dataset, split=split, index=idx, run_id=run_id,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        batch_size=cfg["batch"], lr=cfg["lr"],
        enc_hidden=cfg["enc_hidden"], enc_layers=cfg["enc_layers"],
        epoch=None, epochs_trained=None, threshold=None,
        precision=None, recall=None, f1=None, accuracy=None, pr_auc=None, loss=None,
        checkpoint_path=""
    )

# -----------------------------
# Training / evaluation
# -----------------------------
def train_epoch(model, loader, optim, crit):
    model.train()
    total = 0.0
    for la,ll,ra,rl,y in loader:
        la,ll,ra,rl,y = la.to(device),ll.to(device),ra.to(device),rl.to(device),y.to(device)
        optim.zero_grad()
        p = model(la,ll,ra,rl)
        loss = crit(p, y)
        loss.backward(); optim.step()
        total += loss.item()
    return total / max(1,len(loader))

@torch.no_grad()
def forward_scores(model, loader):
    model.eval()
    scores, labels = [], []
    for la,ll,ra,rl,y in loader:
        la,ll,ra,rl = la.to(device),ll.to(device),ra.to(device),rl.to(device)
        p = model(la,ll,ra,rl)
        scores.extend(p.cpu().tolist())
        labels.extend(y.tolist())
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)

def metrics_at(scores: np.ndarray, y: np.ndarray, thr: float):
    yp = (scores >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yp, average="binary", zero_division=0)
    acc = accuracy_score(y, yp)
    try: pra = roc_auc_score(y, scores)
    except: pra = float("nan")
    loss = -np.mean(y*np.log(np.clip(scores,1e-12,1-1e-12)) + (1-y)*np.log(np.clip(1-scores,1e-12,1)))
    return prec, rec, f1, acc, pra, loss

def sweep_best(scores: np.ndarray, y: np.ndarray):
    best = None
    for thr in np.linspace(0.05, 0.95, 91):
        prec, rec, f1, acc, pra, loss = metrics_at(scores, y, thr)
        if best is None or f1 > best["f1"]:
            best = dict(thr=float(thr), prec=prec, rec=rec, f1=f1, acc=acc, pra=pra, loss=loss)
    return best

# -----------------------------
# Protocol & grid
# -----------------------------
MAX_LEN   = 256
EPOCHS    = 64
PATIENCE  = 5
MIN_DELTA = 1e-6

GRID = []
for b in [16,32,64]:
    for lr in [0.001, 0.0001]:
        for h in [128, 256, 512]:
            for l in [1,2,3]:
                GRID.append(dict(
                    e="lstm", s="hadamard", h="linear",
                    batch=b, lr=lr, wd=1e-3, dropout=0.0,
                    enc_hidden=h, enc_layers=l,
                    sim_hidden=50, finetune_emb=True
                ))

# -----------------------------
# Main
# -----------------------------
def main():
    print(f"Using device: {device}")
    cols = ensure_results_csv(RESULTS_CSV)

    for dataset in DATASETS:
        ds_slug = dataset_slug(dataset)
        for fold in range(10):
            # Load splits & normalize schema
            tr_df, va_df, te_df = load_splits(dataset, fold)
            tr_df = ensure_left_right_and_label(tr_df)
            va_df = ensure_left_right_and_label(va_df)
            te_df = ensure_left_right_and_label(te_df)

            # Build vocab on all text of this fold
            all_text = pd.concat([
                tr_df["left_"], tr_df["right_"],
                va_df["left_"], va_df["right_"],
                te_df["left_"], te_df["right_"]
            ]).astype(str)
            vocab = build_vocab(preprocess_series(all_text), min_freq=1)

            # Embeddings (per-dataset cache)
            embedding = fast_load_embeddings(
                vocab=vocab, emb_path="wiki.en.kv", dim=300,
                cache_dir=os.path.join(dataset, "emb_cache"),
                finetune=True
            )

            # Datasets / loaders
            tr_ds = ERPairDataset(tr_df, vocab, MAX_LEN)
            va_ds = ERPairDataset(va_df, vocab, MAX_LEN)
            te_ds = ERPairDataset(te_df, vocab, MAX_LEN)
            num_attrs = len(tr_ds.attr_names)

            # ---- Track fold-best across the grid ----
            fold_best = dict(f1=-1.0, epoch=None, thr=None, cfg=None)

            for cfg in GRID:
                run_id = make_run_id(dataset, fold, cfg)
                ckpt_dir = ensure_dir(os.path.join(CKPT_ROOT, ds_slug, f"idx{fold}"))
                best_pth  = os.path.join(ckpt_dir, f"{run_id}_best.pth")
                best_meta = os.path.join(ckpt_dir, f"{run_id}_best.json")

                tr_loader = DataLoader(tr_ds, batch_size=cfg["batch"], shuffle=True)
                va_loader = DataLoader(va_ds, batch_size=cfg["batch"], shuffle=False)

                # Model
                model = DeepERModel(
                    embedding=embedding,
                    num_attrs=num_attrs,
                    encoder=cfg["e"], sim=cfg["s"], head=cfg["h"],
                    enc_hidden=cfg["enc_hidden"], enc_layers=cfg["enc_layers"],
                    sim_hidden=cfg["sim_hidden"], dropout=cfg["dropout"]
                ).to(device)

                optim = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=cfg["lr"], weight_decay=cfg["wd"]
                )
                crit = nn.BCELoss()

                best_f1, best_epoch, best_thr = -1.0, None, None
                stale = 0

                # ----- Train epochs (log TRAIN + VALID, choose best on VALID) -----
                for epoch in range(1, EPOCHS+1):
                    tr_loss = train_epoch(model, tr_loader, optim, crit)

                    # TRAIN metrics (quick)
                    tr_scores, tr_y = forward_scores(model, tr_loader)
                    tr_prec, tr_rec, tr_f1, tr_acc, _, _ = metrics_at(tr_scores, tr_y, thr=0.5)

                    # VALID sweep
                    va_scores, va_y = forward_scores(model, va_loader)
                    best = sweep_best(va_scores, va_y)

                    # TRAIN row
                    row = base_row(dataset, "train", fold, run_id, cfg)
                    row.update(dict(epoch=epoch, precision=tr_prec*100, recall=tr_rec*100, f1=tr_f1*100,
                                    accuracy=tr_acc*100, loss=tr_loss))
                    append_row(RESULTS_CSV, cols, row)

                    # VALID row (swept threshold)
                    row = base_row(dataset, "valid", fold, run_id, cfg)
                    row.update(dict(epoch=epoch, threshold=best["thr"],
                                    precision=best["prec"]*100, recall=best["rec"]*100, f1=best["f1"]*100,
                                    accuracy=best["acc"]*100, pr_auc=best["pra"], loss=best["loss"]))
                    append_row(RESULTS_CSV, cols, row)

                    # Early stop + save per-run best
                    if best["f1"] > best_f1 + MIN_DELTA:
                        best_f1, best_epoch, best_thr = best["f1"], epoch, float(best["thr"])
                        stale = 0
                        torch.save(model.state_dict(), best_pth)
                        with open(best_meta, "w", encoding="utf-8") as f:
                            json.dump({"epoch": best_epoch, "threshold": best_thr}, f)
                    else:
                        stale += 1
                        if stale >= PATIENCE:
                            break

                # Keep fold-best across grid
                if best_epoch is not None and best_f1 > fold_best["f1"]:
                    fold_best.update(dict(f1=best_f1, epoch=best_epoch, thr=best_thr, cfg=cfg))

            # ----- After grid: Refit on TRAIN+VALID with fold-best cfg and write test_refit -----
            if fold_best["epoch"] is None:
                # nothing improved on any grid setting for this fold
                continue

            refit_on_trainvalid_and_test(
                dataset=dataset,
                fold=fold,
                cfg=fold_best["cfg"],
                best_epoch=int(fold_best["epoch"]),
                best_thr=float(fold_best["thr"]),
                embedding=embedding,
                vocab=vocab,
                MAX_LEN=MAX_LEN,
                RESULTS_CSV=RESULTS_CSV,
                CKPT_ROOT=CKPT_ROOT,
                cols=cols,
                ds_slug=ds_slug,
                te_df=te_df,
            )

if __name__ == "__main__":
    main()