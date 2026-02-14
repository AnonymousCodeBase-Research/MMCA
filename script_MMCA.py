# script_mmca_simple.py
"""
MMCA tuning → selection → refit → test pipeline (MCA-style; no SIGUSR1 or resume trees).

- Prefers train_temp_*.csv when present (for BOTH tuning and refit).
- No sampling: always full splits.
- Unified outputs under runs/ (results.csv, checkpoints/).
- Keeps MMCA columns alpha/gamma in CSV.
- MCA-style embeddings resolution: .bin → .kv once and reuse its basename.
- Selection by VALID F1 after threshold sweep.
"""

from __future__ import annotations
import os, csv, json, time, hashlib, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

# ───────────────────────────── 1) USER CONFIG ─────────────────────────────
DATASETS = [
    "datasets/PII_Textual/CC_Textual/",   # ← edit your dataset roots here
]

# Choose ONE ignore policy (ER vs PII)
IGNORE_COLUMNS = ['query', 'source']           # PII
# IGNORE_COLUMNS = ['left_id', 'right_id']         # ER

GRID: List[Dict[str, Any]] = [
    dict(batch=batch, lr=lr, dropout=0.1, lbl_smooth=0.05, attn_heads=1, alpha=alpha, gamma=gamma)
    for batch in [16, 32]
    for lr in [1e-3, 1e-4]
    for alpha in [0.7]
    for gamma in [0.1]
]

PAPER_EPOCHS        = 3
EARLY_STOP_PATIENCE = 5
WEIGHT_DECAY        = 1e-6
PAPER_BETA1         = 0.9
PAPER_BETA2         = 0.99
PAPER_DROPOUT       = 0.1

EMBEDDINGS_BIN = "wiki.en.bin"  # will be resolved to .kv once

RESULTS_CSV = os.path.join("runs", "results.csv")
CKPT_ROOT   = os.path.join("runs", "checkpoints")
OS_CWD      = os.getcwd()

# ───────────────────────────── 2) MODELS/RUNNER ───────────────────────────
import mcan
from mcan.models.core import MCANModel
from mcan.optim import Optimizer, SoftNLLLoss
from mcan.runner import Runner as MMCARunner

# ───────────────────────────── 3) DEVICE/SEED ─────────────────────────────
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] Using {DEVICE} (cuda_available={torch.cuda.is_available()})")

# ───────────────────── 4) EMBEDDINGS CACHE (.bin → .kv once) ──────────────
from gensim.models.fasttext import load_facebook_vectors, FastTextKeyedVectors
_EMB_PATH_CACHE: Dict[Tuple[str, str, int | None], str] = {}
EMBEDDINGS_NAME: str = ""   # set in main()

def get_embeddings_path(bin_or_kv_path: str, cache_dir: str, limit: int | None = None) -> str:
    key = (bin_or_kv_path, cache_dir, limit)
    if key in _EMB_PATH_CACHE: return _EMB_PATH_CACHE[key]
    if bin_or_kv_path.endswith(".kv"):
        out = bin_or_kv_path
    elif bin_or_kv_path.endswith(".bin"):
        out = build_fasttext_kv_cache(bin_or_kv_path, cache_dir, limit=limit)
    else:
        out = bin_or_kv_path
    _EMB_PATH_CACHE[key] = out
    return out

def build_fasttext_kv_cache(bin_path: str, cache_dir: str, limit: int | None = None) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(bin_path))[0]
    kv_path = os.path.join(cache_dir, f"{base}.kv")
    if os.path.exists(kv_path): return kv_path
    kv: FastTextKeyedVectors = load_facebook_vectors(bin_path)
    kv.save(kv_path)
    return kv_path

# ───────────────────────────── 5) CSV IO ──────────────────────────────────
CSV_COLUMNS = [
    'dataset','split','index','run_id','timestamp',
    'batch_size','lr','alpha','gamma',
    'epoch','epochs_trained','threshold',
    'precision','recall','f1','accuracy','pr_auc','loss',
    'checkpoint_path'
]

def ensure_csv(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def append_csv_row(path: str, row: Dict[str, Any]):
    ensure_csv(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

# ───────────────────────────── 6) METRICS ─────────────────────────────────
def precision_recall_f1_acc_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Tuple[float,float,float,float]:
    y_pred = (scores >= thr).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    acc       = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    return precision, recall, f1, acc

def pr_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    y = y_true[order]
    tp = fp = 0
    precisions, recalls = [], []
    P = y.sum()
    for i in range(len(y)):
        if y[i] == 1: tp += 1
        else:         fp += 1
        precisions.append(tp / (tp + fp + 1e-12))
        recalls.append(tp / (P + 1e-12))
    auc = 0.0
    for i in range(1, len(recalls)):
        auc += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2
    return float(auc)

def binary_log_loss(scores: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(scores, eps, 1.0 - eps)
    y = y_true.astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

@dataclass
class ThresholdResult:
    value: float
    best_f1: float
    precision: float
    recall: float
    accuracy: float
    pr_auc: float

def sweep_threshold(valid_df: pd.DataFrame, y_true: np.ndarray) -> ThresholdResult:
    scores = valid_df['match_score'].astype(float).to_numpy()
    best: ThresholdResult | None = None
    for thr in np.linspace(0.05, 0.95, 91):
        p, r, f1, acc = precision_recall_f1_acc_at_threshold(y_true, scores, thr)
        if (best is None) or (f1 > best.best_f1):
            best = ThresholdResult(thr, f1, p, r, acc, pr_auc(scores, y_true))
    return best  # type: ignore

# ───────────────────────────── 7) DATA HELPERS ────────────────────────────
def _stable_row_hash(series: pd.Series) -> str:
    parts = []
    for v in series:
        s = "" if pd.isna(v) else str(v)
        parts.append(s.strip().lower())
    blob = "||".join(parts)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()

def ensure_pair_id(csv_path: str):
    if not os.path.isfile(csv_path): return
    df = pd.read_csv(csv_path)
    if 'id' in df.columns:
        df.to_csv(csv_path, index=False); return
    if 'left_id' in df.columns and 'right_id' in df.columns:
        df['id'] = df['left_id'].astype(str) + '||' + df['right_id'].astype(str)
        df.to_csv(csv_path, index=False); return
    left_cols  = [c for c in df.columns if c.startswith('left_')]
    right_cols = [c for c in df.columns if c.startswith('right_')]
    left_bases  = {c[len('left_'):]: c for c in left_cols}
    right_bases = {c[len('right_'):]: c for c in right_cols}
    common_bases = [b for b in left_bases if b in right_bases]
    preferred = ['name','address','phone','email','email_list','bio','country','state','city','zip','aliases']
    ordered = [b for b in preferred if b in common_bases] or common_bases
    if ordered:
        pair_cols = []
        for b in ordered:
            pair_cols += [left_bases[b], right_bases[b]]
        df['id'] = df[pair_cols].apply(_stable_row_hash, axis=1)
    else:
        df['id'] = df.reset_index().index.astype(str)
    df.to_csv(csv_path, index=False)

def _dm_style_align_columns(csv_path: str, left_prefix="left_", right_prefix="right_"):
    if not os.path.isfile(csv_path): return
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in IGNORE_COLUMNS if c in df.columns]
    if drop_cols: df = df.drop(columns=drop_cols)
    cols = list(df.columns)
    left_cols  = [c for c in cols if c.startswith(left_prefix)]
    right_cols = [c for c in cols if c.startswith(right_prefix)]
    left_bases  = {c[len(left_prefix):]: c for c in left_cols}
    right_bases = {c[len(right_prefix):]: c for c in right_cols}
    bases_in_csv_order = [c[len(left_prefix):] for c in left_cols]
    common_bases = [b for b in bases_in_csv_order if b in right_bases]
    if common_bases:
        keep = [c for c in ["id","label"] if c in df.columns]
        for b in common_bases:
            keep.extend([left_bases[b], right_bases[b]])
        df = df[keep]
        df.to_csv(csv_path, index=False); return
    if not left_cols and not right_cols:
        df = df[[c for c in ["id","label"] if c in df.columns]]
        df.to_csv(csv_path, index=False); return
    def _join_text(row, cols):
        parts = []
        for c in cols:
            val = row.get(c, "")
            if pd.isna(val): continue
            s = str(val).strip()
            if s: parts.append(s)
        return " | ".join(parts) if parts else ""
    df['left_concat']  = df.apply(lambda r: _join_text(r, left_cols),  axis=1)
    df['right_concat'] = df.apply(lambda r: _join_text(r, right_cols), axis=1)
    keep_cols = [c for c in ["id","label","left_concat","right_concat"] if c in df.columns]
    df = df[keep_cols]
    df.to_csv(csv_path, index=False)

def dataset_slug(dataset: str) -> str:
    return dataset.strip('/').replace('/', '_')

def make_run_id(dataset: str, idx: int, params: Dict[str, Any]) -> str:
    blob = json.dumps({'batch': params['batch'], 'lr': params['lr'],
                       'alpha': params.get('alpha'), 'gamma': params.get('gamma')}, sort_keys=True)
    return f"{dataset_slug(dataset)}_{idx}_{hashlib.md5(blob.encode()).hexdigest()[:6]}"

def get_labels_for(dataset_obj: Any) -> np.ndarray:
    raw = pd.read_csv(dataset_obj.path)
    label_col = getattr(dataset_obj, 'label_field', 'label')
    return raw[label_col].astype(int).to_numpy()

def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# Prefer train_temp if present
def _pick_train_name(dataset_dir: str, dataset_index: int) -> str:
    tt = os.path.join(dataset_dir, f"train_temp_{dataset_index}.csv")
    t  = os.path.join(dataset_dir, f"train_{dataset_index}.csv")
    return os.path.basename(tt if os.path.isfile(tt) else t)

def _prepare_full_splits(dataset_dir: str, dataset_index: int) -> Tuple[str, str, str]:
    """Ensure IDs & aligned columns on train(_temp)/valid/test; return basenames (NO SAMPLING)."""
    # Always run on existing files (train_temp preferred, but we prep both if they exist)
    for split in ['train_temp', 'train', 'valid', 'test']:
        p = os.path.join(dataset_dir, f"{split}_{dataset_index}.csv")
        if os.path.isfile(p):
            ensure_pair_id(p)
            _dm_style_align_columns(p)
    train_csv = _pick_train_name(dataset_dir, dataset_index)
    valid_csv = f"valid_{dataset_index}.csv"
    test_csv  = f"test_{dataset_index}.csv"
    return train_csv, valid_csv, test_csv

# ───────────────────────────── 8) CSV ROW WRITERS ────────────────────────
def _base_row(dataset, split, idx, run_id, params):
    return dict(
        dataset=dataset, split=split, index=idx, run_id=run_id,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        batch_size=params['batch'], lr=params['lr'],
        alpha=params.get('alpha', None), gamma=params.get('gamma', None),
        epoch=None, epochs_trained=None, threshold=None,
        precision=None, recall=None, f1=None, accuracy=None, pr_auc=None, loss=None,
        checkpoint_path=''
    )

def write_train_epoch_row(csv_path, dataset, idx, run_id, params, epoch, precision, recall, f1, accuracy, loss):
    row = _base_row(dataset, 'train', idx, run_id, params)
    row.update(dict(epoch=epoch, precision=precision, recall=recall, f1=f1, accuracy=accuracy, loss=loss))
    append_csv_row(csv_path, row)

def write_valid_epoch_row(csv_path, dataset, idx, run_id, params, epoch, precision, recall, f1, accuracy, prauc, loss, threshold):
    row = _base_row(dataset, 'valid', idx, run_id, params)
    row.update(dict(epoch=epoch, precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                    pr_auc=prauc, loss=loss, threshold=threshold))
    append_csv_row(csv_path, row)

def write_tune_summary(csv_path, dataset, idx, run_id, params, best_epoch, best_threshold, best_f1, best_prec, best_rec, best_acc, best_prauc):
    row = _base_row(dataset, 'tune_best', idx, run_id, params)
    row.update(dict(epoch=best_epoch, epochs_trained=best_epoch, threshold=best_threshold,
                    precision=best_prec*100.0, recall=best_rec*100.0, f1=best_f1*100.0,
                    accuracy=best_acc*100.0, pr_auc=best_prauc, loss=None))
    append_csv_row(csv_path, row)

def write_test_summary(csv_path, dataset, idx, run_id, params, best_epoch, threshold, precision, recall, f1, accuracy, prauc, loss, checkpoint_path: str, split_label: str = 'test_refit'):
    row = _base_row(dataset, split_label, idx, run_id, params)
    row.update(dict(epoch=best_epoch, epochs_trained=best_epoch, threshold=threshold,
                    precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                    pr_auc=prauc, loss=loss, checkpoint_path=checkpoint_path))
    append_csv_row(csv_path, row)

# ───────────────────────────── 9) RUNNER HELPERS ─────────────────────────
def mmca_get_scores(model, dataset_obj, batch_size, path, dataset_index) -> pd.DataFrame:
    preds = MMCARunner._run(
        path, dataset_index, 'TEST', model, dataset_obj,
        train=False, batch_size=batch_size, return_predictions=True,
        log_freq=10**9, verbose=False, device=DEVICE,
    )
    return pd.DataFrame(preds, columns=('id', 'match_score'))

# ───────────────────────────── 10) TRAIN/TUNE ────────────────────────────
def train_one_setting(dataset_dir: str, dataset_index: int, params: Dict[str, Any]) -> Tuple[float,int,float,str,Dict[str,Any]]:
    """Train one grid point; returns (best_f1, best_epoch, best_thr, best_ckpt, params)."""
    train_csv, valid_csv, test_csv = _prepare_full_splits(dataset_dir, dataset_index)

    # Build datasets (use global EMBEDDINGS_NAME = basename of resolved .kv)
    train, valid, test = mcan.data.process(
        path=dataset_dir,
        train=train_csv, validation=valid_csv, test=test_csv,
        embeddings=EMBEDDINGS_NAME, embeddings_cache_path=OS_CWD,
        ignore_columns=IGNORE_COLUMNS, id_attr='id', label_attr='label',
        left_prefix='left_', right_prefix='right_', pca=True
    )

    valid_labels = get_labels_for(valid)
    train_labels = get_labels_for(train)

    model = MCANModel(
        hidden_size=300,
        attn_heads=params.get('attn_heads', 1),
        rnn_dropout=params.get('dropout', PAPER_DROPOUT),
        gate_dropout=params.get('dropout', PAPER_DROPOUT),
        global_attn_dropout=params.get('dropout', PAPER_DROPOUT),
    )
    model.initialize(train)

    opt = Optimizer(lr=params['lr'], beta1=PAPER_BETA1, beta2=PAPER_BETA2, weight_decay=WEIGHT_DECAY)

    crit = None
    if params.get('lbl_smooth', None) is not None:
        pos_neg_ratio = 1.0
        pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
        neg_weight = 2 - pos_weight
        crit = SoftNLLLoss(params['lbl_smooth'], torch.Tensor([neg_weight, pos_weight]),
                           alpha=params.get('alpha', 0.0), gamma=params.get('gamma', 0.5))

    best = dict(f1=-1.0, epoch=0, thr=0.5, ckpt="", prec=None, rec=None, acc=None, prauc=None)
    no_improve = 0

    ds_slug = dataset_slug(dataset_dir)
    ckpt_dir = os.path.join(CKPT_ROOT, ds_slug, f"idx{dataset_index}")
    os.makedirs(ckpt_dir, exist_ok=True)
    per_grid_best_path = os.path.join(ckpt_dir, f"{make_run_id(dataset_dir, dataset_index, params)}_best.pth")

    for epoch in range(1, PAPER_EPOCHS + 1):
        t0 = time.time()
        MMCARunner._run(dataset_dir, dataset_index, 'TRAIN', model, train,
                        criterion=crit, optimizer=opt, train=True,
                        batch_size=params['batch'], log_freq=10**9, verbose=False, device=DEVICE)
        t1 = time.time()

        tr_scores = mmca_get_scores(model, train, batch_size=params['batch'], path=dataset_dir, dataset_index=dataset_index)
        tr_probs  = tr_scores['match_score'].astype(float).to_numpy()
        tp, rp, tf, ta = precision_recall_f1_acc_at_threshold(train_labels, tr_probs, 0.5)
        tr_loss = binary_log_loss(tr_probs, train_labels)

        print(f"[{_ts()}] idx={dataset_index} ep={epoch:02d} | TRAIN: f1={tf*100:.2f} P={tp*100:.2f} R={rp*100:.2f} acc={ta*100:.2f} loss={tr_loss:.4f} ({t1-t0:.1f}s)")
        write_train_epoch_row(RESULTS_CSV, dataset_dir, dataset_index, make_run_id(dataset_dir, dataset_index, params),
                              params, epoch, precision=tp*100.0, recall=rp*100.0, f1=tf*100.0, accuracy=ta*100.0, loss=tr_loss)

        t2 = time.time()
        va_scores = mmca_get_scores(model, valid, batch_size=params['batch'], path=dataset_dir, dataset_index=dataset_index)
        thrres = sweep_threshold(va_scores, valid_labels)
        va_probs = va_scores['match_score'].astype(float).to_numpy()
        va_loss = binary_log_loss(va_probs, valid_labels)
        t3 = time.time()

        print(f"[{_ts()}] idx={dataset_index} ep={epoch:02d} | VALID: f1={thrres.best_f1*100:.2f} thr={thrres.value:.2f} P={thrres.precision*100:.2f} R={thrres.recall*100:.2f} acc={thrres.accuracy*100:.2f} prAUC={thrres.pr_auc:.3f} loss={va_loss:.4f} ({t3-t2:.1f}s)")
        write_valid_epoch_row(RESULTS_CSV, dataset_dir, dataset_index, make_run_id(dataset_dir, dataset_index, params),
                              params, epoch, precision=thrres.precision*100.0, recall=thrres.recall*100.0,
                              f1=thrres.best_f1*100.0, accuracy=thrres.accuracy*100.0,
                              prauc=thrres.pr_auc, loss=va_loss, threshold=thrres.value)

        if thrres.best_f1 > best['f1'] + 1e-12:
            best.update(dict(f1=thrres.best_f1, epoch=epoch, thr=thrres.value,
                             prec=thrres.precision, rec=thrres.recall, acc=thrres.accuracy, prauc=thrres.pr_auc))
            model.save_state(per_grid_best_path)
            no_improve = 0
        else:
            no_improve += 1
            if EARLY_STOP_PATIENCE and no_improve >= EARLY_STOP_PATIENCE:
                break

    run_id = make_run_id(dataset_dir, dataset_index, params)
    write_tune_summary(RESULTS_CSV, dataset_dir, dataset_index, run_id, params,
                       best_epoch=best['epoch'], best_threshold=best['thr'], best_f1=best['f1'],
                       best_prec=best['prec'], best_rec=best['rec'], best_acc=best['acc'], best_prauc=best['prauc'])
    return best['f1'], best['epoch'], best['thr'], per_grid_best_path, params

# ───────────────────────────── 11) REFIT + TEST ───────────────────────────
def refit_and_test(dataset_dir: str, dataset_index: int, best_cfg: Dict[str, Any]):
    train_csv, valid_csv, test_csv = _prepare_full_splits(dataset_dir, dataset_index)

    train, valid, test = mcan.data.process(
        path=dataset_dir,
        train=train_csv, validation=valid_csv, test=test_csv,
        embeddings=EMBEDDINGS_NAME, embeddings_cache_path=OS_CWD,
        ignore_columns=IGNORE_COLUMNS, id_attr='id', label_attr='label',
        left_prefix='left_', right_prefix='right_', pca=True
    )

    model = MCANModel(
        hidden_size=300,
        attn_heads=best_cfg.get('attn_heads', 1),
        rnn_dropout=best_cfg.get('dropout', PAPER_DROPOUT),
        gate_dropout=best_cfg.get('dropout', PAPER_DROPOUT),
        global_attn_dropout=best_cfg.get('dropout', PAPER_DROPOUT),
    )
    model.initialize(train)

    opt  = Optimizer(lr=best_cfg['lr'], beta1=PAPER_BETA1, beta2=PAPER_BETA2, weight_decay=WEIGHT_DECAY)
    crit = SoftNLLLoss(0.05, torch.Tensor([1.0, 1.0]), alpha=best_cfg.get('alpha', 0.0), gamma=best_cfg.get('gamma', 0.5))

    for e in range(best_cfg['best_epoch']):
        t0 = time.time()
        MMCARunner._run(dataset_dir, dataset_index, 'TRAIN', model, train,
                        criterion=crit, optimizer=opt, train=True,
                        batch_size=best_cfg['batch'], log_freq=10 ** 9, verbose=False, device=DEVICE)
        MMCARunner._run(dataset_dir, dataset_index, 'TRAIN', model, valid,
                        criterion=crit, optimizer=opt, train=True,
                        batch_size=best_cfg['batch'], log_freq=10 ** 9, verbose=False, device=DEVICE)
        t1 = time.time()
        print(f"[{_ts()}] REFIT idx={dataset_index} ep={e + 1:02d} | TRAIN: (train+valid) done in {t1 - t0:.1f}s")

    ds_slug = dataset_slug(dataset_dir)
    ckpt_dir = os.path.join(CKPT_ROOT, ds_slug, f"idx{dataset_index}")
    os.makedirs(ckpt_dir, exist_ok=True)
    refit_ckpt = os.path.join(ckpt_dir, f"{make_run_id(dataset_dir, dataset_index, best_cfg)}_refit_best.pth")
    model.save_state(refit_ckpt)

    test_scores = mmca_get_scores(model, test, batch_size=best_cfg['batch'], path=dataset_dir, dataset_index=dataset_index)
    test_labels = get_labels_for(test)
    probs = test_scores['match_score'].astype(float).to_numpy()
    p, r, f1, acc = precision_recall_f1_acc_at_threshold(test_labels, probs, best_cfg['best_thr'])
    pra = pr_auc(probs, test_labels)
    test_loss = binary_log_loss(probs, test_labels)

    print(f"===> TEST_REFIT (best @ epoch {best_cfg['best_epoch']}, thr={best_cfg['best_thr']:.2f}) | Prec {p*100:.2f} | Rec {r*100:.2f} | F1 {f1*100:.2f}")
    write_test_summary(RESULTS_CSV, dataset_dir, dataset_index, make_run_id(dataset_dir, dataset_index, best_cfg), best_cfg,
                       best_epoch=best_cfg['best_epoch'], threshold=best_cfg['best_thr'],
                       precision=p*100.0, recall=r*100.0, f1=f1*100.0, accuracy=acc*100.0,
                       prauc=pra, loss=test_loss, checkpoint_path=refit_ckpt, split_label='test_refit')

# ───────────────────────────── 12) MAIN ───────────────────────────────────
def main():
    ensure_csv(RESULTS_CSV)
    set_seed(42, deterministic=False)

    # Resolve embeddings .bin → .kv ONCE and reuse basename everywhere
    kv_path = get_embeddings_path(EMBEDDINGS_BIN, cache_dir=OS_CWD, limit=200_000)
    global EMBEDDINGS_NAME
    EMBEDDINGS_NAME = os.path.basename(kv_path)

    for dataset in DATASETS:
        for dataset_index in range(10):
            grid_results: List[Tuple[float,int,float,str,Dict[str,Any]]] = []
            for params in GRID:
                print(f"[grid] idx={dataset_index} batch={params['batch']} lr={params['lr']} alpha={params['alpha']} gamma={params['gamma']}")
                best_f1, best_epoch, best_thr, best_ckpt, returned_params = train_one_setting(dataset, dataset_index, params)
                grid_results.append((best_f1, best_epoch, best_thr, best_ckpt, returned_params))

            grid_results.sort(key=lambda x: x[0], reverse=True)
            best_f1, best_epoch, best_thr, best_ckpt, best_cfg = grid_results[0]
            best_cfg.update({'best_epoch': best_epoch, 'best_thr': best_thr})
            refit_and_test(dataset, dataset_index, best_cfg)

if __name__ == "__main__":
    main()
