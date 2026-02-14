from __future__ import annotations
import os, csv, json, time, hashlib, random, warnings
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import hashlib as _hashlib
from gensim.models.fasttext import load_facebook_vectors, FastTextKeyedVectors
import mcan
from mcan.models.core import MCANModel
from mcan.optim import Optimizer, SoftNLLLoss
from mcan.runner import Runner as MCARunner

# -----------------------------
# Embeddings cache helpers
# -----------------------------
# global singleton cache to avoid repeated loads in-process
_EMB_PATH_CACHE = {}

def get_embeddings_path(bin_or_kv_path: str, cache_dir: str, limit: int | None = None) -> str:
    """
    Returns a path that mcan can load. If given .bin, ensure a .kv cache exists and return that.
    Uses a simple dict so we don’t rebuild or decide this more than once.
    """
    key = (bin_or_kv_path, cache_dir, limit)
    if key in _EMB_PATH_CACHE:
        return _EMB_PATH_CACHE[key]

    if bin_or_kv_path.endswith(".kv"):
        out = bin_or_kv_path
    elif bin_or_kv_path.endswith(".bin"):
        out = build_fasttext_kv_cache(bin_or_kv_path, cache_dir, limit=limit)
    else:
        out = bin_or_kv_path

    _EMB_PATH_CACHE[key] = out
    return out


def build_fasttext_kv_cache(bin_path: str, cache_dir: str, limit: int | None = None) -> str:
    """
    Convert wiki.en.bin -> FastTextKeyedVectors .kv (memory-mappable on later loads).
    NOTE: We ignore `limit` to preserve subword info (keeps OOV synthesis working).
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(bin_path))[0]
    kv_path = os.path.join(cache_dir, f"{base}.kv")

    if os.path.exists(kv_path):
        return kv_path

    kv: FastTextKeyedVectors = load_facebook_vectors(bin_path)
    kv.save(kv_path)
    return kv_path

# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

# -----------------------------
# Configurable knobs
# -----------------------------
RESULTS_CSV = os.path.join("runs", "results.csv")
CKPT_ROOT   = os.path.join("runs", "checkpoints")
OS_CWD      = os.getcwd()

DATASETS = ["datasets/PII_Textual/CC_Textual/"]

# ---- fixed defaults ----
PAPER_EPOCHS  = 64
PAPER_BETA1   = 0.9
PAPER_BETA2   = 0.99
PAPER_DROPOUT = 0.1
WEIGHT_DECAY = 1e-6

# === grid ===
GRID: List[Dict[str, Any]] = [
    dict(batch=batch, lr=lr, wd=WEIGHT_DECAY, dropout=PAPER_DROPOUT, lbl_smooth=0.05, attn_heads=1)
    for batch in [16, 32, 64]
    for lr in [0.001, 0.0001]
]

MAX_EPOCHS = PAPER_EPOCHS
EARLY_STOP_PATIENCE = 5
EMBEDDINGS_NAME = 'wiki.en.bin'
#PII Dataset
IGNORE_COLUMNS = ['query', 'source']
#ER Dataset
#IGNORE_COLUMNS = ['left_id', 'right_id']

# -----------------------------
# CSV schema utilities (same schema as your DM runner)
# -----------------------------
CSV_COLUMNS = [
    'dataset','split','index','run_id','timestamp',
    'batch_size','lr',
    # metrics
    'epoch','epochs_trained','threshold',
    'precision','recall','f1','accuracy','pr_auc','loss',
    # bookkeeping
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

# -----------------------------
# Metrics + threshold sweeps
# -----------------------------
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
    best = None
    for thr in np.linspace(0.05, 0.95, 91):
        p, r, f1, acc = precision_recall_f1_acc_at_threshold(y_true, scores, thr)
        if (best is None) or (f1 > best.best_f1):
            best = ThresholdResult(thr, f1, p, r, acc, pr_auc(scores, y_true))
    return best

# -----------------------------
# Data helpers (IDs/columns)
# -----------------------------
def _stable_row_hash(series: pd.Series) -> str:
    parts = []
    for v in series:
        s = "" if pd.isna(v) else str(v)
        parts.append(s.strip().lower())
    blob = "||".join(parts)
    return _hashlib.md5(blob.encode("utf-8")).hexdigest()

def ensure_pair_id(csv_path: str):
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
    common_bases = [b for b in left_bases.keys() if b in right_bases]
    preferred_order = ['name','address','phone','email','email_list','bio','country','state','city','zip','aliases']
    ordered_bases = [b for b in preferred_order if b in common_bases] or common_bases
    if ordered_bases:
        concat_cols = []
        for b in ordered_bases:
            concat_cols.append(left_bases[b]); concat_cols.append(right_bases[b])
        df['id'] = df[concat_cols].apply(_stable_row_hash, axis=1)
        df.to_csv(csv_path, index=False); return
    df['id'] = df.reset_index().index.astype(str)
    df.to_csv(csv_path, index=False)

def align_left_right_columns(csv_path: str, left_prefix="left_", right_prefix="right_"):
    df = pd.read_csv(csv_path)
    left_cols  = [c for c in df.columns if c.startswith(left_prefix)]
    right_cols = [c for c in df.columns if c.startswith(right_prefix)]
    left_bases  = {c[len(left_prefix):]: c for c in left_cols}
    right_bases = {c[len(right_prefix):]: c for c in right_cols}
    common_bases = set(left_bases.keys()) & set(right_bases.keys())
    keep_cols = ["id", "label", "source"] \
                + [left_prefix + b for b in common_bases] \
                + [right_prefix + b for b in common_bases]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.to_csv(csv_path, index=False)

def dataset_slug(dataset: str) -> str:
    return dataset.strip('/').replace('/', '_')

def make_run_id(dataset: str, idx: int, params: Dict[str, Any]) -> str:
    blob = json.dumps({'batch': params['batch'], 'lr': params['lr']}, sort_keys=True)
    return f"{dataset_slug(dataset)}_{idx}_{_hashlib.md5(blob.encode()).hexdigest()[:6]}"

def get_labels_for(dataset_obj: Any) -> np.ndarray:
    raw = pd.read_csv(dataset_obj.path)
    label_col = getattr(dataset_obj, 'label_field', 'label')
    return raw[label_col].astype(int).to_numpy()

def make_trainvalid_csv_from_files(dataset_dir: str, train_file: str, valid_file: str, out_basename: str) -> str:
    tr = pd.read_csv(os.path.join(dataset_dir, train_file))
    va = pd.read_csv(os.path.join(dataset_dir, valid_file))
    tv = pd.concat([tr, va], axis=0, ignore_index=True)
    out = os.path.join(dataset_dir, out_basename)
    tv.to_csv(out, index=False)
    return out

# -----------------------------
# NEW: Always use full splits (no sampling)
# -----------------------------
def _prepare_full_splits(dataset_dir: str, dataset_index: int) -> Tuple[str, str, str]:
    """
    Ensure IDs/columns on the original CSVs and return their basenames.
    This uses the FULL train/valid/test without any sampling.
    """
    for split in ['train', 'valid', 'test']:
        p = os.path.join(dataset_dir, f"{split}_{dataset_index}.csv")
        ensure_pair_id(p)
        align_left_right_columns(p)
    return (f"train_{dataset_index}.csv",
            f"valid_{dataset_index}.csv",
            f"test_{dataset_index}.csv")

# -----------------------------
# CSV writing helpers
# -----------------------------
def base_row(dataset, split, idx, run_id, params):
    return dict(
        dataset=dataset, split=split, index=idx, run_id=run_id,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        batch_size=params['batch'],
        lr=params['lr'],
        epoch=None, epochs_trained=None,
        threshold=None,
        precision=None, recall=None, f1=None, accuracy=None, pr_auc=None, loss=None,
        checkpoint_path=''
    )

def write_train_epoch_row(csv_path, dataset, idx, run_id, params,
                          epoch, precision, recall, f1, accuracy, loss):
    row = base_row(dataset, 'train', idx, run_id, params)
    row.update(dict(
        epoch=epoch,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy, loss=loss,
    ))
    append_csv_row(csv_path, row)

def write_valid_epoch_row(csv_path, dataset, idx, run_id, params,
                          epoch, precision, recall, f1, accuracy, prauc, loss, threshold):
    row = base_row(dataset, 'valid', idx, run_id, params)
    row.update(dict(
        epoch=epoch,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        pr_auc=prauc, loss=loss, threshold=threshold
    ))
    append_csv_row(csv_path, row)

def write_tune_summary(csv_path, dataset, idx, run_id, params,
                       best_epoch, best_threshold, best_f1, best_prec, best_rec, best_acc, best_prauc):
    row = base_row(dataset, 'tune_best', idx, run_id, params)
    row.update(dict(
        epoch=best_epoch, epochs_trained=best_epoch,
        threshold=best_threshold,
        precision=best_prec*100.0, recall=best_rec*100.0, f1=best_f1*100.0,
        accuracy=best_acc*100.0, pr_auc=best_prauc, loss=None
    ))
    append_csv_row(csv_path, row)

def write_test_summary(csv_path, dataset, idx, run_id, params,
                       best_epoch, threshold, precision, recall, f1, accuracy, prauc, loss,
                       checkpoint_path: str, split_label: str = 'test'):
    row = base_row(dataset, split_label, idx, run_id, params)
    row.update(dict(
        epoch=best_epoch, epochs_trained=best_epoch,
        threshold=threshold, precision=precision, recall=recall, f1=f1,
        accuracy=accuracy, pr_auc=prauc, loss=loss,
        checkpoint_path=checkpoint_path
    ))
    append_csv_row(csv_path, row)

# -----------------------------
# MCA helpers
# -----------------------------
def mca_get_scores(model, dataset_obj, batch_size, path, dataset_index) -> pd.DataFrame:
    preds = MCARunner._run(
        path, dataset_index, 'TEST', model, dataset_obj,
        train=False,
        batch_size=batch_size,
        return_predictions=True,
        log_freq=10**9,
        verbose=False,  # silence MCA prints; we control console output
    )
    # list of (id, prob_of_class_1)
    return pd.DataFrame(preds, columns=('id', 'match_score'))

# -----------------------------
# Main tuning + refit loop
# -----------------------------
def run_for_datasets(dataset_name_list: List[str]):
    ensure_csv(RESULTS_CSV)

    for dataset in dataset_name_list:
        ds_slug = dataset_slug(dataset)

        for dataset_index in range(10):
            # Always use full data
            train_csv, valid_csv, test_csv = _prepare_full_splits(
                dataset_dir=dataset,
                dataset_index=dataset_index,
            )

            train, valid, test = mcan.data.process(
                path=dataset,
                train=train_csv,
                validation=valid_csv,
                test=test_csv,
                embeddings=EMBEDDINGS_NAME,
                embeddings_cache_path=OS_CWD,
                ignore_columns=IGNORE_COLUMNS,
                id_attr='id',
                label_attr='label',
                left_prefix='left_',
                right_prefix='right_',
                pca=True
            )

            valid_labels = get_labels_for(valid)
            test_labels  = get_labels_for(test)
            train_labels = get_labels_for(train)

            # === Grid search on (train -> update), (valid -> model selection) ===
            fold_best: Dict[str, Any] = dict(f1=-1.0, epoch=None, threshold=None, params=None,
                                             prec=None, rec=None, acc=None, prauc=None,
                                             ckpt_path=None, run_id=None)

            for params in GRID:
                run_id = make_run_id(dataset, dataset_index, params)
                per_run_seed = int(_hashlib.md5(run_id.encode()).hexdigest()[:8], 16)
                set_seed(per_run_seed, deterministic=False)

                ckpt_dir = os.path.join(CKPT_ROOT, ds_slug, f"idx{dataset_index}")
                os.makedirs(ckpt_dir, exist_ok=True)
                per_grid_best_path = os.path.join(ckpt_dir, f"{run_id}_best.pth")
                meta_json_path     = os.path.join(ckpt_dir, f"{run_id}_best.json")

                # Build MCA model
                model = MCANModel(
                    hidden_size=300,
                    attn_heads=params.get('attn_heads', 1),
                    rnn_dropout=params.get('dropout', 0.1),
                    gate_dropout=params.get('dropout', 0.1),
                    global_attn_dropout=params.get('dropout', 0.1),
                )
                model.initialize(train)

                # Optimizer & criterion
                opt = Optimizer(lr=params['lr'], beta1=PAPER_BETA1, beta2=PAPER_BETA2,
                                weight_decay=WEIGHT_DECAY)
                crit = None
                if params.get('lbl_smooth', None) is not None:
                    pos_neg_ratio = 1.0
                    pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
                    neg_weight = 2 - pos_weight
                    crit = SoftNLLLoss(params['lbl_smooth'],
                                       torch.Tensor([neg_weight, pos_weight]))

                run_best = {'f1': -1.0, 'epoch': None, 'threshold': None,
                            'prec': None, 'rec': None, 'acc': None, 'prauc': None}
                no_improve_count = 0

                for epoch in range(PAPER_EPOCHS):
                    model.epoch = epoch

                    # TRAIN update pass
                    MCARunner._run(
                        dataset, dataset_index, 'TRAIN', model, train,
                        criterion=crit,
                        optimizer=opt,
                        train=True,
                        batch_size=params['batch'],
                        log_freq=10**9,
                        verbose=False,
                    )

                    # TRAIN metrics (fixed thr=0.5)
                    t0_tr = time.time()
                    train_scores = mca_get_scores(model, train, batch_size=params['batch'],
                                                  path=dataset, dataset_index=dataset_index)
                    elapsed_tr = time.time() - t0_tr
                    train_probs = train_scores['match_score'].astype(float).to_numpy()
                    tp, rp, tf, ta = precision_recall_f1_acc_at_threshold(train_labels, train_probs, 0.5)
                    train_loss = binary_log_loss(train_probs, train_labels)

                    print(f"\n===>  TRAIN Epoch {epoch+1}")
                    print("0% [██████] 100% | ETA: 00:00:00")
                    mm = int(elapsed_tr // 60); ss = int(round(elapsed_tr % 60))
                    print(f"Total time elapsed: 00:{mm:02d}:{ss:02d}")
                    print((
                        "Finished Epoch {epoch} || Run Time: {runtime:6.1f} | Load Time: {datatime:6.1f} || "
                        "Prec: {prec:6.2f} | Rec: {rec:6.2f} | F1: {f1:6.2f} || Ex/s: {eps:6.2f}"
                    ).format(
                        epoch=epoch+1,
                        runtime=elapsed_tr, datatime=0.0,
                        prec=tp*100.0, rec=rp*100.0, f1=tf*100.0,
                        eps=(len(train_scores) / max(elapsed_tr, 1e-8))
                    ))
                    print()

                    write_train_epoch_row(
                        RESULTS_CSV, dataset, dataset_index, run_id, params,
                        epoch+1,
                        precision=tp*100.0, recall=rp*100.0, f1=tf*100.0,
                        accuracy=ta*100.0, loss=train_loss
                    )

                    # VALID sweep
                    t0 = time.time()
                    valid_scores = mca_get_scores(model, valid, batch_size=params['batch'],
                                                  path=dataset, dataset_index=dataset_index)
                    thrres = sweep_threshold(valid_scores, valid_labels)
                    elapsed = time.time() - t0
                    probs = valid_scores['match_score'].astype(float).to_numpy()

                    print(f"===>  EVAL Epoch {epoch+1}")
                    print("0% [██████] 100% | ETA: 00:00:00")
                    mm = int(elapsed // 60); ss = int(round(elapsed % 60))
                    print(f"Total time elapsed: 00:{mm:02d}:{ss:02d}")
                    print((
                        "Finished Epoch {epoch} || Run Time: {runtime:6.1f} | Load Time: {datatime:6.1f} || "
                        "Prec: {prec:6.2f} | Rec: {rec:6.2f} | F1: {f1:6.2f} || Ex/s: {eps:6.2f}"
                    ).format(
                        epoch=epoch+1,
                        runtime=elapsed, datatime=0.0,
                        prec=thrres.precision*100.0,
                        rec=thrres.recall*100.0,
                        f1=thrres.best_f1*100.0,
                        eps=(len(valid_scores) / max(elapsed, 1e-8))
                    ))
                    print()

                    write_valid_epoch_row(
                        RESULTS_CSV, dataset, dataset_index, run_id, params,
                        epoch+1,
                        thrres.precision * 100.0,
                        thrres.recall * 100.0,
                        thrres.best_f1 * 100.0,
                        thrres.accuracy * 100.0,
                        thrres.pr_auc,
                        binary_log_loss(probs, valid_labels),
                        thrres.value
                    )

                    # Model selection on VALID F1
                    improved = thrres.best_f1 > run_best['f1']
                    if improved:
                        model.save_state(per_grid_best_path)
                        with open(meta_json_path, "w", encoding="utf-8") as f:
                            json.dump({"epoch": int(epoch+1),
                                       "threshold": float(thrres.value)}, f)
                        run_best.update({'f1': thrres.best_f1,
                                         'epoch': epoch+1,
                                         'threshold': thrres.value,
                                         'prec': thrres.precision,
                                         'rec': thrres.recall,
                                         'acc': thrres.accuracy,
                                         'prauc': thrres.pr_auc})
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if EARLY_STOP_PATIENCE and no_improve_count >= EARLY_STOP_PATIENCE:
                            break

                # Record tuning summary for this run_id
                if run_best['epoch'] is not None:
                    write_tune_summary(
                        RESULTS_CSV, dataset, dataset_index, run_id, params,
                        best_epoch=run_best['epoch'],
                        best_threshold=run_best['threshold'],
                        best_f1=run_best['f1'],
                        best_prec=run_best['prec'],
                        best_rec=run_best['rec'],
                        best_acc=run_best['acc'],
                        best_prauc=run_best['prauc']
                    )

                # Keep track of the overall fold winner across grid
                if run_best['epoch'] is not None and run_best['f1'] > fold_best['f1']:
                    fold_best.update(run_best)
                    fold_best['params'] = params
                    fold_best['ckpt_path'] = per_grid_best_path
                    fold_best['run_id'] = run_id

            # === Refit on train+valid with locked hyperparams/epoch/threshold ===
            if fold_best['epoch'] is None:
                continue

            best_params   = fold_best['params']
            best_epoch    = int(fold_best['epoch'])
            best_thr      = float(fold_best['threshold'])
            best_run_id   = str(fold_best['run_id'])

            merged_basename = f"trainvalid_{dataset_index}__from_used.csv"
            merged_path = make_trainvalid_csv_from_files(dataset, train_csv, valid_csv, merged_basename)
            ensure_pair_id(merged_path)
            align_left_right_columns(merged_path)

            train_refit, _valid_dummy, test_refit = mcan.data.process(
                path=dataset,
                train=merged_basename,
                validation=valid_csv,  # placeholder; not used but must exist
                test=test_csv,
                embeddings=EMBEDDINGS_NAME,
                embeddings_cache_path=OS_CWD,
                ignore_columns=IGNORE_COLUMNS,
                id_attr='id',
                label_attr='label',
                left_prefix='left_',
                right_prefix='right_',
                pca=True
            )

            # Build fresh model and optimizer with the chosen hyperparams
            refit_model = MCANModel(
                hidden_size=300,
                attn_heads=best_params.get('attn_heads', 1),
                rnn_dropout=best_params.get('dropout', 0.1),
                gate_dropout=best_params.get('dropout', 0.1),
                global_attn_dropout=best_params.get('dropout', 0.1),
            )
            refit_model.initialize(train_refit)
            refit_opt = Optimizer(lr=best_params['lr'], beta1=PAPER_BETA1, beta2=PAPER_BETA2,
                                  weight_decay=WEIGHT_DECAY)
            refit_crit = None
            if best_params.get('lbl_smooth', None) is not None:
                pos_neg_ratio = 1.0
                pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
                neg_weight = 2 - pos_weight
                refit_crit = SoftNLLLoss(best_params['lbl_smooth'],
                                         torch.Tensor([neg_weight, pos_weight]))

            # Train exactly 'best_epoch' epochs on train+valid (no peeking)
            for e in range(best_epoch):
                refit_model.epoch = e
                MCARunner._run(
                    dataset, dataset_index, 'TRAIN', refit_model, train_refit,
                    criterion=refit_crit,
                    optimizer=refit_opt,
                    train=True,
                    batch_size=best_params['batch'],
                    log_freq=10**9,
                    verbose=False,
                )

            # Save refit checkpoint
            ckpt_dir = os.path.join(CKPT_ROOT, ds_slug, f"idx{dataset_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            refit_ckpt = os.path.join(ckpt_dir, f"{best_run_id}_refit_best.pth")
            refit_model.save_state(refit_ckpt)

            # Evaluate once on TEST at locked threshold
            test_scores = mca_get_scores(refit_model, test_refit, batch_size=best_params['batch'],
                                         path=dataset, dataset_index=dataset_index)
            probs = test_scores['match_score'].astype(float).to_numpy()
            p, r, f1, acc = precision_recall_f1_acc_at_threshold(test_labels, probs, best_thr)
            pra = pr_auc(probs, test_labels)
            test_loss = binary_log_loss(probs, test_labels)

            print(f"===>  TEST_REFIT (best @ epoch {best_epoch}, thr={best_thr:.2f}) | "
                  f"Prec: {p*100:.2f} | Rec: {r*100:.2f} | F1: {f1*100:.2f}")

            write_test_summary(
                RESULTS_CSV, dataset, dataset_index, best_run_id, best_params,
                best_epoch, best_thr,
                p * 100.0, r * 100.0, f1 * 100.0, acc * 100.0,
                pra, test_loss,
                checkpoint_path=refit_ckpt,
                split_label='test_refit'
            )

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    set_seed(42, deterministic=False)

    FASTTEXT_LIMIT = 200_000  # or None for full vocab
    EMBEDDINGS_BIN = 'wiki.en.bin'
    EMBEDDINGS_KV = get_embeddings_path(EMBEDDINGS_BIN, cache_dir=OS_CWD, limit=FASTTEXT_LIMIT)

    # Pass just the basename to let MatchingField join with embeddings_cache_path consistently
    EMBEDDINGS_NAME = os.path.basename(EMBEDDINGS_KV)

    run_for_datasets(DATASETS)
