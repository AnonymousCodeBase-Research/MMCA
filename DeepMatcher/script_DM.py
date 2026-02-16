from __future__ import annotations
import os, csv, json, time, hashlib, random
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from gensim.models.fasttext import load_facebook_vectors, FastTextKeyedVectors

import deepmatcher as dm
from deepmatcher.runner import Runner
from deepmatcher.optim import SoftNLLLoss

# =========================================================
# Embeddings cache helpers (optional .bin -> .kv conversion)
# =========================================================
_EMB_PATH_CACHE: Dict[Tuple[str, str, int | None], str] = {}

def get_embeddings_path(bin_or_kv_path: str, cache_dir: str, limit: int | None = None) -> str:
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
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(bin_path))[0]
    kv_path = os.path.join(cache_dir, f"{base}.kv")
    if os.path.exists(kv_path):
        return kv_path
    kv: FastTextKeyedVectors = load_facebook_vectors(bin_path)  # keep subword info
    kv.save(kv_path)
    return kv_path

# ========== Repro ==========
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try: torch.use_deterministic_algorithms(True)
        except Exception: pass

# ================== Config / Defaults ==================
RESULTS_CSV = os.path.join("runs", "results.csv")
CKPT_ROOT   = os.path.join("runs", "checkpoints")
OS_CWD      = os.getcwd()

DATASETS = ["datasets/Structured_Textual/AB/","datasets/Structured_Textual/AG/","datasets/Structured_Textual/DA/",
	"datasets/Structured_Textual/DS/", "datasets/Structured_Textual/WA/",
	"datasets/Textual/AB/","datasets/Textual/AG/","datasets/Textual/DA/",
	"datasets/Textual/DS/", "datasets/Textual/WA/",
	"datasets/Dirty/DA/","datasets/Dirty/DS/","datasets/Dirty/WA/"
]

# Grid (kept simple; rnn_layers is NOT passed to Hybrid)
GRID: List[Dict[str, Any]] = [
    dict(batch=batch, lr=lr, wd=1e-6, dropout=0.1,
         rnn_hidden=300, attn_heads=1,
         clip=0.5,
         lbl_smooth=0.05)
    for batch in [16, 32, 64]
    for lr in [1e-2, 1e-3, 1e-4]
]

MAX_EPOCHS = 64
EARLY_STOP_PATIENCE = 5
EMBEDDINGS_BIN = 'wiki.en.kv'
#PII Dataset
#IGNORE_COLUMNS = ['query', 'source']
#ER Dataset
IGNORE_COLUMNS = ['left_id', 'right_id']

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[device] Using {DEVICE} (cuda_available={torch.cuda.is_available()})")

class EarlyStopRequested(Exception): pass

# ========================= CSV schema & I/O helpers =========================
CSV_COLUMNS = [
    'dataset','split','index','run_id','timestamp',
    'batch_size','lr',                    # removed enc_hidden/enc_layers
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

# ========================= Metrics & sweep helpers =========================
def precision_recall_f1_acc_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float):
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
    for yi in y:
        if yi == 1: tp += 1
        else:       fp += 1
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

# ========================= CSV / ID helpers =========================
import hashlib as _hashlib

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
    preferred_order = ['name','address','phone','email','email_list','bio',
                       'country','state','city','zip','aliases']
    ordered_bases = [b for b in preferred_order if b in common_bases] or common_bases
    if ordered_bases:
        concat_cols = []
        for b in ordered_bases:
            concat_cols.extend([left_bases[b], right_bases[b]])
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
    # removed enc_hidden/enc_layers from run_id key
    blob = json.dumps({'batch': params['batch'], 'lr': params['lr']}, sort_keys=True)
    return f"{dataset_slug(dataset)}_{idx}_{hashlib.md5(blob.encode()).hexdigest()[:6]}"

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

def _prepare_full_splits(dataset_dir: str, dataset_index: int) -> Tuple[str, str, str]:
    for split in ['train', 'valid', 'test']:
        p = os.path.join(dataset_dir, f"{split}_{dataset_index}.csv")
        ensure_pair_id(p)
        align_left_right_columns(p)
    return (f"train_{dataset_index}.csv",
            f"valid_{dataset_index}.csv",
            f"test_{dataset_index}.csv")

# ============== CSV writers ==============
def _base_row(dataset, split, idx, run_id, params):
    return dict(
        dataset=dataset, split=split, index=idx, run_id=run_id,
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        batch_size=params['batch'], lr=params['lr'],
        epoch=None, epochs_trained=None, threshold=None,
        precision=None, recall=None, f1=None, accuracy=None, pr_auc=None, loss=None,
        checkpoint_path=''
    )

def write_train_epoch_row(csv_path, dataset, idx, run_id, params,
                          epoch, precision, recall, f1, accuracy, loss):
    row = _base_row(dataset, 'train', idx, run_id, params)
    row.update(dict(epoch=epoch, precision=precision, recall=recall, f1=f1,
                    accuracy=accuracy, loss=loss))
    append_csv_row(csv_path, row)

def write_valid_epoch_row(csv_path, dataset, idx, run_id, params,
                          epoch, precision, recall, f1, accuracy, prauc, loss, threshold):
    row = _base_row(dataset, 'valid', idx, run_id, params)
    row.update(dict(epoch=epoch, precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                    pr_auc=prauc, loss=loss, threshold=threshold))
    append_csv_row(csv_path, row)

def write_tune_summary(csv_path, dataset, idx, run_id, params,
                       best_epoch, best_threshold, best_f1, best_prec, best_rec, best_acc, best_prauc):
    row = _base_row(dataset, 'tune_best', idx, run_id, params)
    row.update(dict(epoch=best_epoch, epochs_trained=best_epoch, threshold=best_threshold,
                    precision=best_prec*100.0, recall=best_rec*100.0, f1=best_f1*100.0,
                    accuracy=best_acc*100.0, pr_auc=best_prauc, loss=None))
    append_csv_row(csv_path, row)

def write_test_summary(csv_path, dataset, idx, run_id, params,
                       best_epoch, threshold, precision, recall, f1, accuracy, prauc, loss,
                       checkpoint_path: str, split_label: str = 'test'):
    row = _base_row(dataset, split_label, idx, run_id, params)
    row.update(dict(epoch=best_epoch, epochs_trained=best_epoch, threshold=threshold,
                    precision=precision, recall=recall, f1=f1, accuracy=accuracy,
                    pr_auc=prauc, loss=loss, checkpoint_path=checkpoint_path))
    append_csv_row(csv_path, row)

# =========================== DeepMatcher score helper ===========================
def dm_get_scores(model, dataset_obj, batch_size) -> pd.DataFrame:
    preds = Runner._run(
        path='', dataset_index=0, run_type='TEST',
        model=model, dataset=dataset_obj,
        train=False, batch_size=batch_size,
        return_predictions=True, log_freq=10**9, verbose=False,
        device=DEVICE  # ensure GPU for scoring path if available
    )
    return pd.DataFrame(preds, columns=('id', 'match_score'))

# =========================== Model builder (no rnn_layers passed) ===========================
def build_dm_model(params) -> dm.MatchingModel:
    comp = dm.word_comparators.Attention(
        heads=params.get('attn_heads', 1),
        hidden_size=params['rnn_hidden'],
        raw_alignment=True,
        alignment_network='decomposable'
    )
    hybrid_kwargs = dict(
        word_contextualizer='gru',
        word_comparator=comp,
        word_aggregator='attention-with-rnn',
        hidden_size=params['rnn_hidden']
        # IMPORTANT: do NOT pass rnn_layers here
    )
    return dm.MatchingModel(attr_summarizer=dm.attr_summarizers.Hybrid(**hybrid_kwargs))

# =========================== Main training flow ===========================
def run_for_datasets(dataset_name_list: List[str],
                     embeddings_name: str):
    ensure_csv(RESULTS_CSV)

    for dataset in dataset_name_list:
        ds_slug = dataset_slug(dataset)

        for dataset_index in range(10):
            train_csv, valid_csv, test_csv = _prepare_full_splits(dataset, dataset_index)

            train, valid, test = dm.data.process(
                path=dataset,
                train=train_csv,
                validation=valid_csv,
                test=test_csv,
                embeddings=embeddings_name,
                embeddings_cache_path=OS_CWD,
                ignore_columns=IGNORE_COLUMNS,
                id_attr='id',
                label_attr='label',
                left_prefix='left_',
                right_prefix='right_',
                pca=True
            )

            train_labels = get_labels_for(train)
            valid_labels = get_labels_for(valid)
            test_labels  = get_labels_for(test)

            # Best across the grid for this fold
            fold_best = dict(f1=-1.0, epoch=None, threshold=None, params=None,
                             prec=None, rec=None, acc=None, prauc=None,
                             ckpt_path=None, run_id=None)

            for params in GRID:
                run_id = make_run_id(dataset, dataset_index, params)
                per_run_seed = int(hashlib.md5(run_id.encode()).hexdigest()[:8], 16)
                set_seed(per_run_seed, deterministic=False)

                ckpt_dir = os.path.join(CKPT_ROOT, ds_slug, f"idx{dataset_index}")
                os.makedirs(ckpt_dir, exist_ok=True)
                per_grid_best_path = os.path.join(ckpt_dir, f"{run_id}_best.pth")
                meta_json_path     = os.path.join(ckpt_dir, f"{run_id}_best.json")

                model = build_dm_model(params)

                criterion = None
                if params.get('lbl_smooth', None) is not None:
                    pos_neg_ratio = 1.0
                    pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
                    neg_weight = 2 - pos_weight
                    criterion = SoftNLLLoss(params['lbl_smooth'],
                                            torch.Tensor([neg_weight, pos_weight]))

                run_best = {'f1': -1.0, 'epoch': None, 'threshold': None,
                            'prec': None, 'rec': None, 'acc': None, 'prauc': None}
                no_improve_count = 0
                last_improve_epoch = -1

                def _epoch_cb(info,
                              _dataset=dataset, _idx=dataset_index, _rid=run_id, _p=params,
                              _model=model, _valid=valid, _valid_labels=valid_labels):
                    nonlocal run_best, last_improve_epoch, no_improve_count

                    # TRAIN row (DM reports percents already)
                    write_train_epoch_row(
                        RESULTS_CSV, _dataset, _idx, _rid, _p,
                        info.get('epoch'),
                        info.get('precision'), info.get('recall'), info.get('f1'),
                        info.get('accuracy'), info.get('loss')
                    )

                    # VALID sweep
                    t0 = time.time()
                    scores_df = _model.run_prediction(_valid, batch_size=_p['batch'])
                    thrres = sweep_threshold(scores_df, _valid_labels)
                    elapsed = time.time() - t0
                    probs = scores_df['match_score'].to_numpy()

                    write_valid_epoch_row(
                        RESULTS_CSV, _dataset, _idx, _rid, _p,
                        info.get('epoch'),
                        thrres.precision * 100.0,
                        thrres.recall * 100.0,
                        thrres.best_f1 * 100.0,
                        thrres.accuracy * 100.0,
                        thrres.pr_auc,
                        binary_log_loss(probs, _valid_labels),
                        thrres.value
                    )

                    print(f"\n===>  EVAL Epoch {info.get('epoch')}")
                    print("0% [██████] 100% | ETA: 00:00:00")
                    mm = int(elapsed // 60); ss = int(round(elapsed % 60))
                    print(f"Total time elapsed: 00:{mm:02d}:{ss:02d}")
                    print(("Finished Epoch {epoch} || Run Time: {runtime:6.1f} | Load Time: {datatime:6.1f} || "
                           "Prec: {prec:6.2f} | Rec: {rec:6.2f} | F1: {f1:6.2f} || Ex/s: {eps:6.2f}").format(
                        epoch=info.get('epoch'),
                        runtime=elapsed, datatime=0.0,
                        prec=thrres.precision * 100.0,
                        rec=thrres.recall * 100.0,
                        f1=thrres.best_f1 * 100.0,
                        eps=(len(scores_df) / max(elapsed, 1e-8))
                    ))
                    print()

                    improved = thrres.best_f1 > run_best['f1']
                    if improved:
                        _model.save_state(per_grid_best_path)
                        with open(meta_json_path, "w", encoding="utf-8") as f:
                            json.dump({"epoch": int(info.get('epoch')),
                                       "threshold": float(thrres.value)}, f)
                        run_best.update(dict(f1=thrres.best_f1,
                                             epoch=info.get('epoch'),
                                             threshold=thrres.value,
                                             prec=thrres.precision,
                                             rec=thrres.recall,
                                             acc=thrres.accuracy,
                                             prauc=thrres.pr_auc))
                        last_improve_epoch = info.get('epoch')
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if EARLY_STOP_PATIENCE and no_improve_count >= EARLY_STOP_PATIENCE:
                            raise EarlyStopRequested()

                try:
                    # Use DM trainer for the update pass; disable its internal eval
                    _ = model.run_train(
                        train, valid,
                        best_save_path=None,
                        epochs=MAX_EPOCHS,
                        batch_size=params['batch'],
                        dropout=params['dropout'],
                        weight_decay=params['wd'],
                        grad_clip=params['clip'],
                        lr_reduce_on_plateau=False,
                        early_stop_metric='f1',
                        early_stop_patience=MAX_EPOCHS,  # we control patience ourselves
                        internal_eval=False,
                        epoch_callback=_epoch_cb
                    )
                except EarlyStopRequested:
                    stop_at = (last_improve_epoch if last_improve_epoch >= 0 else 0) + no_improve_count
                    print(f"Early stopping (sweep-based) at epoch {stop_at} after "
                          f"{no_improve_count} epochs without improvement.")

                # Track fold-best
                if run_best['f1'] > fold_best['f1']:
                    fold_best.update(run_best)
                    fold_best['params']  = params
                    fold_best['ckpt_path'] = per_grid_best_path
                    fold_best['run_id']  = run_id

                # Record per-run tuning summary
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

            # =========================== Refit on TRAIN+VALID, final TEST ===========================
            if fold_best['epoch'] is None:
                continue

            best_params   = fold_best['params']
            best_epoch    = int(fold_best['epoch'])
            best_thr      = float(fold_best['threshold'])
            best_run_id   = str(fold_best['run_id'])

            merged_basename = f"trainvalid_{dataset_index}__from_used.csv"
            merged_path = make_trainvalid_csv_from_files(dataset, train_csv, valid_csv, merged_basename)
            ensure_pair_id(merged_path); align_left_right_columns(merged_path)

            train_refit, _valid_dummy, test_refit = dm.data.process(
                path=dataset,
                train=merged_basename,
                validation=valid_csv,  # placeholder; not used
                test=test_csv,
                embeddings=embeddings_name,
                embeddings_cache_path=OS_CWD,
                ignore_columns=IGNORE_COLUMNS,
                id_attr='id',
                label_attr='label',
                left_prefix='left_',
                right_prefix='right_',
                pca=True
            )

            refit_model = build_dm_model(best_params)

            _ = refit_model.run_train(
                train_refit, _valid_dummy,  # validation is a placeholder
                best_save_path=None,
                epochs=best_epoch,
                batch_size=best_params['batch'],
                dropout=best_params['dropout'],
                weight_decay=best_params['wd'],
                grad_clip=best_params['clip'],
                lr_reduce_on_plateau=False,
                early_stop_metric='f1',
                early_stop_patience=best_epoch,
                internal_eval=False,
                epoch_callback=None
            )

            # Save refit checkpoint
            ckpt_dir = os.path.join(CKPT_ROOT, ds_slug, f"idx{dataset_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            refit_ckpt = os.path.join(ckpt_dir, f"{best_run_id}_refit_best.pth")
            refit_model.save_state(refit_ckpt)

            # Final TEST at locked threshold
            test_scores_final = refit_model.run_prediction(test_refit, batch_size=best_params['batch'])
            probs_final = test_scores_final['match_score'].to_numpy()
            p, r, f1, acc = precision_recall_f1_acc_at_threshold(test_labels, probs_final, best_thr)
            pra = pr_auc(probs_final, test_labels)
            test_loss = binary_log_loss(probs_final, test_labels)

            print(f"===>  TEST_REFIT (best @ epoch {best_epoch}, thr={best_thr:.2f}) | "
                  f"Prec: {p*100:.2f} | Rec: {r*100:.2f} | F1: {f1*100:.2f}")

            write_test_summary(
                RESULTS_CSV, dataset, dataset_index, best_run_id, best_params,
                best_epoch, best_thr,
                p*100.0, r*100.0, f1*100.0, acc*100.0,
                pra, test_loss,
                checkpoint_path=refit_ckpt,
                split_label='test_refit'
            )

# ========== Main ==========
if __name__ == "__main__":
    set_seed(42, deterministic=False)

    # Resolve embeddings to a load-friendly name (basename)
    FASTTEXT_LIMIT = 200_000  # or None
    EMBEDDINGS_KV = get_embeddings_path(EMBEDDINGS_BIN, cache_dir=OS_CWD, limit=FASTTEXT_LIMIT)
    EMBEDDINGS_NAME = os.path.basename(EMBEDDINGS_KV)

    run_for_datasets(DATASETS, embeddings_name=EMBEDDINGS_NAME)
