# load_embeddings.py
# -----------------------------------------------------------
# Fast, cache-friendly embedding loader for DeepER-like code
# - One-time convert: .txt/.vec  ->  .kv (memory-mapped)
# - Fast load:        vocab -> torch matrix (cached .pt)
# - Fallback:         filtered text reader if .kv not present
# -----------------------------------------------------------

import os
import io
import json
import hashlib
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Optional: gensim for KeyedVectors memory-mapping
try:
    from gensim.models import KeyedVectors
    _HAS_GENSIM = True
except Exception:
    _HAS_GENSIM = False


# ---------------------------
# One-time converter (.txt/.vec -> .kv)
# ---------------------------
def convert_txt_to_kv(src_txt_path: str, dst_kv_path: Optional[str] = None, has_header: Optional[bool] = None):
    """
    Convert a text-format embedding file (e.g., GloVe or word2vec .vec) into a
    gensim KeyedVectors .kv file, which supports memory-mapped reads.

    Args:
        src_txt_path: path to .txt/.vec (space-separated)
        dst_kv_path:  path to write .kv (default: same name with .kv)
        has_header:   if None, auto-detect; True if first line is "N DIM"
    """
    if not _HAS_GENSIM:
        raise ImportError("gensim is required for convert_txt_to_kv(). pip install gensim")

    if dst_kv_path is None:
        base, _ = os.path.splitext(src_txt_path)
        dst_kv_path = base + ".kv"

    # Auto-detect header by peeking first line
    if has_header is None:
        with io.open(src_txt_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            first = f.readline().strip().split()
        has_header = (len(first) == 2 and all(tok.isdigit() for tok in first))
# 
    print(f"[convert] Loading {src_txt_path} (has_header={has_header}) ...")
    kv = KeyedVectors.load_word2vec_format(src_txt_path, binary=False, no_header=(not has_header))
    kv.fill_norms()  # Precompute norms for faster cosine ops
#     print(f"[convert] Saving memory-mappable KeyedVectors to {dst_kv_path} ...")
    kv.save(dst_kv_path)
#     print("[convert] Done.")


# ---------------------------
# Helper: cache key for matrix cache
# ---------------------------
def _cache_key(emb_source_path: str, dim: int, vocab: Dict[str, int]) -> str:
    meta = {
        "emb_path": os.path.abspath(emb_source_path),
        "dim": dim,
        "vocab_size": len(vocab),
        "vocab_hash": hashlib.md5(" ".join(sorted(vocab.keys())).encode('utf-8')).hexdigest(),
    }
    return hashlib.md5(json.dumps(meta, sort_keys=True).encode('utf-8')).hexdigest()


# ---------------------------
# Fast loader using .kv (memory-mapped) + torch cache
# ---------------------------
def _load_from_kv(vocab: Dict[str, int],
                  kv_path: str,
                  dim: int,
                  cache_dir: str,
                  pad_token: str = "<pad>",
                  finetune: bool = True) -> nn.Embedding:
    if not _HAS_GENSIM:
        raise ImportError("gensim is required to load .kv files. pip install gensim or use text loader.")

    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(kv_path, dim, vocab)
    cache_file = os.path.join(cache_dir, f"{key}.pt")

    if os.path.isfile(cache_file):
        mat = torch.load(cache_file, map_location='cpu')
    else:
        kv = KeyedVectors.load(kv_path, mmap='r')
        mat = torch.randn(len(vocab), dim) * 0.01
        if pad_token in vocab:
            mat[vocab[pad_token]] = torch.zeros(dim)

        # inside _load_from_kv
        for w, idx in vocab.items():
            if kv.has_index_for(w):
                vec = kv.get_vector(w, norm=False)
                if vec.shape[0] == dim:
                    mat[idx] = torch.from_numpy(vec)

        torch.save(mat, cache_file)

    emb = nn.Embedding(len(vocab), dim, padding_idx=vocab.get(pad_token, None))
    emb.weight.data.copy_(mat)
    emb.weight.requires_grad = bool(finetune)
    return emb


# ---------------------------
# Fallback: filtered text reader (no gensim required)
# ---------------------------
def _load_from_txt_filtered(vocab: Dict[str, int],
                            txt_path: str,
                            dim: int,
                            cache_dir: str,
                            pad_token: str = "<pad>",
                            has_header: Optional[bool] = None,
                            finetune: bool = True) -> nn.Embedding:
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(txt_path, dim, vocab)
    cache_file = os.path.join(cache_dir, f"{key}.pt")

    if os.path.isfile(cache_file):
        mat = torch.load(cache_file, map_location='cpu')
    else:
        want = set(vocab.keys())
        mat = torch.randn(len(vocab), dim) * 0.01
        if pad_token in vocab:
            mat[vocab[pad_token]] = torch.zeros(dim)

        with io.open(txt_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            # header auto-detect
            first = f.readline().rstrip('\n')
            toks = first.split()
            if has_header is None:
                has_header = (len(toks) == 2 and all(tok.isdigit() for tok in toks))

            # if no header, parse first line
            if not has_header:
                word = toks[0]
                vec = np.asarray(toks[1:], dtype=np.float32)
                if word in want and vec.shape[0] == dim:
                    mat[vocab[word]] = torch.from_numpy(vec)

            # stream remaining
            for line in f:
                parts = line.rstrip('\n').split()  # <-- use split() (not ' ')
                if not parts:
                    continue
                word, *vals = parts
                if word in want:
                    vec = np.asarray(vals, dtype=np.float32)
                    if vec.shape[0] == dim:
                        mat[vocab[word]] = torch.from_numpy(vec)

        torch.save(mat, cache_file)

    emb = nn.Embedding(len(vocab), dim, padding_idx=vocab.get(pad_token, None))
    emb.weight.data.copy_(mat)
    emb.weight.requires_grad = bool(finetune)
    return emb


# ---------------------------
# Public API
# ---------------------------
def load_embeddings(vocab: Dict[str, int],
                    emb_path: str,
                    dim: int = 300,
                    cache_dir: str = "emb_cache",
                    pad_token: str = "<pad>",
                    finetune: bool = True) -> nn.Embedding:
    """
    Fast, cache-friendly embedding loader.

    Usage:
        emb = load_embeddings(vocab, "glove.840B.300d.kv")      # fastest (memory-mapped)
        emb = load_embeddings(vocab, "glove.840B.300d.txt")     # filtered reader + cache
        emb = load_embeddings(vocab, "wiki.en.vec")             # filtered reader + cache

    Args:
        vocab:     token -> index mapping (must include pad_token if you want padding_idx)
        emb_path:  path to .kv (recommended) or .txt/.vec
        dim:       expected dimension
        cache_dir: where to store per-vocab torch matrices (.pt)
        pad_token: token used for padding; if present in vocab, its row is zeros
        finetune:  if True, embedding weights are trainable

    Returns:
        torch.nn.Embedding with weights initialized from emb_path and cached.
    """
    ext = os.path.splitext(emb_path)[1].lower()

    # Prefer .kv if available
    if ext == ".kv":
        return _load_from_kv(vocab, emb_path, dim, cache_dir, pad_token, finetune)

    # If a same-name .kv exists next to txt/vec, use it automatically
    kv_candidate = os.path.splitext(emb_path)[0] + ".kv"
    if os.path.isfile(kv_candidate):
        return _load_from_kv(vocab, kv_candidate, dim, cache_dir, pad_token, finetune)

    # Fallback to filtered text reader
    return _load_from_txt_filtered(vocab, emb_path, dim, cache_dir, pad_token, finetune=finetune)
