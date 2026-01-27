# DeepER.py
# Paper-faithful, configurable DeepER model variants.
# Encoders: avg | lstm
# Similarities: cosine | hadamard | diff | cosine+hadamard
# Heads: linear | mlp (with per-attribute 50-d projection)

from typing import Optional
import torch
import torch.nn as nn

# -----------------------------
# Encoders
# -----------------------------
class AvgEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, dropout: float = 0.0):
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.LongTensor, lengths: torch.LongTensor):
        """
        token_ids: [B, A, L]
        lengths:   [B, A]
        return:    [B, A, D]
        """
        B, A, L = token_ids.shape
        flat = token_ids.view(B * A, L)                        # [B*A, L]
        x = self.embedding(flat)                               # [B*A, L, D]
        lens = lengths.view(B * A).clamp(min=1).float()        # [B*A]
        pad_idx = self.embedding.padding_idx
        if pad_idx is None:
            # if no explicit padding_idx, assume 0 means PAD
            pad_idx = 0
        mask = (flat != pad_idx).unsqueeze(-1)                 # [B*A, L, 1]
        x = (x * mask).sum(dim=1) / lens.unsqueeze(-1)         # [B*A, D]
        x = self.dropout(x)
        return x.view(B, A, -1)                                # [B, A, D]


class SharedBiLSTMEncoder(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden: int = 128, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = embedding
        self.lstm = nn.LSTM(
            embedding.embedding_dim,
            hidden,
            num_layers=layers,
            bidirectional=True,
            batch_first=True,
            # cuDNN requires dropout=0 if layers==1
            dropout=0.0 if layers == 1 else dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.LongTensor, lengths: torch.LongTensor):
        """
        token_ids: [B, A, L]
        lengths:   [B, A]
        return:    [B, A, 2H]
        """
        B, A, L = token_ids.shape
        flat = token_ids.view(B * A, L)                        # [B*A, L]
        x = self.embedding(flat)                               # [B*A, L, D]
        lens = lengths.view(B * A).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        out, (hn, cn) = self.lstm(packed)                      # hn: [2*layers, B*A, H]
        # take last layer's forward/backward states and concat
        fwd = hn[-2]                                           # [B*A, H]
        bwd = hn[-1]                                           # [B*A, H]
        rep = torch.cat([fwd, bwd], dim=-1)                    # [B*A, 2H]
        rep = self.dropout(rep)
        return rep.view(B, A, -1)                              # [B, A, 2H]


# -----------------------------
# Similarity feature builders
# -----------------------------
def sim_features(u: torch.Tensor, v: torch.Tensor, kind: str, eps: float = 1e-8) -> torch.Tensor:
    """
    u, v: [B, A, D]
    return per-attribute similarity features [B, A, F]
    """
    if kind == "cosine":
        un = u / (u.norm(dim=-1, keepdim=True) + eps)
        vn = v / (v.norm(dim=-1, keepdim=True) + eps)
        cos = (un * vn).sum(-1, keepdim=True)                  # [B, A, 1]
        return cos

    if kind == "hadamard":
        return u * v                                           # [B, A, D]

    if kind == "diff":
        return (u - v).abs()                                   # [B, A, D]

    if kind == "cosine+hadamard":
        un = u / (u.norm(dim=-1, keepdim=True) + eps)
        vn = v / (v.norm(dim=-1, keepdim=True) + eps)
        cos = (un * vn).sum(-1, keepdim=True)                  # [B, A, 1]
        had = (u * v)                                          # [B, A, D]
        return torch.cat([cos, had], dim=-1)                   # [B, A, 1 + D]

    raise ValueError(f"Unknown sim kind: {kind}")


# -----------------------------
# Heads
# -----------------------------
class LinearHead(nn.Module):
    def __init__(self, per_attr_dim: int, num_attrs: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(per_attr_dim * num_attrs, 1)

    def forward(self, sim_attr: torch.Tensor):
        B, A, F = sim_attr.shape
        x = self.dropout(sim_attr).reshape(B, A * F)
        logits = self.fc(x).squeeze(-1)
        return torch.sigmoid(logits)


class MLPHead(nn.Module):
    def __init__(self, per_attr_dim: int, num_attrs: int, sim_hidden: int = 50, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(per_attr_dim, sim_hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        in_dim = sim_hidden * num_attrs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, sim_attr: torch.Tensor):
        B, A, F = sim_attr.shape
        z = self.act(self.proj(sim_attr))
        z = self.drop(z)
        x = z.reshape(B, -1)
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)


# -----------------------------
# Configurable DeepER model
# -----------------------------
class DeepERModel(nn.Module):
    """
    encoder: 'avg' | 'lstm'
    sim:     'cosine' | 'hadamard' | 'diff' | 'cosine+hadamard'
    head:    'linear' | 'mlp'
    """
    def __init__(
        self,
        embedding: nn.Embedding,
        num_attrs: int,
        encoder: str = "avg",
        sim: str = "cosine",
        head: str = "linear",
        enc_hidden: int = 128,
        enc_layers: int = 1,
        sim_hidden: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.sim_kind = sim

        # encoder
        if encoder == "avg":
            self.encoder = AvgEncoder(embedding, dropout=0.0)
            rep_dim = embedding.embedding_dim
        elif encoder == "lstm":
            self.encoder = SharedBiLSTMEncoder(embedding, hidden=enc_hidden, layers=enc_layers, dropout=dropout)
            rep_dim = 2 * enc_hidden
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

        # per-attribute similarity dimension
        if sim == "cosine":
            per_attr_dim = 1
        elif sim in ("hadamard", "diff"):
            per_attr_dim = rep_dim
        elif sim == "cosine+hadamard":
            per_attr_dim = 1 + rep_dim
        else:
            raise ValueError(f"Unknown sim: {sim}")

        # head
        if head == "linear":
            self.head = LinearHead(per_attr_dim, num_attrs, dropout=0.0)
        elif head == "mlp":
            self.head = MLPHead(per_attr_dim, num_attrs, sim_hidden=sim_hidden, hidden=256, dropout=dropout)
        else:
            raise ValueError(f"Unknown head: {head}")

    def forward(self, left_ids, left_lens, right_ids, right_lens):
        L = self.encoder(left_ids, left_lens)              # [B, A, D]
        R = self.encoder(right_ids, right_lens)            # [B, A, D]
        S = sim_features(L, R, kind=self.sim_kind)         # [B, A, F]
        return self.head(S)                                # [B]
