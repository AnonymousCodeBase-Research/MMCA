import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class SoftNLLLoss(nn.NLLLoss):
    def __init__(self, label_smoothing=0, weight=None, num_classes=2, alpha=0.1, gamma=0.5, **kwargs):
        super(SoftNLLLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        self.alpha = alpha          # margin for distance-based term
        self.gamma = gamma          # weight between distance term and CE term
        assert 0.0 <= label_smoothing <= 1.0

        # keep your criterion choice; expects log-probs as input
        self.criterion = nn.KLDivLoss(**kwargs)

        # keep buffer name; avoid deprecated Variable(...)
        if weight is not None:
            self.register_buffer('weight', weight.clone().detach())
        else:
            self.weight = None

    def forward(self, input, target, left_representation, right_representation):
        # Similarity in [0,1]
        sim = (1 + nn.functional.cosine_similarity(left_representation, right_representation)) / 2.0
        y = target.float()
        one_minus_y = 1.0 - y

        # ---- Contrastive (metric) part: hinge on SIMILARITY ----
        # Pull positives together: loss small when sim is high
        pos_loss = y * (1.0 - sim)

        # Push negatives apart ONLY if they are too similar
        # Interpret alpha as a similarity margin m_sim in [0,1]
        m_sim = self.alpha  # e.g., 0.9 → act only when sim > 0.9
        neg_loss = one_minus_y * torch.clamp(sim - m_sim, min=0.0)

        distance_loss = (pos_loss + neg_loss).mean()

        # ---- CE part with label smoothing (unchanged) ----
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        if self.weight is not None:
            one_hot = one_hot * self.weight
        ce_loss = self.criterion(input, one_hot)

        # ---- Combine (gamma for distance term, 1-gamma for CE) ----
        return self.gamma * distance_loss + (1.0 - self.gamma) * ce_loss


class Optimizer(object):
    def __init__(self, method='adam', lr=0.001,
                 max_grad_norm=5,
                 start_decay_at=1,
                 beta1=0.9,
                 beta2=0.999,
                 adagrad_accum=0.0,
                 lr_decay=0.8,
                 weight_decay=0.0):     # <-- NEW
        self.last_acc = None
        self.lr = lr
        self.original_lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = [beta1, beta2]
        self.adagrad_accum = adagrad_accum
        self.params = None
        self.weight_decay = float(weight_decay)  # <-- NEW

    def set_parameters(self, params):
        self.params = []
        for k, p in params:
            if p.requires_grad:
                self.params.append(p)
        self.base_optimizer = optim.Adam(
            self.params,
            lr=self.lr,
            betas=self.betas,
            eps=1e-9,
            weight_decay=self.weight_decay  # <-- NEW
        )

    def _set_rate(self, lr):
        # keep incoming lr param for clarity, but set internal self.lr
        self.lr = lr
        for param_group in self.base_optimizer.param_groups:
            param_group['lr'] = self.lr

    def set_weight_decay(self, wd):  # <-- NEW (optional runtime change)
        self.weight_decay = float(wd)
        for param_group in self.base_optimizer.param_groups:
            param_group['weight_decay'] = self.weight_decay

    def step(self):
        self._step += 1
        if self.max_grad_norm:
            # if you prefer the in-place variant:
            # torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
            clip_grad_norm(self.params, self.max_grad_norm)
        self.base_optimizer.step()

    def update_learning_rate(self, acc, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_acc is not None and acc < self.last_acc:
            self.start_decay = True
        if self.start_decay:
            self.lr = self.lr * self.lr_decay
        self.last_acc = acc
        self._set_rate(self.lr)
