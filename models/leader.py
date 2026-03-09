"""
LEADER model for medication recommendation.

Architecture:
  - Per-visit set encoder: separate Transformer blocks for diagnosis, procedure, medication
  - Visit-level encoder:   shared Transformer blocks across visits
  - Profile encoder:       embeds patient demographics into soft prompt embeddings
                           (falls back to PaddingEncoder when profile is unavailable)
  - Medication head:       MLP mapping [diag || proc || med] -> multi-hot drug prediction
  - Optional modules:      knowledge distillation, profile-medication alignment, DDI loss
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic Transformer building blocks
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.d_k = hidden_size // num_heads
        self.num_heads = num_heads

        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x    : (..., seq, hidden)
        mask : broadcastable bool mask, 0 = pad position
        """
        *batch_dims, seq_len, _ = x.shape

        def split_heads(t):
            t = t.view(*batch_dims, seq_len, self.num_heads, self.d_k)
            # move head dim before seq dim: (..., heads, seq, d_k)
            dims = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims), len(batch_dims) + 2]
            return t.permute(*dims)

        q, k, v = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # (..., heads, seq, d_k)

        # merge heads back
        dims = list(range(len(batch_dims))) + [len(batch_dims) + 1, len(batch_dims), len(batch_dims) + 2]
        out = out.permute(*dims).contiguous().view(*batch_dims, seq_len, self.num_heads * self.d_k)
        return self.out(out)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.4):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int,
                 intermediate_size: int, hidden_dropout: float = 0.4,
                 attn_dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_heads, attn_dropout)
        self.ff = FeedForward(hidden_size, intermediate_size, hidden_dropout)
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Profile encoders
# ---------------------------------------------------------------------------

class PaddingEncoder(nn.Module):
    """Learnable bias — used when patient profile is unavailable."""

    def __init__(self, device: torch.device, emb_dim: int, prompt_num: int = 1):
        super().__init__()
        self.device = device
        self.prompt_num = prompt_num
        self.padding_emb = nn.Embedding(1, emb_dim)
        self.diag_head = nn.Linear(emb_dim, emb_dim * prompt_num)
        self.proc_head = nn.Linear(emb_dim, emb_dim * prompt_num)
        self.med_head = nn.Linear(emb_dim, emb_dim * prompt_num)

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]
        idx = torch.zeros(bs, dtype=torch.long, device=self.device)
        e = self.padding_emb(idx)  # (bs, emb_dim)
        return self.diag_head(e), self.proc_head(e), self.med_head(e)


class ProfileEncoder(nn.Module):
    """Embeds discrete patient profile features (insurance, ethnicity, etc.)."""

    def __init__(self, device: torch.device, emb_dim: int,
                 profile_tokenizer: dict, prompt_num: int = 1):
        super().__init__()
        self.device = device
        self.prompt_num = prompt_num

        self.encoders = nn.ModuleList(
            nn.Embedding(len(tok), emb_dim)
            for tok in profile_tokenizer["word2idx"].values()
        )
        n = len(profile_tokenizer["word2idx"])
        in_dim = n * emb_dim
        self.diag_head = nn.Linear(in_dim, emb_dim * prompt_num)
        self.proc_head = nn.Linear(in_dim, emb_dim * prompt_num)
        self.med_head = nn.Linear(in_dim, emb_dim * prompt_num)

    def forward(self, profile: torch.Tensor):
        vecs = [enc(profile[:, i]) for i, enc in enumerate(self.encoders)]
        e = torch.cat(vecs, dim=-1)  # (bs, n*emb_dim)
        return self.diag_head(e), self.proc_head(e), self.med_head(e)


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    def __init__(self, tau: float = 1.0,
                 in_dim_1: int = None, in_dim_2: int = None, out_dim: int = None):
        super().__init__()
        self.tau = tau
        self.proj_x = nn.Linear(in_dim_1, out_dim)
        self.proj_y = nn.Linear(in_dim_2, out_dim)

    def _cl(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        sim = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
        pos = torch.exp(torch.diag(sim) / self.tau).unsqueeze(0)
        neg = torch.sum(torch.exp(sim / self.tau), dim=0) - pos
        return -torch.log(pos / neg).view(X.shape[0], -1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X, Y = self.proj_x(X), self.proj_y(Y)
        return self._cl(X, Y) + self._cl(Y, X)


# ---------------------------------------------------------------------------
# LEADER
# ---------------------------------------------------------------------------

class LEADER(nn.Module):
    """
    Parameters
    ----------
    vocab_size     : total unified vocabulary size (diag + proc + med + special tokens)
    med_voc_size   : number of distinct medication classes (output dimension)
    hidden_size    : embedding / transformer hidden size
    num_trm_layers : number of stacked TransformerBlock layers
    num_heads      : attention heads
    intermediate_size : FFN hidden size
    hidden_dropout : dropout on residual connections and FFN
    attn_dropout   : dropout on attention weights
    prompt_num     : number of profile prompt tokens prepended per modality
    distill        : enable knowledge-distillation loss (requires llm_output at train time)
    d_loss         : distillation loss type — only "mse" supported
    alpha          : weight of distillation loss term
    temperature    : temperature for soft-label distillation
    align          : enable profile-medication contrastive alignment
    align_weight   : weight of alignment loss term
    ml_weight      : weight of multi-label margin loss
    """

    def __init__(
        self,
        vocab_size: int,
        med_voc_size: int,
        device: torch.device,
        hidden_size: int = 64,
        num_trm_layers: int = 1,
        num_heads: int = 4,
        intermediate_size: int = 64,
        hidden_dropout: float = 0.4,
        attn_dropout: float = 0.1,
        prompt_num: int = 1,
        profile_tokenizer: dict | None = None,
        distill: bool = False,
        d_loss: str = "mse",
        alpha: float = 0.1,
        temperature: float = 5.0,
        align: bool = False,
        align_weight: float = 0.1,
        ml_weight: float = 0.05,
    ):
        super().__init__()
        self.device = device
        self.emb_dim = hidden_size
        self.num_trm_layers = num_trm_layers
        self.prompt_num = prompt_num
        self.distill = distill
        self.d_loss = d_loss
        self.alpha = alpha
        self.temperature = temperature
        self.align = align
        self.align_weight = align_weight
        self.ml_weight = ml_weight

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5)

        # Profile encoder
        if profile_tokenizer is not None:
            self.profile_encoder = ProfileEncoder(device, hidden_size, profile_tokenizer, prompt_num)
        else:
            self.profile_encoder = PaddingEncoder(device, hidden_size, prompt_num)

        # Per-item set transformers
        trm_kwargs = dict(hidden_size=hidden_size, num_heads=num_heads,
                          intermediate_size=intermediate_size,
                          hidden_dropout=hidden_dropout, attn_dropout=attn_dropout)
        self.diag_trm = nn.ModuleList(TransformerBlock(**trm_kwargs) for _ in range(num_trm_layers))
        self.proc_trm = nn.ModuleList(TransformerBlock(**trm_kwargs) for _ in range(num_trm_layers))
        self.med_trm  = nn.ModuleList(TransformerBlock(**trm_kwargs) for _ in range(num_trm_layers))

        # Visit-level transformer (shared across modalities)
        self.visit_trm = nn.ModuleList(TransformerBlock(**trm_kwargs) for _ in range(num_trm_layers))

        # Medication prediction head
        self.medrec = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3 * hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, med_voc_size),
        )

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.ml_loss  = nn.MultiLabelMarginLoss(reduction="none")

        if distill:
            if d_loss == "mse":
                self.projector = nn.Linear(2 * hidden_size, 4096)
                self.mse_loss = nn.MSELoss(reduction="none")
            else:
                raise ValueError(f"Unsupported distillation loss: {d_loss}")

        if align:
            self.align_loss = ContrastiveLoss(
                tau=1.0,
                in_dim_1=hidden_size,
                in_dim_2=hidden_size * prompt_num,
                out_dim=hidden_size,
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        diag_seq: torch.Tensor,
        proc_seq: torch.Tensor,
        med_seq: torch.Tensor,
        seq_mask: torch.Tensor,
        labels: torch.Tensor,
        profile: torch.Tensor | None = None,
        multi_label: torch.Tensor | None = None,
        llm_output: dict | None = None,
        **kwargs,
    ):
        """
        Shapes
        ------
        diag_seq / proc_seq / med_seq : (bs, max_visits, max_codes)
        seq_mask                       : (bs, max_visits)  — 1 = real visit, 0 = pad
        labels                         : (bs, med_voc_size)
        profile                        : (bs, num_profile_features)  [optional]
        multi_label                    : (bs, med_voc_size) with -1 padding  [optional]
        llm_output                     : dict with "hidden_states" key         [optional]
        """
        # ---- set-level encoding ----
        def set_lengths(seq):
            l = torch.sum(seq > 0, dim=2, keepdim=True)  # (bs, visits, 1)
            l[l == 0] = 1
            return l

        diag_len = set_lengths(diag_seq)
        proc_len = set_lengths(proc_seq)
        med_len  = set_lengths(med_seq)

        diag_emb = self.embeddings(diag_seq)  # (bs, visits, codes, dim)
        proc_emb = self.embeddings(proc_seq)
        med_emb  = self.embeddings(med_seq)

        def set_mask(seq):
            # (bs, visits, codes) -> (bs, visits, 1, codes, codes)
            return (seq > 0).unsqueeze(2).expand_as(
                seq.unsqueeze(2).expand(*seq.shape[:2], seq.shape[2], seq.shape[2])
            ).unsqueeze(2)

        diag_mask = (diag_seq > 0).unsqueeze(2).repeat(1, 1, diag_seq.shape[2], 1).unsqueeze(2)
        proc_mask = (proc_seq > 0).unsqueeze(2).repeat(1, 1, proc_seq.shape[2], 1).unsqueeze(2)
        med_mask  = (med_seq  > 0).unsqueeze(2).repeat(1, 1, med_seq.shape[2],  1).unsqueeze(2)

        for i in range(self.num_trm_layers):
            diag_emb = self.diag_trm[i](diag_emb, diag_mask)
            proc_emb = self.proc_trm[i](proc_emb, proc_mask)
            med_emb  = self.med_trm[i](med_emb,  med_mask)

        # mean pool over code dimension -> (bs, visits, dim)
        diag_emb = torch.sum(diag_emb * (diag_seq > 0).unsqueeze(-1), dim=2) / diag_len
        proc_emb = torch.sum(proc_emb * (proc_seq > 0).unsqueeze(-1), dim=2) / proc_len
        med_emb  = torch.sum(med_emb  * (med_seq  > 0).unsqueeze(-1), dim=2) / med_len

        # ---- profile prompt ----
        if isinstance(self.profile_encoder, ProfileEncoder) and profile is not None:
            _, _, med_pp = self.profile_encoder(profile)
        else:
            _, _, med_pp = self.profile_encoder(seq_mask)
        med_pp = med_pp.view(med_pp.shape[0], self.prompt_num, -1)  # (bs, prompt_num, dim)

        # prepend prompt to medication history; remove last visit (is the label)
        med_emb = torch.cat([med_pp, med_emb], dim=1)[:, :-self.prompt_num, :]

        # ---- visit-level encoding ----
        visit_mask = seq_mask.unsqueeze(1).repeat(1, seq_mask.shape[1], 1).unsqueeze(1)
        for i in range(self.num_trm_layers):
            diag_emb = self.visit_trm[i](diag_emb, visit_mask)
        for i in range(self.num_trm_layers):
            proc_emb = self.visit_trm[i](proc_emb, visit_mask)
        for i in range(self.num_trm_layers):
            med_emb  = self.visit_trm[i](med_emb,  visit_mask)

        # ---- mean pool over visits ----
        visit_len = torch.sum(seq_mask, dim=1, keepdim=True)  # (bs, 1)
        diag_rep = torch.sum(diag_emb * seq_mask.unsqueeze(-1), dim=1) / visit_len
        proc_rep = torch.sum(proc_emb * seq_mask.unsqueeze(-1), dim=1) / visit_len
        med_rep  = torch.sum(med_emb  * seq_mask.unsqueeze(-1), dim=1) / visit_len

        output = self.medrec(torch.cat([diag_rep, proc_rep, med_rep], dim=1))

        if not self.training:
            return output

        # ---- training losses ----
        loss = self.bce_loss(output, labels).mean(dim=-1)

        if self.align and multi_label is not None:
            med_pp_flat = med_pp.view(med_pp.shape[0], -1)
            align_loss = self._align_profile(multi_label, med_pp_flat)
            loss = loss + self.align_weight * align_loss.mean(dim=-1)

        if self.distill and llm_output is not None:
            mediator = self.medrec[1](self.medrec[0](
                torch.cat([diag_rep, proc_rep, med_rep], dim=1)))
            mediator = self.projector(mediator)
            pseudo = llm_output["hidden_states"].float().detach()
            distill_loss = self.mse_loss(mediator, pseudo).mean(dim=-1)
            loss = loss + self.alpha * distill_loss

        return loss, 0, output

    def get_loss(self, **kwargs) -> torch.Tensor:
        loss, _, _ = self(**kwargs)
        return loss.mean()

    # ------------------------------------------------------------------
    # Profile alignment helper
    # ------------------------------------------------------------------

    def _align_profile(self, med_seq: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        med_seq = med_seq.unsqueeze(1)
        med_seq[med_seq < 0] = 0

        med_len = torch.sum(med_seq > 0, dim=2, keepdim=True)
        med_len[med_len == 0] = 1

        med_emb = self.embeddings(med_seq)
        med_mask = (med_seq > 0).unsqueeze(2).repeat(1, 1, med_seq.shape[2], 1).unsqueeze(2)

        for i in range(self.num_trm_layers):
            med_emb = self.med_trm[i](med_emb, med_mask)

        med_emb = torch.sum(med_emb * (med_seq > 0).unsqueeze(-1), dim=2) / med_len
        return self.align_loss(med_emb.squeeze(), encoder_output)
