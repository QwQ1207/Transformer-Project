# add all your Encoder and Decoder code here
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Reusable MHA for encoder/decoder.
    If is_causal=True, applies causal mask (for Part 2 decoder).
    Also supports key padding mask to ignore <pad>.
    Optionally supports AliBi (Part 3).
    """
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        is_causal: bool = False,
        attn_dropout: float = 0.0,
        use_alibi: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.is_causal = is_causal
        self.use_alibi = use_alibi

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # AliBi slopes (not trainable)
        if self.use_alibi:
            slopes = self._get_alibi_slopes(n_head)  # (n_head,)
            self.register_buffer("alibi_slopes", slopes, persistent=False)
        else:
            self.alibi_slopes = None

    @staticmethod
    def _get_alibi_slopes(n_head: int) -> torch.Tensor:
        """
        Standard AliBi slope recipe (works for any n_head).
        Returns slopes shape (n_head,)
        """
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if (n_head & (n_head - 1)) == 0:
            slopes = get_slopes_power_of_2(n_head)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes += extra[0::2][: n_head - closest_power_of_2]

        return torch.tensor(slopes, dtype=torch.float32)

    def _build_alibi_bias(self, T: int, device) -> torch.Tensor:
        """
        Build AliBi bias of shape (1, n_head, T, T)
        bias[h, i, j] = -slope[h] * (i - j) for i>=j (past), else 0 (future will be masked anyway if causal)
        """
        i = torch.arange(T, device=device).view(T, 1)
        j = torch.arange(T, device=device).view(1, T)
        dist = (i - j).clamp(min=0).float()  # (T,T)

        slopes = self.alibi_slopes.view(1, self.n_head, 1, 1)  # (1,nh,1,1)
        bias = -slopes * dist.view(1, 1, T, T)  # (1,nh,T,T)
        return bias

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, T, C)
        key_padding_mask: (B, T) bool tensor, True means PAD positions to mask out as keys
        return:
          out: (B, T, C)
          attn_avg: (B, T, T)  # averaged over heads for Utilities
        """
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)

        # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scores: (B, nh, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # ---- AliBi bias: add BEFORE masking + softmax ----
        if self.use_alibi:
            scores = scores + self._build_alibi_bias(T, x.device)

        # causal mask (decoder)
        if self.is_causal:
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal, float("-inf"))

        # key padding mask: mask out PAD as KEYS
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True => pad
            # expand to (B, 1, 1, T)
            mask = key_padding_mask.view(B, 1, 1, T)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, nh, T, T)
        attn = self.attn_dropout(attn)

        out = attn @ v  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.proj(out)

        # Utilities expects (B, T, T) and checks row sums
        attn_avg = attn.mean(dim=1)  # average over heads -> (B, T, T)
        return out, attn_avg


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, n_hidden: int = 100, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden=100, is_causal=False, dropout=0.1, use_alibi=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(
            n_embd,
            n_head,
            is_causal=is_causal,
            attn_dropout=dropout,
            use_alibi=use_alibi
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, n_hidden=n_hidden, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        a, attn_map = self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x, attn_map


class Encoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer,
                 n_hidden=100, pad_id=0, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd, n_head,
                n_hidden=n_hidden,
                is_causal=False,
                dropout=dropout,
                use_alibi=False  # Encoder 保持 learned pos emb，不在这里用 AliBi（报告也更清晰）
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, idx):
        """
        idx: (B, T)
        returns:
          h: (B, T, C)
          attn_maps: list of length n_layer, each (B, T, T)
        """
        B, T = idx.shape
        assert T <= self.block_size

        key_padding_mask = (idx == self.pad_id)  # (B,T) True means PAD

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1,T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        attn_maps = []
        for blk in self.blocks:
            x, attn_map = blk(x, key_padding_mask=key_padding_mask)  # attn_map: (B,T,T)

            # ---------- FIX: make PAD query rows a valid distribution (sum to 1) ----------
            if key_padding_mask is not None:
                pad_q = key_padding_mask  # (B,T)
                attn_map = torch.nan_to_num(attn_map, nan=0.0, posinf=0.0, neginf=0.0)
                default_cols = torch.zeros((B, T), dtype=torch.long, device=idx.device)
                one_hot = F.one_hot(default_cols, num_classes=T).float()  # (B,T,T)
                attn_map = torch.where(pad_q.unsqueeze(-1), one_hot, attn_map)

            attn_maps.append(attn_map)
            # ---------------------------------------------------------------------------

        x = self.ln_f(x)
        return x, attn_maps


class EncoderClassifier(nn.Module):
    """
    Classification model for Part 1.
    forward returns logits: (B, 3)
    """
    def __init__(self, encoder: Encoder, n_hidden: int = 100, n_output: int = 3, pad_id: int = 0):
        super().__init__()
        self.encoder = encoder
        self.pad_id = pad_id
        self.classifier = nn.Sequential(
            nn.Linear(encoder.n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def masked_mean_pool(self, h, idx):
        """
        h: (B, T, C)
        idx: (B, T) token ids
        mean over non-pad tokens
        """
        mask = (idx != self.pad_id).float()  # (B, T)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B,C)
        return pooled

    def forward(self, idx):
        h, _ = self.encoder(idx)  # h: (B,T,C)
        pooled = self.masked_mean_pool(h, idx)  # (B,C)
        logits = self.classifier(pooled)  # (B,3)
        return logits


class DecoderLM(nn.Module):
    """
    Transformer Decoder for LM: predicts next token.
    """
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd=64,
        n_head=2,
        n_layer=4,
        n_hidden=100,
        dropout=0.1,
        pad_id=0,
        pos_encoding="learned",   # "learned" / "none" / "alibi"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.pad_id = pad_id
        self.pos_encoding = pos_encoding

        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        # learned: use pos_emb
        # none/alibi: no pos_emb (alibi uses bias inside attention)
        self.pos_emb = nn.Embedding(block_size, n_embd) if pos_encoding == "learned" else None

        self.drop = nn.Dropout(dropout)

        use_alibi = (pos_encoding == "alibi")

        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd, n_head,
                n_hidden=n_hidden,
                is_causal=True,
                dropout=dropout,
                use_alibi=use_alibi
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, return_attn=False):
        """
        idx: (B,T) token ids
        targets: (B,T) next-token ids, optional
        return_attn: if True, also return attention maps (for utilities sanity check)

        If targets is not None: return loss (scalar)
        Else: return logits (B,T,V) (and maybe attn_maps)
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        key_padding_mask = (idx == self.pad_id)

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1,T)
        x = self.tok_emb(idx)

        if self.pos_emb is not None:
            x = x + self.pos_emb(pos)

        x = self.drop(x)

        attn_maps = []
        for blk in self.blocks:
            x, attn_map = blk(x, key_padding_mask=key_padding_mask)
            attn_maps.append(attn_map)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        if targets is None:
            if return_attn:
                return logits, attn_maps
            return logits

        loss = F.cross_entropy(
            logits.view(B * T, self.vocab_size),
            targets.view(B * T),
            ignore_index=self.pad_id # ignore PAD tokens in loss (very important for Part 2 when we have padding)
        )
        return loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
