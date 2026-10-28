"""Adapter modules for S-LoRA, LoRA, and VeRA.

RowSpaceLinear
    Replaces a non-square linear layer with a factored form::

        wide (m>n):  y = W₂( X̄·x_rest + x_sel )       input mixing before square
        tall (n>m):  y = [h ; X̄·h] reordered            output expansion after square

    W₂ is the trainable square factor; X̄ is frozen.  Optionally W₂ itself is
    frozen and a LoRA adapter is applied to it (``rank > 0``).

LoRALinear / LoRAConv1D
    Standard LoRA:  y = base(x) + (α/r)(x A) B.

VeRALayer
    VeRA (Kopiczko et al., ICLR 2024):  y = base(x) + (((x A) ⊙ d) B) ⊙ b,
    with A, B frozen shared random matrices and d, b trainable vectors.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ======================================================================= S-LoRA

class RowSpaceLinear(nn.Module):
    """Linear layer reparametrised into frozen rectangular + trainable square.

    Training modes
    --------------
    rank = 0   Train W₂ directly (k² parameters per layer).
    rank > 0   Freeze W₂, train LoRA on it (2 r k parameters per layer).

    Forward path
    -------------
    pre-orientation  (wide weights):  h = in_proj(x) → square(h)
    post-orientation (tall weights):  h = square(x)  → out_proj(h)
    """

    def __init__(self, fac, orientation, bias, out_dtype, rank=0, alpha=None, norm=None):
        super().__init__()
        self.orientation = orientation
        self.rank = rank
        self.structured = (fac["kind"] == "struct")

        # W₂ᵀ in row-vector layout: forward does  h @ C  instead of  C @ h
        C = fac["W2"].T.contiguous().to(out_dtype)
        k = C.shape[0]
        self.k = k

        # --- frozen rectangular factor ---
        if self.structured:
            self.register_buffer("X",    fac["X"].to(out_dtype))
            self.register_buffer("sel",  fac["sel"])
            self.register_buffer("rest", fac["rest"])
            if orientation == "post":
                perm = torch.cat([fac["sel"], fac["rest"]])
                self.register_buffer("inv", torch.argsort(perm))
                self.nf = perm.numel()
            else:
                self.nf = k
        else:
            self.register_buffer("P", fac["P"].T.contiguous().to(out_dtype))
            self.nf = self.P.shape[1] if orientation == "post" else k

        # --- bias ---
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        # --- trainable square factor ---
        if rank == 0:
            self.C     = nn.Parameter(C)
            self.scale = 1.0
        else:
            self.register_buffer("C", C)
            self.lora_down = nn.Parameter(
                torch.empty(k, rank, dtype=C.dtype, device=C.device))
            self.lora_up = nn.Parameter(
                torch.zeros(rank, k, dtype=C.dtype, device=C.device))
            nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
            self.scale = (alpha if alpha is not None else rank) / rank
            # norm="rms": P = [I | X̄] khuech dai delta len ‖P‖_F/√k lan (do tren
            # LLaMA3-8B: 17x o k/v, 167x o up_proj), nen cung mot lr se manh hon
            # LoRA hang tram lan. Chia scale cho he so do de E‖ΔW‖ = ‖ΔC‖.
            if norm == "rms" and self.structured:
                xf2 = fac["X"].double().pow(2).sum().item()
                self.scale *= math.sqrt(k / (k + xf2))

    # ---- class method kept for backward compat (GPT-2 CLI) ----

    @classmethod
    def from_conv1d(cls, conv, basis="colperm", rank=0, alpha=None,
                    dtype=torch.float64):
        from slora.factorize import factorize
        fac, orient, info = factorize(conv.weight.data, basis, dtype)
        bias = conv.bias.data if conv.bias is not None else None
        return cls(fac, orient, bias, conv.weight.dtype, rank, alpha), info

    # ---- forward: three composable stages ----

    def _square(self, h):
        """Apply the trainable square factor (with optional LoRA)."""
        out = h @ self.C
        if self.rank > 0:
            out = out + self.scale * ((h @ self.lora_down) @ self.lora_up)
        return out

    def _in_proj(self, x):
        """d_in → k:  mix input dimensions before the square factor (wide).

        Structured:  h = x[sel] + x[rest] @ X̄ᵀ   (identity = index gather)
        Dense:       h = x @ P
        """
        if self.structured:
            return (x.index_select(-1, self.sel)
                    + x.index_select(-1, self.rest) @ self.X.T)
        return x @ self.P

    def _out_proj(self, h):
        """k → d_out:  expand output after the square factor (tall).

        Structured:  [h, h @ X̄ᵀ] reordered by inverse permutation
        Dense:       h @ P
        """
        if self.structured:
            return torch.cat([h, h @ self.X.T], dim=-1).index_select(-1, self.inv)
        return h @ self.P

    def forward(self, x):
        if self.orientation == "pre":
            out = self._square(self._in_proj(x))
        else:
            out = self._out_proj(self._square(x))
        if self.bias is not None:
            out = out + self.bias
        return out

    # ---- utilities ----

    def effective_C(self):
        """W₂ᵀ with LoRA merged, in float64 for precision."""
        C = self.C.double()
        if self.rank > 0:
            C = C + self.scale * (self.lora_down.double() @ self.lora_up.double())
        return C

    @torch.no_grad()
    def merged_weight(self):
        """Reconstruct the full [d_in, d_out] weight — zero inference overhead."""
        C = self.effective_C()

        if not self.structured:
            P = self.P.double()
            return (P @ C if self.orientation == "pre" else C @ P).to(self.C.dtype)

        X = self.X.double()
        if self.orientation == "pre":
            W = torch.zeros(self.sel.numel() + self.rest.numel(), C.shape[1],
                            dtype=torch.float64, device=C.device)
            W[self.sel]  = C
            W[self.rest] = X.T @ C
        else:
            W = torch.zeros(C.shape[0], self.sel.numel() + self.rest.numel(),
                            dtype=torch.float64, device=C.device)
            W[:, self.sel]  = C
            W[:, self.rest] = C @ X.T
        return W.to(self.C.dtype)

    def frozen_numel(self):
        """Count frozen buffer parameters (X/P, plus C when using LoRA)."""
        n = self.X.numel() if self.structured else self.P.numel()
        if self.rank > 0:
            n += self.C.numel()
        return n

    def extra_repr(self):
        basis = "structured [I|X̄]" if self.structured else "dense P"
        return f"k={self.k}, orient={self.orientation}, {basis}, rank={self.rank}"


# ======================================================================== LoRA

class LoRALinear(nn.Module):
    """LoRA for ``nn.Linear``:  y = base(x) + (α/r)(x·A)·B."""

    def __init__(self, lin, rank, alpha):
        super().__init__()
        self.base = lin
        d_in, d_out = lin.in_features, lin.out_features
        dt, dv = lin.weight.dtype, lin.weight.device
        self.down  = nn.Parameter(torch.empty(d_in, rank, dtype=dt, device=dv))
        self.up    = nn.Parameter(torch.zeros(rank, d_out, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)


class LoRAConv1D(nn.Module):
    """LoRA for GPT-2's Conv1D:  y = base(x) + (α/r)(x·A)·B."""

    def __init__(self, conv, rank, alpha):
        super().__init__()
        self.base = conv
        d_in, d_out = conv.weight.shape
        dt, dv = conv.weight.dtype, conv.weight.device
        self.down  = nn.Parameter(torch.empty(d_in, rank, dtype=dt, device=dv))
        self.up    = nn.Parameter(torch.zeros(rank, d_out, dtype=dt, device=dv))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank
        self.nf    = conv.nf

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)


# ======================================================================== VeRA

class VeRAShared(nn.Module):
    """Shared frozen random matrices A and B for all VeRA layers.

    Registered as non-persistent buffers (reproducible from seed, not saved).
    ``VeRALayer`` references this via ``object.__setattr__`` to avoid
    ``nn.Module`` ownership — otherwise ``model.to(device)`` would create
    separate copies per layer.
    """

    def __init__(self, A, B):
        super().__init__()
        self.register_buffer("A", A, persistent=False)
        self.register_buffer("B", B, persistent=False)


class VeRALayer(nn.Module):
    """VeRA adapter:  y = base(x) + (((x·A) ⊙ d) · B) ⊙ b.

    A [d_in, r] and B [r, d_out] are frozen and shared across all layers.
    Only *d* (r,) and *b* (d_out,) are trainable — total r + d_out params/layer.
    *b* is zero-initialised → the adapter starts as identity.
    """

    def __init__(self, base, shared, d_in, d_out, rank, d_init):
        super().__init__()
        self.base = base
        object.__setattr__(self, "shared", shared)
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        w = base.weight
        self.vera_d = nn.Parameter(
            torch.full((rank,), d_init, dtype=w.dtype, device=w.device))
        self.vera_b = nn.Parameter(
            torch.zeros(d_out, dtype=w.dtype, device=w.device))
        if not isinstance(base, nn.Linear):
            self.nf = base.nf

    def _AB(self):
        return (self.shared.A[:self.d_in, :self.rank],
                self.shared.B[:self.rank, :self.d_out])

    def forward(self, x):
        A, B = self._AB()
        return self.base(x) + ((((x @ A) * self.vera_d) @ B) * self.vera_b)
