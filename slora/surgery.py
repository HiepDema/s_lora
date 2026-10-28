"""Model surgery: inject adapters, freeze/unfreeze, merge back.

All functions operate **in-place** on the model.  Typical workflow::

    report = convert_rowspace(model, targets, rank=4)   # inject
    # ... train ...
    merge_all(model, targets)                            # fold back
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from slora.adapters import (
    LoRAConv1D,
    LoRALinear,
    RowSpaceLinear,
    VeRALayer,
    VeRAShared,
)
from slora.factorize import factorize, factorize_cached


# ========================================================= target definitions

TARGETS = {
    # GPT-2 (Conv1D layers)
    "gpt2": ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"],

    # Qwen2 / Llama / Mistral — shared module names
    "qwen2_kv":  ["self_attn.k_proj", "self_attn.v_proj"],
    "qwen2_mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "qwen2_all": ["self_attn.k_proj", "self_attn.v_proj",
                   "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "qwen2_qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],

    "llama_kv":  ["self_attn.k_proj", "self_attn.v_proj"],
    "llama_mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],

    # All 7 linear layers (PiSSA/PMSS setup).  Module names happen to be
    # identical across Llama, Mistral, and Qwen2.
    "all7": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],

    # LLM-Adapters / DoRA commonsense setup: q,k,v,up,down (no o_proj, no
    # gate_proj).  This is the configuration behind every LoRA* and DoRA* row
    # quoted for commonsense reasoning, including the MoC paper's Table 1 —
    # LoRA r=32 on these five gives exactly the 0.83/0.71/0.78/0.63 % it lists.
    "qkvud": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "mlp.up_proj", "mlp.down_proj"],
}


# ========================================================= architecture helpers

def get_blocks(model):
    """Return the transformer block list, regardless of architecture.

    Tries common paths in order.  The last two handle multimodal checkpoints
    (Gemma 3 4B/12B/27B, LLaVA) where the language model is nested.
    """
    for path in ("transformer.h",                   # GPT-2
                 "model.layers",                     # Qwen2, Llama, Mistral, Gemma
                 "model.decoder.layers",             # OPT, BART
                 "model.language_model.layers",      # Gemma 3 multimodal
                 "language_model.model.layers"):     # LLaVA-style
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot find transformer blocks in {type(model).__name__}")


def resolve(root, path):
    """Resolve ``'self_attn.k_proj'`` into ``(parent_module, 'k_proj')``."""
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def conv1d_weight(mod):
    """Return ``(weight [d_in, d_out], is_linear)``.

    Normalises ``nn.Linear`` [out, in] and Conv1D [in, out] to a common
    [d_in, d_out] layout so that the same factorization code handles both.
    """
    if isinstance(mod, nn.Linear):
        return mod.weight.data.T.contiguous(), True
    return mod.weight.data, False


def _target_shapes(model, targets):
    """Return ``[(d_in, d_out), ...]`` for each target across all layers."""
    out = []
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            w, _ = conv1d_weight(getattr(parent, attr))
            out.append(tuple(w.shape))
    return out


# ======================================================= inject: S-LoRA

def convert_rowspace(model, targets, basis="colperm", rank=0, alpha=None,
                     dtype=torch.float64, verbose=True, cache=None):
    """Replace target layers with :class:`RowSpaceLinear`.

    Returns a list of per-layer diagnostic dicts (residual, max|X̄|, cond(W₂)).
    """
    report = []
    t0 = time.perf_counter()
    blocks = get_blocks(model)

    for i, block in enumerate(blocks):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, is_lin = conv1d_weight(mod)
            if w.shape[0] == w.shape[1]:
                raise ValueError(
                    f"h.{i}.{name} is square {tuple(w.shape)} — remove from targets")

            cp = f"{cache}/{name}.{i}.{basis}.pt" if cache else None
            fac, orient, info = factorize_cached(w, basis, dtype, cp)
            bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
            layer = RowSpaceLinear(fac, orient, bias, mod.weight.dtype, rank, alpha)

            info.update(layer=f"h.{i}.{name}", target=name, k=layer.k,
                        orient=orient, shape=tuple(w.shape), is_linear=is_lin)
            report.append(info)
            setattr(parent, attr, layer)

        if verbose and (i + 1) % max(1, len(blocks) // 4) == 0:
            print(f"    {i + 1}/{len(blocks)} blocks  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    _freeze_then_unfreeze_rowspace(model)
    return report


def _freeze_then_unfreeze_rowspace(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, RowSpaceLinear):
            if mod.rank == 0:
                mod.C.requires_grad_(True)
            else:
                mod.lora_down.requires_grad_(True)
                mod.lora_up.requires_grad_(True)


# ======================================================= inject: LoRA

def apply_lora(model, targets, rank, alpha, rank_square=None, alpha_square=None):
    """Standard LoRA.

    ``rank_square`` / ``alpha_square`` allow a separate rank for square
    matrices, for fair comparison with hybrid mode.
    """
    rs  = rank  if rank_square  is None else rank_square
    als = alpha if alpha_square is None else alpha_square

    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, _ = conv1d_weight(mod)
            r, al = (rs, als) if w.shape[0] == w.shape[1] else (rank, alpha)
            wrap = LoRALinear if isinstance(mod, nn.Linear) else LoRAConv1D
            setattr(parent, attr, wrap(mod, r, al))

    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (LoRALinear, LoRAConv1D)):
            m.down.requires_grad_(True)
            m.up.requires_grad_(True)
    return model


# ======================================================= inject: VeRA

def apply_vera(model, targets, rank, seed=0, d_init=0.1):
    """VeRA: shared frozen random A, B with per-layer trainable d, b."""
    shapes  = _target_shapes(model, targets)
    max_in  = max(s[0] for s in shapes)
    max_out = max(s[1] for s in shapes)
    ref = next(model.parameters())
    g = torch.Generator(device="cpu").manual_seed(seed)

    A = torch.empty(max_in,  rank,    dtype=ref.dtype).uniform_(
        -1 / math.sqrt(max_in), 1 / math.sqrt(max_in), generator=g)
    B = torch.empty(rank,    max_out, dtype=ref.dtype).uniform_(
        -1 / math.sqrt(rank),   1 / math.sqrt(rank),   generator=g)
    shared = VeRAShared(A.to(ref.device), B.to(ref.device))
    model.add_module("_vera_shared", shared)

    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, _ = conv1d_weight(mod)
            setattr(parent, attr,
                    VeRALayer(mod, shared, w.shape[0], w.shape[1], rank, d_init))

    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, VeRALayer):
            m.vera_d.requires_grad_(True)
            m.vera_b.requires_grad_(True)
    return model


# ======================================================= inject: hybrid

def apply_hybrid(model, targets, rank, rank_square, alpha=None,
                 alpha_square=None, basis="colperm", dtype=torch.float64,
                 verbose=True, cache=None, norm=None):
    """S-LoRA on non-square layers, standard LoRA on square ones.

    Square matrices (q_proj, o_proj) admit no useful factorization — [I|X̄]
    with k = n = m gives an empty X̄.  This mode uses LoRA there and S-LoRA
    everywhere else, with separate ranks for each.
    """
    report = []
    t0 = time.perf_counter()
    blocks = get_blocks(model)
    n_sq = n_fac = 0

    for i, block in enumerate(blocks):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            w, _ = conv1d_weight(mod)

            if w.shape[0] == w.shape[1]:                    # square → LoRA
                wrap = LoRALinear if isinstance(mod, nn.Linear) else LoRAConv1D
                setattr(parent, attr,
                        wrap(mod, rank_square, alpha_square or rank_square))
                n_sq += 1
            else:                                           # non-square → S-LoRA
                cp = f"{cache}/{name}.{i}.{basis}.pt" if cache else None
                fac, orient, info = factorize_cached(w, basis, dtype, cp)
                bias = mod.bias.data if getattr(mod, "bias", None) is not None else None
                layer = RowSpaceLinear(fac, orient, bias, mod.weight.dtype, rank, alpha,
                                       norm=norm)
                info.update(layer=f"h.{i}.{name}", target=name, k=layer.k,
                            orient=orient, shape=tuple(w.shape))
                report.append(info)
                setattr(parent, attr, layer)
                n_fac += 1

        if verbose and (i + 1) % max(1, len(blocks) // 4) == 0:
            print(f"    {i + 1}/{len(blocks)} blocks  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    if verbose:
        print(f"    hybrid: {n_fac} S-LoRA (r={rank}), "
              f"{n_sq} square LoRA (r={rank_square})")

    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, RowSpaceLinear):
            if mod.rank == 0:
                mod.C.requires_grad_(True)
            else:
                mod.lora_down.requires_grad_(True)
                mod.lora_up.requires_grad_(True)
        elif isinstance(mod, (LoRALinear, LoRAConv1D)):
            mod.down.requires_grad_(True)
            mod.up.requires_grad_(True)
    return report


# ======================================================= merge back

@torch.no_grad()
def merge_rowspace(model, targets):
    """Merge :class:`RowSpaceLinear` back into ``nn.Linear``."""
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, RowSpaceLinear):
                continue
            W   = mod.merged_weight()                   # [d_in, d_out]
            lin = nn.Linear(W.shape[0], W.shape[1], bias=mod.bias is not None)
            lin = lin.to(W.device, W.dtype)
            lin.weight.data.copy_(W.T)                  # Linear expects [out, in]
            if mod.bias is not None:
                lin.bias.data.copy_(mod.bias.data)
            setattr(parent, attr, lin)
    return model


@torch.no_grad()
def merge_lora(model, targets):
    """Merge LoRA into base weights — restores original architecture."""
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, (LoRALinear, LoRAConv1D)):
                continue
            delta = mod.scale * (mod.down.data.double() @ mod.up.data.double())
            w = mod.base.weight.data
            w += (delta.T if isinstance(mod.base, nn.Linear) else delta).to(w.dtype)
            setattr(parent, attr, mod.base)
    return model


@torch.no_grad()
def merge_vera(model, targets):
    """Merge VeRA into base weights — restores original architecture."""
    for block in get_blocks(model):
        for name in targets:
            parent, attr = resolve(block, name)
            mod = getattr(parent, attr)
            if not isinstance(mod, VeRALayer):
                continue
            A, B = mod._AB()
            delta = ((A.double() * mod.vera_d.double()) @ B.double()
                     ) * mod.vera_b.double()
            w = mod.base.weight.data
            w += (delta.T if isinstance(mod.base, nn.Linear) else delta).to(w.dtype)
            setattr(parent, attr, mod.base)
    if hasattr(model, "_vera_shared"):
        del model._vera_shared
    return model


def merge_all(model, targets):
    """Merge any adapter type back into original layers."""
    merge_rowspace(model, targets)
    merge_vera(model, targets)
    return merge_lora(model, targets)


# ======================================================= parameter counting

def param_counts(model, n_orig):
    """Return dict with trainable / frozen / total parameter counts."""
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_buf = sum(m.frozen_numel() for m in model.modules()
                     if isinstance(m, RowSpaceLinear))
    total = sum(p.numel() for p in model.parameters()) + frozen_buf
    return dict(trainable=trainable, frozen=frozen_buf, total=total, n_orig=n_orig)


def predict_params(shapes, rank, n_layers, method):
    """Predict trainable-parameter count before building the model.

    *shapes*: ``[(out_features, in_features), ...]`` per target per layer.
    """
    per_layer = 0
    for out_dim, in_dim in shapes:
        k = min(out_dim, in_dim)
        if method == "rowspace":
            per_layer += 2 * k * rank
        elif method == "vera":
            per_layer += rank + out_dim
        elif method == "hybrid":
            per_layer += 2 * k * rank if out_dim != in_dim else rank * (out_dim + in_dim)
        else:                                               # lora
            per_layer += rank * (out_dim + in_dim)
    return per_layer * n_layers


def vera_rank_for(shapes, n_layers, budget):
    """Choose VeRA rank to match a given parameter budget."""
    n = len(shapes) * n_layers
    return max(1, round((budget - sum(o for o, _ in shapes) * n_layers) / n))
