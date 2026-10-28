#!/usr/bin/env python
"""Architecture-agnostic PEFT layer — backward-compatibility shim.

Core code has moved to the ``slora`` package:
    slora/factorize.py  — pivoted-QR / SVD factorization
    slora/adapters.py   — RowSpaceLinear, LoRA, VeRA modules
    slora/surgery.py    — inject / merge / count

This file re-exports everything so that existing imports like
``from peft_generic import TARGETS, apply_lora, …`` keep working.
"""
from __future__ import annotations

# ===== Re-export the full public API from slora ============================
from slora.factorize import (                          # noqa: F401
    factorize as factorize_conv1d,
    factorize_cached,
)
from slora.adapters import (                           # noqa: F401
    LoRAConv1D,
    LoRALinear,
    RowSpaceLinear,
    VeRALayer,
    VeRAShared,
)
from slora.surgery import (                            # noqa: F401
    TARGETS,
    apply_hybrid,
    apply_lora,
    apply_vera,
    conv1d_weight,
    convert_rowspace,
    get_blocks,
    merge_all,
    merge_lora,
    merge_rowspace,
    merge_vera,
    param_counts,
    predict_params,
    resolve,
    vera_rank_for,
)

# ===== Backward-compat aliases =============================================
# peft_generic.py used to call the rowspace merge "merge_back"
merge_back = merge_rowspace

# peft_generic.py used to export this as "_conv1d_style_weight"
_conv1d_style_weight = conv1d_weight
