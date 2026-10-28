"""S-LoRA (Squared LoRA): parameter-efficient fine-tuning via exact row-space factorization.

Reparametrize each non-square weight W₀ into a trainable square factor W₂
and a frozen rectangular factor [X̄ | I], confining updates to row(W₀).

Quickstart::

    from slora import convert_rowspace, merge_all, TARGETS

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
    report = convert_rowspace(model, TARGETS["qwen2_kv"], rank=4)
    # ... train ...
    merge_all(model, TARGETS["qwen2_kv"])   # fold back for deployment

Modules
-------
slora.factorize  — pivoted-QR / SVD factorization of non-square weights
slora.adapters   — RowSpaceLinear, LoRA, VeRA adapter modules
slora.surgery    — inject / merge adapters, target sets, parameter counting
"""

from slora.factorize import (
    factorize,
    factorize_cached,
    pivoted_qr,
    rebuild_from_pivots,
    reconstruct_P,
)
from slora.adapters import (
    LoRAConv1D,
    LoRALinear,
    RowSpaceLinear,
    VeRALayer,
    VeRAShared,
)
from slora.surgery import (
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
