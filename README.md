# S-LoRA (Squared LoRA)

Parameter-efficient fine-tuning by **reparametrizing** a layer instead of adding to it.

Every PEFT method in the LoRA family learns an additive correction, `W₀ + BA`, whose cost
is `O(r·(n+m))` — it scales with *both* dimensions of the weight. That is wasteful for
lopsided matrices, and grouped-query attention produces plenty of them: in Qwen2.5-1.5B the
key and value projections map 1536 inputs to 256 outputs, an aspect ratio of 6.

S-LoRA takes a different route. Column-pivoted QR gives an **exact** factorization of any
non-square weight:

```
W·Π = W₂ · [ I | X ]        X = W₂⁻¹W₁,  W₂ square of side k = min(n, m)
```

Freeze `[I | X]`, adapt only `W₂`. Three consequences:

- **The update is confined to the row space of `W₀`** (column space for tall matrices).
- **Parameter count is preserved exactly** — only `X` is stored, never the identity block,
  which is an index selection rather than a matmul.
- **The adapted matrix is square**, so any adapter whose cost scales with matrix dimensions
  becomes cheaper by `(1+a)/2`, where `a = max(n,m)/min(n,m)`. For full fine-tuning the
  factor is `a`.

After training the layer folds back into a single matrix, so inference cost is unchanged.

## Results

Qwen2.5-1.5B, adapting `k_proj` + `v_proj` only (28 layers, `a = 6`), E2E NLG,
5 epochs with best-epoch selection, beam 10, 630 test sentences.

| Method    |    r | Trainable | BLEU-4 |   sd | seeds |
|-----------|-----:|----------:|-------:|-----:|------:|
| S-LoRA    |    1 |   0.029 M |  62.45 | 0.33 |     2 |
| S-LoRA    |    2 |   0.057 M |  65.22 | 0.22 |     2 |
| S-LoRA    |    4 |   0.115 M | **65.53** | 0.14 |  3 |
| S-LoRA    |    8 |   0.229 M |  65.45 | 0.38 |     3 |
| S-LoRA    |   16 |   0.459 M |  65.70 | 0.38 |     2 |
| LoRA      |    1 |   0.100 M |  62.78 | 0.97 |     2 |
| LoRA      |    2 |   0.201 M |  64.24 | 0.50 |     2 |
| LoRA      |    4 |   0.401 M |  64.98 | 0.70 |     3 |
| LoRA      |    8 |   0.803 M |  65.61 | 0.07 |     3 |
| LoRA      |   16 |   1.606 M |  65.75 | 0.06 |     2 |
| VeRA      |  768 |   0.057 M |  65.03 | 1.18 |     2 |
| target-ft |    — |  22.034 M |  64.26 |    — |     1 |
| zero-shot |    — |         0 |  33.12 |    — |     1 |

**At matched parameter budget:**

| Budget | S-LoRA | LoRA | Δ | σ |
|--------|-------:|-----:|--:|--:|
| ~0.1 M | 65.53 (r=4)  | 62.78 (r=1) | **+2.75** | 4.0 |
| ~0.2 M | 65.45 (r=8)  | 64.24 (r=2) | **+1.21** | 2.7 |
| ~0.4 M | 65.70 (r=16) | 64.98 (r=4) | +0.72 | 1.3 |

Two things worth reading off this table beyond the headline:

- S-LoRA saturates at **0.057 M** parameters (65.22 BLEU). LoRA does not reach that until
  **0.803 M**, 14× larger. It also beats `target-ft` — unconstrained training of those same
  two matrices — which scores 64.26 using 22.0 M parameters, 384× more.
- **The advantage is bounded.** It shrinks as the budget grows (1.3σ by 0.4 M, i.e. gone),
  and at rank 1 S-LoRA *loses* to LoRA. This is a low-budget instrument, not a universal
  replacement.

### Does the advantage survive scale?

Same task, same two projections, same recipe — no TF32, 5 epochs, best-epoch selection,
beam 10, 630 test sentences — across the Qwen2.5 family. One seed per cell. The sweep was
run twice end to end, once in float32 and once in bfloat16, because the two do not agree.

Three runs per model give both comparisons. Since S-LoRA costs `2kr` where LoRA costs
`r(n+m)`, the saving is `(1+a)/2`, so S-LoRA at roughly 4× the rank occupies the same
budget as LoRA r=2:

**float32**

| Model | `a` | S-LoRA r=2 | S-LoRA (budget-matched) | LoRA r=2 | same rank | same budget |
|-------|----:|-----------:|------------------------:|---------:|----------:|------------:|
| 0.5B | 7 | 63.07 (0.025 M) | **64.19** (r=8, 0.098 M) | 62.31 (0.098 M) | +0.76 | **+1.87** |
| 1.5B | 6 | 65.22 (0.057 M) | 65.45 (r=8, 0.229 M) | 64.24 (0.201 M) | +0.98 | **+1.21** |
| 3B | 8 | 64.37 (0.074 M) | 65.33 (r=9, 0.332 M) | **65.38** (0.332 M) | −1.01 | **−0.05** |
| 7B | 7 | — | — | — | — | — |

**bfloat16**

| Model | `a` | S-LoRA r=2 | S-LoRA (budget-matched) | LoRA r=2 | same rank | same budget |
|-------|----:|-----------:|------------------------:|---------:|----------:|------------:|
| 0.5B | 7 | 61.25 (0.025 M) | **63.03** (r=8, 0.098 M) | 60.22 (0.098 M) | +1.03 | **+2.82** |
| 1.5B | 6 | 63.69 (0.057 M) | **65.07** (r=7, 0.201 M) | 62.70 (0.201 M) | +0.99 | **+2.37** |
| 3B | 8 | 63.95 (0.074 M) | **65.08** (r=9, 0.332 M) | 61.16 (0.332 M) | +2.79 | **+3.93** |
| 7B | 7 | **65.07** (0.115 M) | 64.68 (r=8, 0.459 M) | 64.77 (0.459 M) | +0.30 | **−0.09** |

![S-LoRA against LoRA across Qwen2.5 sizes on E2E NLG in bfloat16: absolute BLEU for the
three configurations, and the matched-budget gap against model size for both precisions](paper/fig_qwen_scale.png)

Left: absolute BLEU in bfloat16. Right: the matched-budget gap against model size, with
the float32 sweep dashed for comparison — it has no 7B point. Regenerate with
`python paper/fig_qwen_scale.py`, which reads the result JSON directly.

**The answer is no, in both.** At matched budget the advantage decays and is gone by the
largest model tested in each sweep: **+1.87 → +1.21 → −0.05** in float32, and
**+2.82 → +2.37 → +3.93 → −0.09** in bfloat16. The point where it vanishes moves with
precision — 3B in float32, 7B in bfloat16 — but the direction does not.

The budget match is exact in bfloat16 at every size (both arms at 0.098 M, 0.201 M,
0.332 M, 0.459 M). In float32 at 1.5B it is not: that sweep used r=8, giving S-LoRA 14%
more budget than LoRA r=2, so its +1.21 flatters S-LoRA. The bfloat16 sweep uses r=7 there,
which matches exactly.

Two points on the 3B bfloat16 row. Its LoRA cell is 61.16 at seed 0 and 62.23 at seed 1,
the widest spread anywhere in either table. And at 7B, S-LoRA r=2 reaches 65.07 on 0.115 M
— above both the budget-matched S-LoRA and LoRA itself, on a quarter of their parameters.

This lines up with what happens further out. On Mistral-7B with MetaMathQA, all seven
projections adapted at a matched 167 M budget, S-LoRA scores 61.94 on GSM8K against LoRA's
72.02 — a 10-point deficit, on a pipeline whose LoRA arm *exceeds* both published LoRA
baselines (69.50 in PiSSA, 67.70 in PMSS), which rules out an implementation fault. Four
points, one direction: the row-space constraint helps at small scale, stops helping around
3B, and hurts at 7B.

**Precision is not neutral between the two methods.** Holding everything else fixed on
Qwen2.5-0.5B, moving from float32 to bfloat16 costs S-LoRA 1.15 BLEU and LoRA 2.10 — so
the gap between them widens from +1.87 to +2.82 purely from the dtype. That asymmetry is
why the two sweeps are reported separately and why no row should be read across them.

It also makes the precision choice a reporting problem for the field. PiSSA states it
(Section 5: "Float32 computation type for both the base model and the adapter in LoRA and
PiSSA"), and its Appendix D shows bf16-vs-fp32 swinging GSM8K by up to 7.2 points on full
fine-tuning. LoRA (Hu et al.) and VeRA (Kopiczko et al.) do not state it anywhere — both
papers were searched in full.

What survives in both sweeps is parameter efficiency. At 3B in float32, S-LoRA r=2 reaches
64.37 on **0.074 M** against LoRA's 65.38 on **0.332 M** — within a point on 4.5× fewer
parameters. Validation loss there still favours S-LoRA (1.1184 against 1.1371) while BLEU
does not, which is why "tied" is the safer reading than "loses".

### VeRA as a baseline

VeRA needs a learning rate ~500× higher than LoRA (its trainable objects are two vectors,
so gradients are far smaller). Sweeping at r=768, 1 epoch:

| lr | val loss | BLEU |
|----|---------:|-----:|
| 3e-3 | 1.2906 | 62.10 |
| 1e-2 | 1.2438 | 63.28 |
| 3e-2 | 1.2106 | 64.25 |
| **1e-1** | **1.2006** | **65.29** |
| 3e-1 | 1.2168 | 64.85 |

At its tuned lr VeRA ties S-LoRA at 0.057 M (65.03 vs 65.22, within noise) — but costs
**343 ms/step against S-LoRA's 163 ms**, because it still multiplies by a full-width frozen
matrix. Fewer trainable parameters does not mean less compute.

Note that LoRA and S-LoRA have never had their learning rate swept; both use 2e-4 borrowed
from the LoRA paper. The current comparison therefore favours VeRA.

## Install

```bash
pip install torch transformers==5.15.1 datasets scipy 'numpy<2' 'Pillow>=10'
```

`numpy<2` is not optional — numpy 2.x breaks the torch ABI on the tested setup.
Reference environment: A10, driver 580.105.08, torch 2.7.0 / CUDA 12.8, scipy 1.8.0.

## Run

```bash
# S-LoRA: LoRA applied to the square factor
python finetune_e2e.py --method rowspace --rank 4 \
  --model Qwen/Qwen2.5-1.5B --target-set qwen2_kv --epochs 5 --best-epoch --seed 0

# LoRA baseline at a matched parameter budget
python finetune_e2e.py --method lora --rank 4 --match-params \
  --model Qwen/Qwen2.5-1.5B --target-set qwen2_kv --epochs 5 --best-epoch --seed 0

# VeRA baseline (note the learning rate)
python finetune_e2e.py --method vera --rank 768 --lr 1e-1 \
  --model Qwen/Qwen2.5-1.5B --target-set qwen2_kv --epochs 5 --best-epoch --seed 0

python summarize.py runs/summary.jsonl
```

Methods: `rowspace` (S-LoRA), `rowspace-full` (train the square factor directly, no LoRA),
`lora`, `vera`, `target-ft` (unconstrained upper bound), `full`, `none` (zero-shot floor).

Compare at **matched parameter budget**, not matched rank — at equal rank LoRA receives
about 3.5× more parameters on these matrices, which makes the comparison meaningless.

## Files

| File | Purpose |
|------|---------|
| `rowspace_peft.py` | The factorization and `RowSpaceLinear`. Structured storage (`X` + indices), `merge_back`. |
| `peft_generic.py` | Architecture-agnostic layer. `nn.Linear.weight.T` *is* the Conv1D layout, so GPT-2 and Qwen/Llama share one code path. LoRA and VeRA live here too. |
| `finetune_e2e.py` | E2E NLG pipeline: length-grouped batching, per-epoch validation, early stopping, timing and latency instrumentation. |
| `finetune_commonsense.py` | Llama + Commonsense-170K pipeline. |
| `split_solve.py` | Pivoted QR and triangular solve, with a numpy fallback. |
| `summarize.py` | Aggregates `summary.jsonl` across seeds. |
| `paper/` | ICLR draft. |

## Implementation notes

**The Q-cancellation trick.** `X = W₂⁻¹W₁ = (QR₁₁)⁻¹(QR₁₂) = R₁₁⁻¹R₁₂` — one triangular
solve, and `Q` is never formed. Residual on Qwen weights is 6.6e-15.

**Parameter preservation is real, not approximate.** An early version stored the full
`P = [I | X]` and inflated gpt2-medium from 354.82 M to 375 M. Storing only `X` with
`index_select` in the forward pass brings it back to 354.82 M exactly.

**Do not borrow LoRA's `α`.** `α=32` from the LoRA paper makes results *worse* here: `P`
already amplifies the update magnitude through the `PᵀP` preconditioner, so `α` and `P`
multiply. Use `α = r`.

## Status

Verified: exact factorization (6.6e-15), exact parameter preservation, zero post-merge
inference overhead, bit-identical reproducibility across runs at a fixed seed.

Not yet done, in order of importance:

1. **Measure where the update actually lives.** Run an unconstrained fine-tune and compute
   `ρ = ‖ΔW·VVᵀ‖²_F / ‖ΔW‖²_F` — the fraction of it already inside `row(W₀)`, against a
   chance level of `1/a`. This is the one number that would explain both the small-model
   win and the large-model loss, and it has not been measured at any scale.
2. **Seeds at the cells the conclusion rests on.** Every cell in the scaling table is one
   seed, and only +1.87 clears the noise individually. The monotone trend is the claim;
   confirming it properly needs 2–3 seeds at 0.5B and 3B.
3. **PMSS as a baseline.** It selects rows/columns of `W₀` by the same pivoted-QR criterion
   and states the same subspace constraint. Without it the novelty claim is unsupported.
4. `--train-x` ablation: freeze `W₂`, train `X`, to show whatever gain exists comes from
   the constraint rather than from merely having a factorization.
5. A non-Qwen model on E2E, to separate the method from this family.
6. Official `e2e-metrics` scoring — every BLEU here comes from an in-repo implementation.
   The runs write `*_hyps.txt` / `*_refs.txt` for exactly this.
7. Tall-orientation (`gate_proj`/`up_proj`) validation on a modern model.

Also unresolved: S-LoRA is **20% slower per optimizer step** than LoRA at matched budget
(0.090 vs 0.108 it/s on Mistral-7B). The factored forward does the same number of
multiplies as the dense one; the cost is the column gather, which is memory-bound and
runs at 30–39% of what the card will give. Folding it into the GEMM was tried and fails —
it means replacing cuBLAS with a hand-written kernel, which loses by 32×. `torch.compile`
recovers about a third of the gap. See `slora_kernels.py`.

The name collides with *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*
(Sheng et al., MLSys 2024), which is unrelated work.
