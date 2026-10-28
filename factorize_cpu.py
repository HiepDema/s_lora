#!/usr/bin/env python
"""Factorize LLaMA3-8B weights on CPU — no torch needed.

Downloads safetensors from HuggingFace, runs pivoted QR on each non-square
weight matrix, and saves pivot indices to a cache directory.  The cache files
are compatible with slora.factorize.factorize_cached() — training on GPU
loads them instantly instead of re-running QR.

Memory: processes ONE matrix at a time.  Peak ~1.5 GB for MLP (4096x14336
in float64).  With 4 GB RAM + 11 GB pagefile this works via swap, albeit
slower (~2-3 hours total).

Usage:
  python factorize_cpu.py
  python factorize_cpu.py --model meta-llama/Llama-3-8B --cache fac_cache_llama3
  python factorize_cpu.py --dtype float32   # half the RAM, slight precision loss
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import time


def pivoted_qr(A):
    """Column-pivoted QR: A[:, piv] = Q R.  Returns (R, piv)."""
    from scipy.linalg import qr
    _, R, piv = qr(A, mode="economic", pivoting=True)
    import numpy as np
    return np.ascontiguousarray(R), np.ascontiguousarray(piv)


def factorize_one(W, dtype):
    """Factorize one non-square weight.  W is [out, in] (nn.Linear layout).

    Returns dict with sel, rest, maxX, condW2, residual.
    """
    import numpy as np

    # Transpose to [d_out, d_in] = [n, m] where y = W x
    # nn.Linear stores [out, in], and our convention is W = weight.T for Conv1D
    # but for Linear, weight IS [out, in], so W = weight means y = W x already
    n, m = W.shape

    # Determine orientation
    if n == m:
        return None  # square, skip

    if m > n:
        # Wide: W[:, piv] = Q [R11 | R12], W2 = W[:, piv[:n]], X = R11^-1 R12
        Wd = W.astype(dtype)
        R, piv = pivoted_qr(Wd)
        R11 = R[:, :n]
        R12 = R[:, n:]
        from scipy.linalg import solve_triangular
        X = solve_triangular(R11, R12, lower=False)
        sel = piv[:n]
        rest = piv[n:]
        orient = "pre"
    else:
        # Tall: transpose to wide, factorize, transpose back
        Wd = W.T.astype(dtype)
        nt, mt = Wd.shape  # nt=m (original in), mt=n (original out), mt > nt
        R, piv = pivoted_qr(Wd)
        R11 = R[:, :nt]
        R12 = R[:, nt:]
        from scipy.linalg import solve_triangular
        X = solve_triangular(R11, R12, lower=False)
        sel = piv[:nt]
        rest = piv[nt:]
        orient = "post"

    # Health metrics
    diag = np.abs(np.diag(R11))
    maxX = float(np.abs(X).max())
    condW2 = float(diag.max() / max(diag.min(), 1e-30))

    # Residual: reconstruct and compare
    if orient == "pre":
        W2 = Wd[:, sel]
        W1_recon = W2 @ X
        W1_orig = Wd[:, rest]
    else:
        W2 = Wd[:, sel]  # this is transposed
        W1_recon = W2 @ X
        W1_orig = Wd[:, rest]
    res = float(np.linalg.norm(W1_recon - W1_orig) / max(np.linalg.norm(W1_orig), 1e-30))

    return dict(sel=sel, rest=rest, orient=orient, maxX=maxX,
                condW2=condW2, res=res, shape=(n, m))


def save_pivot_torch_format(path, sel, rest):
    """Save pivot indices in torch.save format (compatible with factorize_cached).

    Uses a minimal reimplementation — no torch needed.
    """
    import numpy as np
    # We save as numpy .npz — the training code will need a small adapter
    # Actually, let's save as .pt using pickle (torch tensors are just numpy + metadata)
    # Simplest: save as .npz and add a loader
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    np.savez(tmp, sel=sel.astype(np.int64), rest=rest.astype(np.int64))
    os.replace(tmp, path)


def load_weight_from_safetensors(model_dir, tensor_name):
    """Load a single tensor from safetensors files without loading the full model."""
    import numpy as np
    from safetensors import safe_open

    # Find which file contains this tensor
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        filename = index["weight_map"].get(tensor_name)
        if filename is None:
            return None
        filepath = os.path.join(model_dir, filename)
    else:
        # Single file model
        filepath = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(filepath):
            return None

    with safe_open(filepath, framework="numpy") as f:
        if tensor_name in f.keys():
            return f.get_tensor(tensor_name)
    return None


def download_model(model_id):
    """Download model safetensors to local cache, return path."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        model_id,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        ignore_patterns=["*.bin", "*.msgpack"],
    )
    return path


def get_target_weights(model_id, target_set):
    """Return list of (layer_idx, target_name, tensor_name, expected_shape)."""
    # LLaMA/Mistral/Qwen naming: model.layers.{i}.{target}.weight
    targets_map = {
        "all7": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "llama_kv": ["self_attn.k_proj", "self_attn.v_proj"],
        "llama_mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "qwen2_kv": ["self_attn.k_proj", "self_attn.v_proj"],
    }
    targets = targets_map.get(target_set)
    if targets is None:
        sys.exit(f"Unknown target set: {target_set}")
    return targets


def main():
    p = argparse.ArgumentParser(
        description="Factorize model weights on CPU (no torch needed)")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--target-set", default="all7",
                   choices=["all7", "llama_kv", "llama_mlp", "qwen2_kv"])
    p.add_argument("--cache", default="fac_cache_llama3",
                   help="directory to save pivot indices")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"],
                   help="compute precision (float32 uses half the RAM)")
    p.add_argument("--n-layers", type=int, default=None,
                   help="override number of layers (auto-detected from config)")
    p.add_argument("--basis", default="colperm")
    a = p.parse_args()

    import numpy as np
    dtype = np.float64 if a.dtype == "float64" else np.float32

    print(f"=== CPU Factorization ===")
    print(f"Model     : {a.model}")
    print(f"Target set: {a.target_set}")
    print(f"Precision : {a.dtype}")
    print(f"Cache dir : {a.cache}")
    print()

    # Download model
    print("Downloading model (safetensors only)...")
    t0 = time.perf_counter()
    model_dir = download_model(a.model)
    print(f"  Model at: {model_dir}  ({time.perf_counter()-t0:.0f}s)")

    # Read config for n_layers
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    n_layers = a.n_layers or config.get("num_hidden_layers", 32)
    print(f"  Layers: {n_layers}")

    targets = get_target_weights(a.model, a.target_set)
    print(f"  Targets: {targets}")

    # Count total factorizations needed (skip square)
    total = 0
    skipped_square = 0
    to_do = []
    print("\n  Scanning weight shapes...")
    for i in range(n_layers):
        for tname in targets:
            tensor_name = f"model.layers.{i}.{tname}.weight"
            cache_path = os.path.join(a.cache, f"{tname}.{i}.{a.basis}.npz")
            if os.path.exists(cache_path):
                total += 1  # already cached
                continue

            # Peek at shape without loading full tensor
            W = load_weight_from_safetensors(model_dir, tensor_name)
            if W is None:
                print(f"    WARNING: {tensor_name} not found, skipping")
                continue
            shape = W.shape
            del W
            gc.collect()

            if shape[0] == shape[1]:
                skipped_square += 1
                continue
            to_do.append((i, tname, tensor_name, cache_path, shape))
            total += 1

    cached = total - len(to_do)
    print(f"  Total non-square: {total}")
    print(f"  Already cached:   {cached}")
    print(f"  To factorize:     {len(to_do)}")
    print(f"  Skipped (square): {skipped_square}")

    if not to_do:
        print("\n  Nothing to do — all pivots already cached!")
        return

    # Estimate time
    n_kv = sum(1 for _, t, _, _, s in to_do if "proj" in t and min(s) <= 1024)
    n_mlp = len(to_do) - n_kv
    est_min = n_kv * 2 / 60 + n_mlp * 60 / 60  # rough: 2s for kv, 60s for MLP
    print(f"\n  Estimated time: ~{est_min:.0f} min "
          f"({n_kv} small + {n_mlp} large matrices)")
    if n_mlp > 0:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        if avail_gb < 1.0:
            print(f"  WARNING: only {avail_gb:.1f} GB RAM free. "
                  f"Large matrices will use pagefile (slower).")
            if a.dtype == "float64":
                print(f"  TIP: use --dtype float32 to halve peak RAM")

    # Factorize
    print(f"\n{'='*70}")
    t_start = time.perf_counter()
    for idx, (i, tname, tensor_name, cache_path, shape) in enumerate(to_do):
        t1 = time.perf_counter()
        k = min(shape)
        a_ratio = max(shape) / k
        print(f"  [{idx+1}/{len(to_do)}] layer {i:2d} {tname:<20s} "
              f"{shape[0]}x{shape[1]} (a={a_ratio:.1f}) ...", end="", flush=True)

        # Load weight
        W = load_weight_from_safetensors(model_dir, tensor_name)
        if W is None:
            print(" SKIP (not found)")
            continue

        # Factorize
        result = factorize_one(W, dtype)
        del W
        gc.collect()

        if result is None:
            print(" SKIP (square)")
            continue

        # Save pivots
        save_pivot_torch_format(cache_path, result["sel"], result["rest"])

        elapsed = time.perf_counter() - t1
        print(f" {elapsed:5.1f}s  res={result['res']:.1e}  "
              f"max|X|={result['maxX']:.2f}  cond={result['condW2']:.1e}")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f"Done: {len(to_do)} matrices in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Cache saved to: {a.cache}/")
    print(f"\nNext: copy {a.cache}/ to GPU machine and run:")
    print(f"  python check_llama3.py --fac-cache {a.cache} --bf16")
    print(f"  bash run_cs_llama3.sh")


if __name__ == "__main__":
    main()
