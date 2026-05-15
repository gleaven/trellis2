"""
Patch TRELLIS.2 for NVIDIA GB10 (sm_121) compatibility.

1. Adds 'sdpa' as a valid sparse attention backend (flash-attn/xformers don't work on GB10)
2. Fixes RMBG-2.0 BiRefNet loading with meta tensors (PyTorch 2.9.1 + transformers 5.x)
"""
import re
import sys

TRELLIS_DIR = sys.argv[1] if len(sys.argv) > 1 else "/app/TRELLIS.2"

# ── Patch 1: sparse/config.py — add 'sdpa' as valid backend ──
config_path = f"{TRELLIS_DIR}/trellis2/modules/sparse/config.py"
with open(config_path, 'r') as f:
    content = f.read()

# Add 'sdpa' to the valid backends list in assert
content = content.replace(
    "assert ATTN in ['xformers', 'flash_attn', 'flash_attn_3']",
    "assert ATTN in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']"
)

# Add 'sdpa' to the __from_env() validation list
content = content.replace(
    "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']",
    "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']"
)

# Add 'sdpa' to the set_attn_backend type hint
content = content.replace(
    "def set_attn_backend(backend: Literal['xformers', 'flash_attn'])",
    "def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'sdpa'])"
)

with open(config_path, 'w') as f:
    f.write(content)
print(f"[PATCH] {config_path}: added 'sdpa' to valid backends + __from_env() + set_attn_backend")

# ── Patch 2: sparse/attention/full_attn.py — add SDPA dispatch branch ──
attn_path = f"{TRELLIS_DIR}/trellis2/modules/sparse/attention/full_attn.py"
with open(attn_path, 'r') as f:
    content = f.read()

# Add import for F at the top
if "import torch.nn.functional as F" not in content:
    content = content.replace(
        "import torch\n",
        "import torch\nimport torch.nn.functional as F\n"
    )

# Add SDPA branch before the else clause
sdpa_branch = """    elif config.ATTN == 'sdpa':
        # PyTorch native SDPA — works on all GPUs including GB10 sm_121
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        # Process each batch element separately (variable-length sequences)
        cu_q = [0] + list(torch.tensor(q_seqlen).cumsum(0).tolist())
        cu_kv = [0] + list(torch.tensor(kv_seqlen).cumsum(0).tolist())
        outs = []
        for i in range(len(q_seqlen)):
            qi = q[cu_q[i]:cu_q[i+1]].unsqueeze(0).transpose(1, 2)   # [1, H, Lq, C]
            ki = k[cu_kv[i]:cu_kv[i+1]].unsqueeze(0).transpose(1, 2) # [1, H, Lk, C]
            vi = v[cu_kv[i]:cu_kv[i+1]].unsqueeze(0).transpose(1, 2) # [1, H, Lk, C]
            oi = F.scaled_dot_product_attention(qi, ki, vi)            # [1, H, Lq, C]
            outs.append(oi.transpose(1, 2).squeeze(0))                 # [Lq, H, C]
        out = torch.cat(outs, dim=0)
"""

content = content.replace(
    "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")",
    sdpa_branch + "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")"
)

with open(attn_path, 'w') as f:
    f.write(content)
print(f"[PATCH] {attn_path}: added SDPA dispatch branch")

print("[PATCH] Done — TRELLIS.2 sparse attention now supports ATTN_BACKEND=sdpa")

# ── Patch 3: rembg/BiRefNet.py — fix meta tensor error with transformers 5.x ──
# AutoModelForImageSegmentation defaults to low_cpu_mem_usage=True in new transformers,
# which uses meta tensors. BiRefNet's custom code calls .item() on torch.linspace
# during __init__, which fails on meta tensors.
birefnet_path = f"{TRELLIS_DIR}/trellis2/pipelines/rembg/BiRefNet.py"
with open(birefnet_path, 'r') as f:
    content = f.read()

content = content.replace(
    'model_name, trust_remote_code=True',
    'model_name, trust_remote_code=True, low_cpu_mem_usage=False'
)

with open(birefnet_path, 'w') as f:
    f.write(content)
print(f"[PATCH] {birefnet_path}: added low_cpu_mem_usage=False to avoid meta tensor error")
