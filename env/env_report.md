# Environment Report — KernelBench (DGX Spark)

**Date:** 2026-05-14

## Hardware
- GPU: `NVIDIA GB10` (Blackwell), reported as **compute capability `sm_121`** (`(12, 1)`).
- Driver: **580.95.05**, advertised CUDA: **13.0**.
- Host: ARM (`aarch64`), Python `3.12.12` (conda-provided; see venv note below).
- Memory: nvidia-smi reports `Memory-Usage: Not Supported` on GB10 because LPDDR5X is unified with the host. Treat the ~128 GB system RAM as the budget — there is no separate VRAM pool to query via nvidia-smi.

## Versions installed
| Component       | Version                            |
|-----------------|------------------------------------|
| torch           | `2.12.0.dev20260408+cu128`         |
| triton          | `3.7.0+git282c8251` (bundled w/ torch nightly) |
| transformers    | `5.8.1`                            |
| accelerate      | `1.13.0`                           |
| safetensors     | `0.7.0`                            |
| huggingface_hub | `1.14.0`                           |
| numpy           | `2.4.4`                            |
| psutil          | `7.2.2`                            |
| nvitop          | `1.6.2`                            |

PyTorch is the CUDA 12.8 nightly (`cp312-cp312-manylinux_2_28_aarch64`). Torch reports `torch.version.cuda = 12.8`; the driver is newer (13.0) which is forward-compatible. `torch.cuda.get_device_capability()` returns `(12, 1)` — i.e. **sm_121**, not sm_120. PyTorch/Triton both accept this device without warnings.

## Venv setup
Located at `/home/ianbarber/Projects/KernelBench/.venv`.

Activation:
```bash
source /home/ianbarber/Projects/KernelBench/.venv/bin/activate
# or call interpreters by absolute path:
/home/ianbarber/Projects/KernelBench/.venv/bin/python
```

Reinstall from scratch:
```bash
/home/ianbarber/miniconda3/envs/qwen/bin/python -m venv /home/ianbarber/Projects/KernelBench/.venv
/home/ianbarber/Projects/KernelBench/.venv/bin/pip install --upgrade pip wheel
/home/ianbarber/Projects/KernelBench/.venv/bin/pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cu128
/home/ianbarber/Projects/KernelBench/.venv/bin/pip install \
    -r /home/ianbarber/Projects/KernelBench/env/requirements.txt
```

### Important: the venv must be built from a Python that ships dev headers

Triton on Blackwell compiles a small `cuda_utils.c` shim at runtime that includes `<Python.h>`. The system `python3` from apt (`/usr/bin/python3`, 3.12.3) is installed **without** `python3-dev`, so its include dir `/usr/include/python3.12` only contains third-party headers (PIL, etc.) — no `Python.h`.

First attempt used `/usr/bin/python3 -m venv .venv` and Triton failed at JIT time with:
```
fatal error: Python.h: No such file or directory
```
`sudo apt install python3-dev` requires a password we don't have, so the venv was instead re-created from `/home/ianbarber/miniconda3/envs/qwen/bin/python` (3.12.12), whose `sysconfig.get_path('include')` resolves to `/home/ianbarber/miniconda3/envs/qwen/include/python3.12` and **does** ship `Python.h`. With that, both smoke tests pass.

Takeaway: do not rebuild this venv from the system python without first getting `python3-dev` installed.

## Verification
`env/verify_env.py` runs three checks:
1. Print versions and device capability.
2. `torch.compile` smoke test: `lambda x: (x*x + x).sum()` on a 1024-element CUDA float32 tensor, mode `reduce-overhead`. Compares eager vs compiled.
3. Triton smoke test: a hand-written vector-add kernel on 4096 floats.

Latest run: **PASS / PASS / overall PASS** (see `verify_env.py` output). No warnings emitted, no `sm_121 not supported` messages, no PTX fallback chatter.

## Surprises / things to know
- **`sm_121` is real.** Some docs assume Blackwell consumer = sm_120; on GB10 we observe sm_121. PyTorch nightly cu128 handles it without complaint; Triton 3.7 codegens fine.
- **Driver/runtime mismatch is fine.** Driver advertises CUDA 13.0, torch is built against 12.8. No issues.
- **nvidia-smi memory is `Not Supported`** on GB10 — expected for unified memory. Use `torch.cuda.memory_allocated()` / `psutil` for memory monitoring; nvitop will still work but won't show VRAM the usual way.
- **HF auth is NOT configured** (`hf auth whoami` → `Not logged in`). Qwen3-1.7B is a public model so this is fine for that download; flag if we move to a gated model.
- **Disk:** 328 GB free on `/` (which holds both the working dir and `~/.cache/huggingface`). The venv itself is **6.7 GB** (torch + CUDA libs dominate). Plenty of headroom for the ~5 GB Qwen3-1.7B weights.

## File map
- `env/requirements.txt` — pip requirements (torch installed separately from the nightly index, see above).
- `env/verify_env.py` — idempotent verification script; exits 0 on PASS, 1 on FAIL, 2 if torch import fails.
- `env/env_report.md` — this file.
