"""Re-run each canonical workload under inductor debug mode and capture artifacts.

For every workload:
  - reset dynamo state
  - reload the model fresh (cleanest isolation between compilations)
  - torch.compile(model, mode='default', dynamic=False) with cudagraphs OFF
  - run one forward (for decode workloads, build the KV cache via the compiled
    model first — the cache-build prefill is itself a graph we want to capture)
  - inductor writes output_code.py, fx_graph_readable.py, output_code-derived
    kernel .py files and torch_compile_debug/ logs into TORCH_INDUCTOR_DEBUG_DIR

After collection, build `extract/manifest.json` mapping (workload, graph_id) ->
captured artifacts, plus a flat index of every fused-Triton kernel emitted with
its workload(s) of origin.

Environment variables MUST be set before `torch` import:
  TORCH_COMPILE_DEBUG=1
  TORCH_LOGS="+inductor,output_code,fusion,schedule"
  TORCHINDUCTOR_CACHE_DIR=<.../extract/inductor_debug/cache>
  TORCH_COMPILE_DEBUG_DIR=<.../extract/inductor_debug/debug>
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = Path(__file__).resolve().parent
DEBUG_ROOT = EXTRACT_DIR / "inductor_debug"
CACHE_DIR = DEBUG_ROOT / "cache"
DEBUG_DIR = DEBUG_ROOT / "debug"
PER_WORKLOAD_DIR = DEBUG_ROOT / "by_workload"

# Wipe + recreate to keep this script idempotent.
for d in (CACHE_DIR, DEBUG_DIR, PER_WORKLOAD_DIR):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# These MUST be set before importing torch.
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_LOGS"] = "+inductor,output_code,fusion,schedule"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR)
os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(DEBUG_DIR)
# Be deterministic about thread count for repeatability of any autotuning.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
import torch._inductor.config as inductor_config  # noqa: E402

# cudagraphs OFF is the primary baseline.
inductor_config.triton.cudagraphs = False
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.cache_size_limit = 64

sys.path.insert(0, str(ROOT))
from workload.inputs import list_workloads, get_workload  # noqa: E402
from workload.model import load_model  # noqa: E402


def _snapshot_debug_dir() -> set:
    """Return set of file paths currently in DEBUG_DIR (relative)."""
    out = set()
    if not DEBUG_DIR.exists():
        return out
    for p in DEBUG_DIR.rglob("*"):
        if p.is_file():
            out.add(p.relative_to(DEBUG_DIR))
    return out


def _input_meta(t):
    if isinstance(t, torch.Tensor):
        return {"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)}
    return {"type": type(t).__name__}


def run_workload(name: str, manifest: dict):
    print(f"\n{'=' * 60}\n[dump] {name}\n{'=' * 60}")
    torch._dynamo.reset()

    before = _snapshot_debug_dir()

    # Load model fresh.
    t0 = time.time()
    model, _ = load_model(dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(model, mode="default", dynamic=False)
    wl = get_workload(name)
    device = "cuda"

    forward_meta = {
        "name": name,
        "mode": wl["mode"],
        "seq_len": wl["seq_len"],
        "batch_size": wl["batch_size"],
        "calls": [],
    }

    try:
        if wl["mode"] == "prefill":
            input_ids = wl["input_ids"].to(device)
            attention_mask = wl["attention_mask"].to(device)
            forward_meta["calls"].append({
                "kind": "prefill",
                "inputs": {
                    "input_ids": _input_meta(input_ids),
                    "attention_mask": _input_meta(attention_mask),
                },
            })
            with torch.no_grad():
                out = compiled(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            torch.cuda.synchronize()
            del out
        elif wl["mode"] == "decode":
            # The cache builder runs a prefill on the *compiled* model. That
            # prefill graph is itself an inductor compilation we want captured.
            forward_meta["calls"].append({
                "kind": "cache_build_prefill",
                "inputs": {"prompt_seq_len": wl["seq_len"], "batch_size": wl["batch_size"]},
            })
            kv_state = wl["kv_cache_builder"](compiled)
            past = kv_state["past_key_values"]
            last_token_ids = kv_state["last_token_ids"]
            attn = kv_state["attention_mask"]
            forward_meta["calls"].append({
                "kind": "decode_step",
                "inputs": {
                    "last_token_ids": _input_meta(last_token_ids),
                    "attention_mask": _input_meta(attn),
                },
            })
            with torch.no_grad():
                out = compiled(
                    input_ids=last_token_ids,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            del out, past
        else:
            raise ValueError(f"unknown workload mode {wl['mode']}")
    except Exception as e:
        forward_meta["error"] = str(e)
        forward_meta["traceback"] = traceback.format_exc()
        print(f"[dump] {name} FAILED: {e}")
        traceback.print_exc()
    finally:
        elapsed = time.time() - t0
        forward_meta["wall_seconds"] = elapsed
        print(f"[dump] {name} done in {elapsed:.1f}s")

    after = _snapshot_debug_dir()
    new_files = sorted(after - before)

    # Copy the new debug files into a per-workload directory so each workload
    # has a self-contained snapshot. Inductor writes into per-graph dirs like
    # torch_compile_debug/run_2024-.../torchinductor/model__N_inference_K/
    # and within those an output_code.py, fx_graph_readable.py, kernel files.
    wl_out = PER_WORKLOAD_DIR / name
    wl_out.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in new_files:
        src = DEBUG_DIR / rel
        dst = wl_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied.append(str(rel))
        except Exception as e:
            print(f"[dump]   warn: copy {src} -> {dst} failed: {e}")

    forward_meta["debug_files"] = copied
    forward_meta["per_workload_dir"] = str(wl_out.relative_to(EXTRACT_DIR))

    # Identify the "graph" subdirs (one per compiled subgraph).
    graphs = []
    for sub in sorted(wl_out.rglob("output_code.py")):
        graph_dir = sub.parent
        graph_id = graph_dir.name
        kernels = sorted(p.name for p in graph_dir.glob("*.py")
                         if p.name not in ("output_code.py", "fx_graph_readable.py",
                                           "fx_graph_transformed.py", "fx_graph_runnable.py",
                                           "ir_pre_fusion.txt", "ir_post_fusion.txt"))
        graphs.append({
            "graph_id": graph_id,
            "graph_dir_rel": str(graph_dir.relative_to(EXTRACT_DIR)),
            "output_code": str((graph_dir / "output_code.py").relative_to(EXTRACT_DIR)),
            "fx_graph_readable": (
                str((graph_dir / "fx_graph_readable.py").relative_to(EXTRACT_DIR))
                if (graph_dir / "fx_graph_readable.py").exists() else None
            ),
            "kernel_files": kernels,
        })
    forward_meta["graphs"] = graphs

    manifest[name] = forward_meta

    # Release memory before next workload.
    del compiled
    del model
    torch.cuda.empty_cache()


def main():
    manifest = {
        "torch_version": torch.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "inductor_debug_dir": str(DEBUG_DIR.relative_to(EXTRACT_DIR)),
        "cache_dir": str(CACHE_DIR.relative_to(EXTRACT_DIR)),
        "per_workload_dir": str(PER_WORKLOAD_DIR.relative_to(EXTRACT_DIR)),
        "workloads": {},
    }
    workloads = list_workloads()
    for name in workloads:
        try:
            run_workload(name, manifest["workloads"])
        except Exception as e:
            print(f"[dump] workload {name} crashed: {e}")
            traceback.print_exc()
            manifest["workloads"][name] = {"error": str(e), "traceback": traceback.format_exc()}

    out = EXTRACT_DIR / "manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[dump] wrote {out}")


if __name__ == "__main__":
    main()
