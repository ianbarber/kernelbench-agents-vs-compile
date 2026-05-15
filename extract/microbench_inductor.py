"""Standalone microbench for inductor-emitted Triton kernels.

Loads each kernel's metadata to discover (a) the cached source file produced by
TorchInductor, (b) the kernel name, (c) its sample input shapes/dtypes, and
(d) the launch kwargs (XBLOCK, num_warps, num_stages). Then we strip inductor's
`@triton_heuristics.pointwise` wrapper so the kernel can be launched directly
via `do_bench`. The wrapper in our torch nightly rejects unknown kwargs like
`XBLOCK` when called via `CachingAutotuner.launcher(...)`, so going around it
is the simplest fix.

Output: `extract/microbench_inductor.json` mapping kernel-name -> median us.
For SwiGLU the dict also surfaces a top-level alias `swiglu_us`.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.testing


HERE = Path(__file__).parent.resolve()
KERNELS_DIR = HERE / "kernels"

# bf16/fp16/fp32 dtype map for inductor signature strings.
_TY_MAP = {
    "*bf16": torch.bfloat16,
    "*fp16": torch.float16,
    "*fp32": torch.float32,
}
_C10_MAP = {
    "c10::BFloat16": torch.bfloat16,
    "c10::Half": torch.float16,
    "float": torch.float32,
}


def _strip_inductor_wrapper(src: str) -> str:
    """Remove the `@triton_heuristics.pointwise(...)` (or `.reduction(...)`)
    decorator that wraps the inductor-emitted kernel. We keep the `@triton.jit`
    decorator that sits below it, plus the function body.
    """
    # Drop multiline `@triton_heuristics.<thing>(...)` (a balanced-paren chunk
    # that ends at the line before `@triton.jit`).
    pat = re.compile(
        r"@triton_heuristics\.[A-Za-z_]+\(.*?\)\s*\n(?=@triton\.jit)",
        re.DOTALL,
    )
    new, n = pat.subn("", src)
    if n == 0:
        # Already plain @triton.jit, nothing to do.
        return src
    return new


def _load_kernel_module(kernel_src_path: Path, kernel_name: str):
    """Load a kernel module from its inductor cache .py file, with the
    inductor-pointwise wrapper stripped. Returns the `@triton.jit` function.
    """
    src = kernel_src_path.read_text()
    src = _strip_inductor_wrapper(src)

    # Write the stripped source to a sibling .py we own, then import.
    stripped_path = HERE / "_stripped" / f"{kernel_src_path.stem}.py"
    stripped_path.parent.mkdir(exist_ok=True)
    stripped_path.write_text(src)

    spec = importlib.util.spec_from_file_location(
        f"_stripped_{kernel_src_path.stem}", str(stripped_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {stripped_path}")
    mod: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = getattr(mod, kernel_name, None)
    if fn is None:
        raise AttributeError(
            f"kernel {kernel_name!r} not found in {stripped_path}"
        )
    return fn


def _parse_launch_kwargs(s: str) -> dict:
    """Parse strings like 'XBLOCK=512' or 'XBLOCK=512,RBLOCK=8'."""
    out = {}
    if not s:
        return out
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, _, v = chunk.partition("=")
        out[k.strip()] = int(v.strip())
    return out


def _bench_reduction_rmsnorm(meta: dict) -> tuple[float, dict]:
    """Benchmark the residual-fused RMSNorm reduction kernel
    `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9`.

    Signature (from inductor): (in_ptr0, in_ptr1, in_ptr2, out_ptr1,
    xnumel, r0_numel, XBLOCK, R0_BLOCK).
      in_ptr0: x         (1, 512, 2048) bf16   -- one tensor view of shape (xnumel, r0_numel)
      in_ptr1: residual  (512, 2048)    bf16
      in_ptr2: weight    (2048,)        bf16
      out_ptr1: out      (1, 512, 2048) bf16

    Returns (median_us, sanity_meta) where sanity_meta contains
    correctness vs eager reference (cos_sim, rmse, l1_rel) so we can
    confirm the stripped kernel is computing the right thing.
    """
    si = meta["sample_inputs"]
    kernel_path = Path(si["kernel_file"])
    if not kernel_path.exists():
        raise FileNotFoundError(f"inductor cache file missing: {kernel_path}")

    kname = meta["name"]
    fn = _load_kernel_module(kernel_path, kname)

    launch_kw = _parse_launch_kwargs(si["kernel_kwargs"])
    XBLOCK = launch_kw.get("XBLOCK", 2)
    R0_BLOCK = launch_kw.get("R0_BLOCK", 1024)
    num_warps = si["num_warps"]
    num_stages = si["num_stages"]

    shape_x = tuple(si["Input Dims"][0])         # (1, 512, 2048)
    shape_residual = tuple(si["Input Dims"][1])  # (512, 2048)
    shape_weight = tuple(si["Input Dims"][2])    # (2048,)

    assert len(shape_x) == 3 and shape_x[0] == 1, shape_x
    xnumel = shape_x[0] * shape_x[1]   # 512
    r0_numel = shape_x[2]              # 2048

    g = torch.Generator(device="cuda")
    g.manual_seed(0xC0FFEE)
    x = torch.randn(shape_x, device="cuda", dtype=torch.bfloat16, generator=g)
    residual = torch.randn(shape_residual, device="cuda", dtype=torch.bfloat16, generator=g)
    weight = (torch.randn(shape_weight, device="cuda", dtype=torch.float32,
                          generator=g) * 0.1 + 1.0).to(torch.bfloat16)
    out = torch.empty(shape_x, device="cuda", dtype=torch.bfloat16)

    grid = (triton.cdiv(xnumel, XBLOCK),)

    # Sanity-launch once.
    fn[grid](x, residual, weight, out, xnumel, r0_numel,
             XBLOCK=XBLOCK, R0_BLOCK=R0_BLOCK,
             num_warps=num_warps, num_stages=num_stages)
    torch.cuda.synchronize()

    # Correctness check vs eager reference.
    sys.path.insert(0, str(HERE.parent))
    try:
        from workload.correctness import check_outputs
    except Exception:
        check_outputs = None  # type: ignore

    s = x.to(torch.float32) + residual.to(torch.float32)
    var = s.pow(2).mean(dim=-1, keepdim=True)
    eager = (s * torch.rsqrt(var + 1e-6) * weight.to(torch.float32)).to(torch.bfloat16)

    sanity = {}
    if check_outputs is not None:
        sanity = check_outputs(eager, out, dtype="bf16", task="strict")
        if not sanity["pass"]:
            print("[rmsnorm] WARNING: stripped kernel disagrees with eager:",
                  sanity, file=sys.stderr)

    def launch():
        fn[grid](x, residual, weight, out, xnumel, r0_numel,
                 XBLOCK=XBLOCK, R0_BLOCK=R0_BLOCK,
                 num_warps=num_warps, num_stages=num_stages)

    ms = triton.testing.do_bench(launch, warmup=25, rep=100, return_mode="median")
    return ms * 1000.0, sanity


def _bench_sdpa_prelude() -> tuple[float, dict]:
    """Benchmark the full SDPA-prelude as inductor emits it for prefill_512_b1.

    The "prelude" is everything between the post-residual-RMSNorm `hidden_states`
    and the `aten._scaled_dot_product_efficient_attention` call. Inductor's
    breakdown:
      * 3x extern_kernels.mm  -- Q/K/V projection GEMMs (cuBLAS).
      * _per_..._1            -- Q per-head RMSNorm + RoPE-apply.
      * _per_..._2            -- K per-head RMSNorm + RoPE-apply.
      * _poi_..._where_3 ×2   -- GQA expansion (K, V: 8 heads -> 16).
      * _poi_..._where_4      -- causal-mask construction (writes 2 buffers).

    We stitch these together in the order inductor calls them in
    `extract/inductor_debug/cache/b2/.../output_code.py`, and use
    `triton.testing.do_bench` over the full sequence. That is the
    codegen-vs-codegen baseline an agent must beat.

    Returns (median_us, breakdown_dict).
    """
    import torch as _t
    K_DIR = KERNELS_DIR

    # --- Locate each component kernel and its launch metadata. ---
    per_q_meta = json.loads((
        K_DIR
        / "triton_per_fused__scaled_dot_product_efficient_attention__to_copy"
          "__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand"
          "_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin"
          "_slice_transpose_unsqueeze_view_where_1"
        / "metadata.json"
    ).read_text())["stats"]["prefill_512_b1"]
    per_k_meta = json.loads((
        K_DIR
        / "triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos"
          "_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2"
        / "metadata.json"
    ).read_text())["stats"]["prefill_512_b1"]
    where_3_meta = json.loads((
        K_DIR
        / "triton_poi_fused__scaled_dot_product_efficient_attention__to_copy"
          "__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand_index"
          "_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_transpose_unsqueeze"
          "_view_where_3"
        / "metadata.json"
    ).read_text())["stats"]["prefill_512_b1"]
    where_4_meta = json.loads((
        K_DIR
        / "triton_poi_fused__scaled_dot_product_efficient_attention__to_copy"
          "__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand_index"
          "_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_transpose_unsqueeze"
          "_view_where_4"
        / "metadata.json"
    ).read_text())["stats"]["prefill_512_b1"]

    per_q_fn = _load_kernel_module(
        Path(per_q_meta["sample_inputs"]["kernel_file"]), per_q_meta["name"]
    )
    per_k_fn = _load_kernel_module(
        Path(per_k_meta["sample_inputs"]["kernel_file"]), per_k_meta["name"]
    )
    where_3_fn = _load_kernel_module(
        Path(where_3_meta["sample_inputs"]["kernel_file"]), where_3_meta["name"]
    )
    where_4_fn = _load_kernel_module(
        Path(where_4_meta["sample_inputs"]["kernel_file"]), where_4_meta["name"]
    )

    # --- Build inputs (canonical prefill_512_b1 shapes). ---
    B, S, H = 1, 512, 2048
    NQ, NKV, D = 16, 8, 128

    g = _t.Generator(device="cuda")
    g.manual_seed(0xC0FFEE)
    hidden_states = _t.randn((B, S, H), device="cuda", dtype=_t.bfloat16, generator=g)
    w_q = (_t.randn((NQ * D, H), device="cuda", dtype=_t.float32, generator=g) * 0.02).to(_t.bfloat16)
    w_k = (_t.randn((NKV * D, H), device="cuda", dtype=_t.float32, generator=g) * 0.02).to(_t.bfloat16)
    w_v = (_t.randn((NKV * D, H), device="cuda", dtype=_t.float32, generator=g) * 0.02).to(_t.bfloat16)
    w_q_norm = ((_t.randn((D,), device="cuda", dtype=_t.float32, generator=g) * 0.1) + 1.0).to(_t.bfloat16)
    w_k_norm = ((_t.randn((D,), device="cuda", dtype=_t.float32, generator=g) * 0.1) + 1.0).to(_t.bfloat16)
    inv_freq = (1.0 / (1_000_000.0 ** (
        _t.arange(0, D, 2, device="cuda", dtype=_t.float32) / D
    ))).contiguous()
    attention_mask = _t.ones((B, S), device="cuda", dtype=_t.int64)

    # Buffers (sized exactly as inductor sizes them in output_code.py).
    buf2 = _t.empty((B * S, NQ * D), device="cuda", dtype=_t.bfloat16)
    buf5 = _t.empty((B * S, NKV * D), device="cuda", dtype=_t.bfloat16)
    buf9 = _t.empty((B * S, NKV * D), device="cuda", dtype=_t.bfloat16)
    # buf8/buf10 are 4D bf16; strides match inductor (transposed view).
    buf8 = _t.empty_strided(
        (B, NKV, S, D), (NKV * S * D, D, NKV * D, 1),
        device="cuda", dtype=_t.bfloat16,
    )  # (1, 8, 512, 128) with stride (524288, 128, 1024, 1)
    buf10 = _t.empty_strided(
        (B, NQ, S, D), (NQ * S * D, D, NQ * D, 1),
        device="cuda", dtype=_t.bfloat16,
    )  # (1, 16, 512, 128) with stride (1048576, 128, 2048, 1)
    buf11 = _t.empty((B, NQ, S, D), device="cuda", dtype=_t.bfloat16)
    buf12 = _t.empty((B, NQ, S, D), device="cuda", dtype=_t.bfloat16)
    buf13 = _t.empty((B, 1, S, S), device="cuda", dtype=_t.bfloat16)
    buf39 = _t.empty((B, 1, S, S), device="cuda", dtype=_t.bfloat16)

    # Parse launch configs.
    per_q_kw = _parse_launch_kwargs(per_q_meta["sample_inputs"]["kernel_kwargs"])
    per_k_kw = _parse_launch_kwargs(per_k_meta["sample_inputs"]["kernel_kwargs"])
    w3_kw = _parse_launch_kwargs(where_3_meta["sample_inputs"]["kernel_kwargs"])
    w4_kw = _parse_launch_kwargs(where_4_meta["sample_inputs"]["kernel_kwargs"])

    per_q_XBLOCK = per_q_kw.get("XBLOCK", 1)
    per_k_XBLOCK = per_k_kw.get("XBLOCK", 1)
    w3_XBLOCK = w3_kw.get("XBLOCK", 1024)
    w4_XBLOCK = w4_kw.get("XBLOCK", 512)

    # Runtime xnumels at prefill_512_b1 (from output_code.py call sites).
    per_q_xnumel = NQ * S  # 8192
    per_k_xnumel = NKV * S  # 4096
    w3_xnumel = NQ * S * D  # 1048576
    w4_xnumel = B * S * S   # 262144

    # Inductor calls cuBLAS via `extern_kernels.mm(input,
    # reinterpret_tensor(w, (in, out), (1, in)))` -- a stride-swap view that
    # cuBLAS recognises as a transposed-B operand and routes to the fast path.
    # Direct `mm(., w.T)` in torch unexpectedly hits a 4-5x slower kernel.
    w_q_T = _t.as_strided(w_q, (H, NQ * D), (1, H))
    w_k_T = _t.as_strided(w_k, (H, NKV * D), (1, H))
    w_v_T = _t.as_strided(w_v, (H, NKV * D), (1, H))
    h_flat = hidden_states.view(B * S, H)

    def _launch_q_proj():
        _t.mm(h_flat, w_q_T, out=buf2)

    def _launch_k_proj():
        _t.mm(h_flat, w_k_T, out=buf5)

    def _launch_v_proj():
        _t.mm(h_flat, w_v_T, out=buf9)

    def _launch_q_rope():
        # in_out=buf10 (Q out), in_ptr0=buf2 (Q proj), in_ptr1=w_q_norm, in_ptr2=inv_freq
        grid = (triton.cdiv(per_q_xnumel, per_q_XBLOCK),)
        per_q_fn[grid](
            buf10, buf2, w_q_norm, inv_freq, per_q_xnumel, 128,
            XBLOCK=per_q_XBLOCK,
            num_warps=per_q_meta["sample_inputs"]["num_warps"],
            num_stages=per_q_meta["sample_inputs"]["num_stages"],
        )

    def _launch_k_rope():
        # in_ptr0=buf5 (K proj), in_ptr1=w_k_norm, in_ptr2=inv_freq, out_ptr2=buf8
        grid = (triton.cdiv(per_k_xnumel, per_k_XBLOCK),)
        per_k_fn[grid](
            buf5, w_k_norm, inv_freq, buf8, per_k_xnumel, 128,
            XBLOCK=per_k_XBLOCK,
            num_warps=per_k_meta["sample_inputs"]["num_warps"],
            num_stages=per_k_meta["sample_inputs"]["num_stages"],
        )

    def _launch_kv_expand_k():
        grid = (triton.cdiv(w3_xnumel, w3_XBLOCK),)
        where_3_fn[grid](
            buf8, buf11, w3_xnumel,
            XBLOCK=w3_XBLOCK,
            num_warps=where_3_meta["sample_inputs"]["num_warps"],
            num_stages=where_3_meta["sample_inputs"]["num_stages"],
        )

    def _launch_kv_expand_v():
        # The V case in inductor uses buf9 viewed as (1, 8, 512, 128) with
        # the same stride pattern, but buf9 is contiguous (B*S, NKV*D).
        # We must reinterpret it the way inductor does: (1, 8, 512, 128)
        # with strides (524288, 128, 1024, 1).
        buf9_v = _t.as_strided(
            buf9, (B, NKV, S, D), (NKV * S * D, D, NKV * D, 1)
        )
        grid = (triton.cdiv(w3_xnumel, w3_XBLOCK),)
        where_3_fn[grid](
            buf9_v, buf12, w3_xnumel,
            XBLOCK=w3_XBLOCK,
            num_warps=where_3_meta["sample_inputs"]["num_warps"],
            num_stages=where_3_meta["sample_inputs"]["num_stages"],
        )

    def _launch_mask():
        grid = (triton.cdiv(w4_xnumel, w4_XBLOCK),)
        where_4_fn[grid](
            attention_mask, buf13, buf39, w4_xnumel,
            XBLOCK=w4_XBLOCK,
            num_warps=where_4_meta["sample_inputs"]["num_warps"],
            num_stages=where_4_meta["sample_inputs"]["num_stages"],
        )

    # Sanity-launch each kernel once.
    _launch_q_proj()
    _launch_k_proj()
    _launch_v_proj()
    _launch_q_rope()
    _launch_k_rope()
    _launch_kv_expand_k()
    _launch_kv_expand_v()
    _launch_mask()
    torch.cuda.synchronize()

    def launch_all():
        _launch_q_proj()
        _launch_q_rope()
        _launch_k_proj()
        _launch_k_rope()
        _launch_v_proj()
        _launch_kv_expand_k()
        _launch_kv_expand_v()
        _launch_mask()

    # Bench each component plus the full chain.
    breakdown = {}
    for name, fn in [
        ("q_proj_mm", _launch_q_proj),
        ("k_proj_mm", _launch_k_proj),
        ("v_proj_mm", _launch_v_proj),
        ("q_rmsnorm_rope", _launch_q_rope),
        ("k_rmsnorm_rope", _launch_k_rope),
        ("kv_expand_k", _launch_kv_expand_k),
        ("kv_expand_v", _launch_kv_expand_v),
        ("causal_mask", _launch_mask),
    ]:
        try:
            ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
            breakdown[name] = ms * 1000.0
        except Exception as exc:  # noqa: BLE001
            breakdown[name] = f"ERROR: {exc}"

    full_ms = triton.testing.do_bench(launch_all, warmup=25, rep=100, return_mode="median")
    full_us = full_ms * 1000.0
    breakdown["full_chain"] = full_us
    return full_us, breakdown


def _bench_pointwise_swiglu(meta: dict) -> float:
    """Benchmark the SwiGLU `triton_poi_fused__unsafe_view_mul_silu_6` kernel.

    Returns median latency in microseconds.
    """
    si = meta["sample_inputs"]
    kernel_path = Path(si["kernel_file"])
    if not kernel_path.exists():
        raise FileNotFoundError(f"inductor cache file missing: {kernel_path}")

    kname = meta["name"]
    fn = _load_kernel_module(kernel_path, kname)

    XBLOCK = _parse_launch_kwargs(si["kernel_kwargs"]).get("XBLOCK", 1024)
    num_warps = si["num_warps"]
    num_stages = si["num_stages"]

    # SwiGLU sample: shapes (1, 512, 6144) bf16 in_out_ptr0, (512, 6144) bf16 in_ptr0.
    # in_out_ptr0 is mutated in place; both inputs same numel (3,145,728).
    shape_out = tuple(si["Input Dims"][0])  # (1, 512, 6144)
    shape_in = tuple(si["Input Dims"][1])   # (512, 6144)

    g = torch.Generator(device="cuda")
    g.manual_seed(0xC0FFEE)
    in_out = torch.randn(shape_out, device="cuda", dtype=torch.bfloat16, generator=g)
    in_a = torch.randn(shape_in, device="cuda", dtype=torch.bfloat16, generator=g)

    xnumel = in_out.numel()
    assert xnumel == in_a.numel(), (xnumel, in_a.numel())
    grid = (triton.cdiv(xnumel, XBLOCK),)

    # Sanity-launch once to make sure it compiles and runs.
    fn[grid](in_out, in_a, xnumel, XBLOCK=XBLOCK,
             num_warps=num_warps, num_stages=num_stages)
    torch.cuda.synchronize()

    # Use a *fresh* in_out each iteration would distort timing with an extra
    # copy; the kernel is mutating but the result-of-silu*y has the same
    # statistics as randn so iterating in place is fine for timing.
    def launch():
        fn[grid](in_out, in_a, xnumel, XBLOCK=XBLOCK,
                 num_warps=num_warps, num_stages=num_stages)

    ms = triton.testing.do_bench(launch, warmup=25, rep=100, return_mode="median")
    return ms * 1000.0  # us


def main() -> int:
    # Try to load existing JSON to preserve other keys (idempotent extension).
    out_path = HERE / "microbench_inductor.json"
    try:
        out: dict = json.loads(out_path.read_text())
    except Exception:
        out = {}

    # --- SwiGLU (the priority) ----------------------------------------------
    swiglu_dir = KERNELS_DIR / "triton_poi_fused__unsafe_view_mul_silu_6"
    swiglu_meta_path = swiglu_dir / "metadata.json"
    swiglu_meta_full = json.loads(swiglu_meta_path.read_text())
    # The metadata has a per-workload stats dict; we grab the prefill_512_b1
    # entry which is the canonical shape for this experiment.
    stats = swiglu_meta_full["stats"]["prefill_512_b1"]
    swiglu_us = _bench_pointwise_swiglu(stats)
    out["triton_poi_fused__unsafe_view_mul_silu_6"] = {
        "median_us": swiglu_us,
        "profiler_aggregate_us": stats["mean_us"],
        "shape": stats["sample_inputs"]["Input Dims"][0],
        "dtype": "bf16",
        "XBLOCK": _parse_launch_kwargs(stats["sample_inputs"]["kernel_kwargs"]).get("XBLOCK"),
        "num_warps": stats["sample_inputs"]["num_warps"],
        "num_stages": stats["sample_inputs"]["num_stages"],
    }
    out["swiglu_us"] = swiglu_us
    print(f"[swiglu] median = {swiglu_us:.2f} us "
          f"(profiler-aggregate mean = {stats['mean_us']:.2f} us)")

    # --- RMSNorm (residual-fused, kernel _9 -- canonical prefill variant) ----
    rms_dir = KERNELS_DIR / "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9"
    rms_meta_path = rms_dir / "metadata.json"
    rms_meta_full = json.loads(rms_meta_path.read_text())
    rms_stats = rms_meta_full["stats"]["prefill_512_b1"]
    rmsnorm_us, rms_sanity = _bench_reduction_rmsnorm(rms_stats)
    out["triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9"] = {
        "median_us": rmsnorm_us,
        "profiler_aggregate_us": rms_stats["mean_us"],
        "shape": rms_stats["sample_inputs"]["Input Dims"][0],
        "dtype": "bf16",
        "XBLOCK": _parse_launch_kwargs(rms_stats["sample_inputs"]["kernel_kwargs"]).get("XBLOCK"),
        "R0_BLOCK": _parse_launch_kwargs(rms_stats["sample_inputs"]["kernel_kwargs"]).get("R0_BLOCK"),
        "num_warps": rms_stats["sample_inputs"]["num_warps"],
        "num_stages": rms_stats["sample_inputs"]["num_stages"],
        "sanity_strict": rms_sanity,
    }
    out["rmsnorm_us"] = rmsnorm_us
    print(f"[rmsnorm] median = {rmsnorm_us:.2f} us "
          f"(profiler-aggregate mean = {rms_stats['mean_us']:.2f} us)")

    # --- SDPA prelude (full chain: 3 GEMMs + 5 fused Triton kernels) --------
    try:
        prelude_us, prelude_breakdown = _bench_sdpa_prelude()
    except Exception as exc:  # noqa: BLE001
        print(f"[sdpa_prelude] FAILED: {exc}", file=sys.stderr)
        import traceback as _tb
        _tb.print_exc()
        prelude_us, prelude_breakdown = None, {"error": str(exc)}
    if prelude_us is not None:
        out["sdpa_prelude"] = {
            "median_us": prelude_us,
            "breakdown_us": prelude_breakdown,
            "shape": {
                "hidden_states": [1, 512, 2048],
                "q_out": [1, 16, 512, 128],
                "k_out": [1, 16, 512, 128],
                "v_out": [1, 16, 512, 128],
                "mask_out": [1, 1, 512, 512],
            },
            "dtype": "bf16",
            "note": (
                "Full prelude as inductor emits it for prefill_512_b1: "
                "3x cuBLAS QKV mm + per_..._1 (Q norm+RoPE) + per_..._2 "
                "(K norm+RoPE) + 2x where_3 (KV GQA-expand) + where_4 "
                "(causal mask). This is the codegen-vs-codegen baseline."
            ),
        }
        out["sdpa_prelude_us"] = prelude_us
        print(f"[sdpa_prelude] median = {prelude_us:.2f} us")
        for k, v in prelude_breakdown.items():
            if isinstance(v, (int, float)):
                print(f"    [{k}] = {v:.2f} us")
            else:
                print(f"    [{k}] = {v}")
    else:
        print("[sdpa_prelude] SKIPPED (see error above)", file=sys.stderr)

    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
