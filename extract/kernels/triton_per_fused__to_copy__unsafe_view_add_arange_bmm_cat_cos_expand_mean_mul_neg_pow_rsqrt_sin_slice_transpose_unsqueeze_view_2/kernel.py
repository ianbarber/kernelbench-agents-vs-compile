
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 11264}}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0).to(tl.float32)
    tmp43 = tl.load(in_ptr2 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = r0_1
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tl.full([1, 1], 64, tl.int64)
    tmp11 = tmp7 < tmp10
    tmp12 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp11 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.full([1, 1], 128.0, tl.float32)
    tmp16 = (tmp6 / tmp15)
    tmp17 = tl.full([1, 1], 1e-06, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp12 * tmp21
    tmp23 = -tmp22
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp11, tmp23, tmp24)
    tmp26 = tmp7 >= tmp10
    tmp27 = tl.full([1, 1], 128, tl.int64)
    tmp28 = tmp7 < tmp27
    tmp29 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp26 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tl.full([1, 1], 128.0, tl.float32)
    tmp33 = (tmp6 / tmp32)
    tmp34 = tl.full([1, 1], 1e-06, tl.float32)
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp29 * tmp38
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp26, tmp39, tmp40)
    tmp42 = tl.where(tmp11, tmp25, tmp41)
    tmp44 = tl.full([1, 1], 512.0, tl.float32)
    tmp45 = tmp43 * tmp44
    tmp46 = tl_math.sin(tmp45)
    tmp47 = tl.full([1, 1], 1.0, tl.float32)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp42 * tmp49
    tmp52 = tl.full([1, 1], 128.0, tl.float32)
    tmp53 = (tmp6 / tmp52)
    tmp54 = tl.full([1, 1], 1e-06, tl.float32)
    tmp55 = tmp53 + tmp54
    tmp56 = libdevice.rsqrt(tmp55)
    tmp57 = tmp1 * tmp56
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp51 * tmp58
    tmp60 = tl_math.cos(tmp45)
    tmp61 = tmp60 * tmp47
    tmp62 = tmp61.to(tl.float32)
    tmp63 = tmp59 * tmp62
    tmp64 = tmp63 + tmp50
    tl.store(out_ptr2 + (r0_1 + 65664*x0), tmp64, xmask)
