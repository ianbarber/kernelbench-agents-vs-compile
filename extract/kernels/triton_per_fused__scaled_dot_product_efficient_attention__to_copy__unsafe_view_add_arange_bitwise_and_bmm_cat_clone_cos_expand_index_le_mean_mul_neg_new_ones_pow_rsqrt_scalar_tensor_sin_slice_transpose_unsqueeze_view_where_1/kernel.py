
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 402654208}}
)
@triton.jit
def triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = ((xindex // 16) % 2048)
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), None).to(tl.float32)
    tmp42 = tl.load(in_ptr2 + ((r0_1 % 64)), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp6 = r0_1
    tmp7 = tl.full([1, 1], 0, tl.int64)
    tmp8 = tmp6 >= tmp7
    tmp9 = tl.full([1, 1], 64, tl.int64)
    tmp10 = tmp6 < tmp9
    tmp11 = tl.load(in_ptr1 + (tl.broadcast_to(64 + (r0_1), [XBLOCK, R0_BLOCK])), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (64 + 128*x0 + (r0_1)), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.full([1, 1], 128.0, tl.float32)
    tmp15 = (tmp5 / tmp14)
    tmp16 = tl.full([1, 1], 1e-06, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp13 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp11 * tmp20
    tmp22 = -tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp6 >= tmp9
    tmp26 = tl.full([1, 1], 128, tl.int64)
    tmp27 = tmp6 < tmp26
    tmp28 = tl.load(in_ptr1 + (tl.broadcast_to((-64) + r0_1, [XBLOCK, R0_BLOCK])), tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tl.load(in_ptr0 + (128*x0 + ((-64) + r0_1)), tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.full([1, 1], 128.0, tl.float32)
    tmp32 = (tmp5 / tmp31)
    tmp33 = tl.full([1, 1], 1e-06, tl.float32)
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp30 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp28 * tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp25, tmp38, tmp39)
    tmp41 = tl.where(tmp10, tmp24, tmp40)
    tmp43 = x3
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp42 * tmp44
    tmp46 = tl_math.sin(tmp45)
    tmp47 = tl.full([1, 1], 1.0, tl.float32)
    tmp48 = tmp46 * tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp41 * tmp49
    tmp52 = tl.full([1, 1], 128.0, tl.float32)
    tmp53 = (tmp5 / tmp52)
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
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp64, None)
