
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 65664}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 513)
    x1 = xindex // 513
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 513, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 512, tl.int64)
    tmp7 = tmp5 <= tmp6
    tmp8 = tl.full([1], True, tl.int1)
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + (513*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = (tmp10 != 0)
    tmp12 = tmp9 & tmp11
    tmp13 = tl.full([1], 0.0, tl.float32)
    tmp14 = tl.full([1], float("-inf"), tl.float32)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 520, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.full([1], 0.0, tl.float32)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp17, tmp23)
    tl.store(out_ptr0 + (x0 + 520*x1), tmp24, xmask)
    tl.store(out_ptr1 + (x0 + 520*x1), tmp24, xmask)
