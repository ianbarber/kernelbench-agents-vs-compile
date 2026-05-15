# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._C._dynamo.guards import copy_misaligned
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/mn/cmn7txeqdpy2cicmgn7i7s25h25ed3utdprx4orjdvbwmvxschuf.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_3, rsqrt, hidden_states_1, to_7, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_3 => add_3
#   hidden_states => convert_element_type_4
#   hidden_states_1 => mul_3
#   hidden_states_2 => mul_4
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_7 => convert_element_type_5
#   variance => mean
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %arg5_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg5_1]
#   %buf0 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf0]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 2), kwargs = {})
#   %mean : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_3 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %mul_3 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %rsqrt), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3, torch.bfloat16), kwargs = {})
#   %mul_4 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg5_1, %convert_element_type_5), kwargs = {})
#   return %buf0,%mul_4
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 12288}}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp2 = tl.full([1, 1], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp14 = tl.load(in_ptr0 + (0))
    tmp15 = tl.broadcast_to(tmp14, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp13 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.full([1, 1], 151936, tl.int32)
        tmp17 = tmp15 + tmp16
        tmp18 = tmp15 < 0
        tmp19 = tl.where(tmp18, tmp17, tmp15)
        tl.device_assert((0 <= tmp19) & (tmp19 < 151936), "index out of bounds: 0 <= tmp19 < 151936")
        tmp21 = tl.load(in_ptr1 + (r0_0 + 2048*tmp19), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tl.full([1, 1], 2048.0, tl.float32)
        tmp24 = (tmp11 / tmp23)
        tmp25 = tl.full([1, 1], 1e-06, tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = libdevice.rsqrt(tmp26)
        tmp28 = tmp22 * tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp13 * tmp29
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp30, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/47/c474y6diye2ip5d3sn25fxrhsekauqxpwyqnhonpurxjuqx4gsed.py
# Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_9, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.slice, aten.neg, aten.sin, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   add_4 => add_4
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => convert_element_type
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default_1, full_default_2, slice_5, where
#   batch_arange => iota_1
#   batch_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   cat_1 => cat
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_14
#   emb => clone, expand_4, unsqueeze_13, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_11 => unsqueeze_16
#   getitem_12 => unsqueeze_17
#   getitem_4 => index
#   getitem_5 => unsqueeze_10, unsqueeze_11
#   getitem_6 => unsqueeze_12
#   hidden_states_3 => convert_element_type_8
#   hidden_states_4 => mul_5
#   hidden_states_7 => expand_5
#   hidden_states_8 => expand_6
#   key => clone_2, view_13
#   kv_arange => add_2
#   kv_indices => unsqueeze_7, unsqueeze_8, unsqueeze_9
#   le => le
#   linear => view_5
#   matmul => mul
#   mul_5 => mul_6
#   mul_8 => mul_9
#   mul_9 => mul_10
#   neg => neg
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type_1
#   pow_2 => pow_2
#   q_arange => add_1
#   q_embed => add_6
#   q_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   query_states => permute_2
#   result => full_default
#   result_1 => bitwise_and
#   result_2 => bitwise_and_1
#   rsqrt_1 => rsqrt_1
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_15
#   to_9 => convert_element_type_9
#   value => clone_3, view_14
#   variance_1 => mean_1
#   view => view_6
#   x1 => slice_1
#   x2 => slice_2
# Graph fragment:
#   %mm : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg7_1 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %buf3 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 16]cuda:0" = PlaceHolder[target=buf3]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mul_10 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0" = PlaceHolder[target=mul_10]
#   %view_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 2048]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 128]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_4 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_5 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %rsqrt_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %convert_element_type_9), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%mul_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_11, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 2048), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_12, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 64][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 1, 1, 64][64, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 64][64, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 1, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_9 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_14), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 16, 1, 64][2048, 128, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 64, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 16, 1, 64][1024, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_2,), kwargs = {})
#   %slice_1 : Tensor "bf16[1, 16, 1, 64][2048, 128, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_2, 3, 0, 64), kwargs = {})
#   %cat : Tensor "bf16[1, 16, 1, 128][2048, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_1], -1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_15 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %mul_10 : Tensor "bf16[1, 16, 1, 128][2048, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_15), kwargs = {})
#   %add_6 : Tensor "bf16[1, 16, 1, 128][2048, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %unsqueeze_16 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_2, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_16, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 16, 2049, 128]), kwargs = {})
#   %unsqueeze_17 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 2), kwargs = {})
#   %expand_6 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_17, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 16, 2049, 128]), kwargs = {})
#   %full_default : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_4 : Tensor "i64[2049][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2049,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[2049][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_7 : Tensor "i64[1, 2049][2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "i64[1, 1, 2049][2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 2), kwargs = {})
#   %iota_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 2048), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %le : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_9, %unsqueeze_6), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %convert_element_type : Tensor "b8[1, 2049][2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bool), kwargs = {})
#   %iota_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %index : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%convert_element_type, [%unsqueeze_3, %unsqueeze_9]), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %index), kwargs = {})
#   %expand : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.aten.expand.default](args = (%bitwise_and_1, [1, -1, 1, 2049]), kwargs = {})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[1, 1, 1, 2056][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, 7], 0.0), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 1, 1, 2049][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, 2049), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 16, 1, 2049][2056, 0, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_5, [1, 16, 1, 2049]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_6, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.08838834764831845})
#   return %buf3,%mul_10,%buf13
triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1 = async_compile.triton('triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 0, 'r0_': 21504}}
)
@triton.jit
def triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp44 = tl.full([1, 1], 2048.0, tl.float32)
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
    tl.store(in_out_ptr0 + (r0_1 + 128*x0), tmp64, xmask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/qt/cqtg3wjjghea37jsx5sn4entk4uia4tkym4wef2i7t4lau5lgqh3.py
# Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, sin, sin_1, sin_2, sin_3, linear_1, view_1, hidden_states_5, pow_3, variance_2, add_5, rsqrt_2, hidden_states_6, to_11, mul_7, key_states, mul_10, x2_1, neg_1, x1_1, cat_2, mul_11, k_embed], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
# Source node to ATen node mapping:
#   add_5 => add_5
#   arange => iota
#   cat_2 => cat_1
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_14
#   emb => clone, expand_4, unsqueeze_13, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_5 => unsqueeze_10, unsqueeze_11
#   getitem_6 => unsqueeze_12
#   hidden_states_5 => convert_element_type_12
#   hidden_states_6 => mul_7
#   k_embed => add_7
#   key_states => permute_4
#   linear_1 => view_8
#   matmul => mul
#   mul_10 => mul_11
#   mul_11 => mul_12
#   mul_7 => mul_8
#   neg_1 => neg_1
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type_1
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   sin => sin
#   sin_1 => mul_2
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_15
#   to_11 => convert_element_type_13
#   variance_2 => mean_2
#   view_1 => view_9
#   x1_1 => slice_3
#   x2_1 => slice_4
# Graph fragment:
#   %mm_1 : Tensor "bf16[1, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %arg9_1 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=arg9_1]
#   %buf6 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 8]cuda:0" = PlaceHolder[target=buf6]
#   %arg4_1 : Tensor "f32[64][1]cuda:0" = PlaceHolder[target=arg4_1]
#   %mul_12 : Tensor "bf16[1, 8, 1, 128][1024, 128, 1024, 1]cuda:0" = PlaceHolder[target=mul_12]
#   %unsqueeze_10 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_11, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 2048), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_12, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 64][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 1, 1, 64][64, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 64][64, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 1, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %sin : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_3,), kwargs = {})
#   %mul_2 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %unsqueeze_15 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %view_8 : Tensor "bf16[1, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 1, 1024]), kwargs = {})
#   %view_9 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 1, -1, 128]), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_12, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_5 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_7 : Tensor "f32[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %rsqrt_2), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %mul_8 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg9_1, %convert_element_type_13), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 8, 1, 128][1024, 128, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%mul_8, [0, 2, 1, 3]), kwargs = {})
#   %mul_11 : Tensor "bf16[1, 8, 1, 128][1024, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_14), kwargs = {})
#   %slice_4 : Tensor "bf16[1, 8, 1, 64][1024, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 64, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "bf16[1, 8, 1, 64][512, 64, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_4,), kwargs = {})
#   %slice_3 : Tensor "bf16[1, 8, 1, 64][1024, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 64), kwargs = {})
#   %cat_1 : Tensor "bf16[1, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_3], -1), kwargs = {})
#   %mul_12 : Tensor "bf16[1, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_15), kwargs = {})
#   %add_7 : Tensor "bf16[1, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   return %buf6,%mul_12,%add_7
triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2', '''
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
    tmp44 = tl.full([1, 1], 2048.0, tl.float32)
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
    tl.store(out_ptr2 + (r0_1 + 262272*x0), tmp64, xmask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/54/c54yi7k4pvbewoeko4mgdhhy44lmkgwkgluhgigfwfkzupjhbn2y.py
# Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, linear_1, view_1, hidden_states_5, pow_3, variance_2, add_5, rsqrt_2, hidden_states_6, to_11, mul_7, key_states, mul_10, k_embed, keys], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   add_5 => add_5
#   arange => iota
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_14
#   emb => clone, expand_4, unsqueeze_13, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_5 => unsqueeze_10, unsqueeze_11
#   getitem_6 => unsqueeze_12
#   hidden_states_5 => convert_element_type_12
#   hidden_states_6 => mul_7
#   k_embed => add_7
#   key_states => permute_4
#   keys => cat_2
#   linear_1 => view_8
#   matmul => mul
#   mul_10 => mul_11
#   mul_7 => mul_8
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type_1
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
#   to_11 => convert_element_type_13
#   variance_2 => mean_2
#   view_1 => view_9
# Graph fragment:
#   %arg2_1 : Tensor "bf16[1, 8, 2048, 128][2097152, 128, 1024, 1]cuda:0" = PlaceHolder[target=arg2_1]
#   %unsqueeze_10 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_11, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 2048), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_12, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 64][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 1, 1, 64][64, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 64][64, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 1, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %view_8 : Tensor "bf16[1, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, 1, 1024]), kwargs = {})
#   %view_9 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_8, [1, 1, -1, 128]), kwargs = {})
#   %convert_element_type_12 : Tensor "f32[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_12, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_5 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, 1, 8, 1][8, 8, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_7 : Tensor "f32[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %rsqrt_2), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7, torch.bfloat16), kwargs = {})
#   %mul_8 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg9_1, %convert_element_type_13), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 8, 1, 128][1024, 128, 1024, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%mul_8, [0, 2, 1, 3]), kwargs = {})
#   %mul_11 : Tensor "bf16[1, 8, 1, 128][1024, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_14), kwargs = {})
#   %add_7 : Tensor "bf16[1, 8, 1, 128][1024, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %mul_12), kwargs = {})
#   %cat_2 : Tensor "bf16[1, 8, 2049, 128][2098176, 262272, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg2_1, %add_7], -2), kwargs = {})
#   return %buf8
triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12582912}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 2048)
    x2 = xindex // 262144
    x3 = (xindex % 262144)
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 1024*x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x3 + 262272*x2), tmp0, None)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/4a/c4a6bawraq6okbpzygbnrgrfjlmihoh2daig4q2odrjqhxcc7dv4.py
# Topologically Sorted Source Nodes: [linear_2, view_2, value_states, values], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
# Source node to ATen node mapping:
#   linear_2 => view_11
#   value_states => permute_6
#   values => cat_3
#   view_2 => view_12
# Graph fragment:
#   %arg11_1 : Tensor "bf16[1, 8, 2048, 128][2097152, 128, 1024, 1]cuda:0" = PlaceHolder[target=arg11_1]
#   %mm_2 : Tensor "bf16[1, 1024][1024, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %view_11 : Tensor "bf16[1, 1, 1024][1024, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_2, [1, 1, 1024]), kwargs = {})
#   %view_12 : Tensor "bf16[1, 1, 8, 128][1024, 1024, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_11, [1, 1, -1, 128]), kwargs = {})
#   %permute_6 : Tensor "bf16[1, 8, 1, 128][1024, 128, 1024, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1, 3]), kwargs = {})
#   %cat_3 : Tensor "bf16[1, 8, 2049, 128][2098176, 262272, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%arg11_1, %permute_6], -2), kwargs = {})
#   return %cat_3
triton_poi_fused__unsafe_view_cat_transpose_view_4 = async_compile.triton('triton_poi_fused__unsafe_view_cat_transpose_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_cat_transpose_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 12589056}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_cat_transpose_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2098176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 128) % 2049)
    x0 = (xindex % 128)
    x2 = xindex // 262272
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 128*x2 + 1024*(x1)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2049, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 128*x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/av/cavdjoucphbcgzsrbohixed2qqtbsko5m554u6lhlsget2iruehh.py
# Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   add_4 => add_4
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => convert_element_type
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default_1, full_default_2, slice_5, where
#   batch_arange => iota_1
#   batch_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_14
#   emb => clone, expand_4, unsqueeze_13, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_11 => unsqueeze_16
#   getitem_12 => unsqueeze_17
#   getitem_4 => index
#   getitem_5 => unsqueeze_10, unsqueeze_11
#   getitem_6 => unsqueeze_12
#   hidden_states_3 => convert_element_type_8
#   hidden_states_4 => mul_5
#   hidden_states_7 => expand_5
#   hidden_states_8 => expand_6
#   key => clone_2, view_13
#   kv_arange => add_2
#   kv_indices => unsqueeze_7, unsqueeze_8, unsqueeze_9
#   le => le
#   linear => view_5
#   matmul => mul
#   mul_5 => mul_6
#   mul_8 => mul_9
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type_1
#   pow_2 => pow_2
#   q_arange => add_1
#   q_embed => add_6
#   q_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   query_states => permute_2
#   result => full_default
#   result_1 => bitwise_and
#   result_2 => bitwise_and_1
#   rsqrt_1 => rsqrt_1
#   to_9 => convert_element_type_9
#   value => clone_3, view_14
#   variance_1 => mean_1
#   view => view_6
# Graph fragment:
#   %cat_2 : Tensor "bf16[1, 8, 2049, 128][2098176, 262272, 128, 1]cuda:0" = PlaceHolder[target=cat_2]
#   %view_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 2048]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 128]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_4 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_5 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %rsqrt_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %convert_element_type_9), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%mul_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_11, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 2048), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_12, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 64][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 1, 1, 64][64, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 64][64, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 1, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_9 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_14), kwargs = {})
#   %add_6 : Tensor "bf16[1, 16, 1, 128][2048, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %unsqueeze_16 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_2, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_16, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 16, 2049, 128]), kwargs = {})
#   %unsqueeze_17 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 2), kwargs = {})
#   %expand_6 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_17, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 16, 2049, 128]), kwargs = {})
#   %full_default : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_4 : Tensor "i64[2049][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2049,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[2049][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_7 : Tensor "i64[1, 2049][2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "i64[1, 1, 2049][2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 2), kwargs = {})
#   %iota_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 2048), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %le : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_9, %unsqueeze_6), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %convert_element_type : Tensor "b8[1, 2049][2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bool), kwargs = {})
#   %iota_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %index : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%convert_element_type, [%unsqueeze_3, %unsqueeze_9]), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %index), kwargs = {})
#   %expand : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.aten.expand.default](args = (%bitwise_and_1, [1, -1, 1, 2049]), kwargs = {})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[1, 1, 1, 2056][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, 7], 0.0), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 1, 1, 2049][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, 2049), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 16, 1, 2049][2056, 0, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_5, [1, 16, 1, 2049]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_6, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.08838834764831845})
#   return %buf14
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 20981760}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4196352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 262272)
    x1 = xindex // 262272
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 262272*(x1 // 2)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/pg/cpgwdksqi2xf2uvrrayq66vp2ckfxnbdltj5r6j5ocxajshhbvkz.py
# Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output, linear_7, view_3, hidden_states_17, pow_6, variance_5, add_12, rsqrt_5, hidden_states_18, to_17, mul_18, query_states_1, cos_4, mul_21, q_embed_1, getitem_17, hidden_states_21, key_1, getitem_18, hidden_states_22, value_1, attn_output_4], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
# Source node to ATen node mapping:
#   add_12 => add_13
#   add_4 => add_4
#   arange => iota
#   arange_3 => iota_3
#   arange_4 => iota_4
#   attention_mask => convert_element_type
#   attention_mask_1 => expand
#   attn_output => _scaled_dot_product_efficient_attention, constant_pad_nd, expand_7, full_default_1, full_default_2, slice_5, where
#   attn_output_4 => _scaled_dot_product_efficient_attention_1, constant_pad_nd_1, expand_10, full_default_3, full_default_4, slice_10, where_1
#   batch_arange => iota_1
#   batch_indices => unsqueeze_1, unsqueeze_2, unsqueeze_3
#   cos => cos
#   cos_1 => mul_1
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_14
#   cos_4 => unsqueeze_18
#   emb => clone, expand_4, unsqueeze_13, view_3
#   expand_1 => expand_1
#   freqs => permute
#   getitem_11 => unsqueeze_16
#   getitem_12 => unsqueeze_17
#   getitem_17 => unsqueeze_20
#   getitem_18 => unsqueeze_21
#   getitem_4 => index
#   getitem_5 => unsqueeze_10, unsqueeze_11
#   getitem_6 => unsqueeze_12
#   hidden_states_17 => convert_element_type_32
#   hidden_states_18 => mul_18
#   hidden_states_21 => expand_8
#   hidden_states_22 => expand_9
#   hidden_states_3 => convert_element_type_8
#   hidden_states_4 => mul_5
#   hidden_states_7 => expand_5
#   hidden_states_8 => expand_6
#   key => clone_2, view_13
#   key_1 => clone_4, view_33
#   kv_arange => add_2
#   kv_indices => unsqueeze_7, unsqueeze_8, unsqueeze_9
#   le => le
#   linear => view_5
#   linear_7 => view_25
#   matmul => mul
#   mul_18 => mul_19
#   mul_21 => mul_22
#   mul_5 => mul_6
#   mul_8 => mul_9
#   position_ids => add
#   position_ids_1 => unsqueeze
#   position_ids_expanded => convert_element_type_1
#   pow_2 => pow_2
#   pow_6 => pow_6
#   q_arange => add_1
#   q_embed => add_6
#   q_embed_1 => add_15
#   q_indices => unsqueeze_4, unsqueeze_5, unsqueeze_6
#   query_states => permute_2
#   query_states_1 => permute_13
#   result => full_default
#   result_1 => bitwise_and
#   result_2 => bitwise_and_1
#   rsqrt_1 => rsqrt_1
#   rsqrt_5 => rsqrt_5
#   to_17 => convert_element_type_33
#   to_9 => convert_element_type_9
#   value => clone_3, view_14
#   value_1 => clone_5, view_34
#   variance_1 => mean_1
#   variance_5 => mean_5
#   view => view_6
#   view_3 => view_26
# Graph fragment:
#   %arg3_1 : Tensor "i64[1, 2049][2049, 1]cuda:0" = PlaceHolder[target=arg3_1]
#   %view_5 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm, [1, 1, 2048]), kwargs = {})
#   %view_6 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_5, [1, 1, -1, 128]), kwargs = {})
#   %convert_element_type_8 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_8, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_4 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %mul_5 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %rsqrt_1), kwargs = {})
#   %convert_element_type_9 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5, torch.bfloat16), kwargs = {})
#   %mul_6 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg7_1, %convert_element_type_9), kwargs = {})
#   %permute_2 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%mul_6, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_10 : Tensor "f32[1, 64][64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%arg4_1, 0), kwargs = {})
#   %unsqueeze_11 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_10, 2), kwargs = {})
#   %expand_1 : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_11, [1, -1, 1]), kwargs = {})
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 2048), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add, 0), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_12, torch.float32), kwargs = {})
#   %mul : Tensor "f32[1, 64, 1][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %expand_3), kwargs = {})
#   %permute : Tensor "f32[1, 1, 64][64, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%mul, [0, 2, 1]), kwargs = {})
#   %unsqueeze_13 : Tensor "f32[1, 1, 1, 64][64, 1, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1, 2, 64][64, 1, 0, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_13, [1, 1, 2, 64]), kwargs = {})
#   %clone : Tensor "f32[1, 1, 2, 64][128, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1, 128]), kwargs = {})
#   %cos : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_3,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1, 128][128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1, 128][128, 128, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_14 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_9 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_14), kwargs = {})
#   %add_6 : Tensor "bf16[1, 16, 1, 128][2048, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %unsqueeze_16 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_2, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_16, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   %view_13 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [1, 16, 2049, 128]), kwargs = {})
#   %unsqueeze_17 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_3, 2), kwargs = {})
#   %expand_6 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_17, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_14 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_3, [1, 16, 2049, 128]), kwargs = {})
#   %full_default : Tensor "b8[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_4 : Tensor "i64[2049][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (2049,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_2 : Tensor "i64[2049][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_4, 0), kwargs = {})
#   %unsqueeze_7 : Tensor "i64[1, 2049][2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_2, 0), kwargs = {})
#   %unsqueeze_8 : Tensor "i64[1, 1, 2049][2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_7, 1), kwargs = {})
#   %unsqueeze_9 : Tensor "i64[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_8, 2), kwargs = {})
#   %iota_3 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %add_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota_3, 2048), kwargs = {})
#   %unsqueeze_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_1, 0), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, 1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, 3), kwargs = {})
#   %le : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%unsqueeze_9, %unsqueeze_6), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %convert_element_type : Tensor "b8[1, 2049][2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg3_1, torch.bool), kwargs = {})
#   %iota_1 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_1, 1), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_1, 2), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, 3), kwargs = {})
#   %index : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%convert_element_type, [%unsqueeze_3, %unsqueeze_9]), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %index), kwargs = {})
#   %expand : Tensor "b8[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=28] = call_function[target=torch.ops.aten.expand.default](args = (%bitwise_and_1, [1, -1, 1, 2049]), kwargs = {})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[1, 1, 1, 2056][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, 7], 0.0), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 1, 1, 2049][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, 2049), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 16, 1, 2049][2056, 0, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_5, [1, 16, 1, 2049]), kwargs = {})
#   %_scaled_dot_product_efficient_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_6, %view_13, %view_14, %expand_7, False), kwargs = {scale: 0.08838834764831845})
#   %view_25 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_7, [1, 1, 2048]), kwargs = {})
#   %view_26 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%view_25, [1, 1, -1, 128]), kwargs = {})
#   %convert_element_type_32 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_26, torch.float32), kwargs = {})
#   %pow_6 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_32, 2), kwargs = {})
#   %mean_5 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
#   %add_13 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_5, 1e-06), kwargs = {})
#   %rsqrt_5 : Tensor "f32[1, 1, 16, 1][16, 16, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_18 : Tensor "f32[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_32, %rsqrt_5), kwargs = {})
#   %convert_element_type_33 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.bfloat16), kwargs = {})
#   %mul_19 : Tensor "bf16[1, 1, 16, 128][2048, 2048, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg19_1, %convert_element_type_33), kwargs = {})
#   %permute_13 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%mul_19, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_18 : Tensor "bf16[1, 1, 1, 128][128, 128, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_22 : Tensor "bf16[1, 16, 1, 128][2048, 128, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_13, %unsqueeze_18), kwargs = {})
#   %add_15 : Tensor "bf16[1, 16, 1, 128][2048, 128, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %mul_23), kwargs = {})
#   %unsqueeze_20 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_6, 2), kwargs = {})
#   %expand_8 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_20, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_4 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_8,), kwargs = {memory_format: torch.contiguous_format})
#   %view_33 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [1, 16, 2049, 128]), kwargs = {})
#   %unsqueeze_21 : Tensor "bf16[1, 8, 1, 2049, 128][2098176, 262272, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%cat_7, 2), kwargs = {})
#   %expand_9 : Tensor "bf16[1, 8, 2, 2049, 128][2098176, 262272, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_21, [1, 8, 2, 2049, 128]), kwargs = {})
#   %clone_5 : Tensor "bf16[1, 8, 2, 2049, 128][4196352, 524544, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_34 : Tensor "bf16[1, 16, 2049, 128][4196352, 262272, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_5, [1, 16, 2049, 128]), kwargs = {})
#   %full_default_4 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "bf16[1, 1, 1, 2049][2049, 2049, 2049, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_4, %full_default_3), kwargs = {})
#   %constant_pad_nd_1 : Tensor "bf16[1, 1, 1, 2056][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where_1, [0, 7], 0.0), kwargs = {})
#   %slice_10 : Tensor "bf16[1, 1, 1, 2049][2056, 2056, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd_1, -1, 0, 2049), kwargs = {})
#   %expand_10 : Tensor "bf16[1, 16, 1, 2049][2056, 0, 2056, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%slice_10, [1, 16, 1, 2049]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_1 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](args = (%add_15, %view_33, %view_34, %expand_10, False), kwargs = {scale: 0.08838834764831845})
#   return %buf16,%buf45
triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32784}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2049
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2049, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 2048, tl.int64)
    tmp7 = tmp5 <= tmp6
    tmp8 = tl.full([1], True, tl.int1)
    tmp9 = tmp8 & tmp7
    tmp10 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = (tmp10 != 0)
    tmp12 = tmp9 & tmp11
    tmp13 = tl.full([1], 0.0, tl.float32)
    tmp14 = tl.full([1], float("-inf"), tl.float32)
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
    tmp17 = tl.where(tmp4, tmp15, tmp16)
    tmp18 = tmp0 >= tmp3
    tmp19 = tl.full([1], 2056, tl.int64)
    tmp20 = tmp0 < tmp19
    tmp21 = tl.full([1], 0.0, tl.float32)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp18, tmp21, tmp22)
    tmp24 = tl.where(tmp4, tmp17, tmp23)
    tl.store(out_ptr0 + (x0), tmp24, xmask)
    tl.store(out_ptr1 + (x0), tmp24, xmask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/63/c63m6yfrasepwyu64qpzoitdxhla5mfaayvmsjtlbdp4vlxpv3cy.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_9, hidden_states_10, pow_4, variance_3, add_9, rsqrt_3, hidden_states_11, to_13, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_9 => add_9
#   attn_output_3 => view_17
#   hidden_states_10 => convert_element_type_18
#   hidden_states_11 => mul_13
#   hidden_states_12 => mul_14
#   hidden_states_9 => add_8
#   inputs_embeds => embedding
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
#   to_13 => convert_element_type_19
#   variance_3 => mean_3
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_3 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %arg13_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg13_1]
#   %buf23 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf23]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 2048]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %convert_element_type_18 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %pow_4 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_18, 2), kwargs = {})
#   %mean_3 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
#   %add_9 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-06), kwargs = {})
#   %rsqrt_3 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_13 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %rsqrt_3), kwargs = {})
#   %convert_element_type_19 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13, torch.bfloat16), kwargs = {})
#   %mul_14 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg13_1, %convert_element_type_19), kwargs = {})
#   return %buf23,%mul_14
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 16384}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(r0_mask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp16 = tl.load(in_ptr0 + (0))
    tmp17 = tl.broadcast_to(tmp16, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp15 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.full([1, 1], 151936, tl.int32)
        tmp19 = tmp17 + tmp18
        tmp20 = tmp17 < 0
        tmp21 = tl.where(tmp20, tmp19, tmp17)
        tl.device_assert((0 <= tmp21) & (tmp21 < 151936), "index out of bounds: 0 <= tmp21 < 151936")
        tmp23 = tl.load(in_ptr1 + (r0_0 + 2048*tmp21), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tl.full([1, 1], 2048.0, tl.float32)
        tmp28 = (tmp13 / tmp27)
        tmp29 = tl.full([1, 1], 1e-06, tl.float32)
        tmp30 = tmp28 + tmp29
        tmp31 = libdevice.rsqrt(tmp30)
        tmp32 = tmp26 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp15 * tmp33
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp34, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/u4/cu4aduude7gb4lbu5iyitl2ymdohyjd6plg5uubw56oc5qd7uz3k.py
# Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   linear_4 => view_19
#   linear_5 => view_21
#   mul_14 => mul_15
#   silu => add_10, convert_element_type_22, convert_element_type_23, div, exp, neg_2
# Graph fragment:
#   %mm_4 : Tensor "bf16[1, 6144][6144, 1]cuda:0" = PlaceHolder[target=mm_4]
#   %mm_5 : Tensor "bf16[1, 6144][6144, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %view_19 : Tensor "bf16[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_4, [1, 1, 6144]), kwargs = {})
#   %convert_element_type_22 : Tensor "f32[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_19, torch.float32), kwargs = {})
#   %neg_2 : Tensor "f32[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_22,), kwargs = {})
#   %exp : Tensor "f32[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_2,), kwargs = {})
#   %add_10 : Tensor "f32[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %div : Tensor "f32[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convert_element_type_22, %add_10), kwargs = {})
#   %convert_element_type_23 : Tensor "bf16[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.bfloat16), kwargs = {})
#   %view_21 : Tensor "bf16[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [1, 1, 6144]), kwargs = {})
#   %mul_15 : Tensor "bf16[1, 1, 6144][6144, 6144, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_23, %view_21), kwargs = {})
#   return %mul_15
triton_poi_fused__unsafe_view_mul_silu_8 = async_compile.triton('triton_poi_fused__unsafe_view_mul_silu_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_mul_silu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 49152}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_mul_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/zd/czdclx5gyj5cvyzatu4g6erc5vjxdjc6tp2rehxyxhhpi3mk7tl7.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_9, down_proj, hidden_states_13, hidden_states_14, pow_5, variance_4, add_11, rsqrt_4, hidden_states_15, to_15, hidden_states_16], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_11 => add_12
#   attn_output_3 => view_17
#   down_proj => view_23
#   hidden_states_13 => add_11
#   hidden_states_14 => convert_element_type_28
#   hidden_states_15 => mul_16
#   hidden_states_16 => mul_17
#   hidden_states_9 => add_8
#   inputs_embeds => embedding
#   pow_5 => pow_5
#   rsqrt_4 => rsqrt_4
#   to_15 => convert_element_type_29
#   variance_4 => mean_4
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_3 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_6 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %arg17_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg17_1]
#   %buf29 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf29]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 2048]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 1, 2048]), kwargs = {})
#   %add_11 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_23), kwargs = {})
#   %convert_element_type_28 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float32), kwargs = {})
#   %pow_5 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_28, 2), kwargs = {})
#   %mean_4 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [-1], True), kwargs = {})
#   %add_12 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-06), kwargs = {})
#   %rsqrt_4 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_12,), kwargs = {})
#   %mul_16 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_28, %rsqrt_4), kwargs = {})
#   %convert_element_type_29 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
#   %mul_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg17_1, %convert_element_type_29), kwargs = {})
#   return %buf29,%mul_17
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 20480}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp18 = tl.load(in_ptr0 + (0))
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp17 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.full([1, 1], 151936, tl.int32)
        tmp21 = tmp19 + tmp20
        tmp22 = tmp19 < 0
        tmp23 = tl.where(tmp22, tmp21, tmp19)
        tl.device_assert((0 <= tmp23) & (tmp23 < 151936), "index out of bounds: 0 <= tmp23 < 151936")
        tmp25 = tl.load(in_ptr1 + (r0_0 + 2048*tmp23), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp27 = tmp25 + tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tl.full([1, 1], 2048.0, tl.float32)
        tmp32 = (tmp15 / tmp31)
        tmp33 = tl.full([1, 1], 1e-06, tl.float32)
        tmp34 = tmp32 + tmp33
        tmp35 = libdevice.rsqrt(tmp34)
        tmp36 = tmp30 * tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp17 * tmp37
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp38, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/74/c74fc46efeqmthjdth5ekxk6ftcjykulvcpldlqrcnzme7c743dx.py
# Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_9, down_proj, hidden_states_13, attn_output_7, hidden_states_23, hidden_states_24, pow_8, variance_7, add_17, rsqrt_7, hidden_states_25, to_21, hidden_states_26], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_17 => add_18
#   attn_output_3 => view_17
#   attn_output_7 => view_37
#   down_proj => view_23
#   hidden_states_13 => add_11
#   hidden_states_23 => add_17
#   hidden_states_24 => convert_element_type_42
#   hidden_states_25 => mul_26
#   hidden_states_26 => mul_27
#   hidden_states_9 => add_8
#   inputs_embeds => embedding
#   pow_8 => pow_8
#   rsqrt_7 => rsqrt_7
#   to_21 => convert_element_type_43
#   variance_7 => mean_7
# Graph fragment:
#   %arg0_1 : Tensor "i64[1, 1][1, 1]cuda:0" = PlaceHolder[target=arg0_1]
#   %arg1_1 : Tensor "bf16[151936, 2048][2048, 1]cuda:0" = PlaceHolder[target=arg1_1]
#   %mm_3 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_6 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %mm_10 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_10]
#   %add_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_17]
#   %arg26_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg26_1]
#   %buf53 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf53]
#   %embedding : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %arg0_1), kwargs = {})
#   %view_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, 1, 2048]), kwargs = {})
#   %add_8 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %view_17), kwargs = {})
#   %view_23 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, 1, 2048]), kwargs = {})
#   %add_11 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_23), kwargs = {})
#   %view_37 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_10, [1, 1, 2048]), kwargs = {})
#   %add_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %view_37), kwargs = {})
#   %convert_element_type_42 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.float32), kwargs = {})
#   %pow_8 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_42, 2), kwargs = {})
#   %mean_7 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_8, [-1], True), kwargs = {})
#   %add_18 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_7, 1e-06), kwargs = {})
#   %rsqrt_7 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_18,), kwargs = {})
#   %mul_26 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_42, %rsqrt_7), kwargs = {})
#   %convert_element_type_43 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
#   %mul_27 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg26_1, %convert_element_type_43), kwargs = {})
#   return %add_17,%buf53,%mul_27
triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 32768}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], 151936, tl.int32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp1 < 0
        tmp5 = tl.where(tmp4, tmp3, tmp1)
        tl.device_assert((0 <= tmp5) & (tmp5 < 151936), "index out of bounds: 0 <= tmp5 < 151936")
        tmp7 = tl.load(in_ptr1 + (r0_0 + 2048*tmp5), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tmp7 + tmp8
        tmp11 = tmp9 + tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp13, r0_mask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp19 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tl.full([1, 1], 2048.0, tl.float32)
        tmp23 = (tmp17 / tmp22)
        tmp24 = tl.full([1, 1], 1e-06, tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp19 * tmp28
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp29, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/va/cva5kvjv3zadnuf3i2lq4powwp3saxfj2rlo6xrlfr4bikmfywsv.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, hidden_states_28, pow_9, variance_8, add_19, rsqrt_8, hidden_states_29, to_23, hidden_states_30], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_19 => add_21
#   down_proj_1 => view_43
#   hidden_states_27 => add_20
#   hidden_states_28 => convert_element_type_52
#   hidden_states_29 => mul_29
#   hidden_states_30 => mul_30
#   pow_9 => pow_9
#   rsqrt_8 => rsqrt_8
#   to_23 => convert_element_type_53
#   variance_8 => mean_8
# Graph fragment:
#   %add_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_17]
#   %mm_13 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %arg30_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg30_1]
#   %buf59 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf59]
#   %view_43 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 1, 2048]), kwargs = {})
#   %add_20 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_43), kwargs = {})
#   %convert_element_type_52 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_20, torch.float32), kwargs = {})
#   %pow_9 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_52, 2), kwargs = {})
#   %mean_8 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [-1], True), kwargs = {})
#   %add_21 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_8, 1e-06), kwargs = {})
#   %rsqrt_8 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_21,), kwargs = {})
#   %mul_29 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %rsqrt_8), kwargs = {})
#   %convert_element_type_53 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_29, torch.bfloat16), kwargs = {})
#   %mul_30 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg30_1, %convert_element_type_53), kwargs = {})
#   return %buf59,%mul_30
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 20480}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 2048.0, tl.float32)
        tmp14 = (tmp6 / tmp13)
        tmp15 = tl.full([1, 1], 1e-06, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp8 * tmp19
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp20, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/ly/clyctsostykdxpugwy5k4l7tancwqv64f5wqqkbubptkhz23zacy.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, attn_output_11, hidden_states_37, hidden_states_38, pow_12, variance_11, add_25, rsqrt_11, hidden_states_39, to_29, hidden_states_40], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_25 => add_27
#   attn_output_11 => view_57
#   down_proj_1 => view_43
#   hidden_states_27 => add_20
#   hidden_states_37 => add_26
#   hidden_states_38 => convert_element_type_66
#   hidden_states_39 => mul_39
#   hidden_states_40 => mul_40
#   pow_12 => pow_12
#   rsqrt_11 => rsqrt_11
#   to_29 => convert_element_type_67
#   variance_11 => mean_11
# Graph fragment:
#   %add_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_17]
#   %mm_13 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %mm_17 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %arg39_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg39_1]
#   %buf82 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf82]
#   %view_43 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 1, 2048]), kwargs = {})
#   %add_20 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_43), kwargs = {})
#   %view_57 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, 1, 2048]), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_57), kwargs = {})
#   %convert_element_type_66 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_26, torch.float32), kwargs = {})
#   %pow_12 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_66, 2), kwargs = {})
#   %mean_11 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_12, [-1], True), kwargs = {})
#   %add_27 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_11, 1e-06), kwargs = {})
#   %rsqrt_11 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_27,), kwargs = {})
#   %mul_39 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_66, %rsqrt_11), kwargs = {})
#   %convert_element_type_67 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_39, torch.bfloat16), kwargs = {})
#   %mul_40 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg39_1, %convert_element_type_67), kwargs = {})
#   return %buf82,%mul_40
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 24576}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp10 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 + tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full([1, 1], 2048.0, tl.float32)
        tmp18 = (tmp8 / tmp17)
        tmp19 = tl.full([1, 1], 1e-06, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp10 * tmp23
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp24, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/va/cvawcle63y25p4fpzgxvbwluz2jmzyppli2av4o4xmbobqukvmgl.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, attn_output_11, hidden_states_37, down_proj_2, hidden_states_41, hidden_states_42, pow_13, variance_12, add_27, rsqrt_12, hidden_states_43, to_31, hidden_states_44], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_27 => add_30
#   attn_output_11 => view_57
#   down_proj_1 => view_43
#   down_proj_2 => view_63
#   hidden_states_27 => add_20
#   hidden_states_37 => add_26
#   hidden_states_41 => add_29
#   hidden_states_42 => convert_element_type_76
#   hidden_states_43 => mul_42
#   hidden_states_44 => mul_43
#   pow_13 => pow_13
#   rsqrt_12 => rsqrt_12
#   to_31 => convert_element_type_77
#   variance_12 => mean_12
# Graph fragment:
#   %add_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_17]
#   %mm_13 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %mm_17 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %mm_20 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %arg43_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg43_1]
#   %buf88 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf88]
#   %view_43 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 1, 2048]), kwargs = {})
#   %add_20 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_43), kwargs = {})
#   %view_57 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, 1, 2048]), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_57), kwargs = {})
#   %view_63 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [1, 1, 2048]), kwargs = {})
#   %add_29 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_63), kwargs = {})
#   %convert_element_type_76 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.float32), kwargs = {})
#   %pow_13 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_76, 2), kwargs = {})
#   %mean_12 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [-1], True), kwargs = {})
#   %add_30 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_12, 1e-06), kwargs = {})
#   %rsqrt_12 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_30,), kwargs = {})
#   %mul_42 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_76, %rsqrt_12), kwargs = {})
#   %convert_element_type_77 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_42, torch.bfloat16), kwargs = {})
#   %mul_43 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg43_1, %convert_element_type_77), kwargs = {})
#   return %buf88,%mul_43
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 28672}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp12 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp14 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.full([1, 1], 2048.0, tl.float32)
        tmp22 = (tmp10 / tmp21)
        tmp23 = tl.full([1, 1], 1e-06, tl.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp12 * tmp27
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp28, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/pl/cplz76nkw7r4yfbfv3f24fjq5hna2jbszuu7ttot5aup5edbvmqt.py
# Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, attn_output_11, hidden_states_37, down_proj_2, hidden_states_41, attn_output_15, hidden_states_51, hidden_states_52, pow_16, variance_15, add_33, rsqrt_15, hidden_states_53, to_37, hidden_states_54], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_33 => add_36
#   attn_output_11 => view_57
#   attn_output_15 => view_77
#   down_proj_1 => view_43
#   down_proj_2 => view_63
#   hidden_states_27 => add_20
#   hidden_states_37 => add_26
#   hidden_states_41 => add_29
#   hidden_states_51 => add_35
#   hidden_states_52 => convert_element_type_90
#   hidden_states_53 => mul_52
#   hidden_states_54 => mul_53
#   pow_16 => pow_16
#   rsqrt_15 => rsqrt_15
#   to_37 => convert_element_type_91
#   variance_15 => mean_15
# Graph fragment:
#   %add_17 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_17]
#   %mm_13 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_13]
#   %mm_17 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %mm_20 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_20]
#   %mm_24 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_24]
#   %add_35 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_35]
#   %arg52_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg52_1]
#   %buf112 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf112]
#   %view_43 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_13, [1, 1, 2048]), kwargs = {})
#   %add_20 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %view_43), kwargs = {})
#   %view_57 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, 1, 2048]), kwargs = {})
#   %add_26 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %view_57), kwargs = {})
#   %view_63 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_20, [1, 1, 2048]), kwargs = {})
#   %add_29 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %view_63), kwargs = {})
#   %view_77 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_24, [1, 1, 2048]), kwargs = {})
#   %add_35 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %view_77), kwargs = {})
#   %convert_element_type_90 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_35, torch.float32), kwargs = {})
#   %pow_16 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_90, 2), kwargs = {})
#   %mean_15 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_16, [-1], True), kwargs = {})
#   %add_36 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_15, 1e-06), kwargs = {})
#   %rsqrt_15 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %mul_52 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_90, %rsqrt_15), kwargs = {})
#   %convert_element_type_91 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_52, torch.bfloat16), kwargs = {})
#   %mul_53 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg52_1, %convert_element_type_91), kwargs = {})
#   return %add_35,%buf112,%mul_53
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 40960}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask, tmp13, _tmp12)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp8, r0_mask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp14 = tl.load(in_ptr4 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tl.full([1, 1], 2048.0, tl.float32)
        tmp18 = (tmp12 / tmp17)
        tmp19 = tl.full([1, 1], 1e-06, tl.float32)
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp14 * tmp23
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp24, r0_mask)
''', device_str='cuda')


# kernel path: /home/ianbarber/Projects/KernelBench/extract/inductor_debug/cache/3k/c3kna6wymkldichn77k4d5ebw2eqv76u4fecfnllnf4voqs6oztl.py
# Topologically Sorted Source Nodes: [down_proj_27, hidden_states_391, hidden_states_392, pow_113, variance_112, add_227, rsqrt_112, hidden_states_393, to_231, hidden_states_394], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_227 => add_255
#   down_proj_27 => view_563
#   hidden_states_391 => add_254
#   hidden_states_392 => convert_element_type_676
#   hidden_states_393 => mul_367
#   hidden_states_394 => mul_368
#   pow_113 => pow_113
#   rsqrt_112 => rsqrt_112
#   to_231 => convert_element_type_677
#   variance_112 => mean_112
# Graph fragment:
#   %add_251 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0" = PlaceHolder[target=add_251]
#   %mm_195 : Tensor "bf16[1, 2048][2048, 1]cuda:0" = PlaceHolder[target=mm_195]
#   %arg368_1 : Tensor "bf16[2048][1]cuda:0" = PlaceHolder[target=arg368_1]
#   %buf826 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0" = PlaceHolder[target=buf826]
#   %view_563 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_195, [1, 1, 2048]), kwargs = {})
#   %add_254 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_251, %view_563), kwargs = {})
#   %convert_element_type_676 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_254, torch.float32), kwargs = {})
#   %pow_113 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_676, 2), kwargs = {})
#   %mean_112 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_113, [-1], True), kwargs = {})
#   %add_255 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_112, 1e-06), kwargs = {})
#   %rsqrt_112 : Tensor "f32[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_255,), kwargs = {})
#   %mul_367 : Tensor "f32[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_676, %rsqrt_112), kwargs = {})
#   %convert_element_type_677 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_367, torch.bfloat16), kwargs = {})
#   %mul_368 : Tensor "bf16[1, 1, 2048][2048, 2048, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg368_1, %convert_element_type_677), kwargs = {})
#   return %buf826,%mul_368
triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=121, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'backend_hash': 'F26D6A7434BEBEE3558734F2AEA99A4FDF9682E2121B0D40DD978F26BD1DF547', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'r0_': 20480}}
)
@triton.jit
def triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2048
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp8 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_out_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.full([1, 1], 2048.0, tl.float32)
        tmp14 = (tmp6 / tmp13)
        tmp15 = tl.full([1, 1], 1e-06, tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp8 * tmp19
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp20, r0_mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1 = args
        args.clear()
        assert_size_stride(arg0_1, (1, 1), (1, 1))
        assert_size_stride(arg1_1, (151936, 2048), (2048, 1))
        assert_size_stride(arg5_1, (2048, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            arg0_1 = copy_misaligned(arg0_1)
            buf1 = empty_strided_cuda((1, 1, 2048), (2048, 2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_3, rsqrt, hidden_states_1, to_7, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0:1
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0.run(arg0_1, arg1_1, arg5_1, buf1, 1, 2048, stream=stream0)
            del arg5_1
            assert_size_stride(arg6_1, (2048, 2048), (2048, 1))
            buf2 = empty_strided_cuda((1, 2048), (2048, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_3, rsqrt, hidden_states_1, to_7, hidden_states_2, linear], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:2
            extern_kernels.mm(reinterpret_tensor(buf1, (1, 2048), (0, 1), 0), reinterpret_tensor(arg6_1, (2048, 2048), (1, 2048), 0), out=buf2)
            del arg6_1
            assert_size_stride(arg7_1, (128, ), (1, ))
            assert_size_stride(arg4_1, (64, ), (1, ))
            buf4 = empty_strided_cuda((1, 16, 1, 128), (2048, 128, 2048, 1), torch.bfloat16)
            buf13 = reinterpret_tensor(buf4, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, x2, neg, x1, cat_1, sin, sin_1, sin_2, sin_3, mul_9, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.slice, aten.neg, aten.sin, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:3
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf13, buf2, arg7_1, arg4_1, 16, 128, stream=stream0)
            del arg7_1
            assert_size_stride(arg8_1, (1024, 2048), (2048, 1))
            buf5 = empty_strided_cuda((1, 1024), (1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:4
            extern_kernels.mm(reinterpret_tensor(buf1, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg8_1, (2048, 1024), (1, 2048), 0), out=buf5)
            del arg8_1
            assert_size_stride(arg9_1, (128, ), (1, ))
            buf10 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf9 = reinterpret_tensor(buf10, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, sin, sin_1, sin_2, sin_3, linear_1, view_1, hidden_states_5, pow_3, variance_2, add_5, rsqrt_2, hidden_states_6, to_11, mul_7, key_states, mul_10, x2_1, neg_1, x1_1, cat_2, mul_11, k_embed], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:5
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf5, arg9_1, arg4_1, buf9, 8, 128, stream=stream0)
            del arg9_1
            assert_size_stride(arg2_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg2_1 = copy_misaligned(arg2_1)
            buf8 = reinterpret_tensor(buf10, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, linear_1, view_1, hidden_states_5, pow_3, variance_2, add_5, rsqrt_2, hidden_states_6, to_11, mul_7, key_states, mul_10, k_embed, keys], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:6
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg2_1, buf8, 2097152, stream=stream0)
            del arg2_1
            assert_size_stride(arg10_1, (1024, 2048), (2048, 1))
            buf11 = buf5; del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:7
            extern_kernels.mm(reinterpret_tensor(buf1, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg10_1, (2048, 1024), (1, 2048), 0), out=buf11)
            del arg10_1
            assert_size_stride(arg11_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg11_1 = copy_misaligned(arg11_1)
            buf12 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2, view_2, value_states, values], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:8
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg11_1, buf11, buf12, 2098176, stream=stream0)
            del arg11_1
            buf14 = empty_strided_cuda((1, 16, 2049, 128), (4196352, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:9
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf10, buf14, 4196352, stream=stream0)
            buf15 = empty_strided_cuda((1, 16, 2049, 128), (4196352, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:10
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf12, buf15, 4196352, stream=stream0)
            assert_size_stride(arg3_1, (1, 2049), (2049, 1))
            arg3_1 = copy_misaligned(arg3_1)
            buf16 = empty_strided_cuda((1, 1, 1, 2049), (2056, 0, 2056, 1), torch.bfloat16)
            buf45 = empty_strided_cuda((1, 1, 1, 2049), (2056, 0, 2056, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output, linear_7, view_3, hidden_states_17, pow_6, variance_5, add_12, rsqrt_5, hidden_states_18, to_17, mul_18, query_states_1, cos_4, mul_21, q_embed_1, getitem_17, hidden_states_21, key_1, getitem_18, hidden_states_22, value_1, attn_output_4], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:11
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf16, buf45, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [linear, view, hidden_states_3, pow_2, variance_1, add_4, rsqrt_1, hidden_states_4, to_9, mul_5, query_states, getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_3, mul_8, q_embed, getitem_11, hidden_states_7, key, getitem_12, hidden_states_8, value, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, attn_output], Original ATen: [aten._unsafe_view, aten.view, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.transpose, aten.unsqueeze, aten.expand, aten.arange, aten.bmm, aten.cat, aten.cos, aten.clone, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:12
            buf17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf13, buf14, buf15, reinterpret_tensor(buf16, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf18 = buf17[0]
            assert_size_stride(buf18, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf18, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf17
            assert_size_stride(arg12_1, (2048, 2048), (2048, 1))
            buf22 = reinterpret_tensor(buf13, (1, 2048), (2048, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, attn_output_3], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:13
            extern_kernels.mm(reinterpret_tensor(buf18, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg12_1, (2048, 2048), (1, 2048), 0), out=buf22)
            del arg12_1
            assert_size_stride(arg13_1, (2048, ), (1, ))
            buf24 = reinterpret_tensor(buf18, (1, 1, 2048), (2048, 2048, 1), 0); del buf18  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_9, hidden_states_10, pow_4, variance_3, add_9, rsqrt_3, hidden_states_11, to_13, hidden_states_12], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7:14
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7.run(arg0_1, arg1_1, buf22, arg13_1, buf24, 1, 2048, stream=stream0)
            del arg13_1
            assert_size_stride(arg14_1, (6144, 2048), (2048, 1))
            buf25 = empty_strided_cuda((1, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:15
            extern_kernels.mm(reinterpret_tensor(buf24, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg14_1, (2048, 6144), (1, 2048), 0), out=buf25)
            del arg14_1
            assert_size_stride(arg15_1, (6144, 2048), (2048, 1))
            buf26 = empty_strided_cuda((1, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:16
            extern_kernels.mm(reinterpret_tensor(buf24, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg15_1, (2048, 6144), (1, 2048), 0), out=buf26)
            del arg15_1
            buf27 = reinterpret_tensor(buf25, (1, 1, 6144), (6144, 6144, 1), 0); del buf25  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:17
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf27, buf26, 6144, stream=stream0)
            assert_size_stride(arg16_1, (2048, 6144), (6144, 1))
            buf28 = reinterpret_tensor(buf24, (1, 2048), (2048, 1), 0); del buf24  # reuse
            # Topologically Sorted Source Nodes: [linear_4, silu, linear_5, mul_14, down_proj], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:18
            extern_kernels.mm(reinterpret_tensor(buf27, (1, 6144), (0, 1), 0), reinterpret_tensor(arg16_1, (6144, 2048), (1, 6144), 0), out=buf28)
            del arg16_1
            assert_size_stride(arg17_1, (2048, ), (1, ))
            buf30 = buf1; del buf1  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_9, down_proj, hidden_states_13, hidden_states_14, pow_5, variance_4, add_11, rsqrt_4, hidden_states_15, to_15, hidden_states_16], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9:19
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9.run(arg0_1, arg1_1, buf22, buf28, arg17_1, buf30, 1, 2048, stream=stream0)
            del arg17_1
            assert_size_stride(arg18_1, (2048, 2048), (2048, 1))
            buf31 = buf2; del buf2  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:20
            extern_kernels.mm(reinterpret_tensor(buf30, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg18_1, (2048, 2048), (1, 2048), 0), out=buf31)
            del arg18_1
            assert_size_stride(arg19_1, (128, ), (1, ))
            buf33 = empty_strided_cuda((1, 16, 1, 128), (2048, 128, 2048, 1), torch.bfloat16)
            buf42 = reinterpret_tensor(buf33, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf33  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_7, view_3, hidden_states_17, pow_6, variance_5, add_12, rsqrt_5, hidden_states_18, to_17, mul_18, query_states_1, cos_4, mul_21, x2_2, neg_2, x1_2, cat_5, sin_4, mul_22, q_embed_1, getitem_17, hidden_states_21, key_1, getitem_18, hidden_states_22, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:21
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf42, buf31, arg19_1, arg4_1, 16, 128, stream=stream0)
            del arg19_1
            assert_size_stride(arg20_1, (1024, 2048), (2048, 1))
            buf34 = buf11; del buf11  # reuse
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:22
            extern_kernels.mm(reinterpret_tensor(buf30, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg20_1, (2048, 1024), (1, 2048), 0), out=buf34)
            del arg20_1
            assert_size_stride(arg21_1, (128, ), (1, ))
            buf39 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf38 = reinterpret_tensor(buf39, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_4, sin_4, linear_8, view_4, hidden_states_19, pow_7, variance_6, add_13, rsqrt_6, hidden_states_20, to_19, mul_20, key_states_1, mul_23, x2_3, neg_3, x1_3, cat_6, mul_24, k_embed_1], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:23
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf34, arg21_1, arg4_1, buf38, 8, 128, stream=stream0)
            del arg21_1
            assert_size_stride(arg23_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg23_1 = copy_misaligned(arg23_1)
            buf37 = reinterpret_tensor(buf39, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_4, linear_8, view_4, hidden_states_19, pow_7, variance_6, add_13, rsqrt_6, hidden_states_20, to_19, mul_20, key_states_1, mul_23, k_embed_1, keys_1], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:24
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg23_1, buf37, 2097152, stream=stream0)
            del arg23_1
            assert_size_stride(arg22_1, (1024, 2048), (2048, 1))
            buf40 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:25
            extern_kernels.mm(reinterpret_tensor(buf30, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg22_1, (2048, 1024), (1, 2048), 0), out=buf40)
            del arg22_1
            assert_size_stride(arg24_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg24_1 = copy_misaligned(arg24_1)
            buf41 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_9, view_5, value_states_1, values_1], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:26
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg24_1, buf40, buf41, 2098176, stream=stream0)
            del arg24_1
            buf43 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_7, view_3, hidden_states_17, pow_6, variance_5, add_12, rsqrt_5, hidden_states_18, to_17, mul_18, query_states_1, cos_4, mul_21, q_embed_1, getitem_17, hidden_states_21, key_1, getitem_18, hidden_states_22, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:27
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf39, buf43, 4196352, stream=stream0)
            buf44 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_7, view_3, hidden_states_17, pow_6, variance_5, add_12, rsqrt_5, hidden_states_18, to_17, mul_18, query_states_1, cos_4, mul_21, q_embed_1, getitem_17, hidden_states_21, key_1, getitem_18, hidden_states_22, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:28
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf41, buf44, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_7, view_3, hidden_states_17, pow_6, variance_5, add_12, rsqrt_5, hidden_states_18, to_17, mul_18, query_states_1, cos_4, mul_21, q_embed_1, getitem_17, hidden_states_21, key_1, getitem_18, hidden_states_22, value_1, attn_output_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:29
            buf46 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf42, buf43, buf44, reinterpret_tensor(buf45, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf47 = buf46[0]
            assert_size_stride(buf47, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf47, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf46
            assert_size_stride(arg25_1, (2048, 2048), (2048, 1))
            buf51 = reinterpret_tensor(buf42, (1, 2048), (2048, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [transpose_8, reshape_5, attn_output_7], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:30
            extern_kernels.mm(reinterpret_tensor(buf47, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg25_1, (2048, 2048), (1, 2048), 0), out=buf51)
            del arg25_1
            assert_size_stride(arg26_1, (2048, ), (1, ))
            buf52 = reinterpret_tensor(buf22, (1, 1, 2048), (2048, 2048, 1), 0); del buf22  # reuse
            buf54 = reinterpret_tensor(buf47, (1, 1, 2048), (2048, 2048, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [inputs_embeds, attn_output_3, hidden_states_9, down_proj, hidden_states_13, attn_output_7, hidden_states_23, hidden_states_24, pow_8, variance_7, add_17, rsqrt_7, hidden_states_25, to_21, hidden_states_26], Original ATen: [aten.embedding, aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10:31
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10.run(buf52, arg0_1, arg1_1, buf28, buf51, arg26_1, buf54, 1, 2048, stream=stream0)
            del arg0_1
            del arg26_1
            assert_size_stride(arg27_1, (6144, 2048), (2048, 1))
            buf55 = reinterpret_tensor(buf27, (1, 6144), (6144, 1), 0); del buf27  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_24, pow_8, variance_7, add_17, rsqrt_7, hidden_states_25, to_21, hidden_states_26, linear_11], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:32
            extern_kernels.mm(reinterpret_tensor(buf54, (1, 2048), (0, 1), 0), reinterpret_tensor(arg27_1, (2048, 6144), (1, 2048), 0), out=buf55)
            del arg27_1
            assert_size_stride(arg28_1, (6144, 2048), (2048, 1))
            buf56 = buf26; del buf26  # reuse
            # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:33
            extern_kernels.mm(reinterpret_tensor(buf54, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg28_1, (2048, 6144), (1, 2048), 0), out=buf56)
            del arg28_1
            buf57 = reinterpret_tensor(buf55, (1, 1, 6144), (6144, 6144, 1), 0); del buf55  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_27], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:34
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf57, buf56, 6144, stream=stream0)
            assert_size_stride(arg29_1, (2048, 6144), (6144, 1))
            buf58 = reinterpret_tensor(buf54, (1, 2048), (2048, 1), 0); del buf54  # reuse
            # Topologically Sorted Source Nodes: [linear_11, silu_1, linear_12, mul_27, down_proj_1], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:35
            extern_kernels.mm(reinterpret_tensor(buf57, (1, 6144), (0, 1), 0), reinterpret_tensor(arg29_1, (6144, 2048), (1, 6144), 0), out=buf58)
            del arg29_1
            assert_size_stride(arg30_1, (2048, ), (1, ))
            buf60 = reinterpret_tensor(buf51, (1, 1, 2048), (2048, 2048, 1), 0); del buf51  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, hidden_states_28, pow_9, variance_8, add_19, rsqrt_8, hidden_states_29, to_23, hidden_states_30], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:36
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf52, buf58, arg30_1, buf60, 1, 2048, stream=stream0)
            del arg30_1
            assert_size_stride(arg31_1, (2048, 2048), (2048, 1))
            buf61 = buf28; del buf28  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, hidden_states_28, pow_9, variance_8, add_19, rsqrt_8, hidden_states_29, to_23, hidden_states_30, linear_14], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:37
            extern_kernels.mm(reinterpret_tensor(buf60, (1, 2048), (0, 1), 0), reinterpret_tensor(arg31_1, (2048, 2048), (1, 2048), 0), out=buf61)
            del arg31_1
            assert_size_stride(arg32_1, (128, ), (1, ))
            buf63 = reinterpret_tensor(buf30, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf30  # reuse
            buf72 = reinterpret_tensor(buf63, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_14, view_6, hidden_states_31, pow_10, variance_9, add_20, rsqrt_9, hidden_states_32, to_25, mul_31, query_states_2, cos_5, mul_34, x2_4, neg_4, x1_4, cat_9, sin_5, mul_35, q_embed_2, getitem_23, hidden_states_35, key_2, getitem_24, hidden_states_36, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:38
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf72, buf61, arg32_1, arg4_1, 16, 128, stream=stream0)
            del arg32_1
            assert_size_stride(arg33_1, (1024, 2048), (2048, 1))
            buf64 = buf40; del buf40  # reuse
            # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:39
            extern_kernels.mm(reinterpret_tensor(buf60, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 1024), (1, 2048), 0), out=buf64)
            del arg33_1
            assert_size_stride(arg34_1, (128, ), (1, ))
            buf69 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf68 = reinterpret_tensor(buf69, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_5, sin_5, linear_15, view_7, hidden_states_33, pow_11, variance_10, add_21, rsqrt_10, hidden_states_34, to_27, mul_33, key_states_2, mul_36, x2_5, neg_5, x1_5, cat_10, mul_37, k_embed_2], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:40
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf64, arg34_1, arg4_1, buf68, 8, 128, stream=stream0)
            del arg34_1
            assert_size_stride(arg36_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg36_1 = copy_misaligned(arg36_1)
            buf67 = reinterpret_tensor(buf69, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_5, linear_15, view_7, hidden_states_33, pow_11, variance_10, add_21, rsqrt_10, hidden_states_34, to_27, mul_33, key_states_2, mul_36, k_embed_2, keys_2], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:41
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg36_1, buf67, 2097152, stream=stream0)
            del arg36_1
            assert_size_stride(arg35_1, (1024, 2048), (2048, 1))
            buf70 = buf64; del buf64  # reuse
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:42
            extern_kernels.mm(reinterpret_tensor(buf60, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg35_1, (2048, 1024), (1, 2048), 0), out=buf70)
            del arg35_1
            assert_size_stride(arg37_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg37_1 = copy_misaligned(arg37_1)
            buf71 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_16, view_8, value_states_2, values_2], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:43
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg37_1, buf70, buf71, 2098176, stream=stream0)
            del arg37_1
            buf73 = buf44; del buf44  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_14, view_6, hidden_states_31, pow_10, variance_9, add_20, rsqrt_9, hidden_states_32, to_25, mul_31, query_states_2, cos_5, mul_34, q_embed_2, getitem_23, hidden_states_35, key_2, getitem_24, hidden_states_36, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:44
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf69, buf73, 4196352, stream=stream0)
            buf74 = buf43; del buf43  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_14, view_6, hidden_states_31, pow_10, variance_9, add_20, rsqrt_9, hidden_states_32, to_25, mul_31, query_states_2, cos_5, mul_34, q_embed_2, getitem_23, hidden_states_35, key_2, getitem_24, hidden_states_36, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:45
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf71, buf74, 4196352, stream=stream0)
            buf75 = buf45; del buf45  # reuse
            buf104 = buf16; del buf16  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_14, view_6, hidden_states_31, pow_10, variance_9, add_20, rsqrt_9, hidden_states_32, to_25, mul_31, query_states_2, cos_5, mul_34, q_embed_2, getitem_23, hidden_states_35, key_2, getitem_24, hidden_states_36, value_2, attn_output_8, linear_21, view_9, hidden_states_45, pow_14, variance_13, add_28, rsqrt_13, hidden_states_46, to_33, mul_44, query_states_3, cos_6, mul_47, q_embed_3, getitem_29, hidden_states_49, key_3, getitem_30, hidden_states_50, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:46
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf75, buf104, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_14, view_6, hidden_states_31, pow_10, variance_9, add_20, rsqrt_9, hidden_states_32, to_25, mul_31, query_states_2, cos_5, mul_34, q_embed_2, getitem_23, hidden_states_35, key_2, getitem_24, hidden_states_36, value_2, attn_output_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:47
            buf76 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf72, buf73, buf74, reinterpret_tensor(buf75, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf77 = buf76[0]
            assert_size_stride(buf77, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf77, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf76
            assert_size_stride(arg38_1, (2048, 2048), (2048, 1))
            buf81 = reinterpret_tensor(buf72, (1, 2048), (2048, 1), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [transpose_12, reshape_8, attn_output_11], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:48
            extern_kernels.mm(reinterpret_tensor(buf77, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg38_1, (2048, 2048), (1, 2048), 0), out=buf81)
            del arg38_1
            assert_size_stride(arg39_1, (2048, ), (1, ))
            buf83 = reinterpret_tensor(buf77, (1, 1, 2048), (2048, 2048, 1), 0); del buf77  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, attn_output_11, hidden_states_37, hidden_states_38, pow_12, variance_11, add_25, rsqrt_11, hidden_states_39, to_29, hidden_states_40], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:49
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf52, buf58, buf81, arg39_1, buf83, 1, 2048, stream=stream0)
            del arg39_1
            assert_size_stride(arg40_1, (6144, 2048), (2048, 1))
            buf84 = reinterpret_tensor(buf57, (1, 6144), (6144, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:50
            extern_kernels.mm(reinterpret_tensor(buf83, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg40_1, (2048, 6144), (1, 2048), 0), out=buf84)
            del arg40_1
            assert_size_stride(arg41_1, (6144, 2048), (2048, 1))
            buf85 = buf56; del buf56  # reuse
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:51
            extern_kernels.mm(reinterpret_tensor(buf83, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg41_1, (2048, 6144), (1, 2048), 0), out=buf85)
            del arg41_1
            buf86 = reinterpret_tensor(buf84, (1, 1, 6144), (6144, 6144, 1), 0); del buf84  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_40], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:52
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf86, buf85, 6144, stream=stream0)
            assert_size_stride(arg42_1, (2048, 6144), (6144, 1))
            buf87 = reinterpret_tensor(buf83, (1, 2048), (2048, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [linear_18, silu_2, linear_19, mul_40, down_proj_2], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:53
            extern_kernels.mm(reinterpret_tensor(buf86, (1, 6144), (0, 1), 0), reinterpret_tensor(arg42_1, (6144, 2048), (1, 6144), 0), out=buf87)
            del arg42_1
            assert_size_stride(arg43_1, (2048, ), (1, ))
            buf89 = buf60; del buf60  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, attn_output_11, hidden_states_37, down_proj_2, hidden_states_41, hidden_states_42, pow_13, variance_12, add_27, rsqrt_12, hidden_states_43, to_31, hidden_states_44], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:54
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf52, buf58, buf81, buf87, arg43_1, buf89, 1, 2048, stream=stream0)
            del arg43_1
            assert_size_stride(arg44_1, (2048, 2048), (2048, 1))
            buf90 = buf61; del buf61  # reuse
            # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:55
            extern_kernels.mm(reinterpret_tensor(buf89, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 2048), (1, 2048), 0), out=buf90)
            del arg44_1
            assert_size_stride(arg45_1, (128, ), (1, ))
            buf92 = reinterpret_tensor(buf31, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf31  # reuse
            buf101 = reinterpret_tensor(buf92, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf92  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_21, view_9, hidden_states_45, pow_14, variance_13, add_28, rsqrt_13, hidden_states_46, to_33, mul_44, query_states_3, cos_6, mul_47, x2_6, neg_6, x1_6, cat_13, sin_6, mul_48, q_embed_3, getitem_29, hidden_states_49, key_3, getitem_30, hidden_states_50, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:56
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf101, buf90, arg45_1, arg4_1, 16, 128, stream=stream0)
            del arg45_1
            del buf90
            assert_size_stride(arg46_1, (1024, 2048), (2048, 1))
            buf93 = buf70; del buf70  # reuse
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:57
            extern_kernels.mm(reinterpret_tensor(buf89, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg46_1, (2048, 1024), (1, 2048), 0), out=buf93)
            del arg46_1
            assert_size_stride(arg47_1, (128, ), (1, ))
            buf98 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf97 = reinterpret_tensor(buf98, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_6, sin_6, linear_22, view_10, hidden_states_47, pow_15, variance_14, add_29, rsqrt_14, hidden_states_48, to_35, mul_46, key_states_3, mul_49, x2_7, neg_7, x1_7, cat_14, mul_50, k_embed_3], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:58
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf93, arg47_1, arg4_1, buf97, 8, 128, stream=stream0)
            del arg47_1
            assert_size_stride(arg49_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg49_1 = copy_misaligned(arg49_1)
            buf96 = reinterpret_tensor(buf98, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_6, linear_22, view_10, hidden_states_47, pow_15, variance_14, add_29, rsqrt_14, hidden_states_48, to_35, mul_46, key_states_3, mul_49, k_embed_3, keys_3], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:59
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg49_1, buf96, 2097152, stream=stream0)
            del arg49_1
            assert_size_stride(arg48_1, (1024, 2048), (2048, 1))
            buf99 = buf93; del buf93  # reuse
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:60
            extern_kernels.mm(reinterpret_tensor(buf89, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg48_1, (2048, 1024), (1, 2048), 0), out=buf99)
            del arg48_1
            del buf89
            assert_size_stride(arg50_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg50_1 = copy_misaligned(arg50_1)
            buf100 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_23, view_11, value_states_3, values_3], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:61
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg50_1, buf99, buf100, 2098176, stream=stream0)
            del arg50_1
            buf102 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_21, view_9, hidden_states_45, pow_14, variance_13, add_28, rsqrt_13, hidden_states_46, to_33, mul_44, query_states_3, cos_6, mul_47, q_embed_3, getitem_29, hidden_states_49, key_3, getitem_30, hidden_states_50, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:62
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf98, buf102, 4196352, stream=stream0)
            buf103 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_21, view_9, hidden_states_45, pow_14, variance_13, add_28, rsqrt_13, hidden_states_46, to_33, mul_44, query_states_3, cos_6, mul_47, q_embed_3, getitem_29, hidden_states_49, key_3, getitem_30, hidden_states_50, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:63
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf100, buf103, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_21, view_9, hidden_states_45, pow_14, variance_13, add_28, rsqrt_13, hidden_states_46, to_33, mul_44, query_states_3, cos_6, mul_47, q_embed_3, getitem_29, hidden_states_49, key_3, getitem_30, hidden_states_50, value_3, attn_output_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:64
            buf105 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf101, buf102, buf103, reinterpret_tensor(buf104, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf106 = buf105[0]
            assert_size_stride(buf106, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf106, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf105
            assert_size_stride(arg51_1, (2048, 2048), (2048, 1))
            buf110 = reinterpret_tensor(buf101, (1, 2048), (2048, 1), 0); del buf101  # reuse
            # Topologically Sorted Source Nodes: [transpose_16, reshape_11, attn_output_15], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:65
            extern_kernels.mm(reinterpret_tensor(buf106, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg51_1, (2048, 2048), (1, 2048), 0), out=buf110)
            del arg51_1
            assert_size_stride(arg52_1, (2048, ), (1, ))
            buf111 = buf52; del buf52  # reuse
            buf113 = reinterpret_tensor(buf106, (1, 1, 2048), (2048, 2048, 1), 0); del buf106  # reuse
            # Topologically Sorted Source Nodes: [down_proj_1, hidden_states_27, attn_output_11, hidden_states_37, down_proj_2, hidden_states_41, attn_output_15, hidden_states_51, hidden_states_52, pow_16, variance_15, add_33, rsqrt_15, hidden_states_53, to_37, hidden_states_54], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:66
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf111, buf58, buf81, buf87, buf110, arg52_1, buf113, 1, 2048, stream=stream0)
            del arg52_1
            assert_size_stride(arg53_1, (6144, 2048), (2048, 1))
            buf114 = reinterpret_tensor(buf86, (1, 6144), (6144, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_52, pow_16, variance_15, add_33, rsqrt_15, hidden_states_53, to_37, hidden_states_54, linear_25], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:67
            extern_kernels.mm(reinterpret_tensor(buf113, (1, 2048), (0, 1), 0), reinterpret_tensor(arg53_1, (2048, 6144), (1, 2048), 0), out=buf114)
            del arg53_1
            assert_size_stride(arg54_1, (6144, 2048), (2048, 1))
            buf115 = buf85; del buf85  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:68
            extern_kernels.mm(reinterpret_tensor(buf113, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg54_1, (2048, 6144), (1, 2048), 0), out=buf115)
            del arg54_1
            buf116 = reinterpret_tensor(buf114, (1, 1, 6144), (6144, 6144, 1), 0); del buf114  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_53], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:69
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf116, buf115, 6144, stream=stream0)
            assert_size_stride(arg55_1, (2048, 6144), (6144, 1))
            buf117 = reinterpret_tensor(buf113, (1, 2048), (2048, 1), 0); del buf113  # reuse
            # Topologically Sorted Source Nodes: [linear_25, silu_3, linear_26, mul_53, down_proj_3], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:70
            extern_kernels.mm(reinterpret_tensor(buf116, (1, 6144), (0, 1), 0), reinterpret_tensor(arg55_1, (6144, 2048), (1, 6144), 0), out=buf117)
            del arg55_1
            assert_size_stride(arg56_1, (2048, ), (1, ))
            buf119 = reinterpret_tensor(buf87, (1, 1, 2048), (2048, 2048, 1), 0); del buf87  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_55, hidden_states_56, pow_17, variance_16, add_35, rsqrt_16, hidden_states_57, to_39, hidden_states_58], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:71
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf111, buf117, arg56_1, buf119, 1, 2048, stream=stream0)
            del arg56_1
            assert_size_stride(arg57_1, (2048, 2048), (2048, 1))
            buf120 = buf81; del buf81  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_55, hidden_states_56, pow_17, variance_16, add_35, rsqrt_16, hidden_states_57, to_39, hidden_states_58, linear_28], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:72
            extern_kernels.mm(reinterpret_tensor(buf119, (1, 2048), (0, 1), 0), reinterpret_tensor(arg57_1, (2048, 2048), (1, 2048), 0), out=buf120)
            del arg57_1
            assert_size_stride(arg58_1, (128, ), (1, ))
            buf122 = reinterpret_tensor(buf58, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf58  # reuse
            buf131 = reinterpret_tensor(buf122, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf122  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_28, view_12, hidden_states_59, pow_18, variance_17, add_36, rsqrt_17, hidden_states_60, to_41, mul_57, query_states_4, cos_7, mul_60, x2_8, neg_8, x1_8, cat_17, sin_7, mul_61, q_embed_4, getitem_35, hidden_states_63, key_4, getitem_36, hidden_states_64, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:73
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf131, buf120, arg58_1, arg4_1, 16, 128, stream=stream0)
            del arg58_1
            assert_size_stride(arg59_1, (1024, 2048), (2048, 1))
            buf123 = buf99; del buf99  # reuse
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:74
            extern_kernels.mm(reinterpret_tensor(buf119, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg59_1, (2048, 1024), (1, 2048), 0), out=buf123)
            del arg59_1
            assert_size_stride(arg60_1, (128, ), (1, ))
            buf128 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf127 = reinterpret_tensor(buf128, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_7, sin_7, linear_29, view_13, hidden_states_61, pow_19, variance_18, add_37, rsqrt_18, hidden_states_62, to_43, mul_59, key_states_4, mul_62, x2_9, neg_9, x1_9, cat_18, mul_63, k_embed_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:75
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf123, arg60_1, arg4_1, buf127, 8, 128, stream=stream0)
            del arg60_1
            assert_size_stride(arg62_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg62_1 = copy_misaligned(arg62_1)
            buf126 = reinterpret_tensor(buf128, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_7, linear_29, view_13, hidden_states_61, pow_19, variance_18, add_37, rsqrt_18, hidden_states_62, to_43, mul_59, key_states_4, mul_62, k_embed_4, keys_4], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:76
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg62_1, buf126, 2097152, stream=stream0)
            del arg62_1
            assert_size_stride(arg61_1, (1024, 2048), (2048, 1))
            buf129 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:77
            extern_kernels.mm(reinterpret_tensor(buf119, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg61_1, (2048, 1024), (1, 2048), 0), out=buf129)
            del arg61_1
            assert_size_stride(arg63_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg63_1 = copy_misaligned(arg63_1)
            buf130 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_30, view_14, value_states_4, values_4], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:78
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg63_1, buf129, buf130, 2098176, stream=stream0)
            del arg63_1
            buf132 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_28, view_12, hidden_states_59, pow_18, variance_17, add_36, rsqrt_17, hidden_states_60, to_41, mul_57, query_states_4, cos_7, mul_60, q_embed_4, getitem_35, hidden_states_63, key_4, getitem_36, hidden_states_64, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:79
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf128, buf132, 4196352, stream=stream0)
            buf133 = buf102; del buf102  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_28, view_12, hidden_states_59, pow_18, variance_17, add_36, rsqrt_17, hidden_states_60, to_41, mul_57, query_states_4, cos_7, mul_60, q_embed_4, getitem_35, hidden_states_63, key_4, getitem_36, hidden_states_64, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:80
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf130, buf133, 4196352, stream=stream0)
            buf134 = buf104; del buf104  # reuse
            buf163 = buf75; del buf75  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_28, view_12, hidden_states_59, pow_18, variance_17, add_36, rsqrt_17, hidden_states_60, to_41, mul_57, query_states_4, cos_7, mul_60, q_embed_4, getitem_35, hidden_states_63, key_4, getitem_36, hidden_states_64, value_4, attn_output_16, linear_35, view_15, hidden_states_73, pow_22, variance_21, add_44, rsqrt_21, hidden_states_74, to_49, mul_70, query_states_5, cos_8, mul_73, q_embed_5, getitem_41, hidden_states_77, key_5, getitem_42, hidden_states_78, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:81
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf134, buf163, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_28, view_12, hidden_states_59, pow_18, variance_17, add_36, rsqrt_17, hidden_states_60, to_41, mul_57, query_states_4, cos_7, mul_60, q_embed_4, getitem_35, hidden_states_63, key_4, getitem_36, hidden_states_64, value_4, attn_output_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:82
            buf135 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf131, buf132, buf133, reinterpret_tensor(buf134, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf136 = buf135[0]
            assert_size_stride(buf136, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf136, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf135
            assert_size_stride(arg64_1, (2048, 2048), (2048, 1))
            buf140 = reinterpret_tensor(buf131, (1, 2048), (2048, 1), 0); del buf131  # reuse
            # Topologically Sorted Source Nodes: [transpose_20, reshape_14, attn_output_19], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:83
            extern_kernels.mm(reinterpret_tensor(buf136, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg64_1, (2048, 2048), (1, 2048), 0), out=buf140)
            del arg64_1
            assert_size_stride(arg65_1, (2048, ), (1, ))
            buf142 = reinterpret_tensor(buf136, (1, 1, 2048), (2048, 2048, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_55, attn_output_19, hidden_states_65, hidden_states_66, pow_20, variance_19, add_41, rsqrt_19, hidden_states_67, to_45, hidden_states_68], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:84
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf111, buf117, buf140, arg65_1, buf142, 1, 2048, stream=stream0)
            del arg65_1
            assert_size_stride(arg66_1, (6144, 2048), (2048, 1))
            buf143 = reinterpret_tensor(buf116, (1, 6144), (6144, 1), 0); del buf116  # reuse
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:85
            extern_kernels.mm(reinterpret_tensor(buf142, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg66_1, (2048, 6144), (1, 2048), 0), out=buf143)
            del arg66_1
            assert_size_stride(arg67_1, (6144, 2048), (2048, 1))
            buf144 = buf115; del buf115  # reuse
            # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:86
            extern_kernels.mm(reinterpret_tensor(buf142, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg67_1, (2048, 6144), (1, 2048), 0), out=buf144)
            del arg67_1
            buf145 = reinterpret_tensor(buf143, (1, 1, 6144), (6144, 6144, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_66], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:87
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf145, buf144, 6144, stream=stream0)
            assert_size_stride(arg68_1, (2048, 6144), (6144, 1))
            buf146 = reinterpret_tensor(buf142, (1, 2048), (2048, 1), 0); del buf142  # reuse
            # Topologically Sorted Source Nodes: [linear_32, silu_4, linear_33, mul_66, down_proj_4], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:88
            extern_kernels.mm(reinterpret_tensor(buf145, (1, 6144), (0, 1), 0), reinterpret_tensor(arg68_1, (6144, 2048), (1, 6144), 0), out=buf146)
            del arg68_1
            assert_size_stride(arg69_1, (2048, ), (1, ))
            buf148 = buf119; del buf119  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_55, attn_output_19, hidden_states_65, down_proj_4, hidden_states_69, hidden_states_70, pow_21, variance_20, add_43, rsqrt_20, hidden_states_71, to_47, hidden_states_72], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:89
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf111, buf117, buf140, buf146, arg69_1, buf148, 1, 2048, stream=stream0)
            del arg69_1
            assert_size_stride(arg70_1, (2048, 2048), (2048, 1))
            buf149 = buf120; del buf120  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:90
            extern_kernels.mm(reinterpret_tensor(buf148, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg70_1, (2048, 2048), (1, 2048), 0), out=buf149)
            del arg70_1
            assert_size_stride(arg71_1, (128, ), (1, ))
            buf151 = reinterpret_tensor(buf110, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf110  # reuse
            buf160 = reinterpret_tensor(buf151, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf151  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_35, view_15, hidden_states_73, pow_22, variance_21, add_44, rsqrt_21, hidden_states_74, to_49, mul_70, query_states_5, cos_8, mul_73, x2_10, neg_10, x1_10, cat_21, sin_8, mul_74, q_embed_5, getitem_41, hidden_states_77, key_5, getitem_42, hidden_states_78, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:91
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf160, buf149, arg71_1, arg4_1, 16, 128, stream=stream0)
            del arg71_1
            del buf149
            assert_size_stride(arg72_1, (1024, 2048), (2048, 1))
            buf152 = buf129; del buf129  # reuse
            # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:92
            extern_kernels.mm(reinterpret_tensor(buf148, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg72_1, (2048, 1024), (1, 2048), 0), out=buf152)
            del arg72_1
            assert_size_stride(arg73_1, (128, ), (1, ))
            buf157 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf156 = reinterpret_tensor(buf157, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_8, sin_8, linear_36, view_16, hidden_states_75, pow_23, variance_22, add_45, rsqrt_22, hidden_states_76, to_51, mul_72, key_states_5, mul_75, x2_11, neg_11, x1_11, cat_22, mul_76, k_embed_5], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:93
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf152, arg73_1, arg4_1, buf156, 8, 128, stream=stream0)
            del arg73_1
            assert_size_stride(arg75_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg75_1 = copy_misaligned(arg75_1)
            buf155 = reinterpret_tensor(buf157, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_8, linear_36, view_16, hidden_states_75, pow_23, variance_22, add_45, rsqrt_22, hidden_states_76, to_51, mul_72, key_states_5, mul_75, k_embed_5, keys_5], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:94
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg75_1, buf155, 2097152, stream=stream0)
            del arg75_1
            assert_size_stride(arg74_1, (1024, 2048), (2048, 1))
            buf158 = buf152; del buf152  # reuse
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:95
            extern_kernels.mm(reinterpret_tensor(buf148, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg74_1, (2048, 1024), (1, 2048), 0), out=buf158)
            del arg74_1
            del buf148
            assert_size_stride(arg76_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg76_1 = copy_misaligned(arg76_1)
            buf159 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_37, view_17, value_states_5, values_5], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:96
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg76_1, buf158, buf159, 2098176, stream=stream0)
            del arg76_1
            buf161 = buf133; del buf133  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_35, view_15, hidden_states_73, pow_22, variance_21, add_44, rsqrt_21, hidden_states_74, to_49, mul_70, query_states_5, cos_8, mul_73, q_embed_5, getitem_41, hidden_states_77, key_5, getitem_42, hidden_states_78, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:97
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf157, buf161, 4196352, stream=stream0)
            buf162 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_35, view_15, hidden_states_73, pow_22, variance_21, add_44, rsqrt_21, hidden_states_74, to_49, mul_70, query_states_5, cos_8, mul_73, q_embed_5, getitem_41, hidden_states_77, key_5, getitem_42, hidden_states_78, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:98
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf159, buf162, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_35, view_15, hidden_states_73, pow_22, variance_21, add_44, rsqrt_21, hidden_states_74, to_49, mul_70, query_states_5, cos_8, mul_73, q_embed_5, getitem_41, hidden_states_77, key_5, getitem_42, hidden_states_78, value_5, attn_output_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:99
            buf164 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf160, buf161, buf162, reinterpret_tensor(buf163, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf165 = buf164[0]
            assert_size_stride(buf165, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf165, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf164
            assert_size_stride(arg77_1, (2048, 2048), (2048, 1))
            buf169 = reinterpret_tensor(buf160, (1, 2048), (2048, 1), 0); del buf160  # reuse
            # Topologically Sorted Source Nodes: [transpose_24, reshape_17, attn_output_23], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:100
            extern_kernels.mm(reinterpret_tensor(buf165, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg77_1, (2048, 2048), (1, 2048), 0), out=buf169)
            del arg77_1
            assert_size_stride(arg78_1, (2048, ), (1, ))
            buf170 = buf111; del buf111  # reuse
            buf172 = reinterpret_tensor(buf165, (1, 1, 2048), (2048, 2048, 1), 0); del buf165  # reuse
            # Topologically Sorted Source Nodes: [down_proj_3, hidden_states_55, attn_output_19, hidden_states_65, down_proj_4, hidden_states_69, attn_output_23, hidden_states_79, hidden_states_80, pow_24, variance_23, add_49, rsqrt_23, hidden_states_81, to_53, hidden_states_82], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:101
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf170, buf117, buf140, buf146, buf169, arg78_1, buf172, 1, 2048, stream=stream0)
            del arg78_1
            assert_size_stride(arg79_1, (6144, 2048), (2048, 1))
            buf173 = reinterpret_tensor(buf145, (1, 6144), (6144, 1), 0); del buf145  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, pow_24, variance_23, add_49, rsqrt_23, hidden_states_81, to_53, hidden_states_82, linear_39], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:102
            extern_kernels.mm(reinterpret_tensor(buf172, (1, 2048), (0, 1), 0), reinterpret_tensor(arg79_1, (2048, 6144), (1, 2048), 0), out=buf173)
            del arg79_1
            assert_size_stride(arg80_1, (6144, 2048), (2048, 1))
            buf174 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:103
            extern_kernels.mm(reinterpret_tensor(buf172, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg80_1, (2048, 6144), (1, 2048), 0), out=buf174)
            del arg80_1
            buf175 = reinterpret_tensor(buf173, (1, 1, 6144), (6144, 6144, 1), 0); del buf173  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_79], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:104
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf175, buf174, 6144, stream=stream0)
            assert_size_stride(arg81_1, (2048, 6144), (6144, 1))
            buf176 = reinterpret_tensor(buf172, (1, 2048), (2048, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [linear_39, silu_5, linear_40, mul_79, down_proj_5], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:105
            extern_kernels.mm(reinterpret_tensor(buf175, (1, 6144), (0, 1), 0), reinterpret_tensor(arg81_1, (6144, 2048), (1, 6144), 0), out=buf176)
            del arg81_1
            assert_size_stride(arg82_1, (2048, ), (1, ))
            buf178 = reinterpret_tensor(buf169, (1, 1, 2048), (2048, 2048, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_83, hidden_states_84, pow_25, variance_24, add_51, rsqrt_24, hidden_states_85, to_55, hidden_states_86], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:106
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf170, buf176, arg82_1, buf178, 1, 2048, stream=stream0)
            del arg82_1
            assert_size_stride(arg83_1, (2048, 2048), (2048, 1))
            buf179 = buf146; del buf146  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_83, hidden_states_84, pow_25, variance_24, add_51, rsqrt_24, hidden_states_85, to_55, hidden_states_86, linear_42], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:107
            extern_kernels.mm(reinterpret_tensor(buf178, (1, 2048), (0, 1), 0), reinterpret_tensor(arg83_1, (2048, 2048), (1, 2048), 0), out=buf179)
            del arg83_1
            assert_size_stride(arg84_1, (128, ), (1, ))
            buf181 = reinterpret_tensor(buf140, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf140  # reuse
            buf190 = reinterpret_tensor(buf181, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_42, view_18, hidden_states_87, pow_26, variance_25, add_52, rsqrt_25, hidden_states_88, to_57, mul_83, query_states_6, cos_9, mul_86, x2_12, neg_12, x1_12, cat_25, sin_9, mul_87, q_embed_6, getitem_47, hidden_states_91, key_6, getitem_48, hidden_states_92, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:108
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf190, buf179, arg84_1, arg4_1, 16, 128, stream=stream0)
            del arg84_1
            assert_size_stride(arg85_1, (1024, 2048), (2048, 1))
            buf182 = buf158; del buf158  # reuse
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:109
            extern_kernels.mm(reinterpret_tensor(buf178, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg85_1, (2048, 1024), (1, 2048), 0), out=buf182)
            del arg85_1
            assert_size_stride(arg86_1, (128, ), (1, ))
            buf187 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf186 = reinterpret_tensor(buf187, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_9, sin_9, linear_43, view_19, hidden_states_89, pow_27, variance_26, add_53, rsqrt_26, hidden_states_90, to_59, mul_85, key_states_6, mul_88, x2_13, neg_13, x1_13, cat_26, mul_89, k_embed_6], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:110
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf182, arg86_1, arg4_1, buf186, 8, 128, stream=stream0)
            del arg86_1
            assert_size_stride(arg88_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg88_1 = copy_misaligned(arg88_1)
            buf185 = reinterpret_tensor(buf187, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_9, linear_43, view_19, hidden_states_89, pow_27, variance_26, add_53, rsqrt_26, hidden_states_90, to_59, mul_85, key_states_6, mul_88, k_embed_6, keys_6], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:111
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg88_1, buf185, 2097152, stream=stream0)
            del arg88_1
            assert_size_stride(arg87_1, (1024, 2048), (2048, 1))
            buf188 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:112
            extern_kernels.mm(reinterpret_tensor(buf178, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg87_1, (2048, 1024), (1, 2048), 0), out=buf188)
            del arg87_1
            assert_size_stride(arg89_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg89_1 = copy_misaligned(arg89_1)
            buf189 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_44, view_20, value_states_6, values_6], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:113
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg89_1, buf188, buf189, 2098176, stream=stream0)
            del arg89_1
            buf191 = buf162; del buf162  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_42, view_18, hidden_states_87, pow_26, variance_25, add_52, rsqrt_25, hidden_states_88, to_57, mul_83, query_states_6, cos_9, mul_86, q_embed_6, getitem_47, hidden_states_91, key_6, getitem_48, hidden_states_92, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:114
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf187, buf191, 4196352, stream=stream0)
            buf192 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_42, view_18, hidden_states_87, pow_26, variance_25, add_52, rsqrt_25, hidden_states_88, to_57, mul_83, query_states_6, cos_9, mul_86, q_embed_6, getitem_47, hidden_states_91, key_6, getitem_48, hidden_states_92, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:115
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf189, buf192, 4196352, stream=stream0)
            buf193 = buf163; del buf163  # reuse
            buf222 = buf134; del buf134  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_42, view_18, hidden_states_87, pow_26, variance_25, add_52, rsqrt_25, hidden_states_88, to_57, mul_83, query_states_6, cos_9, mul_86, q_embed_6, getitem_47, hidden_states_91, key_6, getitem_48, hidden_states_92, value_6, attn_output_24, linear_49, view_21, hidden_states_101, pow_30, variance_29, add_60, rsqrt_29, hidden_states_102, to_65, mul_96, query_states_7, cos_10, mul_99, q_embed_7, getitem_53, hidden_states_105, key_7, getitem_54, hidden_states_106, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:116
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf193, buf222, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_42, view_18, hidden_states_87, pow_26, variance_25, add_52, rsqrt_25, hidden_states_88, to_57, mul_83, query_states_6, cos_9, mul_86, q_embed_6, getitem_47, hidden_states_91, key_6, getitem_48, hidden_states_92, value_6, attn_output_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:117
            buf194 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf190, buf191, buf192, reinterpret_tensor(buf193, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf195 = buf194[0]
            assert_size_stride(buf195, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf195, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf194
            assert_size_stride(arg90_1, (2048, 2048), (2048, 1))
            buf199 = reinterpret_tensor(buf190, (1, 2048), (2048, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [transpose_28, reshape_20, attn_output_27], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:118
            extern_kernels.mm(reinterpret_tensor(buf195, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 2048), (1, 2048), 0), out=buf199)
            del arg90_1
            assert_size_stride(arg91_1, (2048, ), (1, ))
            buf201 = reinterpret_tensor(buf195, (1, 1, 2048), (2048, 2048, 1), 0); del buf195  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_83, attn_output_27, hidden_states_93, hidden_states_94, pow_28, variance_27, add_57, rsqrt_27, hidden_states_95, to_61, hidden_states_96], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:119
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf170, buf176, buf199, arg91_1, buf201, 1, 2048, stream=stream0)
            del arg91_1
            assert_size_stride(arg92_1, (6144, 2048), (2048, 1))
            buf202 = reinterpret_tensor(buf175, (1, 6144), (6144, 1), 0); del buf175  # reuse
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:120
            extern_kernels.mm(reinterpret_tensor(buf201, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg92_1, (2048, 6144), (1, 2048), 0), out=buf202)
            del arg92_1
            assert_size_stride(arg93_1, (6144, 2048), (2048, 1))
            buf203 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:121
            extern_kernels.mm(reinterpret_tensor(buf201, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg93_1, (2048, 6144), (1, 2048), 0), out=buf203)
            del arg93_1
            buf204 = reinterpret_tensor(buf202, (1, 1, 6144), (6144, 6144, 1), 0); del buf202  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_92], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:122
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf204, buf203, 6144, stream=stream0)
            assert_size_stride(arg94_1, (2048, 6144), (6144, 1))
            buf205 = reinterpret_tensor(buf201, (1, 2048), (2048, 1), 0); del buf201  # reuse
            # Topologically Sorted Source Nodes: [linear_46, silu_6, linear_47, mul_92, down_proj_6], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:123
            extern_kernels.mm(reinterpret_tensor(buf204, (1, 6144), (0, 1), 0), reinterpret_tensor(arg94_1, (6144, 2048), (1, 6144), 0), out=buf205)
            del arg94_1
            assert_size_stride(arg95_1, (2048, ), (1, ))
            buf207 = buf178; del buf178  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_83, attn_output_27, hidden_states_93, down_proj_6, hidden_states_97, hidden_states_98, pow_29, variance_28, add_59, rsqrt_28, hidden_states_99, to_63, hidden_states_100], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:124
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf170, buf176, buf199, buf205, arg95_1, buf207, 1, 2048, stream=stream0)
            del arg95_1
            assert_size_stride(arg96_1, (2048, 2048), (2048, 1))
            buf208 = buf179; del buf179  # reuse
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:125
            extern_kernels.mm(reinterpret_tensor(buf207, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 2048), (1, 2048), 0), out=buf208)
            del arg96_1
            assert_size_stride(arg97_1, (128, ), (1, ))
            buf210 = reinterpret_tensor(buf117, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf117  # reuse
            buf219 = reinterpret_tensor(buf210, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_49, view_21, hidden_states_101, pow_30, variance_29, add_60, rsqrt_29, hidden_states_102, to_65, mul_96, query_states_7, cos_10, mul_99, x2_14, neg_14, x1_14, cat_29, sin_10, mul_100, q_embed_7, getitem_53, hidden_states_105, key_7, getitem_54, hidden_states_106, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:126
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf219, buf208, arg97_1, arg4_1, 16, 128, stream=stream0)
            del arg97_1
            del buf208
            assert_size_stride(arg98_1, (1024, 2048), (2048, 1))
            buf211 = buf188; del buf188  # reuse
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:127
            extern_kernels.mm(reinterpret_tensor(buf207, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg98_1, (2048, 1024), (1, 2048), 0), out=buf211)
            del arg98_1
            assert_size_stride(arg99_1, (128, ), (1, ))
            buf216 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf215 = reinterpret_tensor(buf216, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_10, sin_10, linear_50, view_22, hidden_states_103, pow_31, variance_30, add_61, rsqrt_30, hidden_states_104, to_67, mul_98, key_states_7, mul_101, x2_15, neg_15, x1_15, cat_30, mul_102, k_embed_7], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:128
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf211, arg99_1, arg4_1, buf215, 8, 128, stream=stream0)
            del arg99_1
            assert_size_stride(arg101_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg101_1 = copy_misaligned(arg101_1)
            buf214 = reinterpret_tensor(buf216, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_10, linear_50, view_22, hidden_states_103, pow_31, variance_30, add_61, rsqrt_30, hidden_states_104, to_67, mul_98, key_states_7, mul_101, k_embed_7, keys_7], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:129
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg101_1, buf214, 2097152, stream=stream0)
            del arg101_1
            assert_size_stride(arg100_1, (1024, 2048), (2048, 1))
            buf217 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:130
            extern_kernels.mm(reinterpret_tensor(buf207, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg100_1, (2048, 1024), (1, 2048), 0), out=buf217)
            del arg100_1
            del buf207
            assert_size_stride(arg102_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg102_1 = copy_misaligned(arg102_1)
            buf218 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_51, view_23, value_states_7, values_7], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:131
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg102_1, buf217, buf218, 2098176, stream=stream0)
            del arg102_1
            buf220 = buf192; del buf192  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_49, view_21, hidden_states_101, pow_30, variance_29, add_60, rsqrt_29, hidden_states_102, to_65, mul_96, query_states_7, cos_10, mul_99, q_embed_7, getitem_53, hidden_states_105, key_7, getitem_54, hidden_states_106, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:132
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf216, buf220, 4196352, stream=stream0)
            buf221 = buf191; del buf191  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_49, view_21, hidden_states_101, pow_30, variance_29, add_60, rsqrt_29, hidden_states_102, to_65, mul_96, query_states_7, cos_10, mul_99, q_embed_7, getitem_53, hidden_states_105, key_7, getitem_54, hidden_states_106, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:133
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf218, buf221, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_49, view_21, hidden_states_101, pow_30, variance_29, add_60, rsqrt_29, hidden_states_102, to_65, mul_96, query_states_7, cos_10, mul_99, q_embed_7, getitem_53, hidden_states_105, key_7, getitem_54, hidden_states_106, value_7, attn_output_28], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:134
            buf223 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf219, buf220, buf221, reinterpret_tensor(buf222, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf224 = buf223[0]
            assert_size_stride(buf224, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf224, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf223
            assert_size_stride(arg103_1, (2048, 2048), (2048, 1))
            buf228 = reinterpret_tensor(buf219, (1, 2048), (2048, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [transpose_32, reshape_23, attn_output_31], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:135
            extern_kernels.mm(reinterpret_tensor(buf224, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg103_1, (2048, 2048), (1, 2048), 0), out=buf228)
            del arg103_1
            assert_size_stride(arg104_1, (2048, ), (1, ))
            buf229 = buf170; del buf170  # reuse
            buf231 = reinterpret_tensor(buf224, (1, 1, 2048), (2048, 2048, 1), 0); del buf224  # reuse
            # Topologically Sorted Source Nodes: [down_proj_5, hidden_states_83, attn_output_27, hidden_states_93, down_proj_6, hidden_states_97, attn_output_31, hidden_states_107, hidden_states_108, pow_32, variance_31, add_65, rsqrt_31, hidden_states_109, to_69, hidden_states_110], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:136
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf229, buf176, buf199, buf205, buf228, arg104_1, buf231, 1, 2048, stream=stream0)
            del arg104_1
            assert_size_stride(arg105_1, (6144, 2048), (2048, 1))
            buf232 = reinterpret_tensor(buf204, (1, 6144), (6144, 1), 0); del buf204  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_108, pow_32, variance_31, add_65, rsqrt_31, hidden_states_109, to_69, hidden_states_110, linear_53], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:137
            extern_kernels.mm(reinterpret_tensor(buf231, (1, 2048), (0, 1), 0), reinterpret_tensor(arg105_1, (2048, 6144), (1, 2048), 0), out=buf232)
            del arg105_1
            assert_size_stride(arg106_1, (6144, 2048), (2048, 1))
            buf233 = buf203; del buf203  # reuse
            # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:138
            extern_kernels.mm(reinterpret_tensor(buf231, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg106_1, (2048, 6144), (1, 2048), 0), out=buf233)
            del arg106_1
            buf234 = reinterpret_tensor(buf232, (1, 1, 6144), (6144, 6144, 1), 0); del buf232  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_105], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:139
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf234, buf233, 6144, stream=stream0)
            assert_size_stride(arg107_1, (2048, 6144), (6144, 1))
            buf235 = reinterpret_tensor(buf231, (1, 2048), (2048, 1), 0); del buf231  # reuse
            # Topologically Sorted Source Nodes: [linear_53, silu_7, linear_54, mul_105, down_proj_7], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:140
            extern_kernels.mm(reinterpret_tensor(buf234, (1, 6144), (0, 1), 0), reinterpret_tensor(arg107_1, (6144, 2048), (1, 6144), 0), out=buf235)
            del arg107_1
            assert_size_stride(arg108_1, (2048, ), (1, ))
            buf237 = reinterpret_tensor(buf228, (1, 1, 2048), (2048, 2048, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_111, hidden_states_112, pow_33, variance_32, add_67, rsqrt_32, hidden_states_113, to_71, hidden_states_114], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:141
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf229, buf235, arg108_1, buf237, 1, 2048, stream=stream0)
            del arg108_1
            assert_size_stride(arg109_1, (2048, 2048), (2048, 1))
            buf238 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_111, hidden_states_112, pow_33, variance_32, add_67, rsqrt_32, hidden_states_113, to_71, hidden_states_114, linear_56], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:142
            extern_kernels.mm(reinterpret_tensor(buf237, (1, 2048), (0, 1), 0), reinterpret_tensor(arg109_1, (2048, 2048), (1, 2048), 0), out=buf238)
            del arg109_1
            assert_size_stride(arg110_1, (128, ), (1, ))
            buf240 = reinterpret_tensor(buf199, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf199  # reuse
            buf249 = reinterpret_tensor(buf240, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_56, view_24, hidden_states_115, pow_34, variance_33, add_68, rsqrt_33, hidden_states_116, to_73, mul_109, query_states_8, cos_11, mul_112, x2_16, neg_16, x1_16, cat_33, sin_11, mul_113, q_embed_8, getitem_59, hidden_states_119, key_8, getitem_60, hidden_states_120, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:143
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf249, buf238, arg110_1, arg4_1, 16, 128, stream=stream0)
            del arg110_1
            assert_size_stride(arg111_1, (1024, 2048), (2048, 1))
            buf241 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:144
            extern_kernels.mm(reinterpret_tensor(buf237, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg111_1, (2048, 1024), (1, 2048), 0), out=buf241)
            del arg111_1
            assert_size_stride(arg112_1, (128, ), (1, ))
            buf246 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf245 = reinterpret_tensor(buf246, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_11, sin_11, linear_57, view_25, hidden_states_117, pow_35, variance_34, add_69, rsqrt_34, hidden_states_118, to_75, mul_111, key_states_8, mul_114, x2_17, neg_17, x1_17, cat_34, mul_115, k_embed_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:145
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf241, arg112_1, arg4_1, buf245, 8, 128, stream=stream0)
            del arg112_1
            assert_size_stride(arg114_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg114_1 = copy_misaligned(arg114_1)
            buf244 = reinterpret_tensor(buf246, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_11, linear_57, view_25, hidden_states_117, pow_35, variance_34, add_69, rsqrt_34, hidden_states_118, to_75, mul_111, key_states_8, mul_114, k_embed_8, keys_8], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:146
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg114_1, buf244, 2097152, stream=stream0)
            del arg114_1
            assert_size_stride(arg113_1, (1024, 2048), (2048, 1))
            buf247 = buf241; del buf241  # reuse
            # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:147
            extern_kernels.mm(reinterpret_tensor(buf237, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg113_1, (2048, 1024), (1, 2048), 0), out=buf247)
            del arg113_1
            assert_size_stride(arg115_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg115_1 = copy_misaligned(arg115_1)
            buf248 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_58, view_26, value_states_8, values_8], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:148
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg115_1, buf247, buf248, 2098176, stream=stream0)
            del arg115_1
            buf250 = buf221; del buf221  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_56, view_24, hidden_states_115, pow_34, variance_33, add_68, rsqrt_33, hidden_states_116, to_73, mul_109, query_states_8, cos_11, mul_112, q_embed_8, getitem_59, hidden_states_119, key_8, getitem_60, hidden_states_120, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:149
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf246, buf250, 4196352, stream=stream0)
            buf251 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_56, view_24, hidden_states_115, pow_34, variance_33, add_68, rsqrt_33, hidden_states_116, to_73, mul_109, query_states_8, cos_11, mul_112, q_embed_8, getitem_59, hidden_states_119, key_8, getitem_60, hidden_states_120, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:150
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf248, buf251, 4196352, stream=stream0)
            buf252 = buf222; del buf222  # reuse
            buf281 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_56, view_24, hidden_states_115, pow_34, variance_33, add_68, rsqrt_33, hidden_states_116, to_73, mul_109, query_states_8, cos_11, mul_112, q_embed_8, getitem_59, hidden_states_119, key_8, getitem_60, hidden_states_120, value_8, attn_output_32, linear_63, view_27, hidden_states_129, pow_38, variance_37, add_76, rsqrt_37, hidden_states_130, to_81, mul_122, query_states_9, cos_12, mul_125, q_embed_9, getitem_65, hidden_states_133, key_9, getitem_66, hidden_states_134, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:151
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf252, buf281, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_56, view_24, hidden_states_115, pow_34, variance_33, add_68, rsqrt_33, hidden_states_116, to_73, mul_109, query_states_8, cos_11, mul_112, q_embed_8, getitem_59, hidden_states_119, key_8, getitem_60, hidden_states_120, value_8, attn_output_32], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:152
            buf253 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf249, buf250, buf251, reinterpret_tensor(buf252, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf254 = buf253[0]
            assert_size_stride(buf254, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf254, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf253
            assert_size_stride(arg116_1, (2048, 2048), (2048, 1))
            buf258 = reinterpret_tensor(buf249, (1, 2048), (2048, 1), 0); del buf249  # reuse
            # Topologically Sorted Source Nodes: [transpose_36, reshape_26, attn_output_35], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:153
            extern_kernels.mm(reinterpret_tensor(buf254, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg116_1, (2048, 2048), (1, 2048), 0), out=buf258)
            del arg116_1
            assert_size_stride(arg117_1, (2048, ), (1, ))
            buf260 = reinterpret_tensor(buf254, (1, 1, 2048), (2048, 2048, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_111, attn_output_35, hidden_states_121, hidden_states_122, pow_36, variance_35, add_73, rsqrt_35, hidden_states_123, to_77, hidden_states_124], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:154
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf229, buf235, buf258, arg117_1, buf260, 1, 2048, stream=stream0)
            del arg117_1
            assert_size_stride(arg118_1, (6144, 2048), (2048, 1))
            buf261 = reinterpret_tensor(buf234, (1, 6144), (6144, 1), 0); del buf234  # reuse
            # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:155
            extern_kernels.mm(reinterpret_tensor(buf260, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg118_1, (2048, 6144), (1, 2048), 0), out=buf261)
            del arg118_1
            assert_size_stride(arg119_1, (6144, 2048), (2048, 1))
            buf262 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:156
            extern_kernels.mm(reinterpret_tensor(buf260, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg119_1, (2048, 6144), (1, 2048), 0), out=buf262)
            del arg119_1
            buf263 = reinterpret_tensor(buf261, (1, 1, 6144), (6144, 6144, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_118], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:157
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf263, buf262, 6144, stream=stream0)
            assert_size_stride(arg120_1, (2048, 6144), (6144, 1))
            buf264 = reinterpret_tensor(buf260, (1, 2048), (2048, 1), 0); del buf260  # reuse
            # Topologically Sorted Source Nodes: [linear_60, silu_8, linear_61, mul_118, down_proj_8], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:158
            extern_kernels.mm(reinterpret_tensor(buf263, (1, 6144), (0, 1), 0), reinterpret_tensor(arg120_1, (6144, 2048), (1, 6144), 0), out=buf264)
            del arg120_1
            assert_size_stride(arg121_1, (2048, ), (1, ))
            buf266 = buf237; del buf237  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_111, attn_output_35, hidden_states_121, down_proj_8, hidden_states_125, hidden_states_126, pow_37, variance_36, add_75, rsqrt_36, hidden_states_127, to_79, hidden_states_128], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:159
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf229, buf235, buf258, buf264, arg121_1, buf266, 1, 2048, stream=stream0)
            del arg121_1
            assert_size_stride(arg122_1, (2048, 2048), (2048, 1))
            buf267 = buf238; del buf238  # reuse
            # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:160
            extern_kernels.mm(reinterpret_tensor(buf266, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg122_1, (2048, 2048), (1, 2048), 0), out=buf267)
            del arg122_1
            assert_size_stride(arg123_1, (128, ), (1, ))
            buf269 = reinterpret_tensor(buf176, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf176  # reuse
            buf278 = reinterpret_tensor(buf269, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf269  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_63, view_27, hidden_states_129, pow_38, variance_37, add_76, rsqrt_37, hidden_states_130, to_81, mul_122, query_states_9, cos_12, mul_125, x2_18, neg_18, x1_18, cat_37, sin_12, mul_126, q_embed_9, getitem_65, hidden_states_133, key_9, getitem_66, hidden_states_134, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:161
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf278, buf267, arg123_1, arg4_1, 16, 128, stream=stream0)
            del arg123_1
            del buf267
            assert_size_stride(arg124_1, (1024, 2048), (2048, 1))
            buf270 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:162
            extern_kernels.mm(reinterpret_tensor(buf266, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg124_1, (2048, 1024), (1, 2048), 0), out=buf270)
            del arg124_1
            assert_size_stride(arg125_1, (128, ), (1, ))
            buf275 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf274 = reinterpret_tensor(buf275, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_12, sin_12, linear_64, view_28, hidden_states_131, pow_39, variance_38, add_77, rsqrt_38, hidden_states_132, to_83, mul_124, key_states_9, mul_127, x2_19, neg_19, x1_19, cat_38, mul_128, k_embed_9], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:163
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf270, arg125_1, arg4_1, buf274, 8, 128, stream=stream0)
            del arg125_1
            assert_size_stride(arg127_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg127_1 = copy_misaligned(arg127_1)
            buf273 = reinterpret_tensor(buf275, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_12, linear_64, view_28, hidden_states_131, pow_39, variance_38, add_77, rsqrt_38, hidden_states_132, to_83, mul_124, key_states_9, mul_127, k_embed_9, keys_9], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:164
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg127_1, buf273, 2097152, stream=stream0)
            del arg127_1
            assert_size_stride(arg126_1, (1024, 2048), (2048, 1))
            buf276 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:165
            extern_kernels.mm(reinterpret_tensor(buf266, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg126_1, (2048, 1024), (1, 2048), 0), out=buf276)
            del arg126_1
            del buf266
            assert_size_stride(arg128_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg128_1 = copy_misaligned(arg128_1)
            buf277 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_65, view_29, value_states_9, values_9], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:166
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg128_1, buf276, buf277, 2098176, stream=stream0)
            del arg128_1
            buf279 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_63, view_27, hidden_states_129, pow_38, variance_37, add_76, rsqrt_37, hidden_states_130, to_81, mul_122, query_states_9, cos_12, mul_125, q_embed_9, getitem_65, hidden_states_133, key_9, getitem_66, hidden_states_134, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:167
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf275, buf279, 4196352, stream=stream0)
            buf280 = buf250; del buf250  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_63, view_27, hidden_states_129, pow_38, variance_37, add_76, rsqrt_37, hidden_states_130, to_81, mul_122, query_states_9, cos_12, mul_125, q_embed_9, getitem_65, hidden_states_133, key_9, getitem_66, hidden_states_134, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:168
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf277, buf280, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_63, view_27, hidden_states_129, pow_38, variance_37, add_76, rsqrt_37, hidden_states_130, to_81, mul_122, query_states_9, cos_12, mul_125, q_embed_9, getitem_65, hidden_states_133, key_9, getitem_66, hidden_states_134, value_9, attn_output_36], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:169
            buf282 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf278, buf279, buf280, reinterpret_tensor(buf281, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf283 = buf282[0]
            assert_size_stride(buf283, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf283, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf282
            assert_size_stride(arg129_1, (2048, 2048), (2048, 1))
            buf287 = reinterpret_tensor(buf278, (1, 2048), (2048, 1), 0); del buf278  # reuse
            # Topologically Sorted Source Nodes: [transpose_40, reshape_29, attn_output_39], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:170
            extern_kernels.mm(reinterpret_tensor(buf283, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg129_1, (2048, 2048), (1, 2048), 0), out=buf287)
            del arg129_1
            assert_size_stride(arg130_1, (2048, ), (1, ))
            buf288 = buf229; del buf229  # reuse
            buf290 = reinterpret_tensor(buf283, (1, 1, 2048), (2048, 2048, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [down_proj_7, hidden_states_111, attn_output_35, hidden_states_121, down_proj_8, hidden_states_125, attn_output_39, hidden_states_135, hidden_states_136, pow_40, variance_39, add_81, rsqrt_39, hidden_states_137, to_85, hidden_states_138], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:171
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf288, buf235, buf258, buf264, buf287, arg130_1, buf290, 1, 2048, stream=stream0)
            del arg130_1
            assert_size_stride(arg131_1, (6144, 2048), (2048, 1))
            buf291 = reinterpret_tensor(buf263, (1, 6144), (6144, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_136, pow_40, variance_39, add_81, rsqrt_39, hidden_states_137, to_85, hidden_states_138, linear_67], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:172
            extern_kernels.mm(reinterpret_tensor(buf290, (1, 2048), (0, 1), 0), reinterpret_tensor(arg131_1, (2048, 6144), (1, 2048), 0), out=buf291)
            del arg131_1
            assert_size_stride(arg132_1, (6144, 2048), (2048, 1))
            buf292 = buf262; del buf262  # reuse
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:173
            extern_kernels.mm(reinterpret_tensor(buf290, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg132_1, (2048, 6144), (1, 2048), 0), out=buf292)
            del arg132_1
            buf293 = reinterpret_tensor(buf291, (1, 1, 6144), (6144, 6144, 1), 0); del buf291  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_131], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:174
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf293, buf292, 6144, stream=stream0)
            assert_size_stride(arg133_1, (2048, 6144), (6144, 1))
            buf294 = reinterpret_tensor(buf290, (1, 2048), (2048, 1), 0); del buf290  # reuse
            # Topologically Sorted Source Nodes: [linear_67, silu_9, linear_68, mul_131, down_proj_9], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:175
            extern_kernels.mm(reinterpret_tensor(buf293, (1, 6144), (0, 1), 0), reinterpret_tensor(arg133_1, (6144, 2048), (1, 6144), 0), out=buf294)
            del arg133_1
            assert_size_stride(arg134_1, (2048, ), (1, ))
            buf296 = reinterpret_tensor(buf287, (1, 1, 2048), (2048, 2048, 1), 0); del buf287  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_139, hidden_states_140, pow_41, variance_40, add_83, rsqrt_40, hidden_states_141, to_87, hidden_states_142], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:176
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf288, buf294, arg134_1, buf296, 1, 2048, stream=stream0)
            del arg134_1
            assert_size_stride(arg135_1, (2048, 2048), (2048, 1))
            buf297 = buf264; del buf264  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_139, hidden_states_140, pow_41, variance_40, add_83, rsqrt_40, hidden_states_141, to_87, hidden_states_142, linear_70], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:177
            extern_kernels.mm(reinterpret_tensor(buf296, (1, 2048), (0, 1), 0), reinterpret_tensor(arg135_1, (2048, 2048), (1, 2048), 0), out=buf297)
            del arg135_1
            assert_size_stride(arg136_1, (128, ), (1, ))
            buf299 = reinterpret_tensor(buf258, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf258  # reuse
            buf308 = reinterpret_tensor(buf299, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf299  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_70, view_30, hidden_states_143, pow_42, variance_41, add_84, rsqrt_41, hidden_states_144, to_89, mul_135, query_states_10, cos_13, mul_138, x2_20, neg_20, x1_20, cat_41, sin_13, mul_139, q_embed_10, getitem_71, hidden_states_147, key_10, getitem_72, hidden_states_148, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:178
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf308, buf297, arg136_1, arg4_1, 16, 128, stream=stream0)
            del arg136_1
            assert_size_stride(arg137_1, (1024, 2048), (2048, 1))
            buf300 = buf276; del buf276  # reuse
            # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:179
            extern_kernels.mm(reinterpret_tensor(buf296, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg137_1, (2048, 1024), (1, 2048), 0), out=buf300)
            del arg137_1
            assert_size_stride(arg138_1, (128, ), (1, ))
            buf305 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf304 = reinterpret_tensor(buf305, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_13, sin_13, linear_71, view_31, hidden_states_145, pow_43, variance_42, add_85, rsqrt_42, hidden_states_146, to_91, mul_137, key_states_10, mul_140, x2_21, neg_21, x1_21, cat_42, mul_141, k_embed_10], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:180
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf300, arg138_1, arg4_1, buf304, 8, 128, stream=stream0)
            del arg138_1
            assert_size_stride(arg140_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg140_1 = copy_misaligned(arg140_1)
            buf303 = reinterpret_tensor(buf305, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_13, linear_71, view_31, hidden_states_145, pow_43, variance_42, add_85, rsqrt_42, hidden_states_146, to_91, mul_137, key_states_10, mul_140, k_embed_10, keys_10], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:181
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg140_1, buf303, 2097152, stream=stream0)
            del arg140_1
            assert_size_stride(arg139_1, (1024, 2048), (2048, 1))
            buf306 = buf300; del buf300  # reuse
            # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:182
            extern_kernels.mm(reinterpret_tensor(buf296, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg139_1, (2048, 1024), (1, 2048), 0), out=buf306)
            del arg139_1
            assert_size_stride(arg141_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg141_1 = copy_misaligned(arg141_1)
            buf307 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_72, view_32, value_states_10, values_10], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:183
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg141_1, buf306, buf307, 2098176, stream=stream0)
            del arg141_1
            buf309 = buf280; del buf280  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_70, view_30, hidden_states_143, pow_42, variance_41, add_84, rsqrt_41, hidden_states_144, to_89, mul_135, query_states_10, cos_13, mul_138, q_embed_10, getitem_71, hidden_states_147, key_10, getitem_72, hidden_states_148, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:184
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf305, buf309, 4196352, stream=stream0)
            buf310 = buf279; del buf279  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_70, view_30, hidden_states_143, pow_42, variance_41, add_84, rsqrt_41, hidden_states_144, to_89, mul_135, query_states_10, cos_13, mul_138, q_embed_10, getitem_71, hidden_states_147, key_10, getitem_72, hidden_states_148, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:185
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf307, buf310, 4196352, stream=stream0)
            buf311 = buf281; del buf281  # reuse
            buf340 = buf252; del buf252  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_70, view_30, hidden_states_143, pow_42, variance_41, add_84, rsqrt_41, hidden_states_144, to_89, mul_135, query_states_10, cos_13, mul_138, q_embed_10, getitem_71, hidden_states_147, key_10, getitem_72, hidden_states_148, value_10, attn_output_40, linear_77, view_33, hidden_states_157, pow_46, variance_45, add_92, rsqrt_45, hidden_states_158, to_97, mul_148, query_states_11, cos_14, mul_151, q_embed_11, getitem_77, hidden_states_161, key_11, getitem_78, hidden_states_162, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:186
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf311, buf340, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_70, view_30, hidden_states_143, pow_42, variance_41, add_84, rsqrt_41, hidden_states_144, to_89, mul_135, query_states_10, cos_13, mul_138, q_embed_10, getitem_71, hidden_states_147, key_10, getitem_72, hidden_states_148, value_10, attn_output_40], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:187
            buf312 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf308, buf309, buf310, reinterpret_tensor(buf311, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf313 = buf312[0]
            assert_size_stride(buf313, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf313, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf312
            assert_size_stride(arg142_1, (2048, 2048), (2048, 1))
            buf317 = reinterpret_tensor(buf308, (1, 2048), (2048, 1), 0); del buf308  # reuse
            # Topologically Sorted Source Nodes: [transpose_44, reshape_32, attn_output_43], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:188
            extern_kernels.mm(reinterpret_tensor(buf313, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg142_1, (2048, 2048), (1, 2048), 0), out=buf317)
            del arg142_1
            assert_size_stride(arg143_1, (2048, ), (1, ))
            buf319 = reinterpret_tensor(buf313, (1, 1, 2048), (2048, 2048, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_139, attn_output_43, hidden_states_149, hidden_states_150, pow_44, variance_43, add_89, rsqrt_43, hidden_states_151, to_93, hidden_states_152], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:189
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf288, buf294, buf317, arg143_1, buf319, 1, 2048, stream=stream0)
            del arg143_1
            assert_size_stride(arg144_1, (6144, 2048), (2048, 1))
            buf320 = reinterpret_tensor(buf293, (1, 6144), (6144, 1), 0); del buf293  # reuse
            # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:190
            extern_kernels.mm(reinterpret_tensor(buf319, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg144_1, (2048, 6144), (1, 2048), 0), out=buf320)
            del arg144_1
            assert_size_stride(arg145_1, (6144, 2048), (2048, 1))
            buf321 = buf292; del buf292  # reuse
            # Topologically Sorted Source Nodes: [linear_75], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:191
            extern_kernels.mm(reinterpret_tensor(buf319, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg145_1, (2048, 6144), (1, 2048), 0), out=buf321)
            del arg145_1
            buf322 = reinterpret_tensor(buf320, (1, 1, 6144), (6144, 6144, 1), 0); del buf320  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_144], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:192
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf322, buf321, 6144, stream=stream0)
            assert_size_stride(arg146_1, (2048, 6144), (6144, 1))
            buf323 = reinterpret_tensor(buf319, (1, 2048), (2048, 1), 0); del buf319  # reuse
            # Topologically Sorted Source Nodes: [linear_74, silu_10, linear_75, mul_144, down_proj_10], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:193
            extern_kernels.mm(reinterpret_tensor(buf322, (1, 6144), (0, 1), 0), reinterpret_tensor(arg146_1, (6144, 2048), (1, 6144), 0), out=buf323)
            del arg146_1
            assert_size_stride(arg147_1, (2048, ), (1, ))
            buf325 = buf296; del buf296  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_139, attn_output_43, hidden_states_149, down_proj_10, hidden_states_153, hidden_states_154, pow_45, variance_44, add_91, rsqrt_44, hidden_states_155, to_95, hidden_states_156], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:194
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf288, buf294, buf317, buf323, arg147_1, buf325, 1, 2048, stream=stream0)
            del arg147_1
            assert_size_stride(arg148_1, (2048, 2048), (2048, 1))
            buf326 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:195
            extern_kernels.mm(reinterpret_tensor(buf325, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg148_1, (2048, 2048), (1, 2048), 0), out=buf326)
            del arg148_1
            assert_size_stride(arg149_1, (128, ), (1, ))
            buf328 = reinterpret_tensor(buf235, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf235  # reuse
            buf337 = reinterpret_tensor(buf328, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf328  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_77, view_33, hidden_states_157, pow_46, variance_45, add_92, rsqrt_45, hidden_states_158, to_97, mul_148, query_states_11, cos_14, mul_151, x2_22, neg_22, x1_22, cat_45, sin_14, mul_152, q_embed_11, getitem_77, hidden_states_161, key_11, getitem_78, hidden_states_162, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:196
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf337, buf326, arg149_1, arg4_1, 16, 128, stream=stream0)
            del arg149_1
            del buf326
            assert_size_stride(arg150_1, (1024, 2048), (2048, 1))
            buf329 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:197
            extern_kernels.mm(reinterpret_tensor(buf325, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg150_1, (2048, 1024), (1, 2048), 0), out=buf329)
            del arg150_1
            assert_size_stride(arg151_1, (128, ), (1, ))
            buf334 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf333 = reinterpret_tensor(buf334, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_14, sin_14, linear_78, view_34, hidden_states_159, pow_47, variance_46, add_93, rsqrt_46, hidden_states_160, to_99, mul_150, key_states_11, mul_153, x2_23, neg_23, x1_23, cat_46, mul_154, k_embed_11], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:198
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf329, arg151_1, arg4_1, buf333, 8, 128, stream=stream0)
            del arg151_1
            assert_size_stride(arg153_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg153_1 = copy_misaligned(arg153_1)
            buf332 = reinterpret_tensor(buf334, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_14, linear_78, view_34, hidden_states_159, pow_47, variance_46, add_93, rsqrt_46, hidden_states_160, to_99, mul_150, key_states_11, mul_153, k_embed_11, keys_11], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:199
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg153_1, buf332, 2097152, stream=stream0)
            del arg153_1
            assert_size_stride(arg152_1, (1024, 2048), (2048, 1))
            buf335 = buf329; del buf329  # reuse
            # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:200
            extern_kernels.mm(reinterpret_tensor(buf325, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg152_1, (2048, 1024), (1, 2048), 0), out=buf335)
            del arg152_1
            del buf325
            assert_size_stride(arg154_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg154_1 = copy_misaligned(arg154_1)
            buf336 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_79, view_35, value_states_11, values_11], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:201
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg154_1, buf335, buf336, 2098176, stream=stream0)
            del arg154_1
            buf338 = buf310; del buf310  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_77, view_33, hidden_states_157, pow_46, variance_45, add_92, rsqrt_45, hidden_states_158, to_97, mul_148, query_states_11, cos_14, mul_151, q_embed_11, getitem_77, hidden_states_161, key_11, getitem_78, hidden_states_162, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:202
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf334, buf338, 4196352, stream=stream0)
            buf339 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_77, view_33, hidden_states_157, pow_46, variance_45, add_92, rsqrt_45, hidden_states_158, to_97, mul_148, query_states_11, cos_14, mul_151, q_embed_11, getitem_77, hidden_states_161, key_11, getitem_78, hidden_states_162, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:203
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf336, buf339, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_77, view_33, hidden_states_157, pow_46, variance_45, add_92, rsqrt_45, hidden_states_158, to_97, mul_148, query_states_11, cos_14, mul_151, q_embed_11, getitem_77, hidden_states_161, key_11, getitem_78, hidden_states_162, value_11, attn_output_44], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:204
            buf341 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf337, buf338, buf339, reinterpret_tensor(buf340, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf342 = buf341[0]
            assert_size_stride(buf342, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf342, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf341
            assert_size_stride(arg155_1, (2048, 2048), (2048, 1))
            buf346 = reinterpret_tensor(buf337, (1, 2048), (2048, 1), 0); del buf337  # reuse
            # Topologically Sorted Source Nodes: [transpose_48, reshape_35, attn_output_47], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:205
            extern_kernels.mm(reinterpret_tensor(buf342, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg155_1, (2048, 2048), (1, 2048), 0), out=buf346)
            del arg155_1
            assert_size_stride(arg156_1, (2048, ), (1, ))
            buf347 = buf288; del buf288  # reuse
            buf349 = reinterpret_tensor(buf342, (1, 1, 2048), (2048, 2048, 1), 0); del buf342  # reuse
            # Topologically Sorted Source Nodes: [down_proj_9, hidden_states_139, attn_output_43, hidden_states_149, down_proj_10, hidden_states_153, attn_output_47, hidden_states_163, hidden_states_164, pow_48, variance_47, add_97, rsqrt_47, hidden_states_165, to_101, hidden_states_166], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:206
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf347, buf294, buf317, buf323, buf346, arg156_1, buf349, 1, 2048, stream=stream0)
            del arg156_1
            assert_size_stride(arg157_1, (6144, 2048), (2048, 1))
            buf350 = reinterpret_tensor(buf322, (1, 6144), (6144, 1), 0); del buf322  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_164, pow_48, variance_47, add_97, rsqrt_47, hidden_states_165, to_101, hidden_states_166, linear_81], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:207
            extern_kernels.mm(reinterpret_tensor(buf349, (1, 2048), (0, 1), 0), reinterpret_tensor(arg157_1, (2048, 6144), (1, 2048), 0), out=buf350)
            del arg157_1
            assert_size_stride(arg158_1, (6144, 2048), (2048, 1))
            buf351 = buf321; del buf321  # reuse
            # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:208
            extern_kernels.mm(reinterpret_tensor(buf349, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg158_1, (2048, 6144), (1, 2048), 0), out=buf351)
            del arg158_1
            buf352 = reinterpret_tensor(buf350, (1, 1, 6144), (6144, 6144, 1), 0); del buf350  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_157], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:209
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf352, buf351, 6144, stream=stream0)
            assert_size_stride(arg159_1, (2048, 6144), (6144, 1))
            buf353 = reinterpret_tensor(buf349, (1, 2048), (2048, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [linear_81, silu_11, linear_82, mul_157, down_proj_11], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:210
            extern_kernels.mm(reinterpret_tensor(buf352, (1, 6144), (0, 1), 0), reinterpret_tensor(arg159_1, (6144, 2048), (1, 6144), 0), out=buf353)
            del arg159_1
            assert_size_stride(arg160_1, (2048, ), (1, ))
            buf355 = reinterpret_tensor(buf346, (1, 1, 2048), (2048, 2048, 1), 0); del buf346  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_167, hidden_states_168, pow_49, variance_48, add_99, rsqrt_48, hidden_states_169, to_103, hidden_states_170], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:211
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf347, buf353, arg160_1, buf355, 1, 2048, stream=stream0)
            del arg160_1
            assert_size_stride(arg161_1, (2048, 2048), (2048, 1))
            buf356 = buf323; del buf323  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_167, hidden_states_168, pow_49, variance_48, add_99, rsqrt_48, hidden_states_169, to_103, hidden_states_170, linear_84], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:212
            extern_kernels.mm(reinterpret_tensor(buf355, (1, 2048), (0, 1), 0), reinterpret_tensor(arg161_1, (2048, 2048), (1, 2048), 0), out=buf356)
            del arg161_1
            assert_size_stride(arg162_1, (128, ), (1, ))
            buf358 = reinterpret_tensor(buf317, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf317  # reuse
            buf367 = reinterpret_tensor(buf358, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_84, view_36, hidden_states_171, pow_50, variance_49, add_100, rsqrt_49, hidden_states_172, to_105, mul_161, query_states_12, cos_15, mul_164, x2_24, neg_24, x1_24, cat_49, sin_15, mul_165, q_embed_12, getitem_83, hidden_states_175, key_12, getitem_84, hidden_states_176, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:213
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf367, buf356, arg162_1, arg4_1, 16, 128, stream=stream0)
            del arg162_1
            assert_size_stride(arg163_1, (1024, 2048), (2048, 1))
            buf359 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:214
            extern_kernels.mm(reinterpret_tensor(buf355, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg163_1, (2048, 1024), (1, 2048), 0), out=buf359)
            del arg163_1
            assert_size_stride(arg164_1, (128, ), (1, ))
            buf364 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf363 = reinterpret_tensor(buf364, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_15, sin_15, linear_85, view_37, hidden_states_173, pow_51, variance_50, add_101, rsqrt_50, hidden_states_174, to_107, mul_163, key_states_12, mul_166, x2_25, neg_25, x1_25, cat_50, mul_167, k_embed_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:215
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf359, arg164_1, arg4_1, buf363, 8, 128, stream=stream0)
            del arg164_1
            assert_size_stride(arg166_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg166_1 = copy_misaligned(arg166_1)
            buf362 = reinterpret_tensor(buf364, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_15, linear_85, view_37, hidden_states_173, pow_51, variance_50, add_101, rsqrt_50, hidden_states_174, to_107, mul_163, key_states_12, mul_166, k_embed_12, keys_12], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:216
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg166_1, buf362, 2097152, stream=stream0)
            del arg166_1
            assert_size_stride(arg165_1, (1024, 2048), (2048, 1))
            buf365 = buf359; del buf359  # reuse
            # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:217
            extern_kernels.mm(reinterpret_tensor(buf355, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg165_1, (2048, 1024), (1, 2048), 0), out=buf365)
            del arg165_1
            assert_size_stride(arg167_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg167_1 = copy_misaligned(arg167_1)
            buf366 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_86, view_38, value_states_12, values_12], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:218
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg167_1, buf365, buf366, 2098176, stream=stream0)
            del arg167_1
            buf368 = buf339; del buf339  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_84, view_36, hidden_states_171, pow_50, variance_49, add_100, rsqrt_49, hidden_states_172, to_105, mul_161, query_states_12, cos_15, mul_164, q_embed_12, getitem_83, hidden_states_175, key_12, getitem_84, hidden_states_176, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:219
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf364, buf368, 4196352, stream=stream0)
            buf369 = buf338; del buf338  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_84, view_36, hidden_states_171, pow_50, variance_49, add_100, rsqrt_49, hidden_states_172, to_105, mul_161, query_states_12, cos_15, mul_164, q_embed_12, getitem_83, hidden_states_175, key_12, getitem_84, hidden_states_176, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:220
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf366, buf369, 4196352, stream=stream0)
            buf370 = buf340; del buf340  # reuse
            buf399 = buf311; del buf311  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_84, view_36, hidden_states_171, pow_50, variance_49, add_100, rsqrt_49, hidden_states_172, to_105, mul_161, query_states_12, cos_15, mul_164, q_embed_12, getitem_83, hidden_states_175, key_12, getitem_84, hidden_states_176, value_12, attn_output_48, linear_91, view_39, hidden_states_185, pow_54, variance_53, add_108, rsqrt_53, hidden_states_186, to_113, mul_174, query_states_13, cos_16, mul_177, q_embed_13, getitem_89, hidden_states_189, key_13, getitem_90, hidden_states_190, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:221
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf370, buf399, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_84, view_36, hidden_states_171, pow_50, variance_49, add_100, rsqrt_49, hidden_states_172, to_105, mul_161, query_states_12, cos_15, mul_164, q_embed_12, getitem_83, hidden_states_175, key_12, getitem_84, hidden_states_176, value_12, attn_output_48], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:222
            buf371 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf367, buf368, buf369, reinterpret_tensor(buf370, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf372 = buf371[0]
            assert_size_stride(buf372, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf372, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf371
            assert_size_stride(arg168_1, (2048, 2048), (2048, 1))
            buf376 = reinterpret_tensor(buf367, (1, 2048), (2048, 1), 0); del buf367  # reuse
            # Topologically Sorted Source Nodes: [transpose_52, reshape_38, attn_output_51], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:223
            extern_kernels.mm(reinterpret_tensor(buf372, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg168_1, (2048, 2048), (1, 2048), 0), out=buf376)
            del arg168_1
            assert_size_stride(arg169_1, (2048, ), (1, ))
            buf378 = reinterpret_tensor(buf372, (1, 1, 2048), (2048, 2048, 1), 0); del buf372  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_167, attn_output_51, hidden_states_177, hidden_states_178, pow_52, variance_51, add_105, rsqrt_51, hidden_states_179, to_109, hidden_states_180], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:224
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf347, buf353, buf376, arg169_1, buf378, 1, 2048, stream=stream0)
            del arg169_1
            assert_size_stride(arg170_1, (6144, 2048), (2048, 1))
            buf379 = reinterpret_tensor(buf352, (1, 6144), (6144, 1), 0); del buf352  # reuse
            # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:225
            extern_kernels.mm(reinterpret_tensor(buf378, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg170_1, (2048, 6144), (1, 2048), 0), out=buf379)
            del arg170_1
            assert_size_stride(arg171_1, (6144, 2048), (2048, 1))
            buf380 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:226
            extern_kernels.mm(reinterpret_tensor(buf378, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg171_1, (2048, 6144), (1, 2048), 0), out=buf380)
            del arg171_1
            buf381 = reinterpret_tensor(buf379, (1, 1, 6144), (6144, 6144, 1), 0); del buf379  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_170], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:227
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf381, buf380, 6144, stream=stream0)
            assert_size_stride(arg172_1, (2048, 6144), (6144, 1))
            buf382 = reinterpret_tensor(buf378, (1, 2048), (2048, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [linear_88, silu_12, linear_89, mul_170, down_proj_12], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:228
            extern_kernels.mm(reinterpret_tensor(buf381, (1, 6144), (0, 1), 0), reinterpret_tensor(arg172_1, (6144, 2048), (1, 6144), 0), out=buf382)
            del arg172_1
            assert_size_stride(arg173_1, (2048, ), (1, ))
            buf384 = buf355; del buf355  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_167, attn_output_51, hidden_states_177, down_proj_12, hidden_states_181, hidden_states_182, pow_53, variance_52, add_107, rsqrt_52, hidden_states_183, to_111, hidden_states_184], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:229
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf347, buf353, buf376, buf382, arg173_1, buf384, 1, 2048, stream=stream0)
            del arg173_1
            assert_size_stride(arg174_1, (2048, 2048), (2048, 1))
            buf385 = buf356; del buf356  # reuse
            # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:230
            extern_kernels.mm(reinterpret_tensor(buf384, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg174_1, (2048, 2048), (1, 2048), 0), out=buf385)
            del arg174_1
            assert_size_stride(arg175_1, (128, ), (1, ))
            buf387 = reinterpret_tensor(buf294, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf294  # reuse
            buf396 = reinterpret_tensor(buf387, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_91, view_39, hidden_states_185, pow_54, variance_53, add_108, rsqrt_53, hidden_states_186, to_113, mul_174, query_states_13, cos_16, mul_177, x2_26, neg_26, x1_26, cat_53, sin_16, mul_178, q_embed_13, getitem_89, hidden_states_189, key_13, getitem_90, hidden_states_190, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:231
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf396, buf385, arg175_1, arg4_1, 16, 128, stream=stream0)
            del arg175_1
            del buf385
            assert_size_stride(arg176_1, (1024, 2048), (2048, 1))
            buf388 = buf365; del buf365  # reuse
            # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:232
            extern_kernels.mm(reinterpret_tensor(buf384, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 1024), (1, 2048), 0), out=buf388)
            del arg176_1
            assert_size_stride(arg177_1, (128, ), (1, ))
            buf393 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf392 = reinterpret_tensor(buf393, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_16, sin_16, linear_92, view_40, hidden_states_187, pow_55, variance_54, add_109, rsqrt_54, hidden_states_188, to_115, mul_176, key_states_13, mul_179, x2_27, neg_27, x1_27, cat_54, mul_180, k_embed_13], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:233
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf388, arg177_1, arg4_1, buf392, 8, 128, stream=stream0)
            del arg177_1
            assert_size_stride(arg179_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg179_1 = copy_misaligned(arg179_1)
            buf391 = reinterpret_tensor(buf393, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_16, linear_92, view_40, hidden_states_187, pow_55, variance_54, add_109, rsqrt_54, hidden_states_188, to_115, mul_176, key_states_13, mul_179, k_embed_13, keys_13], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:234
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg179_1, buf391, 2097152, stream=stream0)
            del arg179_1
            assert_size_stride(arg178_1, (1024, 2048), (2048, 1))
            buf394 = buf388; del buf388  # reuse
            # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:235
            extern_kernels.mm(reinterpret_tensor(buf384, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg178_1, (2048, 1024), (1, 2048), 0), out=buf394)
            del arg178_1
            del buf384
            assert_size_stride(arg180_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg180_1 = copy_misaligned(arg180_1)
            buf395 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_93, view_41, value_states_13, values_13], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:236
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg180_1, buf394, buf395, 2098176, stream=stream0)
            del arg180_1
            buf397 = buf369; del buf369  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_91, view_39, hidden_states_185, pow_54, variance_53, add_108, rsqrt_53, hidden_states_186, to_113, mul_174, query_states_13, cos_16, mul_177, q_embed_13, getitem_89, hidden_states_189, key_13, getitem_90, hidden_states_190, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:237
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf393, buf397, 4196352, stream=stream0)
            buf398 = buf368; del buf368  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_91, view_39, hidden_states_185, pow_54, variance_53, add_108, rsqrt_53, hidden_states_186, to_113, mul_174, query_states_13, cos_16, mul_177, q_embed_13, getitem_89, hidden_states_189, key_13, getitem_90, hidden_states_190, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:238
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf395, buf398, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_91, view_39, hidden_states_185, pow_54, variance_53, add_108, rsqrt_53, hidden_states_186, to_113, mul_174, query_states_13, cos_16, mul_177, q_embed_13, getitem_89, hidden_states_189, key_13, getitem_90, hidden_states_190, value_13, attn_output_52], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:239
            buf400 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf396, buf397, buf398, reinterpret_tensor(buf399, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf401 = buf400[0]
            assert_size_stride(buf401, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf401, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf400
            assert_size_stride(arg181_1, (2048, 2048), (2048, 1))
            buf405 = reinterpret_tensor(buf396, (1, 2048), (2048, 1), 0); del buf396  # reuse
            # Topologically Sorted Source Nodes: [transpose_56, reshape_41, attn_output_55], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:240
            extern_kernels.mm(reinterpret_tensor(buf401, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg181_1, (2048, 2048), (1, 2048), 0), out=buf405)
            del arg181_1
            assert_size_stride(arg182_1, (2048, ), (1, ))
            buf406 = buf347; del buf347  # reuse
            buf408 = reinterpret_tensor(buf401, (1, 1, 2048), (2048, 2048, 1), 0); del buf401  # reuse
            # Topologically Sorted Source Nodes: [down_proj_11, hidden_states_167, attn_output_51, hidden_states_177, down_proj_12, hidden_states_181, attn_output_55, hidden_states_191, hidden_states_192, pow_56, variance_55, add_113, rsqrt_55, hidden_states_193, to_117, hidden_states_194], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:241
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf406, buf353, buf376, buf382, buf405, arg182_1, buf408, 1, 2048, stream=stream0)
            del arg182_1
            assert_size_stride(arg183_1, (6144, 2048), (2048, 1))
            buf409 = reinterpret_tensor(buf381, (1, 6144), (6144, 1), 0); del buf381  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_192, pow_56, variance_55, add_113, rsqrt_55, hidden_states_193, to_117, hidden_states_194, linear_95], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:242
            extern_kernels.mm(reinterpret_tensor(buf408, (1, 2048), (0, 1), 0), reinterpret_tensor(arg183_1, (2048, 6144), (1, 2048), 0), out=buf409)
            del arg183_1
            assert_size_stride(arg184_1, (6144, 2048), (2048, 1))
            buf410 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [linear_96], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:243
            extern_kernels.mm(reinterpret_tensor(buf408, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg184_1, (2048, 6144), (1, 2048), 0), out=buf410)
            del arg184_1
            buf411 = reinterpret_tensor(buf409, (1, 1, 6144), (6144, 6144, 1), 0); del buf409  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_183], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:244
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf411, buf410, 6144, stream=stream0)
            assert_size_stride(arg185_1, (2048, 6144), (6144, 1))
            buf412 = reinterpret_tensor(buf408, (1, 2048), (2048, 1), 0); del buf408  # reuse
            # Topologically Sorted Source Nodes: [linear_95, silu_13, linear_96, mul_183, down_proj_13], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:245
            extern_kernels.mm(reinterpret_tensor(buf411, (1, 6144), (0, 1), 0), reinterpret_tensor(arg185_1, (6144, 2048), (1, 6144), 0), out=buf412)
            del arg185_1
            assert_size_stride(arg186_1, (2048, ), (1, ))
            buf414 = reinterpret_tensor(buf405, (1, 1, 2048), (2048, 2048, 1), 0); del buf405  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_195, hidden_states_196, pow_57, variance_56, add_115, rsqrt_56, hidden_states_197, to_119, hidden_states_198], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:246
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf406, buf412, arg186_1, buf414, 1, 2048, stream=stream0)
            del arg186_1
            assert_size_stride(arg187_1, (2048, 2048), (2048, 1))
            buf415 = buf382; del buf382  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_195, hidden_states_196, pow_57, variance_56, add_115, rsqrt_56, hidden_states_197, to_119, hidden_states_198, linear_98], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:247
            extern_kernels.mm(reinterpret_tensor(buf414, (1, 2048), (0, 1), 0), reinterpret_tensor(arg187_1, (2048, 2048), (1, 2048), 0), out=buf415)
            del arg187_1
            assert_size_stride(arg188_1, (128, ), (1, ))
            buf417 = reinterpret_tensor(buf376, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf376  # reuse
            buf426 = reinterpret_tensor(buf417, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf417  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_98, view_42, hidden_states_199, pow_58, variance_57, add_116, rsqrt_57, hidden_states_200, to_121, mul_187, query_states_14, cos_17, mul_190, x2_28, neg_28, x1_28, cat_57, sin_17, mul_191, q_embed_14, getitem_95, hidden_states_203, key_14, getitem_96, hidden_states_204, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:248
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf426, buf415, arg188_1, arg4_1, 16, 128, stream=stream0)
            del arg188_1
            assert_size_stride(arg189_1, (1024, 2048), (2048, 1))
            buf418 = buf394; del buf394  # reuse
            # Topologically Sorted Source Nodes: [linear_99], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:249
            extern_kernels.mm(reinterpret_tensor(buf414, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg189_1, (2048, 1024), (1, 2048), 0), out=buf418)
            del arg189_1
            assert_size_stride(arg190_1, (128, ), (1, ))
            buf423 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf422 = reinterpret_tensor(buf423, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_17, sin_17, linear_99, view_43, hidden_states_201, pow_59, variance_58, add_117, rsqrt_58, hidden_states_202, to_123, mul_189, key_states_14, mul_192, x2_29, neg_29, x1_29, cat_58, mul_193, k_embed_14], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:250
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf418, arg190_1, arg4_1, buf422, 8, 128, stream=stream0)
            del arg190_1
            assert_size_stride(arg192_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg192_1 = copy_misaligned(arg192_1)
            buf421 = reinterpret_tensor(buf423, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_17, linear_99, view_43, hidden_states_201, pow_59, variance_58, add_117, rsqrt_58, hidden_states_202, to_123, mul_189, key_states_14, mul_192, k_embed_14, keys_14], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:251
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg192_1, buf421, 2097152, stream=stream0)
            del arg192_1
            assert_size_stride(arg191_1, (1024, 2048), (2048, 1))
            buf424 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:252
            extern_kernels.mm(reinterpret_tensor(buf414, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg191_1, (2048, 1024), (1, 2048), 0), out=buf424)
            del arg191_1
            assert_size_stride(arg193_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg193_1 = copy_misaligned(arg193_1)
            buf425 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_100, view_44, value_states_14, values_14], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:253
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg193_1, buf424, buf425, 2098176, stream=stream0)
            del arg193_1
            buf427 = buf398; del buf398  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_98, view_42, hidden_states_199, pow_58, variance_57, add_116, rsqrt_57, hidden_states_200, to_121, mul_187, query_states_14, cos_17, mul_190, q_embed_14, getitem_95, hidden_states_203, key_14, getitem_96, hidden_states_204, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:254
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf423, buf427, 4196352, stream=stream0)
            buf428 = buf397; del buf397  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_98, view_42, hidden_states_199, pow_58, variance_57, add_116, rsqrt_57, hidden_states_200, to_121, mul_187, query_states_14, cos_17, mul_190, q_embed_14, getitem_95, hidden_states_203, key_14, getitem_96, hidden_states_204, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:255
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf425, buf428, 4196352, stream=stream0)
            buf429 = buf399; del buf399  # reuse
            buf458 = buf370; del buf370  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_98, view_42, hidden_states_199, pow_58, variance_57, add_116, rsqrt_57, hidden_states_200, to_121, mul_187, query_states_14, cos_17, mul_190, q_embed_14, getitem_95, hidden_states_203, key_14, getitem_96, hidden_states_204, value_14, attn_output_56, linear_105, view_45, hidden_states_213, pow_62, variance_61, add_124, rsqrt_61, hidden_states_214, to_129, mul_200, query_states_15, cos_18, mul_203, q_embed_15, getitem_101, hidden_states_217, key_15, getitem_102, hidden_states_218, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:256
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf429, buf458, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_98, view_42, hidden_states_199, pow_58, variance_57, add_116, rsqrt_57, hidden_states_200, to_121, mul_187, query_states_14, cos_17, mul_190, q_embed_14, getitem_95, hidden_states_203, key_14, getitem_96, hidden_states_204, value_14, attn_output_56], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:257
            buf430 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf426, buf427, buf428, reinterpret_tensor(buf429, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf431 = buf430[0]
            assert_size_stride(buf431, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf431, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf430
            assert_size_stride(arg194_1, (2048, 2048), (2048, 1))
            buf435 = reinterpret_tensor(buf426, (1, 2048), (2048, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [transpose_60, reshape_44, attn_output_59], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:258
            extern_kernels.mm(reinterpret_tensor(buf431, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg194_1, (2048, 2048), (1, 2048), 0), out=buf435)
            del arg194_1
            assert_size_stride(arg195_1, (2048, ), (1, ))
            buf437 = reinterpret_tensor(buf431, (1, 1, 2048), (2048, 2048, 1), 0); del buf431  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_195, attn_output_59, hidden_states_205, hidden_states_206, pow_60, variance_59, add_121, rsqrt_59, hidden_states_207, to_125, hidden_states_208], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:259
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf406, buf412, buf435, arg195_1, buf437, 1, 2048, stream=stream0)
            del arg195_1
            assert_size_stride(arg196_1, (6144, 2048), (2048, 1))
            buf438 = reinterpret_tensor(buf411, (1, 6144), (6144, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [linear_102], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:260
            extern_kernels.mm(reinterpret_tensor(buf437, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg196_1, (2048, 6144), (1, 2048), 0), out=buf438)
            del arg196_1
            assert_size_stride(arg197_1, (6144, 2048), (2048, 1))
            buf439 = buf410; del buf410  # reuse
            # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:261
            extern_kernels.mm(reinterpret_tensor(buf437, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg197_1, (2048, 6144), (1, 2048), 0), out=buf439)
            del arg197_1
            buf440 = reinterpret_tensor(buf438, (1, 1, 6144), (6144, 6144, 1), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_196], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:262
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf440, buf439, 6144, stream=stream0)
            assert_size_stride(arg198_1, (2048, 6144), (6144, 1))
            buf441 = reinterpret_tensor(buf437, (1, 2048), (2048, 1), 0); del buf437  # reuse
            # Topologically Sorted Source Nodes: [linear_102, silu_14, linear_103, mul_196, down_proj_14], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:263
            extern_kernels.mm(reinterpret_tensor(buf440, (1, 6144), (0, 1), 0), reinterpret_tensor(arg198_1, (6144, 2048), (1, 6144), 0), out=buf441)
            del arg198_1
            assert_size_stride(arg199_1, (2048, ), (1, ))
            buf443 = buf414; del buf414  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_195, attn_output_59, hidden_states_205, down_proj_14, hidden_states_209, hidden_states_210, pow_61, variance_60, add_123, rsqrt_60, hidden_states_211, to_127, hidden_states_212], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:264
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf406, buf412, buf435, buf441, arg199_1, buf443, 1, 2048, stream=stream0)
            del arg199_1
            assert_size_stride(arg200_1, (2048, 2048), (2048, 1))
            buf444 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [linear_105], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:265
            extern_kernels.mm(reinterpret_tensor(buf443, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg200_1, (2048, 2048), (1, 2048), 0), out=buf444)
            del arg200_1
            assert_size_stride(arg201_1, (128, ), (1, ))
            buf446 = reinterpret_tensor(buf353, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf353  # reuse
            buf455 = reinterpret_tensor(buf446, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf446  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_105, view_45, hidden_states_213, pow_62, variance_61, add_124, rsqrt_61, hidden_states_214, to_129, mul_200, query_states_15, cos_18, mul_203, x2_30, neg_30, x1_30, cat_61, sin_18, mul_204, q_embed_15, getitem_101, hidden_states_217, key_15, getitem_102, hidden_states_218, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:266
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf455, buf444, arg201_1, arg4_1, 16, 128, stream=stream0)
            del arg201_1
            del buf444
            assert_size_stride(arg202_1, (1024, 2048), (2048, 1))
            buf447 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:267
            extern_kernels.mm(reinterpret_tensor(buf443, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg202_1, (2048, 1024), (1, 2048), 0), out=buf447)
            del arg202_1
            assert_size_stride(arg203_1, (128, ), (1, ))
            buf452 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf451 = reinterpret_tensor(buf452, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_18, sin_18, linear_106, view_46, hidden_states_215, pow_63, variance_62, add_125, rsqrt_62, hidden_states_216, to_131, mul_202, key_states_15, mul_205, x2_31, neg_31, x1_31, cat_62, mul_206, k_embed_15], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:268
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf447, arg203_1, arg4_1, buf451, 8, 128, stream=stream0)
            del arg203_1
            assert_size_stride(arg205_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg205_1 = copy_misaligned(arg205_1)
            buf450 = reinterpret_tensor(buf452, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_18, linear_106, view_46, hidden_states_215, pow_63, variance_62, add_125, rsqrt_62, hidden_states_216, to_131, mul_202, key_states_15, mul_205, k_embed_15, keys_15], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:269
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg205_1, buf450, 2097152, stream=stream0)
            del arg205_1
            assert_size_stride(arg204_1, (1024, 2048), (2048, 1))
            buf453 = buf447; del buf447  # reuse
            # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:270
            extern_kernels.mm(reinterpret_tensor(buf443, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg204_1, (2048, 1024), (1, 2048), 0), out=buf453)
            del arg204_1
            del buf443
            assert_size_stride(arg206_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg206_1 = copy_misaligned(arg206_1)
            buf454 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_107, view_47, value_states_15, values_15], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:271
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg206_1, buf453, buf454, 2098176, stream=stream0)
            del arg206_1
            buf456 = buf428; del buf428  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_105, view_45, hidden_states_213, pow_62, variance_61, add_124, rsqrt_61, hidden_states_214, to_129, mul_200, query_states_15, cos_18, mul_203, q_embed_15, getitem_101, hidden_states_217, key_15, getitem_102, hidden_states_218, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:272
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf452, buf456, 4196352, stream=stream0)
            buf457 = buf427; del buf427  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_105, view_45, hidden_states_213, pow_62, variance_61, add_124, rsqrt_61, hidden_states_214, to_129, mul_200, query_states_15, cos_18, mul_203, q_embed_15, getitem_101, hidden_states_217, key_15, getitem_102, hidden_states_218, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:273
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf454, buf457, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_105, view_45, hidden_states_213, pow_62, variance_61, add_124, rsqrt_61, hidden_states_214, to_129, mul_200, query_states_15, cos_18, mul_203, q_embed_15, getitem_101, hidden_states_217, key_15, getitem_102, hidden_states_218, value_15, attn_output_60], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:274
            buf459 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf455, buf456, buf457, reinterpret_tensor(buf458, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf460 = buf459[0]
            assert_size_stride(buf460, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf460, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf459
            assert_size_stride(arg207_1, (2048, 2048), (2048, 1))
            buf464 = reinterpret_tensor(buf455, (1, 2048), (2048, 1), 0); del buf455  # reuse
            # Topologically Sorted Source Nodes: [transpose_64, reshape_47, attn_output_63], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:275
            extern_kernels.mm(reinterpret_tensor(buf460, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg207_1, (2048, 2048), (1, 2048), 0), out=buf464)
            del arg207_1
            assert_size_stride(arg208_1, (2048, ), (1, ))
            buf465 = buf406; del buf406  # reuse
            buf467 = reinterpret_tensor(buf460, (1, 1, 2048), (2048, 2048, 1), 0); del buf460  # reuse
            # Topologically Sorted Source Nodes: [down_proj_13, hidden_states_195, attn_output_59, hidden_states_205, down_proj_14, hidden_states_209, attn_output_63, hidden_states_219, hidden_states_220, pow_64, variance_63, add_129, rsqrt_63, hidden_states_221, to_133, hidden_states_222], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:276
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf465, buf412, buf435, buf441, buf464, arg208_1, buf467, 1, 2048, stream=stream0)
            del arg208_1
            assert_size_stride(arg209_1, (6144, 2048), (2048, 1))
            buf468 = reinterpret_tensor(buf440, (1, 6144), (6144, 1), 0); del buf440  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_220, pow_64, variance_63, add_129, rsqrt_63, hidden_states_221, to_133, hidden_states_222, linear_109], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:277
            extern_kernels.mm(reinterpret_tensor(buf467, (1, 2048), (0, 1), 0), reinterpret_tensor(arg209_1, (2048, 6144), (1, 2048), 0), out=buf468)
            del arg209_1
            assert_size_stride(arg210_1, (6144, 2048), (2048, 1))
            buf469 = buf439; del buf439  # reuse
            # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:278
            extern_kernels.mm(reinterpret_tensor(buf467, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg210_1, (2048, 6144), (1, 2048), 0), out=buf469)
            del arg210_1
            buf470 = reinterpret_tensor(buf468, (1, 1, 6144), (6144, 6144, 1), 0); del buf468  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_209], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:279
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf470, buf469, 6144, stream=stream0)
            assert_size_stride(arg211_1, (2048, 6144), (6144, 1))
            buf471 = reinterpret_tensor(buf467, (1, 2048), (2048, 1), 0); del buf467  # reuse
            # Topologically Sorted Source Nodes: [linear_109, silu_15, linear_110, mul_209, down_proj_15], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:280
            extern_kernels.mm(reinterpret_tensor(buf470, (1, 6144), (0, 1), 0), reinterpret_tensor(arg211_1, (6144, 2048), (1, 6144), 0), out=buf471)
            del arg211_1
            assert_size_stride(arg212_1, (2048, ), (1, ))
            buf473 = reinterpret_tensor(buf464, (1, 1, 2048), (2048, 2048, 1), 0); del buf464  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_223, hidden_states_224, pow_65, variance_64, add_131, rsqrt_64, hidden_states_225, to_135, hidden_states_226], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:281
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf465, buf471, arg212_1, buf473, 1, 2048, stream=stream0)
            del arg212_1
            assert_size_stride(arg213_1, (2048, 2048), (2048, 1))
            buf474 = buf441; del buf441  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_223, hidden_states_224, pow_65, variance_64, add_131, rsqrt_64, hidden_states_225, to_135, hidden_states_226, linear_112], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:282
            extern_kernels.mm(reinterpret_tensor(buf473, (1, 2048), (0, 1), 0), reinterpret_tensor(arg213_1, (2048, 2048), (1, 2048), 0), out=buf474)
            del arg213_1
            assert_size_stride(arg214_1, (128, ), (1, ))
            buf476 = reinterpret_tensor(buf435, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf435  # reuse
            buf485 = reinterpret_tensor(buf476, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf476  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_112, view_48, hidden_states_227, pow_66, variance_65, add_132, rsqrt_65, hidden_states_228, to_137, mul_213, query_states_16, cos_19, mul_216, x2_32, neg_32, x1_32, cat_65, sin_19, mul_217, q_embed_16, getitem_107, hidden_states_231, key_16, getitem_108, hidden_states_232, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:283
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf485, buf474, arg214_1, arg4_1, 16, 128, stream=stream0)
            del arg214_1
            assert_size_stride(arg215_1, (1024, 2048), (2048, 1))
            buf477 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:284
            extern_kernels.mm(reinterpret_tensor(buf473, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg215_1, (2048, 1024), (1, 2048), 0), out=buf477)
            del arg215_1
            assert_size_stride(arg216_1, (128, ), (1, ))
            buf482 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf481 = reinterpret_tensor(buf482, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_19, sin_19, linear_113, view_49, hidden_states_229, pow_67, variance_66, add_133, rsqrt_66, hidden_states_230, to_139, mul_215, key_states_16, mul_218, x2_33, neg_33, x1_33, cat_66, mul_219, k_embed_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:285
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf477, arg216_1, arg4_1, buf481, 8, 128, stream=stream0)
            del arg216_1
            assert_size_stride(arg218_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg218_1 = copy_misaligned(arg218_1)
            buf480 = reinterpret_tensor(buf482, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_19, linear_113, view_49, hidden_states_229, pow_67, variance_66, add_133, rsqrt_66, hidden_states_230, to_139, mul_215, key_states_16, mul_218, k_embed_16, keys_16], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:286
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg218_1, buf480, 2097152, stream=stream0)
            del arg218_1
            assert_size_stride(arg217_1, (1024, 2048), (2048, 1))
            buf483 = buf477; del buf477  # reuse
            # Topologically Sorted Source Nodes: [linear_114], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:287
            extern_kernels.mm(reinterpret_tensor(buf473, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg217_1, (2048, 1024), (1, 2048), 0), out=buf483)
            del arg217_1
            assert_size_stride(arg219_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg219_1 = copy_misaligned(arg219_1)
            buf484 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_114, view_50, value_states_16, values_16], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:288
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg219_1, buf483, buf484, 2098176, stream=stream0)
            del arg219_1
            buf486 = buf457; del buf457  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_112, view_48, hidden_states_227, pow_66, variance_65, add_132, rsqrt_65, hidden_states_228, to_137, mul_213, query_states_16, cos_19, mul_216, q_embed_16, getitem_107, hidden_states_231, key_16, getitem_108, hidden_states_232, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:289
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf482, buf486, 4196352, stream=stream0)
            buf487 = buf456; del buf456  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_112, view_48, hidden_states_227, pow_66, variance_65, add_132, rsqrt_65, hidden_states_228, to_137, mul_213, query_states_16, cos_19, mul_216, q_embed_16, getitem_107, hidden_states_231, key_16, getitem_108, hidden_states_232, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:290
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf484, buf487, 4196352, stream=stream0)
            buf488 = buf458; del buf458  # reuse
            buf517 = buf429; del buf429  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_112, view_48, hidden_states_227, pow_66, variance_65, add_132, rsqrt_65, hidden_states_228, to_137, mul_213, query_states_16, cos_19, mul_216, q_embed_16, getitem_107, hidden_states_231, key_16, getitem_108, hidden_states_232, value_16, attn_output_64, linear_119, view_51, hidden_states_241, pow_70, variance_69, add_140, rsqrt_69, hidden_states_242, to_145, mul_226, query_states_17, cos_20, mul_229, q_embed_17, getitem_113, hidden_states_245, key_17, getitem_114, hidden_states_246, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:291
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf488, buf517, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_112, view_48, hidden_states_227, pow_66, variance_65, add_132, rsqrt_65, hidden_states_228, to_137, mul_213, query_states_16, cos_19, mul_216, q_embed_16, getitem_107, hidden_states_231, key_16, getitem_108, hidden_states_232, value_16, attn_output_64], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:292
            buf489 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf485, buf486, buf487, reinterpret_tensor(buf488, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf490 = buf489[0]
            assert_size_stride(buf490, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf490, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf489
            assert_size_stride(arg220_1, (2048, 2048), (2048, 1))
            buf494 = reinterpret_tensor(buf485, (1, 2048), (2048, 1), 0); del buf485  # reuse
            # Topologically Sorted Source Nodes: [transpose_68, reshape_50, attn_output_67], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:293
            extern_kernels.mm(reinterpret_tensor(buf490, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg220_1, (2048, 2048), (1, 2048), 0), out=buf494)
            del arg220_1
            assert_size_stride(arg221_1, (2048, ), (1, ))
            buf496 = reinterpret_tensor(buf490, (1, 1, 2048), (2048, 2048, 1), 0); del buf490  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_223, attn_output_67, hidden_states_233, hidden_states_234, pow_68, variance_67, add_137, rsqrt_67, hidden_states_235, to_141, hidden_states_236], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:294
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf465, buf471, buf494, arg221_1, buf496, 1, 2048, stream=stream0)
            del arg221_1
            assert_size_stride(arg222_1, (6144, 2048), (2048, 1))
            buf497 = reinterpret_tensor(buf470, (1, 6144), (6144, 1), 0); del buf470  # reuse
            # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:295
            extern_kernels.mm(reinterpret_tensor(buf496, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg222_1, (2048, 6144), (1, 2048), 0), out=buf497)
            del arg222_1
            assert_size_stride(arg223_1, (6144, 2048), (2048, 1))
            buf498 = buf469; del buf469  # reuse
            # Topologically Sorted Source Nodes: [linear_117], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:296
            extern_kernels.mm(reinterpret_tensor(buf496, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg223_1, (2048, 6144), (1, 2048), 0), out=buf498)
            del arg223_1
            buf499 = reinterpret_tensor(buf497, (1, 1, 6144), (6144, 6144, 1), 0); del buf497  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_222], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:297
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf499, buf498, 6144, stream=stream0)
            assert_size_stride(arg224_1, (2048, 6144), (6144, 1))
            buf500 = reinterpret_tensor(buf496, (1, 2048), (2048, 1), 0); del buf496  # reuse
            # Topologically Sorted Source Nodes: [linear_116, silu_16, linear_117, mul_222, down_proj_16], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:298
            extern_kernels.mm(reinterpret_tensor(buf499, (1, 6144), (0, 1), 0), reinterpret_tensor(arg224_1, (6144, 2048), (1, 6144), 0), out=buf500)
            del arg224_1
            assert_size_stride(arg225_1, (2048, ), (1, ))
            buf502 = buf473; del buf473  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_223, attn_output_67, hidden_states_233, down_proj_16, hidden_states_237, hidden_states_238, pow_69, variance_68, add_139, rsqrt_68, hidden_states_239, to_143, hidden_states_240], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:299
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf465, buf471, buf494, buf500, arg225_1, buf502, 1, 2048, stream=stream0)
            del arg225_1
            assert_size_stride(arg226_1, (2048, 2048), (2048, 1))
            buf503 = buf474; del buf474  # reuse
            # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:300
            extern_kernels.mm(reinterpret_tensor(buf502, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 2048), (1, 2048), 0), out=buf503)
            del arg226_1
            assert_size_stride(arg227_1, (128, ), (1, ))
            buf505 = reinterpret_tensor(buf412, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf412  # reuse
            buf514 = reinterpret_tensor(buf505, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_119, view_51, hidden_states_241, pow_70, variance_69, add_140, rsqrt_69, hidden_states_242, to_145, mul_226, query_states_17, cos_20, mul_229, x2_34, neg_34, x1_34, cat_69, sin_20, mul_230, q_embed_17, getitem_113, hidden_states_245, key_17, getitem_114, hidden_states_246, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:301
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf514, buf503, arg227_1, arg4_1, 16, 128, stream=stream0)
            del arg227_1
            del buf503
            assert_size_stride(arg228_1, (1024, 2048), (2048, 1))
            buf506 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [linear_120], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:302
            extern_kernels.mm(reinterpret_tensor(buf502, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg228_1, (2048, 1024), (1, 2048), 0), out=buf506)
            del arg228_1
            assert_size_stride(arg229_1, (128, ), (1, ))
            buf511 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf510 = reinterpret_tensor(buf511, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_20, sin_20, linear_120, view_52, hidden_states_243, pow_71, variance_70, add_141, rsqrt_70, hidden_states_244, to_147, mul_228, key_states_17, mul_231, x2_35, neg_35, x1_35, cat_70, mul_232, k_embed_17], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:303
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf506, arg229_1, arg4_1, buf510, 8, 128, stream=stream0)
            del arg229_1
            assert_size_stride(arg231_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg231_1 = copy_misaligned(arg231_1)
            buf509 = reinterpret_tensor(buf511, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_20, linear_120, view_52, hidden_states_243, pow_71, variance_70, add_141, rsqrt_70, hidden_states_244, to_147, mul_228, key_states_17, mul_231, k_embed_17, keys_17], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:304
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg231_1, buf509, 2097152, stream=stream0)
            del arg231_1
            assert_size_stride(arg230_1, (1024, 2048), (2048, 1))
            buf512 = buf506; del buf506  # reuse
            # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:305
            extern_kernels.mm(reinterpret_tensor(buf502, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg230_1, (2048, 1024), (1, 2048), 0), out=buf512)
            del arg230_1
            del buf502
            assert_size_stride(arg232_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg232_1 = copy_misaligned(arg232_1)
            buf513 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_121, view_53, value_states_17, values_17], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:306
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg232_1, buf512, buf513, 2098176, stream=stream0)
            del arg232_1
            buf515 = buf487; del buf487  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_119, view_51, hidden_states_241, pow_70, variance_69, add_140, rsqrt_69, hidden_states_242, to_145, mul_226, query_states_17, cos_20, mul_229, q_embed_17, getitem_113, hidden_states_245, key_17, getitem_114, hidden_states_246, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:307
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf511, buf515, 4196352, stream=stream0)
            buf516 = buf486; del buf486  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_119, view_51, hidden_states_241, pow_70, variance_69, add_140, rsqrt_69, hidden_states_242, to_145, mul_226, query_states_17, cos_20, mul_229, q_embed_17, getitem_113, hidden_states_245, key_17, getitem_114, hidden_states_246, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:308
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf513, buf516, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_119, view_51, hidden_states_241, pow_70, variance_69, add_140, rsqrt_69, hidden_states_242, to_145, mul_226, query_states_17, cos_20, mul_229, q_embed_17, getitem_113, hidden_states_245, key_17, getitem_114, hidden_states_246, value_17, attn_output_68], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:309
            buf518 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf514, buf515, buf516, reinterpret_tensor(buf517, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf519 = buf518[0]
            assert_size_stride(buf519, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf519, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf518
            assert_size_stride(arg233_1, (2048, 2048), (2048, 1))
            buf523 = reinterpret_tensor(buf514, (1, 2048), (2048, 1), 0); del buf514  # reuse
            # Topologically Sorted Source Nodes: [transpose_72, reshape_53, attn_output_71], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:310
            extern_kernels.mm(reinterpret_tensor(buf519, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg233_1, (2048, 2048), (1, 2048), 0), out=buf523)
            del arg233_1
            assert_size_stride(arg234_1, (2048, ), (1, ))
            buf524 = buf465; del buf465  # reuse
            buf526 = reinterpret_tensor(buf519, (1, 1, 2048), (2048, 2048, 1), 0); del buf519  # reuse
            # Topologically Sorted Source Nodes: [down_proj_15, hidden_states_223, attn_output_67, hidden_states_233, down_proj_16, hidden_states_237, attn_output_71, hidden_states_247, hidden_states_248, pow_72, variance_71, add_145, rsqrt_71, hidden_states_249, to_149, hidden_states_250], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:311
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf524, buf471, buf494, buf500, buf523, arg234_1, buf526, 1, 2048, stream=stream0)
            del arg234_1
            assert_size_stride(arg235_1, (6144, 2048), (2048, 1))
            buf527 = reinterpret_tensor(buf499, (1, 6144), (6144, 1), 0); del buf499  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_248, pow_72, variance_71, add_145, rsqrt_71, hidden_states_249, to_149, hidden_states_250, linear_123], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:312
            extern_kernels.mm(reinterpret_tensor(buf526, (1, 2048), (0, 1), 0), reinterpret_tensor(arg235_1, (2048, 6144), (1, 2048), 0), out=buf527)
            del arg235_1
            assert_size_stride(arg236_1, (6144, 2048), (2048, 1))
            buf528 = buf498; del buf498  # reuse
            # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:313
            extern_kernels.mm(reinterpret_tensor(buf526, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg236_1, (2048, 6144), (1, 2048), 0), out=buf528)
            del arg236_1
            buf529 = reinterpret_tensor(buf527, (1, 1, 6144), (6144, 6144, 1), 0); del buf527  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_235], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:314
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf529, buf528, 6144, stream=stream0)
            assert_size_stride(arg237_1, (2048, 6144), (6144, 1))
            buf530 = reinterpret_tensor(buf526, (1, 2048), (2048, 1), 0); del buf526  # reuse
            # Topologically Sorted Source Nodes: [linear_123, silu_17, linear_124, mul_235, down_proj_17], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:315
            extern_kernels.mm(reinterpret_tensor(buf529, (1, 6144), (0, 1), 0), reinterpret_tensor(arg237_1, (6144, 2048), (1, 6144), 0), out=buf530)
            del arg237_1
            assert_size_stride(arg238_1, (2048, ), (1, ))
            buf532 = reinterpret_tensor(buf523, (1, 1, 2048), (2048, 2048, 1), 0); del buf523  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_251, hidden_states_252, pow_73, variance_72, add_147, rsqrt_72, hidden_states_253, to_151, hidden_states_254], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:316
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf524, buf530, arg238_1, buf532, 1, 2048, stream=stream0)
            del arg238_1
            assert_size_stride(arg239_1, (2048, 2048), (2048, 1))
            buf533 = buf500; del buf500  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_251, hidden_states_252, pow_73, variance_72, add_147, rsqrt_72, hidden_states_253, to_151, hidden_states_254, linear_126], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:317
            extern_kernels.mm(reinterpret_tensor(buf532, (1, 2048), (0, 1), 0), reinterpret_tensor(arg239_1, (2048, 2048), (1, 2048), 0), out=buf533)
            del arg239_1
            assert_size_stride(arg240_1, (128, ), (1, ))
            buf535 = reinterpret_tensor(buf494, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf494  # reuse
            buf544 = reinterpret_tensor(buf535, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf535  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_126, view_54, hidden_states_255, pow_74, variance_73, add_148, rsqrt_73, hidden_states_256, to_153, mul_239, query_states_18, cos_21, mul_242, x2_36, neg_36, x1_36, cat_73, sin_21, mul_243, q_embed_18, getitem_119, hidden_states_259, key_18, getitem_120, hidden_states_260, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:318
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf544, buf533, arg240_1, arg4_1, 16, 128, stream=stream0)
            del arg240_1
            assert_size_stride(arg241_1, (1024, 2048), (2048, 1))
            buf536 = buf512; del buf512  # reuse
            # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:319
            extern_kernels.mm(reinterpret_tensor(buf532, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg241_1, (2048, 1024), (1, 2048), 0), out=buf536)
            del arg241_1
            assert_size_stride(arg242_1, (128, ), (1, ))
            buf541 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf540 = reinterpret_tensor(buf541, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_21, sin_21, linear_127, view_55, hidden_states_257, pow_75, variance_74, add_149, rsqrt_74, hidden_states_258, to_155, mul_241, key_states_18, mul_244, x2_37, neg_37, x1_37, cat_74, mul_245, k_embed_18], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:320
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf536, arg242_1, arg4_1, buf540, 8, 128, stream=stream0)
            del arg242_1
            assert_size_stride(arg244_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg244_1 = copy_misaligned(arg244_1)
            buf539 = reinterpret_tensor(buf541, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_21, linear_127, view_55, hidden_states_257, pow_75, variance_74, add_149, rsqrt_74, hidden_states_258, to_155, mul_241, key_states_18, mul_244, k_embed_18, keys_18], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:321
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg244_1, buf539, 2097152, stream=stream0)
            del arg244_1
            assert_size_stride(arg243_1, (1024, 2048), (2048, 1))
            buf542 = buf536; del buf536  # reuse
            # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:322
            extern_kernels.mm(reinterpret_tensor(buf532, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg243_1, (2048, 1024), (1, 2048), 0), out=buf542)
            del arg243_1
            assert_size_stride(arg245_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg245_1 = copy_misaligned(arg245_1)
            buf543 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_128, view_56, value_states_18, values_18], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:323
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg245_1, buf542, buf543, 2098176, stream=stream0)
            del arg245_1
            buf545 = buf516; del buf516  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_126, view_54, hidden_states_255, pow_74, variance_73, add_148, rsqrt_73, hidden_states_256, to_153, mul_239, query_states_18, cos_21, mul_242, q_embed_18, getitem_119, hidden_states_259, key_18, getitem_120, hidden_states_260, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:324
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf541, buf545, 4196352, stream=stream0)
            buf546 = buf515; del buf515  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_126, view_54, hidden_states_255, pow_74, variance_73, add_148, rsqrt_73, hidden_states_256, to_153, mul_239, query_states_18, cos_21, mul_242, q_embed_18, getitem_119, hidden_states_259, key_18, getitem_120, hidden_states_260, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:325
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf543, buf546, 4196352, stream=stream0)
            buf547 = buf517; del buf517  # reuse
            buf576 = buf488; del buf488  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_126, view_54, hidden_states_255, pow_74, variance_73, add_148, rsqrt_73, hidden_states_256, to_153, mul_239, query_states_18, cos_21, mul_242, q_embed_18, getitem_119, hidden_states_259, key_18, getitem_120, hidden_states_260, value_18, attn_output_72, linear_133, view_57, hidden_states_269, pow_78, variance_77, add_156, rsqrt_77, hidden_states_270, to_161, mul_252, query_states_19, cos_22, mul_255, q_embed_19, getitem_125, hidden_states_273, key_19, getitem_126, hidden_states_274, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:326
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf547, buf576, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_126, view_54, hidden_states_255, pow_74, variance_73, add_148, rsqrt_73, hidden_states_256, to_153, mul_239, query_states_18, cos_21, mul_242, q_embed_18, getitem_119, hidden_states_259, key_18, getitem_120, hidden_states_260, value_18, attn_output_72], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:327
            buf548 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf544, buf545, buf546, reinterpret_tensor(buf547, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf549 = buf548[0]
            assert_size_stride(buf549, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf549, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf548
            assert_size_stride(arg246_1, (2048, 2048), (2048, 1))
            buf553 = reinterpret_tensor(buf544, (1, 2048), (2048, 1), 0); del buf544  # reuse
            # Topologically Sorted Source Nodes: [transpose_76, reshape_56, attn_output_75], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:328
            extern_kernels.mm(reinterpret_tensor(buf549, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg246_1, (2048, 2048), (1, 2048), 0), out=buf553)
            del arg246_1
            assert_size_stride(arg247_1, (2048, ), (1, ))
            buf555 = reinterpret_tensor(buf549, (1, 1, 2048), (2048, 2048, 1), 0); del buf549  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_251, attn_output_75, hidden_states_261, hidden_states_262, pow_76, variance_75, add_153, rsqrt_75, hidden_states_263, to_157, hidden_states_264], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:329
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf524, buf530, buf553, arg247_1, buf555, 1, 2048, stream=stream0)
            del arg247_1
            assert_size_stride(arg248_1, (6144, 2048), (2048, 1))
            buf556 = reinterpret_tensor(buf529, (1, 6144), (6144, 1), 0); del buf529  # reuse
            # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:330
            extern_kernels.mm(reinterpret_tensor(buf555, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg248_1, (2048, 6144), (1, 2048), 0), out=buf556)
            del arg248_1
            assert_size_stride(arg249_1, (6144, 2048), (2048, 1))
            buf557 = buf528; del buf528  # reuse
            # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:331
            extern_kernels.mm(reinterpret_tensor(buf555, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg249_1, (2048, 6144), (1, 2048), 0), out=buf557)
            del arg249_1
            buf558 = reinterpret_tensor(buf556, (1, 1, 6144), (6144, 6144, 1), 0); del buf556  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_248], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:332
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf558, buf557, 6144, stream=stream0)
            assert_size_stride(arg250_1, (2048, 6144), (6144, 1))
            buf559 = reinterpret_tensor(buf555, (1, 2048), (2048, 1), 0); del buf555  # reuse
            # Topologically Sorted Source Nodes: [linear_130, silu_18, linear_131, mul_248, down_proj_18], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:333
            extern_kernels.mm(reinterpret_tensor(buf558, (1, 6144), (0, 1), 0), reinterpret_tensor(arg250_1, (6144, 2048), (1, 6144), 0), out=buf559)
            del arg250_1
            assert_size_stride(arg251_1, (2048, ), (1, ))
            buf561 = buf532; del buf532  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_251, attn_output_75, hidden_states_261, down_proj_18, hidden_states_265, hidden_states_266, pow_77, variance_76, add_155, rsqrt_76, hidden_states_267, to_159, hidden_states_268], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:334
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf524, buf530, buf553, buf559, arg251_1, buf561, 1, 2048, stream=stream0)
            del arg251_1
            assert_size_stride(arg252_1, (2048, 2048), (2048, 1))
            buf562 = buf533; del buf533  # reuse
            # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:335
            extern_kernels.mm(reinterpret_tensor(buf561, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg252_1, (2048, 2048), (1, 2048), 0), out=buf562)
            del arg252_1
            assert_size_stride(arg253_1, (128, ), (1, ))
            buf564 = reinterpret_tensor(buf471, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf471  # reuse
            buf573 = reinterpret_tensor(buf564, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf564  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_133, view_57, hidden_states_269, pow_78, variance_77, add_156, rsqrt_77, hidden_states_270, to_161, mul_252, query_states_19, cos_22, mul_255, x2_38, neg_38, x1_38, cat_77, sin_22, mul_256, q_embed_19, getitem_125, hidden_states_273, key_19, getitem_126, hidden_states_274, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:336
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf573, buf562, arg253_1, arg4_1, 16, 128, stream=stream0)
            del arg253_1
            del buf562
            assert_size_stride(arg254_1, (1024, 2048), (2048, 1))
            buf565 = buf542; del buf542  # reuse
            # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:337
            extern_kernels.mm(reinterpret_tensor(buf561, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg254_1, (2048, 1024), (1, 2048), 0), out=buf565)
            del arg254_1
            assert_size_stride(arg255_1, (128, ), (1, ))
            buf570 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf569 = reinterpret_tensor(buf570, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_22, sin_22, linear_134, view_58, hidden_states_271, pow_79, variance_78, add_157, rsqrt_78, hidden_states_272, to_163, mul_254, key_states_19, mul_257, x2_39, neg_39, x1_39, cat_78, mul_258, k_embed_19], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:338
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf565, arg255_1, arg4_1, buf569, 8, 128, stream=stream0)
            del arg255_1
            assert_size_stride(arg257_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg257_1 = copy_misaligned(arg257_1)
            buf568 = reinterpret_tensor(buf570, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_22, linear_134, view_58, hidden_states_271, pow_79, variance_78, add_157, rsqrt_78, hidden_states_272, to_163, mul_254, key_states_19, mul_257, k_embed_19, keys_19], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:339
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg257_1, buf568, 2097152, stream=stream0)
            del arg257_1
            assert_size_stride(arg256_1, (1024, 2048), (2048, 1))
            buf571 = buf565; del buf565  # reuse
            # Topologically Sorted Source Nodes: [linear_135], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:340
            extern_kernels.mm(reinterpret_tensor(buf561, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg256_1, (2048, 1024), (1, 2048), 0), out=buf571)
            del arg256_1
            del buf561
            assert_size_stride(arg258_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg258_1 = copy_misaligned(arg258_1)
            buf572 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_135, view_59, value_states_19, values_19], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:341
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg258_1, buf571, buf572, 2098176, stream=stream0)
            del arg258_1
            buf574 = buf546; del buf546  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_133, view_57, hidden_states_269, pow_78, variance_77, add_156, rsqrt_77, hidden_states_270, to_161, mul_252, query_states_19, cos_22, mul_255, q_embed_19, getitem_125, hidden_states_273, key_19, getitem_126, hidden_states_274, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:342
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf570, buf574, 4196352, stream=stream0)
            buf575 = buf545; del buf545  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_133, view_57, hidden_states_269, pow_78, variance_77, add_156, rsqrt_77, hidden_states_270, to_161, mul_252, query_states_19, cos_22, mul_255, q_embed_19, getitem_125, hidden_states_273, key_19, getitem_126, hidden_states_274, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:343
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf572, buf575, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_133, view_57, hidden_states_269, pow_78, variance_77, add_156, rsqrt_77, hidden_states_270, to_161, mul_252, query_states_19, cos_22, mul_255, q_embed_19, getitem_125, hidden_states_273, key_19, getitem_126, hidden_states_274, value_19, attn_output_76], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:344
            buf577 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf573, buf574, buf575, reinterpret_tensor(buf576, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf578 = buf577[0]
            assert_size_stride(buf578, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf578, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf577
            assert_size_stride(arg259_1, (2048, 2048), (2048, 1))
            buf582 = reinterpret_tensor(buf573, (1, 2048), (2048, 1), 0); del buf573  # reuse
            # Topologically Sorted Source Nodes: [transpose_80, reshape_59, attn_output_79], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:345
            extern_kernels.mm(reinterpret_tensor(buf578, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg259_1, (2048, 2048), (1, 2048), 0), out=buf582)
            del arg259_1
            assert_size_stride(arg260_1, (2048, ), (1, ))
            buf583 = buf524; del buf524  # reuse
            buf585 = reinterpret_tensor(buf578, (1, 1, 2048), (2048, 2048, 1), 0); del buf578  # reuse
            # Topologically Sorted Source Nodes: [down_proj_17, hidden_states_251, attn_output_75, hidden_states_261, down_proj_18, hidden_states_265, attn_output_79, hidden_states_275, hidden_states_276, pow_80, variance_79, add_161, rsqrt_79, hidden_states_277, to_165, hidden_states_278], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:346
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf583, buf530, buf553, buf559, buf582, arg260_1, buf585, 1, 2048, stream=stream0)
            del arg260_1
            assert_size_stride(arg261_1, (6144, 2048), (2048, 1))
            buf586 = reinterpret_tensor(buf558, (1, 6144), (6144, 1), 0); del buf558  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_276, pow_80, variance_79, add_161, rsqrt_79, hidden_states_277, to_165, hidden_states_278, linear_137], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:347
            extern_kernels.mm(reinterpret_tensor(buf585, (1, 2048), (0, 1), 0), reinterpret_tensor(arg261_1, (2048, 6144), (1, 2048), 0), out=buf586)
            del arg261_1
            assert_size_stride(arg262_1, (6144, 2048), (2048, 1))
            buf587 = buf557; del buf557  # reuse
            # Topologically Sorted Source Nodes: [linear_138], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:348
            extern_kernels.mm(reinterpret_tensor(buf585, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg262_1, (2048, 6144), (1, 2048), 0), out=buf587)
            del arg262_1
            buf588 = reinterpret_tensor(buf586, (1, 1, 6144), (6144, 6144, 1), 0); del buf586  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_261], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:349
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf588, buf587, 6144, stream=stream0)
            assert_size_stride(arg263_1, (2048, 6144), (6144, 1))
            buf589 = reinterpret_tensor(buf585, (1, 2048), (2048, 1), 0); del buf585  # reuse
            # Topologically Sorted Source Nodes: [linear_137, silu_19, linear_138, mul_261, down_proj_19], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:350
            extern_kernels.mm(reinterpret_tensor(buf588, (1, 6144), (0, 1), 0), reinterpret_tensor(arg263_1, (6144, 2048), (1, 6144), 0), out=buf589)
            del arg263_1
            assert_size_stride(arg264_1, (2048, ), (1, ))
            buf591 = reinterpret_tensor(buf582, (1, 1, 2048), (2048, 2048, 1), 0); del buf582  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_279, hidden_states_280, pow_81, variance_80, add_163, rsqrt_80, hidden_states_281, to_167, hidden_states_282], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:351
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf583, buf589, arg264_1, buf591, 1, 2048, stream=stream0)
            del arg264_1
            assert_size_stride(arg265_1, (2048, 2048), (2048, 1))
            buf592 = buf559; del buf559  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_279, hidden_states_280, pow_81, variance_80, add_163, rsqrt_80, hidden_states_281, to_167, hidden_states_282, linear_140], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:352
            extern_kernels.mm(reinterpret_tensor(buf591, (1, 2048), (0, 1), 0), reinterpret_tensor(arg265_1, (2048, 2048), (1, 2048), 0), out=buf592)
            del arg265_1
            assert_size_stride(arg266_1, (128, ), (1, ))
            buf594 = reinterpret_tensor(buf553, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf553  # reuse
            buf603 = reinterpret_tensor(buf594, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf594  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_140, view_60, hidden_states_283, pow_82, variance_81, add_164, rsqrt_81, hidden_states_284, to_169, mul_265, query_states_20, cos_23, mul_268, x2_40, neg_40, x1_40, cat_81, sin_23, mul_269, q_embed_20, getitem_131, hidden_states_287, key_20, getitem_132, hidden_states_288, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:353
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf603, buf592, arg266_1, arg4_1, 16, 128, stream=stream0)
            del arg266_1
            assert_size_stride(arg267_1, (1024, 2048), (2048, 1))
            buf595 = buf571; del buf571  # reuse
            # Topologically Sorted Source Nodes: [linear_141], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:354
            extern_kernels.mm(reinterpret_tensor(buf591, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg267_1, (2048, 1024), (1, 2048), 0), out=buf595)
            del arg267_1
            assert_size_stride(arg268_1, (128, ), (1, ))
            buf600 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf599 = reinterpret_tensor(buf600, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_23, sin_23, linear_141, view_61, hidden_states_285, pow_83, variance_82, add_165, rsqrt_82, hidden_states_286, to_171, mul_267, key_states_20, mul_270, x2_41, neg_41, x1_41, cat_82, mul_271, k_embed_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:355
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf595, arg268_1, arg4_1, buf599, 8, 128, stream=stream0)
            del arg268_1
            assert_size_stride(arg270_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg270_1 = copy_misaligned(arg270_1)
            buf598 = reinterpret_tensor(buf600, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_23, linear_141, view_61, hidden_states_285, pow_83, variance_82, add_165, rsqrt_82, hidden_states_286, to_171, mul_267, key_states_20, mul_270, k_embed_20, keys_20], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:356
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg270_1, buf598, 2097152, stream=stream0)
            del arg270_1
            assert_size_stride(arg269_1, (1024, 2048), (2048, 1))
            buf601 = buf595; del buf595  # reuse
            # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:357
            extern_kernels.mm(reinterpret_tensor(buf591, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg269_1, (2048, 1024), (1, 2048), 0), out=buf601)
            del arg269_1
            assert_size_stride(arg271_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg271_1 = copy_misaligned(arg271_1)
            buf602 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_142, view_62, value_states_20, values_20], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:358
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg271_1, buf601, buf602, 2098176, stream=stream0)
            del arg271_1
            buf604 = buf575; del buf575  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_140, view_60, hidden_states_283, pow_82, variance_81, add_164, rsqrt_81, hidden_states_284, to_169, mul_265, query_states_20, cos_23, mul_268, q_embed_20, getitem_131, hidden_states_287, key_20, getitem_132, hidden_states_288, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:359
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf600, buf604, 4196352, stream=stream0)
            buf605 = buf574; del buf574  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_140, view_60, hidden_states_283, pow_82, variance_81, add_164, rsqrt_81, hidden_states_284, to_169, mul_265, query_states_20, cos_23, mul_268, q_embed_20, getitem_131, hidden_states_287, key_20, getitem_132, hidden_states_288, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:360
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf602, buf605, 4196352, stream=stream0)
            buf606 = buf576; del buf576  # reuse
            buf635 = buf547; del buf547  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_140, view_60, hidden_states_283, pow_82, variance_81, add_164, rsqrt_81, hidden_states_284, to_169, mul_265, query_states_20, cos_23, mul_268, q_embed_20, getitem_131, hidden_states_287, key_20, getitem_132, hidden_states_288, value_20, attn_output_80, linear_147, view_63, hidden_states_297, pow_86, variance_85, add_172, rsqrt_85, hidden_states_298, to_177, mul_278, query_states_21, cos_24, mul_281, q_embed_21, getitem_137, hidden_states_301, key_21, getitem_138, hidden_states_302, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:361
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf606, buf635, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_140, view_60, hidden_states_283, pow_82, variance_81, add_164, rsqrt_81, hidden_states_284, to_169, mul_265, query_states_20, cos_23, mul_268, q_embed_20, getitem_131, hidden_states_287, key_20, getitem_132, hidden_states_288, value_20, attn_output_80], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:362
            buf607 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf603, buf604, buf605, reinterpret_tensor(buf606, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf608 = buf607[0]
            assert_size_stride(buf608, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf608, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf607
            assert_size_stride(arg272_1, (2048, 2048), (2048, 1))
            buf612 = reinterpret_tensor(buf603, (1, 2048), (2048, 1), 0); del buf603  # reuse
            # Topologically Sorted Source Nodes: [transpose_84, reshape_62, attn_output_83], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:363
            extern_kernels.mm(reinterpret_tensor(buf608, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg272_1, (2048, 2048), (1, 2048), 0), out=buf612)
            del arg272_1
            assert_size_stride(arg273_1, (2048, ), (1, ))
            buf614 = reinterpret_tensor(buf608, (1, 1, 2048), (2048, 2048, 1), 0); del buf608  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_279, attn_output_83, hidden_states_289, hidden_states_290, pow_84, variance_83, add_169, rsqrt_83, hidden_states_291, to_173, hidden_states_292], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:364
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf583, buf589, buf612, arg273_1, buf614, 1, 2048, stream=stream0)
            del arg273_1
            assert_size_stride(arg274_1, (6144, 2048), (2048, 1))
            buf615 = reinterpret_tensor(buf588, (1, 6144), (6144, 1), 0); del buf588  # reuse
            # Topologically Sorted Source Nodes: [linear_144], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:365
            extern_kernels.mm(reinterpret_tensor(buf614, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg274_1, (2048, 6144), (1, 2048), 0), out=buf615)
            del arg274_1
            assert_size_stride(arg275_1, (6144, 2048), (2048, 1))
            buf616 = buf587; del buf587  # reuse
            # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:366
            extern_kernels.mm(reinterpret_tensor(buf614, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg275_1, (2048, 6144), (1, 2048), 0), out=buf616)
            del arg275_1
            buf617 = reinterpret_tensor(buf615, (1, 1, 6144), (6144, 6144, 1), 0); del buf615  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_274], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:367
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf617, buf616, 6144, stream=stream0)
            assert_size_stride(arg276_1, (2048, 6144), (6144, 1))
            buf618 = reinterpret_tensor(buf614, (1, 2048), (2048, 1), 0); del buf614  # reuse
            # Topologically Sorted Source Nodes: [linear_144, silu_20, linear_145, mul_274, down_proj_20], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:368
            extern_kernels.mm(reinterpret_tensor(buf617, (1, 6144), (0, 1), 0), reinterpret_tensor(arg276_1, (6144, 2048), (1, 6144), 0), out=buf618)
            del arg276_1
            assert_size_stride(arg277_1, (2048, ), (1, ))
            buf620 = buf591; del buf591  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_279, attn_output_83, hidden_states_289, down_proj_20, hidden_states_293, hidden_states_294, pow_85, variance_84, add_171, rsqrt_84, hidden_states_295, to_175, hidden_states_296], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:369
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf583, buf589, buf612, buf618, arg277_1, buf620, 1, 2048, stream=stream0)
            del arg277_1
            assert_size_stride(arg278_1, (2048, 2048), (2048, 1))
            buf621 = buf592; del buf592  # reuse
            # Topologically Sorted Source Nodes: [linear_147], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:370
            extern_kernels.mm(reinterpret_tensor(buf620, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg278_1, (2048, 2048), (1, 2048), 0), out=buf621)
            del arg278_1
            assert_size_stride(arg279_1, (128, ), (1, ))
            buf623 = reinterpret_tensor(buf530, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf530  # reuse
            buf632 = reinterpret_tensor(buf623, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf623  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_147, view_63, hidden_states_297, pow_86, variance_85, add_172, rsqrt_85, hidden_states_298, to_177, mul_278, query_states_21, cos_24, mul_281, x2_42, neg_42, x1_42, cat_85, sin_24, mul_282, q_embed_21, getitem_137, hidden_states_301, key_21, getitem_138, hidden_states_302, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:371
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf632, buf621, arg279_1, arg4_1, 16, 128, stream=stream0)
            del arg279_1
            del buf621
            assert_size_stride(arg280_1, (1024, 2048), (2048, 1))
            buf624 = buf601; del buf601  # reuse
            # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:372
            extern_kernels.mm(reinterpret_tensor(buf620, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg280_1, (2048, 1024), (1, 2048), 0), out=buf624)
            del arg280_1
            assert_size_stride(arg281_1, (128, ), (1, ))
            buf629 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf628 = reinterpret_tensor(buf629, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_24, sin_24, linear_148, view_64, hidden_states_299, pow_87, variance_86, add_173, rsqrt_86, hidden_states_300, to_179, mul_280, key_states_21, mul_283, x2_43, neg_43, x1_43, cat_86, mul_284, k_embed_21], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:373
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf624, arg281_1, arg4_1, buf628, 8, 128, stream=stream0)
            del arg281_1
            assert_size_stride(arg283_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg283_1 = copy_misaligned(arg283_1)
            buf627 = reinterpret_tensor(buf629, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_24, linear_148, view_64, hidden_states_299, pow_87, variance_86, add_173, rsqrt_86, hidden_states_300, to_179, mul_280, key_states_21, mul_283, k_embed_21, keys_21], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:374
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg283_1, buf627, 2097152, stream=stream0)
            del arg283_1
            assert_size_stride(arg282_1, (1024, 2048), (2048, 1))
            buf630 = buf624; del buf624  # reuse
            # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:375
            extern_kernels.mm(reinterpret_tensor(buf620, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg282_1, (2048, 1024), (1, 2048), 0), out=buf630)
            del arg282_1
            del buf620
            assert_size_stride(arg284_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg284_1 = copy_misaligned(arg284_1)
            buf631 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_149, view_65, value_states_21, values_21], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:376
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg284_1, buf630, buf631, 2098176, stream=stream0)
            del arg284_1
            buf633 = buf605; del buf605  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_147, view_63, hidden_states_297, pow_86, variance_85, add_172, rsqrt_85, hidden_states_298, to_177, mul_278, query_states_21, cos_24, mul_281, q_embed_21, getitem_137, hidden_states_301, key_21, getitem_138, hidden_states_302, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:377
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf629, buf633, 4196352, stream=stream0)
            buf634 = buf604; del buf604  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_147, view_63, hidden_states_297, pow_86, variance_85, add_172, rsqrt_85, hidden_states_298, to_177, mul_278, query_states_21, cos_24, mul_281, q_embed_21, getitem_137, hidden_states_301, key_21, getitem_138, hidden_states_302, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:378
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf631, buf634, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_147, view_63, hidden_states_297, pow_86, variance_85, add_172, rsqrt_85, hidden_states_298, to_177, mul_278, query_states_21, cos_24, mul_281, q_embed_21, getitem_137, hidden_states_301, key_21, getitem_138, hidden_states_302, value_21, attn_output_84], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:379
            buf636 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf632, buf633, buf634, reinterpret_tensor(buf635, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf637 = buf636[0]
            assert_size_stride(buf637, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf637, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf636
            assert_size_stride(arg285_1, (2048, 2048), (2048, 1))
            buf641 = reinterpret_tensor(buf632, (1, 2048), (2048, 1), 0); del buf632  # reuse
            # Topologically Sorted Source Nodes: [transpose_88, reshape_65, attn_output_87], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:380
            extern_kernels.mm(reinterpret_tensor(buf637, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg285_1, (2048, 2048), (1, 2048), 0), out=buf641)
            del arg285_1
            assert_size_stride(arg286_1, (2048, ), (1, ))
            buf642 = buf583; del buf583  # reuse
            buf644 = reinterpret_tensor(buf637, (1, 1, 2048), (2048, 2048, 1), 0); del buf637  # reuse
            # Topologically Sorted Source Nodes: [down_proj_19, hidden_states_279, attn_output_83, hidden_states_289, down_proj_20, hidden_states_293, attn_output_87, hidden_states_303, hidden_states_304, pow_88, variance_87, add_177, rsqrt_87, hidden_states_305, to_181, hidden_states_306], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:381
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf642, buf589, buf612, buf618, buf641, arg286_1, buf644, 1, 2048, stream=stream0)
            del arg286_1
            assert_size_stride(arg287_1, (6144, 2048), (2048, 1))
            buf645 = reinterpret_tensor(buf617, (1, 6144), (6144, 1), 0); del buf617  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_304, pow_88, variance_87, add_177, rsqrt_87, hidden_states_305, to_181, hidden_states_306, linear_151], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:382
            extern_kernels.mm(reinterpret_tensor(buf644, (1, 2048), (0, 1), 0), reinterpret_tensor(arg287_1, (2048, 6144), (1, 2048), 0), out=buf645)
            del arg287_1
            assert_size_stride(arg288_1, (6144, 2048), (2048, 1))
            buf646 = buf616; del buf616  # reuse
            # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:383
            extern_kernels.mm(reinterpret_tensor(buf644, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg288_1, (2048, 6144), (1, 2048), 0), out=buf646)
            del arg288_1
            buf647 = reinterpret_tensor(buf645, (1, 1, 6144), (6144, 6144, 1), 0); del buf645  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_287], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:384
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf647, buf646, 6144, stream=stream0)
            assert_size_stride(arg289_1, (2048, 6144), (6144, 1))
            buf648 = reinterpret_tensor(buf644, (1, 2048), (2048, 1), 0); del buf644  # reuse
            # Topologically Sorted Source Nodes: [linear_151, silu_21, linear_152, mul_287, down_proj_21], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:385
            extern_kernels.mm(reinterpret_tensor(buf647, (1, 6144), (0, 1), 0), reinterpret_tensor(arg289_1, (6144, 2048), (1, 6144), 0), out=buf648)
            del arg289_1
            assert_size_stride(arg290_1, (2048, ), (1, ))
            buf650 = reinterpret_tensor(buf641, (1, 1, 2048), (2048, 2048, 1), 0); del buf641  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_307, hidden_states_308, pow_89, variance_88, add_179, rsqrt_88, hidden_states_309, to_183, hidden_states_310], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:386
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf642, buf648, arg290_1, buf650, 1, 2048, stream=stream0)
            del arg290_1
            assert_size_stride(arg291_1, (2048, 2048), (2048, 1))
            buf651 = buf618; del buf618  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_307, hidden_states_308, pow_89, variance_88, add_179, rsqrt_88, hidden_states_309, to_183, hidden_states_310, linear_154], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:387
            extern_kernels.mm(reinterpret_tensor(buf650, (1, 2048), (0, 1), 0), reinterpret_tensor(arg291_1, (2048, 2048), (1, 2048), 0), out=buf651)
            del arg291_1
            assert_size_stride(arg292_1, (128, ), (1, ))
            buf653 = reinterpret_tensor(buf612, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf612  # reuse
            buf662 = reinterpret_tensor(buf653, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf653  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_154, view_66, hidden_states_311, pow_90, variance_89, add_180, rsqrt_89, hidden_states_312, to_185, mul_291, query_states_22, cos_25, mul_294, x2_44, neg_44, x1_44, cat_89, sin_25, mul_295, q_embed_22, getitem_143, hidden_states_315, key_22, getitem_144, hidden_states_316, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:388
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf662, buf651, arg292_1, arg4_1, 16, 128, stream=stream0)
            del arg292_1
            assert_size_stride(arg293_1, (1024, 2048), (2048, 1))
            buf654 = buf630; del buf630  # reuse
            # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:389
            extern_kernels.mm(reinterpret_tensor(buf650, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg293_1, (2048, 1024), (1, 2048), 0), out=buf654)
            del arg293_1
            assert_size_stride(arg294_1, (128, ), (1, ))
            buf659 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf658 = reinterpret_tensor(buf659, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_25, sin_25, linear_155, view_67, hidden_states_313, pow_91, variance_90, add_181, rsqrt_90, hidden_states_314, to_187, mul_293, key_states_22, mul_296, x2_45, neg_45, x1_45, cat_90, mul_297, k_embed_22], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:390
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf654, arg294_1, arg4_1, buf658, 8, 128, stream=stream0)
            del arg294_1
            assert_size_stride(arg296_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg296_1 = copy_misaligned(arg296_1)
            buf657 = reinterpret_tensor(buf659, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_25, linear_155, view_67, hidden_states_313, pow_91, variance_90, add_181, rsqrt_90, hidden_states_314, to_187, mul_293, key_states_22, mul_296, k_embed_22, keys_22], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:391
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg296_1, buf657, 2097152, stream=stream0)
            del arg296_1
            assert_size_stride(arg295_1, (1024, 2048), (2048, 1))
            buf660 = buf654; del buf654  # reuse
            # Topologically Sorted Source Nodes: [linear_156], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:392
            extern_kernels.mm(reinterpret_tensor(buf650, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg295_1, (2048, 1024), (1, 2048), 0), out=buf660)
            del arg295_1
            assert_size_stride(arg297_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg297_1 = copy_misaligned(arg297_1)
            buf661 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_156, view_68, value_states_22, values_22], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:393
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg297_1, buf660, buf661, 2098176, stream=stream0)
            del arg297_1
            buf663 = buf634; del buf634  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_154, view_66, hidden_states_311, pow_90, variance_89, add_180, rsqrt_89, hidden_states_312, to_185, mul_291, query_states_22, cos_25, mul_294, q_embed_22, getitem_143, hidden_states_315, key_22, getitem_144, hidden_states_316, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:394
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf659, buf663, 4196352, stream=stream0)
            buf664 = buf633; del buf633  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_154, view_66, hidden_states_311, pow_90, variance_89, add_180, rsqrt_89, hidden_states_312, to_185, mul_291, query_states_22, cos_25, mul_294, q_embed_22, getitem_143, hidden_states_315, key_22, getitem_144, hidden_states_316, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:395
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf661, buf664, 4196352, stream=stream0)
            buf665 = buf635; del buf635  # reuse
            buf694 = buf606; del buf606  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_154, view_66, hidden_states_311, pow_90, variance_89, add_180, rsqrt_89, hidden_states_312, to_185, mul_291, query_states_22, cos_25, mul_294, q_embed_22, getitem_143, hidden_states_315, key_22, getitem_144, hidden_states_316, value_22, attn_output_88, linear_161, view_69, hidden_states_325, pow_94, variance_93, add_188, rsqrt_93, hidden_states_326, to_193, mul_304, query_states_23, cos_26, mul_307, q_embed_23, getitem_149, hidden_states_329, key_23, getitem_150, hidden_states_330, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:396
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf665, buf694, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_154, view_66, hidden_states_311, pow_90, variance_89, add_180, rsqrt_89, hidden_states_312, to_185, mul_291, query_states_22, cos_25, mul_294, q_embed_22, getitem_143, hidden_states_315, key_22, getitem_144, hidden_states_316, value_22, attn_output_88], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:397
            buf666 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf662, buf663, buf664, reinterpret_tensor(buf665, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf667 = buf666[0]
            assert_size_stride(buf667, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf667, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf666
            assert_size_stride(arg298_1, (2048, 2048), (2048, 1))
            buf671 = reinterpret_tensor(buf662, (1, 2048), (2048, 1), 0); del buf662  # reuse
            # Topologically Sorted Source Nodes: [transpose_92, reshape_68, attn_output_91], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:398
            extern_kernels.mm(reinterpret_tensor(buf667, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg298_1, (2048, 2048), (1, 2048), 0), out=buf671)
            del arg298_1
            assert_size_stride(arg299_1, (2048, ), (1, ))
            buf673 = reinterpret_tensor(buf667, (1, 1, 2048), (2048, 2048, 1), 0); del buf667  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_307, attn_output_91, hidden_states_317, hidden_states_318, pow_92, variance_91, add_185, rsqrt_91, hidden_states_319, to_189, hidden_states_320], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:399
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf642, buf648, buf671, arg299_1, buf673, 1, 2048, stream=stream0)
            del arg299_1
            assert_size_stride(arg300_1, (6144, 2048), (2048, 1))
            buf674 = reinterpret_tensor(buf647, (1, 6144), (6144, 1), 0); del buf647  # reuse
            # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:400
            extern_kernels.mm(reinterpret_tensor(buf673, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg300_1, (2048, 6144), (1, 2048), 0), out=buf674)
            del arg300_1
            assert_size_stride(arg301_1, (6144, 2048), (2048, 1))
            buf675 = buf646; del buf646  # reuse
            # Topologically Sorted Source Nodes: [linear_159], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:401
            extern_kernels.mm(reinterpret_tensor(buf673, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg301_1, (2048, 6144), (1, 2048), 0), out=buf675)
            del arg301_1
            buf676 = reinterpret_tensor(buf674, (1, 1, 6144), (6144, 6144, 1), 0); del buf674  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_300], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:402
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf676, buf675, 6144, stream=stream0)
            assert_size_stride(arg302_1, (2048, 6144), (6144, 1))
            buf677 = reinterpret_tensor(buf673, (1, 2048), (2048, 1), 0); del buf673  # reuse
            # Topologically Sorted Source Nodes: [linear_158, silu_22, linear_159, mul_300, down_proj_22], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:403
            extern_kernels.mm(reinterpret_tensor(buf676, (1, 6144), (0, 1), 0), reinterpret_tensor(arg302_1, (6144, 2048), (1, 6144), 0), out=buf677)
            del arg302_1
            assert_size_stride(arg303_1, (2048, ), (1, ))
            buf679 = buf650; del buf650  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_307, attn_output_91, hidden_states_317, down_proj_22, hidden_states_321, hidden_states_322, pow_93, variance_92, add_187, rsqrt_92, hidden_states_323, to_191, hidden_states_324], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:404
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf642, buf648, buf671, buf677, arg303_1, buf679, 1, 2048, stream=stream0)
            del arg303_1
            assert_size_stride(arg304_1, (2048, 2048), (2048, 1))
            buf680 = buf651; del buf651  # reuse
            # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:405
            extern_kernels.mm(reinterpret_tensor(buf679, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg304_1, (2048, 2048), (1, 2048), 0), out=buf680)
            del arg304_1
            assert_size_stride(arg305_1, (128, ), (1, ))
            buf682 = reinterpret_tensor(buf589, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf589  # reuse
            buf691 = reinterpret_tensor(buf682, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf682  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_161, view_69, hidden_states_325, pow_94, variance_93, add_188, rsqrt_93, hidden_states_326, to_193, mul_304, query_states_23, cos_26, mul_307, x2_46, neg_46, x1_46, cat_93, sin_26, mul_308, q_embed_23, getitem_149, hidden_states_329, key_23, getitem_150, hidden_states_330, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:406
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf691, buf680, arg305_1, arg4_1, 16, 128, stream=stream0)
            del arg305_1
            del buf680
            assert_size_stride(arg306_1, (1024, 2048), (2048, 1))
            buf683 = buf660; del buf660  # reuse
            # Topologically Sorted Source Nodes: [linear_162], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:407
            extern_kernels.mm(reinterpret_tensor(buf679, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg306_1, (2048, 1024), (1, 2048), 0), out=buf683)
            del arg306_1
            assert_size_stride(arg307_1, (128, ), (1, ))
            buf688 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf687 = reinterpret_tensor(buf688, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_26, sin_26, linear_162, view_70, hidden_states_327, pow_95, variance_94, add_189, rsqrt_94, hidden_states_328, to_195, mul_306, key_states_23, mul_309, x2_47, neg_47, x1_47, cat_94, mul_310, k_embed_23], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:408
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf683, arg307_1, arg4_1, buf687, 8, 128, stream=stream0)
            del arg307_1
            assert_size_stride(arg309_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg309_1 = copy_misaligned(arg309_1)
            buf686 = reinterpret_tensor(buf688, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_26, linear_162, view_70, hidden_states_327, pow_95, variance_94, add_189, rsqrt_94, hidden_states_328, to_195, mul_306, key_states_23, mul_309, k_embed_23, keys_23], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:409
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg309_1, buf686, 2097152, stream=stream0)
            del arg309_1
            assert_size_stride(arg308_1, (1024, 2048), (2048, 1))
            buf689 = buf683; del buf683  # reuse
            # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:410
            extern_kernels.mm(reinterpret_tensor(buf679, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg308_1, (2048, 1024), (1, 2048), 0), out=buf689)
            del arg308_1
            del buf679
            assert_size_stride(arg310_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg310_1 = copy_misaligned(arg310_1)
            buf690 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_163, view_71, value_states_23, values_23], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:411
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg310_1, buf689, buf690, 2098176, stream=stream0)
            del arg310_1
            buf692 = buf664; del buf664  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_161, view_69, hidden_states_325, pow_94, variance_93, add_188, rsqrt_93, hidden_states_326, to_193, mul_304, query_states_23, cos_26, mul_307, q_embed_23, getitem_149, hidden_states_329, key_23, getitem_150, hidden_states_330, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:412
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf688, buf692, 4196352, stream=stream0)
            buf693 = buf663; del buf663  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_161, view_69, hidden_states_325, pow_94, variance_93, add_188, rsqrt_93, hidden_states_326, to_193, mul_304, query_states_23, cos_26, mul_307, q_embed_23, getitem_149, hidden_states_329, key_23, getitem_150, hidden_states_330, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:413
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf690, buf693, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_161, view_69, hidden_states_325, pow_94, variance_93, add_188, rsqrt_93, hidden_states_326, to_193, mul_304, query_states_23, cos_26, mul_307, q_embed_23, getitem_149, hidden_states_329, key_23, getitem_150, hidden_states_330, value_23, attn_output_92], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:414
            buf695 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf691, buf692, buf693, reinterpret_tensor(buf694, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf696 = buf695[0]
            assert_size_stride(buf696, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf696, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf695
            assert_size_stride(arg311_1, (2048, 2048), (2048, 1))
            buf700 = reinterpret_tensor(buf691, (1, 2048), (2048, 1), 0); del buf691  # reuse
            # Topologically Sorted Source Nodes: [transpose_96, reshape_71, attn_output_95], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:415
            extern_kernels.mm(reinterpret_tensor(buf696, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg311_1, (2048, 2048), (1, 2048), 0), out=buf700)
            del arg311_1
            assert_size_stride(arg312_1, (2048, ), (1, ))
            buf701 = buf642; del buf642  # reuse
            buf703 = reinterpret_tensor(buf696, (1, 1, 2048), (2048, 2048, 1), 0); del buf696  # reuse
            # Topologically Sorted Source Nodes: [down_proj_21, hidden_states_307, attn_output_91, hidden_states_317, down_proj_22, hidden_states_321, attn_output_95, hidden_states_331, hidden_states_332, pow_96, variance_95, add_193, rsqrt_95, hidden_states_333, to_197, hidden_states_334], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:416
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf701, buf648, buf671, buf677, buf700, arg312_1, buf703, 1, 2048, stream=stream0)
            del arg312_1
            assert_size_stride(arg313_1, (6144, 2048), (2048, 1))
            buf704 = reinterpret_tensor(buf676, (1, 6144), (6144, 1), 0); del buf676  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_332, pow_96, variance_95, add_193, rsqrt_95, hidden_states_333, to_197, hidden_states_334, linear_165], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:417
            extern_kernels.mm(reinterpret_tensor(buf703, (1, 2048), (0, 1), 0), reinterpret_tensor(arg313_1, (2048, 6144), (1, 2048), 0), out=buf704)
            del arg313_1
            assert_size_stride(arg314_1, (6144, 2048), (2048, 1))
            buf705 = buf675; del buf675  # reuse
            # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:418
            extern_kernels.mm(reinterpret_tensor(buf703, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg314_1, (2048, 6144), (1, 2048), 0), out=buf705)
            del arg314_1
            buf706 = reinterpret_tensor(buf704, (1, 1, 6144), (6144, 6144, 1), 0); del buf704  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_313], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:419
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf706, buf705, 6144, stream=stream0)
            assert_size_stride(arg315_1, (2048, 6144), (6144, 1))
            buf707 = reinterpret_tensor(buf703, (1, 2048), (2048, 1), 0); del buf703  # reuse
            # Topologically Sorted Source Nodes: [linear_165, silu_23, linear_166, mul_313, down_proj_23], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:420
            extern_kernels.mm(reinterpret_tensor(buf706, (1, 6144), (0, 1), 0), reinterpret_tensor(arg315_1, (6144, 2048), (1, 6144), 0), out=buf707)
            del arg315_1
            assert_size_stride(arg316_1, (2048, ), (1, ))
            buf709 = reinterpret_tensor(buf700, (1, 1, 2048), (2048, 2048, 1), 0); del buf700  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_335, hidden_states_336, pow_97, variance_96, add_195, rsqrt_96, hidden_states_337, to_199, hidden_states_338], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:421
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf701, buf707, arg316_1, buf709, 1, 2048, stream=stream0)
            del arg316_1
            assert_size_stride(arg317_1, (2048, 2048), (2048, 1))
            buf710 = buf677; del buf677  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_335, hidden_states_336, pow_97, variance_96, add_195, rsqrt_96, hidden_states_337, to_199, hidden_states_338, linear_168], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:422
            extern_kernels.mm(reinterpret_tensor(buf709, (1, 2048), (0, 1), 0), reinterpret_tensor(arg317_1, (2048, 2048), (1, 2048), 0), out=buf710)
            del arg317_1
            assert_size_stride(arg318_1, (128, ), (1, ))
            buf712 = reinterpret_tensor(buf671, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf671  # reuse
            buf721 = reinterpret_tensor(buf712, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf712  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_168, view_72, hidden_states_339, pow_98, variance_97, add_196, rsqrt_97, hidden_states_340, to_201, mul_317, query_states_24, cos_27, mul_320, x2_48, neg_48, x1_48, cat_97, sin_27, mul_321, q_embed_24, getitem_155, hidden_states_343, key_24, getitem_156, hidden_states_344, value_24, attn_output_96], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:423
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf721, buf710, arg318_1, arg4_1, 16, 128, stream=stream0)
            del arg318_1
            assert_size_stride(arg319_1, (1024, 2048), (2048, 1))
            buf713 = buf689; del buf689  # reuse
            # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:424
            extern_kernels.mm(reinterpret_tensor(buf709, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg319_1, (2048, 1024), (1, 2048), 0), out=buf713)
            del arg319_1
            assert_size_stride(arg320_1, (128, ), (1, ))
            buf718 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf717 = reinterpret_tensor(buf718, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_27, sin_27, linear_169, view_73, hidden_states_341, pow_99, variance_98, add_197, rsqrt_98, hidden_states_342, to_203, mul_319, key_states_24, mul_322, x2_49, neg_49, x1_49, cat_98, mul_323, k_embed_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:425
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf713, arg320_1, arg4_1, buf717, 8, 128, stream=stream0)
            del arg320_1
            assert_size_stride(arg322_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg322_1 = copy_misaligned(arg322_1)
            buf716 = reinterpret_tensor(buf718, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_27, linear_169, view_73, hidden_states_341, pow_99, variance_98, add_197, rsqrt_98, hidden_states_342, to_203, mul_319, key_states_24, mul_322, k_embed_24, keys_24], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:426
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg322_1, buf716, 2097152, stream=stream0)
            del arg322_1
            assert_size_stride(arg321_1, (1024, 2048), (2048, 1))
            buf719 = buf713; del buf713  # reuse
            # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:427
            extern_kernels.mm(reinterpret_tensor(buf709, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg321_1, (2048, 1024), (1, 2048), 0), out=buf719)
            del arg321_1
            assert_size_stride(arg323_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg323_1 = copy_misaligned(arg323_1)
            buf720 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_170, view_74, value_states_24, values_24], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:428
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg323_1, buf719, buf720, 2098176, stream=stream0)
            del arg323_1
            buf722 = buf693; del buf693  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_168, view_72, hidden_states_339, pow_98, variance_97, add_196, rsqrt_97, hidden_states_340, to_201, mul_317, query_states_24, cos_27, mul_320, q_embed_24, getitem_155, hidden_states_343, key_24, getitem_156, hidden_states_344, value_24, attn_output_96], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:429
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf718, buf722, 4196352, stream=stream0)
            buf723 = buf692; del buf692  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_168, view_72, hidden_states_339, pow_98, variance_97, add_196, rsqrt_97, hidden_states_340, to_201, mul_317, query_states_24, cos_27, mul_320, q_embed_24, getitem_155, hidden_states_343, key_24, getitem_156, hidden_states_344, value_24, attn_output_96], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:430
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf720, buf723, 4196352, stream=stream0)
            buf724 = buf694; del buf694  # reuse
            buf753 = buf665; del buf665  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_168, view_72, hidden_states_339, pow_98, variance_97, add_196, rsqrt_97, hidden_states_340, to_201, mul_317, query_states_24, cos_27, mul_320, q_embed_24, getitem_155, hidden_states_343, key_24, getitem_156, hidden_states_344, value_24, attn_output_96, linear_175, view_75, hidden_states_353, pow_102, variance_101, add_204, rsqrt_101, hidden_states_354, to_209, mul_330, query_states_25, cos_28, mul_333, q_embed_25, getitem_161, hidden_states_357, key_25, getitem_162, hidden_states_358, value_25, attn_output_100], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:431
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf724, buf753, 2049, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_168, view_72, hidden_states_339, pow_98, variance_97, add_196, rsqrt_97, hidden_states_340, to_201, mul_317, query_states_24, cos_27, mul_320, q_embed_24, getitem_155, hidden_states_343, key_24, getitem_156, hidden_states_344, value_24, attn_output_96], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:432
            buf725 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf721, buf722, buf723, reinterpret_tensor(buf724, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf726 = buf725[0]
            assert_size_stride(buf726, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf726, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf725
            assert_size_stride(arg324_1, (2048, 2048), (2048, 1))
            buf730 = reinterpret_tensor(buf721, (1, 2048), (2048, 1), 0); del buf721  # reuse
            # Topologically Sorted Source Nodes: [transpose_100, reshape_74, attn_output_99], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:433
            extern_kernels.mm(reinterpret_tensor(buf726, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg324_1, (2048, 2048), (1, 2048), 0), out=buf730)
            del arg324_1
            assert_size_stride(arg325_1, (2048, ), (1, ))
            buf732 = reinterpret_tensor(buf726, (1, 1, 2048), (2048, 2048, 1), 0); del buf726  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_335, attn_output_99, hidden_states_345, hidden_states_346, pow_100, variance_99, add_201, rsqrt_99, hidden_states_347, to_205, hidden_states_348], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:434
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf701, buf707, buf730, arg325_1, buf732, 1, 2048, stream=stream0)
            del arg325_1
            assert_size_stride(arg326_1, (6144, 2048), (2048, 1))
            buf733 = reinterpret_tensor(buf706, (1, 6144), (6144, 1), 0); del buf706  # reuse
            # Topologically Sorted Source Nodes: [linear_172], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:435
            extern_kernels.mm(reinterpret_tensor(buf732, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg326_1, (2048, 6144), (1, 2048), 0), out=buf733)
            del arg326_1
            assert_size_stride(arg327_1, (6144, 2048), (2048, 1))
            buf734 = buf705; del buf705  # reuse
            # Topologically Sorted Source Nodes: [linear_173], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:436
            extern_kernels.mm(reinterpret_tensor(buf732, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg327_1, (2048, 6144), (1, 2048), 0), out=buf734)
            del arg327_1
            buf735 = reinterpret_tensor(buf733, (1, 1, 6144), (6144, 6144, 1), 0); del buf733  # reuse
            # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_326], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:437
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf735, buf734, 6144, stream=stream0)
            assert_size_stride(arg328_1, (2048, 6144), (6144, 1))
            buf736 = reinterpret_tensor(buf732, (1, 2048), (2048, 1), 0); del buf732  # reuse
            # Topologically Sorted Source Nodes: [linear_172, silu_24, linear_173, mul_326, down_proj_24], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:438
            extern_kernels.mm(reinterpret_tensor(buf735, (1, 6144), (0, 1), 0), reinterpret_tensor(arg328_1, (6144, 2048), (1, 6144), 0), out=buf736)
            del arg328_1
            assert_size_stride(arg329_1, (2048, ), (1, ))
            buf738 = buf709; del buf709  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_335, attn_output_99, hidden_states_345, down_proj_24, hidden_states_349, hidden_states_350, pow_101, variance_100, add_203, rsqrt_100, hidden_states_351, to_207, hidden_states_352], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:439
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf701, buf707, buf730, buf736, arg329_1, buf738, 1, 2048, stream=stream0)
            del arg329_1
            assert_size_stride(arg330_1, (2048, 2048), (2048, 1))
            buf739 = buf710; del buf710  # reuse
            # Topologically Sorted Source Nodes: [linear_175], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:440
            extern_kernels.mm(reinterpret_tensor(buf738, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg330_1, (2048, 2048), (1, 2048), 0), out=buf739)
            del arg330_1
            assert_size_stride(arg331_1, (128, ), (1, ))
            buf741 = reinterpret_tensor(buf648, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf648  # reuse
            buf750 = reinterpret_tensor(buf741, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf741  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_175, view_75, hidden_states_353, pow_102, variance_101, add_204, rsqrt_101, hidden_states_354, to_209, mul_330, query_states_25, cos_28, mul_333, x2_50, neg_50, x1_50, cat_101, sin_28, mul_334, q_embed_25, getitem_161, hidden_states_357, key_25, getitem_162, hidden_states_358, value_25, attn_output_100], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:441
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf750, buf739, arg331_1, arg4_1, 16, 128, stream=stream0)
            del arg331_1
            del buf739
            assert_size_stride(arg332_1, (1024, 2048), (2048, 1))
            buf742 = buf719; del buf719  # reuse
            # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:442
            extern_kernels.mm(reinterpret_tensor(buf738, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg332_1, (2048, 1024), (1, 2048), 0), out=buf742)
            del arg332_1
            assert_size_stride(arg333_1, (128, ), (1, ))
            buf747 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf746 = reinterpret_tensor(buf747, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_28, sin_28, linear_176, view_76, hidden_states_355, pow_103, variance_102, add_205, rsqrt_102, hidden_states_356, to_211, mul_332, key_states_25, mul_335, x2_51, neg_51, x1_51, cat_102, mul_336, k_embed_25], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:443
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf742, arg333_1, arg4_1, buf746, 8, 128, stream=stream0)
            del arg333_1
            assert_size_stride(arg335_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg335_1 = copy_misaligned(arg335_1)
            buf745 = reinterpret_tensor(buf747, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_28, linear_176, view_76, hidden_states_355, pow_103, variance_102, add_205, rsqrt_102, hidden_states_356, to_211, mul_332, key_states_25, mul_335, k_embed_25, keys_25], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:444
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg335_1, buf745, 2097152, stream=stream0)
            del arg335_1
            assert_size_stride(arg334_1, (1024, 2048), (2048, 1))
            buf748 = buf742; del buf742  # reuse
            # Topologically Sorted Source Nodes: [linear_177], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:445
            extern_kernels.mm(reinterpret_tensor(buf738, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg334_1, (2048, 1024), (1, 2048), 0), out=buf748)
            del arg334_1
            del buf738
            assert_size_stride(arg336_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg336_1 = copy_misaligned(arg336_1)
            buf749 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_177, view_77, value_states_25, values_25], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:446
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg336_1, buf748, buf749, 2098176, stream=stream0)
            del arg336_1
            buf751 = buf723; del buf723  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_175, view_75, hidden_states_353, pow_102, variance_101, add_204, rsqrt_101, hidden_states_354, to_209, mul_330, query_states_25, cos_28, mul_333, q_embed_25, getitem_161, hidden_states_357, key_25, getitem_162, hidden_states_358, value_25, attn_output_100], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:447
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf747, buf751, 4196352, stream=stream0)
            buf752 = buf722; del buf722  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_175, view_75, hidden_states_353, pow_102, variance_101, add_204, rsqrt_101, hidden_states_354, to_209, mul_330, query_states_25, cos_28, mul_333, q_embed_25, getitem_161, hidden_states_357, key_25, getitem_162, hidden_states_358, value_25, attn_output_100], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:448
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf749, buf752, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_175, view_75, hidden_states_353, pow_102, variance_101, add_204, rsqrt_101, hidden_states_354, to_209, mul_330, query_states_25, cos_28, mul_333, q_embed_25, getitem_161, hidden_states_357, key_25, getitem_162, hidden_states_358, value_25, attn_output_100], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:449
            buf754 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf750, buf751, buf752, reinterpret_tensor(buf753, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            buf755 = buf754[0]
            assert_size_stride(buf755, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf755, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf754
            assert_size_stride(arg337_1, (2048, 2048), (2048, 1))
            buf759 = reinterpret_tensor(buf750, (1, 2048), (2048, 1), 0); del buf750  # reuse
            # Topologically Sorted Source Nodes: [transpose_104, reshape_77, attn_output_103], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:450
            extern_kernels.mm(reinterpret_tensor(buf755, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg337_1, (2048, 2048), (1, 2048), 0), out=buf759)
            del arg337_1
            assert_size_stride(arg338_1, (2048, ), (1, ))
            buf760 = buf701; del buf701  # reuse
            buf762 = reinterpret_tensor(buf755, (1, 1, 2048), (2048, 2048, 1), 0); del buf755  # reuse
            # Topologically Sorted Source Nodes: [down_proj_23, hidden_states_335, attn_output_99, hidden_states_345, down_proj_24, hidden_states_349, attn_output_103, hidden_states_359, hidden_states_360, pow_104, variance_103, add_209, rsqrt_103, hidden_states_361, to_213, hidden_states_362], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:451
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf760, buf707, buf730, buf736, buf759, arg338_1, buf762, 1, 2048, stream=stream0)
            del arg338_1
            assert_size_stride(arg339_1, (6144, 2048), (2048, 1))
            buf763 = reinterpret_tensor(buf735, (1, 6144), (6144, 1), 0); del buf735  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_360, pow_104, variance_103, add_209, rsqrt_103, hidden_states_361, to_213, hidden_states_362, linear_179], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:452
            extern_kernels.mm(reinterpret_tensor(buf762, (1, 2048), (0, 1), 0), reinterpret_tensor(arg339_1, (2048, 6144), (1, 2048), 0), out=buf763)
            del arg339_1
            assert_size_stride(arg340_1, (6144, 2048), (2048, 1))
            buf764 = buf734; del buf734  # reuse
            # Topologically Sorted Source Nodes: [linear_180], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:453
            extern_kernels.mm(reinterpret_tensor(buf762, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg340_1, (2048, 6144), (1, 2048), 0), out=buf764)
            del arg340_1
            buf765 = reinterpret_tensor(buf763, (1, 1, 6144), (6144, 6144, 1), 0); del buf763  # reuse
            # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_339], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:454
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf765, buf764, 6144, stream=stream0)
            assert_size_stride(arg341_1, (2048, 6144), (6144, 1))
            buf766 = reinterpret_tensor(buf762, (1, 2048), (2048, 1), 0); del buf762  # reuse
            # Topologically Sorted Source Nodes: [linear_179, silu_25, linear_180, mul_339, down_proj_25], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:455
            extern_kernels.mm(reinterpret_tensor(buf765, (1, 6144), (0, 1), 0), reinterpret_tensor(arg341_1, (6144, 2048), (1, 6144), 0), out=buf766)
            del arg341_1
            assert_size_stride(arg342_1, (2048, ), (1, ))
            buf768 = reinterpret_tensor(buf759, (1, 1, 2048), (2048, 2048, 1), 0); del buf759  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_363, hidden_states_364, pow_105, variance_104, add_211, rsqrt_104, hidden_states_365, to_215, hidden_states_366], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11:456
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11.run(buf760, buf766, arg342_1, buf768, 1, 2048, stream=stream0)
            del arg342_1
            assert_size_stride(arg343_1, (2048, 2048), (2048, 1))
            buf769 = buf736; del buf736  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_363, hidden_states_364, pow_105, variance_104, add_211, rsqrt_104, hidden_states_365, to_215, hidden_states_366, linear_182], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:457
            extern_kernels.mm(reinterpret_tensor(buf768, (1, 2048), (0, 1), 0), reinterpret_tensor(arg343_1, (2048, 2048), (1, 2048), 0), out=buf769)
            del arg343_1
            assert_size_stride(arg344_1, (128, ), (1, ))
            buf771 = reinterpret_tensor(buf730, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf730  # reuse
            buf780 = reinterpret_tensor(buf771, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf771  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_182, view_78, hidden_states_367, pow_106, variance_105, add_212, rsqrt_105, hidden_states_368, to_217, mul_343, query_states_26, cos_29, mul_346, x2_52, neg_52, x1_52, cat_105, sin_29, mul_347, q_embed_26, getitem_167, hidden_states_371, key_26, getitem_168, hidden_states_372, value_26, attn_output_104], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:458
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf780, buf769, arg344_1, arg4_1, 16, 128, stream=stream0)
            del arg344_1
            assert_size_stride(arg345_1, (1024, 2048), (2048, 1))
            buf772 = buf748; del buf748  # reuse
            # Topologically Sorted Source Nodes: [linear_183], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:459
            extern_kernels.mm(reinterpret_tensor(buf768, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg345_1, (2048, 1024), (1, 2048), 0), out=buf772)
            del arg345_1
            assert_size_stride(arg346_1, (128, ), (1, ))
            buf777 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf776 = reinterpret_tensor(buf777, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_29, sin_29, linear_183, view_79, hidden_states_369, pow_107, variance_106, add_213, rsqrt_106, hidden_states_370, to_219, mul_345, key_states_26, mul_348, x2_53, neg_53, x1_53, cat_106, mul_349, k_embed_26], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:460
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf772, arg346_1, arg4_1, buf776, 8, 128, stream=stream0)
            del arg346_1
            assert_size_stride(arg348_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg348_1 = copy_misaligned(arg348_1)
            buf775 = reinterpret_tensor(buf777, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_29, linear_183, view_79, hidden_states_369, pow_107, variance_106, add_213, rsqrt_106, hidden_states_370, to_219, mul_345, key_states_26, mul_348, k_embed_26, keys_26], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:461
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg348_1, buf775, 2097152, stream=stream0)
            del arg348_1
            assert_size_stride(arg347_1, (1024, 2048), (2048, 1))
            buf778 = buf772; del buf772  # reuse
            # Topologically Sorted Source Nodes: [linear_184], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:462
            extern_kernels.mm(reinterpret_tensor(buf768, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg347_1, (2048, 1024), (1, 2048), 0), out=buf778)
            del arg347_1
            assert_size_stride(arg349_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg349_1 = copy_misaligned(arg349_1)
            buf779 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_184, view_80, value_states_26, values_26], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:463
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg349_1, buf778, buf779, 2098176, stream=stream0)
            del arg349_1
            buf781 = buf752; del buf752  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_182, view_78, hidden_states_367, pow_106, variance_105, add_212, rsqrt_105, hidden_states_368, to_217, mul_343, query_states_26, cos_29, mul_346, q_embed_26, getitem_167, hidden_states_371, key_26, getitem_168, hidden_states_372, value_26, attn_output_104], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:464
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf777, buf781, 4196352, stream=stream0)
            buf782 = buf751; del buf751  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_182, view_78, hidden_states_367, pow_106, variance_105, add_212, rsqrt_105, hidden_states_368, to_217, mul_343, query_states_26, cos_29, mul_346, q_embed_26, getitem_167, hidden_states_371, key_26, getitem_168, hidden_states_372, value_26, attn_output_104], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:465
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf779, buf782, 4196352, stream=stream0)
            buf783 = buf753; del buf753  # reuse
            buf812 = buf724; del buf724  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_182, view_78, hidden_states_367, pow_106, variance_105, add_212, rsqrt_105, hidden_states_368, to_217, mul_343, query_states_26, cos_29, mul_346, q_embed_26, getitem_167, hidden_states_371, key_26, getitem_168, hidden_states_372, value_26, attn_output_104, linear_189, view_81, hidden_states_381, pow_110, variance_109, add_220, rsqrt_109, hidden_states_382, to_225, mul_356, query_states_27, cos_30, mul_359, q_embed_27, getitem_173, hidden_states_385, key_27, getitem_174, hidden_states_386, value_27, attn_output_108], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6:466
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_6.run(arg3_1, buf783, buf812, 2049, stream=stream0)
            del arg3_1
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_182, view_78, hidden_states_367, pow_106, variance_105, add_212, rsqrt_105, hidden_states_368, to_217, mul_343, query_states_26, cos_29, mul_346, q_embed_26, getitem_167, hidden_states_371, key_26, getitem_168, hidden_states_372, value_26, attn_output_104], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:467
            buf784 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf780, buf781, buf782, reinterpret_tensor(buf783, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            del buf781
            del buf783
            buf785 = buf784[0]
            assert_size_stride(buf785, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf785, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf784
            assert_size_stride(arg350_1, (2048, 2048), (2048, 1))
            buf789 = reinterpret_tensor(buf780, (1, 2048), (2048, 1), 0); del buf780  # reuse
            # Topologically Sorted Source Nodes: [transpose_108, reshape_80, attn_output_107], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:468
            extern_kernels.mm(reinterpret_tensor(buf785, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg350_1, (2048, 2048), (1, 2048), 0), out=buf789)
            del arg350_1
            assert_size_stride(arg351_1, (2048, ), (1, ))
            buf791 = reinterpret_tensor(buf785, (1, 1, 2048), (2048, 2048, 1), 0); del buf785  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_363, attn_output_107, hidden_states_373, hidden_states_374, pow_108, variance_107, add_217, rsqrt_107, hidden_states_375, to_221, hidden_states_376], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12:469
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12.run(buf760, buf766, buf789, arg351_1, buf791, 1, 2048, stream=stream0)
            del arg351_1
            assert_size_stride(arg352_1, (6144, 2048), (2048, 1))
            buf792 = reinterpret_tensor(buf765, (1, 6144), (6144, 1), 0); del buf765  # reuse
            # Topologically Sorted Source Nodes: [linear_186], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:470
            extern_kernels.mm(reinterpret_tensor(buf791, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg352_1, (2048, 6144), (1, 2048), 0), out=buf792)
            del arg352_1
            assert_size_stride(arg353_1, (6144, 2048), (2048, 1))
            buf793 = buf764; del buf764  # reuse
            # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:471
            extern_kernels.mm(reinterpret_tensor(buf791, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg353_1, (2048, 6144), (1, 2048), 0), out=buf793)
            del arg353_1
            buf794 = reinterpret_tensor(buf792, (1, 1, 6144), (6144, 6144, 1), 0); del buf792  # reuse
            # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_352], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:472
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf794, buf793, 6144, stream=stream0)
            del buf793
            assert_size_stride(arg354_1, (2048, 6144), (6144, 1))
            buf795 = reinterpret_tensor(buf791, (1, 2048), (2048, 1), 0); del buf791  # reuse
            # Topologically Sorted Source Nodes: [linear_186, silu_26, linear_187, mul_352, down_proj_26], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:473
            extern_kernels.mm(reinterpret_tensor(buf794, (1, 6144), (0, 1), 0), reinterpret_tensor(arg354_1, (6144, 2048), (1, 6144), 0), out=buf795)
            del arg354_1
            del buf794
            assert_size_stride(arg355_1, (2048, ), (1, ))
            buf797 = buf768; del buf768  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_363, attn_output_107, hidden_states_373, down_proj_26, hidden_states_377, hidden_states_378, pow_109, variance_108, add_219, rsqrt_108, hidden_states_379, to_223, hidden_states_380], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13:474
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13.run(buf760, buf766, buf789, buf795, arg355_1, buf797, 1, 2048, stream=stream0)
            del arg355_1
            assert_size_stride(arg356_1, (2048, 2048), (2048, 1))
            buf798 = buf769; del buf769  # reuse
            # Topologically Sorted Source Nodes: [linear_189], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:475
            extern_kernels.mm(reinterpret_tensor(buf797, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg356_1, (2048, 2048), (1, 2048), 0), out=buf798)
            del arg356_1
            assert_size_stride(arg357_1, (128, ), (1, ))
            buf800 = reinterpret_tensor(buf707, (1, 16, 1, 128), (2048, 128, 2048, 1), 0); del buf707  # reuse
            buf809 = reinterpret_tensor(buf800, (1, 16, 1, 128), (2048, 128, 128, 1), 0); del buf800  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_189, view_81, hidden_states_381, pow_110, variance_109, add_220, rsqrt_109, hidden_states_382, to_225, mul_356, query_states_27, cos_30, mul_359, x2_54, neg_54, x1_54, cat_109, sin_30, mul_360, q_embed_27, getitem_173, hidden_states_385, key_27, getitem_174, hidden_states_386, value_27, attn_output_108], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1:476
            stream0 = get_raw_stream(0)
            triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_neg_new_ones_pow_rsqrt_scalar_tensor_sin_slice_transpose_unsqueeze_view_where_1.run(buf809, buf798, arg357_1, arg4_1, 16, 128, stream=stream0)
            del arg357_1
            del buf798
            assert_size_stride(arg358_1, (1024, 2048), (2048, 1))
            buf801 = buf778; del buf778  # reuse
            # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:477
            extern_kernels.mm(reinterpret_tensor(buf797, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg358_1, (2048, 1024), (1, 2048), 0), out=buf801)
            del arg358_1
            assert_size_stride(arg359_1, (128, ), (1, ))
            buf806 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            buf805 = reinterpret_tensor(buf806, (1, 8, 1, 128), (2098176, 262272, 128, 1), 262144)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, sin, sin_1, sin_2, cos_30, sin_30, linear_190, view_82, hidden_states_383, pow_111, variance_110, add_221, rsqrt_110, hidden_states_384, to_227, mul_358, key_states_27, mul_361, x2_55, neg_55, x1_55, cat_110, mul_362, k_embed_27], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.sin, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.slice, aten.neg]
            # [Provenance debug handles] triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2:478
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_rsqrt_sin_slice_transpose_unsqueeze_view_2.run(buf801, arg359_1, arg4_1, buf805, 8, 128, stream=stream0)
            del arg359_1
            del arg4_1
            assert_size_stride(arg361_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg361_1 = copy_misaligned(arg361_1)
            buf804 = reinterpret_tensor(buf806, (1, 8, 2048, 128), (2098176, 262272, 128, 1), 0)  # alias
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, cos_30, linear_190, view_82, hidden_states_383, pow_111, variance_110, add_221, rsqrt_110, hidden_states_384, to_227, mul_358, key_states_27, mul_361, k_embed_27, keys_27], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt]
            # [Provenance debug handles] triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3:479
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt_transpose_unsqueeze_view_3.run(arg361_1, buf804, 2097152, stream=stream0)
            del arg361_1
            assert_size_stride(arg360_1, (1024, 2048), (2048, 1))
            buf807 = buf801; del buf801  # reuse
            # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:480
            extern_kernels.mm(reinterpret_tensor(buf797, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg360_1, (2048, 1024), (1, 2048), 0), out=buf807)
            del arg360_1
            del buf797
            assert_size_stride(arg362_1, (1, 8, 2048, 128), (2097152, 128, 1024, 1))
            arg362_1 = copy_misaligned(arg362_1)
            buf808 = empty_strided_cuda((1, 8, 2049, 128), (2098176, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_191, view_83, value_states_27, values_27], Original ATen: [aten._unsafe_view, aten.view, aten.transpose, aten.cat]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_cat_transpose_view_4:481
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_cat_transpose_view_4.run(arg362_1, buf807, buf808, 2098176, stream=stream0)
            del arg362_1
            del buf807
            buf810 = buf782; del buf782  # reuse
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_189, view_81, hidden_states_381, pow_110, variance_109, add_220, rsqrt_109, hidden_states_382, to_225, mul_356, query_states_27, cos_30, mul_359, q_embed_27, getitem_173, hidden_states_385, key_27, getitem_174, hidden_states_386, value_27, attn_output_108], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:482
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf806, buf810, 4196352, stream=stream0)
            buf811 = empty_strided_cuda((1, 16, 2049, 128), (4196352, 262272, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_189, view_81, hidden_states_381, pow_110, variance_109, add_220, rsqrt_109, hidden_states_382, to_225, mul_356, query_states_27, cos_30, mul_359, q_embed_27, getitem_173, hidden_states_385, key_27, getitem_174, hidden_states_386, value_27, attn_output_108], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5:483
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_arange_bitwise_and_bmm_cat_clone_constant_pad_nd_cos_expand_index_le_mean_mul_new_ones_pow_rsqrt_scalar_tensor_slice_transpose_unsqueeze_view_where_5.run(buf808, buf811, 4196352, stream=stream0)
            # Topologically Sorted Source Nodes: [getitem_5, expand_1, arange, position_ids, position_ids_1, getitem_6, position_ids_expanded, matmul, freqs, emb, cos, cos_1, cos_2, result, arange_4, kv_arange, kv_indices, arange_3, q_arange, q_indices, le, result_1, attention_mask, batch_arange, batch_indices, getitem_4, result_2, attention_mask_1, linear_189, view_81, hidden_states_381, pow_110, variance_109, add_220, rsqrt_109, hidden_states_382, to_225, mul_356, query_states_27, cos_30, mul_359, q_embed_27, getitem_173, hidden_states_385, key_27, getitem_174, hidden_states_386, value_27, attn_output_108], Original ATen: [aten.unsqueeze, aten.expand, aten.arange, aten.add, aten._to_copy, aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten.new_ones, aten.le, aten.bitwise_and, aten.index, aten._unsafe_view, aten.view, aten.pow, aten.mean, aten.rsqrt, aten.clone, aten.scalar_tensor, aten.where, aten.constant_pad_nd, aten.slice, aten._scaled_dot_product_efficient_attention]
            # [Provenance debug handles] torch.ops.aten._scaled_dot_product_efficient_attention.default:484
            buf813 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf809, buf810, buf811, reinterpret_tensor(buf812, (1, 16, 1, 2049), (2056, 0, 2056, 1), 0), False, scale=0.08838834764831845)
            del buf810
            del buf811
            del buf812
            buf814 = buf813[0]
            assert_size_stride(buf814, (1, 16, 1, 128), (2048, 128, 2048, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf814, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf813
            assert_size_stride(arg363_1, (2048, 2048), (2048, 1))
            buf818 = reinterpret_tensor(buf809, (1, 2048), (2048, 1), 0); del buf809  # reuse
            # Topologically Sorted Source Nodes: [transpose_112, reshape_83, attn_output_111], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:485
            extern_kernels.mm(reinterpret_tensor(buf814, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg363_1, (2048, 2048), (1, 2048), 0), out=buf818)
            del arg363_1
            assert_size_stride(arg364_1, (2048, ), (1, ))
            buf819 = buf760; del buf760  # reuse
            buf821 = reinterpret_tensor(buf814, (1, 1, 2048), (2048, 2048, 1), 0); del buf814  # reuse
            # Topologically Sorted Source Nodes: [down_proj_25, hidden_states_363, attn_output_107, hidden_states_373, down_proj_26, hidden_states_377, attn_output_111, hidden_states_387, hidden_states_388, pow_112, variance_111, add_225, rsqrt_111, hidden_states_389, to_229, hidden_states_390], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14:486
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14.run(buf819, buf766, buf789, buf795, buf818, arg364_1, buf821, 1, 2048, stream=stream0)
            del arg364_1
            del buf766
            del buf789
            del buf795
            del buf818
            assert_size_stride(arg365_1, (6144, 2048), (2048, 1))
            buf822 = empty_strided_cuda((1, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_388, pow_112, variance_111, add_225, rsqrt_111, hidden_states_389, to_229, hidden_states_390, linear_193], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:487
            extern_kernels.mm(reinterpret_tensor(buf821, (1, 2048), (0, 1), 0), reinterpret_tensor(arg365_1, (2048, 6144), (1, 2048), 0), out=buf822)
            del arg365_1
            assert_size_stride(arg366_1, (6144, 2048), (2048, 1))
            buf823 = empty_strided_cuda((1, 6144), (6144, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_194], Original ATen: [aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:488
            extern_kernels.mm(reinterpret_tensor(buf821, (1, 2048), (2048, 1), 0), reinterpret_tensor(arg366_1, (2048, 6144), (1, 2048), 0), out=buf823)
            del arg366_1
            buf824 = reinterpret_tensor(buf822, (1, 1, 6144), (6144, 6144, 1), 0); del buf822  # reuse
            # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_365], Original ATen: [aten._unsafe_view, aten.silu, aten.mul]
            # [Provenance debug handles] triton_poi_fused__unsafe_view_mul_silu_8:489
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_mul_silu_8.run(buf824, buf823, 6144, stream=stream0)
            del buf823
            assert_size_stride(arg367_1, (2048, 6144), (6144, 1))
            buf825 = reinterpret_tensor(buf821, (1, 2048), (2048, 1), 0); del buf821  # reuse
            # Topologically Sorted Source Nodes: [linear_193, silu_27, linear_194, mul_365, down_proj_27], Original ATen: [aten._unsafe_view, aten.silu, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:490
            extern_kernels.mm(reinterpret_tensor(buf824, (1, 6144), (0, 1), 0), reinterpret_tensor(arg367_1, (6144, 2048), (1, 6144), 0), out=buf825)
            del arg367_1
            del buf824
            assert_size_stride(arg368_1, (2048, ), (1, ))
            buf827 = buf819; del buf819  # reuse
            # Topologically Sorted Source Nodes: [down_proj_27, hidden_states_391, hidden_states_392, pow_113, variance_112, add_227, rsqrt_112, hidden_states_393, to_231, hidden_states_394], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            # [Provenance debug handles] triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15:491
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf827, buf825, arg368_1, 1, 2048, stream=stream0)
            del arg368_1
            del buf825
            buf828 = empty_strided_cuda((1, 151936), (151936, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [down_proj_27, hidden_states_391, hidden_states_392, pow_113, variance_112, add_227, rsqrt_112, hidden_states_393, to_231, hidden_states_394, logits], Original ATen: [aten._unsafe_view, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.view, aten.t, aten.mm]
            # [Provenance debug handles] extern_kernels.mm:492
            extern_kernels.mm(reinterpret_tensor(buf827, (1, 2048), (0, 1), 0), reinterpret_tensor(arg1_1, (2048, 151936), (1, 2048), 0), out=buf828)
            del arg1_1
            del buf827
        return (buf12, buf10, buf41, buf39, buf71, buf69, buf100, buf98, buf130, buf128, buf159, buf157, buf189, buf187, buf218, buf216, buf248, buf246, buf277, buf275, buf307, buf305, buf336, buf334, buf366, buf364, buf395, buf393, buf425, buf423, buf454, buf452, buf484, buf482, buf513, buf511, buf543, buf541, buf572, buf570, buf602, buf600, buf631, buf629, buf661, buf659, buf690, buf688, buf720, buf718, buf749, buf747, buf779, buf777, buf808, buf806, reinterpret_tensor(buf828, (1, 1, 151936), (151936, 151936, 1), 0), )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((151936, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 2049), (2049, 1), device='cuda:0', dtype=torch.int64)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg198_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg199_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg200_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg201_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg202_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg203_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg204_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg205_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg206_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg207_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg208_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg209_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg210_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg211_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg212_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg213_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg214_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg215_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg216_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg217_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg218_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg219_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg220_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg221_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg222_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg223_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg224_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg225_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg226_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg227_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg228_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg230_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg231_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg232_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg233_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg234_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg235_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg236_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg237_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg238_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg239_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg240_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg241_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg242_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg243_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg244_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg245_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg246_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg247_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg248_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg249_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg250_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg251_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg252_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg254_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg255_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg256_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg257_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg258_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg259_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg260_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg261_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg262_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg263_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg264_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg265_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg266_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg267_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg268_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg269_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg270_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg271_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg272_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg273_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg274_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg275_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg276_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg277_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg278_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg280_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg281_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg282_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg283_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg284_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg285_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg286_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg287_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg288_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg289_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg290_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg291_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg292_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg293_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg294_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg295_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg296_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg297_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg298_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg299_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg300_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg301_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg302_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg303_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg304_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg305_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg306_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg307_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg308_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg309_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg310_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg311_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg312_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg313_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg314_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg315_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg316_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg317_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg318_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg319_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg320_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg321_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg322_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg323_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg324_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg325_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg326_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg327_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg328_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg329_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg330_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg331_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg332_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg334_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg335_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg336_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg337_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg338_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg339_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg340_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg341_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg342_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg343_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg344_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg345_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg346_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg347_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg348_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg349_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg350_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg351_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg352_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg353_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg354_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg355_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg356_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg358_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg359_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg360_1 = rand_strided((1024, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg361_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg362_1 = rand_strided((1, 8, 2048, 128), (2097152, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    arg363_1 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg364_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg365_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg366_1 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.bfloat16)
    arg367_1 = rand_strided((2048, 6144), (6144, 1), device='cuda:0', dtype=torch.bfloat16)
    arg368_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
