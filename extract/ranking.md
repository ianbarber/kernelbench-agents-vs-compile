# Kernel wall-time ranking (cudagraphs OFF profiler traces)

## prefill_512_b1

- Trace: `baselines/results/traces/default_prefill_512_b1.json`
- Total kernel-event time: 1785.05 ms
- Triton (inductor codegen): 902.35 ms (50.6%), 14 distinct kernels
- Aten (cuBLAS/cuDNN/SDPA backends): 882.70 ms (49.4%), 2 distinct ops
- Coverage: top-5 = 94.4%  top-10 = 99.9%  top-20 = 100.0%

### Top 20 by walltime

| rank | kind | walltime_ms | %total | calls | mean_us | name |
|---:|---|---:|---:|---:|---:|---|
| 1 | aten | 876.098 | 49.08 | 1970 | 444.72 | `aten::mm` |
| 2 | triton | 446.017 | 24.99 | 560 | 796.46 | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_aran...` |
| 3 | triton | 164.875 | 9.24 | 140 | 1177.68 | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_aran...` |
| 4 | triton | 101.168 | 5.67 | 280 | 361.32 | `triton_poi_fused__unsafe_view_mul_silu_6` |
| 5 | triton | 96.094 | 5.38 | 280 | 343.19 | `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_r...` |
| 6 | triton | 33.766 | 1.89 | 130 | 259.74 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11` |
| 7 | triton | 33.647 | 1.88 | 130 | 258.83 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_9` |
| 8 | triton | 20.785 | 1.16 | 280 | 74.23 | `triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_aran...` |
| 9 | aten | 6.604 | 0.37 | 280 | 23.59 | `aten::_efficient_attention_forward` |
| 10 | triton | 3.753 | 0.21 | 10 | 375.33 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13` |
| 11 | triton | 1.144 | 0.06 | 130 | 8.80 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_10` |
| 12 | triton | 0.792 | 0.04 | 130 | 6.09 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12` |
| 13 | triton | 0.115 | 0.01 | 10 | 11.48 | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0` |
| 14 | triton | 0.065 | 0.00 | 10 | 6.48 | `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_8` |
| 15 | triton | 0.063 | 0.00 | 10 | 6.29 | `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7` |
| 16 | triton | 0.062 | 0.00 | 10 | 6.24 | `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_5` |

## decode_ctx512_b1

- Trace: `baselines/results/traces/default_decode_ctx512_b1.json`
- Total kernel-event time: 188.23 ms
- Triton (inductor codegen): 76.54 ms (40.7%), 16 distinct kernels
- Aten (cuBLAS/cuDNN/SDPA backends): 111.69 ms (59.3%), 4 distinct ops
- Coverage: top-5 = 70.5%  top-10 = 95.3%  top-20 = 100.0%

### Top 20 by walltime

| rank | kind | walltime_ms | %total | calls | mean_us | name |
|---:|---|---:|---:|---:|---:|---|
| 1 | aten | 52.817 | 28.06 | 1970 | 26.81 | `aten::mm` |
| 2 | aten | 26.567 | 14.11 | 280 | 94.88 | `aten::_efficient_attention_forward` |
| 3 | triton | 21.000 | 11.16 | 560 | 37.50 | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_aran...` |
| 4 | aten | 16.717 | 8.88 | 560 | 29.85 | `aten::clone` |
| 5 | aten | 15.588 | 8.28 | 560 | 27.84 | `aten::copy_` |
| 6 | triton | 14.416 | 7.66 | 140 | 102.97 | `triton_poi_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_aran...` |
| 7 | triton | 11.598 | 6.16 | 130 | 89.22 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_14` |
| 8 | triton | 11.577 | 6.15 | 130 | 89.05 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_12` |
| 9 | triton | 5.538 | 2.94 | 280 | 19.78 | `triton_poi_fused__unsafe_view_cat_transpose_view_4` |
| 10 | triton | 3.516 | 1.87 | 280 | 12.56 | `triton_per_fused__scaled_dot_product_efficient_attention__to_copy__unsafe_view_add_aran...` |
| 11 | triton | 2.677 | 1.42 | 280 | 9.56 | `triton_poi_fused__unsafe_view_mul_silu_8` |
| 12 | triton | 1.563 | 0.83 | 280 | 5.58 | `triton_per_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_neg_pow_r...` |
| 13 | triton | 1.145 | 0.61 | 280 | 4.09 | `triton_poi_fused__to_copy__unsafe_view_add_arange_bmm_cat_cos_expand_mean_mul_pow_rsqrt...` |
| 14 | triton | 0.994 | 0.53 | 10 | 99.45 | `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_7` |
| 15 | triton | 0.982 | 0.52 | 10 | 98.16 | `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_10` |
| 16 | triton | 0.694 | 0.37 | 130 | 5.34 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_13` |
| 17 | triton | 0.629 | 0.33 | 130 | 4.84 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_11` |
| 18 | triton | 0.097 | 0.05 | 10 | 9.66 | `triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_0` |
| 19 | triton | 0.064 | 0.03 | 10 | 6.43 | `triton_red_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_9` |
| 20 | triton | 0.054 | 0.03 | 10 | 5.44 | `triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15` |

## Summary

| workload | triton% | aten% | top5% | top10% | top20% | #triton | #aten |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill_512_b1 | 50.6 | 49.4 | 94.4 | 99.9 | 100.0 | 14 | 2 |
| decode_ctx512_b1 | 40.7 | 59.3 | 70.5 | 95.3 | 100.0 | 16 | 4 |
