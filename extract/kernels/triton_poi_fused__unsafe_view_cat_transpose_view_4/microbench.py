"""Microbenchmark stub for triton_poi_fused__unsafe_view_cat_transpose_view_4.

This is a *template* — input construction is not wired up yet. The next stage
of the experiment will fill in the launch grid + buffer allocation.
"""
import torch
import triton

SAMPLE_INPUTS = {
  "Input Dims": [
    [
      1,
      8,
      512,
      128
    ],
    [
      1,
      1024
    ],
    [
      1,
      8,
      513,
      128
    ],
    []
  ],
  "Input Strides": [
    [
      524288,
      128,
      1024,
      1
    ],
    [
      1024,
      1
    ],
    [
      525312,
      65664,
      128,
      1
    ],
    []
  ],
  "Input type": [
    "c10::BFloat16",
    "c10::BFloat16",
    "c10::BFloat16",
    "Scalar"
  ],
  "kernel_file": "/tmp/torchinductor_ianbarber/m4/cm4fu53a23u3fldseoh5m6vuhajdnwtyeak3mnnbkcpvkxqperp7.py",
  "kernel_hash": "cm4fu53a23u3fldseoh5m6vuhajdnwtyeak3mnnbkcpvkxqperp7",
  "num_warps": 4,
  "num_stages": 1,
  "kernel_kwargs": "XBLOCK=1024"
}

# TODO: import kernel from kernel.py once the launcher is generated.
# from kernel import triton_poi_fused__unsafe_view_cat_transpose_view_4

def make_inputs():
    raise NotImplementedError("Stage 3 will generate input tensors here.")

def reference(*args):
    raise NotImplementedError("Reference: load inductor's output_code.py and call its launcher.")

def candidate(*args):
    raise NotImplementedError("Candidate: agent-generated replacement.")
