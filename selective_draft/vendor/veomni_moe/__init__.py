"""Fused MoE kernels vendored from VeOmni (https://github.com/ByteDance-Seed/VeOmni, Apache-2.0, see LICENSE) at
commit 600fe6d7442392fd3ddefad4b6c8d3c0002fed1c, the submodule pinned by DMax (czg1225/DMax @43fafda, dFactory/VeOmni).
DMax trains LLaDA2.0-mini through veomni.ops.fused_moe_forward on stacked expert weights
(dFactory/models/llada2_moe/modeling_llada2_moe.py: LLaDA2MoeExperts); this package keeps that function and the
Triton group-GEMM / scatter / gather kernels it needs.

Files under ops/group_gemm/ are unmodified copies. Modified files (each says how in its header): ops/fused_moe.py
(single-device path only), utils/device.py and utils/logging.py (minimal replacements for veomni.utils).
Installing the VeOmni package itself is not an option here: it pins torch 2.7.1 and transformers < 4.57.
"""
