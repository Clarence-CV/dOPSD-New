"""CPU-test copy of the official LLaDA2.0-mini remote code (Apache-2.0, inclusionAI).

Source: https://huggingface.co/inclusionAI/LLaDA2.0-mini at revision
dad945cac317da394b390f82c7b40691d8a881ed, files modeling_llada2_moe.py and configuration_llada2_moe.py,
copied unmodified. Only `selective_draft.testing.make_fake_llada2` imports it, to build a tiny randomly
initialised model with the real block-causal attention and MoE code. Real runs load the remote code
from the HF cache (trust_remote_code), never this copy.
"""
