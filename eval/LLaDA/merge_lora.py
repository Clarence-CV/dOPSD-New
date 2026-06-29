"""Merge a trained LLaDA LoRA adapter into the base model for evaluation.

eval_llada.py loads a FULL model via LLaDAModelLM.from_pretrained, so the LoRA
adapter must be merged first. The adapter was trained on the model loaded with
AutoModelForCausalLM (see opsd_dllm_trajectory_train.py), so we merge onto the
same class to keep the LM head intact.

Usage:
    python merge_lora.py \
        --adapter /abs/path/to/lora_checkpoint \
        --output  /abs/path/to/merged_instruct_merge \
        [--base GSAI-ML/LLaDA-8B-Instruct]

Keep "instruct" in the --output dir name so eval_llada.py auto-detects the
instruct chat template.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Base model the LoRA was trained on.")
    parser.add_argument("--adapter", required=True,
                        help="Path to the trained LoRA adapter checkpoint dir.")
    parser.add_argument("--output", required=True,
                        help="Where to save the merged full model (keep 'instruct' in the name).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"[merge_lora] base={args.base}\n[merge_lora] adapter={args.adapter}\n[merge_lora] output={args.output}")

    # Match training: LLaDA is loaded as a causal LM (keeps the LM head).
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[merge_lora] Saved merged model + tokenizer to {args.output}")


if __name__ == "__main__":
    main()
