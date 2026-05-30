import torch
import yaml
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from peft import (
    PeftModel,
)

from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():

    name = "Dream-org/Dream-v0-Instruct-7B"

    device = 'cuda'

    base_model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    peft_model = PeftModel.from_pretrained(base_model, "/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/dream7b-fix075-2epochs-30k/sft_dllm_dream7b_fix075_2epochs_30k/checkpoint-6600")

    merged_model = peft_model.merge_and_unload()
    
    merged_model.save_pretrained("/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/merged_checkpoint_6k6")
    tokenizer.save_pretrained("/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/merged_checkpoint_6k6")


if __name__ == "__main__":
    main()