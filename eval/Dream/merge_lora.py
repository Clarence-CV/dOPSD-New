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

    peft_model = PeftModel.from_pretrained(base_model, "/home/tuan/local_home/ptuandat/on_policy_self_distill_dLLM/outputs/opsd_dllm_trajectory/dream7b_traj05-least-allfut_mixchain_reverseKL_filterwrong_v2/checkpoint-600")

    merged_model = peft_model.merge_and_unload()
    
    merged_model.save_pretrained("/home/tuan/local_home/ptuandat/on_policy_self_distill_dLLM/outputs/opsd_dllm_trajectory/eval_checkpoint/Dream/dream7b_traj05-least-allfut_mixchain_reverseKL_filterwrong_v2_500")
    tokenizer.save_pretrained("/home/tuan/local_home/ptuandat/on_policy_self_distill_dLLM/outputs/opsd_dllm_trajectory/eval_checkpoint/Dream/dream7b_traj05-least-allfut_mixchain_reverseKL_filterwrong_v2_500")


if __name__ == "__main__":
    main()
