import torch
from transformers import AutoModel, AutoTokenizer

model_name = 'GSAI-ML/LLaDA-8B-Base'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).to("cuda")


prompt = "Implement a PyTorch trainer class for a toy dataset.\n\npython\nimport torch\nimport torch.nn as nn\n"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# Giải pháp mạnh: Gọi trực tiếp hàm generate của Class LLaDA để tránh trình kiểm tra của transformers
output_ids = model.generate(
    input_ids,
    mask_token_id=tokenizer.mask_token_id,
    temperature=0.7
)

print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])