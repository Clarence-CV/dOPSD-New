############################################### gsm8k evaluations ###############################################

## Original dllm
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks gsm8k_cot_zeroshot \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/gsm8k \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

## our dParallel
# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/merged_checkpoint_6k6",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy" \
#     --tasks gsm8k_cot_zeroshot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

############################################### minerva_math evaluations ###############################################

## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# ## our dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.45 \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template


############################################### humaneval evaluations ###############################################

## Original dllm
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks humaneval_instruct \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce/humaneval \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

## our dParallel
# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. accelerate launch --num_processes 4 --num_machines 1 --mixed_precision no --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="/home/stud_dat/on_policy_self_distill_dLLM/outputs/sft_dllm/merged_checkpoint_6k6",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.5 \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template



############################################### mbpp evaluations ###############################################

## Original dllm
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# ## our dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained="Zigeng/dParallel_Dream_7B_Instruct",trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.5 \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template





