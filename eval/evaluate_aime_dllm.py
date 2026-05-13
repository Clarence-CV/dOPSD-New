import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last answer inside a LaTeX \\boxed{...} block."""
    if not text:
        return None

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        elif text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None

    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[7:-1].strip()
    return None


def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    """Grade predicted answer against ground truth with math_verify fallback."""
    if predicted is None:
        return False

    try:
        pred_latex = predicted if "$" in predicted else f"${predicted}$"
        gt_latex = ground_truth if "$" in ground_truth else f"${ground_truth}$"
        pred_parsed = parse(pred_latex, fallback_mode="no_fallback")
        gt_parsed = parse(gt_latex, fallback_mode="no_fallback")
        return verify(gt_parsed, pred_parsed, timeout_seconds=5)
    except Exception:
        pred_norm = predicted.replace("$", "").replace(" ", "").lower().strip()
        gt_norm = ground_truth.replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def resolve_dtype(dtype_name: str):
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name == "auto":
        return "auto"
    try:
        return dtype_map[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported --torch_dtype {dtype_name!r}") from exc


def load_aime_dataset(dataset_name: str, num_samples: int | None = None):
    dataset_key = dataset_name.lower()
    if dataset_key == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        print(f"Loaded HuggingFaceH4/aime_2024 with {len(dataset)} problems")
    elif dataset_key == "aime25":
        dataset = load_dataset("yentinglin/aime_2025", split="train", trust_remote_code=True)
        print(f"Loaded yentinglin/aime_2025 with {len(dataset)} problems")
    else:
        raise ValueError("Only AIME datasets are supported here. Choose 'aime24' or 'aime25'.")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    examples = []
    for idx, row in enumerate(dataset):
        if dataset_key == "aime24":
            question_id = row.get("id", idx)
        else:
            question_id = row.get("problem_idx", idx)
        examples.append(
            {
                "problem_id": question_id,
                "problem": row["problem"],
                "ground_truth": str(row["answer"]),
            }
        )
    return examples


def build_prompt(tokenizer, problem: str, use_chat_template: bool = True) -> str:
    user_message = (
        f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    if not use_chat_template:
        return user_message

    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        print(f"Warning: chat template failed ({exc}); falling back to raw prompt.")
        return user_message


def load_dllm_model(
    base_model: str,
    checkpoint_dir: str | None = None,
    torch_dtype: str = "bfloat16",
    attn_implementation: str | None = "sdpa",
    device_map: str | None = None,
):
    print(f"Loading dLLM base model from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(torch_dtype),
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if device_map:
        model_kwargs["device_map"] = device_map

    try:
        model = AutoModel.from_pretrained(base_model, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(base_model, **model_kwargs)

    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        print(f"Loading PEFT adapter from: {checkpoint_dir}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint_dir)

    if not device_map:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}; dtype={next(model.parameters()).dtype}")
    print(f"mask_token_id={tokenizer.mask_token_id}, pad_token_id={tokenizer.pad_token_id}")
    return model, tokenizer


def trim_after_eos(token_ids: torch.Tensor, eos_token_id: int | None, pad_token_id: int | None):
    keep = torch.ones_like(token_ids, dtype=torch.bool)
    if eos_token_id is not None:
        eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            first = int(eos_positions[0].item())
            keep[first + 1 :] = False
    if pad_token_id is not None:
        keep[token_ids == pad_token_id] = False
    return token_ids[keep]


def decode_completion(tokenizer, full_sequence: torch.Tensor, prompt_len: int) -> str:
    completion_ids = full_sequence[prompt_len:].detach().cpu()
    completion_ids = trim_after_eos(completion_ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    diffusion_steps: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
    generator: str,
):
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    ).to(device)
    prompt_len = encoded["input_ids"].shape[1]

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if generator == "auto":
            generator = "diffusion" if hasattr(model, "diffusion_generate") else "generate"

        if generator == "diffusion":
            if not hasattr(model, "diffusion_generate"):
                raise AttributeError("Model does not expose diffusion_generate; use --generator generate.")
            gen_out = model.diffusion_generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=diffusion_steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
            )
            sequences = gen_out.sequences
        else:
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
            if tokenizer.mask_token_id is not None:
                generate_kwargs["mask_token_id"] = tokenizer.mask_token_id
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
            sequences = model.generate(encoded["input_ids"], **generate_kwargs)

    return [decode_completion(tokenizer, seq, prompt_len) for seq in sequences]


def evaluate_aime_dllm(
    model,
    tokenizer,
    dataset_name: str,
    max_new_tokens: int,
    diffusion_steps: int,
    temperature: float,
    top_p: float,
    alg: str,
    alg_temp: float,
    val_n: int,
    batch_size: int,
    num_samples: int | None,
    output_file: str | None,
    base_model: str,
    checkpoint_dir: str | None,
    use_chat_template: bool,
    generator: str,
):
    examples = load_aime_dataset(dataset_name, num_samples)
    prompts = [build_prompt(tokenizer, ex["problem"], use_chat_template) for ex in examples]

    print("\n" + "=" * 70)
    print("DLLM AIME EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Problems: {len(examples)}")
    print(f"Val-N: {val_n}")
    print(f"Batch size: {batch_size}")
    print(f"Generator: {generator}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Diffusion steps: {diffusion_steps}")
    print(f"Temperature: {temperature}")
    print(f"Top-p: {top_p}")
    print(f"Alg: {alg}, alg_temp: {alg_temp}")
    print("=" * 70 + "\n")

    results = []
    pass_at_n = 0
    total_correct = 0
    formatted_count = 0
    total_generations = 0

    for start in tqdm(range(0, len(examples), batch_size), desc="AIME problems"):
        batch_examples = examples[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]

        repeated_prompts = []
        owner_indices = []
        for local_idx, prompt in enumerate(batch_prompts):
            repeated_prompts.extend([prompt] * val_n)
            owner_indices.extend([local_idx] * val_n)

        generations_by_problem = [[] for _ in batch_examples]
        for gen_start in range(0, len(repeated_prompts), batch_size):
            gen_prompts = repeated_prompts[gen_start : gen_start + batch_size]
            gen_owner_indices = owner_indices[gen_start : gen_start + batch_size]
            texts = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=gen_prompts,
                max_new_tokens=max_new_tokens,
                diffusion_steps=diffusion_steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                generator=generator,
            )
            for owner_idx, text in zip(gen_owner_indices, texts):
                generations_by_problem[owner_idx].append(text)

        for local_idx, (example, generations) in enumerate(zip(batch_examples, generations_by_problem)):
            predicted_answers = []
            is_correct_list = []
            is_formatted_list = []

            for text in generations:
                predicted_answer = extract_boxed_answer(text)
                is_formatted = predicted_answer is not None
                is_correct = grade_answer(predicted_answer, example["ground_truth"])
                predicted_answers.append(predicted_answer if predicted_answer else "[No boxed answer found]")
                is_formatted_list.append(is_formatted)
                is_correct_list.append(is_correct)

            num_correct = sum(is_correct_list)
            num_formatted = sum(is_formatted_list)
            has_correct = any(is_correct_list)
            majority_vote_correct = False
            formatted_predictions = [
                pred for pred, is_formatted in zip(predicted_answers, is_formatted_list) if is_formatted
            ]
            if formatted_predictions:
                majority_answer = Counter(formatted_predictions).most_common(1)[0][0]
                majority_vote_correct = grade_answer(majority_answer, example["ground_truth"])

            if has_correct:
                pass_at_n += 1
            total_correct += num_correct
            formatted_count += num_formatted
            total_generations += val_n

            global_idx = start + local_idx
            results.append(
                {
                    "problem_id": example["problem_id"],
                    "problem": example["problem"],
                    "ground_truth": example["ground_truth"],
                    "val_n": val_n,
                    "generations": [
                        {
                            "predicted_answer": pred,
                            "full_generation": gen,
                            "correct": corr,
                            "formatted": fmt,
                        }
                        for pred, gen, corr, fmt in zip(
                            predicted_answers, generations, is_correct_list, is_formatted_list
                        )
                    ],
                    "num_correct": num_correct,
                    "pass_at_n": has_correct,
                    "majority_vote_correct": majority_vote_correct,
                    "predicted_answer": predicted_answers[0],
                    "full_generation": generations[0],
                    "correct": is_correct_list[0],
                    "formatted": is_formatted_list[0],
                }
            )

            current_pass = pass_at_n / (global_idx + 1) * 100
            current_avg = total_correct / total_generations * 100
            current_format = formatted_count / total_generations * 100
            status = "correct" if has_correct else "wrong"
            print(
                f"[{global_idx + 1}/{len(examples)}] {status} | "
                f"Pass@{val_n}: {current_pass:.1f}% | "
                f"Avg@{val_n}: {current_avg:.1f}% | "
                f"Formatted: {current_format:.1f}%"
            )

    num_problems = len(examples)
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct / total_generations * 100
    majority_vote_count = sum(1 for row in results if row["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_count / num_problems * 100
    format_rate = formatted_count / total_generations * 100

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Dataset: {dataset_name.upper()}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct}/{total_generations})")
    print(f"Majority Vote@{val_n}: {majority_vote_at_n_pct:.2f}% ({majority_vote_count}/{num_problems})")
    print(f"Format rate: {format_rate:.2f}% ({formatted_count}/{total_generations})")
    print("=" * 70)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "base_model": base_model,
            "checkpoint_dir": checkpoint_dir,
            "dataset": dataset_name,
            "generator": generator,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "diffusion_steps": diffusion_steps,
            "alg": alg,
            "alg_temp": alg_temp,
            "val_n": val_n,
            "batch_size": batch_size,
            "num_problems": num_problems,
            "total_solutions": total_generations,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "results": results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion LLMs on AIME.")
    parser.add_argument("--base_model", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="aime24", choices=["aime24", "aime25"])
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--diffusion_steps", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg_temp", type=float, default=0.0)
    parser.add_argument("--val_n", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument(
        "--generator",
        type=str,
        default="auto",
        choices=["auto", "diffusion", "generate"],
        help="auto uses diffusion_generate when the model exposes it, otherwise generate.",
    )
    parser.add_argument("--no_chat_template", action="store_true")
    args = parser.parse_args()

    if args.output_file is None:
        parts = ["dllm_eval_results", args.dataset, Path(args.base_model).name]
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            parts += [checkpoint_path.parent.name, checkpoint_path.name]
        parts += [
            f"temp{args.temperature}",
            f"steps{args.diffusion_steps}",
            f"valn{args.val_n}",
        ]
        args.output_file = str(Path("eval_results") / ("_".join(parts) + ".json"))

    model, tokenizer = load_dllm_model(
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )

    average_at_n_pct, _ = evaluate_aime_dllm(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        max_new_tokens=args.max_new_tokens,
        diffusion_steps=args.diffusion_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        alg=args.alg,
        alg_temp=args.alg_temp,
        val_n=args.val_n,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_file=args.output_file,
        base_model=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        use_chat_template=not args.no_chat_template,
        generator=args.generator,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
