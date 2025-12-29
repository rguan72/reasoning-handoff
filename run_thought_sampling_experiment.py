"""
Experiment: Thought Sampling Consistency Analysis

For each question in rguan72/chicken-scratch-tests:
1. Sample 10 CoTs per question for qwen and nvidia
2. Run analyze_thought_sampling on each CoT
3. Order results by difference in answer spread between on-policy and off-policy

"Answer spread" is measured as the entropy of the answer distribution - higher means
more diverse answers. We compare on-policy spread vs off-policy spread.
"""

import json
import math
import gc
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model_configs import MODEL_CONFIGS, VALID_MODEL_NAMES
from thought_sampling_handoff import (
    analyze_thought_sampling,
    split_sentences,
    build_histogram,
    extract_answer,
)


@dataclass
class CoTSample:
    """A single chain-of-thought sample."""
    question_id: str
    problem: str
    ground_truth_answer: str
    source_model: str  # Model that generated this CoT
    cot_text: str
    extracted_answer: Optional[str]


@dataclass
class ThoughtSamplingResult:
    """Result from analyze_thought_sampling for a single CoT."""
    cot_sample: CoTSample
    summary: Dict[str, Any]
    on_policy_spread: float  # Entropy of on-policy answers
    off_policy_spread: float  # Entropy of off-policy answers
    spread_difference: float  # on_policy - off_policy


def compute_entropy(histogram: Dict[str, int]) -> float:
    """
    Compute Shannon entropy of answer distribution.

    Higher entropy = more spread/diverse answers.
    """
    total = sum(histogram.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in histogram.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def compute_answer_spread(histogram: Dict[str, int]) -> float:
    """
    Compute answer spread metric.

    Uses entropy as the spread measure.
    """
    return compute_entropy(histogram)


def construct_prompt(problem: str, model_key: str, tokenizer=None) -> str:
    """Construct prompt for CoT generation."""
    config = MODEL_CONFIGS[model_key]
    user_content = f"Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.\n\nProblem: {problem}"

    if config["use_chat_template"] and tokenizer is not None:
        messages = []
        if config["system_prompt"]:
            messages.append({"role": "system", "content": config["system_prompt"]})
        messages.append({"role": "user", "content": user_content})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = f"{user_content}\n<think>"

    return prompt


def extract_cot_from_completion(completion: str, model_key: str) -> str:
    """
    Extract the chain-of-thought portion from a completion.

    For models that use <think>...</think> tags, extract content inside.
    Otherwise, return the full completion.
    """
    # Check for think tags
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")

    if think_start != -1 and think_end != -1 and think_end > think_start:
        # Extract content between tags
        return completion[think_start + len("<think>"):think_end].strip()
    elif think_start != -1:
        # Has opening but no closing - take everything after opening
        return completion[think_start + len("<think>"):].strip()
    else:
        # No think tags - use everything before boxed answer or full text
        boxed_pos = completion.find("\\boxed")
        if boxed_pos != -1:
            return completion[:boxed_pos].strip()
        return completion.strip()


def sample_cots_for_questions(
    questions: List[Dict],
    model_key: str,
    num_samples: int = 10,
    seed: Optional[int] = None,
) -> List[CoTSample]:
    """
    Sample CoTs for all questions using a single model.

    Uses vLLM for efficient batched inference.
    """
    config = MODEL_CONFIGS[model_key]

    print(f"\nLoading {config['name']} for CoT sampling...")

    llm_kwargs = {
        "model": config["model_path"],
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "dtype": config["dtype"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "max_model_len": config["max_model_len"],
        "max_num_batched_tokens": config["max_num_batched_tokens"],
    }
    if seed is not None:
        llm_kwargs["seed"] = seed

    llm = LLM(**llm_kwargs)

    tokenizer = None
    if config["use_chat_template"]:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True
        )

    sampling_params = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
    )

    # Build all prompts
    all_prompts = []
    prompt_metadata = []  # Track which question each prompt belongs to

    for q in questions:
        prompt = construct_prompt(q["problem"], model_key, tokenizer)
        for sample_idx in range(num_samples):
            all_prompts.append(prompt)
            prompt_metadata.append({
                "question_id": q["unique_id"],
                "problem": q["problem"],
                "answer": q["answer"],
                "sample_idx": sample_idx,
            })

    print(f"Generating {len(all_prompts)} completions...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Process outputs
    cot_samples = []
    for i, output in enumerate(outputs):
        meta = prompt_metadata[i]
        completion = output.outputs[0].text

        cot_text = extract_cot_from_completion(completion, model_key)
        extracted_answer = extract_answer(completion)

        cot_samples.append(CoTSample(
            question_id=meta["question_id"],
            problem=meta["problem"],
            ground_truth_answer=meta["answer"],
            source_model=model_key,
            cot_text=cot_text,
            extracted_answer=extracted_answer,
        ))

    # Cleanup
    del llm
    if tokenizer:
        del tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    return cot_samples


def run_thought_sampling_analysis(
    cot_sample: CoTSample,
    model_a: str,
    model_b: str,
    num_samples: int = 10,
    fractions: List[float] = None,
    out_dir: str = "./thought_sampling_out",
    seed: Optional[int] = None,
) -> ThoughtSamplingResult:
    """
    Run thought sampling analysis on a single CoT sample.
    """
    if fractions is None:
        fractions = [0.25, 0.5, 0.75]

    result = analyze_thought_sampling(
        prompt=cot_sample.problem,
        chain_of_thought_text=cot_sample.cot_text,
        model_name_a=model_a,
        model_name_b=model_b,
        num_samples=num_samples,
        fractions=fractions,
        out_dir=out_dir,
        seed=seed,
    )

    summary = result["summary"]

    # Aggregate spread across all fractions
    on_policy_spreads = []
    off_policy_spreads = []

    for f in fractions:
        f_key = str(f)
        on_hist = summary["fractions"][f_key]["on_policy"]["histogram"]
        off_hist = summary["fractions"][f_key]["off_policy"]["histogram"]

        on_policy_spreads.append(compute_answer_spread(on_hist))
        off_policy_spreads.append(compute_answer_spread(off_hist))

    # Average spread across fractions
    avg_on_policy_spread = sum(on_policy_spreads) / len(on_policy_spreads)
    avg_off_policy_spread = sum(off_policy_spreads) / len(off_policy_spreads)

    return ThoughtSamplingResult(
        cot_sample=cot_sample,
        summary=summary,
        on_policy_spread=avg_on_policy_spread,
        off_policy_spread=avg_off_policy_spread,
        spread_difference=avg_on_policy_spread - avg_off_policy_spread,
    )


def run_experiment(
    num_cot_samples: int = 10,
    num_resample: int = 10,
    fractions: List[float] = None,
    output_dir: str = "./experiment_results",
    seed: Optional[int] = 42,
):
    """
    Run the full experiment.

    Args:
        num_cot_samples: Number of CoT samples per question per model
        num_resample: Number of resamples for thought sampling analysis
        fractions: CoT prefix fractions to test
        output_dir: Directory for output files
        seed: Random seed
    """
    if fractions is None:
        fractions = [0.25, 0.5, 0.75]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset rguan72/chicken-scratch-tests...")
    dataset = load_dataset("rguan72/chicken-scratch-tests", split="test")
    questions = list(dataset)
    print(f"Loaded {len(questions)} questions")

    # Step 1: Sample CoTs for all questions from both models
    print("\n" + "="*60)
    print("STEP 1: Sampling CoTs from both models")
    print("="*60)

    all_cot_samples = []

    for model_key in ["qwen", "nvidia"]:
        cot_samples = sample_cots_for_questions(
            questions=questions,
            model_key=model_key,
            num_samples=num_cot_samples,
            seed=seed,
        )
        all_cot_samples.extend(cot_samples)
        print(f"Generated {len(cot_samples)} CoT samples from {model_key}")

    # Save CoT samples
    cot_samples_path = output_path / "cot_samples.jsonl"
    with open(cot_samples_path, "w") as f:
        for sample in all_cot_samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_cot_samples)} CoT samples to {cot_samples_path}")

    # Step 2: Run thought sampling analysis on each CoT
    print("\n" + "="*60)
    print("STEP 2: Running thought sampling analysis on each CoT")
    print("="*60)

    all_results = []

    for i, cot_sample in enumerate(tqdm(all_cot_samples, desc="Analyzing CoTs")):
        # For each CoT, model_a is the source model (on-policy)
        # and model_b is the other model (off-policy)
        model_a = cot_sample.source_model
        model_b = "nvidia" if model_a == "qwen" else "qwen"

        sample_out_dir = output_path / "thought_sampling" / f"sample_{i:04d}"

        try:
            result = run_thought_sampling_analysis(
                cot_sample=cot_sample,
                model_a=model_a,
                model_b=model_b,
                num_samples=num_resample,
                fractions=fractions,
                out_dir=str(sample_out_dir),
                seed=seed,
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nError analyzing sample {i}: {e}")
            continue

    # Step 3: Order results by spread difference
    print("\n" + "="*60)
    print("STEP 3: Ordering results by answer spread difference")
    print("="*60)

    # Sort by spread difference (largest positive = most consistency benefit from on-policy)
    all_results.sort(key=lambda r: r.spread_difference, reverse=True)

    # Save ordered results
    ordered_results_path = output_path / "ordered_results.jsonl"
    with open(ordered_results_path, "w") as f:
        for result in all_results:
            entry = {
                "question_id": result.cot_sample.question_id,
                "source_model": result.cot_sample.source_model,
                "on_policy_spread": result.on_policy_spread,
                "off_policy_spread": result.off_policy_spread,
                "spread_difference": result.spread_difference,
                "cot_text_preview": result.cot_sample.cot_text[:200] + "..." if len(result.cot_sample.cot_text) > 200 else result.cot_sample.cot_text,
                "extracted_answer": result.cot_sample.extracted_answer,
                "ground_truth": result.cot_sample.ground_truth_answer,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save full summary
    summary = {
        "num_questions": len(questions),
        "num_cot_samples_per_model": num_cot_samples,
        "num_resample": num_resample,
        "fractions": fractions,
        "total_cot_samples": len(all_cot_samples),
        "total_analyzed": len(all_results),
        "results_by_spread_difference": [
            {
                "rank": i + 1,
                "question_id": r.cot_sample.question_id,
                "source_model": r.cot_sample.source_model,
                "on_policy_spread": r.on_policy_spread,
                "off_policy_spread": r.off_policy_spread,
                "spread_difference": r.spread_difference,
            }
            for i, r in enumerate(all_results)
        ],
    }

    summary_path = output_path / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total CoT samples analyzed: {len(all_results)}")
    print(f"\nTop 10 by spread difference (on-policy - off-policy):")
    print("-" * 60)

    for i, result in enumerate(all_results[:10]):
        print(f"{i+1:2d}. {result.cot_sample.question_id}")
        print(f"    Source: {result.cot_sample.source_model}")
        print(f"    On-policy spread:  {result.on_policy_spread:.3f}")
        print(f"    Off-policy spread: {result.off_policy_spread:.3f}")
        print(f"    Difference: {result.spread_difference:+.3f}")
        print()

    print(f"\nResults saved to: {output_path}")
    print(f"  - {cot_samples_path}")
    print(f"  - {ordered_results_path}")
    print(f"  - {summary_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run thought sampling experiment")
    parser.add_argument("--num-cot-samples", type=int, default=10,
                        help="Number of CoT samples per question per model")
    parser.add_argument("--num-resample", type=int, default=10,
                        help="Number of resamples for thought sampling")
    parser.add_argument("--fractions", type=str, default="0.25,0.5,0.75",
                        help="Comma-separated fractions for CoT prefixes")
    parser.add_argument("--output-dir", type=str, default="./experiment_results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    fractions = [float(f.strip()) for f in args.fractions.split(",")]

    run_experiment(
        num_cot_samples=args.num_cot_samples,
        num_resample=args.num_resample,
        fractions=fractions,
        output_dir=args.output_dir,
        seed=args.seed,
    )
