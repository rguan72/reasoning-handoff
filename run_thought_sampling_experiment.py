"""
Experiment: Thought Sampling Consistency Analysis

For each question in rguan72/chicken-scratch-tests:
1. Sample 10 CoTs per question for qwen and nvidia
2. Run thought sampling resamples on each CoT
3. Order results by difference in answer spread between on-policy and off-policy

Optimized for maximum vLLM throughput:
- Each model loaded only ONCE
- All prompts batched together for parallel processing
- GPU memory utilization maximized for RTX 6000 (96GB VRAM)
"""

import json
import math
import gc
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from model_configs import MODEL_CONFIGS, VALID_MODEL_NAMES
from thought_sampling_handoff import (
    split_sentences,
    build_histogram,
    extract_answer,
)


@dataclass
class CoTSample:
    """A single chain-of-thought sample."""
    sample_id: int
    question_id: str
    problem: str
    ground_truth_answer: str
    source_model: str
    cot_text: str
    extracted_answer: Optional[str]


@dataclass
class ResampleTask:
    """A resampling task to be batched."""
    cot_sample_id: int
    fraction: float
    cot_prefix: str
    problem: str


@dataclass
class ResampleResult:
    """Result from a single resample."""
    cot_sample_id: int
    fraction: float
    model: str
    completion: str
    extracted_answer: Optional[str]


@dataclass
class ThoughtSamplingResult:
    """Final result for a single CoT."""
    cot_sample: CoTSample
    on_policy_histograms: Dict[float, Dict[str, int]]
    off_policy_histograms: Dict[float, Dict[str, int]]
    on_policy_spread: float
    off_policy_spread: float
    spread_difference: float


def compute_entropy(histogram: Dict[str, int]) -> float:
    """Compute Shannon entropy of answer distribution."""
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def construct_cot_prompt(problem: str, model_key: str, tokenizer=None) -> str:
    """Construct prompt for initial CoT generation."""
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


def construct_resample_prompt(problem: str, cot_prefix: str, model_key: str, tokenizer=None) -> str:
    """Construct prompt for resampling with CoT prefix."""
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
        prompt = prompt + cot_prefix
    else:
        prompt = user_content + cot_prefix
    return prompt


def extract_cot_from_completion(completion: str) -> str:
    """Extract the chain-of-thought portion from a completion."""
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")

    if think_start != -1 and think_end != -1 and think_end > think_start:
        return completion[think_start + len("<think>"):think_end].strip()
    elif think_start != -1:
        return completion[think_start + len("<think>"):].strip()
    else:
        boxed_pos = completion.find("\\boxed")
        if boxed_pos != -1:
            return completion[:boxed_pos].strip()
        return completion.strip()


def cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass


def run_experiment(
    num_cot_samples: int = 10,
    num_resample: int = 10,
    fractions: List[float] = None,
    output_dir: str = "./experiment_results",
    seed: Optional[int] = 42,
    gpu_memory_utilization: float = 0.90,
):
    """
    Run the full experiment with maximum vLLM throughput.

    Strategy:
    1. Load qwen ONCE -> generate ALL qwen CoTs + ALL resamples needing qwen
    2. Load nvidia ONCE -> generate ALL nvidia CoTs + ALL resamples needing nvidia
    3. Combine results and compute metrics

    Args:
        num_cot_samples: Number of CoT samples per question per model
        num_resample: Number of resamples per fraction per model
        fractions: CoT prefix fractions to test
        output_dir: Directory for output files
        seed: Random seed
        gpu_memory_utilization: Fraction of GPU memory to use (0.90 for 96GB)
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

    # Storage for all data
    all_cot_samples: Dict[int, CoTSample] = {}  # sample_id -> CoTSample
    all_resample_results: Dict[str, List[ResampleResult]] = defaultdict(list)  # model -> results

    sample_id_counter = 0

    # ========================================================================
    # PHASE 1: Process with QWEN (load once, do all qwen work)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Loading Qwen and running ALL Qwen inference")
    print("=" * 70)

    qwen_config = MODEL_CONFIGS["qwen"]
    print(f"Loading {qwen_config['name']}...")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")

    qwen_llm = LLM(
        model=qwen_config["model_path"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=qwen_config["dtype"],
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=qwen_config["max_model_len"],
        max_num_batched_tokens=qwen_config["max_num_batched_tokens"],
        seed=seed,
    )

    qwen_tokenizer = None
    if qwen_config["use_chat_template"]:
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_config["model_path"], trust_remote_code=True
        )

    qwen_sampling_params = SamplingParams(
        temperature=qwen_config["temperature"],
        max_tokens=qwen_config["max_tokens"],
        top_p=qwen_config["top_p"],
    )

    # 1a. Generate CoTs from qwen
    print(f"\n[Qwen] Generating {len(questions) * num_cot_samples} CoTs...")
    qwen_cot_prompts = []
    qwen_cot_metadata = []

    for q in questions:
        prompt = construct_cot_prompt(q["problem"], "qwen", qwen_tokenizer)
        for _ in range(num_cot_samples):
            qwen_cot_prompts.append(prompt)
            qwen_cot_metadata.append({
                "question_id": q["unique_id"],
                "problem": q["problem"],
                "answer": q["answer"],
            })

    qwen_cot_outputs = qwen_llm.generate(qwen_cot_prompts, qwen_sampling_params)

    # Process qwen CoT outputs
    qwen_cot_samples = []
    for i, output in enumerate(qwen_cot_outputs):
        meta = qwen_cot_metadata[i]
        completion = output.outputs[0].text
        cot_text = extract_cot_from_completion(completion)
        extracted = extract_answer(completion)

        sample = CoTSample(
            sample_id=sample_id_counter,
            question_id=meta["question_id"],
            problem=meta["problem"],
            ground_truth_answer=meta["answer"],
            source_model="qwen",
            cot_text=cot_text,
            extracted_answer=extracted,
        )
        all_cot_samples[sample_id_counter] = sample
        qwen_cot_samples.append(sample)
        sample_id_counter += 1

    print(f"[Qwen] Generated {len(qwen_cot_samples)} CoTs")

    # 1b. Build ALL resample prompts that need qwen
    # - Qwen CoTs need qwen resamples (on-policy)
    # - Nvidia CoTs will need qwen resamples (off-policy) - but we don't have nvidia CoTs yet
    #   So we'll do qwen on-policy now, and qwen off-policy for nvidia CoTs later

    print(f"\n[Qwen] Building resample prompts for qwen CoTs (on-policy)...")
    qwen_resample_prompts = []
    qwen_resample_metadata = []

    for sample in qwen_cot_samples:
        sentences = split_sentences(sample.cot_text)
        n = len(sentences)
        if n == 0:
            continue

        for f in fractions:
            k = max(1, round(f * n))
            cot_prefix = " ".join(sentences[:k])

            for _ in range(num_resample):
                prompt = construct_resample_prompt(
                    sample.problem, cot_prefix, "qwen", qwen_tokenizer
                )
                qwen_resample_prompts.append(prompt)
                qwen_resample_metadata.append({
                    "cot_sample_id": sample.sample_id,
                    "fraction": f,
                })

    print(f"[Qwen] Running {len(qwen_resample_prompts)} resamples...")
    if qwen_resample_prompts:
        qwen_resample_outputs = qwen_llm.generate(qwen_resample_prompts, qwen_sampling_params)

        for i, output in enumerate(qwen_resample_outputs):
            meta = qwen_resample_metadata[i]
            completion = output.outputs[0].text
            extracted = extract_answer(completion)

            all_resample_results["qwen"].append(ResampleResult(
                cot_sample_id=meta["cot_sample_id"],
                fraction=meta["fraction"],
                model="qwen",
                completion=completion,
                extracted_answer=extracted,
            ))

    print(f"[Qwen] Completed {len(all_resample_results['qwen'])} resamples")

    # Cleanup qwen
    del qwen_llm
    if qwen_tokenizer:
        del qwen_tokenizer
    cleanup_gpu()

    # ========================================================================
    # PHASE 2: Process with NVIDIA (load once, do all nvidia work)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Loading Nvidia and running ALL Nvidia inference")
    print("=" * 70)

    nvidia_config = MODEL_CONFIGS["nvidia"]
    print(f"Loading {nvidia_config['name']}...")

    nvidia_llm = LLM(
        model=nvidia_config["model_path"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=nvidia_config["dtype"],
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=nvidia_config["max_model_len"],
        max_num_batched_tokens=nvidia_config["max_num_batched_tokens"],
        seed=seed,
    )

    nvidia_tokenizer = None
    if nvidia_config["use_chat_template"]:
        nvidia_tokenizer = AutoTokenizer.from_pretrained(
            nvidia_config["model_path"], trust_remote_code=True
        )

    nvidia_sampling_params = SamplingParams(
        temperature=nvidia_config["temperature"],
        max_tokens=nvidia_config["max_tokens"],
        top_p=nvidia_config["top_p"],
    )

    # 2a. Generate CoTs from nvidia
    print(f"\n[Nvidia] Generating {len(questions) * num_cot_samples} CoTs...")
    nvidia_cot_prompts = []
    nvidia_cot_metadata = []

    for q in questions:
        prompt = construct_cot_prompt(q["problem"], "nvidia", nvidia_tokenizer)
        for _ in range(num_cot_samples):
            nvidia_cot_prompts.append(prompt)
            nvidia_cot_metadata.append({
                "question_id": q["unique_id"],
                "problem": q["problem"],
                "answer": q["answer"],
            })

    nvidia_cot_outputs = nvidia_llm.generate(nvidia_cot_prompts, nvidia_sampling_params)

    # Process nvidia CoT outputs
    nvidia_cot_samples = []
    for i, output in enumerate(nvidia_cot_outputs):
        meta = nvidia_cot_metadata[i]
        completion = output.outputs[0].text
        cot_text = extract_cot_from_completion(completion)
        extracted = extract_answer(completion)

        sample = CoTSample(
            sample_id=sample_id_counter,
            question_id=meta["question_id"],
            problem=meta["problem"],
            ground_truth_answer=meta["answer"],
            source_model="nvidia",
            cot_text=cot_text,
            extracted_answer=extracted,
        )
        all_cot_samples[sample_id_counter] = sample
        nvidia_cot_samples.append(sample)
        sample_id_counter += 1

    print(f"[Nvidia] Generated {len(nvidia_cot_samples)} CoTs")

    # 2b. Build ALL resample prompts that need nvidia
    # - Nvidia CoTs need nvidia resamples (on-policy)
    # - Qwen CoTs need nvidia resamples (off-policy)

    print(f"\n[Nvidia] Building resample prompts...")
    nvidia_resample_prompts = []
    nvidia_resample_metadata = []

    # Nvidia on-policy (for nvidia CoTs)
    for sample in nvidia_cot_samples:
        sentences = split_sentences(sample.cot_text)
        n = len(sentences)
        if n == 0:
            continue

        for f in fractions:
            k = max(1, round(f * n))
            cot_prefix = " ".join(sentences[:k])

            for _ in range(num_resample):
                prompt = construct_resample_prompt(
                    sample.problem, cot_prefix, "nvidia", nvidia_tokenizer
                )
                nvidia_resample_prompts.append(prompt)
                nvidia_resample_metadata.append({
                    "cot_sample_id": sample.sample_id,
                    "fraction": f,
                })

    # Nvidia off-policy (for qwen CoTs)
    for sample in qwen_cot_samples:
        sentences = split_sentences(sample.cot_text)
        n = len(sentences)
        if n == 0:
            continue

        for f in fractions:
            k = max(1, round(f * n))
            cot_prefix = " ".join(sentences[:k])

            for _ in range(num_resample):
                prompt = construct_resample_prompt(
                    sample.problem, cot_prefix, "nvidia", nvidia_tokenizer
                )
                nvidia_resample_prompts.append(prompt)
                nvidia_resample_metadata.append({
                    "cot_sample_id": sample.sample_id,
                    "fraction": f,
                })

    print(f"[Nvidia] Running {len(nvidia_resample_prompts)} resamples...")
    if nvidia_resample_prompts:
        nvidia_resample_outputs = nvidia_llm.generate(nvidia_resample_prompts, nvidia_sampling_params)

        for i, output in enumerate(nvidia_resample_outputs):
            meta = nvidia_resample_metadata[i]
            completion = output.outputs[0].text
            extracted = extract_answer(completion)

            all_resample_results["nvidia"].append(ResampleResult(
                cot_sample_id=meta["cot_sample_id"],
                fraction=meta["fraction"],
                model="nvidia",
                completion=completion,
                extracted_answer=extracted,
            ))

    print(f"[Nvidia] Completed {len(all_resample_results['nvidia'])} resamples")

    # Cleanup nvidia
    del nvidia_llm
    if nvidia_tokenizer:
        del nvidia_tokenizer
    cleanup_gpu()

    # ========================================================================
    # PHASE 3: Load qwen again for off-policy resamples on nvidia CoTs
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Loading Qwen for off-policy resamples on Nvidia CoTs")
    print("=" * 70)

    qwen_llm = LLM(
        model=qwen_config["model_path"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=qwen_config["dtype"],
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=qwen_config["max_model_len"],
        max_num_batched_tokens=qwen_config["max_num_batched_tokens"],
        seed=seed,
    )

    if qwen_config["use_chat_template"]:
        qwen_tokenizer = AutoTokenizer.from_pretrained(
            qwen_config["model_path"], trust_remote_code=True
        )
    else:
        qwen_tokenizer = None

    # Build off-policy prompts for nvidia CoTs
    print(f"\n[Qwen] Building off-policy resample prompts for nvidia CoTs...")
    qwen_offpolicy_prompts = []
    qwen_offpolicy_metadata = []

    for sample in nvidia_cot_samples:
        sentences = split_sentences(sample.cot_text)
        n = len(sentences)
        if n == 0:
            continue

        for f in fractions:
            k = max(1, round(f * n))
            cot_prefix = " ".join(sentences[:k])

            for _ in range(num_resample):
                prompt = construct_resample_prompt(
                    sample.problem, cot_prefix, "qwen", qwen_tokenizer
                )
                qwen_offpolicy_prompts.append(prompt)
                qwen_offpolicy_metadata.append({
                    "cot_sample_id": sample.sample_id,
                    "fraction": f,
                })

    print(f"[Qwen] Running {len(qwen_offpolicy_prompts)} off-policy resamples...")
    if qwen_offpolicy_prompts:
        qwen_offpolicy_outputs = qwen_llm.generate(qwen_offpolicy_prompts, qwen_sampling_params)

        for i, output in enumerate(qwen_offpolicy_outputs):
            meta = qwen_offpolicy_metadata[i]
            completion = output.outputs[0].text
            extracted = extract_answer(completion)

            all_resample_results["qwen"].append(ResampleResult(
                cot_sample_id=meta["cot_sample_id"],
                fraction=meta["fraction"],
                model="qwen",
                completion=completion,
                extracted_answer=extracted,
            ))

    print(f"[Qwen] Total qwen resamples: {len(all_resample_results['qwen'])}")

    # Final cleanup
    del qwen_llm
    if qwen_tokenizer:
        del qwen_tokenizer
    cleanup_gpu()

    # ========================================================================
    # PHASE 4: Aggregate results and compute metrics
    # ========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Aggregating results and computing metrics")
    print("=" * 70)

    # Save CoT samples
    cot_samples_path = output_path / "cot_samples.jsonl"
    with open(cot_samples_path, "w") as f:
        for sample in all_cot_samples.values():
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    print(f"Saved {len(all_cot_samples)} CoT samples to {cot_samples_path}")

    # Save raw resample results
    resamples_path = output_path / "resample_results.jsonl"
    with open(resamples_path, "w") as f:
        for model, results in all_resample_results.items():
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    total_resamples = sum(len(r) for r in all_resample_results.values())
    print(f"Saved {total_resamples} resample results to {resamples_path}")

    # Build histograms per CoT sample
    # Structure: cot_sample_id -> fraction -> model -> list of answers
    resample_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for model, results in all_resample_results.items():
        for r in results:
            resample_answers[r.cot_sample_id][r.fraction][model].append(r.extracted_answer)

    # Compute final results
    final_results = []

    for sample_id, sample in all_cot_samples.items():
        source_model = sample.source_model
        other_model = "nvidia" if source_model == "qwen" else "qwen"

        on_policy_histograms = {}
        off_policy_histograms = {}
        on_policy_spreads = []
        off_policy_spreads = []

        for f in fractions:
            on_answers = resample_answers[sample_id][f][source_model]
            off_answers = resample_answers[sample_id][f][other_model]

            on_hist = build_histogram(on_answers)
            off_hist = build_histogram(off_answers)

            on_policy_histograms[f] = on_hist
            off_policy_histograms[f] = off_hist

            on_policy_spreads.append(compute_entropy(on_hist))
            off_policy_spreads.append(compute_entropy(off_hist))

        avg_on_spread = sum(on_policy_spreads) / len(on_policy_spreads) if on_policy_spreads else 0
        avg_off_spread = sum(off_policy_spreads) / len(off_policy_spreads) if off_policy_spreads else 0

        final_results.append(ThoughtSamplingResult(
            cot_sample=sample,
            on_policy_histograms=on_policy_histograms,
            off_policy_histograms=off_policy_histograms,
            on_policy_spread=avg_on_spread,
            off_policy_spread=avg_off_spread,
            spread_difference=avg_on_spread - avg_off_spread,
        ))

    # Sort by spread difference
    final_results.sort(key=lambda r: r.spread_difference, reverse=True)

    # Save ordered results
    ordered_results_path = output_path / "ordered_results.jsonl"
    with open(ordered_results_path, "w") as f:
        for result in final_results:
            entry = {
                "sample_id": result.cot_sample.sample_id,
                "question_id": result.cot_sample.question_id,
                "source_model": result.cot_sample.source_model,
                "on_policy_spread": result.on_policy_spread,
                "off_policy_spread": result.off_policy_spread,
                "spread_difference": result.spread_difference,
                "on_policy_histograms": {str(k): v for k, v in result.on_policy_histograms.items()},
                "off_policy_histograms": {str(k): v for k, v in result.off_policy_histograms.items()},
                "cot_text_preview": result.cot_sample.cot_text[:300] + "..." if len(result.cot_sample.cot_text) > 300 else result.cot_sample.cot_text,
                "extracted_answer": result.cot_sample.extracted_answer,
                "ground_truth": result.cot_sample.ground_truth_answer,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "config": {
            "num_questions": len(questions),
            "num_cot_samples_per_model": num_cot_samples,
            "num_resample": num_resample,
            "fractions": fractions,
            "seed": seed,
            "gpu_memory_utilization": gpu_memory_utilization,
        },
        "totals": {
            "total_cot_samples": len(all_cot_samples),
            "total_resamples": total_resamples,
            "qwen_cots": len(qwen_cot_samples),
            "nvidia_cots": len(nvidia_cot_samples),
        },
        "top_20_by_spread_difference": [
            {
                "rank": i + 1,
                "sample_id": r.cot_sample.sample_id,
                "question_id": r.cot_sample.question_id,
                "source_model": r.cot_sample.source_model,
                "on_policy_spread": round(r.on_policy_spread, 4),
                "off_policy_spread": round(r.off_policy_spread, 4),
                "spread_difference": round(r.spread_difference, 4),
            }
            for i, r in enumerate(final_results[:20])
        ],
    }

    summary_path = output_path / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total CoT samples: {len(all_cot_samples)}")
    print(f"Total resamples: {total_resamples}")
    print(f"\nTop 10 by spread difference (on-policy - off-policy):")
    print("-" * 70)

    for i, result in enumerate(final_results[:10]):
        print(f"{i+1:2d}. Sample {result.cot_sample.sample_id} | {result.cot_sample.question_id}")
        print(f"    Source: {result.cot_sample.source_model}")
        print(f"    On-policy spread:  {result.on_policy_spread:.4f}")
        print(f"    Off-policy spread: {result.off_policy_spread:.4f}")
        print(f"    Difference: {result.spread_difference:+.4f}")

    print(f"\nResults saved to: {output_path}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run thought sampling experiment")
    parser.add_argument("--num-cot-samples", type=int, default=10,
                        help="Number of CoT samples per question per model")
    parser.add_argument("--num-resample", type=int, default=10,
                        help="Number of resamples per fraction per model")
    parser.add_argument("--fractions", type=str, default="0.25,0.5,0.75",
                        help="Comma-separated fractions for CoT prefixes")
    parser.add_argument("--output-dir", type=str, default="./experiment_results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu-memory", type=float, default=0.90,
                        help="GPU memory utilization (default: 0.90 for 96GB)")

    args = parser.parse_args()
    fractions = [float(f.strip()) for f in args.fractions.split(",")]

    run_experiment(
        num_cot_samples=args.num_cot_samples,
        num_resample=args.num_resample,
        fractions=fractions,
        output_dir=args.output_dir,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory,
    )
