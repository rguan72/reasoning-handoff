"""
Thought Sampling Consistency Analysis Utility

Measures whether resampling on the same model produces the same/similar final answers
more often than resampling with a different model ("off-policy").
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_math500 import extract_answer
from model_configs import MODEL_CONFIGS, VALID_MODEL_NAMES


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a robust heuristic.

    Args:
        text: Plain text to split

    Returns:
        List of sentences (whitespace-stripped)
    """
    if not text or not text.strip():
        return []

    text = ' '.join(text.split())

    # Common abbreviations that shouldn't trigger sentence breaks
    abbrevs = {'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Inc', 'Ltd', 'St'}

    # First pass: use lookbehind/lookahead to split on sentence boundaries
    # Split after .!? followed by space and uppercase letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    parts = re.split(pattern, text)

    # Second pass: merge back sentences that were incorrectly split on abbreviations
    sentences = []
    i = 0
    while i < len(parts):
        part = parts[i]
        # Check if this part ends with an abbreviation
        words = part.split()
        if words:
            last_word = words[-1].rstrip('.!?')
            # If ends with abbreviation and there's a next part, merge
            if last_word in abbrevs and i + 1 < len(parts):
                parts[i + 1] = part + ' ' + parts[i + 1]
            else:
                sentences.append(part.strip())
        i += 1

    sentences = [s for s in sentences if s.strip()]

    # Fallback for text that didn't split well
    if len(sentences) <= 1 and len(text) > 100:
        simple_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(simple_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def build_histogram(answers: List[Optional[str]]) -> Dict[str, int]:
    """
    Build a histogram of answer occurrences.

    Args:
        answers: List of extracted answers (may include None for failed extractions)

    Returns:
        Dictionary mapping answer string to count (None represented as "__NONE__")
    """
    normalized = [a if a is not None else "__NONE__" for a in answers]
    return dict(Counter(normalized))


def run_samples(
    llm,
    sampling_params,
    prompts: List[str],
) -> List[str]:
    """
    Run sampling to generate completions using vLLM.

    Args:
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams
        prompts: List of prompts to complete

    Returns:
        List of generated completion texts
    """
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def construct_prompt_with_cot_prefix(
    original_prompt: str,
    cot_prefix: str,
    model_key: str,
    tokenizer=None,
) -> str:
    """
    Construct the input prompt with CoT prefix for sampling.

    Args:
        original_prompt: The original problem prompt
        cot_prefix: The chain-of-thought prefix to continue from
        model_key: Model key ('qwen' or 'nvidia')
        tokenizer: Optional tokenizer for chat template

    Returns:
        Constructed prompt string
    """
    config = MODEL_CONFIGS[model_key]
    user_content = f"Solve the following math problem step by step. Put your final answer inside \\boxed{{}}.\n\nProblem: {original_prompt}"

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
        prompt = f"{user_content}\n<think>{cot_prefix}"

    return prompt


def analyze_thought_sampling(
    prompt: str,
    chain_of_thought_text: str,
    model_name_a: str,
    model_name_b: str,
    num_samples: int = 10,
    fractions: List[float] = None,
    out_dir: str = "./thought_sampling_out",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze thought-sampling consistency for a single example.

    Args:
        prompt: The original problem prompt
        chain_of_thought_text: The chain-of-thought text to sample from
        model_name_a: Model A name ('qwen' or 'nvidia')
        model_name_b: Model B name ('qwen' or 'nvidia')
        num_samples: Number of samples per model per fraction (default: 10)
        fractions: List of fractions for CoT prefixes (default: [0.25, 0.5, 0.75])
        out_dir: Output directory for results
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - 'results': List of all sample results
            - 'summary': Summary with histograms per fraction
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import gc
    import platform
    import torch

    # Detect if we're on macOS or if CUDA is not available
    is_macos = platform.system() == "Darwin"
    has_cuda = torch.cuda.is_available()
    use_cpu = is_macos or not has_cuda

    if is_macos and not has_cuda:
        print("WARNING: Running on macOS without CUDA. vLLM may not work correctly.")
        print("For macOS, vLLM must be built with CPU target: VLLM_TARGET_DEVICE=cpu pip install -e .")
        print("Attempting to continue with CPU configuration...")

    if fractions is None:
        fractions = [0.25, 0.5, 0.75]
    fractions = sorted(fractions)

    if model_name_a not in MODEL_CONFIGS or model_name_b not in MODEL_CONFIGS:
        raise ValueError(f"model_name_a and model_name_b must be one of {VALID_MODEL_NAMES}, got {model_name_a} and {model_name_b}")

    # Split CoT into sentences
    sentences = split_sentences(chain_of_thought_text)
    n = len(sentences)

    if n == 0:
        raise ValueError("Could not split chain_of_thought_text into sentences")

    # Create CoT prefixes
    cot_prefixes = {}
    for f in fractions:
        k = max(1, round(f * n))
        cot_prefixes[f] = " ".join(sentences[:k])

    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    model_completions = {}

    # Process each model
    for model_key in [model_name_a, model_name_b]:
        config = MODEL_CONFIGS[model_key]

        # Build LLM
        print(config)
        llm_kwargs = {
            "model": config["model_path"],
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "max_model_len": config["max_model_len"],
            "max_num_batched_tokens": config["max_num_batched_tokens"],
        }
        
        # Configure device-specific settings
        if use_cpu:
            # For CPU mode, use float32 and remove GPU-specific parameters
            llm_kwargs["dtype"] = "float32"
            # Note: vLLM may not fully support CPU mode on macOS
            # GPU-specific parameters are omitted for CPU mode
        else:
            llm_kwargs["dtype"] = config["dtype"]
            llm_kwargs["gpu_memory_utilization"] = config["gpu_memory_utilization"]
        
        if seed is not None:
            llm_kwargs["seed"] = seed

        try:
            llm = LLM(**llm_kwargs)
        except (AssertionError, RuntimeError, TypeError) as e:
            error_str = str(e).lower()
            if "cuda" in error_str or "device_config" in error_str or "unexpected keyword" in error_str:
                error_msg = (
                    "\n" + "="*60 + "\n"
                    "ERROR: vLLM configuration issue.\n"
                    "On macOS, vLLM must be built with CPU target:\n"
                    "  git clone https://github.com/vllm-project/vllm.git\n"
                    "  cd vllm\n"
                    "  VLLM_TARGET_DEVICE=cpu pip install -e .\n\n"
                    "Alternatively, vLLM may not support CPU mode on macOS.\n"
                    "Consider using a GPU-enabled system or a different inference backend.\n"
                    "="*60 + "\n"
                )
                raise RuntimeError(error_msg) from e
            raise

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

        model_completions[model_key] = {}

        for f in fractions:
            cot_prefix = cot_prefixes[f]
            input_prompt = construct_prompt_with_cot_prefix(
                prompt, cot_prefix, model_key, tokenizer
            )

            # Batch all samples together for vLLM efficiency
            prompts = [input_prompt] * num_samples
            completions = run_samples(llm, sampling_params, prompts)

            fraction_results = []
            for i, completion in enumerate(completions):
                answer = extract_answer(completion)
                if not answer or answer == completion.strip():
                    answer = None

                fraction_results.append((completion, answer))
                all_results.append({
                    "fraction": f,
                    "model": model_key,
                    "model_name": config["name"],
                    "sample_index": i,
                    "raw_completion": completion,
                    "extracted_answer": answer,
                })

            model_completions[model_key][f] = fraction_results

        # Cleanup
        del llm
        if tokenizer:
            del tokenizer
        gc.collect()
        if not use_cpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    # Build summary
    summary = {"fractions": {}}
    for f in fractions:
        on_policy_answers = [r[1] for r in model_completions[model_name_a][f]]
        off_policy_answers = [r[1] for r in model_completions[model_name_b][f]]

        summary["fractions"][str(f)] = {
            "on_policy": {
                "model": model_name_a,
                "model_name": MODEL_CONFIGS[model_name_a]["name"],
                "histogram": build_histogram(on_policy_answers),
                "num_samples": len(on_policy_answers),
            },
            "off_policy": {
                "model": model_name_b,
                "model_name": MODEL_CONFIGS[model_name_b]["name"],
                "histogram": build_histogram(off_policy_answers),
                "num_samples": len(off_policy_answers),
            },
        }

    summary["metadata"] = {
        "model_a": model_name_a,
        "model_b": model_name_b,
        "num_samples": num_samples,
        "fractions": fractions,
        "seed": seed,
        "num_sentences": n,
    }

    # Write outputs
    results_path = out_path / "results.jsonl"
    with open(results_path, "w") as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary_path = out_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print human-readable summary
    print("\n" + "=" * 60)
    print("THOUGHT SAMPLING CONSISTENCY ANALYSIS")
    print("=" * 60)
    print(f"On-policy model (A):  {MODEL_CONFIGS[model_name_a]['name']}")
    print(f"Off-policy model (B): {MODEL_CONFIGS[model_name_b]['name']}")
    print(f"Samples per condition: {num_samples}")
    print(f"CoT sentences: {n}")

    for f in fractions:
        k = max(1, round(f * n))
        print(f"\n--- Fraction {f} ({k} sentences) ---")

        on_hist = summary["fractions"][str(f)]["on_policy"]["histogram"]
        off_hist = summary["fractions"][str(f)]["off_policy"]["histogram"]

        print(f"  On-policy ({model_name_a}):")
        for answer, count in sorted(on_hist.items(), key=lambda x: -x[1]):
            display = answer if answer != "__NONE__" else "(extraction failed)"
            print(f"    {count:3d}x: {display[:60]}{'...' if len(str(display)) > 60 else ''}")

        print(f"  Off-policy ({model_name_b}):")
        for answer, count in sorted(off_hist.items(), key=lambda x: -x[1]):
            display = answer if answer != "__NONE__" else "(extraction failed)"
            print(f"    {count:3d}x: {display[:60]}{'...' if len(str(display)) > 60 else ''}")

    return {"results": all_results, "summary": summary}
