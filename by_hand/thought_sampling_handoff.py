"""
Utility for thought sampling handoff method.

Samples multiple times from a prompt with a chain-of-thought prefix,
extracts answers, and computes histogram and entropy statistics.
"""
import math
from collections import Counter
from typing import Dict, List, Optional

from by_hand.answer_extraction import extract_answer
from by_hand.inference import run_inference, cleanup_model_memory
from by_hand.long_data import correct_cot_prefix
from by_hand.model_configs import MODEL_CONFIGS
from by_hand.prompts import construct_prompt


def construct_prompt_with_cot_prefix(
    original_prompt: str,
    cot_prefix: str,
    model_key: str,
) -> str:
    """
    Construct the input prompt with CoT prefix for sampling.
    
    Uses construct_prompt from prompts.py and inserts the CoT prefix after <think>.
    
    Args:
        original_prompt: The original problem prompt
        cot_prefix: The chain-of-thought prefix to continue from
        model_key: Model key from MODEL_CONFIGS
    
    Returns:
        Constructed prompt string with CoT prefix appended
    """
    # Use construct_prompt to get the base prompt format
    # It returns: "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. Problem: {problem} Solution: \n<think>\n"
    base_prompt = construct_prompt(original_prompt)
    
    # Replace the trailing "\n<think>\n" with "\n<think>{cot_prefix}"
    # to insert the CoT prefix
    base_prompt_text = base_prompt.replace("\n<think>\n", f"\n<think>{cot_prefix}")
    
    config = MODEL_CONFIGS[model_key]
    
    if config.get("use_chat_template", False):
        # For chat template models, format the prompt using the chat template
        from by_hand.inference import format_prompt
        prompt = format_prompt(model_key, base_prompt_text)
    else:
        # For non-chat-template models, use the prompt as-is
        prompt = base_prompt_text
    
    return prompt


def compute_entropy(histogram: Dict[str, int], total_count: Optional[int] = None) -> float:
    """
    Compute Shannon entropy from a histogram.
    
    Args:
        histogram: Dictionary mapping answer strings to counts
        total_count: Total number of samples (if None, sum of histogram values)
    
    Returns:
        Shannon entropy in bits
    """
    if total_count is None:
        total_count = sum(histogram.values())
    
    if total_count == 0:
        return 0.0
    
    entropy = 0.0
    for count in histogram.values():
        if count > 0:
            prob = count / total_count
            entropy -= prob * math.log2(prob)
    
    return entropy


def analyze_answer_distribution(
    prompt: str,
    cot_prefix: str,
    model_key: str,
    num_samples: int,
) -> Dict:
    """
    Perform thought sampling handoff for a single prompt and CoT prefix.
    
    Samples multiple times, extracts answers, and computes histogram and entropy.
    
    Args:
        prompt: The original problem prompt
        cot_prefix: The chain-of-thought prefix to continue from
        model_key: Model key from MODEL_CONFIGS
        num_samples: Number of times to sample
    
    Returns:
        Dictionary containing:
            - 'histogram': Dictionary mapping answer strings to counts (None -> "__NONE__")
            - 'entropy_with_none': Entropy including None values
            - 'entropy_without_none': Entropy excluding None values
            - 'num_samples': Total number of samples
            - 'num_none': Number of None values
            - 'num_valid': Number of non-None values
            - 'raw_completions': List of raw completion texts
            - 'extracted_answers': List of extracted answers (may contain None)
    """
    # Construct the full prompt with CoT prefix
    full_prompt = construct_prompt_with_cot_prefix(prompt, cot_prefix, model_key)
    
    # For chat-template models, we've already formatted the prompt, so we need to
    # use the lower-level API to avoid double-formatting. For non-chat-template models,
    # run_inference will work fine since format_prompt returns the prompt as-is.
    config = MODEL_CONFIGS[model_key]
    if config.get("use_chat_template", False):
        # Use lower-level API to avoid double formatting
        from vllm import SamplingParams
        from by_hand.inference import get_llm
        
        llm = get_llm(model_key)
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
        )
        prompts = [full_prompt] * num_samples
        outputs = llm.generate(prompts, sampling_params)
        raw_completions = [output.outputs[0].text for output in outputs]
    else:
        # For non-chat-template models, run_inference will work fine
        raw_completions = run_inference(
            model_key=model_key,
            prompt=full_prompt,
            num_runs=num_samples,
        )
    
    # Extract answers from each completion
    extracted_answers = [extract_answer(completion) for completion in raw_completions]
    
    # Build histogram (bucket None values together as "__NONE__")
    normalized_answers = [ans if ans is not None else "__NONE__" for ans in extracted_answers]
    histogram = dict(Counter(normalized_answers))
    
    # Compute entropy with None values
    entropy_with_none = compute_entropy(histogram, total_count=num_samples)
    
    # Compute entropy without None values
    histogram_without_none = {
        k: v for k, v in histogram.items() if k != "__NONE__"
    }
    num_valid = sum(histogram_without_none.values())
    entropy_without_none = compute_entropy(histogram_without_none, total_count=num_valid) if num_valid > 0 else 0.0
    
    num_none = histogram.get("__NONE__", 0)
    
    return {
        'histogram': histogram,
        'entropy_with_none': entropy_with_none,
        'entropy_without_none': entropy_without_none,
        'num_samples': num_samples,
        'num_none': num_none,
        'num_valid': num_valid,
        'raw_completions': raw_completions,
        'extracted_answers': extracted_answers,
    }


def print_analysis(results: Dict):
    """
    Print a formatted analysis of the answer distribution.
    
    Args:
        results: Results dictionary from analyze_answer_distribution
    """
    print("\n" + "=" * 60)
    print("THOUGHT SAMPLING HANDOFF ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {results['num_samples']}")
    print(f"Valid answers: {results['num_valid']}")
    print(f"None values: {results['num_none']}")
    print(f"\nEntropy (with None): {results['entropy_with_none']:.4f} bits")
    print(f"Entropy (without None): {results['entropy_without_none']:.4f} bits")
    
    print(f"\nAnswer Distribution (Histogram):")
    histogram = results['histogram']
    sorted_items = sorted(histogram.items(), key=lambda x: -x[1])
    
    for answer, count in sorted_items:
        display = answer if answer != "__NONE__" else "(None - extraction failed)"
        percentage = (count / results['num_samples']) * 100
        print(f"  {count:3d}x ({percentage:5.1f}%): {display[:70]}{'...' if len(display) > 70 else ''}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    split = correct_cot_prefix.split(". ")
    num_sentences = len(split)
    num_to_take = int(num_sentences * 2 / 4)
    cot_prefix = ". ".join(split[:num_to_take])
    prompt = f"In $\\triangle ABC$ points $D$ and $E$ lie on $\\overline{{AB}}$ so that $AD < AE < AB$, while points $F$ and $G$ lie on $\\overline{{AC}}$ so that $AF < AG < AC$. Suppose $AD = 4$, $DE = 16$, $EB = 8$, $AF = 13$, $FG = 52$, and $GC = 26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. The area of quadrilateral $DEGF$ is $288$. Find the area of heptagon $AFNBCEM$."
    results = analyze_answer_distribution(
        prompt=prompt,
        cot_prefix=cot_prefix,
        model_key="deep-llama",
        num_samples=20,
    )
    print("========Off Policy========")
    print_analysis(results)

    # Clean up model memory after first analysis
    cleanup_model_memory("deep-llama")

    results2 = analyze_answer_distribution(
        prompt=prompt,
        cot_prefix=cot_prefix,
        model_key="deep-qwen",
        num_samples=20,
    )
    print("========On Policy========")
    print_analysis(results2)
    
    # Clean up model memory after second analysis
    cleanup_model_memory("deep-qwen")

