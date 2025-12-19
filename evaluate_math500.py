"""
Evaluate Qwen3-14B and NVIDIA-Nemotron-Nano-12B-v2 on MATH-500 dataset.
Finds problems that are solved correctly 25%-75% of the time.

Requirements:
- CUDA-capable GPU with sufficient memory (models are ~14B and ~12B parameters)
- vllm library for efficient inference
- HuggingFace datasets library

Memory Optimization:
- Uses FP16 (float16) quantization by default to reduce memory usage by ~50%
- Can use AWQ or GPTQ quantization for ~75% memory reduction (if quantized models available)
- GPU memory utilization set to 85% to prevent OOM errors
- Context length limited to 8192 tokens

Usage:
    python evaluate_math500.py [options]

Options:
    --model {nvidia,qwen,both}  Which model(s) to evaluate (default: both)
    --debug-log FILE            Output file for debug log (default: math500_debug_log.jsonl)
    --output FILE               Output file for evaluation results (default: math500_evaluation_results.json)

Examples:
    python evaluate_math500.py --model nvidia --output nvidia_results.json --debug-log nvidia_debug.jsonl
    python evaluate_math500.py --model qwen
    python evaluate_math500.py --model both

Output:
    Creates evaluation results JSON with problems that have 25%-75% accuracy
    for at least one of the models.
"""

import argparse
import json
import re
from typing import Dict, List
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import sympy
from sympy import sympify, simplify, N


def extract_answer(text: str) -> str:
    """
    Extract the final answer from model output.
    Looks for boxed answers or final numerical/expression answers.
    """
    # Helper function to extract content with balanced braces
    def extract_balanced_braces(text: str, start_pos: int) -> str:
        """Extract content from opening brace at start_pos, handling nested braces."""
        if start_pos >= len(text) or text[start_pos] != '{':
            return None
        
        depth = 0
        start = start_pos + 1  # Skip opening brace
        i = start_pos
        
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    # Found matching closing brace
                    return text[start:i]
            i += 1
        
        return None
    
    # Try to find boxed answer first (common in MATH dataset format)
    # Look for \boxed{ or \boxed\{ patterns
    boxed_patterns = [
        r'\\boxed\{',
        r'\\boxed{',
        r'\boxed\{',
        r'\boxed{',
    ]
    
    for pattern in boxed_patterns:
        match = re.search(pattern, text)
        if match:
            # Find the position after the pattern
            start_pos = match.end()
            # Extract balanced braces content
            content = extract_balanced_braces(text, start_pos - 1)
            if content:
                return content.strip()
    
    # If no boxed answer, try to find the last line or expression
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove common prefixes
            for prefix in ['Answer:', 'answer:', 'The answer is', 'Therefore']:
                if line.lower().startswith(prefix.lower()):
                    line = line[len(prefix):].strip()
                    if line.startswith(':'):
                        line = line[1:].strip()
                    return line
            return line
    
    return text.strip()


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    Removes extra whitespace and normalizes LaTeX formatting.
    """
    # Normalize common LaTeX patterns first (before whitespace normalization)
    answer = answer.replace('\\left(', '(').replace('\\right)', ')')
    answer = answer.replace('\\left[', '[').replace('\\right]', ']')
    answer = answer.replace('\\left{', '{').replace('\\right}', '}')
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    # Normalize spacing around parentheses, brackets, and commas
    # Remove spaces immediately after opening parentheses/brackets
    answer = re.sub(r'\(\s+', '(', answer)
    answer = re.sub(r'\[\s+', '[', answer)
    # Remove spaces immediately before closing parentheses/brackets
    answer = re.sub(r'\s+\)', ')', answer)
    answer = re.sub(r'\s+\]', ']', answer)
    # Normalize spacing around commas (remove spaces before, keep one after)
    answer = re.sub(r'\s*,\s*', ', ', answer)
    
    return answer.strip()


def compare_answers(predicted: str, ground_truth: str) -> bool:
    """
    Compare predicted answer with ground truth answer.
    Handles both exact match and symbolic equality.
    """
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    # Try exact match first
    if pred_norm == gt_norm:
        return True
    
    # Try symbolic comparison for mathematical expressions
    try:
        # Remove text wrappers like \text{}
        pred_clean = re.sub(r'\\text\{([^}]+)\}', r'\1', pred_norm)
        gt_clean = re.sub(r'\\text\{([^}]+)\}', r'\1', gt_norm)
        
        # Try to parse as symbolic expressions
        pred_expr = sympify(pred_clean, evaluate=False)
        gt_expr = sympify(gt_clean, evaluate=False)
        
        # Check if they're equal
        if simplify(pred_expr - gt_expr) == 0:
            return True
        
        # Try numerical comparison if both are numbers
        pred_num = N(pred_expr)
        gt_num = N(gt_expr)
        if abs(float(pred_num) - float(gt_num)) < 1e-6:
            return True
    except:
        pass
    
    # Try string similarity (for text answers)
    if pred_norm.lower() == gt_norm.lower():
        return True
    
    # Check if ground truth is contained in predicted (for verbose answers)
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return True
    
    return False


def evaluate_model(
    model: LLM,
    sampling_params: SamplingParams,
    problems: List[Dict],
    model_name: str,
    num_runs: int = 10,
    debug_log_file: str = None,
    tokenizer = None,
    use_chat_template: bool = False,
    system_prompt: str = None
) -> Dict[str, List[bool]]:
    """
    Evaluate a model on all problems, running each problem num_runs times.

    Returns a dictionary mapping problem_id to list of correctness results.
    """
    results = {}

    # Open debug log file if specified
    debug_log = None
    if debug_log_file:
        debug_log = open(debug_log_file, 'a', encoding='utf-8')

    # Prepare prompts
    prompts = []
    problem_ids = []

    for item in problems:
        problem_text = item['problem']
        user_content = f"Solve the following math problem step by step.\n\nProblem: {problem_text}"

        if use_chat_template and tokenizer is not None:
            # Use chat template for models that require it (e.g., Nemotron)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"{user_content}\n\nSolution:"

        # Add each problem num_runs times
        for _ in range(num_runs):
            prompts.append(prompt)
            problem_ids.append(item['unique_id'])
    
    # Run inference in batches
    print(f"\nRunning inference for {model_name}...")
    outputs = model.generate(prompts, sampling_params)
    
    # Process results
    current_idx = 0
    for item in tqdm(problems, desc=f"Processing {model_name}"):
        problem_id = item['unique_id']
        ground_truth = item['answer']
        problem_text = item['problem']
        
        correctness_list = []
        for run_idx in range(num_runs):
            output = outputs[current_idx]
            generated_text = output.outputs[0].text
            extracted_answer = extract_answer(generated_text)
            is_correct = compare_answers(extracted_answer, ground_truth)
            correctness_list.append(is_correct)
            
            # Log transcript and extracted answer for debugging
            if debug_log:
                log_entry = {
                    'model_name': model_name,
                    'problem_id': problem_id,
                    'run_index': run_idx,
                    'problem': problem_text,
                    'ground_truth': ground_truth,
                    'transcript': generated_text,
                    'extracted_answer': extracted_answer,
                    'is_correct': is_correct
                }
                debug_log.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                debug_log.flush()
            
            current_idx += 1
        
        results[problem_id] = correctness_list
    
    # Close debug log file
    if debug_log:
        debug_log.close()
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate models on MATH-500 dataset"
    )
    parser.add_argument(
        "--model",
        choices=["nvidia", "qwen", "both"],
        default="both",
        help="Which model(s) to evaluate: 'nvidia', 'qwen', or 'both' (default: both)"
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        default="math500_debug_log.jsonl",
        help="Output file for debug log (default: math500_debug_log.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="math500_evaluation_results.json",
        help="Output file for evaluation results (default: math500_evaluation_results.json)"
    )
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    print("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = list(dataset)
    
    print(f"Loaded {len(problems)} problems")
    
    # Model configurations
    # Note: Model paths may need to be adjusted based on actual HuggingFace model names
    # Common alternatives:
    # - Qwen3-14B: "Qwen/Qwen3-14B-Base" or "Qwen/Qwen3-14B"
    # - Nemotron: "nvidia/NVIDIA-Nemotron-Nano-12B-v2" or "nvidia/Nemotron-Nano-12B-v2"
    # Quantization options:
    # - dtype: "float16" (half precision, ~2x memory reduction) or "bfloat16"
    # - quantization: "awq" or "gptq" (if quantized models available, ~4x memory reduction)
    # - gpu_memory_utilization: 0.0-1.0 (fraction of GPU memory to use)
    all_models_config = {
        "Qwen3-14B": {
            "model_path": "Qwen/Qwen3-14B",
            "max_tokens": 4096,
            "temperature": 0.7,
            "dtype": "float16",  # Use half precision to reduce memory
            "quantization": None,  # Set to "awq" or "gptq" if quantized model available
            "gpu_memory_utilization": 0.85,  # Use 85% of GPU memory
            "max_model_len": 8192,  # Limit context length to save memory
        },
        "NVIDIA-Nemotron-Nano-12B-v2": {
            "model_path": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            "max_tokens": 4096,
            "temperature": 0.7,
            "dtype": "float16",  # Use half precision to reduce memory
            "quantization": None,  # Set to "awq" or "gptq" if quantized model available
            "gpu_memory_utilization": 0.85,  # Use 85% of GPU memory
            "max_model_len": 8192,  # Limit context length to save memory
            "use_chat_template": True,  # Nemotron requires chat template
            "system_prompt": "/think",  # Enable reasoning mode
        }
    }

    # Filter models based on CLI argument
    if args.model == "nvidia":
        models_config = {"NVIDIA-Nemotron-Nano-12B-v2": all_models_config["NVIDIA-Nemotron-Nano-12B-v2"]}
    elif args.model == "qwen":
        models_config = {"Qwen3-14B": all_models_config["Qwen3-14B"]}
    else:
        models_config = all_models_config

    print(f"Models to evaluate: {list(models_config.keys())}")
    
    all_results = {}
    num_runs = 10
    
    # Set up debug log file for transcripts and extracted answers
    debug_log_file = Path(args.debug_log)
    # Clear/create the debug log file
    if debug_log_file.exists():
        debug_log_file.unlink()
    print(f"\nDebug logging enabled: transcripts and extracted answers will be saved to {debug_log_file}")
    
    # Evaluate each model
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        try:
            # Initialize model with quantization settings
            print(f"Loading model: {config['model_path']}")
            print(f"  Quantization: {config.get('quantization', 'None (using dtype={})'.format(config.get('dtype', 'auto')))}")
            print(f"  GPU memory utilization: {config.get('gpu_memory_utilization', 0.9)}")
            
            # Build LLM arguments
            llm_kwargs = {
                "model": config['model_path'],
                "trust_remote_code": True,
                "tensor_parallel_size": 1,  # Adjust based on available GPUs
                "seed": 42,
                "dtype": config.get('dtype', 'auto'),
                "gpu_memory_utilization": config.get('gpu_memory_utilization', 0.9),
                "max_model_len": config.get('max_model_len', None),
            }
            
            # Add quantization if specified
            if config.get('quantization'):
                llm_kwargs["quantization"] = config['quantization']
            
            # Remove None values
            llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
            
            llm = LLM(**llm_kwargs)

            # Load tokenizer if chat template is needed
            tokenizer = None
            use_chat_template = config.get('use_chat_template', False)
            if use_chat_template:
                print(f"Loading tokenizer for chat template...")
                tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)

            # Set up sampling parameters
            # Note: seed is not set to allow different reasoning trajectories for each run
            sampling_params = SamplingParams(
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                top_p=0.95,
            )

            # Evaluate model
            results = evaluate_model(
                llm, sampling_params, problems, model_name, num_runs, str(debug_log_file),
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                system_prompt=config.get('system_prompt')
            )
            all_results[model_name] = results

            # Clean up
            del llm
            if tokenizer is not None:
                del tokenizer
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            print(f"Skipping {model_name}...")
            continue
    
    # Process all evaluated problems and save results
    print(f"\n{'='*60}")
    print("Analyzing results...")
    print(f"{'='*60}")
    
    all_evaluated_problems = []
    problems_in_range = []
    
    # Get all problem IDs that were evaluated (may be subset if some models failed)
    evaluated_problem_ids = set()
    for results in all_results.values():
        evaluated_problem_ids.update(results.keys())
    
    # Create a mapping of problem_id to problem data
    problem_map = {p['unique_id']: p for p in problems}
    
    for problem_id in evaluated_problem_ids:
        if problem_id not in problem_map:
            continue
            
        problem = problem_map[problem_id]
        problem_data = {
            'unique_id': problem_id,
            'problem': problem['problem'],
            'answer': problem['answer'],
            'subject': problem.get('subject', 'Unknown'),
            'level': problem.get('level', 'Unknown'),
            'model_results': {}
        }
        
        in_range = False
        
        for model_name, results in all_results.items():
            if problem_id in results:
                correctness_list = results[problem_id]
                accuracy = sum(correctness_list) / len(correctness_list)
                problem_data['model_results'][model_name] = {
                    'accuracy': accuracy,
                    'correct_count': sum(correctness_list),
                    'total_runs': len(correctness_list),
                    'correctness_list': correctness_list
                }
                
                # Check if accuracy is in 25%-75% range
                if 0.25 <= accuracy <= 0.75:
                    in_range = True
        
        # Save all evaluated problems (not just filtered ones)
        # This allows load_problems_in_range.py to filter by any range
        all_evaluated_problems.append(problem_data)
        if in_range:
            problems_in_range.append(problem_data)
    
    # Save results
    output_file = Path(args.output)
    print(f"\nSaving results to {output_file}...")
    
    output_data = {
        'summary': {
            'total_problems': len(problems),
            'problems_evaluated': len(all_evaluated_problems),
            'problems_in_range': len(problems_in_range),
            'models_evaluated': list(all_results.keys()),
            'runs_per_problem': num_runs,
        },
        'problems': all_evaluated_problems
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Total problems in dataset: {len(problems)}")
    print(f"Problems evaluated: {len(all_evaluated_problems)}")
    print(f"Problems with 25%-75% accuracy: {len(problems_in_range)}")
    print(f"Results saved to: {output_file}")
    print(f"Debug log saved to: {debug_log_file}")
    print(f"\nNote: All evaluated problems are saved (not just filtered ones).")
    print(f"      Use load_problems_in_range.py to filter by different accuracy ranges.")
    
    # Print summary statistics
    for model_name in all_results.keys():
        print(f"\n{model_name}:")
        model_problems_evaluated = sum(
            1 for p in all_evaluated_problems
            if model_name in p['model_results']
        )
        model_problems_in_range = sum(
            1 for p in all_evaluated_problems
            if model_name in p['model_results'] and 
            0.25 <= p['model_results'][model_name]['accuracy'] <= 0.75
        )
        print(f"  Problems evaluated: {model_problems_evaluated}")
        print(f"  Problems in 25%-75% range: {model_problems_in_range}")


if __name__ == "__main__":
    main()

