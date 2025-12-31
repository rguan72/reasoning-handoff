import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from datasets import load_dataset

from by_hand.answer_extraction import compare_answers, extract_answer
from by_hand.inference import run_inference_batch, cleanup_model_memory
from by_hand.prompts import construct_prompt


def load_aime25_problems(limit: int = 100) -> List[Dict]:
    """
    Load AIME25 dataset from HuggingFace.
    
    Args:
        limit: Maximum number of problems to return
    
    Returns:
        List of problem dictionaries with fields: id, problem, answer
    """
    print(f"Loading AIME25 dataset...")
    dataset = load_dataset("math-ai/aime25", split="test")
    
    problems = []
    for i, item in enumerate(dataset):
        problems.append({
            'unique_id': item.get('id'),
            'problem': item.get('problem'),
            'answer': item.get('answer'),
        })
        if len(problems) >= limit:
            break
    
    print(f"Found {len(problems)} problems (limited to {limit})")
    return problems


def evaluate_model(model_key: str, problems: List[Dict] = None, num_runs: int = 20):
    """
    Evaluate a model on math problems.
    
    Args:
        model_key: Model key from model_configs
        problems: List of problem dictionaries. If None, loads first 10 problems from AIME25
        num_runs: Number of runs per problem
    
    Returns:
        Dictionary with overall_accuracy, extracted_accuracy, and per-problem results
    """
    # Load problems if not provided
    if problems is None:
        problems = load_aime25_problems()
    
    # Prepare all prompts for batch inference
    # Each problem gets num_runs prompts, so we create a list of (problem_idx, prompt) tuples
    all_prompts = []
    problem_indices = []  # Maps each prompt to its problem index
    
    for problem_idx, problem in enumerate(problems):
        problem_text = problem['problem']
        prompt = construct_prompt(problem_text)
        # Add num_runs copies of this prompt
        for _ in range(num_runs):
            all_prompts.append(prompt)
            problem_indices.append(problem_idx)
    
    print(f"Running batch inference on {len(all_prompts)} prompts ({len(problems)} problems × {num_runs} runs)...")
    
    # Run batch inference on all prompts at once
    all_results_raw = run_inference_batch(model_key=model_key, prompts=all_prompts)
    
    # Group results by problem
    all_results = []
    all_extracted_answers = []
    all_correct_answers = []
    
    for problem_idx, problem in enumerate(problems):
        # Get all results for this problem
        problem_result_indices = [i for i, p_idx in enumerate(problem_indices) if p_idx == problem_idx]
        results = [all_results_raw[i] for i in problem_result_indices]
        
        problem_text = problem['problem']
        ground_truth = problem['answer']
        problem_id = problem.get('unique_id', 'unknown')
        
        extracted_answers = [extract_answer(result) for result in results]
        
        # Handle None cases: filter out None values for extracted-only accuracy
        # For overall accuracy, None counts as incorrect
        correct_answers = []
        for extracted_answer in extracted_answers:
            if extracted_answer is None:
                correct_answers.append(False)  # None counts as incorrect for overall accuracy
            else:
                correct_answers.append(compare_answers(extracted_answer, ground_truth))
        
        # Store results for this problem
        problem_results = {
            'problem_id': problem_id,
            'problem': problem_text,
            'ground_truth': ground_truth,
            'results': results,
            'extracted_answers': extracted_answers,
            'correct_answers': correct_answers,
            'overall_accuracy': sum(correct_answers) / len(correct_answers) if correct_answers else 0.0,
        }
        
        # Calculate extracted-only accuracy for this problem
        extracted_only = [(ans, correct) for ans, correct in zip(extracted_answers, correct_answers) if ans is not None]
        if extracted_only:
            problem_results['extracted_accuracy'] = sum(correct for _, correct in extracted_only) / len(extracted_only)
        else:
            problem_results['extracted_accuracy'] = 0.0
        
        all_results.append(problem_results)
        all_extracted_answers.extend(extracted_answers)
        all_correct_answers.extend(correct_answers)
    
    # Calculate overall statistics
    overall_accuracy = sum(all_correct_answers) / len(all_correct_answers) if all_correct_answers else 0.0
    
    # Accuracy only for extracted answers (excluding None cases)
    extracted_only = [(ans, correct) for ans, correct in zip(all_extracted_answers, all_correct_answers) if ans is not None]
    if extracted_only:
        extracted_accuracy = sum(correct for _, correct in extracted_only) / len(extracted_only)
    else:
        extracted_accuracy = 0.0
    
    # Save raw results to file in by_hand/runs directory
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = runs_dir / f"{model_key}_{timestamp}.json"
    
    # Save results with metadata
    output_data = {
        'model_key': model_key,
        'timestamp': timestamp,
        'num_problems': len(problems),
        'num_runs_per_problem': num_runs,
        'overall_accuracy': overall_accuracy,
        'extracted_accuracy': extracted_accuracy,
        'extraction_rate': len(extracted_only) / len(all_extracted_answers) if all_extracted_answers else 0.0,
        'problems': all_results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print(f"\n=== Overall Statistics ===")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({sum(all_correct_answers)}/{len(all_correct_answers)})")
    print(f"Accuracy (extracted only): {extracted_accuracy:.4f} ({sum(correct for _, correct in extracted_only)}/{len(extracted_only)})")
    print(f"Extraction rate: {len(extracted_only)}/{len(all_extracted_answers)} ({len(extracted_only)/len(all_extracted_answers):.4f})")
    
    # Print per-problem accuracies
    print_per_problem_accuracies(all_results)
    
    # Clean up model memory to free GPU resources
    print(f"\nCleaning up model memory for {model_key}...")
    cleanup_model_memory(model_key)
    
    return {
        'overall_accuracy': overall_accuracy,
        'extracted_accuracy': extracted_accuracy,
        'results': all_results
    }


def print_per_problem_accuracies(problem_results: List[Dict]):
    """
    Print accuracy statistics for each individual problem.
    
    Args:
        problem_results: List of problem result dictionaries from evaluate_model
    """
    print(f"\n=== Per-Problem Accuracies ===")
    for i, result in enumerate(problem_results, 1):
        problem_id = result.get('problem_id', f'problem_{i}')
        overall_acc = result.get('overall_accuracy', 0.0)
        extracted_acc = result.get('extracted_accuracy', 0.0)
        num_correct = sum(result.get('correct_answers', []))
        num_total = len(result.get('correct_answers', []))
        num_extracted = len([a for a in result.get('extracted_answers', []) if a is not None])
        
        print(f"Problem {i} (ID: {problem_id}):")
        print(f"  Overall: {overall_acc:.4f} ({num_correct}/{num_total})")
        print(f"  Extracted only: {extracted_acc:.4f} ({num_correct}/{num_extracted})")
        print(f"  Extraction rate: {num_extracted}/{num_total} ({num_extracted/num_total:.4f})")


def filter_problems_by_accuracy(
    problem_results: List[Dict],
    min_overall_acc: Optional[float] = None,
    max_overall_acc: Optional[float] = None,
    min_extracted_acc: Optional[float] = None,
    max_extracted_acc: Optional[float] = None,
    accuracy_type: str = 'overall'
) -> List[Dict]:
    """
    Filter problems by accuracy range.
    
    Args:
        problem_results: List of problem result dictionaries from evaluate_model
        min_overall_acc: Minimum overall accuracy (inclusive)
        max_overall_acc: Maximum overall accuracy (inclusive)
        min_extracted_acc: Minimum extracted-only accuracy (inclusive)
        max_extracted_acc: Maximum extracted-only accuracy (inclusive)
        accuracy_type: Which accuracy to filter by ('overall', 'extracted', or 'both')
    
    Returns:
        Filtered list of problem results
    """
    filtered = []
    
    for result in problem_results:
        overall_acc = result.get('overall_accuracy', 0.0)
        extracted_acc = result.get('extracted_accuracy', 0.0)
        
        # Check overall accuracy filters
        if accuracy_type in ('overall', 'both'):
            if min_overall_acc is not None and overall_acc < min_overall_acc:
                continue
            if max_overall_acc is not None and overall_acc > max_overall_acc:
                continue
        
        # Check extracted accuracy filters
        if accuracy_type in ('extracted', 'both'):
            if min_extracted_acc is not None and extracted_acc < min_extracted_acc:
                continue
            if max_extracted_acc is not None and extracted_acc > max_extracted_acc:
                continue
        
        filtered.append(result)
    
    return filtered


def analyze_results_file(
    results_file: str,
    min_overall_acc: Optional[float] = None,
    max_overall_acc: Optional[float] = None,
    min_extracted_acc: Optional[float] = None,
    max_extracted_acc: Optional[float] = None,
    accuracy_type: str = 'overall'
):
    """
    Load and analyze results from a saved JSON file, optionally filtering by accuracy range.
    
    Args:
        results_file: Path to the results JSON file
        min_overall_acc: Minimum overall accuracy (inclusive)
        max_overall_acc: Maximum overall accuracy (inclusive)
        min_extracted_acc: Minimum extracted-only accuracy (inclusive)
        max_extracted_acc: Maximum extracted-only accuracy (inclusive)
        accuracy_type: Which accuracy to filter by ('overall', 'extracted', or 'both')
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problem_results = data.get('problems', [])
    
    # Apply filters if provided
    if any([min_overall_acc is not None, max_overall_acc is not None,
            min_extracted_acc is not None, max_extracted_acc is not None]):
        problem_results = filter_problems_by_accuracy(
            problem_results,
            min_overall_acc=min_overall_acc,
            max_overall_acc=max_overall_acc,
            min_extracted_acc=min_extracted_acc,
            max_extracted_acc=max_extracted_acc,
            accuracy_type=accuracy_type
        )
        print(f"\nFiltered to {len(problem_results)} problems matching criteria")
    
    # Print statistics
    print(f"\n=== Results Analysis ===")
    print(f"Model: {data.get('model_key', 'unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'unknown')}")
    print(f"Number of problems: {len(problem_results)}")
    print(f"Runs per problem: {data.get('num_runs_per_problem', 'unknown')}")
    
    if problem_results:
        overall_accs = [p.get('overall_accuracy', 0.0) for p in problem_results]
        extracted_accs = [p.get('extracted_accuracy', 0.0) for p in problem_results]
        
        print(f"\nOverall Accuracy:")
        print(f"  Mean: {sum(overall_accs)/len(overall_accs):.4f}")
        print(f"  Min: {min(overall_accs):.4f}")
        print(f"  Max: {max(overall_accs):.4f}")
        
        print(f"\nExtracted-Only Accuracy:")
        print(f"  Mean: {sum(extracted_accs)/len(extracted_accs):.4f}")
        print(f"  Min: {min(extracted_accs):.4f}")
        print(f"  Max: {max(extracted_accs):.4f}")
        
        # Print per-problem details
        print_per_problem_accuracies(problem_results)
    
    return problem_results

if __name__ == "__main__":
    import sys
    
    # Check if we're analyzing an existing results file
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        if len(sys.argv) < 3:
            print("Usage: python script.py analyze <results_file> [--min-overall FLOAT] [--max-overall FLOAT] [--min-extracted FLOAT] [--max-extracted FLOAT] [--type overall|extracted|both]")
            sys.exit(1)
        
        results_file = sys.argv[2]
        min_overall = None
        max_overall = None
        min_extracted = None
        max_extracted = None
        accuracy_type = 'overall'
        
        # Parse command-line arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == '--min-overall' and i + 1 < len(sys.argv):
                min_overall = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--max-overall' and i + 1 < len(sys.argv):
                max_overall = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--min-extracted' and i + 1 < len(sys.argv):
                min_extracted = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--max-extracted' and i + 1 < len(sys.argv):
                max_extracted = float(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--type' and i + 1 < len(sys.argv):
                accuracy_type = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        analyze_results_file(
            results_file,
            min_overall_acc=min_overall,
            max_overall_acc=max_overall,
            min_extracted_acc=min_extracted,
            max_extracted_acc=max_extracted,
            accuracy_type=accuracy_type
        )
    else:
        # Run evaluation
        evaluate_model("deep-llama")
        evaluate_model("deep-qwen")