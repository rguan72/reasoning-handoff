import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset

from by_hand.answer_extraction import compare_answers, extract_answer
from by_hand.inference import run_inference
from by_hand.prompts import construct_prompt


def load_math500_problems(limit: int = 10, levels: List[int] = [3, 4]) -> List[Dict]:
    """
    Load MATH-500 dataset and filter for level 3/4 problems.
    
    Args:
        limit: Maximum number of problems to return
        levels: List of levels to filter for (default: [3, 4])
    
    Returns:
        List of problem dictionaries with fields: unique_id, problem, answer, level, subject
    """
    print(f"Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    # Filter for level 3 or 4 problems
    filtered_problems = []
    for item in dataset:
        level = item.get('level')
        if level in levels:
            filtered_problems.append({
                'unique_id': item.get('unique_id'),
                'problem': item.get('problem'),
                'answer': item.get('answer'),
                'level': level,
                'subject': item.get('subject', 'Unknown')
            })
            if len(filtered_problems) >= limit:
                break
    
    print(f"Found {len(filtered_problems)} level {levels} problems (limited to {limit})")
    return filtered_problems


def evaluate_model(model_key: str, problems: List[Dict] = None, num_runs: int = 20):
    """
    Evaluate a model on math problems.
    
    Args:
        model_key: Model key from model_configs
        problems: List of problem dictionaries. If None, loads first 10 level 3/4 problems from MATH-500
        num_runs: Number of runs per problem
    
    Returns:
        Dictionary with overall_accuracy, extracted_accuracy, and per-problem results
    """
    # Load problems if not provided
    if problems is None:
        problems = load_math500_problems(limit=10, levels=[3, 4])
    
    all_results = []
    all_extracted_answers = []
    all_correct_answers = []
    
    # Evaluate each problem
    for problem in problems:
        problem_text = problem['problem']
        ground_truth = problem['answer']
        problem_id = problem.get('unique_id', 'unknown')
        
        prompt = construct_prompt(problem_text)
        results = run_inference(model_key=model_key, prompt=prompt, num_runs=num_runs)
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
            'level': problem.get('level'),
            'subject': problem.get('subject'),
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
    print(f"Overall accuracy: {overall_accuracy:.4f} ({sum(all_correct_answers)}/{len(all_correct_answers)})")
    print(f"Accuracy (extracted only): {extracted_accuracy:.4f} ({sum(correct for _, correct in extracted_only)}/{len(extracted_only)})")
    print(f"Extraction rate: {len(extracted_only)}/{len(all_extracted_answers)} ({len(extracted_only)/len(all_extracted_answers):.4f})")
    
    return {
        'overall_accuracy': overall_accuracy,
        'extracted_accuracy': extracted_accuracy,
        'results': all_results
    }

if __name__ == "__main__":
    evaluate_model("qwen8")
    evaluate_model("deep")