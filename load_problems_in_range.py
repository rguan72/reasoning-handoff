"""
Load problems from MATH-500 evaluation results that fall within a specific accuracy range.

This script loads problems from the evaluation results JSON file and filters them
based on accuracy ranges (e.g., 25%-75%) for use in other experiments.

Usage:
    python load_problems_in_range.py [--min-accuracy MIN] [--max-accuracy MAX] [--input-file FILE] [--output-file FILE]

Examples:
    # Load problems with 25%-75% accuracy (default)
    python load_problems_in_range.py

    # Load problems with 30%-70% accuracy
    python load_problems_in_range.py --min-accuracy 0.3 --max-accuracy 0.7

    # Specify custom input/output files
    python load_problems_in_range.py --input-file results.json --output-file filtered_problems.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def load_problems_in_range(
    input_file: str = "math500_evaluation_results.json",
    min_accuracy: float = 0.25,
    max_accuracy: float = 0.75,
    model_name: Optional[str] = None
) -> List[Dict]:
    """
    Load problems from evaluation results that fall within the specified accuracy range.
    
    Args:
        input_file: Path to the evaluation results JSON file
        min_accuracy: Minimum accuracy threshold (default: 0.25)
        max_accuracy: Maximum accuracy threshold (default: 0.75)
        model_name: Specific model to filter by (if None, checks all models)
    
    Returns:
        List of problem dictionaries that meet the criteria
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading results from {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = data.get('problems', [])
    print(f"Found {len(problems)} problems in results file")
    
    filtered_problems = []
    
    for problem in problems:
        problem_id = problem.get('unique_id')
        model_results = problem.get('model_results', {})
        
        # If specific model requested, only check that model
        models_to_check = [model_name] if model_name else list(model_results.keys())
        
        include_problem = False
        matching_models = []
        
        for model in models_to_check:
            if model in model_results:
                accuracy = model_results[model].get('accuracy', 0.0)
                
                if min_accuracy <= accuracy <= max_accuracy:
                    include_problem = True
                    matching_models.append({
                        'model': model,
                        'accuracy': accuracy,
                        'correct_count': model_results[model].get('correct_count', 0),
                        'total_runs': model_results[model].get('total_runs', 0)
                    })
        
        if include_problem:
            # Create a clean problem entry for experiments
            filtered_problem = {
                'unique_id': problem_id,
                'problem': problem.get('problem'),
                'answer': problem.get('answer'),
                'subject': problem.get('subject', 'Unknown'),
                'level': problem.get('level', 'Unknown'),
                'matching_models': matching_models
            }
            
            # Optionally include full model results
            filtered_problem['model_results'] = model_results
            
            filtered_problems.append(filtered_problem)
    
    return filtered_problems


def save_problems(problems: List[Dict], output_file: str):
    """
    Save filtered problems to a JSON file.
    
    Args:
        problems: List of problem dictionaries
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    
    output_data = {
        'summary': {
            'total_problems': len(problems),
            'problem_ids': [p['unique_id'] for p in problems]
        },
        'problems': problems
    }
    
    print(f"\nSaving {len(problems)} problems to {output_file}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Load problems from MATH-500 evaluation results within a specific accuracy range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load problems with 25%-75% accuracy (default)
  python load_problems_in_range.py

  # Load problems with 30%-70% accuracy
  python load_problems_in_range.py --min-accuracy 0.3 --max-accuracy 0.7

  # Filter by specific model
  python load_problems_in_range.py --model Qwen3-14B

  # Custom input/output files
  python load_problems_in_range.py --input-file results.json --output-file filtered.json
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        default='math500_evaluation_results.json',
        help='Path to input evaluation results JSON file (default: math500_evaluation_results.json)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='problems_in_range.json',
        help='Path to output JSON file (default: problems_in_range.json)'
    )
    
    parser.add_argument(
        '--min-accuracy',
        type=float,
        default=0.25,
        help='Minimum accuracy threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--max-accuracy',
        type=float,
        default=0.75,
        help='Maximum accuracy threshold (default: 0.75)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Filter by specific model name (if not specified, checks all models)'
    )
    
    args = parser.parse_args()
    
    # Validate accuracy range
    if not 0.0 <= args.min_accuracy <= 1.0:
        raise ValueError(f"min_accuracy must be between 0.0 and 1.0, got {args.min_accuracy}")
    if not 0.0 <= args.max_accuracy <= 1.0:
        raise ValueError(f"max_accuracy must be between 0.0 and 1.0, got {args.max_accuracy}")
    if args.min_accuracy >= args.max_accuracy:
        raise ValueError(f"min_accuracy ({args.min_accuracy}) must be less than max_accuracy ({args.max_accuracy})")
    
    print(f"{'='*60}")
    print("Loading Problems in Range")
    print(f"{'='*60}")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Accuracy range: {args.min_accuracy:.1%} - {args.max_accuracy:.1%}")
    if args.model:
        print(f"Model filter: {args.model}")
    print(f"{'='*60}\n")
    
    try:
        # Load problems
        problems = load_problems_in_range(
            input_file=args.input_file,
            min_accuracy=args.min_accuracy,
            max_accuracy=args.max_accuracy,
            model_name=args.model
        )
        
        # Save results
        save_problems(problems, args.output_file)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Total problems loaded: {len(problems)}")
        
        if problems:
            # Show breakdown by model
            model_counts = {}
            for problem in problems:
                for match in problem.get('matching_models', []):
                    model = match['model']
                    model_counts[model] = model_counts.get(model, 0) + 1
            
            print("\nProblems by model:")
            for model, count in model_counts.items():
                print(f"  {model}: {count} problems")
            
            # Show sample problem IDs
            print(f"\nSample problem IDs (first 10):")
            for problem in problems[:10]:
                print(f"  {problem['unique_id']}")
        
        print(f"\nProblems saved to: {args.output_file}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

