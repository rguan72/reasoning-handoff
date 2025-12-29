#!/usr/bin/env python3
"""
One-time use script to create a Hugging Face dataset subset.

This script:
1. Loads qwen_results.json and nvidia_results.json
2. Finds problems where BOTH models achieved 25-75% accuracy
3. Loads the original HuggingFaceH4/MATH-500 dataset
4. Filters it to only include those problems
5. Uploads it as a new dataset to your Hugging Face account

Usage:
    python create_subset_dataset.py --dataset-name YOUR_USERNAME/MATH-500-subset-25-75
"""

import json
import argparse
from pathlib import Path
from typing import Set, Dict, List
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login, whoami


def load_results_file(filepath: str) -> Dict[str, Dict]:
    """
    Load results JSON file and return a mapping of problem_id -> model_results.
    
    Args:
        filepath: Path to results JSON file
        
    Returns:
        Dictionary mapping problem_id to model_results dict
    """
    print(f"Loading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    for problem in data.get('problems', []):
        problem_id = problem.get('unique_id')
        if problem_id:
            results[problem_id] = problem.get('model_results', {})
    
    print(f"  Loaded {len(results)} problems")
    return results


def find_matching_problems(
    qwen_results: Dict[str, Dict],
    nvidia_results: Dict[str, Dict],
    min_accuracy: float = 0.25,
    max_accuracy: float = 0.75
) -> Set[str]:
    """
    Find problem IDs where both models have accuracy in the specified range.
    
    Args:
        qwen_results: Dictionary mapping problem_id -> model_results from qwen_results.json
        nvidia_results: Dictionary mapping problem_id -> model_results from nvidia_results.json
        min_accuracy: Minimum accuracy threshold (default: 0.25)
        max_accuracy: Maximum accuracy threshold (default: 0.75)
        
    Returns:
        Set of problem IDs that match the criteria
    """
    matching_ids = set()
    
    # Model names as they appear in the results
    qwen_model_name = "Qwen3-14B"
    nvidia_model_name = "NVIDIA-Nemotron-Nano-12B-v2"
    
    # Get all problem IDs that exist in both results
    common_ids = set(qwen_results.keys()) & set(nvidia_results.keys())
    print(f"\nFound {len(common_ids)} problems evaluated by both models")
    
    for problem_id in common_ids:
        qwen_model_results = qwen_results[problem_id].get(qwen_model_name, {})
        nvidia_model_results = nvidia_results[problem_id].get(nvidia_model_name, {})
        
        qwen_accuracy = qwen_model_results.get('accuracy', 0.0)
        nvidia_accuracy = nvidia_model_results.get('accuracy', 0.0)
        
        # Check if both models have accuracy in range
        qwen_in_range = min_accuracy <= qwen_accuracy <= max_accuracy
        nvidia_in_range = min_accuracy <= nvidia_accuracy <= max_accuracy
        
        if qwen_in_range and nvidia_in_range:
            matching_ids.add(problem_id)
            print(f"  Match: {problem_id} - Qwen: {qwen_accuracy:.2%}, NVIDIA: {nvidia_accuracy:.2%}")
    
    return matching_ids


def create_subset_dataset(problem_ids: Set[str], dataset_name: str):
    """
    Create a subset of HuggingFaceH4/MATH-500 and upload it to Hugging Face.
    
    Args:
        problem_ids: Set of problem IDs to include in the subset
        dataset_name: Name for the new dataset (e.g., "username/MATH-500-subset-25-75")
    """
    print(f"\n{'='*60}")
    print("Loading original HuggingFaceH4/MATH-500 dataset...")
    print(f"{'='*60}")
    
    # Load the original dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    print(f"Original dataset has {len(dataset)} problems")
    print(f"Filtering to {len(problem_ids)} matching problems...")
    
    # Filter the dataset to only include matching problems
    # The dataset uses 'unique_id' field to identify problems
    filtered_data = []
    for item in dataset:
        if item.get('unique_id') in problem_ids:
            filtered_data.append(item)
    
    print(f"Found {len(filtered_data)} problems in original dataset")
    
    if len(filtered_data) != len(problem_ids):
        missing = problem_ids - {item.get('unique_id') for item in filtered_data}
        if missing:
            print(f"Warning: {len(missing)} problem IDs not found in original dataset:")
            for pid in list(missing)[:10]:  # Show first 10
                print(f"  {pid}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
    
    # Create a new dataset from the filtered data
    print(f"\nCreating new dataset with {len(filtered_data)} problems...")
    subset_dataset = Dataset.from_list(filtered_data)
    
    # Create DatasetDict with test split (matching original structure)
    dataset_dict = DatasetDict({"test": subset_dataset})
    
    # Check if user is logged in
    print(f"\n{'='*60}")
    print(f"Uploading dataset to Hugging Face: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        user_info = whoami()
        print(f"Logged in as: {user_info.get('name', 'Unknown')}")
    except Exception:
        print("⚠ Not logged in to Hugging Face.")
        print("Please run: huggingface-cli login")
        print("Or use: python -c 'from huggingface_hub import login; login()'")
        raise RuntimeError("Not logged in to Hugging Face. Please login first.")
    
    print(f"{'='*60}\n")
    
    try:
        print("Uploading dataset (this may take a few minutes)...")
        dataset_dict.push_to_hub(dataset_name)
        print(f"\n✓ Successfully uploaded dataset to: https://huggingface.co/datasets/{dataset_name}")
    except Exception as e:
        print(f"\n✗ Error uploading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Verify the dataset name follows the format: username/dataset-name")
        print("3. Ensure you have write access to create the dataset repository")
        print("4. Check your internet connection")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Create a Hugging Face dataset subset from MATH-500 evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--qwen-results',
        type=str,
        default='qwen_results.json',
        help='Path to qwen_results.json file (default: qwen_results.json)'
    )
    
    parser.add_argument(
        '--nvidia-results',
        type=str,
        default='nvidia_results.json',
        help='Path to nvidia_results.json file (default: nvidia_results.json)'
    )
    
    parser.add_argument(
        '--dataset-name',
        type=str,
        required=True,
        help='Name for the new dataset (e.g., "username/MATH-500-subset-25-75")'
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.qwen_results).exists():
        raise FileNotFoundError(f"Qwen results file not found: {args.qwen_results}")
    if not Path(args.nvidia_results).exists():
        raise FileNotFoundError(f"NVIDIA results file not found: {args.nvidia_results}")
    
    if not 0.0 <= args.min_accuracy <= 1.0:
        raise ValueError(f"min_accuracy must be between 0.0 and 1.0, got {args.min_accuracy}")
    if not 0.0 <= args.max_accuracy <= 1.0:
        raise ValueError(f"max_accuracy must be between 0.0 and 1.0, got {args.max_accuracy}")
    if args.min_accuracy >= args.max_accuracy:
        raise ValueError(f"min_accuracy ({args.min_accuracy}) must be less than max_accuracy ({args.max_accuracy})")
    
    print(f"{'='*60}")
    print("Creating MATH-500 Subset Dataset")
    print(f"{'='*60}")
    print(f"Qwen results: {args.qwen_results}")
    print(f"NVIDIA results: {args.nvidia_results}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Accuracy range: {args.min_accuracy:.1%} - {args.max_accuracy:.1%}")
    print(f"{'='*60}\n")
    
    # Load results files
    qwen_results = load_results_file(args.qwen_results)
    nvidia_results = load_results_file(args.nvidia_results)
    
    # Find matching problems
    matching_ids = find_matching_problems(
        qwen_results,
        nvidia_results,
        args.min_accuracy,
        args.max_accuracy
    )
    
    print(f"\n{'='*60}")
    print(f"Found {len(matching_ids)} problems matching criteria")
    print(f"{'='*60}")
    
    if not matching_ids:
        print("No matching problems found. Exiting.")
        return
    
    # Create and upload dataset
    create_subset_dataset(matching_ids, args.dataset_name)
    
    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")
    print(f"Dataset available at: https://huggingface.co/datasets/{args.dataset_name}")


if __name__ == "__main__":
    main()

