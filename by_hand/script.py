import json
from datetime import datetime
from pathlib import Path

from by_hand.answer_extraction import compare_answers, extract_answer
from by_hand.inference import run_inference
from by_hand.prompts import construct_prompt, ground_truth, math_prompt

def evaluate_model(model_key: str):
    results = run_inference(model_key=model_key, prompt=construct_prompt(math_prompt), num_runs=20)
    extracted_answers = [extract_answer(result) for result in results]
    
    # Handle None cases: filter out None values for extracted-only accuracy
    # For overall accuracy, None counts as incorrect
    correct_answers = []
    for extracted_answer in extracted_answers:
        if extracted_answer is None:
            correct_answers.append(False)  # None counts as incorrect for overall accuracy
        else:
            correct_answers.append(compare_answers(extracted_answer, ground_truth))
    
    # Overall accuracy (including None cases as incorrect)
    overall_accuracy = sum(correct_answers) / len(correct_answers)
    
    # Accuracy only for extracted answers (excluding None cases)
    extracted_only = [(ans, correct) for ans, correct in zip(extracted_answers, correct_answers) if ans is not None]
    if extracted_only:
        extracted_accuracy = sum(correct for _, correct in extracted_only) / len(extracted_only)
    else:
        extracted_accuracy = 0.0
    
    # Save raw results to file in by_hand/runs directory
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = runs_dir / f"{model_key}_{timestamp}.json"
    
    # Save just the raw results (list of transcripts)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    print(f"Overall accuracy: {overall_accuracy:.4f} ({sum(correct_answers)}/{len(correct_answers)})")
    print(f"Accuracy (extracted only): {extracted_accuracy:.4f} ({sum(correct for _, correct in extracted_only)}/{len(extracted_only)})")
    print(f"Extraction rate: {len(extracted_only)}/{len(extracted_answers)} ({len(extracted_only)/len(extracted_answers):.4f})")
    
    return overall_accuracy, extracted_accuracy

evaluate_model("qwen8")
evaluate_model("deep")