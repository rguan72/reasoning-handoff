import json
from datetime import datetime
from pathlib import Path

from by_hand.answer_extraction import compare_answers, extract_answer
from by_hand.inference import run_inference
from by_hand.prompts import ground_truth, math_prompt

def evaluate_model(model_key: str):
    results = run_inference(model_key=model_key, prompt=math_prompt, num_runs=20)
    extracted_answers = [extract_answer(result) for result in results]
    correct_answers = [compare_answers(extracted_answer, ground_truth) for extracted_answer in extracted_answers]
    accuracy = sum(correct_answers) / len(correct_answers)
    
    # Save raw results to file in by_hand/runs directory
    runs_dir = Path(__file__).parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = runs_dir / f"{model_key}_{timestamp}.json"
    
    # Save just the raw results (list of transcripts)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")
    
    return accuracy

print(evaluate_model("qwen8"))
print(evaluate_model("deep"))