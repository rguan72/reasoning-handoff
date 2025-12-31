from by_hand.answer_extraction import compare_answers, extract_answer
from by_hand.inference import run_inference
from by_hand.prompts import ground_truth, math_prompt

def evaluate_model(model_key: str):
    results = run_inference(model_key=model_key, prompt=math_prompt, num_runs=20)
    extracted_answers = [extract_answer(result) for result in results]
    correct_answers = [compare_answers(extracted_answer, ground_truth) for extracted_answer in extracted_answers]
    return sum(correct_answers) / len(correct_answers)

print(evaluate_model("qwen8"))
print(evaluate_model("deep"))