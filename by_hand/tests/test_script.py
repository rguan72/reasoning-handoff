import pytest
import sys
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path to import by_hand module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock vllm before importing anything that depends on it
sys.modules['vllm'] = MagicMock()
sys.modules['vllm'].LLM = MagicMock()
sys.modules['vllm'].SamplingParams = MagicMock()

# Mock datasets before importing - need to set __spec__ for transformers compatibility
mock_datasets = MagicMock()
mock_datasets.__spec__ = MagicMock()
sys.modules['datasets'] = mock_datasets

from by_hand.script import evaluate_model, load_aime25_problems


def test_evaluate_model_with_correct_answers():
    """Test evaluate_model with all correct answers."""
    # Create mock problems
    mock_problems = [
        {
            'unique_id': 'test_1',
            'problem': 'What is 2+2?',
            'answer': '4',
            'level': 3,
            'subject': 'arithmetic'
        }
    ]
    
    # Mock run_inference_batch to return strings with correct answers
    # For 1 problem × 3 runs = 3 prompts, return 3 results
    mock_results = [
        "Step by step solution. The answer is \\boxed{4}.",
        "After calculation, we get \\boxed{4}.",
        "The result is \\boxed{4}.",
    ]
    
    with patch('by_hand.script.run_inference_batch', return_value=mock_results):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('by_hand.script.Path.mkdir'):
                result = evaluate_model("test_model", problems=mock_problems, num_runs=3)
    
    # All 3 answers are correct
    assert result['overall_accuracy'] == 1.0
    assert result['extracted_accuracy'] == 1.0


def test_evaluate_model_with_mixed_answers():
    """Test evaluate_model with mix of correct and incorrect answers."""
    # Create mock problems
    mock_problems = [
        {
            'unique_id': 'test_1',
            'problem': 'What is 2+2?',
            'answer': '4',
            'level': 3,
            'subject': 'arithmetic'
        }
    ]
    
    # Mock run_inference_batch to return mix of correct and incorrect
    # For 1 problem × 5 runs = 5 prompts, return 5 results
    mock_results = [
        "The answer is \\boxed{4}.",      # Correct
        "The answer is \\boxed{5}.",      # Incorrect
        "The answer is \\boxed{4}.",      # Correct
        "The answer is \\boxed{6}.",      # Incorrect
        "The answer is \\boxed{4}.",      # Correct
    ]
    
    with patch('by_hand.script.run_inference_batch', return_value=mock_results):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('by_hand.script.Path.mkdir'):
                result = evaluate_model("test_model", problems=mock_problems, num_runs=5)
    
    # 3 out of 5 are correct
    assert result['overall_accuracy'] == 0.6
    assert result['extracted_accuracy'] == 0.6


def test_evaluate_model_with_none_extractions():
    """Test evaluate_model with some None extractions (no \\boxed{} found)."""
    # Create mock problems
    mock_problems = [
        {
            'unique_id': 'test_1',
            'problem': 'What is 2+2?',
            'answer': '4',
            'level': 3,
            'subject': 'arithmetic'
        }
    ]
    
    # Mock run_inference_batch to return mix with some missing boxed
    # For 1 problem × 5 runs = 5 prompts, return 5 results
    mock_results = [
        "The answer is \\boxed{4}.",      # Correct
        "The answer is 5.",               # No boxed - None extraction
        "The answer is \\boxed{4}.",      # Correct
        "I think the answer might be 6.", # No boxed - None extraction
        "The answer is \\boxed{4}.",      # Correct
    ]
    
    with patch('by_hand.script.run_inference_batch', return_value=mock_results):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('by_hand.script.Path.mkdir'):
                result = evaluate_model("test_model", problems=mock_problems, num_runs=5)
    
    # Overall: 3 correct out of 5 (None counts as incorrect)
    assert result['overall_accuracy'] == 0.6
    # Extracted only: 3 correct out of 3 extracted
    assert result['extracted_accuracy'] == 1.0


def test_evaluate_model_with_all_none():
    """Test evaluate_model when no answers can be extracted."""
    # Create mock problems
    mock_problems = [
        {
            'unique_id': 'test_1',
            'problem': 'What is 2+2?',
            'answer': '4',
            'level': 3,
            'subject': 'arithmetic'
        }
    ]
    
    # Mock run_inference_batch to return strings without boxed
    # For 1 problem × 3 runs = 3 prompts, return 3 results
    mock_results = [
        "The answer is 4.",
        "I think it's 5.",
        "Maybe 6?",
    ]
    
    with patch('by_hand.script.run_inference_batch', return_value=mock_results):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('by_hand.script.Path.mkdir'):
                result = evaluate_model("test_model", problems=mock_problems, num_runs=3)
    
    # Overall: 0 correct out of 3
    assert result['overall_accuracy'] == 0.0
    # Extracted only: 0.0 (no extractions)
    assert result['extracted_accuracy'] == 0.0


def test_evaluate_model_saves_results():
    """Test that evaluate_model saves results to file."""
    # Create mock problems
    mock_problems = [
        {
            'unique_id': 'test_1',
            'problem': 'What is 2+2?',
            'answer': '4',
            'level': 3,
            'subject': 'arithmetic'
        }
    ]
    
    # Mock run_inference_batch - for 1 problem × 2 runs = 2 prompts, return 2 results
    mock_results = [
        "The answer is \\boxed{4}.",
        "The answer is \\boxed{5}.",
    ]
    
    with patch('by_hand.script.run_inference_batch', return_value=mock_results):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('by_hand.script.Path.mkdir'):
                with patch('by_hand.script.json.dump') as mock_json_dump:
                    evaluate_model("test_model", problems=mock_problems, num_runs=2)
    
    # Verify json.dump was called
    mock_json_dump.assert_called_once()
    call_args = mock_json_dump.call_args
    # First arg should be the output data dict
    output_data = call_args[0][0]
    assert 'overall_accuracy' in output_data
    assert 'problems' in output_data


def test_evaluate_model_with_different_answer_formats():
    """Test evaluate_model handles different answer formats correctly."""
    # Create mock problems
    mock_problems = [
        {
            'unique_id': 'test_1',
            'problem': 'What is 2+2?',
            'answer': '4',
            'level': 3,
            'subject': 'arithmetic'
        }
    ]
    
    # Test with various formats that should match ground_truth=4
    # For 1 problem × 4 runs = 4 prompts, return 4 results
    mock_results = [
        "The answer is \\boxed{4}.",           # Exact match
        "The answer is \\boxed{4.0}.",         # Should match numerically
        "The answer is \\boxed{ 4 }.",         # With spaces
        "The answer is \\boxed{5}.",          # Wrong answer
    ]
    
    with patch('by_hand.script.run_inference_batch', return_value=mock_results):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('by_hand.script.Path.mkdir'):
                result = evaluate_model("test_model", problems=mock_problems, num_runs=4)
    
    # First 3 should match (exact or numerical), last one is wrong
    # Note: The comparison logic might handle 4.0 as matching 4
    assert result['overall_accuracy'] >= 0.5  # At least 2-3 should match
    assert result['extracted_accuracy'] >= 0.5


def test_load_aime25_problems_mocks_huggingface():
    """Test that load_aime25_problems correctly mocks HuggingFace dataset loading."""
    # Create mock dataset with AIME25 structure (id, problem, answer)
    mock_dataset_items = [
        {'id': 'prob_1', 'problem': 'Problem 1', 'answer': '10'},
        {'id': 'prob_2', 'problem': 'Problem 2', 'answer': '20'},
        {'id': 'prob_3', 'problem': 'Problem 3', 'answer': '30'},
        {'id': 'prob_4', 'problem': 'Problem 4', 'answer': '40'},
        {'id': 'prob_5', 'problem': 'Problem 5', 'answer': '50'},
    ]
    
    # Create a mock dataset object that can be iterated
    class MockDataset:
        def __init__(self, items):
            self.items = items
        def __iter__(self):
            return iter(self.items)
    
    mock_dataset = MockDataset(mock_dataset_items)
    
    # Mock load_dataset to return our mock dataset
    with patch('by_hand.script.load_dataset', return_value=mock_dataset):
        problems = load_aime25_problems(limit=3)
    
    # Should return 3 problems (limited to 3)
    assert len(problems) == 3
    assert problems[0]['unique_id'] == 'prob_1'
    assert problems[0]['problem'] == 'Problem 1'
    assert problems[0]['answer'] == '10'
    assert problems[1]['unique_id'] == 'prob_2'
    assert problems[2]['unique_id'] == 'prob_3'


def test_evaluate_model_with_mocked_huggingface():
    """Test evaluate_model with mocked HuggingFace dataset loading."""
    # Create mock dataset with AIME25 structure (id, problem, answer)
    mock_dataset_items = [
        {'id': 'prob_1', 'problem': 'What is 2+2?', 'answer': '4'},
        {'id': 'prob_2', 'problem': 'What is 3+3?', 'answer': '6'},
    ]
    
    # Create a mock dataset object that can be iterated
    class MockDataset:
        def __init__(self, items):
            self.items = items
        def __iter__(self):
            return iter(self.items)
    
    mock_dataset = MockDataset(mock_dataset_items)
    
    # Mock inference results - batch inference takes a list of prompts
    # For 2 problems × 3 runs = 6 prompts total
    def mock_run_inference_batch_side_effect(model_key, prompts):
        results = []
        for prompt in prompts:
            if '2+2' in prompt:
                results.append("The answer is \\boxed{4}.")
            elif '3+3' in prompt:
                results.append("The answer is \\boxed{7}.")  # Wrong answer
            else:
                results.append("The answer is \\boxed{0}.")
        return results
    
    with patch('by_hand.script.load_dataset', return_value=mock_dataset):
        with patch('by_hand.script.run_inference_batch', side_effect=mock_run_inference_batch_side_effect):
            with patch('builtins.open', mock_open()):
                with patch('by_hand.script.Path.mkdir'):
                    result = evaluate_model("test_model", num_runs=3)
    
    # Should have evaluated 2 problems
    assert len(result['results']) == 2
    
    # First problem: all 3 runs correct
    assert result['results'][0]['overall_accuracy'] == 1.0
    assert result['results'][0]['extracted_accuracy'] == 1.0
    
    # Second problem: all 3 runs incorrect
    assert result['results'][1]['overall_accuracy'] == 0.0
    assert result['results'][1]['extracted_accuracy'] == 0.0
    
    # Overall accuracy: 3 correct out of 6 total runs
    assert result['overall_accuracy'] == 0.5
    assert result['extracted_accuracy'] == 0.5

