import pytest
import sys
from pathlib import Path

# Add parent directory to path to import by_hand module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from by_hand.answer_extraction import extract_answer


def test_extract_answer_found():
    """Test extracting answer when \\boxed{} is present."""
    # Simple case
    text1 = "The answer is \\boxed{42}."
    assert extract_answer(text1) == "42"
    
    # Multiple boxed occurrences - should get the last one
    text2 = "First \\boxed{10}, then \\boxed{20}, finally \\boxed{30}."
    assert extract_answer(text2) == "30"
    
    # Boxed with nested braces
    text3 = "The solution is \\boxed{\\frac{1}{2}}."
    assert extract_answer(text3) == "\\frac{1}{2}"
    
    # Boxed with spaces
    text4 = "Answer: \\boxed { 100 }"
    assert extract_answer(text4) == " 100 "
    
    # Complex nested expression
    text5 = "Result: \\boxed{\\left(\\frac{a+b}{c}\\right)}"
    assert extract_answer(text5) == "\\left(\\frac{a+b}{c}\\right)"
    
    # Boxed at the end
    text6 = "After solving, we get \\boxed{x = 5}"
    assert extract_answer(text6) == "x = 5"


def test_extract_answer_not_found():
    """Test extracting answer when \\boxed{} is not present."""
    # No boxed at all
    text1 = "The answer is 42."
    assert extract_answer(text1) is None
    
    # Empty string
    text2 = ""
    assert extract_answer(text2) is None
    
    # Text with boxed-like but not actual boxed
    text3 = "This has boxed but not \\boxed."
    assert extract_answer(text3) is None
    
    # Incomplete boxed (no closing brace)
    text4 = "The answer is \\boxed{42"
    assert extract_answer(text4) is None
    
    # Only opening brace, no content
    text5 = "\\boxed{"
    assert extract_answer(text5) is None

