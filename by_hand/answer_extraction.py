import re
from sympy import sympify, simplify, N


def extract_answer(text: str) -> str | None:
    """
    Extract the answer from text by finding the last \\boxed{} occurrence.
    
    Args:
        text: The text to search for an answer
        
    Returns:
        The content inside the last \\boxed{} if found, None otherwise
    """
    # Find all occurrences of \boxed
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return None
    
    # Get the last match
    last_match = matches[-1]
    start = last_match.end() - 1  # Position of the opening brace
    
    # Find the matching closing brace, handling nested braces
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    
    # If we never found a matching closing brace, return None
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    Removes extra whitespace and normalizes LaTeX formatting.
    """
    # Convert to string if not already
    if not isinstance(answer, str):
        answer = str(answer)
    
    # Strip \boxed{} wrapper if present (as a fallback if extraction didn't remove it)
    boxed_match = re.search(r'\\boxed\s*\{', answer)
    if boxed_match:
        # Find the content inside \boxed{}
        start = boxed_match.end() - 1  # Position of the opening brace
        depth = 0
        for i in range(start, len(answer)):
            if answer[i] == '{':
                depth += 1
            elif answer[i] == '}':
                depth -= 1
                if depth == 0:
                    answer = answer[start + 1:i]
                    break

    # Normalize delimiter sizing commands: \left, \right, \big, \Big, \bigg, \Bigg, etc.
    # Handle with optional space between command and delimiter
    answer = re.sub(r'\\left\s*\(', '(', answer)
    answer = re.sub(r'\\right\s*\)', ')', answer)
    answer = re.sub(r'\\left\s*\[', '[', answer)
    answer = re.sub(r'\\right\s*\]', ']', answer)
    answer = re.sub(r'\\left\s*\\{', '{', answer)
    answer = re.sub(r'\\right\s*\\}', '}', answer)
    answer = re.sub(r'\\left\s*\|', '|', answer)
    answer = re.sub(r'\\right\s*\|', '|', answer)
    # Also handle \bigl, \bigr, \Bigl, \Bigr, etc.
    answer = re.sub(r'\\[Bb]ig[lr]?\s*\(', '(', answer)
    answer = re.sub(r'\\[Bb]ig[lr]?\s*\)', ')', answer)
    answer = re.sub(r'\\[Bb]ig[lr]?\s*\[', '[', answer)
    answer = re.sub(r'\\[Bb]ig[lr]?\s*\]', ']', answer)

    # Normalize fraction commands: \dfrac, \cfrac, \tfrac -> \frac (they're equivalent)
    # Use word boundary to ensure we match the full command name
    answer = re.sub(r'\\dfrac\b', r'\\frac', answer)
    answer = re.sub(r'\\cfrac\b', r'\\frac', answer)
    answer = re.sub(r'\\tfrac\b', r'\\frac', answer)

    # Normalize LaTeX spacing commands: \  (non-breaking space), \, \; \: \! -> regular space or nothing
    answer = re.sub(r'\\[,;:!]\s*', ' ', answer)  # thin/medium/thick space -> space
    answer = re.sub(r'\\\s+', ' ', answer)  # \ followed by whitespace -> space

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


def compare_answers(predicted: str, ground_truth: str | int | float) -> bool:
    """
    Compare predicted answer with ground truth answer.
    Handles both exact match and symbolic equality.
    """
    # Convert ground_truth to string if it's not already
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    
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
