math_prompt = "Each page number of a 488-page book is printed one time in the book. The first page is page 1 and the last page is page 488. When printing all of the page numbers, how many more 4's are printed than 8's?"
ground_truth = 90

def construct_prompt(problem: str) -> str:
    return f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem} Solution: \n<think>\n"
