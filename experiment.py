# %%
from thought_sampling_handoff.__main__ import analyze_thought_sampling
# %%
if __name__ == '__main__':
    analyze_thought_sampling(
        prompt="Evaluate $(1+2i)(6-3i)$.",                    # Original problem prompt
        chain_of_thought_text="""Okay, let's evaluate the expression (1 + 2i) multiplied by 6 and then subtract 3i. Hmm, first step is to distribute the 6 across the terms inside the parentheses. So, 6 times 1 is 6, and 6 times 2i is 12i. That gives me 6 + 12i. Then, I need to subtract 3i from that result.

Subtracting 3i from 12i would be 12i - 3i, which is 9i. So, combining the real part and the imaginary part, the result should be 6 + 9i. Let me double-check that.

Wait, is there any chance I mixed up the signs? The original expression is (1 + 2i) * 6 - 3i. Yes, so after multiplying, it's 6 + 12i, then subtracting 3i. So 12i - 3i is indeed 9i. Therefore, the final answer is 6 + 9i.

I think that's right. Let me write it out step by step again. Multiply 1 by 6 gives 6, multiply 2i by 6 gives 12i. Then, subtract 3i: 12i - 3i = 9i. So, 6 + 9i. Yep, that seems correct. No mistakes there.""",     # CoT text to sample from
        model_name_a="qwen-small",
        model_name_b="qwen-small",
        num_samples=5,
        fractions=[0.25, 0.5, 0.75],
        out_dir="./thought_sampling_out",
        seed=None,
    )
# %%
