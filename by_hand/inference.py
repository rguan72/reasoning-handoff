"""
Basic inference function for running vLLM inference with models from model_configs.
"""
from typing import List, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from model_configs import MODEL_CONFIGS, VALID_MODEL_NAMES


def run_inference(
    model_key: str,
    prompt: str,
    num_runs: int = 1,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Run inference with vLLM using a model from the model config.
    
    Args:
        model_key: Model key from MODEL_CONFIGS (e.g., 'qwen', 'nvidia', 'qwen-small')
        prompt: The prompt to run inference on
        num_runs: Number of times to run inference on the prompt (default: 1)
        temperature: Override temperature from config (optional)
        max_tokens: Override max_tokens from config (optional)
        top_p: Override top_p from config (optional)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        List of generated completion texts (length = num_runs)
    
    Example:
        >>> outputs = run_inference('qwen', 'What is 2+2?', num_runs=3)
        >>> print(outputs)
        ['The answer is 4.', '2 + 2 equals 4.', 'Four.']
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key: {model_key}. Valid keys: {VALID_MODEL_NAMES}"
        )
    
    config = MODEL_CONFIGS[model_key]
    
    # Build LLM initialization arguments
    llm_kwargs = {
        "model": config["model_path"],
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "max_model_len": config["max_model_len"],
        "max_num_batched_tokens": config["max_num_batched_tokens"],
        "dtype": config["dtype"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
    }
    
    if seed is not None:
        llm_kwargs["seed"] = seed
    
    # Add quantization if specified
    if config.get("quantization"):
        llm_kwargs["quantization"] = config["quantization"]
    
    # Initialize LLM
    llm = LLM(**llm_kwargs)
    
    # Load tokenizer if chat template is needed
    tokenizer = None
    use_chat_template = config.get("use_chat_template", False)
    if use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
    
    # Prepare prompt (apply chat template if needed)
    if use_chat_template and tokenizer is not None:
        messages = []
        if config.get("system_prompt"):
            messages.append({"role": "system", "content": config["system_prompt"]})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature if temperature is not None else config["temperature"],
        max_tokens=max_tokens if max_tokens is not None else config["max_tokens"],
        top_p=top_p if top_p is not None else config["top_p"],
    )
    
    # Create list of prompts (same prompt repeated num_runs times)
    prompts = [formatted_prompt] * num_runs
    
    # Run inference
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated texts
    completions = [output.outputs[0].text for output in outputs]
    
    return completions

