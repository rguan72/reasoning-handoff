"""
Basic inference function for running vLLM inference with models from model_configs.
"""
from typing import List, Optional, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from by_hand.model_configs import MODEL_CONFIGS, VALID_MODEL_NAMES


# Global cache for LLM instances and tokenizers
_llm_cache: Dict[str, LLM] = {}
_tokenizer_cache: Dict[str, AutoTokenizer] = {}


def get_llm(model_key: str, seed: Optional[int] = None) -> LLM:
    """
    Get or create an LLM instance for the given model key.
    Uses a global cache to avoid reinitializing models.
    
    Args:
        model_key: Model key from MODEL_CONFIGS
        seed: Random seed for reproducibility (optional)
    
    Returns:
        LLM instance
    """
    cache_key = f"{model_key}_{seed}" if seed is not None else model_key
    
    if cache_key not in _llm_cache:
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
        
        print(f"Initializing LLM for {model_key}...")
        _llm_cache[cache_key] = LLM(**llm_kwargs)
        print(f"LLM initialized for {model_key}")
    
    return _llm_cache[cache_key]


def get_tokenizer(model_key: str) -> Optional[AutoTokenizer]:
    """
    Get or create a tokenizer for the given model key.
    Uses a global cache to avoid reloading tokenizers.
    
    Args:
        model_key: Model key from MODEL_CONFIGS
    
    Returns:
        Tokenizer instance or None if chat template is not needed
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key: {model_key}. Valid keys: {VALID_MODEL_NAMES}"
        )
    
    config = MODEL_CONFIGS[model_key]
    use_chat_template = config.get("use_chat_template", False)
    
    if not use_chat_template:
        return None
    
    if model_key not in _tokenizer_cache:
        _tokenizer_cache[model_key] = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
    
    return _tokenizer_cache[model_key]


def format_prompt(model_key: str, prompt: str) -> str:
    """
    Format a prompt using the model's chat template if needed.
    
    Args:
        model_key: Model key from MODEL_CONFIGS
        prompt: The raw prompt text
    
    Returns:
        Formatted prompt string
    """
    config = MODEL_CONFIGS[model_key]
    use_chat_template = config.get("use_chat_template", False)
    
    if use_chat_template:
        tokenizer = get_tokenizer(model_key)
        if tokenizer is not None:
            messages = []
            if config.get("system_prompt"):
                messages.append({"role": "system", "content": config["system_prompt"]})
            messages.append({"role": "user", "content": prompt})
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    
    return prompt


def run_inference(
    model_key: str,
    prompt: str,
    num_runs: int = 1,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    llm: Optional[LLM] = None,
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
        llm: Optional pre-initialized LLM instance (for efficiency)
    
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
    
    # Use provided LLM or get from cache
    if llm is None:
        llm = get_llm(model_key, seed)
    
    # Format prompt
    formatted_prompt = format_prompt(model_key, prompt)
    
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


def run_inference_batch(
    model_key: str,
    prompts: List[str],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Run inference on a batch of prompts efficiently.
    This function batches all prompts together for maximum throughput.
    
    Args:
        model_key: Model key from MODEL_CONFIGS
        prompts: List of prompts to run inference on
        temperature: Override temperature from config (optional)
        max_tokens: Override max_tokens from config (optional)
        top_p: Override top_p from config (optional)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        List of generated completion texts (same order as prompts)
    
    Example:
        >>> outputs = run_inference_batch('qwen', ['What is 2+2?', 'What is 3+3?'])
        >>> print(outputs)
        ['The answer is 4.', 'The answer is 6.']
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key: {model_key}. Valid keys: {VALID_MODEL_NAMES}"
        )
    
    config = MODEL_CONFIGS[model_key]
    
    # Get LLM from cache
    llm = get_llm(model_key, seed)
    
    # Format all prompts
    formatted_prompts = [format_prompt(model_key, prompt) for prompt in prompts]
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature if temperature is not None else config["temperature"],
        max_tokens=max_tokens if max_tokens is not None else config["max_tokens"],
        top_p=top_p if top_p is not None else config["top_p"],
    )
    
    # Run inference on all prompts at once
    print(f"Running inference on batch of {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Extract generated texts
    completions = [output.outputs[0].text for output in outputs]
    
    return completions


def clear_llm_cache(model_key: Optional[str] = None, seed: Optional[int] = None):
    """
    Clear LLM instances from cache to free GPU memory.
    
    Args:
        model_key: If provided, only clear this specific model. If None, clear all models.
        seed: If provided along with model_key, clear specific seed variant.
    """
    if model_key is None:
        # Clear all models
        for cache_key in list(_llm_cache.keys()):
            llm = _llm_cache.pop(cache_key)
            del llm
        print("Cleared all LLM instances from cache")
    else:
        # Clear specific model
        cache_key = f"{model_key}_{seed}" if seed is not None else model_key
        if cache_key in _llm_cache:
            llm = _llm_cache.pop(cache_key)
            del llm
            print(f"Cleared LLM instance for {cache_key} from cache")
        else:
            print(f"No LLM instance found for {cache_key} in cache")


def clear_tokenizer_cache(model_key: Optional[str] = None):
    """
    Clear tokenizer instances from cache to free memory.
    
    Args:
        model_key: If provided, only clear this specific tokenizer. If None, clear all tokenizers.
    """
    if model_key is None:
        # Clear all tokenizers
        _tokenizer_cache.clear()
        print("Cleared all tokenizer instances from cache")
    else:
        # Clear specific tokenizer
        if model_key in _tokenizer_cache:
            del _tokenizer_cache[model_key]
            print(f"Cleared tokenizer instance for {model_key} from cache")


def cleanup_model_memory(model_key: str, seed: Optional[int] = None):
    """
    Clean up all memory associated with a model (LLM and tokenizer).
    This should be called after finishing evaluation to free GPU memory.
    
    Args:
        model_key: Model key to clean up
        seed: Optional seed variant to clean up
    """
    clear_llm_cache(model_key=model_key, seed=seed)
    clear_tokenizer_cache(model_key=model_key)
    
    # Force garbage collection to ensure memory is freed
    import gc
    gc.collect()
    
    # Try to clear CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Cleared CUDA cache for {model_key}")
    except ImportError:
        pass  # torch not available, skip CUDA cache clearing

