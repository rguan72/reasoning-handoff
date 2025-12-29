"""
Model configurations for Qwen3-14B and NVIDIA-Nemotron-Nano-12B-v2.

This module provides unified model configurations used across the codebase.
"""
# Model configurations
    # Note: Model paths may need to be adjusted based on actual HuggingFace model names
    # Common alternatives:
    # - Qwen3-14B: "Qwen/Qwen3-14B-Base" or "Qwen/Qwen3-14B"
    # - Nemotron: "nvidia/NVIDIA-Nemotron-Nano-12B-v2" or "nvidia/Nemotron-Nano-12B-v2"
    # Quantization options:
    # - dtype: "float16" (half precision, ~2x memory reduction) or "bfloat16"
    # - quantization: "awq" or "gptq" (if quantized models available, ~4x memory reduction)
    # - gpu_memory_utilization: 0.0-1.0 (fraction of GPU memory to use)
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen3-14B",
        "model_path": "Qwen/Qwen3-14B",
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.95,
        "dtype": "float16",
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 8192,
        "max_model_len": 8192,
        "use_chat_template": False,
        "system_prompt": None,
        "quantization": None,
    },
    "qwen-small": {
        "name": "Qwen3-0.6B",
        "model_path": "Qwen/Qwen3-0.6B",
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.95,
        "dtype": "float16",
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 8192,
        "max_model_len": 8192,
        "use_chat_template": False,
        "system_prompt": None,
        "quantization": None,
    },
    "nvidia": {
        "name": "NVIDIA-Nemotron-Nano-12B-v2",
        "model_path": "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
        "max_tokens": 8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "dtype": "float16",
        "gpu_memory_utilization": 0.85,
        "max_num_batched_tokens": 16384,
        "max_model_len": 16384,
        "use_chat_template": True,
        "system_prompt": "/think",
        "quantization": None,
    },
}

VALID_MODEL_NAMES = list(MODEL_CONFIGS.keys())

# Mapping from full model names to short keys (for backward compatibility)
MODEL_NAME_TO_KEY = {
    "Qwen3-14B": "qwen",
    "NVIDIA-Nemotron-Nano-12B-v2": "nvidia",
}


def get_model_config_by_name(model_name: str) -> dict:
    """
    Get model configuration by full model name.
    
    Args:
        model_name: Full model name (e.g., "Qwen3-14B")
    
    Returns:
        Model configuration dictionary
    """
    key = MODEL_NAME_TO_KEY.get(model_name)
    if key is None:
        raise ValueError(f"Unknown model name: {model_name}. Valid names: {list(MODEL_NAME_TO_KEY.keys())}")
    return MODEL_CONFIGS[key]


def get_all_models_config() -> dict:
    """
    Get all model configurations keyed by full model name.
    This is for backward compatibility with evaluate_math500.py.
    
    Returns:
        Dictionary mapping full model names to configurations
    """
    return {
        config["name"]: {
            "model_path": config["model_path"],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"],
            "top_p": config.get("top_p", 0.95),
            "dtype": config["dtype"],
            "quantization": config.get("quantization"),
            "gpu_memory_utilization": config["gpu_memory_utilization"],
            "max_model_len": config["max_model_len"],
            "use_chat_template": config.get("use_chat_template", False),
            "system_prompt": config.get("system_prompt"),
        }
        for config in MODEL_CONFIGS.values()
    }

