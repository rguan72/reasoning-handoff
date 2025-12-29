"""Thought sampling consistency analysis utility."""

__version__ = "0.1.0"

from .__main__ import (
    analyze_thought_sampling,
    split_sentences,
    run_samples,
    build_histogram,
    extract_answer,
    construct_prompt_with_cot_prefix,
)

# Import model configs from centralized configs module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model_configs import MODEL_CONFIGS, VALID_MODEL_NAMES
