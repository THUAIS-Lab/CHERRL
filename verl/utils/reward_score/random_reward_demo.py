import asyncio
import json
import re
from typing import Optional
import random

from transformers import PreTrainedTokenizer


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Optional[PreTrainedTokenizer] = None,
) -> dict:
    """
    Compute reward score by returning a random value (for testing purposes).

    This function is designed to work with the Reward Loop framework for parallel
    reward computation. It returns random scores for testing the reward loop
    infrastructure without actually calling a reward model.

    Args:
        data_source: Source identifier for the data sample
        solution_str: The model-generated solution string
        ground_truth: The ground truth answer
        extra_info: Additional context information (should contain "question" key)
        reward_router_address: HTTP router endpoint address for GenRM
            (not used in test mode)
        reward_model_tokenizer: Tokenizer for the reward model
            (not used in test mode)

    Returns:
        Dictionary containing:
            - "score": float reward score between -1.0 and 1.0 (random)
            - "acc": bool indicating if the solution is correct (random)
            - "genrm_response": str indicating this is a test response

    """
    # Generate random score between -1.0 and 1.0
    score = random.uniform(-1.0, 1.0)
    
    # Randomly determine if correct (50% chance)
    is_correct = random.choice([True, False])
    
    # Optionally, you can make score match the correctness
    # Uncomment the following line to make score positive when correct, negative when incorrect
    # score = random.uniform(0.0, 1.0) if is_correct else random.uniform(-1.0, 0.0)
    
    result = {
        "score": score,
        "acc": is_correct,
        "genrm_response": f"Test mode: random score = {score:.4f}",
    }
    # Only include "pred" if it's not None to avoid issues in validation metrics calculation
    # For testing, we don't need pred, so we omit it
    # If you need pred, uncomment the following line:
    # result["pred"] = None
    
    return result