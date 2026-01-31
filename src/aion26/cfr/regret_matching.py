"""Regret matching utilities for CFR."""

import numpy as np
import numpy.typing as npt


def regret_matching(regrets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert regrets to a strategy using Regret Matching.

    Regret Matching is the core strategy update rule in CFR:
    - Strategy for action a is proportional to max(0, regret_a)
    - CRITICAL FIX: If all regrets are non-positive, use argmax to pick
      the "least bad" action instead of uniform random (which causes
      suicidal bluffs like All-In with trash hands)

    Args:
        regrets: Array of cumulative regrets for each action

    Returns:
        Probability distribution over actions (sums to 1.0)
    """
    # Only use positive regrets
    positive_regrets = np.maximum(regrets, 0.0)

    # Sum of positive regrets
    regret_sum = positive_regrets.sum()

    # If no positive regrets, use argmax fallback (deterministic, picks "least bad")
    if regret_sum <= 0.0:
        num_actions = len(regrets)
        result = np.zeros(num_actions, dtype=np.float64)
        best_idx = np.argmax(regrets)  # Pick action with highest (least negative) regret
        result[best_idx] = 1.0
        return result

    # Normalize positive regrets to get strategy
    return positive_regrets / regret_sum


def sample_action(strategy: npt.NDArray[np.float64], rng: np.random.Generator) -> int:
    """Sample an action according to a strategy.

    Args:
        strategy: Probability distribution over actions
        rng: NumPy random number generator

    Returns:
        Sampled action index
    """
    # Normalize strategy to ensure it sums to exactly 1.0 (fixes floating point precision issues)
    strategy_sum = strategy.sum()
    if strategy_sum > 0:
        normalized_strategy = strategy / strategy_sum
    else:
        # Fallback to uniform if all zeros
        normalized_strategy = np.ones_like(strategy) / len(strategy)

    return int(rng.choice(len(normalized_strategy), p=normalized_strategy))
