"""Regret matching utilities for CFR."""

import numpy as np
import numpy.typing as npt


def regret_matching(regrets: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert regrets to a strategy using Regret Matching.

    Regret Matching is the core strategy update rule in CFR:
    - Strategy for action a is proportional to max(0, regret_a)
    - If all regrets are non-positive, use uniform random strategy

    Args:
        regrets: Array of cumulative regrets for each action

    Returns:
        Probability distribution over actions (sums to 1.0)
    """
    # Only use positive regrets
    positive_regrets = np.maximum(regrets, 0.0)

    # Sum of positive regrets
    regret_sum = positive_regrets.sum()

    # If no positive regrets, use uniform strategy
    if regret_sum <= 0.0:
        num_actions = len(regrets)
        return np.ones(num_actions, dtype=np.float64) / num_actions

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
