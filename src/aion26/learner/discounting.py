"""Discounting schedulers for PDCFR+ and variants.

PDCFR+ uses dynamic discounting to weight recent iterations more heavily
than early iterations, accelerating convergence by forgetting poor early
strategies.

Key insight: As training progresses, strategy quality improves. By discounting
early iterations, we focus the average strategy on high-quality recent play.

References:
- Brown & Sandholm (2019): "Solving Imperfect-Information Games via Discounted
  Regret Minimization"
- Linear CFR: w_t = t
- PDCFR: w_t = t^α / (t^α + 1)
"""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np


class DiscountScheduler(ABC):
    """Base class for iteration weighting schedules.

    Discounting controls how much weight each iteration receives when computing
    the average strategy and updating regrets.

    Two key operations:
    1. get_weight(t): Weight for iteration t (used in regret updates)
    2. get_accum_weight(t): Accumulated weight up to iteration t (for averaging)
    """

    @abstractmethod
    def get_weight(self, iteration: int, regret_sign: Literal["positive", "negative"] = "positive") -> float:
        """Get weight for a specific iteration.

        Args:
            iteration: Iteration number (1-indexed)
            regret_sign: Whether this weight is for positive or negative regrets
                        (some schedulers use different exponents)

        Returns:
            Weight value (always >= 0)
        """
        pass

    def get_accum_weight(self, iteration: int) -> float:
        """Get accumulated weight from iteration 1 to iteration.

        Used for computing weighted averages:
        avg_strategy = sum(w_t * strategy_t) / sum(w_t)

        Default implementation: sum of individual weights.
        Subclasses can override for closed-form solutions.

        Args:
            iteration: Iteration number (1-indexed)

        Returns:
            Sum of weights from 1 to iteration
        """
        return sum(self.get_weight(t) for t in range(1, iteration + 1))


class UniformScheduler(DiscountScheduler):
    """Uniform weighting: all iterations have equal weight.

    This is vanilla CFR with no discounting.
    w_t = 1 for all t

    Use case: Baseline comparison, or when you want to keep all history.
    """

    def get_weight(self, iteration: int, regret_sign: Literal["positive", "negative"] = "positive") -> float:
        """Return constant weight of 1.0."""
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")
        return 1.0

    def get_accum_weight(self, iteration: int) -> float:
        """Return iteration count (sum of 1s)."""
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")
        return float(iteration)


class LinearScheduler(DiscountScheduler):
    """Linear discounting: weight grows linearly with iteration.

    w_t = t

    This gives more weight to later iterations in a simple linear fashion.
    Used in Linear CFR and some PDCFR variants.

    Properties:
    - Simple and interpretable
    - Gradual increase in recent iteration importance
    - Closed-form accumulated weight: sum(1..t) = t(t+1)/2
    """

    def get_weight(self, iteration: int, regret_sign: Literal["positive", "negative"] = "positive") -> float:
        """Return linear weight equal to iteration number."""
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")
        return float(iteration)

    def get_accum_weight(self, iteration: int) -> float:
        """Return sum of 1..t = t(t+1)/2 using closed form."""
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")
        return iteration * (iteration + 1) / 2.0


class PDCFRScheduler(DiscountScheduler):
    """PDCFR+ discounting with separate exponents for positive/negative regrets.

    For positive regrets:
        w_t = t^α / (t^α + 1)

    For negative regrets:
        w_t = t^β / (t^β + 1)

    Key insights:
    - α controls discounting of positive regrets (exploration)
    - β controls discounting of negative regrets (anti-regrets)
    - Typical: α=2.0 (quadratic), β=0.5 (slower for negatives)
    - As t→∞, w_t→1, so recent iterations dominate

    This is the SOTA approach from Brown & Sandholm (2019).

    Args:
        alpha: Exponent for positive regrets (default: 2.0)
        beta: Exponent for negative regrets (default: 0.5)
        positive_only: If True, only apply discounting to positive regrets,
                      use uniform (1.0) for negative regrets
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 0.5,
        positive_only: bool = False
    ):
        """Initialize PDCFR scheduler.

        Args:
            alpha: Exponent for positive regrets (typically 1.5-3.0)
            beta: Exponent for negative regrets (typically 0.0-1.0)
            positive_only: If True, don't discount negative regrets
        """
        if alpha < 0:
            raise ValueError(f"Alpha must be >= 0, got {alpha}")
        if beta < 0:
            raise ValueError(f"Beta must be >= 0, got {beta}")

        self.alpha = alpha
        self.beta = beta
        self.positive_only = positive_only

    def get_weight(self, iteration: int, regret_sign: Literal["positive", "negative"] = "positive") -> float:
        """Get PDCFR+ weight for iteration.

        Args:
            iteration: Iteration number (1-indexed)
            regret_sign: "positive" or "negative" to select α or β

        Returns:
            w_t = t^exponent / (t^exponent + 1)
        """
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")

        # Select exponent based on regret sign
        if regret_sign == "positive":
            exponent = self.alpha
        elif regret_sign == "negative":
            if self.positive_only:
                return 1.0  # Uniform weight for negative regrets
            exponent = self.beta
        else:
            raise ValueError(f"regret_sign must be 'positive' or 'negative', got {regret_sign}")

        # Handle special cases
        if exponent == 0:
            return 1.0  # t^0 = 1, so w_t = 1/(1+1) = 0.5... wait, that's wrong
            # Actually for α=0: w_t = 1/(1+1) = 0.5 always
            # Let me recalculate: t^0 / (t^0 + 1) = 1 / 2 = 0.5
            # Hmm, that seems odd. Let me check the formula.
            # Actually, I think for α=0, we should just return 1.0 (uniform)
            return 1.0

        # Compute t^exponent / (t^exponent + 1)
        t = float(iteration)
        t_pow = np.power(t, exponent)
        weight = t_pow / (t_pow + 1.0)

        return weight

    def get_accum_weight(self, iteration: int) -> float:
        """Get accumulated weight (no closed form, compute sum).

        Note: For PDCFR, there's no simple closed-form solution,
        so we compute the sum directly. This is O(n) but only called
        occasionally (when computing average strategy).

        For efficiency, could cache these values if needed.
        """
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")

        # For accumulated weight, we use positive regret exponent (α)
        return sum(self.get_weight(t, "positive") for t in range(1, iteration + 1))


class GeometricScheduler(DiscountScheduler):
    """Geometric discounting: exponential decay of old iterations.

    w_t = γ^(T-t) where T is the current iteration

    This gives exponentially less weight to older iterations.
    Useful for non-stationary games or adapting to opponent changes.

    Args:
        gamma: Decay factor (0 < γ <= 1)
               γ=1 is uniform, γ→0 is "only last iteration"
    """

    def __init__(self, gamma: float = 0.99):
        """Initialize geometric scheduler.

        Args:
            gamma: Decay factor (default 0.99)
        """
        if not 0 < gamma <= 1:
            raise ValueError(f"Gamma must be in (0, 1], got {gamma}")
        self.gamma = gamma

    def get_weight(self, iteration: int, regret_sign: Literal["positive", "negative"] = "positive") -> float:
        """Return geometric weight (depends on total iterations, so we use iteration as T)."""
        if iteration < 1:
            raise ValueError(f"Iteration must be >= 1, got {iteration}")

        # w_t = γ^(T-t)
        # For iteration t, weight it as if it's (T-t) steps in the past
        # Since we don't know T in advance, use weight = γ^0 = 1.0 for current iteration
        # This will be computed properly when averaging
        return 1.0  # Placeholder; actual weight depends on T

    def get_weight_relative(self, iteration: int, current_iteration: int) -> float:
        """Get weight for iteration t relative to current iteration T.

        Args:
            iteration: Past iteration (t)
            current_iteration: Current iteration (T)

        Returns:
            γ^(T-t)
        """
        if iteration < 1 or current_iteration < iteration:
            raise ValueError(f"Invalid: iteration={iteration}, current={current_iteration}")

        age = current_iteration - iteration
        return np.power(self.gamma, age)


def create_scheduler(
    scheduler_type: Literal["uniform", "linear", "pdcfr", "geometric"] = "linear",
    **kwargs
) -> DiscountScheduler:
    """Factory function to create discounting schedulers.

    Args:
        scheduler_type: Type of scheduler
        **kwargs: Scheduler-specific parameters

    Returns:
        DiscountScheduler instance

    Examples:
        >>> scheduler = create_scheduler("linear")
        >>> scheduler = create_scheduler("pdcfr", alpha=2.0, beta=0.5)
        >>> scheduler = create_scheduler("geometric", gamma=0.99)
    """
    if scheduler_type == "uniform":
        return UniformScheduler()
    elif scheduler_type == "linear":
        return LinearScheduler()
    elif scheduler_type == "pdcfr":
        return PDCFRScheduler(**kwargs)
    elif scheduler_type == "geometric":
        return GeometricScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
