"""Baseline bots for benchmarking learned strategies.

These simple heuristic agents provide baselines to evaluate Deep CFR performance:
- RandomBot: Uniform random policy
- CallingStation: Always calls/checks (passive)
- HonestBot: Bets based on hand strength (exploitable but reasonable)

Used for head-to-head evaluation when NashConv is computationally infeasible
(e.g., Texas Hold'em with 52 cards).
"""

import random
import numpy as np
from typing import Protocol, Optional

try:
    from treys import Evaluator
    TREYS_AVAILABLE = True
except ImportError:
    TREYS_AVAILABLE = False


class GameState(Protocol):
    """Protocol for game states that baseline bots can play."""

    def legal_actions(self) -> list[int]: ...
    def current_player(self) -> int: ...
    def is_terminal(self) -> bool: ...


class BaselineBot:
    """Base class for baseline bots."""

    def get_action(self, state: GameState) -> int:
        """Get action for current state.

        Args:
            state: Current game state

        Returns:
            Action index
        """
        raise NotImplementedError


class RandomBot(BaselineBot):
    """Bot that picks uniformly random legal action.

    This is the weakest baseline - any learned strategy should easily beat it.
    Expected performance: ~0 EV in self-play (purely random).

    Example:
        bot = RandomBot()
        action = bot.get_action(state)  # Random legal action
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize RandomBot.

        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.rng = random.Random(seed)

    def get_action(self, state: GameState) -> int:
        """Pick random legal action.

        Args:
            state: Current game state

        Returns:
            Random action from legal actions
        """
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available")
        return self.rng.choice(legal)

    def __repr__(self) -> str:
        return "RandomBot()"


class CallingStation(BaselineBot):
    """Bot that always checks/calls unless forced to fold.

    This is a passive baseline that never bluffs or value bets.
    A learned strategy should exploit this by betting strong hands.

    Action priority:
    1. Check/Call if available
    2. Fold if must (e.g., facing all-in with no chips)
    3. Random otherwise (shouldn't happen in well-designed games)

    Example:
        bot = CallingStation()
        action = bot.get_action(state)  # Always tries to call/check
    """

    def get_action(self, state: GameState) -> int:
        """Always check/call if possible.

        Args:
            state: Current game state

        Returns:
            Check/Call action (1) if available, else first legal action
        """
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available")

        # Action indices (from river_holdem.py):
        # 0: Fold
        # 1: Check/Call
        # 2: Bet Pot
        # 3: All-In

        CHECK_CALL = 1

        # Always check/call if available
        if CHECK_CALL in legal:
            return CHECK_CALL

        # Fallback to first legal action (shouldn't happen normally)
        return legal[0]

    def __repr__(self) -> str:
        return "CallingStation()"


class HonestBot(BaselineBot):
    """Bot that plays based on hand strength using treys evaluator.

    This is an exploitable but reasonable baseline that:
    - Bets strong hands (>80th percentile)
    - Calls medium hands (50-80th percentile)
    - Folds weak hands (<50th percentile)

    Hand strength is normalized to [0, 1] using treys rank ranges.

    Strategy:
    - strength > 0.8: Bet/Raise (aggressive value betting)
    - strength > 0.5: Call (protect medium strength)
    - strength ≤ 0.5: Check/Fold (give up weak hands)

    Exploitability:
    - Too honest (no bluffing)
    - Predictable (strength-based only)
    - Doesn't adjust to opponent

    But still stronger than RandomBot or CallingStation.

    Example:
        bot = HonestBot()
        action = bot.get_action(river_state)  # Strength-based decision
    """

    def __init__(self):
        """Initialize HonestBot with treys evaluator."""
        if not TREYS_AVAILABLE:
            raise ImportError("treys library required for HonestBot. Install with: pip install treys")
        self.evaluator = Evaluator()

        # Treys rank ranges (lower is better)
        # Royal Flush: 1
        # High Card: 7462 (worst)
        self.BEST_RANK = 1
        self.WORST_RANK = 7462

    def _get_hand_strength(self, state) -> float:
        """Calculate hand strength from 0 (worst) to 1 (best).

        Args:
            state: TexasHoldemRiver game state

        Returns:
            Normalized hand strength in [0, 1]
        """
        # Get current player's hand and board
        player = state.current_player()
        if player == -1:
            # Terminal/chance node
            return 0.5

        hand = state.hands[player]
        board = state.board

        # Evaluate hand (lower rank = better hand in treys)
        rank = self.evaluator.evaluate(board, hand)

        # Normalize to [0, 1] where 1 is best
        # strength = (WORST - rank) / (WORST - BEST)
        strength = (self.WORST_RANK - rank) / (self.WORST_RANK - self.BEST_RANK)

        return strength

    def get_action(self, state: GameState) -> int:
        """Choose action based on hand strength.

        Args:
            state: Current game state (must be TexasHoldemRiver)

        Returns:
            Action based on strength thresholds
        """
        legal = state.legal_actions()
        if not legal:
            raise ValueError("No legal actions available")

        # Action indices:
        FOLD = 0
        CHECK_CALL = 1
        BET_POT = 2
        ALL_IN = 3

        # Calculate hand strength
        strength = self._get_hand_strength(state)

        # Strong hand (>80th percentile): Bet/Raise
        if strength > 0.8:
            # Prefer aggressive actions
            if BET_POT in legal:
                return BET_POT
            elif ALL_IN in legal:
                return ALL_IN
            elif CHECK_CALL in legal:
                return CHECK_CALL
            else:
                return legal[0]

        # Medium hand (50-80th percentile): Call
        elif strength > 0.5:
            # Prefer passive actions
            if CHECK_CALL in legal:
                return CHECK_CALL
            elif FOLD in legal:
                return FOLD
            else:
                return legal[0]

        # Weak hand (≤50th percentile): Check/Fold
        else:
            # Prefer defensive actions
            if CHECK_CALL in legal and state.current_bet == 0:
                # Free check
                return CHECK_CALL
            elif FOLD in legal:
                # Fold to bet
                return FOLD
            elif CHECK_CALL in legal:
                # Forced to call (shouldn't happen with fold available)
                return CHECK_CALL
            else:
                return legal[0]

    def __repr__(self) -> str:
        return "HonestBot(strength_based)"


# Factory function for easy bot creation
def create_bot(name: str) -> BaselineBot:
    """Create a baseline bot by name.

    Args:
        name: Bot name ("random", "calling_station", "honest")

    Returns:
        Baseline bot instance

    Raises:
        ValueError: If bot name is unknown
    """
    name = name.lower()

    if name == "random":
        return RandomBot()
    elif name in ["calling_station", "callingstation", "calling"]:
        return CallingStation()
    elif name in ["honest", "honest_bot"]:
        return HonestBot()
    else:
        raise ValueError(
            f"Unknown bot name: {name}. "
            f"Available: random, calling_station, honest"
        )
