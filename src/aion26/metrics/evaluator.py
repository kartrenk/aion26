"""Head-to-head evaluation for learned strategies.

When NashConv is computationally infeasible (e.g., Texas Hold'em with 52 cards),
we evaluate learned strategies by playing head-to-head matches against baseline bots.

Metrics:
- Win rate in milli-big-blinds per hand (mbb/h)
- Standard error of the mean
- Win rate confidence intervals
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from aion26.baselines import BaselineBot
from aion26.cfr.regret_matching import regret_matching


@dataclass
class HeadToHeadResult:
    """Results from head-to-head evaluation.

    Attributes:
        num_hands: Number of hands played
        agent_winnings: Total winnings for the agent (big blinds)
        bot_winnings: Total winnings for the bot (big blinds)
        avg_mbb_per_hand: Average winnings in milli-big-blinds per hand
        std_error: Standard error of the mean
        confidence_95: 95% confidence interval (±mbb)
    """

    num_hands: int
    agent_winnings: float
    bot_winnings: float
    avg_mbb_per_hand: float
    std_error: float
    confidence_95: float

    def __str__(self) -> str:
        """Pretty print results."""
        return (
            f"HeadToHeadResult(\n"
            f"  Hands: {self.num_hands}\n"
            f"  Agent: {self.agent_winnings:+.1f} BB\n"
            f"  Bot: {self.bot_winnings:+.1f} BB\n"
            f"  Average: {self.avg_mbb_per_hand:+.0f} mbb/h ± {self.confidence_95:.0f}\n"
            f")"
        )


class HeadToHeadEvaluator:
    """Evaluator for playing head-to-head matches.

    This class plays matches between a learned strategy (from Deep CFR) and
    baseline bots to evaluate performance.

    Example:
        evaluator = HeadToHeadEvaluator()

        # Get strategy from trainer
        strategy = trainer.get_all_average_strategies()

        # Play 1000 hands vs RandomBot
        result = evaluator.evaluate(
            initial_state=new_river_holdem_game(),
            strategy=strategy,
            opponent=RandomBot(),
            num_hands=1000
        )

        print(f"Win rate: {result.avg_mbb_per_hand:+.0f} mbb/h")
    """

    def __init__(self, big_blind: float = 2.0):
        """Initialize evaluator.

        Args:
            big_blind: Size of big blind for mbb calculations (default: 2.0)
        """
        self.big_blind = big_blind

    def _get_action_from_strategy(
        self,
        state,
        strategy: dict[str, np.ndarray],
        player: int
    ) -> int:
        """Get action from learned strategy.

        Args:
            state: Current game state
            strategy: Strategy dictionary (info_state -> action probabilities)
            player: Player index

        Returns:
            Action index (greedy - argmax of strategy)
        """
        # Get information state string
        info_state = state.information_state_string()

        # Get strategy for this info state
        if info_state not in strategy:
            # Unseen state - use uniform random
            legal = state.legal_actions()
            return np.random.choice(legal)

        action_probs = strategy[info_state]

        # Greedy action (argmax)
        return int(np.argmax(action_probs))

    def _play_single_hand(
        self,
        initial_state,
        strategy: dict[str, np.ndarray],
        opponent: BaselineBot,
        agent_is_p0: bool
    ) -> tuple[float, float]:
        """Play a single hand.

        Args:
            initial_state: Initial game state (before dealing)
            strategy: Learned strategy dictionary
            opponent: Baseline bot opponent
            agent_is_p0: True if agent plays as player 0, False if player 1

        Returns:
            Tuple of (agent_return, bot_return) in big blinds
        """
        # Deal cards (apply chance action)
        state = initial_state

        # Deal if needed
        if state.is_chance_node():
            chance_outcomes = state.chance_outcomes()
            if chance_outcomes:
                action, _ = chance_outcomes[0]
                state = state.apply_action(action)

        # Play until terminal
        while not state.is_terminal():
            current = state.current_player()

            if current == -1:
                # Shouldn't happen after initial deal
                break

            # Determine who acts
            if (agent_is_p0 and current == 0) or (not agent_is_p0 and current == 1):
                # Agent acts
                action = self._get_action_from_strategy(state, strategy, current)
            else:
                # Bot acts
                action = opponent.get_action(state)

            state = state.apply_action(action)

        # Get returns
        returns = state.returns()

        # Convert to big blinds
        agent_return = returns[0 if agent_is_p0 else 1] / self.big_blind
        bot_return = returns[1 if agent_is_p0 else 0] / self.big_blind

        return agent_return, bot_return

    def evaluate(
        self,
        initial_state,
        strategy: dict[str, np.ndarray],
        opponent: BaselineBot,
        num_hands: int = 1000,
        alternate_positions: bool = True
    ) -> HeadToHeadResult:
        """Evaluate learned strategy against baseline bot.

        Args:
            initial_state: Initial game state (will be reset for each hand)
            strategy: Learned strategy dictionary
            opponent: Baseline bot to play against
            num_hands: Number of hands to play (default: 1000)
            alternate_positions: Alternate who plays P0/P1 (default: True)

        Returns:
            HeadToHeadResult with statistics
        """
        agent_winnings_list = []

        for hand_num in range(num_hands):
            # Alternate positions if enabled
            agent_is_p0 = (hand_num % 2 == 0) if alternate_positions else True

            # Play hand
            agent_return, bot_return = self._play_single_hand(
                initial_state,
                strategy,
                opponent,
                agent_is_p0
            )

            agent_winnings_list.append(agent_return)

        # Calculate statistics
        agent_winnings_array = np.array(agent_winnings_list)
        total_agent_winnings = agent_winnings_array.sum()
        total_bot_winnings = -total_agent_winnings  # Zero-sum

        # Average winnings per hand (in big blinds)
        avg_bb_per_hand = total_agent_winnings / num_hands

        # Convert to milli-big-blinds (mbb)
        avg_mbb_per_hand = avg_bb_per_hand * 1000

        # Calculate standard error
        std_dev = agent_winnings_array.std()
        std_error = std_dev / np.sqrt(num_hands)

        # 95% confidence interval (±1.96 * SE)
        confidence_95_bb = 1.96 * std_error
        confidence_95_mbb = confidence_95_bb * 1000

        return HeadToHeadResult(
            num_hands=num_hands,
            agent_winnings=total_agent_winnings,
            bot_winnings=total_bot_winnings,
            avg_mbb_per_hand=avg_mbb_per_hand,
            std_error=std_error * 1000,  # Convert to mbb
            confidence_95=confidence_95_mbb
        )

    def evaluate_against_multiple(
        self,
        initial_state,
        strategy: dict[str, np.ndarray],
        opponents: dict[str, BaselineBot],
        num_hands: int = 1000
    ) -> dict[str, HeadToHeadResult]:
        """Evaluate against multiple baseline bots.

        Args:
            initial_state: Initial game state
            strategy: Learned strategy
            opponents: Dictionary of {name: bot}
            num_hands: Number of hands per opponent

        Returns:
            Dictionary of {name: HeadToHeadResult}
        """
        results = {}

        for name, bot in opponents.items():
            results[name] = self.evaluate(
                initial_state,
                strategy,
                bot,
                num_hands
            )

        return results
