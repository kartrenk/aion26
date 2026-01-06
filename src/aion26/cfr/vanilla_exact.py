"""Exact Vanilla CFR - Full Tree Traversal (No Sampling).

This is a reference implementation of textbook vanilla CFR that performs
full tree traversal at every iteration. Use this as ground truth to debug
the external sampling MCCFR implementation.

Reference: Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information"
"""

from collections import defaultdict
from typing import Protocol
import numpy as np
import numpy.typing as npt

from aion26.cfr.regret_matching import regret_matching


class GameState(Protocol):
    """Protocol for game states (duck typing)."""

    def apply_action(self, action: int) -> "GameState": ...
    def legal_actions(self) -> list[int]: ...
    def is_terminal(self) -> bool: ...
    def returns(self) -> tuple[float, float]: ...
    def current_player(self) -> int: ...
    def information_state_string(self) -> str: ...
    def is_chance_node(self) -> bool: ...
    def chance_outcomes(self) -> list[tuple[int, float]]: ...


class VanillaCFR_Exact:
    """Exact Vanilla CFR with full tree traversal (no sampling).

    This is a textbook implementation that:
    - Evaluates ALL actions at ALL nodes
    - Uses exact expected values (no sampling)
    - Updates regrets using standard CFR formula
    - Serves as ground truth for debugging MCCFR variants

    Attributes:
        regret_sum: Dictionary mapping info_state -> cumulative regrets per action
        strategy_sum: Dictionary mapping info_state -> cumulative strategy per action
        num_actions: Number of actions for each information state
        iteration: Current iteration count
    """

    def __init__(self, initial_state: GameState):
        """Initialize the exact CFR solver.

        Args:
            initial_state: Initial game state (used to infer action space size)
        """
        self.initial_state = initial_state
        self.iteration = 0

        # Regret tables: info_state -> np.array of regrets per action
        self.regret_sum: dict[str, npt.NDArray[np.float64]] = {}
        # Strategy sum for average strategy computation
        self.strategy_sum: dict[str, npt.NDArray[np.float64]] = {}

        # Infer number of actions from initial state
        self.num_actions = 2  # CHECK/BET for Kuhn Poker

    def get_strategy(self, info_state: str) -> npt.NDArray[np.float64]:
        """Get current strategy for an information state using regret matching.

        Args:
            info_state: Information state string

        Returns:
            Current strategy (probability distribution over actions)
        """
        # Get cumulative regrets (or initialize to zeros)
        if info_state not in self.regret_sum:
            self.regret_sum[info_state] = np.zeros(self.num_actions, dtype=np.float64)

        regrets = self.regret_sum[info_state]
        return regret_matching(regrets)

    def get_average_strategy(self, info_state: str) -> npt.NDArray[np.float64]:
        """Get average strategy for an information state.

        The average strategy is the blueprint strategy that converges to Nash equilibrium.

        Args:
            info_state: Information state string

        Returns:
            Average strategy (probability distribution over actions)
        """
        if info_state not in self.strategy_sum:
            # If never visited, return uniform
            return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        strat_sum = self.strategy_sum[info_state]
        total = strat_sum.sum()

        if total <= 0.0:
            # If sum is zero, return uniform
            return np.ones(self.num_actions, dtype=np.float64) / self.num_actions

        return strat_sum / total

    def cfr(
        self,
        state: GameState,
        reach_prob_0: float,
        reach_prob_1: float,
        update_player: int,
    ) -> float:
        """Vanilla CFR traversal - computes expected value for update_player.

        This is the textbook CFR algorithm from Zinkevich et al. (2007).
        Unlike MCCFR, this evaluates the full game tree at every iteration.

        Args:
            state: Current game state
            reach_prob_0: Probability that player 0 reaches this state
            reach_prob_1: Probability that player 1 reaches this state
            update_player: Player whose expected value we're computing (0 or 1)

        Returns:
            Expected value for update_player at this state
        """
        # Terminal node: return payoff for update_player
        if state.is_terminal():
            returns = state.returns()
            return returns[update_player]

        # Chance node: weighted average over outcomes
        if state.is_chance_node():
            value = 0.0
            for action, probability in state.chance_outcomes():
                next_state = state.apply_action(action)
                value += probability * self.cfr(next_state, reach_prob_0, reach_prob_1, update_player)
            return value

        # Player node
        current_player = state.current_player()
        info_state = state.information_state_string()
        legal_actions = state.legal_actions()
        num_legal = len(legal_actions)

        # Get current strategy
        strategy = self.get_strategy(info_state)

        # Compute value for each action (counterfactual values)
        action_values = np.zeros(num_legal, dtype=np.float64)

        for i, action in enumerate(legal_actions):
            next_state = state.apply_action(action)

            # Recursive call with updated reach probabilities
            if current_player == 0:
                # Player 0's node: multiply player 0's reach by strategy[i]
                action_values[i] = self.cfr(
                    next_state,
                    reach_prob_0 * strategy[i],
                    reach_prob_1,
                    update_player,
                )
            else:
                # Player 1's node: multiply player 1's reach by strategy[i]
                action_values[i] = self.cfr(
                    next_state,
                    reach_prob_0,
                    reach_prob_1 * strategy[i],
                    update_player,
                )

        # Expected value of this node (playing according to current strategy)
        node_value = np.dot(strategy[:num_legal], action_values)

        # Only update regrets and strategy for the player being updated
        if current_player == update_player:
            # Compute counterfactual regrets
            # Regret = (value if we always played action a) - (value of current strategy)
            counterfactual_regrets = action_values - node_value

            # Update regret sum (weighted by opponent reach probability)
            opponent_reach = reach_prob_1 if current_player == 0 else reach_prob_0

            if info_state not in self.regret_sum:
                self.regret_sum[info_state] = np.zeros(self.num_actions, dtype=np.float64)

            # Update regrets for legal actions
            for i in range(num_legal):
                self.regret_sum[info_state][i] += opponent_reach * counterfactual_regrets[i]

            # Update strategy sum (weighted by own reach probability)
            own_reach = reach_prob_0 if current_player == 0 else reach_prob_1

            if info_state not in self.strategy_sum:
                self.strategy_sum[info_state] = np.zeros(self.num_actions, dtype=np.float64)

            # Accumulate strategy
            for i in range(num_legal):
                self.strategy_sum[info_state][i] += own_reach * strategy[i]

        # Return expected value for update_player
        return node_value

    def run_iteration(self) -> float:
        """Run one iteration of exact vanilla CFR (updates both players).

        Returns:
            Expected game value for player 0 at the root
        """
        self.iteration += 1

        # Update player 0
        value_p0 = self.cfr(
            self.initial_state,
            reach_prob_0=1.0,
            reach_prob_1=1.0,
            update_player=0,
        )

        # Update player 1
        self.cfr(
            self.initial_state,
            reach_prob_0=1.0,
            reach_prob_1=1.0,
            update_player=1,
        )

        return value_p0

    def get_all_average_strategies(self) -> dict[str, npt.NDArray[np.float64]]:
        """Get average strategies for all information states.

        Returns:
            Dictionary mapping info_state -> average strategy
        """
        return {
            info_state: self.get_average_strategy(info_state)
            for info_state in self.strategy_sum.keys()
        }
