"""Vanilla CFR (Counterfactual Regret Minimization) implementation.

This is a tabular implementation of the vanilla CFR algorithm for small games
like Kuhn Poker. It maintains dictionaries of regrets and strategies keyed by
information state strings.

Reference:
- Zinkevich et al. (2007): "Regret Minimization in Games with Incomplete Information"
"""

from typing import Protocol
import numpy as np
import numpy.typing as npt

from aion26.cfr.regret_matching import regret_matching, sample_action


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


class VanillaCFR:
    """Vanilla CFR solver using tabular regret storage.

    Attributes:
        regret_sum: Dictionary mapping info_state -> cumulative regrets per action
        strategy_sum: Dictionary mapping info_state -> cumulative strategy per action
        num_actions: Number of actions for each information state
        iteration: Current iteration count
        rng: Random number generator for sampling
    """

    def __init__(self, initial_state: GameState, seed: int = 42):
        """Initialize the CFR solver.

        Args:
            initial_state: Initial game state (used to infer action space size)
            seed: Random seed for reproducibility
        """
        self.initial_state = initial_state
        self.rng = np.random.default_rng(seed)
        self.iteration = 0

        # Regret tables: info_state -> np.array of regrets per action
        # Using defaultdict so we automatically initialize to zeros
        self.regret_sum: dict[str, npt.NDArray[np.float64]] = {}
        # Strategy sum for average strategy computation
        self.strategy_sum: dict[str, npt.NDArray[np.float64]] = {}

        # Infer number of actions from initial state
        # (Kuhn Poker has 2 actions everywhere except chance nodes)
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

    def traverse(
        self,
        state: GameState,
        update_player: int,
        reach_prob_0: float,
        reach_prob_1: float,
    ) -> float:
        """Recursively traverse the game tree and update regrets.

        This is the core CFR algorithm. It performs a depth-first traversal,
        computing counterfactual values and updating regrets.

        Args:
            state: Current game state
            update_player: The player whose regrets we're updating (0 or 1)
            reach_prob_0: Probability that player 0 reaches this state
            reach_prob_1: Probability that player 1 reaches this state

        Returns:
            Expected value for the update_player at this state
        """
        # Terminal node: return payoff
        if state.is_terminal():
            returns = state.returns()
            return returns[update_player]

        # Chance node: weighted average over outcomes
        if state.is_chance_node():
            expected_value = 0.0
            for action, probability in state.chance_outcomes():
                next_state = state.apply_action(action)
                value = self.traverse(next_state, update_player, reach_prob_0, reach_prob_1)
                expected_value += probability * value
            return expected_value

        # Player node
        current_player = state.current_player()
        info_state = state.information_state_string()
        legal_actions = state.legal_actions()
        num_legal = len(legal_actions)

        # Get current strategy
        strategy = self.get_strategy(info_state)

        # If this is the player we're updating, compute counterfactual values
        if current_player == update_player:
            # Compute value for each action
            action_values = np.zeros(num_legal, dtype=np.float64)
            for i, action in enumerate(legal_actions):
                next_state = state.apply_action(action)

                # In external sampling, we need to update reach probabilities when
                # traversing our own actions so that downstream strategy_sum updates
                # are weighted correctly by the probability of reaching those nodes
                if current_player == 0:
                    action_values[i] = self.traverse(
                        next_state,
                        update_player,
                        reach_prob_0 * strategy[i],  # ✅ FIXED: Weight by action probability
                        reach_prob_1,
                    )
                else:
                    action_values[i] = self.traverse(
                        next_state,
                        update_player,
                        reach_prob_0,
                        reach_prob_1 * strategy[i],  # ✅ FIXED: Weight by action probability
                    )

            # Expected value of this information state
            node_value = np.dot(strategy, action_values)

            # Compute counterfactual regrets
            # Regret = (value if we always played action a) - (value of current strategy)
            counterfactual_regrets = action_values - node_value

            # Weight regrets by opponent reach probability (CFR weighting)
            opponent_reach = reach_prob_1 if current_player == 0 else reach_prob_0

            # Update regret sums
            if info_state not in self.regret_sum:
                self.regret_sum[info_state] = np.zeros(self.num_actions, dtype=np.float64)

            self.regret_sum[info_state] += opponent_reach * counterfactual_regrets

            # Update strategy sum (weighted by own reach probability)
            own_reach = reach_prob_0 if current_player == 0 else reach_prob_1

            if info_state not in self.strategy_sum:
                self.strategy_sum[info_state] = np.zeros(self.num_actions, dtype=np.float64)

            self.strategy_sum[info_state] += own_reach * strategy

            return node_value

        else:
            # Opponent's node: sample according to their strategy
            # In external sampling MCCFR, we sample the opponent's action
            # and do NOT multiply reach probability (sampling already accounts for it)
            action_idx = sample_action(strategy[:num_legal], self.rng)
            action = legal_actions[action_idx]
            next_state = state.apply_action(action)

            # In external sampling: don't multiply reach prob when sampling
            # The reach probability represents the probability of reaching this node
            # in the SAMPLED trajectory, which is always 1.0 for sampled actions
            return self.traverse(
                next_state,
                update_player,
                reach_prob_0,
                reach_prob_1,
            )

    def run_iteration(self) -> None:
        """Run one iteration of CFR (update both players)."""
        self.iteration += 1

        # Update player 0
        self.traverse(self.initial_state, update_player=0, reach_prob_0=1.0, reach_prob_1=1.0)

        # Update player 1
        self.traverse(self.initial_state, update_player=1, reach_prob_0=1.0, reach_prob_1=1.0)

    def get_all_average_strategies(self) -> dict[str, npt.NDArray[np.float64]]:
        """Get average strategies for all information states.

        Returns:
            Dictionary mapping info_state -> average strategy
        """
        return {
            info_state: self.get_average_strategy(info_state)
            for info_state in self.strategy_sum.keys()
        }
