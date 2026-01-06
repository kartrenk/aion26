"""Base game interface for imperfect information games."""

from typing import Protocol
import numpy as np
import numpy.typing as npt


class GameState(Protocol):
    """Protocol defining the interface for imperfect information game states.

    This protocol must be implemented by all games (Kuhn Poker, Leduc, RPS, etc.).
    It provides a standardized interface for the CFR algorithm to interact with
    any game without knowing its specific implementation details.
    """

    def apply_action(self, action: int) -> "GameState":
        """Apply an action and return the resulting game state.

        Args:
            action: Integer index of the action to apply

        Returns:
            New GameState after applying the action (immutable pattern)
        """
        ...

    def legal_actions(self) -> list[int]:
        """Get list of legal action indices for the current player.

        Returns:
            List of valid action indices (e.g., [0, 1] for check/bet)
        """
        ...

    def is_terminal(self) -> bool:
        """Check if this state is a terminal (game over) state.

        Returns:
            True if the game has ended, False otherwise
        """
        ...

    def returns(self) -> tuple[float, float]:
        """Get the final payoffs for both players.

        Only valid to call when is_terminal() is True.

        Returns:
            Tuple of (player_0_payoff, player_1_payoff)
        """
        ...

    def current_player(self) -> int:
        """Get the index of the player to act.

        Returns:
            0 for player 0, 1 for player 1, -1 for chance/terminal nodes
        """
        ...

    def information_state_tensor(self) -> npt.NDArray[np.float32]:
        """Get the information state representation as a numpy array.

        This is the observation visible to the current player, excluding
        hidden information. Used as input to neural networks in Deep CFR.

        Returns:
            1D numpy array representing the information state
        """
        ...

    def information_state_string(self) -> str:
        """Get a unique string identifier for the information state.

        Used as dictionary key in tabular CFR. Must be the same for all
        game histories that are indistinguishable to the current player.

        Returns:
            String uniquely identifying the information set
        """
        ...

    def is_chance_node(self) -> bool:
        """Check if this is a chance node (random event, not player action).

        Returns:
            True if this is a chance node (e.g., card deal), False otherwise
        """
        ...

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Get possible chance outcomes and their probabilities.

        Only valid when is_chance_node() is True.

        Returns:
            List of (action, probability) tuples for chance events
        """
        ...
