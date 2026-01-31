"""Python wrapper for Rust River Hold'em implementation.

Provides a GameState-compatible interface for the blazing-fast Rust backend.
"""

from typing import Optional

try:
    import aion26_rust

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class RustRiverWrapper:
    """Wrapper for Rust RiverHoldem implementation.

    This class provides a Python-friendly interface that matches the GameState
    protocol used by DeepCFRTrainer, while delegating all heavy computation to Rust.

    Performance: ~1000x faster than pure Python implementation.
    """

    def __init__(self, rust_game=None):
        """Initialize wrapper.

        Args:
            rust_game: Optional aion26_rust.RustRiverHoldem instance.
                      If None, creates a new game.
        """
        if not RUST_AVAILABLE:
            raise ImportError(
                "aion26_rust module not found. Build with: uv pip install --editable src/aion26_rust"
            )

        if rust_game is None:
            self._game = aion26_rust.RustRiverHoldem()
        else:
            self._game = rust_game

    def legal_actions(self) -> list[int]:
        """Get legal actions for current player.

        Returns:
            List of legal action indices
        """
        return self._game.legal_actions()

    def apply_action(self, action: int) -> "RustRiverWrapper":
        """Apply an action and return new game state.

        Args:
            action: Action index to apply

        Returns:
            New RustRiverWrapper with updated state
        """
        new_rust_game = self._game.apply_action(action)
        return RustRiverWrapper(new_rust_game)

    def is_terminal(self) -> bool:
        """Check if game is in a terminal state.

        Returns:
            True if terminal, False otherwise
        """
        return self._game.is_terminal()

    def current_player(self) -> int:
        """Get current player.

        Returns:
            Player index (0 or 1), or -1 for chance/terminal nodes
        """
        return self._game.current_player()

    def is_chance_node(self) -> bool:
        """Check if this is a chance node (pre-deal).

        Returns:
            True if chance node, False otherwise
        """
        return self._game.is_chance_node()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Get chance outcomes (for chance nodes).

        Returns:
            List of (action, probability) tuples
            For River Hold'em, only one outcome: deal cards with probability 1.0
        """
        if self.is_chance_node():
            return [(0, 1.0)]  # Single deal action
        return []

    def returns(self) -> tuple[float, float]:
        """Get returns for terminal state.

        Returns:
            Tuple of (player_0_return, player_1_return) in chips
        """
        rust_returns = self._game.returns()
        return (rust_returns[0], rust_returns[1])

    def information_state_string(self, player: Optional[int] = None) -> str:
        """Get information state string for player.

        Args:
            player: Player index (0 or 1). If None, uses current player.

        Returns:
            String encoding the player's information state
        """
        if player is None:
            player = self.current_player()
            if player == -1:
                return "chance" if self.is_chance_node() else "terminal"

        return self._game.information_state_string(player)

    # Properties for encoder access
    @property
    def pot(self) -> float:
        """Current pot size."""
        return self._game.pot

    @property
    def stacks(self) -> list[float]:
        """Player stacks."""
        return self._game.stacks

    @property
    def current_bet(self) -> float:
        """Current bet to call."""
        return self._game.current_bet

    @property
    def player_0_invested(self) -> float:
        """Amount player 0 has invested."""
        return self._game.player_0_invested

    @property
    def player_1_invested(self) -> float:
        """Amount player 1 has invested."""
        return self._game.player_1_invested

    @property
    def board(self) -> list[int]:
        """Board cards (5 cards, 0-51 encoding)."""
        return self._game.board

    @property
    def hands(self) -> list[list[int]]:
        """Player hole cards (2 cards each, 0-51 encoding)."""
        return self._game.hands

    @property
    def is_dealt(self) -> bool:
        """Whether cards have been dealt."""
        return self._game.is_dealt

    @property
    def history(self) -> str:
        """Betting history string."""
        return self._game.history

    def __repr__(self) -> str:
        """String representation."""
        return f"RustRiverWrapper(pot={self.pot}, is_dealt={self.is_dealt}, terminal={self.is_terminal()})"


def new_rust_river_game(fixed_board: Optional[list[int]] = None) -> RustRiverWrapper:
    """Create a new Rust-backed River Hold'em game.

    Args:
        fixed_board: Optional list of 5 card indices (0-51) for fixed board mode.
                    If None, uses random boards (full game).
                    Example: [12, 11, 10, 9, 13] = [As, Ks, Qs, Js, 2h]

    Returns:
        RustRiverWrapper instance
    """
    if fixed_board is not None:
        # Fixed board mode
        import aion26_rust

        rust_game = aion26_rust.RustRiverHoldem(
            stacks=[100.0, 100.0],
            pot=2.0,
            current_bet=0.0,
            player_0_invested=1.0,
            player_1_invested=1.0,
            fixed_board=fixed_board,
        )
        return RustRiverWrapper(rust_game)
    else:
        # Random board mode (backward compatible)
        return RustRiverWrapper()
