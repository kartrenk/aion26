"""Kuhn Poker implementation.

Kuhn Poker is the simplest non-trivial poker game, with 3 cards (J, Q, K),
2 players, and a simple betting structure. It has 12 information sets and
a known analytical Nash equilibrium solution.

Rules:
- Each player antes 1 chip
- Each player is dealt one card from {J, Q, K}
- Player 0 acts first, then Player 1, then possibly Player 0 again
- Actions: Check (0) or Bet/Call (1), or Fold (implied by game state)
- Bet size is fixed at 1 chip
- Showdown: Higher card wins

Information Sets (12 total):
- Player 0: J/Q/K at first action (3 infosets)
- Player 0: J/Q/K after checking, facing a bet (3 infosets)
- Player 1: J/Q/K after opponent checked (3 infosets)
- Player 1: J/Q/K after opponent bet (3 infosets)
"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt


# Card constants
JACK = 0
QUEEN = 1
KING = 2
CARD_NAMES = ["J", "Q", "K"]

# Action constants
CHECK = 0
BET = 1
ACTION_NAMES = ["Check", "Bet"]


@dataclass(frozen=True)
class KuhnPoker:
    """Kuhn Poker game state (immutable).

    Attributes:
        cards: Tuple of (player_0_card, player_1_card) or (None, None) before deal
        history: Sequence of actions as a string (e.g., "cb" = check then bet)
        pot: Total chips in the pot
        player_0_invested: Chips player 0 has put in (beyond ante)
        player_1_invested: Chips player 1 has put in (beyond ante)
    """

    cards: tuple[int | None, int | None] = (None, None)
    history: str = ""
    pot: int = 2  # Both players ante 1
    player_0_invested: int = 1  # Ante
    player_1_invested: int = 1  # Ante

    def apply_action(self, action: int) -> "KuhnPoker":
        """Apply an action and return new state."""
        if self.is_terminal():
            raise ValueError("Cannot apply action to terminal state")

        if self.is_chance_node():
            # Chance action is the deal index (0-5 for the 6 possible deals)
            deals = [
                (JACK, QUEEN),
                (JACK, KING),
                (QUEEN, JACK),
                (QUEEN, KING),
                (KING, JACK),
                (KING, QUEEN),
            ]
            if action < 0 or action >= len(deals):
                raise ValueError(f"Invalid deal action: {action}")
            return KuhnPoker(
                cards=deals[action],
                history=self.history,
                pot=self.pot,
                player_0_invested=self.player_0_invested,
                player_1_invested=self.player_1_invested,
            )

        # Player action
        if action not in [CHECK, BET]:
            raise ValueError(f"Invalid action: {action}")

        action_char = "c" if action == CHECK else "b"
        new_history = self.history + action_char

        # Calculate new pot and investments
        new_p0_inv = self.player_0_invested
        new_p1_inv = self.player_1_invested
        new_pot = self.pot

        current = self.current_player()
        if action == BET:
            # Player bets 1 chip
            if current == 0:
                new_p0_inv += 1
            else:
                new_p1_inv += 1
            new_pot += 1

        return KuhnPoker(
            cards=self.cards,
            history=new_history,
            pot=new_pot,
            player_0_invested=new_p0_inv,
            player_1_invested=new_p1_inv,
        )

    def legal_actions(self) -> list[int]:
        """Get legal actions for current player."""
        if self.is_terminal():
            return []

        if self.is_chance_node():
            # All 6 possible deals are legal
            return list(range(6))

        # Players can always check or bet
        # (In Kuhn, check = fold when facing a bet, bet = call when facing a bet)
        return [CHECK, BET]

    def is_terminal(self) -> bool:
        """Check if game has ended."""
        h = self.history

        # Need cards to be dealt
        if self.cards == (None, None):
            return False

        # Game ends after:
        # - "cc": Both check (showdown)
        # - "bc": Bet then fold
        # - "bb": Bet then call (showdown)
        # - "cbc": Check, bet, fold
        # - "cbb": Check, bet, call (showdown)

        if h in ["cc", "bb", "cbb"]:  # Showdown
            return True
        if h in ["bc", "cbc"]:  # Fold
            return True

        return False

    def returns(self) -> tuple[float, float]:
        """Get terminal payoffs."""
        if not self.is_terminal():
            raise ValueError("Cannot get returns for non-terminal state")

        h = self.history
        p0_card, p1_card = self.cards
        assert p0_card is not None and p1_card is not None

        # Fold cases (player who didn't bet wins the pot)
        if h == "bc":  # P0 bet, P1 folded
            # P0 wins the pot minus their investment, P1 loses their investment
            return (float(self.pot - self.player_0_invested), float(-self.player_1_invested))
        if h == "cbc":  # P0 checked, P1 bet, P0 folded
            return (float(-self.player_0_invested), float(self.pot - self.player_1_invested))

        # Showdown cases (higher card wins)
        if p0_card > p1_card:
            # P0 wins the pot
            return (float(self.pot - self.player_0_invested), float(-self.player_1_invested))
        else:
            # P1 wins the pot
            return (float(-self.player_0_invested), float(self.pot - self.player_1_invested))

    def current_player(self) -> int:
        """Get current player to act."""
        if self.is_terminal():
            return -1

        if self.cards == (None, None):
            return -1  # Chance node

        h = self.history
        # P0 acts first, then P1, then possibly P0 again
        # - "": P0's first action
        # - "c": P1 acts after P0 checked
        # - "b": P1 acts after P0 bet
        # - "cb": P0 acts after checking then facing a bet

        if h == "":
            return 0
        elif h == "c" or h == "b":
            return 1
        elif h == "cb":
            return 0
        else:
            return -1  # Terminal

    def information_state_tensor(self) -> npt.NDArray[np.float32]:
        """Get information state as numpy array.

        Encoding:
        - One-hot card (3 dims): [is_jack, is_queen, is_king]
        - Betting history (max 3 actions, 2 dims each = 6 dims): [is_check, is_bet] repeated
        Total: 9 dimensions
        """
        current = self.current_player()
        if current == -1:
            # Terminal or chance node - return zeros
            return np.zeros(9, dtype=np.float32)

        # One-hot encode current player's card
        card = self.cards[current]
        assert card is not None
        card_encoding = np.zeros(3, dtype=np.float32)
        card_encoding[card] = 1.0

        # Encode betting history (up to 3 actions in Kuhn)
        history_encoding = np.zeros(6, dtype=np.float32)
        for i, action_char in enumerate(self.history):
            if i >= 3:  # Safety check
                break
            if action_char == "c":
                history_encoding[i * 2] = 1.0  # Check
            else:  # "b"
                history_encoding[i * 2 + 1] = 1.0  # Bet

        return np.concatenate([card_encoding, history_encoding])

    def information_state_string(self) -> str:
        """Get information state as string (for tabular CFR).

        Format: "<card><history>" e.g., "J", "Qc", "Kcb"
        """
        current = self.current_player()
        if current == -1:
            return "terminal"

        card = self.cards[current]
        assert card is not None
        return CARD_NAMES[card] + self.history

    def is_chance_node(self) -> bool:
        """Check if this is a chance node (card deal)."""
        return self.cards == (None, None) and not self.is_terminal()

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Get chance outcomes (all 6 deals are equally likely)."""
        if not self.is_chance_node():
            return []
        # 6 possible deals, each with probability 1/6
        return [(i, 1.0 / 6.0) for i in range(6)]

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.cards == (None, None):
            return "Kuhn(undealt)"
        p0_card = CARD_NAMES[self.cards[0]] if self.cards[0] is not None else "?"
        p1_card = CARD_NAMES[self.cards[1]] if self.cards[1] is not None else "?"
        history = self.history if self.history else "start"
        return f"Kuhn(P0:{p0_card}, P1:{p1_card}, history:{history})"


def new_kuhn_game() -> KuhnPoker:
    """Create a new Kuhn Poker game at the initial state."""
    return KuhnPoker()
