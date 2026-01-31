"""Texas Hold'em River - Single Street Endgame Solver.

This implements the River subgame of Texas Hold'em with full 52-card logic.
This is the first step in the "Endgame Solving" strategy before expanding to
the full game.

Game Setup:
- 52-card deck (standard poker deck)
- 2 players
- 5 community cards dealt face-up (the "River")
- 2 private cards per player
- Pot starts at 10.0 (simulating pre-river action)
- Stacks: 200.0 each

Actions:
- 0: Fold
- 1: Check/Call
- 2: Bet Pot (bet equal to current pot size)
- 3: All-In (bet remaining stack)

Rules:
- Single betting round only (River street)
- If both players check or one calls, hands are evaluated
- Best 5-card hand wins (using 2 hole cards + 5 board cards)
- Hand evaluation using treys library
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from treys import Card, Evaluator, Deck

# Action constants
FOLD = 0
CHECK_CALL = 1
BET_POT = 2
ALL_IN = 3
ACTION_NAMES = ["Fold", "Check/Call", "Bet Pot", "All-In"]


@dataclass(frozen=True)
class TexasHoldemRiver:
    """Texas Hold'em River game state (immutable).

    Attributes:
        board: 5 community cards (River already dealt)
        hands: Tuple of (player_0_hand, player_1_hand) - 2 cards each
        pot: Current pot size
        stacks: Tuple of (player_0_stack, player_1_stack)
        current_bet: Current bet to call
        player_0_invested: Amount player 0 has invested this street
        player_1_invested: Amount player 1 has invested this street
        history: Action history as string
        is_dealt: Whether cards have been dealt (for chance node)
    """

    board: Optional[list[int]] = None
    hands: Optional[tuple[list[int], list[int]]] = None
    pot: float = 10.0  # Simulating pre-river action
    stacks: tuple[float, float] = (200.0, 200.0)
    current_bet: float = 0.0
    player_0_invested: float = 0.0
    player_1_invested: float = 0.0
    history: str = ""
    is_dealt: bool = False

    def __post_init__(self):
        """Validate state after initialization."""
        if self.is_dealt:
            assert self.board is not None and len(self.board) == 5, "Board must have 5 cards"
            assert self.hands is not None, "Hands must be dealt"
            assert len(self.hands[0]) == 2 and len(self.hands[1]) == 2, (
                "Each player must have 2 cards"
            )

    def is_chance_node(self) -> bool:
        """Check if this is a chance node (cards need to be dealt)."""
        return not self.is_dealt

    def apply_action(self, action: int) -> "TexasHoldemRiver":
        """Apply an action and return new state."""
        if self.is_terminal():
            raise ValueError("Cannot apply action to terminal state")

        if self.is_chance_node():
            # Chance action: deal all cards
            # Action 0 means deal cards randomly
            deck = Deck()
            board = deck.draw(5)
            hand_0 = deck.draw(2)
            hand_1 = deck.draw(2)

            return TexasHoldemRiver(
                board=board,
                hands=(hand_0, hand_1),
                pot=self.pot,
                stacks=self.stacks,
                current_bet=self.current_bet,
                player_0_invested=self.player_0_invested,
                player_1_invested=self.player_1_invested,
                history=self.history,
                is_dealt=True,
            )

        # Player action
        if action not in [FOLD, CHECK_CALL, BET_POT, ALL_IN]:
            raise ValueError(f"Invalid action: {action}")

        current = self.current_player()
        new_history = self.history + ["f", "c", "p", "a"][action]

        new_pot = self.pot
        new_stacks = list(self.stacks)
        new_p0_inv = self.player_0_invested
        new_p1_inv = self.player_1_invested
        new_bet = self.current_bet

        if action == FOLD:
            # Fold - no money changes
            pass

        elif action == CHECK_CALL:
            # Check if no bet, otherwise call
            if self.current_bet > 0:
                # Call the bet
                call_amount = self.current_bet - (new_p0_inv if current == 0 else new_p1_inv)
                call_amount = min(call_amount, new_stacks[current])

                new_stacks[current] -= call_amount
                new_pot += call_amount

                if current == 0:
                    new_p0_inv += call_amount
                else:
                    new_p1_inv += call_amount
            # else: check (no money changes)

        elif action == BET_POT:
            # Bet amount equal to pot
            bet_amount = self.pot
            bet_amount = min(bet_amount, new_stacks[current])

            new_stacks[current] -= bet_amount
            new_pot += bet_amount
            new_bet = bet_amount

            if current == 0:
                new_p0_inv += bet_amount
            else:
                new_p1_inv += bet_amount

        elif action == ALL_IN:
            # Bet entire remaining stack
            all_in_amount = new_stacks[current]

            new_stacks[current] = 0.0
            new_pot += all_in_amount
            new_bet = max(new_bet, all_in_amount)

            if current == 0:
                new_p0_inv += all_in_amount
            else:
                new_p1_inv += all_in_amount

        return TexasHoldemRiver(
            board=self.board,
            hands=self.hands,
            pot=new_pot,
            stacks=tuple(new_stacks),
            current_bet=new_bet,
            player_0_invested=new_p0_inv,
            player_1_invested=new_p1_inv,
            history=new_history,
            is_dealt=self.is_dealt,
        )

    def legal_actions(self) -> list[int]:
        """Get legal actions for current player."""
        if self.is_terminal():
            return []

        if self.is_chance_node():
            return [0]  # Only one chance outcome (random deal)

        actions = []
        current = self.current_player()
        current_stack = self.stacks[current]
        current_invested = self.player_0_invested if current == 0 else self.player_1_invested

        # Can always fold (except if we can check for free)
        if self.current_bet > current_invested:
            actions.append(FOLD)

        # Can always check/call
        actions.append(CHECK_CALL)

        # Can bet pot if we have chips and haven't bet yet this round
        if current_stack > 0 and self.current_bet == 0:
            actions.append(BET_POT)
            actions.append(ALL_IN)

        # Can raise if facing a bet and have chips
        if current_stack > 0 and self.current_bet > 0:
            # In this simple version, we allow all-in as a raise option
            actions.append(ALL_IN)

        return actions

    def is_terminal(self) -> bool:
        """Check if game has ended."""
        if not self.is_dealt:
            return False

        h = self.history

        # Someone folded
        if "f" in h:
            return True

        # Both checked (cc)
        if h == "cc":
            return True

        # Someone bet and other called
        # Patterns: pc (bet pot, call), ac (all-in, call), ppc (bet pot, raise pot, call), etc.
        # Simple check: if last action is call and there was a bet before
        if h.endswith("c") and len(h) >= 2 and any(x in h[:-1] for x in ["p", "a"]):
            # Check if bets are matched
            if self.player_0_invested == self.player_1_invested:
                return True

        # Both all-in
        if self.stacks[0] == 0 and self.stacks[1] == 0:
            return True

        return False

    def returns(self) -> tuple[float, float]:
        """Get terminal payoffs."""
        if not self.is_terminal():
            raise ValueError("Cannot get returns for non-terminal state")

        h = self.history

        # Someone folded
        if "f" in h:
            # Last player to act folded
            folder = self.current_player()
            winner = 1 - folder

            # Winner gets the pot
            if winner == 0:
                return (self.pot, -self.pot)
            else:
                return (-self.pot, self.pot)

        # Showdown - evaluate hands
        evaluator = Evaluator()

        # Get hand strengths (lower is better in treys)
        rank_0 = evaluator.evaluate(self.board, self.hands[0])
        rank_1 = evaluator.evaluate(self.board, self.hands[1])

        if rank_0 < rank_1:
            # Player 0 wins
            winnings = self.pot / 2
            return (winnings, -winnings)
        elif rank_1 < rank_0:
            # Player 1 wins
            winnings = self.pot / 2
            return (-winnings, winnings)
        else:
            # Tie (split pot)
            return (0.0, 0.0)

    def current_player(self) -> int:
        """Get current player to act."""
        if not self.is_dealt:
            return -1  # Chance node

        if self.is_terminal():
            return -1

        # Simple betting round: players alternate
        # Player 0 acts first, then player 1, then player 0 again if needed
        num_actions = len(self.history)

        # If bets are not matched, it's the player who needs to respond
        if self.current_bet > 0:
            if self.player_0_invested < self.current_bet:
                return 0
            elif self.player_1_invested < self.current_bet:
                return 1

        # Otherwise, alternate based on action count
        return num_actions % 2

    def information_state_string(self) -> str:
        """Get information state string for current player."""
        if not self.is_dealt or self.current_player() == -1:
            return "chance_or_terminal"

        current = self.current_player()
        hand = self.hands[current]

        # Format: "hand|board|history|pot|stacks|bet"
        # Convert cards to compact string representation
        hand_str = "".join([Card.int_to_str(c) for c in hand])
        board_str = "".join([Card.int_to_str(c) for c in self.board])

        return f"{hand_str}|{board_str}|{self.history}|{self.pot:.0f}|{self.stacks[0]:.0f},{self.stacks[1]:.0f}|{self.current_bet:.0f}"

    def information_state_tensor(self) -> np.ndarray:
        """Get information state as tensor for neural network input."""
        if not self.is_dealt or self.current_player() == -1:
            # Return zero vector for chance/terminal nodes
            return np.zeros(52 + 52 + 10 + 1)  # 2 cards + 5 cards encoding + context

        current = self.current_player()
        hand = self.hands[current]

        # One-hot encoding for 2 hole cards (52 * 2 = 104 bits total, but we'll use 52 bits per card)
        # Actually, let's use a simpler encoding: rank + suit features
        features = []

        # Hole cards (2 cards * 2 features = 4 features)
        for card in hand:
            rank = Card.get_rank_int(card)  # 0-12
            suit = Card.get_suit_int(card)  # 0-3
            features.append(rank / 12.0)  # Normalize rank
            features.append(suit / 3.0)  # Normalize suit

        # Board cards (5 cards * 2 features = 10 features)
        for card in self.board:
            rank = Card.get_rank_int(card)
            suit = Card.get_suit_int(card)
            features.append(rank / 12.0)
            features.append(suit / 3.0)

        # Context features
        features.append(self.pot / 500.0)  # Normalize pot
        features.append(self.stacks[0] / 200.0)  # Normalize stack
        features.append(self.stacks[1] / 200.0)
        features.append(self.current_bet / 200.0)  # Normalize bet
        features.append(self.player_0_invested / 200.0)
        features.append(self.player_1_invested / 200.0)
        features.append(len(self.history) / 10.0)  # Normalize history length

        return np.array(features, dtype=np.float32)

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Get possible chance outcomes with probabilities."""
        if not self.is_chance_node():
            return []

        # Only one outcome: deal random cards
        return [(0, 1.0)]

    def __str__(self) -> str:
        """String representation for debugging."""
        if not self.is_dealt:
            return "TexasHoldemRiver(cards not dealt)"

        board_str = Card.print_pretty_cards(self.board)
        hand_0_str = Card.print_pretty_cards(self.hands[0])
        hand_1_str = Card.print_pretty_cards(self.hands[1])

        return (
            f"TexasHoldemRiver(\n"
            f"  Board: {board_str}\n"
            f"  P0 Hand: {hand_0_str}\n"
            f"  P1 Hand: {hand_1_str}\n"
            f"  Pot: {self.pot}\n"
            f"  Stacks: P0={self.stacks[0]}, P1={self.stacks[1]}\n"
            f"  Current Bet: {self.current_bet}\n"
            f"  History: {self.history}\n"
            f")"
        )


def new_river_holdem_game() -> TexasHoldemRiver:
    """Create a new River Hold'em game in initial state."""
    return TexasHoldemRiver()


def new_river_holdem_with_cards(
    board: list[int],
    hand_0: list[int],
    hand_1: list[int],
    pot: float = 10.0,
    stacks: tuple[float, float] = (200.0, 200.0),
) -> TexasHoldemRiver:
    """Create a River Hold'em game with specific cards (for testing)."""
    return TexasHoldemRiver(
        board=board,
        hands=(hand_0, hand_1),
        pot=pot,
        stacks=stacks,
        current_bet=0.0,
        player_0_invested=0.0,
        player_1_invested=0.0,
        history="",
        is_dealt=True,
    )
