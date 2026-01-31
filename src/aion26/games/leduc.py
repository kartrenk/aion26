"""Leduc Poker implementation.

Leduc Poker is a simplified poker game with:
- 6 cards: Jack, Queen, King (2 suits each)
- 2 players
- 2 betting rounds (preflop and flop)
- Antes: 1 chip each (pot starts at 2)
- Bet sizes: 2 chips in round 1, 4 chips in round 2
- Public card revealed after round 1

Hand rankings:
1. Pair (same rank in hand + board)
2. High card

Reference:
- Southey et al. (2005): "Bayes' Bluff: Opponent Modelling in Poker"
"""

from dataclasses import dataclass
from typing import Optional

# Card constants (rank, suit)
# Ranks: 0=Jack, 1=Queen, 2=King
# Suits: 0=Spades, 1=Hearts
JACK = 0
QUEEN = 1
KING = 2

SPADES = 0
HEARTS = 1


@dataclass(frozen=True)
class Card:
    """A card with rank and suit."""

    rank: int  # 0=J, 1=Q, 2=K
    suit: int  # 0=♠, 1=♥

    def __str__(self):
        rank_str = ["J", "Q", "K"][self.rank]
        suit_str = ["♠", "♥"][self.suit]
        return f"{rank_str}{suit_str}"

    def __repr__(self):
        return str(self)


# Standard 6-card deck for Leduc
LEDUC_DECK = [
    Card(JACK, SPADES),
    Card(JACK, HEARTS),
    Card(QUEEN, SPADES),
    Card(QUEEN, HEARTS),
    Card(KING, SPADES),
    Card(KING, HEARTS),
]


class LeducPoker:
    """Leduc Poker game state.

    Game flow:
    1. Chance: Deal 2 private cards (one to each player)
    2. Round 1: Betting (bet size = 2)
    3. Chance: Deal 1 public card
    4. Round 2: Betting (bet size = 4)
    5. Showdown: Best hand wins

    Attributes:
        cards: Tuple of (player_0_card, player_1_card, public_card)
               public_card is None until after round 1
        history: String of actions taken ("" initially)
        pot: Current pot size (starts at 2 from antes)
        player_bets: Tuple of (player_0_bet, player_1_bet) for current round
        round: Current betting round (1 or 2)
    """

    def __init__(
        self,
        cards: Optional[tuple[Optional[Card], Optional[Card], Optional[Card]]] = None,
        history: str = "",
        pot: int = 2,
        player_bets: tuple[int, int] = (1, 1),  # Antes
        round: int = 1,
    ):
        """Initialize Leduc Poker state.

        Args:
            cards: (p0_card, p1_card, public_card). None values indicate chance nodes.
            history: String of actions (c=check, b=bet, f=fold)
            pot: Current pot size
            player_bets: Current round bets for each player
            round: Current betting round (1 or 2)
        """
        self.cards = cards if cards is not None else (None, None, None)
        self.history = history
        self.pot = pot
        self.player_bets = player_bets
        self.round = round

    def is_chance_node(self) -> bool:
        """Check if this is a chance node (cards need to be dealt)."""
        # Initial state: need to deal private cards
        if self.cards[0] is None or self.cards[1] is None:
            return True

        # After round 1, need to deal public card
        # Check if we're in round 2 and public card hasn't been dealt yet
        if self.round == 2 and self.cards[2] is None:
            # Make sure the game hasn't ended with a fold
            # A fold is indicated by "bc" pattern in history
            actions_this_round = (
                self.history.split("/")[-1] if "/" in self.history else self.history
            )
            if (
                len(actions_this_round) >= 2
                and actions_this_round[-2] == "b"
                and actions_this_round[-1] == "c"
            ):
                return False  # Game ended with fold
            return True

        return False

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Get possible chance outcomes with probabilities.

        Returns:
            List of (action, probability) tuples
            Action is the card index in LEDUC_DECK
        """
        if self.cards[0] is None:
            # Deal first private card (all 6 cards equally likely)
            return [(i, 1.0 / 6.0) for i in range(6)]

        if self.cards[1] is None:
            # Deal second private card (5 remaining cards)
            remaining_cards = [i for i in range(6) if LEDUC_DECK[i] != self.cards[0]]
            prob = 1.0 / len(remaining_cards)
            return [(i, prob) for i in remaining_cards]

        if self.round == 2 and self.cards[2] is None:
            # Deal public card (4 remaining cards)
            remaining_cards = [
                i for i in range(6) if LEDUC_DECK[i] not in [self.cards[0], self.cards[1]]
            ]
            prob = 1.0 / len(remaining_cards)
            return [(i, prob) for i in remaining_cards]

        return []

    def apply_action(self, action: int) -> "LeducPoker":
        """Apply an action and return the new state.

        Args:
            action: If chance node, card index in LEDUC_DECK.
                   If player node, 0=check/fold, 1=bet/call

        Returns:
            New game state
        """
        # Chance node: deal card
        if self.is_chance_node():
            card = LEDUC_DECK[action]

            if self.cards[0] is None:
                # Deal to player 0
                return LeducPoker(
                    cards=(card, None, None),
                    history=self.history,
                    pot=self.pot,
                    player_bets=self.player_bets,
                    round=self.round,
                )
            elif self.cards[1] is None:
                # Deal to player 1
                return LeducPoker(
                    cards=(self.cards[0], card, None),
                    history=self.history,
                    pot=self.pot,
                    player_bets=self.player_bets,
                    round=self.round,
                )
            else:
                # Deal public card
                return LeducPoker(
                    cards=(self.cards[0], self.cards[1], card),
                    history=self.history,
                    pot=self.pot,
                    player_bets=self.player_bets,
                    round=self.round,
                )

        # Player action
        action_char = "c" if action == 0 else "b"
        new_history = self.history + action_char

        # Get current player
        player = self.current_player()

        # Calculate new bets and pot
        bet_size = 2 if self.round == 1 else 4
        new_bets = list(self.player_bets)
        new_pot = self.pot

        if action == 0:  # check/fold
            # If facing a bet, this is a fold
            if self.player_bets[1 - player] > self.player_bets[player]:
                # Fold: terminal state
                return LeducPoker(
                    cards=self.cards,
                    history=new_history,
                    pot=self.pot,  # Pot doesn't change (folder loses their bet)
                    player_bets=tuple(new_bets),
                    round=self.round,
                )
            # Otherwise, it's a check
        else:  # bet/call
            # If facing a bet, this is a call
            if self.player_bets[1 - player] > self.player_bets[player]:
                # Call
                chips_to_call = self.player_bets[1 - player] - self.player_bets[player]
                new_bets[player] += chips_to_call
                new_pot += chips_to_call
            else:
                # Bet/raise
                new_bets[player] += bet_size
                new_pot += bet_size

        # Check if round ends
        # Round ends when both players have acted and bets are equal
        actions_this_round = new_history.split("/")[-1] if "/" in new_history else new_history

        # Count actions in current round
        round_actions = len(actions_this_round)

        # Both players acted and bets equal -> round over
        if round_actions >= 2 and new_bets[0] == new_bets[1]:
            if self.round == 1:
                # Move to round 2
                return LeducPoker(
                    cards=self.cards,
                    history=new_history + "/",  # Separator between rounds
                    pot=new_pot,
                    player_bets=(0, 0),  # Reset bets for new round
                    round=2,
                )
            else:
                # Round 2 complete -> showdown
                return LeducPoker(
                    cards=self.cards,
                    history=new_history,
                    pot=new_pot,
                    player_bets=tuple(new_bets),
                    round=2,
                )

        return LeducPoker(
            cards=self.cards,
            history=new_history,
            pot=new_pot,
            player_bets=tuple(new_bets),
            round=self.round,
        )

    def current_player(self) -> int:
        """Get the current player to act.

        Returns:
            0 or 1 for player, -1 for terminal/chance
        """
        if self.is_terminal() or self.is_chance_node():
            return -1

        # Get current round actions
        actions_this_round = self.history.split("/")[-1] if "/" in self.history else self.history

        # Alternate between players
        return len(actions_this_round) % 2

    def is_terminal(self) -> bool:
        """Check if game is over."""
        if self.is_chance_node():
            return False

        # Check for fold
        if "c" in self.history:
            # Check if last action was a fold (check after bet)
            actions_this_round = (
                self.history.split("/")[-1] if "/" in self.history else self.history
            )

            # Pattern "bc" means bet then fold
            if len(actions_this_round) >= 2:
                if actions_this_round[-2] == "b" and actions_this_round[-1] == "c":
                    # Player faced a bet and folded
                    return True

        # Check for showdown (round 2 complete)
        if self.round == 2 and self.cards[2] is not None:
            actions_this_round = (
                self.history.split("/")[-1] if "/" in self.history else self.history
            )

            # Both players acted and bets are equal
            if len(actions_this_round) >= 2 and self.player_bets[0] == self.player_bets[1]:
                return True

        return False

    def returns(self) -> tuple[float, float]:
        """Get payoffs for terminal states.

        Returns:
            (player_0_payoff, player_1_payoff)
        """
        if not self.is_terminal():
            raise ValueError("Cannot get returns for non-terminal state")

        # Check for fold
        actions_this_round = self.history.split("/")[-1] if "/" in self.history else self.history

        if (
            len(actions_this_round) >= 2
            and actions_this_round[-2] == "b"
            and actions_this_round[-1] == "c"
        ):
            # Last player folded
            folder = (len(actions_this_round) - 1) % 2
            winner = 1 - folder

            if winner == 0:
                return (self.pot / 2.0, -self.pot / 2.0)
            else:
                return (-self.pot / 2.0, self.pot / 2.0)

        # Showdown: compare hands
        hand_value_0 = self._hand_value(0)
        hand_value_1 = self._hand_value(1)

        if hand_value_0 > hand_value_1:
            return (self.pot / 2.0, -self.pot / 2.0)
        elif hand_value_1 > hand_value_0:
            return (-self.pot / 2.0, self.pot / 2.0)
        else:
            return (0.0, 0.0)  # Tie

    def _hand_value(self, player: int) -> int:
        """Calculate hand value for a player.

        Returns:
            Hand value: pair=100+rank, high_card=rank
        """
        private_card = self.cards[player]
        public_card = self.cards[2]

        if private_card is None or public_card is None:
            raise ValueError("Cannot evaluate hand without all cards")

        # Check for pair
        if private_card.rank == public_card.rank:
            return 100 + private_card.rank  # Pair

        # High card
        return private_card.rank

    def legal_actions(self) -> list[int]:
        """Get legal actions for current player.

        Returns:
            [0, 1] for check/fold and bet/call
        """
        if self.is_terminal() or self.is_chance_node():
            return []
        return [0, 1]

    def information_state_string(self) -> str:
        """Get information state string for current player.

        Returns:
            String representation of observable information
        """
        player = self.current_player()
        if player == -1:
            return "terminal_or_chance"

        # Private card
        private_card = self.cards[player]
        card_str = str(private_card)

        # Public card (if dealt)
        if self.cards[2] is not None:
            public_str = str(self.cards[2])
        else:
            public_str = ""

        # Betting history
        history_str = self.history

        return f"{card_str}|{public_str}|{history_str}"

    def __str__(self):
        return (
            f"Leduc(cards={self.cards}, history={self.history}, pot={self.pot}, round={self.round})"
        )

    def __repr__(self):
        return str(self)
