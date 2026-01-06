"""Tests for Leduc Poker implementation.

This module tests the complete Leduc Poker game logic including:
- Card dealing and deck management
- Betting rounds and pot accounting
- Hand evaluation (pairs, high cards)
- Terminal states and payoffs
- Information state representation
"""

import pytest
from aion26.games.leduc import (
    LeducPoker, Card, LEDUC_DECK,
    JACK, QUEEN, KING, SPADES, HEARTS
)


class TestLeducBasics:
    """Test basic Leduc Poker functionality."""

    def test_initial_state(self):
        """Test that initial state is a chance node."""
        game = LeducPoker()
        assert game.is_chance_node()
        assert not game.is_terminal()
        assert game.current_player() == -1
        assert game.pot == 2  # Antes

    def test_deck_size(self):
        """Test that Leduc deck has 6 cards."""
        assert len(LEDUC_DECK) == 6

        # Check all cards are unique
        assert len(set(LEDUC_DECK)) == 6

        # Check ranks and suits
        ranks = [card.rank for card in LEDUC_DECK]
        suits = [card.suit for card in LEDUC_DECK]

        assert ranks.count(JACK) == 2
        assert ranks.count(QUEEN) == 2
        assert ranks.count(KING) == 2

        assert suits.count(SPADES) == 3
        assert suits.count(HEARTS) == 3

    def test_deal_private_cards(self):
        """Test dealing private cards to both players."""
        game = LeducPoker()

        # Deal to player 0
        game = game.apply_action(0)  # Deal J♠
        assert game.cards[0] == LEDUC_DECK[0]
        assert game.cards[1] is None
        assert game.is_chance_node()

        # Deal to player 1
        game = game.apply_action(2)  # Deal Q♠
        assert game.cards[0] == LEDUC_DECK[0]
        assert game.cards[1] == LEDUC_DECK[2]
        assert not game.is_chance_node()
        assert game.current_player() == 0  # Player 0 acts first


class TestLeducBettingRound1:
    """Test first betting round mechanics."""

    def test_both_check_round1(self):
        """Test when both players check in round 1."""
        # J♠ vs Q♠
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            pot=2,
            round=1
        )

        # P0 checks
        game = game.apply_action(0)
        assert game.history == "c"
        assert game.pot == 2

        # P1 checks
        game = game.apply_action(0)
        assert game.history == "cc/"
        assert game.round == 2
        assert game.is_chance_node()  # Need to deal public card

    def test_bet_call_round1(self):
        """Test bet and call in round 1."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            pot=2,
            round=1
        )

        # P0 bets (2 chips)
        game = game.apply_action(1)
        assert game.history == "b"
        assert game.pot == 4
        assert game.player_bets == (3, 1)  # P0 has 1 ante + 2 bet

        # P1 calls
        game = game.apply_action(1)
        assert game.history == "bb/"
        assert game.pot == 6
        assert game.player_bets == (0, 0)  # Reset for round 2
        assert game.round == 2

    def test_bet_fold_round1(self):
        """Test bet and fold in round 1."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            pot=2,
            round=1
        )

        # P0 bets
        game = game.apply_action(1)
        assert game.pot == 4

        # P1 folds
        game = game.apply_action(0)
        assert game.history == "bc"
        assert game.is_terminal()

        # P0 wins the pot
        returns = game.returns()
        assert returns == (2.0, -2.0)  # P0 wins 2 chips (pot=4, each invested 2)


class TestLeducBettingRound2:
    """Test second betting round mechanics."""

    def test_both_check_round2_showdown(self):
        """Test showdown after both check in round 2."""
        # J♠ vs Q♠, public K♠
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], LEDUC_DECK[4]),
            history="cc/",
            pot=2,
            round=2
        )

        # P0 checks
        game = game.apply_action(0)
        assert game.history == "cc/c"

        # P1 checks
        game = game.apply_action(0)
        assert game.history == "cc/cc"
        assert game.is_terminal()

        # Q♠ beats J♠ (high card)
        returns = game.returns()
        assert returns == (-1.0, 1.0)

    def test_bet_call_round2_showdown(self):
        """Test showdown after bet-call in round 2."""
        # J♠ vs Q♠, public K♠
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], LEDUC_DECK[4]),
            history="cc/",
            pot=2,
            round=2
        )

        # P0 bets (4 chips in round 2)
        game = game.apply_action(1)
        assert game.pot == 6

        # P1 calls
        game = game.apply_action(1)
        assert game.is_terminal()
        assert game.pot == 10

        # Q beats J
        returns = game.returns()
        assert returns == (-5.0, 5.0)


class TestLeducHandEvaluation:
    """Test hand value calculation."""

    def test_pair_beats_high_card(self):
        """Test that a pair beats high card."""
        # J♠ vs K♠, public J♥ -> P0 has pair of Jacks
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[4], LEDUC_DECK[1]),
            history="cc/cc",
            pot=2,
            round=2
        )

        assert game.is_terminal()
        returns = game.returns()
        assert returns == (1.0, -1.0)  # P0 wins with pair

    def test_higher_pair_wins(self):
        """Test that higher pair wins."""
        # Q♠ vs K♠, public Q♥ -> P0 has pair of Queens
        game = LeducPoker(
            cards=(LEDUC_DECK[2], LEDUC_DECK[4], LEDUC_DECK[3]),
            history="cc/cc",
            pot=2,
            round=2
        )

        returns = game.returns()
        assert returns == (1.0, -1.0)  # P0 wins with pair of Q

        # K♠ vs J♠, public K♥ -> P0 has pair of Kings
        game = LeducPoker(
            cards=(LEDUC_DECK[4], LEDUC_DECK[0], LEDUC_DECK[5]),
            history="cc/cc",
            pot=2,
            round=2
        )

        returns = game.returns()
        assert returns == (1.0, -1.0)  # P0 wins with pair of K

    def test_high_card_comparison(self):
        """Test high card tiebreaker."""
        # K♠ vs Q♠, public J♥ -> K beats Q
        game = LeducPoker(
            cards=(LEDUC_DECK[4], LEDUC_DECK[2], LEDUC_DECK[1]),
            history="cc/cc",
            pot=2,
            round=2
        )

        returns = game.returns()
        assert returns == (1.0, -1.0)  # P0 wins with K

    def test_tie_same_rank(self):
        """Test tie when both have same rank (different suits)."""
        # J♠ vs J♥, public Q♠ -> Tie (both have J)
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[1], LEDUC_DECK[2]),
            history="cc/cc",
            pot=2,
            round=2
        )

        returns = game.returns()
        assert returns == (0.0, 0.0)  # Tie


class TestLeducInformationStates:
    """Test information state representation."""

    def test_information_state_round1(self):
        """Test info state in round 1."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            round=1
        )

        # Player 0's perspective
        info_state = game.information_state_string()
        assert "J♠" in info_state
        assert "|" in info_state
        assert "||" in info_state  # No public card yet

    def test_information_state_round2(self):
        """Test info state in round 2 with public card."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], LEDUC_DECK[4]),
            history="cc/",
            round=2
        )

        info_state = game.information_state_string()
        assert "J♠" in info_state
        assert "K♠" in info_state  # Public card
        assert "cc/" in info_state  # History

    def test_information_state_different_players(self):
        """Test that different players see different info states."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="b",
            round=1
        )

        # Player 1's turn
        assert game.current_player() == 1

        # Different private card
        info_state_p1 = game.information_state_string()
        assert "Q♠" in info_state_p1


class TestLeducPotAccounting:
    """Test pot size calculations."""

    def test_pot_after_round1_bet_call(self):
        """Test pot after bet-call in round 1."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            pot=2,
            round=1
        )

        game = game.apply_action(1)  # P0 bets 2
        assert game.pot == 4

        game = game.apply_action(1)  # P1 calls 2
        assert game.pot == 6

    def test_pot_after_round2_bet_call(self):
        """Test pot after bet-call in round 2."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], LEDUC_DECK[4]),
            history="cc/",
            pot=2,
            round=2
        )

        game = game.apply_action(1)  # P0 bets 4
        assert game.pot == 6

        game = game.apply_action(1)  # P1 calls 4
        assert game.pot == 10

    def test_pot_multiple_bets(self):
        """Test pot with bet-raise sequence."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            pot=2,
            round=1
        )

        # P0 bets
        game = game.apply_action(1)
        assert game.pot == 4
        assert game.player_bets == (3, 1)

        # P1 calls (completes the round)
        game = game.apply_action(1)
        assert game.pot == 6
        assert game.player_bets == (0, 0)  # Reset for round 2
        assert game.round == 2


class TestLeducEdgeCases:
    """Test edge cases and error handling."""

    def test_cannot_get_returns_from_nonterminal(self):
        """Test that returns() raises error for non-terminal states."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="",
            round=1
        )

        with pytest.raises(ValueError):
            game.returns()

    def test_legal_actions_terminal(self):
        """Test that terminal states have no legal actions."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="bc",
            round=1
        )

        assert game.is_terminal()
        assert game.legal_actions() == []

    def test_legal_actions_chance(self):
        """Test that chance nodes have no legal actions."""
        game = LeducPoker()
        assert game.is_chance_node()
        assert game.legal_actions() == []

    def test_chance_outcomes_initial(self):
        """Test chance outcomes at initial state."""
        game = LeducPoker()
        outcomes = game.chance_outcomes()

        assert len(outcomes) == 6  # All 6 cards
        assert all(prob == 1.0/6.0 for _, prob in outcomes)

    def test_chance_outcomes_second_card(self):
        """Test chance outcomes for second private card."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], None, None),
            history="",
            round=1
        )

        outcomes = game.chance_outcomes()
        assert len(outcomes) == 5  # 5 remaining cards

        # Check that dealt card is not in outcomes
        action_indices = [action for action, _ in outcomes]
        assert 0 not in action_indices  # J♠ already dealt

    def test_chance_outcomes_public_card(self):
        """Test chance outcomes for public card."""
        game = LeducPoker(
            cards=(LEDUC_DECK[0], LEDUC_DECK[2], None),
            history="cc/",
            round=2
        )

        outcomes = game.chance_outcomes()
        assert len(outcomes) == 4  # 4 remaining cards

        action_indices = [action for action, _ in outcomes]
        assert 0 not in action_indices  # J♠ dealt
        assert 2 not in action_indices  # Q♠ dealt


class TestLeducCompleteGames:
    """Test complete game scenarios."""

    def test_quick_fold_game(self):
        """Test a game ending in immediate fold."""
        game = LeducPoker()

        # Deal cards
        game = game.apply_action(0)  # J♠ to P0
        game = game.apply_action(2)  # Q♠ to P1

        # P0 bets
        game = game.apply_action(1)

        # P1 folds
        game = game.apply_action(0)

        assert game.is_terminal()
        assert game.returns() == (2.0, -2.0)

    def test_full_game_with_showdown(self):
        """Test a complete game going to showdown."""
        game = LeducPoker()

        # Deal private cards
        game = game.apply_action(0)  # J♠ to P0
        game = game.apply_action(2)  # Q♠ to P1

        # Round 1: both check
        game = game.apply_action(0)
        game = game.apply_action(0)

        assert game.round == 2
        assert game.is_chance_node()

        # Deal public card
        game = game.apply_action(4)  # K♠

        # Round 2: both check
        game = game.apply_action(0)
        game = game.apply_action(0)

        assert game.is_terminal()

        # Q beats J
        assert game.returns() == (-1.0, 1.0)

    def test_game_with_pair(self):
        """Test a game where one player makes a pair."""
        game = LeducPoker()

        # Deal J♠ vs Q♠
        game = game.apply_action(0)
        game = game.apply_action(2)

        # Round 1: both check
        game = game.apply_action(0)
        game = game.apply_action(0)

        # Deal J♥ (P0 makes pair)
        game = game.apply_action(1)

        # Round 2: P0 bets, P1 calls
        game = game.apply_action(1)
        game = game.apply_action(1)

        assert game.is_terminal()

        # P0 wins with pair
        assert game.returns() == (5.0, -5.0)
