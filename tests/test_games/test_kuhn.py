"""Tests for Kuhn Poker implementation."""

import pytest
import numpy as np
from aion26.games.kuhn import (
    KuhnPoker,
    new_kuhn_game,
    JACK,
    QUEEN,
    KING,
    CHECK,
    BET,
)


class TestKuhnPokerBasics:
    """Test basic game mechanics."""

    def test_initial_state(self):
        """Test game starts in correct initial state."""
        game = new_kuhn_game()
        assert game.cards == (None, None)
        assert game.history == ""
        assert game.pot == 2  # Both antes
        assert game.is_chance_node()
        assert not game.is_terminal()

    def test_chance_outcomes(self):
        """Test chance node has 6 equally likely outcomes."""
        game = new_kuhn_game()
        outcomes = game.chance_outcomes()
        assert len(outcomes) == 6
        assert all(prob == 1.0 / 6.0 for _, prob in outcomes)

    def test_deal_cards(self):
        """Test dealing cards moves from chance node to player action."""
        game = new_kuhn_game()
        # Deal J to P0, Q to P1 (action 0)
        game = game.apply_action(0)
        assert game.cards == (JACK, QUEEN)
        assert not game.is_chance_node()
        assert game.current_player() == 0  # P0 acts first


class TestKuhnPokerTerminalStates:
    """Test terminal conditions and payoffs."""

    def test_both_check_showdown(self):
        """Test cc (both check) goes to showdown."""
        # J vs Q, both check -> Q wins
        game = KuhnPoker(cards=(JACK, QUEEN))
        game = game.apply_action(CHECK)  # P0 checks
        game = game.apply_action(CHECK)  # P1 checks
        assert game.is_terminal()
        returns = game.returns()
        # P0 has J, P1 has Q -> P1 wins the 2-chip pot
        # P0 loses their ante (1), P1 wins 1 net
        assert returns == (-1.0, 1.0)

    def test_both_check_showdown_reverse(self):
        """Test cc with P0 having higher card."""
        # Q vs J, both check -> Q wins
        game = KuhnPoker(cards=(QUEEN, JACK))
        game = game.apply_action(CHECK).apply_action(CHECK)
        assert game.is_terminal()
        assert game.returns() == (1.0, -1.0)

    def test_bet_fold(self):
        """Test bc (bet then fold)."""
        # P0 bets, P1 folds
        game = KuhnPoker(cards=(JACK, QUEEN))
        game = game.apply_action(BET)  # P0 bets 1
        assert game.pot == 3  # 2 antes + 1 bet
        game = game.apply_action(CHECK)  # P1 folds (check when facing bet = fold)
        assert game.is_terminal()
        # P0 wins the pot (3), invested 2 total (ante + bet) -> +1 net
        # P1 loses ante -> -1 net
        assert game.returns() == (1.0, -1.0)

    def test_bet_call_showdown(self):
        """Test bb (bet then call goes to showdown)."""
        # J vs Q, P0 bets, P1 calls -> Q wins
        game = KuhnPoker(cards=(JACK, QUEEN))
        game = game.apply_action(BET)  # P0 bets
        game = game.apply_action(BET)  # P1 calls
        assert game.is_terminal()
        assert game.pot == 4  # 2 antes + 2 bets
        # P1 (Q) wins pot of 4, invested 2 -> +2 net
        # P0 (J) loses 2 -> -2 net
        assert game.returns() == (-2.0, 2.0)

    def test_check_bet_fold(self):
        """Test cbc (check, bet, fold)."""
        # P0 checks, P1 bets, P0 folds
        game = KuhnPoker(cards=(JACK, QUEEN))
        game = game.apply_action(CHECK)  # P0 checks
        game = game.apply_action(BET)  # P1 bets
        assert game.pot == 3
        game = game.apply_action(CHECK)  # P0 folds
        assert game.is_terminal()
        # P1 wins pot of 3, invested 2 (ante + bet) -> +1 net
        # P0 loses ante -> -1 net
        assert game.returns() == (-1.0, 1.0)

    def test_check_bet_call_showdown(self):
        """Test cbb (check, bet, call goes to showdown)."""
        # J vs Q: P0 checks, P1 bets, P0 calls -> Q wins
        game = KuhnPoker(cards=(JACK, QUEEN))
        game = game.apply_action(CHECK)
        game = game.apply_action(BET)
        game = game.apply_action(BET)  # P0 calls
        assert game.is_terminal()
        assert game.pot == 4
        # Q wins
        assert game.returns() == (-2.0, 2.0)


class TestKuhnPokerInformationStates:
    """Test information state representations."""

    def test_information_state_string(self):
        """Test information state string encoding."""
        # P0 with Jack at start
        game = KuhnPoker(cards=(JACK, QUEEN))
        assert game.information_state_string() == "J"

        # P0 with Jack after checking and facing a bet
        game = game.apply_action(CHECK)  # P0 checks
        game = game.apply_action(BET)  # P1 bets
        assert game.current_player() == 0
        assert game.information_state_string() == "Jcb"

        # P1 with Queen after P0 checked
        game2 = KuhnPoker(cards=(JACK, QUEEN))
        game2 = game2.apply_action(CHECK)
        assert game2.current_player() == 1
        assert game2.information_state_string() == "Qc"

    def test_information_state_tensor_shape(self):
        """Test information state tensor has correct shape."""
        game = KuhnPoker(cards=(JACK, QUEEN))
        tensor = game.information_state_tensor()
        assert tensor.shape == (9,)  # 3 for card + 6 for history
        assert tensor.dtype == np.float32

    def test_information_state_tensor_encoding(self):
        """Test information state tensor encoding."""
        # P0 with Jack at start
        game = KuhnPoker(cards=(JACK, QUEEN))
        tensor = game.information_state_tensor()
        # Jack is card 0 -> one-hot [1, 0, 0]
        assert np.array_equal(tensor[:3], [1.0, 0.0, 0.0])
        # No history -> all zeros
        assert np.array_equal(tensor[3:], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # P1 with Jack after P0 checked (history = "c")
        # When history is "c", current player is 1 (P1's turn)
        game2 = KuhnPoker(cards=(QUEEN, JACK), history="c")
        assert game2.current_player() == 1  # Verify it's P1's turn
        tensor2 = game2.information_state_tensor()
        # P1 has Jack (card 0) -> one-hot [1, 0, 0]
        assert np.array_equal(tensor2[:3], [1.0, 0.0, 0.0])
        # History "c" -> [1, 0, 0, 0, 0, 0] (check in first position)
        assert tensor2[3] == 1.0  # Check
        assert tensor2[4] == 0.0  # Not bet


class TestKuhnPokerLegalActions:
    """Test legal actions at different game states."""

    def test_legal_actions_at_start(self):
        """Test chance node has 6 legal actions."""
        game = new_kuhn_game()
        assert game.legal_actions() == [0, 1, 2, 3, 4, 5]

    def test_legal_actions_after_deal(self):
        """Test players can check or bet."""
        game = KuhnPoker(cards=(JACK, QUEEN))
        assert game.legal_actions() == [CHECK, BET]

    def test_legal_actions_terminal(self):
        """Test terminal states have no legal actions."""
        game = KuhnPoker(cards=(JACK, QUEEN), history="cc")
        assert game.is_terminal()
        assert game.legal_actions() == []


class TestKuhnPokerEdgeCases:
    """Test edge cases and error handling."""

    def test_cannot_get_returns_from_nonterminal(self):
        """Test error when getting returns from non-terminal state."""
        game = KuhnPoker(cards=(JACK, QUEEN))
        with pytest.raises(ValueError, match="non-terminal"):
            game.returns()

    def test_cannot_apply_action_to_terminal(self):
        """Test error when applying action to terminal state."""
        game = KuhnPoker(cards=(JACK, QUEEN), history="cc")
        with pytest.raises(ValueError, match="terminal"):
            game.apply_action(CHECK)

    def test_all_six_deals_valid(self):
        """Test all 6 possible card deals."""
        game = new_kuhn_game()
        expected_deals = [
            (JACK, QUEEN),
            (JACK, KING),
            (QUEEN, JACK),
            (QUEEN, KING),
            (KING, JACK),
            (KING, QUEEN),
        ]
        for i, expected in enumerate(expected_deals):
            dealt_game = game.apply_action(i)
            assert dealt_game.cards == expected

    def test_pot_accounting(self):
        """Test pot size is correctly tracked."""
        game = KuhnPoker(cards=(JACK, QUEEN))
        assert game.pot == 2  # Initial antes

        # P0 bets
        game = game.apply_action(BET)
        assert game.pot == 3
        assert game.player_0_invested == 2  # Ante + bet

        # P1 calls
        game = game.apply_action(BET)
        assert game.pot == 4
        assert game.player_1_invested == 2


class TestKuhnPokerAnalyticalSolution:
    """Test against known analytical Nash equilibrium properties."""

    def test_king_always_bets(self):
        """In Nash equilibrium, King should always bet at first action.

        This is a sanity check - the optimal strategy isn't tested here,
        but we verify the game allows this action.
        """
        game = KuhnPoker(cards=(KING, QUEEN))
        # King can bet
        assert BET in game.legal_actions()

    def test_jack_always_folds_to_bet(self):
        """Jack should fold when facing a bet (strategy check, not enforced)."""
        # P1 has Jack, P0 bet
        game = KuhnPoker(cards=(QUEEN, JACK), history="b")
        assert game.current_player() == 1
        # Jack can fold (CHECK when facing bet)
        assert CHECK in game.legal_actions()

    def test_game_value(self):
        """The game value for P0 under Nash equilibrium is -1/18.

        This will be validated by the CFR algorithm, not the game mechanics.
        Here we just ensure the game structure allows for this outcome.
        """
        # This is a placeholder - actual validation happens in CFR convergence
        game = KuhnPoker(cards=(JACK, QUEEN))
        # Verify game can be played
        assert not game.is_terminal()
        assert game.current_player() == 0
