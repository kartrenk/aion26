"""Tests for Vanilla CFR implementation."""

import pytest
import numpy as np
from aion26.cfr.vanilla import VanillaCFR
from aion26.cfr.regret_matching import regret_matching, sample_action
from aion26.games.kuhn import new_kuhn_game


class TestRegretMatching:
    """Test regret matching utility function."""

    def test_regret_matching_all_positive(self):
        """Test regret matching with all positive regrets."""
        regrets = np.array([1.0, 2.0, 3.0])
        strategy = regret_matching(regrets)
        # Strategy should be proportional to regrets
        expected = np.array([1.0, 2.0, 3.0]) / 6.0
        np.testing.assert_array_almost_equal(strategy, expected)
        # Should sum to 1
        assert abs(strategy.sum() - 1.0) < 1e-10

    def test_regret_matching_mixed(self):
        """Test regret matching with mixed positive/negative regrets."""
        regrets = np.array([-1.0, 2.0, 3.0])
        strategy = regret_matching(regrets)
        # Only positive regrets contribute
        expected = np.array([0.0, 2.0, 3.0]) / 5.0
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_regret_matching_all_negative(self):
        """Test regret matching with all negative regrets.

        When all regrets are negative, we use argmax to pick the "least bad"
        action rather than uniform random. This prevents suicidal bluffs with
        trash hands during early training.
        """
        regrets = np.array([-1.0, -2.0, -3.0])
        strategy = regret_matching(regrets)
        # Should pick action with highest (least negative) regret
        expected = np.array([1.0, 0.0, 0.0])  # -1.0 is the highest
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_regret_matching_zeros(self):
        """Test regret matching with all zeros.

        When all regrets are zero, argmax picks the first action.
        """
        regrets = np.array([0.0, 0.0, 0.0])
        strategy = regret_matching(regrets)
        # argmax on ties returns first index
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(strategy, expected)


class TestSampleAction:
    """Test action sampling utility."""

    def test_sample_action_deterministic(self):
        """Test sampling from deterministic strategy."""
        strategy = np.array([0.0, 1.0, 0.0])
        rng = np.random.default_rng(42)
        # Should always sample action 1
        for _ in range(10):
            action = sample_action(strategy, rng)
            assert action == 1

    def test_sample_action_uniform(self):
        """Test sampling from uniform strategy."""
        strategy = np.array([1.0, 1.0, 1.0]) / 3.0
        rng = np.random.default_rng(42)
        # Sample many times and check distribution
        samples = [sample_action(strategy, rng) for _ in range(1000)]
        # Each action should be sampled roughly 1/3 of the time
        counts = np.bincount(samples)
        proportions = counts / len(samples)
        for p in proportions:
            assert 0.25 < p < 0.42  # Rough check for ~1/3


class TestVanillaCFRBasics:
    """Test basic CFR functionality."""

    def test_initialization(self):
        """Test CFR solver initialization."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)
        assert solver.iteration == 0
        assert len(solver.regret_sum) == 0
        assert len(solver.strategy_sum) == 0

    def test_get_strategy_initial(self):
        """Test getting strategy for unvisited information state.

        With argmax fallback, unvisited states (zero regrets) pick the first action.
        """
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)
        # Unvisited state with zero regrets -> argmax picks first action
        strategy = solver.get_strategy("J")
        expected = np.array([1.0, 0.0])  # First action (check) selected
        np.testing.assert_array_almost_equal(strategy, expected)

    def test_run_iteration(self):
        """Test running a single CFR iteration."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)
        solver.run_iteration()
        assert solver.iteration == 1
        # After one iteration, should have visited some states
        assert len(solver.regret_sum) > 0


class TestKuhnCFRConvergence:
    """Test CFR convergence on Kuhn Poker."""

    def test_cfr_runs_without_error(self):
        """Test CFR runs for many iterations without crashing."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)
        for _ in range(100):
            solver.run_iteration()
        assert solver.iteration == 100

    def test_cfr_visits_all_information_sets(self):
        """Test CFR visits all 12 Kuhn Poker information sets."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)
        # Run CFR for enough iterations to visit all states
        for _ in range(1000):
            solver.run_iteration()

        # Kuhn has 12 information sets:
        # P0: J, Q, K (first action)
        # P0: Jcb, Qcb, Kcb (after check, facing bet)
        # P1: Jc, Qc, Kc (after opponent checked)
        # P1: Jb, Qb, Kb (after opponent bet)
        expected_infostates = {
            # P0 first action
            "J", "Q", "K",
            # P0 facing a bet after checking
            "Jcb", "Qcb", "Kcb",
            # P1 after opponent checked
            "Jc", "Qc", "Kc",
            # P1 after opponent bet
            "Jb", "Qb", "Kb",
        }

        visited_infostates = set(solver.strategy_sum.keys())
        assert expected_infostates == visited_infostates

    def test_cfr_strategy_sum_increases(self):
        """Test that strategy sums accumulate over iterations."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)

        # Run 100 iterations
        for _ in range(100):
            solver.run_iteration()

        # Get strategy sum for some infostate
        strat_sum_100 = solver.strategy_sum.get("J", np.zeros(2)).copy()

        # Run 100 more iterations
        for _ in range(100):
            solver.run_iteration()

        strat_sum_200 = solver.strategy_sum.get("J", np.zeros(2))

        # Strategy sum should have increased
        assert strat_sum_200.sum() > strat_sum_100.sum()

    def test_average_strategy_probabilities(self):
        """Test average strategies are valid probability distributions."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)

        for _ in range(1000):
            solver.run_iteration()

        # Check all average strategies
        for info_state in solver.strategy_sum.keys():
            avg_strategy = solver.get_average_strategy(info_state)
            # Should sum to 1
            assert abs(avg_strategy.sum() - 1.0) < 1e-10
            # All probabilities should be non-negative
            assert (avg_strategy >= 0).all()
            # All probabilities should be <= 1
            assert (avg_strategy <= 1.0).all()


class TestKuhnNashEquilibrium:
    """Test CFR convergence properties on Kuhn Poker.

    Note: This implementation uses outcome sampling (MCCFR) which converges
    slowly. These tests verify basic convergence properties, not exact Nash equilibrium.
    Full convergence testing will be done with the exploitability calculator.
    """

    def test_strategies_are_not_random(self):
        """Test that strategies deviate from uniform random after training."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)

        for _ in range(10000):
            solver.run_iteration()

        # Get strategies for various states
        king_strategy = solver.get_average_strategy("K")
        queen_strategy = solver.get_average_strategy("Q")
        jack_strategy = solver.get_average_strategy("J")

        # After 10k iterations, strategies should not be exactly uniform (0.5, 0.5)
        # Allow for some deviation from uniform
        assert abs(king_strategy[0] - 0.5) > 0.01 or abs(king_strategy[1] - 0.5) > 0.01
        assert abs(queen_strategy[0] - 0.5) > 0.01 or abs(queen_strategy[1] - 0.5) > 0.01
        assert abs(jack_strategy[0] - 0.5) > 0.01 or abs(jack_strategy[1] - 0.5) > 0.01

    def test_jack_tends_to_fold(self):
        """Test that Jack has some tendency to fold when facing a bet.

        This is a weak test - we just check that Jack doesn't always call.
        """
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)

        for _ in range(10000):
            solver.run_iteration()

        # Get average strategy for Jack facing a bet
        jack_vs_bet_strategy = solver.get_average_strategy("Jb")
        # jack_vs_bet_strategy[0] is probability of folding
        # Just check it's not purely calling (strategy should not be [0, 1])
        assert jack_vs_bet_strategy[0] > 0.5, f"Jack should tend to fold, but strategy is {jack_vs_bet_strategy}"

    def test_strategies_converge_over_time(self):
        """Test that strategies change less as training progresses."""
        game = new_kuhn_game()
        solver = VanillaCFR(game, seed=42)

        # Train for 1000 iterations
        for _ in range(1000):
            solver.run_iteration()
        early_king = solver.get_average_strategy("K").copy()

        # Train for 9000 more iterations (total 10000)
        for _ in range(9000):
            solver.run_iteration()
        mid_king = solver.get_average_strategy("K").copy()

        # Train for 10000 more iterations (total 20000)
        for _ in range(10000):
            solver.run_iteration()
        late_king = solver.get_average_strategy("K").copy()

        # Strategies should change less over time (convergence)
        early_to_mid_change = np.abs(mid_king - early_king).sum()
        mid_to_late_change = np.abs(late_king - mid_king).sum()

        # Later changes should generally be smaller (convergence property)
        # We allow some variance due to sampling, so this is a weak test
        # Just check that strategies don't diverge wildly
        assert mid_to_late_change < early_to_mid_change * 2.0
