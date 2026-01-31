"""Leduc Poker convergence validation for CI.

Trains tabular VanillaCFR on Leduc Poker and verifies that
exploitability (NashConv) converges toward zero, confirming the game logic,
CFR algorithm, and metrics pipeline all work correctly end-to-end.

Runs in ~90 seconds on a standard CI runner (no GPU needed).
"""

import pytest

from aion26.games.leduc import LeducPoker
from aion26.cfr.vanilla import VanillaCFR
from aion26.metrics.exploitability import compute_exploitability


@pytest.fixture(scope="module")
def trained_cfr():
    """Train VanillaCFR on Leduc for 5000 iterations, recording NashConv at checkpoints."""
    initial_state = LeducPoker()
    cfr = VanillaCFR(initial_state, seed=42)

    checkpoints = [200, 500, 1000, 2000, 3000, 5000]
    nashconv_history = {}

    for target in checkpoints:
        while cfr.iteration < target:
            cfr.run_iteration()

        avg_strategy = cfr.get_all_average_strategies()
        nashconv = compute_exploitability(initial_state, avg_strategy)
        nashconv_history[target] = nashconv

    return cfr, nashconv_history


class TestLeducConvergence:
    """Validate that CFR converges on Leduc Poker."""

    def test_nashconv_converges_toward_zero(self, trained_cfr):
        """Absolute NashConv at iteration 5000 should be much smaller than at 200."""
        _, history = trained_cfr

        early_abs = abs(history[200])
        late_abs = abs(history[5000])

        assert late_abs < early_abs, (
            f"|NashConv| did not decrease: early={early_abs:.4f}, late={late_abs:.4f}"
        )
        reduction = (early_abs - late_abs) / early_abs
        assert reduction > 0.3, (
            f"|NashConv| reduction only {reduction:.1%}, expected >30%: "
            f"early={early_abs:.4f}, late={late_abs:.4f}"
        )

    def test_nashconv_near_zero_at_end(self, trained_cfr):
        """After 5000 iterations, NashConv should be close to zero (< 0.05)."""
        _, history = trained_cfr

        final_abs = abs(history[5000])
        assert final_abs < 0.05, (
            f"|NashConv| at iter 5000 is {final_abs:.4f}, expected < 0.05"
        )

    def test_early_convergence_visible(self, trained_cfr):
        """NashConv should show clear improvement between iter 200 and 1000."""
        _, history = trained_cfr

        abs_200 = abs(history[200])
        abs_1000 = abs(history[1000])

        assert abs_1000 < abs_200, (
            f"|NashConv| did not improve from iter 200 ({abs_200:.4f}) "
            f"to iter 1000 ({abs_1000:.4f})"
        )

    def test_info_sets_discovered(self, trained_cfr):
        """After training, the solver should have discovered many Leduc info sets."""
        cfr, _ = trained_cfr

        num_info_sets = len(cfr.strategy_sum)

        # Leduc has ~288 theoretical info sets; with external sampling
        # we should discover a substantial fraction
        assert num_info_sets > 50, (
            f"Only {num_info_sets} info sets discovered, expected >50"
        )
