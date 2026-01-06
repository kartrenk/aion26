"""Unit tests for discounting schedulers."""

import pytest
import numpy as np

from aion26.learner.discounting import (
    UniformScheduler,
    LinearScheduler,
    PDCFRScheduler,
    GeometricScheduler,
    create_scheduler,
)


class TestUniformScheduler:
    """Tests for uniform (no discounting) scheduler."""

    def test_constant_weight(self):
        """All iterations should have weight 1.0."""
        scheduler = UniformScheduler()

        for t in [1, 10, 100, 1000]:
            assert scheduler.get_weight(t) == 1.0

    def test_accumulated_weight(self):
        """Accumulated weight should equal iteration count."""
        scheduler = UniformScheduler()

        assert scheduler.get_accum_weight(1) == 1.0
        assert scheduler.get_accum_weight(10) == 10.0
        assert scheduler.get_accum_weight(100) == 100.0

    def test_invalid_iteration(self):
        """Should raise error for iteration < 1."""
        scheduler = UniformScheduler()

        with pytest.raises(ValueError):
            scheduler.get_weight(0)

        with pytest.raises(ValueError):
            scheduler.get_weight(-5)


class TestLinearScheduler:
    """Tests for linear discounting scheduler."""

    def test_linear_weight(self):
        """Weight should equal iteration number."""
        scheduler = LinearScheduler()

        assert scheduler.get_weight(1) == 1.0
        assert scheduler.get_weight(10) == 10.0
        assert scheduler.get_weight(100) == 100.0
        assert scheduler.get_weight(1000) == 1000.0

    def test_accumulated_weight_closed_form(self):
        """Accumulated weight should use closed form: t(t+1)/2."""
        scheduler = LinearScheduler()

        # sum(1..1) = 1
        assert scheduler.get_accum_weight(1) == 1.0

        # sum(1..10) = 55
        assert scheduler.get_accum_weight(10) == 55.0

        # sum(1..100) = 5050
        assert scheduler.get_accum_weight(100) == 5050.0

    def test_accumulated_weight_formula(self):
        """Verify accumulated weight matches manual calculation."""
        scheduler = LinearScheduler()

        for t in [1, 5, 10, 50]:
            expected = t * (t + 1) / 2.0
            assert scheduler.get_accum_weight(t) == expected

    def test_increasing_weights(self):
        """Weights should increase with iteration."""
        scheduler = LinearScheduler()

        weights = [scheduler.get_weight(t) for t in range(1, 101)]

        # Check monotonically increasing
        for i in range(len(weights) - 1):
            assert weights[i + 1] > weights[i]

    def test_invalid_iteration(self):
        """Should raise error for iteration < 1."""
        scheduler = LinearScheduler()

        with pytest.raises(ValueError):
            scheduler.get_weight(0)


class TestPDCFRScheduler:
    """Tests for PDCFR+ discounting scheduler."""

    def test_default_parameters(self):
        """Test with default α=2.0, β=0.5."""
        scheduler = PDCFRScheduler()

        assert scheduler.alpha == 2.0
        assert scheduler.beta == 0.5

    def test_positive_regret_weights(self):
        """Test weights for positive regrets (α=2.0)."""
        scheduler = PDCFRScheduler(alpha=2.0)

        # w_1 = 1^2 / (1^2 + 1) = 1/2 = 0.5
        assert abs(scheduler.get_weight(1, "positive") - 0.5) < 1e-6

        # w_10 = 10^2 / (10^2 + 1) = 100/101 ≈ 0.9901
        expected = 100.0 / 101.0
        assert abs(scheduler.get_weight(10, "positive") - expected) < 1e-6

        # w_100 = 100^2 / (100^2 + 1) = 10000/10001 ≈ 0.9999
        expected = 10000.0 / 10001.0
        assert abs(scheduler.get_weight(100, "positive") - expected) < 1e-6

    def test_negative_regret_weights(self):
        """Test weights for negative regrets (β=0.5)."""
        scheduler = PDCFRScheduler(beta=0.5)

        # w_1 = 1^0.5 / (1^0.5 + 1) = 1/2 = 0.5
        assert abs(scheduler.get_weight(1, "negative") - 0.5) < 1e-6

        # w_100 = 100^0.5 / (100^0.5 + 1) = 10/11 ≈ 0.9091
        expected = 10.0 / 11.0
        assert abs(scheduler.get_weight(100, "negative") - expected) < 1e-6

    def test_weights_approach_one(self):
        """Weights should approach 1.0 as iteration increases."""
        scheduler = PDCFRScheduler(alpha=2.0)

        # For large t, t^α / (t^α + 1) → 1
        w_1000 = scheduler.get_weight(1000, "positive")
        assert w_1000 > 0.999

        w_10000 = scheduler.get_weight(10000, "positive")
        assert w_10000 > 0.9999

    def test_positive_only_mode(self):
        """Test positive_only flag (no discounting for negative regrets)."""
        scheduler = PDCFRScheduler(alpha=2.0, beta=0.5, positive_only=True)

        # Positive regrets use α
        w_pos = scheduler.get_weight(10, "positive")
        assert abs(w_pos - 100.0 / 101.0) < 1e-6

        # Negative regrets use uniform (1.0)
        w_neg = scheduler.get_weight(10, "negative")
        assert w_neg == 1.0

        w_neg_100 = scheduler.get_weight(100, "negative")
        assert w_neg_100 == 1.0

    def test_different_exponents(self):
        """Test with different α and β values."""
        scheduler = PDCFRScheduler(alpha=1.5, beta=1.0)

        # α=1.5: w_10 = 10^1.5 / (10^1.5 + 1)
        t_pow = np.power(10, 1.5)
        expected_pos = t_pow / (t_pow + 1)
        assert abs(scheduler.get_weight(10, "positive") - expected_pos) < 1e-6

        # β=1.0: w_10 = 10^1 / (10^1 + 1) = 10/11
        expected_neg = 10.0 / 11.0
        assert abs(scheduler.get_weight(10, "negative") - expected_neg) < 1e-6

    def test_accumulated_weight(self):
        """Test accumulated weight computation."""
        scheduler = PDCFRScheduler(alpha=2.0)

        # Should sum weights from 1 to t
        accum_10 = scheduler.get_accum_weight(10)

        # Manual calculation
        manual_sum = sum(scheduler.get_weight(t, "positive") for t in range(1, 11))
        assert abs(accum_10 - manual_sum) < 1e-6

    def test_non_linear_growth(self):
        """Weights should grow non-linearly, approaching 1.0 asymptotically."""
        scheduler = PDCFRScheduler(alpha=2.0)

        weights = [scheduler.get_weight(t, "positive") for t in range(1, 21)]

        # Weights should be monotonically increasing
        for i in range(len(weights) - 1):
            assert weights[i + 1] > weights[i]

        # First weight should be 0.5, last should approach 1.0
        assert abs(weights[0] - 0.5) < 1e-6
        assert weights[-1] > 0.9
        assert weights[-1] < 1.0

        # Check non-linear: not equal spacing
        # For linear, deltas would be constant
        deltas = [weights[i + 1] - weights[i] for i in range(len(weights) - 1)]

        # Deltas should decrease as we approach 1.0 (asymptotic behavior)
        # Early deltas > later deltas
        assert deltas[0] > deltas[-1]

    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError):
            PDCFRScheduler(alpha=-1.0)

        with pytest.raises(ValueError):
            PDCFRScheduler(beta=-0.5)

        scheduler = PDCFRScheduler()
        with pytest.raises(ValueError):
            scheduler.get_weight(0)

        with pytest.raises(ValueError):
            scheduler.get_weight(10, "invalid")  # type: ignore


class TestGeometricScheduler:
    """Tests for geometric discounting scheduler."""

    def test_default_gamma(self):
        """Test default γ=0.99."""
        scheduler = GeometricScheduler()
        assert scheduler.gamma == 0.99

    def test_custom_gamma(self):
        """Test custom γ values."""
        scheduler = GeometricScheduler(gamma=0.95)
        assert scheduler.gamma == 0.95

    def test_relative_weights(self):
        """Test relative weighting (exponential decay)."""
        scheduler = GeometricScheduler(gamma=0.9)

        # Weight for iteration 1 when at iteration 10
        # age = 10 - 1 = 9, weight = 0.9^9
        w = scheduler.get_weight_relative(1, 10)
        expected = np.power(0.9, 9)
        assert abs(w - expected) < 1e-6

        # Weight for iteration 10 when at iteration 10 (current)
        # age = 0, weight = 0.9^0 = 1.0
        w = scheduler.get_weight_relative(10, 10)
        assert abs(w - 1.0) < 1e-6

    def test_invalid_gamma(self):
        """Test error handling for invalid γ."""
        with pytest.raises(ValueError):
            GeometricScheduler(gamma=0.0)

        with pytest.raises(ValueError):
            GeometricScheduler(gamma=1.1)

        with pytest.raises(ValueError):
            GeometricScheduler(gamma=-0.5)


class TestSchedulerFactory:
    """Tests for create_scheduler factory function."""

    def test_create_uniform(self):
        """Create uniform scheduler."""
        scheduler = create_scheduler("uniform")
        assert isinstance(scheduler, UniformScheduler)
        assert scheduler.get_weight(10) == 1.0

    def test_create_linear(self):
        """Create linear scheduler."""
        scheduler = create_scheduler("linear")
        assert isinstance(scheduler, LinearScheduler)
        assert scheduler.get_weight(100) == 100.0

    def test_create_pdcfr(self):
        """Create PDCFR scheduler with custom parameters."""
        scheduler = create_scheduler("pdcfr", alpha=1.5, beta=1.0)
        assert isinstance(scheduler, PDCFRScheduler)
        assert scheduler.alpha == 1.5
        assert scheduler.beta == 1.0

    def test_create_geometric(self):
        """Create geometric scheduler."""
        scheduler = create_scheduler("geometric", gamma=0.95)
        assert isinstance(scheduler, GeometricScheduler)
        assert scheduler.gamma == 0.95

    def test_invalid_type(self):
        """Test error for invalid scheduler type."""
        with pytest.raises(ValueError):
            create_scheduler("invalid_type")  # type: ignore


class TestSchedulerComparison:
    """Comparative tests across different schedulers."""

    def test_weight_ordering_at_iteration_100(self):
        """Compare weights at iteration 100."""
        uniform = UniformScheduler()
        linear = LinearScheduler()
        pdcfr = PDCFRScheduler(alpha=2.0)

        w_uniform = uniform.get_weight(100)
        w_linear = linear.get_weight(100)
        w_pdcfr = pdcfr.get_weight(100, "positive")

        # Uniform: 1.0
        # Linear: 100.0
        # PDCFR: 10000/10001 ≈ 0.9999

        # Linear should be much larger
        assert w_linear > w_uniform
        assert w_linear > w_pdcfr

        # PDCFR should be close to but less than 1.0
        assert w_pdcfr < 1.0
        assert w_pdcfr > 0.99

    def test_accumulated_weight_growth_rates(self):
        """Compare how accumulated weights grow."""
        uniform = UniformScheduler()
        linear = LinearScheduler()
        pdcfr = PDCFRScheduler(alpha=2.0)

        # At iteration 10
        accum_uniform_10 = uniform.get_accum_weight(10)
        accum_linear_10 = linear.get_accum_weight(10)
        accum_pdcfr_10 = pdcfr.get_accum_weight(10)

        # Uniform: 10 (sum of 1s)
        # Linear: 55 (sum of 1..10)
        # PDCFR: ~9 (sum of fractional weights < 1)

        assert accum_uniform_10 == 10.0
        assert accum_linear_10 == 55.0

        # PDCFR weights are fractional (< 1), so accumulated sum < uniform
        # But should be positive and reasonable
        assert 0 < accum_pdcfr_10 < accum_uniform_10

        # Linear should be largest
        assert accum_linear_10 > accum_uniform_10 > accum_pdcfr_10


class TestNumericalStability:
    """Test numerical stability for edge cases."""

    def test_very_large_iterations(self):
        """Test with very large iteration numbers."""
        scheduler = PDCFRScheduler(alpha=2.0)

        # Should not overflow or produce NaN
        w = scheduler.get_weight(100000, "positive")
        assert not np.isnan(w)
        assert not np.isinf(w)
        assert 0 < w <= 1.0

    def test_alpha_zero(self):
        """Test with α=0 (should give uniform weighting)."""
        scheduler = PDCFRScheduler(alpha=0.0, beta=0.0)

        # With α=0, should return uniform weight
        assert scheduler.get_weight(1, "positive") == 1.0
        assert scheduler.get_weight(100, "positive") == 1.0

    def test_very_small_alpha(self):
        """Test with very small α."""
        scheduler = PDCFRScheduler(alpha=0.01)

        # Should still produce valid weights
        w1 = scheduler.get_weight(1, "positive")
        w100 = scheduler.get_weight(100, "positive")

        assert 0 < w1 <= 1.0
        assert 0 < w100 <= 1.0
        # Small α means slow growth
        assert w100 > w1
        assert w100 - w1 < 0.5  # Growth should be slow


class TestDocumentedExamples:
    """Validate examples from docstrings."""

    def test_pdcfr_standard_config(self):
        """Test standard PDCFR+ configuration from paper."""
        # Brown & Sandholm (2019) standard settings
        scheduler = PDCFRScheduler(alpha=2.0, beta=0.5)

        # Iteration 1: both should be 0.5
        assert abs(scheduler.get_weight(1, "positive") - 0.5) < 1e-6
        assert abs(scheduler.get_weight(1, "negative") - 0.5) < 1e-6

        # Iteration 100
        w_pos_100 = scheduler.get_weight(100, "positive")
        w_neg_100 = scheduler.get_weight(100, "negative")

        # Positive (α=2): 10000/10001 ≈ 0.9999
        assert w_pos_100 > 0.999

        # Negative (β=0.5): 10/11 ≈ 0.909
        assert abs(w_neg_100 - 10.0 / 11.0) < 1e-6

        # Positive should converge faster than negative
        assert w_pos_100 > w_neg_100

    def test_linear_cfr_equivalence(self):
        """Linear scheduler should match Linear CFR."""
        scheduler = LinearScheduler()

        # In Linear CFR, iteration t has weight t
        for t in [1, 5, 10, 50, 100]:
            assert scheduler.get_weight(t) == float(t)
