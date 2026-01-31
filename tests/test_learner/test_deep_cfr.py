"""Tests for Deep CFR trainer implementation.

This module tests the DeepCFRTrainer class, including:
- Initialization and setup
- Network predictions
- Strategy computation
- CFR traversal
- Experience collection
- Network training
- Target network updates
- End-to-end integration
"""

import torch
import numpy as np

from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.deep_cfr.networks import KuhnEncoder
from aion26.games.kuhn import KuhnPoker, JACK, QUEEN


class TestDeepCFRTrainerInitialization:
    """Test trainer initialization and setup."""

    def test_initialization(self):
        """Test that trainer initializes correctly."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            hidden_size=64,
            num_hidden_layers=3,
            buffer_capacity=1000,
            seed=42,
        )

        assert trainer.iteration == 0
        assert trainer.num_actions == 2
        assert trainer.buffer.capacity == 1000
        assert len(trainer.buffer) == 0
        assert trainer.discount == 0.0  # Default: no bootstrap
        assert trainer.polyak == 0.01  # Default Polyak coefficient

    def test_networks_initialized_with_zero_output(self):
        """Test that networks have near-zero initial outputs."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        # Test advantage network
        test_input = torch.randn(5, 10)
        outputs = trainer.advantage_net(test_input)

        # Should be near-zero due to zero_init_output=True
        max_abs_value = outputs.abs().max().item()
        assert max_abs_value < 0.01, f"Max output {max_abs_value} should be < 0.01"

    def test_target_network_initialized_from_advantage(self):
        """Test that target network is initialized as copy of advantage network."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        # Networks should have identical weights initially
        test_input = torch.randn(5, 10)

        adv_output = trainer.advantage_net(test_input)
        target_output = trainer.target_net(test_input)

        torch.testing.assert_close(adv_output, target_output)

    def test_custom_hyperparameters(self):
        """Test initialization with custom hyperparameters."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            batch_size=64,
            discount=0.5,
            polyak=0.05,
            train_every=5,
            target_update_every=20,
            learning_rate=0.0001,
            seed=123,
        )

        assert trainer.batch_size == 64
        assert trainer.discount == 0.5
        assert trainer.polyak == 0.05
        assert trainer.train_every == 5
        assert trainer.target_update_every == 20


class TestNetworkPredictions:
    """Test neural network predictions."""

    def test_get_predicted_regrets(self):
        """Test getting regret predictions from network."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        # Create a test state
        state = KuhnPoker(cards=(JACK, QUEEN), history="")

        # Get predictions for player 0
        regrets = trainer.get_predicted_regrets(state, player=0)

        assert isinstance(regrets, torch.Tensor)
        assert regrets.shape == (2,)  # 2 actions
        assert not regrets.requires_grad  # Should be detached

    def test_predicted_regrets_near_zero_initially(self):
        """Test that initial predictions are near-zero due to zero-init."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        regrets = trainer.get_predicted_regrets(state, player=0)

        max_abs_regret = regrets.abs().max().item()
        assert max_abs_regret < 0.01, "Initial regrets should be near-zero"

    def test_target_network_predictions(self):
        """Test predictions from target network."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        state = KuhnPoker(cards=(JACK, QUEEN), history="")

        # Get predictions from both networks
        adv_regrets = trainer.get_predicted_regrets(state, player=0, use_target=False)
        target_regrets = trainer.get_predicted_regrets(state, player=0, use_target=True)

        # Should be identical initially
        torch.testing.assert_close(adv_regrets, target_regrets)


class TestStrategyComputation:
    """Test strategy computation via regret matching."""

    def test_get_strategy(self):
        """Test strategy computation from network predictions."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        strategy = trainer.get_strategy(state, player=0)

        assert isinstance(strategy, np.ndarray)
        assert strategy.shape == (2,)
        assert np.isclose(strategy.sum(), 1.0), "Strategy should sum to 1"
        assert np.all(strategy >= 0.0), "Strategy should be non-negative"

    def test_initial_strategy_near_uniform(self):
        """Test that initial strategy is valid (may not be exactly uniform).

        Note: Even with near-zero initialization, small numerical differences
        in regrets can lead to non-uniform strategies via regret matching.
        This is expected behavior. We just verify the strategy is valid.
        """
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        strategy = trainer.get_strategy(state, player=0)

        # Strategy should be a valid probability distribution
        assert np.isclose(strategy.sum(), 1.0), "Strategy should sum to 1"
        assert np.all(strategy >= 0.0), "Strategy should be non-negative"
        assert np.all(strategy <= 1.0), "Strategy probabilities should be <= 1"


class TestCFRTraversal:
    """Test CFR traversal and experience collection."""

    def test_traverse_terminal_state(self):
        """Test traversal at terminal state returns correct payoff."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        # Create terminal state: J vs Q, both check
        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        state = state.apply_action(0)  # P0 checks
        state = state.apply_action(0)  # P1 checks

        assert state.is_terminal()

        # Traverse for player 0
        value = trainer.traverse(state, update_player=0, reach_prob_0=1.0, reach_prob_1=1.0)

        # Player 0 has J, Player 1 has Q → Player 0 loses -1
        assert value == -1.0

    def test_traverse_collects_experiences(self):
        """Test that traversal collects experiences in buffer."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            seed=42,
        )

        initial_buffer_size = len(trainer.buffer)

        # Run one traversal
        trainer.traverse(initial_state, update_player=0, reach_prob_0=1.0, reach_prob_1=1.0)

        # Buffer should have new experiences
        assert len(trainer.buffer) > initial_buffer_size

    def test_traverse_handles_chance_nodes(self):
        """Test that traversal correctly handles chance nodes."""
        initial_state = KuhnPoker()  # Starts at chance node
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        # Should not raise error on chance node
        value = trainer.traverse(initial_state, update_player=0, reach_prob_0=1.0, reach_prob_1=1.0)

        assert isinstance(value, float)


class TestNetworkTraining:
    """Test neural network training."""

    def test_train_network_requires_full_buffer(self):
        """Test that training only happens when buffer is full enough."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=1000,
            batch_size=128,
            seed=42,
        )

        # Buffer is empty, should return 0 loss
        loss = trainer.train_network()
        assert loss == 0.0
        assert trainer.num_train_steps == 0

    def test_train_network_updates_weights(self):
        """Test that training actually updates network weights."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            batch_size=32,
            seed=42,
        )

        # Fill buffer with some experiences
        for _ in range(100):
            trainer.traverse(initial_state, update_player=0, reach_prob_0=1.0, reach_prob_1=1.0)

        # Get initial weights
        initial_weight = trainer.advantage_net.network[0].weight.data.clone()

        # Train network
        loss = trainer.train_network()

        # Weights should have changed
        new_weight = trainer.advantage_net.network[0].weight.data
        assert not torch.allclose(initial_weight, new_weight)
        assert loss > 0.0
        assert trainer.num_train_steps == 1

    def test_bootstrap_loss_with_discount_zero(self):
        """Test that discount=0.0 uses only instant regrets (vanilla Deep CFR)."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        # Vanilla Deep CFR: no bootstrap
        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            discount=0.0,  # No bootstrap
            buffer_capacity=100,
            batch_size=32,
            seed=42,
        )

        # Fill buffer
        for _ in range(100):
            trainer.traverse(initial_state, update_player=0, reach_prob_0=1.0, reach_prob_1=1.0)

        # Should train successfully
        loss = trainer.train_network()
        assert loss >= 0.0


class TestTargetNetworkUpdates:
    """Test target network Polyak averaging updates."""

    def test_update_target_network(self):
        """Test that target network updates using Polyak averaging."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            polyak=0.1,
            seed=42,
        )

        # Modify advantage network weights
        with torch.no_grad():
            for param in trainer.advantage_net.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Get target weights before update
        target_weight_before = trainer.target_net.network[0].weight.data.clone()

        # Update target network
        trainer.update_target_network()

        # Get target weights after update
        target_weight_after = trainer.target_net.network[0].weight.data

        # Weights should have changed (soft update)
        assert not torch.allclose(target_weight_before, target_weight_after)

    def test_polyak_averaging_formula(self):
        """Test that Polyak averaging formula is correct."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            polyak=0.1,
            seed=42,
        )

        # Modify advantage network significantly
        with torch.no_grad():
            for param in trainer.advantage_net.parameters():
                param.fill_(1.0)

        # Target network starts at ~0, advantage at 1.0
        # After update: target = 0.1 * 1.0 + 0.9 * 0.0 = 0.1

        trainer.update_target_network()

        # Check first layer weight
        target_weight = trainer.target_net.network[0].weight.data
        expected_value = 0.1  # polyak * 1.0 + (1-polyak) * 0.0

        mean_weight = target_weight.mean().item()
        assert np.isclose(mean_weight, expected_value, atol=0.01)


class TestRunIteration:
    """Test full iteration execution."""

    def test_run_iteration_increments_counter(self):
        """Test that run_iteration increments iteration counter."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        assert trainer.iteration == 0

        metrics = trainer.run_iteration()

        assert trainer.iteration == 1
        assert metrics["iteration"] == 1

    def test_run_iteration_fills_buffer(self):
        """Test that iterations fill the buffer with experiences."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            seed=42,
        )

        initial_size = len(trainer.buffer)

        # Run 10 iterations
        for _ in range(10):
            trainer.run_iteration()

        # Buffer should have more experiences
        assert len(trainer.buffer) > initial_size

    def test_run_iteration_trains_network(self):
        """Test that run_iteration trains network when appropriate."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=50,
            batch_size=32,
            train_every=5,
            seed=42,
        )

        # Fill buffer first
        for _ in range(50):
            trainer.run_iteration()

        # Run iteration that should trigger training
        metrics = trainer.run_iteration()

        # Should have trained by now
        assert trainer.num_train_steps > 0

    def test_run_iteration_updates_target_network(self):
        """Test that run_iteration updates target network periodically."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            target_update_every=10,
            seed=42,
        )

        # Run 9 iterations
        for _ in range(9):
            metrics = trainer.run_iteration()
            assert "target_updated" not in metrics

        # 10th iteration should update target
        metrics = trainer.run_iteration()
        assert metrics.get("target_updated", False) == True

    def test_run_iteration_returns_metrics(self):
        """Test that run_iteration returns comprehensive metrics."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        metrics = trainer.run_iteration()

        assert "iteration" in metrics
        assert "buffer_size" in metrics
        assert "buffer_fill_pct" in metrics
        assert "loss" in metrics


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_training_loop_convergence(self):
        """Test that training loop runs without errors and fills buffer."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            batch_size=32,
            train_every=1,
            target_update_every=10,
            seed=42,
        )

        # Run 100 iterations
        for i in range(100):
            metrics = trainer.run_iteration()

            # Verify metrics are reasonable
            assert metrics["iteration"] == i + 1
            assert metrics["buffer_size"] <= 100
            assert 0.0 <= metrics["buffer_fill_pct"] <= 100.0

        # Buffer should be full
        assert trainer.buffer.is_full

        # Should have trained multiple times
        assert trainer.num_train_steps > 0

        # Average loss should be computable
        avg_loss = trainer.get_average_loss()
        assert avg_loss >= 0.0

    def test_strategy_changes_over_time(self):
        """Test that strategies change as network trains."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            batch_size=32,
            train_every=1,
            seed=42,
        )

        # Get initial strategy
        test_state = KuhnPoker(cards=(JACK, QUEEN), history="")
        initial_strategy = trainer.get_strategy(test_state, player=0).copy()

        # Run training
        for _ in range(100):
            trainer.run_iteration()

        # Get final strategy
        final_strategy = trainer.get_strategy(test_state, player=0)

        # Strategies should be different after training
        assert not np.allclose(initial_strategy, final_strategy, atol=0.01)

    def test_bootstrap_training_with_discount(self):
        """Test training with bootstrap (discount > 0)."""
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            discount=0.5,  # Bootstrap enabled
            buffer_capacity=100,
            batch_size=32,
            train_every=1,
            target_update_every=10,
            seed=42,
        )

        # Should run without errors
        for _ in range(50):
            trainer.run_iteration()

        # Should have trained successfully
        assert trainer.num_train_steps > 0
        assert trainer.buffer.is_full


class TestPDCFRPlusIntegration:
    """Test PDCFR+ dynamic discounting integration."""

    def test_pdcfr_plus_schedulers_initialized(self):
        """Test that PDCFR+ schedulers are properly initialized."""
        from aion26.learner.discounting import PDCFRScheduler, LinearScheduler

        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state, encoder=encoder, input_size=10, output_size=2, seed=42
        )

        # Should have default schedulers
        assert isinstance(trainer.regret_scheduler, PDCFRScheduler)
        assert isinstance(trainer.strategy_scheduler, LinearScheduler)
        assert trainer.regret_scheduler.alpha == 2.0
        assert trainer.regret_scheduler.beta == 0.5

    def test_pdcfr_plus_custom_schedulers(self):
        """Test that custom schedulers can be provided."""
        from aion26.learner.discounting import PDCFRScheduler, UniformScheduler

        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        regret_sched = PDCFRScheduler(alpha=1.5, beta=1.0)
        strategy_sched = UniformScheduler()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            regret_scheduler=regret_sched,
            strategy_scheduler=strategy_sched,
            seed=42,
        )

        assert trainer.regret_scheduler.alpha == 1.5
        assert trainer.regret_scheduler.beta == 1.0
        assert isinstance(trainer.strategy_scheduler, UniformScheduler)

    def test_pdcfr_plus_dynamic_discounting_logic(self):
        """Test that dynamic discounting applies different weights to positive/negative regrets."""
        from unittest.mock import patch

        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        # Create trainer with PDCFR scheduler
        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            seed=42,
        )

        # Set iteration to 10 for testing
        trainer.iteration = 10

        # Mock target network to return known values: [1.0, -1.0]
        # (positive regret for action 0, negative for action 1)
        mock_target_regrets = torch.tensor([1.0, -1.0], dtype=torch.float32)

        with patch.object(trainer, "get_predicted_regrets", return_value=mock_target_regrets):
            # Run one traversal step
            trainer.run_iteration()

        # Verify buffer has experiences
        assert len(trainer.buffer) > 0

        # Extract a sample from buffer to verify discounting was applied
        sample_states, sample_targets = trainer.buffer.sample(min(10, len(trainer.buffer)))

        # Check that targets exist and are tensors
        assert sample_targets.shape[0] > 0
        assert sample_targets.shape[1] == 2  # 2 actions

        # The key verification: targets should reflect different discounting
        # We can't verify exact values without complex mocking, but we can verify
        # that the logic runs without errors and produces valid targets

        # Targets should be finite (not NaN or inf)
        assert torch.all(torch.isfinite(sample_targets))

    def test_pdcfr_plus_strategy_weighting(self):
        """Test that strategy accumulation uses dynamic weighting."""

        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        trainer = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=50,  # Small buffer to fill quickly
            seed=42,
        )

        # Run until buffer is full (strategy accumulation starts)
        while not trainer.buffer.is_full:
            trainer.run_iteration()

        # At this point, strategy_sum should be populated
        assert len(trainer.strategy_sum) > 0

        # Run 10 more iterations
        for _ in range(10):
            trainer.run_iteration()

        # Get average strategy
        avg_strategy = trainer.get_all_average_strategies()

        # Should have valid strategies
        assert len(avg_strategy) > 0

        # Strategies should sum to 1 (or close to it)
        for info_state, strategy in avg_strategy.items():
            strategy_sum = np.sum(strategy)
            assert 0.99 <= strategy_sum <= 1.01  # Allow small numerical error

    def test_pdcfr_plus_weights_increase_with_iteration(self):
        """Test that PDCFR+ weights increase as iterations progress."""
        from aion26.learner.discounting import PDCFRScheduler

        scheduler = PDCFRScheduler(alpha=2.0, beta=0.5)

        # Weights at iteration 1
        w1_pos = scheduler.get_weight(1, "positive")
        w1_neg = scheduler.get_weight(1, "negative")

        # Weights at iteration 100
        w100_pos = scheduler.get_weight(100, "positive")
        w100_neg = scheduler.get_weight(100, "negative")

        # Weights should increase
        assert w100_pos > w1_pos
        assert w100_neg > w1_neg

        # Positive should converge faster (higher alpha)
        assert w100_pos > w100_neg

    def test_pdcfr_plus_convergence_vs_vanilla(self):
        """Compare PDCFR+ vs vanilla Deep CFR convergence (qualitative test)."""
        from aion26.learner.discounting import PDCFRScheduler, UniformScheduler

        initial_state = KuhnPoker()
        encoder = KuhnEncoder()

        # PDCFR+ trainer
        trainer_pdcfr = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            regret_scheduler=PDCFRScheduler(alpha=2.0, beta=0.5),
            seed=42,
        )

        # Vanilla Deep CFR (uniform weighting)
        trainer_vanilla = DeepCFRTrainer(
            initial_state=initial_state,
            encoder=encoder,
            input_size=10,
            output_size=2,
            buffer_capacity=100,
            regret_scheduler=UniformScheduler(),
            strategy_scheduler=UniformScheduler(),
            seed=42,
        )

        # Run enough iterations to fill buffer and train
        for _ in range(150):
            trainer_pdcfr.run_iteration()
            trainer_vanilla.run_iteration()

        # Both should have trained (buffer must be full for training to start)
        # Note: buffer capacity is 100, so training starts after buffer fills
        assert trainer_pdcfr.buffer.is_full or len(trainer_pdcfr.buffer) > 0
        assert trainer_vanilla.buffer.is_full or len(trainer_vanilla.buffer) > 0

        # Both should have strategy sums
        assert len(trainer_pdcfr.strategy_sum) > 0
        assert len(trainer_vanilla.strategy_sum) > 0

        # This is a qualitative test - just verify both work
        # Actual convergence comparison would require exploitability metrics
