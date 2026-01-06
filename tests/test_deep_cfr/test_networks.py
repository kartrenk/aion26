"""Tests for Deep CFR neural network components."""

import pytest
import torch
import numpy as np

from aion26.games.kuhn import KuhnPoker, JACK, QUEEN, KING
from aion26.deep_cfr.networks import CardEmbedding, KuhnEncoder, DeepCFRNetwork


class TestCardEmbedding:
    """Tests for CardEmbedding utility."""

    def test_encode_jack(self):
        """Test encoding of Jack card."""
        one_hot = CardEmbedding.encode(JACK)
        assert one_hot.shape == (3,)
        assert one_hot.dtype == np.float32
        np.testing.assert_array_equal(one_hot, [1.0, 0.0, 0.0])

    def test_encode_queen(self):
        """Test encoding of Queen card."""
        one_hot = CardEmbedding.encode(QUEEN)
        assert one_hot.shape == (3,)
        np.testing.assert_array_equal(one_hot, [0.0, 1.0, 0.0])

    def test_encode_king(self):
        """Test encoding of King card."""
        one_hot = CardEmbedding.encode(KING)
        assert one_hot.shape == (3,)
        np.testing.assert_array_equal(one_hot, [0.0, 0.0, 1.0])

    def test_encode_invalid_card(self):
        """Test that invalid card raises ValueError."""
        with pytest.raises(ValueError, match="Invalid card"):
            CardEmbedding.encode(3)

        with pytest.raises(ValueError, match="Invalid card"):
            CardEmbedding.encode(-1)

    def test_to_tensor_jack(self):
        """Test tensor conversion for Jack."""
        tensor = CardEmbedding.to_tensor(JACK)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3,)
        assert tensor.dtype == torch.float32
        torch.testing.assert_close(tensor, torch.tensor([1.0, 0.0, 0.0]))

    def test_to_tensor_queen(self):
        """Test tensor conversion for Queen."""
        tensor = CardEmbedding.to_tensor(QUEEN)
        torch.testing.assert_close(tensor, torch.tensor([0.0, 1.0, 0.0]))

    def test_to_tensor_king(self):
        """Test tensor conversion for King."""
        tensor = CardEmbedding.to_tensor(KING)
        torch.testing.assert_close(tensor, torch.tensor([0.0, 0.0, 1.0]))


class TestKuhnEncoder:
    """Tests for KuhnEncoder state encoder."""

    def test_encoder_initialization(self):
        """Test encoder initializes with correct max pot."""
        encoder = KuhnEncoder(max_pot=5.0)
        assert encoder.max_pot == 5.0
        assert encoder.feature_size() == 10

    def test_encoder_default_max_pot(self):
        """Test encoder uses default max pot of 5.0."""
        encoder = KuhnEncoder()
        assert encoder.max_pot == 5.0

    def test_encode_initial_state_jack(self):
        """Test encoding initial state with Jack (history='')."""
        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        encoder = KuhnEncoder()
        features = encoder.encode(state, player=0)

        # Check shape and type
        assert features.shape == (10,)
        assert features.dtype == torch.float32

        # Expected features:
        # Card (J): [1, 0, 0]
        # History (empty): [0, 0, 0, 0, 0, 0]
        # Pot (2/5): [0.4]
        expected = torch.tensor([
            1.0, 0.0, 0.0,  # Jack
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # No history
            2.0 / 5.0  # Pot = 2 (antes)
        ])
        torch.testing.assert_close(features, expected)

    def test_encode_state_with_history_jb(self):
        """Test encoding state with Jack and history 'b' (bet)."""
        # Create state by applying actions to get correct pot
        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        state = state.apply_action(1)  # Player 0 bets -> history="b", pot=3

        encoder = KuhnEncoder()
        features = encoder.encode(state, player=1)  # Player 1's turn

        # Expected features:
        # Card (Q): [0, 1, 0]
        # History ('b'): [0, 1, 0, 0, 0, 0]
        # Pot (3/5): [0.6]
        expected = torch.tensor([
            0.0, 1.0, 0.0,  # Queen
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,  # Bet
            3.0 / 5.0  # Pot = 3 (2 ante + 1 bet)
        ])
        torch.testing.assert_close(features, expected)

    def test_encode_state_with_history_cb(self):
        """Test encoding state with history 'cb' (check, bet)."""
        # Create state by applying actions to get correct pot
        state = KuhnPoker(cards=(KING, JACK), history="")
        state = state.apply_action(0)  # Player 0 checks -> history="c"
        state = state.apply_action(1)  # Player 1 bets -> history="cb", pot=3

        encoder = KuhnEncoder()
        features = encoder.encode(state, player=0)  # Player 0 facing bet

        # Expected features:
        # Card (K): [0, 0, 1]
        # History ('cb'): [1, 0, 0, 1, 0, 0]
        # Pot (3/5): [0.6]
        expected = torch.tensor([
            0.0, 0.0, 1.0,  # King
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0,  # Check, Bet
            3.0 / 5.0  # Pot = 3 (2 ante + 1 bet)
        ])
        torch.testing.assert_close(features, expected)

    def test_encode_state_with_history_cbb(self):
        """Test encoding terminal state with history 'cbb' (check, bet, call)."""
        # Note: This is a terminal state, but we test encoding before terminal check
        state = KuhnPoker(cards=(QUEEN, KING), history="cbb")
        encoder = KuhnEncoder()

        # This should raise an error because it's terminal
        with pytest.raises(ValueError, match="Cannot encode terminal"):
            encoder.encode(state)

    def test_encode_uses_current_player_by_default(self):
        """Test that encode uses current_player() when player not specified."""
        state = KuhnPoker(cards=(JACK, QUEEN), history="c")
        encoder = KuhnEncoder()

        # Player 1's turn (after player 0 checked)
        features = encoder.encode(state)  # No player specified

        # Should encode from player 1's perspective (Queen)
        assert features[0] == 0.0  # Not Jack
        assert features[1] == 1.0  # Is Queen
        assert features[2] == 0.0  # Not King

    def test_encode_chance_node_raises_error(self):
        """Test that encoding chance node raises ValueError."""
        state = KuhnPoker()  # Initial state (chance node)
        encoder = KuhnEncoder()

        with pytest.raises(ValueError, match="Cannot encode terminal or chance"):
            encoder.encode(state)

    def test_encode_terminal_state_raises_error(self):
        """Test that encoding terminal state raises ValueError."""
        state = KuhnPoker(cards=(JACK, QUEEN), history="cc")  # Terminal
        encoder = KuhnEncoder()

        with pytest.raises(ValueError, match="Cannot encode terminal"):
            encoder.encode(state)

    def test_encode_different_pot_sizes(self):
        """Test encoding captures different pot sizes correctly."""
        encoder = KuhnEncoder()

        # Initial state: pot = 2
        state1 = KuhnPoker(cards=(JACK, QUEEN), history="")
        features1 = encoder.encode(state1, player=0)
        assert features1[-1] == 2.0 / 5.0  # 0.4

        # After bet: pot = 3
        state2 = KuhnPoker(cards=(JACK, QUEEN), history="")
        state2 = state2.apply_action(1)  # Bet
        features2 = encoder.encode(state2, player=1)
        assert features2[-1] == 3.0 / 5.0  # 0.6

        # After check, bet: pot = 3
        state3 = KuhnPoker(cards=(JACK, QUEEN), history="")
        state3 = state3.apply_action(0)  # Check
        state3 = state3.apply_action(1)  # Bet
        features3 = encoder.encode(state3, player=0)
        assert features3[-1] == 3.0 / 5.0  # 0.6

    def test_feature_size(self):
        """Test feature_size() returns correct dimension."""
        encoder = KuhnEncoder()
        assert encoder.feature_size() == 10


class TestDeepCFRNetwork:
    """Tests for DeepCFRNetwork MLP."""

    def test_network_initialization(self):
        """Test network initializes with correct architecture."""
        network = DeepCFRNetwork(input_size=10, output_size=2)

        assert network.input_size == 10
        assert network.output_size == 2
        assert network.hidden_size == 64
        assert network.num_hidden_layers == 3

    def test_network_custom_hidden_size(self):
        """Test network with custom hidden layer size."""
        network = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            hidden_size=128,
            num_hidden_layers=2
        )

        assert network.hidden_size == 128
        assert network.num_hidden_layers == 2

    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        network = DeepCFRNetwork(input_size=10, output_size=2)

        # Create random input
        x = torch.randn(1, 10)
        output = network(x)

        # Check output shape
        assert output.shape == (1, 2)
        assert output.dtype == torch.float32

    def test_forward_pass_batch(self):
        """Test forward pass with batch of samples."""
        network = DeepCFRNetwork(input_size=10, output_size=2)

        # Batch of 32 samples
        x = torch.randn(32, 10)
        output = network(x)

        # Check output shape
        assert output.shape == (32, 2)

    def test_forward_pass_with_real_state(self):
        """Test forward pass with real encoded Kuhn state."""
        encoder = KuhnEncoder()
        network = DeepCFRNetwork(
            input_size=encoder.feature_size(),
            output_size=2
        )

        # Encode a real state
        state = KuhnPoker(cards=(JACK, QUEEN), history="")
        features = encoder.encode(state, player=0)

        # Add batch dimension
        features_batch = features.unsqueeze(0)  # Shape: (1, 10)

        # Forward pass
        regrets = network(features_batch)

        # Check output
        assert regrets.shape == (1, 2)
        assert not torch.isnan(regrets).any()
        assert not torch.isinf(regrets).any()

    def test_forward_pass_deterministic_with_seed(self):
        """Test that forward pass is deterministic with same seed."""
        torch.manual_seed(42)
        network1 = DeepCFRNetwork(input_size=10, output_size=2)

        torch.manual_seed(42)
        network2 = DeepCFRNetwork(input_size=10, output_size=2)

        # Same input
        x = torch.randn(1, 10)

        output1 = network1(x)
        output2 = network2(x)

        torch.testing.assert_close(output1, output2)

    def test_network_has_correct_number_of_layers(self):
        """Test network has correct number of layers."""
        network = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            hidden_size=64,
            num_hidden_layers=3
        )

        # Count Linear layers
        linear_layers = [m for m in network.modules() if isinstance(m, torch.nn.Linear)]

        # Should have: 1 input + 3 hidden + 1 output = 4 Linear layers
        # (input->hidden counts as first hidden layer)
        assert len(linear_layers) == 4

    def test_network_repr(self):
        """Test string representation of network."""
        network = DeepCFRNetwork(input_size=10, output_size=2)
        repr_str = repr(network)

        assert "DeepCFRNetwork" in repr_str
        assert "input=10" in repr_str
        assert "hidden=64x3" in repr_str
        assert "output=2" in repr_str

    def test_network_different_output_sizes(self):
        """Test network works with different output sizes."""
        # Kuhn has 2 actions
        network_kuhn = DeepCFRNetwork(input_size=10, output_size=2)
        x = torch.randn(1, 10)
        output = network_kuhn(x)
        assert output.shape == (1, 2)

        # Leduc might have 3 actions (fold, call, raise)
        network_leduc = DeepCFRNetwork(input_size=20, output_size=3)
        x = torch.randn(1, 20)
        output = network_leduc(x)
        assert output.shape == (1, 3)

    def test_network_gradients_flow(self):
        """Test that gradients flow through the network."""
        network = DeepCFRNetwork(input_size=10, output_size=2)

        x = torch.randn(1, 10, requires_grad=True)
        output = network(x)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that network parameters have gradients
        for param in network.parameters():
            assert param.grad is not None


class TestIntegration:
    """Integration tests combining encoder and network."""

    def test_encoder_network_integration(self):
        """Test full pipeline: state -> encoder -> network."""
        # Setup
        encoder = KuhnEncoder()
        network = DeepCFRNetwork(
            input_size=encoder.feature_size(),
            output_size=2
        )

        # Test all 12 information sets
        test_states = [
            # Player 0 initial actions
            (KuhnPoker(cards=(JACK, QUEEN), history=""), 0),
            (KuhnPoker(cards=(QUEEN, JACK), history=""), 0),
            (KuhnPoker(cards=(KING, JACK), history=""), 0),

            # Player 1 after check
            (KuhnPoker(cards=(JACK, QUEEN), history="c"), 1),
            (KuhnPoker(cards=(QUEEN, JACK), history="c"), 1),
            (KuhnPoker(cards=(KING, QUEEN), history="c"), 1),

            # Player 1 after bet
            (KuhnPoker(cards=(JACK, QUEEN), history="b"), 1),
            (KuhnPoker(cards=(QUEEN, JACK), history="b"), 1),
            (KuhnPoker(cards=(KING, JACK), history="b"), 1),

            # Player 0 callback
            (KuhnPoker(cards=(JACK, QUEEN), history="cb"), 0),
            (KuhnPoker(cards=(QUEEN, KING), history="cb"), 0),
            (KuhnPoker(cards=(KING, JACK), history="cb"), 0),
        ]

        for state, player in test_states:
            # Encode
            features = encoder.encode(state, player=player)
            assert features.shape == (10,)

            # Forward pass
            features_batch = features.unsqueeze(0)
            regrets = network(features_batch)

            # Verify output
            assert regrets.shape == (1, 2)
            assert not torch.isnan(regrets).any()
            assert not torch.isinf(regrets).any()

    def test_batch_encoding_and_prediction(self):
        """Test batched encoding and network prediction."""
        encoder = KuhnEncoder()
        network = DeepCFRNetwork(input_size=encoder.feature_size(), output_size=2)

        # Create batch of states
        states = [
            (KuhnPoker(cards=(JACK, QUEEN), history=""), 0),
            (KuhnPoker(cards=(QUEEN, JACK), history="c"), 1),
            (KuhnPoker(cards=(KING, JACK), history="b"), 1),
        ]

        # Encode all states
        features_list = [encoder.encode(state, player) for state, player in states]
        features_batch = torch.stack(features_list)  # Shape: (3, 10)

        # Forward pass
        regrets = network(features_batch)

        # Verify output
        assert regrets.shape == (3, 2)
        assert not torch.isnan(regrets).any()

    def test_network_output_different_for_different_states(self):
        """Test that network produces different outputs for different states."""
        torch.manual_seed(42)
        encoder = KuhnEncoder()
        network = DeepCFRNetwork(input_size=encoder.feature_size(), output_size=2)

        # Two different states
        state1 = KuhnPoker(cards=(JACK, QUEEN), history="")
        state2 = KuhnPoker(cards=(KING, JACK), history="b")

        features1 = encoder.encode(state1, player=0).unsqueeze(0)
        features2 = encoder.encode(state2, player=1).unsqueeze(0)

        output1 = network(features1)
        output2 = network(features2)

        # Outputs should be different (with high probability)
        assert not torch.allclose(output1, output2, atol=1e-6)


class TestPDCFRConformity:
    """Tests for PDCFR+ specific requirements.

    These tests ensure the network supports the critical features needed
    for Deep PDCFR+:
    1. Zero-initialized output layer for uniform exploration
    2. Target network support with Polyak averaging
    """

    def test_zero_init_output_produces_near_zero_values(self):
        """Test that zero-initialized network outputs near-zero values.

        This is critical for PDCFR+ to ensure near-uniform exploration
        at the start of training, avoiding premature convergence.
        """
        torch.manual_seed(42)
        network = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            zero_init_output=True
        )

        # Create random inputs
        x = torch.randn(10, 10)  # Batch of 10 samples

        # Forward pass
        output = network(x)

        # All outputs should be very close to zero
        max_abs_value = output.abs().max().item()

        print(f"\n  Zero-init network output statistics:")
        print(f"    Max absolute value: {max_abs_value:.6f}")
        print(f"    Mean: {output.mean().item():.6f}")
        print(f"    Std:  {output.std().item():.6f}")

        # Pass condition: all values < 0.01
        assert max_abs_value < 0.01, (
            f"Zero-initialized network produced values > 0.01 "
            f"(max = {max_abs_value})"
        )

    def test_non_zero_init_produces_larger_values(self):
        """Test that non-zero initialization produces larger values.

        This verifies that zero_init_output flag actually has an effect.
        """
        torch.manual_seed(42)

        # Network with zero init
        net_zero = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            zero_init_output=True
        )

        # Network without zero init (default PyTorch initialization)
        net_normal = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            zero_init_output=False
        )

        # Same input
        x = torch.randn(10, 10)

        output_zero = net_zero(x)
        output_normal = net_normal(x)

        max_zero = output_zero.abs().max().item()
        max_normal = output_normal.abs().max().item()

        print(f"\n  Comparison:")
        print(f"    Zero-init max:   {max_zero:.6f}")
        print(f"    Normal-init max: {max_normal:.6f}")

        # Normal init should produce much larger values
        assert max_normal > 10 * max_zero, (
            "Normal initialization should produce larger values than zero init"
        )

    def test_copy_weights_hard_copy(self):
        """Test hard copy (polyak=1.0) for target network.

        Hard copy replaces all target weights with source weights.
        """
        torch.manual_seed(42)

        # Create source and target networks
        source = DeepCFRNetwork(input_size=10, output_size=2)
        target = DeepCFRNetwork(input_size=10, output_size=2)

        # Verify they start with different weights
        x = torch.randn(5, 10)
        output_source_before = source(x)
        output_target_before = target(x)

        assert not torch.allclose(output_source_before, output_target_before, atol=1e-6)

        # Hard copy: target <- source
        target.copy_weights_from(source, polyak=1.0)

        # Now outputs should be identical
        output_source_after = source(x)
        output_target_after = target(x)

        torch.testing.assert_close(output_target_after, output_source_after)

        print("\n  Hard copy verification:")
        print(f"    Before: max diff = {(output_target_before - output_source_before).abs().max():.6f}")
        print(f"    After:  max diff = {(output_target_after - output_source_after).abs().max():.6e}")

    def test_copy_weights_soft_update(self):
        """Test soft update (polyak<1.0) for target network.

        Soft update uses exponential moving average:
        target = polyak * source + (1-polyak) * target
        """
        torch.manual_seed(42)

        # Create networks with different random initializations
        source = DeepCFRNetwork(input_size=10, output_size=2, zero_init_output=False)

        torch.manual_seed(123)  # Different seed for target
        target = DeepCFRNetwork(input_size=10, output_size=2, zero_init_output=False)

        # Save original target weights
        target_weights_before = [p.clone() for p in target.parameters()]

        # Soft update with polyak=0.1
        polyak = 0.1
        target.copy_weights_from(source, polyak=polyak)

        # Verify soft update formula: target = 0.1 * source + 0.9 * target_old
        for i, (target_param, source_param, old_target) in enumerate(
            zip(target.parameters(), source.parameters(), target_weights_before)
        ):
            expected = polyak * source_param + (1 - polyak) * old_target
            torch.testing.assert_close(target_param, expected, rtol=1e-5, atol=1e-7)

        print(f"\n  Soft update (polyak={polyak}) verification:")
        print(f"    Formula verified: target = {polyak} * source + {1-polyak} * target_old")
        print(f"    All parameters match expected values ✓")

    def test_copy_weights_multiple_soft_updates(self):
        """Test multiple soft updates converge target to source.

        With repeated soft updates, target should gradually approach source.
        """
        torch.manual_seed(42)

        source = DeepCFRNetwork(input_size=10, output_size=2)
        target = DeepCFRNetwork(input_size=10, output_size=2)

        x = torch.randn(5, 10)

        # Initial difference
        initial_diff = (target(x) - source(x)).abs().max().item()

        # Apply multiple soft updates
        polyak = 0.3
        num_updates = 20

        print(f"\n  Multiple soft updates (polyak={polyak}, updates={num_updates}):")
        print(f"    Initial diff: {initial_diff:.6f}")

        for i in range(num_updates):
            target.copy_weights_from(source, polyak=polyak)

            if (i + 1) % 5 == 0:
                diff = (target(x) - source(x)).abs().max().item()
                print(f"    After {i+1:2d} updates: diff = {diff:.6f}")

        # Final difference should be much smaller
        final_diff = (target(x) - source(x)).abs().max().item()

        assert final_diff < initial_diff * 0.01, (
            f"Target should converge to source after {num_updates} updates"
        )

    def test_copy_weights_incompatible_networks_raises_error(self):
        """Test that copying between incompatible networks raises error."""
        source = DeepCFRNetwork(input_size=10, output_size=2)
        target = DeepCFRNetwork(input_size=20, output_size=2)  # Different input size

        with pytest.raises(ValueError, match="incompatible networks"):
            target.copy_weights_from(source, polyak=1.0)

    def test_copy_weights_invalid_polyak_raises_error(self):
        """Test that invalid polyak coefficient raises error."""
        source = DeepCFRNetwork(input_size=10, output_size=2)
        target = DeepCFRNetwork(input_size=10, output_size=2)

        # Polyak must be in [0, 1]
        with pytest.raises(ValueError, match="Polyak must be in"):
            target.copy_weights_from(source, polyak=1.5)

        with pytest.raises(ValueError, match="Polyak must be in"):
            target.copy_weights_from(source, polyak=-0.1)

    def test_copy_weights_polyak_zero_no_change(self):
        """Test that polyak=0 doesn't change target network."""
        torch.manual_seed(42)

        source = DeepCFRNetwork(input_size=10, output_size=2)
        target = DeepCFRNetwork(input_size=10, output_size=2)

        # Save target weights
        target_weights_before = [p.clone() for p in target.parameters()]

        # Copy with polyak=0 (should not change target)
        target.copy_weights_from(source, polyak=0.0)

        # Verify target unchanged
        for param_before, param_after in zip(target_weights_before, target.parameters()):
            torch.testing.assert_close(param_after, param_before)

    def test_pdcfr_ready_integration(self):
        """Integration test: Verify network is ready for PDCFR+ training.

        This test combines all PDCFR+ requirements:
        1. Zero-initialized output
        2. Near-uniform initial predictions
        3. Target network support
        """
        torch.manual_seed(42)

        # Create advantage network (for regrets)
        advantage_net = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            zero_init_output=True
        )

        # Create target network (for bootstrap)
        target_net = DeepCFRNetwork(
            input_size=10,
            output_size=2,
            zero_init_output=True
        )

        # Initialize target from advantage (hard copy)
        target_net.copy_weights_from(advantage_net, polyak=1.0)

        # Test on random states
        states = torch.randn(100, 10)

        # Both networks should produce near-zero regrets initially
        regrets = advantage_net(states)
        target_regrets = target_net(states)

        assert regrets.abs().max() < 0.01
        assert target_regrets.abs().max() < 0.01
        torch.testing.assert_close(regrets, target_regrets)

        print("\n  PDCFR+ Integration Test:")
        print(f"    ✓ Zero-initialized networks")
        print(f"    ✓ Near-uniform initial regrets (max = {regrets.abs().max():.6f})")
        print(f"    ✓ Target network initialized")
        print(f"    ✓ Ready for PDCFR+ bootstrap training!")
