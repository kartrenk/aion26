"""Verification script for Deep CFR network components.

This script performs 4 stress tests to ensure network robustness:
1. Overfitting/Memorization Test - Can the network learn?
2. Feature Uniqueness Test - Are encodings distinct?
3. Gradient Flow Test - Do gradients flow properly?
4. Batch Invariance Test - Does batching work correctly?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import combinations

from aion26.games.kuhn import KuhnPoker, JACK, QUEEN, KING
from aion26.deep_cfr.networks import KuhnEncoder, DeepCFRNetwork


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} - {test_name}")
    if details:
        print(f"  {details}")


def test_overfitting_memorization():
    """Test 1: Overfitting/Memorization Test.

    Generate random batch and train network to memorize it.
    Pass condition: Loss drops to < 0.01
    """
    print_section("TEST 1: OVERFITTING/MEMORIZATION TEST")
    print("Testing if network has capacity to learn by overfitting to a small dataset...")

    # Setup
    batch_size = 10
    input_dim = 10
    output_dim = 2
    num_iterations = 100

    # Generate random data
    torch.manual_seed(42)
    random_inputs = torch.randn(batch_size, input_dim)
    random_targets = torch.randn(batch_size, output_dim)

    # Create network
    network = DeepCFRNetwork(input_size=input_dim, output_size=output_dim)
    optimizer = optim.SGD(network.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    # Train
    initial_loss = None
    final_loss = None

    print(f"\nTraining on {batch_size} random samples for {num_iterations} iterations...")

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        predictions = network(random_inputs)
        loss = criterion(predictions, random_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        if iteration == 0:
            initial_loss = loss.item()

        if (iteration + 1) % 20 == 0:
            print(f"  Iteration {iteration + 1:3d}: Loss = {loss.item():.6f}")

    final_loss = loss.item()

    # Check pass condition
    passed = final_loss < 0.01

    print(f"\nInitial Loss: {initial_loss:.6f}")
    print(f"Final Loss:   {final_loss:.6f}")
    print(f"Reduction:    {((initial_loss - final_loss) / initial_loss * 100):.1f}%")

    print_result(
        "Overfitting/Memorization Test",
        passed,
        f"Network {'CAN' if passed else 'CANNOT'} learn (final loss = {final_loss:.6f})"
    )

    return passed


def test_feature_uniqueness():
    """Test 2: Feature Uniqueness Test.

    Encode all 12 Kuhn information sets and verify they are distinct.
    Pass condition: All pairwise distances > 0
    """
    print_section("TEST 2: FEATURE UNIQUENESS TEST")
    print("Testing if all 12 Kuhn information sets have distinct encodings...")

    encoder = KuhnEncoder()

    # Manually construct all 12 information sets
    # Format: (state, player, info_set_name)
    info_sets = []

    # Player 0 initial actions (3 sets)
    for card in [JACK, QUEEN, KING]:
        state = KuhnPoker(cards=(card, (card + 1) % 3), history="")
        card_name = ["J", "Q", "K"][card]
        info_sets.append((state, 0, card_name))

    # Player 0 callback after check-bet (3 sets)
    for card in [JACK, QUEEN, KING]:
        state = KuhnPoker(cards=(card, (card + 1) % 3), history="")
        state = state.apply_action(0)  # Check
        state = state.apply_action(1)  # Bet (from P1)
        card_name = ["J", "Q", "K"][card]
        info_sets.append((state, 0, f"{card_name}cb"))

    # Player 1 after check (3 sets)
    for card in [JACK, QUEEN, KING]:
        state = KuhnPoker(cards=((card + 1) % 3, card), history="")
        state = state.apply_action(0)  # P0 checks
        card_name = ["J", "Q", "K"][card]
        info_sets.append((state, 1, f"{card_name}c"))

    # Player 1 after bet (3 sets)
    for card in [JACK, QUEEN, KING]:
        state = KuhnPoker(cards=((card + 1) % 3, card), history="")
        state = state.apply_action(1)  # P0 bets
        card_name = ["J", "Q", "K"][card]
        info_sets.append((state, 1, f"{card_name}b"))

    # Encode all information sets
    print(f"\nEncoding {len(info_sets)} information sets...")
    encodings = []
    names = []

    for state, player, name in info_sets:
        encoding = encoder.encode(state, player)
        encodings.append(encoding)
        names.append(name)
        print(f"  {name:6s}: {encoding.numpy()}")

    # Compute pairwise distances
    print(f"\nComputing pairwise distances...")
    encodings_tensor = torch.stack(encodings)

    min_distance = float('inf')
    min_pair = None
    zero_distance_pairs = []

    for i, j in combinations(range(len(encodings)), 2):
        distance = torch.dist(encodings_tensor[i], encodings_tensor[j]).item()

        if distance < min_distance:
            min_distance = distance
            min_pair = (names[i], names[j])

        if distance == 0.0:
            zero_distance_pairs.append((names[i], names[j], distance))

    # Check pass condition
    passed = len(zero_distance_pairs) == 0 and min_distance > 0

    print(f"\nMinimum pairwise distance: {min_distance:.6f}")
    print(f"  Between: {min_pair[0]} and {min_pair[1]}")

    if zero_distance_pairs:
        print(f"\n⚠️  Found {len(zero_distance_pairs)} pairs with zero distance:")
        for name1, name2, dist in zero_distance_pairs:
            print(f"    {name1} ≡ {name2} (distance = {dist})")
    else:
        print(f"\n✓ All {len(info_sets)} information sets have unique encodings")

    # Compute statistics
    all_distances = []
    for i, j in combinations(range(len(encodings)), 2):
        distance = torch.dist(encodings_tensor[i], encodings_tensor[j]).item()
        all_distances.append(distance)

    print(f"\nDistance statistics:")
    print(f"  Min:    {min(all_distances):.6f}")
    print(f"  Max:    {max(all_distances):.6f}")
    print(f"  Mean:   {np.mean(all_distances):.6f}")
    print(f"  Median: {np.median(all_distances):.6f}")

    print_result(
        "Feature Uniqueness Test",
        passed,
        f"All encodings are {'distinct' if passed else 'NOT distinct'}"
    )

    return passed


def test_gradient_flow():
    """Test 3: Gradient Flow Test.

    Check that gradients flow properly through the network.
    Pass condition: No gradients are zero or NaN
    """
    print_section("TEST 3: GRADIENT FLOW TEST")
    print("Testing gradient flow through all network parameters...")

    # Setup
    torch.manual_seed(42)
    network = DeepCFRNetwork(input_size=10, output_size=2)

    # Create input and target
    x = torch.randn(4, 10, requires_grad=True)
    target = torch.randn(4, 2)

    # Forward pass
    output = network(x)
    loss = nn.MSELoss()(output, target)

    print(f"\nForward pass:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Loss:         {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients
    print(f"\nChecking gradients for all {len(list(network.parameters()))} parameter tensors...")

    zero_grad_params = []
    nan_grad_params = []
    all_grads_ok = True

    for i, (name, param) in enumerate(network.named_parameters()):
        if param.grad is None:
            print(f"  ⚠️  Parameter {i} ({name}): No gradient!")
            all_grads_ok = False
            continue

        has_zero = (param.grad == 0.0).all().item()
        has_nan = torch.isnan(param.grad).any().item()
        grad_norm = param.grad.norm().item()

        status = "✓"
        if has_zero:
            zero_grad_params.append(name)
            status = "✗ ZERO"
            all_grads_ok = False
        elif has_nan:
            nan_grad_params.append(name)
            status = "✗ NaN"
            all_grads_ok = False

        print(f"  {status} Parameter {i:2d} ({name:30s}): "
              f"shape={list(param.shape)}, grad_norm={grad_norm:.6f}")

    # Check input gradient
    if x.grad is not None:
        input_grad_norm = x.grad.norm().item()
        print(f"\n  Input gradient norm: {input_grad_norm:.6f}")

    # Check pass condition
    passed = all_grads_ok and len(zero_grad_params) == 0 and len(nan_grad_params) == 0

    if zero_grad_params:
        print(f"\n⚠️  Found {len(zero_grad_params)} parameters with all-zero gradients:")
        for name in zero_grad_params:
            print(f"    - {name}")

    if nan_grad_params:
        print(f"\n⚠️  Found {len(nan_grad_params)} parameters with NaN gradients:")
        for name in nan_grad_params:
            print(f"    - {name}")

    if passed:
        print(f"\n✓ All gradients flow properly (no zeros, no NaNs)")

    print_result(
        "Gradient Flow Test",
        passed,
        f"Gradients {'flow properly' if passed else 'have issues'}"
    )

    return passed


def test_batch_invariance():
    """Test 4: Batch Invariance Test.

    Verify that batch processing gives same results as individual processing.
    Pass condition: Outputs match within 1e-6
    """
    print_section("TEST 4: BATCH INVARIANCE TEST")
    print("Testing if batched inference matches individual inference...")

    # Setup
    torch.manual_seed(42)
    encoder = KuhnEncoder()
    network = DeepCFRNetwork(input_size=encoder.feature_size(), output_size=2)
    network.eval()  # Set to eval mode (in case we add BatchNorm later)

    # Create 3 different states
    states = [
        (KuhnPoker(cards=(JACK, QUEEN), history=""), 0),
        (KuhnPoker(cards=(QUEEN, KING), history="").apply_action(0), 1),
        (KuhnPoker(cards=(KING, JACK), history="").apply_action(1), 1),
    ]

    print(f"\nEncoding {len(states)} different states...")

    # Encode individually
    individual_outputs = []
    for i, (state, player) in enumerate(states):
        features = encoder.encode(state, player).unsqueeze(0)  # Add batch dim
        output = network(features)
        individual_outputs.append(output.squeeze(0))  # Remove batch dim
        print(f"  State {i}: {state.information_state_string()} "
              f"→ output shape {output.shape}")

    # Encode as batch
    batch_features = torch.stack([
        encoder.encode(state, player) for state, player in states
    ])
    batch_output = network(batch_features)

    print(f"\nBatch encoding:")
    print(f"  Input shape:  {batch_features.shape}")
    print(f"  Output shape: {batch_output.shape}")

    # Compare outputs
    print(f"\nComparing outputs:")
    max_diff = 0.0
    all_match = True

    for i in range(len(states)):
        individual = individual_outputs[i]
        batched = batch_output[i]

        diff = torch.abs(individual - batched).max().item()
        max_diff = max(max_diff, diff)

        matches = diff < 1e-6
        all_match = all_match and matches

        status = "✓" if matches else "✗"
        print(f"  {status} State {i}: max_diff = {diff:.10f} "
              f"({'MATCH' if matches else 'MISMATCH'})")

        if not matches:
            print(f"      Individual: {individual.detach().numpy()}")
            print(f"      Batched:    {batched.detach().numpy()}")

    # Check pass condition
    passed = all_match and max_diff < 1e-6

    print(f"\nMaximum difference across all outputs: {max_diff:.10e}")

    print_result(
        "Batch Invariance Test",
        passed,
        f"Batch processing {'matches' if passed else 'DOES NOT match'} individual processing"
    )

    return passed


def main():
    """Run all verification tests."""
    print("\n" + "=" * 80)
    print("  DEEP CFR NETWORK VERIFICATION")
    print("  Testing network robustness beyond standard unit tests")
    print("=" * 80)

    results = {}

    # Run all tests
    results['overfitting'] = test_overfitting_memorization()
    results['uniqueness'] = test_feature_uniqueness()
    results['gradient_flow'] = test_gradient_flow()
    results['batch_invariance'] = test_batch_invariance()

    # Summary
    print_section("VERIFICATION SUMMARY")

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"\nResults:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")

    print(f"\n{'=' * 80}")
    print(f"  TOTAL: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print(f"  🎉 ALL VERIFICATION TESTS PASSED!")
        print(f"  Network is robust and ready for Deep CFR training.")
    else:
        print(f"  ⚠️  Some tests failed. Review issues above.")
        failed = [name for name, passed in results.items() if not passed]
        print(f"  Failed tests: {', '.join(failed)}")

    print(f"{'=' * 80}\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
