#!/usr/bin/env python3
"""Test script for GUI strategy heatmap visualization."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from aion26.gui.app import _convert_strategy_to_heatmap


def test_kuhn_heatmap():
    """Test heatmap conversion for Kuhn Poker."""
    print("Testing Kuhn Poker heatmap conversion...")

    # Sample Kuhn strategy (3 cards × betting rounds)
    strategy_dict = {
        "J": np.array([0.8, 0.2]),  # Jack: mostly check
        "Q": np.array([0.6, 0.4]),  # Queen: mixed
        "K": np.array([0.3, 0.7]),  # King: mostly bet
        "J pb": np.array([0.9, 0.1]),  # Jack after opponent bet
        "Q pb": np.array([0.5, 0.5]),
        "K pb": np.array([0.1, 0.9]),
    }

    heatmap_data, row_labels, col_labels = _convert_strategy_to_heatmap(strategy_dict, "kuhn")

    print(f"✓ Heatmap shape: {heatmap_data.shape}")
    print(f"✓ Row labels ({len(row_labels)}): {row_labels}")
    print(f"✓ Col labels ({len(col_labels)}): {col_labels}")
    print(f"✓ Data range: [{heatmap_data.min():.2f}, {heatmap_data.max():.2f}]")
    print()

    assert heatmap_data.shape == (6, 2), f"Expected (6, 2), got {heatmap_data.shape}"
    assert col_labels == ["Check", "Bet"], f"Expected ['Check', 'Bet'], got {col_labels}"
    print("✓ Kuhn heatmap test PASSED\n")


def test_leduc_heatmap():
    """Test heatmap conversion for Leduc Poker."""
    print("Testing Leduc Poker heatmap conversion...")

    # Sample Leduc strategy (6 cards × betting rounds × board cards)
    strategy_dict = {
        "Js": np.array([0.1, 0.5, 0.4]),  # Jack spades: fold 10%, call 50%, raise 40%
        "Qs": np.array([0.2, 0.4, 0.4]),
        "Ks": np.array([0.05, 0.3, 0.65]),  # King spades: mostly raise
        "Js Qh": np.array([0.3, 0.6, 0.1]),  # Jack spades, board Queen hearts
        "Qs Qh": np.array([0.0, 0.3, 0.7]),  # Queen pair! mostly raise
        "Ks Qh": np.array([0.1, 0.4, 0.5]),
    }

    heatmap_data, row_labels, col_labels = _convert_strategy_to_heatmap(strategy_dict, "leduc")

    print(f"✓ Heatmap shape: {heatmap_data.shape}")
    print(f"✓ Row labels ({len(row_labels)}): {row_labels}")
    print(f"✓ Col labels ({len(col_labels)}): {col_labels}")
    print(f"✓ Data range: [{heatmap_data.min():.2f}, {heatmap_data.max():.2f}]")
    print()

    assert heatmap_data.shape == (6, 3), f"Expected (6, 3), got {heatmap_data.shape}"
    assert col_labels == ["Fold", "Call", "Raise"], f"Expected ['Fold', 'Call', 'Raise'], got {col_labels}"
    print("✓ Leduc heatmap test PASSED\n")


def test_large_state_space():
    """Test heatmap conversion with large state space (sampling)."""
    print("Testing large state space sampling...")

    # Create 100 dummy states
    strategy_dict = {}
    for i in range(100):
        if i < 50:
            # Round 1 states (single card)
            strategy_dict[f"state_{i}"] = np.array([0.3, 0.4, 0.3])
        else:
            # Round 2 states (card + board)
            strategy_dict[f"state_{i} board"] = np.array([0.2, 0.5, 0.3])

    heatmap_data, row_labels, col_labels = _convert_strategy_to_heatmap(strategy_dict, "leduc")

    print(f"✓ Original states: 100")
    print(f"✓ Sampled states: {len(row_labels)}")
    print(f"✓ Heatmap shape: {heatmap_data.shape}")
    print()

    assert len(row_labels) == 50, f"Expected 50 sampled states, got {len(row_labels)}"
    print("✓ Large state space sampling test PASSED\n")


def test_empty_strategy():
    """Test heatmap conversion with empty strategy."""
    print("Testing empty strategy...")

    strategy_dict = {}
    heatmap_data, row_labels, col_labels = _convert_strategy_to_heatmap(strategy_dict, "kuhn")

    print(f"✓ Heatmap shape: {heatmap_data.shape}")
    print(f"✓ Row labels: {row_labels}")
    print(f"✓ Col labels: {col_labels}")
    print()

    assert heatmap_data.size == 0, f"Expected empty array, got size {heatmap_data.size}"
    print("✓ Empty strategy test PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("Strategy Heatmap Conversion Tests")
    print("="*60)
    print()

    try:
        test_kuhn_heatmap()
        test_leduc_heatmap()
        test_large_state_space()
        test_empty_strategy()

        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
