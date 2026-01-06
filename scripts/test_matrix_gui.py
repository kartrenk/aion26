#!/usr/bin/env python3
"""Test script for GUI strategy matrix visualization."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from aion26.gui.app import _convert_strategy_to_matrix


def test_leduc_matrix():
    """Test matrix conversion for Leduc Poker."""
    print("Testing Leduc Poker matrix conversion...")

    # Sample Leduc Round 2 strategies (private × board)
    strategy_dict = {
        # Round 1 states (should be filtered out)
        "Js": np.array([0.1, 0.5, 0.4]),
        "Qs": np.array([0.2, 0.4, 0.4]),
        "Ks": np.array([0.05, 0.3, 0.65]),

        # Round 2 states (private card + board card)
        "Js Jh": np.array([0.1, 0.2, 0.7]),  # Pair of Jacks!
        "Js Qh": np.array([0.3, 0.6, 0.1]),  # Jack with Queen board
        "Js Kh": np.array([0.4, 0.5, 0.1]),  # Jack with King board

        "Qs Jh": np.array([0.2, 0.5, 0.3]),
        "Qs Qh": np.array([0.0, 0.3, 0.7]),  # Pair of Queens!
        "Qs Kh": np.array([0.3, 0.5, 0.2]),

        "Ks Jh": np.array([0.1, 0.4, 0.5]),
        "Ks Qh": np.array([0.1, 0.4, 0.5]),
        "Ks Kh": np.array([0.0, 0.2, 0.8]),  # Pair of Kings!

        # States with betting history (should also work)
        "Js Jh p": np.array([0.05, 0.15, 0.8]),
        "Qs Qh b": np.array([0.0, 0.2, 0.8]),
    }

    matrix_data = _convert_strategy_to_matrix(strategy_dict, "leduc")

    print(f"✓ Game: {matrix_data['game']}")
    print(f"✓ Ranks: {matrix_data['ranks']}")
    print(f"✓ Matrix keys: {list(matrix_data['matrix'].keys())}")
    print()

    # Check that we have a 3×3 matrix
    assert matrix_data["game"] == "leduc"
    assert matrix_data["ranks"] == ["J", "Q", "K"]
    assert len(matrix_data["matrix"]) == 9  # 3×3 = 9 cells

    # Check specific cells
    jj_key = ("J", "J")
    if jj_key in matrix_data["matrix"]:
        jj_strategy = matrix_data["matrix"][jj_key]
        print(f"✓ J♠/J♥ strategy: Fold={jj_strategy['fold']:.2f}, Call={jj_strategy['call']:.2f}, Raise={jj_strategy['raise']:.2f}")
        # Should be aggressive with a pair
        assert jj_strategy["raise"] > 0.5, "Pair should raise more"

    qq_key = ("Q", "Q")
    if qq_key in matrix_data["matrix"]:
        qq_strategy = matrix_data["matrix"][qq_key]
        print(f"✓ Q♠/Q♥ strategy: Fold={qq_strategy['fold']:.2f}, Call={qq_strategy['call']:.2f}, Raise={qq_strategy['raise']:.2f}")

    print("\n✓ Leduc matrix test PASSED\n")


def test_kuhn_tree():
    """Test tree conversion for Kuhn Poker."""
    print("Testing Kuhn Poker tree conversion...")

    # Sample Kuhn strategies
    strategy_dict = {
        "J": np.array([0.8, 0.2]),    # Jack: mostly check
        "Q": np.array([0.6, 0.4]),    # Queen: mixed
        "K": np.array([0.3, 0.7]),    # King: mostly bet

        "J pb": np.array([0.9, 0.1]),  # Jack after opponent bet: mostly fold (check)
        "Q pb": np.array([0.5, 0.5]),  # Queen: mixed
        "K pb": np.array([0.1, 0.9]),  # King: mostly call/raise (bet)
    }

    tree_data = _convert_strategy_to_matrix(strategy_dict, "kuhn")

    print(f"✓ Game: {tree_data['game']}")
    print(f"✓ Tree keys: {list(tree_data['tree'].keys())}")
    print()

    assert tree_data["game"] == "kuhn"
    assert "J" in tree_data["tree"]
    assert "Q" in tree_data["tree"]
    assert "K" in tree_data["tree"]

    # Check that strategies are present
    if "" in tree_data["tree"]["J"]:
        j_strategy = tree_data["tree"]["J"][""]
        print(f"✓ Jack initial: Check={j_strategy['check']:.2f}, Bet={j_strategy['bet']:.2f}")
        assert j_strategy["check"] > 0.5, "Jack should check more often"

    if "" in tree_data["tree"]["K"]:
        k_strategy = tree_data["tree"]["K"][""]
        print(f"✓ King initial: Check={k_strategy['check']:.2f}, Bet={k_strategy['bet']:.2f}")
        assert k_strategy["bet"] > 0.5, "King should bet more often"

    print("\n✓ Kuhn tree test PASSED\n")


def test_empty_matrix():
    """Test matrix conversion with empty strategy."""
    print("Testing empty strategy...")

    strategy_dict = {}

    leduc_data = _convert_strategy_to_matrix(strategy_dict, "leduc")
    kuhn_data = _convert_strategy_to_matrix(strategy_dict, "kuhn")

    print(f"✓ Leduc empty: matrix={leduc_data['matrix']}")
    print(f"✓ Kuhn empty: matrix={kuhn_data['matrix']}")

    assert leduc_data["matrix"] is None
    assert kuhn_data["matrix"] is None

    print("\n✓ Empty matrix test PASSED\n")


def test_averaging():
    """Test that multiple states get averaged correctly."""
    print("Testing strategy averaging...")

    # Multiple states for same cell (should be averaged)
    strategy_dict = {
        "Js Jh": np.array([0.1, 0.2, 0.7]),
        "Jh Js": np.array([0.1, 0.2, 0.7]),
        "Js Jh p": np.array([0.2, 0.3, 0.5]),
        "Js Jh b": np.array([0.0, 0.1, 0.9]),
    }

    matrix_data = _convert_strategy_to_matrix(strategy_dict, "leduc")
    jj_key = ("J", "J")

    if jj_key in matrix_data["matrix"]:
        jj_strategy = matrix_data["matrix"][jj_key]
        print(f"✓ Averaged J♠/J♥: Fold={jj_strategy['fold']:.3f}, Call={jj_strategy['call']:.3f}, Raise={jj_strategy['raise']:.3f}")

        # Should be average of [0.1, 0.1, 0.2, 0.0] = 0.1
        expected_fold = (0.1 + 0.1 + 0.2 + 0.0) / 4
        assert abs(jj_strategy["fold"] - expected_fold) < 0.01, f"Expected {expected_fold}, got {jj_strategy['fold']}"

    print("\n✓ Averaging test PASSED\n")


if __name__ == "__main__":
    print("="*60)
    print("Strategy Matrix Conversion Tests")
    print("="*60)
    print()

    try:
        test_leduc_matrix()
        test_kuhn_tree()
        test_empty_matrix()
        test_averaging()

        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
