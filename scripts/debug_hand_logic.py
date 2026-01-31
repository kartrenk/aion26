#!/usr/bin/env python3
"""
DEBUG SCRIPT: Hand Evaluation Logic Test
=========================================
Tests the Rust game engine to identify why a Broadway Straight is folding.

Test Case:
  Player Hand: 9h, Kc
  Board: Jh, Ah, Tc, Ts, Qd
  Expected: A-K-Q-J-T Broadway Straight (Rank 5 = Straight)
  Bug: Strategy Inspector shows 100% Fold

This script isolates whether the bug is in:
1. Hand evaluation (rank calculation)
2. Payoff/reward calculation (inverted signs)
3. Legal actions (masking issue)
4. State encoding (wrong features)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Windows DLL fix
if sys.platform == 'win32':
    import ctypes
    import importlib.util
    site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "aion26_rust"
    pyd_file = site_packages / "aion26_rust.cp312-win_amd64.pyd"
    if pyd_file.exists():
        try:
            ctypes.CDLL(str(pyd_file))
            spec = importlib.util.spec_from_file_location('aion26_rust', str(pyd_file))
            _rust_module = importlib.util.module_from_spec(spec)
            sys.modules['aion26_rust'] = _rust_module
            spec.loader.exec_module(_rust_module)
            print(f"[OK] Loaded Rust module: {pyd_file}")
        except Exception as e:
            print(f"[ERROR] Could not load Rust module: {e}")
            sys.exit(1)

import numpy as np

try:
    from aion26_rust import RustRiverHoldem, ParallelTrainer
    print("[OK] Imported RustRiverHoldem and ParallelTrainer")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Also import treys for independent hand evaluation
try:
    from treys import Card, Evaluator
    TREYS_AVAILABLE = True
    print("[OK] Treys evaluator available for comparison")
except ImportError:
    TREYS_AVAILABLE = False
    print("[WARN] Treys not available - skipping comparison")


def card_to_index(card_str: str) -> int:
    """Convert card string like '9h' to 0-51 index."""
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}

    rank = card_str[0].upper()
    suit = card_str[1].lower()

    return rank_map[rank] * 4 + suit_map[suit]


def test_hand_evaluation():
    """Test the specific hand that's showing 100% fold."""
    print("\n" + "="*60)
    print("TEST CASE: Broadway Straight (should NEVER fold)")
    print("="*60)

    # The problematic hand from Strategy Inspector
    player_hand = ['9h', 'Kc']
    board = ['Jh', 'Ah', 'Tc', 'Ts', 'Qd']

    print(f"\nPlayer Hand: {player_hand}")
    print(f"Board: {board}")
    print(f"Best 5 cards: A-K-Q-J-T (Broadway Straight)")

    # Convert to indices
    hand_indices = [card_to_index(c) for c in player_hand]
    board_indices = [card_to_index(c) for c in board]

    print(f"\nCard indices:")
    print(f"  Hand: {hand_indices} ({player_hand})")
    print(f"  Board: {board_indices} ({board})")

    # Test with Treys for ground truth
    if TREYS_AVAILABLE:
        evaluator = Evaluator()
        treys_hand = [Card.new(c) for c in player_hand]
        treys_board = [Card.new(c) for c in board]
        treys_rank = evaluator.evaluate(treys_board, treys_hand)
        treys_class = evaluator.get_rank_class(treys_rank)
        treys_class_str = evaluator.class_to_string(treys_class)

        print(f"\n[TREYS GROUND TRUTH]")
        print(f"  Rank score: {treys_rank} (lower is better)")
        print(f"  Hand class: {treys_class} ({treys_class_str})")
        print(f"  Is Straight: {treys_class == 5}")  # 5 = Straight in treys


def test_rust_game_state():
    """Test the Rust game engine directly."""
    print("\n" + "="*60)
    print("TEST: RustRiverHoldem Game State (Properly Initialized)")
    print("="*60)

    try:
        # Create a PROPERLY INITIALIZED game (like train_webapp_pro.py does)
        game = RustRiverHoldem(
            stacks=[100.0, 100.0],
            pot=2.0,
            current_bet=0.0,
            player_0_invested=1.0,
            player_1_invested=1.0,
        )
        print(f"\n[OK] Created RustRiverHoldem with proper init")

        # Check available methods
        print(f"\nAvailable methods:")
        methods = [m for m in dir(game) if not m.startswith('_')]
        for m in methods:
            print(f"  - {m}")

        # Deal cards (action 0 is the chance action)
        print(f"\n[TEST] Dealing cards (action 0)...")
        game = game.apply_action(0)
        print(f"[OK] Cards dealt")

        # Check dealt cards
        print(f"\n[TEST] Checking dealt cards...")
        hands = game.hands
        board = game.board
        print(f"  Hands raw: {hands}")
        print(f"  Board raw: {board}")

        # Decode cards to understand format
        print(f"\n[CARD FORMAT ANALYSIS]")
        if len(hands) > 0 and len(hands[0]) > 0:
            card = hands[0][0]
            print(f"  First card value: {card}")
            print(f"  If format is rank*4+suit: rank={card//4}, suit={card%4}")
            print(f"  If format is suit*13+rank: rank={card%13}, suit={card//13}")

        # Check legal actions AFTER dealing
        print(f"\n[TEST] Legal actions after deal...")
        legal = game.legal_actions()
        print(f"  Legal actions: {legal}")
        action_names = ['Fold', 'Call', 'Raise', 'All-in']
        if hasattr(legal, '__len__'):
            print(f"  Count: {len(legal)} actions")
            if len(legal) == 4:
                for i, (name, val) in enumerate(zip(action_names, legal)):
                    print(f"    {i}={name}: {'LEGAL' if val else 'ILLEGAL'}")
            else:
                # legal might be list of action indices
                print(f"  Legal action indices: {legal}")

        # Current player
        player = game.current_player()
        print(f"\n[TEST] Current player: {player}")

        # Hand strength (this is key!)
        print(f"\n[TEST] Hand strength...")
        try:
            strength = game.get_hand_strength(0)  # Player 0
            print(f"  Player 0 hand strength: {strength}")
            strength1 = game.get_hand_strength(1)  # Player 1
            print(f"  Player 1 hand strength: {strength1}")
        except Exception as e:
            print(f"  [ERROR] get_hand_strength failed: {e}")

        # Pot and stacks
        print(f"\n[TEST] Game state values...")
        print(f"  Pot: {game.pot}")
        print(f"  Stacks: {game.stacks}")
        print(f"  Current bet: {game.current_bet}")
        print(f"  P0 invested: {game.player_0_invested}")
        print(f"  P1 invested: {game.player_1_invested}")

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"[ERROR] Failed to create RustRiverHoldem: {e}")
        import traceback
        traceback.print_exc()


def test_action_sequence():
    """Test a sequence of actions to see payoffs."""
    print("\n" + "="*60)
    print("TEST: Action Sequence & Payoffs")
    print("="*60)

    try:
        game = RustRiverHoldem()

        print("\n[TEST] Playing through a game...")
        step = 0
        while not game.is_terminal() and step < 10:
            state = np.array(game.get_state())
            legal = game.legal_actions()
            player = game.current_player()

            print(f"\nStep {step}:")
            print(f"  Player: {player}")
            print(f"  Legal: {legal}")
            print(f"  State sum: {state.sum():.4f}")

            # Take the first legal action (usually fold=0 or call=1)
            action = 1 if legal[1] else 0  # Prefer call over fold
            print(f"  Taking action: {action}")

            game.apply_action(action)
            step += 1

        if game.is_terminal():
            returns = game.returns()
            print(f"\n[TERMINAL] Returns: {returns}")
            print(f"  Player 0: {returns[0]:+.4f}")
            print(f"  Player 1: {returns[1]:+.4f}")

            # Check for sign issues
            if returns[0] == -returns[1]:
                print("  [OK] Zero-sum game confirmed")
            else:
                print("  [WARNING] Not zero-sum!")

    except Exception as e:
        print(f"[ERROR] Action sequence failed: {e}")
        import traceback
        traceback.print_exc()


def test_regret_calculation():
    """Test if regrets are being calculated correctly."""
    print("\n" + "="*60)
    print("TEST: Regret Calculation Logic")
    print("="*60)

    print("""
In CFR, regret for action 'a' should be:
  regret(a) = value(a) - value(current_strategy)

For a Broadway Straight:
  - Fold regret should be VERY NEGATIVE (folding nuts is terrible)
  - Call/Raise regret should be POSITIVE or ZERO

If the network outputs NEGATIVE values for Call/Raise on strong hands,
the regret calculation might be inverted.
""")

    # Check what the network is outputting
    print("\nFrom CODE RED logs, network predictions for good hands:")
    print("  preds mean: -0.05 to -0.07 (NEGATIVE)")
    print("  preds range: -1.0 to +0.2")
    print("")
    print("This suggests the network is learning that ALL actions have")
    print("negative expected value, which is WRONG for strong hands.")
    print("")
    print("HYPOTHESIS: The reward/payoff signs might be inverted in Rust,")
    print("OR the regret targets are computed from the wrong player's perspective.")


def main():
    print("="*60)
    print("AION-26 HAND LOGIC DEBUG SCRIPT")
    print("="*60)

    test_hand_evaluation()
    test_rust_game_state()
    test_action_sequence()
    test_regret_calculation()

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)
    print("""
NEXT STEPS:
1. If Treys shows Straight but Rust doesn't -> Bug in Rust evaluator
2. If legal_actions masks Call/Raise -> Bug in action masking
3. If returns are inverted -> Bug in payoff calculation
4. If all regrets negative -> Bug in CFR regret formula
""")


if __name__ == "__main__":
    main()
