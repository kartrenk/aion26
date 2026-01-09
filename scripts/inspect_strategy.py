#!/usr/bin/env python3
"""Sanity Check: Inspect Agent Strategy on Specific River Scenarios.

This script forces the agent into specific game states to verify that
the learned policy makes sense. If the agent folds a Royal Flush or
bets with air, the algorithm is fundamentally broken.

Test Scenarios:
1. Royal Flush: Board [As, Ks, Qs, Js, 2h], Hero [Ts, 3d] (absolute nuts)
   Expected: BET 100% (or close to it)

2. Weak Hand: Same board, Hero [2c, 3c] (pair of 2s, very weak)
   Expected: CHECK or FOLD (defensive play)
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from treys import Card

from aion26.games.river_holdem import TexasHoldemRiver
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.deep_cfr.networks import HoldemEncoder
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler


def create_specific_river_state(
    board_cards: list[str],
    hero_cards: list[str],
    pot: float = 100.0,
    stacks: tuple[float, float] = (100.0, 100.0),
    current_bet: float = 0.0,
    player_0_invested: float = 50.0,
    player_1_invested: float = 50.0,
) -> TexasHoldemRiver:
    """Create a specific River Hold'em state with known cards.

    Args:
        board_cards: List of 5 card strings (e.g., ["As", "Ks", "Qs", "Js", "2h"])
        hero_cards: List of 2 card strings for hero (player 0)
        pot: Pot size
        stacks: (stack_0, stack_1)
        current_bet: Current bet to call
        player_0_invested: Amount player 0 has invested
        player_1_invested: Amount player 1 has invested

    Returns:
        TexasHoldemRiver state with specified configuration
    """
    # Convert card strings to treys format
    board = [Card.new(c) for c in board_cards]
    hero_hand = [Card.new(c) for c in hero_cards]

    # Create a dummy opponent hand (not used for policy inspection)
    # We'll use two cards that aren't in board or hero hand
    used_cards = set(board_cards + hero_cards)
    remaining_deck = [
        f"{r}{s}"
        for r in "23456789TJQKA"
        for s in "shdc"
        if f"{r}{s}" not in used_cards
    ]
    villain_hand = [Card.new(remaining_deck[0]), Card.new(remaining_deck[1])]

    # Create state manually (bypass chance node)
    state = TexasHoldemRiver(
        stacks=list(stacks),
        pot=pot,
        current_bet=current_bet,
        player_0_invested=player_0_invested,
        player_1_invested=player_1_invested,
        history="",
        board=board,
        hands=[hero_hand, villain_hand],
        is_dealt=True,
        deck=[],  # Already dealt
    )

    return state


def main():
    print("="*80)
    print("POLICY SANITY CHECK")
    print("="*80)
    print()

    # Create encoder and trainer (load trained model)
    encoder = HoldemEncoder()

    # Note: This requires a trained model checkpoint
    # For now, we'll create a fresh trainer to show the structure
    # In production, you'd load from checkpoint

    print("⚠️  WARNING: This script requires a trained model checkpoint.")
    print("For demonstration, we'll show expected vs random initialization.")
    print()

    # Scenario 1: Royal Flush
    print("="*80)
    print("SCENARIO 1: ROYAL FLUSH (Absolute Nuts)")
    print("="*80)
    print()
    print("Board: [A♠, K♠, Q♠, J♠, 2♥]")
    print("Hero:  [T♠, 3♦]  ← Royal Flush")
    print("Pot:   100 BB")
    print("Stacks: 100 BB each")
    print()

    royal_flush_state = create_specific_river_state(
        board_cards=["As", "Ks", "Qs", "Js", "2h"],
        hero_cards=["Ts", "3d"],
        pot=100.0,
        stacks=(100.0, 100.0),
        current_bet=0.0,
        player_0_invested=50.0,
        player_1_invested=50.0,
    )

    print("Legal Actions:")
    legal = royal_flush_state.legal_actions()
    action_names = ["Fold", "Check/Call", "Bet Pot", "All-In"]
    for action in legal:
        print(f"  {action}: {action_names[action]}")
    print()

    # For demonstration, show what a random policy would do
    random_policy = np.ones(4) / 4  # Uniform random
    print("Random Policy (Baseline):")
    for action in legal:
        print(f"  {action_names[action]}: {random_policy[action]*100:.1f}%")
    print()
    print("Expected (Trained Agent):")
    print("  Fold: ~0%")
    print("  Check/Call: ~0%")
    print("  Bet Pot: ~50-60%")
    print("  All-In: ~40-50%")
    print()

    # Scenario 2: Weak Hand
    print("="*80)
    print("SCENARIO 2: WEAK HAND (Pair of 2s)")
    print("="*80)
    print()
    print("Board: [A♠, K♠, Q♠, J♠, 2♥]")
    print("Hero:  [2♣, 3♣]  ← Pair of 2s (very weak)")
    print("Pot:   100 BB")
    print("Stacks: 100 BB each")
    print()

    weak_hand_state = create_specific_river_state(
        board_cards=["As", "Ks", "Qs", "Js", "2h"],
        hero_cards=["2c", "3c"],
        pot=100.0,
        stacks=(100.0, 100.0),
        current_bet=0.0,
        player_0_invested=50.0,
        player_1_invested=50.0,
    )

    print("Legal Actions:")
    for action in legal:
        print(f"  {action}: {action_names[action]}")
    print()

    print("Random Policy (Baseline):")
    for action in legal:
        print(f"  {action_names[action]}: {random_policy[action]*100:.1f}%")
    print()
    print("Expected (Trained Agent):")
    print("  Fold: ~0% (no bet to call)")
    print("  Check/Call: ~95%")
    print("  Bet Pot: ~5%")
    print("  All-In: ~0%")
    print()

    print("="*80)
    print("TO USE THIS SCRIPT WITH TRAINED MODEL:")
    print("="*80)
    print()
    print("1. Train for at least 5k iterations")
    print("2. Save model checkpoint to 'checkpoints/river_model.pt'")
    print("3. Load checkpoint in this script:")
    print("   trainer = DeepCFRTrainer.load('checkpoints/river_model.pt')")
    print("4. Get strategy: strategy = trainer.get_strategy(state, player=0)")
    print("5. Compare strategy to expected behavior above")
    print()
    print("FAILURE MODE:")
    print("If Royal Flush strategy is: Fold=80%, the algorithm is broken.")
    print()


if __name__ == "__main__":
    main()
