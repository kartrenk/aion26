#!/usr/bin/env python3
"""Quick test of baseline bots and head-to-head evaluator."""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from treys import Card

from aion26.games.river_holdem import new_river_holdem_with_cards
from aion26.baselines import RandomBot, CallingStation, HonestBot
from aion26.metrics.evaluator import HeadToHeadEvaluator

print("="*80)
print("BASELINE BOTS & EVALUATOR TEST")
print("="*80)
print()

# Create a simple scenario
board = [
    Card.new("Ah"),
    Card.new("Kh"),
    Card.new("Qh"),
    Card.new("Jh"),
    Card.new("2s"),
]

hand_0 = [Card.new("Th"), Card.new("2c")]  # Royal Flush
hand_1 = [Card.new("As"), Card.new("Ks")]  # Two Pair

game = new_river_holdem_with_cards(
    board=board,
    hand_0=hand_0,
    hand_1=hand_1,
)

print("Test Scenario:")
print(f"  Board: {Card.print_pretty_cards(board)}")
print(f"  P0: {Card.print_pretty_cards(hand_0)} (Royal Flush)")
print(f"  P1: {Card.print_pretty_cards(hand_1)} (Two Pair)")
print()

# Test 1: RandomBot
print("Test 1: RandomBot")
print("-" * 80)
bot = RandomBot(seed=42)
action = bot.get_action(game)
print(f"  Action: {action} ({['Fold', 'Check/Call', 'Bet Pot', 'All-In'][action]})")
assert action in game.legal_actions()
print("  ✅ PASS: RandomBot picks legal action")
print()

# Test 2: CallingStation
print("Test 2: CallingStation")
print("-" * 80)
bot = CallingStation()
action = bot.get_action(game)
print(f"  Action: {action} ({['Fold', 'Check/Call', 'Bet Pot', 'All-In'][action]})")
assert action == 1  # CHECK_CALL
print("  ✅ PASS: CallingStation always checks/calls")
print()

# Test 3: HonestBot with strong hand
print("Test 3: HonestBot")
print("-" * 80)
bot = HonestBot()

# Test with P0 (Royal Flush - should bet)
action_p0 = bot.get_action(game)
print(f"  P0 action (Royal Flush): {action_p0} ({['Fold', 'Check/Call', 'Bet Pot', 'All-In'][action_p0]})")
assert action_p0 in [2, 3], "Royal Flush should bet or all-in"

# Test with P1 (Two Pair - medium strength, should call)
game_p1 = game.apply_action(1)  # P0 checks
action_p1 = bot.get_action(game_p1)
print(f"  P1 action (Two Pair): {action_p1} ({['Fold', 'Check/Call', 'Bet Pot', 'All-In'][action_p1]})")

print("  ✅ PASS: HonestBot makes strength-based decisions")
print()

# Test 4: Head-to-Head Evaluator
print("Test 4: HeadToHeadEvaluator")
print("-" * 80)

# Create a simple always-call strategy
strategy = {}
evaluator = HeadToHeadEvaluator(big_blind=2.0)

# Create a dummy strategy that always checks/calls
# This is just for testing - doesn't need to be good
from aion26.games.river_holdem import new_river_holdem_game

test_game = new_river_holdem_game()

print("  Playing 10 hands vs RandomBot (quick test)...")
try:
    result = evaluator.evaluate(
        initial_state=test_game,
        strategy={},  # Empty strategy will default to random
        opponent=RandomBot(),
        num_hands=10,
        alternate_positions=True
    )
    print(f"    Win rate: {result.avg_mbb_per_hand:+.0f} mbb/h ± {result.confidence_95:.0f}")
    print(f"    Agent: {result.agent_winnings:+.2f} BB")
    print("  ✅ PASS: Evaluator completes without errors")
except Exception as e:
    print(f"  ❌ FAIL: {e}")
    raise

print()
print("="*80)
print("🎉 ALL BASELINE TESTS PASSED! 🎉")
print("="*80)
print()
print("Baseline bots and evaluator are working correctly!")
print()
