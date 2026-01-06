#!/usr/bin/env python3
"""Smoke test for Texas Hold'em River implementation.

This tests:
1. Game initialization
2. Card dealing and representation
3. Hand evaluation using treys
4. Encoder functionality

Test Scenario:
- Board: [Ah, Kh, Qh, Jh, 2s] (4 hearts + 1 spade)
- Player 0: [Th, 2c] (Ten of Hearts + Two of Clubs) → Royal Flush!
- Player 1: [As, Ks] (Ace of Spades + King of Spades) → Top Two Pair

Expected: Player 0 (Royal Flush) beats Player 1 (Two Pair)
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from treys import Card, Evaluator

from aion26.games.river_holdem import TexasHoldemRiver, new_river_holdem_with_cards
from aion26.deep_cfr.networks import HoldemEncoder

print("="*80)
print("TEXAS HOLD'EM RIVER - SMOKE TEST")
print("="*80)
print()

# ============================================================================
# Test 1: Card Representation and Evaluation
# ============================================================================

print("Test 1: Card Representation and Evaluation")
print("-" * 80)

# Create the board: [Ah, Kh, Qh, Jh, 2s]
board = [
    Card.new("Ah"),  # Ace of Hearts
    Card.new("Kh"),  # King of Hearts
    Card.new("Qh"),  # Queen of Hearts
    Card.new("Jh"),  # Jack of Hearts
    Card.new("2s"),  # Two of Spades
]

# Player 0 hand: [Th, 2c] - completes Royal Flush with board
hand_0 = [
    Card.new("Th"),  # Ten of Hearts (completes Royal Flush!)
    Card.new("2c"),  # Two of Clubs
]

# Player 1 hand: [As, Ks] - has top two pair
hand_1 = [
    Card.new("As"),  # Ace of Spades
    Card.new("Ks"),  # King of Spades
]

print(f"Board: {Card.print_pretty_cards(board)}")
print(f"Player 0 Hand: {Card.print_pretty_cards(hand_0)}")
print(f"Player 1 Hand: {Card.print_pretty_cards(hand_1)}")
print()

# Evaluate hands
evaluator = Evaluator()
rank_0 = evaluator.evaluate(board, hand_0)
rank_1 = evaluator.evaluate(board, hand_1)

print(f"Player 0 Rank: {rank_0} ({evaluator.class_to_string(evaluator.get_rank_class(rank_0))})")
print(f"Player 1 Rank: {rank_1} ({evaluator.class_to_string(evaluator.get_rank_class(rank_1))})")
print()

# Verify Player 0 wins (lower rank value = better hand)
assert rank_0 < rank_1, f"Player 0 should win! rank_0={rank_0}, rank_1={rank_1}"
assert rank_0 == 1, f"Player 0 should have Royal Flush (rank=1), got {rank_0}"
print("✅ PASS: Player 0 (Royal Flush) beats Player 1 (Two Pair)")
print()

# ============================================================================
# Test 2: Game State Creation
# ============================================================================

print("Test 2: Game State Creation")
print("-" * 80)

# Create game state with the specific scenario
game = new_river_holdem_with_cards(
    board=board,
    hand_0=hand_0,
    hand_1=hand_1,
    pot=10.0,
    stacks=(200.0, 200.0)
)

print(f"Game State:")
print(game)
print()

# Verify state
assert game.is_dealt, "Cards should be dealt"
assert not game.is_terminal(), "Game should not be terminal yet"
assert game.current_player() == 0, "Player 0 should act first"
print("✅ PASS: Game state created correctly")
print()

# ============================================================================
# Test 3: Legal Actions
# ============================================================================

print("Test 3: Legal Actions")
print("-" * 80)

actions = game.legal_actions()
print(f"Legal actions for Player 0: {actions}")
print(f"Action names: {[['Fold', 'Check/Call', 'Bet Pot', 'All-In'][a] for a in actions]}")
print()

# Player 0 should be able to check/call, bet pot, or all-in (no fold since no bet to face)
assert 1 in actions, "Check/Call should be legal"  # CHECK_CALL
assert 2 in actions, "Bet Pot should be legal"     # BET_POT
assert 3 in actions, "All-In should be legal"      # ALL_IN
assert 0 not in actions, "Fold should not be legal (nothing to fold to)"
print("✅ PASS: Legal actions are correct")
print()

# ============================================================================
# Test 4: Game Tree Traversal
# ============================================================================

print("Test 4: Game Tree Traversal")
print("-" * 80)

# Player 0 checks
game_after_check = game.apply_action(1)  # CHECK_CALL
print(f"After P0 checks:")
print(f"  Current player: {game_after_check.current_player()}")
print(f"  History: '{game_after_check.history}'")
print()

# Player 1 checks (both check → showdown)
game_showdown = game_after_check.apply_action(1)  # CHECK_CALL
print(f"After P1 checks:")
print(f"  Terminal: {game_showdown.is_terminal()}")
print(f"  History: '{game_showdown.history}'")
print()

assert game_showdown.is_terminal(), "Game should be terminal after both check"
print("✅ PASS: Game tree traversal works (check-check)")
print()

# ============================================================================
# Test 5: Payoffs
# ============================================================================

print("Test 5: Payoffs")
print("-" * 80)

returns = game_showdown.returns()
print(f"Returns: Player 0 = {returns[0]}, Player 1 = {returns[1]}")
print()

# Player 0 should win (Royal Flush beats Two Pair)
assert returns[0] > 0, f"Player 0 should win, got returns={returns}"
assert returns[1] < 0, f"Player 1 should lose, got returns={returns}"
assert returns[0] == -returns[1], "Returns should sum to zero"
print("✅ PASS: Correct payoffs (Player 0 wins with Royal Flush)")
print()

# ============================================================================
# Test 6: Encoder
# ============================================================================

print("Test 6: HoldemEncoder")
print("-" * 80)

encoder = HoldemEncoder()
print(f"Encoder input size: {encoder.input_size}")
print()

# Encode from Player 0's perspective
features = encoder.encode(game, player=0)
print(f"Encoded features shape: {features.shape}")
print(f"Expected shape: ({encoder.input_size},)")
print()

assert features.shape == (31,), f"Features should be 31-dim, got {features.shape}"
print("Feature breakdown:")
print(f"  Hand rank (one-hot): {features[:10].numpy()}")
print(f"  Hole cards: {features[10:14].numpy()}")
print(f"  Board cards: {features[14:24].numpy()}")
print(f"  Context: {features[24:].numpy()}")
print()

# Verify hand rank is Royal Flush (category 9)
rank_one_hot = features[:10].numpy()
assert rank_one_hot[9] == 1.0, f"Should encode Royal Flush (category 9), got {rank_one_hot}"
assert rank_one_hot.sum() == 1.0, "Hand rank should be one-hot"
print("✅ PASS: Encoder correctly identifies Royal Flush")
print()

# Encode from Player 1's perspective
features_p1 = encoder.encode(game, player=1)
rank_one_hot_p1 = features_p1[:10].numpy()
print(f"Player 1 hand rank (one-hot): {rank_one_hot_p1}")

# Player 1 has Two Pair (category 2)
assert rank_one_hot_p1[2] == 1.0, f"Should encode Two Pair (category 2), got {rank_one_hot_p1}"
print("✅ PASS: Encoder correctly identifies Two Pair for Player 1")
print()

# ============================================================================
# Test 7: Betting Scenario
# ============================================================================

print("Test 7: Betting Scenario")
print("-" * 80)

# Player 0 bets pot
game_after_bet = game.apply_action(2)  # BET_POT
print(f"After P0 bets pot:")
print(f"  Pot: {game_after_bet.pot}")
print(f"  P0 stack: {game_after_bet.stacks[0]}")
print(f"  P0 invested: {game_after_bet.player_0_invested}")
print(f"  Current bet: {game_after_bet.current_bet}")
print()

# Pot was 10, so P0 bet 10
assert game_after_bet.pot == 20.0, f"Pot should be 20 (10 + 10), got {game_after_bet.pot}"
assert game_after_bet.stacks[0] == 190.0, f"P0 stack should be 190 (200 - 10), got {game_after_bet.stacks[0]}"
assert game_after_bet.player_0_invested == 10.0, "P0 should have invested 10"
print("✅ PASS: Pot bet works correctly")
print()

# Player 1 calls
game_after_call = game_after_bet.apply_action(1)  # CHECK_CALL (calls the bet)
print(f"After P1 calls:")
print(f"  Pot: {game_after_call.pot}")
print(f"  P1 stack: {game_after_call.stacks[1]}")
print(f"  P1 invested: {game_after_call.player_1_invested}")
print(f"  Terminal: {game_after_call.is_terminal()}")
print()

assert game_after_call.pot == 30.0, f"Pot should be 30 (20 + 10), got {game_after_call.pot}"
assert game_after_call.stacks[1] == 190.0, f"P1 stack should be 190, got {game_after_call.stacks[1]}"
assert game_after_call.is_terminal(), "Game should be terminal after call"
print("✅ PASS: Call works correctly")
print()

# Check final payoffs
final_returns = game_after_call.returns()
print(f"Final returns: Player 0 = {final_returns[0]}, Player 1 = {final_returns[1]}")

# Player 0 wins pot/2 = 15.0
assert final_returns[0] == 15.0, f"P0 should win 15.0, got {final_returns[0]}"
assert final_returns[1] == -15.0, f"P1 should lose 15.0, got {final_returns[1]}"
print("✅ PASS: Final payoffs correct")
print()

# ============================================================================
# Test 8: All-In Scenario
# ============================================================================

print("Test 8: All-In Scenario")
print("-" * 80)

# Player 0 goes all-in
game_all_in = game.apply_action(3)  # ALL_IN
print(f"After P0 goes all-in:")
print(f"  Pot: {game_all_in.pot}")
print(f"  P0 stack: {game_all_in.stacks[0]}")
print(f"  Current bet: {game_all_in.current_bet}")
print()

assert game_all_in.stacks[0] == 0.0, "P0 should have 0 stack after all-in"
assert game_all_in.pot == 210.0, f"Pot should be 210 (10 + 200), got {game_all_in.pot}"
print("✅ PASS: All-in works correctly")
print()

# ============================================================================
# Summary
# ============================================================================

print("="*80)
print("🎉 ALL TESTS PASSED! 🎉")
print("="*80)
print()
print("Summary:")
print("  ✅ Card representation and evaluation (treys)")
print("  ✅ Game state creation")
print("  ✅ Legal actions generation")
print("  ✅ Game tree traversal")
print("  ✅ Correct payoffs")
print("  ✅ HoldemEncoder (31-dim features)")
print("  ✅ Hand rank detection (Royal Flush vs Two Pair)")
print("  ✅ Betting mechanics (pot bet, call)")
print("  ✅ All-in mechanics")
print()
print("Texas Hold'em River implementation is ready! 🚀")
print()
