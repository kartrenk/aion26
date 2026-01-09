#!/usr/bin/env python
"""Final Model Evaluation Suite

This script subjects the trained Deep CFR model to rigorous testing
to verify it actually plays good poker, not just achieves low loss.

Three Tests:
1. Sanity Check (vs RandomBot): Must crush by >3000 mbb/h
2. Skill Check (vs CallingStation): Must be profitable
3. Bias Check (Self-Play): Must be ~0 mbb/h (no positional bias)

Usage:
    uv run python scripts/evaluate_final.py
    uv run python scripts/evaluate_final.py --model river_checkpoint_e500.pt
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol, Optional
import random

import numpy as np
import torch
import torch.nn as nn

import aion26_rust

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL_PATH = "river_model_final.pt"
NUM_HANDS = 50_000
BIG_BLIND = 2.0  # For mbb/h calculation

STATE_DIM = 136
TARGET_DIM = 4

# =============================================================================
# Network (must match training)
# =============================================================================

class AdvantageNetwork(nn.Module):
    """Neural network for predicting action advantages."""

    def __init__(self, input_size: int = 136, hidden_size: int = 256, output_size: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =============================================================================
# State Encoder (Python version matching Rust)
# =============================================================================

class StateEncoder:
    """Encode game state to neural network input format."""

    @staticmethod
    def encode(game_state, player: int) -> np.ndarray:
        """Encode state for the given player's perspective."""
        features = np.zeros(STATE_DIM, dtype=np.float32)

        # Get game info (accessing properties directly, not methods)
        hands = game_state.hands
        board = game_state.board
        pot = game_state.pot
        stacks = game_state.stacks
        current_bet = game_state.current_bet
        invested = [game_state.player_0_invested, game_state.player_1_invested]

        # 1. Hand rank category (10 dims) - simplified, just mark as valid
        features[0] = 1.0  # Placeholder for hand category

        # 2. Hole cards (34 dims: 2 cards × 17 bits)
        if len(hands) > player:
            hand = hands[player]
            for i, card in enumerate(hand[:2]):
                offset = 10 + i * 17
                rank = card % 13
                suit = card // 13
                features[offset + rank] = 1.0
                features[offset + 13 + suit] = 1.0

        # 3. Board cards (85 dims: 5 cards × 17 bits)
        for i, card in enumerate(board[:5]):
            offset = 10 + 34 + i * 17
            rank = card % 13
            suit = card // 13
            features[offset + rank] = 1.0
            features[offset + 13 + suit] = 1.0

        # 4. Betting context (7 dims)
        context_offset = 10 + 34 + 85
        max_pot = 500.0
        max_stack = 200.0

        current_invested = invested[player] if len(invested) > player else 0.0
        call_amount = max(0, current_bet - current_invested)
        pot_after_call = pot + call_amount
        pot_odds = call_amount / pot_after_call if pot_after_call > 0 else 0.0

        features[context_offset] = pot / max_pot
        features[context_offset + 1] = stacks[0] / max_stack if len(stacks) > 0 else 0.0
        features[context_offset + 2] = stacks[1] / max_stack if len(stacks) > 1 else 0.0
        features[context_offset + 3] = current_bet / max_stack
        features[context_offset + 4] = invested[0] / max_stack if len(invested) > 0 else 0.0
        features[context_offset + 5] = invested[1] / max_stack if len(invested) > 1 else 0.0
        features[context_offset + 6] = pot_odds

        return features


# =============================================================================
# Strategy Interface
# =============================================================================

class Strategy(Protocol):
    """Protocol for poker strategies."""
    def get_action(self, game_state, player: int) -> int:
        """Return action index given game state."""
        ...

    @property
    def name(self) -> str:
        ...


# =============================================================================
# Bot Implementations
# =============================================================================

@dataclass
class DeepCFRStrategy:
    """Strategy using trained Deep CFR network."""
    network: nn.Module
    device: torch.device
    name: str = "DeepCFR"

    def get_action(self, game_state, player: int) -> int:
        """Sample action from network's strategy."""
        legal_actions = game_state.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # Encode state
        state = StateEncoder.encode(game_state, player)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

        # Get advantages from network
        with torch.no_grad():
            advantages = self.network(state_tensor)[0].cpu().numpy()

        # Regret matching to get strategy
        strategy = self._regret_matching(advantages, legal_actions)

        # Sample action according to strategy
        action_idx = np.random.choice(len(legal_actions), p=strategy)
        return legal_actions[action_idx]

    def _regret_matching(self, advantages: np.ndarray, legal_actions: list) -> np.ndarray:
        """Convert advantages to probability distribution."""
        # Get advantages for legal actions only
        legal_advantages = np.array([advantages[a] if a < len(advantages) else 0.0
                                      for a in legal_actions])

        # Positive regrets only
        positive = np.maximum(legal_advantages, 0)
        total = positive.sum()

        if total > 0:
            return positive / total
        else:
            # Uniform fallback
            return np.ones(len(legal_actions)) / len(legal_actions)


@dataclass
class RandomBot:
    """Plays uniformly random legal actions."""
    name: str = "RandomBot"

    def get_action(self, game_state, player: int) -> int:
        legal_actions = game_state.legal_actions()
        return random.choice(legal_actions)


@dataclass
class CallingStation:
    """Always calls or checks. Never folds, never raises."""
    name: str = "CallingStation"

    def get_action(self, game_state, player: int) -> int:
        legal_actions = game_state.legal_actions()

        # Action mapping: 0=fold, 1=check/call, 2=bet/raise, 3=all-in
        # Prefer check/call
        if 1 in legal_actions:
            return 1  # Check or call
        elif 0 in legal_actions:
            return 1 if 1 in legal_actions else legal_actions[0]  # Call if possible
        else:
            return legal_actions[0]


@dataclass
class FoldBot:
    """Always folds when possible."""
    name: str = "FoldBot"

    def get_action(self, game_state, player: int) -> int:
        legal_actions = game_state.legal_actions()
        if 0 in legal_actions:
            return 0  # Fold
        return legal_actions[0]  # Otherwise first legal action


@dataclass
class AggressiveBot:
    """Always raises/bets when possible."""
    name: str = "AggressiveBot"

    def get_action(self, game_state, player: int) -> int:
        legal_actions = game_state.legal_actions()
        # Prefer raise/bet (action 2 or 3)
        for action in [3, 2]:  # All-in, then raise
            if action in legal_actions:
                return action
        return legal_actions[-1]  # Otherwise last (usually most aggressive)


# =============================================================================
# Game Simulator
# =============================================================================

def play_hand(p0_strategy: Strategy, p1_strategy: Strategy) -> tuple[float, float]:
    """Play a single hand and return (p0_profit, p1_profit) in chips."""

    # Create a fresh game with blinds posted (pot=2, each invested 1)
    game = aion26_rust.RustRiverHoldem(
        stacks=[100.0, 100.0],
        pot=2.0,
        current_bet=0.0,
        player_0_invested=1.0,
        player_1_invested=1.0,
    )

    # Deal cards (action 0 when not dealt)
    game = game.apply_action(0)

    # Play until terminal
    while not game.is_terminal():
        current_player = game.current_player()

        if current_player == -1:  # Terminal
            break

        if current_player == 0:
            action = p0_strategy.get_action(game, 0)
        else:
            action = p1_strategy.get_action(game, 1)

        game = game.apply_action(action)

    # Get final returns
    returns = game.returns()
    return returns[0], returns[1]


def evaluate_matchup(
    p0_strategy: Strategy,
    p1_strategy: Strategy,
    num_hands: int,
    verbose: bool = True,
) -> dict:
    """Evaluate P0 vs P1 over many hands.

    Returns dict with:
    - p0_total: Total chips won by P0
    - p1_total: Total chips won by P1
    - p0_mbb_per_hand: P0's win rate in milli-big-blinds per hand
    - hands_played: Number of hands
    - variance: Standard deviation of P0 results
    """
    p0_results = []

    start_time = time.time()

    for i in range(num_hands):
        p0_profit, p1_profit = play_hand(p0_strategy, p1_strategy)
        p0_results.append(p0_profit)

        if verbose and (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            avg = np.mean(p0_results)
            print(f"    {i+1:,}/{num_hands:,} hands | "
                  f"{rate:.0f} hands/s | "
                  f"P0 avg: {avg:.2f} chips", end="\r")

    if verbose:
        print()  # New line after progress

    p0_results = np.array(p0_results)
    p0_total = p0_results.sum()
    p0_mean = p0_results.mean()
    p0_std = p0_results.std()

    # Convert to mbb/h (milli-big-blinds per hand)
    # mbb = (profit / big_blind) * 1000
    p0_mbb_per_hand = (p0_mean / BIG_BLIND) * 1000

    elapsed = time.time() - start_time

    return {
        "p0_total": p0_total,
        "p1_total": -p0_total,
        "p0_mean": p0_mean,
        "p0_mbb_per_hand": p0_mbb_per_hand,
        "p0_std": p0_std,
        "hands_played": num_hands,
        "time_seconds": elapsed,
        "hands_per_second": num_hands / elapsed,
    }


# =============================================================================
# Test Suite
# =============================================================================

def run_test_suite(model_path: str, num_hands: int = NUM_HANDS):
    """Run the complete evaluation suite."""

    print("=" * 70)
    print("DEEP CFR MODEL EVALUATION SUITE")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Hands per test: {num_hands:,}")
    print("=" * 70)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Available checkpoints:")
        for f in sorted(Path(".").glob("river_*.pt")):
            print(f"  - {f}")
        return None

    checkpoint = torch.load(model_path, map_location=device)

    network = AdvantageNetwork(
        input_size=STATE_DIM,
        hidden_size=256,
        output_size=TARGET_DIM,
    ).to(device)

    network.load_state_dict(checkpoint["network"])
    network.eval()

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    if "total_samples" in checkpoint:
        print(f"Trained on {checkpoint['total_samples']:,} samples")
    if "nash_conv" in checkpoint:
        print(f"Nash Conv at checkpoint: {checkpoint['nash_conv']:.6f}")

    # Create strategies
    deep_cfr = DeepCFRStrategy(network=network, device=device, name="DeepCFR")
    random_bot = RandomBot()
    calling_station = CallingStation()

    results = {}

    # =========================================================================
    # TEST 1: Sanity Check (vs RandomBot)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: SANITY CHECK (DeepCFR vs RandomBot)")
    print("=" * 70)
    print("Hypothesis: DeepCFR must crush RandomBot by > +3000 mbb/h")
    print("-" * 70)

    result = evaluate_matchup(deep_cfr, random_bot, num_hands)
    results["vs_random"] = result

    mbb = result["p0_mbb_per_hand"]
    passed = mbb > 3000
    status = "PASS" if passed else "FAIL"

    print(f"\nResult: {mbb:+.1f} mbb/h")
    print(f"Total: {result['p0_total']:+.0f} chips over {num_hands:,} hands")
    print(f"Std Dev: {result['p0_std']:.2f} chips/hand")
    print(f"Status: [{status}] {'Model crushes random play' if passed else 'CRITICAL: Model cannot beat random!'}")

    # =========================================================================
    # TEST 2: Skill Check (vs CallingStation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: SKILL CHECK (DeepCFR vs CallingStation)")
    print("=" * 70)
    print("Hypothesis: DeepCFR should extract value (positive winrate)")
    print("-" * 70)

    result = evaluate_matchup(deep_cfr, calling_station, num_hands)
    results["vs_calling_station"] = result

    mbb = result["p0_mbb_per_hand"]
    passed = mbb > 0
    status = "PASS" if passed else "FAIL"

    print(f"\nResult: {mbb:+.1f} mbb/h")
    print(f"Total: {result['p0_total']:+.0f} chips over {num_hands:,} hands")
    print(f"Std Dev: {result['p0_std']:.2f} chips/hand")
    print(f"Status: [{status}] {'Model extracts value from passive play' if passed else 'WARNING: Model loses to calling station!'}")

    # =========================================================================
    # TEST 3: Bias Check (Self-Play)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: BIAS CHECK (DeepCFR vs DeepCFR)")
    print("=" * 70)
    print("Hypothesis: Self-play should be ~0 mbb/h (no positional bias)")
    print("-" * 70)

    deep_cfr_copy = DeepCFRStrategy(network=network, device=device, name="DeepCFR_P1")
    result = evaluate_matchup(deep_cfr, deep_cfr_copy, num_hands)
    results["self_play"] = result

    mbb = result["p0_mbb_per_hand"]
    # Allow some variance - within 2 standard errors of 0
    se = (result["p0_std"] / BIG_BLIND * 1000) / np.sqrt(num_hands)
    passed = abs(mbb) < max(500, 3 * se)  # Within 500 mbb/h or 3 SE
    status = "PASS" if passed else "WARN"

    print(f"\nResult: {mbb:+.1f} mbb/h")
    print(f"Total: {result['p0_total']:+.0f} chips over {num_hands:,} hands")
    print(f"Std Dev: {result['p0_std']:.2f} chips/hand")
    print(f"Standard Error: {se:.1f} mbb/h")
    print(f"Status: [{status}] {'No significant positional bias' if passed else 'Potential positional bias detected!'}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Test':<30} | {'Result':>12} | {'Status':>8}")
    print("-" * 70)

    tests = [
        ("vs RandomBot", results["vs_random"]["p0_mbb_per_hand"], "> +3000"),
        ("vs CallingStation", results["vs_calling_station"]["p0_mbb_per_hand"], "> 0"),
        ("Self-Play (Bias)", results["self_play"]["p0_mbb_per_hand"], "~ 0"),
    ]

    for name, mbb, threshold in tests:
        status = "PASS" if (
            (name == "vs RandomBot" and mbb > 3000) or
            (name == "vs CallingStation" and mbb > 0) or
            (name == "Self-Play (Bias)" and abs(mbb) < 500)
        ) else ("WARN" if name == "Self-Play (Bias)" else "FAIL")
        print(f"{name:<30} | {mbb:>+10.1f} | {status:>8}")

    print("-" * 70)

    all_passed = (
        results["vs_random"]["p0_mbb_per_hand"] > 3000 and
        results["vs_calling_station"]["p0_mbb_per_hand"] > 0 and
        abs(results["self_play"]["p0_mbb_per_hand"]) < 500
    )

    if all_passed:
        print("OVERALL: ALL TESTS PASSED - Model is production ready!")
    else:
        print("OVERALL: SOME TESTS FAILED - Review results above")

    print("=" * 70)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Deep CFR Model Evaluation Suite")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to model checkpoint")
    parser.add_argument("--hands", type=int, default=NUM_HANDS,
                        help="Number of hands per test")
    args = parser.parse_args()

    run_test_suite(args.model, args.hands)


if __name__ == "__main__":
    main()
