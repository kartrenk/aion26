#!/usr/bin/env python3
"""
Analyze losses against CallingStation to diagnose the -206 mbb/h leak.

This script plays hands against CallingStation on the training flop (Ac Kd Qh)
and categorizes big losses (>20 BB) into:
- Bad Bluff: Agent bet/raised with weak hand (< Pair) on river
- Value Own-Goal: Agent value bet worse hand into better hand
- Cooler: Unavoidable loss (strong vs stronger hand)

Usage:
    python scripts/analyze_station_losses.py [--hands 1000] [--threshold 20]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from collections import defaultdict

# Import Rust extension
try:
    import aion26_rust
    from aion26_rust import RustFullHoldem
except ImportError:
    print("ERROR: aion26_rust not found. Run: cd src/aion26_rust && maturin develop --release")
    sys.exit(1)

# Import baselines
from aion26.baselines import CallingStationBot

# ============================================================================
# Constants
# ============================================================================

# Training flop: Ac, Kd, Qh -> card indices
# Card encoding: suit * 13 + rank (A=12, K=11, Q=10)
# Ac = 0*13 + 12 = 12, Kd = 1*13 + 11 = 24, Qh = 2*13 + 10 = 36
TRAINING_FLOP = [12, 24, 36]

# Action names for Full HUNL (8 actions)
ACTION_NAMES = ['Fold', 'Check/Call', 'Bet 0.5x', 'Bet 0.75x', 'Bet Pot', 'Bet 1.5x', 'Bet 2x', 'All-In']

# Hand categories (from evaluator)
HAND_CATEGORIES = [
    'High Card',      # 0
    'Pair',           # 1
    'Two Pair',       # 2
    'Three of Kind',  # 3
    'Straight',       # 4
    'Flush',          # 5
    'Full House',     # 6
    'Four of Kind',   # 7
    'Straight Flush', # 8
    'Royal Flush',    # 9 (same as straight flush in most systems)
]

# State/Target dimensions for Full HUNL
STATE_DIM = 220
TARGET_DIM = 8

# ============================================================================
# Network Definition (must match training)
# ============================================================================

class AdvantageNetworkFull(torch.nn.Module):
    """Larger advantage network for Full HUNL (220 dims, 8 actions).
    Must match train_webapp_pro.py architecture exactly.
    """

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = TARGET_DIM, hidden_dim: int = 512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)

# ============================================================================
# Card Utilities
# ============================================================================

RANKS = '23456789TJQKA'
SUITS = 'cdhs'  # clubs, diamonds, hearts, spades

def card_to_string(card: int) -> str:
    """Convert card index (0-51) to string like 'Ac', 'Kd'."""
    rank = card % 13
    suit = card // 13
    return f"{RANKS[rank]}{SUITS[suit]}"

def cards_to_string(cards: List[int]) -> str:
    """Convert list of cards to string."""
    return ' '.join(card_to_string(c) for c in cards)

def get_hand_category_name(category: int) -> str:
    """Get human-readable hand category name."""
    if 0 <= category < len(HAND_CATEGORIES):
        return HAND_CATEGORIES[category]
    return f"Unknown({category})"

# ============================================================================
# State Encoding (must match training)
# ============================================================================

def encode_card(card: int) -> np.ndarray:
    """Encode single card as 17-dim vector: 13 rank one-hot + 4 suit one-hot."""
    features = np.zeros(17, dtype=np.float32)
    rank = card % 13
    suit = card // 13
    features[rank] = 1.0
    features[13 + suit] = 1.0
    return features

def encode_state(game, player: int) -> np.ndarray:
    """Encode game state for network input (220 dims for Full HUNL)."""
    features = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # [0-9]: Hand rank category one-hot (10 dims) - only valid on river
    try:
        if game.street == 3 and len(game.board) == 5:
            hand_category = game.get_hand_strength(player)
            if 0 <= hand_category < 10:
                features[hand_category] = 1.0
    except:
        pass
    idx = 10

    # [10-43]: Hole cards (2 × 17 = 34 dims)
    hand = game.hands[player]
    for i, card in enumerate(hand[:2]):
        features[idx:idx+17] = encode_card(card)
        idx += 17

    # [44-128]: Board cards (5 × 17 = 85 dims)
    board = game.board
    for i in range(5):
        if i < len(board):
            features[idx:idx+17] = encode_card(board[i])
        idx += 17

    # [129-132]: Street one-hot (4 dims)
    street = game.street
    if 0 <= street < 4:
        features[129 + street] = 1.0
    idx = 133

    # [133-196]: Action history (8 × 8 = 64 dims)
    action_history = game.get_action_history()
    for i, action in enumerate(action_history[-8:]):
        if 0 <= action < 8:
            features[idx + i * 8 + action] = 1.0
    idx = 197

    # [197-219]: Betting context (23 dims)
    pot = game.pot
    stacks = game.stacks
    current_bet = game.current_bet
    invested = game.invested_street

    # Normalize by starting stack (100 BB)
    starting_stack = 100.0
    features[idx] = pot / (2 * starting_stack)
    features[idx+1] = stacks[player] / starting_stack
    features[idx+2] = stacks[1-player] / starting_stack
    features[idx+3] = current_bet / starting_stack
    features[idx+4] = invested[player] / starting_stack
    features[idx+5] = invested[1-player] / starting_stack

    return features

# ============================================================================
# Regret Matching
# ============================================================================

def regret_matching(advantages: np.ndarray, legal_actions: List[int]) -> np.ndarray:
    """Convert advantages to strategy using regret matching."""
    legal_advantages = np.array([advantages[a] for a in legal_actions])

    # Clamp to positive (regret matching)
    positive = np.maximum(legal_advantages, 0)
    total = positive.sum()

    if total > 1e-9:
        strategy = positive / total
    else:
        # Uniform if all non-positive - but prefer Check/Call over Fold
        # to avoid suicide shoves
        strategy = np.ones(len(legal_actions)) / len(legal_actions)

        # If Check/Call is available and Fold is available, prefer Check/Call
        if 1 in legal_actions and 0 in legal_actions:
            check_idx = legal_actions.index(1)
            fold_idx = legal_actions.index(0)
            strategy[fold_idx] = 0.1 / len(legal_actions)
            strategy[check_idx] = 0.9
            strategy = strategy / strategy.sum()

    return strategy

# ============================================================================
# Hand History Tracking
# ============================================================================

@dataclass
class HandAction:
    """Single action in a hand."""
    street: int
    player: int
    action: int
    action_name: str
    pot_before: float
    pot_after: float
    stack_before: float
    stack_after: float

@dataclass
class HandHistory:
    """Complete hand history for analysis."""
    hand_num: int
    hero_cards: List[int]
    villain_cards: List[int]
    board: List[int]
    actions: List[HandAction] = field(default_factory=list)
    hero_result: float = 0.0
    villain_result: float = 0.0
    hero_hand_category: int = 0
    villain_hand_category: int = 0
    loss_category: str = ""

    def add_action(self, street: int, player: int, action: int, pot_before: float,
                   pot_after: float, stack_before: float, stack_after: float):
        self.actions.append(HandAction(
            street=street,
            player=player,
            action=action,
            action_name=ACTION_NAMES[action] if action < len(ACTION_NAMES) else f"Action{action}",
            pot_before=pot_before,
            pot_after=pot_after,
            stack_before=stack_before,
            stack_after=stack_after,
        ))

    def print_history(self):
        """Print formatted hand history."""
        print(f"\n{'='*60}")
        print(f"Hand #{self.hand_num} - Loss Category: {self.loss_category}")
        print(f"{'='*60}")
        print(f"Hero:    {cards_to_string(self.hero_cards)} ({get_hand_category_name(self.hero_hand_category)})")
        print(f"Villain: {cards_to_string(self.villain_cards)} ({get_hand_category_name(self.villain_hand_category)})")
        print(f"Board:   {cards_to_string(self.board)}")
        print(f"Result:  Hero {self.hero_result:+.1f} BB | Villain {self.villain_result:+.1f} BB")
        print()

        current_street = -1
        street_names = ['Preflop', 'Flop', 'Turn', 'River']

        for action in self.actions:
            if action.street != current_street:
                current_street = action.street
                street_name = street_names[current_street] if current_street < 4 else f"Street{current_street}"
                print(f"\n--- {street_name} ---")

            player_name = "Hero" if action.player == 0 else "Villain"
            print(f"  {player_name}: {action.action_name} (pot: {action.pot_before:.1f} -> {action.pot_after:.1f})")

        print()

# ============================================================================
# Loss Categorization
# ============================================================================

def categorize_loss(history: HandHistory) -> str:
    """
    Categorize the loss type:
    - Bad Bluff: Hero bet/raised with weak hand (< Pair) when villain called
    - Value Own-Goal: Hero value bet but had worse hand
    - Cooler: Strong vs stronger hand (unavoidable)
    """
    hero_cat = history.hero_hand_category
    villain_cat = history.villain_hand_category

    # Check if hero made aggressive actions on river with weak hand
    river_actions = [a for a in history.actions if a.street == 3 and a.player == 0]
    hero_bet_river = any(a.action >= 2 for a in river_actions)  # Actions 2-7 are bets/raises

    # Bad Bluff: Hero bet with less than a pair on river
    if hero_bet_river and hero_cat < 1:  # High card only
        return "Bad Bluff (High Card)"

    if hero_bet_river and hero_cat == 1 and villain_cat >= 2:  # Pair vs better
        return "Bad Bluff (Weak Pair)"

    # Value Own-Goal: Hero bet with decent hand but villain had better
    if hero_bet_river and hero_cat >= 1 and villain_cat > hero_cat:
        return f"Value Own-Goal ({get_hand_category_name(hero_cat)} vs {get_hand_category_name(villain_cat)})"

    # Cooler: Both have strong hands
    if hero_cat >= 2 and villain_cat > hero_cat:
        return f"Cooler ({get_hand_category_name(hero_cat)} vs {get_hand_category_name(villain_cat)})"

    # Check all-street aggression
    all_hero_bets = [a for a in history.actions if a.player == 0 and a.action >= 2]
    if all_hero_bets and villain_cat > hero_cat:
        return f"Overplayed Hand ({get_hand_category_name(hero_cat)} vs {get_hand_category_name(villain_cat)})"

    return f"Other ({get_hand_category_name(hero_cat)} vs {get_hand_category_name(villain_cat)})"

# ============================================================================
# Main Analysis
# ============================================================================

def find_latest_model(data_dir: str = "/tmp/full_hunl_dcfr") -> Optional[str]:
    """Find the most recent model checkpoint."""
    models_dir = Path(data_dir) / "models"
    if not models_dir.exists():
        return None

    model_files = list(models_dir.glob("model_*.pt"))
    if not model_files:
        return None

    # Sort by modification time, get latest
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(model_files[0])

def load_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    network = AdvantageNetworkFull(state_dim=STATE_DIM, num_actions=TARGET_DIM, hidden_dim=512)
    network.load_state_dict(checkpoint['network_state_dict'])
    network.to(device)
    network.eval()

    print(f"Loaded model: {model_path}")
    if 'metadata' in checkpoint:
        meta = checkpoint['metadata']
        print(f"  Epoch: {meta.get('epoch', '?')}")
        print(f"  Loss: {meta.get('loss', '?')}")

    return network

def play_hand(network, device, baseline, hand_num: int) -> HandHistory:
    """Play single hand and return history."""
    # Create game with training flop
    game = RustFullHoldem(
        stacks=[100.0, 100.0],
        small_blind=0.5,
        big_blind=1.0,
        fixed_flop=TRAINING_FLOP,
    )

    # Deal
    game = game.apply_action(0)

    history = HandHistory(
        hand_num=hand_num,
        hero_cards=list(game.hands[0]),
        villain_cards=list(game.hands[1]),
        board=[],
    )

    while not game.is_terminal():
        current_player = game.current_player()
        if current_player == -1:
            # Chance node - deal next street
            game = game.apply_action(0)
            continue

        legal_actions = list(game.legal_actions())
        pot_before = game.pot
        stack_before = game.stacks[current_player]

        if len(legal_actions) == 1:
            action = legal_actions[0]
        elif current_player == 0:
            # Hero (our agent)
            state_enc = encode_state(game, current_player)
            state_tensor = torch.from_numpy(state_enc).unsqueeze(0).to(device)
            with torch.no_grad():
                advantages = network(state_tensor)[0].cpu().numpy()
            strategy = regret_matching(advantages, legal_actions)
            action = legal_actions[np.random.choice(len(legal_actions), p=strategy)]

            # Debug: Log preflop all-ins
            if game.street == 0 and action == 7:  # Preflop All-In
                hero_cards = cards_to_string(game.hands[0])
                print(f"\n[DEBUG] Hand #{hand_num} Preflop All-In with {hero_cards}")
                print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
                print(f"  Legal: {legal_actions}, Strategy: {[f'{s:.3f}' for s in strategy]}")
                print(f"  Max adv: {np.argmax(advantages)} = {advantages[np.argmax(advantages)]:.3f}")
        else:
            # Villain (CallingStation)
            action = baseline.get_action(game)

        game = game.apply_action(action)

        history.add_action(
            street=game.street,
            player=current_player,
            action=action,
            pot_before=pot_before,
            pot_after=game.pot,
            stack_before=stack_before,
            stack_after=game.stacks[current_player],
        )

    # Record final results
    returns = game.returns()
    history.hero_result = returns[0]
    history.villain_result = returns[1]
    history.board = list(game.board)

    # Get hand categories
    if len(game.board) == 5:
        history.hero_hand_category = game.get_hand_strength(0)
        history.villain_hand_category = game.get_hand_strength(1)

    return history

def analyze_losses(num_hands: int = 1000, loss_threshold: float = 20.0):
    """Main analysis function."""
    print("=" * 60)
    print("Aion-26 CallingStation Loss Analysis")
    print("=" * 60)
    print(f"Training Flop: {cards_to_string(TRAINING_FLOP)}")
    print(f"Hands to play: {num_hands}")
    print(f"Loss threshold: {loss_threshold} BB")
    print()

    # Find and load model
    model_path = find_latest_model()
    if model_path is None:
        print("ERROR: No model found in /tmp/full_hunl_dcfr/models/")
        print("Run training first or specify model path.")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    network = load_model(model_path, device)
    baseline = CallingStationBot()

    # Play hands and collect big losses
    results = []
    big_losses = []
    loss_categories = defaultdict(list)

    print(f"\nPlaying {num_hands} hands...")

    for i in range(num_hands):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_hands}")

        history = play_hand(network, device, baseline, i + 1)
        results.append(history.hero_result)

        # Check for big loss
        if history.hero_result < -loss_threshold:
            history.loss_category = categorize_loss(history)
            big_losses.append(history)
            loss_categories[history.loss_category].append(history)

    # Summary statistics
    total_result = sum(results)
    avg_result = total_result / num_hands
    mbb_per_hand = (avg_result / 2.0) * 1000  # Convert to mbb/h

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total hands:     {num_hands}")
    print(f"Total result:    {total_result:+.1f} BB")
    print(f"Average result:  {avg_result:+.2f} BB/hand")
    print(f"Win rate:        {mbb_per_hand:+.0f} mbb/h")
    print()
    print(f"Big losses (>{loss_threshold} BB): {len(big_losses)}")
    print()

    # Loss category breakdown
    print("LOSS CATEGORIES:")
    print("-" * 40)
    for category, hands in sorted(loss_categories.items(), key=lambda x: -len(x[1])):
        total_loss = sum(h.hero_result for h in hands)
        print(f"  {category}: {len(hands)} hands ({total_loss:+.1f} BB)")

    # Print detailed hand histories for big losses
    if big_losses:
        print("\n" + "=" * 60)
        print("BIG LOSS HAND HISTORIES")
        print("=" * 60)

        # Sort by loss amount (most negative first)
        big_losses.sort(key=lambda h: h.hero_result)

        # Print worst 10 hands
        for history in big_losses[:10]:
            history.print_history()

    # Analysis recommendations
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    bad_bluffs = sum(1 for h in big_losses if 'Bad Bluff' in h.loss_category)
    value_own_goals = sum(1 for h in big_losses if 'Value Own-Goal' in h.loss_category)
    coolers = sum(1 for h in big_losses if 'Cooler' in h.loss_category)
    overplayed = sum(1 for h in big_losses if 'Overplayed' in h.loss_category)

    if bad_bluffs > len(big_losses) * 0.3:
        print("PROBLEM: Excessive bluffing into CallingStation")
        print("  - CallingStation never folds, so bluffs have 0% fold equity")
        print("  - Agent should check weak hands instead of betting")

    if value_own_goals > len(big_losses) * 0.3:
        print("PROBLEM: Value betting into better hands")
        print("  - Agent betting medium-strength hands into calling range")
        print("  - Need better hand reading / pot control")

    if coolers > len(big_losses) * 0.5:
        print("NOTE: Many cooler situations (strong vs stronger)")
        print("  - These are often unavoidable")
        print("  - Check bet sizing in these spots")

    if overplayed > len(big_losses) * 0.3:
        print("PROBLEM: Overplaying marginal hands")
        print("  - Agent putting in too much money with mediocre holdings")

    print()
    print("Done.")

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze losses against CallingStation")
    parser.add_argument("--hands", type=int, default=1000, help="Number of hands to play")
    parser.add_argument("--threshold", type=float, default=20.0, help="Loss threshold in BB")

    args = parser.parse_args()

    analyze_losses(num_hands=args.hands, loss_threshold=args.threshold)
