#!/usr/bin/env python3
"""
DEBUG: Why is agent losing to CallingStation?

This script plays hands against CallingStation and logs:
1. Network advantages (raw output)
2. Resulting strategy (after regret matching)
3. Actions taken by agent
4. Showdown outcomes and hand strengths

Hypothesis: Agent is bluffing too much (betting/raising with weak hands)
"""

import sys
from pathlib import Path

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
        except Exception as e:
            print(f"[ERROR] Could not load Rust module: {e}")
            sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

from aion26_rust import RustRiverHoldem
from aion26.baselines import CallingStationBot

# Constants
STATE_DIM = 136
ACTION_DIM = 4
ACTION_NAMES = ['Fold', 'Call', 'Raise', 'AllIn']
HAND_CATEGORIES = ['High Card', 'Pair', 'Two Pair', 'Trips', 'Straight',
                   'Flush', 'Full House', 'Quads', 'Str Flush', 'Royal']


class AdvantageNetwork(nn.Module):
    """Same network architecture as train_webapp_pro.py"""
    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = ACTION_DIM, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def encode_state(game_state, player: int) -> np.ndarray:
    """Encode game state - MUST MATCH train_webapp_pro.py"""
    features = np.zeros(STATE_DIM, dtype=np.float32)

    hands = game_state.hands
    board = game_state.board
    pot = game_state.pot
    stacks = game_state.stacks
    current_bet = game_state.current_bet
    invested = [game_state.player_0_invested, game_state.player_1_invested]

    # 1. Hand category ONE-HOT (10 dims)
    try:
        hand_category = game_state.get_hand_strength(player)
        if 0 <= hand_category < 10:
            features[hand_category] = 1.0
    except:
        features[0] = 1.0

    # 2. Hole cards (34 dims)
    if len(hands) > player:
        hand = hands[player]
        for i, card in enumerate(hand[:2]):
            offset = 10 + i * 17
            rank = card % 13
            suit = card // 13
            if rank < 13:
                features[offset + rank] = 1.0
            if suit < 4:
                features[offset + 13 + suit] = 1.0

    # 3. Board cards (85 dims)
    for i, card in enumerate(board[:5]):
        offset = 10 + 34 + i * 17
        rank = card % 13
        suit = card // 13
        if rank < 13:
            features[offset + rank] = 1.0
        if suit < 4:
            features[offset + 13 + suit] = 1.0

    # 4. Betting context (7 dims)
    total_stack = sum(stacks) + pot
    if total_stack > 0:
        features[129] = pot / total_stack
        features[130] = stacks[player] / total_stack
        features[131] = stacks[1-player] / total_stack
        features[132] = current_bet / total_stack if total_stack > 0 else 0
        features[133] = invested[player] / total_stack
        features[134] = invested[1-player] / total_stack
        features[135] = 1.0 if player == 0 else 0.0

    return features


def regret_matching(advantages: np.ndarray, legal_actions: list) -> np.ndarray:
    """Convert advantages to strategy via regret matching.

    CRITICAL FIX: When all advantages are negative, use argmax to pick
    the "least bad" action instead of uniform random (which causes
    suicidal bluffs like All-In with trash hands).
    """
    legal_advantages = np.array([advantages[a] if a < len(advantages) else 0.0 for a in legal_actions])
    positive = np.maximum(legal_advantages, 0)
    total = positive.sum()
    if total > 0:
        return positive / total
    else:
        # Argmax fallback: pick the action with highest (least negative) advantage
        best_idx = np.argmax(legal_advantages)
        result = np.zeros(len(legal_actions))
        result[best_idx] = 1.0
        return result


def load_latest_model(model_dir: str = "/tmp/vr_dcfr_pro/models"):
    """Load most recent model checkpoint."""
    model_path = Path(model_dir)
    if not model_path.exists():
        return None

    checkpoints = list(model_path.glob("*.pt"))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Loading model: {latest}")

    network = AdvantageNetwork()
    checkpoint = torch.load(latest, map_location='cpu')
    network.load_state_dict(checkpoint['network_state_dict'])
    network.eval()

    return network


def debug_single_hand(network, device, hand_num: int):
    """Play one hand with detailed logging."""
    game = RustRiverHoldem(
        stacks=[100.0, 100.0],
        pot=2.0,
        current_bet=0.0,
        player_0_invested=1.0,
        player_1_invested=1.0,
    )
    game = game.apply_action(0)  # Deal

    baseline = CallingStationBot()

    # Get hand info
    p0_category = game.get_hand_strength(0)
    p1_category = game.get_hand_strength(1)

    print(f"\n{'='*60}")
    print(f"HAND #{hand_num}")
    print(f"{'='*60}")
    print(f"P0 (Agent) hand category: {p0_category} ({HAND_CATEGORIES[p0_category]})")
    print(f"P1 (Station) hand category: {p1_category} ({HAND_CATEGORIES[p1_category]})")
    print(f"P0 cards: {game.hands[0]}")
    print(f"P1 cards: {game.hands[1]}")
    print(f"Board: {game.board}")

    actions_taken = []

    while not game.is_terminal():
        current_player = game.current_player()
        if current_player == -1:
            break

        legal_actions = game.legal_actions()

        if current_player == 0:
            # Our agent
            state_enc = encode_state(game, current_player)
            state_tensor = torch.from_numpy(state_enc).unsqueeze(0).to(device)

            with torch.no_grad():
                advantages = network(state_tensor)[0].cpu().numpy()

            strategy = regret_matching(advantages, legal_actions)
            action = legal_actions[np.random.choice(len(legal_actions), p=strategy)]

            print(f"\n  P0 turn (Agent):")
            print(f"    Legal actions: {[ACTION_NAMES[a] for a in legal_actions]}")
            print(f"    Network advantages: {dict(zip([ACTION_NAMES[a] for a in legal_actions], [f'{advantages[a]:.4f}' for a in legal_actions]))}")
            print(f"    Strategy: {dict(zip([ACTION_NAMES[a] for a in legal_actions], [f'{s:.2%}' for s in strategy]))}")
            print(f"    Chose: {ACTION_NAMES[action]}")

            actions_taken.append(('Agent', ACTION_NAMES[action]))
        else:
            # CallingStation
            action = baseline.get_action(game)
            print(f"\n  P1 turn (CallingStation):")
            print(f"    Legal actions: {[ACTION_NAMES[a] for a in legal_actions]}")
            print(f"    Chose: {ACTION_NAMES[action]}")

            actions_taken.append(('Station', ACTION_NAMES[action]))

        game = game.apply_action(action)

    # Results
    returns = game.returns()
    print(f"\n  RESULT:")
    print(f"    History: {game.history}")
    print(f"    P0 (Agent) return: {returns[0]:+.2f} BB")
    print(f"    P1 (Station) return: {returns[1]:+.2f} BB")

    if 'f' in game.history:
        print(f"    Ended by: FOLD")
    else:
        print(f"    Ended by: SHOWDOWN")
        if returns[0] > 0:
            print(f"    Winner: Agent (better hand)")
        elif returns[0] < 0:
            print(f"    Winner: CallingStation (better hand)")
        else:
            print(f"    Result: TIE")

    return {
        'p0_category': p0_category,
        'p1_category': p1_category,
        'return': returns[0],
        'actions': actions_taken,
        'ended_fold': 'f' in game.history,
    }


def main():
    print("="*60)
    print("DEBUG: Agent vs CallingStation")
    print("="*60)

    device = torch.device('cpu')
    network = load_latest_model()

    if network is None:
        print("[ERROR] No model found. Run training first.")
        return

    network = network.to(device)

    # Play some hands
    NUM_HANDS = 20
    results = []

    for i in range(NUM_HANDS):
        result = debug_single_hand(network, device, i+1)
        results.append(result)

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total_return = sum(r['return'] for r in results)
    print(f"Total return: {total_return:+.2f} BB over {NUM_HANDS} hands")
    print(f"Average: {total_return/NUM_HANDS:+.4f} BB/hand = {total_return/NUM_HANDS*1000:+.2f} mbb/h")

    # Action distribution
    action_counts = defaultdict(int)
    for r in results:
        for actor, action in r['actions']:
            if actor == 'Agent':
                action_counts[action] += 1

    total_actions = sum(action_counts.values())
    print(f"\nAgent action distribution:")
    for action in ACTION_NAMES:
        count = action_counts[action]
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action}: {count} ({pct:.1f}%)")

    # Win rate by hand strength
    print(f"\nResults by agent hand strength:")
    for cat in range(10):
        cat_results = [r for r in results if r['p0_category'] == cat]
        if cat_results:
            avg_return = sum(r['return'] for r in cat_results) / len(cat_results)
            print(f"  {HAND_CATEGORIES[cat]}: {len(cat_results)} hands, avg return {avg_return:+.2f} BB")

    # Bluff detection
    bluffs = [r for r in results if r['p0_category'] <= 1]  # High card or pair
    bluff_bets = sum(1 for r in bluffs for actor, action in r['actions']
                    if actor == 'Agent' and action in ['Raise', 'AllIn'])

    print(f"\nBLUFF ANALYSIS:")
    print(f"  Hands with weak hands (high card/pair): {len(bluffs)}")
    print(f"  Times agent bet/raised with weak hand: {bluff_bets}")
    if bluffs:
        print(f"  This is COSTLY vs CallingStation who never folds!")


if __name__ == "__main__":
    main()
