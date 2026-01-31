#!/usr/bin/env python3
"""Production-Ready Training Dashboard for VR-Deep PDCFR+.

Features (Option A - Quick Win):
- Real-time training visualization
- Strategy inspector (view learned policies)
- Model save/load management
- Baseline evaluation suite (RandomBot, CallingStation, AlwaysFold)
- Structured logging with proper error handling
- Clean API endpoints

Run with: python scripts/train_webapp_pro.py
Then open: http://localhost:5001
"""

import sys
import os
from pathlib import Path

# Windows DLL loading fix: Load .pyd directly with ctypes + importlib
if sys.platform == 'win32':
    import ctypes
    import importlib.util

    # Find the .pyd file in site-packages
    site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "aion26_rust"
    pyd_file = site_packages / "aion26_rust.cp312-win_amd64.pyd"

    if pyd_file.exists():
        try:
            # Pre-load with ctypes first
            ctypes.CDLL(str(pyd_file))

            # Then load as module using importlib
            spec = importlib.util.spec_from_file_location('aion26_rust', str(pyd_file))
            _rust_module = importlib.util.module_from_spec(spec)
            sys.modules['aion26_rust'] = _rust_module
            spec.loader.exec_module(_rust_module)
            print(f"[DLL] Loaded: {pyd_file}")
        except Exception as e:
            print(f"[DLL] Warning: Could not load {pyd_file}: {e}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import threading
import logging

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Annotated
from datetime import datetime
import json

import torch
import torch.nn as nn
# CODE RED: AMP DISABLED for debugging - force float32 everywhere
# from torch.cuda.amp import autocast, GradScaler
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO

# Rich and Typer for CLI and logging
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# OPTIMIZED: Enable TF32 for faster GPU computation on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
torch.set_default_dtype(torch.float32)

from aion26_rust import ParallelTrainer, RustRiverHoldem, ParallelTrainerFull, RustFullHoldem
from aion26.baselines import RandomBot, CallingStationBot, AlwaysFoldBot

# ============================================================================
# Game Mode Configuration
# ============================================================================

# Global game mode flag - set via CLI
GAME_MODE = "river"  # "river" or "full"
APP_START_TIME = time.time()

def get_state_dim():
    return 220 if GAME_MODE == "full" else 136

def get_target_dim():
    return 8 if GAME_MODE == "full" else 4

def get_action_names():
    if GAME_MODE == "full":
        return ['Fold', 'Check/Call', 'Bet 0.5x', 'Bet 0.75x', 'Bet Pot', 'Bet 1.5x', 'Bet 2x', 'All-In']
    else:
        return ['Fold', 'Call', 'Raise', 'All-In']

# ============================================================================
# Solver Database - Export solved flops for reuse
# ============================================================================

SOLVER_DB_PATH = Path("E:/solver_data")
FORCE_RETRAIN = False  # Set via --force CLI flag

def get_flop_key(flop_cards: list) -> str:
    """Generate a canonical key for a flop (sorted cards)."""
    if not flop_cards or len(flop_cards) != 3:
        return "random"
    sorted_flop = sorted(flop_cards)
    return f"flop_{sorted_flop[0]}_{sorted_flop[1]}_{sorted_flop[2]}"

def get_config_hash(config) -> str:
    """Generate a hash of training config for comparison."""
    import hashlib
    config_str = f"{config.epochs}_{config.traversals_per_epoch}_{config.learning_rate}_{config.hidden_dim if hasattr(config, 'hidden_dim') else 512}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def is_flop_solved(flop_cards: list, config) -> tuple[bool, Optional[Path]]:
    """Check if a flop has already been solved with compatible config."""
    if not SOLVER_DB_PATH.exists():
        return False, None

    flop_key = get_flop_key(flop_cards)
    flop_dir = SOLVER_DB_PATH / flop_key

    if not flop_dir.exists():
        return False, None

    # Check for config file
    config_file = flop_dir / "config.json"
    if not config_file.exists():
        return False, None

    try:
        with open(config_file, 'r') as f:
            saved_config = json.load(f)

        # Check if config matches (epochs and key params)
        if (saved_config.get('epochs', 0) >= config.epochs and
            saved_config.get('traversals_per_epoch', 0) >= config.traversals_per_epoch):
            model_file = flop_dir / "model.pt"
            if model_file.exists():
                return True, model_file
    except Exception as e:
        print(f"Error reading solver config: {e}")

    return False, None

def export_solved_flop(flop_cards: list, config, network, metrics: dict):
    """Export a solved flop to the solver database."""
    SOLVER_DB_PATH.mkdir(parents=True, exist_ok=True)

    flop_key = get_flop_key(flop_cards)
    flop_dir = SOLVER_DB_PATH / flop_key
    flop_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = flop_dir / "model.pt"
    torch.save({
        'network_state_dict': network.state_dict(),
        'flop_cards': flop_cards,
    }, model_path)

    # Save config
    config_data = {
        'flop_cards': flop_cards,
        'flop_key': flop_key,
        'epochs': config.epochs,
        'traversals_per_epoch': config.traversals_per_epoch,
        'learning_rate': config.learning_rate,
        'train_batch_size': config.train_batch_size,
        'train_steps_per_epoch': config.train_steps_per_epoch,
        'small_blind': config.small_blind,
        'big_blind': config.big_blind,
        'starting_stack': config.starting_stack,
        'final_loss': metrics.get('loss', 0),
        'final_win_rate': metrics.get('win_rate', 0),
        'total_samples': metrics.get('total_samples', 0),
        'exported_at': datetime.now().isoformat(),
        'config_hash': get_config_hash(config),
    }

    config_path = flop_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"[SOLVER DB] Exported {flop_key} to {flop_dir}")
    return flop_dir

def list_solved_flops() -> list:
    """List all solved flops in the database."""
    if not SOLVER_DB_PATH.exists():
        return []

    solved = []
    for flop_dir in SOLVER_DB_PATH.iterdir():
        if flop_dir.is_dir() and (flop_dir / "config.json").exists():
            try:
                with open(flop_dir / "config.json", 'r') as f:
                    config = json.load(f)
                solved.append(config)
            except:
                pass
    return solved

# ============================================================================
# Logging Setup with Rich
# ============================================================================

# Global console and verbosity setting
console = Console()
VERBOSE = False  # Set via CLI

def setup_logging(verbose: bool = False):
    """Configure logging with Rich handler."""
    global VERBOSE
    VERBOSE = verbose

    level = logging.DEBUG if verbose else logging.INFO

    # Remove existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Add Rich handler
    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=verbose,
        )]
    )

    # Suppress noisy loggers unless verbose
    if not verbose:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('engineio').setLevel(logging.WARNING)
        logging.getLogger('socketio').setLevel(logging.WARNING)

# Default setup (can be overridden by CLI)
setup_logging(verbose=False)
logger = logging.getLogger(__name__)


def ts() -> str:
    """Get current timestamp string for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# Configuration
# ============================================================================

# These are now computed dynamically based on GAME_MODE
def get_record_size():
    return (get_state_dim() + get_target_dim()) * 4

def get_data_dir():
    """Get data directory based on game mode."""
    if GAME_MODE == "full":
        return "/tmp/full_hunl_dcfr"
    return "/tmp/vr_dcfr_pro"


@dataclass
class TrainConfig:
    epochs: int = 150  # Increased for better convergence with LR decay
    traversals_per_epoch: int = 200_000       # Original value
    num_workers: int = 2048                   # Original value
    query_buffer_size: int = 16384            # Original value
    train_batch_size: int = 16384             # Original value
    train_steps_per_epoch: int = 200          # Original value
    learning_rate: float = 5e-4               # Original value
    lr_end: float = 1e-5                      # Final learning rate (cosine annealing)
    polyak_tau: float = 0.005
    history_alpha: float = 0.5
    eval_interval: int = 2  # Run win rate eval every 2 epochs
    eval_hands: int = 100_000  # 100K hands (~100 sec per eval)
    data_dir: str = ""  # Set dynamically based on game mode
    # Full HUNL specific
    small_blind: float = 0.5
    big_blind: float = 1.0
    starting_stack: float = 100.0
    # Single-flop training mode (for debugging or focused training)
    fixed_flop: list = None  # If set, train only on this flop (list of 3 card indices 0-51)
                             # Example: [12, 25, 38] = Ac, Kd, Ah (A-K-A rainbow)
    # EXPLORATION: Temperature annealing and entropy regularization
    temp_start: float = 2.0           # High temp early = more exploration
    temp_end: float = 1.0             # Final temperature
    temp_anneal_epochs: int = 20      # Epochs to anneal from start to end
    entropy_coef: float = 0.01        # Entropy bonus coefficient ("Don't Be A Nit" tax)
    # FORCED AGGRESSION: DISABLED - breaks CFR convergence
    # These were added to fix passivity but corrupt training data
    aggression_force: float = 0.0     # DISABLED: Initial forced aggression rate
    aggression_end: float = 0.0       # DISABLED: Final forced aggression rate
    aggression_decay_epochs: int = 50 # Epochs to decay aggression
    aggression_boost: float = 2.0     # Boost added to bet/raise advantages when forcing
    prefer_large_bet: float = 0.5     # 50% chance to prefer largest bet size when forcing
    # AWR LOSS: Asymmetric weighting - prioritize learning positive regrets
    positive_regret_weight: float = 5.0  # Weight for positive regrets (missed opportunities)
    # "I don't care if you lose, but if you miss a winning bet, you're fired"

    def __post_init__(self):
        # Set data_dir dynamically if not already set
        if not self.data_dir:
            self.data_dir = get_data_dir()

# Global config instance - single source of truth
ACTIVE_CONFIG = TrainConfig()

# Global epoch tracker for temperature annealing
CURRENT_EPOCH = 0


def get_temperature() -> float:
    """Get current temperature based on epoch for exploration annealing.

    High temperature early = more exploration (try "bad" actions like overbets).
    Decays linearly to 1.0 over temp_anneal_epochs.
    """
    config = ACTIVE_CONFIG
    if CURRENT_EPOCH >= config.temp_anneal_epochs:
        return config.temp_end

    # Linear decay from temp_start to temp_end
    progress = CURRENT_EPOCH / config.temp_anneal_epochs
    return config.temp_start + (config.temp_end - config.temp_start) * progress


def get_aggression_rate() -> float:
    """Get current aggression forcing rate based on epoch.

    High aggression early = force exploration of aggressive lines.
    Decays linearly after model has learned to be aggressive.
    """
    config = ACTIVE_CONFIG
    if CURRENT_EPOCH >= config.aggression_decay_epochs:
        return config.aggression_end

    # Linear decay from aggression_force to aggression_end
    progress = CURRENT_EPOCH / config.aggression_decay_epochs
    return config.aggression_force + (config.aggression_end - config.aggression_force) * progress


def print_active_config():
    """CODE RED: Print actual config values being used in training."""
    print("\n" + "="*60)
    print("CODE RED: ACTIVE CONFIGURATION DUMP")
    print("="*60)
    print(f"  epochs:              {ACTIVE_CONFIG.epochs}")
    print(f"  traversals_per_epoch: {ACTIVE_CONFIG.traversals_per_epoch:,}")
    print(f"  num_workers:         {ACTIVE_CONFIG.num_workers:,}")
    print(f"  query_buffer_size:   {ACTIVE_CONFIG.query_buffer_size:,}")
    print(f"  train_batch_size:    {ACTIVE_CONFIG.train_batch_size:,}")  # <-- THE SUSPECT
    print(f"  train_steps_per_epoch: {ACTIVE_CONFIG.train_steps_per_epoch}")
    print(f"  learning_rate:       {ACTIVE_CONFIG.learning_rate}")
    print(f"  polyak_tau:          {ACTIVE_CONFIG.polyak_tau}")
    print(f"  history_alpha:       {ACTIVE_CONFIG.history_alpha}")
    print(f"  eval_interval:       {ACTIVE_CONFIG.eval_interval}")
    print(f"  data_dir:            {ACTIVE_CONFIG.data_dir}")
    if ACTIVE_CONFIG.fixed_flop:
        print(f"  fixed_flop:          {ACTIVE_CONFIG.fixed_flop} (SINGLE-FLOP MODE)")
    else:
        print(f"  fixed_flop:          None (random flops)")
    print(f"  temp_start:          {ACTIVE_CONFIG.temp_start} (exploration temperature)")
    print(f"  temp_end:            {ACTIVE_CONFIG.temp_end}")
    print(f"  temp_anneal_epochs:  {ACTIVE_CONFIG.temp_anneal_epochs}")
    print(f"  entropy_coef:        {ACTIVE_CONFIG.entropy_coef} (Don't Be A Nit tax)")
    print(f"  aggression_force:    {ACTIVE_CONFIG.aggression_force} -> {ACTIVE_CONFIG.aggression_end} over {ACTIVE_CONFIG.aggression_decay_epochs} epochs")
    print(f"  aggression_boost:    {ACTIVE_CONFIG.aggression_boost} (boost for bet/raise)")
    print(f"  prefer_large_bet:    {ACTIVE_CONFIG.prefer_large_bet} (prefer max bet size)")
    print(f"  positive_regret_wt:  {ACTIVE_CONFIG.positive_regret_weight} (AWR: upside > safety)")
    print("="*60 + "\n")
    logger.info(f"CONFIG: batch_size={ACTIVE_CONFIG.train_batch_size}, lr={ACTIVE_CONFIG.learning_rate}, tau={ACTIVE_CONFIG.polyak_tau}")

def sanity_check_tensor(tensor: torch.Tensor, name: str, batch_num: int = -1):
    """CODE RED: Verify tensor integrity - check for NaN/Inf and log stats."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        raise ValueError(f"CRITICAL: {name} contains NaN={has_nan}, Inf={has_inf} at batch {batch_num}")

    # Log stats for first 5 batches
    if 0 <= batch_num < 5:
        print(f"[SANITY] {name} batch={batch_num}: shape={tuple(tensor.shape)}, "
              f"dtype={tensor.dtype}, mean={tensor.mean().item():.4f}, "
              f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

# ============================================================================
# Utility Functions
# ============================================================================

def polyak_update(target_net: nn.Module, online_net: nn.Module, tau: float):
    """Soft update: target = tau * online + (1 - tau) * target"""
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

def load_epoch_data(data_dir: str, epoch: int) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from binary file."""
    path = Path(data_dir) / f"epoch_{epoch}.bin"
    if not path.exists():
        return np.array([]), np.array([])

    state_dim = get_state_dim()
    target_dim = get_target_dim()
    record_size = get_record_size()

    file_size = path.stat().st_size
    num_samples = file_size // record_size

    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(num_samples, state_dim + target_dim)

    states = data[:, :state_dim]
    targets = data[:, state_dim:]

    return states, targets

def card_str(card: int) -> str:
    """Convert card index to string."""
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks[card % 13] + suits[card // 13]

# ============================================================================
# Network
# ============================================================================

class AdvantageNetwork(nn.Module):
    """Advantage network for River-only (136 dims, 4 actions)."""
    def __init__(self, state_dim: int = 136, num_actions: int = 4, hidden_dim: int = 256):
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


class AdvantageNetworkFull(nn.Module):
    """Larger advantage network for Full HUNL (220 dims, 8 actions)."""
    def __init__(self, state_dim: int = 220, num_actions: int = 8, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def create_network(device):
    """Create appropriate network based on game mode."""
    if GAME_MODE == "full":
        return AdvantageNetworkFull(
            state_dim=get_state_dim(),
            num_actions=get_target_dim(),
            hidden_dim=512
        ).to(device)
    else:
        return AdvantageNetwork(
            state_dim=get_state_dim(),
            num_actions=get_target_dim(),
            hidden_dim=256
        ).to(device)

# ============================================================================
# Preflop Hand Strength (Chen Formula)
# ============================================================================

def chen_formula(card1: int, card2: int) -> float:
    """Calculate Chen formula score for preflop hand strength.

    Returns normalized score 0-1 (1 = best like AA, 0 = worst like 72o).
    Used to prevent Suicide Shove bug where model goes all-in preflop with garbage.
    """
    rank1 = card1 % 13  # 0=2, 1=3, ..., 12=A
    rank2 = card2 % 13
    suit1 = card1 // 13
    suit2 = card2 // 13

    def chen_rank(r):
        """Convert rank to Chen point value."""
        if r == 12: return 10    # Ace
        elif r == 11: return 8   # King
        elif r == 10: return 7   # Queen
        elif r == 9: return 6    # Jack
        else: return (r + 2) / 2  # 2-10 get half their face value

    high, low = max(rank1, rank2), min(rank1, rank2)
    score = chen_rank(high)

    # Pair bonus: double the score, minimum 5
    if rank1 == rank2:
        score = max(score * 2, 5)

    # Suited bonus: +2 for suited hands
    if suit1 == suit2:
        score += 2

    # Gap penalty: deduct points for gaps between cards
    gap = high - low - 1
    if gap == 1:
        score -= 1
    elif gap == 2:
        score -= 2
    elif gap == 3:
        score -= 4
    elif gap >= 4:
        score -= 5

    # Connector bonus: +1 for no gap if both cards < Q
    if gap <= 0 and high < 10:
        score += 1

    # Normalize to 0-1 range (Chen scores range from -1 to 20)
    return max(0.0, min(1.0, (score + 1) / 21.0))


def preflop_hand_strength(hand: list) -> tuple:
    """Get (chen_score, suited_indicator) for preflop hand.

    Returns:
        chen_score: 0-1 normalized Chen formula score
        suited: 1.0 if suited, 0.0 if offsuit
    """
    if len(hand) < 2:
        return 0.5, 0.0
    return chen_formula(hand[0], hand[1]), 1.0 if (hand[0] // 13) == (hand[1] // 13) else 0.0


# ============================================================================
# State Encoder
# ============================================================================

def canonicalize_suits(cards: list) -> dict:
    """Canonicalize suits for isomorphic state representation.

    Maps suits to canonical form (0,1,2,3) based on order of first appearance.
    This is a LOSSLESS compression - Ah Kh 2s becomes equivalent to Ad Kd 2c.
    MUST MATCH RUST canonicalize_suits() exactly!
    """
    suit_map = {0: -1, 1: -1, 2: -1, 3: -1}
    next_canonical = 0

    for card in cards:
        suit = card // 13
        if suit < 4 and suit_map[suit] == -1:
            suit_map[suit] = next_canonical
            next_canonical += 1

    # Fill remaining unmapped suits
    for s in range(4):
        if suit_map[s] == -1:
            suit_map[s] = next_canonical
            next_canonical += 1

    return suit_map


def encode_state(game_state, player: int) -> np.ndarray:
    """Encode game state for neural network.

    MUST MATCH RUST encode_state() exactly!
    Includes SUIT ISOMORPHISM for lossless state compression.
    River mode (136 dims):
      [0-9]:   Hand category one-hot (10 dims)
      [10-43]: Hole cards (2 * 17 = 34 dims)
      [44-128]: Board cards (5 * 17 = 85 dims)
      [129-135]: Betting context (7 dims)
    Full HUNL mode (220 dims):
      Additional street/action history encoding
    """
    features = np.zeros(get_state_dim(), dtype=np.float32)

    hands = game_state.hands
    board = game_state.board
    pot = game_state.pot
    stacks = game_state.stacks
    current_bet = game_state.current_bet

    # Handle different attribute names between RustRiverHoldem and RustFullHoldem
    if hasattr(game_state, 'player_0_invested'):
        # RustRiverHoldem
        invested = [game_state.player_0_invested, game_state.player_1_invested]
    elif hasattr(game_state, 'invested_total'):
        # RustFullHoldem
        invested = game_state.invested_total
    else:
        invested = [0.0, 0.0]

    # SUIT ISOMORPHISM: Build canonical suit mapping from all visible cards
    # Order: player's hole cards first, then board cards (MUST MATCH RUST!)
    hand = hands[player] if len(hands) > player else []
    all_cards = list(hand[:2]) + list(board[:5])
    suit_map = canonicalize_suits(all_cards)

    # 1. Hand category ONE-HOT (10 dims) - MUST MATCH RUST!
    # Uses actual suits for flush detection (game logic unchanged)
    try:
        hand_category = game_state.get_hand_strength(player)
        if 0 <= hand_category < 10:
            features[hand_category] = 1.0  # One-hot encoding!
    except:
        features[0] = 1.0  # Default to high card

    # 2. Hole cards (34 dims) - use CANONICAL suits
    if len(hands) > player:
        hand = hands[player]
        for i, card in enumerate(hand[:2]):
            offset = 10 + i * 17
            rank = card % 13
            actual_suit = card // 13
            canonical_suit = suit_map.get(actual_suit, actual_suit)
            if rank < 13:
                features[offset + rank] = 1.0
            if canonical_suit < 4:
                features[offset + 13 + canonical_suit] = 1.0

    # 3. Board cards (85 dims) - use CANONICAL suits
    for i, card in enumerate(board[:5]):
        offset = 10 + 34 + i * 17
        rank = card % 13
        actual_suit = card // 13
        canonical_suit = suit_map.get(actual_suit, actual_suit)
        if rank < 13:
            features[offset + rank] = 1.0
        if canonical_suit < 4:
            features[offset + 13 + canonical_suit] = 1.0

    # 4. Betting context (7 dims) - using dynamic normalization
    context_offset = 10 + 34 + 85

    # CRITICAL FIX: Use dynamic normalization based on total money in play
    # This ensures features are scale-invariant and consistent between training/inference
    total_money = sum(stacks) + pot
    normalizer = total_money if total_money > 0 else 1.0

    current_invested = invested[player] if len(invested) > player else 0.0
    call_amount = max(0, current_bet - current_invested)
    pot_after_call = pot + call_amount
    pot_odds = call_amount / pot_after_call if pot_after_call > 0 else 0.0

    features[context_offset] = pot / normalizer
    features[context_offset + 1] = stacks[0] / normalizer if len(stacks) > 0 else 0.0
    features[context_offset + 2] = stacks[1] / normalizer if len(stacks) > 1 else 0.0
    features[context_offset + 3] = current_bet / normalizer
    features[context_offset + 4] = invested[0] / normalizer if len(invested) > 0 else 0.0
    features[context_offset + 5] = invested[1] / normalizer if len(invested) > 1 else 0.0
    features[context_offset + 6] = pot_odds

    # FLOP ABSTRACTION: Board texture bucket for full HUNL mode
    # Must match Rust compute_board_texture_bucket() exactly
    if GAME_MODE == "full" and len(board) >= 3:
        texture_bucket = compute_board_texture_bucket(board[:3])
        # In full mode, context_offset is 197 and texture bucket is at +12
        features[197 + 12] = texture_bucket / 200.0

    # PREFLOP HAND STRENGTH: Chen formula features to prevent Suicide Shove bug
    # These features help the network understand raw preflop equity
    # Added at indices 197+13 and 197+14 (within 23-dim betting context for full mode)
    if GAME_MODE == "full" and len(hands) > player and len(hands[player]) >= 2:
        chen_score, suited = preflop_hand_strength(list(hands[player]))
        features[197 + 13] = chen_score   # 0-1 normalized Chen formula score
        features[197 + 14] = suited       # 1.0 if suited, 0.0 if offsuit

    return features


def compute_board_texture_bucket(flop_cards) -> int:
    """Compute board texture bucket ID from flop cards.

    Must match Rust compute_board_texture_bucket() exactly!
    Features: suit pattern, connectedness, high card strength, paired status.
    """
    NUM_TEXTURE_BUCKETS = 200

    ranks = sorted([c % 13 for c in flop_cards], reverse=True)
    suits = [c // 13 for c in flop_cards]

    # Feature 1: Suit pattern (0-2)
    unique_suits = len(set(suits))
    if unique_suits == 1:
        suit_feature = 2  # Monotone
    elif unique_suits == 2:
        suit_feature = 1  # Two-tone
    else:
        suit_feature = 0  # Rainbow

    # Feature 2: Connectedness (0-9)
    gap1 = abs(ranks[0] - ranks[1])
    gap2 = abs(ranks[1] - ranks[2])
    connected_feature = min(10 - min(gap1 + gap2, 10), 9)

    # Feature 3: High card (0-3)
    high_feature = min(ranks[0] // 4, 3)

    # Feature 4: Paired (0-1)
    paired_feature = 1 if (ranks[0] == ranks[1] or ranks[1] == ranks[2]) else 0

    # Combine features into bucket ID
    bucket = suit_feature * 80 + connected_feature * 8 + high_feature * 2 + paired_feature
    return bucket % NUM_TEXTURE_BUCKETS

def regret_matching(advantages: np.ndarray, legal_actions: list, use_temperature: bool = True) -> np.ndarray:
    """Convert advantages to strategy via proper CFR regret matching.

    CFR REGRET MATCHING (correct algorithm for Nash equilibrium convergence):
    1. Take only positive regrets (negative regrets contribute 0 probability)
    2. Normalize positive regrets to form a probability distribution
    3. If all regrets are negative/zero, play the least negative action

    This is the ONLY correct way to compute strategies in CFR.
    Softmax does NOT have Nash equilibrium convergence guarantees.

    Args:
        advantages: Raw advantage values from network (cumulative regrets)
        legal_actions: List of legal action indices
        use_temperature: If True, add epsilon exploration during training
    """
    legal_advantages = np.array([advantages[a] if a < len(advantages) else 0.0 for a in legal_actions])

    # PROPER CFR REGRET MATCHING
    # Step 1: Take only positive regrets
    positive_regrets = np.maximum(legal_advantages, 0.0)
    regret_sum = positive_regrets.sum()

    if regret_sum > 1e-10:
        # Step 2: Normalize positive regrets to get strategy
        strategy = positive_regrets / regret_sum
    else:
        # Step 3: All regrets are negative - play best (least negative) action
        strategy = np.zeros(len(legal_actions))
        best_idx = np.argmax(legal_advantages)
        strategy[best_idx] = 1.0

    # Optional: Add epsilon-greedy exploration during training
    # This is theoretically sound (unlike softmax) and helps discover aggressive lines
    if use_temperature:
        epsilon = get_aggression_rate()  # Reuse aggression rate as exploration rate
        if epsilon > 0 and len(legal_actions) > 1:
            # Mix with uniform distribution for exploration
            uniform = np.ones(len(legal_actions)) / len(legal_actions)
            strategy = (1 - epsilon) * strategy + epsilon * uniform

    return strategy

# ============================================================================
# Card Parsing Utilities
# ============================================================================

def parse_card(card_str: str) -> int:
    """Parse a card string like 'Ac', 'Kd', '2h' to card index 0-51.

    Card encoding: suit * 13 + rank
    Ranks: 2=0, 3=1, ..., K=11, A=12
    Suits: c=0, d=1, h=2, s=3
    """
    card_str = card_str.strip().upper()

    # Handle numeric input
    if card_str.isdigit():
        return int(card_str)

    if len(card_str) < 2:
        raise ValueError(f"Invalid card: {card_str}")

    rank_char = card_str[:-1]
    suit_char = card_str[-1].lower()

    # Parse rank
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                '9': 7, 'T': 8, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    rank = rank_map.get(rank_char.upper())
    if rank is None:
        raise ValueError(f"Invalid rank: {rank_char}")

    # Parse suit
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    suit = suit_map.get(suit_char)
    if suit is None:
        raise ValueError(f"Invalid suit: {suit_char}")

    return suit * 13 + rank


def parse_flop_string(flop_str: str) -> list:
    """Parse a flop string like 'Ac,Kd,Qh' or '12,25,38' to list of card indices.

    Examples:
        'Ac,Kd,Qh' -> [12, 24, 36]  (Ace of clubs, King of diamonds, Queen of hearts)
        '12,25,38' -> [12, 25, 38]  (raw indices)
        'As Ks Qs' -> [51, 50, 49]  (space-separated)
    """
    # Split by comma or space
    if ',' in flop_str:
        cards = flop_str.split(',')
    else:
        cards = flop_str.split()

    if len(cards) != 3:
        raise ValueError(f"Flop must have exactly 3 cards, got {len(cards)}: {flop_str}")

    return [parse_card(c) for c in cards]


def card_to_string(card_idx: int) -> str:
    """Convert card index to readable string like 'Ac', 'Kd'."""
    rank = card_idx % 13
    suit = card_idx // 13
    rank_chars = '23456789TJQKA'
    suit_chars = 'cdhs'
    return f"{rank_chars[rank]}{suit_chars[suit]}"


# ============================================================================
# Model Management
# ============================================================================

class ModelRegistry:
    """Simple model registry for checkpoint management."""

    def __init__(self, base_dir: str = "/tmp/vr_dcfr_pro/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model registry initialized at {self.base_dir}")

    def save_model(self, network: nn.Module, optimizer, metadata: dict) -> str:
        """Save model checkpoint with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{timestamp}"
        model_path = self.base_dir / f"{model_id}.pt"

        checkpoint = {
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata,
            'timestamp': timestamp,
        }

        torch.save(checkpoint, model_path)

        # Save metadata separately for easy listing
        meta_path = self.base_dir / f"{model_id}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved: {model_id}")
        return model_id

    def load_model(self, model_id: str, device: str = 'cpu') -> tuple[nn.Module, dict]:
        """Load model checkpoint."""
        model_path = self.base_dir / f"{model_id}.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_id}")

        checkpoint = torch.load(model_path, map_location=device)

        # Use appropriate network based on game mode
        network = create_network(device)
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()

        logger.info(f"Model loaded: {model_id}")
        return network, checkpoint['metadata']

    def list_models(self) -> List[Dict]:
        """List all available models."""
        models = []
        for meta_file in self.base_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                model_id = meta_file.stem.replace('_meta', '')
                models.append({
                    'id': model_id,
                    'metadata': metadata,
                })
            except Exception as e:
                logger.error(f"Error reading metadata for {meta_file}: {e}")

        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['id'], reverse=True)
        return models

# ============================================================================
# Training State
# ============================================================================

@dataclass
class TrainingState:
    running: bool = False
    epoch: int = 0
    step: int = 0
    samples: int = 0
    total_samples: int = 0
    samples_per_sec: float = 0.0
    loss: float = 0.0
    batch_size: int = 0
    elapsed: float = 0.0

    # Win rate tracking
    win_rate_mbb: float = 0.0
    win_rate_history: list = field(default_factory=list)

    # Loss tracking
    loss_history: list = field(default_factory=list)

    # Action distribution
    action_counts: dict = field(default_factory=lambda: {'fold': 0, 'call': 0, 'raise': 0, 'allin': 0})

    # Recent hands log
    recent_hands: list = field(default_factory=list)

    # Throughput history
    throughput_history: list = field(default_factory=list)

    # Baseline evaluations
    baseline_results: dict = field(default_factory=dict)

    # Strategy samples
    strategy_samples: list = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

            if self.samples_per_sec > 0:
                self.throughput_history.append({
                    'epoch': self.epoch,
                    'value': self.samples_per_sec,
                    'time': time.time()
                })
                if len(self.throughput_history) > 200:
                    self.throughput_history = self.throughput_history[-200:]

    def add_loss(self, loss: float):
        with self.lock:
            self.loss_history.append({
                'epoch': self.epoch,
                'loss': loss,
                'time': time.time()
            })
            if len(self.loss_history) > 500:
                self.loss_history = self.loss_history[-500:]

    def add_win_rate(self, mbb: float):
        with self.lock:
            self.win_rate_mbb = mbb
            self.win_rate_history.append({
                'epoch': self.epoch,
                'mbb': mbb,
                'time': time.time()
            })

    def add_action(self, action: int):
        with self.lock:
            action_names = ['fold', 'call', 'raise', 'allin']
            if 0 <= action < len(action_names):
                self.action_counts[action_names[action]] += 1

    def add_hand(self, hand_info: dict):
        with self.lock:
            self.recent_hands.append(hand_info)
            if len(self.recent_hands) > 20:
                self.recent_hands = self.recent_hands[-20:]

    def add_baseline_result(self, baseline_name: str, mbb: float, hands: int):
        with self.lock:
            self.baseline_results[baseline_name] = {
                'mbb': mbb,
                'hands': hands,
                'bb_per_hand': mbb / 1000 * 2,
                'timestamp': time.time()
            }

    def add_strategy_sample(self, info_set: str, strategy: dict):
        with self.lock:
            self.strategy_samples.append({
                'info_set': info_set,
                'strategy': strategy,
                'epoch': self.epoch,
                'timestamp': time.time()
            })
            if len(self.strategy_samples) > 50:
                self.strategy_samples = self.strategy_samples[-50:]

    def to_dict(self):
        with self.lock:
            total_actions = sum(self.action_counts.values()) or 1
            return {
                'running': self.running,
                'epoch': self.epoch,
                'step': self.step,
                'samples': self.samples,
                'total_samples': self.total_samples,
                'samples_per_sec': self.samples_per_sec,
                'loss': self.loss,
                'batch_size': self.batch_size,
                'elapsed': self.elapsed,
                'win_rate_mbb': self.win_rate_mbb,
                'win_rate_history': self.win_rate_history[-50:],
                'loss_history': self.loss_history[-100:],
                'action_distribution': {
                    k: v / total_actions * 100 for k, v in self.action_counts.items()
                },
                'recent_hands': self.recent_hands[-10:],
                'throughput_history': self.throughput_history[-100:],
                'baseline_results': self.baseline_results,
                'strategy_samples': self.strategy_samples[-10:],
            }

state = TrainingState()
model_registry = ModelRegistry()

# ============================================================================
# Evaluation
# ============================================================================

def create_game(config: TrainConfig = None):
    """Create appropriate game based on game mode."""
    if config is None:
        config = ACTIVE_CONFIG

    if GAME_MODE == "full":
        return RustFullHoldem(
            stacks=[config.starting_stack, config.starting_stack],
            small_blind=config.small_blind,
            big_blind=config.big_blind,
        )
    else:
        return RustRiverHoldem(
            stacks=[100.0, 100.0],
            pot=2.0,
            current_bet=0.0,
            player_0_invested=1.0,
            player_1_invested=1.0,
        )


def evaluate_vs_baseline(network, device, baseline, baseline_name: str, num_hands: int = 2000) -> float:
    """Evaluate model against a baseline bot using BATCHED inference for speed."""
    logger.info(f"Evaluating vs {baseline_name} ({num_hands} hands) [mode: {GAME_MODE}]")
    network.eval()

    # BATCHED EVALUATION: Run many hands in parallel with batched network calls
    BATCH_SIZE = 512  # Number of hands to simulate in parallel
    results = []
    start_time = time.time()
    total_completed = 0

    # AGGRESSION TRACKING: Count bot's actions
    total_bot_actions = 0
    aggressive_actions = 0  # Bets/raises (actions 2-7 in full mode)

    # Pre-allocate state buffer for batched inference
    state_dim = get_state_dim()

    while total_completed < num_hands:
        batch_count = min(BATCH_SIZE, num_hands - total_completed)

        # Initialize batch of games
        games = []
        for _ in range(batch_count):
            game = create_game()
            game = game.apply_action(0)  # Deal
            games.append(game)

        active_mask = [True] * batch_count
        hand_results = [0.0] * batch_count

        # Play all hands in parallel until all are terminal
        max_actions = 50  # Safety limit
        for _ in range(max_actions):
            # Collect games that need our bot's decision
            bot_indices = []
            bot_games = []
            bot_legal_actions = []

            for i, (game, active) in enumerate(zip(games, active_mask)):
                if not active:
                    continue
                if game.is_terminal():
                    hand_results[i] = game.returns()[0]
                    active_mask[i] = False
                    continue

                current_player = game.current_player()
                if current_player == -1:
                    active_mask[i] = False
                    continue

                legal_actions = game.legal_actions()
                if len(legal_actions) == 1:
                    games[i] = game.apply_action(legal_actions[0])
                elif current_player == 0:
                    # Queue for batched inference
                    bot_indices.append(i)
                    bot_games.append(game)
                    bot_legal_actions.append(legal_actions)
                else:
                    # Baseline opponent acts immediately
                    action = baseline.get_action(game)
                    games[i] = game.apply_action(action)

            # BATCHED INFERENCE for all bot decisions
            if bot_indices:
                # Encode all states at once
                states_np = np.zeros((len(bot_indices), state_dim), dtype=np.float32)
                for j, game in enumerate(bot_games):
                    states_np[j] = encode_state(game, 0)

                # Single batched forward pass
                states_tensor = torch.from_numpy(states_np).to(device)
                with torch.no_grad():
                    all_advantages = network(states_tensor).cpu().numpy()

                # Apply actions (use_temperature=False during evaluation - no exploration)
                for j, (idx, legal_actions) in enumerate(zip(bot_indices, bot_legal_actions)):
                    advantages = all_advantages[j]
                    strategy = regret_matching(advantages, legal_actions, use_temperature=False)
                    action = legal_actions[np.random.choice(len(legal_actions), p=strategy)]
                    games[idx] = games[idx].apply_action(action)

                    # AGGRESSION TRACKING (Fixed: track for ALL game modes)
                    total_bot_actions += 1
                    if action >= 2:  # Actions 2+ are bets/raises in both modes
                        aggressive_actions += 1

            # Check if all hands are done
            if not any(active_mask):
                break

        results.extend(hand_results)
        total_completed += batch_count

        # Progress logging every 1/10 of total hands
        log_interval = max(BATCH_SIZE, num_hands // 10)
        if total_completed % log_interval < BATCH_SIZE:
            elapsed = time.time() - start_time
            rate = total_completed / elapsed if elapsed > 0 else 0
            pct = 100 * total_completed / num_hands
            logger.info(f"  {baseline_name}: {total_completed}/{num_hands} ({pct:.0f}%) - {rate:.0f} hands/s")

    mbb = (np.mean(results) / 2.0) * 1000
    elapsed = time.time() - start_time
    rate = num_hands / elapsed if elapsed > 0 else 0

    # AGGRESSION FREQUENCY REPORT
    agg_freq = aggressive_actions / total_bot_actions if total_bot_actions > 0 else 0
    agg_pct = agg_freq * 100

    if agg_freq < 0.05:
        logger.warning(f"[WARNING] AGGRESSION CRITICALLY LOW: {agg_pct:.1f}% (target: >5%)")
    elif agg_freq < 0.15:
        logger.info(f"  Aggression: {agg_pct:.1f}% (low - consider increasing aggression_force)")
    else:
        logger.info(f"  Aggression: {agg_pct:.1f}% (healthy)")

    logger.info(f"Evaluation complete: {baseline_name} = {mbb:+.0f} mbb/h ({rate:.0f} hands/s, agg={agg_pct:.1f}%)")
    return mbb

def run_baseline_evaluations(network, device, num_hands: int = 10_000):
    """Run full baseline evaluation suite. Reduced from 100K to 10K for faster feedback."""
    logger.info(f"Starting baseline evaluation suite ({num_hands:,} hands each)")

    baselines = {
        'RandomBot': RandomBot(),
        'CallingStation': CallingStationBot(),
        'AlwaysFold': AlwaysFoldBot(),
    }

    for name, bot in baselines.items():
        try:
            logger.info(f"Evaluating vs {name} ({num_hands:,} hands)...")
            mbb = evaluate_vs_baseline(network, device, bot, name, num_hands=num_hands)
            state.add_baseline_result(name, mbb, num_hands)
            broadcast_state()
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")

    logger.info("Baseline evaluation suite complete")

# ============================================================================
# Strategy Inspector
# ============================================================================

def sample_strategies(network, device, num_samples: int = 5):
    """Sample random game states and record strategies."""
    logger.info(f"Sampling {num_samples} strategies")
    network.eval()

    for _ in range(num_samples):
        try:
            game = create_game()
            game = game.apply_action(0)  # Deal

            # Take a few random actions to get to interesting state
            for _ in range(np.random.randint(0, 4)):
                if game.is_terminal():
                    break
                legal_actions = game.legal_actions()
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
                    game = game.apply_action(action)

            if not game.is_terminal():
                state_enc = encode_state(game, 0)
                state_tensor = torch.from_numpy(state_enc).unsqueeze(0).to(device)

                with torch.no_grad():
                    advantages = network(state_tensor)[0].cpu().numpy()

                legal_actions = game.legal_actions()
                strategy = regret_matching(advantages, legal_actions, use_temperature=False)

                # Create readable info set
                hands = game.hands
                board = game.board
                pot = game.pot

                info_set = f"Hand: {card_str(hands[0][0])}{card_str(hands[0][1])} | "
                info_set += f"Board: {' '.join([card_str(c) for c in board])} | "
                info_set += f"Pot: {pot:.0f}"

                action_names = get_action_names()
                strategy_dict = {
                    action_names[a] if a < len(action_names) else f"Action{a}": float(strategy[i])
                    for i, a in enumerate(legal_actions)
                }

                state.add_strategy_sample(info_set, strategy_dict)
        except Exception as e:
            logger.error(f"Error sampling strategy: {e}")
            continue

# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Aion-26 Pro Training Dashboard</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1800px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff6b6b, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
            font-size: 0.9em;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-value.positive { background: linear-gradient(90deg, #00ff88, #00d4ff); -webkit-background-clip: text; }
        .stat-value.negative { background: linear-gradient(90deg, #ff6b6b, #ff9f43); -webkit-background-clip: text; }
        .stat-label {
            color: #666;
            margin-top: 8px;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .chart-title {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart-wrapper { height: 200px; }
        .status {
            text-align: center;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 1.1em;
            font-weight: 500;
        }
        .status.running {
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.15), rgba(123, 44, 191, 0.15));
            color: #00d4ff;
            border: 1px solid rgba(0, 212, 255, 0.3);
        }
        .status.stopped {
            background: rgba(255, 107, 107, 0.1);
            color: #ff6b6b;
            border: 1px solid rgba(255, 107, 107, 0.3);
        }
        .btn {
            display: inline-block;
            padding: 12px 35px;
            font-size: 1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: 500;
        }
        .btn:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(0,0,0,0.3); }
        .btn-start { background: linear-gradient(90deg, #00d4ff, #7b2cbf); color: white; }
        .btn-stop { background: linear-gradient(90deg, #ff6b6b, #ff9f43); color: white; }
        .btn-eval { background: linear-gradient(90deg, #00ff88, #00d4ff); color: #111; }
        .btn-save { background: linear-gradient(90deg, #7b2cbf, #ff00ff); color: white; }
        .controls { text-align: center; margin-bottom: 20px; display: flex; justify-content: center; align-items: center; flex-wrap: wrap; }

        select {
            padding: 12px 20px;
            font-size: 1em;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.05);
            color: #eee;
            margin: 8px;
            cursor: pointer;
        }
        select:hover {
            background: rgba(255,255,255,0.08);
        }

        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .baseline-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .baseline-table th {
            text-align: left;
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            color: #888;
            font-size: 0.85em;
        }
        .baseline-table td {
            padding: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .baseline-mbb {
            font-weight: bold;
            font-size: 1.2em;
        }
        .baseline-mbb.positive { color: #00ff88; }
        .baseline-mbb.negative { color: #ff6b6b; }

        .strategy-entry {
            padding: 10px;
            margin: 5px 0;
            background: rgba(255,255,255,0.02);
            border-radius: 6px;
            font-size: 0.85em;
            border-left: 3px solid #00d4ff;
        }
        .strategy-info {
            color: #888;
            margin-bottom: 5px;
        }
        .strategy-actions {
            display: flex;
            gap: 10px;
            margin-top: 5px;
        }
        .action-prob {
            flex: 1;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 4px;
            padding: 5px;
            text-align: center;
        }
        .action-prob .label {
            font-size: 0.75em;
            color: #666;
        }
        .action-prob .value {
            font-size: 1.1em;
            font-weight: bold;
            color: #00d4ff;
        }

        .action-dist {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 10px;
        }
        .action-bar {
            flex: 1;
            text-align: center;
        }
        .action-bar-fill {
            height: 100px;
            border-radius: 6px;
            display: flex;
            align-items: flex-end;
            justify-content: center;
            transition: height 0.3s;
        }
        .action-bar-inner {
            width: 100%;
            border-radius: 6px;
            transition: height 0.3s;
        }
        .action-label {
            margin-top: 8px;
            font-size: 0.75em;
            color: #888;
            text-transform: uppercase;
        }
        .action-value {
            font-size: 0.9em;
            font-weight: bold;
            margin-top: 4px;
        }
        .fold .action-bar-inner { background: linear-gradient(to top, #ff6b6b, #ff9f43); }
        .call .action-bar-inner { background: linear-gradient(to top, #00d4ff, #7b2cbf); }
        .raise .action-bar-inner { background: linear-gradient(to top, #00ff88, #00d4ff); }
        .allin .action-bar-inner { background: linear-gradient(to top, #ff00ff, #7b2cbf); }

        .scrollable {
            max-height: 300px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Aion-26 Pro Training Dashboard</h1>
        <p class="subtitle">Production-ready poker AI training with strategy inspection & baseline evaluation</p>

        <div id="status" class="status stopped">IDLE - Click Start to begin training</div>

        <div class="controls">
            <button class="btn btn-start" onclick="startTraining()">Start Training</button>
            <button class="btn btn-stop" onclick="stopTraining()">Stop Training</button>
            <button class="btn btn-eval" onclick="runBaselines()">Run Baselines</button>
            <button class="btn btn-save" onclick="saveModel()">Save Model</button>
            <select id="model-select" onchange="loadModel()">
                <option value="">-- Load Model --</option>
            </select>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="win-rate">0</div>
                <div class="stat-label">Win Rate (mbb/h)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="win-rate-bb100">0.00</div>
                <div class="stat-label">Win Rate (bb/100)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="samples-sec">0</div>
                <div class="stat-label">Samples/Second</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="total-samples">0</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="epoch">0</div>
                <div class="stat-label">Epoch</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="loss">0.00</div>
                <div class="stat-label">Loss</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="batch-size">0</div>
                <div class="stat-label">Batch Size</div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Win Rate Over Training (mbb/h)</div>
                <div class="chart-wrapper">
                    <canvas id="winRateChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Action Distribution</div>
                <div class="action-dist">
                    <div class="action-bar fold">
                        <div class="action-bar-fill"><div class="action-bar-inner" id="fold-bar"></div></div>
                        <div class="action-label">Fold</div>
                        <div class="action-value" id="fold-pct">0%</div>
                    </div>
                    <div class="action-bar call">
                        <div class="action-bar-fill"><div class="action-bar-inner" id="call-bar"></div></div>
                        <div class="action-label">Call</div>
                        <div class="action-value" id="call-pct">0%</div>
                    </div>
                    <div class="action-bar raise">
                        <div class="action-bar-fill"><div class="action-bar-inner" id="raise-bar"></div></div>
                        <div class="action-label">Raise</div>
                        <div class="action-value" id="raise-pct">0%</div>
                    </div>
                    <div class="action-bar allin">
                        <div class="action-bar-fill"><div class="action-bar-inner" id="allin-bar"></div></div>
                        <div class="action-label">All-In</div>
                        <div class="action-value" id="allin-pct">0%</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Training Throughput</div>
                <div class="chart-wrapper">
                    <canvas id="throughputChart"></canvas>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Loss Curve</div>
                <div class="chart-wrapper">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
        </div>

        <div class="bottom-grid">
            <div class="panel">
                <div class="chart-title">Baseline Evaluations</div>
                <table class="baseline-table">
                    <thead>
                        <tr>
                            <th>Baseline</th>
                            <th>mbb/h</th>
                            <th>bb/100</th>
                            <th>Hands</th>
                        </tr>
                    </thead>
                    <tbody id="baseline-body">
                        <tr><td colspan="4" style="text-align:center;color:#666;">Run evaluation to see results</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="panel">
                <div class="chart-title">Strategy Inspector</div>
                <div class="scrollable" id="strategy-samples">
                    <p style="text-align:center;color:#666;">Strategy samples will appear here</p>
                </div>
            </div>

            <div class="panel">
                <div class="chart-title">Performance Summary</div>
                <div style="padding: 20px; font-size: 0.9em; line-height: 1.8;">
                    <div><strong>Training Time:</strong> <span id="train-time">0:00</span></div>
                    <div><strong>Epochs Completed:</strong> <span id="epochs-done">0</span></div>
                    <div><strong>Samples Processed:</strong> <span id="samples-done">0</span></div>
                    <div><strong>Available Models:</strong> <span id="model-count">0</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let startTime = null;

        // Chart configurations
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#666', maxTicksLimit: 8 } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#666' } }
            }
        };

        // Win Rate Chart
        const winRateCtx = document.getElementById('winRateChart').getContext('2d');
        const winRateChart = new Chart(winRateCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: '#00ff88'
                }]
            },
            options: { ...chartOptions, scales: { ...chartOptions.scales, y: { ...chartOptions.scales.y, beginAtZero: false } } }
        });

        // Throughput Chart
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        const throughputChart = new Chart(throughputCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: chartOptions
        });

        // Loss Chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: chartOptions
        });

        // Function to update UI with state data
        function updateUI(data) {
            // Update stat cards
            const wrEl = document.getElementById('win-rate');
            wrEl.textContent = data.win_rate_mbb.toFixed(0);
            wrEl.className = 'stat-value ' + (data.win_rate_mbb >= 0 ? 'positive' : 'negative');

            // bb/100 = mbb/h / 10 (milli-big-blinds to big-blinds, per-hand to per-100-hands)
            const bb100 = data.win_rate_mbb / 10;
            const bb100El = document.getElementById('win-rate-bb100');
            bb100El.textContent = bb100.toFixed(2);
            bb100El.className = 'stat-value ' + (bb100 >= 0 ? 'positive' : 'negative');

            document.getElementById('samples-sec').textContent =
                data.samples_per_sec.toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById('total-samples').textContent =
                (data.total_samples / 1000000).toFixed(1) + 'M';
            document.getElementById('epoch').textContent = data.epoch;
            document.getElementById('loss').textContent = data.loss.toFixed(4);
            document.getElementById('batch-size').textContent = data.batch_size.toLocaleString();

            // Status
            const statusEl = document.getElementById('status');
            if (data.running) {
                statusEl.className = 'status running';
                statusEl.textContent = 'TRAINING - Epoch ' + data.epoch + ' | ' +
                    data.samples_per_sec.toLocaleString(undefined, {maximumFractionDigits: 0}) + ' samples/s';
                if (!startTime) startTime = Date.now();
            } else {
                statusEl.className = 'status stopped';
                statusEl.textContent = 'IDLE - Click Start to begin training';
            }

            // Update loss chart - extract loss values from history objects
            if (data.loss_history && data.loss_history.length > 0) {
                lossChart.data.labels = data.loss_history.map((_, i) => i);
                lossChart.data.datasets[0].data = data.loss_history.map(h => h.loss);
                lossChart.update('none');
            }

            // Update throughput chart
            if (data.throughput_history && data.throughput_history.length > 0) {
                throughputChart.data.labels = data.throughput_history.map((_, i) => i);
                throughputChart.data.datasets[0].data = data.throughput_history.map(h => h.value);
                throughputChart.update('none');
            }

            // Update win rate chart
            if (data.win_rate_history && data.win_rate_history.length > 0) {
                winRateChart.data.labels = data.win_rate_history.map(h => 'E' + h.epoch);
                winRateChart.data.datasets[0].data = data.win_rate_history.map(h => h.mbb);
                winRateChart.update('none');
            }

            // Update action distribution bars
            if (data.action_distribution) {
                const dist = data.action_distribution;
                ['fold', 'call', 'raise', 'allin'].forEach(action => {
                    const pct = dist[action] || 0;
                    const barEl = document.getElementById(action + '-bar');
                    const pctEl = document.getElementById(action + '-pct');
                    if (barEl) barEl.style.height = pct + '%';
                    if (pctEl) pctEl.textContent = pct.toFixed(1) + '%';
                });
            }

            // Update baseline results table
            if (data.baseline_results && Object.keys(data.baseline_results).length > 0) {
                const tbody = document.getElementById('baseline-body');
                if (tbody) {
                    tbody.innerHTML = Object.entries(data.baseline_results).map(([name, result]) => {
                        const mbbClass = result.mbb >= 0 ? 'positive' : 'negative';
                        const bb100 = result.mbb / 10;
                        return `<tr>
                            <td>${name}</td>
                            <td class="baseline-mbb ${mbbClass}">${result.mbb >= 0 ? '+' : ''}${result.mbb.toFixed(0)}</td>
                            <td>${bb100.toFixed(2)}</td>
                            <td>${result.hands}</td>
                        </tr>`;
                    }).join('');
                }
            }

            // Update strategy samples
            if (data.strategy_samples && data.strategy_samples.length > 0) {
                let html = '';
                data.strategy_samples.slice(-5).forEach(s => {
                    html += '<div class="strategy-item">' +
                        '<div class="info-set">' + s.info_set + '</div>' +
                        '<div class="strategy">';
                    for (const [action, prob] of Object.entries(s.strategy)) {
                        html += '<span>' + action + ': ' + (prob * 100).toFixed(0) + '%</span> ';
                    }
                    html += '</div></div>';
                });
                document.getElementById('strategy-samples').innerHTML = html;
            }
        }

        // Fetch initial state on page load
        fetch('/state')
            .then(response => response.json())
            .then(data => updateUI(data))
            .catch(err => console.error('Failed to fetch initial state:', err));

        socket.on('update', function(data) {
            // updateUI handles all stat cards, charts, action dist, and baseline results
            updateUI(data);

            // Strategy samples
            if (data.strategy_samples && data.strategy_samples.length > 0) {
                const container = document.getElementById('strategy-samples');
                container.innerHTML = data.strategy_samples.slice().reverse().map(sample => {
                    const actions = Object.entries(sample.strategy).map(([action, prob]) =>
                        `<div class="action-prob">
                            <div class="label">${action}</div>
                            <div class="value">${(prob * 100).toFixed(1)}%</div>
                        </div>`
                    ).join('');

                    return `<div class="strategy-entry">
                        <div class="strategy-info">${sample.info_set}</div>
                        <div class="strategy-actions">${actions}</div>
                    </div>`;
                }).join('');
            }

            // Summary
            document.getElementById('epochs-done').textContent = data.epoch;
            document.getElementById('samples-done').textContent =
                (data.total_samples / 1000000).toFixed(2) + 'M';

            if (startTime) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                document.getElementById('train-time').textContent =
                    mins + ':' + secs.toString().padStart(2, '0');
            }
        });

        function startTraining() {
            fetch('/start', { method: 'POST' });
            startTime = Date.now();
        }

        function stopTraining() {
            fetch('/stop', { method: 'POST' });
        }

        function runBaselines() {
            fetch('/eval/baselines', { method: 'POST' })
                .then(r => r.json())
                .then(data => console.log('Baseline eval started:', data));
        }

        function saveModel() {
            fetch('/models/save', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    alert('Model saved: ' + data.model_id);
                    refreshModelList();
                });
        }

        function loadModel() {
            const select = document.getElementById('model-select');
            const modelId = select.value;
            if (modelId) {
                fetch('/models/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_id: modelId })
                })
                .then(r => r.json())
                .then(data => {
                    alert('Model loaded: ' + modelId);
                    console.log('Metadata:', data.metadata);
                });
            }
        }

        function refreshModelList() {
            fetch('/models/list')
                .then(r => r.json())
                .then(data => {
                    const select = document.getElementById('model-select');
                    select.innerHTML = '<option value="">-- Load Model --</option>' +
                        data.models.map(m =>
                            `<option value="${m.id}">${m.id} (Epoch ${m.metadata.epoch || '?'})</option>`
                        ).join('');
                    document.getElementById('model-count').textContent = data.models.length;
                });
        }

        // Initial load
        refreshModelList();
        setInterval(refreshModelList, 10000); // Refresh every 10s
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start', methods=['POST'])
def start_training():
    with state.lock:
        if not state.running:
            state.running = True
            thread = threading.Thread(target=training_loop, daemon=True)
            thread.start()
            logger.info("Training started")
            return {'status': 'started'}
        else:
            return {'status': 'already_running'}

@app.route('/stop', methods=['POST'])
def stop_training():
    state.running = False
    logger.info("Training stopped")
    return {'status': 'stopped'}

@app.route('/state', methods=['GET'])
def get_state():
    """Return current training state for initial page load."""
    return state.to_dict()

@app.route('/eval/baselines', methods=['POST'])
def eval_baselines():
    if hasattr(state, '_network') and state._network is not None:
        thread = threading.Thread(
            target=lambda: run_baseline_evaluations(state._network, state._device),
            daemon=True
        )
        thread.start()
        logger.info("Baseline evaluation started")
        return {'status': 'evaluating'}
    return {'status': 'no_model'}, 400

@app.route('/models/save', methods=['POST'])
def save_model_endpoint():
    if hasattr(state, '_network') and state._network is not None:
        try:
            metadata = {
                'epoch': state.epoch,
                'total_samples': state.total_samples,
                'loss': state.loss,
                'win_rate_mbb': state.win_rate_mbb,
                'timestamp': datetime.now().isoformat(),
            }
            model_id = model_registry.save_model(state._network, state._optimizer, metadata)
            return {'status': 'saved', 'model_id': model_id}
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return {'status': 'error', 'message': str(e)}, 500
    return {'status': 'no_model'}, 400

@app.route('/models/load', methods=['POST'])
def load_model_endpoint():
    try:
        data = request.get_json()
        model_id = data.get('model_id')

        if not model_id:
            return {'status': 'error', 'message': 'model_id required'}, 400

        device = state._device if hasattr(state, '_device') else 'cpu'
        network, metadata = model_registry.load_model(model_id, device)

        state._network = network
        logger.info(f"Model loaded: {model_id}")

        # Sample some strategies from loaded model
        sample_strategies(network, device, num_samples=5)
        broadcast_state()

        return {'status': 'loaded', 'model_id': model_id, 'metadata': metadata}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/models/list', methods=['GET'])
def list_models_endpoint():
    try:
        models = model_registry.list_models()
        return {'models': models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/health')
def health():
    """Health check endpoint for CI and monitoring."""
    return jsonify({
        'status': 'ok',
        'version': '0.1.0',
        'training': state.running,
        'epoch': state.epoch,
        'game_mode': GAME_MODE,
        'uptime_seconds': round(time.time() - APP_START_TIME, 1),
    })

def broadcast_state():
    socketio.emit('update', state.to_dict())

# ============================================================================
# Training Loop
# ============================================================================

def training_loop():
    try:
        logger.info("Starting training loop with Polyak averaging")

        # CODE RED: Use global config, NOT a new instance
        config = ACTIVE_CONFIG
        print_active_config()  # Dump config to console immediately

        # Check if this flop is already solved in solver database
        if config.fixed_flop and not FORCE_RETRAIN:
            is_solved, model_path = is_flop_solved(config.fixed_flop, config)
            if is_solved:
                flop_key = get_flop_key(config.fixed_flop)
                logger.info(f"[SOLVER DB] Flop {flop_key} already solved at {model_path}")
                logger.info(f"[SOLVER DB] Skipping training. Use --force to retrain.")
                state.running = False
                broadcast_state()
                return

        # Prefer CUDA (NVIDIA GPU), then MPS (Apple Silicon), then CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        logger.info(f"Device: {device}")
        logger.info(f"Game mode: {GAME_MODE}")
        logger.info(f"State dim: {get_state_dim()}, Target dim: {get_target_dim()}")
        logger.info(f"Polyak tau: {config.polyak_tau}")
        logger.info(f"History alpha: {config.history_alpha}")

        # Online network (trained via gradient descent)
        online_net = create_network(device)
        optimizer = torch.optim.Adam(online_net.parameters(), lr=config.learning_rate)

        # Learning rate scheduler: Cosine annealing from lr to lr_end
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr_end
        )
        logger.info(f"LR scheduler: Cosine annealing {config.learning_rate} -> {config.lr_end}")

        # OPTIMIZED: AMP enabled for faster training
        logger.info("OPTIMIZED: TF32 + AMP enabled for maximum GPU throughput")

        # Target network (updated via Polyak averaging)
        target_net = create_network(device)
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()

        # Store network reference
        state._network = target_net
        state._device = device
        state._optimizer = optimizer

        # Note: state.running is already True (set in start_training endpoint)
        state.update(epoch=0, step=0, samples=0, total_samples=0)
        broadcast_state()

        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Data dir: {config.data_dir}")

        total_samples = 0

        for epoch in range(config.epochs):
            if not state.running:
                break

            # Update global epoch for temperature annealing
            global CURRENT_EPOCH
            CURRENT_EPOCH = epoch
            temp = get_temperature()

            logger.info(f"[{ts()}] === EPOCH {epoch+1}/{config.epochs} START === (temp={temp:.2f})")

            # Use appropriate trainer based on game mode
            if GAME_MODE == "full":
                trainer = ParallelTrainerFull(
                    config.data_dir,
                    query_buffer_size=config.query_buffer_size,
                    num_workers=config.num_workers,
                    small_blind=config.small_blind,
                    big_blind=config.big_blind,
                    starting_stack=config.starting_stack,
                    fixed_flop=config.fixed_flop,  # Single-flop training mode
                )
            else:
                # River-only mode: fixed_board is the full 5-card board
                fixed_board = config.fixed_flop  # In river mode, we need 5 cards
                trainer = ParallelTrainer(
                    config.data_dir,
                    query_buffer_size=config.query_buffer_size,
                    num_workers=config.num_workers,
                    fixed_board=fixed_board,
                )
            trainer.start_epoch(epoch)

            epoch_start = time.time()
            step = 0

            result = trainer.step(None, num_traversals=config.traversals_per_epoch)

            inference_batch_num = 0  # For sanity check logging
            forced_agg_samples = 0  # Track forced aggression for diagnostic

            while not result.is_finished() and state.running:
                if result.is_request_inference() and result.count() > 0:
                    states = trainer.get_query_buffer()
                    states_np = np.asarray(states, dtype=np.float32)
                    states_tensor = torch.from_numpy(states_np).to(device)

                    # Fast inference with AMP
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        preds = target_net(states_tensor)

                    preds_np = preds.float().cpu().numpy()

                    # FORCED AGGRESSION: Break passive convergence loop
                    # Randomly boost bet/raise advantages to force exploration of aggressive lines
                    # Use decaying aggression rate
                    current_agg_rate = get_aggression_rate()
                    if current_agg_rate > 0:
                        batch_size = preds_np.shape[0]
                        # Random mask: which samples get forced aggression
                        force_mask = np.random.random(batch_size) < current_agg_rate
                        num_forced = force_mask.sum()
                        forced_agg_samples += num_forced

                        if num_forced > 0:
                            if GAME_MODE == "full":
                                # Full HUNL: Actions 0=Fold, 1=Check/Call, 2-7=Bets
                                preds_np[force_mask, 2:8] += config.aggression_boost
                                # 50% prefer All-In (index 7)
                                prefer_max_mask = force_mask & (np.random.random(batch_size) < config.prefer_large_bet)
                                if prefer_max_mask.any():
                                    preds_np[prefer_max_mask, 7] += config.aggression_boost
                            else:
                                # River mode: Actions 0=Fold, 1=Check/Call, 2=Bet, 3=All-In
                                preds_np[force_mask, 2:4] += config.aggression_boost
                                # 50% prefer All-In (index 3)
                                prefer_max_mask = force_mask & (np.random.random(batch_size) < config.prefer_large_bet)
                                if prefer_max_mask.any():
                                    preds_np[prefer_max_mask, 3] += config.aggression_boost

                    result = trainer.step(preds_np)
                    inference_batch_num += 1
                else:
                    result = trainer.step(None)

                step += 1
                current_samples = result.samples()
                elapsed = time.time() - epoch_start

                if step % 10 == 0:
                    samples_per_sec = current_samples / elapsed if elapsed > 0 else 0
                    state.update(
                        epoch=epoch + 1,
                        step=step,
                        samples=current_samples,
                        total_samples=total_samples + current_samples,
                        samples_per_sec=samples_per_sec,
                        # CODE RED: Show TRAINING batch size, not inference count
                        batch_size=config.train_batch_size,
                        elapsed=elapsed
                    )
                    broadcast_state()

            epoch_samples = result.samples()
            total_samples += epoch_samples
            trainer.end_epoch()

            elapsed = time.time() - epoch_start
            # Fixed: Calculate forced aggression percentage for ALL game modes
            agg_pct = 100 * forced_agg_samples / max(1, epoch_samples)
            logger.info(f"[{ts()}] Epoch {epoch+1} traversal complete: {epoch_samples:,} samples, {epoch_samples/elapsed:.0f}/s, forced_agg={agg_pct:.1f}%")

            # Network Training Phase with Historical Mixing
            logger.info(f"[{ts()}] Training network with historical mixing...")
            train_start = time.time()

            alpha = config.history_alpha
            all_states = []
            all_targets = []
            all_weights = []

            # FIX: Cap samples BEFORE concatenation to avoid OOM during np.concatenate
            MAX_SAMPLES = 5_000_000
            total_samples = 0

            for past_epoch in range(epoch + 1):
                past_states, past_targets = load_epoch_data(config.data_dir, past_epoch)
                if len(past_states) > 0:
                    weight = alpha ** (epoch - past_epoch)

                    # Subsample if this would exceed MAX_SAMPLES
                    if total_samples + len(past_states) > MAX_SAMPLES:
                        keep = MAX_SAMPLES - total_samples
                        if keep > 0:
                            indices = np.random.choice(len(past_states), keep, replace=False)
                            past_states = past_states[indices]
                            past_targets = past_targets[indices]
                            all_states.append(past_states)
                            all_targets.append(past_targets)
                            all_weights.extend([weight] * len(past_states))
                            total_samples += len(past_states)
                        break  # Stop loading more epochs
                    else:
                        all_states.append(past_states)
                        all_targets.append(past_targets)
                        all_weights.extend([weight] * len(past_states))
                        total_samples += len(past_states)

            if len(all_states) == 0:
                logger.warning(f"No data for epoch {epoch}")
                continue

            logger.info(f"Total samples for training: {total_samples:,}")

            # CODE RED: Force float32, no async transfer during debugging
            all_states_np = np.concatenate(all_states).astype(np.float32)
            all_targets_np = np.concatenate(all_targets).astype(np.float32)

            # CODE RED: Sanity check the REGRET TARGETS from Rust
            print(f"\n[CODE RED] REGRET TARGET STATS (epoch {epoch+1}):")
            print(f"  shape: {all_targets_np.shape}")
            print(f"  dtype: {all_targets_np.dtype}")
            print(f"  mean:  {all_targets_np.mean():.6f}")
            print(f"  std:   {all_targets_np.std():.6f}")
            print(f"  min:   {all_targets_np.min():.6f}")
            print(f"  max:   {all_targets_np.max():.6f}")
            print(f"  has_nan: {np.isnan(all_targets_np).any()}")
            print(f"  has_inf: {np.isinf(all_targets_np).any()}")
            print(f"  all_zero: {(all_targets_np == 0).all()}")
            # AWR stats: how many positive regrets (missed opportunities)?
            positive_count = (all_targets_np > 0).sum()
            total_elements = all_targets_np.size
            print(f"  positive_regrets: {positive_count:,} / {total_elements:,} ({100*positive_count/total_elements:.1f}%)")
            print(f"  AWR weight: {config.positive_regret_weight}x for positive regrets")

            # CODE RED: Check for corrupt data
            if np.isnan(all_targets_np).any() or np.isinf(all_targets_np).any():
                raise ValueError("CRITICAL: Rust returned NaN/Inf in regret targets!")
            if (all_targets_np == 0).all():
                logger.warning("WARNING: All regret targets are zero - Rust solver might be broken!")

            # ILLEGAL ACTION STATS: Track how many actions are legal vs illegal
            legal_count = (all_targets_np != 0).sum()
            total_count = all_targets_np.size
            legal_pct = 100 * legal_count / total_count
            print(f"  legal_actions: {legal_count:,} / {total_count:,} ({legal_pct:.1f}%)")
            if GAME_MODE == "river":
                # Per-action breakdown for river mode
                actions = ['Fold', 'Check/Call', 'Bet', 'All-In']
                for i, name in enumerate(actions):
                    col = all_targets_np[:, i] if i < all_targets_np.shape[1] else np.array([])
                    if len(col) > 0:
                        leg = (col != 0).sum()
                        pos = (col > 0).sum()
                        print(f"    {name}: {100*leg/len(col):.0f}% legal, {100*pos/leg:.0f}% positive when legal" if leg > 0 else f"    {name}: 0% legal")

            # Note: Sample limit already applied during loading (MAX_SAMPLES = 5M)

            train_states = torch.from_numpy(all_states_np).to(device)
            train_targets = torch.from_numpy(all_targets_np).to(device)
            sample_weights = torch.tensor(all_weights, dtype=torch.float32, device=device)
            sample_weights = sample_weights / sample_weights.sum()

            logger.info(f"Historical data: {len(train_states):,} samples from {epoch+1} epochs")

            online_net.train()
            total_loss = 0.0
            num_batches = 0

            sample_weights_cpu = sample_weights.cpu()

            # OPTIMIZED: Use AMP for faster training on GPU
            use_amp = device.type == 'cuda'
            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

            for train_step in range(config.train_steps_per_epoch):
                if not state.running:
                    break

                indices = torch.multinomial(sample_weights_cpu, config.train_batch_size, replacement=True)
                batch_states = train_states[indices]
                batch_targets = train_targets[indices]

                # OPTIMIZED: Mixed precision forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    predictions = online_net(batch_states)

                    # STANDARD CFR LOSS: Simple MSE on all actions
                    # Zero regret is VALID (action as good as expected value)
                    # No asymmetric weighting - this breaks CFR convergence
                    # No entropy regularization - this prevents Nash convergence
                    squared_errors = (predictions - batch_targets) ** 2
                    loss = squared_errors.mean()  # Standard MSE loss

                    # Standard CFR: No hand penalty, no entropy regularization
                    # These modifications break CFR convergence theory

                # OPTIMIZED: Scaled backward pass
                optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                polyak_update(target_net, online_net, config.polyak_tau)

                total_loss += loss.item()
                num_batches += 1

                if train_step % 20 == 0:
                    avg_loss = total_loss / max(1, num_batches)
                    state.update(loss=avg_loss)
                    state.add_loss(avg_loss)
                    broadcast_state()

            online_net.eval()
            train_elapsed = time.time() - train_start
            avg_loss = total_loss / max(1, num_batches)

            # Step learning rate scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            logger.info(f"[{ts()}] Epoch {epoch+1} network trained: loss={avg_loss:.4f}, time={train_elapsed:.1f}s, lr={current_lr:.2e}")
            state.update(
                epoch=epoch + 1,
                samples=epoch_samples,
                total_samples=total_samples,
                loss=avg_loss,
            )
            broadcast_state()

            # Periodic evaluation and strategy sampling
            if (epoch + 1) % config.eval_interval == 0:
                logger.info(f"[{ts()}] Running periodic evaluation...")

                # Sample strategies
                sample_strategies(target_net, device, num_samples=5)

                # Quick win rate check
                try:
                    mbb = evaluate_vs_baseline(target_net, device, RandomBot(), "RandomBot", num_hands=config.eval_hands)
                    state.add_win_rate(mbb)
                except Exception as e:
                    logger.error(f"Evaluation error: {e}")

                broadcast_state()

        # Save final checkpoint
        metadata = {
            'epoch': config.epochs,
            'total_samples': total_samples,
            'loss': avg_loss,
            'final_model': True,
        }
        model_id = model_registry.save_model(target_net, optimizer, metadata)
        logger.info(f"Final model saved: {model_id}")

        # Export to solver database if training a specific flop
        if config.fixed_flop:
            export_metrics = {
                'loss': avg_loss,
                'win_rate': state.win_rate_mbb,
                'total_samples': total_samples,
            }
            export_solved_flop(config.fixed_flop, config, target_net, export_metrics)
            logger.info(f"[SOLVER DB] Flop exported to E:/solver_data")

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
    finally:
        state.running = False
        broadcast_state()
        logger.info("Training loop ended")

# ============================================================================
# Typer CLI
# ============================================================================

cli = typer.Typer(
    name="aion26-train",
    help="Aion-26 Deep PDCFR+ Training Dashboard",
    add_completion=False,
)

@cli.command()
def serve(
    port: Annotated[int, typer.Option("--port", "-p", help="Server port")] = 5001,
    host: Annotated[str, typer.Option("--host", "-h", help="Server host")] = "0.0.0.0",
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable Flask debug mode")] = False,
    full: Annotated[bool, typer.Option("--full", "-f", help="Full HUNL mode (4 streets, 8 actions)")] = False,
    flop: Annotated[str, typer.Option("--flop", help="Fixed flop for single-flop training (e.g., 'Ac,Kd,Qh' or '12,25,38')")] = None,
    force: Annotated[bool, typer.Option("--force", help="Force retrain even if flop already solved")] = False,
):
    """Start the training dashboard web server."""
    global GAME_MODE, ACTIVE_CONFIG, model_registry, FORCE_RETRAIN
    GAME_MODE = "full" if full else "river"
    FORCE_RETRAIN = force

    # Reinitialize config with correct data_dir for game mode
    ACTIVE_CONFIG = TrainConfig()
    model_registry = ModelRegistry(base_dir=f"{ACTIVE_CONFIG.data_dir}/models")

    # Parse fixed flop if provided
    if flop:
        ACTIVE_CONFIG.fixed_flop = parse_flop_string(flop)
        console.print(f"[yellow]Single-flop training mode: {flop} -> {ACTIVE_CONFIG.fixed_flop}[/yellow]")

        # Check solver database
        is_solved, model_path = is_flop_solved(ACTIVE_CONFIG.fixed_flop, ACTIVE_CONFIG)
        if is_solved and not force:
            console.print(f"[green]Flop already solved: {model_path}[/green]")
            console.print(f"[dim]Use --force to retrain[/dim]")
        elif is_solved and force:
            console.print(f"[yellow]Force retraining (--force flag set)[/yellow]")

    # Show solver database status
    solved_flops = list_solved_flops()
    if solved_flops:
        console.print(f"[cyan]Solver DB: {len(solved_flops)} flops solved in E:/solver_data[/cyan]")

    setup_logging(verbose=verbose)

    # Display startup banner with Rich
    mode_str = "Full HUNL (4 streets)" if full else "River-only (1 street)"
    console.print(Panel.fit(
        f"[bold cyan]Aion-26 Pro Training Dashboard[/bold cyan]\n"
        f"[dim]Deep PDCFR+ for Imperfect Information Games[/dim]\n"
        f"[yellow]Mode: {mode_str}[/yellow]",
        border_style="cyan"
    ))

    # Show configuration table
    table = Table(title="Configuration", show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Host", host)
    table.add_row("Port", str(port))
    table.add_row("Game Mode", GAME_MODE)
    table.add_row("State Dim", str(get_state_dim()))
    table.add_row("Action Dim", str(get_target_dim()))
    table.add_row("Data Dir", ACTIVE_CONFIG.data_dir)
    table.add_row("Verbose", str(verbose))
    table.add_row("Debug", str(debug))
    table.add_row("Device", "CUDA" if torch.cuda.is_available() else "CPU")
    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
    console.print(table)

    console.print("\n[bold green]Features:[/bold green]")
    console.print("  [dim]-[/dim] Real-time training visualization")
    console.print("  [dim]-[/dim] Strategy inspector")
    console.print("  [dim]-[/dim] Model save/load management")
    console.print("  [dim]-[/dim] Baseline evaluation suite")
    console.print("  [dim]-[/dim] Rich logging with --verbose flag")

    console.print(f"\n[bold]Open [link=http://localhost:{port}]http://localhost:{port}[/link] in your browser[/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)

@cli.command()
def info():
    """Show system and configuration information."""
    setup_logging(verbose=True)

    console.print(Panel.fit("[bold]Aion-26 System Info[/bold]", border_style="blue"))

    table = Table(title="Environment")
    table.add_column("Component", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", torch.version.cuda or "N/A")
        table.add_row("GPU", torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        table.add_row("GPU Memory", f"{mem:.1f} GB")

    table.add_row("Game Mode", GAME_MODE)
    table.add_row("State Dim", str(get_state_dim()))
    table.add_row("Target Dim", str(get_target_dim()))

    console.print(table)

    # Show default training config
    config = TrainConfig()
    cfg_table = Table(title="Default Training Config")
    cfg_table.add_column("Parameter", style="cyan")
    cfg_table.add_column("Value", style="yellow")

    cfg_table.add_row("Epochs", str(config.epochs))
    cfg_table.add_row("Traversals/Epoch", f"{config.traversals_per_epoch:,}")
    cfg_table.add_row("Workers", f"{config.num_workers:,}")
    cfg_table.add_row("Train Batch Size", f"{config.train_batch_size:,}")
    cfg_table.add_row("Learning Rate", str(config.learning_rate))
    cfg_table.add_row("Polyak Tau", str(config.polyak_tau))

    console.print(cfg_table)

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    cli()
