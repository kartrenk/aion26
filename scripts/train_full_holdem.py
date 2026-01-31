#!/usr/bin/env python3
"""Training Script for Full HUNL Deep CFR.

This script trains a poker AI for full Heads-Up No-Limit Texas Hold'em
using Deep CFR with Monte Carlo sampling.

Features:
- 4 betting streets (Preflop, Flop, Turn, River)
- 8 action types with bet sizing
- 220-dimensional state encoding
- Parallel game tree traversal

Run with: python scripts/train_full_holdem.py
"""

import sys
import os
from pathlib import Path

# Windows DLL loading fix
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
            print(f"[DLL] Loaded: {pyd_file}")
        except Exception as e:
            print(f"[DLL] Warning: Could not load {pyd_file}: {e}")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import json

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO

# Rich and Typer for CLI and logging
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Disable TF32 for consistent results
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)

from aion26_rust import ParallelTrainerFull, RustFullHoldem
from aion26.baselines import RandomBot, CallingStationBot, AlwaysFoldBot

# ============================================================================
# Logging Setup
# ============================================================================

console = Console()
VERBOSE = False

def setup_logging(verbose: bool = False):
    global VERBOSE
    VERBOSE = verbose

    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

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

    if not verbose:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('engineio').setLevel(logging.WARNING)
        logging.getLogger('socketio').setLevel(logging.WARNING)

setup_logging(verbose=False)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

STATE_DIM = 220
TARGET_DIM = 8
RECORD_SIZE = (STATE_DIM + TARGET_DIM) * 4  # 912 bytes

ACTION_NAMES = ['Fold', 'Check/Call', 'Bet 0.5x', 'Bet 0.75x', 'Bet Pot', 'Bet 1.5x', 'Bet 2x', 'All-In']
STREET_NAMES = ['Preflop', 'Flop', 'Turn', 'River']

@dataclass
class TrainConfig:
    epochs: int = 200
    traversals_per_epoch: int = 500_000      # More traversals needed for multi-street
    num_workers: int = 2048
    query_buffer_size: int = 16384
    train_batch_size: int = 32768            # Larger batch for more stable gradients
    train_steps_per_epoch: int = 300
    learning_rate: float = 3e-4              # Slightly lower LR
    polyak_tau: float = 0.005
    history_alpha: float = 0.5
    eval_interval: int = 10
    eval_hands: int = 5000
    data_dir: str = "/tmp/full_hunl_dcfr"
    small_blind: float = 0.5
    big_blind: float = 1.0
    starting_stack: float = 100.0

ACTIVE_CONFIG = TrainConfig()

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

    file_size = path.stat().st_size
    num_samples = file_size // RECORD_SIZE

    data = np.fromfile(path, dtype=np.float32)
    data = data.reshape(num_samples, STATE_DIM + TARGET_DIM)

    states = data[:, :STATE_DIM]
    targets = data[:, STATE_DIM:]

    return states, targets

def card_str(card: int) -> str:
    """Convert card index to string."""
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks[card % 13] + suits[card // 13]

# ============================================================================
# Network (Larger for Full HUNL)
# ============================================================================

class AdvantageNetworkFull(nn.Module):
    """Advantage network for Full HUNL with 8 actions."""

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

# ============================================================================
# State Encoder (220 dims)
# ============================================================================

def encode_state(game_state, player: int) -> np.ndarray:
    """Encode game state for neural network.

    MUST MATCH RUST encode_state() exactly!
    Layout (220 dims):
      [0-9]:     Hand rank category one-hot (10 dims) - zeros if not river
      [10-43]:   Hole cards (2 * 17 = 34 dims)
      [44-128]:  Board cards (5 * 17 = 85 dims) - zeros for undealt slots
      [129-132]: Street one-hot (4 dims)
      [133-196]: Action history (8 * 8 = 64 dims)
      [197-219]: Betting context (23 dims)
    """
    features = np.zeros(STATE_DIM, dtype=np.float32)

    hands = game_state.hands
    board = game_state.board
    street = game_state.street

    # 1. Hand rank category (10 dims) - only on river
    if len(board) >= 5:
        try:
            hand_category = game_state.get_hand_strength(player)
            if 0 <= hand_category < 10:
                features[hand_category] = 1.0
        except:
            pass

    # 2. Hole cards (34 dims): indices 10-43
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

    # 3. Board cards (85 dims): indices 44-128
    for i, card in enumerate(board[:5]):
        offset = 44 + i * 17
        rank = card % 13
        suit = card // 13
        if rank < 13:
            features[offset + rank] = 1.0
        if suit < 4:
            features[offset + 13 + suit] = 1.0

    # 4. Street one-hot (4 dims): indices 129-132
    if 0 <= street < 4:
        features[129 + street] = 1.0

    # 5. Action history (64 dims): indices 133-196
    action_hist = game_state.get_action_history()
    hist_len = len(action_hist)
    start = max(0, hist_len - 8)

    for slot, action in enumerate(action_hist[start:]):
        if slot < 8:
            offset = 133 + slot * 8
            action_idx = min(action, 7)
            features[offset + action_idx] = 1.0

    # 6. Betting context (23 dims): indices 197-219
    ctx_offset = 197
    max_pot = 500.0
    max_stack = 200.0
    max_bet = 200.0

    pot = game_state.pot
    stacks = game_state.stacks
    current_bet = game_state.current_bet
    invested_street = game_state.invested_street
    invested_total = game_state.invested_total

    # Pot (normalized)
    features[ctx_offset] = min(pot / max_pot, 1.0)

    # Player stacks (2 dims)
    features[ctx_offset + 1] = min(stacks[0] / max_stack, 1.0) if len(stacks) > 0 else 0.0
    features[ctx_offset + 2] = min(stacks[1] / max_stack, 1.0) if len(stacks) > 1 else 0.0

    # Current bet (1 dim)
    features[ctx_offset + 3] = min(current_bet / max_bet, 1.0)

    # Invested this street (2 dims)
    features[ctx_offset + 4] = min(invested_street[0] / max_bet, 1.0) if len(invested_street) > 0 else 0.0
    features[ctx_offset + 5] = min(invested_street[1] / max_bet, 1.0) if len(invested_street) > 1 else 0.0

    # Invested total (2 dims)
    features[ctx_offset + 6] = min(invested_total[0] / max_stack, 1.0) if len(invested_total) > 0 else 0.0
    features[ctx_offset + 7] = min(invested_total[1] / max_stack, 1.0) if len(invested_total) > 1 else 0.0

    # Pot odds
    my_invested = invested_street[player] if len(invested_street) > player else 0.0
    to_call = max(0, current_bet - my_invested)
    pot_after = pot + to_call
    pot_odds = to_call / pot_after if pot_after > 0 else 0.0
    features[ctx_offset + 8] = pot_odds

    # Stack-to-pot ratio
    my_stack = stacks[player] if len(stacks) > player else 0.0
    spr = my_stack / pot if pot > 0 else 10.0
    features[ctx_offset + 9] = min(spr / 10.0, 1.0)

    # Position indicator
    features[ctx_offset + 10] = 1.0 if player == 0 else 0.0

    # Actions this street
    features[ctx_offset + 11] = min(game_state.actions_this_street / 10.0, 1.0)

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

# ============================================================================
# Model Management
# ============================================================================

class ModelRegistry:
    """Model checkpoint management."""

    def __init__(self, base_dir: str = "/tmp/full_hunl_dcfr/models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model registry at {self.base_dir}")

    def save_model(self, network: nn.Module, optimizer, metadata: dict) -> str:
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

        meta_path = self.base_dir / f"{model_id}_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved: {model_id}")
        return model_id

    def load_latest(self, device: str = 'cpu') -> tuple[nn.Module, dict]:
        """Load most recent checkpoint."""
        checkpoints = list(self.base_dir.glob("model_*.pt"))
        if not checkpoints:
            return None, {}

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        model_id = latest.stem

        checkpoint = torch.load(latest, map_location=device)

        network = AdvantageNetworkFull().to(device)
        network.load_state_dict(checkpoint['network_state_dict'])
        network.eval()

        logger.info(f"Loaded model: {model_id}")
        return network, checkpoint.get('metadata', {})

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
    elapsed: float = 0.0

    win_rate_mbb: float = 0.0
    loss_history: list = field(default_factory=list)
    action_counts: dict = field(default_factory=lambda: {name: 0 for name in ACTION_NAMES})
    baseline_results: dict = field(default_factory=dict)

    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def add_loss(self, loss: float):
        with self.lock:
            self.loss_history.append({'epoch': self.epoch, 'loss': loss, 'time': time.time()})
            if len(self.loss_history) > 500:
                self.loss_history = self.loss_history[-500:]

    def to_dict(self):
        with self.lock:
            return {
                'running': self.running,
                'epoch': self.epoch,
                'step': self.step,
                'samples': self.samples,
                'total_samples': self.total_samples,
                'samples_per_sec': round(self.samples_per_sec, 1),
                'loss': round(self.loss, 6),
                'elapsed': round(self.elapsed, 1),
                'win_rate_mbb': round(self.win_rate_mbb, 1),
                'loss_history': self.loss_history[-100:],
                'action_counts': dict(self.action_counts),
                'baseline_results': self.baseline_results,
            }

# Global state
training_state = TrainingState()

# ============================================================================
# Training Engine
# ============================================================================

class FullHoldemTrainer:
    """Training engine for Full HUNL."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Create directories
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)

        # Networks
        self.network = AdvantageNetworkFull().to(self.device)
        self.target_network = AdvantageNetworkFull().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Rust trainer
        self.trainer = ParallelTrainerFull(
            data_dir=config.data_dir,
            query_buffer_size=config.query_buffer_size,
            num_workers=config.num_workers,
            small_blind=config.small_blind,
            big_blind=config.big_blind,
            starting_stack=config.starting_stack,
        )

        # Model registry
        self.registry = ModelRegistry(base_dir=f"{config.data_dir}/models")

        # Historical data
        self.all_states = []
        self.all_targets = []
        self.all_weights = []

        logger.info(f"Trainer initialized: {config.traversals_per_epoch:,} traversals/epoch")

    def run_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        epoch_start = time.time()

        # Start epoch in Rust
        self.trainer.start_epoch(epoch)

        # Data generation phase
        gen_start = time.time()
        result = self.trainer.step(num_traversals=self.config.traversals_per_epoch)

        while not result.is_finished():
            if result.is_request_inference():
                # Get query buffer
                query_states = self.trainer.get_query_buffer()
                query_tensor = torch.from_numpy(np.asarray(query_states, dtype=np.float32)).to(self.device)

                # Neural network inference
                with torch.no_grad():
                    advantages = self.target_network(query_tensor)

                # Return predictions
                result = self.trainer.step(inference_results=advantages.cpu().numpy())
            else:
                result = self.trainer.step()

        samples_generated = self.trainer.end_epoch()
        gen_time = time.time() - gen_start

        # Load generated data
        states, targets = load_epoch_data(self.config.data_dir, epoch)
        if len(states) == 0:
            logger.warning(f"No data generated for epoch {epoch}")
            return {'samples': 0, 'loss': 0.0}

        # Add to historical buffer with weight
        weight = 1.0 / (1 + epoch * 0.1)  # DCFR discounting
        self.all_states.append(states)
        self.all_targets.append(targets)
        self.all_weights.extend([weight] * len(states))

        # Training phase
        train_start = time.time()
        epoch_losses = []

        for step in range(self.config.train_steps_per_epoch):
            # Sample batch from historical data
            batch_states, batch_targets = self._sample_batch()

            if len(batch_states) == 0:
                continue

            # Forward pass
            state_tensor = torch.from_numpy(batch_states).to(self.device)
            target_tensor = torch.from_numpy(batch_targets).to(self.device)

            pred = self.network(state_tensor)
            loss = nn.functional.mse_loss(pred, target_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

            # Polyak update
            polyak_update(self.target_network, self.network, self.config.polyak_tau)

            epoch_losses.append(loss.item())

        train_time = time.time() - train_start
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # Update state
        training_state.add_loss(avg_loss)
        training_state.update(
            epoch=epoch,
            samples=samples_generated,
            total_samples=sum(len(s) for s in self.all_states),
            loss=avg_loss,
        )

        # Cleanup old data
        self._cleanup_old_data()

        total_time = time.time() - epoch_start

        return {
            'samples': samples_generated,
            'loss': avg_loss,
            'gen_time': gen_time,
            'train_time': train_time,
            'total_time': total_time,
        }

    def _sample_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a batch from historical data with weighting."""
        if not self.all_states:
            return np.array([]), np.array([])

        # Concatenate all data
        all_states = np.concatenate(self.all_states, axis=0)
        all_targets = np.concatenate(self.all_targets, axis=0)

        n_samples = len(all_states)
        batch_size = min(self.config.train_batch_size, n_samples)

        # Weighted sampling
        weights = np.array(self.all_weights[:n_samples], dtype=np.float64)
        weights = weights / weights.sum()

        indices = np.random.choice(n_samples, size=batch_size, replace=False, p=weights)

        return all_states[indices], all_targets[indices]

    def _cleanup_old_data(self):
        """Remove old data to limit memory usage."""
        max_samples = 10_000_000  # 10M samples max

        total = sum(len(s) for s in self.all_states)
        while total > max_samples and len(self.all_states) > 1:
            removed = len(self.all_states[0])
            self.all_states.pop(0)
            self.all_targets.pop(0)
            self.all_weights = self.all_weights[removed:]
            total -= removed

    def save_checkpoint(self) -> str:
        """Save current model."""
        metadata = {
            'epoch': training_state.epoch,
            'total_samples': training_state.total_samples,
            'loss': training_state.loss,
            'config': {
                'state_dim': STATE_DIM,
                'target_dim': TARGET_DIM,
                'hidden_dim': 512,
            }
        }
        return self.registry.save_model(self.network, self.optimizer, metadata)

# ============================================================================
# Flask Web Dashboard
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'full-hunl-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

trainer: Optional[FullHoldemTrainer] = None
training_thread: Optional[threading.Thread] = None

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Full HUNL Training Dashboard</title>
    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
    <style>
        body { font-family: 'Monaco', monospace; margin: 20px; background: #1a1a2e; color: #eee; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #4ecca3; }
        .card { background: #16213e; padding: 20px; border-radius: 8px; margin: 10px 0; }
        .stat { display: inline-block; margin: 10px 20px; }
        .stat-value { font-size: 24px; color: #4ecca3; }
        .stat-label { color: #888; font-size: 12px; }
        button { background: #4ecca3; color: #1a1a2e; border: none; padding: 10px 20px;
                 border-radius: 5px; cursor: pointer; margin: 5px; font-weight: bold; }
        button:hover { background: #6ee7b7; }
        button.stop { background: #e94560; }
        #log { background: #0f0f1a; padding: 10px; border-radius: 5px;
               max-height: 300px; overflow-y: auto; font-size: 12px; }
        .log-entry { margin: 2px 0; }
        .street-name { color: #ff9f1c; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Full HUNL Training Dashboard</h1>

        <div class="card">
            <button onclick="startTraining()">Start Training</button>
            <button onclick="stopTraining()" class="stop">Stop</button>
            <button onclick="saveModel()">Save Model</button>
        </div>

        <div class="card">
            <div class="stat">
                <div class="stat-value" id="epoch">0</div>
                <div class="stat-label">Epoch</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="samples">0</div>
                <div class="stat-label">Samples</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="loss">0.0000</div>
                <div class="stat-label">Loss</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="throughput">0</div>
                <div class="stat-label">Samples/sec</div>
            </div>
        </div>

        <div class="card">
            <h3>Training Log</h3>
            <div id="log"></div>
        </div>
    </div>

    <script>
        const socket = io();

        socket.on('training_update', (data) => {
            document.getElementById('epoch').textContent = data.epoch;
            document.getElementById('samples').textContent = data.total_samples.toLocaleString();
            document.getElementById('loss').textContent = data.loss.toFixed(6);
            document.getElementById('throughput').textContent = data.samples_per_sec.toFixed(0);
        });

        socket.on('log_message', (msg) => {
            const log = document.getElementById('log');
            log.innerHTML += '<div class="log-entry">' + msg + '</div>';
            log.scrollTop = log.scrollHeight;
        });

        function startTraining() {
            fetch('/api/train/start', { method: 'POST' });
        }

        function stopTraining() {
            fetch('/api/train/stop', { method: 'POST' });
        }

        function saveModel() {
            fetch('/api/model/save', { method: 'POST' })
                .then(r => r.json())
                .then(data => alert('Saved: ' + data.model_id));
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def api_status():
    return jsonify(training_state.to_dict())

@app.route('/api/train/start', methods=['POST'])
def api_start_training():
    global trainer, training_thread

    if training_state.running:
        return jsonify({'error': 'Already running'}), 400

    trainer = FullHoldemTrainer(ACTIVE_CONFIG)
    training_state.update(running=True)

    def train_loop():
        epoch = 0
        while training_state.running:
            result = trainer.run_epoch(epoch)

            msg = f"Epoch {epoch}: {result['samples']:,} samples, loss={result['loss']:.6f}"
            socketio.emit('log_message', msg)
            socketio.emit('training_update', training_state.to_dict())

            epoch += 1

            # Auto-save every 10 epochs
            if epoch % 10 == 0:
                trainer.save_checkpoint()

        training_state.update(running=False)

    training_thread = threading.Thread(target=train_loop, daemon=True)
    training_thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/train/stop', methods=['POST'])
def api_stop_training():
    training_state.update(running=False)
    return jsonify({'status': 'stopping'})

@app.route('/api/model/save', methods=['POST'])
def api_save_model():
    if trainer is None:
        return jsonify({'error': 'No trainer'}), 400

    model_id = trainer.save_checkpoint()
    return jsonify({'model_id': model_id})

# ============================================================================
# CLI
# ============================================================================

cli = typer.Typer(help="Full HUNL Deep CFR Training")

@cli.command()
def serve(
    port: int = typer.Option(5002, "--port", "-p", help="Server port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Start training dashboard."""
    setup_logging(verbose=verbose)

    console.print(Panel.fit(
        "[bold green]Full HUNL Training Dashboard[/bold green]\n"
        f"[dim]220-dim state, 8 actions, 4 streets[/dim]",
        title="Aion-26"
    ))

    console.print(f"\nDashboard: [cyan]http://localhost:{port}[/cyan]")

    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)

@cli.command()
def info():
    """Show configuration info."""
    table = Table(title="Full HUNL Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("State Dimension", str(STATE_DIM))
    table.add_row("Action Dimension", str(TARGET_DIM))
    table.add_row("Traversals/Epoch", f"{ACTIVE_CONFIG.traversals_per_epoch:,}")
    table.add_row("Batch Size", f"{ACTIVE_CONFIG.train_batch_size:,}")
    table.add_row("Learning Rate", str(ACTIVE_CONFIG.learning_rate))
    table.add_row("Hidden Dim", "512")
    table.add_row("Streets", ", ".join(STREET_NAMES))
    table.add_row("Actions", ", ".join(ACTION_NAMES))

    console.print(table)

if __name__ == "__main__":
    cli()
