#!/usr/bin/env python3
"""Enhanced Real-time Training Dashboard for VR-Deep PDCFR+.

Features:
- Live throughput and loss charts
- Win rate evaluation (mbb/h) during training
- Action distribution visualization
- Recent hand log viewer
- Strategy heatmap

Run with: python scripts/train_webapp_enhanced.py
Then open: http://localhost:5001
"""

import sys
import time
import threading
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aion26_rust import ParallelTrainer, RustRiverHoldem

# ============================================================================
# Configuration
# ============================================================================

STATE_DIM = 136
TARGET_DIM = 4
RECORD_SIZE = (STATE_DIM + TARGET_DIM) * 4

@dataclass
class TrainConfig:
    epochs: int = 100
    traversals_per_epoch: int = 200_000
    num_workers: int = 2048
    query_buffer_size: int = 8192
    train_batch_size: int = 8192
    train_steps_per_epoch: int = 200
    learning_rate: float = 3e-4
    polyak_tau: float = 0.005  # Soft update rate for target network
    history_alpha: float = 0.5  # 50% recent / 50% history mixing
    eval_interval: int = 5  # Evaluate every N epochs
    eval_hands: int = 2000  # Hands per evaluation
    data_dir: str = "/tmp/vr_dcfr_enhanced"


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


# ============================================================================
# Network
# ============================================================================

class AdvantageNetwork(nn.Module):
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


# ============================================================================
# State Encoder
# ============================================================================

def encode_state(game_state, player: int) -> np.ndarray:
    """Encode game state for neural network."""
    features = np.zeros(STATE_DIM, dtype=np.float32)

    hands = game_state.hands
    board = game_state.board
    pot = game_state.pot
    stacks = game_state.stacks
    current_bet = game_state.current_bet
    invested = [game_state.player_0_invested, game_state.player_1_invested]

    features[0] = 1.0

    if len(hands) > player:
        hand = hands[player]
        for i, card in enumerate(hand[:2]):
            offset = 10 + i * 17
            rank = card % 13
            suit = card // 13
            features[offset + rank] = 1.0
            features[offset + 13 + suit] = 1.0

    for i, card in enumerate(board[:5]):
        offset = 10 + 34 + i * 17
        rank = card % 13
        suit = card // 13
        features[offset + rank] = 1.0
        features[offset + 13 + suit] = 1.0

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

    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

            # Track throughput history
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
            }


state = TrainingState()

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
    <title>VR-Deep PDCFR+ Training Dashboard</title>
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
        .container { max-width: 1600px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.2em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf, #ff6b6b);
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
        .controls { text-align: center; margin-bottom: 20px; }

        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .hand-log {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.08);
            max-height: 300px;
            overflow-y: auto;
        }
        .hand-entry {
            padding: 10px;
            margin: 5px 0;
            background: rgba(255,255,255,0.02);
            border-radius: 6px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85em;
            border-left: 3px solid #00d4ff;
        }
        .hand-entry.win { border-left-color: #00ff88; }
        .hand-entry.loss { border-left-color: #ff6b6b; }
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
    </style>
</head>
<body>
    <div class="container">
        <h1>VR-Deep PDCFR+ Training Dashboard</h1>
        <p class="subtitle">Real-time poker AI training visualization</p>

        <div id="status" class="status stopped">IDLE - Click Start to begin training</div>

        <div class="controls">
            <button class="btn btn-start" onclick="startTraining()">Start Training</button>
            <button class="btn btn-stop" onclick="stopTraining()">Stop Training</button>
            <button class="btn btn-eval" onclick="runEval()">Run Evaluation</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="win-rate">0</div>
                <div class="stat-label">Win Rate (mbb/h)</div>
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
            <div class="hand-log">
                <div class="chart-title">Recent Hands</div>
                <div id="hand-log-entries"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Performance Summary</div>
                <div style="padding: 20px; font-size: 0.9em; line-height: 1.8;">
                    <div><strong>Best Win Rate:</strong> <span id="best-wr">0</span> mbb/h</div>
                    <div><strong>Training Time:</strong> <span id="train-time">0:00</span></div>
                    <div><strong>Epochs Completed:</strong> <span id="epochs-done">0</span></div>
                    <div><strong>Samples Processed:</strong> <span id="samples-done">0</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let bestWinRate = 0;
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

        socket.on('update', function(data) {
            // Update stat cards
            const wrEl = document.getElementById('win-rate');
            wrEl.textContent = data.win_rate_mbb.toFixed(0);
            wrEl.className = 'stat-value ' + (data.win_rate_mbb >= 0 ? 'positive' : 'negative');

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

            // Action distribution bars
            const dist = data.action_distribution;
            ['fold', 'call', 'raise', 'allin'].forEach(action => {
                const pct = dist[action] || 0;
                document.getElementById(action + '-bar').style.height = pct + '%';
                document.getElementById(action + '-pct').textContent = pct.toFixed(1) + '%';
            });

            // Win rate chart
            if (data.win_rate_history && data.win_rate_history.length > 0) {
                winRateChart.data.labels = data.win_rate_history.map(h => 'E' + h.epoch);
                winRateChart.data.datasets[0].data = data.win_rate_history.map(h => h.mbb);
                winRateChart.update('none');

                bestWinRate = Math.max(bestWinRate, ...data.win_rate_history.map(h => h.mbb));
                document.getElementById('best-wr').textContent = bestWinRate.toFixed(0);
            }

            // Throughput chart
            if (data.throughput_history && data.throughput_history.length > 0) {
                throughputChart.data.labels = data.throughput_history.map((_, i) => i);
                throughputChart.data.datasets[0].data = data.throughput_history.map(h => h.value);
                throughputChart.update('none');
            }

            // Loss chart
            if (data.loss_history && data.loss_history.length > 0) {
                lossChart.data.labels = data.loss_history.map((_, i) => i);
                lossChart.data.datasets[0].data = data.loss_history.map(h => h.loss);
                lossChart.update('none');
            }

            // Recent hands
            const handLog = document.getElementById('hand-log-entries');
            if (data.recent_hands && data.recent_hands.length > 0) {
                handLog.innerHTML = data.recent_hands.slice().reverse().map(h =>
                    `<div class="hand-entry ${h.result > 0 ? 'win' : 'loss'}">
                        ${h.hole_cards} | ${h.action} | ${h.result > 0 ? '+' : ''}${h.result.toFixed(0)} chips
                    </div>`
                ).join('');
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

        function startTraining() { fetch('/start', { method: 'POST' }); startTime = Date.now(); }
        function stopTraining() { fetch('/stop', { method: 'POST' }); }
        function runEval() { fetch('/eval', { method: 'POST' }); }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/start', methods=['POST'])
def start_training():
    if not state.running:
        thread = threading.Thread(target=training_loop, daemon=True)
        thread.start()
    return {'status': 'started'}


@app.route('/stop', methods=['POST'])
def stop_training():
    state.running = False
    return {'status': 'stopped'}


@app.route('/eval', methods=['POST'])
def run_evaluation():
    if hasattr(state, '_network') and state._network is not None:
        thread = threading.Thread(target=lambda: evaluate_model(state._network, state._device), daemon=True)
        thread.start()
        return {'status': 'evaluating'}
    return {'status': 'no_model'}


def broadcast_state():
    socketio.emit('update', state.to_dict())


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(network, device, num_hands: int = 2000) -> float:
    """Evaluate model win rate against RandomBot."""
    network.eval()
    results = []

    for _ in range(num_hands):
        game = RustRiverHoldem(
            stacks=[100.0, 100.0],
            pot=2.0,
            current_bet=0.0,
            player_0_invested=1.0,
            player_1_invested=1.0,
        )
        game = game.apply_action(0)  # Deal

        while not game.is_terminal():
            current_player = game.current_player()
            if current_player == -1:
                break

            legal_actions = game.legal_actions()
            if len(legal_actions) == 1:
                action = legal_actions[0]
            elif current_player == 0:
                # Our bot
                state_enc = encode_state(game, current_player)
                state_tensor = torch.from_numpy(state_enc).unsqueeze(0).to(device)
                with torch.no_grad():
                    advantages = network(state_tensor)[0].cpu().numpy()
                strategy = regret_matching(advantages, legal_actions)
                action = legal_actions[np.random.choice(len(legal_actions), p=strategy)]
                state.add_action(action)
            else:
                # Random opponent
                action = random.choice(legal_actions)

            game = game.apply_action(action)

        returns = game.returns()
        results.append(returns[0])

        # Log hand
        if len(results) <= 20:
            hands = game.hands
            hole_str = "??" if len(hands) == 0 else f"{card_str(hands[0][0])}{card_str(hands[0][1])}"
            state.add_hand({
                'hole_cards': hole_str,
                'action': 'played',
                'result': returns[0]
            })

    mbb = (np.mean(results) / 2.0) * 1000
    state.add_win_rate(mbb)
    broadcast_state()
    return mbb


def card_str(card: int) -> str:
    """Convert card index to string."""
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return ranks[card % 13] + suits[card // 13]


# ============================================================================
# Training Loop
# ============================================================================

def training_loop():
    try:
        print("[Training] Starting training loop with Polyak averaging...", flush=True)
        config = TrainConfig()

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[Training] Device: {device}", flush=True)
        print(f"[Training] Polyak tau: {config.polyak_tau}", flush=True)
        print(f"[Training] History alpha: {config.history_alpha} (lower = more history)", flush=True)

        # Online network (trained via gradient descent)
        online_net = AdvantageNetwork().to(device)
        optimizer = torch.optim.Adam(online_net.parameters(), lr=config.learning_rate)

        # Target network (updated via Polyak averaging - NO HARD UPDATES)
        target_net = AdvantageNetwork().to(device)
        target_net.load_state_dict(online_net.state_dict())
        target_net.eval()  # Target network is always in eval mode

        # Store network reference for evaluation (use online for eval)
        state._network = online_net
        state._device = device

        state.running = True
        state.update(epoch=0, step=0, samples=0, total_samples=0)
        broadcast_state()

        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Training] Data dir: {config.data_dir}", flush=True)

        total_samples = 0

        for epoch in range(config.epochs):
            if not state.running:
                break

            print(f"[Training] Starting epoch {epoch+1}/{config.epochs}", flush=True)

            # Note: We keep old epoch files for historical mixing

            trainer = ParallelTrainer(
                config.data_dir,
                query_buffer_size=config.query_buffer_size,
                num_workers=config.num_workers
            )
            trainer.start_epoch(epoch)

            epoch_start = time.time()
            step = 0

            result = trainer.step(None, num_traversals=config.traversals_per_epoch)

            while not result.is_finished() and state.running:
                if result.is_request_inference() and result.count() > 0:
                    states = trainer.get_query_buffer()
                    states_tensor = torch.from_numpy(np.asarray(states)).to(device)

                    # Use TARGET network for traversal (stable, slowly updated)
                    with torch.no_grad():
                        preds = target_net(states_tensor)

                    result = trainer.step(preds.cpu().numpy())
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
                        batch_size=result.count() if result.count() > 0 else state.batch_size,
                        elapsed=elapsed
                    )
                    broadcast_state()

            epoch_samples = result.samples()
            total_samples += epoch_samples
            trainer.end_epoch()

            elapsed = time.time() - epoch_start
            print(f"[Training] Epoch {epoch+1} traversal: {epoch_samples} samples, {epoch_samples/elapsed:.0f}/s", flush=True)

            # Network Training Phase with Historical Mixing
            print(f"[Training] Training network with historical mixing...", flush=True)
            train_start = time.time()

            # Load data from multiple epochs with recency weighting
            # alpha=0.5 means ~50% recent, ~50% history (prevents catastrophic forgetting)
            alpha = config.history_alpha
            all_states = []
            all_targets = []
            all_weights = []

            for past_epoch in range(epoch + 1):
                past_states, past_targets = load_epoch_data(config.data_dir, past_epoch)
                if len(past_states) > 0:
                    # Weight = alpha^(num_epochs - epoch - 1)
                    weight = alpha ** (epoch - past_epoch)
                    all_states.append(past_states)
                    all_targets.append(past_targets)
                    all_weights.extend([weight] * len(past_states))

            if len(all_states) == 0:
                print(f"[Training] WARNING: No data for epoch {epoch}", flush=True)
                continue

            # Concatenate all historical data
            train_states = torch.from_numpy(np.concatenate(all_states)).to(device)
            train_targets = torch.from_numpy(np.concatenate(all_targets)).to(device)
            sample_weights = torch.tensor(all_weights, dtype=torch.float32, device=device)
            sample_weights = sample_weights / sample_weights.sum()  # Normalize

            print(f"[Training] Historical data: {len(train_states)} samples from {epoch+1} epochs", flush=True)

            online_net.train()
            total_loss = 0.0
            num_batches = 0

            # Move weights to CPU for multinomial (MPS has issues with large multinomial)
            sample_weights_cpu = sample_weights.cpu()

            for train_step in range(config.train_steps_per_epoch):
                if not state.running:
                    break

                # Weighted sampling on CPU (MPS multinomial crashes with large tensors)
                indices = torch.multinomial(sample_weights_cpu, config.train_batch_size, replacement=True)
                batch_states = train_states[indices]
                batch_targets = train_targets[indices]

                # Forward pass on ONLINE network
                predictions = online_net(batch_states)
                loss = torch.nn.functional.mse_loss(predictions, batch_targets)

                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()

                # POLYAK UPDATE: Soft update target network after EVERY batch
                # This prevents the sawtooth pattern by keeping targets stable
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

            print(f"[Training] Network: loss={avg_loss:.4f}, time={train_elapsed:.1f}s (tau={config.polyak_tau})", flush=True)
            state.update(
                epoch=epoch + 1,
                samples=epoch_samples,
                total_samples=total_samples,
                loss=avg_loss,
            )
            broadcast_state()

            # Periodic evaluation
            if (epoch + 1) % config.eval_interval == 0:
                print(f"[Training] Running evaluation...", flush=True)
                mbb = evaluate_model(target_net, device, config.eval_hands)
                print(f"[Training] Win rate: {mbb:+.0f} mbb/h", flush=True)

        # Save checkpoint
        checkpoint_path = Path(config.data_dir) / "model_final.pt"
        torch.save({
            'network_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': config.epochs,
            'total_samples': total_samples,
        }, checkpoint_path)
        print(f"[Training] Model saved to {checkpoint_path}", flush=True)

    except Exception as e:
        print(f"[Training] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        state.running = False
        broadcast_state()
        print("[Training] Training loop ended", flush=True)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("VR-Deep PDCFR+ Enhanced Training Dashboard")
    print("="*60)
    print("\nOpen http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop\n")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
