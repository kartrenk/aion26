#!/usr/bin/env python3
"""Real-time Training Dashboard for VR-Deep PDCFR+.

This webapp provides a live dashboard showing training metrics:
- Samples/second throughput
- Loss curves
- Win rate progress
- GPU utilization

Run with: python scripts/train_webapp.py
Then open: http://localhost:5000
"""

import sys
import time
import threading
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template_string
from flask_socketio import SocketIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aion26_rust import ParallelTrainer

# ============================================================================
# Configuration
# ============================================================================

STATE_DIM = 136
TARGET_DIM = 4
RECORD_SIZE = (STATE_DIM + TARGET_DIM) * 4  # 560 bytes

@dataclass
class TrainConfig:
    epochs: int = 30  # More epochs for better convergence
    traversals_per_epoch: int = 200_000
    num_workers: int = 2048
    query_buffer_size: int = 8192
    train_batch_size: int = 8192  # Larger batch for stability
    train_steps_per_epoch: int = 200  # More training per epoch
    learning_rate: float = 3e-4  # Lower LR for stability
    data_dir: str = "/tmp/vr_dcfr_webapp"


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
# Training State
# ============================================================================

class TrainingState:
    def __init__(self):
        self.running = False
        self.epoch = 0
        self.step = 0
        self.samples = 0
        self.samples_per_sec = 0.0
        self.loss = 0.0
        self.batch_size = 0
        self.elapsed = 0.0
        self.history = deque(maxlen=100)  # Last 100 data points
        self.lock = threading.Lock()

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.history.append({
                'epoch': self.epoch,
                'step': self.step,
                'samples': self.samples,
                'samples_per_sec': self.samples_per_sec,
                'loss': self.loss,
                'time': time.time(),
            })

    def to_dict(self):
        with self.lock:
            return {
                'running': self.running,
                'epoch': self.epoch,
                'step': self.step,
                'samples': self.samples,
                'samples_per_sec': self.samples_per_sec,
                'loss': self.loss,
                'batch_size': self.batch_size,
                'elapsed': self.elapsed,
                'history': list(self.history),
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
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        .stat-value {
            font-size: 3em;
            font-weight: bold;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label {
            color: #888;
            margin-top: 10px;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart-container {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .status {
            text-align: center;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        .status.running { background: rgba(0, 212, 255, 0.2); color: #00d4ff; }
        .status.stopped { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }
        .btn {
            display: inline-block;
            padding: 15px 40px;
            font-size: 1.1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin: 10px;
            transition: transform 0.2s;
        }
        .btn:hover { transform: scale(1.05); }
        .btn-start { background: linear-gradient(90deg, #00d4ff, #7b2cbf); color: white; }
        .btn-stop { background: #ff6b6b; color: white; }
        .controls { text-align: center; margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>VR-Deep PDCFR+ Training Dashboard</h1>

        <div id="status" class="status stopped">IDLE</div>

        <div class="controls">
            <button class="btn btn-start" onclick="startTraining()">Start Training</button>
            <button class="btn btn-stop" onclick="stopTraining()">Stop Training</button>
        </div>

        <div class="stats-grid">
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
                <div class="stat-value" id="batch-size">0</div>
                <div class="stat-label">Batch Size</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="throughputChart"></canvas>
        </div>
    </div>

    <script>
        const socket = io();

        // Chart setup
        const ctx = document.getElementById('throughputChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Samples/Second',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#888' } }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888' }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888' },
                        beginAtZero: true
                    }
                }
            }
        });
        document.getElementById('throughputChart').parentElement.style.height = '300px';

        socket.on('update', function(data) {
            document.getElementById('samples-sec').textContent =
                data.samples_per_sec.toLocaleString(undefined, {maximumFractionDigits: 0});
            document.getElementById('total-samples').textContent =
                data.samples.toLocaleString();
            document.getElementById('epoch').textContent = data.epoch;
            document.getElementById('batch-size').textContent = data.batch_size;

            const statusEl = document.getElementById('status');
            if (data.running) {
                statusEl.className = 'status running';
                statusEl.textContent = 'TRAINING - Epoch ' + data.epoch;
            } else {
                statusEl.className = 'status stopped';
                statusEl.textContent = 'IDLE';
            }

            // Update chart
            if (data.history && data.history.length > 0) {
                chart.data.labels = data.history.map((_, i) => i);
                chart.data.datasets[0].data = data.history.map(h => h.samples_per_sec);
                chart.update('none');
            }
        });

        function startTraining() {
            fetch('/start', { method: 'POST' });
        }

        function stopTraining() {
            fetch('/stop', { method: 'POST' });
        }
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


def broadcast_state():
    socketio.emit('update', state.to_dict())


# ============================================================================
# Training Loop
# ============================================================================

def training_loop():
    import sys
    try:
        print("[Training] Starting training loop...", flush=True)
        config = TrainConfig()

        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[Training] Device: {device}", flush=True)
        net = AdvantageNetwork().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

        state.running = True
        state.update(epoch=0, step=0, samples=0)
        broadcast_state()

        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        print(f"[Training] Data dir: {config.data_dir}", flush=True)

        for epoch in range(config.epochs):
            if not state.running:
                break

            print(f"[Training] Starting epoch {epoch+1}/{config.epochs}", flush=True)
            trainer = ParallelTrainer(
                config.data_dir,
                query_buffer_size=config.query_buffer_size,
                num_workers=config.num_workers
            )
            trainer.start_epoch(epoch)

            epoch_start = time.time()
            epoch_samples = 0
            step = 0

            result = trainer.step(None, num_traversals=config.traversals_per_epoch)

            while not result.is_finished() and state.running:
                if result.is_request_inference() and result.count() > 0:
                    states = trainer.get_query_buffer()
                    states_tensor = torch.from_numpy(np.asarray(states)).to(device)

                    with torch.no_grad():
                        preds = net(states_tensor)

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
                        samples_per_sec=samples_per_sec,
                        batch_size=result.count() if result.count() > 0 else state.batch_size,
                        elapsed=elapsed
                    )
                    broadcast_state()

            epoch_samples = result.samples()
            trainer.end_epoch()

            # Traversal phase complete
            elapsed = time.time() - epoch_start
            print(f"[Training] Epoch {epoch+1} traversal complete: {epoch_samples} samples, {epoch_samples/elapsed:.0f} samples/s", flush=True)

            # ================================================================
            # Network Training Phase
            # ================================================================
            print(f"[Training] Starting network training...", flush=True)
            train_start = time.time()

            # Load samples from this epoch
            train_states, train_targets = load_epoch_data(config.data_dir, epoch)
            if len(train_states) == 0:
                print(f"[Training] WARNING: No training data for epoch {epoch}", flush=True)
                continue

            train_states = torch.from_numpy(train_states).to(device)
            train_targets = torch.from_numpy(train_targets).to(device)

            # Train network on collected samples
            net.train()
            total_loss = 0.0
            num_batches = 0

            for train_step in range(config.train_steps_per_epoch):
                if not state.running:
                    break

                # Random batch
                indices = torch.randint(0, len(train_states), (config.train_batch_size,))
                batch_states = train_states[indices]
                batch_targets = train_targets[indices]

                # Forward pass
                predictions = net(batch_states)

                # MSE loss on regret prediction
                loss = torch.nn.functional.mse_loss(predictions, batch_targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Update state
                if train_step % 10 == 0:
                    avg_loss = total_loss / max(1, num_batches)
                    state.update(
                        epoch=epoch + 1,
                        step=train_step,
                        loss=avg_loss,
                    )
                    broadcast_state()

            net.eval()
            train_elapsed = time.time() - train_start
            avg_loss = total_loss / max(1, num_batches)

            print(f"[Training] Network training complete: loss={avg_loss:.4f}, time={train_elapsed:.1f}s", flush=True)
            state.update(
                epoch=epoch + 1,
                samples=epoch_samples,
                samples_per_sec=epoch_samples / elapsed if elapsed > 0 else 0,
                loss=avg_loss,
                elapsed=elapsed + train_elapsed
            )
            broadcast_state()

        # Save checkpoint
        checkpoint_path = Path(config.data_dir) / "model_final.pt"
        torch.save({
            'network_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': config.epochs,
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
    print("VR-Deep PDCFR+ Training Dashboard")
    print("="*60)
    print("\nOpen http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop\n")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)
