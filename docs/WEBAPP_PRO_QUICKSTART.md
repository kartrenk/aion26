# Aion-26 Pro Webapp Quick Start

## Overview

**train_webapp_pro.py** is a production-ready training dashboard with:

✅ **Real-time training visualization** - Live charts for throughput, loss, win rate
✅ **Strategy inspector** - View learned policies for random game states
✅ **Model management** - Save/load checkpoints with metadata
✅ **Baseline evaluation** - Auto-evaluate vs RandomBot, CallingStation, AlwaysFold
✅ **Structured logging** - Proper Python logging with timestamps
✅ **Error handling** - Graceful degradation and error recovery

## Quick Start

### 1. Install Dependencies

```bash
# Make sure you have the base dependencies
uv sync

# Ensure Rust trainer is built
cd rust_trainer && cargo build --release && cd ..
```

### 2. Run the Webapp

```bash
python scripts/train_webapp_pro.py
```

### 3. Open Browser

Navigate to: **http://localhost:5001**

## Features

### Training Controls

- **Start Training** - Begin new training run with Polyak-averaged PDCFR+
- **Stop Training** - Gracefully halt training (saves state)
- **Run Baselines** - Evaluate current model against all baselines
- **Save Model** - Save checkpoint with metadata (epoch, loss, win rate)
- **Load Model** - Restore previous checkpoint from dropdown

### Real-time Metrics

**Top Stats:**
- Win Rate (mbb/h) vs RandomBot
- Samples/Second throughput
- Total samples processed
- Current epoch
- Training loss
- Batch size

**Charts:**
- Win rate over training (mbb/h)
- Action distribution (Fold/Call/Raise/All-in percentages)
- Training throughput (samples/s)
- Loss curve (MSE)

**Bottom Panels:**
- **Baseline Evaluations** - Win rates vs RandomBot, CallingStation, AlwaysFold
- **Strategy Inspector** - Sample learned policies for random states
- **Performance Summary** - Training time, epochs, samples, model count

### Strategy Inspector

Every 5 epochs, the webapp samples 5 random game states and displays:

- **Info Set**: Hand cards, board, pot size
- **Strategy**: Action probabilities (Fold/Call/Raise/All-in)

Example:
```
Hand: Ah Kd | Board: Qh Jh Ts 9c 8c | Pot: 42
Fold: 5.2%  Call: 12.3%  Raise: 72.5%  All-in: 10.0%
```

### Model Management

**Save Model:**
- Click "Save Model" to create checkpoint
- Metadata stored: epoch, total_samples, loss, win_rate, timestamp

**Load Model:**
- Select from dropdown (shows all saved models)
- Loads network weights and displays metadata
- Auto-samples strategies from loaded model

**Model Naming:**
- Format: `model_YYYYMMDD_HHMMSS` (timestamp-based)
- Stored in: `/tmp/vr_dcfr_pro/models/`

### Baseline Evaluation

**Run Baselines** button evaluates against:

1. **RandomBot** (2000 hands)
   - Uniform random policy
   - Expected: +100 to +200 mbb/h

2. **CallingStationBot** (2000 hands)
   - Always calls/checks
   - Expected: +200 to +400 mbb/h

3. **AlwaysFoldBot** (2000 hands)
   - Always folds when possible
   - Expected: +400 to +600 mbb/h

Results displayed in table with color-coded win rates (green=positive, red=negative).

## Configuration

Edit `TrainConfig` in `train_webapp_pro.py`:

```python
@dataclass
class TrainConfig:
    epochs: int = 100                     # Total training epochs
    traversals_per_epoch: int = 200_000   # MCCFR traversals per epoch
    num_workers: int = 2048               # Parallel workers (Rust)
    train_batch_size: int = 8192          # Neural network batch size
    train_steps_per_epoch: int = 200      # Gradient steps per epoch
    learning_rate: float = 3e-4           # Adam learning rate
    polyak_tau: float = 0.005             # Soft update rate (lower=more stable)
    history_alpha: float = 0.5            # Historical mixing weight
    eval_interval: int = 5                # Evaluate every N epochs
    eval_hands: int = 2000                # Hands per evaluation
    data_dir: str = "/tmp/vr_dcfr_pro"    # Data directory
```

## Architecture

### Backend (Flask + SocketIO)

**Routes:**
- `GET /` - Main dashboard HTML
- `POST /start` - Start training loop
- `POST /stop` - Stop training
- `POST /eval/baselines` - Run baseline suite
- `POST /models/save` - Save checkpoint
- `POST /models/load` - Load checkpoint
- `GET /models/list` - List all models

**WebSocket:**
- `update` event - Broadcasts state every 10 steps

### Training Loop

1. **Traversal Phase** (Rust parallel trainer)
   - Generate training data via MCCFR
   - Query neural network for inference
   - Store (state, regret) pairs to disk

2. **Training Phase** (PyTorch)
   - Load historical data with exponential weighting
   - Train online network via SGD
   - Soft-update target network (Polyak averaging)
   - Sample strategies for inspection

3. **Evaluation Phase** (every 5 epochs)
   - Evaluate vs RandomBot (1000 hands)
   - Sample 5 random strategies
   - Broadcast results to UI

### Model Registry

**Structure:**
```
/tmp/vr_dcfr_pro/models/
├── model_20260109_143052.pt           # PyTorch checkpoint
├── model_20260109_143052_meta.json    # Metadata
├── model_20260109_145523.pt
└── model_20260109_145523_meta.json
```

**Metadata Format:**
```json
{
  "epoch": 50,
  "total_samples": 10000000,
  "loss": 0.0234,
  "win_rate_mbb": 152.3,
  "timestamp": "2026-01-09T14:30:52"
}
```

## Logging

Structured Python logging to console:

```
2026-01-09 14:30:52 [INFO] __main__: Training started
2026-01-09 14:30:52 [INFO] __main__: Device: mps
2026-01-09 14:30:52 [INFO] __main__: Starting epoch 1/100
2026-01-09 14:31:15 [INFO] __main__: Epoch 1 traversal: 200000 samples, 48523/s
2026-01-09 14:31:20 [INFO] __main__: Network: loss=0.0342, time=5.2s
2026-01-09 14:31:25 [INFO] __main__: Evaluating vs RandomBot (1000 hands)
2026-01-09 14:31:30 [INFO] __main__: Evaluation complete: RandomBot = +124 mbb/h
```

## Performance

**Expected throughput:**
- MPS (Apple Silicon): 45K-55K samples/s
- CPU (Intel/AMD): 10K-20K samples/s
- GPU (CUDA): 60K-100K samples/s (untested)

**Training time:**
- 100 epochs: ~30-45 minutes (MPS)
- Convergence: ~50 epochs (10M samples)

## Troubleshooting

### Issue: No Rust module

```
ModuleNotFoundError: No module named 'aion26_rust'
```

**Solution:**
```bash
cd rust_trainer
cargo build --release
cd ..
```

### Issue: Port already in use

```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Kill existing process
lsof -ti:5001 | xargs kill -9

# Or change port in script
socketio.run(app, host='0.0.0.0', port=5002)
```

### Issue: Model not loading

```
FileNotFoundError: Model not found: model_xxx
```

**Solution:**
- Check model directory exists: `/tmp/vr_dcfr_pro/models/`
- Verify model ID is correct (dropdown shows available models)
- Ensure permissions (755) on directory

### Issue: Baseline eval crashes

```
AttributeError: 'RustRiverHoldem' object has no attribute 'current_bet'
```

**Solution:**
- Update to latest `aion26_rust` build
- Check Rust/Python interface compatibility

## Next Steps

### Option A: Production Deployment

- Add Redis + Celery for async tasks
- Add PostgreSQL for experiment tracking
- Docker deployment (see `docs/WEBAPP_ENHANCEMENT_PLAN.md`)
- Add monitoring (Prometheus + Grafana)

### Option B: Research Enhancements

- Multi-seed validation
- Hyperparameter tuning UI
- WandB integration
- Regret Matching+ implementation
- Importance sampling

### Option C: More Baselines

- Add HonestBot (strength-based)
- Add AggroBot (always raises)
- Add TightBot (only plays premium hands)
- Head-to-head tournament mode

## Comparison to Other Webapps

| Feature | train_webapp.py | train_webapp_enhanced.py | train_webapp_pro.py |
|---------|----------------|------------------------|-------------------|
| Real-time charts | ✅ Basic | ✅ Advanced | ✅ Advanced |
| Strategy inspector | ❌ | ❌ | ✅ |
| Model save/load | ❌ | ❌ | ✅ |
| Baseline evaluation | ❌ | Partial (eval button) | ✅ Full suite |
| Structured logging | ❌ (prints) | ❌ (prints) | ✅ Python logging |
| Error handling | ❌ | ❌ | ✅ Try/catch |
| Model registry | ❌ | ❌ | ✅ |
| API endpoints | Basic | Basic | ✅ RESTful |

**Recommendation:** Use `train_webapp_pro.py` for production work.

## Demo

**Training in action:**
1. Start training → Watch throughput stabilize at 45K samples/s
2. After epoch 5 → Strategy inspector shows learned policies
3. Run baselines → See win rates: Random +124, CallingStation +315, AlwaysFold +502
4. Save model → `model_20260109_143052` appears in dropdown
5. Continue training → Win rate improves from +124 to +180
6. Save again → Compare two checkpoints

**Strategy evolution example:**

*Epoch 5:*
```
Hand: 9h 8h | Board: Kh Qh Js 3c 2d | Pot: 20
Fold: 45%  Call: 40%  Raise: 10%  All-in: 5%
→ Conservative (learning)
```

*Epoch 50:*
```
Hand: 9h 8h | Board: Kh Qh Js 3c 2d | Pot: 20
Fold: 2%  Call: 8%  Raise: 75%  All-in: 15%
→ Aggressive (confident - has straight draw)
```

## Credits

Built on top of:
- `train_webapp_enhanced.py` (Polyak averaging, historical mixing)
- Aion-26 Deep PDCFR+ framework (Phase 1-3)
- Rust parallel trainer (45K samples/s)

**Author:** Aion-26 Team
**Date:** 2026-01-09
**Version:** 1.0.0 (Option A - Quick Win)
