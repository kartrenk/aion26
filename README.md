# Aion-26: Deep PDCFR+ Framework

A lean, agent-based framework for solving imperfect information games using Deep Predictive Discounted Counterfactual Regret Minimization (Deep PDCFR+).

## Overview

Aion-26 implements state-of-the-art game theory algorithms to find Nash equilibrium strategies in games like poker. The project follows a three-phase lean development approach:

- **Phase 1** ✅ COMPLETE: Vanilla CFR on Kuhn Poker - Validate algorithmic correctness
- **Phase 2** ✅ COMPLETE: Deep CFR with neural networks - Scale to larger games
- **Phase 3** ✅ COMPLETE: VR-DDCFR+ with variance reduction and dynamic discounting

**Latest**: Now includes a **GUI visualizer** for real-time training monitoring!

## Installation

```bash
# Install dependencies
uv sync

# Install with optional features
uv sync --extra deep      # Neural network support (PyTorch)
uv sync --extra metrics   # WandB, matplotlib for tracking
```

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/aion26 --cov-report=term-missing

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type checking
uv run mypy src/
```

## Project Structure

```
aion26/
├── src/aion26/
│   ├── games/          # Game implementations (Kuhn, Leduc, RPS)
│   ├── cfr/            # CFR algorithm engine
│   ├── deep_cfr/       # Neural networks (Advantage, Value, Encoders)
│   ├── memory/         # Experience replay buffers
│   ├── learner/        # Training loop and optimizers
│   ├── metrics/        # Exploitability and validation
│   ├── gui/            # GUI visualizer (Tkinter)
│   ├── config.py       # YAML configuration system
│   └── utils/          # Utilities
├── tests/              # Unit and integration tests
├── scripts/            # Training scripts and GUI launcher
└── configs/            # Experiment configurations (YAML)
```

## Phase 1: Kuhn Poker ✅ COMPLETE

The initial phase implements tabular vanilla CFR on Kuhn Poker:

```bash
# Train vanilla CFR on Kuhn Poker
PYTHONPATH=src:$PYTHONPATH .venv/bin/python scripts/train_kuhn.py --iterations 10000

# Generate interactive HTML report
PYTHONPATH=src:$PYTHONPATH .venv/bin/python scripts/generate_report.py

# View reports
open phase1_report.html              # Interactive dashboard
open docs/phase1_report.md           # Detailed markdown report
```

**Results**:
- ✅ Test coverage: 87% (47 tests passing)
- ✅ Convergence time: 6.8s for 10,000 iterations (~1,478 it/s)
- ⚠️ Exploitability: ~0.54 (needs tuning, MCCFR converges slowly)
- ✅ Code quality: mypy + ruff clean

**Phase 1 Reports**:
- 📊 [Interactive HTML Dashboard](file://phase1_report.html) - Visual training metrics
- 📄 [Detailed Markdown Report](docs/phase1_report.md) - Complete analysis & next steps

## Phase 2 & 3: Deep PDCFR+ ✅ COMPLETE

Successfully implemented Deep CFR with neural networks, variance reduction, and dynamic discounting:

**Achievements**:
- ✅ **34.5× speedup** with External Sampling MCCFR
- ✅ **42.6% NashConv improvement** with VR-DDCFR+ (0.7848 → 0.4502)
- ✅ Variance Reduction with Value Network baseline
- ✅ DDCFR strategy weighting (t^γ)
- ✅ Bootstrapped target networks with Polyak averaging
- ✅ Reservoir sampling buffers

**Reports**:
- 📄 [VR-DDCFR Completion Report](VR_DDCFR_COMPLETION.md) - Full implementation details

## GUI Visualizer 🎨 NEW

Launch the interactive GUI for real-time training visualization:

```bash
# Launch GUI
python scripts/launch_gui.py

# Or with uv
PYTHONPATH=src uv run python scripts/launch_gui.py
```

**Features**:
- 🎛️ **Configuration Panel**: Game selection, algorithm tuning, hyperparameters
- 📊 **Real-time Plotting**: Live NashConv convergence visualization
- 🔍 **Strategy Inspector**: View strategy evolution for all information sets
- 💾 **Config Management**: Save/load experiments as YAML files
- ⚡ **Non-blocking Training**: Background threads keep UI responsive

**Sample Configs**:
- `configs/leduc_vr_ddcfr.yaml` - Leduc Poker with VR-DDCFR+ (SOTA)
- `configs/kuhn_vanilla.yaml` - Kuhn Poker baseline

**Note**: Requires Python with tkinter support. See [GUI_IMPLEMENTATION_REPORT.md](GUI_IMPLEMENTATION_REPORT.md) for details.

## References

Based on cutting-edge research in:
- Discounted CFR (DCFR)
- Deep CFR with neural network approximation
- Predictive CFR+ (PCFR+)
- Dynamic discounting optimization

## License

MIT License - See LICENSE file for details
