# Aion-26: Deep PDCFR+ Framework

A lean, agent-based framework for solving imperfect information games using Deep Predictive Discounted Counterfactual Regret Minimization (Deep PDCFR+).

## Overview

Aion-26 implements state-of-the-art game theory algorithms to find Nash equilibrium strategies in games like poker. The project follows a three-phase lean development approach:

- **Phase 1** (Current): Vanilla CFR on Kuhn Poker - Validate algorithmic correctness
- **Phase 2**: Deep CFR with neural networks - Scale to larger games
- **Phase 3**: Full PDCFR+ with dynamic discounting and predictive updates

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
│   ├── memory/         # Experience replay buffers
│   ├── networks/       # Neural networks (Phase 2+)
│   ├── learner/        # Training loop and optimizers (Phase 2+)
│   ├── metrics/        # Exploitability and validation
│   └── utils/          # Configuration and utilities
├── tests/              # Unit and integration tests
├── scripts/            # Training scripts
└── configs/            # Experiment configurations
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

## References

Based on cutting-edge research in:
- Discounted CFR (DCFR)
- Deep CFR with neural network approximation
- Predictive CFR+ (PCFR+)
- Dynamic discounting optimization

## License

MIT License - See LICENSE file for details
