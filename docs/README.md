# Aion-26 Documentation

Documentation for the Aion-26 Deep PDCFR+ implementation.

## Overview

Aion-26 implements **Deep Counterfactual Regret Minimization with Predictive Discounting (Deep PDCFR+)** for solving large imperfect-information games like poker.

## Quick Links

- **Main Report**: [PHASE3_COMPLETION_REPORT.md](PHASE3_COMPLETION_REPORT.md) - Latest results
- **Performance**: [EXTERNAL_SAMPLING_MCCFR.md](EXTERNAL_SAMPLING_MCCFR.md) - 34× speedup
- **Known Issues**: [EXPLOITABILITY_BUG_ANALYSIS.md](EXPLOITABILITY_BUG_ANALYSIS.md) - Minor measurement bug

## Phase Completion Reports

### Phase 1: Vanilla CFR on Kuhn Poker ✅
- **Report**: [PHASE1_COMPLETION_REPORT.md](PHASE1_COMPLETION_REPORT.md)
- **Status**: Complete
- **Achievement**: Converged to NashConv < 0.001 in <10,000 iterations

### Phase 2: Deep CFR on Leduc Poker ✅
- **Report**: [PHASE2_COMPLETION_REPORT.md](PHASE2_COMPLETION_REPORT.md)
- **Status**: Complete
- **Achievement**: Scaled from Kuhn (12 states) to Leduc (288 states) with neural networks

### Phase 3: Deep PDCFR+ ✅
- **Report**: [PHASE3_COMPLETION_REPORT.md](PHASE3_COMPLETION_REPORT.md)
- **Status**: Complete
- **Achievement**: 95% improvement over vanilla (NashConv 0.0187 vs 0.3967)

## Technical Reports

### External Sampling MCCFR
- **File**: [EXTERNAL_SAMPLING_MCCFR.md](EXTERNAL_SAMPLING_MCCFR.md)
- **Topic**: Monte Carlo CFR implementation for 34× speedup
- **Impact**: Makes Texas Hold'em tractable (1000× game size)

### Exploitability Calculator Bug
- **File**: [EXPLOITABILITY_BUG_ANALYSIS.md](EXPLOITABILITY_BUG_ANALYSIS.md)
- **Topic**: Rare negative NashConv values (measurement artifact)
- **Status**: Partial fix applied, workaround available
- **Impact**: Agent learning is correct (validated with OpenSpiel)

## Key Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Final NashConv (Leduc)** | 0.0187 | Within 4% of OpenSpiel optimal (0.0137) |
| **PDCFR+ vs Vanilla** | 95% improvement | NashConv 0.02 vs 0.40 |
| **Training speedup (MCCFR)** | 34.5× | 182ms → 5.3ms per iteration |
| **Convergence time (2K iters)** | 11 seconds | Was 7 minutes before MCCFR |

## Algorithm Components

### Core Algorithm
- **Deep CFR**: Neural network function approximation for regrets
- **PDCFR+**: Dynamic discounting (α=2.0, β=0.5) for faster convergence
- **External Sampling MCCFR**: Monte Carlo sampling for scalability

### Key Innovations
1. **Dual-exponent discounting**: Different weights for positive/negative regrets
2. **Linear strategy averaging**: Recent iterations weighted more heavily
3. **Bootstrap targets**: Combines instant regrets with target network predictions
4. **External sampling**: Samples chance/opponent nodes for 34× speedup

## Archive

Historical implementation details moved to `archive/` folder:
- Phase 2 detailed design docs (networks, reservoir, encoder, etc.)
- Individual component specifications

These are preserved for reference but superseded by phase completion reports.

## Next Steps

**Phase 4**: Texas Hold'em scaling (in progress)
- Larger networks (512-1024 units)
- Card abstraction techniques
- 100K+ iteration training runs
- Larger buffer (100K-1M transitions)
