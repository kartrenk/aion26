# Aion-26 Deep PDCFR+ Specs

## Performance Metrics
- **Throughput**: 45,000-55,000 samples/s (MPS/Apple Silicon)
- **Training target**: 100M samples over 2000 epochs
- **Batch size**: 4096
- **Query buffer**: 512 states per GPU inference call

## Architecture
- **Algorithm**: Deep PDCFR+ with External Sampling MCCFR
- **Network**: 3-layer MLP, 256 hidden units, 167K parameters
- **State dim**: 136 features
- **Action dim**: 4 (fold, check/call, bet/raise, all-in)
- **Game**: Heads-up River Hold'em

## Convergence (Epoch 100, ~4M samples)
- **Loss**: 0.003 (MSE on regret targets)
- **Nash Conv proxy**: 0.0048
- **vs RandomBot**: +102 mbb/h
- **vs CallingStation**: +326 mbb/h
- **Self-play**: -64 mbb/h (~0, balanced)

## Reference Comparisons
- Libratus (2017): ~15M core-hours for full HUNL
- Pluribus (2019): 8 days on 64-core server
- ReBeL (2020): Claims 1000x speedup over Libratus
- OpenSpiel Deep CFR: ~1000 samples/s typical on CPU
