# River Hold'em Divergence Fix - Validation Guide

**Date**: 2026-01-07
**Commit**: `800c08f` - "CRITICAL FIX: Prevent River Hold'em training divergence"
**Status**: ✅ Fixes implemented, awaiting validation

---

## Summary of Fixes

Based on the extensive analysis in `CONVERGENCE_FAILURE_ANALYSIS.md`, we identified and fixed the root cause of training divergence:

### Root Cause
- **Batch size too small** (128) for buffer size (100k)
- **Poor sample coverage** (0.41% per batch vs required 3-10%)
- **No gradient clipping** → exploding gradients

### Fixes Applied

| Component | Fix | Impact |
|-----------|-----|--------|
| **Config** | Buffer 100k → 30k | Ensures buffer fills |
| **Config** | Batch 1024 (enforced) | 3.4% coverage ✅ |
| **Trainer** | Gradient clipping (norm=1.0) | Prevents divergence |
| **GUI** | Auto-config to 30k buffer | Prevents user error |

---

## Validation Plan

### Quick Validation Run (Recommended)

**Goal**: Verify fixes work without waiting 7+ minutes

**Configuration**:
```python
Game: river_holdem
Iterations: 2,000  # Reduced for quick test
Buffer: 30,000
Batch Size: 1,024
Learning Rate: 0.001
Algorithm: DDCFR (α=1.5, β=0.0, γ=2.0)
```

**Expected Duration**: ~90 seconds

**Success Criteria**:
- ✅ Loss starts at ~8,000-9,000
- ✅ Loss decreases to ~6,000-7,000 by iter 2,000
- ✅ NO upward trend after iter 1,000
- ✅ Buffer reaches ~6,000/30,000 (20%)
- ✅ No NaN or Inf values in loss

**How to Run**:

1. **Option A: GUI** (easiest)
   ```bash
   python gui/main.py

   # In GUI:
   - Game: river_holdem
   - Iterations: 2000
   - Batch Size: 1024 (should auto-populate)
   - Buffer: 30000 (should auto-populate)
   - Click "Start Training"
   ```

2. **Option B: Training Script**
   ```bash
   PYTHONPATH=src python scripts/train_river.py --iterations 2000
   ```

3. **Check the log**:
   ```bash
   # After training completes
   tail -100 logs/gui_*.log

   # Look for:
   # - Final loss ~6,000-7,000
   # - No "Iter XXXX: loss=9000+" after iter 1000
   # - Buffer size ~6,000
   ```

---

### Full Validation Run (If Quick Test Passes)

**Goal**: Validate full convergence to target loss

**Configuration**:
```python
Iterations: 10,000
Buffer: 30,000
Batch: 1,024
# Everything else same
```

**Expected Duration**: ~7-8 minutes

**Success Criteria**:
- ✅ Loss decreases monotonically after iter 1,500
- ✅ Final loss <3,000 (target: ~2,500)
- ✅ Buffer fills to 100% by iter ~10,000
- ✅ Variance decreases (stable convergence)
- ✅ Matrix view shows strong hands bet more

**Loss Checkpoints** (compare to previous run):

| Iteration | Old Loss (DIVERGED) | Expected New Loss | Status |
|-----------|---------------------|-------------------|--------|
| 500 | 8,220 | 7,500-8,500 | Should be similar |
| 1,000 | 7,565 | 6,500-7,500 | Should be similar |
| 1,500 | 6,763 | 5,500-6,500 | **CRITICAL: Should continue down** |
| 2,000 | 7,389 ❌ | 5,000-6,000 ✅ | Should NOT increase |
| 5,000 | 8,533 ❌ | 3,500-4,500 ✅ | Should decrease |
| 10,000 | 8,782 ❌ | 2,000-3,000 ✅ | **Target** |

**Key Difference**: After iter 1,500, loss should **continue decreasing** instead of increasing.

---

## What to Look For

### 🟢 Good Signs (Training Working)

1. **Loss Curve**:
   - Starts high (~8,000-9,000)
   - Decreases steadily
   - No major spikes after iter 1,000
   - Reaches <3,000 by iter 10,000

2. **Logs**:
   ```
   [INFO] Iter 1000: loss=7,245, buffer=3,012/30,000
   [INFO] Iter 2000: loss=6,180, buffer=6,024/30,000
   [INFO] Iter 5000: loss=3,890, buffer=15,060/30,000
   [INFO] Iter 10000: loss=2,456, buffer=30,000/30,000 ← Full!
   ```

3. **Buffer Fill**:
   - Progresses smoothly: 3%, 6%, 15%, 30%, ... 100%
   - Reaches 100% by iter ~10,000
   - Sample rate: ~3 samples/iter (consistent)

4. **Matrix View** (GUI):
   - High Card: Fold % high (>50%)
   - Flush/Full House: Bet % high (>40%)
   - Clear upward trend: stronger hands bet more

### 🔴 Bad Signs (Still Broken)

1. **Loss Curve**:
   - Increases after iter 1,500
   - Wild oscillations (range >2,000)
   - Ends higher than iter 2,000
   - NaN or Inf values

2. **Logs**:
   ```
   [ERROR] NaN detected in loss
   [WARNING] Gradient norm: 1e6 (clipped to 1.0)  ← Too many clips
   [INFO] Iter 5000: loss=9,500  ← Should be decreasing!
   ```

3. **Buffer**:
   - Never reaches 100% (stuck at 31%)
   - Erratic fill rate (1-5 samples/iter variance)

4. **Matrix View**:
   - Flat strategies (all hands play same)
   - High Card bets >20% (should fold!)
   - No clear pattern by hand strength

---

## Troubleshooting

### Issue 1: Loss Still Diverging

**Symptoms**: Loss increases after iter 1,500 (same as before)

**Possible Causes**:
1. GUI didn't apply auto-config (still using old batch size 128)
2. Config file saved with old values
3. Wrong algorithm selected

**Fixes**:
1. **Check GUI values BEFORE clicking "Start Training"**:
   ```
   Batch Size: 1024 ← MUST be this!
   Buffer: 30000 ← MUST be this!
   ```

2. **Manually set if auto-config didn't apply**:
   - Change game dropdown away and back to river_holdem
   - Or manually type 1024 and 30000

3. **Use training script instead**:
   ```bash
   PYTHONPATH=src python scripts/train_river.py
   # This uses river_holdem_config() directly
   ```

### Issue 2: NaN or Inf Loss

**Symptoms**: Log shows `loss=nan` or `loss=inf`

**Possible Causes**:
1. Gradient clipping not applied (code didn't save?)
2. Learning rate too high
3. Corrupted checkpoint loaded

**Fixes**:
1. **Verify gradient clipping is in code**:
   ```bash
   grep -A 2 "clip_grad_norm_" src/aion26/learner/deep_cfr.py

   # Should show:
   # torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)
   ```

2. **Reduce learning rate**:
   - Change 0.001 → 0.0005
   - Or use GUI to set 0.0005

3. **Start fresh** (no checkpoint loading)

### Issue 3: Buffer Not Filling

**Symptoms**: Buffer stuck at <10% after 2,000 iterations

**Possible Causes**:
1. Sample collection broken
2. Wrong game selected
3. Iterations too low

**Fixes**:
1. **Check game selected**: MUST be "river_holdem"
2. **Check sample rate in logs**:
   ```bash
   # Should see buffer growing every 10 iterations
   grep "buffer=" logs/gui_*.log | tail -20
   ```

3. **Increase iterations** to 5,000+ to see buffer fill

---

## Expected Output (Success)

### Terminal/Log Output

```
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] TrainingThread created: game=river_holdem, algo=ddcfr, iters=10000
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] Trainer initialized successfully
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 100: loss=0.0000, buffer=300/30000
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 500: loss=8124.5, buffer=1503/30000
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] Skipping NashConv for river_holdem
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 1000: loss=7021.3, buffer=3006/30000
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 1500: loss=6128.7, buffer=4509/30000
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 2000: loss=5453.2, buffer=6012/30000
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] Skipping NashConv for river_holdem
...
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 5000: loss=3876.4, buffer=15030/30000
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 7500: loss=3124.8, buffer=22545/30000
[2026-01-07 XX:XX:XX] DEBUG [aion26.gui.model] Iter 10000: loss=2587.1, buffer=30000/30000  ← FULL!
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] Training completed
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] Skipping final NashConv for river_holdem
[2026-01-07 XX:XX:XX] INFO [aion26.gui.model] Training thread completed successfully
```

**Key Indicators**:
- ✅ Loss decreases from 8,124 → 2,587 (68% improvement)
- ✅ Buffer fills smoothly: 3k → 6k → 15k → 30k
- ✅ No NaN or error messages
- ✅ Completed successfully

### Loss Curve (Conceptual)

```
Loss
^
9k |****                                    (Old run - diverged)
8k |    ****                            ****
7k |        ****                    ****
6k |            ****   (New run) ****
5k |                ****      ****
4k |                    *******
3k |                         ****
2k |                             ****  (Target: converged!)
1k |
   +-------------------------------------------------> Iterations
   0     2k    4k    6k    8k    10k
```

---

## After Validation

### If Validation PASSES ✅

1. **Document results**:
   ```bash
   # Create success report
   echo "Validation PASSED - $(date)" >> VALIDATION_RESULTS.txt
   tail -100 logs/gui_*.log >> VALIDATION_RESULTS.txt
   ```

2. **Run extended training** (optional):
   - 30,000 iterations with 100k buffer
   - Target: Final loss <1,500
   - Duration: ~20-25 minutes

3. **Head-to-head evaluation**:
   ```bash
   PYTHONPATH=src python scripts/train_river.py
   # Includes head-to-head vs baseline bots
   # Expected: +2,000-3,000 mbb/h vs RandomBot
   ```

4. **Update documentation**:
   - Mark CONVERGENCE_FAILURE_ANALYSIS.md as "RESOLVED"
   - Add link to validation results

### If Validation FAILS ❌

1. **Capture diagnostic info**:
   ```bash
   # Save full log
   cp logs/gui_*.log FAILED_VALIDATION_$(date +%Y%m%d_%H%M%S).log

   # Check gradient clipping
   grep "clip_grad" src/aion26/learner/deep_cfr.py

   # Check config values
   python3 -c "from src.aion26.config import river_holdem_config; c=river_holdem_config(); print(f'Batch: {c.training.batch_size}, Buffer: {c.training.buffer_capacity}')"
   ```

2. **Report issue** with:
   - Log file
   - Screenshot of GUI settings
   - Final loss value
   - Buffer fill percentage

3. **Try alternative fixes**:
   - Reduce learning rate: 0.001 → 0.0005
   - Increase gradient clip: 1.0 → 0.5
   - Reduce buffer: 30k → 10k (for 2k iterations)

---

## Quick Commands Reference

```bash
# Run quick validation (2k iterations)
python gui/main.py
# Set: river_holdem, 2000 iters, 1024 batch, 30k buffer

# Run full validation (10k iterations)
PYTHONPATH=src python scripts/train_river.py

# Check latest log
tail -100 logs/gui_$(ls -t logs/gui_*.log | head -1)

# Verify fixes applied
grep "clip_grad_norm_" src/aion26/learner/deep_cfr.py
grep "buffer_capacity=30000" src/aion26/config.py

# Test syntax
python3 -m py_compile src/aion26/{config,learner/deep_cfr,gui/app}.py

# Analyze loss trend
python3 <<EOF
import re
with open('logs/gui_20260107_XXXXXX.log') as f:  # Replace with actual log
    losses = [float(re.search(r'loss=([\d.]+)', line).group(1))
              for line in f if 'loss=' in line and 'loss=0.0000' not in line]
    print(f"Initial: {losses[0]:.0f}, Final: {losses[-1]:.0f}, Change: {losses[-1]-losses[0]:.0f} ({(losses[-1]-losses[0])/losses[0]*100:+.1f}%)")
EOF
```

---

## Success Definition

**Primary Goal**: Loss decreases monotonically after iter 1,500

**Quantitative Targets**:
- 2k iterations: Final loss <7,000 (baseline)
- 10k iterations: Final loss <3,000 (target)
- 30k iterations: Final loss <1,500 (stretch goal)

**Qualitative Indicators**:
- No divergence or oscillation
- Smooth loss curve
- Buffer fills as expected
- Strategy matrix shows sensible patterns

---

**Status**: ⏳ Awaiting user validation run
**Next Step**: Run quick validation (2k iterations) and report results
**Expected Outcome**: Loss ~6,000-7,000 (vs previous 7,389) ✅
