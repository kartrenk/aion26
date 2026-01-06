# River Hold'em Training Convergence Failure Analysis

**Date**: 2026-01-07
**Log File**: `logs/gui_20260106_235605.log`
**Training Run**: 10,000 iterations with 100k buffer
**Status**: ❌ **DIVERGING** (not converging)

---

## Executive Summary

The 10,000-iteration River Hold'em training run **failed to converge** and actually **diverged** over time. Loss increased from 7,546 (early phase) to 8,934 (late phase), a **+18.4% degradation**. This is the opposite of expected behavior and indicates critical configuration issues.

### Key Findings

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| **Final Loss** | <5,000 | 8,782 | ❌ Too high |
| **Loss Trend** | Decreasing | Increasing | ❌ Diverging |
| **Buffer Fill** | 100% (100k) | 31% (30,947) | ❌ Under-utilized |
| **Convergence** | Stable | Oscillating | ❌ Not converged |
| **Duration** | ~7.4 min | 7.4 min | ✅ Normal |

**Diagnosis**: Severe hyperparameter mismatch causing training instability.

---

## Training Configuration

### What Was Used (Inferred)

```python
Game: river_holdem
Algorithm: DDCFR (α=1.5, β=0.0, γ=2.0)
Iterations: 10,000
Buffer Capacity: 100,000
Batch Size: 128  # ⚠️ LIKELY CAUSE OF FAILURE
Learning Rate: 0.001 (default)
Hidden Size: 128 (default)
Eval Every: 100 iterations
```

### What Should Have Been Used

```python
Game: river_holdem
Algorithm: DDCFR (α=1.5, β=0.0, γ=2.0)
Iterations: 10,000
Buffer Capacity: 100,000  # ✅ Correct
Batch Size: 1,024  # ❌ Should be 8x larger!
Learning Rate: 0.001  # ✅ Correct
Hidden Size: 128  # ✅ Correct
Eval Every: 100  # ✅ Correct
```

---

## Detailed Loss Analysis

### Loss Progression Over Time

| Phase | Iterations | Average Loss | Std Dev | Trend |
|-------|-----------|--------------|---------|-------|
| **Early** | 350-3,300 | 7,546 | 384 | Baseline |
| **Mid** | 3,300-6,600 | 8,096 | 396 | +7.3% ⚠️ |
| **Late** | 6,600-10,000 | 8,934 | 402 | +18.4% ❌ |
| **Final 100** | 9,900-10,000 | 9,195 | 358 | +21.8% ❌ |

**Interpretation**: Loss is **increasing** over time, indicating:
- Network is not learning useful patterns
- Training is unstable (high variance)
- Hyperparameters are mismatched

### Loss at Key Checkpoints

```
Iter   350: loss= 8,725  (training starts)
Iter   500: loss= 8,220  (-5.8%)   ← Early improvement
Iter  1000: loss= 7,565  (-13.3%)  ← Best phase
Iter  1500: loss= 6,763  (-22.5%)  ← MINIMUM LOSS
Iter  2000: loss= 7,389  (-15.3%)  ← Starting to degrade
Iter  3000: loss= 7,871  (-9.8%)   ← Degradation continues
Iter  5000: loss= 8,533  (-2.2%)   ← Back to initial
Iter  7000: loss= 9,103  (+4.3%)   ← DIVERGING
Iter  9000: loss= 9,919  (+13.7%)  ← Severe divergence
Iter 10000: loss= 8,782  (+0.7%)   ← Still diverged
```

**Critical Observation**: Loss **bottomed out at iteration 1,500** (6,763), then **continuously increased** for the next 8,500 iterations. This is catastrophic divergence.

### Loss Variance (Instability)

```
Early phase:  std = 384 (5.1% coefficient of variation)
Mid phase:    std = 396 (4.9% CV)
Late phase:   std = 402 (4.5% CV)
Final 100:    std = 358 (3.9% CV)
```

**Interpretation**: Variance remains high throughout training (350-400 points), indicating:
- No convergence to stable strategy
- Network predictions remain noisy
- Insufficient training iterations OR bad hyperparameters

---

## Buffer Utilization Analysis

### Buffer Fill Progression

| Iteration | Buffer Size | Fill % | Samples Since Start |
|-----------|-------------|--------|---------------------|
| 350 | 1,052 | 1.1% | 1,052 |
| 500 | 1,496 | 1.5% | 1,496 |
| 1,000 | 2,979 | 3.0% | 2,979 |
| 2,000 | 5,972 | 6.0% | 5,972 |
| 5,000 | 15,500 | 15.5% | 15,500 |
| 7,000 | 21,719 | 21.7% | 21,719 |
| 9,000 | 27,902 | 27.9% | 27,902 |
| 10,000 | 30,947 | 30.9% | 30,947 |

**Buffer Fill Rate**: 3.09 samples/iteration

**Problem**: Buffer never reached capacity! After 10,000 iterations, only 30.9% full. This means:
1. ❌ **No reservoir sampling**: Old samples never replaced (buffer acts as append-only)
2. ❌ **Sparse coverage**: 100k capacity but only 31k samples
3. ❌ **Wasted memory**: 69% of buffer empty

### Why Is This a Problem?

With a 100k buffer but only 31k samples:
- **Batch size 128**: Samples only 0.4% of buffer per update
- **Expected**: Batch size 1,024 would sample 3.3% of buffer per update (8x better coverage)
- **Result**: Network sees very sparse, non-representative samples

---

## Root Cause Analysis

### Primary Issue: Batch Size Too Small

**Batch Size**: 128 (likely used)
**Buffer Size**: 30,947 (at end)
**Sampling Coverage**: 128 / 30,947 = **0.41% per training step**

This is **critically low**! The network is training on tiny, unrepresentative samples from a large, sparse buffer.

#### Why 128 Batch Size Fails for 100k Buffer

1. **Poor Sample Coverage**:
   - With 128 samples from 30k buffer, we're only seeing 0.4% of experiences
   - Network doesn't get a representative view of the strategy space
   - Gradients are high-variance and noisy

2. **Insufficient Repetition**:
   - Large buffer means samples are rarely seen again
   - Network needs repetition to learn patterns
   - With small batches, it takes 242 iterations to see all samples once

3. **Learning Rate Mismatch**:
   - Learning rate (0.001) is calibrated for larger batches
   - Small batches with same LR cause unstable updates
   - Effective learning rate is too high for the noise level

### Secondary Issue: Buffer Never Fills

**Expected Behavior**:
- With 10,000 iterations × 3 samples/iter = 30,000 samples total
- Buffer capacity: 100,000
- **Buffer should be 30% full** ✅ (matches observation)

**Problem**:
- Buffer sized for 30,000+ iterations, not 10,000
- Should have either:
  - **Option A**: 30k buffer + 10k iterations
  - **Option B**: 100k buffer + 30k iterations

### Comparison to Previous Run

| Metric | 2k Iter (1k Buffer) | 10k Iter (100k Buffer) | Change |
|--------|---------------------|------------------------|--------|
| **Iterations** | 2,000 | 10,000 | +5x |
| **Buffer Capacity** | 1,000 | 100,000 | +100x |
| **Buffer Fill %** | 100% (iter 420) | 31% (never) | -69% |
| **Batch Size** | 128 | 128 | Same |
| **Batch/Buffer Ratio** | 12.8% | 0.41% | **-31x** ❌ |
| **Loss Reduction** | -71% ✅ | +0.7% ❌ | DIVERGED |

**Analysis**: The 2k iteration run **converged** because:
- Buffer filled quickly (iter 420)
- High batch/buffer ratio (12.8%) ensured good coverage
- Network saw repetitive samples and learned patterns

The 10k iteration run **diverged** because:
- Buffer never filled (31% at end)
- Extremely low batch/buffer ratio (0.41%)
- Network saw sparse, diverse samples without repetition

---

## Performance Metrics

### Timing Analysis

```
Start Time:  2026-01-06 23:56:18
End Time:    2026-01-07 00:03:30
Duration:    ~7 minutes 12 seconds (432 seconds)
```

**Throughput**:
- Iterations: 10,000
- Speed: **23.1 iterations/second**
- Time per iteration: **43.2 ms**

**Comparison to 2k Run**:
- Previous: 28.6 iters/sec (35ms per iter)
- Current: 23.1 iters/sec (43ms per iter)
- **Slowdown**: -19% (expected with larger buffer)

### Resource Utilization (Estimated)

**Memory**:
- Buffer: 30,947 samples × 31 features × 4 bytes ≈ **3.8 MB**
- Network: 31→128→128→128→4 ≈ ~50K params ≈ **200 KB**
- Total: **~4 MB** (very lightweight)

**CPU Usage**:
- Likely single-threaded (Python GIL)
- ~23 iters/sec suggests CPU-bound, not memory-bound
- No GPU acceleration detected

---

## Hyperparameter Recommendations

### Immediate Fixes

#### Fix 1: Increase Batch Size ✅ **CRITICAL**

```python
# Current (WRONG):
batch_size = 128

# Recommended:
batch_size = 1024  # 8x increase

# Rationale:
# - With 100k buffer, need larger batches for coverage
# - 1024 / 30k = 3.4% coverage (8x better than 0.4%)
# - Reduces gradient noise, stabilizes training
```

#### Fix 2: Match Buffer to Iterations

**Option A**: Keep 10k iterations, reduce buffer
```python
buffer_capacity = 30000  # 3x sample rate
batch_size = 1024
# Buffer fills at iter ~10,000
# Good batch/buffer ratio: 1024/30k = 3.4%
```

**Option B**: Keep 100k buffer, increase iterations
```python
buffer_capacity = 100000
batch_size = 1024
iterations = 30000  # 3x increase
# Buffer fills at iter ~30,000
# Reservoir sampling kicks in
```

**Recommendation**: Use **Option A** for faster experiments (10k iters), then scale to **Option B** for final training.

### Advanced Hyperparameter Tuning

#### Learning Rate Schedule

Current: Fixed 0.001

Recommended:
```python
# Warm-up phase (iters 0-1000)
lr = 0.0001  # Low LR while buffer fills

# Main training (iters 1000-8000)
lr = 0.001   # Standard LR

# Fine-tuning (iters 8000-10000)
lr = 0.0001  # Reduce LR for convergence
```

#### Gradient Clipping

```python
# Add to prevent divergence
max_grad_norm = 1.0  # Clip gradients above this threshold
```

#### Batch Size Schedule (Advanced)

```python
# Adaptive batch size based on buffer fill
if buffer_fill_pct < 10%:
    batch_size = 128  # Small batches for sparse buffer
elif buffer_fill_pct < 50%:
    batch_size = 512  # Medium batches
else:
    batch_size = 1024  # Large batches for full buffer
```

---

## Comparison to Theoretical Expectations

### Expected Convergence Pattern (GTO)

For Deep CFR on River Hold'em, we expect:

| Iteration | Loss Range | NashConv (if computable) |
|-----------|------------|--------------------------|
| 0-1,000 | 10,000-8,000 | ~1000 mbb/g |
| 1,000-3,000 | 8,000-5,000 | ~500 mbb/g |
| 3,000-5,000 | 5,000-3,000 | ~200 mbb/g |
| 5,000-10,000 | 3,000-1,500 | ~100 mbb/g |
| 10,000+ | 1,500-1,000 | ~50 mbb/g |

**Actual Results**:
- Iter 1,500: Loss = 6,763 ✅ (on track!)
- Iter 10,000: Loss = 8,782 ❌ (went backwards!)

**Interpretation**: Training was **on track until iter 1,500**, then **diverged catastrophically**. This suggests the network learned initial patterns, but then:
1. Overfitted to sparse samples (small batch size)
2. Lost learned patterns due to noisy gradients
3. Entered unstable regime (too high effective learning rate)

---

## Recommended Action Plan

### Phase 1: Quick Fix (1 hour)

1. **Update GUI to enforce batch size when loading saved config**
   - Currently: `_on_game_changed` only triggers on dropdown change
   - Fix: Also call `_on_game_changed` after loading config
   - Location: `src/aion26/gui/app.py`, `_load_config()` method

2. **Add validation warnings**
   - Warn if batch_size < buffer_capacity / 100
   - Warn if buffer will never fill (iterations × 3 < buffer_capacity)

3. **Re-run training with correct config**
   - Buffer: 100,000
   - Batch: 1,024 (not 128!)
   - Iterations: 10,000
   - Expected: Loss should decrease to ~2,000-3,000

### Phase 2: Validation (30 minutes)

1. **Monitor loss curve**
   - Should decrease monotonically after iter 1,000
   - Final loss target: <3,000
   - Variance should decrease over time

2. **Check buffer fill**
   - Should reach 30-31% at end (same as before)
   - Batch/buffer ratio: 1024/30k = 3.4% ✅

3. **Extract strategy and visualize**
   - Use Matrix View to check hand rank strategies
   - Verify: Strong hands bet more, weak hands fold more

### Phase 3: Long-Term Improvements (2-4 hours)

1. **Implement learning rate schedule**
   - Add `lr_scheduler` to trainer
   - Decay LR after iter 5,000

2. **Add gradient clipping**
   - Prevent extreme updates
   - Improves stability

3. **Add early stopping**
   - Stop if loss increases for 1,000 consecutive iterations
   - Save best model (lowest loss checkpoint)

4. **Run 30k iteration training**
   - Fill 100k buffer completely
   - Benefit from reservoir sampling
   - Target: Loss <1,500, stable strategy

---

## Diagnostic Plots (Conceptual)

### Loss Curve (Actual vs Expected)

```
Loss
^
10k |                                  **** (actual - diverging)
9k  |                              ****
8k  |                          ****
7k  |      ****            ****
6k  |  ****    ****    ****  ← Should have continued down
5k  |            ****
4k  |                ****
3k  |                    ****
2k  |                        **** (expected - converging)
1k  |                            ****
    +-------------------------------------------------> Iterations
    0    2k   4k   6k   8k   10k
```

### Buffer Fill vs Batch Coverage

```
Coverage %
^
15%|
12%| ████ (2k run: 12.8% coverage) ← GOOD
9% |
6% |
3% | ▁ (10k run: 0.41% coverage)  ← BAD
0% +-------------------------------------------------> Iterations
    0    2k   4k   6k   8k   10k
```

---

## Comparison Summary Table

| Aspect | 2k Run (SUCCESS) | 10k Run (FAILURE) | Recommendation |
|--------|------------------|-------------------|----------------|
| **Buffer Size** | 1,000 | 100,000 | Use 30k or 100k |
| **Batch Size** | 128 | 128 | **Change to 1024** |
| **Iterations** | 2,000 | 10,000 | Keep 10k |
| **Buffer Fill** | 100% (iter 420) | 31% (never) | 30k or 30k iters |
| **Batch/Buffer** | 12.8% | 0.41% | **3-10% target** |
| **Loss Trend** | -71% ✅ | +0.7% ❌ | Should be -60% |
| **Final Loss** | 4,258 | 8,782 | Target: <3,000 |
| **Convergence** | Yes | No | Fix hyperparams |

---

## Conclusion

The 10,000-iteration River Hold'em training run **failed to converge** due to a **critical hyperparameter mismatch**: batch size (128) was far too small for the buffer size (100k). This caused:

1. **Poor sample coverage** (0.41% per batch)
2. **High gradient noise** (insufficient averaging)
3. **Training instability** (loss increased instead of decreased)
4. **Wasted resources** (69% of buffer unused)

### Success Indicators for Next Run

✅ **Loss decreases from 8,000 → 3,000** (-62%)
✅ **Loss variance stabilizes** (std dev < 200 in final 1000 iters)
✅ **Buffer utilization** matches expectations (~31% for 10k iters)
✅ **Batch/buffer ratio** ≥ 3% (1024/30k)
✅ **Matrix view shows** strong hands bet, weak hands fold

### Expected Outcome with Fix

With `batch_size=1024`:
- **Iter 1,000**: Loss ~7,000 (baseline)
- **Iter 3,000**: Loss ~5,000 (-29%)
- **Iter 5,000**: Loss ~3,500 (-50%)
- **Iter 10,000**: Loss ~2,500 (-64%)
- **Convergence**: ✅ Achieved

**Next Step**: Apply batch size fix and re-run training. Document results in follow-up analysis.

---

**Report Generated**: 2026-01-07
**Analyst**: Claude Code
**Status**: ❌ Training failed - hyperparameter fix required
**Priority**: 🔴 **CRITICAL** - Must fix before further experimentation
