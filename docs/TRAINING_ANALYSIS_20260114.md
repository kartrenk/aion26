# Training Analysis - January 14, 2026 Run

## Executive Summary

The training run completed 100 epochs in river mode. Analysis reveals:

1. **The regret distribution is actually BALANCED** when accounting for action legality
2. **The 0% aggression is NOT caused by biased regrets** - it's caused by other issues
3. **The forced aggression code did NOT run** (forced_agg=0.0% in logs)

---

## Key Finding: Regrets Are Balanced

### Raw vs Conditional Positive Rates

| Action | Raw Positive % | Legal States | **Conditional Positive %** |
|--------|---------------|--------------|---------------------------|
| Fold | 38.2% | 98.5% | 38.8% |
| Check/Call | 49.2% | 96.2% | **51.1%** |
| Bet | 28.7% | 59.2% | **48.4%** |
| All-In | 14.3% | 28.7% | **49.7%** |

**Interpretation**: When Bet/All-In ARE legal, they have positive regret ~50% of the time - perfectly balanced! The low raw percentages come from these actions being **illegal** in many states (zeros in the regret targets).

---

## Why The Model Shows 0% Aggression

### Issue #1: Forced Aggression Never Ran
```
forced_agg=0.0%   <- Every epoch
```
The code fix was made but the webapp was running old code, or there's a bug preventing the aggression boost from taking effect.

### Issue #2: Model Learning From Raw Zeros
The network sees:
- Bet column: 40.8% zeros
- All-In column: 71.3% zeros

Without understanding these mean "illegal" (not "bad"), the network learns:
- "Bet often has zero regret → Bet is rarely good"
- "All-In often has zero regret → All-In is rarely good"

### Issue #3: Softmax Favors Check/Call
With temperature-scaled softmax on advantages:
- Check/Call has the highest mean advantage (+0.017 vs others negative)
- Softmax concentrates probability on highest value
- Model always picks Check/Call

---

## Epoch-over-Epoch Analysis

### Sample Counts (Stable)
| Epoch | Samples | File Size |
|-------|---------|-----------|
| 0 | 426,797 | 239 MB |
| 50 | 422,887 | 237 MB |
| 99 | 424,596 | 238 MB |

### "Best Action" Distribution (Converging to Passivity)
| Epoch | Fold | Check/Call | Bet | All-In |
|-------|------|------------|-----|--------|
| 0 | 36.8% | 26.0% | 15.6% | **21.6%** |
| 50 | 29.1% | **31.9%** | 24.6% | 14.4% |
| 99 | 29.2% | **32.3%** | 24.4% | 14.2% |

The model is shifting from "Fold-heavy" (epoch 0) to "Check-heavy" (epoch 99).
All-In preference dropped from 21.6% → 14.2%.

---

## Regret Statistics (Epoch 99)

```
Overall:
  Mean:  -0.033
  Std:   0.273
  Min:   -1.44
  Max:   +1.19

Per-Action (when legal):
  Fold:       mean=-0.066, positive 38.8%
  Check/Call: mean=-0.017, positive 51.1%  ← Highest
  Bet:        mean=-0.034, positive 48.4%
  All-In:     mean=-0.013, positive 49.7%
```

**Note**: Check/Call has the least negative mean advantage, which is why the network prefers it.

---

## Root Cause Analysis

### The Zero-Regret Problem
The network training uses MSE loss on raw regret targets. When All-In is illegal:
- Target = 0.0
- Network predicts ≈ 0.0
- Loss = 0 (correct!)

But this teaches the network: "All-In advantage ≈ 0 most of the time"

When the network then plays poker:
```python
advantages = network(state)  # Returns [Fold: -0.1, Check: +0.1, Bet: 0.0, All-In: 0.0]
strategy = softmax(advantages)  # Check gets highest probability
action = sample(strategy)  # Always picks Check
```

### The Fix
**Don't use zeros for illegal actions.** Options:
1. Mask illegal actions in loss computation
2. Use -∞ (or large negative) for illegal actions
3. Output a legal action mask alongside regrets

---

## Recommendations

### Immediate
1. **Verify aggression forcing is working**
   - The logs show `forced_agg=0.0%` which means the code isn't running
   - Check that the webapp was restarted after code changes

2. **Clear historical data**
   ```bash
   rm /tmp/vr_dcfr_pro/epoch_*.bin
   ```

### Code Changes Needed
1. **Handle illegal actions properly**
   ```python
   # In training, mask out illegal actions
   legal_mask = (batch_targets != 0).float()
   loss = ((predictions - batch_targets) ** 2 * legal_mask).sum() / legal_mask.sum()
   ```

2. **Use negative infinity for illegal actions during inference**
   ```python
   # When evaluating
   advantages[illegal_actions] = -1e9  # Force softmax to ignore
   ```

---

## Conclusion

The solver is producing **correct, balanced regrets**. The passivity problem stems from:

1. The network learning that Bet/All-In often have zero regret (because they're illegal)
2. The forced aggression code not executing
3. Historical data from 100 epochs of passive play

**The regret data is fine. The training pipeline needs to handle illegal actions properly.**

---

*Analysis generated: 2026-01-15*
