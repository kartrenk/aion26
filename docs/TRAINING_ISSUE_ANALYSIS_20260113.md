# Training Issue Analysis - January 13, 2026

## Executive Summary

**CRITICAL BUG FOUND**: The training is running in **RIVER MODE** (`mode: river`), not **FULL HUNL MODE**. All aggression forcing code is gated behind `if GAME_MODE == "full"` and is therefore **NOT EXECUTING**.

---

## Issue #1: WRONG GAME MODE (ROOT CAUSE)

### Evidence
```
Evaluating vs RandomBot (100000 hands) [mode: river]
                                        ^^^^^^^^^^^^
shape: (5000000, 4)   <- Only 4 actions, not 8!
              ^^^
```

### Impact
The following features are **DISABLED** because they check `GAME_MODE == "full"`:
1. **Forced Aggression** - `forced_agg=0.0%` every epoch
2. **Hand Strength Penalty** - Chen score not applied
3. **Entropy Regularization** - Not applied
4. **Preflop Features** - Not encoded

### Fix Required
```bash
# Start webapp in FULL mode, not river mode
python scripts/train_webapp_pro.py serve --game full
```

---

## Issue #2: Forced Aggression Not Working

### Evidence
```
forced_agg=0.0%   <- Every single epoch shows 0%
```

### Root Cause
The aggression forcing code is wrapped in:
```python
if GAME_MODE == "full" and config.aggression_force > 0:
    # This never runs because GAME_MODE == "river"
```

### Secondary Issue
Even if running in Full mode, the aggression forcing only modifies **predictions** sent to Rust, not the actual action selection. The Rust engine may still be selecting passive actions based on its internal logic.

---

## Issue #3: Model Converged to Passive Local Minimum

### Evidence
```
Aggression: 0.0%  <- Literally never bets
Win rate:   +252 -> +63 mbb/h (declining)
Loss:       0.072 -> 0.070 (barely improving)
```

### Regret Stats Stagnation
| Epoch | Mean    | Std   | Positive% |
|-------|---------|-------|-----------|
| 81    | -0.0330 | 0.267 | 34.8%     |
| 90    | -0.0330 | 0.267 | 34.8%     |
| 100   | -0.0330 | 0.267 | 34.8%     |

**The regret distribution is FROZEN** - the model stopped learning around epoch 80.

---

## Issue #4: Win Rate Declining Despite Training

### Win Rate Progression (vs RandomBot)
| Epoch | Win Rate (mbb/h) |
|-------|------------------|
| 81    | +252             |
| 84    | +123             |
| 90    | +117             |
| 94    | +80              |
| 96    | +86              |
| 98    | +63              |
| 100   | +151             |

**Trend**: Win rate collapsed from +252 to +63 over 20 epochs. The model is getting WORSE at exploiting RandomBot.

---

## Issue #5: AWR Loss Not Breaking Passivity

### Evidence
```
AWR weight: 5.0x for positive regrets
positive_regrets: 34.8%  <- Unchanged
```

The AWR loss is applied, but:
1. It's not being applied in river mode (no hand penalty, no entropy)
2. The model has already converged to a passive strategy
3. Historical data contains 80+ epochs of passive behavior that drowns out any new exploration

---

## Recommendations

### Immediate Actions

1. **Switch to Full HUNL Mode**
   ```bash
   python scripts/train_webapp_pro.py serve --game full
   ```

2. **Clear Historical Data**
   Delete old epoch files to prevent passive history from dominating:
   ```bash
   rm -rf /tmp/vr_dcfr_pro/epoch_*.bin
   rm -rf /tmp/full_hunl_dcfr/epoch_*.bin
   ```

3. **Increase Aggression Force**
   Consider increasing from 0.25 to 0.50 for first 20 epochs.

### Code Fixes Required

1. **Add River Mode Aggression Forcing**
   The aggression code only works in Full mode. Add support for River mode:
   ```python
   # River mode has 4 actions: 0=Fold, 1=Check/Call, 2=Bet, 3=All-In
   if GAME_MODE == "river" and config.aggression_force > 0:
       force_mask = np.random.random(batch_size) < config.aggression_force
       preds_np[force_mask, 2:4] += config.aggression_boost  # Boost Bet and All-In
   ```

2. **Apply AWR/Entropy/Hand Penalty to River Mode**
   Remove the `if GAME_MODE == "full"` gates or add equivalent logic for river mode.

3. **Reduce Historical Mixing Weight**
   Current `history_alpha=0.5` means old passive data heavily influences training:
   ```python
   history_alpha: float = 0.3  # Reduce to favor recent data
   ```

---

## Log Anomalies Summary

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Game Mode | full | river | BUG |
| Forced Aggression | ~25% | 0.0% | BUG |
| Evaluation Aggression | >5% | 0.0% | CRITICAL |
| Win Rate Trend | Increasing | Decreasing | PROBLEM |
| Regret Distribution | Evolving | Frozen | PROBLEM |
| Loss Trend | Decreasing | Stagnant (0.07) | PROBLEM |

---

## Conclusion

The training run was fundamentally misconfigured. It ran in **River mode** while all the anti-passivity features (aggression forcing, entropy bonus, hand penalty) are gated for **Full HUNL mode only**.

The bot learned to check 100% of the time because:
1. No forced exploration was applied
2. 80+ epochs of historical "check always" data dominate training
3. AWR loss alone cannot break a deeply entrenched passive strategy

**Action Required**: Restart training in Full mode with cleared historical data.

---

*Analysis generated: 2026-01-13 20:10*
