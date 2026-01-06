# GUI Fixes for River Hold'em Training

**Date**: 2026-01-06
**Context**: Fixes based on log analysis from `LOG_ANALYSIS_RIVER_TRAINING.md`

---

## Issues Fixed

### 1. Final NashConv Computation for River Hold'em ✅

**Problem**: Final NashConv was being computed even for River Hold'em, which is computationally infeasible for 52-card games.

**Location**: `src/aion26/gui/model.py` lines 237-240

**Fix**:
```python
# Before:
logger.info("Training completed, computing final NashConv...")
final_strategy = self._get_average_strategy()
final_nash_conv = compute_nash_conv(self.trainer.initial_state, final_strategy)

# After:
logger.info("Training completed")
final_strategy = self._get_average_strategy()

# Only compute final NashConv for small games (Kuhn, Leduc)
if self.config.game.name in ["kuhn", "leduc"]:
    logger.info("Computing final NashConv...")
    final_nash_conv = compute_nash_conv(self.trainer.initial_state, final_strategy)
    logger.info(f"Final NashConv: {final_nash_conv:.6f}")
else:
    logger.info(f"Skipping final NashConv for {self.config.game.name} (use head-to-head evaluation instead)")
    final_nash_conv = None
```

**Impact**:
- River Hold'em training no longer attempts infeasible NashConv computation at the end
- Consistent with periodic NashConv skipping during training (already implemented)
- Final log message now correctly indicates NashConv is skipped

---

### 2. Game-Specific Buffer Sizes in GUI ✅

**Problem**: GUI was using default buffer capacity of 1000 for all games, but River Hold'em requires 100,000 samples due to large state space.

**Location**: `src/aion26/gui/app.py`

**Changes**:

1. **Added game selection callback** (line 268):
```python
game_combo.bind("<<ComboboxSelected>>", self._on_game_changed)
```

2. **Implemented `_on_game_changed` method** (lines 475-494):
```python
def _on_game_changed(self, event):
    """Update recommended settings when game selection changes."""
    game = self.game_var.get()

    # Update buffer capacity and iterations based on game
    if game == "river_holdem":
        # River Hold'em needs large buffer due to 52-card state space
        self.buffer_capacity_var.set("100000")
        self.iterations_var.set("10000")
        self.batch_size_var.set("1024")
    elif game == "leduc":
        # Leduc poker - medium complexity
        self.buffer_capacity_var.set("10000")
        self.iterations_var.set("5000")
        self.batch_size_var.set("256")
    else:  # kuhn
        # Kuhn poker - simplest game
        self.buffer_capacity_var.set("1000")
        self.iterations_var.set("2000")
        self.batch_size_var.set("128")
```

3. **Updated default values** to match Leduc (default game):
   - Buffer capacity: `"1000"` → `"10000"` (line 310)
   - Iterations: `"2000"` → `"5000"` (line 296)
   - Batch size: `"128"` → `"256"` (line 303)

**Impact**:
- When user selects "river_holdem" from dropdown, buffer automatically increases to 100k
- Training parameters automatically adjust to game-specific recommended values
- Prevents underfitting due to insufficient buffer capacity
- User can still manually override values if needed

---

## Recommended Settings by Game

| Game | Buffer Capacity | Iterations | Batch Size | Rationale |
|------|----------------|------------|------------|-----------|
| **Kuhn** | 1,000 | 2,000 | 128 | 12 information sets, small state space |
| **Leduc** | 10,000 | 5,000 | 256 | ~144 information sets, medium complexity |
| **River Hold'em** | 100,000 | 10,000 | 1,024 | 52-card deck, large state space |

---

## Verification

### Syntax Verification ✅
```bash
$ PYTHONPATH=src python3 -m py_compile src/aion26/gui/model.py src/aion26/gui/app.py
✓ GUI files compile successfully
```

### Test Results ✅
- 130/131 tests passing
- 1 pre-existing test failure (unrelated to GUI changes)
- Test failure: `test_pdcfr_plus_schedulers_initialized` expects LinearScheduler but gets DDCFRStrategyScheduler

---

## Expected Behavior

### Before Fixes:
```
[GUI] User selects "river_holdem"
→ Buffer capacity: 1,000 (too small!)
→ Training runs, buffer fills at iter ~420
→ Limited sample diversity, higher variance
→ Final NashConv: 167.53 (meaningless, shouldn't compute)
```

### After Fixes:
```
[GUI] User selects "river_holdem"
→ Buffer capacity auto-updates to 100,000
→ Iterations auto-updates to 10,000
→ Batch size auto-updates to 1,024
→ Training runs with appropriate buffer size
→ Final NashConv skipped (correct!)
→ Log: "Skipping final NashConv for river_holdem (use head-to-head evaluation instead)"
```

---

## Related Files

- `src/aion26/gui/model.py` - Training backend (NashConv fix)
- `src/aion26/gui/app.py` - GUI frontend (buffer size fix)
- `LOG_ANALYSIS_RIVER_TRAINING.md` - Original analysis
- `RIVER_TRAINING_INFRASTRUCTURE.md` - River Hold'em docs
- `src/aion26/config.py` - Contains `river_holdem_config()` with correct values

---

## Future Improvements

1. **Add buffer fill indicator** to GUI status bar
2. **Display head-to-head win rates** for River Hold'em instead of NashConv
3. **Add tooltips** explaining why each game needs different buffer sizes
4. **Validate user inputs** (warn if buffer too small for selected game)

---

**Commit Message**:
```
Fix GUI config for River Hold'em training

- Skip final NashConv computation for river_holdem (infeasible for 52-card games)
- Auto-set game-specific buffer sizes when game dropdown changes
- River Hold'em: 100k buffer, 10k iters, 1024 batch
- Leduc: 10k buffer, 5k iters, 256 batch
- Kuhn: 1k buffer, 2k iters, 128 batch

Fixes issues identified in LOG_ANALYSIS_RIVER_TRAINING.md
```
