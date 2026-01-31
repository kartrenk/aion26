# Bug Analysis: Losing to CallingStation (-4425 mbb/h)

## Executive Summary

The agent loses **-4425 mbb/h** against a CallingStation, which is a severe bug. A CallingStation never folds and never raises - they only call. Against this opponent:

- **Optimal strategy**: Value bet strong hands, check weak hands, never bluff
- **Expected win rate**: +500 to +2000 mbb/h
- **Actual win rate**: -4425 mbb/h (losing ~4.4 BB per hand!)

This indicates a fundamental flaw in either the game logic, reward calculation, or learned strategy.

---

## Possible Error Sources

### 1. Reward/Return Sign Inversion
**Likelihood: HIGH**

The `game.returns()` function might return rewards from the wrong player's perspective, or the sign might be inverted somewhere in the pipeline.

**Symptoms:**
- Agent learns to minimize its own winnings
- Strong hands fold, weak hands bet
- Winning against RandomBot could be accidental

**Code to check:**
```
src/aion26_rust/src/river_holdem.rs  - returns() implementation
scripts/train_webapp_pro.py:528     - results.append(returns[0])
```

---

### 2. Showdown Winner Determination Bug
**Likelihood: HIGH**

At showdown (when CallingStation calls), the winner might be determined incorrectly. The hand evaluator might:
- Compare hands incorrectly (lower rank wins instead of higher)
- Use wrong cards for evaluation
- Have inverted comparison logic

**Symptoms:**
- Losing at showdown with winning hands
- Consistent losses when hands go to showdown

**Code to check:**
```
src/aion26_rust/src/evaluator.rs     - Hand comparison logic
src/aion26_rust/src/river_holdem.rs  - Showdown resolution
```

---

### 3. Pot Award Direction Bug
**Likelihood: MEDIUM-HIGH**

When the pot is awarded, it might go to the wrong player:
- Player indices swapped
- Pot subtracted instead of added
- Wrong player marked as winner

**Symptoms:**
- Returns sum to zero (zero-sum preserved) but assigned to wrong player

**Code to check:**
```
src/aion26_rust/src/river_holdem.rs  - Terminal state payoff calculation
```

---

### 4. CallingStation Implementation Bug
**Likelihood: MEDIUM**

The CallingStation bot might not behave as expected:
- Might be raising instead of calling
- Might be folding sometimes
- Action selection could be wrong

**Symptoms:**
- CallingStation taking unexpected actions

**Code to check:**
```
src/aion26/baselines.py              - CallingStationBot.get_action()
```

---

### 5. Action Index Mapping Error
**Likelihood: MEDIUM**

Actions might be mapped incorrectly between Python and Rust:
- Fold=0, Call=1, Raise=2, AllIn=3 might not match
- Legal action masking could be wrong

**Symptoms:**
- Agent thinks it's calling but actually folding
- Agent thinks it's value betting but actually checking

**Code to check:**
```
scripts/train_webapp_pro.py:519-520  - Action selection in eval
src/aion26_rust/src/river_holdem.rs  - Action enum definition
```

---

### 6. Evaluation Player Perspective Error
**Likelihood: MEDIUM**

During baseline evaluation, the agent plays as Player 0. If returns are calculated from Player 1's perspective, the results would be inverted.

**Symptoms:**
- Winning/losing patterns exactly inverted

**Code to check:**
```
scripts/train_webapp_pro.py:528      - returns[0] vs returns[1]
scripts/train_webapp_pro.py:541      - Which player is the agent?
```

---

### 7. Strategy Learned Backwards (CFR Bug)
**Likelihood: MEDIUM**

The CFR regret calculation might have sign errors:
- Regret = value(action) - value(strategy) might be inverted
- Network learns to maximize regret instead of minimize

**Symptoms:**
- Network outputs negative values for good actions
- Strategy Inspector shows high fold % with strong hands

**Code to check:**
```
src/aion26_rust/src/parallel_trainer.rs  - Regret calculation
scripts/train_webapp_pro.py              - Network training targets
```

---

### 8. Hand Strength Encoding Still Wrong
**Likelihood: LOW-MEDIUM**

Despite the previous fix, the hand strength encoding might still be misaligned between training (Rust) and evaluation (Python).

**Symptoms:**
- Network can't distinguish hand strengths
- Random-looking strategy regardless of cards

**Code to check:**
```
scripts/train_webapp_pro.py:219-227   - Python encode_state()
src/aion26_rust/src/parallel_trainer.rs - Rust encode_state()
```

---

## Diagnostic Tests

### Test 1: Verify Returns Sign
```python
# In evaluate_vs_baseline, add logging:
returns = game.returns()
print(f"Returns: P0={returns[0]:+.2f}, P1={returns[1]:+.2f}, sum={sum(returns):.4f}")
```

### Test 2: Verify Showdown Winner
```python
# At terminal state, log hand strengths and winner:
if game.is_terminal():
    p0_strength = game.get_hand_strength(0)
    p1_strength = game.get_hand_strength(1)
    returns = game.returns()
    winner = 0 if returns[0] > 0 else 1
    print(f"P0 hand: {p0_strength}, P1 hand: {p1_strength}, Winner: P{winner}")
```

### Test 3: Verify CallingStation Behavior
```python
# Log CallingStation actions:
action = baseline.get_action(game)
print(f"CallingStation chose action {action} from legal {game.legal_actions()}")
```

### Test 4: Manual Hand Verification
```python
# Play specific known hand and verify outcome:
# P0 has nuts, P1 has nothing, P0 bets, P1 calls -> P0 should win
```

---

## Code References

| File | Line | Function | What to Check |
|------|------|----------|---------------|
| `scripts/train_webapp_pro.py` | 528 | `evaluate_vs_baseline` | `returns[0]` - correct player? |
| `scripts/train_webapp_pro.py` | 534 | `evaluate_vs_baseline` | mbb calculation |
| `src/aion26/baselines.py` | ~20 | `CallingStationBot.get_action` | Always returns call action? |
| `src/aion26_rust/src/river_holdem.rs` | - | `returns()` | Sign of returns |
| `src/aion26_rust/src/river_holdem.rs` | - | Terminal payoff | Pot goes to right player? |
| `src/aion26_rust/src/evaluator.rs` | - | `compare_hands` | Lower rank = better? |

---

## ROOT CAUSE IDENTIFIED (via debug_calling_station.py)

### Diagnostic Results (20 hands):
- Total return: **-96 BB = -4800 mbb/h** (matches baseline eval!)
- Agent action distribution: 85% Check, 15% AllIn, 0% Fold/Raise

### Problem 1: Catastrophic ALL-IN with weak hands
```
Hand #2:  Agent=Pair, Station=TwoPair → Agent ALL-IN → -101 BB
Hand #13: Agent=Pair, Station=Trips  → Agent ALL-IN → -101 BB
```
The network has ~20% probability of going all-in even with weak hands.
Against CallingStation who NEVER folds, this is -EV every time.

### Problem 2: NOT value betting with strong hands
```
Hand #3:  Agent=Trips vs Pair → Agent CHECKS → wins only +1 BB
Hand #8:  Agent=Trips vs Pair → Agent CHECKS → wins only +1 BB
Hand #14: Agent=Trips vs Pair → Agent CHECKS → wins only +1 BB
```
With Trips vs CallingStation, agent should bet big (they'll call with worse).
Instead it checks, leaving ~100 BB on the table each hand.

### Why This Happens

The network learned from **CFR self-play** where:
- Opponents sometimes fold to aggression → bluffing has +EV
- Opponents sometimes have monsters → caution with value hands

Against CallingStation:
- Bluffs ALWAYS lose (they never fold) → bluffing is -EV
- Value bets ALWAYS get called → aggressive value betting is +EV

**The Nash equilibrium strategy is suboptimal against exploitable opponents.**

### Is This Actually a Bug?

**Yes, partially.** The extreme losses suggest the network hasn't converged:
- A true Nash strategy shouldn't lose -4400 mbb/h to ANY opponent
- The 20% all-in probability with weak hands is too high
- The lack of value betting with strong hands is suboptimal even for Nash

**The game logic is CORRECT** (verified via debug script):
- Hand evaluation works correctly
- Pot awarded to correct winner at showdown
- CallingStation behaves correctly

**The network strategy is the problem:**
- Not enough training iterations
- Or regret calculation has issues

---

## Recommendations

1. **Train longer** - Current model may not have converged
2. **Check regret signs** in `parallel_trainer.rs` - ensure regrets have correct direction
3. **Add value betting bias** - Network should learn to bet strong hands more
4. **Monitor convergence** - Track exploitability metric during training

---

## Verified Working Components

- ✅ Hand evaluation (lower rank = better)
- ✅ Pot distribution at showdown
- ✅ CallingStation always calls
- ✅ Returns[0] is agent's return
- ✅ mbb/h calculation

