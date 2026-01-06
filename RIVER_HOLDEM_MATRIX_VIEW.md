# River Hold'em Matrix View Implementation

**Date**: 2026-01-06
**Feature**: Strategy visualization by hand strength category for River Hold'em

---

## Overview

The Matrix View for River Hold'em provides a grouped bar chart showing how the agent's strategy varies across different hand strength categories. This is a critical visualization for understanding whether the agent has learned fundamental poker concepts like "bet strong hands, fold weak hands."

---

## Visualization Design

### Hand Rank Categories (from treys Evaluator)

The visualization groups strategies by 10 hand rank categories based on the treys poker hand evaluator:

| Category | Name | treys Rank Range | Example |
|----------|------|------------------|---------|
| 0 | **High Card** | 6186-7462 | A♠ K♦ with board Q♣ J♥ 7♠ 3♦ 2♣ |
| 1 | **One Pair** | 3326-6185 | A♠ A♦ with board K♣ Q♥ J♠ 7♦ 2♣ |
| 2 | **Two Pair** | 2468-3325 | A♠ K♦ with board A♣ K♥ Q♠ 7♦ 2♣ |
| 3 | **Three of a Kind** | 1610-2467 | A♠ A♦ with board A♣ K♥ Q♠ 7♦ 2♣ |
| 4 | **Straight** | 1600-1609 | A♠ K♦ with board Q♣ J♥ T♠ 7♦ 2♣ |
| 5 | **Flush** | 323-1599 | A♠ K♠ with board Q♠ J♠ 7♠ 3♦ 2♣ |
| 6 | **Full House** | 167-322 | A♠ A♦ with board A♣ K♥ K♠ 7♦ 2♣ |
| 7 | **Four of a Kind** | 11-166 | A♠ A♦ with board A♣ A♥ K♠ 7♦ 2♣ |
| 8 | **Straight Flush** | 2-10 | A♠ K♠ with board Q♠ J♠ T♠ 7♦ 2♣ |
| 9 | **Royal Flush** | 1 | A♠ K♠ with board Q♠ J♠ T♠ 7♦ 2♣ |

**Note**: In treys, lower rank = better hand. The encoder inverts this for neural network input.

---

## Chart Components

### 1. Grouped Bars (4 actions per category)

Each hand rank category shows 4 bars representing action probabilities:

- **Red (Fold)**: Probability of folding
- **Teal (Check/Call)**: Probability of checking/calling
- **Light Green (Bet Pot)**: Probability of betting pot-sized
- **Yellow (All-In)**: Probability of going all-in

### 2. Sample Counts

Above each group, a gray label shows `n=XXX` indicating how many training samples fell into that category. Categories with low sample counts may have noisy/unreliable strategies.

### 3. Background Regions (Strategy Zones)

The chart highlights three strategic zones with colored backgrounds:

- **Red Background**: Weak hands (High Card, One Pair, Two Pair)
  - Expected: High fold probability, low bet/raise probability
- **Yellow Background**: Medium hands (Three of a Kind, Straight, Flush, Full House)
  - Expected: Balanced check/call, some betting
- **Green Background**: Strong hands (Four of a Kind, Straight Flush, Royal Flush)
  - Expected: High bet/all-in probability, low fold probability

---

## Implementation Details

### Data Processing Pipeline

```python
def _convert_strategy_to_matrix(strategy_dict: dict, game_name: str) -> dict:
    """Convert River Hold'em strategies to hand rank categories."""

    # For each information state:
    for info_state, strategy in strategy_dict.items():
        # 1. Parse info state: "hand|board|history|pot|stacks|bet"
        parts = info_state.split("|")
        hand_str = parts[0]  # e.g., "Ah2c"
        board_str = parts[1]  # e.g., "KhQhJh2s3d"

        # 2. Convert to treys card objects
        hand = [Card.new(hand_str[0:2]), Card.new(hand_str[2:4])]
        board = [Card.new(board_str[i:i+2]) for i in range(0, 10, 2)]

        # 3. Evaluate hand rank using treys
        rank = evaluator.evaluate(board, hand)

        # 4. Map rank to category (0-9)
        category = _map_rank_to_category(rank)

        # 5. Accumulate action probabilities for category
        category_data[category]["fold"].append(strategy[0])
        category_data[category]["check_call"].append(strategy[1])
        category_data[category]["bet_pot"].append(strategy[2])
        category_data[category]["all_in"].append(strategy[3])

    # 6. Average probabilities for each category
    category_avg = {
        cat: {
            "fold": np.mean(actions["fold"]),
            "check_call": np.mean(actions["check_call"]),
            "bet_pot": np.mean(actions["bet_pot"]),
            "all_in": np.mean(actions["all_in"]),
            "count": len(actions["fold"]),
        }
        for cat, actions in category_data.items() if actions["fold"]
    }
```

### Visualization Rendering

```python
def _draw_river_holdem_bars(self, matrix_data: dict):
    """Draw grouped bar chart with hand rank categories."""

    # Extract data
    category_indices = sorted(matrix.keys())
    category_labels = [categories[i] for i in category_indices]
    fold_probs = [matrix[i]["fold"] for i in category_indices]
    # ... (other actions)

    # Create grouped bars
    x = np.arange(len(category_labels))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, fold_probs, width, label='Fold', color='#ff6b6b')
    bars2 = ax.bar(x - 0.5*width, check_call_probs, width, label='Check/Call', color='#4ecdc4')
    bars3 = ax.bar(x + 0.5*width, bet_pot_probs, width, label='Bet Pot', color='#95e1d3')
    bars4 = ax.bar(x + 1.5*width, all_in_probs, width, label='All-In', color='#f9ca24')

    # Add background zones
    ax.axvspan(-0.5, weak_end + 0.5, alpha=0.1, color='red')  # Weak hands
    ax.axvspan(medium_start - 0.5, medium_end + 0.5, alpha=0.1, color='yellow')  # Medium
    ax.axvspan(strong_start - 0.5, len(category_indices) - 0.5, alpha=0.1, color='green')  # Strong
```

---

## Expected Behavior (GTO-like Strategy)

A well-trained agent should exhibit the following patterns:

### Weak Hands (0-2: High Card, One Pair, Two Pair)
- **High Fold %**: 60-80% when facing bets
- **Low Bet/All-In %**: 0-20%
- **Moderate Check/Call %**: 20-40% when getting good pot odds

### Medium Hands (3-6: Three of a Kind, Straight, Flush, Full House)
- **Balanced Check/Call %**: 40-60%
- **Moderate Bet Pot %**: 30-50%
- **Low Fold %**: 0-20%
- **Strategy depends on board texture and pot odds**

### Strong Hands (7-9: Four of a Kind, Straight Flush, Royal Flush)
- **High Bet/All-In %**: 70-90%
- **Very Low Fold %**: 0-5%
- **Strategy: Extract maximum value**

---

## Interpreting the Visualization

### Good Learning Indicators ✅

1. **Upward Trend**: Bet/All-In probability increases with hand strength
2. **Downward Trend**: Fold probability decreases with hand strength
3. **Clear Separation**: Strong hands show significantly different strategy than weak hands
4. **Balanced Sample Counts**: Sufficient samples (n > 50) across all categories

### Poor Learning Indicators ⚠️

1. **Flat Lines**: All hand categories using same strategy (not exploiting hand strength)
2. **Inverted Trends**: Folding strong hands, betting weak hands
3. **Extreme Imbalance**: Some categories have n < 10 samples
4. **High Variance**: Strategies oscillate wildly between evaluations

---

## Example Interpretation

```
Chart Shows:
- High Card: Fold=0.75, Check/Call=0.20, Bet=0.03, All-In=0.02
- One Pair: Fold=0.60, Check/Call=0.30, Bet=0.08, All-In=0.02
- Two Pair: Fold=0.35, Check/Call=0.45, Bet=0.15, All-In=0.05
- Three of a Kind: Fold=0.10, Check/Call=0.40, Bet=0.35, All-In=0.15
- Flush: Fold=0.05, Check/Call=0.25, Bet=0.50, All-In=0.20
- Four of a Kind: Fold=0.01, Check/Call=0.09, Bet=0.40, All-In=0.50

Analysis:
✅ Agent has learned basic hand strength ranking
✅ Weak hands mostly fold (75% for high card)
✅ Strong hands mostly bet/raise (90% for four of a kind)
✅ Medium hands show balanced play (flush: 70% bet/all-in)
✅ Clear upward trend in aggression with hand strength

Conclusion: Agent exhibits fundamental poker understanding!
```

---

## Technical Considerations

### Performance

- **Parsing**: O(n) where n = number of information states
- **Evaluation**: Uses cached treys evaluator (fast hand ranking)
- **Memory**: Stores only averaged probabilities (10 categories × 4 actions × 2 floats = 320 bytes)

### Edge Cases

1. **No samples for category**: Category not shown in chart (e.g., Royal Flush rarely appears)
2. **Invalid card strings**: Skipped with try/except (logs error)
3. **Incomplete strategies**: Requires len(strategy) >= 4, otherwise skipped

### Limitations

1. **No position awareness**: Aggregates across all betting histories
2. **No pot odds context**: Doesn't separate "facing bet" vs "first to act"
3. **No board texture**: Flush on wet board vs dry board treated same
4. **Sample bias**: Categories with more samples have more stable estimates

---

## Future Enhancements

### Planned Improvements

1. **Stratify by position**: Show separate charts for "First to Act" vs "Facing Bet"
2. **Pot odds breakdown**: Group by pot odds ranges (< 0.3, 0.3-0.5, > 0.5)
3. **Confidence intervals**: Show error bars based on sample variance
4. **Board texture**: Separate flush draws, paired boards, connected boards
5. **Clickable bars**: Click a category to show example hands in that range

### Advanced Analytics

1. **Exploitability**: Compute local exploitability for each category
2. **Comparison to GTO**: Overlay theoretical optimal strategy
3. **Evolution over time**: Animation showing strategy changes across iterations
4. **Opponent modeling**: Compare strategy vs different baseline bots

---

## Code Locations

### Files Modified

- `src/aion26/gui/app.py`:
  - `_convert_strategy_to_matrix()`: Lines 85-204 (added river_holdem case)
  - `_update_strategy_matrix()`: Line 897 (added river_holdem dispatch)
  - `_draw_river_holdem_bars()`: Lines 1054-1130 (new function)

### Dependencies

- `treys`: Card parsing and hand evaluation
- `numpy`: Probability averaging
- `matplotlib`: Bar chart rendering

---

## Testing

### Manual Testing Checklist

- [ ] Select "river_holdem" from game dropdown
- [ ] Start training (use 100k buffer, 10k iterations)
- [ ] Wait for first evaluation (iter 100)
- [ ] Click "Matrix View" tab
- [ ] Verify grouped bars appear
- [ ] Verify sample counts shown (n=XXX)
- [ ] Verify background zones colored correctly
- [ ] Verify legend shows 4 actions
- [ ] Verify axes labeled correctly
- [ ] Continue training to iter 1000+
- [ ] Verify strategy evolves (bars change height)

### Expected Timeline

- **Iter 100**: Noisy, mostly uniform probabilities
- **Iter 500**: Clear trends emerging (weak hands fold more)
- **Iter 2000**: Strong differentiation by hand rank
- **Iter 10000**: Converged strategy (smooth curves)

---

## Comparison to Other Games

| Game | Matrix View | Categories | Visualization |
|------|-------------|------------|---------------|
| **Kuhn** | Betting tree | 3 cards (J, Q, K) | Tree diagram |
| **Leduc** | 3×3 grid | Private × Board card | Pie charts in cells |
| **River Hold'em** | Grouped bars | 10 hand ranks | Bar chart with zones |

---

## Conclusion

The River Hold'em Matrix View provides an intuitive way to verify that the Deep PDCFR+ agent is learning fundamental poker concepts. By grouping strategies by hand strength and visualizing action distributions, users can quickly assess whether the agent has learned to:

1. **Bet strong hands** (high bet/all-in % for categories 7-9)
2. **Fold weak hands** (high fold % for categories 0-2)
3. **Play medium hands carefully** (balanced check/call for categories 3-6)

This visualization is critical for debugging training issues and validating that the agent is converging to a reasonable poker strategy before running expensive head-to-head evaluations.

---

**Status**: ✅ Implemented and ready for testing
**Next**: Run full 10k iteration training and validate strategy evolution
