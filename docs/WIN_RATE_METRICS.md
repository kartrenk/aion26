# Win Rate Metrics in Poker AI Evaluation

## Abstract

This document provides a formal definition of the win rate metrics used to evaluate poker AI agents in the Aion-26 framework. We define milli-big-blinds per hand (mbb/h) and big blinds per 100 hands (bb/100), establish their mathematical relationship, and provide interpretation guidelines based on established poker AI research.

---

## 1. Introduction

Evaluating the performance of poker AI agents requires standardized metrics that account for the stochastic nature of the game and allow comparison across different stake levels. The poker research community has converged on **big blind-normalized win rates** as the standard measure of agent strength.

Two equivalent representations are commonly used:

| Metric | Full Name | Units |
|--------|-----------|-------|
| **mbb/h** | Milli-big-blinds per hand | 1/1000 BB per hand |
| **bb/100** | Big blinds per 100 hands | BB per 100 hands |

---

## 2. Formal Definitions

### 2.1 Big Blind Normalization

Let $W_i$ denote the profit (or loss) in chips for hand $i$, and let $BB$ denote the size of the big blind. The **normalized profit** for hand $i$ is:

$$w_i = \frac{W_i}{BB}$$

This normalization allows comparison across games with different stake levels.

### 2.2 Milli-Big-Blinds per Hand (mbb/h)

Given $N$ hands played, the win rate in mbb/h is defined as:

$$\text{mbb/h} = \frac{1000}{N} \sum_{i=1}^{N} w_i = 1000 \cdot \bar{w}$$

where $\bar{w}$ is the mean normalized profit per hand.

The factor of 1000 converts big blinds to milli-big-blinds, providing finer granularity for small edges.

### 2.3 Big Blinds per 100 Hands (bb/100)

The equivalent metric in bb/100 is:

$$\text{bb/100} = \frac{100}{N} \sum_{i=1}^{N} w_i = 100 \cdot \bar{w}$$

### 2.4 Conversion Formula

The two metrics are related by a factor of 10:

$$\text{bb/100} = \frac{\text{mbb/h}}{10}$$

**Example:** A win rate of +150 mbb/h equals +15.0 bb/100.

---

## 3. Interpretation Guidelines

### 3.1 Performance Benchmarks

Based on established poker AI literature (Brown & Sandholm, 2019; Moravčík et al., 2017), we provide the following interpretation guidelines for heads-up No-Limit Texas Hold'em:

| Win Rate (mbb/h) | Win Rate (bb/100) | Interpretation |
|------------------|-------------------|----------------|
| < -100 | < -10.0 | Severely exploitable strategy |
| -100 to -20 | -10.0 to -2.0 | Weak strategy with significant leaks |
| -20 to +20 | -2.0 to +2.0 | Near break-even / within variance |
| +20 to +100 | +2.0 to +10.0 | Moderate edge |
| +100 to +500 | +10.0 to +50.0 | Strong edge against weak opponents |
| +500 to +2000 | +50.0 to +200.0 | Dominant against non-adaptive bots |
| > +2000 | > +200.0 | Exploitation of trivial strategies |

### 3.2 Opponent-Relative Interpretation

Win rates must be interpreted relative to opponent strength:

- **vs. Random Bot**: Expected win rate for competent agent: +1000 to +5000 mbb/h
- **vs. Calling Station**: Expected win rate: +500 to +2000 mbb/h
- **vs. Always Fold**: Expected: ~+500 mbb/h (winning blinds each hand)
- **vs. Nash Equilibrium**: Expected: ~0 mbb/h (by definition)
- **vs. Human Professional**: State-of-the-art AI: +10 to +100 mbb/h

### 3.3 Context: River-Only vs Full Game

The Aion-26 framework currently implements **river-only** Hold'em, which has different dynamics than the full game:

- Reduced decision points (single betting round)
- Simpler information structure
- Faster convergence of CFR algorithms
- Higher expected win rates against weak opponents

Win rates in river-only games are **not directly comparable** to full-game statistics.

---

## 4. Statistical Considerations

### 4.1 Variance and Sample Size

Poker is a high-variance game. The standard deviation of single-hand outcomes typically ranges from 50-150 BB depending on game format. This implies:

$$\sigma_{\bar{w}} = \frac{\sigma_w}{\sqrt{N}}$$

For a typical $\sigma_w \approx 100$ mbb:

| Sample Size (N) | Standard Error (mbb/h) | 95% CI Width |
|-----------------|------------------------|--------------|
| 1,000 | ±3.16 | ±6.2 |
| 10,000 | ±1.00 | ±2.0 |
| 100,000 | ±0.32 | ±0.6 |
| 1,000,000 | ±0.10 | ±0.2 |

**Recommendation:** Minimum 10,000 hands for preliminary evaluation; 100,000+ hands for statistically significant comparisons.

### 4.2 Confidence Intervals

The 95% confidence interval for win rate is approximately:

$$\text{mbb/h} \pm 1.96 \cdot \frac{\sigma_w}{\sqrt{N}}$$

where $\sigma_w$ is the sample standard deviation of per-hand results.

### 4.3 Statistical Significance Testing

To determine if Agent A is significantly better than Agent B, compute:

$$z = \frac{\bar{w}_A - \bar{w}_B}{\sqrt{\frac{\sigma_A^2}{N_A} + \frac{\sigma_B^2}{N_B}}}$$

Reject the null hypothesis (equal strength) if $|z| > 1.96$ for $\alpha = 0.05$.

---

## 5. Calculation in Aion-26

### 5.1 Implementation

In the Aion-26 framework, win rate is calculated as:

```python
# Per-hand profit is in big blinds (pot normalized to 2 BB)
results = [game.returns()[0] for game in completed_games]

# Convert to mbb/h
mbb_per_hand = (np.mean(results) / 2.0) * 1000

# Convert to bb/100
bb_per_100 = mbb_per_hand / 10
```

The division by 2.0 accounts for the pot being initialized with 2 BB (1 BB from each player as blinds).

### 5.2 Baseline Evaluation Protocol

The evaluation suite tests against three baseline opponents:

1. **RandomBot**: Uniform random action selection
2. **CallingStation**: Always calls (never folds, never raises)
3. **AlwaysFold**: Always folds when legal, otherwise checks

Expected win rates against these baselines provide diagnostic information about strategy quality.

---

## 6. References

1. Brown, N., & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885-890.

2. Moravčík, M., et al. (2017). DeepStack: Expert-level artificial intelligence in heads-up no-limit poker. *Science*, 356(6337), 508-513.

3. Zinkevich, M., et al. (2007). Regret minimization in games with incomplete information. *Advances in Neural Information Processing Systems*, 20.

4. Lanctot, M., et al. (2009). Monte Carlo sampling for regret minimization in extensive games. *Advances in Neural Information Processing Systems*, 22.

---

## Appendix A: Quick Reference

| To Convert | Formula |
|------------|---------|
| mbb/h → bb/100 | Divide by 10 |
| bb/100 → mbb/h | Multiply by 10 |
| mbb/h → BB/hand | Divide by 1000 |
| bb/100 → BB/hand | Divide by 100 |

| Win Rate | Meaning |
|----------|---------|
| +100 mbb/h | Winning 0.1 BB per hand on average |
| +10 bb/100 | Winning 10 BB per 100 hands |
| +1000 mbb/h | Winning 1 BB per hand (very high) |

---

*Document Version: 1.0*
*Aion-26 Deep PDCFR+ Framework*
