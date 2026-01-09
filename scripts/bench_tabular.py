#!/usr/bin/env python3
"""Benchmark Tabular CFR vs Baseline Bots.

This script solves River Hold'em exactly using tabular vanilla CFR in Rust,
then evaluates the Nash equilibrium strategy against baseline bots to establish
the theoretical performance ceiling.

Expected outcomes:
- vs RandomBot: +3,000 to +6,000 mbb/h (exploiting random play)
- vs CallingStation: +1,000 to +3,000 mbb/h (value betting)
- vs HonestBot: ~0 to +500 mbb/h (both reasonable strategies)

This provides ground truth for comparing Deep CFR performance.
"""

import sys
from pathlib import Path
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np

try:
    import aion26_rust
except ImportError:
    print("ERROR: aion26_rust not available. Build with:")
    print("  cd /Users/vincentfraillon/Desktop/DPDCFR/aion26")
    print("  uv run maturin develop --manifest-path src/aion26_rust/Cargo.toml --release")
    sys.exit(1)

from aion26.games.rust_wrapper import new_rust_river_game
from aion26.baselines import RandomBot, CallingStation, HonestBot
from aion26.metrics.evaluator import HeadToHeadEvaluator

# ============================================================================
# Configuration
# ============================================================================

ITERATIONS = 1_000_000  # 1M iterations for convergence
EVAL_HANDS = 10_000     # 10k hands for tight confidence intervals
EVAL_EVERY = 100_000    # Evaluate every 100k iterations

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("TABULAR CFR - GROUND TRUTH SOLVER")
    print("="*80)
    print()

    print("Configuration:")
    print(f"  Algorithm: Vanilla CFR (tabular, exact)")
    print(f"  Iterations: {ITERATIONS:,}")
    print(f"  Backend: Rust (aion26_rust.TabularCFR)")
    print(f"  Evaluation: {EVAL_HANDS:,} hands per bot")
    print()

    # Create tabular CFR solver
    print("Initializing tabular CFR solver...")
    solver = aion26_rust.TabularCFR()
    print("✅ Solver initialized")
    print()

    # Training
    print("="*80)
    print("TRAINING")
    print("="*80)
    print()

    start_time = time.time()

    for iteration in range(1, ITERATIONS + 1):
        solver.run_iteration()

        # Progress reporting
        if iteration % 10000 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            iters_per_sec = iteration / elapsed if elapsed > 0 else 0
            print(f"Iter {iteration:7,} | States: {solver.num_states():6,} | {iters_per_sec:6.0f} it/s")

        # Intermediate evaluation
        if iteration % EVAL_EVERY == 0:
            print()
            print(f"{'='*80}")
            print(f"EVALUATION AT ITERATION {iteration:,}")
            print(f"{'='*80}")
            print()

            # Get average strategy
            strategies_dict = solver.get_all_strategies()
            print(f"Strategy size: {len(strategies_dict):,} information states")
            print()

            # Convert to numpy arrays
            strategy = {
                k: np.array(v, dtype=np.float64)
                for k, v in strategies_dict.items()
            }

            # Evaluate
            game = new_rust_river_game()
            evaluator = HeadToHeadEvaluator(big_blind=2.0)

            opponents = {
                "RandomBot": RandomBot(),
                "CallingStation": CallingStation(),
                "HonestBot": HonestBot(),
            }

            for name, bot in opponents.items():
                result = evaluator.evaluate(
                    initial_state=game,
                    strategy=strategy,
                    opponent=bot,
                    num_hands=EVAL_HANDS
                )

                print(f"{name}:")
                print(f"  {result.avg_mbb_per_hand:+7.0f} mbb/h ± {result.confidence_95:.0f} (95% CI)")
                print(f"  Total: {result.agent_winnings:+.1f} BB over {result.num_hands:,} hands")
                print()

            print(f"{'='*80}")
            print()

    # Final timing
    elapsed_total = time.time() - start_time
    print()
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print()
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"Total iterations: {ITERATIONS:,}")
    print(f"Final states explored: {solver.num_states():,}")
    print()

    # Final evaluation
    print("="*80)
    print("FINAL EVALUATION - NASH EQUILIBRIUM")
    print("="*80)
    print()

    # Get final strategy
    strategies_dict = solver.get_all_strategies()
    strategy = {
        k: np.array(v, dtype=np.float64)
        for k, v in strategies_dict.items()
    }

    print(f"Strategy size: {len(strategy):,} information states")
    print()

    # Evaluate against all bots
    game = new_rust_river_game()
    evaluator = HeadToHeadEvaluator(big_blind=2.0)

    opponents = {
        "RandomBot": RandomBot(),
        "CallingStation": CallingStation(),
        "HonestBot": HonestBot(),
    }

    print("Performance vs Baseline Bots:")
    print()

    results = {}
    for name, bot in opponents.items():
        result = evaluator.evaluate(
            initial_state=game,
            strategy=strategy,
            opponent=bot,
            num_hands=EVAL_HANDS
        )

        results[name] = result.avg_mbb_per_hand

        print(f"{name}:")
        print(f"  {result.avg_mbb_per_hand:+7.0f} mbb/h ± {result.confidence_95:.0f}")
        print(f"  Total: {result.agent_winnings:+.1f} BB over {result.num_hands:,} hands")
        print()

    print("="*80)
    print("GROUND TRUTH ESTABLISHED ✅")
    print("="*80)
    print()
    print("These are the THEORETICAL CEILINGS for River Hold'em:")
    print(f"  RandomBot:       {results['RandomBot']:+7.0f} mbb/h")
    print(f"  CallingStation:  {results['CallingStation']:+7.0f} mbb/h")
    print(f"  HonestBot:       {results['HonestBot']:+7.0f} mbb/h")
    print()
    print("Use these to evaluate Deep CFR performance!")
    print()


if __name__ == "__main__":
    main()
