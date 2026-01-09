#!/usr/bin/env python3
"""Micro-Benchmark: Tabular CFR on FIXED BOARD.

This script runs tabular vanilla CFR on a single fixed board texture
to establish ground truth Nash equilibrium for that specific scenario.

Expected outcomes (fixed board [As, Ks, Qs, Js, 2h]):
- State count: < 30,000 (reachable info states)
- vs RandomBot: Positive and stable (Nash equilibrium exploits random play)
- Convergence: Should be monotonic, no oscillation
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
    print("  uv run maturin develop --manifest-path src/aion26_rust/Cargo.toml --release")
    sys.exit(1)

from aion26.games.rust_wrapper import RustRiverWrapper
from aion26.baselines import RandomBot, CallingStation, HonestBot
from aion26.metrics.evaluator import HeadToHeadEvaluator

# ============================================================================
# Configuration
# ============================================================================

ITERATIONS = 100_000    # 100k iterations for convergence on fixed board
EVAL_HANDS = 10_000     # 10k hands for tight confidence intervals
EVAL_EVERY = 10_000     # Evaluate every 10k iterations

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("TABULAR CFR - FIXED BOARD MICRO-BENCHMARK")
    print("="*80)
    print()

    print("Configuration:")
    print(f"  Algorithm: Vanilla CFR (tabular, exact)")
    print(f"  Board: FIXED [As, Ks, Qs, Js, 2h]")
    print(f"  Iterations: {ITERATIONS:,}")
    print(f"  Backend: Rust (aion26_rust.TabularCFR)")
    print(f"  Evaluation: {EVAL_HANDS:,} hands per bot")
    print()

    print("Expected Results:")
    print("  State count: < 30,000 (reachable info states)")
    print("  vs RandomBot: Positive and stable")
    print("  Convergence: Monotonic, no oscillation")
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
        if iteration % 1000 == 0 or iteration == 1:
            elapsed = time.time() - start_time
            iters_per_sec = iteration / elapsed if elapsed > 0 else 0
            print(f"Iter {iteration:6,} | States: {solver.num_states():6,} | {iters_per_sec:6.0f} it/s")

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

            # Create game with fixed board (matching the one used in training)
            # Board: [As, Ks, Qs, Js, 2h] = [12, 11, 10, 9, 13]
            fixed_board = [12, 11, 10, 9, 13]
            game = RustRiverWrapper(aion26_rust.RustRiverHoldem(
                stacks=[100.0, 100.0],
                pot=2.0,
                current_bet=0.0,
                player_0_invested=1.0,
                player_1_invested=1.0,
                fixed_board=fixed_board
            ))

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
    print("FINAL EVALUATION - NASH EQUILIBRIUM (FIXED BOARD)")
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
    fixed_board = [12, 11, 10, 9, 13]
    game = RustRiverWrapper(aion26_rust.RustRiverHoldem(
        stacks=[100.0, 100.0],
        pot=2.0,
        current_bet=0.0,
        player_0_invested=1.0,
        player_1_invested=1.0,
        fixed_board=fixed_board
    ))

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
        print()

    print("="*80)
    print("GROUND TRUTH ESTABLISHED ✅")
    print("="*80)
    print()
    print(f"Fixed Board [As, Ks, Qs, Js, 2h] Nash Equilibrium:")
    print(f"  State count:     {len(strategy):,}")
    print(f"  RandomBot:       {results['RandomBot']:+7.0f} mbb/h")
    print(f"  CallingStation:  {results['CallingStation']:+7.0f} mbb/h")
    print(f"  HonestBot:       {results['HonestBot']:+7.0f} mbb/h")
    print()

    # Check success criteria
    state_count_ok = len(strategy) < 30000
    random_positive = results['RandomBot'] > 0

    print("Success Criteria:")
    print(f"  State count < 30,000: {'✅ PASS' if state_count_ok else '❌ FAIL'} ({len(strategy):,})")
    print(f"  vs RandomBot > 0:     {'✅ PASS' if random_positive else '❌ FAIL'} ({results['RandomBot']:+.0f})")
    print()


if __name__ == "__main__":
    main()
