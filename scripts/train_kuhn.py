#!/usr/bin/env python3
"""Training script for Kuhn Poker with Vanilla CFR.

This script trains a CFR agent on Kuhn Poker and logs convergence metrics.
It demonstrates Phase 1 of the Aion-26 project: tabular CFR on a simple game.

Example usage:
    uv run python scripts/train_kuhn.py --iterations 10000 --log-every 1000
"""

import argparse
import csv
import time
from pathlib import Path

from aion26.games.kuhn import new_kuhn_game
from aion26.cfr.vanilla import VanillaCFR
from aion26.metrics.exploitability import compute_exploitability, evaluate_strategy_profile


def main():
    parser = argparse.ArgumentParser(description="Train CFR on Kuhn Poker")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Number of CFR iterations to run (default: 10000)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log metrics every N iterations (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kuhn_cfr_training.csv",
        help="Output CSV file for metrics (default: kuhn_cfr_training.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output",
    )

    args = parser.parse_args()

    print(f"Aion-26: Kuhn Poker CFR Training")
    print(f"=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Log frequency: every {args.log_every} iterations")
    print(f"Random seed: {args.seed}")
    print(f"Output file: {args.output}")
    print(f"=" * 60)
    print()

    # Initialize game and solver
    game = new_kuhn_game()
    solver = VanillaCFR(game, seed=args.seed)

    # Prepare CSV output
    output_path = Path(args.output)
    csv_file = open(output_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "iteration",
            "exploitability",
            "best_response_value_p0",
            "best_response_value_p1",
            "elapsed_time",
        ],
    )
    csv_writer.writeheader()

    # Training loop
    start_time = time.time()
    try:
        for i in range(1, args.iterations + 1):
            solver.run_iteration()

            # Log metrics periodically
            if i % args.log_every == 0 or i == args.iterations:
                strategy = solver.get_all_average_strategies()
                metrics = evaluate_strategy_profile(game, strategy)

                elapsed = time.time() - start_time

                # Write to CSV
                csv_writer.writerow({
                    "iteration": i,
                    "exploitability": metrics["exploitability"],
                    "best_response_value_p0": metrics["best_response_value_p0"],
                    "best_response_value_p1": metrics["best_response_value_p1"],
                    "elapsed_time": elapsed,
                })
                csv_file.flush()

                if args.verbose:
                    print(f"Iteration {i:6d}:")
                    print(f"  Exploitability: {metrics['exploitability']:.6f}")
                    print(f"  BR Value P0: {metrics['best_response_value_p0']:+.6f}")
                    print(f"  BR Value P1: {metrics['best_response_value_p1']:+.6f}")
                    print(f"  Elapsed: {elapsed:.2f}s")
                    print()
                else:
                    print(f"[{i:6d}/{args.iterations}] Exploit: {metrics['exploitability']:.6f}, Elapsed: {elapsed:.2f}s")

    finally:
        csv_file.close()

    # Print final statistics
    print()
    print(f"=" * 60)
    print(f"Training complete!")
    print(f"Final exploitability: {metrics['exploitability']:.6f}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Iterations/second: {args.iterations / elapsed:.1f}")
    print(f"Results saved to: {output_path}")
    print()

    # Print learned strategies for key information states
    print("Learned Strategies (Average):")
    print("-" * 60)
    key_states = ["J", "Q", "K", "Jb", "Qb", "Kb"]
    for state in key_states:
        if state in strategy:
            strat = strategy[state]
            print(f"  {state:4s}: Check={strat[0]:.3f}, Bet={strat[1]:.3f}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
