"""Benchmark: Deep PDCFR+ vs Vanilla Deep CFR on Leduc Poker.

This script compares two agents:
- Agent A: Vanilla Deep CFR (uniform weighting)
- Agent B: Deep PDCFR+ (dynamic discounting)

Both agents train on Leduc Poker for 2,000 iterations, and we measure
exploitability (NashConv) every 200 iterations.

Success criteria:
1. PDCFR+ should converge faster than vanilla
2. PDCFR+ should break the 0.20 NashConv barrier
3. Final PDCFR+ NashConv should be significantly lower than vanilla

Reference: Phase 2 baseline showed vanilla Deep CFR plateauing around 0.20-0.22
"""

import time
import numpy as np
import torch

from aion26.games.leduc import LeducPoker
from aion26.deep_cfr.networks import LeducEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import UniformScheduler, PDCFRScheduler, LinearScheduler
from aion26.metrics.exploitability import compute_exploitability


def train_agent(
    name: str,
    regret_scheduler,
    strategy_scheduler,
    num_iterations: int,
    eval_every: int,
    seed: int = 42
):
    """Train a single agent and return exploitability metrics.

    Args:
        name: Agent name for logging
        regret_scheduler: Scheduler for regret discounting
        strategy_scheduler: Scheduler for strategy accumulation
        num_iterations: Total training iterations
        eval_every: Evaluate exploitability every N iterations
        seed: Random seed

    Returns:
        List of (iteration, nashconv) tuples
    """
    print(f"\n{'=' * 70}")
    print(f"Training: {name}")
    print(f"{'=' * 70}")

    # Initialize game and encoder
    initial_state = LeducPoker()
    encoder = LeducEncoder()

    # Create trainer
    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=26,  # LeducEncoder output
        output_size=2,  # fold/check, bet/call
        hidden_size=128,
        num_hidden_layers=3,
        buffer_capacity=10000,
        learning_rate=0.001,
        batch_size=128,
        polyak=0.01,
        train_every=1,
        target_update_every=10,
        seed=seed,
        device="cpu",
        regret_scheduler=regret_scheduler,
        strategy_scheduler=strategy_scheduler,
    )

    # Track metrics
    results = []
    start_time = time.time()

    # Training loop
    for i in range(1, num_iterations + 1):
        trainer.run_iteration()

        # Evaluate exploitability
        if i % eval_every == 0 or i == 1:
            avg_strategies = trainer.get_all_average_strategies()
            nashconv = compute_exploitability(initial_state, avg_strategies)
            results.append((i, nashconv))

            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0

            print(f"  {i:5d}  {len(trainer.buffer):5d}  "
                  f"{trainer.get_average_loss():8.4f}  {nashconv:8.4f}  "
                  f"({rate:.1f} iter/s)")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({num_iterations/total_time:.1f} iter/s)")

    return results


def print_comparison_table(vanilla_results, pdcfr_results):
    """Print markdown comparison table.

    Args:
        vanilla_results: List of (iteration, nashconv) for vanilla
        pdcfr_results: List of (iteration, nashconv) for PDCFR+
    """
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print("\n| Iteration | Vanilla Deep CFR | Deep PDCFR+ | Improvement |")
    print("|-----------|------------------|-------------|-------------|")

    for (iter_v, nashconv_v), (iter_p, nashconv_p) in zip(vanilla_results, pdcfr_results):
        assert iter_v == iter_p, "Iteration mismatch"

        if nashconv_v > 0:
            improvement_pct = ((nashconv_v - nashconv_p) / nashconv_v) * 100
            improvement_str = f"{improvement_pct:+.1f}%"
        else:
            improvement_str = "N/A"

        print(f"| {iter_v:9d} | {nashconv_v:16.4f} | {nashconv_p:11.4f} | {improvement_str:11s} |")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)

    _, vanilla_final = vanilla_results[-1]
    _, pdcfr_final = pdcfr_results[-1]

    print(f"\nVanilla Deep CFR Final NashConv:  {vanilla_final:.4f}")
    print(f"Deep PDCFR+ Final NashConv:       {pdcfr_final:.4f}")

    if vanilla_final > 0:
        improvement = ((vanilla_final - pdcfr_final) / vanilla_final) * 100
        print(f"Improvement:                      {improvement:+.1f}%")

    # Success criteria
    print("\n" + "-" * 70)
    print("SUCCESS CRITERIA")
    print("-" * 70)

    # Criterion 1: Break 0.20 barrier
    broke_barrier = pdcfr_final < 0.20
    print(f"1. PDCFR+ breaks 0.20 NashConv barrier:  {'✅ YES' if broke_barrier else '❌ NO'} ({pdcfr_final:.4f})")

    # Criterion 2: Better than vanilla
    better_than_vanilla = pdcfr_final < vanilla_final
    print(f"2. PDCFR+ converges lower than vanilla:  {'✅ YES' if better_than_vanilla else '❌ NO'}")

    # Criterion 3: Convergence speed (check at midpoint)
    midpoint_idx = len(pdcfr_results) // 2
    _, vanilla_mid = vanilla_results[midpoint_idx]
    _, pdcfr_mid = pdcfr_results[midpoint_idx]
    faster_convergence = pdcfr_mid < vanilla_mid
    print(f"3. PDCFR+ converges faster (midpoint):   {'✅ YES' if faster_convergence else '❌ NO'}")

    # Overall verdict
    all_pass = broke_barrier and better_than_vanilla and faster_convergence
    print("\n" + "=" * 70)
    if all_pass:
        print("VERDICT: ✅ PDCFR+ WINS - All criteria met!")
    else:
        print("VERDICT: ⚠️  Mixed results - Further tuning needed")
    print("=" * 70)


def main():
    """Run the comparison benchmark."""
    print("\n" + "=" * 70)
    print("DEEP PDCFR+ vs VANILLA DEEP CFR - LEDUC POKER BENCHMARK")
    print("=" * 70)
    print("\nExperiment Configuration:")
    print("  Game:            Leduc Poker (288 information sets)")
    print("  Iterations:      2,000")
    print("  Evaluation:      Every 200 iterations")
    print("  Network:         3 layers × 128 units")
    print("  Buffer:          10,000 transitions")
    print("  Device:          CPU")
    print("\nAgent A (Vanilla):")
    print("  Regret:          UniformScheduler (w=1)")
    print("  Strategy:        UniformScheduler (w=1)")
    print("\nAgent B (PDCFR+):")
    print("  Regret:          PDCFRScheduler (α=2.0, β=0.5)")
    print("  Strategy:        LinearScheduler (w=t)")
    print("=" * 70)

    NUM_ITERATIONS = 2000
    EVAL_EVERY = 200
    SEED = 42

    # Train Agent A: Vanilla Deep CFR
    vanilla_results = train_agent(
        name="Agent A: Vanilla Deep CFR",
        regret_scheduler=UniformScheduler(),
        strategy_scheduler=UniformScheduler(),
        num_iterations=NUM_ITERATIONS,
        eval_every=EVAL_EVERY,
        seed=SEED
    )

    # Train Agent B: Deep PDCFR+
    pdcfr_results = train_agent(
        name="Agent B: Deep PDCFR+",
        regret_scheduler=PDCFRScheduler(alpha=2.0, beta=0.5),
        strategy_scheduler=LinearScheduler(),
        num_iterations=NUM_ITERATIONS,
        eval_every=EVAL_EVERY,
        seed=SEED
    )

    # Print comparison
    print_comparison_table(vanilla_results, pdcfr_results)


if __name__ == "__main__":
    main()
