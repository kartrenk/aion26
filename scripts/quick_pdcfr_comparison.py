"""Fast PDCFR+ vs Vanilla comparison (1000 iterations, eval every 200).

Optimized for quick validation while still demonstrating PDCFR+ superiority.
"""

import time
import numpy as np
from aion26.games.leduc import LeducPoker
from aion26.deep_cfr.networks import LeducEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import UniformScheduler, PDCFRScheduler, LinearScheduler
from aion26.metrics.exploitability import compute_exploitability


def train_and_eval(name, regret_sched, strategy_sched, iterations=1000, eval_every=200, seed=42):
    """Train agent and return eval history."""
    initial_state = LeducPoker()
    encoder = LeducEncoder()

    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=26,
        output_size=2,
        hidden_size=128,
        num_hidden_layers=3,
        buffer_capacity=10000,
        learning_rate=0.001,
        batch_size=128,
        seed=seed,
        device="cpu",
        regret_scheduler=regret_sched,
        strategy_scheduler=strategy_sched,
    )

    results = []
    for i in range(1, iterations + 1):
        trainer.run_iteration()

        if i % eval_every == 0 or i == 1:
            avg_strat = trainer.get_all_average_strategies()
            nashconv = compute_exploitability(initial_state, avg_strat)
            results.append((i, nashconv))

    return results


print("=" * 70)
print("QUICK COMPARISON: PDCFR+ vs Vanilla (1000 iterations)")
print("=" * 70)
print()

start = time.time()

# Vanilla
print("Training Vanilla Deep CFR...")
vanilla_results = train_and_eval(
    "Vanilla",
    UniformScheduler(),
    UniformScheduler(),
    iterations=1000,
    eval_every=200
)
print(f"  Done in {time.time() - start:.1f}s\n")

# PDCFR+
print("Training Deep PDCFR+...")
pdcfr_start = time.time()
pdcfr_results = train_and_eval(
    "PDCFR+",
    PDCFRScheduler(alpha=2.0, beta=0.5),
    LinearScheduler(),
    iterations=1000,
    eval_every=200
)
print(f"  Done in {time.time() - pdcfr_start:.1f}s\n")

total_time = time.time() - start

# Print comparison
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print("| Iteration | Vanilla | PDCFR+ | Improvement |")
print("|-----------|---------|--------|-------------|")

for (iter_v, nash_v), (iter_p, nash_p) in zip(vanilla_results, pdcfr_results):
    if nash_v > 0:
        improvement = ((nash_v - nash_p) / nash_v) * 100
        imp_str = f"{improvement:+.1f}%"
    else:
        imp_str = "N/A"
    print(f"| {iter_v:9d} | {nash_v:7.4f} | {nash_p:6.4f} | {imp_str:11s} |")

print()
_, vanilla_final = vanilla_results[-1]
_, pdcfr_final = pdcfr_results[-1]

improvement = ((vanilla_final - pdcfr_final) / vanilla_final) * 100 if vanilla_final > 0 else 0

print(f"Final NashConv:")
print(f"  Vanilla: {vanilla_final:.4f}")
print(f"  PDCFR+:  {pdcfr_final:.4f}")
print(f"  Improvement: {improvement:+.1f}%")
print()
print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
print()

if pdcfr_final < vanilla_final and pdcfr_final < 0.30:
    print("✅ PDCFR+ WINS - Faster convergence confirmed!")
else:
    print("⚠️  Results mixed - may need more iterations")

print("=" * 70)
