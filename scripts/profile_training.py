"""Profile Deep PDCFR+ training to identify performance bottlenecks.

This script measures time spent in different components during training.
"""

import time
import numpy as np
from aion26.games.leduc import LeducPoker
from aion26.deep_cfr.networks import LeducEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler
from aion26.metrics.exploitability import compute_exploitability


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name):
        self.name = name
        self.times = []

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        self.times.append(elapsed)

    @property
    def total(self):
        return sum(self.times)

    @property
    def mean(self):
        return np.mean(self.times) if self.times else 0

    @property
    def count(self):
        return len(self.times)


print("=" * 70)
print("PERFORMANCE PROFILING: Deep PDCFR+ Training")
print("=" * 70)
print()

# Initialize
print("Initializing...")
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
    seed=42,
    device="cpu",
    regret_scheduler=PDCFRScheduler(alpha=2.0, beta=0.5),
    strategy_scheduler=LinearScheduler(),
)

# Timers
timers = {
    'traversal': Timer('CFR Traversal'),
    'network_train': Timer('Network Training'),
    'target_update': Timer('Target Update'),
    'exploitability': Timer('Exploitability Calc'),
    'total_iteration': Timer('Total Iteration'),
}

print("Training 200 iterations with detailed profiling...\n")

total_start = time.time()

for i in range(1, 201):
    with timers['total_iteration']:
        # Traverse for both players
        trav_start = time.time()
        trainer.traverse(trainer.initial_state, 0, 1.0, 1.0)
        trainer.traverse(trainer.initial_state, 1, 1.0, 1.0)
        timers['traversal'].times.append(time.time() - trav_start)

        trainer.iteration += 1

        # Train network
        if trainer.iteration % trainer.train_every == 0:
            train_start = time.time()
            trainer.train_network()
            timers['network_train'].times.append(time.time() - train_start)

        # Update target
        if trainer.iteration % trainer.target_update_every == 0:
            update_start = time.time()
            trainer.update_target_network()
            timers['target_update'].times.append(time.time() - update_start)

    # Periodic exploitability (expensive, do less often)
    if i % 100 == 0:
        with timers['exploitability']:
            avg_strat = trainer.get_all_average_strategies()
            nashconv = compute_exploitability(initial_state, avg_strat)

        print(f"Iteration {i:3d}: NashConv = {nashconv:.4f}")

total_time = time.time() - total_start

print()
print("=" * 70)
print("PROFILING RESULTS")
print("=" * 70)
print()

# Per-iteration breakdown
print("Time per iteration (average):")
print(f"  Total:            {timers['total_iteration'].mean * 1000:6.2f} ms")
print(f"    CFR Traversal:  {timers['traversal'].mean * 1000:6.2f} ms ({timers['traversal'].total / timers['total_iteration'].total * 100:5.1f}%)")
print(f"    Network Train:  {timers['network_train'].mean * 1000:6.2f} ms ({timers['network_train'].total / timers['total_iteration'].total * 100:5.1f}%)")
print(f"    Target Update:  {timers['target_update'].mean * 1000:6.2f} ms ({timers['target_update'].total / timers['total_iteration'].total * 100:5.1f}%)")
print()

# Total time breakdown
print("Total time distribution:")
total_accounted = timers['traversal'].total + timers['network_train'].total + timers['target_update'].total
print(f"  CFR Traversal:    {timers['traversal'].total:6.2f}s ({timers['traversal'].total / total_time * 100:5.1f}%)")
print(f"  Network Training: {timers['network_train'].total:6.2f}s ({timers['network_train'].total / total_time * 100:5.1f}%)")
print(f"  Target Updates:   {timers['target_update'].total:6.2f}s ({timers['target_update'].total / total_time * 100:5.1f}%)")
print(f"  Exploitability:   {timers['exploitability'].total:6.2f}s ({timers['exploitability'].total / total_time * 100:5.1f}%)")
print(f"  Other overhead:   {total_time - total_accounted - timers['exploitability'].total:6.2f}s ({(total_time - total_accounted - timers['exploitability'].total) / total_time * 100:5.1f}%)")
print(f"  TOTAL:            {total_time:6.2f}s")
print()

# Throughput
print("Throughput:")
print(f"  Iterations/second: {200 / total_time:.2f}")
print(f"  Seconds/iteration: {total_time / 200:.3f}")
print()

# Network stats
print("Network training stats:")
print(f"  Train calls:       {timers['network_train'].count}")
print(f"  Frequency:         Every {trainer.train_every} iteration(s)")
print(f"  Batch size:        {trainer.batch_size}")
print(f"  Buffer size:       {len(trainer.buffer)} / {trainer.buffer.capacity}")
print(f"  Average loss:      {trainer.get_average_loss():.4f}")
print()

# Exploitability stats
print("Exploitability calculation:")
print(f"  Calls:             {timers['exploitability'].count}")
print(f"  Mean time:         {timers['exploitability'].mean:.3f}s")
print(f"  Total time:        {timers['exploitability'].total:.3f}s")
print(f"  % of total:        {timers['exploitability'].total / total_time * 100:.1f}%")
print()

# Recommendations
print("=" * 70)
print("PERFORMANCE RECOMMENDATIONS")
print("=" * 70)
print()

traversal_pct = timers['traversal'].total / total_time * 100
network_pct = timers['network_train'].total / total_time * 100
exploit_pct = timers['exploitability'].total / total_time * 100

if traversal_pct > 50:
    print("🔥 BOTTLENECK: CFR Traversal")
    print(f"   {traversal_pct:.1f}% of time spent in traversal")
    print("   Optimizations:")
    print("   - Use outcome sampling (MCCFR) instead of full traversal")
    print("   - Vectorize state encoding")
    print("   - Cache information state strings")
    print()

if network_pct > 30:
    print("⚡ BOTTLENECK: Network Training")
    print(f"   {network_pct:.1f}% of time spent training network")
    print("   Optimizations:")
    print("   - Move to GPU (use device='cuda')")
    print("   - Reduce train_every (train less frequently)")
    print("   - Use smaller batch size")
    print()

if exploit_pct > 20:
    print("📊 BOTTLENECK: Exploitability Calculation")
    print(f"   {exploit_pct:.1f}% of time in exploitability")
    print("   Optimizations:")
    print("   - Evaluate less frequently (e.g., every 500 iterations)")
    print("   - Use approximate metrics (loss, strategy change)")
    print("   - Skip during training, only evaluate at end")
    print()

if max(traversal_pct, network_pct, exploit_pct) < 40:
    print("✅ BALANCED: No major bottlenecks detected")
    print("   All components take reasonable time")
    print()

print("Projected scaling:")
print(f"  Time for 1,000 iters:  ~{total_time * 5:.0f}s  (~{total_time * 5 / 60:.1f} min)")
print(f"  Time for 10,000 iters: ~{total_time * 50:.0f}s (~{total_time * 50 / 60:.1f} min)")
print()
print("=" * 70)
