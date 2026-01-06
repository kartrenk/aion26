"""Benchmark External Sampling MCCFR traversal performance.

This script measures the speedup from implementing External Sampling
in our Deep PDCFR+ trainer.

Expected results:
- Full traversal: ~182ms per iteration
- External Sampling: <20ms per iteration (9× speedup!)
"""

import time
import numpy as np
from aion26.games.leduc import LeducPoker
from aion26.deep_cfr.networks import LeducEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler

print("=" * 70)
print("BENCHMARK: External Sampling MCCFR Traversal")
print("=" * 70)
print()

# Initialize trainer
print("Initializing Deep PDCFR+ trainer...")
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

print("Running 100 iterations to measure traversal performance...\n")

# Measure traversal time
traversal_times = []

start_time = time.time()

for i in range(1, 101):
    # Measure just the traversal (2 calls per iteration)
    trav_start = time.time()

    trainer.traverse(trainer.initial_state, 0, 1.0, 1.0)
    trainer.traverse(trainer.initial_state, 1, 1.0, 1.0)

    trav_time = time.time() - trav_start
    traversal_times.append(trav_time)

    trainer.iteration += 1

    # Train network (to keep behavior consistent)
    if i % trainer.train_every == 0:
        trainer.train_network()

    # Update target
    if i % trainer.target_update_every == 0:
        trainer.update_target_network()

total_time = time.time() - start_time

# Calculate statistics
traversal_times = np.array(traversal_times) * 1000  # Convert to ms

print("=" * 70)
print("RESULTS")
print("=" * 70)
print()

print("Traversal Performance:")
print(f"  Mean time per iteration:  {np.mean(traversal_times):6.2f} ms")
print(f"  Median time:              {np.median(traversal_times):6.2f} ms")
print(f"  Std deviation:            {np.std(traversal_times):6.2f} ms")
print(f"  Min time:                 {np.min(traversal_times):6.2f} ms")
print(f"  Max time:                 {np.max(traversal_times):6.2f} ms")
print()

print("Overall Performance:")
print(f"  Total time (100 iters):   {total_time:6.2f} s")
print(f"  Time per iteration:       {total_time / 100 * 1000:6.2f} ms")
print(f"  Iterations per second:    {100 / total_time:6.2f}")
print()

# Compare to baseline
baseline_traversal_ms = 182.22  # From profiling (full traversal)
speedup = baseline_traversal_ms / np.mean(traversal_times)

print("Comparison to Full Traversal Baseline:")
print(f"  Baseline (full tree):     {baseline_traversal_ms:6.2f} ms/iter")
print(f"  Current (external samp):  {np.mean(traversal_times):6.2f} ms/iter")
print(f"  Speedup:                  {speedup:6.2f}×")
print()

# Target check
target_ms = 20.0
meets_target = np.mean(traversal_times) < target_ms

if meets_target:
    print(f"✅ SUCCESS: Traversal time {np.mean(traversal_times):.2f}ms < {target_ms:.2f}ms target!")
else:
    print(f"⚠️  CLOSE: Traversal time {np.mean(traversal_times):.2f}ms (target was {target_ms:.2f}ms)")

print()

# Buffer and training stats
print("Training Stats:")
print(f"  Buffer fill:              {len(trainer.buffer)} / {trainer.buffer.capacity} ({trainer.buffer.fill_percentage:.1f}%)")
print(f"  Average loss:             {trainer.get_average_loss():.4f}")
print(f"  Info states tracked:      {len(trainer.strategy_sum)}")
print()

# Projected performance for larger runs
print("Projected Performance:")
print(f"  1,000 iterations:         ~{(total_time / 100) * 1000:.1f}s  (~{(total_time / 100) * 1000 / 60:.1f} min)")
print(f"  10,000 iterations:        ~{(total_time / 100) * 10000:.0f}s (~{(total_time / 100) * 10000 / 60:.1f} min)")
print()

# Variance analysis (important for MCCFR)
print("Variance Analysis:")
print(f"  Coefficient of variation: {np.std(traversal_times) / np.mean(traversal_times) * 100:.1f}%")

if np.std(traversal_times) / np.mean(traversal_times) < 0.2:
    print("  ✅ Low variance - consistent performance")
elif np.std(traversal_times) / np.mean(traversal_times) < 0.5:
    print("  ✅ Moderate variance - acceptable for MCCFR")
else:
    print("  ⚠️  High variance - may need investigation")

print()
print("=" * 70)
print("EXTERNAL SAMPLING MCCFR VALIDATION")
print("=" * 70)
print()

if meets_target and speedup > 5:
    print("🚀 EXCELLENT PERFORMANCE!")
    print(f"   - {speedup:.1f}× faster than full traversal")
    print(f"   - {np.mean(traversal_times):.1f}ms per iteration")
    print("   - Ready for Texas Hold'em scaling")
elif speedup > 3:
    print("✅ GOOD PERFORMANCE")
    print(f"   - {speedup:.1f}× faster than full traversal")
    print("   - Sufficient for Leduc, may need tuning for Hold'em")
else:
    print("⚠️  MODERATE IMPROVEMENT")
    print(f"   - {speedup:.1f}× speedup (expected >5×)")
    print("   - May need further optimization")

print()
print("=" * 70)
