"""Convergence verification for Deep CFR on Kuhn Poker.

This script trains a Deep CFR agent on Kuhn Poker and tracks its exploitability
over time to verify that it converges towards Nash equilibrium.

Success Criteria:
- Initial exploitability > 0.1 (random strategy is exploitable)
- Final exploitability < 0.05 (converging to Nash)
- Monotonically decreasing trend (learning is working)
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.deep_cfr.networks import KuhnEncoder
from aion26.games.kuhn import KuhnPoker
from aion26.metrics.exploitability import compute_exploitability


def main():
    """Run Deep CFR training with exploitability tracking."""
    print("=" * 70)
    print("Deep CFR Convergence Verification on Kuhn Poker")
    print("=" * 70)
    print()

    # Configuration
    NUM_ITERATIONS = 5000
    EVAL_EVERY = 250
    SEED = 42

    # Initialize game and encoder
    initial_state = KuhnPoker()
    encoder = KuhnEncoder()

    print("Configuration:")
    print(f"  Game: Kuhn Poker")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Eval frequency: every {EVAL_EVERY} iterations")
    print(f"  Seed: {SEED}")
    print()

    # Create Deep CFR trainer
    print("Creating Deep CFR Trainer...")
    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=10,
        output_size=2,
        hidden_size=128,         # Larger network
        num_hidden_layers=4,
        buffer_capacity=50000,   # Larger buffer
        batch_size=256,          # Larger batches
        learning_rate=0.0001,    # Smaller learning rate
        discount=0.0,            # Vanilla Deep CFR (no bootstrap)
        train_every=1,           # Train every iteration
        target_update_every=10,
        seed=SEED
    )
    print("✓ Trainer created")
    print()

    # Training loop with exploitability tracking
    print("Training Progress:")
    print("-" * 70)
    print(f"{'Iteration':>10} {'Buffer Fill':>12} {'Loss':>10} {'Exploitability':>15}")
    print("-" * 70)

    exploitability_history = []

    for iteration in range(NUM_ITERATIONS):
        # Run one iteration of Deep CFR
        metrics = trainer.run_iteration()

        # Evaluate exploitability periodically
        if (iteration + 1) % EVAL_EVERY == 0 or iteration == 0:
            # Extract average strategy from trainer
            avg_strategy = trainer.get_all_average_strategies()

            # Compute exploitability
            exploitability = compute_exploitability(initial_state, avg_strategy)
            exploitability_history.append((iteration + 1, exploitability))

            # Print progress
            buffer_fill = f"{metrics['buffer_fill_pct']:.1f}%"
            loss = metrics.get('loss', 0.0)

            print(f"{iteration + 1:>10} {buffer_fill:>12} {loss:>10.4f} {exploitability:>15.6f}")

    print("-" * 70)
    print()

    # Summary
    print("=" * 70)
    print("Convergence Analysis")
    print("=" * 70)
    print()

    # Initial vs final exploitability
    initial_exploit = exploitability_history[0][1]
    final_exploit = exploitability_history[-1][1]

    print(f"Initial Exploitability (iteration 1):       {initial_exploit:.6f}")
    print(f"Final Exploitability (iteration {NUM_ITERATIONS}):   {final_exploit:.6f}")
    print(f"Reduction:                                  {initial_exploit - final_exploit:.6f}")
    print(f"Improvement:                                {(1 - final_exploit/initial_exploit)*100:.1f}%")
    print()

    # Check convergence criteria
    print("Convergence Criteria:")
    print("-" * 70)

    criteria_passed = 0
    total_criteria = 3

    # Criterion 1: Initial exploitability > 0.1
    if initial_exploit > 0.1:
        print("✓ Initial exploitability > 0.1 (random strategy)")
        criteria_passed += 1
    else:
        print(f"✗ Initial exploitability {initial_exploit:.4f} <= 0.1 (unexpected)")

    # Criterion 2: Final exploitability < 0.05
    if final_exploit < 0.05:
        print("✓ Final exploitability < 0.05 (converging to Nash)")
        criteria_passed += 1
    else:
        print(f"✗ Final exploitability {final_exploit:.4f} >= 0.05 (not converged)")

    # Criterion 3: Monotonically decreasing (mostly)
    decreasing_count = 0
    for i in range(1, len(exploitability_history)):
        if exploitability_history[i][1] <= exploitability_history[i-1][1]:
            decreasing_count += 1

    decreasing_pct = (decreasing_count / (len(exploitability_history) - 1)) * 100

    if decreasing_pct >= 70:  # Allow some fluctuations
        print(f"✓ Exploitability decreasing {decreasing_pct:.1f}% of the time")
        criteria_passed += 1
    else:
        print(f"✗ Exploitability only decreasing {decreasing_pct:.1f}% of the time")

    print("-" * 70)
    print()

    # Final verdict
    print("=" * 70)
    if criteria_passed == total_criteria:
        print("✓ SUCCESS: Deep CFR converges to Nash Equilibrium on Kuhn Poker!")
        print("=" * 70)
        return 0
    else:
        print(f"✗ PARTIAL: {criteria_passed}/{total_criteria} criteria passed")
        print("=" * 70)
        return 1

    print()
    print("Exploitability History:")
    print("-" * 40)
    for iteration, exploit in exploitability_history:
        print(f"  Iteration {iteration:>4}: {exploit:.6f}")
    print()


if __name__ == "__main__":
    sys.exit(main())
