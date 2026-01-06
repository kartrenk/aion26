"""Convergence verification for Deep CFR on Leduc Poker.

Leduc Poker is significantly more complex than Kuhn:
- 6 cards (vs 3 in Kuhn)
- 2 betting rounds (vs 1 in Kuhn)
- ~288 information sets (vs 12 in Kuhn)

This script verifies that Deep CFR can scale to this larger game.

Success Criteria:
- Exploitability decreases over time
- Final exploitability reasonable for the game size
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.deep_cfr.networks import LeducEncoder
from aion26.games.leduc import LeducPoker
from aion26.metrics.exploitability import compute_exploitability


def main():
    """Run Deep CFR training on Leduc Poker with exploitability tracking."""
    print("=" * 70)
    print("Deep CFR Convergence Verification on Leduc Poker")
    print("=" * 70)
    print()

    # Configuration (adapted for larger game)
    NUM_ITERATIONS = 10000
    EVAL_EVERY = 500
    SEED = 42

    # Initialize game and encoder
    initial_state = LeducPoker()
    encoder = LeducEncoder()

    print("Configuration:")
    print(f"  Game: Leduc Poker (~288 information sets)")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print(f"  Eval frequency: every {EVAL_EVERY} iterations")
    print(f"  Seed: {SEED}")
    print()

    # Create Deep CFR trainer with larger hyperparameters
    print("Creating Deep CFR Trainer...")
    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=26,           # LeducEncoder output size
        output_size=2,           # check/fold or bet/call
        hidden_size=256,         # Larger network for larger game
        num_hidden_layers=5,     # Deeper network
        buffer_capacity=100000,  # Much larger buffer
        batch_size=512,          # Larger batches
        learning_rate=0.00005,   # Smaller LR for stability
        discount=0.0,            # Vanilla Deep CFR
        train_every=1,
        target_update_every=20,
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
            try:
                exploitability = compute_exploitability(initial_state, avg_strategy)
                exploitability_history.append((iteration + 1, exploitability))

                # Print progress
                buffer_fill = f"{metrics['buffer_fill_pct']:.1f}%"
                loss = metrics.get('loss', 0.0)

                print(f"{iteration + 1:>10} {buffer_fill:>12} {loss:>10.4f} {exploitability:>15.6f}")
            except Exception as e:
                print(f"{iteration + 1:>10} {'Error':>12} {'':<10} Computing exploitability...")

    print("-" * 70)
    print()

    # Summary
    print("=" * 70)
    print("Convergence Analysis")
    print("=" * 70)
    print()

    if len(exploitability_history) >= 2:
        # Initial vs final exploitability
        initial_exploit = exploitability_history[0][1]
        final_exploit = exploitability_history[-1][1]

        print(f"Initial Exploitability (iteration 1):         {initial_exploit:.6f}")
        print(f"Final Exploitability (iteration {NUM_ITERATIONS}):   {final_exploit:.6f}")
        print(f"Reduction:                                    {initial_exploit - final_exploit:.6f}")

        if initial_exploit > 0:
            improvement = (1 - final_exploit/initial_exploit) * 100
            print(f"Improvement:                                  {improvement:.1f}%")
        print()

        # Check if exploitability decreased
        print("Convergence Criteria:")
        print("-" * 70)

        if final_exploit < initial_exploit:
            print("✓ Exploitability decreased (learning is working)")
        else:
            print("✗ Exploitability did not decrease")

        # Check trend
        decreasing_count = 0
        for i in range(1, len(exploitability_history)):
            if exploitability_history[i][1] <= exploitability_history[i-1][1]:
                decreasing_count += 1

        if len(exploitability_history) > 1:
            decreasing_pct = (decreasing_count / (len(exploitability_history) - 1)) * 100
            if decreasing_pct >= 60:  # Allow more variance for larger game
                print(f"✓ Exploitability trending down ({decreasing_pct:.1f}% of evals)")
            else:
                print(f"✗ Exploitability not trending down ({decreasing_pct:.1f}% of evals)")

        print("-" * 70)
        print()

        # Exploitability history
        print("Exploitability History:")
        print("-" * 40)
        for iteration, exploit in exploitability_history:
            print(f"  Iteration {iteration:>5}: {exploit:.6f}")
        print()

        # Final verdict
        print("=" * 70)
        if final_exploit < initial_exploit:
            print("✓ Deep CFR is learning on Leduc Poker!")
            print("  (Convergence is slower for larger games - this is expected)")
        else:
            print("⚠ Deep CFR may need hyperparameter tuning for Leduc")
        print("=" * 70)
        return 0 if final_exploit < initial_exploit else 1
    else:
        print("Not enough data collected for analysis")
        return 1


if __name__ == "__main__":
    sys.exit(main())
