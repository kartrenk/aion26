"""Training script for Deep CFR on Leduc Poker.

This script trains a Deep CFR agent on Leduc Poker and tracks convergence
towards Nash equilibrium via exploitability (NashConv).

Expected Results:
- Initial NashConv: ~4.0 (near-uniform random strategy)
- After 10k iterations: ~1.0-2.0 (significant learning)
- Vanilla Deep CFR may not fully converge (that's what PDCFR+ is for!)

The goal is to demonstrate that the agent LEARNS, not necessarily that
it fully converges to Nash equilibrium in 10k iterations.
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.deep_cfr.networks import LeducEncoder
from aion26.games.leduc import LeducPoker
from aion26.metrics.exploitability import compute_exploitability


def print_header():
    """Print training header."""
    print("=" * 80)
    print("Deep CFR Training on Leduc Poker")
    print("=" * 80)
    print()


def print_config(config: dict):
    """Print training configuration."""
    print("Configuration:")
    print("-" * 80)
    for key, value in config.items():
        print(f"  {key:<25}: {value}")
    print("-" * 80)
    print()


def optimize_network(network: nn.Module) -> nn.Module:
    """Apply optimizations to network if available.

    Args:
        network: Neural network to optimize

    Returns:
        Optimized network (or original if optimizations unavailable)
    """
    # Try to use torch.compile (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            print("  ✓ Applying torch.compile() optimization...")
            return torch.compile(network)
        except Exception as e:
            print(f"  ⚠ torch.compile() failed ({e}), using standard network")
            return network
    else:
        print("  ⚠ torch.compile() not available (PyTorch < 2.0)")
        return network


def main():
    """Run Deep CFR training on Leduc Poker."""
    print_header()

    # ==================== Configuration ====================

    config = {
        "Game": "Leduc Poker (~288 info sets)",
        "Iterations": 10000,
        "Eval Frequency": 500,
        "Seed": 42,
        "Network Hidden Size": 256,
        "Network Layers": 5,
        "Buffer Capacity": 100000,
        "Batch Size": 512,
        "Initial Learning Rate": 0.00005,
        "LR Decay Step": 5000,
        "LR Decay Gamma": 0.5,
        "Device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    print_config(config)

    # ==================== Setup ====================

    print("Initializing...")

    device = config["Device"]
    initial_state = LeducPoker()
    encoder = LeducEncoder()

    # Create trainer
    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=26,
        output_size=2,
        hidden_size=config["Network Hidden Size"],
        num_hidden_layers=config["Network Layers"],
        buffer_capacity=config["Buffer Capacity"],
        batch_size=config["Batch Size"],
        learning_rate=config["Initial Learning Rate"],
        discount=0.0,  # Vanilla Deep CFR (no bootstrap for now)
        train_every=1,
        target_update_every=20,
        seed=config["Seed"],
        device=device
    )

    # Apply network optimizations
    print()
    print("Applying optimizations:")
    trainer.advantage_net = optimize_network(trainer.advantage_net)
    trainer.target_net = optimize_network(trainer.target_net)

    # Setup learning rate scheduler
    scheduler = StepLR(
        trainer.optimizer,
        step_size=config["LR Decay Step"],
        gamma=config["LR Decay Gamma"]
    )

    print("  ✓ Learning rate scheduler configured")
    print(f"    Initial LR: {config['Initial Learning Rate']}")
    print(f"    Decay every {config['LR Decay Step']} iterations")
    print(f"    Decay factor: {config['LR Decay Gamma']}")
    print()

    print("✓ Trainer initialized successfully")
    print()

    # ==================== Training Loop ====================

    print("Training Progress:")
    print("=" * 80)
    print(f"{'Iter':>6} {'Time':>8} {'Buffer':>8} {'Loss':>10} {'LR':>10} {'NashConv':>12} {'Status':>15}")
    print("=" * 80)

    start_time = time.time()
    history = []

    for iteration in range(1, config["Iterations"] + 1):
        # Run one iteration
        metrics = trainer.run_iteration()

        # Step learning rate scheduler
        if iteration % config["LR Decay Step"] == 0:
            scheduler.step()

        # Evaluate periodically
        if iteration % config["Eval Frequency"] == 0 or iteration == 1:
            elapsed = time.time() - start_time

            # Extract average strategy
            avg_strategy = trainer.get_all_average_strategies()

            # Compute exploitability (NashConv)
            try:
                nashconv = compute_exploitability(initial_state, avg_strategy)

                # Get current learning rate
                current_lr = trainer.optimizer.param_groups[0]['lr']

                # Print progress
                buffer_pct = f"{metrics['buffer_fill_pct']:.1f}%"
                loss = metrics.get('loss', 0.0)

                # Determine status
                if iteration == 1:
                    status = "Initial"
                elif len(history) > 0 and nashconv < history[-1][1]:
                    status = "↓ Improving"
                elif len(history) > 0 and nashconv > history[-1][1]:
                    status = "↑ Degrading"
                else:
                    status = "→ Stable"

                print(f"{iteration:>6} {elapsed:>7.1f}s {buffer_pct:>8} {loss:>10.4f} {current_lr:>10.6f} {nashconv:>12.4f} {status:>15}")

                # Store history
                history.append((iteration, nashconv, loss))

            except Exception as e:
                print(f"{iteration:>6} {'Error':>8} {'':<10} Computing exploitability: {e}")

    print("=" * 80)
    print()

    # ==================== Analysis ====================

    print("=" * 80)
    print("Training Complete - Analysis")
    print("=" * 80)
    print()

    if len(history) >= 2:
        initial_nashconv = history[0][1]
        final_nashconv = history[-1][1]

        print(f"Initial NashConv (iter 1):          {initial_nashconv:.4f}")
        print(f"Final NashConv (iter {config['Iterations']}):    {final_nashconv:.4f}")
        print(f"Absolute Reduction:                 {initial_nashconv - final_nashconv:.4f}")

        if initial_nashconv > 0:
            improvement_pct = (1 - final_nashconv / initial_nashconv) * 100
            print(f"Relative Improvement:               {improvement_pct:.1f}%")

        print()

        # Check if learning occurred
        print("Learning Criteria:")
        print("-" * 80)

        criteria_met = 0
        total_criteria = 3

        # Criterion 1: NashConv decreased
        if final_nashconv < initial_nashconv:
            print("✓ NashConv decreased (agent is learning)")
            criteria_met += 1
        else:
            print("✗ NashConv did not decrease")

        # Criterion 2: Final NashConv < 2.0 (reasonable for 10k iterations)
        if final_nashconv < 2.0:
            print(f"✓ Final NashConv < 2.0 (good progress for vanilla Deep CFR)")
            criteria_met += 1
        else:
            print(f"⚠ Final NashConv {final_nashconv:.2f} >= 2.0 (may need more iterations)")

        # Criterion 3: Trend is decreasing
        decreasing_count = 0
        for i in range(1, len(history)):
            if history[i][1] <= history[i-1][1]:
                decreasing_count += 1

        if len(history) > 1:
            decreasing_pct = (decreasing_count / (len(history) - 1)) * 100
            if decreasing_pct >= 60:
                print(f"✓ NashConv trending down ({decreasing_pct:.1f}% of evaluations)")
                criteria_met += 1
            else:
                print(f"⚠ NashConv not consistently decreasing ({decreasing_pct:.1f}%)")

        print("-" * 80)
        print()

        # Final verdict
        print("=" * 80)
        if criteria_met >= 2:
            print("✓ SUCCESS: Deep CFR learns on Leduc Poker!")
            print()
            print("  The agent demonstrates clear learning progress.")
            print("  For full convergence to Nash equilibrium, use:")
            print("    1. More iterations (20k-50k)")
            print("    2. PDCFR+ with dynamic discounting (Phase 3)")
            print("    3. Larger networks or hyperparameter tuning")
        else:
            print("⚠ PARTIAL: Learning detected but convergence incomplete")
            print()
            print("  Suggestions:")
            print("    - Increase iterations")
            print("    - Tune hyperparameters (LR, network size)")
            print("    - Verify buffer is filling properly")

        print("=" * 80)
        print()

        # NashConv history
        print("NashConv History:")
        print("-" * 40)
        for iter_num, nashconv, loss in history:
            print(f"  Iteration {iter_num:>5}: NashConv={nashconv:>8.4f}, Loss={loss:>8.4f}")
        print()

        # Training statistics
        total_time = time.time() - start_time
        iterations_per_sec = config["Iterations"] / total_time

        print("Training Statistics:")
        print("-" * 40)
        print(f"  Total time:        {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Iterations/sec:    {iterations_per_sec:.2f}")
        print(f"  Info sets visited: {len(trainer.strategy_sum)}")
        print(f"  Buffer size:       {len(trainer.buffer)}")
        print(f"  Training steps:    {trainer.num_train_steps}")
        print(f"  Average loss:      {trainer.get_average_loss():.4f}")
        print()

        return 0 if criteria_met >= 2 else 1

    else:
        print("✗ ERROR: Insufficient data collected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
