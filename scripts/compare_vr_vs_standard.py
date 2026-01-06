"""Compare VR-DDCFR+ vs Standard PDCFR+ on Leduc Poker.

This script validates the algorithmic improvements:
1. Variance Reduction with Value Network baseline
2. Full DDCFR strategy weighting (t^γ vs linear)

Training Setup:
- Game: Leduc Poker (288 information states)
- Iterations: 1,000 (configurable)
- Evaluation: NashConv every 100 iterations

Comparison:
- Standard PDCFR+: LinearScheduler for strategy (Phase 3 baseline)
- VR-DDCFR+: DDCFRStrategyScheduler + Value Network

Success Criteria:
- VR-DDCFR+ should show equal or better final NashConv
- VR-DDCFR+ should show smoother loss curves (lower variance)
- Both should converge (NashConv < 0.05 is good for 1K iters)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from aion26.games.leduc import LeducPoker
from aion26.deep_cfr.networks import LeducEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler, DDCFRStrategyScheduler
from aion26.metrics.exploitability import compute_exploitability


def train_agent(
    name: str,
    use_ddcfr_strategy: bool = True,
    num_iterations: int = 1000,
    eval_every: int = 100,
    seed: int = 42
):
    """Train a single agent and return metrics.

    Args:
        name: Agent name for logging
        use_ddcfr_strategy: If True, use DDCFRStrategyScheduler; else LinearScheduler
        num_iterations: Number of training iterations
        eval_every: Evaluate exploitability every N iterations
        seed: Random seed

    Returns:
        Dictionary with training history
    """
    print(f"\n{'=' * 70}")
    print(f"Training {name}")
    print(f"{'=' * 70}")

    # Game setup
    initial_state = LeducPoker()
    encoder = LeducEncoder()

    # Strategy scheduler selection
    if use_ddcfr_strategy:
        strategy_scheduler = DDCFRStrategyScheduler(gamma=2.0)
        print("Strategy weighting: DDCFR (t^2.0)")
    else:
        strategy_scheduler = LinearScheduler()
        print("Strategy weighting: Linear (t)")

    # Create trainer (use smaller buffer for faster convergence in validation)
    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=26,
        output_size=2,
        hidden_size=128,
        num_hidden_layers=3,
        buffer_capacity=1000,  # Smaller buffer fills faster for validation
        learning_rate=0.001,
        batch_size=128,
        seed=seed,
        device="cpu",
        regret_scheduler=PDCFRScheduler(alpha=2.0, beta=0.5),
        strategy_scheduler=strategy_scheduler,
    )

    # Training history
    history = {
        'iterations': [],
        'loss': [],
        'value_loss': [],
        'nashconv': [],
        'nashconv_iters': [],
    }

    print(f"\nStarting training for {num_iterations} iterations...")
    start_time = time.time()

    for i in range(1, num_iterations + 1):
        # Run iteration
        metrics = trainer.run_iteration()

        # Track losses
        history['iterations'].append(i)
        history['loss'].append(metrics.get('loss', 0.0))
        history['value_loss'].append(metrics.get('value_loss', 0.0))

        # Evaluate exploitability periodically
        if i % eval_every == 0 or i == 1:
            avg_strat = trainer.get_all_average_strategies()
            if len(avg_strat) > 0:
                nashconv = compute_exploitability(initial_state, avg_strat)
                history['nashconv'].append(nashconv)
                history['nashconv_iters'].append(i)
                print(f"  Iter {i:4d}: Loss={metrics['loss']:.4f}, "
                      f"ValLoss={metrics['value_loss']:.4f}, "
                      f"NashConv={nashconv:.4f}")
            else:
                print(f"  Iter {i:4d}: Loss={metrics['loss']:.4f}, "
                      f"ValLoss={metrics['value_loss']:.4f} (no strategy yet)")

    training_time = time.time() - start_time

    # Final evaluation
    print("\nComputing final exploitability...")
    final_strat = trainer.get_all_average_strategies()
    final_nashconv = compute_exploitability(initial_state, final_strat) if final_strat else None

    print(f"\nTraining Summary:")
    print(f"  Total time:        {training_time:.2f}s")
    print(f"  Iterations/sec:    {num_iterations / training_time:.2f}")
    print(f"  Final loss:        {history['loss'][-1]:.4f}")
    print(f"  Final value loss:  {history['value_loss'][-1]:.4f}")
    if final_nashconv is not None:
        print(f"  Final NashConv:    {final_nashconv:.4f}")
    else:
        print(f"  Final NashConv:    N/A (no strategy accumulated)")
    print(f"  Buffer fill:       {trainer.buffer.fill_percentage:.1f}%")

    history['final_nashconv'] = final_nashconv
    history['training_time'] = training_time

    return history


def create_comparison_plots(standard_history, vr_history, output_dir='plots'):
    """Create comparison plots.

    Args:
        standard_history: Training history from standard PDCFR+
        vr_history: Training history from VR-DDCFR+
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Figure 1: Loss comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Advantage network loss
    ax1.plot(standard_history['iterations'], standard_history['loss'],
             'b-', alpha=0.6, label='Standard PDCFR+ (Linear)')
    ax1.plot(vr_history['iterations'], vr_history['loss'],
             'r-', alpha=0.6, label='VR-DDCFR+ (t^2.0)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Advantage Loss')
    ax1.set_title('Advantage Network Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Value network loss
    ax2.plot(standard_history['iterations'], standard_history['value_loss'],
             'b-', alpha=0.6, label='Standard PDCFR+')
    ax2.plot(vr_history['iterations'], vr_history['value_loss'],
             'r-', alpha=0.6, label='VR-DDCFR+')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value Loss')
    ax2.set_title('Value Network Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/loss_comparison.png', dpi=150)
    print(f"\nSaved: {output_dir}/loss_comparison.png")
    plt.close()

    # Figure 2: NashConv comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(standard_history['nashconv_iters'], standard_history['nashconv'],
            'b-o', label='Standard PDCFR+ (Linear)', markersize=5)
    ax.plot(vr_history['nashconv_iters'], vr_history['nashconv'],
            'r-o', label='VR-DDCFR+ (t^2.0)', markersize=5)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('NashConv (Exploitability)', fontsize=12)
    ax.set_title('Convergence to Nash Equilibrium\n(Lower is better)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add improvement annotation if VR is better
    if vr_history['final_nashconv'] < standard_history['final_nashconv']:
        improvement = (standard_history['final_nashconv'] - vr_history['final_nashconv']) / \
                      standard_history['final_nashconv'] * 100
        ax.annotate(f'{improvement:.1f}% improvement',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/nashconv_comparison.png', dpi=150)
    print(f"Saved: {output_dir}/nashconv_comparison.png")
    plt.close()


def print_comparison_summary(standard_history, vr_history):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print()

    # NashConv comparison
    standard_final = standard_history['final_nashconv']
    vr_final = vr_history['final_nashconv']

    print("Final NashConv:")
    if standard_final is not None and vr_final is not None:
        print(f"  Standard PDCFR+ (Linear):  {standard_final:.4f}")
        print(f"  VR-DDCFR+ (t^2.0):         {vr_final:.4f}")

        if vr_final < standard_final:
            improvement = (standard_final - vr_final) / standard_final * 100
            print(f"  ✅ VR-DDCFR+ is {improvement:.1f}% better!")
        elif vr_final > standard_final:
            degradation = (vr_final - standard_final) / standard_final * 100
            print(f"  ⚠️  VR-DDCFR+ is {degradation:.1f}% worse")
        else:
            print(f"  ✅ Both converged to same value")
    else:
        print(f"  Standard PDCFR+ (Linear):  {standard_final if standard_final else 'N/A'}")
        print(f"  VR-DDCFR+ (t^2.0):         {vr_final if vr_final else 'N/A'}")
        print(f"  ⚠️  One or both strategies not accumulated (buffer not full)")
    print()

    # Loss comparison
    print("Final Losses:")
    print(f"  Standard advantage loss:  {standard_history['loss'][-1]:.4f}")
    print(f"  VR-DDCFR+ advantage loss: {vr_history['loss'][-1]:.4f}")
    print(f"  Standard value loss:      {standard_history['value_loss'][-1]:.4f}")
    print(f"  VR-DDCFR+ value loss:     {vr_history['value_loss'][-1]:.4f}")
    print()

    # Training time
    print("Training Time:")
    print(f"  Standard PDCFR+:  {standard_history['training_time']:.2f}s")
    print(f"  VR-DDCFR+:        {vr_history['training_time']:.2f}s")
    overhead = ((vr_history['training_time'] - standard_history['training_time']) /
                standard_history['training_time'] * 100)
    print(f"  Overhead:         {overhead:+.1f}%")
    print()


def main():
    """Main entry point."""
    print("=" * 70)
    print("VR-DDCFR+ VALIDATION ON LEDUC POKER")
    print("=" * 70)
    print()
    print("This script compares:")
    print("1. Standard PDCFR+ with Linear strategy weighting (Phase 3 baseline)")
    print("2. VR-DDCFR+ with t^γ strategy weighting + Value Network")
    print()

    # Train both agents
    num_iterations = 1000
    eval_every = 100

    standard_history = train_agent(
        name="Standard PDCFR+ (Linear)",
        use_ddcfr_strategy=False,
        num_iterations=num_iterations,
        eval_every=eval_every,
        seed=42
    )

    vr_history = train_agent(
        name="VR-DDCFR+ (t^2.0)",
        use_ddcfr_strategy=True,
        num_iterations=num_iterations,
        eval_every=eval_every,
        seed=42
    )

    # Create plots
    print("\nGenerating comparison plots...")
    create_comparison_plots(standard_history, vr_history)

    # Print summary
    print_comparison_summary(standard_history, vr_history)

    print("\n" + "=" * 70)
    print("✅ Validation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
