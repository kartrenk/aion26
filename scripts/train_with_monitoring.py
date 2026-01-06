"""Enhanced Deep PDCFR+ training with comprehensive monitoring.

This script provides detailed real-time statistics during training:
- Iteration timing and throughput
- Buffer fill percentage
- Network loss (training quality)
- Strategy change (convergence indicator)
- Periodic exploitability (ground truth metric)

Usage:
    python scripts/train_with_monitoring.py --iterations 2000 --game leduc
"""

import argparse
import time
import numpy as np
import sys
from pathlib import Path

from aion26.games.leduc import LeducPoker
from aion26.games.kuhn import KuhnPoker
from aion26.deep_cfr.networks import LeducEncoder, KuhnEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler, UniformScheduler
from aion26.metrics.exploitability import compute_exploitability


class TrainingMonitor:
    """Monitor training progress with detailed statistics."""

    def __init__(self, eval_exploitability_every=500):
        self.eval_exploitability_every = eval_exploitability_every
        self.iteration_times = []
        self.losses = []
        self.exploitabilities = []
        self.buffer_fills = []
        self.strategy_changes = []
        self.prev_strategies = {}

    def log_iteration(self, iteration, metrics, trainer, initial_state):
        """Log statistics for current iteration."""
        iter_time = metrics.get('iter_time', 0)
        loss = trainer.get_average_loss()
        buffer_fill = len(trainer.buffer) / trainer.buffer.capacity * 100

        self.iteration_times.append(iter_time)
        self.losses.append(loss)
        self.buffer_fills.append(buffer_fill)

        # Compute strategy change (convergence indicator)
        if self.prev_strategies:
            strategy_change = self._compute_strategy_change(
                trainer.get_all_average_strategies(),
                self.prev_strategies
            )
            self.strategy_changes.append(strategy_change)
        else:
            strategy_change = 0.0

        self.prev_strategies = trainer.get_all_average_strategies().copy()

        # Compute exploitability periodically (expensive)
        if iteration % self.eval_exploitability_every == 0 or iteration == 1:
            avg_strat = trainer.get_all_average_strategies()
            if len(avg_strat) > 0:
                nashconv = compute_exploitability(initial_state, avg_strat)
                self.exploitabilities.append((iteration, nashconv))
            else:
                nashconv = None
        else:
            nashconv = None

        return {
            'iter_time': iter_time,
            'loss': loss,
            'buffer_fill': buffer_fill,
            'strategy_change': strategy_change,
            'nashconv': nashconv,
        }

    def _compute_strategy_change(self, current, previous):
        """Compute L2 distance between strategy profiles."""
        total_diff = 0.0
        count = 0

        for info_state in current:
            if info_state in previous:
                diff = np.linalg.norm(current[info_state] - previous[info_state])
                total_diff += diff
                count += 1

        return total_diff / max(count, 1)

    def print_header(self):
        """Print monitoring header."""
        print()
        print("=" * 90)
        print("  Iter     Time    Buf%     Loss    ΔStrat    NashConv   Rate    Status")
        print("=" * 90)

    def print_stats(self, iteration, stats):
        """Print current iteration statistics."""
        status = self._get_status(stats)

        # Format NashConv
        if stats['nashconv'] is not None:
            nashconv_str = f"{stats['nashconv']:8.4f}"
        else:
            nashconv_str = "    —   "

        # Compute iteration rate
        if len(self.iteration_times) > 10:
            recent_times = self.iteration_times[-10:]
            rate = 1.0 / np.mean(recent_times) if np.mean(recent_times) > 0 else 0
        else:
            rate = 0

        print(f"  {iteration:5d}   "
              f"{stats['iter_time']*1000:5.1f}ms  "
              f"{stats['buffer_fill']:5.1f}%  "
              f"{stats['loss']:7.4f}  "
              f"{stats['strategy_change']:7.4f}  "
              f"{nashconv_str}  "
              f"{rate:5.1f}/s  "
              f"{status}")

    def _get_status(self, stats):
        """Get training status indicator."""
        if stats['buffer_fill'] < 100:
            return "Filling buffer..."
        elif stats['loss'] > 10:
            return "Training..."
        elif stats['strategy_change'] > 0.1:
            return "Converging..."
        elif stats['strategy_change'] > 0.01:
            return "Refining..."
        else:
            return "Converged?"

    def print_summary(self, total_time, final_nashconv=None):
        """Print training summary."""
        print()
        print("=" * 90)
        print("TRAINING SUMMARY")
        print("=" * 90)
        print()

        print("Performance:")
        print(f"  Total time:              {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"  Mean iter time:          {np.mean(self.iteration_times)*1000:.2f}ms")
        print(f"  Iterations/second:       {len(self.iteration_times) / total_time:.2f}")
        print()

        print("Convergence:")
        print(f"  Final loss:              {self.losses[-1]:.4f}")
        print(f"  Final strategy change:   {self.strategy_changes[-1] if self.strategy_changes else 0:.4f}")
        if final_nashconv is not None:
            print(f"  Final NashConv:          {final_nashconv:.4f}")
        print()

        if len(self.exploitabilities) > 1:
            print("Exploitability Progress:")
            for iter_num, nash in self.exploitabilities[-5:]:
                print(f"  Iteration {iter_num:5d}:         {nash:.4f}")
            print()

            # Compute improvement
            initial_nash = self.exploitabilities[0][1]
            final_nash = self.exploitabilities[-1][1]
            if initial_nash > 0:
                improvement = (initial_nash - final_nash) / initial_nash * 100
                print(f"Overall improvement:       {improvement:.1f}%")
                print(f"  From: {initial_nash:.4f}")
                print(f"  To:   {final_nash:.4f}")
                print()


def train(
    game_name='leduc',
    num_iterations=2000,
    eval_every=500,
    algorithm='pdcfr',
    seed=42,
    **kwargs
):
    """Train Deep PDCFR+ with monitoring."""

    print("=" * 90)
    print(f"DEEP PDCFR+ TRAINING: {game_name.upper()} POKER")
    print("=" * 90)
    print()

    # Initialize game
    if game_name == 'leduc':
        initial_state = LeducPoker()
        encoder = LeducEncoder()
        input_size = 26
        output_size = 2
    elif game_name == 'kuhn':
        initial_state = KuhnPoker()
        encoder = KuhnEncoder()
        input_size = 10
        output_size = 2
    else:
        raise ValueError(f"Unknown game: {game_name}")

    # Configure algorithm
    if algorithm == 'pdcfr':
        regret_scheduler = PDCFRScheduler(alpha=2.0, beta=0.5)
        strategy_scheduler = LinearScheduler()
        algo_desc = "Deep PDCFR+ (α=2.0, β=0.5, linear averaging)"
    elif algorithm == 'vanilla':
        regret_scheduler = UniformScheduler()
        strategy_scheduler = UniformScheduler()
        algo_desc = "Vanilla Deep CFR (no discounting)"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print("Configuration:")
    print(f"  Game:                {game_name.capitalize()} Poker")
    print(f"  Algorithm:           {algo_desc}")
    print(f"  Iterations:          {num_iterations:,}")
    print(f"  Eval frequency:      Every {eval_every} iterations")
    print(f"  Seed:                {seed}")
    print()

    # Create trainer
    trainer = DeepCFRTrainer(
        initial_state=initial_state,
        encoder=encoder,
        input_size=input_size,
        output_size=output_size,
        hidden_size=kwargs.get('hidden_size', 128),
        num_hidden_layers=kwargs.get('num_hidden', 3),
        buffer_capacity=kwargs.get('buffer_size', 10000),
        learning_rate=kwargs.get('lr', 0.001),
        batch_size=kwargs.get('batch_size', 128),
        seed=seed,
        device=kwargs.get('device', 'cpu'),
        regret_scheduler=regret_scheduler,
        strategy_scheduler=strategy_scheduler,
    )

    print("Network:")
    print(f"  Architecture:        {input_size} → {kwargs.get('num_hidden', 3)}×{kwargs.get('hidden_size', 128)} → {output_size}")
    print(f"  Buffer capacity:     {kwargs.get('buffer_size', 10000):,}")
    print(f"  Batch size:          {kwargs.get('batch_size', 128)}")
    print(f"  Learning rate:       {kwargs.get('lr', 0.001)}")

    # Training loop with monitoring
    monitor = TrainingMonitor(eval_exploitability_every=eval_every)
    monitor.print_header()

    start_time = time.time()

    for i in range(1, num_iterations + 1):
        iter_start = time.time()

        # Run iteration
        metrics = trainer.run_iteration()
        metrics['iter_time'] = time.time() - iter_start

        # Log and print
        stats = monitor.log_iteration(i, metrics, trainer, initial_state)

        # Print every 100 iterations or when evaluating exploitability
        if i % 100 == 0 or stats['nashconv'] is not None or i == 1:
            monitor.print_stats(i, stats)

    total_time = time.time() - start_time

    # Final evaluation
    print()
    print("Computing final exploitability...")
    final_strat = trainer.get_all_average_strategies()
    final_nashconv = compute_exploitability(initial_state, final_strat) if final_strat else None

    # Print summary
    monitor.print_summary(total_time, final_nashconv)

    return trainer, monitor


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Deep PDCFR+ with monitoring')

    parser.add_argument('--game', type=str, default='leduc',
                       choices=['kuhn', 'leduc'],
                       help='Poker game variant')
    parser.add_argument('--iterations', type=int, default=2000,
                       help='Number of training iterations')
    parser.add_argument('--eval-every', type=int, default=500,
                       help='Evaluate exploitability every N iterations')
    parser.add_argument('--algorithm', type=str, default='pdcfr',
                       choices=['pdcfr', 'vanilla'],
                       help='Algorithm variant')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--num-hidden', type=int, default=3,
                       help='Number of hidden layers')
    parser.add_argument('--buffer-size', type=int, default=10000,
                       help='Reservoir buffer capacity')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Training batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    try:
        trainer, monitor = train(
            game_name=args.game,
            num_iterations=args.iterations,
            eval_every=args.eval_every,
            algorithm=args.algorithm,
            seed=args.seed,
            hidden_size=args.hidden_size,
            num_hidden=args.num_hidden,
            buffer_size=args.buffer_size,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )

        print()
        print("✅ Training complete!")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
