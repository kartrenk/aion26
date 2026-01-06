"""Visualize profiling results for Deep PDCFR+ training.

Creates plots showing:
1. Time distribution (pie chart)
2. Iteration time over iterations (line plot)
3. Before/after MCCFR comparison (bar chart)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from aion26.games.leduc import LeducPoker
from aion26.deep_cfr.networks import LeducEncoder
from aion26.learner.deep_cfr import DeepCFRTrainer
from aion26.learner.discounting import PDCFRScheduler, LinearScheduler


def profile_training(num_iterations=200):
    """Profile training and collect timing data."""
    print("Profiling training...")

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

    # Collect timing data
    traversal_times = []
    network_times = []
    target_times = []
    iteration_times = []

    for i in range(1, num_iterations + 1):
        iter_start = time.time()

        # Traversal
        trav_start = time.time()
        trainer.traverse(trainer.initial_state, 0, 1.0, 1.0)
        trainer.traverse(trainer.initial_state, 1, 1.0, 1.0)
        traversal_times.append(time.time() - trav_start)

        trainer.iteration += 1

        # Network training
        if i % trainer.train_every == 0:
            net_start = time.time()
            trainer.train_network()
            network_times.append(time.time() - net_start)

        # Target update
        if i % trainer.target_update_every == 0:
            target_start = time.time()
            trainer.update_target_network()
            target_times.append(time.time() - target_start)

        iteration_times.append(time.time() - iter_start)

    return {
        'traversal': np.array(traversal_times),
        'network': np.array(network_times),
        'target': np.array(target_times),
        'iteration': np.array(iteration_times),
    }


def create_visualizations(timing_data, output_dir='plots'):
    """Create profiling visualization plots."""
    Path(output_dir).mkdir(exist_ok=True)

    # Figure 1: Time distribution pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    total_traversal = timing_data['traversal'].sum()
    total_network = timing_data['network'].sum()
    total_target = timing_data['target'].sum()
    total_iteration = timing_data['iteration'].sum()
    other = total_iteration - (total_traversal + total_network + total_target)

    sizes = [total_traversal, total_network, total_target, other]
    labels = ['CFR Traversal', 'Network Training', 'Target Updates', 'Other']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.1, 0, 0, 0)  # Explode the traversal slice

    ax.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
    ax.set_title('Time Distribution in Deep PDCFR+ Training\n(External Sampling MCCFR)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/time_distribution.png', dpi=150)
    print(f"Saved: {output_dir}/time_distribution.png")
    plt.close()

    # Figure 2: Iteration time over iterations
    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = np.arange(1, len(timing_data['iteration']) + 1)
    iter_times_ms = timing_data['iteration'] * 1000

    ax.plot(iterations, iter_times_ms, 'b-', alpha=0.6, linewidth=0.5, label='Per iteration')

    # Add rolling average
    window = 20
    if len(iter_times_ms) >= window:
        rolling_avg = np.convolve(iter_times_ms, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window, len(iterations) + 1), rolling_avg,
               'r-', linewidth=2, label=f'{window}-iteration moving average')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Iteration Time During Training\n(External Sampling MCCFR on Leduc Poker)',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/iteration_time.png', dpi=150)
    print(f"Saved: {output_dir}/iteration_time.png")
    plt.close()

    # Figure 3: Component timing comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    components = ['Traversal', 'Network Train', 'Target Update']
    mean_times = [
        timing_data['traversal'].mean() * 1000,
        timing_data['network'].mean() * 1000,
        timing_data['target'].mean() * 1000
    ]
    std_times = [
        timing_data['traversal'].std() * 1000,
        timing_data['network'].std() * 1000,
        timing_data['target'].std() * 1000
    ]

    x = np.arange(len(components))
    bars = ax.bar(x, mean_times, yerr=std_times, capsize=5,
                  color=['#ff9999', '#66b3ff', '#99ff99'],
                  edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Mean Component Time per Iteration\n(Error bars show standard deviation)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_times)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}ms',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_timing.png', dpi=150)
    print(f"Saved: {output_dir}/component_timing.png")
    plt.close()

    # Figure 4: Before/After MCCFR comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data from profiling
    before_traversal = 182.22  # Full traversal baseline
    after_traversal = timing_data['traversal'].mean() * 1000

    categories = ['Before MCCFR\n(Full Traversal)', 'After MCCFR\n(External Sampling)']
    values = [before_traversal, after_traversal]
    colors_comp = ['#ff6b6b', '#51cf66']

    bars = ax.bar(categories, values, color=colors_comp, edgecolor='black', linewidth=2)

    # Add speedup annotation
    speedup = before_traversal / after_traversal
    ax.annotate(f'{speedup:.1f}× speedup',
               xy=(0.5, max(values) * 0.9),
               ha='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_ylabel('Traversal Time (ms)', fontsize=12)
    ax.set_title('Impact of External Sampling MCCFR\n(Leduc Poker, mean traversal time per iteration)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}ms',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/mccfr_comparison.png', dpi=150)
    print(f"Saved: {output_dir}/mccfr_comparison.png")
    plt.close()


def print_summary(timing_data):
    """Print profiling summary statistics."""
    print()
    print("=" * 70)
    print("PROFILING SUMMARY")
    print("=" * 70)
    print()

    print("Component Timing (mean ± std):")
    print(f"  Traversal:       {timing_data['traversal'].mean()*1000:6.2f} ± {timing_data['traversal'].std()*1000:5.2f} ms")
    print(f"  Network Train:   {timing_data['network'].mean()*1000:6.2f} ± {timing_data['network'].std()*1000:5.2f} ms")
    print(f"  Target Update:   {timing_data['target'].mean()*1000:6.2f} ± {timing_data['target'].std()*1000:5.2f} ms")
    print(f"  Total Iteration: {timing_data['iteration'].mean()*1000:6.2f} ± {timing_data['iteration'].std()*1000:5.2f} ms")
    print()

    total = timing_data['iteration'].sum()
    trav_pct = timing_data['traversal'].sum() / total * 100
    net_pct = timing_data['network'].sum() / total * 100
    target_pct = timing_data['target'].sum() / total * 100

    print("Time Distribution:")
    print(f"  Traversal:       {trav_pct:5.1f}%")
    print(f"  Network Train:   {net_pct:5.1f}%")
    print(f"  Target Update:   {target_pct:5.1f}%")
    print(f"  Other:           {100 - trav_pct - net_pct - target_pct:5.1f}%")
    print()

    print("Throughput:")
    print(f"  Iterations/sec:  {len(timing_data['iteration']) / total:.2f}")
    print()


def main():
    """Main entry point."""
    print("=" * 70)
    print("PROFILING VISUALIZATION")
    print("=" * 70)
    print()

    # Profile training
    timing_data = profile_training(num_iterations=200)

    # Print summary
    print_summary(timing_data)

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(timing_data)

    print()
    print("=" * 70)
    print("✅ Profiling complete! Check plots/ directory for visualizations.")
    print("=" * 70)


if __name__ == '__main__':
    main()
