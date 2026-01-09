#!/usr/bin/env python3
"""Real-Time Training Dashboard - Monitor River 5M Run.

Tails production_5m.log and plots live metrics every 30 seconds.

Critical Metrics:
1. Loss Curve: Must decrease (not explode)
2. Buffer Fill Rate: Track when we hit 100%
3. Strategy Size: Should grow linearly (millions expected)
4. Speed: Iterations per second

Usage:
    python scripts/monitor_training.py
"""

import sys
import re
import time
from pathlib import Path
from collections import deque
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TrainingMonitor:
    """Real-time monitor for production_5m.log."""

    def __init__(self, log_path: Path, history_size: int = 1000):
        self.log_path = log_path
        self.history_size = history_size

        # Metric histories (deque for efficient rolling window)
        self.iterations = deque(maxlen=history_size)
        self.losses = deque(maxlen=history_size)
        self.buffer_fills = deque(maxlen=history_size)
        self.speeds = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

        # Checkpoint data
        self.checkpoints = []  # [(iteration, strategy_size)]

        # Current position in log file
        self.log_position = 0

        # Start time
        self.start_time = time.time()

    def parse_log_line(self, line: str) -> Optional[dict]:
        """Parse a log line for metrics.

        Example lines:
            Iter  5000 | Loss: 0.0007 | Buffer:  24741/2000000 (  1.2%) | 221.5 it/s
            Strategy size: 28,890 information states
        """
        metrics = {}

        # Parse iteration line
        iter_match = re.search(r'Iter\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Buffer:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)\s+\|\s+([\d.]+)\s+it/s', line)
        if iter_match:
            metrics['iteration'] = int(iter_match.group(1))
            metrics['loss'] = float(iter_match.group(2))
            metrics['buffer_current'] = int(iter_match.group(3))
            metrics['buffer_capacity'] = int(iter_match.group(4))
            metrics['buffer_fill'] = float(iter_match.group(5))
            metrics['speed'] = float(iter_match.group(6))
            return metrics

        # Parse strategy size (at checkpoints)
        strategy_match = re.search(r'Strategy size:\s+([\d,]+)\s+information states', line)
        if strategy_match:
            strategy_size = int(strategy_match.group(1).replace(',', ''))
            metrics['strategy_size'] = strategy_size
            return metrics

        return None

    def tail_log(self) -> list[dict]:
        """Tail log file and extract new metrics.

        Returns:
            List of metric dictionaries
        """
        if not self.log_path.exists():
            return []

        metrics = []

        with open(self.log_path, 'r') as f:
            # Seek to last position
            f.seek(self.log_position)

            # Read new lines
            for line in f:
                parsed = self.parse_log_line(line)
                if parsed:
                    metrics.append(parsed)

            # Update position
            self.log_position = f.tell()

        return metrics

    def update(self):
        """Update metrics from log."""
        new_metrics = self.tail_log()

        for m in new_metrics:
            if 'iteration' in m:
                self.iterations.append(m['iteration'])
                self.losses.append(m['loss'])
                self.buffer_fills.append(m['buffer_fill'])
                self.speeds.append(m['speed'])
                self.timestamps.append(time.time() - self.start_time)

            if 'strategy_size' in m:
                # Associate with last iteration
                if self.iterations:
                    last_iter = self.iterations[-1]
                    self.checkpoints.append((last_iter, m['strategy_size']))

    def get_eta(self) -> Tuple[float, str]:
        """Estimate time to completion.

        Returns:
            (seconds_remaining, human_readable)
        """
        if len(self.iterations) < 2 or len(self.speeds) < 10:
            return 0, "Calculating..."

        # Use average speed over last 100 samples
        recent_speed = np.mean(list(self.speeds)[-100:])
        current_iter = self.iterations[-1]
        remaining = 5_000_000 - current_iter

        if recent_speed <= 0:
            return 0, "Unknown"

        seconds_remaining = remaining / recent_speed

        # Convert to human readable
        hours = int(seconds_remaining // 3600)
        minutes = int((seconds_remaining % 3600) // 60)

        return seconds_remaining, f"{hours}h {minutes}m"

    def get_stats(self) -> dict:
        """Get current statistics."""
        if not self.iterations:
            return {
                'current_iter': 0,
                'total_iters': 5_000_000,
                'progress_pct': 0.0,
                'current_loss': 0.0,
                'buffer_fill': 0.0,
                'speed': 0.0,
                'eta': 'Calculating...',
            }

        _, eta_str = self.get_eta()

        return {
            'current_iter': self.iterations[-1],
            'total_iters': 5_000_000,
            'progress_pct': (self.iterations[-1] / 5_000_000) * 100,
            'current_loss': self.losses[-1],
            'buffer_fill': self.buffer_fills[-1],
            'speed': self.speeds[-1],
            'eta': eta_str,
            'checkpoints': len(self.checkpoints),
        }


class LiveDashboard:
    """Live matplotlib dashboard."""

    def __init__(self, monitor: TrainingMonitor, update_interval: int = 30):
        self.monitor = monitor
        self.update_interval = update_interval

        # Create figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('River 5M Training - Live Monitor', fontsize=16, fontweight='bold')

        # Subplot references
        self.ax_loss = self.axes[0, 0]
        self.ax_buffer = self.axes[0, 1]
        self.ax_speed = self.axes[1, 0]
        self.ax_strategy = self.axes[1, 1]

        # Initialize plots
        self._setup_plots()

    def _setup_plots(self):
        """Setup plot layouts."""
        # Loss plot
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Curve (MUST DECREASE)')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.set_yscale('log')

        # Buffer plot
        self.ax_buffer.set_xlabel('Iteration')
        self.ax_buffer.set_ylabel('Buffer Fill (%)')
        self.ax_buffer.set_title('Buffer Fill Rate')
        self.ax_buffer.grid(True, alpha=0.3)
        self.ax_buffer.set_ylim(0, 105)

        # Speed plot
        self.ax_speed.set_xlabel('Time (seconds)')
        self.ax_speed.set_ylabel('Iterations/sec')
        self.ax_speed.set_title('Training Speed')
        self.ax_speed.grid(True, alpha=0.3)

        # Strategy size plot
        self.ax_strategy.set_xlabel('Iteration')
        self.ax_strategy.set_ylabel('Strategy Size (states)')
        self.ax_strategy.set_title('Strategy Growth (expect millions)')
        self.ax_strategy.grid(True, alpha=0.3)

    def update_plots(self, frame):
        """Update plots with new data."""
        # Fetch new data
        self.monitor.update()

        if not self.monitor.iterations:
            return

        # Convert to numpy arrays
        iters = np.array(list(self.monitor.iterations))
        losses = np.array(list(self.monitor.losses))
        buffer_fills = np.array(list(self.monitor.buffer_fills))
        speeds = np.array(list(self.monitor.speeds))
        timestamps = np.array(list(self.monitor.timestamps))

        # Clear axes
        self.ax_loss.clear()
        self.ax_buffer.clear()
        self.ax_speed.clear()
        self.ax_strategy.clear()

        # Re-setup (labels get cleared)
        self._setup_plots()

        # Plot loss
        self.ax_loss.plot(iters, losses, 'b-', linewidth=1.5, alpha=0.7)
        self.ax_loss.set_yscale('log')

        # Add horizontal line at loss=0.05 (failure threshold)
        self.ax_loss.axhline(y=0.05, color='r', linestyle='--', label='Failure threshold (0.05)')
        self.ax_loss.legend()

        # Plot buffer fill
        self.ax_buffer.plot(iters, buffer_fills, 'g-', linewidth=1.5, alpha=0.7)
        self.ax_buffer.axhline(y=100, color='r', linestyle='--', label='100% full')
        self.ax_buffer.legend()

        # Plot speed
        self.ax_speed.plot(timestamps, speeds, 'orange', linewidth=1.5, alpha=0.7)

        # Plot strategy size (from checkpoints)
        if self.monitor.checkpoints:
            cp_iters = [c[0] for c in self.monitor.checkpoints]
            cp_sizes = [c[1] for c in self.monitor.checkpoints]
            self.ax_strategy.plot(cp_iters, cp_sizes, 'mo-', linewidth=2, markersize=8, label='Checkpoints')
            self.ax_strategy.legend()

        # Add stats text
        stats = self.monitor.get_stats()
        stats_text = (
            f"Progress: {stats['progress_pct']:.1f}% ({stats['current_iter']:,} / {stats['total_iters']:,})\n"
            f"Loss: {stats['current_loss']:.4f} | Buffer: {stats['buffer_fill']:.1f}% | Speed: {stats['speed']:.1f} it/s\n"
            f"ETA: {stats['eta']} | Checkpoints: {stats['checkpoints']}"
        )

        self.fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    def run(self):
        """Start live dashboard."""
        print("="*80)
        print("RIVER 5M TRAINING - LIVE DASHBOARD")
        print("="*80)
        print()
        print(f"Monitoring: {self.monitor.log_path}")
        print(f"Update interval: {self.update_interval} seconds")
        print()
        print("Critical Alerts:")
        print("  🔴 Loss > 0.05: Training failing")
        print("  🟡 Speed < 100 it/s: Bottleneck")
        print("  🟢 Strategy size growing: Healthy")
        print()
        print("Press Ctrl+C to exit")
        print("="*80)
        print()

        # Start animation
        ani = FuncAnimation(
            self.fig,
            self.update_plots,
            interval=self.update_interval * 1000,  # Convert to milliseconds
            cache_frame_data=False
        )

        plt.show()


def main():
    """Run live dashboard."""
    # Paths
    project_root = Path(__file__).parent.parent
    log_path = project_root / "production_5m.log"

    if not log_path.exists():
        print(f"❌ ERROR: Log file not found: {log_path}")
        print()
        print("Make sure the training is running:")
        print("  uv run python scripts/train_river.py --use-rust --iterations 5000000 ...")
        return 1

    # Create monitor
    monitor = TrainingMonitor(log_path, history_size=5000)

    # Create dashboard
    dashboard = LiveDashboard(monitor, update_interval=30)

    # Run
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard closed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
