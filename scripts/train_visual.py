#!/usr/bin/env python
"""Visual Training Dashboard with Real-Time Metrics

Uses rich library for beautiful terminal UI showing:
- Live throughput graph
- Training progress
- GPU/CPU utilization
- Loss and Nash Conv metrics

Usage:
    uv run python scripts/train_visual.py
    uv run python scripts/train_visual.py --epochs 100 --traversals 50000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import shutil
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import box

import aion26_rust
from aion26.memory.disk import TrajectoryDataset, WeightedEpochSampler

# Configuration
DEFAULT_CONFIG = {
    "num_epochs": 500,
    "traversals_per_epoch": 100_000,
    "batch_size": 4096,
    "max_train_steps": 1000,
    "learning_rate": 1e-3,
    "recency_alpha": 1.5,
    "hidden_size": 256,
    "state_dim": 136,
    "target_dim": 4,
    "data_dir": "data/visual_train",
    "num_workers": 8,
}

console = Console()


class AdvantageNetwork(nn.Module):
    def __init__(self, input_size=136, hidden_size=256, output_size=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        return self.network(x)


class MetricsTracker:
    def __init__(self, window_size=50):
        self.throughput_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.total_samples = 0
        self.total_time = 0
        self.epoch = 0
        self.current_loss = 0.0
        self.current_throughput = 0
        self.peak_throughput = 0
        self.start_time = time.time()

    def update(self, samples, gen_time, loss):
        throughput = samples / gen_time if gen_time > 0 else 0
        self.throughput_history.append(throughput)
        self.loss_history.append(loss)
        self.total_samples += samples
        self.total_time += gen_time
        self.current_loss = loss
        self.current_throughput = throughput
        self.peak_throughput = max(self.peak_throughput, throughput)

    def get_sparkline(self, data, width=20):
        """Generate ASCII sparkline for data."""
        if not data:
            return "─" * width

        values = list(data)
        if len(values) < 2:
            return "─" * width

        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return "─" * width

        chars = "▁▂▃▄▅▆▇█"
        result = []

        # Resample to fit width
        step = max(1, len(values) // width)
        sampled = values[::step][:width]

        for v in sampled:
            idx = int((v - min_val) / (max_val - min_val) * (len(chars) - 1))
            result.append(chars[idx])

        return "".join(result).ljust(width, "─")


def make_dashboard(metrics: MetricsTracker, config: dict, device: str) -> Layout:
    """Create the dashboard layout."""
    layout = Layout()

    # Header
    header = Table.grid(expand=True)
    header.add_column(justify="center", ratio=1)
    header.add_row(
        Text("AION-26 DEEP PDCFR+ TRAINER", style="bold magenta", justify="center")
    )

    # Stats table
    stats = Table(box=box.ROUNDED, expand=True, show_header=False)
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green", justify="right")
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green", justify="right")

    elapsed = time.time() - metrics.start_time
    avg_throughput = metrics.total_samples / elapsed if elapsed > 0 else 0

    stats.add_row(
        "Epoch", f"{metrics.epoch}",
        "Device", device.upper(),
    )
    stats.add_row(
        "Total Samples", f"{metrics.total_samples:,}",
        "Workers", f"{config['num_workers']}",
    )
    stats.add_row(
        "Throughput", f"{metrics.current_throughput:,.0f}/s",
        "Peak", f"{metrics.peak_throughput:,.0f}/s",
    )
    stats.add_row(
        "Avg Throughput", f"{avg_throughput:,.0f}/s",
        "Loss", f"{metrics.current_loss:.6f}",
    )
    stats.add_row(
        "Elapsed", f"{elapsed:.1f}s",
        "ETA", f"{(config['num_epochs'] - metrics.epoch) * elapsed / max(metrics.epoch, 1):.0f}s",
    )

    # Throughput chart
    sparkline = metrics.get_sparkline(metrics.throughput_history, width=40)

    chart_text = Text()
    chart_text.append("Throughput: ", style="cyan")
    chart_text.append(sparkline, style="green")
    chart_text.append(f" {metrics.current_throughput:,.0f}/s", style="bold green")

    # Loss chart
    loss_sparkline = metrics.get_sparkline(metrics.loss_history, width=40)
    loss_text = Text()
    loss_text.append("Loss:       ", style="cyan")
    loss_text.append(loss_sparkline, style="yellow")
    loss_text.append(f" {metrics.current_loss:.6f}", style="bold yellow")

    charts = Panel(
        Text.assemble(chart_text, "\n", loss_text),
        title="Real-Time Metrics",
        border_style="blue"
    )

    layout.split_column(
        Layout(Panel(header, border_style="magenta"), size=3),
        Layout(Panel(stats, title="Status", border_style="green"), size=9),
        Layout(charts, size=5),
    )

    return layout


def train(config: dict):
    """Run training with visual dashboard."""

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Clean data dir
    data_dir = Path(config["data_dir"])
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Initialize
    trainer = aion26_rust.ParallelTrainer(
        data_dir=str(data_dir),
        query_buffer_size=4096,
        num_workers=config["num_workers"],
    )

    network = AdvantageNetwork(
        input_size=config["state_dim"],
        hidden_size=config["hidden_size"],
        output_size=config["target_dim"],
    ).to(device)

    optimizer = optim.Adam(network.parameters(), lr=config["learning_rate"])
    metrics = MetricsTracker()
    dataset = TrajectoryDataset(str(data_dir))

    console.print(f"\n[bold green]Starting training on {device.type.upper()}[/bold green]")
    console.print(f"[dim]Epochs: {config['num_epochs']} | Traversals: {config['traversals_per_epoch']:,} | Workers: {config['num_workers']}[/dim]\n")

    # Check if we're in a TTY for live display
    use_live = sys.stdout.isatty()

    live_context = Live(make_dashboard(metrics, config, device.type), refresh_per_second=4, console=console) if use_live else None

    if live_context:
        live_context.start()

    try:
        for epoch in range(config["num_epochs"]):
            metrics.epoch = epoch

            # === GENERATION ===
            trainer.start_epoch(epoch)
            gen_start = time.time()

            network.eval()
            predictions = None
            samples = 0

            while True:
                result = trainer.step(predictions, num_traversals=config["traversals_per_epoch"] if predictions is None else None)

                if result.is_finished():
                    samples = result.samples()
                    break

                query_buffer = trainer.get_query_buffer()
                queries_tensor = torch.from_numpy(query_buffer).to(device)

                with torch.no_grad():
                    predictions_tensor = network(queries_tensor)

                predictions = predictions_tensor.cpu().numpy()

            gen_time = time.time() - gen_start
            trainer.end_epoch()

            # === TRAINING ===
            dataset.refresh()

            if len(dataset) > 0:
                network.train()
                sampler = WeightedEpochSampler(dataset, batch_size=config["batch_size"], alpha=config["recency_alpha"])

                total_loss = 0.0
                steps = 0

                for batch_indices in sampler:
                    if steps >= config["max_train_steps"]:
                        break

                    indices = np.array(batch_indices)
                    states, targets = dataset.get_batch_from_indices(indices)
                    states = states.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    preds = network(states)
                    loss = nn.functional.mse_loss(preds, targets)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    steps += 1

                avg_loss = total_loss / steps if steps > 0 else 0.0
            else:
                avg_loss = 0.0

            # Update metrics
            metrics.update(samples, gen_time, avg_loss)

            # Update display
            if live_context:
                live_context.update(make_dashboard(metrics, config, device.type))
            elif epoch % 10 == 0:
                # Print progress for non-TTY mode
                print(f"Epoch {epoch}: {metrics.current_throughput:,.0f} samples/s | Loss: {avg_loss:.6f}")

            # Checkpoint
            if (epoch + 1) % 100 == 0:
                torch.save({
                    "epoch": epoch,
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "total_samples": metrics.total_samples,
                }, f"visual_checkpoint_e{epoch+1}.pt")

    finally:
        if live_context:
            live_context.stop()

    # Final save
    torch.save({
        "epoch": config["num_epochs"] - 1,
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "total_samples": metrics.total_samples,
    }, "visual_model_final.pt")

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"Total samples: {metrics.total_samples:,}")
    console.print(f"Peak throughput: {metrics.peak_throughput:,.0f} samples/s")
    console.print(f"Model saved to visual_model_final.pt")


def main():
    parser = argparse.ArgumentParser(description="Visual Training Dashboard")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--traversals", type=int, default=DEFAULT_CONFIG["traversals_per_epoch"])
    parser.add_argument("--workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["num_epochs"] = args.epochs
    config["traversals_per_epoch"] = args.traversals
    config["num_workers"] = args.workers
    config["learning_rate"] = args.lr

    train(config)


if __name__ == "__main__":
    main()
