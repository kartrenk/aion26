#!/usr/bin/env python
"""Phase 6: Rust Driver + Neural Network + Disk Streaming.

This is the "Architecture of Victory" - the complete Deep CFR system:

1. GENERATION PHASE (Rust + GPU Inference)
   - Rust traverses game tree, pauses when it needs strategy predictions
   - Python runs batch inference on GPU
   - Rust resumes with predictions, writes samples to disk

2. TRAINING PHASE (Disk Streaming)
   - Load all historical epochs via memory-mapped files
   - Sample with recency weighting (newer epochs get more weight)
   - Train network on batches

Key Insight: Deep CFR NEEDS all historical samples. Reservoir sampling
causes catastrophic forgetting. Solution: Store everything on disk.

Usage:
    uv run python scripts/train_phase6.py --epochs 100 --traversals-per-epoch 1000
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import aion26_rust
from aion26.memory.disk import TrajectoryDataset, WeightedEpochSampler

# Constants from Rust (must match io.rs)
STATE_DIM = 136
TARGET_DIM = 4


class AdvantageNetwork(nn.Module):
    """Neural network for predicting action advantages.

    Input: State encoding (136 dims)
    Output: Advantage for each action (4 dims)

    Architecture: Simple MLP with 3 hidden layers.
    """

    def __init__(self, input_size: int = STATE_DIM, hidden_size: int = 256, output_size: int = TARGET_DIM):
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

        # Initialize output layer with small weights for uniform initial strategy
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_step(
    network: nn.Module,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Execute a single training step.

    Args:
        network: The advantage network
        optimizer: Optimizer for the network
        states: Batch of state encodings (batch_size, STATE_DIM)
        targets: Batch of regret targets (batch_size, TARGET_DIM)

    Returns:
        Training loss value
    """
    network.train()
    optimizer.zero_grad()

    predictions = network(states)
    loss = nn.functional.mse_loss(predictions, targets)

    loss.backward()
    optimizer.step()

    return loss.item()


def run_epoch_generation(
    trainer: aion26_rust.RustTrainer,
    network: nn.Module,
    device: torch.device,
    num_traversals: int,
    verbose: bool = False,
) -> tuple[int, int, float, dict]:
    """Run the generation phase for one epoch.

    This is the cooperative inference loop:
    1. Rust traverses game tree
    2. When it needs predictions, returns RequestInference
    3. Python runs batch inference on GPU
    4. Loop back with predictions until Finished

    Args:
        trainer: The Rust trainer with active epoch
        network: The advantage network for inference
        device: PyTorch device (CPU or CUDA)
        num_traversals: Number of game traversals to complete
        verbose: Print detailed timing breakdown

    Returns:
        Tuple of (samples_written, inference_calls, generation_time, timing_breakdown)
    """
    network.eval()

    start_time = time.time()
    inference_calls = 0
    predictions = None

    # Timing breakdown
    rust_time = 0.0
    inference_time = 0.0
    transfer_time = 0.0
    total_queries = 0

    while True:
        t0 = time.time()
        result = trainer.step(predictions, num_traversals=num_traversals if predictions is None else None)
        rust_time += time.time() - t0

        if result.is_finished():
            break

        elif result.is_request_inference():
            # Get query buffer from Rust (zero-copy view)
            t1 = time.time()
            query_buffer = trainer.get_query_buffer()
            batch_size = query_buffer.shape[0]
            total_queries += batch_size

            # Transfer to GPU
            queries_tensor = torch.from_numpy(query_buffer).to(device)
            transfer_time += time.time() - t1

            # Run batch inference on GPU
            t2 = time.time()
            with torch.no_grad():
                predictions_tensor = network(queries_tensor)
            inference_time += time.time() - t2

            # Convert back to NumPy for Rust
            t3 = time.time()
            predictions = predictions_tensor.cpu().numpy()
            transfer_time += time.time() - t3

            inference_calls += 1

    generation_time = time.time() - start_time
    samples_written = result.count()  # count() is a method

    timing = {
        'rust_time': rust_time,
        'inference_time': inference_time,
        'transfer_time': transfer_time,
        'total_queries': total_queries,
        'inference_calls': inference_calls,
        'queries_per_call': total_queries / inference_calls if inference_calls > 0 else 0,
    }

    if verbose:
        avg_batch = total_queries // inference_calls if inference_calls > 0 else 0
        print(f"    [Timing] Rust: {rust_time:.1f}s ({100*rust_time/generation_time:.0f}%) | "
              f"GPU: {inference_time:.1f}s ({100*inference_time/generation_time:.0f}%) | "
              f"Transfer: {transfer_time:.1f}s ({100*transfer_time/generation_time:.0f}%) | "
              f"Calls: {inference_calls} | Batch: {avg_batch}", flush=True)

    return samples_written, inference_calls, generation_time, timing


def run_epoch_training(
    network: nn.Module,
    optimizer: optim.Optimizer,
    dataset: TrajectoryDataset,
    device: torch.device,
    batch_size: int,
    max_steps: int,
    alpha: float,
) -> tuple[float, int, float]:
    """Run the training phase for one epoch.

    Samples from all historical epochs with recency weighting,
    then trains the network.

    Args:
        network: The advantage network to train
        optimizer: Optimizer for the network
        dataset: Dataset containing all historical samples
        device: PyTorch device
        batch_size: Training batch size
        max_steps: Maximum training steps per epoch
        alpha: Recency weighting parameter (0.5 = 2x weight for newest epoch)

    Returns:
        Tuple of (avg_loss, steps_taken, training_time)
    """
    if len(dataset) == 0:
        return 0.0, 0, 0.0

    start_time = time.time()
    sampler = WeightedEpochSampler(dataset, batch_size=batch_size, alpha=alpha)

    total_loss = 0.0
    steps = 0

    for batch_indices in sampler:
        if steps >= max_steps:
            break

        # Get batch from dataset
        indices = np.array(batch_indices)
        states, targets = dataset.get_batch_from_indices(indices)

        # Move to device
        states = states.to(device)
        targets = targets.to(device)

        # Training step
        loss = train_step(network, optimizer, states, targets)
        total_loss += loss
        steps += 1

    training_time = time.time() - start_time
    avg_loss = total_loss / steps if steps > 0 else 0.0

    return avg_loss, steps, training_time


def compute_vintage_loss_on_epoch0(
    network: nn.Module,
    dataset: TrajectoryDataset,
    device: torch.device,
    epoch0_sample_count: int,
    num_batches: int = 10,
    batch_size: int = 1024,
) -> float:
    """Compute validation loss on vintage (epoch 0) samples.

    This is the "forgetting detector" - if this loss spikes,
    the network is forgetting early lessons.

    Args:
        network: The advantage network
        dataset: Full dataset (we'll sample from first epoch0_sample_count indices)
        device: PyTorch device
        epoch0_sample_count: Number of samples in epoch 0
        num_batches: Number of batches to evaluate
        batch_size: Samples per batch

    Returns:
        Average MSE loss on vintage (epoch 0) samples
    """
    if epoch0_sample_count == 0 or len(dataset) == 0:
        return 0.0

    network.eval()
    total_loss = 0.0
    batches_evaluated = 0

    with torch.no_grad():
        for _ in range(num_batches):
            # Random sample ONLY from epoch 0 indices (0 to epoch0_sample_count-1)
            indices = np.random.randint(0, epoch0_sample_count, size=min(batch_size, epoch0_sample_count))
            states, targets = dataset.get_batch_from_indices(indices)

            states = states.to(device)
            targets = targets.to(device)

            predictions = network(states)
            loss = nn.functional.mse_loss(predictions, targets)
            total_loss += loss.item()
            batches_evaluated += 1

    return total_loss / batches_evaluated if batches_evaluated > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Deep CFR with Rust Driver + Disk Streaming")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--traversals-per-epoch", type=int, default=50000, help="Game traversals per epoch")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--max-train-steps", type=int, default=1000, help="Max training steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Recency weighting (0.5 = 2x for newest)")
    parser.add_argument("--data-dir", type=str, default="data/phase6", help="Directory for epoch files")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--query-buffer-size", type=int, default=4096, help="Rust query buffer size")
    parser.add_argument("--clean", action="store_true", help="Delete existing data directory")
    parser.add_argument("--vintage-interval", type=int, default=10, help="Epochs between vintage validation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed timing breakdown per epoch")
    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Phase 6: Rust Driver + Disk Streaming")
    print(f"=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Traversals/epoch: {args.traversals_per_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max train steps/epoch: {args.max_train_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Recency alpha: {args.alpha}")
    print(f"Data directory: {args.data_dir}")
    print(f"=" * 60, flush=True)

    # Clean data directory if requested
    if args.clean and Path(args.data_dir).exists():
        print(f"Cleaning data directory: {args.data_dir}", flush=True)
        shutil.rmtree(args.data_dir)

    # Initialize network and optimizer
    network = AdvantageNetwork(
        input_size=STATE_DIM,
        hidden_size=args.hidden_size,
        output_size=TARGET_DIM,
    ).to(device)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    print(f"Network: {sum(p.numel() for p in network.parameters())} parameters", flush=True)

    # Initialize Rust trainer
    trainer = aion26_rust.RustTrainer(
        data_dir=args.data_dir,
        query_buffer_size=args.query_buffer_size,
    )

    print(f"Rust trainer initialized: {trainer.data_dir()}", flush=True)

    # Initialize dataset (empty initially)
    dataset = TrajectoryDataset(args.data_dir)

    # Vintage validation: track epoch 0 samples to detect forgetting
    vintage_sample_count = 0
    vintage_loss_history = []

    # Training loop
    total_samples = 0
    total_start = time.time()

    print(f"\n{'Epoch':>6} | {'Samples':>10} | {'Gen(s)':>8} | {'Train Loss':>12} | {'Vintage Loss':>12} | {'Disk(MB)':>10}", flush=True)
    print("-" * 85, flush=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # === PHASE A: GENERATION (Rust + GPU Inference) ===
        trainer.start_epoch(epoch)

        samples, inf_calls, gen_time, timing = run_epoch_generation(
            trainer=trainer,
            network=network,
            device=device,
            num_traversals=args.traversals_per_epoch,
            verbose=args.verbose,
        )

        trainer.end_epoch()
        total_samples += samples

        # === PHASE B: TRAINING (Disk Streaming) ===
        dataset.refresh()  # Discover new epoch file

        avg_loss, train_steps, train_time = run_epoch_training(
            network=network,
            optimizer=optimizer,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_steps=args.max_train_steps,
            alpha=args.alpha,
        )

        # === PHASE C: VINTAGE VALIDATION ===
        # After epoch 0, record how many samples are in epoch 0 (for vintage reference)
        if epoch == 0:
            vintage_sample_count = samples
            print(f"  [Vintage] Recorded {vintage_sample_count} samples from epoch 0 as reference", flush=True)

        # Compute vintage loss periodically (sample from epoch 0 portion only)
        vintage_loss = 0.0
        if epoch > 0 and (epoch + 1) % args.vintage_interval == 0:
            # Sample specifically from the first epoch's indices (0 to vintage_sample_count-1)
            vintage_loss = compute_vintage_loss_on_epoch0(
                network, dataset, device, vintage_sample_count
            )
            vintage_loss_history.append((epoch, vintage_loss))

        # Compute disk usage
        disk_usage = sum(f.stat().st_size for f in Path(args.data_dir).glob("*.bin")) / (1024 * 1024)

        # Log progress
        vintage_str = f"{vintage_loss:>12.4f}" if vintage_loss > 0 else f"{'--':>12}"
        print(f"{epoch:>6} | {samples:>10} | {gen_time:>8.1f} | {avg_loss:>12.4f} | {vintage_str} | {disk_usage:>10.1f}", flush=True)

        # Periodic detailed status
        if (epoch + 1) % args.vintage_interval == 0:
            elapsed = time.time() - total_start
            samples_per_sec = total_samples / elapsed

            # Check for forgetting: vintage loss should stay stable
            if len(vintage_loss_history) >= 2:
                first_vintage = vintage_loss_history[0][1]
                latest_vintage = vintage_loss_history[-1][1]
                vintage_ratio = latest_vintage / first_vintage if first_vintage > 0 else 1.0
                forgetting_status = "OK" if vintage_ratio < 2.0 else "WARN: Possible forgetting!"
            else:
                forgetting_status = "Collecting baseline..."

            print(f"  [Status] Total: {total_samples:,} | Rate: {samples_per_sec:.0f}/s | Disk: {disk_usage:.1f} MB | {forgetting_status}", flush=True)

    # Final summary
    total_time = time.time() - total_start
    disk_usage = sum(f.stat().st_size for f in Path(args.data_dir).glob("*.bin")) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print(f"Phase 6 Training Complete")
    print(f"{'=' * 70}")
    print(f"Total epochs: {args.epochs}")
    print(f"Total samples: {total_samples:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Throughput: {total_samples / total_time:.0f} samples/s")
    print(f"Disk usage: {disk_usage:.1f} MB")
    print(f"Epoch files: {dataset.num_epochs}")

    # Vintage validation summary
    if len(vintage_loss_history) >= 2:
        first_vintage = vintage_loss_history[0][1]
        last_vintage = vintage_loss_history[-1][1]
        vintage_ratio = last_vintage / first_vintage if first_vintage > 0 else 1.0
        print(f"\n--- Vintage Validation (No-Forgetting Test) ---")
        print(f"Epoch 0 reference samples: {vintage_sample_count:,}")
        print(f"First vintage loss (epoch {vintage_loss_history[0][0]}): {first_vintage:.4f}")
        print(f"Final vintage loss (epoch {vintage_loss_history[-1][0]}): {last_vintage:.4f}")
        print(f"Ratio (final/first): {vintage_ratio:.2f}x")
        if vintage_ratio < 1.5:
            print(f"Result: PASSED - Network maintains memory of early lessons")
        elif vintage_ratio < 2.0:
            print(f"Result: WARNING - Some forgetting detected")
        else:
            print(f"Result: FAILED - Significant forgetting detected")

    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
