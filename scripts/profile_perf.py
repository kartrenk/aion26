#!/usr/bin/env python
"""Hardware-Agnostic Performance Profiler

Detects accelerator (CUDA vs MPS vs CPU) and instruments the
training loop to identify bottlenecks.

Usage:
    uv run python scripts/profile_perf.py            # Single-threaded
    uv run python scripts/profile_perf.py --parallel # 8-worker parallel
"""

import argparse
import time
import platform
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import aion26_rust

# Configuration
PROFILE_STEPS = 200
TRAVERSALS_PER_STEP = 200_000
NUM_WORKERS = 8


class DeepCFRNetwork(nn.Module):
    """Simple network matching training architecture."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_device_and_sync():
    """Auto-detects hardware and returns (device, sync_function)."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.cuda.synchronize
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.mps.synchronize
    else:
        return torch.device("cpu"), lambda: None


def profile_run(use_parallel: bool = False):
    device, sync_fn = get_device_and_sync()
    system_info = f"{platform.system()} {platform.processor()}"

    trainer_type = "PARALLEL (8 workers)" if use_parallel else "SINGLE-THREADED"
    print(f"Hardware: {system_info}")
    print(f"Device: {device.type.upper()}")
    print(f"Trainer: {trainer_type}")

    # Clean profile directory
    profile_dir = Path("data/profile_test")
    if profile_dir.exists():
        shutil.rmtree(profile_dir)

    # Initialize trainer
    if use_parallel:
        trainer = aion26_rust.ParallelTrainer(
            data_dir=str(profile_dir),
            query_buffer_size=4096,
            num_workers=NUM_WORKERS,
        )
    else:
        trainer = aion26_rust.RustTrainer(
            data_dir=str(profile_dir),
            query_buffer_size=4096,
        )

    # Network
    network = DeepCFRNetwork(136, 4, 256).to(device)
    network.eval()

    # Timers
    t_rust_total = 0
    t_transfer_total = 0
    t_gpu_total = 0
    total_samples = 0

    epoch = 0
    trainer.start_epoch(epoch)

    print(f"\nProfiling ({PROFILE_STEPS} batches, {TRAVERSALS_PER_STEP:,} traversals)...")
    step_count = 0
    predictions = None

    sync_fn()
    start_time = time.perf_counter_ns()

    while step_count < PROFILE_STEPS:
        # --- 1. Rust Logic (CPU) ---
        t0 = time.perf_counter_ns()
        result = trainer.step(predictions, num_traversals=TRAVERSALS_PER_STEP if predictions is None else None)
        t1 = time.perf_counter_ns()
        t_rust_total += (t1 - t0)

        if result.is_finished():
            trainer.end_epoch()
            epoch += 1
            trainer.start_epoch(epoch)
            predictions = None
            continue

        # --- 2. Data Transfer ---
        t2 = time.perf_counter_ns()
        query_buffer = trainer.get_query_buffer()
        queries_tensor = torch.from_numpy(query_buffer).to(device, non_blocking=True)
        t3 = time.perf_counter_ns()
        t_transfer_total += (t3 - t2)

        # --- 3. Inference ---
        t4 = time.perf_counter_ns()
        with torch.no_grad():
            pred_tensor = network(queries_tensor)
        sync_fn()
        t5 = time.perf_counter_ns()
        t_gpu_total += (t5 - t4)

        predictions = pred_tensor.cpu().numpy()
        step_count += 1
        total_samples += query_buffer.shape[0]

        if step_count % 50 == 0:
            elapsed = (time.perf_counter_ns() - start_time) / 1e9
            rate = total_samples / elapsed
            print(f"  Step {step_count}/{PROFILE_STEPS} | {rate:,.0f} samples/s | Batch: {query_buffer.shape[0]}")

    end_time = time.perf_counter_ns()
    total_time_ms = (end_time - start_time) / 1e6

    trainer.end_epoch()

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"PERFORMANCE AUDIT - {trainer_type}")
    print("=" * 60)
    print(f"Device:          {device.type.upper()}")
    print(f"Total Time:      {total_time_ms/1000:.2f} s")
    print(f"Throughput:      {total_samples / (total_time_ms/1000):,.0f} samples/sec")
    print(f"Batches:         {step_count}")
    print(f"Avg Batch Size:  {total_samples / step_count:,.0f}")
    print("-" * 60)

    avg_rust = (t_rust_total / step_count) / 1e6
    avg_trans = (t_transfer_total / step_count) / 1e6
    avg_gpu = (t_gpu_total / step_count) / 1e6
    total_measured = t_rust_total + t_transfer_total + t_gpu_total

    print(f"1. Rust Logic:    {avg_rust:>8.3f} ms  ({(t_rust_total/total_measured)*100:>5.1f}%)")
    print(f"2. Data Transfer: {avg_trans:>8.3f} ms  ({(t_transfer_total/total_measured)*100:>5.1f}%)")
    print(f"3. AI Inference:  {avg_gpu:>8.3f} ms  ({(t_gpu_total/total_measured)*100:>5.1f}%)")
    print("-" * 60)

    # Bottleneck analysis
    if avg_rust > avg_gpu * 2:
        print("BOTTLENECK: CPU-bound. GPU is starving.")
        if not use_parallel:
            print("  TRY: --parallel flag for 8-worker parallelism")
        else:
            print("  Solution: More workers or faster CPU")
    elif avg_gpu > avg_rust * 2:
        print("BOTTLENECK: GPU-bound. Model inference is the limit.")
        print("  Solution: Smaller model or larger batch size")
    elif avg_trans > avg_rust and avg_trans > avg_gpu:
        print("BOTTLENECK: Memory bandwidth.")
        print("  Solution: Pin memory or reduce transfer size")
    else:
        print("STATUS: Balanced pipeline - optimal utilization!")

    print("=" * 60)

    # Cleanup
    shutil.rmtree(profile_dir, ignore_errors=True)

    return total_samples / (total_time_ms / 1000)


def main():
    parser = argparse.ArgumentParser(description="Performance Profiler")
    parser.add_argument("--parallel", action="store_true", help="Use parallel trainer (8 workers)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("AION-26 PERFORMANCE PROFILER")
    print("=" * 60)

    if args.parallel:
        rate = profile_run(use_parallel=True)
    else:
        # Run both and compare
        print("\n[1/2] Single-threaded baseline...")
        rate_single = profile_run(use_parallel=False)

        print("\n[2/2] Parallel (8 workers)...")
        rate_parallel = profile_run(use_parallel=True)

        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Single-threaded: {rate_single:>10,.0f} samples/s")
        print(f"Parallel (8w):   {rate_parallel:>10,.0f} samples/s")
        speedup = rate_parallel / rate_single if rate_single > 0 else 0
        print(f"Speedup:         {speedup:>10.1f}x")
        print("=" * 60)


if __name__ == "__main__":
    main()
