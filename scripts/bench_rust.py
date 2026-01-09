#!/usr/bin/env python3
"""Benchmark: Python vs Rust for River Hold'em simulation

Measures the speedup achieved by porting game logic to Rust.
Target: >50x speedup for game state transitions.
"""

import sys
from pathlib import Path
import time

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "aion26_rust"))

from aion26.games.river_holdem import new_river_holdem_game
import aion26_rust

def bench_python_game(num_playouts: int = 10000) -> float:
    """Benchmark Python TexasHoldemRiver implementation"""
    start = time.time()

    for _ in range(num_playouts):
        game = new_river_holdem_game()

        # Deal cards
        game = game.apply_action(0)

        # Play random actions until terminal
        while not game.is_terminal():
            actions = game.legal_actions()
            if not actions:
                break
            # Always choose first legal action for consistency
            game = game.apply_action(actions[0])

        # Get returns (force evaluation)
        _ = game.returns()

    elapsed = time.time() - start
    return elapsed


def bench_rust_game(num_playouts: int = 10000) -> float:
    """Benchmark Rust RustRiverHoldem implementation"""
    start = time.time()

    for _ in range(num_playouts):
        game = aion26_rust.RustRiverHoldem()

        # Deal cards
        game = game.apply_action(0)

        # Play random actions until terminal
        while not game.is_terminal():
            actions = game.legal_actions()
            if not actions:
                break
            # Always choose first legal action for consistency
            game = game.apply_action(actions[0])

        # Get returns (force evaluation)
        _ = game.returns()

    elapsed = time.time() - start
    return elapsed


def main():
    print("="*80)
    print("BENCHMARK: Python vs Rust for River Hold'em")
    print("="*80)
    print()

    # Warm-up
    print("Warming up...")
    bench_python_game(100)
    bench_rust_game(100)
    print("✅ Warm-up complete")
    print()

    # Benchmark
    num_playouts = 10000
    print(f"Running {num_playouts:,} playouts for each implementation...")
    print()

    print("Python (baseline):")
    python_time = bench_python_game(num_playouts)
    python_rate = num_playouts / python_time
    print(f"  Time: {python_time:.2f}s")
    print(f"  Rate: {python_rate:.0f} playouts/sec")
    print()

    print("Rust (optimized):")
    rust_time = bench_rust_game(num_playouts)
    rust_rate = num_playouts / rust_time
    print(f"  Time: {rust_time:.2f}s")
    print(f"  Rate: {rust_rate:.0f} playouts/sec")
    print()

    speedup = python_time / rust_time
    print("="*80)
    print(f"SPEEDUP: {speedup:.1f}x")
    print("="*80)
    print()

    if speedup >= 50:
        print(f"✅ SUCCESS! Achieved {speedup:.1f}x speedup (target: >50x)")
    elif speedup >= 20:
        print(f"⚠️  PARTIAL: Achieved {speedup:.1f}x speedup (target: >50x)")
        print("   Consider enabling more aggressive optimizations.")
    else:
        print(f"❌ FAILED: Only {speedup:.1f}x speedup (target: >50x)")
        print("   Check Rust implementation for bottlenecks.")

    print()
    print("Impact on training:")
    print(f"  50k iterations at Python speed: {50000 / python_rate / 60:.0f} minutes")
    print(f"  50k iterations at Rust speed: {50000 / rust_rate / 60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
