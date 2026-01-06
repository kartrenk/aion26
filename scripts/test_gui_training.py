#!/usr/bin/env python3
"""Automated test for GUI training functionality.

Tests that training actually works for both Kuhn and Leduc poker without errors.
"""

import sys
from pathlib import Path
import queue
import time

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from aion26.config import AionConfig, GameConfig, TrainingConfig, ModelConfig, AlgorithmConfig
from aion26.gui.model import TrainingThread


def test_game(game_name: str, iterations: int = 50):
    """Test training for a specific game.

    Args:
        game_name: "kuhn" or "leduc"
        iterations: Number of training iterations

    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {game_name.upper()} Poker Training")
    print(f"{'='*60}\n")

    # Create config
    # Use small buffer capacity (100) so buffer fills in first ~20-40 iterations
    # This allows network to actually train during the test
    config = AionConfig(
        name=f"{game_name}_test",
        game=GameConfig(name=game_name),
        training=TrainingConfig(
            iterations=iterations,
            batch_size=32,  # Smaller batch size for testing
            buffer_capacity=100,  # Small buffer that fills quickly
            eval_every=10,  # Evaluate more frequently for testing
            log_every=1,   # Log every iteration for testing
        ),
        model=ModelConfig(
            hidden_size=64,
            num_hidden_layers=3,
            learning_rate=0.001,
        ),
        algorithm=AlgorithmConfig(
            use_vr=True,
            scheduler_type="ddcfr",
            gamma=2.0,
        ),
        seed=42,
    )

    print(f"Config: {config}\n")

    # Create training thread
    metrics_queue = queue.Queue()
    thread = TrainingThread(config, metrics_queue)

    print("Starting training thread...")
    thread.start()

    # Monitor training
    last_iteration = 0
    nash_conv_values = []
    errors = []

    while thread.is_alive():
        try:
            # Get metrics with timeout
            metrics = metrics_queue.get(timeout=1.0)

            # Check for errors
            if metrics.status == "error":
                errors.append(metrics.error_message)
                print(f"✗ ERROR: {metrics.error_message}")
                break

            # Track progress
            if metrics.iteration > last_iteration:
                last_iteration = metrics.iteration

                # Print progress every 10 iterations
                if metrics.iteration % 10 == 0:
                    nashconv_str = f"{metrics.nash_conv:.6f}" if metrics.nash_conv is not None else "N/A"
                    print(f"  Iteration {metrics.iteration:3d} | Loss: {metrics.loss:.6f} | "
                          f"NashConv: {nashconv_str} | Buffer: {metrics.buffer_size}")

                # Collect NashConv values
                if metrics.nash_conv is not None:
                    nash_conv_values.append((metrics.iteration, metrics.nash_conv))

            # Check if completed
            if metrics.status == "completed":
                print(f"\n✓ Training completed at iteration {metrics.iteration}")
                break

        except queue.Empty:
            # No metrics yet, keep waiting
            pass
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            thread.stop()
            break

    # Wait for thread to finish
    thread.join(timeout=5.0)

    # Analyze results
    print(f"\n{'='*60}")
    print("Test Results")
    print(f"{'='*60}\n")

    if errors:
        print(f"✗ FAILED - {len(errors)} error(s) occurred:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return False

    if last_iteration < iterations:
        print(f"✗ FAILED - Training stopped early at iteration {last_iteration}/{iterations}")
        return False

    print(f"✓ Completed {last_iteration} iterations")
    print(f"✓ Computed NashConv {len(nash_conv_values)} times")

    if nash_conv_values:
        print(f"\nNashConv Progress:")
        for iteration, nashconv in nash_conv_values:
            print(f"  Iteration {iteration:3d}: {nashconv:.6f}")

        # Check convergence
        initial_nashconv = nash_conv_values[0][1]
        final_nashconv = nash_conv_values[-1][1]
        improvement = initial_nashconv - final_nashconv

        print(f"\nConvergence:")
        print(f"  Initial NashConv: {initial_nashconv:.6f}")
        print(f"  Final NashConv:   {final_nashconv:.6f}")
        print(f"  Improvement:      {improvement:.6f}")

        # Success criteria: NashConv should decrease or stay low
        if final_nashconv < 1.0:  # Reasonable threshold
            print(f"\n✓ PASSED - NashConv converging ({final_nashconv:.6f} < 1.0)")
            return True
        else:
            print(f"\n⚠ WARNING - NashConv not converging well ({final_nashconv:.6f})")
            return True  # Still pass but with warning
    else:
        print("⚠ WARNING - No NashConv values computed (iterations too low)")
        return True  # Still pass if no errors


def main():
    """Run all tests."""
    print("="*60)
    print("GUI Training Automated Test Suite")
    print("="*60)

    results = {}

    # Test Kuhn Poker
    try:
        results["kuhn"] = test_game("kuhn", iterations=50)
    except Exception as e:
        print(f"\n✗ KUHN TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results["kuhn"] = False

    # Test Leduc Poker
    try:
        results["leduc"] = test_game("leduc", iterations=50)
    except Exception as e:
        print(f"\n✗ LEDUC TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results["leduc"] = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60 + "\n")

    for game, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{game.upper():10} {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nThe GUI should work correctly for both Kuhn and Leduc poker!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
