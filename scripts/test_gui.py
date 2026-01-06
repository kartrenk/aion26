#!/usr/bin/env python3
"""Test GUI components without launching the window.

This script verifies that all GUI components can be instantiated correctly
without requiring a display (headless testing).
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test all GUI imports."""
    print("Testing imports...")

    try:
        from aion26.config import AionConfig, leduc_vr_ddcfr_config
        print("  ✓ Config module")
    except ImportError as e:
        print(f"  ✗ Config module: {e}")
        return False

    try:
        from aion26.gui.model import TrainingThread, MetricsUpdate
        print("  ✓ GUI model")
    except ImportError as e:
        print(f"  ✗ GUI model: {e}")
        return False

    try:
        import tkinter
        print(f"  ✓ Tkinter (version {tkinter.TkVersion})")
    except ImportError as e:
        print(f"  ✗ Tkinter: {e}")
        return False

    try:
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("  ✓ Matplotlib TkAgg backend")
    except ImportError as e:
        print(f"  ✗ Matplotlib TkAgg: {e}")
        return False

    return True


def test_config():
    """Test config system."""
    print("\nTesting config system...")

    from aion26.config import AionConfig, leduc_vr_ddcfr_config
    import tempfile

    try:
        # Create config
        config = leduc_vr_ddcfr_config()
        print(f"  ✓ Created config: {config}")

        # Test YAML save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        config.to_yaml(temp_path)
        print(f"  ✓ Saved config to {temp_path}")

        loaded = AionConfig.from_yaml(temp_path)
        print(f"  ✓ Loaded config: {loaded}")

        # Cleanup
        Path(temp_path).unlink()

        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        return False


def test_training_thread():
    """Test training thread initialization."""
    print("\nTesting training thread...")

    from aion26.gui.model import TrainingThread, MetricsUpdate
    from aion26.config import kuhn_vanilla_config
    import queue

    try:
        config = kuhn_vanilla_config()
        # Reduce iterations for quick test
        config.training.iterations = 10

        metrics_queue = queue.Queue()
        thread = TrainingThread(config, metrics_queue)
        print(f"  ✓ Created TrainingThread: {thread}")

        return True
    except Exception as e:
        print(f"  ✗ TrainingThread test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GUI Component Verification")
    print("=" * 60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test config
    results.append(("Config", test_config()))

    # Test training thread
    results.append(("TrainingThread", test_training_thread()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe GUI is ready to launch:")
        print("  PYTHONPATH=src .venv-system/bin/python scripts/launch_gui.py")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
