#!/usr/bin/env python3
"""Launch script for Aion-26 Deep PDCFR+ Visualizer GUI.

This script launches the Tkinter-based GUI for training Deep CFR agents
with real-time visualization of convergence and strategy evolution.

Usage:
    # With system Python venv (recommended for Tkinter support)
    PYTHONPATH=src .venv-system/bin/python scripts/launch_gui.py

    # Or with uv (if Tkinter is available)
    PYTHONPATH=src uv run python scripts/launch_gui.py

    # Or direct execution
    ./scripts/launch_gui.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Create logs directory
project_root = Path(__file__).parent.parent
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = logs_dir / f"gui_{timestamp}.log"

# Configure logging to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# Add src to path for development
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
    logger.info(f"Added {src_path} to Python path")

try:
    from aion26.gui.app import launch_gui
    logger.info("Successfully imported launch_gui")
except ImportError as e:
    logger.error(f"Failed to import GUI: {e}")
    raise


def main():
    """Main entry point."""
    logger.info("Starting Aion-26 Visualizer")
    print("Launching Aion-26 Deep PDCFR+ Visualizer...")
    print("=" * 60)
    print("Features:")
    print("  - Real-time NashConv convergence plotting")
    print("  - Strategy inspector for information sets")
    print("  - Configuration management (save/load YAML)")
    print("  - Background training with non-blocking UI")
    print("=" * 60)
    print()
    print("Check console for detailed logs...")
    print()

    try:
        logger.info("Calling launch_gui()...")
        launch_gui()
        logger.info("GUI exited normally")
    except KeyboardInterrupt:
        logger.info("GUI closed by user (Ctrl+C)")
        print("\n\nGUI closed by user.")
    except Exception as e:
        logger.exception(f"GUI crashed: {e}")
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
