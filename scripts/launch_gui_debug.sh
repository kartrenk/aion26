#!/bin/bash
# Debug launcher for Aion-26 GUI with full logging

echo "====================================================================="
echo "Aion-26 GUI Debug Launcher"
echo "====================================================================="
echo ""
echo "This will launch the GUI with detailed logging to help debug issues."
echo ""
echo "Logs will be saved to:"
echo "  logs/gui_YYYYMMDD_HHMMSS.log"
echo ""
echo "You can also watch the console output in real-time."
echo ""
echo "Press Ctrl+C to stop."
echo ""
echo "====================================================================="
echo ""

cd "$(dirname "$0")/.."

# Run with Python path and logging
PYTHONPATH=src .venv-system/bin/python scripts/launch_gui.py

echo ""
echo "====================================================================="
echo "GUI closed."
echo ""
echo "Log file saved to: logs/"
echo "Check the log file for detailed session information."
echo "====================================================================="
