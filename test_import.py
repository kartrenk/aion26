#!/usr/bin/env python3
"""Minimal test of aion26_rust import"""
import sys
from pathlib import Path

# Add DLL directories
if sys.platform == 'win32':
    import os
    rust_dll = Path(__file__).parent / "src" / "aion26_rust" / "target" / "release"
    if rust_dll.exists():
        os.add_dll_directory(str(rust_dll))

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"Platform: {sys.platform}")

try:
    from aion26_rust import ParallelTrainer, RustRiverHoldem
    print("SUCCESS: aion26_rust imported!")
    print(f"ParallelTrainer: {ParallelTrainer}")
    print(f"RustRiverHoldem: {RustRiverHoldem}")
except ImportError as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
