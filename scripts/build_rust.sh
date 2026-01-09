#!/usr/bin/env bash
# Build Rust extension for Aion-26

set -e

cd "$(dirname "$0")/.."

echo "Building Rust extension with maturin..."
cd src/aion26_rust

# Build in release mode with full optimizations
uv run maturin develop --release

echo ""
echo "✅ Rust extension built successfully!"
echo ""
echo "Test import:"
python3 -c "import aion26_rust; print(f'  aion26_rust.evaluate_7_cards: {aion26_rust.evaluate_7_cards}')"
