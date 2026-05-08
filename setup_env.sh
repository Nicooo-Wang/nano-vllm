#!/bin/bash
set -e

echo "=== Setting up nano-vllm uv environment ==="

# 1. Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating uv venv..."
    uv venv --python 3.10
fi

# 2. Detect CUDA version and install compatible torch
# This system has CUDA 12.8, so we use torch 2.6.0+cu124 (CUDA 12.x compatible)
echo "Installing torch with CUDA 12.4 support..."
uv pip install torch==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124

# 3. Clear flash-attn cache to avoid stale prebuilt wheels
echo "Clearing flash-attn build cache..."
rm -rf ~/.cache/uv/sdists-v9/index/*/flash-attn 2>/dev/null || true
find ~/.cache/uv/archive-v0 -maxdepth 1 -type d | while read dir; do
    if ls "$dir" 2>/dev/null | grep -qi flash_attn; then
        rm -rf "$dir"
    fi
done

# 4. Install project dependencies with no-build-isolation
# FLASH_ATTENTION_FORCE_BUILD forces flash-attn to compile from source
# instead of downloading potentially ABI-incompatible prebuilt wheels
echo "Installing nano-vllm (flash-attn will build from source, this takes ~10-15 min)..."
FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install -e . --no-build-isolation

echo "=== Setup complete! ==="
echo ""
echo "You can now run:"
echo "  python example.py"
echo "  python bench.py"
echo ""
echo "Note: example.py and bench.py expect a local model at ~/huggingface/Qwen3-0.6B/"
