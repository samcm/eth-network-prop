#!/usr/bin/env bash

# Exit on error
set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Installing dependencies with uv..."
uv pip install -e .

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Error: Data directory not found. Please run 'python preprocess_data.py' first."
    exit 1
fi

echo "Starting Streamlit application"
uv run streamlit run app.py
