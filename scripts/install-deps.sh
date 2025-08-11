#!/bin/bash

# GraphRAG Dependency Installation Script
# This script ensures proper dependency installation with version constraints

set -e

echo "🚀 Installing GraphRAG dependencies with compatibility constraints..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install with constraints
echo "📚 Installing dependencies with version constraints..."
pip install -r constraints.txt

# Check if this is a development installation
if [ "$1" = "--dev" ]; then
    echo "🔬 Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Verify installation
echo "✅ Verifying GraphRAG installation..."
python -c "import graphrag; print(f'GraphRAG {graphrag.__version__} installed successfully!')"

echo ""
echo "🎉 Installation complete! Activate the virtual environment with:"
echo "   source .venv/bin/activate"
echo ""
echo "📖 For more information, see the README.md file."
echo "🔧 For maintenance records and dependency fixes, see docs/maintenance/README.md"
