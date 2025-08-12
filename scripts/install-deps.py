#!/usr/bin/env python3
"""
GraphRAG Dependency Installation Script
This script ensures proper dependency installation with version constraints
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        sys.exit(1)
    return result


def main():
    print("🚀 Installing GraphRAG dependencies with compatibility constraints...")
    
    # Check if virtual environment exists
    venv_path = Path(".venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        venv.create(".venv", with_pip=True)
    
    # Determine the Python executable in the virtual environment
    if os.name == 'nt':  # Windows
        python_exe = ".venv\\Scripts\\python.exe"
        pip_exe = ".venv\\Scripts\\pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = ".venv/bin/python"
        pip_exe = ".venv/bin/pip"
    
    # Upgrade pip
    print("⬆️  Upgrading pip...")
    run_command(f"{python_exe} -m pip install --upgrade pip")
    
    # Install with constraints
    print("📚 Installing dependencies with version constraints...")
    run_command(f"{pip_exe} install -r constraints.txt")
    
    # Check if this is a development installation
    if len(sys.argv) > 1 and sys.argv[1] == "--dev":
        print("🔬 Installing development dependencies...")
        run_command(f"{pip_exe} install -r requirements-dev.txt")
    
    # Verify installation
    print("✅ Verifying GraphRAG installation...")
    try:
        result = run_command(f"{python_exe} -c \"import graphrag; print(f'GraphRAG {{graphrag.__version__}} installed successfully!')\"", check=False)
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print("⚠️  Warning: Could not verify GraphRAG import")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify GraphRAG import: {e}")
    
    print("")
    print("🎉 Installation complete!")
    if os.name == 'nt':  # Windows
        print("   Activate the virtual environment with:")
        print("   .venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   Activate the virtual environment with:")
        print("   source .venv/bin/activate")
    print("")
    print("📖 For more information, see the README.md file.")
    print("🔧 For maintenance records and dependency fixes, see docs/maintenance/README.md")


if __name__ == "__main__":
    main()
