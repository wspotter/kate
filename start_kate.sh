#!/bin/bash

# Kate LLM Client - Smart Startup Script
# Automatically handles dependency conflicts and launches Kate

set -e  # Exit on any error

KATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$KATE_DIR"

echo "🚀 Kate LLM Client - Smart Startup"
echo "=================================="
echo "📍 Directory: $KATE_DIR"
echo ""

# Function to check if Kate can import
check_kate_imports() {
    echo "🔍 Testing Kate imports..."
    if poetry run python -c "
import sys
sys.path.insert(0, 'app')
try:
    from app.core.application import KateApplication
    print('✅ Kate imports successful')
    exit(0)
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install minimal dependencies
install_minimal_deps() {
    echo "📦 Installing minimal dependencies..."
    
    # Backup original pyproject.toml
    if [ -f "pyproject.toml" ] && [ ! -f "pyproject_original.toml" ]; then
        cp pyproject.toml pyproject_original.toml
        echo "💾 Backed up original pyproject.toml"
    fi
    
    # Use minimal configuration
    cp pyproject_minimal.toml pyproject.toml
    echo "📝 Using minimal dependency configuration"
    
    # Clean and install
    rm -rf poetry.lock .venv 2>/dev/null || true
    poetry install --no-dev
    
    echo "✅ Minimal dependencies installed"
}

# Function to try original Kate backup
try_original_kate() {
    echo "🔄 Trying original Kate backup..."
    cd original_kate
    
    # Install essential packages
    echo "📦 Installing essential packages..."
    pip3 install PySide6 loguru pydantic sqlalchemy aiosqlite click 2>/dev/null || {
        echo "❌ Failed to install packages"
        cd ..
        return 1
    }
    
    # Test import
    if python3 -c "
import sys
sys.path.insert(0, 'app')
try:
    from app.core.application import KateApplication
    print('✅ Original Kate imports working')
except Exception as e:
    print(f'❌ Original Kate import failed: {e}')
    exit(1)
" 2>/dev/null; then
        echo "🚀 Launching original Kate..."
        python3 app/main.py
        return 0
    else
        cd ..
        return 1
    fi
}

# Function to launch Kate
launch_kate() {
    echo "🚀 Launching Kate LLM Client..."
    echo ""
    
    # Set display for GUI
    export DISPLAY="${DISPLAY:-:0}"
    
    # Launch with proper error handling
    if poetry run python app/main.py; then
        echo ""
        echo "✅ Kate launched successfully!"
        return 0
    else
        echo ""
        echo "❌ Kate failed to launch"
        return 1
    fi
}

# Main startup logic
main() {
    echo "🔧 Step 1: Checking current state..."
    
    # Check if Poetry is available
    if ! command -v poetry &> /dev/null; then
        echo "❌ Poetry not found. Please install Poetry first:"
        echo "   curl -sSL https://install.python-poetry.org | python3 -"
        exit 1
    fi
    
    # Try current setup first
    if poetry env info &>/dev/null && check_kate_imports; then
        echo "✅ Kate dependencies already working!"
        launch_kate && exit 0
    fi
    
    echo ""
    echo "🔧 Step 2: Fixing dependencies..."
    
    # Try minimal dependencies approach
    if install_minimal_deps && check_kate_imports; then
        echo "✅ Minimal dependencies working!"
        launch_kate && exit 0
    fi
    
    echo ""
    echo "🔧 Step 3: Trying backup version..."
    
    # Try original Kate backup
    if try_original_kate; then
        exit 0
    fi
    
    # If all else fails
    echo ""
    echo "❌ All startup methods failed!"
    echo ""
    echo "🔧 Manual steps to try:"
    echo "1. Check system dependencies:"
    echo "   sudo apt install python3-dev build-essential qt6-base-dev libgl1-mesa-dev"
    echo ""
    echo "2. Try manual Poetry setup:"
    echo "   poetry install --no-dev"
    echo ""
    echo "3. Check the logs above for specific errors"
    echo ""
    exit 1
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n👋 Kate startup cancelled"; exit 130' INT

# Run main function
main "$@"