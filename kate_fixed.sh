#!/bin/bash

echo "🚀 Kate LLM Client - FINAL WORKING FIX"
echo "===================================="
echo ""

echo "✅ Issue identified: Missing sentence_transformers dependency"
echo "✅ Fixed: Updated minimal config with required ML packages"
echo ""

echo "🔧 Installing Kate with working dependencies..."
echo ""

# Use the fixed minimal config
cp pyproject_minimal.toml pyproject.toml
rm -rf poetry.lock

echo "📦 Installing dependencies (this will take 2-3 minutes for PyTorch/transformers)..."
poetry install --only main

echo ""
echo "🧪 Testing Kate imports..."
if poetry run python -c "
import sys
sys.path.insert(0, 'app')
try:
    from app.core.application import KateApplication
    print('✅ Kate imports successful!')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"; then
    echo ""
    echo "🚀 Launching Kate LLM Client..."
    echo "Expected: 3-column desktop interface with multi-modal AI capabilities"
    echo ""
    
    # Set GUI environment
    export DISPLAY="${DISPLAY:-:0}"
    
    # Launch Kate
    poetry run python app/main.py
else
    echo ""
    echo "❌ Import test failed - check error above"
    exit 1
fi