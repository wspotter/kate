#!/bin/bash

echo "🔍 Kate Debug - Step by Step Test"
echo "================================="
echo ""

echo "1. Checking Poetry..."
if command -v poetry &> /dev/null; then
    echo "✅ Poetry found: $(poetry --version)"
else
    echo "❌ Poetry not found"
    exit 1
fi

echo ""
echo "2. Checking Poetry environment..."
if poetry env info &>/dev/null; then
    echo "✅ Poetry environment exists"
    poetry env info
else
    echo "❌ Poetry environment not found"
fi

echo ""
echo "3. Testing Python import..."
timeout 3s poetry run python -c "print('✅ Python works')" || echo "❌ Python test failed or timed out"

echo ""
echo "4. Testing Kate minimal import..."
timeout 3s poetry run python -c "
import sys
sys.path.insert(0, 'app')
try:
    import click
    print('✅ Click import works')
except Exception as e:
    print(f'❌ Click import failed: {e}')
" || echo "❌ Import test failed or timed out"

echo ""
echo "5. Checking files..."
ls -la pyproject*.toml

echo ""
echo "🎯 Debug complete"