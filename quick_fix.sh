#!/bin/bash
echo "🔧 Kate Quick Fix Script"
echo "======================"
echo ""

echo "📋 Step 1: Install system dependencies"
echo "Run manually: sudo apt update && sudo apt install -y python3-dev build-essential qt6-base-dev libgl1-mesa-dev"
echo ""

echo "📋 Step 2: Install Python dependencies"
echo "Run manually: poetry install"
echo "(This will take 5-10 minutes due to 120+ packages)"
echo ""

echo "📋 Step 3: Test Kate startup"
echo "Run manually: poetry run python app/main.py"
echo ""

echo "🚀 Alternative: Quick test with backup"
echo "If issues persist, copy original_kate/app to test_kate and install minimal deps:"
echo "pip3 install PySide6 loguru pydantic sqlalchemy aiosqlite"
echo ""

echo "✅ Commands ready - run them manually in terminal"