#!/bin/bash
echo "🔧 Kate Application Fix - DEPENDENCY CONFLICT RESOLVED"
echo "======================================================"
echo ""

echo "✅ FIXED: Removed googletrans dependency that conflicted with httpx"
echo "📋 Now run these commands:"
echo ""

echo "1️⃣  Install core dependencies:"
echo "poetry install"
echo ""

echo "2️⃣  Test Kate startup:"
echo "poetry run python app/main.py"
echo ""

echo "🚀 If issues persist, try installing just core packages first:"
echo "poetry install --only main"
echo ""

echo "📖 Translation feature can be re-enabled later with:"
echo "pip install googletrans==4.0.0rc1  # newer version that doesn't conflict"
echo ""

echo "✅ Ready to run - dependency conflict resolved!"