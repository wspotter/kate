#!/bin/bash

echo "=== Testing Kate with Qt Environment Fixes ==="

# Set Qt environment variables for proper rendering
export QT_QPA_PLATFORMTHEME=qt5ct
export DISPLAY=:0

# Additional Qt environment variables for better compatibility
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1
export QT_SCREEN_SCALE_FACTORS=1

echo "Environment variables set:"
echo "QT_QPA_PLATFORMTHEME: $QT_QPA_PLATFORMTHEME"
echo "DISPLAY: $DISPLAY"
echo "QT_AUTO_SCREEN_SCALE_FACTOR: $QT_AUTO_SCREEN_SCALE_FACTOR"

echo ""
echo "=== Test 1: Basic Qt with environment variables ==="
cd /home/stacy/Desktop/kate
poetry run python test_qt_basic.py

echo ""
echo "=== Test 2: Kate with xcb platform plugin explicitly ==="
poetry run python -m app.main -platform xcb

echo ""
echo "=== Test 3: Kate with environment variables (if Test 2 fails) ==="
poetry run python -m app.main