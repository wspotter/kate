#!/usr/bin/env python3
"""
Basic Qt test to check if PySide6 works and window can be shown.
This bypasses all Kate services to test if the GUI "screenshot" issue 
is related to Qt rendering or import hanging.
"""

import sys
from pathlib import Path

# Add Kate to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
    
    print("✓ PySide6 imports successful")
    
    app = QApplication(sys.argv)
    print("✓ QApplication created")
    
    # Create a simple test window
    window = QMainWindow()
    window.setWindowTitle("Kate Qt Test - NOT a screenshot!")
    window.setGeometry(100, 100, 400, 300)
    
    # Add a label to make it clear this is live GUI
    label = QLabel("This is a LIVE Qt window - not a screenshot!")
    label.setAlignment(Qt.AlignCenter)
    window.setCentralWidget(label)
    
    print("✓ Test window created")
    
    window.show()
    print("✓ Window shown - you should see a live Qt window")
    print("✓ If you see a screenshot instead of live GUI, the issue is Qt rendering")
    print("✓ If you see a live window, the issue is Kate's service imports")
    
    # Run for 3 seconds then exit
    import time
    time.sleep(3)
    
    print("✓ Qt test completed successfully")
    
except Exception as e:
    print(f"✗ Qt test failed: {e}")
    import traceback
    traceback.print_exc()