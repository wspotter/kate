#!/usr/bin/env python3
"""
Qt Display Test - Validate Qt/PySide6 GUI functionality
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from loguru import logger
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QLabel,
        QMainWindow,
        QVBoxLayout,
        QWidget,
    )
    
    print("✅ PySide6 imports successful")
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Qt Display Test")
            self.setGeometry(100, 100, 400, 300)
            
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Create layout
            layout = QVBoxLayout(central_widget)
            
            # Add test labels
            label1 = QLabel("✅ Qt Display Test - SUCCESS!")
            label1.setAlignment(Qt.AlignCenter)
            label1.setStyleSheet("font-size: 18px; color: green; font-weight: bold;")
            
            label2 = QLabel("If you can see this window, Qt GUI is working properly")
            label2.setAlignment(Qt.AlignCenter)
            
            label3 = QLabel("This window will close automatically in 5 seconds")
            label3.setAlignment(Qt.AlignCenter)
            label3.setStyleSheet("color: blue;")
            
            layout.addWidget(label1)
            layout.addWidget(label2)
            layout.addWidget(label3)
            
            # Auto-close timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.close)
            self.timer.start(5000)  # 5 seconds
            
            print("✅ Test window created and displayed")
            
    def main():
        print("🔧 Starting Qt Display Test...")
        
        # Check display environment
        display = os.environ.get('DISPLAY')
        wayland_display = os.environ.get('WAYLAND_DISPLAY')
        
        print(f"DISPLAY: {display}")
        print(f"WAYLAND_DISPLAY: {wayland_display}")
        
        if not display and not wayland_display:
            print("⚠️  WARNING: No display server detected (DISPLAY/WAYLAND_DISPLAY not set)")
            print("This may indicate a headless environment")
        
        # Create Qt application
        app = QApplication([])
        print("✅ QApplication created")
        
        # Create test window
        window = TestWindow()
        window.show()
        print("✅ Test window shown")
        
        # Run event loop
        print("🚀 Starting Qt event loop...")
        result = app.exec()
        print(f"✅ Qt event loop exited with code: {result}")
        
        return result

    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)