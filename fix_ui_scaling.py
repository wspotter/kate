#!/usr/bin/env python3
"""
UI Scaling Fix for Kate LLM Client

This script adjusts the UI scaling settings to fix issues with widget sizing and layout.
Run this script before starting Kate to apply the fixes.
"""

import os
import subprocess
import sys
from pathlib import Path

# Set environment variables to fix scaling
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'  # Enable auto scaling
os.environ['QT_SCALE_FACTOR'] = '1.0'  # Set base scale factor
os.environ['QT_SCREEN_SCALE_FACTORS'] = ''  # Clear any per-screen factors
os.environ['QT_FONT_DPI'] = '96'  # Set standard DPI for fonts

# Path to Kate main.py
kate_dir = Path(__file__).parent
kate_main = kate_dir / "app" / "main.py"

def apply_temporary_fixes():
    """Apply temporary fixes to the codebase to address UI scaling issues."""
    
    # Fix main window sizing
    main_window_path = kate_dir / "app" / "ui" / "main_window.py"
    if main_window_path.exists():
        with open(main_window_path, 'r') as f:
            content = f.read()
        
        # Increase minimum window size
        content = content.replace(
            "self.setMinimumSize(1200, 800)",
            "self.setMinimumSize(1280, 900)"
        )
        
        # Increase initial window size
        content = content.replace(
            "self.resize(1400, 900)",
            "self.resize(1600, 1000)"
        )
        
        # Adjust panel widths
        content = content.replace(
            "self.conversation_sidebar.setMinimumWidth(250)",
            "self.conversation_sidebar.setMinimumWidth(280)"
        )
        
        content = content.replace(
            "self.assistant_panel.setMinimumWidth(280)",
            "self.assistant_panel.setMinimumWidth(320)"
        )
        
        content = content.replace(
            "self.assistant_panel.setMaximumWidth(350)",
            "self.assistant_panel.setMaximumWidth(400)"
        )
        
        # Adjust splitter proportions
        content = content.replace(
            "self.main_splitter.setSizes([300, 700, 300])",
            "self.main_splitter.setSizes([320, 800, 320])"
        )
        
        with open(main_window_path, 'w') as f:
            f.write(content)
        print(f"✅ Applied fixes to {main_window_path}")
    
    # Fix assistant panel sizing
    assistant_panel_path = kate_dir / "app" / "ui" / "components" / "assistant_panel.py"
    if assistant_panel_path.exists():
        with open(assistant_panel_path, 'r') as f:
            content = f.read()
        
        # Increase description height
        content = content.replace(
            "desc_label.setMaximumHeight(60)",
            "desc_label.setMaximumHeight(80)"
        )
        
        # Increase combo box height
        content = content.replace(
            "self.assistant_combo.setMinimumHeight(36)",
            "self.assistant_combo.setMinimumHeight(40)"
        )
        
        # Adjust margins and spacing
        content = content.replace(
            "layout.setContentsMargins(8, 8, 8, 8)",
            "layout.setContentsMargins(10, 10, 10, 10)"
        )
        
        content = content.replace(
            "layout.setSpacing(12)",
            "layout.setSpacing(14)"
        )
        
        with open(assistant_panel_path, 'w') as f:
            f.write(content)
        print(f"✅ Applied fixes to {assistant_panel_path}")
    
    # Fix settings window sizing
    settings_window_path = kate_dir / "app" / "ui" / "components" / "settings_window.py"
    if settings_window_path.exists():
        with open(settings_window_path, 'r') as f:
            content = f.read()
        
        # Increase window size
        content = content.replace(
            "self.resize(600, 500)",
            "self.resize(700, 600)"
        )
        
        # Increase text area heights
        content = content.replace(
            "self.test_text.setMaximumHeight(80)",
            "self.test_text.setMaximumHeight(100)"
        )
        
        content = content.replace(
            "self.agent_description.setMaximumHeight(120)",
            "self.agent_description.setMaximumHeight(150)"
        )
        
        with open(settings_window_path, 'w') as f:
            f.write(content)
        print(f"✅ Applied fixes to {settings_window_path}")

def run_kate():
    """Run Kate with the fixed scaling settings."""
    print("🚀 Starting Kate with fixed UI scaling...")
    
    # Check if Poetry is available
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        # Run with Poetry
        subprocess.run(["poetry", "run", "python", "-m", "app.main"], cwd=kate_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to direct Python execution
        subprocess.run([sys.executable, str(kate_main)], cwd=kate_dir)

if __name__ == "__main__":
    print("🔧 Kate UI Scaling Fix")
    print("=====================")
    
    choice = input("Apply temporary code fixes to improve UI scaling? (y/n): ").strip().lower()
    if choice == 'y':
        apply_temporary_fixes()
    
    run_kate()