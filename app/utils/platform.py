"""
Platform-specific utilities for Kate LLM Client.
"""

import sys
import platform
from typing import Dict, Any


def setup_platform() -> None:
    """
    Setup platform-specific configurations.
    """
    if sys.platform == "win32":
        # Windows-specific setup
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("kate.llm.client")
        except:
            pass
    elif sys.platform == "darwin":
        # macOS-specific setup
        pass
    else:
        # Linux-specific setup
        pass


def get_platform_info() -> Dict[str, Any]:
    """
    Get comprehensive platform information.
    
    Returns:
        Dictionary containing platform details
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.architecture(),
    }