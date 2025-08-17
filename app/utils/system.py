"""
System information utilities for Kate LLM Client.
"""

import sys
import platform
from pathlib import Path
from typing import Dict, Any
import platformdirs
from PySide6.QtCore import QT_VERSION_STR
from PySide6 import __version__ as pyside_version


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for Kate.
    
    Returns:
        Dictionary containing system details
    """
    return {
        "version": "1.0.0",
        "platform": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "qt_version": QT_VERSION_STR,
        "pyside_version": pyside_version,
        "config_dir": str(Path(platformdirs.user_config_dir("Kate"))),
        "data_dir": str(Path(platformdirs.user_data_dir("Kate"))),
        "cache_dir": str(Path(platformdirs.user_cache_dir("Kate"))),
        "logs_dir": str(Path(platformdirs.user_log_dir("Kate"))),
    }