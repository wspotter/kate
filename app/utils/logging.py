"""
Logging configuration for Kate LLM Client.
"""

import os
import sys
from pathlib import Path
from loguru import logger
import platformdirs


def setup_logging() -> None:
    """
    Setup application logging with loguru.
    
    Configures both console and file logging with appropriate levels
    and formatting for the Kate application.
    """
    # Remove default handler
    logger.remove()
    
    # Get log level from environment or default to INFO
    log_level = os.getenv("KATE_LOG_LEVEL", "INFO").upper()
    debug_mode = os.getenv("KATE_DEBUG", "").lower() in ("1", "true", "yes")
    
    if debug_mode:
        log_level = "DEBUG"
    
    # Console logging with color
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=debug_mode,
        diagnose=debug_mode
    )
    
    # File logging
    try:
        logs_dir = Path(platformdirs.user_log_dir("Kate"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "kate.log"
        
        logger.add(
            log_file,
            level="DEBUG" if debug_mode else "INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="gz",
            backtrace=debug_mode,
            diagnose=debug_mode
        )
        
        logger.info(f"Logging to file: {log_file}")
        
    except Exception as e:
        logger.warning(f"Failed to setup file logging: {e}")
    
    logger.info(f"Logging configured with level: {log_level}")