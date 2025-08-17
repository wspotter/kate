"""
Main entry point for Kate LLM Client.

This module provides the main application entry point and handles:
- Application initialization
- Command-line argument parsing
- Platform-specific setup
- Error handling and crash reporting
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.application import KateApplication
from app.core.config import get_settings
from app.utils.logging import setup_logging
from app.utils.platform import get_platform_info, setup_platform


def setup_qt_application() -> QApplication:
    """
    Setup and configure the Qt application.
    
    Returns:
        Configured QApplication instance
    """
    logger.info("Setting up Qt application...")
    
    # Set Qt environment variables for proper rendering
    os.environ['QT_QPA_PLATFORMTHEME'] = 'qt5ct'
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
    os.environ['QT_SCALE_FACTOR'] = '1'
    os.environ['QT_SCREEN_SCALE_FACTORS'] = '1'
    logger.info("Qt environment variables set for proper rendering")
    
    # Set application attributes before creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    
    logger.info("Qt attributes set - High DPI scaling, pixmaps, and OpenGL context sharing enabled")
    
    # Create Qt application
    app = QApplication(sys.argv)
    logger.info("QApplication instance created successfully")
    
    # Set application metadata
    app.setApplicationName("Kate")
    app.setApplicationDisplayName("Kate LLM Client")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Kate Team")
    app.setOrganizationDomain("kate-llm.com")
    
    logger.info("Qt application metadata configured")
    
    # Set application icon
    # TODO: Add actual icon file
    # icon_path = Path(__file__).parent / "resources" / "icons" / "kate.ico"
    # if icon_path.exists():
    #     app.setWindowIcon(QIcon(str(icon_path)))
    
    return app


async def main_async() -> None:
    """
    Main async entry point for the application.
    """
    try:
        # Setup platform-specific configurations
        setup_platform()
        
        # Initialize and start Kate application
        kate_app = KateApplication()
        await kate_app.startup()
        
        # Setup graceful shutdown handlers
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating shutdown...")
            asyncio.create_task(kate_app.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Keep the application running until shutdown
        while not kate_app.is_shutting_down():
            await asyncio.sleep(0.1)
            
        logger.info("Kate application shutdown complete")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in Kate application: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


def main() -> None:
    """
    Main entry point for the Kate application.
    
    This function:
    1. Sets up logging
    2. Creates the Qt application
    3. Initializes and runs the Kate application
    4. Handles graceful shutdown
    """
    try:
        # Setup logging first
        setup_logging()
        logger.info("Starting Kate LLM Client...")
        
        # Log platform information
        platform_info = get_platform_info()
        logger.info(f"Platform: {platform_info}")
        
        # Create Qt application
        qt_app = setup_qt_application()
        
        # Create and run the async event loop
        if sys.platform == "win32":
            # Use ProactorEventLoop on Windows for better compatibility
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Start the async main function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Setup Qt integration with asyncio
        timer = QTimer()
        timer.timeout.connect(lambda: None)  # Keep the Qt event loop active
        timer.start(10)  # 10ms intervals
        
        try:
            # Run the main async function
            loop.run_until_complete(main_async())
        finally:
            # Cleanup
            timer.stop()
            loop.close()
            qt_app.quit()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


@click.group()
@click.version_option(version="1.0.0", prog_name="Kate")
def cli():
    """Kate LLM Client - A modern desktop client for multiple LLM providers."""
    pass


@cli.command()
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--config", type=click.Path(), help="Path to configuration file")
@click.option("--log-level", default="INFO", help="Set logging level")
def run(debug: bool, config: Optional[str], log_level: str):
    """Run the Kate LLM Client."""
    if debug:
        os.environ["KATE_DEBUG"] = "1"
        
    if config:
        os.environ["KATE_CONFIG"] = config
        
    if log_level:
        os.environ["KATE_LOG_LEVEL"] = log_level
        
    main()


@cli.command()
def dev():
    """Run Kate in development mode."""
    os.environ["KATE_DEBUG"] = "1"
    os.environ["KATE_LOG_LEVEL"] = "DEBUG"
    main()


@cli.command()
def test():
    """Run Kate tests."""
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/"], cwd=project_root)
    sys.exit(result.returncode)


@cli.command()
def info():
    """Show Kate system information."""
    setup_logging()
    
    from app.utils.system import get_system_info
    
    info = get_system_info()
    click.echo("Kate LLM Client System Information:")
    click.echo(f"Version: {info.get('version', 'Unknown')}")
    click.echo(f"Platform: {info.get('platform', 'Unknown')}")
    click.echo(f"Python: {info.get('python_version', 'Unknown')}")
    click.echo(f"Qt: {info.get('qt_version', 'Unknown')}")
    click.echo(f"Config Dir: {info.get('config_dir', 'Unknown')}")
    click.echo(f"Data Dir: {info.get('data_dir', 'Unknown')}")


def dev_main() -> None:
    """Development entry point."""
    os.environ["KATE_DEBUG"] = "1"
    os.environ["KATE_LOG_LEVEL"] = "DEBUG"
    main()


def test_main() -> None:
    """Test entry point."""
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/"], cwd=project_root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    # If run directly, start the main application
    if len(sys.argv) == 1:
        main()
    else:
        # Otherwise, use the CLI interface
        cli()