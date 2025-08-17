#!/usr/bin/env python3
"""
Diagnostic test to isolate Kate's UI component rendering issues.
This will test each component individually to identify where the screenshot issue occurs.
"""

import sys

from loguru import logger
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

# Set up logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

def test_basic_window():
    """Test 1: Basic window creation"""
    logger.info("=== TEST 1: Basic Window Creation ===")
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Kate UI Test - Basic Window")
    window.setGeometry(100, 100, 400, 300)
    
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel("Basic Window Test - If you see this text, basic Qt works")
    layout.addWidget(label)
    
    window.setCentralWidget(central)
    window.show()
    
    logger.info("✓ Basic window created and shown")
    
    # Process events briefly
    app.processEvents()
    QTimer.singleShot(2000, lambda: (logger.info("Basic window test complete"), app.quit()))
    
    return app.exec()

def test_kate_imports():
    """Test 2: Test Kate component imports without initialization"""
    logger.info("=== TEST 2: Kate Component Imports ===")
    
    try:
        logger.info("Importing Kate components...")
        
        # Test imports one by one
        logger.info("Importing ConversationSidebar...")
        from app.ui.components.conversation_sidebar import ConversationSidebar
        logger.info("✓ ConversationSidebar imported")
        
        logger.info("Importing ChatArea...")
        from app.ui.components.chat_area import ChatArea
        logger.info("✓ ChatArea imported")
        
        logger.info("Importing AssistantPanel...")
        from app.ui.components.assistant_panel import AssistantPanel
        logger.info("✓ AssistantPanel imported")
        
        logger.info("All Kate components imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"Kate component import failed: {e}")
        return False

def test_kate_component_creation():
    """Test 3: Test individual Kate component creation"""
    logger.info("=== TEST 3: Kate Component Creation ===")
    
    if not test_kate_imports():
        logger.error("Cannot proceed - imports failed")
        return False
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    try:
        # Create a mock event bus
        from app.core.events import EventBus
        event_bus = EventBus()
        logger.info("✓ EventBus created")
        
        # Test each component individually
        window = QMainWindow()
        window.setWindowTitle("Kate UI Test - Component Creation")
        window.setGeometry(200, 200, 800, 600)
        
        # Test ConversationSidebar
        logger.info("Creating ConversationSidebar...")
        from app.ui.components.conversation_sidebar import ConversationSidebar
        sidebar = ConversationSidebar(event_bus)
        logger.info("✓ ConversationSidebar created successfully")
        
        # Test ChatArea  
        logger.info("Creating ChatArea...")
        from app.ui.components.chat_area import ChatArea
        chat_area = ChatArea(event_bus)
        logger.info("✓ ChatArea created successfully")
        
        # Test AssistantPanel
        logger.info("Creating AssistantPanel...")
        from app.ui.components.assistant_panel import AssistantPanel
        assistant_panel = AssistantPanel(event_bus)
        logger.info("✓ AssistantPanel created successfully")
        
        # Set just one component to test rendering
        window.setCentralWidget(chat_area)
        window.show()
        
        logger.info("Component creation test window shown")
        app.processEvents()
        
        QTimer.singleShot(3000, lambda: (logger.info("Component creation test complete"), app.quit()))
        
        return app.exec()
        
    except Exception as e:
        logger.error(f"Kate component creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_simplified_main_window():
    """Test 4: Simplified version of Kate's main window"""
    logger.info("=== TEST 4: Simplified Main Window ===")
    
    app = QApplication.instance() or QApplication(sys.argv)
    
    try:
        # Import necessary components
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QSplitter

        from app.core.events import EventBus
        from app.ui.components.assistant_panel import AssistantPanel
        from app.ui.components.chat_area import ChatArea
        from app.ui.components.conversation_sidebar import ConversationSidebar
        
        # Create components WITHOUT evaluation service setup
        logger.info("Creating simplified main window...")
        
        window = QMainWindow()
        window.setWindowTitle("Kate UI Test - Simplified Main Window")
        window.setGeometry(300, 300, 1200, 800)
        
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        event_bus = EventBus()
        
        # Create splitter and components
        splitter = QSplitter(Qt.Horizontal)
        
        logger.info("Creating components without evaluation service...")
        sidebar = ConversationSidebar(event_bus)
        chat_area = ChatArea(event_bus)
        assistant_panel = AssistantPanel(event_bus)
        
        splitter.addWidget(sidebar)
        splitter.addWidget(chat_area)
        splitter.addWidget(assistant_panel)
        
        main_layout.addWidget(splitter)
        window.setCentralWidget(central_widget)
        
        logger.info("Simplified main window setup complete, showing...")
        window.show()
        app.processEvents()
        
        logger.info("If this shows properly, the issue is in evaluation service setup")
        logger.info("If this shows screenshot, the issue is in basic component rendering")
        
        QTimer.singleShot(5000, lambda: (logger.info("Simplified main window test complete"), app.quit()))
        
        return app.exec()
        
    except Exception as e:
        logger.error(f"Simplified main window failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all diagnostic tests"""
    logger.info("Starting Kate UI component diagnostic tests...")
    
    # Test 1: Basic Qt functionality (should work)
    logger.info("Running Test 1...")
    test_basic_window()
    
    # Test 2: Import Kate components
    logger.info("Running Test 2...")
    if not test_kate_imports():
        logger.error("Cannot continue - component imports failed")
        return
    
    # Test 3: Create individual components
    logger.info("Running Test 3...")
    test_kate_component_creation()
    
    # Test 4: Simplified main window
    logger.info("Running Test 4...")
    test_simplified_main_window()
    
    logger.info("All diagnostic tests completed")

if __name__ == "__main__":
    main()