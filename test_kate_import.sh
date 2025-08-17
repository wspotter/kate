#!/bin/bash

echo "🧪 Testing Kate Import After pyqtSignal Fix"
echo "========================================="
echo ""

echo "📋 Testing fixed imports..."
poetry run python -c "
import sys
sys.path.insert(0, 'app')
try:
    print('Testing PySide6 imports...')
    from PySide6.QtCore import Signal, QObject
    print('✅ PySide6.QtCore imports work')
    
    print('Testing Kate core import...')
    from app.core.application import KateApplication
    print('✅ Kate core imports work')
    
    print('Testing fixed UI components...')
    from app.ui.components.chat_area import ChatArea
    print('✅ ChatArea import works')
    
    from app.ui.components.progress_indicators import DocumentIndexingProgressItem
    print('✅ Progress indicators import works')
    
    from app.ui.components.document_manager import DocumentManager
    print('✅ Document manager import works')
    
    print('')
    print('🎉 ALL IMPORTS SUCCESSFUL!')
    print('Kate is ready to launch!')
    
except Exception as e:
    print(f'❌ Import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

echo ""
echo "🚀 If successful, Kate can now be launched!"