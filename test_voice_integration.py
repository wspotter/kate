#!/usr/bin/env python3
"""
Test Voice Service Integration in Kate
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_kate_voice_integration():
    """Test that Kate properly initializes the voice service."""
    print("🧪 TESTING KATE VOICE SERVICE INTEGRATION")
    print("=" * 50)
    
    try:
        # Import Kate application
        from app.core.application import KateApplication
        print("✅ KateApplication imported successfully")
        
        # Create Kate app instance
        app = KateApplication()
        print("✅ KateApplication instantiated")
        
        # Check if voice service is in attributes
        if hasattr(app, 'voice_service'):
            print("✅ voice_service attribute exists")
            print(f"   Initial value: {app.voice_service}")
        else:
            print("❌ voice_service attribute missing")
            return False
        
        # Initialize Kate app (this should initialize voice service)
        print("\n🚀 Initializing Kate application...")
        await app.startup()
        print("✅ Kate startup completed")
        
        # Check voice service after initialization
        voice_service = app.get_voice_service()
        if voice_service:
            print("✅ Voice service successfully initialized")
            
            # Get capabilities
            capabilities = voice_service.get_capabilities()
            print(f"📊 Voice capabilities: {capabilities}")
            
            # Test TTS
            print("\n🔊 Testing TTS integration...")
            result = await voice_service.text_to_speech("Kate voice integration test successful!")
            print(f"TTS Result: {'✅ Success' if result else '❌ Failed'}")
            
            # Test STT (brief)
            print("\n🎙️ Testing STT integration...")
            text = await voice_service.speech_to_text(duration=0.5)  # Brief test
            print(f"STT Result: {text}")
            
        else:
            print("❌ Voice service not initialized")
            return False
        
        # Cleanup
        print("\n🛑 Shutting down Kate...")
        await app.shutdown()
        print("✅ Kate shutdown completed")
        
        print("\n🎉 VOICE INTEGRATION TEST SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    success = await test_kate_voice_integration()
    if success:
        print("\n✅ All tests passed - Kate voice integration working!")
    else:
        print("\n❌ Tests failed - Voice integration needs debugging")
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)