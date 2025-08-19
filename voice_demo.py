#!/usr/bin/env python3
"""
🎤 Kate Voice Demo - Your 3 Groovy Voice Options
A standalone demonstration of the simplified voice interface you requested
"""

import sys
import time

import pyttsx3


def demo_voice_styles():
    """Demonstrate the 3 groovy voice styles with actual audio output."""
    
    print("🎙️  Kate Voice Assistant - Demo")
    print("=" * 50)
    print("You requested 3 groovy options instead of complex settings.")
    print("Here are your voice style choices:\n")
    
    try:
        engine = pyttsx3.init()
        
        # Voice styles configuration (matching your SimpleVoiceWidget)
        voice_styles = {
            "1": {
                "name": "🌟 Natural & Friendly",
                "description": "Warm, conversational tone perfect for everyday chats",
                "rate": 180,
                "volume": 0.9
            },
            "2": {
                "name": "💼 Professional & Clear", 
                "description": "Crisp, authoritative voice ideal for business",
                "rate": 160,
                "volume": 0.8
            },
            "3": {
                "name": "⚡ Energetic & Fun",
                "description": "Upbeat, enthusiastic tone that brings energy",
                "rate": 200,
                "volume": 1.0
            }
        }
        
        # Display options
        for key, style in voice_styles.items():
            print(f"{key}. {style['name']}")
            print(f"   {style['description']}")
            print()
        
        # Get user choice
        while True:
            choice = input("Choose your groovy voice style (1-3) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Thanks for trying Kate's voice options!")
                break
                
            if choice in voice_styles:
                style = voice_styles[choice]
                print(f"\n🎵 Testing: {style['name']}")
                
                # Configure engine with style settings
                engine.setProperty('rate', style['rate'])
                engine.setProperty('volume', style['volume'])
                
                # Test message
                test_message = "Hey there! I'm Kate, your AI assistant. How can I help you today?"
                
                print(f"💬 Speaking: '{test_message}'")
                engine.say(test_message)
                engine.runAndWait()
                
                print("✅ Voice test complete!")
                print("-" * 30)
                
            else:
                print("❌ Please choose 1, 2, 3, or 'q' to quit")
                
    except Exception as e:
        print(f"❌ Voice engine error: {e}")
        print("💡 Make sure your system has audio output enabled")
        return False
        
    return True

def main():
    """Main demo function."""
    print("🚀 Starting Kate Voice Demo...")
    print("Note: This demonstrates the NEW simplified voice interface")
    print("that replaces the complex 4-tab settings you didn't like.\n")
    
    # Check if TTS is available
    try:
        engine = pyttsx3.init()
        print("✅ TTS engine ready")
        engine.stop()
    except Exception as e:
        print(f"❌ TTS engine not available: {e}")
        print("💡 Install espeak: sudo apt install espeak espeak-data")
        return
    
    # Run demo
    demo_voice_styles()

if __name__ == "__main__":
    main()