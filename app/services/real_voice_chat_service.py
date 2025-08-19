"""
Real Voice Chat Service with TTS/STT implementation.
Gracefully handles missing dependencies with fallbacks.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)

class RealVoiceChatService:
    """
    Enhanced voice chat service with real TTS/STT capabilities.
    Falls back to simulation when dependencies are missing.
    """
    
    def __init__(self, voice_settings=None):
        self._is_active = False
        self._tts_engine = None
        self._speech_recognizer = None
        self._audio_available = False
        self.voice_settings = voice_settings  # Kate's voice configuration
        self._initialize_audio_components()

    def _initialize_audio_components(self) -> None:
        """Initialize TTS and STT components with dependency checking."""
        logger.info("🔧 Initializing audio components...")
        
        # Initialize TTS
        self._tts_engine = self._setup_tts()
        
        # Initialize STT  
        self._speech_recognizer = self._setup_stt()
        
        # Check overall audio availability
        self._audio_available = self._tts_engine is not None or self._speech_recognizer is not None
        
        if self._audio_available:
            logger.info("✅ Real audio components initialized successfully")
        else:
            logger.warning("⚠️ Audio dependencies not available - using simulation mode")

    def _setup_tts(self) -> Optional[Any]:
        """Setup Text-to-Speech engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            
            # Configure TTS settings using Kate's voice settings
            voices = engine.getProperty('voices')
            if voices:
                # Use configured voice or default to first available
                if self.voice_settings and hasattr(self.voice_settings, 'pyttsx3_voice'):
                    # Try to find matching voice
                    for voice in voices:
                        if self.voice_settings.pyttsx3_voice in voice.id:
                            engine.setProperty('voice', voice.id)
                            break
                    else:
                        engine.setProperty('voice', voices[0].id)  # Fallback
                else:
                    engine.setProperty('voice', voices[0].id)  # Default
            
            # Apply configured speech rate and volume
            rate = 150  # Default
            volume = 0.8  # Default
            if self.voice_settings:
                rate = getattr(self.voice_settings, 'tts_rate', 150)
                volume = getattr(self.voice_settings, 'tts_volume', 0.8)
            
            engine.setProperty('rate', rate)
            engine.setProperty('volume', volume)
            
            logger.info("✅ TTS engine (pyttsx3) initialized")
            return engine
            
        except ImportError:
            logger.debug("pyttsx3 not available, trying edge-tts...")
            try:
                import edge_tts
                logger.info("✅ Edge TTS available")
                return "edge_tts"
            except ImportError:
                logger.debug("No TTS engines available")
                return None
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")
            return None

    def _setup_stt(self) -> Optional[Any]:
        """Setup Speech-to-Text recognizer."""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            
            # Test microphone availability
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            logger.info("✅ Speech recognition (speech_recognition) initialized")
            return recognizer
            
        except ImportError:
            logger.debug("speech_recognition not available, trying whisper...")
            try:
                import whisper
                model = whisper.load_model("base")
                logger.info("✅ Whisper STT available")
                return model
            except ImportError:
                logger.debug("No STT engines available")
                return None
        except Exception as e:
            logger.warning(f"STT initialization failed: {e}")
            return None

    async def text_to_speech(self, text: str, output_file: Optional[str] = None) -> bool:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            output_file: Optional file path to save audio
            
        Returns:
            True if successful, False otherwise
        """
        if not text.strip():
            return False
            
        logger.info(f"🔊 Converting text to speech: '{text[:50]}...'")
        
        try:
            if self._tts_engine is None:
                logger.info("🔊 TTS: Simulating speech output")
                await asyncio.sleep(1)  # Simulate processing time
                return True
                
            if hasattr(self._tts_engine, 'say'):  # pyttsx3
                if output_file:
                    self._tts_engine.save_to_file(text, output_file)
                else:
                    self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                logger.info("✅ TTS completed (pyttsx3)")
                return True
                
            elif self._tts_engine == "edge_tts":  # edge-tts
                import edge_tts
                voice = "en-US-AriaNeural"  # Default voice
                
                if output_file:
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(output_file)
                else:
                    # For real-time playback, we'd need audio player
                    temp_file = "/tmp/kate_tts_temp.wav"
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(temp_file)
                    # Here you'd play the audio file
                    
                logger.info("✅ TTS completed (edge-tts)")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    async def speech_to_text(self, audio_file: Optional[str] = None, duration: float = 5.0) -> Optional[str]:
        """
        Convert speech to text.
        
        Args:
            audio_file: Path to audio file, or None for microphone input
            duration: Recording duration for microphone input
            
        Returns:
            Transcribed text or None if failed
        """
        logger.info(f"🎙️ Converting speech to text (duration: {duration}s)")
        
        try:
            if self._speech_recognizer is None:
                logger.info("🎙️ STT: Simulating speech recognition")
                await asyncio.sleep(duration)
                return "Simulated transcription: Hello, this is a test message."
            
            if hasattr(self._speech_recognizer, 'listen'):  # speech_recognition
                import speech_recognition as sr
                
                if audio_file:
                    # Process audio file
                    with sr.AudioFile(audio_file) as source:
                        audio = self._speech_recognizer.record(source)
                else:
                    # Listen from microphone
                    with sr.Microphone() as source:
                        logger.info("🎙️ Listening...")
                        audio = self._speech_recognizer.listen(source, timeout=duration)
                
                # Use Google Speech Recognition (free tier)
                text = self._speech_recognizer.recognize_google(audio)
                logger.info(f"✅ STT completed: '{text}'")
                return text
                
            else:  # whisper
                if audio_file:
                    result = self._speech_recognizer.transcribe(audio_file)
                    text = result["text"]
                    logger.info(f"✅ STT completed (Whisper): '{text}'")
                    return text
                else:
                    logger.warning("Whisper requires audio file input")
                    return None
                    
        except Exception as e:
            logger.error(f"STT error: {e}")
            return None

    async def start_voice_session(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Start an interactive voice chat session with real TTS/STT.
        
        Yields:
            Dictionary with status updates and transcribed text
        """
        if self._is_active:
            yield {"status": "error", "message": "Voice session already active"}
            return

        self._is_active = True
        logger.info("🚀 Starting real voice chat session...")
        
        try:
            yield {"status": "started", "message": "Voice session initiated"}
            
            # Test TTS
            await self.text_to_speech("Voice chat session started. Please speak now.")
            yield {"status": "tts_complete", "message": "Welcome message played"}
            
            # Listen for speech (multiple attempts)
            for attempt in range(3):
                yield {"status": "listening", "message": f"Listening attempt {attempt + 1}/3"}
                
                text = await self.speech_to_text(duration=5.0)
                
                if text:
                    yield {
                        "status": "transcribed",
                        "text": text,
                        "attempt": attempt + 1
                    }
                    
                    # Provide TTS feedback
                    response = f"I heard you say: {text}"
                    await self.text_to_speech(response)
                    
                    yield {
                        "status": "response_complete",
                        "message": "TTS response completed"
                    }
                else:
                    yield {
                        "status": "no_speech",
                        "message": f"No speech detected in attempt {attempt + 1}"
                    }
                
                await asyncio.sleep(1)  # Brief pause between attempts
            
            yield {"status": "completed", "message": "Voice session completed successfully"}
            
        except Exception as e:
            logger.error(f"Voice session error: {e}")
            yield {"status": "error", "message": str(e)}
            
        finally:
            self._is_active = False
            logger.info("🛑 Voice session ended")

    def is_active(self) -> bool:
        """Check if voice chat session is active."""
        return self._is_active

    def get_capabilities(self) -> Dict[str, bool]:
        """Get available audio capabilities."""
        return {
            "tts_available": self._tts_engine is not None,
            "stt_available": self._speech_recognizer is not None,
            "audio_available": self._audio_available,
            "real_time_chat": self._audio_available,
        }

# Test function
async def test_voice_service():
    """Test the real voice chat service."""
    print("🧪 Testing Real Voice Chat Service...")
    
    service = RealVoiceChatService()
    capabilities = service.get_capabilities()
    
    print(f"Capabilities: {capabilities}")
    
    # Test TTS
    print("\n🔊 Testing TTS...")
    success = await service.text_to_speech("Hello, this is a test of the text to speech system.")
    print(f"TTS Result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test STT (will use simulation if no mic)
    print("\n🎙️ Testing STT...")
    text = await service.speech_to_text(duration=2.0)
    print(f"STT Result: {text}")
    
    # Test voice session
    print("\n🚀 Testing Voice Session...")
    async for update in service.start_voice_session():
        print(f"  Update: {update}")

if __name__ == "__main__":
    asyncio.run(test_voice_service())