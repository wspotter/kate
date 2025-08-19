---
description: Repository Information Overview
alwaysApply: true
---

# Kate LLM Client Information

## Summary

Kate is a modern, professional desktop LLM client built with Python and PySide6. It features a sleek 3-column interface supporting multiple AI providers with enterprise-grade architecture. The application provides a unified interface for interacting with various LLM providers including OpenAI, Anthropic, Ollama, Google, Groq, Cohere, and Mistral AI.

## Structure

- **app/**: Main application source code
  - **core/**: Core framework components
  - **database/**: Database layer with SQLAlchemy
  - **ui/**: User interface components
  - **providers/**: LLM provider integrations
  - **services/**: Business logic services
  - **themes/**: Theme system
  - **utils/**: Utility functions
- **tests/**: Test suite
- **docs/**: Documentation
- **build/**: Build configuration

## Language & Runtime

**Language**: Python
**Version**: >=3.9, <3.13
**Build System**: Poetry
**Package Manager**: Poetry

## Dependencies

**Main Dependencies**:

- PySide6 (>=6.0.0): Qt-based GUI framework
- SQLAlchemy (>=2.0.0): Database ORM
- Pydantic (>=2.0.0): Data validation
- qasync (>=0.24.0): Async/Qt integration
- httpx (>=0.24.0): Async HTTP client
- sentence-transformers (>=2.0.0): Embedding models
- torch (>=2.0.0): Deep learning framework
- SpeechRecognition (>=3.10.0): Voice processing
- edge-tts (>=6.1.0): Text-to-speech

**Development Dependencies**:

- pytest: Testing framework
- black: Code formatting
- ruff: Linting
- mypy: Type checking
- pre-commit: Automated quality checks

## Build & Installation

```bash
# Install dependencies
poetry install

# Run the application
poetry run kate-llm
# or
poetry run python -m app.main

# Optional web server
poetry run python -m app.web_server
```

## Testing

**Framework**: pytest
**Test Location**: tests/
**Naming Convention**: test\_\*.py
**Configuration**: tests/pytest.ini
**Run Command**:

```bash
poetry run pytest
poetry run pytest --cov=app --cov-report=html
poetry run pytest -m "unit"
poetry run pytest -m "integration"
```

## Architecture

Kate follows an enterprise software architecture pattern:

- **Core Application Framework**: Async lifecycle management, event-driven communication
- **Database Layer**: SQLAlchemy 2.0 for conversation management and user preferences
- **LLM Provider System**: Unified interface with streaming support
- **Modern UI**: PySide6/Qt6-based 3-column layout with theme system
- **Advanced Features**: Voice processing, document handling, search capabilities

## Configuration

Kate uses configuration files stored in `~/.config/Kate/settings.json` and supports multiple predefined assistants loaded from `app/config/assistants.json`. The application also supports environment variables for configuration:

- KATE_DEBUG: Enable debug mode
- KATE_LOG_LEVEL: Set logging level
- KATE_WATCH_ASSISTANTS: Enable hot-reload for assistant configurations
- KATE_CONFIG: Path to configuration file
