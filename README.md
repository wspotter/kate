# Kate LLM Desktop Client

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, professional desktop LLM client built with Python and PySide6. Features a sleek 3-column interface supporting multiple AI providers with enterprise-grade architecture.

> Recovery Status (Aug 2025): Core GUI + async stack stabilized (qasync + SQLAlchemy async). Timezone-aware timestamps enforced (`now_utc()`), legacy compatibility layer (model + settings alias shims) active pending formal deprecation. See `docs/adr/ADR-0001-timezone-alias-strategy.md`.

## 🎯 Key Features

### Multi-Provider LLM Support

- **OpenAI**: GPT-4, GPT-3.5-turbo with streaming
- **Anthropic**: Claude 3 Sonnet, Haiku, Opus
- **Local Models**: Ollama integration
- **Google**: Gemini Pro support
- **Additional**: Groq, Cohere, Mistral AI

### Modern Desktop Interface

- **3-Column Layout**: Conversation sidebar, chat area, assistant panel
- **Dark/Light Themes**: Professional Qt-based styling
- **Responsive Design**: Adjustable panels with saved layouts
- **Message Bubbles**: Rich text formatting with copy functionality

### Enterprise Architecture

- **Async Framework**: Full async/await patterns for responsiveness
- **Type Safety**: Pydantic models with comprehensive validation
- **Event-Driven**: Decoupled components via EventBus
- **Database**: SQLAlchemy 2.0 with async support
- **Plugin System**: Extensible architecture with security validation

### Advanced Capabilities

- **Voice Processing**: Speech-to-text and text-to-speech
- **Document Handling**: PDF, image, and OCR processing
- **Search**: Full-text and semantic search across conversations
- **Translation**: Multi-language support
- **Auto-Updates**: Secure update mechanism with rollback
- **Web API (Minimal)**: FastAPI health + WebSocket channel (optional background server)

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd kate

# Install dependencies
poetry install

# Run the application (preferred script name avoids conflict with KDE's 'kate' editor)
poetry run kate-llm

# Alternative explicit module invocation (qasync-integrated GUI)
poetry run python -m app.main

# (Optional) Start minimal web API (in another shell)
poetry run python -m app.web_server
```

### Configuration

Kate uses intelligent defaults but can be customized via `~/.config/Kate/settings.json`:

```json
{
  "llm_providers": {
    "openai": {
      "api_key": "your-openai-key",
      "model": "gpt-4"
    },
    "anthropic": {
      "api_key": "your-anthropic-key",
      "model": "claude-3-sonnet-20240229"
    }
  },
  "ui": {
    "theme": "Kate Dark",
    "font_size": 11
  }
}
```

## 📖 Documentation

### User Guide

- [Getting Started](docs/user/getting-started.md)
- [Configuration](docs/user/configuration.md)
- [LLM Providers](docs/user/providers.md)
- [Keyboard Shortcuts](docs/user/shortcuts.md)

### Developer Documentation

- [Architecture Overview](docs/developer/architecture.md)
- [API Reference](docs/developer/api.md)
- [Plugin Development](docs/developer/plugins.md)
- [Contributing](docs/developer/contributing.md)

### Assistant Definitions & Service

Kate supports multiple predefined assistants (personas) that can be selected in the right-hand Assistant Panel. These are loaded by the `AssistantService` from a JSON configuration file (currently `app/config/assistants.json` or a user override path if implemented later).

Example `assistants.json` structure:

```jsonc
{
  "general": {
    "name": "General Assistant",
    "description": "Balanced assistant for everyday tasks",
    "provider": "openai",
    "model": "gpt-4-turbo"
  },
  "coding": {
    "name": "Code Assistant",
    "description": "Focused on programming help and code generation",
    "provider": "openai",
    "model": "gpt-4o"
  }
}
```

Key points:

- Missing or invalid config falls back to a single built-in general assistant.
- The service is created once by the application and injected into UI components to avoid duplicate loading.
- Hot reload (optional) can be enabled by setting `KATE_WATCH_ASSISTANTS=1` before launching; the panel will auto-reload when the file's mtime changes.
- Future enhancements may add per-assistant model parameters (temperature, max tokens) and UI editing.

Environment example:

```bash
export KATE_WATCH_ASSISTANTS=1
poetry run python -m app.main
```

If the file changes and the env var is set, the Assistant Panel will refresh its list and re-select the current assistant if still present.

### Local Ollama Auto-Connect (Experimental)

If a local [Ollama](https://ollama.ai) server is running on `http://127.0.0.1:11434`, the application now attempts a best-effort connection during startup:

- Discovers available local models via `/api/tags`.
- Populates an internal `available_models` list and sets `selected_model` (prefers a model starting with `mistral`, otherwise the first).
- Updates the status bar with connection and selected model info.
- Fails gracefully (UI still loads) if the server is unreachable.

You can refresh models later programmatically (future UI button planned). This provides immediate local LLM functionality for chat without external API keys.

Example (ensure Ollama running):

```bash
ollama run mistral  # pull model if first time
poetry run python -m app.main
```

If an assistant definition specifies `{"provider": "ollama", "model": "mistral"}`, selecting that assistant will auto-switch to the matching local model when found.

> Note: This feature is marked experimental; advanced provider management UI (connect/disconnect, model refresh) will follow.

### Session Progress (Development Snapshot)

Current implementation milestones from the active refactor session:

- Added centralized `AssistantService` with hot-reload (optional via `KATE_WATCH_ASSISTANTS`).
- Repaired indentation / structural issues in `AssistantPanel` and `MainWindow`.
- Integrated evaluation metrics panel & dashboard wiring (basic metrics display).
- Introduced experimental Ollama auto-connect & model selection logic.
- Added placeholder indexing status hook (future document processing integration).
- Improved model settings propagation (temperature, max tokens, top_p, streaming flag).
- Replaced naive UTC defaults with timezone-aware `now_utc()` helper across models.
- Added legacy compatibility alias layer for database models & settings (temporary shim documented).
- Implemented graceful shutdown handling around qasync event loop (suppressed benign loop-stop RuntimeError).

Upcoming (planned next passes):

- UI control to refresh local Ollama models.
- Persist last selected assistant & model between sessions.
- Refine type hints to reduce mypy noise around dynamic Qt attributes.
- Expose evaluation export & details dialog polish.

This progress list is transient and intended to help collaborators track in-flight functionality work.

## 🏗️ Architecture

Kate follows enterprise software patterns for maintainability and extensibility:

```
Kate LLM Client
├── Core Application Framework
│   ├── Async Lifecycle Management
│   ├── Event-Driven Communication
│   └── Configuration Management
├── Database Layer (SQLAlchemy 2.0)
│   ├── Conversation Management
│   ├── Message History
│   └── User Preferences
├── LLM Provider System
│   ├── Unified Interface
│   ├── Streaming Support
│   └── Error Handling
├── Modern UI (PySide6/Qt6)
│   ├── 3-Column Layout
│   ├── Theme System
│   └── Responsive Design
└── Advanced Features
    ├── Plugin Architecture
    ├── Voice Processing
    ├── Document Processing
    └── Auto-Updates
```

## 🧪 Testing

Kate includes comprehensive testing coverage:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test categories
poetry run pytest -m "unit"
poetry run pytest -m "integration"
poetry run pytest -m "ui"
```

## 📦 Building & Distribution

### Development Build

```bash
# Preferred (console script)
poetry run kate-llm

# Or directly (module form)
poetry run python -m app.main
```

### Production Build

```bash
# PyInstaller (recommended)
poetry run pyinstaller build/kate.spec

# Alternative: cx_Freeze
poetry run python build/setup_cx.py build

# Alternative: Nuitka
poetry run python -m nuitka --onefile app/main.py
```

## 🔧 Development

### Project Structure

```
kate/
├── app/                    # Application source code
│   ├── core/              # Core framework
│   ├── database/          # Database layer
│   ├── ui/                # User interface
│   ├── providers/         # LLM providers
│   ├── services/          # Business logic
│   ├── themes/            # Theme system
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── docs/                  # Documentation
├── build/                 # Build configuration
└── pyproject.toml         # Project configuration
```

### Code Quality

- **Type Checking**: MyPy with strict configuration
- **Linting**: Ruff for fast, modern Python linting
- **Formatting**: Black for consistent code style
- **Pre-commit**: Automated quality checks

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run quality checks: `poetry run pre-commit run --all-files`
5. Submit a pull request

## 📋 Roadmap

### Current Version (1.0.0)

- ✅ Core 3-column UI
- ✅ Multi-provider LLM support
- ✅ Conversation management
- ✅ Theme system
- ✅ Plugin architecture

### Planned Features

- [ ] Advanced RAG integration (post-stabilization re-validation underway)
- [ ] Team collaboration features
- [ ] Mobile companion app
- [ ] Cloud synchronization
- [ ] Advanced analytics

## 🐛 Troubleshooting

### Common Issues

**GUI not displaying on Linux:**

```bash
export DISPLAY=:0
poetry run kate-llm
```

**Command name conflict (Linux/KDE):**
If you have the KDE text editor `kate` installed, its executable name conflicts with this project's original script entry (`kate`). Use the provided alternate script name instead:

```bash
poetry run kate-llm
```

If installed globally (not recommended for dev) ensure your PATH does not shadow the KDE editor or invoke via `python -m app.main`.

**Permission errors:**

```bash
# Ensure proper permissions
chmod +x app/main.py
```

**Missing dependencies:**

```bash
# Reinstall dependencies
poetry install --no-cache
```

### Getting Help

- **Issues**: [GitHub Issues](issues-url)
- **Discussions**: [GitHub Discussions](discussions-url)
- **Documentation**: [Full Documentation](docs-url)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PySide6](https://wiki.qt.io/Qt_for_Python) for the modern desktop UI
- Database powered by [SQLAlchemy 2.0](https://www.sqlalchemy.org/)
- Configuration management via [Pydantic](https://pydantic-docs.helpmanual.io/)
- Dependency management with [Poetry](https://python-poetry.org/)

---

**Kate LLM Client** - Professional desktop AI assistant for power users and enterprises.
