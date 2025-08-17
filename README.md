# Kate LLM Desktop Client

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, professional desktop LLM client built with Python and PySide6. Features a sleek 3-column interface supporting multiple AI providers with enterprise-grade architecture.

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

# Run the application
poetry run python app/main.py
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
poetry run python app/main.py
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
- [ ] Advanced RAG integration
- [ ] Team collaboration features
- [ ] Mobile companion app
- [ ] Cloud synchronization
- [ ] Advanced analytics

## 🐛 Troubleshooting

### Common Issues

**GUI not displaying on Linux:**
```bash
export DISPLAY=:0
poetry run python app/main.py
```

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